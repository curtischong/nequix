import itertools
import time
from collections import defaultdict
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import optax
from nequix.config import PFTTrainerConfig, checkpoint_dir, config_values
from nequix.hardware import peak_device_memory_bytes
from nequix.data import (
    AtomPackDataset,
    ConcatDataset,
    DataLoader,
    prefetch,
)
from nequix.model import load_model, node_graph_idx, weight_decay_mask
from nequix.pft.data import PhononDataset
from nequix.run_summary import build_run_summary, print_run_summary_csv
from nequix.train import evaluate as efs_evaluate
from nequix.train import load_training_state, loss as efs_loss, save_checkpoint

import wandb


def loss(
    model,
    graph,
    energy_weight=20.0,
    force_weight=20.0,
    stress_weight=5.0,
    hessian_weight=100.0,
    checkpoint_grad_energy=False,
):
    def total_energy_fn(positions_eps: tuple[jax.Array, jax.Array]):
        positions, eps = positions_eps
        eps_sym = (eps + eps.swapaxes(1, 2)) / 2
        eps_sym_per_node = jnp.repeat(
            eps_sym,
            graph.n_node,
            axis=0,
            total_repeat_length=graph.nodes["positions"].shape[0],
        )
        positions = positions + jnp.einsum("ik,ikj->ij", positions, eps_sym_per_node)
        cell_per_edge = jnp.repeat(
            graph.globals["cell"],
            graph.n_edge,
            axis=0,
            total_repeat_length=graph.edges["shifts"].shape[0],
        )
        offsets = jnp.einsum("ij,ijk->ik", graph.edges["shifts"], cell_per_edge)
        r = positions[graph.senders] - positions[graph.receivers] + offsets
        node_energies = model.node_energies(
            r, graph.nodes["species"], graph.senders, graph.receivers
        )
        return jnp.sum(node_energies), node_energies

    eps = jnp.zeros_like(graph.globals["cell"])

    node_mask = jraph.get_node_padding_mask(graph)
    graph_mask = jraph.get_graph_padding_mask(graph)

    if checkpoint_grad_energy:
        # checkpoint grad energy (useful for larger models/smaller GPUs)
        grad_energy = jax.grad(jax.checkpoint(total_energy_fn), has_aux=True)
    else:
        grad_energy = jax.grad(total_energy_fn, has_aux=True)
    (minus_forces, virial), node_energies = grad_energy((graph.nodes["positions"], eps))

    # hessian column is jacobian vector product of grad energy w.r.t. node
    # degree of freedoms in vs
    # _, hvp, _ = jax.linearize(grad_energy, (graph.nodes["positions"], eps), has_aux=True)
    # hvp = jax.jit(hvp)
    # hessian_col, _ = hvp((graph.nodes["vs"], eps))
    hessian_col, _ = jax.jvp(
        grad_energy,
        ((graph.nodes["positions"], eps),),
        ((graph.nodes["vs"], eps),),
        has_aux=True,
    )[1]

    graph_energies = jraph.segment_sum(
        node_energies,
        node_graph_idx(graph),
        num_segments=graph.n_node.shape[0],
        indices_are_sorted=True,
    )
    det = jnp.abs(jnp.linalg.det(graph.globals["cell"]))[:, None, None]
    det = jnp.where(det > 0.0, det, 1.0)  # padded graphs have det = 0
    stress = virial / det

    # mask out padding nodes
    hessian_col = jnp.where(node_mask[:, None], hessian_col, 0.0)
    minus_forces = jnp.where(node_mask[:, None], minus_forces, 0.0)
    stress = jnp.where(graph_mask[:, None, None], stress, 0.0)

    energy = graph_energies[:, 0]
    forces = -minus_forces

    energy_loss_per_atom = jnp.sum(
        jnp.abs(energy / graph.n_node - graph.globals["energy"] / graph.n_node) * graph_mask
    ) / jnp.sum(graph_mask)

    force_diff_squared = jnp.sum((forces - graph.nodes["forces"]) ** 2, axis=-1)
    safe_force_diff_squared = jnp.where(force_diff_squared == 0.0, 1.0, force_diff_squared)
    force_loss = jnp.sum(
        jnp.where(force_diff_squared == 0.0, 0.0, jnp.sqrt(safe_force_diff_squared)) * node_mask
    ) / jnp.sum(node_mask)

    stress_loss = jnp.sum(jnp.abs(stress - graph.globals["stress"]) * graph_mask[:, None, None]) / (
        9 * jnp.sum(graph_mask)
    )

    # MAE
    hessian_loss = jnp.sum(
        jnp.abs(hessian_col - graph.nodes["hessian_col"]) * node_mask[:, None]
    ) / (3 * jnp.sum(node_mask))

    total_loss = (
        energy_weight * energy_loss_per_atom
        + force_weight * force_loss
        + stress_weight * stress_loss
        + hessian_weight * hessian_loss
    )

    # metrics:
    energy_mae_per_atom = jnp.sum(
        jnp.abs(energy / graph.n_node - graph.globals["energy"] / graph.n_node) * graph_mask
    ) / jnp.sum(graph_mask)

    # MAE forces
    force_mae = jnp.sum(jnp.abs(forces - graph.nodes["forces"]) * node_mask[:, None]) / (
        3 * jnp.sum(node_mask)
    )

    # MAE stress
    stress_mae_per_atom = jnp.sum(
        jnp.abs(stress - graph.globals["stress"])
        / jnp.where(graph.n_node > 0, graph.n_node, 1.0)[:, None, None]
        * graph_mask[:, None, None]
    ) / (9 * jnp.sum(graph_mask))

    # MAE hessian
    hessian_mae_per_atom = jnp.sum(
        jnp.abs(hessian_col - graph.nodes["hessian_col"]) * node_mask[:, None]
    ) / (3 * jnp.sum(node_mask))

    return total_loss, {
        "energy_mae_per_atom": energy_mae_per_atom,
        "force_mae": force_mae,
        "stress_mae_per_atom": stress_mae_per_atom,
        "hessian_mae_per_atom": hessian_mae_per_atom,
    }


def evaluate(
    model,
    dataloader,
    energy_weight=20.0,
    force_weight=20.0,
    stress_weight=5.0,
    hessian_weight=100.0,
    checkpoint_grad_energy=False,
):
    total_metrics = defaultdict(int)
    total_count = 0
    for batch in prefetch(dataloader):
        n_graphs = jnp.sum(jraph.get_graph_padding_mask(batch))
        val_loss, metrics = eqx.filter_jit(loss)(
            model,
            batch,
            energy_weight,
            force_weight,
            stress_weight,
            hessian_weight,
            checkpoint_grad_energy,
        )
        total_metrics["loss"] += val_loss * n_graphs
        for k, v in metrics.items():
            total_metrics[k] += v * n_graphs
        total_count += n_graphs

    for k, v in total_metrics.items():
        total_metrics[k] = v / total_count
    return total_metrics


def train(run_config: PFTTrainerConfig):
    invocation_start = time.perf_counter()
    config = run_config

    model, metadata = load_model(config.finetune_from, config.kernel)
    param_count = sum(p.size for p in jax.tree.flatten(eqx.filter(model, eqx.is_array))[0])

    if config.optimizer == "muon":
        optim = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            optax.contrib.muon(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay if config.weight_decay != 0.0 else None,
                weight_decay_mask=weight_decay_mask(model),
            ),
        )
    elif config.optimizer == "adam":
        optim = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            optax.adamw(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay if config.weight_decay != 0.0 else None,
                mask=weight_decay_mask(model),
            ),
        )
    else:
        raise ValueError(f"optimizer {config.optimizer!r} is not supported")

    ema_model = jax.tree.map(lambda x: x.copy(), model)  # copy model

    opt_state = optim.init(model)

    train_dataset = PhononDataset(
        file_path=config.train_path,
        atomic_numbers=metadata.atomic_numbers,
        cutoff=metadata.model_config.cutoff,
        random_col=True,
    )

    val_dataset = PhononDataset(
        file_path=config.val_path,
        atomic_numbers=metadata.atomic_numbers,
        cutoff=metadata.model_config.cutoff,
        random_col=False,  # always use first column for validation
    )

    if isinstance(config.extra_train_path, tuple):
        extra_train_dataset = ConcatDataset(
            [
                AtomPackDataset(
                    file_path=path,
                    atomic_numbers=metadata.atomic_numbers,
                    cutoff=metadata.model_config.cutoff,
                )
                for path in config.extra_train_path
            ]
        )
    else:
        extra_train_dataset = AtomPackDataset(
            file_path=config.extra_train_path,
            atomic_numbers=metadata.atomic_numbers,
            cutoff=metadata.model_config.cutoff,
        )
    if config.extra_val_frac is not None:
        extra_train_dataset, extra_val_dataset = extra_train_dataset.split(
            valid_frac=config.extra_val_frac
        )
    else:
        if config.extra_val_path is None:
            raise ValueError("extra_val_path is required when extra_val_frac is not provided")
        extra_val_dataset = AtomPackDataset(
            file_path=config.extra_val_path,
            cutoff=metadata.model_config.cutoff,
            atomic_numbers=metadata.atomic_numbers,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        n_graph=config.n_graph,
        shuffle=True,
        max_n_nodes=config.max_n_nodes,
        max_n_edges=config.max_n_edges,
        avg_n_nodes=config.avg_n_nodes,
        avg_n_edges=config.avg_n_edges,
        num_workers=8,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        n_graph=config.n_graph,
        shuffle=False,
        max_n_nodes=config.max_n_nodes,
        max_n_edges=config.max_n_edges,
        avg_n_nodes=config.avg_n_nodes,
        avg_n_edges=config.avg_n_edges,
        num_workers=8,
    )

    extra_val_loader = DataLoader(
        extra_val_dataset,
        batch_size=config.extra_batch_size,
        shuffle=False,
        max_n_nodes=config.extra_max_n_nodes,
        max_n_edges=config.extra_max_n_edges,
        avg_n_nodes=config.extra_avg_n_nodes,
        avg_n_edges=config.extra_avg_n_edges,
        num_workers=8,
    )

    extra_train_loader = DataLoader(
        extra_train_dataset,
        batch_size=config.extra_batch_size,
        shuffle=True,
        max_n_nodes=config.extra_max_n_nodes,
        max_n_edges=config.extra_max_n_edges,
        avg_n_nodes=config.extra_avg_n_nodes,
        avg_n_edges=config.extra_avg_n_edges,
        num_workers=8,
    )

    loss_fn = partial(
        loss,
        energy_weight=config.energy_weight,
        force_weight=config.force_weight,
        stress_weight=config.stress_weight,
        hessian_weight=config.hessian_weight,
        checkpoint_grad_energy=config.checkpoint_grad_energy,
    )

    extra_train_loss_fn = partial(
        efs_loss,
        energy_weight=config.extra_energy_weight,
        force_weight=config.extra_force_weight,
        stress_weight=config.extra_stress_weight,
    )

    @eqx.filter_jit(donate="all")
    def train_step(model, ema_model, loss_fn, step, opt_state, graph):
        (total_loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, graph
        )
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)

        # update EMA model
        # don't weight early steps as much (from https://github.com/fadel/pytorch_ema)
        decay = jnp.minimum(config.ema_decay, (1 + step) / (10 + step))
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        model_params = eqx.filter(model, eqx.is_array)
        new_ema_params = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, model_params
        )
        ema_model = eqx.combine(ema_static, new_ema_params)
        return model, ema_model, opt_state, total_loss, metrics

    step = jnp.array(0)
    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = None
    training_runtime_seconds = 0.0
    validation_runtime_seconds = 0.0
    final_val_metrics = {}
    final_extra_val_metrics = {}

    run_dir = checkpoint_dir(config)
    resume_path = run_dir / "latest.pkl"
    if resume_path.exists():
        (
            model,
            ema_model,
            optim,
            opt_state,
            step,
            start_epoch,
            best_val_loss,
            wandb_run_id,
            training_runtime_seconds,
            validation_runtime_seconds,
        ) = load_training_state(resume_path)

    wandb_init_kwargs = {
        "entity": config.wandb_entity,
        "project": config.wandb_project,
        "name": config.name,
        "config": config_values(config),
    }
    if wandb_run_id:
        wandb_init_kwargs.update({"id": wandb_run_id, "resume": "allow"})
    wandb_run = wandb.init(**wandb_init_kwargs)
    wandb_run_url = getattr(wandb_run, "url", None)
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb_run_id = getattr(wandb.run, "id", None)

    for epoch in range(start_epoch, config.n_epochs):
        training_start = time.perf_counter()
        train_loader.set_epoch(epoch)
        extra_train_loader.set_epoch(epoch)

        # infinite iterator over extra train loader
        extra_iter = itertools.chain.from_iterable(iter(lambda: prefetch(extra_train_loader), None))

        for batch in prefetch(train_loader):
            start = time.time()
            model, ema_model, opt_state, total_loss, metrics = train_step(
                model, ema_model, loss_fn, step.copy(), opt_state, batch
            )
            total_loss.block_until_ready()
            step_time = time.time() - start

            # extra train step with original training data/loss fn
            for _ in range(config.extra_train_steps):
                extra_batch = next(extra_iter)
                model, ema_model, opt_state, extra_total_loss, extra_metrics = train_step(
                    model, ema_model, extra_train_loss_fn, step.copy(), opt_state, extra_batch
                )

            step = step + 1
            if step % config.log_every == 0:
                logs = {}
                for k, v in metrics.items():
                    logs[f"train/{k}"] = v.mean().item()
                logs["train/loss"] = total_loss.mean().item()
                logs["train/batch_size"] = jraph.get_graph_padding_mask(batch).sum().item()
                logs["train/time"] = step_time
                if config.extra_train_steps:
                    for k, v in extra_metrics.items():
                        logs[f"extra_train/{k}"] = v.mean().item()
                    logs["extra_train/loss"] = extra_total_loss.mean().item()
                wandb.log(logs, step=step)
                print(f"step {step:03d} logs: {logs}")

        jax.block_until_ready(model)
        training_runtime_seconds += time.perf_counter() - training_start

        if epoch % config.val_every == 0 or epoch == config.n_epochs - 1:
            validation_start = time.perf_counter()
            val_metrics = evaluate(
                ema_model,
                val_loader,
                config.energy_weight,
                config.force_weight,
                config.stress_weight,
                config.hessian_weight,
                config.checkpoint_grad_energy,
            )

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]

            final_val_metrics = {key: value.mean().item() for key, value in val_metrics.items()}
            logs = {f"val/{key}": value for key, value in final_val_metrics.items()}
            logs["epoch"] = epoch
            wandb.log(logs, step=step)
            print(f"epoch {epoch:03d} val metrics: {logs}")

            extra_val_metrics = efs_evaluate(
                ema_model,
                extra_val_loader,
                energy_weight=config.extra_energy_weight,
                force_weight=config.extra_force_weight,
                stress_weight=config.extra_stress_weight,
            )
            final_extra_val_metrics = {
                key: value.mean().item() for key, value in extra_val_metrics.items()
            }
            extra_logs = {
                f"extra_val/{key}": value for key, value in final_extra_val_metrics.items()
            }
            wandb.log(extra_logs, step=step)
            print(f"epoch {epoch:03d} extra val metrics: {extra_logs}")

            validation_runtime_seconds += time.perf_counter() - validation_start
            save_checkpoint(
                run_dir,
                model,
                ema_model,
                optim,
                opt_state,
                step,
                epoch + 1,
                best_val_loss,
                best=improved,
                wandb_run_id=wandb_run_id,
                training_runtime_seconds=training_runtime_seconds,
                validation_runtime_seconds=validation_runtime_seconds,
            )

    if not final_val_metrics:
        validation_start = time.perf_counter()
        val_metrics = evaluate(
            ema_model,
            val_loader,
            config.energy_weight,
            config.force_weight,
            config.stress_weight,
            config.hessian_weight,
            config.checkpoint_grad_energy,
        )
        extra_val_metrics = efs_evaluate(
            ema_model,
            extra_val_loader,
            energy_weight=config.extra_energy_weight,
            force_weight=config.extra_force_weight,
            stress_weight=config.extra_stress_weight,
        )
        final_val_metrics = {key: value.mean().item() for key, value in val_metrics.items()}
        final_extra_val_metrics = {
            key: value.mean().item() for key, value in extra_val_metrics.items()
        }
        validation_runtime_seconds += time.perf_counter() - validation_start

    for loader in (train_loader, val_loader, extra_train_loader, extra_val_loader):
        loader.shutdown()

    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.run.summary.update(
            {
                "runtime/training_seconds": training_runtime_seconds,
                "runtime/validation_seconds": validation_runtime_seconds,
            }
        )
    wandb.finish()

    devices = jax.devices()
    summary = build_run_summary(
        config,
        run_name=config.name,
        trainer="pft",
        final_metrics=final_val_metrics,
        best_val_loss=best_val_loss,
        steps_completed=int(step.item()),
        epochs_completed=config.n_epochs,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        param_count=param_count,
        accelerator_count=len(devices),
        accelerator_type=";".join(sorted({device.device_kind for device in devices})),
        backend=jax.default_backend(),
        training_runtime_seconds=training_runtime_seconds,
        validation_runtime_seconds=validation_runtime_seconds,
        invocation_runtime_seconds=time.perf_counter() - invocation_start,
        peak_accelerator_memory_bytes=peak_device_memory_bytes(),
        run_id=wandb_run_id,
        run_url=wandb_run_url,
        extra_values={
            **{
                f"final_extra_val_{key}": value
                for key, value in final_extra_val_metrics.items()
            },
            "extra_train_size": len(extra_train_dataset),
            "extra_val_size": len(extra_val_dataset),
            "training_examples_seen": len(train_dataset) * config.n_epochs,
            "batch_size": config.batch_size,
            "n_graph": config.n_graph,
            "extra_batch_size": config.extra_batch_size,
        },
    )
    print_run_summary_csv(summary)


if __name__ == "__main__":
    from nequix.cli import main

    main()
