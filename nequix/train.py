import functools
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import optax
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from nequix.autobatch import probe_summary, tune_training_batch
from nequix.config import TrainerConfig, config_dict
from nequix.data import (
    ConcatDataset,
    DataLoader,
    ParallelLoader,
    average_atom_energies,
    dataset_from_path,
    dataset_stats,
    prefetch,
)
from nequix.model import (
    DirectForceNequix,
    Nequix,
    conservative_backbone,
    load_model,
    replace_normalization,
    save_model,
    weight_decay_mask,
)
from nequix.train_utils import wandb_run_name


def _loss_statistics(model, batch, loss_type="huber"):
    """Return unnormalized loss/metric sums and their real-sample counts."""
    energy, forces, stress = model(batch)
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    config = {
        "mse": {"energy": "mse", "force": "mse", "stress": "mse"},
        "huber": {"energy": "huber", "force": "huber", "stress": "huber"},
        "mae": {"energy": "mae", "force": "l2", "stress": "mae"},
    }[loss_type]

    loss_fns = {
        "mae": lambda pred, true: jnp.abs(pred - true),
        "mse": lambda pred, true: (pred - true) ** 2,
        "huber": lambda pred, true: optax.losses.huber_loss(pred, true, delta=0.1),
    }

    # energy per atom (see eq. 30 https://www.nature.com/articles/s41467-023-36329-y)
    # can be achieved by dividing predictied and true energy by number of atoms
    safe_n_node = jnp.where(batch.n_node > 0, batch.n_node, 1)
    energy_loss_sum = jnp.sum(
        loss_fns[config["energy"]](energy / safe_n_node, batch.globals["energy"] / safe_n_node)
        * graph_mask
    )
    graph_count = jnp.sum(graph_mask)
    node_count = jnp.sum(node_mask)

    if config["force"] == "l2":
        # l2 norm loss for forces
        # NOTE: double where trick is needed to avoid nan's
        force_diff_squared = jnp.sum((forces - batch.nodes["forces"]) ** 2, axis=-1)
        safe_force_diff_squared = jnp.where(force_diff_squared == 0.0, 1.0, force_diff_squared)
        force_loss_sum = jnp.sum(
            jnp.where(force_diff_squared == 0.0, 0.0, jnp.sqrt(safe_force_diff_squared)) * node_mask
        )
        force_loss_count = node_count
    else:
        force_loss_sum = jnp.sum(
            loss_fns[config["force"]](forces, batch.nodes["forces"]) * node_mask[:, None]
        )
        force_loss_count = 3 * node_count

    if stress is not None and batch.globals["stress"] is not None:
        stress_loss_sum = jnp.sum(
            loss_fns[config["stress"]](stress, batch.globals["stress"]) * graph_mask[:, None, None]
        )
        stress_loss_count = 9 * graph_count
    else:
        stress_loss_sum = jnp.array(0.0)
        stress_loss_count = jnp.array(0)

    # metrics:

    # MAE energy
    energy_mae_sum = jnp.sum(
        jnp.abs(energy / safe_n_node - batch.globals["energy"] / safe_n_node) * graph_mask
    )

    # MAE forces
    force_mae_sum = jnp.sum(jnp.abs(forces - batch.nodes["forces"]) * node_mask[:, None])

    # MAE stress
    if stress is not None and batch.globals["stress"] is not None:
        stress_mae_sum = jnp.sum(
            jnp.abs(stress - batch.globals["stress"])
            / safe_n_node[:, None, None]
            * graph_mask[:, None, None]
        )
        stress_mae_count = 9 * graph_count
    else:
        stress_mae_sum = jnp.array(0.0)
        stress_mae_count = jnp.array(0)

    return {
        "energy_loss": (energy_loss_sum, graph_count),
        "force_loss": (force_loss_sum, force_loss_count),
        "stress_loss": (stress_loss_sum, stress_loss_count),
        "energy_mae_per_atom": (energy_mae_sum, graph_count),
        "force_mae": (force_mae_sum, 3 * node_count),
        "stress_mae_per_atom": (stress_mae_sum, stress_mae_count),
    }


def _safe_mean(total, count):
    return total / jnp.maximum(count, 1)


def _total_and_metrics(statistics, energy_weight, force_weight, stress_weight):
    total_loss = (
        energy_weight * _safe_mean(*statistics["energy_loss"])
        + force_weight * _safe_mean(*statistics["force_loss"])
        + stress_weight * _safe_mean(*statistics["stress_loss"])
    )
    metrics = {
        key: _safe_mean(*statistics[key])
        for key in ("energy_mae_per_atom", "force_mae", "stress_mae_per_atom")
    }
    return total_loss, metrics


@eqx.filter_jit
def loss(model, batch, energy_weight, force_weight, stress_weight, loss_type="huber"):
    """Return globally normalized loss and MAEs for one padded batch."""
    statistics = _loss_statistics(model, batch, loss_type)
    return _total_and_metrics(statistics, energy_weight, force_weight, stress_weight)


def _distributed_loss(model, batch, energy_weight, force_weight, stress_weight, loss_type="huber"):
    """Return a local loss contribution and globally weighted metrics.

    The local contribution is differentiated on each device and its gradients
    are summed by ``make_train_step``. This is equivalent to computing each
    component over one combined multi-device batch, including when devices
    contain different real graph and node counts.
    """
    local = _loss_statistics(model, batch, loss_type)
    global_statistics = jax.tree.map(lambda value: jax.lax.psum(value, axis_name="device"), local)
    total_loss, metrics = _total_and_metrics(
        global_statistics, energy_weight, force_weight, stress_weight
    )

    def local_contribution(name, weight):
        numerator = local[name][0]
        denominator = jax.lax.stop_gradient(jnp.maximum(global_statistics[name][1], 1))
        return weight * numerator / denominator

    differentiable_loss = (
        local_contribution("energy_loss", energy_weight)
        + local_contribution("force_loss", force_weight)
        + local_contribution("stress_loss", stress_weight)
    )
    return differentiable_loss, (total_loss, metrics)


def evaluate(
    model, dataloader, energy_weight=1.0, force_weight=1.0, stress_weight=1.0, loss_type="huber"
):
    """Return loss and RMSE of energy and force in eV and eV/Å respectively"""
    total_metrics = defaultdict(int)
    total_count = 0
    for batch in prefetch(dataloader):
        n_graphs = jnp.sum(jraph.get_graph_padding_mask(batch))
        val_loss, metrics = loss(
            model, batch, energy_weight, force_weight, stress_weight, loss_type
        )
        total_metrics["loss"] += val_loss * n_graphs
        for key, value in metrics.items():
            total_metrics[key] += value * n_graphs
        total_count += n_graphs

    for key, value in total_metrics.items():
        total_metrics[key] = value / total_count

    return total_metrics


def save_training_state(
    path,
    model,
    ema_model,
    optim,
    opt_state,
    step,
    epoch,
    best_val_loss,
    wandb_run_id=None,
    training_runtime_seconds=0.0,
    validation_runtime_seconds=0.0,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model,
        "ema_model": ema_model,
        "optim": optim,
        "opt_state": opt_state,
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "wandb_run_id": wandb_run_id,
        "training_runtime_seconds": training_runtime_seconds,
        "validation_runtime_seconds": validation_runtime_seconds,
    }
    with path.open("wb") as f:
        cloudpickle.dump(state, f)


def load_training_state(path):
    with open(path, "rb") as f:
        state = cloudpickle.load(f)
    return (
        state["model"],
        state["ema_model"],
        state["optim"],
        state["opt_state"],
        state["step"],
        state["epoch"],
        state["best_val_loss"],
        state.get("wandb_run_id"),
        state.get("training_runtime_seconds", 0.0),
        state.get("validation_runtime_seconds", 0.0),
    )


def build_model(config, stats, atom_energies):
    """Construct the training architecture shared by real runs and probes."""
    key = jax.random.key(0)
    model = Nequix(
        key,
        n_species=len(config["atomic_numbers"]),
        hidden_irreps=config["hidden_irreps"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        index_weights=config["index_weights"],
        layer_norm=config["layer_norm"],
        shift=stats["shift"],
        scale=stats["scale"],
        avg_n_neighbors=stats["avg_n_neighbors"],
        atom_energies=atom_energies,
        kernel=config["kernel"],
    )
    if config["force_mode"] == "direct":
        model = DirectForceNequix(
            model,
            config["hidden_irreps"],
            key=jax.random.fold_in(key, 1),
        )
    return model


def build_optimizer(config, model, steps_per_epoch):
    """Construct the optimizer and schedule shared by real runs and probes."""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config["learning_rate"] * config["warmup_factor"],
        peak_value=config["learning_rate"],
        end_value=1e-6,
        warmup_steps=config["warmup_epochs"] * steps_per_epoch,
        decay_steps=config["n_epochs"] * steps_per_epoch,
    )
    if config["optimizer"] == "adamw":
        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=config["weight_decay"],
                mask=weight_decay_mask(model),
            ),
        )
    elif config["optimizer"] == "muon":
        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.contrib.muon(
                learning_rate=schedule,
                weight_decay=config["weight_decay"] if config["weight_decay"] != 0.0 else None,
                weight_decay_mask=weight_decay_mask(model),
            ),
        )
    else:
        raise ValueError(f"optimizer {config['optimizer']} not supported")
    return optim, schedule


def make_train_step(optim, config):
    """Build the exact pmapped step used for both training and capacity probes."""

    @functools.partial(eqx.filter_pmap, in_axes=(0, 0, None, 0, 0), axis_name="device")
    def train_step(model, ema_model, step, opt_state, batch):
        (_, (total_loss, metrics)), grads = eqx.filter_value_and_grad(
            _distributed_loss, has_aux=True
        )(
            model,
            batch,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
        )
        # Each local loss is normalized by global real-sample counts, so a sum
        # produces the gradient of the equivalent combined batch.
        grads = jax.lax.psum(grads, axis_name="device")
        metrics["grad_norm"] = optax.global_norm(grads)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)

        decay = jnp.minimum(config["ema_decay"], (1 + step) / (10 + step))
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        model_params = eqx.filter(model, eqx.is_array)
        new_ema_params = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, model_params
        )
        ema_model = eqx.combine(ema_static, new_ema_params)
        return model, ema_model, opt_state, total_loss, metrics

    return train_step


def _peak_device_memory_bytes():
    peaks = []
    for device in jax.devices():
        try:
            statistics = device.memory_stats() or {}
        except (AttributeError, RuntimeError):
            continue
        for key in ("peak_bytes_in_use", "peak_bytes_in_use_limit", "bytes_in_use"):
            if key in statistics:
                peaks.append(int(statistics[key]))
                break
    return max(peaks, default=0)


def train(run_config: TrainerConfig):
    """Train a JAX Nequix model from a registered Python config."""
    if run_config.trainer != "jax":
        raise ValueError(
            f"JAX trainer cannot run {run_config.trainer!r} config {run_config.name!r}"
        )
    config = config_dict(run_config)
    if config["force_mode"] not in {"conservative", "direct"}:
        raise ValueError(f"force mode {config['force_mode']!r} is not supported")
    val_every_steps = config.get("val_every_steps")
    if val_every_steps is not None and val_every_steps <= 0:
        raise ValueError("val_every_steps must be greater than zero")

    # use TMPDIR for slurm jobs if available
    config["cache_dir"] = config.get("cache_dir") or os.environ.get("TMPDIR")

    def make_dataset(path):
        return dataset_from_path(
            file_path=path,
            atomic_numbers=config["atomic_numbers"],
            cutoff=config["cutoff"],
            backend="jax",
        )

    if isinstance(config["train_path"], list):
        train_dataset = ConcatDataset([make_dataset(path) for path in config["train_path"]])
    else:
        train_dataset = make_dataset(config["train_path"])
    if "valid_frac" in config:
        train_dataset, val_dataset = train_dataset.split(
            valid_frac=config["valid_frac"], seed=config.get("seed", 42)
        )
    else:
        assert "valid_path" in config, "valid_path must be specified if valid_frac is not provided"
        val_dataset = make_dataset(config["valid_path"])

    train_dataset = train_dataset.subset(
        float(config.get("train_frac", 1.0)), seed=config.get("seed", 0)
    )

    if "atom_energies" in config:
        atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    else:
        atom_energies = average_atom_energies(train_dataset)

    stats_keys = [
        "shift",
        "scale",
        "avg_n_neighbors",
        "max_n_edges",
        "max_n_nodes",
        "avg_n_nodes",
        "avg_n_edges",
    ]
    if all(key in config for key in stats_keys):
        stats = {key: config[key] for key in stats_keys}
    else:
        stats = dataset_stats(train_dataset, atom_energies)

    tune_result = tune_training_batch(config, stats, train_dataset, atom_energies)
    selected_shape = tune_result.shape
    print(probe_summary(tune_result))

    num_devices = len(jax.devices())
    per_device_train_loader = DataLoader(
        train_dataset,
        batch_size=selected_shape.batch_size,
        n_graph=selected_shape.n_graph,
        shuffle=True,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        seed=config.get("seed", 0),
        num_workers=16,
        packing="best_fit" if config.get("autobatch", True) else "next_fit",
    )
    if (
        per_device_train_loader.n_node != selected_shape.n_node
        or per_device_train_loader.n_edge != selected_shape.n_edge
    ):
        raise RuntimeError("selected autobatch shape does not match the training loader shape")
    train_loader = ParallelLoader(per_device_train_loader, num_devices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=selected_shape.batch_size,
        n_graph=selected_shape.n_graph,
        shuffle=False,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=16,
        packing="best_fit" if config.get("autobatch", True) else "next_fit",
    )
    if val_loader.n_node != selected_shape.n_node or val_loader.n_edge != selected_shape.n_edge:
        raise RuntimeError("selected autobatch shape does not match the validation loader shape")

    wandb_sync = (
        TriggerWandbSyncHook() if os.environ.get("WANDB_MODE") == "offline" else lambda: None
    )

    model_config = {**config, "force_mode": "conservative"}
    model = build_model(model_config, stats, atom_energies)
    resume_exists = "resume_from" in config and Path(config["resume_from"]).exists()
    if "finetune_from" in config and Path(config["finetune_from"]).exists():
        model, checkpoint_config = load_model(config["finetune_from"], config["kernel"])
        if checkpoint_config["atomic_numbers"] != config["atomic_numbers"]:
            raise ValueError("fine-tuning checkpoint and run config use different elements")
        model = replace_normalization(
            model,
            atom_energies=atom_energies,
            shift=stats["shift"],
            scale=stats["scale"],
        )
    elif "finetune_from" in config and not resume_exists:
        raise FileNotFoundError(f"fine-tuning checkpoint does not exist: {config['finetune_from']}")

    if config["force_mode"] == "direct":
        key = jax.random.key(0)
        model = DirectForceNequix(
            model,
            config["hidden_irreps"],
            key=jax.random.fold_in(key, 1),
        )

    param_count = sum(p.size for p in jax.tree.flatten(eqx.filter(model, eqx.is_array))[0])

    steps_per_epoch = max(
        1,
        math.ceil(len(train_dataset) / (selected_shape.batch_size * jax.device_count())),
    )
    optim, schedule = build_optimizer(config, model, steps_per_epoch)

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    model = jax.device_put_replicated(model, list(jax.devices()))
    opt_state = jax.device_put_replicated(opt_state, list(jax.devices()))
    ema_model = jax.tree.map(lambda x: x.copy(), model)  # copy model
    step = jnp.array(0)
    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = None
    training_runtime_seconds = 0.0
    validation_runtime_seconds = 0.0

    if resume_exists:
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
        ) = load_training_state(config["resume_from"])

    run_name = wandb_run_name(run_config.name, config)
    wandb_config = {
        **config,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "autobatch_selected_batch_size": selected_shape.batch_size,
        "autobatch_n_graph": selected_shape.n_graph,
        "autobatch_n_node": selected_shape.n_node,
        "autobatch_n_edge": selected_shape.n_edge,
        "autobatch_cached": tune_result.cached,
    }
    wandb_init_kwargs = {
        "project": config.get("wandb_project", "nequix"),
        "name": run_name,
        "config": wandb_config,
    }
    if config.get("wandb_entity"):
        wandb_init_kwargs["entity"] = config["wandb_entity"]
    if config.get("wandb_mode"):
        wandb_init_kwargs["mode"] = config["wandb_mode"]
    if wandb_run_id:
        wandb_init_kwargs.update({"id": wandb_run_id, "resume": "allow"})
    wandb_run = wandb.init(**wandb_init_kwargs)
    wandb_run.define_metric("runtime/training_seconds", summary="last")
    wandb_run.define_metric("runtime/training_hours", summary="last")
    wandb_run.define_metric("runtime/validation_seconds", summary="last")
    for metric_glob in ("train/*", "val/*"):
        wandb_run.define_metric(metric_glob, step_metric="runtime/training_hours")
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.run.summary["param_count"] = param_count
        wandb.run.summary["train_size"] = len(train_dataset)
        wandb.run.summary["val_size"] = len(val_dataset)
        wandb.run.summary["autobatch/selected_shape"] = {
            "n_graph": selected_shape.n_graph,
            "n_node": selected_shape.n_node,
            "n_edge": selected_shape.n_edge,
        }
        wandb.run.summary["autobatch/probes"] = [
            {
                "batch_size": probe.shape.batch_size,
                "status": probe.status,
                "graphs_per_second": probe.graphs_per_second,
                "nodes_per_second": probe.nodes_per_second,
                "edges_per_second": probe.edges_per_second,
                "peak_memory_bytes": probe.peak_memory_bytes,
                "final_loss": probe.final_loss,
            }
            for probe in tune_result.probes
        ]
        wandb_run_id = getattr(wandb.run, "id", None)

    def runtime_metrics(training_seconds):
        return {
            "runtime/training_seconds": training_seconds,
            "runtime/training_hours": training_seconds / 3600.0,
            "runtime/validation_seconds": validation_runtime_seconds,
        }

    def run_validation(epoch, step_in_epoch, training_seconds):
        nonlocal best_val_loss, validation_runtime_seconds

        validation_start = time.perf_counter()
        ema_model_single = jax.tree.map(lambda x: x[0], ema_model)
        val_metrics = evaluate(
            ema_model_single,
            val_loader,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_model = conservative_backbone(ema_model_single)
            save_model(Path(wandb.run.dir) / "checkpoint.nqx", checkpoint_model, config)
            if "checkpoint_path" in config:
                save_model(config["checkpoint_path"], checkpoint_model, config)

        validation_runtime_seconds += time.perf_counter() - validation_start

        logs = {f"val/{key}": value.item() for key, value in val_metrics.items()}
        logs["epoch"] = epoch
        logs["step_in_epoch"] = step_in_epoch
        logs.update(runtime_metrics(training_seconds))
        global_step = int(step.item())
        wandb.log(logs, step=global_step)
        print(
            f"validation at epoch: {epoch}, step in epoch: {step_in_epoch}, "
            f"global step: {global_step}, logs: {logs}"
        )
        wandb_sync()

    train_step = make_train_step(optim, config)

    for epoch in range(start_epoch, config["n_epochs"]):
        train_segment_start = time.perf_counter()
        start_time = time.perf_counter()
        train_loader.loader.set_epoch(epoch)
        last_validation_step_in_epoch = None
        step_in_epoch = 0
        for step_in_epoch, batch in enumerate(prefetch(train_loader), start=1):
            batch_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            (model, ema_model, opt_state, total_loss, metrics) = train_step(
                model, ema_model, step, opt_state, batch
            )
            jax.block_until_ready(total_loss)
            train_time = time.perf_counter() - start_time
            step = step + 1
            if step % config["log_every"] == 0:
                graph_masks = jax.vmap(jraph.get_graph_padding_mask)(batch)
                node_masks = jax.vmap(jraph.get_node_padding_mask)(batch)
                graph_count = graph_masks.sum().item()
                node_count = node_masks.sum().item()
                edge_count = (batch.n_edge * graph_masks).sum().item()
                logs = {}
                logs["train/loss"] = total_loss.mean().item()
                logs["learning_rate"] = schedule(step).item()
                logs["train/batch_time"] = batch_time
                logs["train/train_time"] = train_time
                for key, value in metrics.items():
                    logs[f"train/{key}"] = value.mean().item()
                logs["train/batch_size"] = graph_count
                logs["train/graphs_per_second"] = graph_count / train_time
                logs["train/nodes_per_second"] = node_count / train_time
                logs["train/edges_per_second"] = edge_count / train_time
                logs["train/graph_padding_utilization"] = graph_count / (
                    num_devices * (selected_shape.n_graph - 1)
                )
                logs["train/node_padding_utilization"] = node_count / (
                    num_devices * (selected_shape.n_node - 1)
                )
                logs["train/edge_padding_utilization"] = edge_count / max(
                    num_devices * selected_shape.n_edge, 1
                )
                logs["train/peak_gpu_memory_bytes"] = _peak_device_memory_bytes()
                current_training_seconds = (
                    training_runtime_seconds + time.perf_counter() - train_segment_start
                )
                logs.update(runtime_metrics(current_training_seconds))
                wandb.log(logs, step=step)
                print(f"step: {step}, logs: {logs}")
                wandb_sync()

            if val_every_steps is not None and step_in_epoch % val_every_steps == 0:
                training_runtime_seconds += time.perf_counter() - train_segment_start
                run_validation(epoch, step_in_epoch, training_runtime_seconds)
                last_validation_step_in_epoch = step_in_epoch
                train_segment_start = time.perf_counter()
            start_time = time.perf_counter()

        training_runtime_seconds += time.perf_counter() - train_segment_start
        if last_validation_step_in_epoch != step_in_epoch:
            run_validation(epoch, step_in_epoch, training_runtime_seconds)

        save_training_state(
            Path(wandb.run.dir) / "state.pkl",
            model,
            ema_model,
            optim,
            opt_state,
            step,
            epoch + 1,
            best_val_loss,
            wandb_run_id=wandb_run_id,
            training_runtime_seconds=training_runtime_seconds,
            validation_runtime_seconds=validation_runtime_seconds,
        )

        if "state_path" in config:
            save_training_state(
                config["state_path"],
                model,
                ema_model,
                optim,
                opt_state,
                step,
                epoch + 1,
                best_val_loss,
                wandb_run_id=wandb_run_id,
                training_runtime_seconds=training_runtime_seconds,
                validation_runtime_seconds=validation_runtime_seconds,
            )

    final_runtime_metrics = runtime_metrics(training_runtime_seconds)
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.run.summary.update(final_runtime_metrics)
    wandb.finish()


def main():
    from nequix.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
