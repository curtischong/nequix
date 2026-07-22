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
from nequix.config import ModelMetadata, TrainerConfig, checkpoint_dir, config_values
from nequix.data import (
    AtomPackDataset,
    ConcatDataset,
    DataLoader,
    ParallelLoader,
    prefetch,
)
from nequix.evaluation import (
    benchmarks_due,
    run_model_evaluations,
    validate_benchmark_config,
    validate_validation_config,
)
from nequix.hardware import peak_device_memory_bytes
from nequix.model import (
    DirectForceNequix,
    conservative_backbone,
    model_from_metadata,
    replace_normalization,
    weight_decay_mask,
)
from nequix.run_summary import build_run_summary, print_run_summary_csv

TRAINING_STATE_FORMAT = "nequix-training-state-v2"


def _format_percentage(fraction: float) -> str:
    percentage = 100.0 * fraction
    if percentage.is_integer():
        return str(int(percentage))
    return f"{percentage:g}".replace(".", "p")


def wandb_run_name(config: TrainerConfig) -> str:
    """Build the data-schedule-prefixed run name used by the training configs."""
    if config.wandb_run_name:
        return config.wandb_run_name

    run_name = config.run_name or config.name
    dataset_name = config.dataset_name
    if not dataset_name:
        return run_name

    train_fraction = float(config.train_frac)
    fraction_suffix = "" if train_fraction == 1.0 else _format_percentage(train_fraction)
    return f"{dataset_name}{fraction_suffix}_{config.n_epochs}ep_{run_name}"


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


def unreplicate(tree):
    """Return the single-device copy of a pmap-replicated pytree."""
    return jax.tree.map(lambda x: x[0], tree)


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
    """Serialize the complete, unreplicated training state to ``path``."""
    state = {
        "format": TRAINING_STATE_FORMAT,
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cloudpickle.dumps(state))


def _repoint_symlink(link: Path, target: Path) -> None:
    """Atomically point ``link`` at a sibling checkpoint file."""
    tmp = link.with_name(link.name + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target.name)
    tmp.replace(link)


def save_checkpoint(
    run_dir,
    model,
    ema_model,
    optim,
    opt_state,
    step,
    epoch,
    best_val_loss,
    *,
    best=False,
    wandb_run_id=None,
    training_runtime_seconds=0.0,
    validation_runtime_seconds=0.0,
) -> Path:
    """Write a step-stamped checkpoint and repoint the latest.pkl/best.pkl symlinks.

    Every checkpoint is kept as ``step_<global step>.pkl``; ``latest.pkl`` always
    points at the newest one and ``best.pkl`` at the best-validation one.
    """
    path = Path(run_dir) / f"step_{int(step):09d}.pkl"
    save_training_state(
        path,
        model,
        ema_model,
        optim,
        opt_state,
        step,
        epoch,
        best_val_loss,
        wandb_run_id=wandb_run_id,
        training_runtime_seconds=training_runtime_seconds,
        validation_runtime_seconds=validation_runtime_seconds,
    )
    _repoint_symlink(path.with_name("latest.pkl"), path)
    if best:
        _repoint_symlink(path.with_name("best.pkl"), path)
    return path


def load_training_state(path):
    with open(path, "rb") as f:
        state = cloudpickle.load(f)
    expected_keys = {
        "format",
        "model",
        "ema_model",
        "optim",
        "opt_state",
        "step",
        "epoch",
        "best_val_loss",
        "wandb_run_id",
        "training_runtime_seconds",
        "validation_runtime_seconds",
    }
    if not isinstance(state, dict) or set(state) != expected_keys:
        raise ValueError("invalid Nequix training state")
    if state["format"] != TRAINING_STATE_FORMAT:
        raise ValueError(f"unsupported Nequix training state format: {state['format']!r}")
    return (
        state["model"],
        state["ema_model"],
        state["optim"],
        state["opt_state"],
        state["step"],
        state["epoch"],
        state["best_val_loss"],
        state["wandb_run_id"],
        state["training_runtime_seconds"],
        state["validation_runtime_seconds"],
    )


def model_metadata(config: TrainerConfig) -> ModelMetadata:
    """Build the checkpoint metadata a training config describes."""
    return ModelMetadata(
        atomic_numbers=config.atomic_numbers,
        atom_energies=config.atom_energy_list(),
        shift=config.shift,
        scale=config.scale,
        avg_n_neighbors=config.avg_n_neighbors,
        model_config=config.model_config,
    )


def same_architecture(a, b) -> bool:
    """Whether two models have identical parameter counts, shapes, and dtypes.

    Model statics hold callables that only compare by identity, so full pytree
    equality is too strict even for two identical builds.
    """
    arrays_a = jax.tree.leaves(eqx.filter(a, eqx.is_array))
    arrays_b = jax.tree.leaves(eqx.filter(b, eqx.is_array))
    if len(arrays_a) != len(arrays_b):
        return False
    return all(x.shape == y.shape and x.dtype == y.dtype for x, y in zip(arrays_a, arrays_b))


def attach_direct_force_head(model, config: TrainerConfig) -> DirectForceNequix:
    return DirectForceNequix(
        model,
        config.model_config.hidden_irreps,
        key=jax.random.fold_in(jax.random.key(config.seed), 1),
    )


def build_model(config: TrainerConfig):
    """Construct the training architecture shared by real runs and probes."""
    model = model_from_metadata(
        model_metadata(config), kernel=config.kernel, key=jax.random.key(config.seed)
    )
    if config.force_mode == "direct":
        model = attach_direct_force_head(model, config)
    return model


def build_optimizer(config, model, steps_per_epoch):
    """Construct the optimizer and schedule shared by real runs and probes."""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * config.warmup_factor,
        peak_value=config.learning_rate,
        end_value=1e-6,
        warmup_steps=config.warmup_epochs * steps_per_epoch,
        decay_steps=config.n_epochs * steps_per_epoch,
    )
    if config.optimizer == "adamw":
        optim = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=config.weight_decay,
                mask=weight_decay_mask(model),
            ),
        )
    elif config.optimizer == "muon":
        optim = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            optax.contrib.muon(
                learning_rate=schedule,
                weight_decay=config.weight_decay if config.weight_decay != 0.0 else None,
                weight_decay_mask=weight_decay_mask(model),
            ),
        )
    else:
        raise ValueError(f"optimizer {config.optimizer} not supported")
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
            config.energy_weight,
            config.force_weight,
            config.stress_weight,
            config.loss_type,
        )
        # Each local loss is normalized by global real-sample counts, so a sum
        # produces the gradient of the equivalent combined batch.
        grads = jax.lax.psum(grads, axis_name="device")
        metrics["grad_norm"] = optax.global_norm(grads)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)

        decay = jnp.minimum(config.ema_decay, (1 + step) / (10 + step))
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        model_params = eqx.filter(model, eqx.is_array)
        new_ema_params = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, model_params
        )
        ema_model = eqx.combine(ema_static, new_ema_params)
        return model, ema_model, opt_state, total_loss, metrics

    return train_step


def _batch_counts(batch):
    """Return the real (graph, node, edge) counts in one stacked multi-device batch."""
    graph_masks = jax.vmap(jraph.get_graph_padding_mask)(batch)
    node_masks = jax.vmap(jraph.get_node_padding_mask)(batch)
    return (
        int(graph_masks.sum().item()),
        int(node_masks.sum().item()),
        int((batch.n_edge * graph_masks).sum().item()),
    )


def load_datasets(config: TrainerConfig):
    """Construct the train and validation datasets shared by training and precomputation."""

    def make_dataset(path):
        return AtomPackDataset(
            file_path=path,
            atomic_numbers=config.atomic_numbers,
            cutoff=config.model_config.cutoff,
            backend="jax",
        )

    if isinstance(config.train_path, tuple):
        train_dataset = ConcatDataset([make_dataset(path) for path in config.train_path])
    else:
        train_dataset = make_dataset(config.train_path)
    if config.valid_frac is not None:
        train_dataset, val_dataset = train_dataset.split(
            valid_frac=config.valid_frac, seed=config.seed
        )
    else:
        if config.valid_path is None:
            raise ValueError("valid_path must be specified when valid_frac is not provided")
        val_dataset = make_dataset(config.valid_path)

    train_dataset = train_dataset.subset(float(config.train_frac), seed=config.seed)
    return train_dataset, val_dataset


def train(run_config: TrainerConfig):
    """Train a JAX Nequix model from a registered Python config."""
    invocation_start = time.perf_counter()
    config = run_config
    if config.force_mode not in {"conservative", "direct"}:
        raise ValueError(f"force mode {config.force_mode!r} is not supported")
    validation_config = config.validation
    benchmark_config = config.benchmarks
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least one")
    validate_validation_config(validation_config)
    validate_benchmark_config(benchmark_config)

    train_dataset, val_dataset = load_datasets(config)
    metadata = model_metadata(config)

    num_devices = len(jax.devices())
    per_device_train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        max_n_nodes=config.max_n_nodes,
        max_n_edges=config.max_n_edges,
        avg_n_nodes=config.avg_n_nodes,
        avg_n_edges=config.avg_n_edges,
        seed=config.seed,
        num_workers=16,
    )
    train_loader = ParallelLoader(per_device_train_loader, num_devices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        max_n_nodes=config.max_n_nodes,
        max_n_edges=config.max_n_edges,
        avg_n_nodes=config.avg_n_nodes,
        avg_n_edges=config.avg_n_edges,
        num_workers=16,
    )
    n_graph = per_device_train_loader.n_graph
    n_node = per_device_train_loader.n_node
    n_edge = per_device_train_loader.n_edge

    wandb_sync = (
        TriggerWandbSyncHook() if os.environ.get("WANDB_MODE") == "offline" else lambda: None
    )

    model = model_from_metadata(metadata, kernel=config.kernel, key=jax.random.key(config.seed))
    run_dir = checkpoint_dir(config)
    resume_path = run_dir / "latest.pkl"
    resume_exists = resume_path.exists()
    if config.finetune_from is not None and Path(config.finetune_from).exists():
        _, finetune_model, *_ = load_training_state(config.finetune_from)
        finetune_model = replace_normalization(
            conservative_backbone(finetune_model),
            atom_energies=metadata.atom_energies,
            shift=config.shift,
            scale=config.scale,
        )
        if not same_architecture(finetune_model, model):
            raise ValueError("fine-tuning checkpoint and run config use different architectures")
        model = finetune_model
    elif config.finetune_from is not None and not resume_exists:
        raise FileNotFoundError(f"fine-tuning checkpoint does not exist: {config.finetune_from}")

    if config.force_mode == "direct":
        model = attach_direct_force_head(model, config)

    param_count = sum(p.size for p in jax.tree.flatten(eqx.filter(model, eqx.is_array))[0])

    steps_per_epoch = max(
        1,
        math.ceil(len(train_dataset) / (config.batch_size * jax.device_count())),
    )
    optim, schedule = build_optimizer(config, model, steps_per_epoch)

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    ema_model = model
    step = jnp.array(0)
    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = None
    training_runtime_seconds = 0.0
    validation_runtime_seconds = 0.0
    evaluation_runtime_seconds = 0.0
    final_val_metrics = {}

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
        ) = load_training_state(resume_path)

    devices = list(jax.devices())
    model = jax.device_put_replicated(model, devices)
    ema_model = jax.device_put_replicated(ema_model, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    run_name = wandb_run_name(config)
    wandb_config = {
        **config_values(config),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "batch_n_graph": n_graph,
        "batch_n_node": n_node,
        "batch_n_edge": n_edge,
    }
    wandb_init_kwargs = {
        "entity": config.wandb_entity,
        "project": config.wandb_project or "nequix",
        "name": run_name,
        "config": wandb_config,
    }
    if config.wandb_mode:
        wandb_init_kwargs["mode"] = config.wandb_mode
    if wandb_run_id:
        wandb_init_kwargs.update({"id": wandb_run_id, "resume": "allow"})
    wandb_run = wandb.init(**wandb_init_kwargs)
    wandb_run_url = getattr(wandb_run, "url", None)
    wandb_run.define_metric("runtime/training_seconds", summary="last")
    wandb_run.define_metric("runtime/training_hours", summary="last")
    wandb_run.define_metric("runtime/validation_seconds", summary="last")
    wandb_run.define_metric("runtime/evaluation_seconds", summary="last")
    for metric_glob in ("train/*", "val/*", "eval/*", "progress/epoch_percent"):
        wandb_run.define_metric(metric_glob, step_metric="runtime/training_hours")
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.run.summary["param_count"] = param_count
        wandb.run.summary["train_size"] = len(train_dataset)
        wandb.run.summary["val_size"] = len(val_dataset)
        wandb.run.summary["batch/shape"] = {
            "n_graph": n_graph,
            "n_node": n_node,
            "n_edge": n_edge,
        }
        wandb_run_id = getattr(wandb.run, "id", None)

    def runtime_metrics(training_seconds):
        return {
            "runtime/training_seconds": training_seconds,
            "runtime/training_hours": training_seconds / 3600.0,
            "runtime/validation_seconds": validation_runtime_seconds,
            "runtime/evaluation_seconds": evaluation_runtime_seconds,
        }

    def run_validation(epoch, step_in_epoch, training_seconds):
        nonlocal best_val_loss, final_val_metrics, validation_runtime_seconds

        validation_start = time.perf_counter()
        ema_model_single = unreplicate(ema_model)
        val_metrics = evaluate(
            ema_model_single,
            val_loader,
            config.energy_weight,
            config.force_weight,
            config.stress_weight,
            config.loss_type,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                run_dir,
                unreplicate(model),
                ema_model_single,
                optim,
                unreplicate(opt_state),
                step,
                epoch,
                best_val_loss,
                best=True,
                wandb_run_id=wandb_run_id,
                training_runtime_seconds=training_seconds,
                validation_runtime_seconds=validation_runtime_seconds,
            )

        validation_runtime_seconds += time.perf_counter() - validation_start

        final_val_metrics = {key: value.item() for key, value in val_metrics.items()}
        logs = {f"val/{key}": value for key, value in final_val_metrics.items()}
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

    for epoch in range(start_epoch, config.n_epochs):
        train_segment_start = time.perf_counter()
        start_time = time.perf_counter()
        train_loader.loader.set_epoch(epoch)
        last_validation_step_in_epoch = None
        step_in_epoch = 0
        for step_in_epoch, batch in enumerate(prefetch(train_loader), start=1):
            batch_time = time.perf_counter() - start_time
            # Only log steps synchronize with the device; other steps dispatch
            # asynchronously so the host can keep feeding the accelerators.
            log_this_step = int(step + 1) % config.log_every == 0
            if log_this_step:
                jax.block_until_ready(model)
            start_time = time.perf_counter()
            (model, ema_model, opt_state, total_loss, metrics) = train_step(
                model, ema_model, step, opt_state, batch
            )
            step = step + 1
            if log_this_step:
                jax.block_until_ready(total_loss)
                train_time = time.perf_counter() - start_time
                graph_count, node_count, edge_count = _batch_counts(batch)
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
                    num_devices * (n_graph - 1)
                )
                logs["train/node_padding_utilization"] = node_count / (num_devices * (n_node - 1))
                logs["train/edge_padding_utilization"] = edge_count / max(num_devices * n_edge, 1)
                logs["train/peak_gpu_memory_bytes"] = peak_device_memory_bytes()
                current_training_seconds = (
                    training_runtime_seconds + time.perf_counter() - train_segment_start
                )
                completed_steps = int(step.item())
                seconds_per_step = current_training_seconds / max(completed_steps, 1)
                epoch_fraction = min(step_in_epoch / steps_per_epoch, 1.0)
                epoch_steps_remaining = max(steps_per_epoch - step_in_epoch, 0)
                stage_steps_remaining = max(
                    config.n_epochs * steps_per_epoch - completed_steps,
                    0,
                )
                epoch_eta_seconds = epoch_steps_remaining * seconds_per_step
                stage_eta_seconds = stage_steps_remaining * seconds_per_step
                stage_fraction = min(
                    completed_steps / (config.n_epochs * steps_per_epoch),
                    1.0,
                )
                logs.update(
                    {
                        "progress/current_epoch": epoch + 1,
                        "progress/epoch_percent": 100.0 * epoch_fraction,
                        "progress/stage_percent": 100.0 * stage_fraction,
                        "progress/steps_per_epoch": steps_per_epoch,
                        "progress/epochs_completed": epoch + epoch_fraction,
                        "progress/epoch_eta_seconds": epoch_eta_seconds,
                        "progress/epoch_eta_hours": epoch_eta_seconds / 3600.0,
                        "progress/stage_eta_hours": stage_eta_seconds / 3600.0,
                    }
                )
                logs.update(runtime_metrics(current_training_seconds))
                wandb.log(logs, step=step)
                print(f"step: {step}, logs: {logs}")
                wandb_sync()

            validation_cadence = validation_config.every_steps
            if validation_cadence is not None and step_in_epoch % validation_cadence == 0:
                jax.block_until_ready(model)
                training_runtime_seconds += time.perf_counter() - train_segment_start
                run_validation(epoch, step_in_epoch, training_runtime_seconds)
                last_validation_step_in_epoch = step_in_epoch
                train_segment_start = time.perf_counter()

            if benchmarks_due(benchmark_config, int(step.item())):
                jax.block_until_ready(model)
                training_runtime_seconds += time.perf_counter() - train_segment_start
                evaluation_start = time.perf_counter()
                ema_model_single = unreplicate(ema_model)
                eval_metrics = run_model_evaluations(
                    conservative_backbone(ema_model_single),
                    metadata,
                    benchmark_config,
                    kernel=config.kernel,
                    step=int(step.item()),
                )
                evaluation_runtime_seconds += time.perf_counter() - evaluation_start
                logs = {f"eval/{key}": value for key, value in eval_metrics.items()}
                logs["epoch"] = epoch
                logs["step_in_epoch"] = step_in_epoch
                logs.update(runtime_metrics(training_runtime_seconds))
                wandb.log(logs, step=int(step.item()))
                print(f"model evaluations at step: {step}, logs: {logs}")
                wandb_sync()
                train_segment_start = time.perf_counter()
            start_time = time.perf_counter()

        jax.block_until_ready(model)
        training_runtime_seconds += time.perf_counter() - train_segment_start
        if last_validation_step_in_epoch != step_in_epoch:
            run_validation(epoch, step_in_epoch, training_runtime_seconds)

        save_checkpoint(
            run_dir,
            unreplicate(model),
            unreplicate(ema_model),
            optim,
            unreplicate(opt_state),
            step,
            epoch + 1,
            best_val_loss,
            wandb_run_id=wandb_run_id,
            training_runtime_seconds=training_runtime_seconds,
            validation_runtime_seconds=validation_runtime_seconds,
        )

    if not final_val_metrics:
        run_validation(max(start_epoch - 1, 0), 0, training_runtime_seconds)

    per_device_train_loader.shutdown()
    val_loader.shutdown()

    final_runtime_metrics = runtime_metrics(training_runtime_seconds)
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.run.summary.update(final_runtime_metrics)
    wandb.finish()

    devices = jax.devices()
    summary = build_run_summary(
        config,
        run_name=run_name,
        trainer="standard",
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
            "steps_per_epoch": steps_per_epoch,
            "training_examples_seen": len(train_dataset) * config.n_epochs,
            "batch_size_per_accelerator": config.batch_size,
            "batch_n_graph_per_accelerator": n_graph,
            "batch_n_node_per_accelerator": n_node,
            "batch_n_edge_per_accelerator": n_edge,
            "evaluation_runtime_seconds": evaluation_runtime_seconds,
        },
    )
    print_run_summary_csv(summary)


if __name__ == "__main__":
    from nequix.cli import main

    main()
