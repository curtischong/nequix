from __future__ import annotations

import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp

from nequix.autobatch import (
    BatchShape,
    ProbeResult,
    _looks_like_oom,
    _shape_from_dict,
    peak_device_memory_bytes,
)
from nequix.config import TrainerConfig
from nequix.data import DataLoader, ParallelLoader
from nequix.train import _batch_counts, build_model, build_optimizer, make_train_step


def _block_step(output):
    jax.block_until_ready(output[3])
    return output


def run_probe(config: TrainerConfig, dataset, shape: BatchShape) -> ProbeResult:
    stats = config.dataset_stats()
    loader = DataLoader(
        dataset,
        batch_size=shape.batch_size,
        n_graph=shape.n_graph,
        shuffle=False,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=min(4, max(1, len(dataset))),
        packing="best_fit",
        neighbor_backend=config.neighbor_backend,
        neighbor_cutoff=config.model_config.cutoff,
        neighbor_batch_size=config.neighbor_batch_size,
        neighbor_max_neighbors=config.neighbor_max_neighbors,
    )
    loader.start_workers()
    devices = list(jax.devices())
    if not devices or devices[0].platform != "gpu":
        return ProbeResult(shape=shape, status="failed", error="probe did not find a JAX GPU")

    parallel_loader = ParallelLoader(loader, len(devices))
    warmup_steps = 3
    timed_steps = 3
    batches = []
    for batch in parallel_loader:
        batches.append(batch)
        if len(batches) >= warmup_steps + timed_steps:
            break
    if not batches:
        return ProbeResult(
            shape=shape, status="failed", error="training dataset produced no batches"
        )

    model = build_model(config)
    steps_per_epoch = max(1, math.ceil(len(dataset) / (shape.batch_size * len(devices))))
    optim, _ = build_optimizer(config, model, steps_per_epoch)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    model = jax.device_put_replicated(model, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    ema_model = jax.tree.map(lambda value: value.copy(), model)
    step = jnp.array(0)
    train_step = make_train_step(optim, config)
    final_loss = None

    for index in range(warmup_steps):
        model, ema_model, opt_state, total_loss, _ = _block_step(
            train_step(model, ema_model, step, opt_state, batches[index % len(batches)])
        )
        final_loss = float(total_loss.mean().item())
        step += 1

    totals = [0, 0, 0]
    timed_rates = []
    node_rates = []
    edge_rates = []
    for index in range(timed_steps):
        batch = batches[(warmup_steps + index) % len(batches)]
        counts = _batch_counts(batch)
        start = time.perf_counter()
        model, ema_model, opt_state, total_loss, _ = _block_step(
            train_step(model, ema_model, step, opt_state, batch)
        )
        final_loss = float(total_loss.mean().item())
        elapsed = time.perf_counter() - start
        step += 1
        totals = [a + b for a, b in zip(totals, counts)]
        timed_rates.append(counts[0] / elapsed)
        node_rates.append(counts[1] / elapsed)
        edge_rates.append(counts[2] / elapsed)

    allocated = (
        timed_steps * len(devices) * (shape.n_graph - 1),
        timed_steps * len(devices) * (shape.n_node - 1),
        timed_steps * len(devices) * shape.n_edge,
    )
    if final_loss is None or not math.isfinite(final_loss):
        return ProbeResult(shape=shape, status="failed", error=f"non-finite loss: {final_loss}")
    return ProbeResult(
        shape=shape,
        status="ok",
        graphs_per_second=statistics.median(timed_rates),
        nodes_per_second=statistics.median(node_rates),
        edges_per_second=statistics.median(edge_rates),
        graph_utilization=totals[0] / allocated[0],
        node_utilization=totals[1] / allocated[1],
        edge_utilization=totals[2] / max(allocated[2], 1),
        peak_memory_bytes=peak_device_memory_bytes(),
        final_loss=final_loss,
        timed_graphs_per_second=tuple(timed_rates),
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    payload_path, shape_path, result_path = map(Path, argv)
    shape = _shape_from_dict(json.loads(shape_path.read_text()))
    with payload_path.open("rb") as payload_file:
        payload = cloudpickle.load(payload_file)
    try:
        result = run_probe(payload["config"], payload["train_dataset"], shape)
    except BaseException:
        error = traceback.format_exc()
        result = ProbeResult(
            shape=shape,
            status="oom" if _looks_like_oom(error) else "failed",
            error=error[-4000:],
        )
    result_path.write_text(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
