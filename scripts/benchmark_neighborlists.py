"""Benchmark Matscipy and AlchemiOps neighbor lists and training throughput.

Run this script from the repository root after syncing the environment::

    uv run python scripts/benchmark_neighborlists.py data/mptrj.atp --max-neighbors 512

The training comparison uses the real data loader, graph packing, JAX model,
optimizer, and pmapped training step. Each backend runs in a fresh subprocess
so compilation caches and retained accelerator state do not cross-contaminate
the measurements. The portable JAX convolutions are used unless
``--train-kernel`` is passed. Use ``--skip-training`` or
``--skip-neighbor-benchmarks`` to run only one half of the benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matscipy.neighbours
import numpy as np
import torch
from ase.geometry import complete_cell
from atompack import Database
from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list


_TRAIN_RESULT_PREFIX = "NEQUIX_TRAIN_RESULT="


@dataclass(frozen=True)
class System:
    """The inputs needed to construct one neighbor list."""

    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray


@dataclass(frozen=True)
class HostBatch:
    """A group of systems in AlchemiOps batch format."""

    positions: np.ndarray
    cells: np.ndarray
    pbc: np.ndarray
    system_idx: np.ndarray


@dataclass(frozen=True)
class DeviceBatch:
    """A host batch copied to a Torch device."""

    positions: torch.Tensor
    cells: torch.Tensor
    pbc: torch.Tensor
    system_idx: torch.Tensor


def load_systems(path: Path, n_samples: int, seed: int) -> tuple[list[System], float]:
    """Load a deterministic random sample from an AtomPack database."""
    database = Database.open(str(path))
    if n_samples > len(database):
        raise ValueError(f"requested {n_samples} samples from a dataset of size {len(database)}")

    indices = np.random.default_rng(seed).choice(len(database), n_samples, replace=False)
    start = time.perf_counter()
    systems = []
    for index in indices:
        molecule = database.get_molecule(int(index))
        systems.append(
            System(
                positions=np.asarray(molecule.positions, dtype=np.float32),
                cell=complete_cell(np.asarray(molecule.cell)).astype(np.float32),
                pbc=np.asarray(molecule.pbc, dtype=bool),
            )
        )
    return systems, time.perf_counter() - start


def make_host_batches(systems: list[System], batch_size: int) -> list[HostBatch]:
    """Pack systems into batches without copying their neighbor-list outputs."""
    batches = []
    for start in range(0, len(systems), batch_size):
        chunk = systems[start : start + batch_size]
        atom_counts = [len(system.positions) for system in chunk]
        batches.append(
            HostBatch(
                positions=np.concatenate([system.positions for system in chunk]),
                cells=np.stack([system.cell for system in chunk]),
                pbc=np.stack([system.pbc for system in chunk]),
                system_idx=np.repeat(
                    np.arange(len(chunk), dtype=np.int32), np.asarray(atom_counts)
                ),
            )
        )
    return batches


def copy_batch_to_device(batch: HostBatch, device: torch.device) -> DeviceBatch:
    """Copy one packed batch to a Torch device."""
    return DeviceBatch(
        positions=torch.from_numpy(batch.positions).to(device),
        cells=torch.from_numpy(batch.cells).to(device),
        pbc=torch.from_numpy(batch.pbc).to(device),
        system_idx=torch.from_numpy(batch.system_idx).to(device),
    )


def alchemi_neighbors(
    method: str, batch: DeviceBatch, cutoff: float, max_neighbors: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run an AlchemiOps neighbor-list implementation."""
    if method == "naive":
        return batch_naive_neighbor_list(
            batch.positions,
            cutoff,
            batch_idx=batch.system_idx,
            cell=batch.cells,
            pbc=batch.pbc,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )
    if method == "cell":
        return batch_cell_list(
            batch.positions,
            cutoff,
            batch.cells,
            batch.pbc,
            batch.system_idx,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )
    raise ValueError(f"unknown AlchemiOps method {method!r}")


def matscipy_corpus(systems: list[System], cutoff: float) -> int:
    """Construct each neighbor list as the current training loader does."""
    n_edges = 0
    for system in systems:
        src, _, _ = matscipy.neighbours.neighbour_list(
            "ijS",
            positions=system.positions,
            cell=system.cell,
            pbc=system.pbc,
            cutoff=cutoff,
        )
        n_edges += len(src)
    return n_edges


def alchemi_corpus_resident(
    method: str,
    batches: list[DeviceBatch],
    cutoff: float,
    max_neighbors: int | None,
) -> int:
    """Run AlchemiOps with inputs and outputs left on the GPU."""
    n_edges = 0
    for batch in batches:
        neighbor_list, _, _ = alchemi_neighbors(method, batch, cutoff, max_neighbors)
        n_edges += neighbor_list.shape[1]
    torch.cuda.synchronize()
    return n_edges


def alchemi_corpus_end_to_end(
    method: str,
    batches: list[HostBatch],
    cutoff: float,
    device: torch.device,
    max_neighbors: int | None,
) -> int:
    """Run AlchemiOps including host-to-device and output-to-host copies."""
    n_edges = 0
    for host_batch in batches:
        batch = copy_batch_to_device(host_batch, device)
        outputs = alchemi_neighbors(method, batch, cutoff, max_neighbors)
        host_outputs = tuple(output.cpu().numpy() for output in outputs)
        n_edges += host_outputs[0].shape[1]
    torch.cuda.synchronize()
    return n_edges


def benchmark(function: Callable[[], int], rounds: int, warmup: bool = False) -> tuple[float, int]:
    """Return the median wall time and edge count for a callable."""
    if warmup:
        function()
    timings = []
    n_edges = 0
    for _ in range(rounds):
        gc.collect()
        start = time.perf_counter()
        n_edges = function()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), n_edges


def sorted_rows(rows: np.ndarray) -> np.ndarray:
    """Lexicographically sort edge-and-shift rows."""
    return rows[np.lexsort(rows.T[::-1])]


def matscipy_edge_rows(systems: list[System], cutoff: float) -> np.ndarray:
    """Build global-index edge-and-shift rows for correctness comparison."""
    rows = []
    offset = 0
    for system in systems:
        src, dst, shifts = matscipy.neighbours.neighbour_list(
            "ijS",
            positions=system.positions,
            cell=system.cell,
            pbc=system.pbc,
            cutoff=cutoff,
        )
        rows.append(np.column_stack((src + offset, dst + offset, shifts)))
        offset += len(system.positions)
    return sorted_rows(np.concatenate(rows).astype(np.int64, copy=False))


def verify_method(
    method: str,
    systems: list[System],
    cutoff: float,
    device: torch.device,
    max_neighbors: int | None,
) -> int:
    """Require AlchemiOps and Matscipy to return identical edges and shifts."""
    expected = matscipy_edge_rows(systems, cutoff)
    device_batch = copy_batch_to_device(make_host_batches(systems, len(systems))[0], device)
    neighbor_list, _, shifts = alchemi_neighbors(method, device_batch, cutoff, max_neighbors)
    actual = np.column_stack(
        (
            neighbor_list.T.cpu().numpy(),
            shifts.cpu().numpy(),
        )
    ).astype(np.int64, copy=False)
    actual = sorted_rows(actual)
    np.testing.assert_array_equal(actual, expected)
    return len(actual)


def _batch_counts(batch) -> tuple[int, int, int]:
    """Count real graphs, nodes, and edges in a NumPy multi-device batch."""
    graph_count = 0
    node_count = 0
    edge_count = 0
    for n_node, n_edge in zip(np.asarray(batch.n_node), np.asarray(batch.n_edge)):
        # jraph padding is one nonempty padding graph followed by zero-node
        # graphs. This is the NumPy equivalent of get_graph_padding_mask and
        # avoids introducing a synchronization into the timed training loop.
        trailing_empty = int(np.argmin(n_node[::-1] == 0))
        real_graphs = len(n_node) - trailing_empty - 1
        graph_count += real_graphs
        node_count += int(n_node[:real_graphs].sum())
        edge_count += int(n_edge[:real_graphs].sum())
    return graph_count, node_count, edge_count


def run_training_worker(args: argparse.Namespace) -> None:
    """Run one end-to-end training measurement in an isolated process."""
    from dataclasses import replace

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from nequix.config import RUNS, TrainerConfig
    from nequix.data import AtomPackDataset, DataLoader, IndexDataset, ParallelLoader, prefetch
    from nequix.train import build_model, build_optimizer, make_train_step

    if args.training_config not in RUNS:
        raise ValueError(f"unknown training config {args.training_config!r}")
    base_config = RUNS[args.training_config]
    if not isinstance(base_config, TrainerConfig):
        raise TypeError("the training benchmark requires a standard TrainerConfig")

    config = replace(
        base_config,
        train_path=str(args.dataset),
        model_config=replace(base_config.model_config, cutoff=args.cutoff),
        kernel=args.train_kernel,
        neighbor_backend=args.training_worker,
        neighbor_batch_size=args.train_neighbor_batch_size,
        neighbor_max_neighbors=args.max_neighbors,
    )
    dataset = AtomPackDataset(
        file_path=str(args.dataset),
        atomic_numbers=config.atomic_numbers,
        cutoff=config.model_config.cutoff,
        backend="jax",
    )
    sample_count = min(args.train_samples, len(dataset))
    if sample_count < 1:
        raise ValueError("the training benchmark requires at least one sample")
    indices = np.random.default_rng(args.seed).permutation(len(dataset))[:sample_count]
    dataset = IndexDataset(dataset, indices)

    stats = config.dataset_stats()
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=min(args.train_workers, sample_count),
        packing="best_fit",
        neighbor_backend=config.neighbor_backend,
        neighbor_cutoff=config.model_config.cutoff,
        neighbor_batch_size=config.neighbor_batch_size,
        neighbor_max_neighbors=config.neighbor_max_neighbors,
    )
    # This must happen before JAX or Alchemi initializes accelerator threads.
    loader.start_workers()
    devices = list(jax.devices())
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError("the training benchmark requires a JAX GPU")

    parallel_loader = ParallelLoader(loader, len(devices))
    model = build_model(config)
    steps_per_epoch = max(
        1,
        math.ceil(sample_count / (args.train_batch_size * len(devices))),
    )
    optim, _ = build_optimizer(config, model, steps_per_epoch)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    model = jax.device_put_replicated(model, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    ema_model = jax.tree.map(lambda value: value.copy(), model)
    train_step = make_train_step(optim, config)
    step = jnp.array(0)
    batches = prefetch(parallel_loader)

    def next_batch():
        try:
            return next(batches)
        except StopIteration as error:
            required = (args.train_warmup_steps + args.train_steps) * args.train_batch_size
            raise RuntimeError(
                f"dataset exhausted before the benchmark finished; use at least about "
                f"{required} samples (currently {sample_count})"
            ) from error

    warmup_start = time.perf_counter()
    for _ in range(args.train_warmup_steps):
        batch = next_batch()
        model, ema_model, opt_state, total_loss, _ = train_step(
            model, ema_model, step, opt_state, batch
        )
        jax.block_until_ready(model)
        step += 1
    warmup_seconds = time.perf_counter() - warmup_start

    totals = [0, 0, 0]
    start = time.perf_counter()
    for _ in range(args.train_steps):
        batch = next_batch()
        totals = [left + right for left, right in zip(totals, _batch_counts(batch))]
        model, ema_model, opt_state, total_loss, _ = train_step(
            model, ema_model, step, opt_state, batch
        )
        step += 1
    jax.block_until_ready(model)
    elapsed = time.perf_counter() - start
    batches.close()

    result = {
        "backend": config.neighbor_backend,
        "device": devices[0].device_kind,
        "config": config.name,
        "sample_count": sample_count,
        "batch_size": args.train_batch_size,
        "warmup_steps": args.train_warmup_steps,
        "steps": args.train_steps,
        "warmup_seconds": warmup_seconds,
        "seconds": elapsed,
        "steps_per_second": args.train_steps / elapsed,
        "graphs_per_second": totals[0] / elapsed,
        "nodes_per_second": totals[1] / elapsed,
        "edges_per_second": totals[2] / elapsed,
        "graphs": totals[0],
        "nodes": totals[1],
        "edges": totals[2],
        "final_loss": float(total_loss.mean().item()),
    }
    print(f"{_TRAIN_RESULT_PREFIX}{json.dumps(result, sort_keys=True)}")


def run_training_subprocess(args: argparse.Namespace, backend: str) -> dict:
    """Run one backend in a clean interpreter and return its result."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.dataset),
        "--training-worker",
        backend,
        "--training-config",
        args.training_config,
        "--cutoff",
        str(args.cutoff),
        "--train-samples",
        str(args.train_samples),
        "--train-batch-size",
        str(args.train_batch_size),
        "--train-neighbor-batch-size",
        str(args.train_neighbor_batch_size),
        "--train-warmup-steps",
        str(args.train_warmup_steps),
        "--train-steps",
        str(args.train_steps),
        "--train-workers",
        str(args.train_workers),
        "--max-neighbors",
        str(args.max_neighbors),
        "--seed",
        str(args.seed),
    ]
    if args.train_kernel:
        command.append("--train-kernel")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=args.train_timeout,
        check=False,
    )
    if completed.returncode:
        output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part)
        raise RuntimeError(f"{backend} training benchmark failed:\n{output[-8000:]}")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_TRAIN_RESULT_PREFIX):
            return json.loads(line.removeprefix(_TRAIN_RESULT_PREFIX))
    raise RuntimeError(
        f"{backend} training benchmark returned no result:\n{completed.stdout[-8000:]}"
    )


def benchmark_training(args: argparse.Namespace) -> None:
    """Compare steady-state end-to-end training throughput for both backends."""
    results = []
    for backend in ("matscipy", "alchemi"):
        print(
            f"training {backend} ({args.train_warmup_steps} warmup + {args.train_steps} steps)..."
        )
        results.append(run_training_subprocess(args, backend))

    baseline = results[0]["graphs_per_second"]
    print(
        "\ntraining_backend,seconds,steps_per_second,graphs_per_second,"
        "nodes_per_second,edges_per_second,speedup_vs_matscipy,warmup_seconds,final_loss"
    )
    for result in results:
        print(
            f"{result['backend']},{result['seconds']:.6f},"
            f"{result['steps_per_second']:.4f},{result['graphs_per_second']:.2f},"
            f"{result['nodes_per_second']:.2f},{result['edges_per_second']:.2f},"
            f"{result['graphs_per_second'] / baseline:.3f},"
            f"{result['warmup_seconds']:.6f},{result['final_loss']:.6g}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument(
        "--methods", choices=("naive", "cell"), nargs="+", default=["naive", "cell"]
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--verify-samples", type=int, default=32)
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=512,
        help="fixed Alchemi capacity per atom; must cover the dataset's densest environment",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="run only correctness and isolated neighbor-list benchmarks",
    )
    parser.add_argument(
        "--skip-neighbor-benchmarks",
        action="store_true",
        help="run only the end-to-end training comparison",
    )
    parser.add_argument("--training-config", default="nequix-mp-1")
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--train-neighbor-batch-size", type=int, default=1024)
    parser.add_argument("--train-warmup-steps", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--train-workers", type=int, default=16)
    parser.add_argument("--train-timeout", type=float, default=1800)
    parser.add_argument(
        "--train-kernel",
        action="store_true",
        help="use the optional OpenEquivariance JAX kernel from the registered config",
    )
    parser.add_argument(
        "--training-worker",
        choices=("matscipy", "alchemi"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    for name in (
        "samples",
        "rounds",
        "max_neighbors",
        "train_samples",
        "train_batch_size",
        "train_neighbor_batch_size",
        "train_steps",
        "train_workers",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least one")
    if args.train_warmup_steps < 0:
        parser.error("--train-warmup-steps cannot be negative")
    if args.skip_training and args.skip_neighbor_benchmarks:
        parser.error("--skip-training and --skip-neighbor-benchmarks are mutually exclusive")
    return args


def main() -> None:
    """Run correctness checks and neighbor-list benchmarks."""
    args = parse_args()
    if args.training_worker is not None:
        run_training_worker(args)
        return

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA device")

    print(f"device: {torch.cuda.get_device_name(device)}")
    if not args.skip_training:
        benchmark_training(args)
    if args.skip_neighbor_benchmarks:
        return

    systems, load_time = load_systems(args.dataset, args.samples, args.seed)
    n_atoms = sum(len(system.positions) for system in systems)
    print(f"samples: {len(systems)}")
    print(f"atoms: {n_atoms} ({n_atoms / len(systems):.2f}/system)")
    print(f"database load: {load_time:.3f} s")

    verification_systems = systems[: min(args.verify_samples, len(systems))]
    for method in args.methods:
        n_verified = verify_method(
            method, verification_systems, args.cutoff, device, args.max_neighbors
        )
        print(f"verified {method}: {n_verified} edges")

    matscipy_time, expected_edges = benchmark(
        lambda: matscipy_corpus(systems, args.cutoff), args.rounds
    )
    print("\nmethod,batch_size,scope,seconds,systems_per_second,speedup_vs_matscipy,edges")
    print(
        f"matscipy,1,host,{matscipy_time:.6f},{len(systems) / matscipy_time:.2f},"
        f"1.000,{expected_edges}"
    )

    for batch_size in args.batch_sizes:
        host_batches = make_host_batches(systems, batch_size)
        device_batches = [copy_batch_to_device(batch, device) for batch in host_batches]
        for method in args.methods:
            resident_time, resident_edges = benchmark(
                lambda method=method: alchemi_corpus_resident(
                    method, device_batches, args.cutoff, args.max_neighbors
                ),
                args.rounds,
                warmup=True,
            )
            end_to_end_time, end_to_end_edges = benchmark(
                lambda method=method: alchemi_corpus_end_to_end(
                    method, host_batches, args.cutoff, device, args.max_neighbors
                ),
                args.rounds,
                warmup=True,
            )
            if resident_edges != expected_edges or end_to_end_edges != expected_edges:
                raise RuntimeError(
                    f"{method} returned {resident_edges}/{end_to_end_edges} edges; "
                    f"Matscipy returned {expected_edges}"
                )
            print(
                f"alchemi-{method},{batch_size},gpu-resident,{resident_time:.6f},"
                f"{len(systems) / resident_time:.2f},{matscipy_time / resident_time:.3f},"
                f"{resident_edges}"
            )
            print(
                f"alchemi-{method},{batch_size},end-to-end,{end_to_end_time:.6f},"
                f"{len(systems) / end_to_end_time:.2f},{matscipy_time / end_to_end_time:.3f},"
                f"{end_to_end_edges}"
            )


if __name__ == "__main__":
    main()
