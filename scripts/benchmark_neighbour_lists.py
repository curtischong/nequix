# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ase>=3.24.0",
#     "atompack-db>=0.4.0",
#     "matscipy>=1.1.1",
#     "numpy",
#     "vesin>=0.6.0",
# ]
# ///
"""Benchmark matscipy vs vesin neighbour lists on real .atp structures (CPU only).

Self-contained: run with `uv run scripts/benchmark_neighbour_lists.py` so it uses an
isolated environment and leaves the project venv (and any running training) untouched.

Reports throughput, agreement between backends, and per-process memory drift
(matscipy <= 1.2.0 leaks ~0.65 kB per neighbour_list call; see
https://github.com/libAtoms/matscipy c/tools.c resize_array).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable

import matscipy.neighbours
import numpy as np
from ase.geometry import complete_cell
from atompack import Database
from vesin import NeighborList

NeighbourFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class Structure:
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray


@dataclass
class BackendResult:
    name: str
    seconds_per_pass: float
    structures_per_second: float
    total_pairs: int
    private_dirty_delta_kb: int


def private_dirty_kb() -> int:
    with open(f"/proc/{os.getpid()}/smaps_rollup") as f:
        for line in f:
            if line.startswith("Private_Dirty:"):
                return int(line.split()[1])
    return 0


def load_structures(path: str, n_structures: int, seed: int) -> list[Structure]:
    database = Database.open(path)
    idxs = np.random.default_rng(seed).integers(0, len(database), size=n_structures)
    structures = []
    for idx in idxs:
        molecule = database.get_molecule(int(idx))
        pbc = np.asarray(molecule.pbc if molecule.pbc is not None else (False, False, False))
        raw_cell = np.asarray(molecule.cell) if molecule.cell is not None else np.zeros((3, 3))
        structures.append(
            Structure(
                positions=np.asarray(molecule.positions, dtype=np.float64),
                cell=complete_cell(raw_cell),
                pbc=pbc,
            )
        )
    return structures


def matscipy_pairs(cutoff: float) -> NeighbourFn:
    def compute(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray):
        i, j, _ = matscipy.neighbours.neighbour_list(
            "ijS", positions=positions, cell=cell, pbc=pbc, cutoff=cutoff
        )
        return i, j

    return compute


def vesin_pairs(cutoff: float, n_threads: int) -> NeighbourFn:
    nl = NeighborList(cutoff=cutoff, full_list=True, n_threads=n_threads)

    def compute(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray):
        i, j = nl.compute(points=positions, box=cell, periodic=pbc, quantities="ij")
        return i, j

    return compute


def run_backend(
    name: str, fn: NeighbourFn, structures: list[Structure], passes: int
) -> BackendResult:
    for structure in structures[:32]:  # warmup
        fn(structure.positions, structure.cell, structure.pbc)
    memory_before = private_dirty_kb()
    total_pairs = 0
    started = time.perf_counter()
    for _ in range(passes):
        for structure in structures:
            i, _ = fn(structure.positions, structure.cell, structure.pbc)
            total_pairs += i.size
    elapsed = time.perf_counter() - started
    return BackendResult(
        name=name,
        seconds_per_pass=elapsed / passes,
        structures_per_second=passes * len(structures) / elapsed,
        total_pairs=total_pairs // passes,
        private_dirty_delta_kb=private_dirty_kb() - memory_before,
    )


def check_agreement(structures: list[Structure], backends: dict[str, NeighbourFn]) -> int:
    """Compare (i, j) pair multisets between backends; return count of mismatched structures."""
    mismatches = 0
    for structure in structures:
        pair_sets = []
        for fn in backends.values():
            i, j = fn(structure.positions, structure.cell, structure.pbc)
            pairs = np.stack([i, j], axis=1)
            pair_sets.append(pairs[np.lexsort((j, i))])
        if not all(np.array_equal(pair_sets[0], pairs) for pairs in pair_sets[1:]):
            mismatches += 1
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atp-path", default="data/omat/train.atp")
    parser.add_argument("--n-structures", type=int, default=2000)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="vesin CPU threads; keep 1 for a fair single-core comparison with matscipy",
    )
    args = parser.parse_args()

    structures = load_structures(args.atp_path, args.n_structures, args.seed)
    n_atoms = [len(s.positions) for s in structures]
    mixed_pbc = sum(1 for s in structures if s.pbc.any() and not s.pbc.all())
    print(
        f"{len(structures)} structures, atoms/structure "
        f"min={min(n_atoms)} median={int(np.median(n_atoms))} max={max(n_atoms)}, "
        f"{mixed_pbc} with mixed pbc"
    )

    backends = {
        "matscipy": matscipy_pairs(args.cutoff),
        "vesin": vesin_pairs(args.cutoff, args.n_threads),
    }
    mismatches = check_agreement(structures[:200], backends)
    print(f"pair-set agreement on 200 structures: {200 - mismatches}/200")

    results = [run_backend(name, fn, structures, args.passes) for name, fn in backends.items()]
    print(json.dumps([asdict(result) for result in results], indent=2))
    baseline, candidate = results
    print(
        f"vesin speedup: {candidate.structures_per_second / baseline.structures_per_second:.2f}x, "
        f"memory drift per pass: matscipy "
        f"{baseline.private_dirty_delta_kb / args.passes:+.0f} kB, "
        f"vesin {candidate.private_dirty_delta_kb / args.passes:+.0f} kB"
    )


if __name__ == "__main__":
    main()
