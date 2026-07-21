import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from nequix.config import ATOMIC_NUMBERS
from nequix.config.models import load_atom_energies
from nequix.data import AtomPackDataset, ConcatDataset, Dataset


@dataclass
class PartialStats:
    n_structures: int = 0
    energy_per_atom_sum: float = 0.0
    force_sq_sum: float = 0.0
    n_force_components: int = 0
    neighbors_sum: float = 0.0
    nodes_sum: int = 0
    edges_sum: int = 0
    max_n_nodes: int = 0
    max_n_edges: int = 0

    def merge(self, other: "PartialStats") -> None:
        self.n_structures += other.n_structures
        self.energy_per_atom_sum += other.energy_per_atom_sum
        self.force_sq_sum += other.force_sq_sum
        self.n_force_components += other.n_force_components
        self.neighbors_sum += other.neighbors_sum
        self.nodes_sum += other.nodes_sum
        self.edges_sum += other.edges_sum
        self.max_n_nodes = max(self.max_n_nodes, other.max_n_nodes)
        self.max_n_edges = max(self.max_n_edges, other.max_n_edges)


def chunk_stats(dataset: Dataset, indices: np.ndarray, atom_energies: np.ndarray) -> PartialStats:
    stats = PartialStats()
    for index in indices:
        graph = dataset._get_graph_dict(int(index))
        n_node = int(graph["n_node"][0])
        n_edge = int(graph["n_edge"][0])
        e0 = float(np.sum(atom_energies[graph["species"]]))
        stats.n_structures += 1
        stats.energy_per_atom_sum += (float(graph["energy"][0]) - e0) / n_node
        stats.force_sq_sum += float(np.sum(graph["forces"].astype(np.float64) ** 2))
        stats.n_force_components += graph["forces"].size
        stats.neighbors_sum += n_edge / n_node
        stats.nodes_sum += n_node
        stats.edges_sum += n_edge
        stats.max_n_nodes = max(stats.max_n_nodes, n_node)
        stats.max_n_edges = max(stats.max_n_edges, n_edge)
    return stats


def parse_path(path: str) -> tuple[str, int]:
    """Parse "path.atp" or "path.atp:REPEAT" into (path, repeat)."""
    base, _, repeat = path.rpartition(":")
    if base and repeat.isdigit():
        return base, int(repeat)
    return path, 1


def compute_dataset_stats(
    paths: list[str],
    atom_energies_name: str,
    cutoff: float,
    sample_frac: float,
    seed: int,
    n_workers: int,
    chunk_size: int,
) -> dict[str, float]:
    """Statistics for a (possibly repeated) mix of AtomPack datasets.

    Matches the training distribution: repeats weight a dataset the same way
    repeating its path in ``TrainerConfig.train_path`` does.
    """
    datasets = []
    for path, repeat in (parse_path(path) for path in paths):
        dataset = AtomPackDataset(
            file_path=path, atomic_numbers=ATOMIC_NUMBERS, cutoff=cutoff, backend="dict"
        )
        datasets.extend([dataset] * repeat)
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    n_samples = int(len(dataset) * sample_frac)
    indices = np.random.default_rng(seed).permutation(len(dataset))[:n_samples]
    e0 = np.array([load_atom_energies(atom_energies_name)[z] for z in sorted(ATOMIC_NUMBERS)])

    chunks = [indices[start : start + chunk_size] for start in range(0, n_samples, chunk_size)]
    stats = PartialStats()
    with tqdm(total=n_samples, unit="structures") as progress:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(chunk_stats, dataset, chunk, e0) for chunk in chunks]
            for future in as_completed(futures):
                partial = future.result()
                stats.merge(partial)
                progress.update(partial.n_structures)

    return {
        "shift": stats.energy_per_atom_sum / stats.n_structures,
        "scale": float(np.sqrt(stats.force_sq_sum / stats.n_force_components)),
        "avg_n_neighbors": stats.neighbors_sum / stats.n_structures,
        "avg_n_nodes": stats.nodes_sum / stats.n_structures,
        "avg_n_edges": stats.edges_sum / stats.n_structures,
        "max_n_nodes": stats.max_n_nodes,
        "max_n_edges": stats.max_n_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=compute_dataset_stats.__doc__)
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        help='atompack file(s), each optionally "path.atp:REPEAT" (e.g. data/mptrj.atp:8)',
    )
    parser.add_argument("--atom-energies", required=True, help="atom energies name: mp, omat, oam")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="fraction of the mix to sample; max_n_* are underestimates when < 1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=None, help="optional output .yml path")
    args = parser.parse_args()

    stats = compute_dataset_stats(
        args.dataset_paths,
        args.atom_energies,
        args.cutoff,
        args.sample_frac,
        args.seed,
        args.n_workers,
        args.chunk_size,
    )
    text = yaml.safe_dump(stats, sort_keys=False)
    print("dataset statistics for the TrainerConfig:")
    print(text)
    if args.output is not None:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
