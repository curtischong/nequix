import bisect
import multiprocessing
import os
import queue
import threading
from collections import deque
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import ase
from atompack import Database
from ase.geometry import complete_cell
from ase.stress import voigt_6_to_full_3x3_stress
import jax
import jraph
import matscipy.neighbours
import numpy as np
from tqdm import tqdm


def preprocess_graph(
    atoms: ase.Atoms,
    atom_indices: dict[int, int],
    cutoff: float,
    targets: bool,
) -> dict:
    cell = complete_cell(atoms.cell)  # avoids singular cell
    src, dst, shift = matscipy.neighbours.neighbour_list(
        "ijS", positions=atoms.positions, cell=cell, pbc=atoms.pbc, cutoff=cutoff
    )
    graph_dict = {
        "n_node": np.array([len(atoms)]).astype(np.int32),
        "n_edge": np.array([len(src)]).astype(np.int32),
        "senders": dst.astype(np.int32),
        "receivers": src.astype(np.int32),
        "species": np.array([atom_indices[n] for n in atoms.get_atomic_numbers()]).astype(np.int32),
        "positions": atoms.positions.astype(np.float32),
        "shifts": shift.astype(np.float32),
        "cell": atoms.cell.astype(np.float32) if atoms.pbc.all() else None,
    }
    if targets:
        graph_dict["forces"] = atoms.get_forces().astype(np.float32)
        graph_dict["energy"] = np.array([atoms.get_potential_energy()]).astype(np.float32)
        try:
            graph_dict["stress"] = atoms.get_stress(voigt=False).astype(np.float32)
        except ase.calculators.calculator.PropertyNotImplementedError:
            pass

    return graph_dict


def dict_to_pytorch_geometric(graph_dict: dict):
    import torch
    from torch_geometric.data import Data

    """Convert graph dictionary to PyTorch Geometric Data object"""
    # Convert numpy arrays to torch tensors
    species = torch.from_numpy(graph_dict["species"]).long()  # Node features (atomic species)
    positions = torch.from_numpy(graph_dict["positions"])  # Node positions

    # Edge indices (PyG expects [2, num_edges] format)
    edge_index = torch.stack(
        [torch.from_numpy(graph_dict["senders"]), torch.from_numpy(graph_dict["receivers"])], dim=0
    ).long()

    energy = None if "energy" not in graph_dict else torch.from_numpy(graph_dict["energy"])
    forces = None if "forces" not in graph_dict else torch.from_numpy(graph_dict["forces"])
    stress = (
        None if "stress" not in graph_dict else torch.from_numpy(graph_dict["stress"])[None, :, :]
    )

    # Edge attributes
    edge_attr = torch.from_numpy(graph_dict["shifts"])

    cell = (
        torch.from_numpy(graph_dict["cell"])[None, :, :] if graph_dict["cell"] is not None else None
    )

    n_node = torch.from_numpy(graph_dict["n_node"])
    n_edge = torch.from_numpy(graph_dict["n_edge"])

    # Create Data object
    data = Data(
        n_node=n_node,
        n_edge=n_edge,
        energy=energy,
        forces=forces,
        stress=stress,
        x=species,
        positions=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell=cell,
    )

    return data


def dict_to_graphstuple(graph_dict: dict):
    import jraph

    return jraph.GraphsTuple(
        n_node=graph_dict["n_node"],
        n_edge=graph_dict["n_edge"],
        nodes={
            "species": graph_dict["species"],
            "positions": graph_dict["positions"],
            "forces": graph_dict["forces"] if "forces" in graph_dict else None,
        },
        edges={"shifts": graph_dict["shifts"]},
        senders=graph_dict["senders"],
        receivers=graph_dict["receivers"],
        globals={
            "cell": graph_dict["cell"][None, ...] if graph_dict["cell"] is not None else None,
            "energy": graph_dict["energy"] if "energy" in graph_dict else None,
            "stress": graph_dict["stress"][None, ...] if "stress" in graph_dict else None,
        },
    )


def atomic_numbers_to_indices(atomic_numbers: list[int]) -> dict[int, int]:
    """Convert list of atomic numbers to dictionary of atomic number to index."""
    return {n: i for i, n in enumerate(sorted(atomic_numbers))}


class Dataset(ABC):
    def __init__(self, backend: str = "jax"):
        self.backend = backend

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def _get_graph_dict(self, idx: int) -> dict: ...

    def __getitem__(self, idx: int):
        graph = self._get_graph_dict(idx)
        if self.backend == "jax":
            return dict_to_graphstuple(graph)
        if self.backend == "torch":
            return dict_to_pytorch_geometric(graph)
        return graph  # "dict"

    def split(self, valid_frac: float, seed: int = 42):
        n = len(self)
        perm = np.random.RandomState(seed).permutation(n)
        n_tr = int(round(n * (1 - valid_frac)))
        return IndexDataset(self, perm[:n_tr]), IndexDataset(self, perm[n_tr:])

    def subset(self, fraction: float, seed: int = 0):
        """Return a deterministic random fraction of this dataset."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"dataset fraction must be in (0, 1], got {fraction}")
        if fraction == 1.0:
            return self

        size = int(len(self) * fraction)
        if size == 0:
            raise ValueError(
                f"dataset fraction {fraction} selects no items from a dataset of size {len(self)}"
            )
        indices = np.random.default_rng(seed).permutation(len(self))[:size]
        return IndexDataset(self, indices)


class IndexDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        super().__init__(backend=base.backend)
        self.base, self.indices = base, np.asarray(indices, dtype=int)

    def __len__(self):
        return self.indices.size

    def _get_graph_dict(self, i: int):
        return self.base._get_graph_dict(int(self.indices[i]))


class ConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        super().__init__(backend=datasets[0].backend)
        self.datasets = datasets
        self.len_cumulative = np.cumsum([len(ds) for ds in datasets])

    def __len__(self):
        return self.len_cumulative[-1]

    def _get_graph_dict(self, idx: int):
        ds_idx = bisect.bisect(self.len_cumulative, idx)
        if ds_idx > 0:
            idx = idx - self.len_cumulative[ds_idx - 1]
        return self.datasets[ds_idx]._get_graph_dict(idx)


class AtomPackDataset(Dataset):
    """Random-access AtomPack dataset, reopened independently in each worker process."""

    def __init__(
        self, file_path: str, atomic_numbers: list[int], cutoff: float = 5.0, backend: str = "jax"
    ):
        super().__init__(backend=backend)
        self.atomic_indices = atomic_numbers_to_indices(atomic_numbers)
        self.file_path = Path(file_path)
        self.cutoff = cutoff
        database = Database.open(str(self.file_path))
        self._length = len(database)
        del database
        self._database = None
        self._database_pid = None

    def __len__(self):
        return self._length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_database"] = None
        state["_database_pid"] = None
        return state

    def _get_database(self):
        pid = os.getpid()
        if self._database is None or self._database_pid != pid:
            self._database = Database.open(str(self.file_path))
            self._database_pid = pid
        return self._database

    def _get_molecule(self, idx: int):
        return self._get_database().get_molecule(idx)

    def _molecule_to_graph_dict(self, molecule, idx: int):
        if molecule.energy is None or molecule.forces is None:
            raise ValueError(
                f"AtomPack training record {idx} in {self.file_path} must contain energy and forces"
            )

        positions = np.asarray(molecule.positions)
        atomic_numbers = np.asarray(molecule.atomic_numbers)
        pbc = np.asarray(molecule.pbc if molecule.pbc is not None else (False, False, False))
        raw_cell = np.asarray(molecule.cell) if molecule.cell is not None else np.zeros((3, 3))
        cell = complete_cell(raw_cell)
        src, dst, shift = matscipy.neighbours.neighbour_list(
            "ijS", positions=positions, cell=cell, pbc=pbc, cutoff=self.cutoff
        )

        stress = molecule.stress
        if stress is not None:
            stress = np.asarray(stress)
            if stress.shape == (6,):
                stress = voigt_6_to_full_3x3_stress(stress)

        graph = {
            "n_node": np.array([len(atomic_numbers)], dtype=np.int32),
            "n_edge": np.array([len(src)], dtype=np.int32),
            "senders": dst.astype(np.int32),
            "receivers": src.astype(np.int32),
            "species": np.array(
                [self.atomic_indices[int(number)] for number in atomic_numbers], dtype=np.int32
            ),
            "positions": positions.astype(np.float32),
            "shifts": shift.astype(np.float32),
            "cell": raw_cell.astype(np.float32) if pbc.all() else None,
            "forces": np.asarray(molecule.forces, dtype=np.float32),
            "energy": np.array([molecule.energy], dtype=np.float32),
        }
        if stress is not None:
            graph["stress"] = stress.astype(np.float32)
        return graph

    def _get_graph_dict(self, idx: int):
        return self._molecule_to_graph_dict(self._get_molecule(idx), idx)


def _dataloader_worker(dataset, index_queue, output_queue):
    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        output_queue.put((index, dataset[index]))


def padded_shape(
    batch_size: int,
    avg_n_nodes: float,
    avg_n_edges: float,
    max_n_nodes: int,
    max_n_edges: int,
    buffer_factor: float = 1.1,
) -> tuple[int, int, int]:
    """Return the static (n_graph, n_node, n_edge) padding for a dynamic batch."""
    n_node = int(max(batch_size * avg_n_nodes * buffer_factor, max_n_nodes)) + 1
    n_edge = int(max(batch_size * avg_n_edges * buffer_factor, max_n_edges))
    return batch_size + 1, n_node, n_edge


# multiprocess data loader with dynamic batching, based on
# https://teddykoker.com/2020/12/dataloader/
# https://github.com/google-deepmind/jraph/blob/51f5990/jraph/ogb_examples/data_utils.py
class DataLoader:
    def __init__(
        self,
        dataset,
        max_n_nodes: int,
        max_n_edges: int,
        avg_n_nodes: int,
        avg_n_edges: int,
        batch_size=1,
        n_graph=None,
        seed=0,
        shuffle=False,
        buffer_factor=1.1,
        num_workers=4,
        prefetch_factor=2,
        packing="next_fit",
        packing_lookahead=64,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.idxs = np.arange(len(self.dataset))
        self.idx = 0
        self._generator = None  # created in __iter__
        default_n_graph, self.n_node, self.n_edge = padded_shape(
            batch_size, avg_n_nodes, avg_n_edges, max_n_nodes, max_n_edges, buffer_factor
        )
        self.n_graph = int(n_graph if n_graph is not None else default_n_graph)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        if packing not in {"next_fit", "best_fit"}:
            raise ValueError(f"unknown packing strategy {packing!r}")
        if packing_lookahead < 1:
            raise ValueError("packing_lookahead must be at least one")
        self.packing = packing
        self.packing_lookahead = packing_lookahead

        self._started = False
        self.index_queue = None
        self.output_queue = None
        self.workers = []
        self.prefetch_idx = 0

    def _start_workers(self):
        if self._started:
            return

        # NB: we can use fork here, only because we are not using jax
        # in the workers (data is just numpy arrays)
        # multiprocessing.set_start_method("spawn", force=True)
        self._started = True
        self.index_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        for _ in range(self.num_workers):
            worker = multiprocessing.Process(
                target=_dataloader_worker,
                args=(self.dataset, self.index_queue, self.output_queue),
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(seed=hash((self.seed, epoch)) % 2**32)

    def _prefetch(self):
        prefetch_limit = self.idx + self.prefetch_factor * self.num_workers * self.batch_size
        while self.prefetch_idx < len(self.dataset) and self.prefetch_idx < prefetch_limit:
            self.index_queue.put(self.idxs[self.prefetch_idx])
            self.prefetch_idx += 1

    def make_generator(self):
        cache = {}
        self.prefetch_idx = 0

        while True:
            if self.idx >= len(self.dataset):
                return

            self._prefetch()

            real_idx = self.idxs[self.idx]

            if real_idx in cache:
                item = cache[real_idx]
                del cache[real_idx]
            else:
                while True:
                    try:
                        (index, data) = self.output_queue.get(timeout=0)
                    except queue.Empty:
                        continue

                    if index == real_idx:
                        item = data
                        break
                    else:
                        cache[index] = data

            yield item
            self.idx += 1

    def __iter__(self):
        self._start_workers()
        self.idx = 0
        if self.shuffle:
            self.idxs = self.rng.permutation(np.arange(len(self.dataset)))
        if self.packing == "best_fit":
            self._generator = best_fit_dynamic_batch(
                self.make_generator(),
                n_node=self.n_node,
                n_edge=self.n_edge,
                n_graph=self.n_graph,
                lookahead=self.packing_lookahead,
            )
        else:
            self._generator = jraph.dynamically_batch(
                self.make_generator(),
                n_node=self.n_node,
                n_edge=self.n_edge,
                n_graph=self.n_graph,
            )
        return self

    def __next__(self):
        return next(self._generator)


class ParallelLoader:
    def __init__(self, loader: DataLoader, n: int):
        self.loader = loader
        self.n = n

    def __iter__(self):
        it = iter(self.loader)
        while True:
            batches = []
            for _ in range(self.n):
                try:
                    batches.append(next(it))
                except StopIteration:
                    break
            if not batches:
                return
            batches.extend(_empty_padded_batch(batches[0]) for _ in range(self.n - len(batches)))
            yield jax.tree.map(lambda *x: np.stack(x), *batches)


def _graph_size(graph):
    return (
        int(np.asarray(graph.n_node).size),
        int(np.asarray(graph.n_node).sum()),
        int(np.asarray(graph.n_edge).sum()),
    )


def _bounded_best_fit(items, capacity, lookahead):
    """Pack (payload, size) items with deterministic, bounded-lookahead best fit.

    ``capacity`` and every size are ``(graphs, nodes, edges)``. Items may move
    only within the rolling lookahead window, and each input item is returned
    exactly once.
    """
    capacity = tuple(int(value) for value in capacity)
    if capacity[0] < 1 or capacity[1] < 1 or capacity[2] < 0:
        raise ValueError(
            f"graph/node capacities must be positive and edges nonnegative: {capacity}"
        )
    if lookahead < 1:
        raise ValueError("lookahead must be at least one")

    items = iter(items)
    waiting = deque()
    exhausted = False

    def refill():
        nonlocal exhausted
        while len(waiting) < lookahead and not exhausted:
            try:
                payload, size = next(items)
            except StopIteration:
                exhausted = True
                break
            size = tuple(int(value) for value in size)
            if any(value < 0 for value in size):
                raise ValueError(f"item has a negative size: {size}")
            if any(value > limit for value, limit in zip(size, capacity)):
                raise ValueError(f"item with size {size} exceeds capacity {capacity}")
            waiting.append((payload, size))

    refill()
    while waiting:
        used = (0, 0, 0)
        packed = []
        while True:
            fitting = []
            for position, (payload, size) in enumerate(waiting):
                added = tuple(a + b for a, b in zip(used, size))
                if all(value <= limit for value, limit in zip(added, capacity)):
                    # Best fit minimizes normalized space left. The original
                    # stream position is a stable tie-breaker.
                    residual = sum(
                        (limit - value) / limit
                        for value, limit in zip(added, capacity)
                        if limit > 0
                    )
                    fitting.append((residual, position, added))
            if not fitting:
                break
            _, position, used = min(fitting)
            packed.append(waiting[position][0])
            del waiting[position]
            refill()
        if not packed:  # guarded by the individual-size check above
            raise RuntimeError("best-fit packer made no progress")
        yield packed


def bounded_best_fit_indices(sizes, capacity, lookahead=64):
    """Pack ordered item sizes, yielding each input index exactly once."""
    yield from _bounded_best_fit(enumerate(sizes), capacity, lookahead)


def best_fit_dynamic_batch(graphs, n_node, n_edge, n_graph, lookahead=64):
    """Batch graphs into a single static shape using bounded best-fit packing."""
    # Jraph needs one spare graph and one spare node for its padding graph.
    capacity = (int(n_graph) - 1, int(n_node) - 1, int(n_edge))
    sized = ((graph, _graph_size(graph)) for graph in graphs)
    for packed in _bounded_best_fit(sized, capacity, lookahead):
        yield jraph.pad_with_graphs(
            jraph.batch_np(packed),
            n_node=int(n_node),
            n_edge=int(n_edge),
            n_graph=int(n_graph),
        )


def _empty_padded_batch(batch):
    """Return an all-padding batch with the same arrays and static shape."""
    empty = jax.tree.map(np.zeros_like, batch)
    n_node = np.zeros_like(batch.n_node)
    n_edge = np.zeros_like(batch.n_edge)
    n_node[0] = sum(np.asarray(batch.n_node))
    n_edge[0] = sum(np.asarray(batch.n_edge))
    return empty._replace(n_node=n_node, n_edge=n_edge)


# simple threaded prefetching for dataloader (lets us build our dyanamic batches async)
def prefetch(loader, queue_size=4):
    q = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()

    def worker():
        try:
            for item in loader:
                if stop_event.is_set():
                    return
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            elif isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        thread.join(timeout=1.0)


def write_atompack_database(
    input_path: str | Path,
    output_path: str | Path,
    glob_pattern: str,
    read_molecules: Callable,
    n_workers: int = 16,
):
    """Convert input files into one AtomPack database, in parallel across files.

    ``read_molecules`` must be a picklable top-level function that maps one input
    file path to a list of AtomPack molecules.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.suffix != ".atp":
        raise ValueError(f"AtomPack output path must end in .atp: {output_path}")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_paths = sorted(input_path.rglob(glob_pattern)) if input_path.is_dir() else [input_path]
    if not file_paths:
        raise ValueError(f"no {glob_pattern} files found in {input_path}")

    database = Database(str(output_path), overwrite=True)
    if n_workers == 1 or len(file_paths) == 1:
        for molecules in tqdm(map(read_molecules, file_paths), total=len(file_paths)):
            database.add_molecules(molecules)
    else:
        with multiprocessing.Pool(min(n_workers, len(file_paths))) as pool:
            for molecules in tqdm(pool.imap(read_molecules, file_paths), total=len(file_paths)):
                database.add_molecules(molecules)
    database.flush()
