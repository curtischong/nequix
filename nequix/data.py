import bisect
import mmap
import multiprocessing
import os
import queue
import threading
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
    inner_cutoff: float | None = None,
) -> dict:
    cell = complete_cell(atoms.cell)  # avoids singular cell
    src, dst, distance, shift = matscipy.neighbours.neighbour_list(
        "ijdS", positions=atoms.positions, cell=cell, pbc=atoms.pbc, cutoff=cutoff
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
    if inner_cutoff is not None:
        graph_dict["inner"] = distance < inner_cutoff
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
        edges={"shifts": graph_dict["shifts"]}
        | ({"inner": graph_dict["inner"]} if "inner" in graph_dict else {}),
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
        self,
        file_path: str,
        atomic_numbers: list[int],
        cutoff: float = 5.0,
        backend: str = "jax",
        inner_cutoff: float | None = None,
    ):
        super().__init__(backend=backend)
        self.atomic_indices = atomic_numbers_to_indices(atomic_numbers)
        self.file_path = Path(file_path)
        self.cutoff = cutoff
        self.inner_cutoff = inner_cutoff
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
        src, dst, distance, shift = matscipy.neighbours.neighbour_list(
            "ijdS", positions=positions, cell=cell, pbc=pbc, cutoff=self.cutoff
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
        if self.inner_cutoff is not None:
            graph["inner"] = distance < self.inner_cutoff
        if stress is not None:
            graph["stress"] = stress.astype(np.float32)
        return graph

    def _get_graph_dict(self, idx: int):
        return self._molecule_to_graph_dict(self._get_molecule(idx), idx)


def _check_inner_budget(batch: jraph.GraphsTuple, n_edge_inner: int) -> jraph.GraphsTuple:
    """Warn when a batch's inner edges overflow the inner edge budget.

    The model sorts inner edges first on device (``sort_inner_edges_first``) and
    truncates the overflow out of the inner layers (those edges keep their
    outer-layer contributions); the warning signals the budget should grow.
    """
    n_inner = int(batch.edges["inner"].sum())
    if n_inner > n_edge_inner:
        print(
            f"WARNING: {n_inner} inner edges exceed the {n_edge_inner} edge budget; "
            "increase inner_edge_fraction"
        )
    return batch


def _batches(
    dataset,
    indices: np.ndarray,
    n_node: int,
    n_edge: int,
    n_graph: int,
    n_edge_inner: int | None,
    abort=None,
):
    """Dynamically batch ``dataset[indices]`` in order, stopping early once ``abort`` is set."""

    def graphs():
        for index in indices:
            if abort is not None and abort.is_set():
                return
            yield dataset[index]

    for batch in jraph.dynamically_batch(graphs(), n_node=n_node, n_edge=n_edge, n_graph=n_graph):
        if n_edge_inner is not None:
            _check_inner_budget(batch, n_edge_inner)
        yield batch


_SHM_DIR = Path("/dev/shm")
_SHM_ALIGN = 64


def _share_batch(batch: jraph.GraphsTuple, name: str) -> tuple:
    """Write a batch's arrays into a fresh shared-memory file and return its handle."""
    leaves, treedef = jax.tree_util.tree_flatten(batch)
    leaves = [np.ascontiguousarray(leaf) for leaf in leaves]
    layout = []
    size = 0
    for leaf in leaves:
        layout.append((leaf.shape, leaf.dtype.str, size))
        size += -(-leaf.nbytes // _SHM_ALIGN) * _SHM_ALIGN
    fd = os.open(_SHM_DIR / name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, max(size, 1))
    shared = mmap.mmap(fd, max(size, 1))
    os.close(fd)
    for leaf, (shape, dtype, offset) in zip(leaves, layout):
        if leaf.size:
            np.frombuffer(shared, dtype=dtype, count=leaf.size, offset=offset)[...] = leaf.ravel()
    shared.close()
    skeleton = jax.tree_util.tree_unflatten(treedef, range(len(leaves)))
    return name, skeleton, layout


def _receive_batch(handle: tuple) -> jraph.GraphsTuple:
    """Map a shared batch as zero-copy array views; the mapping lives as long as they do."""
    name, skeleton, layout = handle
    fd = os.open(_SHM_DIR / name, os.O_RDWR)
    os.unlink(_SHM_DIR / name)
    shared = mmap.mmap(fd, os.fstat(fd).st_size)
    os.close(fd)
    arrays = [
        np.frombuffer(shared, dtype=dtype, count=int(np.prod(shape)), offset=offset).reshape(shape)
        for shape, dtype, offset in layout
    ]
    return jax.tree_util.tree_map(lambda index: arrays[index], skeleton)


def _discard_batch(handle: tuple) -> None:
    os.unlink(_SHM_DIR / handle[0])


def _dataloader_worker(
    dataset, index_queue, output_queue, abort, n_node, n_edge, n_graph, n_edge_inner
):
    """Batch every shard of indices the parent sends, ending each shard with ``None``."""
    count = 0
    while True:
        indices = index_queue.get()
        if indices is None:
            break
        for batch in _batches(dataset, indices, n_node, n_edge, n_graph, n_edge_inner, abort):
            output_queue.put(_share_batch(batch, f"nequix-{os.getpid()}-{count}"))
            count += 1
        output_queue.put(None)
    # allow exit without flushing queued results, otherwise a mid-iteration
    # shutdown deadlocks the join on unflushed results
    output_queue.cancel_join_thread()


# multiprocess data loader with dynamic batching, based on
# https://teddykoker.com/2020/12/dataloader/
# https://github.com/google-deepmind/jraph/blob/51f5990/jraph/ogb_examples/data_utils.py
# Each worker dynamically batches its own interleaved shard of the epoch's index
# order and hands whole padded batches over through /dev/shm, so the parent only
# round-robins the worker queues and maps the arrays: per-graph work in the
# parent capped MPtrj loading at ~4.8k graphs/s however many workers fed it,
# and pickling the padded batches through pipes at ~8.4k.
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
        inner_edge_fraction: float | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.idxs = np.arange(len(self.dataset))
        self._generator = None  # created in __iter__
        self.n_node = max(batch_size * avg_n_nodes * buffer_factor, max_n_nodes) + 1
        self.n_edge = max(batch_size * avg_n_edges * buffer_factor, max_n_edges)
        self.n_edge_inner = (
            int(round(self.n_edge * inner_edge_fraction))
            if inner_edge_fraction is not None
            else None
        )
        self.n_graph = n_graph if n_graph is not None else batch_size + 1
        self.num_workers = num_workers
        # batches buffered per worker
        self.prefetch_factor = prefetch_factor

        self._started = False
        self.abort = None
        self.index_queues = []
        self.output_queues = []
        self.workers = []
        # workers still batching a shard whose ``None`` end marker we have not read
        self._pending: set[int] = set()

    def _start_workers(self):
        if self._started or self.num_workers == 0:
            return

        # Workers start once training is already iterating, by which point JAX
        # has initialized its multithreaded runtime. Forking that runtime
        # deadlocks the workers (and the parent) on inherited locks, so fork
        # from a clean forkserver instead. The dataset reopens its database per
        # process (see __getstate__), so it pickles cheaply to each worker.
        self._started = True
        ctx = multiprocessing.get_context("forkserver")
        self.abort = ctx.Event()

        for _ in range(self.num_workers):
            index_queue = ctx.Queue()
            output_queue = ctx.Queue(maxsize=self.prefetch_factor)
            worker = ctx.Process(
                target=_dataloader_worker,
                args=(
                    self.dataset,
                    index_queue,
                    output_queue,
                    self.abort,
                    self.n_node,
                    self.n_edge,
                    self.n_graph,
                    self.n_edge_inner,
                ),
            )
            worker.daemon = True
            worker.start()
            self.index_queues.append(index_queue)
            self.output_queues.append(output_queue)
            self.workers.append(worker)

    def _drain_pending(self):
        """Abort and discard the shards of an epoch that was not iterated to its end."""
        if not self._pending:
            return
        self.abort.set()
        for worker in self._pending:
            while (handle := self.output_queues[worker].get()) is not None:
                _discard_batch(handle)
        self._pending = set()
        self.abort.clear()

    def shutdown(self):
        if not self._started:
            return
        self._drain_pending()
        for index_queue in self.index_queues:
            index_queue.put(None)
        for worker in self.workers:
            worker.join()
        for q in (*self.index_queues, *self.output_queues):
            q.close()
        self.index_queues, self.output_queues, self.workers = [], [], []
        self._started = False

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(seed=hash((self.seed, epoch)) % 2**32)

    def make_generator(self):
        if self.num_workers == 0:
            yield from _batches(
                self.dataset, self.idxs, self.n_node, self.n_edge, self.n_graph, self.n_edge_inner
            )
            return

        self._drain_pending()
        for worker, index_queue in enumerate(self.index_queues):
            index_queue.put(self.idxs[worker :: self.num_workers])
        self._pending = set(range(self.num_workers))

        active = list(range(self.num_workers))
        while active:
            for worker in list(active):
                handle = self.output_queues[worker].get()
                if handle is None:
                    active.remove(worker)
                    self._pending.discard(worker)
                else:
                    yield _receive_batch(handle)

    def __iter__(self):
        self._start_workers()
        if self.shuffle:
            self.idxs = self.rng.permutation(np.arange(len(self.dataset)))
        self._generator = self.make_generator()
        return self

    def __next__(self):
        return next(self._generator)


class ParallelLoader:
    """Group consecutive batches into one per-device stacked batch.

    With ``devices`` the group is placed on them directly (``device_put_sharded``),
    so the host-to-device copy happens wherever the loader is iterated (the
    prefetch thread) instead of inside the training step's dispatch, and the
    per-device arrays are never stacked on the host.
    """

    def __init__(self, loader: DataLoader, n: int, devices: list | None = None):
        self.loader = loader
        self.n = n
        self.devices = devices
        if devices is not None and len(devices) != n:
            raise ValueError(f"{len(devices)} devices for {n} batches per step")

    def __iter__(self):
        it = iter(self.loader)
        while True:
            try:
                batches = [next(it) for _ in range(self.n)]
            except StopIteration:
                return
            if self.devices is None:
                yield jax.tree.map(lambda *x: np.stack(x), *batches)
            else:
                yield jax.device_put_sharded(batches, self.devices)


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
