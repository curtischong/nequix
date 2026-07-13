import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import write
from atompack import Database, Molecule

from nequix.data import AtomPackDataset, Dataset, dataset_from_path
from nequix.pft.data import PhononDataset
from nequix.train_utils import wandb_run_name
from scripts.preprocess_ase_db import preprocess as preprocess_ase_db
from scripts.preprocess_data import preprocess


class _RangeDataset(Dataset):
    def __init__(self, size):
        super().__init__(backend="dict")
        self.size = size

    def __len__(self):
        return self.size

    def _get_graph_dict(self, idx):
        return {"idx": idx}


def test_dataset_fraction_is_deterministic():
    dataset = _RangeDataset(20)

    first = dataset.subset(0.25, seed=7)
    second = dataset.subset(0.25, seed=7)

    assert len(first) == 5
    np.testing.assert_array_equal(first.indices, second.indices)
    with pytest.raises(ValueError, match="fraction must be in"):
        dataset.subset(0.0)


def test_atompack_dataset_builds_graph(tmp_path):
    path = tmp_path / "train.atp"
    database = Database(str(path), overwrite=True)
    database.add_molecule(
        Molecule.from_arrays(
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            atomic_numbers=np.array([1, 8], dtype=np.uint8),
            energy=-1.5,
            forces=np.zeros((2, 3), dtype=np.float32),
            cell=np.eye(3, dtype=np.float64) * 5.0,
            stress=np.eye(3, dtype=np.float32),
            pbc=(True, True, True),
        )
    )
    database.flush()

    dataset = dataset_from_path(str(path), [1, 8], cutoff=1.5, backend="dict")
    graph = dataset[0]

    assert isinstance(dataset, AtomPackDataset)
    np.testing.assert_array_equal(graph["species"], [0, 1])
    np.testing.assert_allclose(graph["energy"], [-1.5])
    assert graph["forces"].shape == (2, 3)
    assert graph["stress"].shape == (3, 3)
    assert graph["n_edge"].item() == 2


def test_extxyz_preprocessor_writes_atompack(tmp_path):
    input_path = tmp_path / "input.extxyz"
    output_path = tmp_path / "output.atp"
    atoms = Atoms("HO", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-2.5,
        forces=np.zeros((2, 3), dtype=np.float32),
    )
    write(input_path, atoms)

    preprocess(input_path, output_path, n_workers=1)

    database = Database.open(str(output_path))
    molecule = database.get_molecule(0)
    assert len(database) == 1
    assert molecule.energy == pytest.approx(-2.5)
    np.testing.assert_allclose(molecule.forces, 0.0)


def test_extxyz_preprocessor_discards_shape_varying_magmoms(tmp_path):
    input_path = tmp_path / "magnetic.extxyz"
    output_path = tmp_path / "output.atp"
    frames = []
    for symbols in ("H", "HH"):
        atoms = Atoms(symbols, positions=np.zeros((len(symbols), 3)))
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-1.0,
            forces=np.zeros((len(symbols), 3)),
            magmoms=np.zeros(len(symbols)),
        )
        frames.append(atoms)
    write(input_path, frames)

    preprocess(input_path, output_path, n_workers=1)

    database = Database.open(str(output_path))
    assert len(database) == 2
    assert not database.get_molecule(0).has_property("magmoms")
    assert not database.get_molecule(1).has_property("magmoms")


def test_ase_db_preprocessor_writes_atompack(tmp_path):
    input_path = tmp_path / "input.aselmdb"
    output_path = tmp_path / "output.atp"
    atoms = Atoms("HO", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-2.5,
        forces=np.zeros((2, 3), dtype=np.float32),
        stress=np.zeros(6, dtype=np.float32),
    )
    with connect(str(input_path), use_lock_file=False) as database:
        database.write(atoms)

    preprocess_ase_db(str(input_path), str(output_path), batch_size=1)

    database = Database.open(str(output_path))
    molecule = database.get_molecule(0)
    assert len(database) == 1
    assert molecule.energy == pytest.approx(-2.5)
    np.testing.assert_allclose(molecule.forces, 0.0)
    np.testing.assert_allclose(molecule.stress, 0.0)


def test_ase_db_preprocessor_merges_parallel_parts(tmp_path):
    input_path = tmp_path / "input"
    input_path.mkdir()
    output_path = tmp_path / "output.atp"
    for index, symbols in enumerate(("H", "HO")):
        atoms = Atoms(symbols, positions=np.zeros((len(symbols), 3)))
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-float(index),
            forces=np.zeros((len(symbols), 3)),
        )
        with connect(str(input_path / f"part-{index}.aselmdb"), use_lock_file=False) as database:
            database.write(atoms)

    preprocess_ase_db(str(input_path), str(output_path), batch_size=1, n_workers=2)

    database = Database.open(str(output_path))
    assert len(database) == 2
    assert database.get_molecule(0).energy == pytest.approx(0.0)
    assert database.get_molecule(1).energy == pytest.approx(-1.0)


def test_phonon_dataset_reads_hessian_from_atompack(tmp_path):
    path = tmp_path / "phonons.atp"
    hessian = np.arange(36, dtype=np.float32).reshape(6, 6)
    molecule = Molecule.from_arrays(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        atomic_numbers=np.array([1, 8], dtype=np.uint8),
        energy=-1.5,
        forces=np.zeros((2, 3), dtype=np.float32),
        cell=np.eye(3, dtype=np.float64) * 5.0,
        pbc=(True, True, True),
    )
    molecule.set_property("hessian", hessian)
    database = Database(str(path), overwrite=True)
    database.add_molecule(molecule)
    database.flush()

    dataset = PhononDataset(str(path), [1, 8], cutoff=1.5, random_col=False)
    graph = dataset[0]

    np.testing.assert_array_equal(graph.nodes["vs"], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(graph.nodes["hessian_col"], hessian[:, 0].reshape(2, 3))


def test_wandb_name_includes_fraction_and_schedule():
    config = {
        "dataset_name": "1m",
        "train_frac": 0.25,
        "n_epochs": 4,
        "run_name": "nequix_orig",
    }

    assert wandb_run_name("unused", config) == "1m25_4ep_nequix_orig"
