import numpy as np
import pytest
from atompack import Database, Molecule

from nequix.data import AtomPackDataset, Dataset, dataset_from_path
from nequix.train_utils import wandb_run_name


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

    # The extensionless split path should prefer its sibling train.atp file.
    dataset = dataset_from_path(str(path.with_suffix("")), [1, 8], cutoff=1.5, backend="dict")
    graph = dataset[0]

    assert isinstance(dataset, AtomPackDataset)
    np.testing.assert_array_equal(graph["species"], [0, 1])
    np.testing.assert_allclose(graph["energy"], [-1.5])
    assert graph["forces"].shape == (2, 3)
    assert graph["stress"].shape == (3, 3)
    assert graph["n_edge"].item() == 2


def test_wandb_name_includes_fraction_and_schedule():
    config = {
        "dataset_name": "1m",
        "train_frac": 0.25,
        "n_epochs": 4,
        "run_name": "nequix_orig",
    }

    assert wandb_run_name("unused.yml", config) == "1m25_4ep_nequix_orig"
