import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nequix.cli import main, run
from nequix.config import RUNS, config_dict


EXPECTED_TRAINERS = {
    "nequix-mp-1": "jax",
    "nequix-mp-1-pft": "pft",
    "nequix-mp-1-pft-no-cotrain": "pft",
    "nequix-oam-1": "jax",
    "nequix-oam-1-pft": "pft",
    "nequix-omat-foundation-conservative": "jax",
    "nequix-omat-foundation-direct": "jax",
    "nequix-omat-1": "jax",
}

DATA_PATH_KEYS = {
    "train_path",
    "valid_path",
    "val_path",
    "extra_train_path",
    "extra_val_path",
}


def test_all_training_configs_are_registered():
    assert {name: config.trainer for name, config in RUNS.items()} == EXPECTED_TRAINERS


def test_config_dict_flattens_model_and_sequence_values():
    config = config_dict(RUNS["nequix-oam-1"])

    assert "model_config" not in config
    assert config["cutoff"] == 6.0
    assert config["hidden_irreps"] == "128x0e + 64x1o + 32x2e + 32x3o"
    assert config["train_path"] == ["data/mptrj.atp"] * 8 + ["data/salex/train.atp"]
    assert config["atomic_numbers"][:3] == [1, 2, 3]
    assert config["finetune_from"] == "models/nequix-omat-1.nqx"
    assert config["resume_from"] == "checkpoints/nequix-oam-1-jax.pkl"
    assert "val_every_steps" not in config
    assert all(value is not None for value in config.values())


def test_omat_foundation_curriculum_configs():
    omat = config_dict(RUNS["nequix-omat-1"])
    direct = config_dict(RUNS["nequix-omat-foundation-direct"])
    conservative = config_dict(RUNS["nequix-omat-foundation-conservative"])

    assert omat["val_every_steps"] == 10_000
    assert direct["val_every_steps"] == conservative["val_every_steps"] == 10_000
    assert direct["train_frac"] == conservative["train_frac"] == 1.0
    assert direct["n_epochs"] == conservative["n_epochs"] == 2
    assert direct["force_mode"] == "direct"
    assert direct["stress_weight"] == 0.0
    assert conservative["force_mode"] == "conservative"
    assert conservative["finetune_from"] == direct["checkpoint_path"]
    assert conservative["resume_from"] != direct["resume_from"]


def test_bundled_model_metadata_uses_atompack_paths():
    model_dir = Path(__file__).parents[1] / "models"
    for model_path in model_dir.glob("*.nqx"):
        with model_path.open("rb") as model_file:
            config = json.loads(model_file.readline())
        for key in DATA_PATH_KEYS & config.keys():
            paths = config[key] if isinstance(config[key], list) else [config[key]]
            assert all(path.endswith(".atp") for path in paths), (model_path, key, paths)


@pytest.mark.parametrize("name", EXPECTED_TRAINERS)
def test_cli_resolves_config_name(monkeypatch, name):
    selected = []
    monkeypatch.setattr("nequix.cli.run", selected.append)

    main([name])

    assert selected == [RUNS[name]]


@pytest.mark.parametrize(
    ("name", "module_name"),
    [
        ("nequix-mp-1", "nequix.train"),
        ("nequix-omat-1", "nequix.train"),
        ("nequix-mp-1-pft", "nequix.pft.train"),
    ],
)
def test_cli_dispatches_to_configured_trainer(monkeypatch, name, module_name):
    calls = []
    trainer_module = ModuleType(module_name)
    trainer_module.train = calls.append
    monkeypatch.setitem(sys.modules, module_name, trainer_module)

    run(RUNS[name])

    assert calls == [RUNS[name]]
