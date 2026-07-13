import sys
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
    "nequix-omat-1": "jax",
}


def test_all_training_configs_are_registered():
    assert {name: config.trainer for name, config in RUNS.items()} == EXPECTED_TRAINERS


def test_config_dict_flattens_model_and_sequence_values():
    config = config_dict(RUNS["nequix-oam-1"])

    assert "model_config" not in config
    assert config["cutoff"] == 6.0
    assert config["hidden_irreps"] == "128x0e + 64x1o + 32x2e + 32x3o"
    assert config["train_path"] == ["data/mptrj-aselmdb"] * 8 + ["data/salex/train"]
    assert config["atomic_numbers"][:3] == [1, 2, 3]
    assert config["finetune_from"] == "models/nequix-omat-1.nqx"
    assert config["resume_from"] == "checkpoints/nequix-oam-1-jax.pkl"
    assert all(value is not None for value in config.values())


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
