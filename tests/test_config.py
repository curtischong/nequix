import sys
from dataclasses import replace
from types import ModuleType

import pytest

from nequix.cli import main, run
from nequix.config import PFTTrainerConfig, RUNS, TrainerConfig, config_values


EXPECTED_RUNS = {
    "nequix-mp-1": TrainerConfig,
    "nequix-mp-1-pft": PFTTrainerConfig,
    "nequix-mp-1-pft-no-cotrain": PFTTrainerConfig,
    "nequix-oam-1": TrainerConfig,
    "nequix-oam-1-pft": PFTTrainerConfig,
    "nequix-omat-foundation-conservative": TrainerConfig,
    "nequix-omat-foundation-direct": TrainerConfig,
    "nequix-omat-1": TrainerConfig,
}


def test_all_training_configs_are_registered():
    assert {name: type(config) for name, config in RUNS.items()} == EXPECTED_RUNS


def test_config_values_preserves_typed_config_structure():
    config = config_values(RUNS["nequix-oam-1"])

    assert config["model_config"]["cutoff"] == 6.0
    assert config["model_config"]["hidden_irreps"] == "128x0e + 64x1o + 32x2e + 32x3o"
    assert config["train_path"] == ["data/mptrj.atp"] * 8 + ["data/salex/train.atp"]
    assert config["atomic_numbers"][:3] == [1, 2, 3]
    assert config["finetune_from"] == "checkpoints/nequix-omat-1.nqx"
    assert config["resume_from"] == "checkpoints/nequix-oam-1-jax.pkl"
    assert config["batch_size"] == 128
    assert config["validation"]["every_steps"] is None
    assert config["validation"]["evaluation_every_steps"] == 25_000
    assert config["validation"]["mlip_arena"]["tasks"] == ["diatomics"]
    assert config["validation"]["mlip_arena"]["elements"] == ["H", "C", "O", "Si", "Cu"]
    assert config["validation"]["long_md"]["tm23_regimes"] == ["melt"]
    assert config["validation"]["long_md"]["max_systems"] == 1


def test_omat_foundation_curriculum_configs():
    mp = RUNS["nequix-mp-1"]
    omat = RUNS["nequix-omat-1"]
    oam = RUNS["nequix-oam-1"]
    direct = RUNS["nequix-omat-foundation-direct"]
    conservative = RUNS["nequix-omat-foundation-conservative"]

    assert mp.batch_size == 64
    assert omat.batch_size == oam.batch_size == 128
    assert omat.validation.every_steps == 10_000
    assert direct.batch_size == conservative.batch_size == 256
    assert direct.validation == conservative.validation
    assert direct.validation == replace(omat.validation, evaluation_every_steps=2_000)
    assert omat.validation.evaluation_every_steps == 25_000
    assert oam.validation == replace(omat.validation, every_steps=None)
    assert direct.train_frac == conservative.train_frac == 1.0
    assert direct.n_epochs == conservative.n_epochs == 2
    assert direct.force_mode == "direct"
    assert direct.stress_weight == 0.0
    assert conservative.force_mode == "conservative"
    assert conservative.finetune_from == direct.checkpoint_path
    assert conservative.resume_from != direct.resume_from
    assert direct.model_config == conservative.model_config
    assert direct.model_config.hidden_irreps == "195x0e + 97x1o + 49x2e + 49x3o"
    assert direct.model_config.lmax == 4
    assert direct.model_config.n_layers == 10


@pytest.mark.parametrize("name", EXPECTED_RUNS)
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
