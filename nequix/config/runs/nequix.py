from __future__ import annotations

from dataclasses import replace

from nequix.config.models import (
    ATOMIC_NUMBERS,
    LongMDEvalConfig,
    MLIPArenaConfig,
    MP_ATOM_ENERGIES,
    OAM_ATOM_ENERGIES,
    OMAT_ATOM_ENERGIES,
    TrainerConfig,
    ValidationConfig,
)


_MP = TrainerConfig(
    name="nequix-mp-1",
    state_path="checkpoints/nequix-mp-1.pkl",
    resume_from="checkpoints/nequix-mp-1.pkl",
    checkpoint_path="checkpoints/nequix-mp-1.nqx",
    batch_size=64,
    train_path="data/mptrj.atp",
    valid_frac=0.05,
    dataset_name="mptrj",
    atomic_numbers=ATOMIC_NUMBERS,
    atom_energies=MP_ATOM_ENERGIES,
    avg_n_edges=1932.8392640079926,
    avg_n_neighbors=57.413687022442645,
    avg_n_nodes=31.196903505120307,
    max_n_edges=34704,
    max_n_nodes=444,
    scale=0.8066479563713074,
    shift=0.16502578765761478,
    n_epochs=100,
)

_TRAINING_VALIDATION = ValidationConfig(
    every_steps=10_000,
    evaluation_every_steps=25_000,
    mlip_arena=MLIPArenaConfig(
        tasks=("diatomics",),
        # Broad chemical coverage while keeping the training interruption
        # under the five-minute evaluation budget.
        elements=("H", "C", "O", "Si", "Cu"),
    ),
    long_md=LongMDEvalConfig(
        dataset="tm23",
        tm23_regimes=("melt",),
        max_systems=1,
    ),
)

_OMAT = replace(
    _MP,
    name="nequix-omat-1",
    state_path="checkpoints/nequix-omat-1-jax.pkl",
    resume_from="checkpoints/nequix-omat-1-jax.pkl",
    checkpoint_path="checkpoints/nequix-omat-1.nqx",
    train_path="data/omat/train.atp",
    valid_frac=None,
    valid_path="data/omat/val.atp",
    dataset_name="omat24",
    atom_energies=OMAT_ATOM_ENERGIES,
    avg_n_edges=736.2363228968411,
    avg_n_neighbors=39.200198903821516,
    avg_n_nodes=18.68197878523378,
    max_n_edges=17940,
    max_n_nodes=236,
    batch_size=128,
    scale=0.8080419656942678,
    shift=-3.513482726416955,
    n_epochs=6,
    validation=_TRAINING_VALIDATION,
)

_OMAT_CURRICULUM_DIRECT = replace(
    _OMAT,
    name="nequix-omat-foundation-direct",
    state_path="checkpoints/nequix-omat-foundation-direct.pkl",
    resume_from="checkpoints/nequix-omat-foundation-direct.pkl",
    checkpoint_path="checkpoints/nequix-omat-foundation-direct.nqx",
    force_mode="direct",
    stress_weight=0.0,
    n_epochs=2,
    batch_size=256,
    validation=replace(_TRAINING_VALIDATION, evaluation_every_steps=2_000),
    model_config=replace(
        _OMAT.model_config,
        hidden_irreps="195x0e + 97x1o + 49x2e + 49x3o",
        lmax=4,
        n_layers=10,
    ),
)

_OMAT_CURRICULUM_CONSERVATIVE = replace(
    _OMAT,
    name="nequix-omat-foundation-conservative",
    state_path="checkpoints/nequix-omat-foundation-conservative.pkl",
    resume_from="checkpoints/nequix-omat-foundation-conservative.pkl",
    finetune_from="checkpoints/nequix-omat-foundation-direct.nqx",
    checkpoint_path="checkpoints/nequix-omat-foundation-conservative.nqx",
    force_mode="conservative",
    n_epochs=2,
    batch_size=256,
    validation=replace(_TRAINING_VALIDATION, evaluation_every_steps=2_000),
    model_config=replace(
        _OMAT.model_config,
        hidden_irreps="195x0e + 97x1o + 49x2e + 49x3o",
        lmax=4,
        n_layers=10,
    ),
)

# Repeat MPtrj so every epoch mixes it 8:1 with sAlex (the eSEN OAM recipe).
OAM_TRAIN_PATHS = ("data/mptrj.atp",) * 8 + ("data/salex/train.atp",)

_OAM = replace(
    _OMAT,
    name="nequix-oam-1",
    state_path="checkpoints/nequix-oam-1-jax.pkl",
    resume_from="checkpoints/nequix-oam-1-jax.pkl",
    finetune_from="checkpoints/nequix-omat-1.nqx",
    checkpoint_path="checkpoints/nequix-oam-1.nqx",
    train_path=OAM_TRAIN_PATHS,
    valid_path="data/salex/val.atp",
    dataset_name="oam",
    atom_energies=OAM_ATOM_ENERGIES,
    shift=-4.3250839528546265,
    learning_rate=0.003,
    warmup_epochs=0.0,
    warmup_factor=0.0,
    n_epochs=3,
    validation=replace(_TRAINING_VALIDATION, every_steps=None),
)


RUNS: list[TrainerConfig] = [
    _MP,
    _OMAT,
    _OMAT_CURRICULUM_DIRECT,
    _OMAT_CURRICULUM_CONSERVATIVE,
    _OAM,
]
