from __future__ import annotations

from dataclasses import replace

from nequix.config.models import (
    ATOMIC_NUMBERS,
    BenchmarkConfig,
    LongMDEvalConfig,
    MLIPArenaConfig,
    MP_ATOM_ENERGIES,
    OAM_ATOM_ENERGIES,
    OMAT_ATOM_ENERGIES,
    TrainerConfig,
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

_TRAINING_BENCHMARKS = BenchmarkConfig(
    mlip_arena=MLIPArenaConfig(
        tasks=("diatomics",),
        # Every element the model supports; the curves are fanned out across
        # all local GPUs in pinned worker processes.
        elements=None,
    ),
    long_md=LongMDEvalConfig(
        dataset="tm23",
        tm23_regimes=("melt",),
        # One 100 ps trajectory per GPU on an 8-GPU node keeps the whole
        # evaluation wave near five minutes; a second trajectory per GPU
        # (max_systems=16) raises it to about 7.5 minutes.
        max_systems=8,
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
    benchmarks=_TRAINING_BENCHMARKS,
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
    model_config=replace(
        _OMAT.model_config,
        hidden_irreps="195x0e + 97x1o + 49x2e + 49x3o",
        lmax=4,
        n_layers=10,
    ),
)

# The eSEN OAM fine-tuning mix: sAlex plus eight copies of MPtrj, sampled
# uniformly over the concatenation. OMat24 is deliberately absent — its VASP 54
# PBE(+U) settings are incompatible with the MP-compatible energies of MPtrj,
# sAlex, and the WBM test set, so it only enters through pre-training.
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
)

# Stage three of the foundation curriculum: one conservative epoch on the OAM
# mix so the energy reference matches the MP DFT settings that Matbench
# Discovery evaluates against.
_OAM_FOUNDATION = replace(
    _OMAT_CURRICULUM_CONSERVATIVE,
    name="nequix-oam-foundation",
    state_path="checkpoints/nequix-oam-foundation.pkl",
    resume_from="checkpoints/nequix-oam-foundation.pkl",
    finetune_from="checkpoints/nequix-omat-foundation-conservative.nqx",
    checkpoint_path="checkpoints/nequix-oam-foundation.nqx",
    train_path=OAM_TRAIN_PATHS,
    valid_path="data/salex/val.atp",
    dataset_name="oam",
    atom_energies=OAM_ATOM_ENERGIES,
    # Mix-size-weighted blend of the MPtrj and OMat stats (8x1.58M MPtrj vs
    # ~10.4M sAlex); replace with exact values from
    # scripts/compute_dataset_stats.py data/mptrj.atp:8 data/salex/train.atp
    # --atom-energies oam once data/salex/*.atp exist.
    avg_n_edges=1394.4,
    avg_n_neighbors=49.2,
    avg_n_nodes=25.6,
    max_n_edges=34704,
    max_n_nodes=444,
    shift=-4.3250839528546265,
    # Halve the OMat-stage batch size: the mix nearly doubles avg_n_edges, so
    # this keeps the dynamic-batch edge budget (and memory) roughly constant.
    batch_size=128,
    learning_rate=0.003,
    warmup_epochs=0.0,
    warmup_factor=0.0,
    n_epochs=1,
)


RUNS: list[TrainerConfig] = [
    _MP,
    _OMAT,
    _OMAT_CURRICULUM_DIRECT,
    _OMAT_CURRICULUM_CONSERVATIVE,
    _OAM,
    _OAM_FOUNDATION,
]
