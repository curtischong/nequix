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
    finetune_from="checkpoints/nequix-omat-foundation-direct/best.pkl",
    force_mode="conservative",
    n_epochs=2,
    # conservative forces + stress differentiate through the network, so the
    # direct stage's 256 OOMs; 240 peaks at 57.4GB of the 63.8GB pool on H100
    batch_size=240,
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

# Stats for the 8x MPtrj + sAlex mix from
# scripts/compute_dataset_stats.py data/mptrj.atp:8 data/salex/train.atp
# --atom-energies oam --cutoff 6.0 --sample-frac 0.005; the script's
# default cutoff is 5.0, not the model's 6.0. The cutoff-independent
# shift/scale/node stats come from the full-dataset run. Sampled runs
# underestimate max_n_*; MPtrj is in the mix, so its full-dataset max at
# this cutoff is a floor.
_OAM_MIX_STATS = dict(
    avg_n_edges=1264.9728203440332,
    avg_n_neighbors=51.27093835667581,
    avg_n_nodes=21.74594045063158,
    max_n_edges=34704,
    max_n_nodes=444,
    shift=-4.089559490159454,
    scale=0.7653674612006598,
)

_OAM = replace(
    _OMAT,
    name="nequix-oam-1",
    finetune_from="checkpoints/nequix-omat-1/best.pkl",
    train_path=OAM_TRAIN_PATHS,
    valid_path="data/salex/val.atp",
    dataset_name="oam",
    atom_energies=OAM_ATOM_ENERGIES,
    **_OAM_MIX_STATS,
    # 74 x 1265 avg edges keeps the OMat stage's edge budget (128 x 736) and
    # per-step memory.
    batch_size=74,
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
    finetune_from="checkpoints/nequix-omat-foundation-conservative/best.pkl",
    train_path=OAM_TRAIN_PATHS,
    valid_path="data/salex/val.atp",
    dataset_name="oam",
    atom_energies=OAM_ATOM_ENERGIES,
    **_OAM_MIX_STATS,
    # 140 x 1265 avg edges matches the conservative stage's probe-confirmed
    # edge budget (240 x 736), keeping per-step memory at its 57.4GB peak.
    batch_size=140,
    learning_rate=0.003,
    warmup_epochs=0.0,
    warmup_factor=0.0,
    n_epochs=1,
)


# Stage three rerun matching the eSEN OAM fine-tuning schedule (Table 6 of
# arXiv:2502.12147): peak LR 2e-4 — half their OMat pre-train LR — with a
# 0.1-epoch warmup at factor 0.2, vs the 3e-3 no-warmup first attempt.
_OAM_FOUNDATION_ESEN_LR = replace(
    _OAM_FOUNDATION,
    name="nequix-oam-foundation-esen-lr",
    learning_rate=2e-4,
    warmup_epochs=0.1,
    warmup_factor=0.2,
)


# LoRA variant of stage three: the conservative OMat foundation stays frozen and
# only rank-64 adapters on every linear layer (plus the layer-norm scales) train
# on the OAM mix. AdamW replaces muon because Newton-Schulz orthogonalization
# rescales the zero-initialized adapters' tiny early gradients too aggressively;
# LoRA's decoupled adapters conventionally take ~5x the full-fine-tune LR.
_OAM_FOUNDATION_LORA = replace(
    _OAM_FOUNDATION_ESEN_LR,
    name="nequix-oam-foundation-lora",
    lora_rank=64,
    lora_alpha=128.0,
    optimizer="adamw",
    learning_rate=1e-3,
    weight_decay=0.0,
)


# Low-parameter LoRA: rank 5 is 307k adapter parameters (~8% of the rank-64
# variant), the closest rank to a 300k budget.
_OAM_FOUNDATION_LORA_R5 = replace(
    _OAM_FOUNDATION_LORA,
    name="nequix-oam-foundation-lora-r5",
    lora_rank=5,
    lora_alpha=10.0,
)


# Doubled-width curriculum: the original Nequix irreps (128/64/32/32) times
# two with l=1..3 doubled again to 256/128/128, extended to l=6 with 64
# channels per extra degree, and spherical harmonics raised to lmax=6.
_2X_IRREPS = "256x0e + 256x1o + 128x2e + 128x3o + 64x4e + 64x5o + 64x6e"
_2X_LMAX = 6

# The pre-training epoch split follows TECE-OAM-RRA: one direct epoch, then
# two conservative epochs.
_OMAT_FOUNDATION_DIRECT_2X = replace(
    _OMAT_CURRICULUM_DIRECT,
    name="nequix-omat-foundation-direct-2x",
    batch_size=65,
    n_epochs=1,
    model_config=replace(
        _OMAT_CURRICULUM_DIRECT.model_config, hidden_irreps=_2X_IRREPS, lmax=_2X_LMAX
    ),
)

_OMAT_FOUNDATION_CONSERVATIVE_2X = replace(
    _OMAT_CURRICULUM_CONSERVATIVE,
    name="nequix-omat-foundation-conservative-2x",
    finetune_from="checkpoints/nequix-omat-foundation-direct-2x/best.pkl",
    batch_size=61,
    n_epochs=2,
    model_config=replace(
        _OMAT_CURRICULUM_CONSERVATIVE.model_config, hidden_irreps=_2X_IRREPS, lmax=_2X_LMAX
    ),
)

# Stage three keeps the esen-lr schedule but runs two OAM epochs like
# TECE-OAM-RRA; 35 x 1265 avg edges matches the conservative stage's
# 61 x 736 edge budget.
_OAM_FOUNDATION_2X = replace(
    _OAM_FOUNDATION_ESEN_LR,
    name="nequix-oam-foundation-2x",
    finetune_from="checkpoints/nequix-omat-foundation-conservative-2x/best.pkl",
    batch_size=35,
    n_epochs=2,
    model_config=replace(
        _OAM_FOUNDATION_ESEN_LR.model_config, hidden_irreps=_2X_IRREPS, lmax=_2X_LMAX
    ),
)

RUNS: list[TrainerConfig] = [
    _MP,
    _OMAT,
    _OMAT_CURRICULUM_DIRECT,
    _OMAT_CURRICULUM_CONSERVATIVE,
    _OAM,
    _OAM_FOUNDATION,
    _OAM_FOUNDATION_ESEN_LR,
    _OAM_FOUNDATION_LORA,
    _OAM_FOUNDATION_LORA_R5,
    _OMAT_FOUNDATION_DIRECT_2X,
    _OMAT_FOUNDATION_CONSERVATIVE_2X,
    _OAM_FOUNDATION_2X,
]
