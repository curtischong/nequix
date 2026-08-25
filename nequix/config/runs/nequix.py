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
    ValidationConfig,
)


# Batch size from single-H100 probes (real train step + real loader, GPM
# counters): graphs/s plateaus from 384 (852) through 704 (874) while SM
# activity sits at 87-91%; 512 is within 2% of the peak at 59.6GB of the
# 77.3GB pool. The zigzag variant plateaus at the same point (630 graphs/s,
# 70.3GB, 90% SM active). Four devices at 512 consume ~3400 graphs/s, more
# than 16 loader workers produce (3024/s); 32 workers produce 3786/s.
_MP = TrainerConfig(
    name="nequix-mp-1",
    batch_size=512,
    num_workers=32,
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

# Zigzag counterpart of the official ~700k-parameter MPtrj model at matched
# parameter count (707,578 vs 707,658): the (6, 4, 4) A block tiles to
# [6, 4, 4, 6, 4] over 5 layers, costing 3.5 full-layer equivalents of edge
# compute vs the baseline's 4, and the width is trimmed so the count matches.
_MP_ZIGZAG = replace(
    _MP,
    name="nequix-mp-1-zigzag",
    model_config=replace(
        _MP.model_config,
        # hidden_irreps="104x0e + 52x1o + 26x2e + 26x3o",
        n_layers=6,
        zigzag_radii=(6.0, 4.0, 4.0),
    ),
)

# Best trial from the one-factor-at-a-time sweep in scripts/tune_zigzag_700k.py
# (9 trials, 2 epochs on 5% of MPtrj, seed 0). Learning rate was the only axis
# that moved validation force MAE outside the noise: 0.03 reached 0.1292 eV/A
# vs the base 0.01 at 0.1351 and 0.003 at 0.1455. Depth and width did not pay
# for themselves -- 8 layers bought 0.0014 eV/A for 63% more step time and 79%
# more parameters, and 6 layers and 1.5x width were no better than the base --
# so both stay put and the parameter count remains matched at 707,578.
_MP_ZIGZAG_TUNED = replace(
    _MP_ZIGZAG,
    name="nequix-mp-1-zigzag-tuned",
    learning_rate=0.03,
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

# Stress-head curriculum: the direct stage trains the 0e+2e stress readout on
# OMat's stress labels (the original direct stage set stress_weight=0, leaving
# them unused), then hands the backbone to conservative training as usual.
# Batch sizes from single-GPU synthetic-batch probes at the 0.97 fraction:
# direct 512 peaks at 40.0GB and 592 graphs/s (256: 556, 1024: 599 at 75.8GB);
# conservative keeps 240 (320 gains 1% for 73GB). cuda_async hands the pool to
# the synchronous step-20k benchmark waves, as in the zigzag/2x runs.
_OMAT_DIRECT_STRESS = replace(
    _OMAT_CURRICULUM_DIRECT,
    name="nequix-omat-foundation-direct-stress",
    stress_weight=5.0,
    batch_size=512,
    allocator="cuda_async",
)

_OMAT_CONSERVATIVE_STRESS = replace(
    _OMAT_CURRICULUM_CONSERVATIVE,
    name="nequix-omat-foundation-conservative-stress",
    finetune_from="checkpoints/nequix-omat-foundation-direct-stress/best.pkl",
    allocator="cuda_async",
)


# Zigzag counterpart of the direct foundation run at matched parameter count:
# the (6, 4, 4) A block over 12 layers costs 8 full-layer equivalents of edge
# compute vs the baseline's 10, and the width is trimmed so the count matches
# (4.78M vs 4.82M). Inner edges are ~29% of the 6 A OMat edge list (p99.9
# per-structure ratio 0.65), so the default 0.5 inner edge budget halves the
# eight inner layers' edge slots with headroom against truncation.
_ZIGZAG_MODEL = replace(
    _OMAT_CURRICULUM_DIRECT.model_config,
    hidden_irreps="172x0e + 85x1o + 43x2e + 43x3o",
    lmax=4,
    n_layers=12,
    zigzag_radii=(6.0, 4.0, 4.0),
)

# cuda_async lets the trainer hand its pool back to the synchronous benchmark
# wave's workers (see the 2x stages); under the default BFC pool the workers
# cannot even create CUDA contexts and the step-20k wave kills training.
_OMAT_DIRECT_ZIGZAG = replace(
    _OMAT_CURRICULUM_DIRECT,
    name="nequix-omat-foundation-direct-zigzag",
    allocator="cuda_async",
    model_config=_ZIGZAG_MODEL,
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


# Stage three of the stress-head curriculum, on the esen-lr schedule.
_OAM_FOUNDATION_STRESS = replace(
    _OAM_FOUNDATION_ESEN_LR,
    name="nequix-oam-foundation-stress",
    finetune_from="checkpoints/nequix-omat-foundation-conservative-stress/best.pkl",
    allocator="cuda_async",
)


# Zigzag post-training: conservative fine-tune on the OAM mix straight from the
# direct zigzag checkpoint (skipping the conservative OMat stage, like the 2ep
# 2x run), on the esen-lr schedule.
_OAM_FOUNDATION_ZIGZAG = replace(
    _OAM_FOUNDATION_ESEN_LR,
    name="nequix-oam-foundation-zigzag",
    finetune_from="checkpoints/nequix-omat-foundation-direct-zigzag/best.pkl",
    allocator="cuda_async",
    model_config=_ZIGZAG_MODEL,
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

# The 2x stages train with the cuda_async allocator so the synchronous wave
# gets the whole GPU: right before each wave the trainer trims its mempool
# (release_device_memory_pools), dropping an idle H100 from ~79 GB held to
# ~1.2 GB, and the pool regrows on the next train step. The default BFC pool
# can't do this - it OOMed the wave workers at every fraction above 0.85.
# The in-training MD is cut to 10 ps: the full 100 ps protocol takes ~55
# minutes per trajectory with this model (measured at the smoke test's
# step-200 wave), and a synchronous wave stops training for its whole
# duration. 10 ps keeps the drift trend and takes ~6 minutes; the full
# protocol remains for offline evaluation.
_2X_BENCHMARKS = replace(
    _TRAINING_BENCHMARKS,
    every_steps=10_000,
    long_md=replace(_TRAINING_BENCHMARKS.long_md, steps=2_000),
)
_2X_VALIDATION = ValidationConfig(every_steps=10_000)
_2X_MEM_FRACTION = 0.97
_2X_ALLOCATOR = "cuda_async"

# The pre-training epoch split follows TECE-OAM-RRA: one direct epoch, then
# two conservative epochs. Batch sizes come from single-GPU capacity probes
# under cuda_async targeting ~92% of the 0.97-fraction limit (76.8 GiB on
# H100); peaks fit linearly in batch size (direct 0.42 GiB/graph,
# conservative 1.21, OAM mix 1.99).
_OMAT_FOUNDATION_DIRECT_2X = replace(
    _OMAT_CURRICULUM_DIRECT,
    name="nequix-omat-foundation-direct-2x",
    batch_size=166,
    n_epochs=1,
    mem_fraction=_2X_MEM_FRACTION,
    allocator=_2X_ALLOCATOR,
    validation=_2X_VALIDATION,
    benchmarks=_2X_BENCHMARKS,
    model_config=replace(
        _OMAT_CURRICULUM_DIRECT.model_config, hidden_irreps=_2X_IRREPS, lmax=_2X_LMAX
    ),
)

_OMAT_FOUNDATION_CONSERVATIVE_2X = replace(
    _OMAT_CURRICULUM_CONSERVATIVE,
    name="nequix-omat-foundation-conservative-2x",
    finetune_from="checkpoints/nequix-omat-foundation-direct-2x/best.pkl",
    batch_size=57,
    n_epochs=2,
    mem_fraction=_2X_MEM_FRACTION,
    allocator=_2X_ALLOCATOR,
    validation=_2X_VALIDATION,
    benchmarks=_2X_BENCHMARKS,
    model_config=replace(
        _OMAT_CURRICULUM_CONSERVATIVE.model_config, hidden_irreps=_2X_IRREPS, lmax=_2X_LMAX
    ),
)

# Zigzag-radii variant of the direct 2x OMat stage: the (6, 4, 4) A block
# tiles across the 10 layers, so six of them convolve only edges within 4 A.
# Inner edges are ~29% of the 6 A OMat edge list (p99.9 per-structure ratio
# 0.65), so the default 0.5 inner edge budget halves those layers' edge slots
# with comfortable headroom against truncation.
_OMAT_FOUNDATION_DIRECT_2X_ZIGZAG = replace(
    _OMAT_FOUNDATION_DIRECT_2X,
    name="nequix-omat-foundation-direct-2x-zigzag",
    model_config=replace(
        _OMAT_FOUNDATION_DIRECT_2X.model_config, zigzag_radii=(6.0, 4.0, 4.0)
    ),
)

# Stage three keeps the esen-lr schedule but runs two OAM epochs like
# TECE-OAM-RRA.
_OAM_FOUNDATION_2X = replace(
    _OAM_FOUNDATION_ESEN_LR,
    name="nequix-oam-foundation-2x",
    finetune_from="checkpoints/nequix-omat-foundation-conservative-2x/best.pkl",
    batch_size=35,
    n_epochs=2,
    mem_fraction=_2X_MEM_FRACTION,
    allocator=_2X_ALLOCATOR,
    validation=_2X_VALIDATION,
    benchmarks=_2X_BENCHMARKS,
    model_config=replace(
        _OAM_FOUNDATION_ESEN_LR.model_config, hidden_irreps=_2X_IRREPS, lmax=_2X_LMAX
    ),
)

# Conservative OAM training initialized straight from the direct 2x OMat
# checkpoint: we can't afford the conservative OMat stage, so all conservative
# training happens on the much smaller 8x MPtrj + sAlex mix instead. The
# fine-tuning source is the mistyped 100-epoch run's best checkpoint (~30k
# steps at peak LR from the same direct 2x model); a resume can't fix that
# run's schedule because resume restores the pickled optimizer, so this run
# restarts the step counter and anneals over a fresh two-epoch cosine.
# Validation is the repo-default 20k cadence: each pass is a ~6.3h
# single-device sweep of the full sAlex val set, which at 10k cadence was 37%
# of wall clock.
_OAM_CONSERVATIVE_2EP_2X = replace(
    _OAM_FOUNDATION_2X,
    name="nequix-oam-conservative-2ep-2x",
    finetune_from="checkpoints/nequix-oam-conservative-100ep-2x/best.pkl",
    validation=ValidationConfig(every_steps=20_000),
)

RUNS: list[TrainerConfig] = [
    _MP,
    _MP_ZIGZAG,
    _MP_ZIGZAG_TUNED,
    _OMAT,
    _OMAT_CURRICULUM_DIRECT,
    _OMAT_CURRICULUM_CONSERVATIVE,
    _OMAT_DIRECT_STRESS,
    _OMAT_CONSERVATIVE_STRESS,
    _OAM_FOUNDATION_STRESS,
    _OMAT_DIRECT_ZIGZAG,
    _OAM_FOUNDATION_ZIGZAG,
    _OAM,
    _OAM_FOUNDATION,
    _OAM_FOUNDATION_ESEN_LR,
    _OAM_FOUNDATION_LORA,
    _OAM_FOUNDATION_LORA_R5,
    _OMAT_FOUNDATION_DIRECT_2X,
    _OMAT_FOUNDATION_DIRECT_2X_ZIGZAG,
    _OMAT_FOUNDATION_CONSERVATIVE_2X,
    _OAM_FOUNDATION_2X,
    _OAM_CONSERVATIVE_2EP_2X,
]
