from __future__ import annotations

from dataclasses import replace

from nequix.config.models import PFTTrainerConfig


_MP_PFT = PFTTrainerConfig(
    name="nequix-mp-1-pft",
    state_path="checkpoints/nequix-mp-1-pft.pkl",
    resume_from="checkpoints/nequix-mp-1-pft.pkl",
    finetune_from="models/nequix-mp-1.nqx",
    train_path="data/pbe-mdr/train-aselmdb",
    val_path="data/pbe-mdr/val-aselmdb",
    extra_train_path="data/mptrj-aselmdb",
    extra_val_frac=0.05,
    avg_n_edges=5418.453349001175,
    avg_n_nodes=104.9225616921269,
    max_n_edges=42600,
    max_n_nodes=306,
)

_MP_PFT_NO_COTRAIN = replace(
    _MP_PFT,
    name="nequix-mp-1-pft-no-cotrain",
    state_path="checkpoints/nequix-mp-1-pft-nocotrain.pkl",
    resume_from="checkpoints/nequix-mp-1-pft-nocotrain.pkl",
    extra_train_steps=0,
    n_epochs=200,
    energy_weight=20.0,
)

_OAM_TRAIN_PATHS = ("data/mptrj-aselmdb",) * 8 + ("data/salex/train",)

_OAM_PFT = replace(
    _MP_PFT,
    name="nequix-oam-1-pft",
    state_path="checkpoints/nequix-oam-1-pft.pkl",
    resume_from="checkpoints/nequix-oam-1-pft.pkl",
    finetune_from="models/nequix-oam-1.nqx",
    extra_train_path=_OAM_TRAIN_PATHS,
    extra_val_frac=None,
    extra_val_path="data/salex/val",
    extra_energy_weight=750.0,
    n_epochs=200,
    val_every=5,
)


RUNS: list[PFTTrainerConfig] = [_MP_PFT, _MP_PFT_NO_COTRAIN, _OAM_PFT]
