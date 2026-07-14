from __future__ import annotations

from dataclasses import replace

from nequix.config.models import PFTTrainerConfig
from nequix.config.runs.nequix import OAM_TRAIN_PATHS


_MP_PFT = PFTTrainerConfig(
    name="nequix-mp-1-pft",
    state_path="checkpoints/nequix-mp-1-pft.pkl",
    resume_from="checkpoints/nequix-mp-1-pft.pkl",
    finetune_from="checkpoints/nequix-mp-1.nqx",
    train_path="data/pbe-mdr/train.atp",
    val_path="data/pbe-mdr/val.atp",
    extra_train_path="data/mptrj.atp",
    extra_val_frac=0.05,
    avg_n_edges=5418.453349001175,
    avg_n_nodes=104.9225616921269,
    max_n_edges=42600,
    max_n_nodes=306,
    extra_avg_n_edges=1932.8392640079926,
    extra_avg_n_nodes=31.196903505120307,
    extra_max_n_edges=34704,
    extra_max_n_nodes=444,
    extra_batch_size=64,
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

_OAM_PFT = replace(
    _MP_PFT,
    name="nequix-oam-1-pft",
    state_path="checkpoints/nequix-oam-1-pft.pkl",
    resume_from="checkpoints/nequix-oam-1-pft.pkl",
    finetune_from="checkpoints/nequix-oam-1.nqx",
    extra_train_path=OAM_TRAIN_PATHS,
    extra_val_frac=None,
    extra_val_path="data/salex/val.atp",
    extra_avg_n_edges=736.2363228968411,
    extra_avg_n_nodes=18.68197878523378,
    extra_max_n_edges=17940,
    extra_max_n_nodes=236,
    extra_batch_size=128,
    extra_energy_weight=750.0,
    n_epochs=200,
    val_every=5,
)


RUNS: list[PFTTrainerConfig] = [_MP_PFT, _MP_PFT_NO_COTRAIN, _OAM_PFT]
