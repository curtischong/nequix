#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from nequix.scaling import ScalingTrial, make_trial_config
from nequix.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated OMat-1M scaling trial.")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--wandb-project", default="nequix-scaling-omat1m")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--no-kernel", action="store_true")
    args = parser.parse_args()

    trial = ScalingTrial(args.depth, args.width, args.seed)
    args.trial_dir.mkdir(parents=True, exist_ok=True)
    config = make_trial_config(
        trial,
        args.trial_dir,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        kernel=not args.no_kernel,
    )
    for dataset_path in (Path(config.train_path), Path(config.valid_path)):
        if not dataset_path.is_file():
            raise FileNotFoundError(f"required scaling dataset does not exist: {dataset_path}")

    print(
        f"scaling trial {trial.trial_id}: depth={trial.depth}, "
        f"width={trial.width_multiplier:g}, seed={trial.seed}, "
        f"hidden_irreps={trial.hidden_irreps}"
    )
    train(config)


if __name__ == "__main__":
    main()
