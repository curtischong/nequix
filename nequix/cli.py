from __future__ import annotations

import argparse
from collections.abc import Sequence

from nequix.config import PFTTrainerConfig, RUNS, RunConfig, TrainerConfig


def run(config: RunConfig) -> None:
    """Dispatch a named config to its JAX or PFT trainer."""
    if isinstance(config, TrainerConfig):
        from nequix.train import train
    elif isinstance(config, PFTTrainerConfig):
        from nequix.pft.train import train
    else:  # pragma: no cover - guarded by the config registry.
        raise TypeError(f"unsupported run config: {type(config).__name__}")
    train(config)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Nequix from a named run config.")
    parser.add_argument(
        "run",
        choices=sorted(RUNS),
        help="Named training config defined in nequix.config.runs.",
    )
    args = parser.parse_args(argv)
    run(RUNS[args.run])


if __name__ == "__main__":
    main()
