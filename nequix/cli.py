from __future__ import annotations

import argparse
from collections.abc import Sequence

from nequix.config import RUNS, RunConfig


def run(config: RunConfig) -> None:
    """Dispatch a named config to its JAX, Torch, or PFT trainer."""
    if config.trainer == "jax":
        from nequix.train import train
    elif config.trainer == "torch":
        from nequix.torch_impl.train import train
    elif config.trainer == "pft":
        from nequix.pft.train import train
    else:  # pragma: no cover - guarded by the config type and registry tests.
        raise ValueError(f"unsupported trainer: {config.trainer}")
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
