from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from nequix.config import PFTTrainerConfig, RUNS, RunConfig, TrainerConfig


def run(config: RunConfig) -> None:
    """Dispatch a named config to its JAX or PFT trainer."""
    if isinstance(config, TrainerConfig):
        # JAX reads these when the backend initializes on first device use,
        # inside train(); values already in the environment win.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.mem_fraction))
        if config.allocator is not None:
            os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", config.allocator)
        # Relaunches and resumes reload identical programs from disk instead
        # of repaying the multi-minute XLA compile.
        os.environ.setdefault(
            "JAX_COMPILATION_CACHE_DIR", str(Path("evaluations/jax_cache").absolute())
        )
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
