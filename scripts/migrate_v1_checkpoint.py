"""Migrate a legacy v1 training checkpoint into the per-run folder layout.

v1 checkpoints are single flat .pkl files whose arrays carry the pmap
device-replica axis. This rewrites one as an unreplicated v2 checkpoint at
<checkpoint-root>/<run-name>/step_<step>.pkl and points latest.pkl and
best.pkl at it. Run with JAX_PLATFORMS=cpu if the GPUs are busy training.
"""

import argparse
from pathlib import Path

import cloudpickle
import jax

from nequix.config import RUNS, TrainerConfig, checkpoint_dir
from nequix.train import TRAINING_STATE_FORMAT

LEGACY_FORMAT = "nequix-training-state-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", choices=sorted(RUNS))
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help="legacy checkpoint, defaults to checkpoints/<run-name>.pkl",
    )
    args = parser.parse_args()

    config = RUNS[args.run_name]
    source = args.pkl or Path(f"checkpoints/{args.run_name}.pkl")
    state = cloudpickle.loads(source.read_bytes())
    if state["format"] != LEGACY_FORMAT:
        raise ValueError(f"expected a {LEGACY_FORMAT} checkpoint, got {state['format']!r}")

    state["format"] = TRAINING_STATE_FORMAT
    if isinstance(config, TrainerConfig):
        # the main trainer pickled its state with the pmap device-replica axis
        for key in ("model", "ema_model", "opt_state"):
            state[key] = jax.tree.map(lambda x: x[0], state[key])

    run_dir = checkpoint_dir(config)
    if (run_dir / "latest.pkl").exists():
        raise FileExistsError(f"{run_dir} already has new-format checkpoints")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"step_{int(state['step']):09d}.pkl"
    path.write_bytes(cloudpickle.dumps(state))
    for link in ("latest.pkl", "best.pkl"):
        (run_dir / link).symlink_to(path.name)
    print(f"wrote {path}; latest.pkl and best.pkl point at it")
    print("note: best.pkl is the final state, not the best-validation state — v1 kept only one")


if __name__ == "__main__":
    main()
