"""Convert a training checkpoint (.pkl) into a serving model (.nqx)."""

import argparse
import json
from pathlib import Path

from nequix.config import RUNS, TrainerConfig, checkpoint_dir
from nequix.config.models import ModelMetadata
from nequix.model import conservative_backbone, save_model
from nequix.train import load_training_state, model_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", choices=sorted(RUNS))
    parser.add_argument(
        "--checkpoint",
        default="best.pkl",
        help="file name inside the run's checkpoint folder, or a path to any .pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to the resolved checkpoint's name with an .nqx suffix",
    )
    args = parser.parse_args()

    config = RUNS[args.run_name]
    path = Path(args.checkpoint)
    if not path.exists():
        path = checkpoint_dir(config) / args.checkpoint
    _, ema_model, *_ = load_training_state(path)

    if isinstance(config, TrainerConfig):
        metadata = model_metadata(config)
    else:
        with open(config.finetune_from, "rb") as f:
            metadata = ModelMetadata.from_header(json.loads(f.readline().decode()))
    model = conservative_backbone(ema_model)

    output = args.output or path.resolve().with_suffix(".nqx")
    save_model(output, model, metadata)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
