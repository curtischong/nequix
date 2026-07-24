"""Time downstream evaluations with fresh, randomly initialized Nequix weights."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from ase.build import bulk
from ase.data import chemical_symbols

from nequix.calculator import NequixCalculator
from nequix.config import (
    BenchmarkConfig,
    LongMDEvalConfig,
    MLIPArenaConfig,
    RUNS,
    TrainerConfig,
)
from nequix.evaluation import (
    run_long_md_evaluation,
    run_mlip_arena_evaluation,
)
from nequix.model import conservative_backbone, save_model
from nequix.train import build_model, model_metadata


def fresh_checkpoint(config: TrainerConfig, path: Path, kernel: bool) -> None:
    config = replace(config, kernel=kernel, force_mode="conservative")
    save_model(path, conservative_backbone(build_model(config)), model_metadata(config))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time MLIP Arena and eSEN long-MD evaluations with fresh weights."
    )
    parser.add_argument("--run-config", default="nequix-oam-1", choices=sorted(RUNS))
    parser.add_argument(
        "--evaluations",
        nargs="+",
        choices=("mlip-arena", "long-md"),
        default=("mlip-arena", "long-md"),
    )
    parser.add_argument(
        "--arena-tasks",
        nargs="+",
        choices=("diatomics", "eos_bulk", "ev"),
        help="Override the tasks in the selected run's training evaluation config.",
    )
    parser.add_argument("--dataset", choices=("tm23", "md22"), default="tm23")
    parser.add_argument("--dataset-root", default="data/md")
    parser.add_argument("--output-dir", default=".benchmarks/evaluation-timing")
    parser.add_argument("--md-steps", type=int)
    parser.add_argument(
        "--real-md-data",
        action="store_true",
        help="Use the configured TM23/MD22 subset instead of one generated Cu system.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use H-only Arena and 20 MD steps to verify the path quickly.",
    )
    parser.add_argument("--no-kernel", action="store_true")
    args = parser.parse_args()

    config = RUNS[args.run_config]
    if not isinstance(config, TrainerConfig):
        parser.error("--run-config must select a standard (non-PFT) training config")
    output_dir = Path(args.output_dir) / f"run-{time.time_ns()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "fresh.nqx"
    kernel = not args.no_kernel
    fresh_checkpoint(config, checkpoint, kernel)
    calculator_kwargs = {"model_path": checkpoint, "use_kernel": kernel}
    training_benchmarks = config.benchmarks
    if training_benchmarks.mlip_arena is None and training_benchmarks.long_md is None:
        training_benchmarks = BenchmarkConfig(
            mlip_arena=MLIPArenaConfig(tasks=("diatomics",)),
            long_md=LongMDEvalConfig(max_systems=1),
        )
    timings = {}
    metrics = {}

    if "mlip-arena" in args.evaluations:
        configured_arena = training_benchmarks.mlip_arena or MLIPArenaConfig(tasks=("diatomics",))
        elements = (
            ("H",)
            if args.smoke
            else configured_arena.elements
            or tuple(chemical_symbols[number] for number in config.atomic_numbers)
        )
        arena_config = replace(
            configured_arena,
            tasks=tuple(args.arena_tasks) if args.arena_tasks else configured_arena.tasks,
            output_dir=str(output_dir / "mlip_arena"),
            elements=elements,
        )
        started = time.perf_counter()
        arena_metrics = run_mlip_arena_evaluation(
            arena_config,
            calculator_kwargs=calculator_kwargs,
        )
        timings["mlip_arena_seconds"] = time.perf_counter() - started
        metrics["mlip_arena"] = arena_metrics

    if "long-md" in args.evaluations:
        md_steps = args.md_steps if args.md_steps is not None else (20 if args.smoke else None)
        configured_md = training_benchmarks.long_md or LongMDEvalConfig(max_systems=1)
        md_config = replace(
            configured_md,
            dataset=args.dataset,
            dataset_root=args.dataset_root,
            output_dir=str(output_dir / "long_md"),
            steps=md_steps,
            relaxation_steps=2 if args.smoke else 1000,
        )
        systems = None
        if not args.real_md_data:
            atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
            del atoms[0]  # Match TM23's single-vacancy character.
            systems = [("dummy-Cu-melt", atoms, 1358.0 * 1.25)]
        calculator = NequixCalculator(checkpoint, use_kernel=kernel)
        started = time.perf_counter()
        md_metrics = run_long_md_evaluation(md_config, calculator, systems=systems)
        timings["long_md_seconds"] = time.perf_counter() - started
        metrics["long_md"] = md_metrics

        if not args.real_md_data and not args.smoke:
            suite_size = configured_md.max_systems
            if suite_size is not None:
                timings["long_md_configured_suite_estimate_seconds"] = (
                    timings["long_md_seconds"] * suite_size
                )

    report = {
        "run_config": args.run_config,
        "fresh_checkpoint": str(checkpoint),
        "timings": timings,
        "metrics": metrics,
    }
    (output_dir / "timing.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
