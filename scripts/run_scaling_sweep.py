#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from nequix.scaling import (
    ScalingTrial,
    extract_final_summary,
    finalist_replicates,
    primary_trials,
    read_trial_summaries,
    select_finalists,
    write_manifest,
    write_trial_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAL_SCRIPT = Path(__file__).with_name("run_scaling_trial.py")
ANALYSIS_SCRIPT = Path(__file__).with_name("analyze_scaling.py")


def discover_gpus(explicit: str | None) -> list[str]:
    if explicit:
        values = explicit.split(",")
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        values = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        values = result.stdout.splitlines()
    values = [value.strip() for value in values if value.strip()]
    if not values:
        raise RuntimeError("no GPUs were provided or discovered")
    if len(values) != len(set(values)):
        raise ValueError("GPU identifiers must be unique")
    return values


def _status_path(output_dir: Path, trial: ScalingTrial) -> Path:
    return output_dir / "trials" / trial.trial_id / "status.json"


def _summary_path(output_dir: Path, trial: ScalingTrial) -> Path:
    return output_dir / "trials" / trial.trial_id / "summary.csv"


def _trial_command(
    trial: ScalingTrial,
    trial_dir: Path,
    wandb_project: str,
    wandb_mode: str | None,
    kernel: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(TRIAL_SCRIPT),
        "--depth",
        str(trial.depth),
        "--width",
        str(trial.width_multiplier),
        "--seed",
        str(trial.seed),
        "--trial-dir",
        str(trial_dir),
        "--wandb-project",
        wandb_project,
    ]
    if wandb_mode:
        command.extend(("--wandb-mode", wandb_mode))
    if not kernel:
        command.append("--no-kernel")
    return command


def run_trials(
    trials: list[ScalingTrial],
    *,
    output_dir: Path,
    gpu_ids: list[str],
    max_parallel: int,
    wandb_project: str,
    wandb_mode: str | None,
    kernel: bool,
    dry_run: bool,
) -> list[ScalingTrial]:
    pending = [trial for trial in trials if not _summary_path(output_dir, trial).is_file()]
    completed = len(trials) - len(pending)
    if completed:
        print(f"skipping {completed} completed trial(s)")

    if dry_run:
        for index, trial in enumerate(pending):
            gpu = gpu_ids[index % min(max_parallel, len(gpu_ids))]
            trial_dir = output_dir / "trials" / trial.trial_id
            command = _trial_command(trial, trial_dir, wandb_project, wandb_mode, kernel)
            print(f"GPU {gpu}: {' '.join(command)}")
        return []

    available = gpu_ids[:max_parallel]
    active: dict[subprocess.Popen, tuple[ScalingTrial, str, object, float]] = {}
    failed: list[ScalingTrial] = []

    try:
        while pending or active:
            while pending and available:
                trial = pending.pop(0)
                gpu = available.pop(0)
                trial_dir = output_dir / "trials" / trial.trial_id
                trial_dir.mkdir(parents=True, exist_ok=True)
                log_path = trial_dir / "train.log"
                log_file = log_path.open("a")
                log_file.write(f"\n--- launching {trial.trial_id} on GPU {gpu} ---\n")
                log_file.flush()
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                process = subprocess.Popen(
                    _trial_command(trial, trial_dir, wandb_project, wandb_mode, kernel),
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[process] = (trial, gpu, log_file, time.time())
                print(f"started {trial.trial_id} on GPU {gpu} (pid {process.pid})")

            finished = [process for process in active if process.poll() is not None]
            for process in finished:
                trial, gpu, log_file, started_at = active.pop(process)
                log_file.close()
                available.append(gpu)
                log_path = output_dir / "trials" / trial.trial_id / "train.log"
                status = {
                    "trial_id": trial.trial_id,
                    "gpu": gpu,
                    "return_code": process.returncode,
                    "elapsed_seconds": time.time() - started_at,
                }
                if process.returncode == 0:
                    try:
                        summary = extract_final_summary(log_path)
                        write_trial_summary(_summary_path(output_dir, trial), trial, summary)
                        status["status"] = "completed"
                        print(f"completed {trial.trial_id} on GPU {gpu}")
                    except ValueError as error:
                        status.update({"status": "failed", "error": str(error)})
                        failed.append(trial)
                        print(f"failed to collect {trial.trial_id}: {error}", file=sys.stderr)
                else:
                    status["status"] = "failed"
                    failed.append(trial)
                    print(
                        f"trial {trial.trial_id} failed with exit code {process.returncode}; "
                        f"see {log_path}",
                        file=sys.stderr,
                    )
                _status_path(output_dir, trial).write_text(json.dumps(status, indent=2) + "\n")

            if active and not finished:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("interrupt received; terminating active trials", file=sys.stderr)
        for process in active:
            process.terminate()
        for process, (trial, gpu, log_file, started_at) in active.items():
            process.wait()
            log_file.close()
            status = {
                "status": "interrupted",
                "trial_id": trial.trial_id,
                "gpu": gpu,
                "return_code": process.returncode,
                "elapsed_seconds": time.time() - started_at,
            }
            _status_path(output_dir, trial).write_text(json.dumps(status, indent=2) + "\n")
        raise
    return failed


def completed_primary_records(output_dir: Path) -> list[dict[str, str]]:
    primary_ids = {trial.trial_id for trial in primary_trials()}
    return [
        record
        for record in read_trial_summaries(output_dir)
        if record.get("trial_id") in primary_ids
    ]


def selected_primary_trials(output_dir: Path) -> list[ScalingTrial]:
    records = completed_primary_records(output_dir)
    expected = len(primary_trials())
    if len(records) != expected:
        raise RuntimeError(
            f"cannot select finalists: completed {len(records)} of {expected} seed-0 trials"
        )
    selected = select_finalists(records)
    return [
        ScalingTrial(
            depth=int(record["scaling_depth"]),
            width_multiplier=float(record["scaling_width_multiplier"]),
            seed=0,
        )
        for record in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resumable OMat-1M Nequix depth/width scaling study."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("scaling_runs/omat1m-depth-width"))
    parser.add_argument(
        "--gpus", help="Comma-separated physical GPU IDs; defaults to visible GPUs."
    )
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--wandb-project", default="nequix-scaling-omat1m")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="Use portable e3nn-jax operations instead of the OpenEquivariance extension.",
    )
    parser.add_argument("--phase", choices=("all", "grid", "finalists"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = discover_gpus(args.gpus)
    max_parallel = args.max_parallel or len(gpu_ids)
    if max_parallel < 1 or max_parallel > len(gpu_ids):
        parser.error("--max-parallel must be between 1 and the number of selected GPUs")
    kernel = not args.no_kernel
    if kernel and not args.dry_run and importlib.util.find_spec("openequivariance_extjax") is None:
        parser.error(
            "OpenEquivariance kernels are enabled but openequivariance_extjax is missing; "
            "install the oeq extra and extension as documented, or pass --no-kernel"
        )

    grid = primary_trials()
    study_config = {
        "format": "nequix-scaling-study-v1",
        "train_path": "data/omat-1m/train.atp",
        "valid_path": "data/omat-1m/val.atp",
        "depths": sorted({trial.depth for trial in grid}),
        "width_multipliers": sorted({trial.width_multiplier for trial in grid}),
        "epochs": 4,
        "batch_size": 128,
        "force_mode": "conservative",
        "kernel": kernel,
    }
    study_config_path = output_dir / "study_config.json"
    if (
        study_config_path.is_file()
        and json.loads(study_config_path.read_text()) != study_config
        and read_trial_summaries(output_dir)
    ):
        parser.error(f"study settings do not match the completed trials in: {study_config_path}")
    study_config_path.write_text(json.dumps(study_config, indent=2) + "\n")
    write_manifest(output_dir / "grid_manifest.json", grid)
    failures: list[ScalingTrial] = []
    if args.phase in ("all", "grid"):
        failures.extend(
            run_trials(
                grid,
                output_dir=output_dir,
                gpu_ids=gpu_ids,
                max_parallel=max_parallel,
                wandb_project=args.wandb_project,
                wandb_mode=args.wandb_mode,
                kernel=kernel,
                dry_run=args.dry_run,
            )
        )

    if args.dry_run:
        if args.phase == "finalists":
            finalists = selected_primary_trials(output_dir)
            repeats = finalist_replicates(finalists)
            write_manifest(output_dir / "finalists.json", finalists)
            write_manifest(output_dir / "finalist_replicates.json", repeats)
            run_trials(
                repeats,
                output_dir=output_dir,
                gpu_ids=gpu_ids,
                max_parallel=max_parallel,
                wandb_project=args.wandb_project,
                wandb_mode=args.wandb_mode,
                kernel=kernel,
                dry_run=True,
            )
        return

    if args.phase in ("all", "finalists") and not failures:
        finalists = selected_primary_trials(output_dir)
        repeats = finalist_replicates(finalists)
        write_manifest(output_dir / "finalists.json", finalists)
        write_manifest(output_dir / "finalist_replicates.json", repeats)
        failures.extend(
            run_trials(
                repeats,
                output_dir=output_dir,
                gpu_ids=gpu_ids,
                max_parallel=max_parallel,
                wandb_project=args.wandb_project,
                wandb_mode=args.wandb_mode,
                kernel=kernel,
                dry_run=False,
            )
        )

    if read_trial_summaries(output_dir):
        subprocess.run(
            [sys.executable, str(ANALYSIS_SCRIPT), str(output_dir)],
            cwd=REPO_ROOT,
            check=True,
        )
    else:
        print("no completed trials are available to analyze", file=sys.stderr)
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed; rerun the same command to resume")


if __name__ == "__main__":
    main()
