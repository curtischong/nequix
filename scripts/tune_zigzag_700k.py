#!/usr/bin/env python3
"""One-factor-at-a-time hyperparameter tuning for the 700k-parameter zigzag model.

Each trial varies exactly one of learning rate, layer count, or embedding width
from the nequix-mp-1-zigzag base, trains on a small MPtrj fraction, and reports
the trainer's final CSV summary. The sweep is resumable (completed trials are
skipped) and finishes by writing trials.csv plus accuracy/time plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from nequix.config import RUNS, TrainerConfig, ValidationConfig
from nequix.scaling import extract_final_summary

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_RUN = "nequix-mp-1-zigzag"
LEARNING_RATES = (0.003, 0.01, 0.03)
N_LAYERS = (4, 5, 6, 8)
WIDTH_MULTIPLIERS = (0.5, 0.75, 1.0, 1.5)

SWEEP_AXES = (
    ("tuning_learning_rate", "Learning rate", "log"),
    ("tuning_n_layers", "Interaction layers", "linear"),
    ("tuning_width_multiplier", "Embedding width multiplier", "linear"),
)


@dataclass(frozen=True)
class Trial:
    learning_rate: float
    n_layers: int
    width_multiplier: float

    @property
    def trial_id(self) -> str:
        label = f"lr{self.learning_rate:g}-l{self.n_layers}-w{self.width_multiplier:g}"
        return label.replace(".", "p")


def base_trial() -> Trial:
    base = RUNS[BASE_RUN]
    return Trial(
        learning_rate=base.learning_rate,
        n_layers=base.model_config.n_layers,
        width_multiplier=1.0,
    )


def sweep_trials() -> list[Trial]:
    base = base_trial()
    trials = [base]
    trials += [replace(base, learning_rate=lr) for lr in LEARNING_RATES]
    trials += [replace(base, n_layers=layers) for layers in N_LAYERS]
    trials += [replace(base, width_multiplier=width) for width in WIDTH_MULTIPLIERS]
    return list(dict.fromkeys(trials))


def scaled_irreps(width_multiplier: float) -> str:
    terms = []
    for term in RUNS[BASE_RUN].model_config.hidden_irreps.split("+"):
        count, irrep = term.strip().split("x")
        terms.append(f"{max(1, round(int(count) * width_multiplier))}x{irrep}")
    return " + ".join(terms)


def trial_config(trial: Trial, args: argparse.Namespace) -> TrainerConfig:
    base = RUNS[BASE_RUN]
    name = f"tune-{trial.trial_id}"
    return replace(
        base,
        name=name,
        seed=args.seed,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        n_epochs=args.epochs,
        learning_rate=trial.learning_rate,
        checkpoint_root=str(args.output_dir / "trials" / trial.trial_id),
        validation=ValidationConfig(every_steps=None),
        kernel=not args.no_kernel,
        model_config=replace(
            base.model_config,
            n_layers=trial.n_layers,
            hidden_irreps=scaled_irreps(trial.width_multiplier),
        ),
        run_name=name,
        wandb_run_name=name,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )


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
    return values


def _summary_path(output_dir: Path, trial: Trial) -> Path:
    return output_dir / "trials" / trial.trial_id / "summary.csv"


def write_trial_summary(output_dir: Path, trial: Trial, summary: dict[str, str]) -> None:
    record = {
        "tuning_trial_id": trial.trial_id,
        "tuning_learning_rate": trial.learning_rate,
        "tuning_n_layers": trial.n_layers,
        "tuning_width_multiplier": trial.width_multiplier,
        **summary,
    }
    path = _summary_path(output_dir, trial)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=record)
        writer.writeheader()
        writer.writerow(record)


def _child_command(trial: Trial, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--trial",
        str(trial.learning_rate),
        str(trial.n_layers),
        str(trial.width_multiplier),
        "--output-dir",
        str(args.output_dir),
        "--train-frac",
        str(args.train_frac),
        "--valid-frac",
        str(args.valid_frac),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
    ]
    if args.no_kernel:
        command.append("--no-kernel")
    return command


def run_sweep(trials: list[Trial], args: argparse.Namespace) -> list[Trial]:
    """Run pending trials one-per-GPU in parallel; return the ones that failed."""
    pending = [trial for trial in trials if not _summary_path(args.output_dir, trial).is_file()]
    if len(pending) < len(trials):
        print(f"skipping {len(trials) - len(pending)} completed trial(s)")

    available = discover_gpus(args.gpus)
    active: dict[subprocess.Popen, tuple[Trial, str, object, float]] = {}
    failed: list[Trial] = []
    while pending or active:
        while pending and available:
            trial = pending.pop(0)
            gpu = available.pop(0)
            trial_dir = args.output_dir / "trials" / trial.trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            log_file = (trial_dir / "train.log").open("a")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                _child_command(trial, args),
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
            trial_dir = args.output_dir / "trials" / trial.trial_id
            status = {
                "trial_id": trial.trial_id,
                "gpu": gpu,
                "return_code": process.returncode,
                "elapsed_seconds": time.time() - started_at,
            }
            if process.returncode == 0:
                summary = extract_final_summary(trial_dir / "train.log")
                write_trial_summary(args.output_dir, trial, summary)
                status["status"] = "completed"
                print(f"completed {trial.trial_id} on GPU {gpu}")
            else:
                status["status"] = "failed"
                failed.append(trial)
                print(
                    f"trial {trial.trial_id} failed with exit code {process.returncode}; "
                    f"see {trial_dir / 'train.log'}",
                    file=sys.stderr,
                )
            (trial_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")

        if active and not finished:
            time.sleep(0.5)
    return failed


NUMERIC_COLUMNS = (
    "tuning_learning_rate",
    "tuning_n_layers",
    "tuning_width_multiplier",
    "parameter_count",
    "final_val_force_mae",
    "final_val_energy_mae_per_atom",
    "training_runtime_seconds",
    "compute_cost_accelerator_hours",
    "peak_accelerator_memory_bytes",
)


def analyze(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    records: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("trials/*/summary.csv")):
        with path.open(newline="") as source:
            records.extend(csv.DictReader(source))
    if not records:
        raise ValueError(f"no completed trial summaries found under {output_dir}")

    frame = pd.DataFrame(records)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column])
    frame = frame.sort_values("tuning_trial_id").reset_index(drop=True)
    frame.to_csv(output_dir / "trials.csv", index=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    base = base_trial()
    base_values = {
        "tuning_learning_rate": base.learning_rate,
        "tuning_n_layers": base.n_layers,
        "tuning_width_multiplier": base.width_multiplier,
    }

    for column, label, x_scale in SWEEP_AXES:
        fixed = [axis for axis, _, _ in SWEEP_AXES if axis != column]
        series = frame[
            (frame[fixed[0]] == base_values[fixed[0]]) & (frame[fixed[1]] == base_values[fixed[1]])
        ].sort_values(column)
        figure, mae_axis = plt.subplots(figsize=(7, 5))
        time_axis = mae_axis.twinx()
        (mae_line,) = mae_axis.plot(
            series[column],
            series["final_val_force_mae"],
            marker="o",
            color="tab:blue",
            label="Force MAE",
        )
        (time_line,) = time_axis.plot(
            series[column],
            series["training_runtime_seconds"] / 3600,
            marker="s",
            linestyle="--",
            color="tab:orange",
            label="Training time",
        )
        mae_axis.set_xscale(x_scale)
        mae_axis.set_xlabel(label)
        mae_axis.set_ylabel("Validation force MAE (eV/Å)", color="tab:blue")
        time_axis.set_ylabel("Training time (hours)", color="tab:orange")
        mae_axis.axvline(base_values[column], color="gray", linestyle=":", alpha=0.6)
        mae_axis.set_title(f"Zigzag 700k tuning: {label.lower()}")
        mae_axis.grid(True, which="both", alpha=0.25)
        mae_axis.legend(handles=[mae_line, time_line], loc="best")
        figure.tight_layout()
        figure.savefig(plots_dir / f"{column.removeprefix('tuning_')}.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        frame["training_runtime_seconds"] / 3600,
        frame["final_val_force_mae"],
        c=frame["parameter_count"],
        s=55,
        cmap="viridis",
    )
    for row in frame.itertuples():
        axis.annotate(
            row.tuning_trial_id,
            (row.training_runtime_seconds / 3600, row.final_val_force_mae),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_xlabel("Training time (hours)")
    axis.set_ylabel("Validation force MAE (eV/Å)")
    axis.set_title("Zigzag 700k tuning: accuracy vs time")
    axis.grid(True, which="both", alpha=0.25)
    figure.colorbar(axis.collections[0], ax=axis, label="Parameter count")
    figure.tight_layout()
    figure.savefig(plots_dir / "force_mae_vs_time.png", dpi=180)
    plt.close(figure)

    best = frame.loc[frame["final_val_force_mae"].idxmin()]
    print(f"wrote {output_dir / 'trials.csv'} and {plots_dir}")
    print(
        f"best trial: {best['tuning_trial_id']} "
        f"(force MAE {best['final_val_force_mae']:.6g} eV/Å, "
        f"{best['training_runtime_seconds'] / 3600:.2f} h)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune lr, layer count, and embedding width for the 700k zigzag model."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tuning_runs/zigzag-700k"))
    parser.add_argument("--gpus", help="Comma-separated physical GPU IDs; defaults to all visible.")
    parser.add_argument("--train-frac", type=float, default=0.05)
    parser.add_argument("--valid-frac", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="nequix-zigzag-tuning")
    parser.add_argument("--wandb-mode", default="disabled", choices=("online", "offline", "disabled"))
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="Use portable e3nn-jax operations instead of the OpenEquivariance extension.",
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--trial", nargs=3, metavar=("LR", "LAYERS", "WIDTH"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    if args.trial:
        trial = Trial(float(args.trial[0]), int(args.trial[1]), float(args.trial[2]))
        config = trial_config(trial, args)
        print(
            f"tuning trial {trial.trial_id}: lr={trial.learning_rate:g}, "
            f"n_layers={trial.n_layers}, width={trial.width_multiplier:g}, "
            f"hidden_irreps={config.model_config.hidden_irreps}"
        )
        from nequix.cli import run

        run(config)
        return

    if not args.analyze_only:
        train_path = REPO_ROOT / RUNS[BASE_RUN].train_path
        if not train_path.is_file():
            raise FileNotFoundError(f"required tuning dataset does not exist: {train_path}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failed = run_sweep(sweep_trials(), args)
        if failed:
            print(
                f"{len(failed)} trial(s) failed; rerun the same command to resume",
                file=sys.stderr,
            )
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
