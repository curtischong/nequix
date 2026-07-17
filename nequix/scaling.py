from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nequix.config import RUNS, TrainerConfig


DEPTHS = (2, 3, 4, 5)
WIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)
BASE_WIDTHS = (128, 64, 32, 32)
IRREPS = ("0e", "1o", "2e", "3o")


@dataclass(frozen=True)
class ScalingTrial:
    depth: int
    width_multiplier: float
    seed: int = 0

    @property
    def width_label(self) -> str:
        return f"{self.width_multiplier:g}".replace(".", "p")

    @property
    def trial_id(self) -> str:
        return f"d{self.depth}-w{self.width_label}-s{self.seed}"

    @property
    def hidden_irreps(self) -> str:
        widths = [int(width * self.width_multiplier) for width in BASE_WIDTHS]
        if any(width < 1 for width in widths):
            raise ValueError("width multiplier produces an empty irrep")
        return " + ".join(f"{width}x{irrep}" for width, irrep in zip(widths, IRREPS, strict=True))


def primary_trials() -> list[ScalingTrial]:
    return [
        ScalingTrial(depth=depth, width_multiplier=width, seed=0)
        for depth in DEPTHS
        for width in WIDTH_MULTIPLIERS
    ]


def finalist_replicates(finalists: Iterable[ScalingTrial]) -> list[ScalingTrial]:
    return [replace(trial, seed=seed) for trial in finalists for seed in (1, 2)]


def make_trial_config(
    trial: ScalingTrial,
    trial_dir: str | Path,
    *,
    wandb_project: str = "nequix-scaling-omat1m",
    wandb_mode: str | None = None,
    kernel: bool = True,
) -> TrainerConfig:
    base = RUNS["nequix-omat-1"]
    if not isinstance(base, TrainerConfig):  # pragma: no cover - registry invariant
        raise TypeError("nequix-omat-1 must be a standard trainer config")

    trial_dir = Path(trial_dir)
    state_path = trial_dir / "state.pkl"
    name = f"omat1m-{trial.trial_id}"
    return replace(
        base,
        name=name,
        train_path="data/omat-1m/train.atp",
        valid_path="data/omat-1m/val.atp",
        valid_frac=None,
        dataset_name="omat1m",
        train_frac=1.0,
        seed=trial.seed,
        n_epochs=4,
        batch_size=128,
        val_every_steps=None,
        force_mode="conservative",
        finetune_from=None,
        state_path=str(state_path),
        resume_from=str(state_path),
        checkpoint_path=str(trial_dir / "checkpoint.nqx"),
        model_config=replace(
            base.model_config,
            n_layers=trial.depth,
            hidden_irreps=trial.hidden_irreps,
        ),
        run_name=name,
        wandb_run_name=name,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        kernel=kernel,
    )


def write_manifest(path: str | Path, trials: Sequence[ScalingTrial]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(trial) for trial in trials], indent=2) + "\n")


def read_manifest(path: str | Path) -> list[ScalingTrial]:
    values = json.loads(Path(path).read_text())
    return [ScalingTrial(**value) for value in values]


def extract_final_summary(log_path: str | Path) -> dict[str, str]:
    """Extract the trainer's documented final two-line CSV from a trial log."""
    lines = [line for line in Path(log_path).read_text(errors="replace").splitlines() if line]
    if len(lines) < 2:
        raise ValueError(f"training log has no final CSV summary: {log_path}")
    for index in range(len(lines) - 2, -1, -1):
        try:
            rows = list(csv.DictReader([lines[index], lines[index + 1]]))
        except csv.Error:
            continue
        if len(rows) == 1 and {
            "run_name",
            "parameter_count",
            "final_val_force_mae",
        }.issubset(rows[0]):
            return dict(rows[0])
    raise ValueError(f"training log has no valid final CSV summary: {log_path}")


def write_trial_summary(path: str | Path, trial: ScalingTrial, summary: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial.trial_id,
        "scaling_depth": trial.depth,
        "scaling_width_multiplier": trial.width_multiplier,
        "scaling_seed": trial.seed,
        **summary,
    }
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=record)
        writer.writeheader()
        writer.writerow(record)


def read_trial_summaries(output_dir: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(Path(output_dir).glob("trials/*/summary.csv")):
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 1:
            raise ValueError(f"expected exactly one summary row in {path}")
        records.append(dict(rows[0]))
    return records


def pareto_frontier(
    records: Sequence[Mapping[str, Any]],
    *,
    compute_key: str = "compute_cost_accelerator_hours",
    error_key: str = "final_val_force_mae",
) -> list[Mapping[str, Any]]:
    """Return records nondominated when both compute and error are minimized."""
    ordered = sorted(
        records,
        key=lambda row: (float(row[compute_key]), float(row[error_key]), str(row.get("trial_id"))),
    )
    frontier: list[Mapping[str, Any]] = []
    best_error = math.inf
    for row in ordered:
        error = float(row[error_key])
        if error < best_error:
            frontier.append(row)
            best_error = error
    return frontier


def _normalized_log_points(
    records: Sequence[Mapping[str, Any]], compute_key: str, error_key: str
) -> np.ndarray:
    points = np.log(
        np.asarray(
            [[float(row[compute_key]), float(row[error_key])] for row in records],
            dtype=float,
        )
    )
    spans = np.ptp(points, axis=0)
    spans[spans == 0.0] = 1.0
    return (points - points.min(axis=0)) / spans


def geometric_knee(
    frontier: Sequence[Mapping[str, Any]],
    *,
    compute_key: str = "compute_cost_accelerator_hours",
    error_key: str = "final_val_force_mae",
) -> Mapping[str, Any]:
    if not frontier:
        raise ValueError("cannot find a knee on an empty frontier")
    if len(frontier) < 3:
        return min(frontier, key=lambda row: float(row[error_key]))

    points = _normalized_log_points(frontier, compute_key, error_key)
    start, end = points[0], points[-1]
    chord = end - start
    norm = np.linalg.norm(chord)
    if norm == 0.0:
        return min(frontier, key=lambda row: float(row[error_key]))
    offsets = points - start
    distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / norm
    return frontier[int(np.argmax(distances))]


def select_finalists(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int = 3,
    compute_key: str = "compute_cost_accelerator_hours",
    error_key: str = "final_val_force_mae",
) -> list[Mapping[str, Any]]:
    if len(records) < count:
        raise ValueError(f"need at least {count} completed trials to select finalists")
    frontier = pareto_frontier(records, compute_key=compute_key, error_key=error_key)
    cheap = frontier[0]
    accurate = min(frontier, key=lambda row: float(row[error_key]))
    knee = geometric_knee(frontier, compute_key=compute_key, error_key=error_key)

    selected: list[Mapping[str, Any]] = []
    for row in (cheap, knee, accurate):
        if row not in selected:
            selected.append(row)

    # Prefer frontier points that add the greatest log-space separation.
    while len(selected) < count and any(row not in selected for row in frontier):
        candidates = [row for row in frontier if row not in selected]
        all_points = _normalized_log_points([*selected, *candidates], compute_key, error_key)
        selected_points = all_points[: len(selected)]
        candidate_points = all_points[len(selected) :]
        distances = [
            min(np.linalg.norm(point - chosen) for chosen in selected_points)
            for point in candidate_points
        ]
        selected.append(candidates[int(np.argmax(distances))])

    if len(selected) < count:
        remaining = sorted(
            (row for row in records if row not in selected),
            key=lambda row: (float(row[error_key]), float(row[compute_key])),
        )
        selected.extend(remaining[: count - len(selected)])
    return selected


def fit_power_law(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Fit y = coefficient * x ** exponent in log space."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) < 2 or np.any(x_values <= 0.0) or np.any(y_values <= 0.0):
        raise ValueError("power-law fit requires at least two positive observations")
    log_x = np.log(x_values)
    log_y = np.log(y_values)
    exponent, intercept = np.polyfit(log_x, log_y, deg=1)
    predicted = intercept + exponent * log_x
    residual = float(np.sum((log_y - predicted) ** 2))
    total = float(np.sum((log_y - log_y.mean()) ** 2))
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return {
        "coefficient": float(np.exp(intercept)),
        "exponent": float(exponent),
        "r_squared": r_squared,
    }
