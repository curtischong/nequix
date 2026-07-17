#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nequix.scaling import (
    ScalingTrial,
    fit_power_law,
    geometric_knee,
    pareto_frontier,
    primary_trials,
    read_manifest,
    read_trial_summaries,
    select_finalists,
)


NUMERIC_COLUMNS = (
    "scaling_depth",
    "scaling_width_multiplier",
    "scaling_seed",
    "parameter_count",
    "final_val_force_mae",
    "final_val_energy_mae_per_atom",
    "final_val_stress_mae_per_atom",
    "compute_cost_accelerator_hours",
    "peak_accelerator_memory_bytes",
    "training_runtime_seconds",
    "validation_runtime_seconds",
    "invocation_runtime_seconds",
    "training_examples_seen",
)


def _numeric_frame(records: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if "config_kernel" in frame and frame["config_kernel"].nunique(dropna=True) > 1:
        raise ValueError("kernel and non-kernel trials cannot be analyzed as one scaling study")
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = {
        "scaling_depth",
        "scaling_width_multiplier",
        "scaling_seed",
        "parameter_count",
        "final_val_force_mae",
        "compute_cost_accelerator_hours",
        "peak_accelerator_memory_bytes",
        "training_runtime_seconds",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"trial summaries are missing required columns: {sorted(missing)}")
    return frame.sort_values(
        ["scaling_seed", "scaling_depth", "scaling_width_multiplier"]
    ).reset_index(drop=True)


def _architecture_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in (
            "final_val_force_mae",
            "final_val_energy_mae_per_atom",
            "final_val_stress_mae_per_atom",
            "compute_cost_accelerator_hours",
            "peak_accelerator_memory_bytes",
            "training_runtime_seconds",
            "validation_runtime_seconds",
            "invocation_runtime_seconds",
        )
        if column in frame
    ]
    grouped = frame.groupby(
        ["scaling_depth", "scaling_width_multiplier", "parameter_count"], as_index=False
    )
    summary = grouped[metric_columns].agg(["mean", "std", "count"])
    summary.columns = [
        "_".join(str(item) for item in column if item != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    # Pandas retains grouping keys as single-level tuple columns after a multi-aggregation.
    summary = summary.rename(
        columns={
            "scaling_depth": "scaling_depth",
            "scaling_width_multiplier": "scaling_width_multiplier",
            "parameter_count": "parameter_count",
        }
    )
    count = summary["final_val_force_mae_count"]
    summary["final_val_force_mae_ci95"] = np.where(
        count > 1,
        4.303 * summary["final_val_force_mae_std"] / np.sqrt(count),
        np.nan,
    )
    return summary.sort_values(["scaling_depth", "scaling_width_multiplier"])


def _fit_scaling_laws(seed_zero: pd.DataFrame) -> pd.DataFrame:
    fits: list[dict[str, float | int | str]] = []
    for depth, values in seed_zero.groupby("scaling_depth"):
        values = values[(values["parameter_count"] > 0) & (values["final_val_force_mae"] > 0)]
        if len(values) < 2:
            continue
        fit = fit_power_law(values["parameter_count"], values["final_val_force_mae"])
        fits.append(
            {
                "relationship": "force_mae_vs_parameters",
                "depth": int(depth),
                "observations": len(values),
                **fit,
            }
        )
    for metric in (
        "compute_cost_accelerator_hours",
        "peak_accelerator_memory_bytes",
        "training_runtime_seconds",
    ):
        values = seed_zero.dropna(subset=[metric, "parameter_count"])
        values = values[(values[metric] > 0) & (values["parameter_count"] > 0)]
        if len(values) < 2:
            continue
        fit = fit_power_law(values["parameter_count"], values[metric])
        fits.append(
            {
                "relationship": f"{metric}_vs_parameters",
                "depth": "all",
                "observations": len(values),
                **fit,
            }
        )
    return pd.DataFrame(fits)


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict(orient="records")


def _trial_from_record(record: dict) -> ScalingTrial:
    return ScalingTrial(
        depth=int(record["scaling_depth"]),
        width_multiplier=float(record["scaling_width_multiplier"]),
        seed=int(record.get("scaling_seed", 0)),
    )


def _recommend_from_means(values: pd.DataFrame) -> dict:
    records = _records(values)
    frontier = pareto_frontier(
        records,
        compute_key="compute_cost_accelerator_hours_mean",
        error_key="final_val_force_mae_mean",
    )
    if len(frontier) >= 3:
        return dict(
            geometric_knee(
                frontier,
                compute_key="compute_cost_accelerator_hours_mean",
                error_key="final_val_force_mae_mean",
            )
        )
    if len(frontier) == 1:
        return dict(frontier[0])

    points = np.log(
        np.asarray(
            [
                [
                    float(row["compute_cost_accelerator_hours_mean"]),
                    float(row["final_val_force_mae_mean"]),
                ]
                for row in frontier
            ]
        )
    )
    spans = np.ptp(points, axis=0)
    spans[spans == 0.0] = 1.0
    normalized = (points - points.min(axis=0)) / spans
    distances = np.linalg.norm(normalized, axis=1)
    best = min(
        range(len(frontier)),
        key=lambda index: (distances[index], float(frontier[index]["final_val_force_mae_mean"])),
    )
    return dict(frontier[best])


def _plot_accuracy_vs_size(seed_zero: pd.DataFrame, plots_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for depth, values in seed_zero.groupby("scaling_depth"):
        values = values.sort_values("parameter_count")
        axis.plot(
            values["parameter_count"],
            values["final_val_force_mae"],
            marker="o",
            label=f"{int(depth)} layers",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Parameter count")
    axis.set_ylabel("Validation force MAE (eV/Å)")
    axis.set_title("OMat-1M accuracy scaling")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "force_mae_vs_parameters.png", dpi=180)
    plt.close(figure)


def _plot_accuracy_vs_compute(seed_zero: pd.DataFrame, plots_dir: Path) -> None:
    records = _records(seed_zero)
    frontier = pareto_frontier(records)
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        seed_zero["compute_cost_accelerator_hours"],
        seed_zero["final_val_force_mae"],
        c=seed_zero["scaling_depth"],
        s=55,
        cmap="viridis",
    )
    if frontier:
        axis.plot(
            [float(row["compute_cost_accelerator_hours"]) for row in frontier],
            [float(row["final_val_force_mae"]) for row in frontier],
            color="black",
            linestyle="--",
            label="Pareto frontier",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Accelerator-hours")
    axis.set_ylabel("Validation force MAE (eV/Å)")
    axis.set_title("Accuracy versus measured compute")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="Interaction layers")
    figure.tight_layout()
    figure.savefig(plots_dir / "force_mae_vs_accelerator_hours.png", dpi=180)
    plt.close(figure)


def _plot_resources(seed_zero: pd.DataFrame, plots_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(seed_zero["parameter_count"], seed_zero["training_runtime_seconds"] / 3600)
    axes[0].set_ylabel("Training duration (hours)")
    axes[1].scatter(
        seed_zero["parameter_count"], seed_zero["peak_accelerator_memory_bytes"] / 2**30
    )
    axes[1].set_ylabel("Peak accelerator memory (GiB)")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Parameter count")
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle("OMat-1M resource scaling")
    figure.tight_layout()
    figure.savefig(plots_dir / "resource_scaling.png", dpi=180)
    plt.close(figure)


def analyze(output_dir: Path) -> dict:
    records = read_trial_summaries(output_dir)
    if not records:
        raise ValueError(f"no completed trial summaries found under {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    frame = _numeric_frame(records)
    frame.to_csv(output_dir / "trials.csv", index=False)
    architecture = _architecture_summary(frame)
    architecture.to_csv(output_dir / "architecture_summary.csv", index=False)
    seed_zero = frame[frame["scaling_seed"] == 0]
    fits = _fit_scaling_laws(seed_zero)
    fits.to_csv(output_dir / "scaling_fits.csv", index=False)

    _plot_accuracy_vs_size(seed_zero, plots_dir)
    _plot_accuracy_vs_compute(seed_zero, plots_dir)
    _plot_resources(seed_zero, plots_dir)

    preliminary = select_finalists(_records(seed_zero), count=min(3, len(seed_zero)))
    recommendation: dict | None = None
    recommendation_status = "preliminary"
    finalists_path = output_dir / "finalists.json"
    if finalists_path.is_file():
        finalist_architectures = {
            (trial.depth, trial.width_multiplier) for trial in read_manifest(finalists_path)
        }
        completed_finalists = architecture[
            architecture.apply(
                lambda row: (
                    (int(row["scaling_depth"]), row["scaling_width_multiplier"])
                    in finalist_architectures
                ),
                axis=1,
            )
            & (architecture["final_val_force_mae_count"] >= 3)
        ]
        if len(completed_finalists) == len(finalist_architectures):
            recommendation = _recommend_from_means(completed_finalists)
            recommendation_status = "final"

    if recommendation is None:
        frontier = pareto_frontier(_records(seed_zero))
        recommendation = dict(geometric_knee(frontier))

    recommended_trial = _trial_from_record(recommendation)
    recommendation_record = {
        "status": recommendation_status,
        "depth": recommended_trial.depth,
        "width_multiplier": recommended_trial.width_multiplier,
        "hidden_irreps": recommended_trial.hidden_irreps,
        "parameter_count": int(recommendation["parameter_count"]),
        "force_mae": float(
            recommendation.get(
                "final_val_force_mae_mean", recommendation.get("final_val_force_mae")
            )
        ),
        "accelerator_hours": float(
            recommendation.get(
                "compute_cost_accelerator_hours_mean",
                recommendation.get("compute_cost_accelerator_hours"),
            )
        ),
    }
    (output_dir / "recommendation.json").write_text(
        json.dumps(recommendation_record, indent=2) + "\n"
    )

    expected_primary = len(primary_trials())
    report = [
        "# OMat-1M depth × width scaling report",
        "",
        f"Completed seed-0 grid trials: {len(seed_zero)}/{expected_primary}.",
        f"Total completed trials: {len(frame)}.",
        "",
        f"## {recommendation_status.title()} recommendation",
        "",
        f"- Depth: {recommendation_record['depth']} interaction layers",
        f"- Width multiplier: {recommendation_record['width_multiplier']:g}",
        f"- Hidden irreps: `{recommendation_record['hidden_irreps']}`",
        f"- Parameters: {recommendation_record['parameter_count']:,}",
        f"- Force MAE: {recommendation_record['force_mae']:.6g} eV/Å",
        f"- Measured compute: {recommendation_record['accelerator_hours']:.4g} accelerator-hours",
        "",
        "## Seed-0 Pareto finalists",
        "",
    ]
    for row in preliminary:
        trial = _trial_from_record(dict(row))
        report.append(
            f"- `{trial.trial_id}`: {float(row['final_val_force_mae']):.6g} eV/Å, "
            f"{float(row['compute_cost_accelerator_hours']):.4g} accelerator-hours"
        )
    report.extend(["", "## Scaling fits", ""])
    for row in fits.to_dict(orient="records"):
        report.append(
            f"- `{row['relationship']}` (depth {row['depth']}): exponent "
            f"{row['exponent']:.4g}, R² {row['r_squared']:.4g}"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    return recommendation_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed Nequix scaling trials.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    recommendation = analyze(args.output_dir.resolve())
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
