from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from nequix.config import RUNS
from nequix.scaling import (
    ScalingTrial,
    extract_final_summary,
    finalist_replicates,
    fit_power_law,
    geometric_knee,
    make_trial_config,
    pareto_frontier,
    primary_trials,
    select_finalists,
    write_manifest,
    write_trial_summary,
)
from nequix.train import build_model


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_primary_grid_and_trial_config(tmp_path):
    trials = primary_trials()
    assert len(trials) == 16
    assert len({trial.trial_id for trial in trials}) == 16
    assert {trial.depth for trial in trials} == {2, 3, 4, 5}
    assert {trial.width_multiplier for trial in trials} == {0.25, 0.5, 1.0, 2.0}

    smallest = trials[0]
    assert smallest.hidden_irreps == "32x0e + 16x1o + 8x2e + 8x3o"
    config = make_trial_config(
        smallest,
        tmp_path / smallest.trial_id,
        wandb_mode="disabled",
        kernel=False,
    )
    assert config.train_path == "data/omat-1m/train.atp"
    assert config.valid_path == "data/omat-1m/val.atp"
    assert config.n_epochs == 4
    assert config.batch_size == 128
    assert config.validation.every_steps is None
    assert config.force_mode == "conservative"
    assert config.finetune_from is None
    assert config.model_config.n_layers == 2
    assert config.model_config.hidden_irreps == smallest.hidden_irreps
    assert config.checkpoint_root == str(tmp_path / smallest.trial_id)
    assert config.wandb_mode == "disabled"
    assert config.kernel is False


def test_finalist_replicates_use_two_new_seeds():
    finalists = [ScalingTrial(2, 0.25), ScalingTrial(4, 1.0), ScalingTrial(5, 2.0)]
    repeats = finalist_replicates(finalists)
    assert len(repeats) == 6
    assert {trial.seed for trial in repeats} == {1, 2}
    assert {(trial.depth, trial.width_multiplier) for trial in repeats} == {
        (2, 0.25),
        (4, 1.0),
        (5, 2.0),
    }


def test_fresh_model_initialization_uses_config_seed():
    base = RUNS["nequix-omat-1"]
    model_config = replace(
        base.model_config,
        hidden_irreps="4x0e + 4x1o",
        lmax=1,
        n_layers=1,
        radial_basis_size=2,
        radial_mlp_size=4,
    )
    seed_zero = replace(base, model_config=model_config, kernel=False, seed=0)
    seed_one = replace(base, model_config=model_config, kernel=False, seed=1)
    arrays_a = jax.tree.leaves(eqx.filter(build_model(seed_zero), eqx.is_array))
    arrays_b = jax.tree.leaves(eqx.filter(build_model(seed_zero), eqx.is_array))
    arrays_c = jax.tree.leaves(eqx.filter(build_model(seed_one), eqx.is_array))

    assert all(np.array_equal(a, b) for a, b in zip(arrays_a, arrays_b, strict=True))
    assert any(not np.array_equal(a, c) for a, c in zip(arrays_a, arrays_c, strict=True))


def test_extract_final_summary_finds_last_csv_pair(tmp_path):
    log_path = tmp_path / "train.log"
    fields = ["run_name", "parameter_count", "final_val_force_mae", "config_model_config"]
    with log_path.open("w", newline="") as output:
        output.write("ordinary training output\n")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "run_name": "trial",
                "parameter_count": 123,
                "final_val_force_mae": 0.04,
                "config_model_config": '{"hidden_irreps":"8x0e, 8x1o"}',
            }
        )
    summary = extract_final_summary(log_path)
    assert summary["run_name"] == "trial"
    assert summary["parameter_count"] == "123"
    assert summary["final_val_force_mae"] == "0.04"


def test_pareto_knee_and_finalist_selection():
    records = [
        {"trial_id": "cheap", "compute_cost_accelerator_hours": 1.0, "final_val_force_mae": 1.0},
        {"trial_id": "knee", "compute_cost_accelerator_hours": 2.0, "final_val_force_mae": 0.62},
        {
            "trial_id": "dominated",
            "compute_cost_accelerator_hours": 3.0,
            "final_val_force_mae": 0.9,
        },
        {
            "trial_id": "accurate",
            "compute_cost_accelerator_hours": 5.0,
            "final_val_force_mae": 0.55,
        },
    ]
    frontier = pareto_frontier(records)
    assert [row["trial_id"] for row in frontier] == ["cheap", "knee", "accurate"]
    assert geometric_knee(frontier)["trial_id"] == "knee"
    assert {row["trial_id"] for row in select_finalists(records)} == {
        "cheap",
        "knee",
        "accurate",
    }


def test_power_law_fit_recovers_exponent():
    fit = fit_power_law([1, 2, 4, 8], [3, 1.5, 0.75, 0.375])
    assert fit["coefficient"] == pytest.approx(3.0)
    assert fit["exponent"] == pytest.approx(-1.0)
    assert fit["r_squared"] == pytest.approx(1.0)


def _synthetic_summary(trial: ScalingTrial) -> dict[str, float | int | str]:
    parameters = int(100_000 * trial.depth * trial.width_multiplier**2)
    force_mae = 0.2 / (trial.depth * trial.width_multiplier**0.4) + trial.seed * 0.0001
    hours = 0.2 * trial.depth * trial.width_multiplier + 0.01
    return {
        "run_name": trial.trial_id,
        "parameter_count": parameters,
        "final_val_force_mae": force_mae,
        "final_val_energy_mae_per_atom": force_mae / 2,
        "final_val_stress_mae_per_atom": force_mae * 2,
        "compute_cost_accelerator_hours": hours,
        "peak_accelerator_memory_bytes": parameters * 100,
        "training_runtime_seconds": hours * 3600,
        "validation_runtime_seconds": 10,
        "invocation_runtime_seconds": hours * 3600 + 20,
        "training_examples_seen": 4_039_400,
    }


def test_analysis_cli_writes_aggregate_artifacts(tmp_path):
    output_dir = tmp_path / "scaling"
    for trial in primary_trials():
        write_trial_summary(
            output_dir / "trials" / trial.trial_id / "summary.csv",
            trial,
            _synthetic_summary(trial),
        )
    finalists = [ScalingTrial(2, 0.25), ScalingTrial(4, 1.0), ScalingTrial(5, 2.0)]
    write_manifest(output_dir / "finalists.json", finalists)
    for trial in finalist_replicates(finalists):
        write_trial_summary(
            output_dir / "trials" / trial.trial_id / "summary.csv",
            trial,
            _synthetic_summary(trial),
        )

    subprocess.run(
        [sys.executable, "scripts/analyze_scaling.py", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for relative_path in (
        "trials.csv",
        "architecture_summary.csv",
        "scaling_fits.csv",
        "recommendation.json",
        "report.md",
        "plots/force_mae_vs_parameters.png",
        "plots/force_mae_vs_accelerator_hours.png",
        "plots/resource_scaling.png",
    ):
        assert (output_dir / relative_path).is_file()


def test_sweep_cli_dry_run_lists_all_primary_trials(tmp_path):
    output_dir = tmp_path / "dry-run"
    command = [
        sys.executable,
        "scripts/run_scaling_sweep.py",
        "--output-dir",
        str(output_dir),
        "--gpus",
        "0,1",
        "--max-parallel",
        "2",
        "--phase",
        "grid",
        "--dry-run",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.count("GPU ") == 16
    assert "d2-w0p25-s0" in result.stdout
    assert "d5-w2-s0" in result.stdout
    assert (output_dir / "study_config.json").is_file()

    completed = primary_trials()[0]
    write_trial_summary(
        output_dir / "trials" / completed.trial_id / "summary.csv",
        completed,
        _synthetic_summary(completed),
    )
    resumed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "skipping 1 completed trial(s)" in resumed.stdout
    assert resumed.stdout.count("GPU ") == 15
