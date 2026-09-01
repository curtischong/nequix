from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from nequix.config import RunConfig, config_values


def _csv_value(value: Any) -> Any:
    """Convert arrays and structured config values to one CSV-safe cell."""
    if value is None:
        return ""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _flatten_config(config: RunConfig) -> dict[str, Any]:
    """Flatten nested dataclass settings while keeping large collections in one cell."""
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping) and prefix != "config_atom_energies":
            for key, item in value.items():
                visit(f"{prefix}_{key}", item)
        else:
            flattened[prefix] = _csv_value(value)

    for key, value in config_values(config).items():
        visit(f"config_{key}", value)
    return flattened


def build_run_summary(
    config: RunConfig,
    *,
    run_name: str,
    trainer: str,
    final_metrics: Mapping[str, Any],
    best_val_loss: Any,
    steps_completed: int,
    epochs_completed: int,
    train_size: int,
    val_size: int,
    param_count: int,
    accelerator_count: int,
    accelerator_type: str,
    backend: str,
    training_runtime_seconds: float,
    validation_runtime_seconds: float,
    invocation_runtime_seconds: float,
    peak_accelerator_memory_bytes: int,
    run_id: str | None = None,
    run_url: str | None = None,
    extra_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat, spreadsheet-oriented record for a completed training run."""
    training_runtime_seconds = float(training_runtime_seconds)
    validation_runtime_seconds = float(validation_runtime_seconds)
    measured_runtime_seconds = training_runtime_seconds + validation_runtime_seconds
    accelerator_hours = measured_runtime_seconds * int(accelerator_count) / 3600.0

    summary: dict[str, Any] = {
        "run_name": run_name,
        "config_name": config.name,
        "run_id": run_id,
        "run_url": run_url,
        "trainer": trainer,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": int(epochs_completed),
        "steps_completed": int(steps_completed),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "parameter_count": int(param_count),
        "backend": backend,
        "accelerator_count": int(accelerator_count),
        "accelerator_type": accelerator_type,
        "peak_accelerator_memory_bytes": int(peak_accelerator_memory_bytes),
        "training_runtime_seconds": training_runtime_seconds,
        "training_runtime_hours": training_runtime_seconds / 3600.0,
        "validation_runtime_seconds": validation_runtime_seconds,
        "validation_runtime_hours": validation_runtime_seconds / 3600.0,
        "measured_runtime_seconds": measured_runtime_seconds,
        "measured_runtime_hours": measured_runtime_seconds / 3600.0,
        "invocation_runtime_seconds": float(invocation_runtime_seconds),
        "invocation_runtime_hours": float(invocation_runtime_seconds) / 3600.0,
        "compute_cost_accelerator_hours": accelerator_hours,
        "accelerator_hours": accelerator_hours,
        "best_val_loss": best_val_loss,
    }
    for key, value in final_metrics.items():
        summary[f"final_val_{key}"] = value
    if extra_values:
        summary.update(extra_values)
    summary.update(_flatten_config(config))
    return {key: _csv_value(value) for key, value in summary.items()}


def format_run_summary_csv(summary: Mapping[str, Any]) -> str:
    """Return one header and one data row, using standard CSV quoting."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=summary.keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerow(summary)
    return output.getvalue()


def print_run_summary_csv(summary: Mapping[str, Any]) -> None:
    """Print a completed run as the final two spreadsheet-ready output lines."""
    print(format_run_summary_csv(summary), end="")
