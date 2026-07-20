"""Expensive downstream evaluations used during foundation-model training."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.data import chemical_symbols
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import LBFGS

from nequix.calculator import NequixCalculator
from nequix.config import EvaluationConfig, LongMDEvalConfig, MLIPArenaConfig, ModelMetadata
from nequix.model import save_model


MD22_TEMPERATURES = {
    "AT-AT-CG-CG": 500.0,
    "AT-AT": 500.0,
    "Ac-Ala3-NHMe": 500.0,
    "DHA": 500.0,
    "buckyball-catcher": 400.0,
    "double-walled_nanotube": 400.0,
    "stachyose": 500.0,
}

TM23_MELTING_TEMPERATURES = {
    "Ag": 1235.0,
    "Au": 1337.0,
    "Cd": 594.0,
    "Co": 1768.0,
    "Cr": 2180.0,
    "Cu": 1358.0,
    "Fe": 1811.0,
    "Hf": 2506.0,
    "Hg": 234.0,
    "Ir": 2739.0,
    "Mn": 1519.0,
    "Mo": 2896.0,
    "Nb": 2750.0,
    "Ni": 1728.0,
    "Os": 3306.0,
    "Pd": 1828.0,
    "Pt": 2041.0,
    "Re": 3459.0,
    "Rh": 2237.0,
    "Ru": 2607.0,
    "Ta": 3290.0,
    "Tc": 2430.0,
    "Ti": 1941.0,
    "V": 2183.0,
    "W": 3695.0,
    "Zn": 693.0,
    "Zr": 2128.0,
}

TM23_TEMPERATURE_FACTORS = {"cold": 0.25, "warm": 0.75, "melt": 1.25}


def validate_evaluation_config(config: EvaluationConfig | None) -> None:
    if config is None:
        return
    if config.every_steps <= 0:
        raise ValueError("evaluations.every_steps must be greater than zero")
    if config.mlip_arena is None and config.long_md is None:
        raise ValueError("evaluations must enable mlip_arena and/or long_md")
    if config.mlip_arena is not None and config.mlip_arena.max_workers <= 0:
        raise ValueError("evaluations.mlip_arena.max_workers must be greater than zero")
    if config.long_md is not None:
        md = config.long_md
        if md.steps is not None and md.steps <= 0:
            raise ValueError("evaluations.long_md.steps must be greater than zero")
        if md.time_step_fs is not None and md.time_step_fs <= 0:
            raise ValueError("evaluations.long_md.time_step_fs must be greater than zero")
        if md.save_frequency <= 0:
            raise ValueError("evaluations.long_md.save_frequency must be greater than zero")
        if md.max_systems is not None and md.max_systems <= 0:
            raise ValueError("evaluations.long_md.max_systems must be greater than zero")


def evaluations_due(config: EvaluationConfig | None, step: int) -> bool:
    """Return whether expensive evaluations are scheduled at this global step."""
    return config is not None and step > 0 and step % config.every_steps == 0


def long_md_protocol(config: LongMDEvalConfig) -> tuple[int, float]:
    default_steps, default_time_step = {
        "tm23": (20_000, 5.0),
        "md22": (100_000, 1.0),
    }[config.dataset]
    return config.steps or default_steps, config.time_step_fs or default_time_step


def load_long_md_systems(config: LongMDEvalConfig) -> list[tuple[str, Atoms, float]]:
    """Load the TM23/MD22 starting frames and protocol temperatures."""
    root = Path(config.dataset_root)
    specifications: list[tuple[str, Path, float]] = []
    if config.dataset == "tm23":
        for element, melting_temperature in sorted(TM23_MELTING_TEMPERATURES.items()):
            for regime in config.tm23_regimes:
                name = f"{element}-{regime}"
                path = root / "tm23" / f"{element}_{regime}_nequip_test.xyz"
                specifications.append(
                    (name, path, melting_temperature * TM23_TEMPERATURE_FACTORS[regime])
                )
    else:
        for molecule, temperature in sorted(MD22_TEMPERATURES.items()):
            path = root / "md22" / f"md22_{molecule}.xyz"
            specifications.append((molecule, path, temperature))
    if config.max_systems is not None:
        specifications = specifications[: config.max_systems]
    return [(name, read(path), temperature) for name, path, temperature in specifications]


def _energy_drift(energies: Sequence[float], times_ps: Sequence[float]) -> float:
    """Return the eSEN benchmark drift in meV / atom / ps."""
    energies = np.asarray(energies, dtype=float)
    times_ps = np.asarray(times_ps, dtype=float)
    equilibration = int(len(energies) * 0.1)
    energies = energies[equilibration:]
    times_ps = times_ps[equilibration:]
    if len(energies) >= 20:
        energies = np.convolve(energies, np.ones(20) / 20.0, mode="valid")
        times_ps = times_ps[19:]
    if len(energies) < 2 or times_ps[-1] <= times_ps[0]:
        raise ValueError("the trajectory is too short to calculate energy drift")
    return float(1000.0 * abs(energies[-1] - energies[0]) / (times_ps[-1] - times_ps[0]))


def run_long_md_evaluation(
    config: LongMDEvalConfig,
    calculator: Any,
    *,
    systems: Sequence[tuple[str, Atoms, float]] | None = None,
) -> dict[str, float]:
    """Run the eSEN 100 ps NVE conservation evaluation with an ASE calculator."""
    steps, time_step_fs = long_md_protocol(config)
    systems = list(systems) if systems is not None else load_long_md_systems(config)
    if config.max_systems is not None:
        systems = systems[: config.max_systems]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(config.seed)
    results = []

    for name, initial_atoms, temperature in systems:
        started = time.perf_counter()
        atoms = initial_atoms.copy()
        atoms.calc = calculator
        try:
            optimizer = LBFGS(atoms, logfile=None)
            converged = optimizer.run(
                fmax=config.relaxation_fmax,
                steps=config.relaxation_steps,
            )
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
            dynamics = VelocityVerlet(atoms, timestep=time_step_fs * units.fs)
            energies: list[float] = []
            times_ps: list[float] = []

            def record_energy() -> None:
                energies.append(float(atoms.get_total_energy()) / len(atoms))
                times_ps.append(float(dynamics.get_time() / (1000.0 * units.fs)))

            record_energy()
            dynamics.attach(record_energy, interval=config.save_frequency)
            dynamics.run(steps)
            if dynamics.nsteps % config.save_frequency:
                record_energy()
            drift = _energy_drift(energies, times_ps)
            result = {
                "name": name,
                "temperature_K": temperature,
                "converged": bool(converged),
                "drift_mev_per_atom_ps": drift,
                "runtime_seconds": time.perf_counter() - started,
            }
        except Exception as error:
            result = {
                "name": name,
                "temperature_K": temperature,
                "converged": False,
                "error": f"{type(error).__name__}: {error}",
                "runtime_seconds": time.perf_counter() - started,
            }
        results.append(result)

    successful_drifts = [
        item["drift_mev_per_atom_ps"] for item in results if "drift_mev_per_atom_ps" in item
    ]
    summary = {
        "drift_mev_per_atom_ps": float(np.mean(successful_drifts))
        if successful_drifts
        else float("nan"),
        "successful_systems": float(len(successful_drifts)),
        "failed_systems": float(len(results) - len(successful_drifts)),
        "runtime_seconds": float(sum(item["runtime_seconds"] for item in results)),
    }
    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "systems": results}, indent=2, allow_nan=True)
    )
    return summary


def _dataframe_summary(frame: Any) -> dict[str, float]:
    summary: dict[str, float] = {}
    if frame is None or getattr(frame, "empty", True):
        return summary
    for column in frame.columns:
        try:
            values = np.asarray(frame[column], dtype=float)
        except (TypeError, ValueError):
            continue
        if values.ndim == 1 and np.isfinite(values).any():
            summary[f"{column}_mean"] = float(np.nanmean(values))
    return summary


def run_mlip_arena_evaluation(
    config: MLIPArenaConfig,
    *,
    calculator: type = NequixCalculator,
    calculator_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Run configured tasks using MLIP Arena's official flow implementations."""
    try:
        from mlip_arena.flows.diatomics import analyze, homonuclear_diatomic, homonuclear_diatomics
        from mlip_arena.flows.eos_bulk import run_db as eos_bulk
        from mlip_arena.flows.ev import run_db as ev
        from prefect.task_runners import ThreadPoolTaskRunner
    except ImportError as error:
        raise ImportError(
            "MLIP Arena evaluation requires Python 3.11+ and the 'evals' extra: "
            "uv sync --python 3.12 --extra evals"
        ) from error

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calculator_kwargs = calculator_kwargs or {}
    summary: dict[str, float] = {}

    for task_name in config.tasks:
        started = time.perf_counter()
        if task_name == "diatomics":
            task_dir = output_dir / "diatomics"
            if config.elements is None:
                homonuclear_diatomics.with_options(
                    task_runner=ThreadPoolTaskRunner(max_workers=config.max_workers)
                )(
                    calculator=calculator,
                    calculator_kwargs=calculator_kwargs,
                    run_dir=task_dir,
                )
            else:
                for element in config.elements:
                    # ``fn`` is Prefect's underlying task function; using it
                    # preserves MLIP Arena's exact calculation without starting
                    # a flow per element.
                    homonuclear_diatomic.fn(
                        symbol=element,
                        calculator=calculator,
                        calculator_kwargs=calculator_kwargs,
                        out_dir=task_dir,
                    )
            task_summary = _dataframe_summary(analyze(task_dir))
        elif task_name == "eos_bulk":
            frame = eos_bulk.with_options(
                task_runner=ThreadPoolTaskRunner(max_workers=config.max_workers)
            )(
                calculator=calculator,
                calculator_kwargs=calculator_kwargs,
                run_dir=output_dir / "eos_bulk",
                dataset=config.dataset,
                dataset_file=config.dataset_file,
            )
            task_summary = _dataframe_summary(frame)
        elif task_name == "ev":
            frame = ev.with_options(
                task_runner=ThreadPoolTaskRunner(max_workers=config.max_workers)
            )(
                calculator=calculator,
                calculator_kwargs=calculator_kwargs,
                run_dir=output_dir / "ev",
                dataset=config.dataset,
                dataset_file=config.dataset_file,
            )
            task_summary = _dataframe_summary(frame)
        else:  # pragma: no cover - protected by the config type.
            raise ValueError(f"unsupported MLIP Arena task: {task_name}")
        task_summary["runtime_seconds"] = time.perf_counter() - started
        summary.update({f"{task_name}/{key}": value for key, value in task_summary.items()})

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def run_model_evaluations(
    model: Any,
    metadata: ModelMetadata,
    config: EvaluationConfig,
    *,
    kernel: bool,
    step: int,
) -> dict[str, float]:
    """Save an EMA snapshot and run all configured downstream evaluations."""
    metrics: dict[str, float] = {}
    if config.mlip_arena is not None:
        arena_dir = Path(config.mlip_arena.output_dir) / f"step-{step}"
        checkpoint = arena_dir / "model.nqx"
        save_model(checkpoint, model, metadata)
        elements = config.mlip_arena.elements
        if elements is None:
            elements = tuple(chemical_symbols[number] for number in metadata.atomic_numbers)
        arena_config = replace(
            config.mlip_arena,
            output_dir=str(arena_dir),
            elements=elements,
        )
        arena_metrics = run_mlip_arena_evaluation(
            arena_config,
            calculator_kwargs={"model_path": checkpoint, "use_kernel": kernel},
        )
        metrics.update({f"mlip_arena/{key}": value for key, value in arena_metrics.items()})

    if config.long_md is not None:
        md_dir = Path(config.long_md.output_dir) / f"step-{step}"
        checkpoint = md_dir / "model.nqx"
        save_model(checkpoint, model, metadata)
        md_config = replace(config.long_md, output_dir=str(md_dir))
        calculator_instance = NequixCalculator(checkpoint, use_kernel=kernel)
        md_metrics = run_long_md_evaluation(md_config, calculator_instance)
        metrics.update({f"long_md/{key}": value for key, value in md_metrics.items()})
    return metrics
