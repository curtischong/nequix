"""Expensive downstream evaluations used during foundation-model training."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
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
from nequix.config import (
    BenchmarkConfig,
    LongMDEvalConfig,
    MLIPArenaConfig,
    ModelMetadata,
    ValidationConfig,
)
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


def validate_validation_config(config: ValidationConfig) -> None:
    if config.every_steps is not None and config.every_steps <= 0:
        raise ValueError("validation.every_steps must be greater than zero")


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    if config.every_steps is not None and config.every_steps <= 0:
        raise ValueError("benchmarks.every_steps must be greater than zero")
    if config.workers_per_gpu <= 0:
        raise ValueError("benchmarks.workers_per_gpu must be greater than zero")
    if config.mlip_arena is not None and config.mlip_arena.max_workers <= 0:
        raise ValueError("benchmarks.mlip_arena.max_workers must be greater than zero")
    if config.long_md is not None:
        md = config.long_md
        if md.steps is not None and md.steps <= 0:
            raise ValueError("benchmarks.long_md.steps must be greater than zero")
        if md.time_step_fs is not None and md.time_step_fs <= 0:
            raise ValueError("benchmarks.long_md.time_step_fs must be greater than zero")
        if md.save_frequency <= 0:
            raise ValueError("benchmarks.long_md.save_frequency must be greater than zero")
        if md.max_systems is not None and md.max_systems <= 0:
            raise ValueError("benchmarks.long_md.max_systems must be greater than zero")


def benchmarks_due(config: BenchmarkConfig, step: int) -> bool:
    """Return whether the downstream benchmarks are scheduled at this global step."""
    if config.mlip_arena is None and config.long_md is None:
        return False
    cadence = config.every_steps
    return cadence is not None and step > 0 and step % cadence == 0


def long_md_protocol(config: LongMDEvalConfig) -> tuple[int, float]:
    default_steps, default_time_step = {
        "tm23": (20_000, 5.0),
        "md22": (100_000, 1.0),
    }[config.dataset]
    return config.steps or default_steps, config.time_step_fs or default_time_step


def _long_md_specifications(config: LongMDEvalConfig) -> list[tuple[str, Path, float]]:
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
    return specifications


def load_long_md_systems(config: LongMDEvalConfig) -> list[tuple[str, Atoms, float]]:
    """Load the TM23/MD22 starting frames and protocol temperatures."""
    return [
        (name, read(path), temperature)
        for name, path, temperature in _long_md_specifications(config)
    ]


def cuda_device_ids() -> tuple[str, ...]:
    """The CUDA device ids visible to this process, for pinning eval workers."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return tuple(part.strip() for part in visible.split(",") if part.strip())
    import jax

    return tuple(
        str(device.local_hardware_id) for device in jax.devices() if device.platform == "gpu"
    )


def _partition(items: Sequence[Any], devices: Sequence[str]) -> list[tuple[str, list[Any]]]:
    chunks: list[tuple[str, list[Any]]] = [(device, []) for device in devices]
    for index, item in enumerate(items):
        chunks[index % len(devices)][1].append(item)
    return [(device, chunk) for device, chunk in chunks if chunk]


def _serializable_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}


MPS_PIPE_DIRECTORY = "/tmp/nequix-eval-mps"
JAX_CACHE_DIRECTORY = "evaluations/jax_cache"


def _mps_environment() -> dict[str, str]:
    """Start (or reuse) a dedicated CUDA MPS daemon for evaluation workers.

    Without MPS, worker processes sharing a GPU time-slice their contexts and
    the tiny evaluation kernels serialize; MPS lets them execute concurrently.
    The private pipe directory keeps the training process outside MPS.
    Set NEQUIX_DISABLE_MPS=1 to run the workers without MPS.
    """
    if os.environ.get("NEQUIX_DISABLE_MPS"):
        return {}
    control = shutil.which("nvidia-cuda-mps-control")
    if control is None:
        return {}
    Path(MPS_PIPE_DIRECTORY).mkdir(parents=True, exist_ok=True)
    Path(f"{MPS_PIPE_DIRECTORY}-log").mkdir(parents=True, exist_ok=True)
    environment = {
        "CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE_DIRECTORY,
        "CUDA_MPS_LOG_DIRECTORY": f"{MPS_PIPE_DIRECTORY}-log",
    }
    # Exits immediately with an error if a daemon already serves this pipe.
    subprocess.run([control, "-d"], env=os.environ | environment, capture_output=True)
    return {"CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE_DIRECTORY}


def _spawn_evaluation_workers(
    payloads: Sequence[dict[str, Any]], scratch_dir: Path
) -> list[tuple[subprocess.Popen, Path]]:
    """Start one pinned-GPU subprocess per payload."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    shared_env = _mps_environment() | {
        # Workers share their GPU with the training process, so they must
        # allocate on demand instead of claiming the XLA default pool.
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        # Identical jit programs recompile in every worker on every trigger
        # without a persistent cache.
        "JAX_COMPILATION_CACHE_DIR": str(Path(JAX_CACHE_DIRECTORY).absolute()),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    }
    processes: list[tuple[subprocess.Popen, Path]] = []
    for index, payload in enumerate(payloads):
        payload_path = scratch_dir / f"worker-{index}.json"
        payload_path.write_text(json.dumps(payload))
        log_path = scratch_dir / f"worker-{index}.log"
        env = os.environ | shared_env | {"CUDA_VISIBLE_DEVICES": payload["device"]}
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                [sys.executable, "-m", "nequix.evaluation", str(payload_path)],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        processes.append((process, log_path))
    return processes


def _wait_for_workers(processes: Sequence[tuple[subprocess.Popen, Path]]) -> None:
    failures = [
        f"{log_path}:\n{log_path.read_text()[-2000:]}"
        for process, log_path in processes
        if process.wait() != 0
    ]
    if failures:
        raise RuntimeError("evaluation workers failed:\n" + "\n".join(failures))


def _run_evaluation_workers(payloads: Sequence[dict[str, Any]], scratch_dir: Path) -> None:
    """Run one pinned-GPU subprocess per payload and wait for all of them."""
    _wait_for_workers(_spawn_evaluation_workers(payloads, scratch_dir))


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


def _long_md_summary(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    successful_drifts = [
        item["drift_mev_per_atom_ps"] for item in results if "drift_mev_per_atom_ps" in item
    ]
    return {
        "drift_mev_per_atom_ps": float(np.mean(successful_drifts))
        if successful_drifts
        else float("nan"),
        "successful_systems": float(len(successful_drifts)),
        "failed_systems": float(len(results) - len(successful_drifts)),
        "runtime_seconds": float(sum(item["runtime_seconds"] for item in results)),
    }


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

    summary = _long_md_summary(results)
    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "systems": results}, indent=2, allow_nan=True)
    )
    return summary


def _system_cost(path: Path) -> int:
    """The atom count from an xyz header, a proxy for per-step cost."""
    with path.open() as handle:
        return int(handle.readline())


def _long_md_payloads(
    config: LongMDEvalConfig,
    calculator_kwargs: dict[str, Any],
    slots: Sequence[str],
) -> list[dict[str, Any]]:
    """Balance the MD systems over worker slots by total atom count.

    System sizes span roughly 30-80 atoms, so unbalanced assignment leaves the
    wave waiting on whichever GPU drew the largest systems.
    """
    output_dir = Path(config.output_dir)
    specifications = _long_md_specifications(config)
    costs = {name: _system_cost(path) for name, path, _ in specifications}
    groups: list[list[str]] = [[] for _ in slots]
    totals = [0] * len(slots)
    for name in sorted(costs, key=lambda key: costs[key], reverse=True):
        index = totals.index(min(totals))
        groups[index].append(name)
        totals[index] += costs[name]
    return [
        {
            "kind": "long_md",
            "device": device,
            "systems": group,
            # The worker already receives an explicit system subset, so its
            # config must not re-apply the prefix limit.
            "config": asdict(
                replace(config, output_dir=str(output_dir / f"worker-{index}"), max_systems=None)
            ),
            "calculator_kwargs": _serializable_kwargs(calculator_kwargs),
        }
        for index, (device, group) in enumerate(zip(slots, groups))
        if group
    ]


def _collect_long_md(output_dir: Path, payloads: Sequence[dict[str, Any]]) -> dict[str, float]:
    results = [
        system
        for payload in payloads
        for system in json.loads(
            (Path(payload["config"]["output_dir"]) / "results.json").read_text()
        )["systems"]
    ]
    summary = _long_md_summary(results)
    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "systems": results}, indent=2, allow_nan=True)
    )
    return summary


def run_long_md_evaluation_parallel(
    config: LongMDEvalConfig,
    *,
    calculator_kwargs: dict[str, Any],
    gpus: Sequence[str],
) -> dict[str, float]:
    """Fan the long-MD systems out across GPUs in pinned worker processes."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = _long_md_payloads(config, calculator_kwargs, gpus)
    _run_evaluation_workers(payloads, output_dir / "workers")
    return _collect_long_md(output_dir, payloads)


def _diatomics_payloads(
    elements: Sequence[str],
    calculator_kwargs: dict[str, Any],
    slots: Sequence[str],
    task_dir: Path,
) -> list[dict[str, Any]]:
    # Each element writes its own trajectory file, so workers can share the
    # task directory and ``analyze`` merges them.
    return [
        {
            "kind": "diatomics",
            "device": device,
            "elements": chunk,
            "out_dir": str(task_dir),
            "calculator_kwargs": _serializable_kwargs(calculator_kwargs),
        }
        for device, chunk in _partition(elements, slots)
    ]


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
    gpus: Sequence[str] | None = None,
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
            parallel = (
                config.elements is not None
                and gpus is not None
                and len(gpus) > 1
                and calculator is NequixCalculator
            )
            if parallel:
                payloads = _diatomics_payloads(config.elements, calculator_kwargs, gpus, task_dir)
                _run_evaluation_workers(payloads, task_dir / "workers")
            elif config.elements is None:
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


@dataclass
class EvaluationWave:
    """A wave of evaluation workers running concurrently with training."""

    step: int
    processes: list[tuple[subprocess.Popen, Path]]
    arena_dir: Path
    task_dir: Path
    md_dir: Path
    md_payloads: list[dict[str, Any]]
    started: float

    def poll(self) -> dict[str, float] | None:
        """The wave's metrics once every worker has exited, None until then."""
        if any(process.poll() is None for process, _ in self.processes):
            return None
        return self._collect()

    def wait(self) -> dict[str, float]:
        for process, _ in self.processes:
            process.wait()
        return self._collect()

    def _collect(self) -> dict[str, float]:
        from mlip_arena.flows.diatomics import analyze

        _wait_for_workers(self.processes)
        wave_seconds = time.perf_counter() - self.started
        arena_summary = _dataframe_summary(analyze(self.task_dir))
        arena_summary["runtime_seconds"] = wave_seconds
        (self.arena_dir / "summary.json").write_text(
            json.dumps(
                {f"diatomics/{key}": value for key, value in arena_summary.items()},
                indent=2,
                allow_nan=True,
            )
        )
        metrics = {f"mlip_arena/diatomics/{key}": value for key, value in arena_summary.items()}
        metrics.update(
            {
                f"long_md/{key}": value
                for key, value in _collect_long_md(self.md_dir, self.md_payloads).items()
            }
        )
        return metrics


def launch_model_evaluations(
    model: Any,
    metadata: ModelMetadata,
    config: BenchmarkConfig,
    *,
    kernel: bool,
    step: int,
) -> EvaluationWave | None:
    """Save an EMA snapshot and start the benchmark worker wave without waiting.

    One wave runs the diatomic curves and MD systems together, paying the
    process/JIT startup tax once. Duplicating the device list oversubscribes
    each GPU: the benchmark systems are far too small to saturate one, so
    concurrent (MPS-shared) workers overlap their host and kernel-launch
    latency while training keeps stepping. Returns None for configurations
    the single-wave path does not cover.
    """
    slots = cuda_device_ids() * config.workers_per_gpu
    arena = config.mlip_arena
    long_md = config.long_md
    if (
        len(slots) < 2
        or arena is None
        or arena.tasks != ("diatomics",)
        or long_md is None
    ):
        return None
    arena_dir = Path(arena.output_dir) / f"step-{step}"
    checkpoint = arena_dir / "model.nqx"
    save_model(checkpoint, model, metadata)
    calculator_kwargs = {"model_path": str(checkpoint), "use_kernel": kernel}
    elements = arena.elements
    if elements is None:
        elements = tuple(chemical_symbols[number] for number in metadata.atomic_numbers)
    task_dir = arena_dir / "diatomics"
    md_dir = Path(long_md.output_dir) / f"step-{step}"
    md_config = replace(long_md, output_dir=str(md_dir))
    md_payloads = _long_md_payloads(md_config, calculator_kwargs, slots)
    payloads = _diatomics_payloads(elements, calculator_kwargs, slots, task_dir) + md_payloads
    started = time.perf_counter()
    processes = _spawn_evaluation_workers(payloads, md_dir / "workers")
    return EvaluationWave(step, processes, arena_dir, task_dir, md_dir, md_payloads, started)


def run_model_evaluations(
    model: Any,
    metadata: ModelMetadata,
    config: BenchmarkConfig,
    *,
    kernel: bool,
    step: int,
) -> dict[str, float]:
    """Save an EMA snapshot and run all configured downstream benchmarks."""
    wave = launch_model_evaluations(model, metadata, config, kernel=kernel, step=step)
    if wave is not None:
        return wave.wait()
    metrics: dict[str, float] = {}
    slots = cuda_device_ids() * config.workers_per_gpu
    arena = config.mlip_arena
    long_md = config.long_md
    if arena is not None:
        arena_dir = Path(arena.output_dir) / f"step-{step}"
        checkpoint = arena_dir / "model.nqx"
        save_model(checkpoint, model, metadata)
        elements = arena.elements
        if elements is None:
            elements = tuple(chemical_symbols[number] for number in metadata.atomic_numbers)
        arena_config = replace(arena, output_dir=str(arena_dir), elements=elements)
        arena_metrics = run_mlip_arena_evaluation(
            arena_config,
            calculator_kwargs={"model_path": checkpoint, "use_kernel": kernel},
            gpus=slots,
        )
        metrics.update({f"mlip_arena/{key}": value for key, value in arena_metrics.items()})

    if long_md is not None:
        md_dir = Path(long_md.output_dir) / f"step-{step}"
        checkpoint = md_dir / "model.nqx"
        save_model(checkpoint, model, metadata)
        md_config = replace(long_md, output_dir=str(md_dir))
        if len(slots) > 1:
            md_metrics = run_long_md_evaluation_parallel(
                md_config,
                calculator_kwargs={"model_path": checkpoint, "use_kernel": kernel},
                gpus=slots,
            )
        else:
            calculator_instance = NequixCalculator(checkpoint, use_kernel=kernel)
            md_metrics = run_long_md_evaluation(md_config, calculator_instance)
        metrics.update({f"long_md/{key}": value for key, value in md_metrics.items()})
    return metrics


def _worker_main(payload_path: str) -> None:
    payload = json.loads(Path(payload_path).read_text())
    calculator_kwargs = payload["calculator_kwargs"]
    if payload["kind"] == "diatomics":
        from mlip_arena.flows.diatomics import homonuclear_diatomic

        for element in payload["elements"]:
            homonuclear_diatomic.fn(
                symbol=element,
                calculator=NequixCalculator,
                calculator_kwargs=calculator_kwargs,
                out_dir=Path(payload["out_dir"]),
            )
    else:
        config = LongMDEvalConfig(
            **{**payload["config"], "tm23_regimes": tuple(payload["config"]["tm23_regimes"])}
        )
        names = set(payload["systems"])
        systems = [
            (name, read(path), temperature)
            for name, path, temperature in _long_md_specifications(config)
            if name in names
        ]
        run_long_md_evaluation(config, NequixCalculator(**calculator_kwargs), systems=systems)


if __name__ == "__main__":
    _worker_main(sys.argv[1])
