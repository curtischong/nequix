from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import cloudpickle
import jax
import jaxlib

from nequix.config import TrainerConfig
from nequix.data import padded_shape

_CACHE_SCHEMA = 2
_OOM_MARKERS = (
    "out of memory",
    "resource_exhausted",
    "resource exhausted",
    "cuda_error_out_of_memory",
    "failed to allocate",
)


@dataclass(frozen=True)
class BatchShape:
    batch_size: int
    n_graph: int
    n_node: int
    n_edge: int


@dataclass(frozen=True)
class ProbeResult:
    shape: BatchShape
    status: str
    graphs_per_second: float = 0.0
    nodes_per_second: float = 0.0
    edges_per_second: float = 0.0
    graph_utilization: float = 0.0
    node_utilization: float = 0.0
    edge_utilization: float = 0.0
    peak_memory_bytes: int = 0
    final_loss: float | None = None
    timed_graphs_per_second: tuple[float, ...] = ()
    error: str | None = None

    @property
    def safe(self):
        return self.status == "ok"


@dataclass(frozen=True)
class TuneResult:
    shape: BatchShape
    probes: tuple[ProbeResult, ...] = ()
    cached: bool = False
    warning: str | None = None
    cache_key: str | None = None


def batch_shape(batch_size: int, stats: dict, buffer_factor: float = 1.1) -> BatchShape:
    """Return the padded shape a DataLoader derives for an autobatch candidate."""
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    n_graph, n_node, n_edge = padded_shape(
        batch_size,
        stats["avg_n_nodes"],
        stats["avg_n_edges"],
        stats["max_n_nodes"],
        stats["max_n_edges"],
        buffer_factor,
    )
    return BatchShape(batch_size=batch_size, n_graph=n_graph, n_node=n_node, n_edge=n_edge)


def peak_device_memory_bytes() -> int:
    """Return the highest peak-memory statistic reported by any JAX device."""
    peaks = []
    for device in jax.devices():
        try:
            memory = device.memory_stats() or {}
        except (AttributeError, RuntimeError):
            continue
        for key in ("peak_bytes_in_use", "peak_bytes_in_use_limit", "bytes_in_use"):
            if key in memory:
                peaks.append(int(memory[key]))
                break
    return max(peaks, default=0)


def query_gpu_hardware() -> dict:
    """Query GPU identity without initializing JAX's accelerator backend."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"gpus": [], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    gpus = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        name, memory_mib, driver = fields
        try:
            memory_bytes = int(memory_mib) * 1024**2
        except ValueError:
            continue
        gpus.append({"name": name, "memory_bytes": memory_bytes, "driver": driver})

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() not in {"", "-1"}:
        identifiers = [identifier.strip() for identifier in visible.split(",")]
        if all(identifier.isdigit() for identifier in identifiers):
            gpus = [
                gpus[int(identifier)] for identifier in identifiers if int(identifier) < len(gpus)
            ]
        else:
            # UUID/MIG masks cannot be matched to this nvidia-smi query reliably,
            # but their count still describes how many devices the child sees.
            gpus = gpus[: len(identifiers)]
    elif visible is not None:
        gpus = []
    return {"gpus": gpus, "cuda_visible_devices": visible}


def autobatch_cache_key(config: TrainerConfig, dataset_size: int, hardware: dict) -> str:
    """Build a stable key for every input that changes capacity or train graphs."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "hardware": hardware,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "os_kernel": platform.release(),
        "model": {"atomic_numbers": config.atomic_numbers, **asdict(config.model_config)},
        "training_kernel": {
            "enabled": config.kernel,
            "openequivariance": _package_version("openequivariance"),
            "openequivariance_extjax": _package_version("openequivariance-extjax"),
        },
        "force_mode": config.force_mode,
        "optimizer": config.optimizer,
        "optimizer_config": {
            "grad_clip_norm": config.grad_clip_norm,
            "weight_decay": config.weight_decay,
            "ema_decay": config.ema_decay,
        },
        "loss": {
            "energy_weight": config.energy_weight,
            "force_weight": config.force_weight,
            "stress_weight": config.stress_weight,
            "loss_type": config.loss_type,
        },
        "dataset_size": int(dataset_size),
        "dataset_stats": config.dataset_stats(),
        "dataset_config": {
            "train_path": config.train_path,
            "train_frac": config.train_frac,
            "seed": config.seed,
        },
        "autobatch_memory_scaling_factor": config.autobatch_memory_scaling_factor,
        "autobatch_minimum_speedup": config.autobatch_minimum_speedup,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def default_cache_path() -> Path:
    override = os.environ.get("NEQUIX_AUTOBATCH_CACHE")
    if override:
        return Path(override).expanduser()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "nequix" / "autobatch-v2.json"


def _load_cache(path: Path) -> dict:
    try:
        contents = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"schema": _CACHE_SCHEMA, "entries": {}}
    if contents.get("schema") != _CACHE_SCHEMA or not isinstance(contents.get("entries"), dict):
        return {"schema": _CACHE_SCHEMA, "entries": {}}
    return contents


def _write_cache(path: Path, contents: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as cache_file:
        json.dump(contents, cache_file, sort_keys=True, indent=2)
        temporary_path = Path(cache_file.name)
    temporary_path.replace(path)


def _shape_from_dict(values: dict) -> BatchShape:
    return BatchShape(**{key: int(value) for key, value in values.items()})


def _probe_from_dict(values: dict) -> ProbeResult:
    values = dict(values)
    values["shape"] = _shape_from_dict(values["shape"])
    values["timed_graphs_per_second"] = tuple(values["timed_graphs_per_second"])
    return ProbeResult(**values)


def cached_tune_result(path: Path, key: str) -> TuneResult | None:
    entry = _load_cache(path)["entries"].get(key)
    if not isinstance(entry, dict):
        return None
    try:
        shape = _shape_from_dict(entry["shape"])
        probes = tuple(_probe_from_dict(probe) for probe in entry["probes"])
    except (KeyError, TypeError, ValueError):
        return None
    return TuneResult(
        shape=shape,
        probes=probes,
        cached=True,
        warning=entry["warning"],
        cache_key=key,
    )


def cache_tune_result(path: Path, key: str, result: TuneResult):
    contents = _load_cache(path)
    contents["entries"][key] = {
        "shape": asdict(result.shape),
        "probes": [asdict(probe) for probe in result.probes],
        "warning": result.warning,
    }
    _write_cache(path, contents)


def tune_batch_shape(
    stats: dict,
    dataset_size: int,
    device_count: int,
    probe: Callable[[BatchShape], ProbeResult],
    *,
    memory_scaling_factor: float = 1.6,
    minimum_speedup: float = 0.02,
) -> TuneResult:
    """Probe a safe capacity range and choose measured graph throughput."""
    if not math.isfinite(memory_scaling_factor) or memory_scaling_factor <= 1:
        raise ValueError("autobatch_memory_scaling_factor must be greater than one")

    initial_shape = batch_shape(1, stats)
    probes = []

    def run(candidate_batch_size):
        shape = batch_shape(candidate_batch_size, stats)
        try:
            result = probe(shape)
        except Exception as error:  # a broken probe must never break real training
            result = ProbeResult(shape=shape, status="failed", error=str(error))
        probes.append(result)
        return result

    def fallback(warning):
        return TuneResult(shape=initial_shape, probes=tuple(probes), warning=warning)

    baseline = run(initial_shape.batch_size)
    if not baseline.safe:
        return fallback(
            "autobatch probing failed at the minimum capacity; using a single-example capacity"
        )

    per_device_examples = max(1, math.ceil(dataset_size / max(device_count, 1)))
    upper_limit = per_device_examples
    last_safe = baseline
    first_unsafe = None
    candidate = initial_shape.batch_size
    while candidate < upper_limit:
        candidate = min(
            max(round(candidate * memory_scaling_factor), candidate + 1),
            upper_limit,
        )
        result = run(candidate)
        if result.status == "failed":
            return fallback("autobatch probing failed; using a single-example capacity")
        if result.safe:
            last_safe = result
        else:
            first_unsafe = result
            break

    # Resolve the safe boundary without compiling every integer batch size.
    if first_unsafe is not None:
        low = last_safe.shape.batch_size
        high = first_unsafe.shape.batch_size
        for _ in range(3):
            if high - low <= max(2, initial_shape.batch_size // 8):
                break
            middle = (low + high) // 2
            result = run(middle)
            if result.status == "failed":
                return fallback("autobatch probing failed; using a single-example capacity")
            if result.safe:
                low = middle
            else:
                high = middle

    safe = [result for result in probes if result.safe]
    best = max(safe, key=lambda result: result.graphs_per_second)
    if best.graphs_per_second <= baseline.graphs_per_second * (1 + minimum_speedup):
        return fallback(
            "no autobatch candidate was measurably faster than one example; "
            "using a single-example capacity"
        )
    return TuneResult(shape=best.shape, probes=tuple(probes))


def _looks_like_oom(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _OOM_MARKERS)


def subprocess_probe(payload_path: Path, shape: BatchShape, timeout: float = 1800) -> ProbeResult:
    """Run one real compiled candidate in a disposable Python process."""
    with tempfile.TemporaryDirectory(prefix="nequix-autobatch-") as directory:
        directory = Path(directory)
        shape_path = directory / "shape.json"
        result_path = directory / "result.json"
        shape_path.write_text(json.dumps(asdict(shape)))
        command = [
            sys.executable,
            "-m",
            "nequix.autobatch_probe",
            str(payload_path),
            str(shape_path),
            str(result_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            return ProbeResult(shape=shape, status="failed", error=f"probe timed out: {error}")

        if result_path.exists():
            try:
                return _probe_from_dict(json.loads(result_path.read_text()))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
                return ProbeResult(
                    shape=shape, status="failed", error=f"invalid probe result: {error}"
                )

        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        status = "oom" if _looks_like_oom(output) or completed.returncode in {-9, 137} else "failed"
        return ProbeResult(
            shape=shape,
            status=status,
            error=output[-2000:] or f"probe exited with status {completed.returncode}",
        )


def tune_training_batch(config: TrainerConfig, train_dataset) -> TuneResult:
    """Load or measure the fixed batch shape for a standard JAX training run."""
    stats = config.dataset_stats()
    hardware = query_gpu_hardware()
    if not hardware["gpus"]:
        message = "autobatch could not identify an NVIDIA GPU; using a single-example capacity"
        warnings.warn(message, RuntimeWarning)
        return TuneResult(shape=batch_shape(1, stats), warning=message)

    key = autobatch_cache_key(config, len(train_dataset), hardware)
    cache_path = default_cache_path()
    cached = cached_tune_result(cache_path, key)
    if cached is not None:
        if cached.warning:
            warnings.warn(cached.warning, RuntimeWarning)
        return cached

    # The payload is identical for every candidate, so it is serialized once.
    with tempfile.TemporaryDirectory(prefix="nequix-autobatch-") as directory:
        payload_path = Path(directory) / "payload.pkl"
        with payload_path.open("wb") as payload_file:
            cloudpickle.dump({"config": config, "train_dataset": train_dataset}, payload_file)
        result = tune_batch_shape(
            stats,
            len(train_dataset),
            len(hardware["gpus"]),
            lambda shape: subprocess_probe(payload_path, shape),
            memory_scaling_factor=float(config.autobatch_memory_scaling_factor),
            minimum_speedup=float(config.autobatch_minimum_speedup),
        )
    result = replace(result, cache_key=key)
    if result.warning:
        warnings.warn(result.warning, RuntimeWarning)
    if not any(probe.status == "failed" for probe in result.probes):
        cache_tune_result(cache_path, key, result)
    return result


def probe_summary(result: TuneResult) -> str:
    source = "cache" if result.cached else "probes"
    selected = result.shape
    lines = [
        f"autobatch selected from {source}: n_graph={selected.n_graph}, "
        f"n_node={selected.n_node}, n_edge={selected.n_edge}"
    ]
    for probe in result.probes:
        lines.append(
            f"  batch_size={probe.shape.batch_size} status={probe.status} "
            f"graphs/s={probe.graphs_per_second:.3f} nodes/s={probe.nodes_per_second:.3f} "
            f"edges/s={probe.edges_per_second:.3f} peak_memory={probe.peak_memory_bytes} "
            f"final_loss={probe.final_loss}"
        )
    if result.warning:
        lines.append(f"  warning: {result.warning}")
    return "\n".join(lines)
