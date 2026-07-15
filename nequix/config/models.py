from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

MODEL_FORMAT = "nequix-model-v1"

ATOMIC_NUMBERS = tuple(range(1, 84)) + tuple(range(89, 95))


ATOM_ENERGIES_DIR = Path(__file__).parent / "atom_energies"


def load_atom_energies(name: str) -> dict[int, float]:
    """Per-dataset isolated-atom energies, computed by scripts/compute_atom_energies.py."""
    values = yaml.safe_load((ATOM_ENERGIES_DIR / f"{name}.yml").read_text())
    if sorted(values) != sorted(ATOMIC_NUMBERS):
        raise ValueError("one isolated-atom energy is required for every configured element")
    return {int(number): float(energy) for number, energy in values.items()}


MP_ATOM_ENERGIES = load_atom_energies("mp")
OMAT_ATOM_ENERGIES = load_atom_energies("omat")
OAM_ATOM_ENERGIES = load_atom_energies("oam")


@dataclass
class NequixConfig:
    cutoff: float = 6.0
    hidden_irreps: str = "128x0e + 64x1o + 32x2e + 32x3o"
    lmax: int = 3
    n_layers: int = 4
    radial_basis_size: int = 8
    radial_mlp_size: int = 64
    radial_mlp_layers: int = 2
    radial_polynomial_p: float = 6.0
    mlp_init_scale: float = 4.0
    index_weights: bool = False
    layer_norm: bool = True


@dataclass(frozen=True)
class ModelMetadata:
    """The complete, backend-independent schema stored with model weights."""

    atomic_numbers: tuple[int, ...]
    atom_energies: tuple[float, ...]
    shift: float
    scale: float
    avg_n_neighbors: float
    model_config: NequixConfig

    def __post_init__(self) -> None:
        if len(self.atomic_numbers) != len(self.atom_energies):
            raise ValueError("atomic_numbers and atom_energies must have the same length")
        if len(set(self.atomic_numbers)) != len(self.atomic_numbers):
            raise ValueError("atomic_numbers must be unique")

    def to_header(self) -> dict[str, Any]:
        return {"format": MODEL_FORMAT, "metadata": asdict(self)}

    @classmethod
    def from_header(cls, header: Any) -> ModelMetadata:
        if not isinstance(header, dict) or set(header) != {"format", "metadata"}:
            raise ValueError("invalid Nequix model header")
        if header["format"] != MODEL_FORMAT:
            raise ValueError(f"unsupported Nequix model format: {header['format']!r}")

        values = header["metadata"]
        expected = {item.name for item in fields(cls)}
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError("invalid Nequix model metadata")

        model_values = values["model_config"]
        model_expected = {item.name for item in fields(NequixConfig)}
        if not isinstance(model_values, dict) or set(model_values) != model_expected:
            raise ValueError("invalid Nequix architecture metadata")

        try:
            return cls(
                atomic_numbers=tuple(int(value) for value in values["atomic_numbers"]),
                atom_energies=tuple(float(value) for value in values["atom_energies"]),
                shift=float(values["shift"]),
                scale=float(values["scale"]),
                avg_n_neighbors=float(values["avg_n_neighbors"]),
                model_config=NequixConfig(**model_values),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("invalid Nequix model metadata") from error


@dataclass
class TrainerConfig:
    name: str
    train_path: str | tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    atom_energies: dict[int, float]
    avg_n_edges: float
    avg_n_neighbors: float
    avg_n_nodes: float
    max_n_edges: int
    max_n_nodes: int
    scale: float
    shift: float
    state_path: str
    resume_from: str
    checkpoint_path: str
    batch_size: int
    model_config: NequixConfig = field(default_factory=NequixConfig)
    kernel: bool = True
    valid_frac: float | None = None
    valid_path: str | None = None
    dataset_name: str | None = None
    train_frac: float = 1.0
    seed: int = 0
    optimizer: str = "muon"
    learning_rate: float = 0.01
    warmup_epochs: float = 0.1
    warmup_factor: float = 0.2
    grad_clip_norm: float = 100.0
    weight_decay: float = 1.0e-3
    n_epochs: int = 100
    energy_weight: float = 20.0
    force_weight: float = 20.0
    stress_weight: float = 5.0
    force_mode: Literal["conservative", "direct"] = "conservative"
    loss_type: str = "mae"
    log_every: int = 100
    val_every_steps: int | None = None
    ema_decay: float = 0.999
    finetune_from: str | None = None
    run_name: str | None = None
    wandb_entity: str = "curtischong"
    wandb_run_name: str | None = None
    wandb_project: str | None = None
    wandb_mode: str | None = None

    def dataset_stats(self) -> dict[str, float]:
        """Precomputed dataset statistics consumed by model construction and batching."""
        return {
            "shift": self.shift,
            "scale": self.scale,
            "avg_n_neighbors": self.avg_n_neighbors,
            "avg_n_nodes": self.avg_n_nodes,
            "avg_n_edges": self.avg_n_edges,
            "max_n_nodes": self.max_n_nodes,
            "max_n_edges": self.max_n_edges,
        }

    def atom_energy_list(self) -> tuple[float, ...]:
        """Isolated-atom energies ordered to match ``atomic_numbers``."""
        return tuple(self.atom_energies[number] for number in self.atomic_numbers)


@dataclass
class PFTTrainerConfig:
    name: str
    state_path: str
    resume_from: str
    finetune_from: str
    train_path: str
    val_path: str
    extra_train_path: str | tuple[str, ...]
    avg_n_edges: float
    avg_n_nodes: float
    max_n_edges: int
    max_n_nodes: int
    extra_avg_n_edges: float
    extra_avg_n_nodes: float
    extra_max_n_edges: int
    extra_max_n_nodes: int
    extra_batch_size: int
    extra_val_frac: float | None = None
    extra_val_path: str | None = None
    extra_train_steps: int = 4
    extra_energy_weight: float = 500.0
    extra_force_weight: float = 200.0
    extra_stress_weight: float = 50.0
    optimizer: str = "adam"
    learning_rate: float = 1.0e-4
    checkpoint_grad_energy: bool = False
    grad_clip_norm: float = 100.0
    weight_decay: float = 1.0e-3
    batch_size: int = 16
    n_graph: int = 18
    n_epochs: int = 150
    energy_weight: float = 0.0
    force_weight: float = 20.0
    stress_weight: float = 5.0
    hessian_weight: float = 100.0
    val_every: int = 2
    log_every: int = 100
    ema_decay: float = 0.999
    kernel: bool = True
    wandb_entity: str = "curtischong"
    wandb_project: str = "nequix-phonon"


RunConfig = TrainerConfig | PFTTrainerConfig


def config_values(config: RunConfig) -> dict[str, Any]:
    """Return the nested, JSON-friendly representation used for run logging."""

    return _plain_value(asdict(config))


def _plain_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    return value
