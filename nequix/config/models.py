from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Trainer = Literal["jax", "torch", "pft"]

ATOMIC_NUMBERS = tuple(range(1, 84)) + tuple(range(89, 95))


def _energy_map(values: tuple[float, ...]) -> dict[int, float]:
    if len(values) != len(ATOMIC_NUMBERS):
        raise ValueError("one isolated-atom energy is required for every configured element")
    return dict(zip(ATOMIC_NUMBERS, values))


MP_ATOM_ENERGIES = _energy_map(
    (
        -3.6683144569396973,
        -1.3163000345230103,
        -3.4831652641296387,
        -4.764586448669434,
        -7.723288536071777,
        -8.40385913848877,
        -7.355093955993652,
        -7.283683776855469,
        -4.895404815673828,
        -0.03005797415971756,
        -2.7473080158233643,
        -2.811192512512207,
        -4.856420993804932,
        -7.702361583709717,
        -6.968142509460449,
        -4.674358367919922,
        -2.8150434494018555,
        -0.06283307820558548,
        -2.622321128845215,
        -5.3866095542907715,
        -7.873931407928467,
        -10.264874458312988,
        -8.66543197631836,
        -9.2335205078125,
        -8.309571266174316,
        -7.048263072967529,
        -5.5792694091796875,
        -5.171901702880859,
        -3.252624273300171,
        -1.2877082824707031,
        -3.528157949447632,
        -4.709354877471924,
        -3.9754624366760254,
        -3.887714147567749,
        -2.516629219055176,
        6.747716426849365,
        -2.577591896057129,
        -4.939635276794434,
        -10.152220726013184,
        -11.842530250549316,
        -12.148360252380371,
        -8.799700736999512,
        -8.797399520874023,
        -7.760889530181885,
        -6.835111618041992,
        -4.8846940994262695,
        -2.0629477500915527,
        -0.6418149471282959,
        -2.785382032394409,
        -3.8185558319091797,
        -3.5863938331604004,
        -2.880044937133789,
        -1.6353932619094849,
        9.806041717529297,
        -2.7564780712127686,
        -4.997802257537842,
        -8.92111873626709,
        -8.735455513000488,
        -8.035662651062012,
        -8.260232925415039,
        -7.5397539138793945,
        -8.160961151123047,
        -13.596965789794922,
        -18.527551651000977,
        -7.6466827392578125,
        -8.13962459564209,
        -7.610777854919434,
        -6.8239006996154785,
        -7.813919544219971,
        -3.5790491104125977,
        -7.460397243499756,
        -12.797314643859863,
        -14.110695838928223,
        -9.345513343811035,
        -11.383716583251953,
        -9.633197784423828,
        -7.3318257331848145,
        -5.296637535095215,
        -2.371084213256836,
        0.25143736600875854,
        -2.319478988647461,
        -3.7398195266723633,
        -3.4386684894561768,
        -5.125858783721924,
        -11.000879287719727,
        -12.322521209716797,
        -13.852907180786133,
        -14.94003963470459,
        -15.294189453125,
    )
)

# OMat24 isolated-atom references from the fairchem UMA training release.
OMAT_ATOM_ENERGIES = _energy_map(
    (
        -1.11700253,
        0.00079886,
        -0.29731164,
        -0.04129868,
        -0.29106192,
        -1.27751531,
        -3.12342715,
        -1.54797136,
        -0.43969356,
        -0.01250908,
        -0.22855413,
        -0.00943179,
        -0.21707638,
        -0.82619133,
        -1.88667434,
        -0.89093583,
        -0.25816211,
        -0.02414768,
        -0.17662425,
        -0.02568319,
        -2.13001165,
        -2.38688845,
        -3.55934233,
        -5.44700879,
        -5.14749562,
        -3.30662847,
        -1.42167737,
        -0.63181379,
        -0.23449167,
        -0.01146636,
        -0.21291259,
        -0.77939897,
        -1.70148487,
        -0.78386705,
        -0.22690657,
        -0.02245409,
        -0.16092396,
        -0.02798717,
        -2.25685695,
        -2.23690495,
        -2.15347771,
        -4.60251809,
        -3.36416792,
        -2.23062607,
        -1.15550917,
        -1.47553527,
        -0.19918102,
        -0.01475888,
        -0.19767692,
        -0.68005773,
        -1.43073368,
        -0.65790462,
        -0.18915279,
        -0.01179476,
        -0.13507902,
        -0.03056979,
        -0.36017439,
        -0.86279246,
        -0.20573327,
        -0.2734463,
        -0.20046965,
        -0.25444338,
        -8.37972664,
        -9.58424928,
        -0.19466184,
        -0.24860115,
        -0.19531288,
        -0.15401392,
        -0.14577898,
        -0.19655747,
        -0.15645898,
        -3.49380556,
        -3.5317097,
        -4.57108006,
        -4.63425205,
        -2.88247063,
        -1.45679675,
        -0.50290184,
        -0.18521704,
        -0.01123956,
        -0.17483649,
        -0.63132037,
        -1.3248562,
        -0.24135757,
        -1.04601971,
        -2.04574044,
        -3.84544799,
        -7.28626119,
        -7.3136314,
    )
)

# Isolated-atom references used by the combined OMat/sAlex/MPtrj run.
OAM_ATOM_ENERGIES = _energy_map(
    (
        -1.1176,
        -0.0005,
        -0.2974,
        -0.0181,
        -0.4447,
        -1.3865,
        -3.1256,
        -1.9067,
        -0.7674,
        -0.0121,
        -0.2285,
        -0.0958,
        -0.3122,
        -0.8689,
        -1.8879,
        -1.0746,
        -0.3714,
        -0.0502,
        -0.2277,
        -0.0927,
        -2.2127,
        -2.6397,
        -3.7438,
        -5.6018,
        -5.3235,
        -3.5955,
        -2.1496,
        -1.0536,
        -0.6027,
        -0.1645,
        -0.4043,
        -0.8916,
        -1.6834,
        -0.8716,
        -0.2651,
        -0.0331,
        -0.1879,
        -0.068,
        -2.2868,
        -2.3603,
        -3.1513,
        -4.6011,
        -3.5438,
        -1.6595,
        -1.6479,
        -1.4776,
        -0.3388,
        -0.1672,
        -0.4087,
        -0.8167,
        -1.4107,
        -0.7239,
        -0.1703,
        -0.0097,
        -0.1369,
        -0.0344,
        -0.8455,
        -1.3876,
        -0.5491,
        -0.5186,
        -0.4895,
        -0.4683,
        -8.3662,
        -10.4088,
        -0.3982,
        -0.3886,
        -0.3834,
        -0.3857,
        -0.3168,
        -0.064,
        -0.3808,
        -3.527,
        -3.7421,
        -4.6555,
        -3.4276,
        -2.8979,
        -1.1789,
        -0.5638,
        -0.2872,
        -0.1235,
        -0.3606,
        -0.7674,
        -1.326,
        -0.3866,
        -1.1045,
        -2.553,
        -4.9889,
        -7.7017,
        -10.8084,
    )
)


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
    kernel: bool = True


@dataclass
class TrainerConfig:
    name: str
    trainer: Literal["jax", "torch"]
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
    model_config: NequixConfig = field(default_factory=NequixConfig)
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
    autobatch_memory_scaling_factor: float = 1.6
    n_epochs: int = 100
    energy_weight: float = 20.0
    force_weight: float = 20.0
    stress_weight: float = 5.0
    force_mode: Literal["conservative", "direct"] = "conservative"
    loss_type: str = "mae"
    log_every: int = 100
    val_every_steps: int | None = None
    ema_decay: float = 0.999
    state_path: str | None = None
    resume_from: str | None = None
    finetune_from: str | None = None
    checkpoint_path: str | None = None
    cache_dir: str | None = None
    run_name: str | None = None
    wandb_run_name: str | None = None
    wandb_project: str | None = None
    wandb_mode: str | None = None


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
    trainer: Literal["pft"] = field(default="pft", init=False)
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


RunConfig = TrainerConfig | PFTTrainerConfig


def _plain_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    return value


def config_dict(config: RunConfig) -> dict[str, Any]:
    """Flatten a named Python config into the legacy trainer dictionary shape."""
    values = asdict(config)
    values.pop("name")
    values.pop("trainer")
    model_config = values.pop("model_config", None)
    if model_config is not None:
        values.update(model_config)
    return {key: _plain_value(value) for key, value in values.items() if value is not None}
