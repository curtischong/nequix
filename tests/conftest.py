import jax
import pytest

from nequix.config import ModelMetadata, NequixConfig
from nequix.model import Nequix, save_model


@pytest.fixture(scope="session")
def model_metadata():
    return ModelMetadata(
        atomic_numbers=(1, 6, 8, 14),
        atom_energies=(-1.0, -2.0, -3.0, -4.0),
        shift=0.1,
        scale=0.8,
        avg_n_neighbors=4.0,
        model_config=NequixConfig(
            cutoff=4.0,
            hidden_irreps="8x0e + 8x1o",
            lmax=1,
            n_layers=2,
            radial_basis_size=4,
            radial_mlp_size=8,
            radial_mlp_layers=2,
            radial_polynomial_p=2.0,
            mlp_init_scale=4.0,
            index_weights=False,
            layer_norm=True,
        ),
    )


@pytest.fixture(scope="session")
def jax_model_path(tmp_path_factory, model_metadata):
    config = model_metadata.model_config
    model = Nequix(
        jax.random.key(0),
        n_species=len(model_metadata.atomic_numbers),
        cutoff=config.cutoff,
        hidden_irreps=config.hidden_irreps,
        lmax=config.lmax,
        n_layers=config.n_layers,
        radial_basis_size=config.radial_basis_size,
        radial_mlp_size=config.radial_mlp_size,
        radial_mlp_layers=config.radial_mlp_layers,
        radial_polynomial_p=config.radial_polynomial_p,
        mlp_init_scale=config.mlp_init_scale,
        index_weights=config.index_weights,
        layer_norm=config.layer_norm,
        shift=model_metadata.shift,
        scale=model_metadata.scale,
        avg_n_neighbors=model_metadata.avg_n_neighbors,
        atom_energies=model_metadata.atom_energies,
    )
    path = tmp_path_factory.mktemp("models") / "fresh.nqx"
    save_model(path, model, model_metadata)
    return path
