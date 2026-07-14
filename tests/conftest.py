import pytest

from nequix.config import ModelMetadata, NequixConfig
from nequix.model import model_from_metadata, save_model


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
    model = model_from_metadata(model_metadata)
    path = tmp_path_factory.mktemp("models") / "fresh.nqx"
    save_model(path, model, model_metadata)
    return path
