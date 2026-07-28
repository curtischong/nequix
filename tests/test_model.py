import tempfile

import cloudpickle
import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from nequix.layer_norm import RMSLayerNorm
from nequix.config import ModelMetadata, NequixConfig
from nequix.model import (
    DirectForceNequix,
    LoRAE3nnLinear,
    LoRALinear,
    Nequix,
    apply_lora,
    conservative_backbone,
    load_model,
    merge_lora,
    model_from_metadata,
    replace_normalization,
    save_model,
    weight_decay_mask,
)
from nequix.train import load_training_state


def dummy_graph():
    return jraph.GraphsTuple(
        n_node=np.array([3]),
        n_edge=np.array([3]),
        nodes={
            "species": np.array([0, 1, 0], dtype=np.int32),
            "positions": np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            ).astype(np.float32),
            "forces": np.ones((3, 3), dtype=np.float32),
        },
        edges={
            "shifts": np.zeros((3, 3), dtype=np.float32),
        },
        senders=jnp.array([0, 1, 2]),
        receivers=jnp.array([1, 2, 0]),
        globals={"energy": np.ones((1,)), "cell": np.eye(3)[None, ...]},
    )


def test_model():
    key = jax.random.key(0)

    # small model for testing
    model = Nequix(
        key,
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
    )

    batch = dummy_graph()
    batch_padded = jraph.pad_with_graphs(batch, n_node=4, n_edge=4, n_graph=2)
    energy, forces, stress = model(batch_padded)
    assert energy.shape == batch_padded.globals["energy"].shape
    assert forces.shape == batch_padded.nodes["forces"].shape
    assert stress.shape == batch_padded.globals["cell"].shape

    energy, forces, stress = eqx.filter_jit(model)(batch_padded)
    assert energy.shape == batch_padded.globals["energy"].shape
    assert forces.shape == batch_padded.nodes["forces"].shape
    assert stress.shape == batch_padded.globals["cell"].shape

    batch = jraph.batch_np([batch, batch])
    batch_padded = jraph.pad_with_graphs(batch, n_node=7, n_edge=7, n_graph=3)
    energy, forces, stress = model(batch_padded)
    assert energy.shape == batch_padded.globals["energy"].shape
    assert forces.shape == batch_padded.nodes["forces"].shape
    assert stress.shape == batch_padded.globals["cell"].shape

    # check that stress is None if cell is None
    batch = dummy_graph()
    batch = batch._replace(globals={**batch.globals, "cell": None})
    batch_padded = jraph.pad_with_graphs(batch, n_node=4, n_edge=4, n_graph=2)
    energy, forces, stress = model(batch_padded)
    assert energy.shape == batch_padded.globals["energy"].shape
    assert forces.shape == batch_padded.nodes["forces"].shape
    assert stress is None


def test_direct_force_training_head_reuses_conservative_backbone():
    hidden_irreps = "8x0e+8x1o"
    backbone = Nequix(
        jax.random.key(0),
        n_species=2,
        lmax=1,
        hidden_irreps=hidden_irreps,
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
    )
    model = DirectForceNequix(backbone, hidden_irreps, key=jax.random.key(1))
    batch = jraph.pad_with_graphs(dummy_graph(), n_node=4, n_edge=4, n_graph=2)

    energy, forces, stress = eqx.filter_jit(model)(batch)

    assert energy.shape == batch.globals["energy"].shape
    assert forces.shape == batch.nodes["forces"].shape
    assert stress is None
    assert conservative_backbone(model) is backbone


def test_lora():
    model = Nequix(
        jax.random.key(0),
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
    )
    batch = jraph.pad_with_graphs(dummy_graph(), n_node=4, n_edge=4, n_graph=2)
    wrapped = apply_lora(model, rank=4, key=jax.random.key(1))

    def is_lora(x):
        return isinstance(x, (LoRALinear, LoRAE3nnLinear))

    # zero-initialized deltas leave the pre-trained model unchanged
    energy, forces, stress = model(batch)
    lora_energy, lora_forces, lora_stress = wrapped(batch)
    assert jnp.allclose(energy, lora_energy, atol=1e-6)
    assert jnp.allclose(forces, lora_forces, atol=1e-6)
    assert jnp.allclose(stress, lora_stress, atol=1e-6)

    # gradients reach the adapters but never the frozen base weights
    def loss_fn(m):
        e, f, _ = m(batch)
        return jnp.sum(e**2) + jnp.sum(f**2)

    grads = eqx.filter_grad(loss_fn)(wrapped)
    lora_b_grads = []
    for leaf in jax.tree.flatten(grads, is_leaf=is_lora)[0]:
        if is_lora(leaf):
            assert all(
                jnp.all(g == 0) for g in jax.tree.leaves(eqx.filter(leaf.base, eqx.is_array))
            )
            lora_b_grads.extend(jax.tree.leaves(leaf.lora_b))
    assert any(jnp.any(g != 0) for g in lora_b_grads)

    # weight decay only targets the adapters inside LoRA wrappers
    mask = weight_decay_mask(wrapped)
    assert all(jax.tree.leaves(mask.layers[0].linear_1.lora_a))
    assert not any(jax.tree.leaves(mask.layers[0].linear_1.base))
    assert mask.layers[0].radial_mlp.layers[0].lora_a
    assert not any(jax.tree.leaves(mask.layers[0].radial_mlp.layers[0].base))

    # after an adapter update, merging reproduces the wrapped model as a plain Nequix
    trained = eqx.apply_updates(wrapped, jax.tree.map(lambda g: -0.01 * g, grads))
    merged = conservative_backbone(trained)
    assert isinstance(merged, Nequix)
    assert not any(is_lora(leaf) for leaf in jax.tree.flatten(merged, is_leaf=is_lora)[0])
    assert jax.tree.structure(eqx.filter(merged, eqx.is_array)) == jax.tree.structure(
        eqx.filter(model, eqx.is_array)
    )
    trained_energy, trained_forces, _ = trained(batch)
    merged_energy, merged_forces, _ = merged(batch)
    assert jnp.allclose(trained_energy, merged_energy, atol=1e-5)
    assert jnp.allclose(trained_forces, merged_forces, atol=1e-5)
    assert merge_lora(model) is model


@pytest.mark.parametrize("centering", [True, False])
def test_layer_norm(centering):
    irreps = e3nn.Irreps("8x0e + 4x1o + 2x2e")
    layer_norm = RMSLayerNorm(irreps=irreps, centering=centering, std_balance_degrees=True)
    x = jax.random.normal(jax.random.key(0), (5, irreps.dim))
    node_input = e3nn.IrrepsArray(irreps, x)
    output = layer_norm(node_input)
    assert output.irreps == node_input.irreps
    assert output.shape == node_input.shape


def test_model_save_load():
    key = jax.random.key(0)

    # Create config for model initialization
    config = {
        "cutoff": 6.0,
        "atomic_numbers": [1, 6],  # H and C
        "hidden_irreps": "8x0e+8x1o",
        "lmax": 1,
        "n_layers": 2,
        "radial_basis_size": 4,
        "radial_mlp_size": 8,
        "radial_mlp_layers": 2,
        "radial_polynomial_p": 2.0,
        "mlp_init_scale": 4.0,
        "index_weights": True,
        "layer_norm": False,
        "shift": 1.0,
        "scale": 2.0,
        "avg_n_neighbors": 2.0,
        "atom_energies": {1: 1.0, 6: 2.0},
    }

    # Create model using config parameters
    atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    metadata = ModelMetadata(
        atomic_numbers=tuple(config["atomic_numbers"]),
        atom_energies=tuple(atom_energies),
        shift=config["shift"],
        scale=config["scale"],
        avg_n_neighbors=config["avg_n_neighbors"],
        model_config=NequixConfig(
            cutoff=config["cutoff"],
            hidden_irreps=config["hidden_irreps"],
            lmax=config["lmax"],
            n_layers=config["n_layers"],
            radial_basis_size=config["radial_basis_size"],
            radial_mlp_size=config["radial_mlp_size"],
            radial_mlp_layers=config["radial_mlp_layers"],
            radial_polynomial_p=config["radial_polynomial_p"],
            mlp_init_scale=config["mlp_init_scale"],
            index_weights=config["index_weights"],
            layer_norm=config["layer_norm"],
        ),
    )
    model = model_from_metadata(metadata, key=key)

    batch = dummy_graph()
    batch_padded = jraph.pad_with_graphs(batch, n_node=4, n_edge=3, n_graph=2)

    original_energy, original_forces, original_stress = model(batch_padded)

    with tempfile.NamedTemporaryFile(suffix=".eqx") as tmp_file:
        save_model(tmp_file.name, model, metadata)
        loaded_model, loaded_metadata = load_model(tmp_file.name)
        assert loaded_metadata == metadata
        assert model.lmax == loaded_model.lmax
        assert model.cutoff == loaded_model.cutoff
        assert model.n_species == loaded_model.n_species
        assert model.radial_basis_size == loaded_model.radial_basis_size
        assert model.radial_polynomial_p == loaded_model.radial_polynomial_p
        assert model.shift == loaded_model.shift
        assert model.scale == loaded_model.scale
        assert model.layers[0].avg_n_neighbors == loaded_model.layers[0].avg_n_neighbors
        np.testing.assert_allclose(model.atom_energies, loaded_model.atom_energies)
        loaded_energy, loaded_forces, loaded_stress = loaded_model(batch_padded)
        np.testing.assert_allclose(original_energy, loaded_energy)
        np.testing.assert_allclose(original_forces, loaded_forces)
        np.testing.assert_allclose(original_stress, loaded_stress)


def test_model_loader_rejects_flat_legacy_header(tmp_path):
    path = tmp_path / "legacy.nqx"
    path.write_bytes(b'{"atomic_numbers": [1], "cutoff": 5.0}\n')

    with pytest.raises(ValueError, match="invalid Nequix model header"):
        load_model(path)


def test_training_state_loader_rejects_unversioned_state(tmp_path):
    path = tmp_path / "unversioned.pkl"
    with path.open("wb") as state_file:
        cloudpickle.dump({}, state_file)

    with pytest.raises(ValueError, match="invalid Nequix training state"):
        load_training_state(path)


def test_weight_decay_mask():
    key = jax.random.key(0)
    model = Nequix(
        key,
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
        layer_norm=True,
    )
    mask = weight_decay_mask(model)
    # should be weight decay on e3nn linear layers and normal linear weights
    assert all(mask.layers[0].linear_1._weights.values())
    assert mask.layers[0].radial_mlp.layers[0].weights

    # no weight decay on atom energies, biases, or layer norms
    assert not mask.atom_energies
    assert not mask.layers[0].radial_mlp.layers[0].bias
    assert not any(mask.layers[0].layer_norm.affine_weight)
    assert not mask.layers[0].layer_norm.affine_bias


def test_replace_normalization_preserves_weights():
    model = Nequix(
        jax.random.key(0),
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=1,
        shift=1.0,
        scale=2.0,
        atom_energies=[3.0, 4.0],
    )

    updated = replace_normalization(
        model,
        atom_energies=[5.0, 6.0],
        shift=7.0,
        scale=8.0,
    )

    assert model.shift == 1.0
    assert model.scale == 2.0
    np.testing.assert_allclose(model.atom_energies, [3.0, 4.0])
    assert updated.shift == 7.0
    assert updated.scale == 8.0
    np.testing.assert_allclose(updated.atom_energies, [5.0, 6.0])
    assert jax.tree.all(
        jax.tree.map(
            lambda old, new: jnp.array_equal(old, new),
            eqx.filter(model.layers, eqx.is_array),
            eqx.filter(updated.layers, eqx.is_array),
        )
    )
