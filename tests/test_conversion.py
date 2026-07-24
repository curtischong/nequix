import numpy as np
import torch

from nequix.model import load_model as load_model_jax
from nequix.model import save_model as save_model_jax
from nequix.torch_impl.model import load_model as load_model_torch
from nequix.torch_impl.model import save_model as save_model_torch
from nequix.torch_impl.utils import convert_model_jax_to_torch, convert_model_torch_to_jax
from tests.test_model import dummy_graph as dummy_graph_jax
from tests.torch_impl.test_model_torch import dummy_graph as dummy_graph_torch


def test_conversion(jax_model_path, tmp_path):
    jax_model, metadata = load_model_jax(jax_model_path)
    torch_model, torch_metadata = convert_model_jax_to_torch(
        jax_model, metadata, use_kernel=False
    )

    torch_path = tmp_path / "converted.pt"
    save_model_torch(torch_path, torch_model, torch_metadata)

    torch_model_loaded, torch_metadata_loaded = load_model_torch(torch_path)
    jax_model_converted, jax_metadata_converted = convert_model_torch_to_jax(
        torch_model_loaded, torch_metadata_loaded, use_kernel=False
    )

    jax_path = tmp_path / "converted.nqx"
    save_model_jax(jax_path, jax_model_converted, jax_metadata_converted)

    jax_model_loaded, jax_metadata_loaded = load_model_jax(jax_path)
    torch_model_converted, torch_metadata_converted = convert_model_jax_to_torch(
        jax_model_loaded, jax_metadata_loaded, use_kernel=False
    )
    assert torch_metadata_converted == metadata

    graph_jax = dummy_graph_jax()

    energy_jax, forces_jax, stress_jax = jax_model(graph_jax)
    energy_converted_jax, forces_converted_jax, stress_converted_jax = jax_model_converted(
        graph_jax
    )

    np.testing.assert_allclose(energy_jax, energy_converted_jax)
    np.testing.assert_allclose(forces_jax, forces_converted_jax)
    np.testing.assert_allclose(stress_jax, stress_converted_jax)

    graph_torch = dummy_graph_torch()
    energy_torch, forces_torch, stress_torch = torch_model_loaded(
        graph_torch.x,
        graph_torch.positions,
        graph_torch.edge_attr,
        graph_torch.edge_index,
        graph_torch.cell,
        graph_torch.n_node,
        graph_torch.n_edge,
        torch.zeros(graph_torch.x.shape[0], dtype=torch.int64),
    )
    energy_converted_torch, forces_converted_torch, stress_converted_torch = torch_model_converted(
        graph_torch.x,
        graph_torch.positions,
        graph_torch.edge_attr,
        graph_torch.edge_index,
        graph_torch.cell,
        graph_torch.n_node,
        graph_torch.n_edge,
        torch.zeros(graph_torch.x.shape[0], dtype=torch.int64),
    )

    np.testing.assert_allclose(
        energy_torch.detach().numpy(), energy_converted_torch.detach().numpy()
    )
    np.testing.assert_allclose(
        forces_torch.detach().numpy(), forces_converted_torch.detach().numpy()
    )
    np.testing.assert_allclose(
        stress_torch.detach().numpy(), stress_converted_torch.detach().numpy()
    )
