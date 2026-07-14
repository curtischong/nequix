from pathlib import Path

import equinox as eqx
import jraph
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequix.data import (
    atomic_numbers_to_indices,
    dict_to_graphstuple,
    dict_to_pytorch_geometric,
    preprocess_graph,
)
from nequix.model import load_model as load_model_jax
from nequix.pft.hessian import hessian_linearized


def model_path_backend(model_path: Path) -> str:
    """Map a checkpoint path to the backend its suffix denotes."""
    try:
        return {".nqx": "jax", ".pt": "torch"}[model_path.suffix]
    except KeyError as error:
        raise ValueError("model checkpoints must use a .nqx or .pt extension") from error


def load_model_for_backend(
    model_path: str | Path, backend: str = "jax", use_kernel: bool = True
):
    """Load an explicit current-format checkpoint for the requested backend."""
    if backend not in {"jax", "torch"}:
        raise ValueError(f"invalid backend: {backend!r}")

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {model_path}")
    path_backend = model_path_backend(model_path)

    if path_backend == backend:
        if backend == "jax":
            return load_model_jax(model_path, use_kernel)
        else:
            from nequix.torch_impl.model import load_model as load_model_torch

            return load_model_torch(model_path, use_kernel)

    if path_backend == "torch":
        from nequix.torch_impl.model import load_model as load_model_torch
        from nequix.torch_impl.utils import convert_model_torch_to_jax

        model, metadata = load_model_torch(model_path, use_kernel)
        return convert_model_torch_to_jax(model, metadata, use_kernel)

    from nequix.torch_impl.utils import convert_model_jax_to_torch

    model, metadata = load_model_jax(model_path, use_kernel)
    return convert_model_jax_to_torch(model, metadata, use_kernel)


class NequixCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str | Path,
        capacity_multiplier: float = 1.1,  # Only for jax backend
        backend: str = "jax",
        use_kernel: bool = True,
        use_compile: bool = False,  # Only for torch backend
        **kwargs,
    ):
        super().__init__(**kwargs)

        if use_kernel and backend == "torch":
            import torch

            assert torch.cuda.is_available(), "Kernels need GPU environment"

        self.model, self.metadata = load_model_for_backend(model_path, backend, use_kernel)

        if backend == "torch":
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            # setting compile_state to True would skip compilation else will compile for the first time
            # Only use compile for GPUs
            self.compile_state = False if use_compile and torch.cuda.is_available() else True

        self.atom_indices = atomic_numbers_to_indices(self.metadata.atomic_numbers)
        self.cutoff = self.metadata.model_config.cutoff
        self._capacity = None
        self._capacity_multiplier = capacity_multiplier
        self.backend = backend

    def _pad_graph_jax(self, graph, numbers_changed=False):
        # maintain edge capacity with _capacity_multiplier over edges,
        # recalculate if numbers (system) changes, or if the capacity is exceeded
        if self._capacity is None or numbers_changed or graph.n_edge[0] > self._capacity:
            raw = int(np.ceil(graph.n_edge[0] * self._capacity_multiplier))
            # round up edges to the nearest multiple of 64
            # NB: this avoids excessive recompilation in high-throughput
            # workflows (e.g.  material relaxtions) but this number may need
            # to be tuned depending on the system sizes
            self._capacity = ((raw + 63) // 64) * 64

        # round up nodes to the nearest multiple of 8
        # NB: this avoids excessive recompilation in high-throughput
        # workflows (e.g. material relaxtions) but this number may need to
        # be tuned depending on the system sizes
        n_node = ((graph.n_node[0] + 8) // 8) * 8

        # pad the graph
        graph = jraph.pad_with_graphs(graph, n_node=n_node, n_edge=self._capacity, n_graph=2)
        return graph

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        processed_graph = preprocess_graph(atoms, self.atom_indices, self.cutoff, False)
        if self.backend == "jax":
            graph = dict_to_graphstuple(processed_graph)
            graph = self._pad_graph_jax(graph, "numbers" in system_changes)
            energy, forces, stress = eqx.filter_jit(self.model)(graph)
            forces = forces[: len(atoms)]

        elif self.backend == "torch":
            import torch

            graph = dict_to_pytorch_geometric(processed_graph)
            graph.n_graph = torch.zeros(graph.x.shape[0], dtype=torch.int64).to(self.device)
            graph = graph.to(self.device)
            if not self.compile_state:
                from torch.fx.experimental.proxy_tensor import make_fx

                self.model = torch.compile(
                    make_fx(
                        self.model,
                        tracing_mode="symbolic",
                        _allow_non_fake_inputs=True,
                        _error_on_data_dependent_ops=True,
                    )(
                        graph.x,
                        graph.positions,
                        graph.edge_attr,
                        graph.edge_index,
                        getattr(graph, "cell", None),
                        graph.n_node,
                        graph.n_edge,
                        graph.n_graph,
                    )
                )
                self.compile_state = True

            # Need to explicitly list out all the tensors because of make_fx
            energy_per_atom, forces, stress = self.model(
                graph.x,
                graph.positions,
                graph.edge_attr,
                graph.edge_index,
                getattr(graph, "cell", None),
                graph.n_node,
                graph.n_edge,
                graph.n_graph,
            )

            # scatter is outside of the model to avoid compile issues
            from nequix.torch_impl.model import scatter

            energy = scatter(energy_per_atom, graph.n_graph, dim=0, dim_size=graph.n_node.size(0))
            energy, forces, stress = (
                energy.detach().cpu(),
                forces.detach().cpu(),
                stress.detach().cpu() if stress is not None else None,
            )

        # take energy and forces without padding
        energy = energy[0].item()
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = np.array(forces)
        self.results["stress"] = (
            full_3x3_to_voigt_6_stress(np.array(stress[0])) if stress is not None else None
        )

    def get_hessian(self, atoms=None):
        assert self.backend == "jax", "Hessian calculation currently only supported for JAX backend"
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        n_atoms = len(atoms)
        processed_graph = preprocess_graph(atoms, self.atom_indices, self.cutoff, False)
        graph = dict_to_graphstuple(processed_graph)
        graph = self._pad_graph_jax(graph, True)
        hessian = eqx.filter_jit(hessian_linearized)(self.model, graph)
        return np.array(hessian[:n_atoms, :n_atoms, :, :], copy=True)
