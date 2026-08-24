import copy
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph

from nequix.layer_norm import RMSLayerNorm
from nequix.config import ModelMetadata, layer_cutoffs


def bessel_basis(x: jax.Array, num_basis: int, r_max: float) -> jax.Array:
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
    x = x[:, None]
    return prefactor * jnp.where(
        x == 0.0,
        bessel_weights / r_max,  # prevent division by zero
        jnp.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: jax.Array, r_max: float, p: float) -> jax.Array:
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(x, p))
    out = out + (p * (p + 2.0) * jnp.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * jnp.power(x, p + 2.0))
    return out * jnp.where(x < 1.0, 1.0, 0.0)


class Sort(eqx.Module):
    irreps: e3nn.Irreps = eqx.field(static=True)
    irreps_sorted: e3nn.Irreps = eqx.field(static=True)
    slices_sorted: list = eqx.field(static=True)

    def __init__(self, irreps: e3nn.Irreps):
        self.irreps = irreps
        slices = list(irreps.slices())
        irreps_sorted, _, inv = irreps.sort()
        self.slices_sorted = [slices[i] for i in inv]
        self.irreps_sorted = irreps_sorted

    def __call__(self, x: jax.Array) -> jax.Array:
        chunks = [x[..., s] for s in self.slices_sorted]
        return jnp.concatenate(chunks, axis=-1)


class Linear(eqx.Module):
    weights: jax.Array
    bias: Optional[jax.Array]
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        scale = math.sqrt(init_scale / in_size)
        self.weights = jax.random.normal(key, (in_size, out_size)) * scale
        self.bias = jnp.zeros(out_size) if use_bias else None
        self.use_bias = use_bias

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


class MLP(eqx.Module):
    layers: list[Linear]
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        sizes,
        activation=jax.nn.silu,
        *,
        init_scale: float = 1.0,
        use_bias: bool = False,
        key: jax.Array,
    ):
        self.activation = activation

        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            Linear(
                sizes[i],
                sizes[i + 1],
                key=keys[i],
                use_bias=use_bias,
                # don't scale last layer since no activation
                init_scale=init_scale if i < len(sizes) - 2 else 1.0,
            )
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class NequixConvolution(eqx.Module):
    output_irreps: e3nn.Irreps = eqx.field(static=True)
    tp_irreps: e3nn.Irreps = eqx.field(static=True)
    index_weights: bool = eqx.field(static=True)
    avg_n_neighbors: float = eqx.field(static=True)
    kernel: bool = eqx.field(static=True)
    tp_conv: Optional[Any] = eqx.field(static=True)

    radial_mlp: MLP
    linear_1: e3nn.equinox.Linear
    linear_2: e3nn.equinox.Linear
    skip: e3nn.equinox.Linear
    layer_norm: Optional[RMSLayerNorm]
    sort: Sort

    def __init__(
        self,
        key: jax.Array,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        index_weights: bool = True,
        layer_norm: bool = False,
        kernel: bool = False,
    ):
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors
        self.index_weights = index_weights
        self.kernel = kernel

        irreps_out_tp = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(input_irreps):
            for j, (_, ir_in2) in enumerate(sh_irreps):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in output_irreps:
                        k = len(irreps_out_tp)
                        irreps_out_tp.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        tp_irreps = e3nn.Irreps(irreps_out_tp)
        _, _, inv = tp_irreps.sort()
        self.tp_irreps = tp_irreps

        if kernel:
            try:
                import torch  # noqa: F401
            except ImportError:
                # allow openequivariance to be imported without torch, but only if it is not
                # installed; otherwise, the torch backend won't work if users want to use
                # both torch and jax.
                os.environ["OEQ_NOTORCH"] = "1"

            try:
                import openequivariance as oeq
                import openequivariance_extjax  # noqa: F401
            except ImportError:
                raise ImportError(
                    "OpenEquivariance with JAX support is required for kernel=True. "
                    "Install both packages:\n"
                    "  uv pip install 'openequivariance[jax]'\n"
                    "  uv pip install 'openequivariance_extjax' --no-build-isolation"
                )

            instructions = [instructions[i] for i in inv]
            problem = oeq.TPProblem(
                str(input_irreps),
                str(sh_irreps),
                str(tp_irreps),
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
            )
            self.tp_conv = oeq.jax.TensorProductConv(problem, deterministic=False)
        else:
            self.tp_conv = None

        self.sort = Sort(tp_irreps)
        tp_irreps = self.sort.irreps_sorted

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.linear_1 = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=input_irreps,
            key=k1,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            key=k2,
        )

        # add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
        output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = e3nn.equinox.Linear(
            irreps_in=tp_irreps,
            irreps_out=output_irreps,
            key=k3,
        )

        # skip connection has per-species weights
        self.skip = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=output_irreps,
            linear_type="indexed" if index_weights else "vanilla",
            num_indexed_weights=n_species if index_weights else None,
            force_irreps_out=True,
            key=k4,
        )

        if layer_norm:
            self.layer_norm = RMSLayerNorm(
                irreps=output_irreps,
                centering=False,
                std_balance_degrees=True,
            )
        else:
            self.layer_norm = None

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        messages = self.linear_1(features)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)

        if self.kernel:
            messages_agg = self.sort(
                self.tp_conv.forward(
                    messages.array,
                    sh.array,
                    radial_message,
                    receivers.astype(jnp.int32),
                    senders.astype(jnp.int32),
                )
            )
            messages_agg = e3nn.IrrepsArray(self.sort.irreps_sorted, messages_agg)
        else:
            messages = messages[senders]
            messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.tp_irreps)
            messages = messages * radial_message
            messages_agg = e3nn.scatter_sum(messages, dst=receivers, output_size=features.shape[0])

        messages_agg = messages_agg / jnp.sqrt(jax.lax.stop_gradient(self.avg_n_neighbors))

        skip = self.skip(species, features) if self.index_weights else self.skip(features)
        features = self.linear_2(messages_agg) + skip

        if self.layer_norm is not None:
            features = self.layer_norm(features)

        return e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.tanh,
            even_gate_act=jax.nn.silu,
        )


def sort_inner_edges_first(data: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Reorder a batch on device so edges within the inner cutoff form a prefix.

    Edge order is irrelevant to message passing, so this only enables inner
    zigzag layers (``n_inner_edges``) to process a prefix of the edge slots.
    Sorting ungroups edges from their graphs, so the batch gains a per-edge
    graph index; the model must pair cells through it rather than positionally.
    Callers run it as its own small device program: a host-side reorder starves
    the dataloader, and inlining the sort into the training step's program
    makes its XLA compile pathologically slow.
    """
    edge_graph_idx = jnp.repeat(
        jnp.arange(data.n_edge.shape[0]),
        data.n_edge,
        total_repeat_length=data.senders.shape[0],
    )
    order = jnp.argsort(~data.edges["inner"], stable=True)
    edges = {key: value[order] for key, value in data.edges.items()}
    edges["graph"] = edge_graph_idx[order]
    return data._replace(
        senders=data.senders[order],
        receivers=data.receivers[order],
        edges=edges,
    )


def _cell_per_edge(data: jraph.GraphsTuple, cell: jax.Array) -> jax.Array:
    """Map each edge to its graph's cell, robust to inner-first edge sorting."""
    if "graph" in data.edges:
        return cell[data.edges["graph"]]
    return jnp.repeat(
        cell,
        data.n_edge,
        axis=0,
        total_repeat_length=data.edges["shifts"].shape[0],
    )


class Nequix(eqx.Module):
    lmax: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)
    radial_basis_size: int = eqx.field(static=True)
    radial_polynomial_p: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    layer_cutoffs: tuple[float, ...] = eqx.field(static=True)
    shift: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    atom_energies: jax.Array
    layers: list[NequixConvolution]
    readout: e3nn.equinox.Linear

    def __init__(
        self,
        key,
        n_species,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "128x0e + 128x1o + 128x2e + 128x3o",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        index_weights: bool = True,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
        layer_norm: bool = False,
        kernel: bool = False,
        zigzag_radii: Optional[Sequence[float]] = None,
    ):
        self.lmax = lmax
        self.cutoff = cutoff
        self.layer_cutoffs = layer_cutoffs(cutoff, n_layers, zigzag_radii)
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.shift = shift
        self.scale = scale
        self.atom_energies = (
            jnp.array(atom_energies)
            if atom_energies is not None
            else jnp.zeros(n_species, dtype=jnp.float32)
        )
        input_irreps = e3nn.Irreps(f"{n_species}x0e")
        sh_irreps = e3nn.s2_irreps(lmax)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        self.layers = []

        key, *subkeys = jax.random.split(key, n_layers + 1)
        for i in range(n_layers):
            self.layers.append(
                NequixConvolution(
                    key=subkeys[i],
                    input_irreps=input_irreps if i == 0 else hidden_irreps,
                    output_irreps=hidden_irreps if i < n_layers - 1 else hidden_irreps.filter("0e"),
                    sh_irreps=sh_irreps,
                    n_species=n_species,
                    radial_basis_size=radial_basis_size,
                    radial_mlp_size=radial_mlp_size,
                    radial_mlp_layers=radial_mlp_layers,
                    mlp_init_scale=mlp_init_scale,
                    # neighbor counts scale with the enclosed volume, so inner
                    # layers normalize by a cubically smaller estimate
                    avg_n_neighbors=avg_n_neighbors * (self.layer_cutoffs[i] / cutoff) ** 3,
                    index_weights=index_weights,
                    layer_norm=layer_norm,
                    kernel=kernel,
                )
            )

        self.readout = e3nn.equinox.Linear(
            irreps_in=hidden_irreps.filter("0e"), irreps_out="0e", key=key
        )

    def node_energies(
        self,
        displacements: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        n_inner_edges: Optional[int] = None,
    ):
        node_energies, _ = self.node_energies_and_penultimate_features(
            displacements, species, senders, receivers, n_inner_edges
        )
        return node_energies

    def node_energies_and_penultimate_features(
        self,
        displacements: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        n_inner_edges: Optional[int] = None,
    ) -> tuple[jax.Array, e3nn.IrrepsArray]:
        """Return energies and the features feeding the final scalar convolution.

        The penultimate equivariant features are useful for auxiliary pre-training
        heads. Keeping those heads outside :class:`Nequix` lets training discard the
        auxiliary parameters and retain the conservative backbone.

        With zigzag radii, inner layers stay correct on any edge ordering because
        the cutoff envelope zeroes edges beyond their radius. ``n_inner_edges``
        additionally restricts inner layers to the first ``n_inner_edges`` edges,
        which saves their compute when the caller has sorted inner edges first
        (see ``sort_inner_edges_first``).
        """
        # input features are one-hot encoded species
        features = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"), jax.nn.one_hot(species, self.n_species)
        )

        # safe norm (avoids nan for r = 0)
        square_r_norm = jnp.sum(displacements**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

        edge_counts = {}
        radial_bases = {}
        for layer_cutoff in set(self.layer_cutoffs):
            n_edges = r_norm.shape[0]
            if layer_cutoff < self.cutoff and n_inner_edges is not None:
                n_edges = min(n_inner_edges, n_edges)
            edge_counts[layer_cutoff] = n_edges
            radial_bases[layer_cutoff] = (
                bessel_basis(r_norm[:n_edges], self.radial_basis_size, layer_cutoff)
                * polynomial_cutoff(
                    r_norm[:n_edges],
                    layer_cutoff,
                    self.radial_polynomial_p,
                )[:, None]
            )

        # compute spherical harmonics of edge displacements
        sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(self.lmax),
            displacements,
            normalize=True,
            normalization="component",
        )

        def apply_layer(layer, layer_cutoff, features):
            n_edges = edge_counts[layer_cutoff]
            return layer(
                features,
                species,
                sh[:n_edges],
                radial_bases[layer_cutoff],
                senders[:n_edges],
                receivers[:n_edges],
            )

        for layer, layer_cutoff in zip(self.layers[:-1], self.layer_cutoffs[:-1]):
            features = apply_layer(layer, layer_cutoff, features)
        penultimate_features = features
        features = apply_layer(self.layers[-1], self.layer_cutoffs[-1], features)

        node_energies = self.readout(features)

        # scale and shift energies
        node_energies = node_energies * jax.lax.stop_gradient(self.scale) + jax.lax.stop_gradient(
            self.shift
        )

        # add isolated atom energies to each node as prior
        node_energies = node_energies + jax.lax.stop_gradient(self.atom_energies[species, None])

        return node_energies.array, penultimate_features

    def __call__(self, data: jraph.GraphsTuple, n_inner_edges: Optional[int] = None):
        if data.globals["cell"] is None:
            # compute forces and stress as gradient of total energy w.r.t positions
            def total_energy_fn(positions: jax.Array):
                r = positions[data.senders] - positions[data.receivers]
                node_energies = self.node_energies(
                    r, data.nodes["species"], data.senders, data.receivers, n_inner_edges
                )
                return jnp.sum(node_energies), node_energies

            minus_forces, node_energies = eqx.filter_grad(total_energy_fn, has_aux=True)(
                data.nodes["positions"]
            )
        else:
            # compute forces and stress as gradient of total energy w.r.t positions and strain
            def total_energy_fn(positions_eps: tuple[jax.Array, jax.Array]):
                positions, eps = positions_eps
                eps_sym = (eps + eps.swapaxes(1, 2)) / 2
                eps_sym_per_node = jnp.repeat(
                    eps_sym,
                    data.n_node,
                    axis=0,
                    total_repeat_length=data.nodes["positions"].shape[0],
                )
                # apply strain to positions and cell
                positions = positions + jnp.einsum("ik,ikj->ij", positions, eps_sym_per_node)
                cell = data.globals["cell"] + jnp.einsum(
                    "bij,bjk->bik", data.globals["cell"], eps_sym
                )
                offsets = jnp.einsum("ij,ijk->ik", data.edges["shifts"], _cell_per_edge(data, cell))
                r = positions[data.senders] - positions[data.receivers] + offsets
                node_energies = self.node_energies(
                    r, data.nodes["species"], data.senders, data.receivers, n_inner_edges
                )
                return jnp.sum(node_energies), node_energies

            eps = jnp.zeros_like(data.globals["cell"])
            (minus_forces, virial), node_energies = eqx.filter_grad(total_energy_fn, has_aux=True)(
                (data.nodes["positions"], eps)
            )

        # padded nodes may have nan forces, so we mask them
        node_mask = jraph.get_node_padding_mask(data)
        minus_forces = jnp.where(node_mask[:, None], minus_forces, 0.0)

        # compute total energies across each subgraph
        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=data.n_node.shape[0],
            indices_are_sorted=True,
        )

        if data.globals["cell"] is None:
            stress = None
        else:
            det = jnp.abs(jnp.linalg.det(data.globals["cell"]))[:, None, None]
            det = jnp.where(det > 0.0, det, 1.0)  # padded graphs have det = 0
            stress = virial / det
            # padded stress may be nan, so we mask them
            graph_mask = jraph.get_graph_padding_mask(data)
            stress = jnp.where(graph_mask[:, None, None], stress, 0.0)

        return graph_energies[:, 0], -minus_forces, stress


class DirectForceNequix(eqx.Module):
    """Training-only direct force and stress heads on a conservative Nequix backbone.

    Both heads consume the equivariant features immediately before the backbone's
    final scalar-only convolution: forces as a per-node 1o readout, stress as a
    per-node 0e+2e readout summed per graph (a symmetric rank-2 tensor's six
    independent components) and normalized by cell volume. They are intentionally
    separate from the backbone so direct pre-training can hand its Nequix weights
    to conservative fine-tuning.
    """

    backbone: Nequix
    force_readout: e3nn.equinox.Linear
    stress_readout: e3nn.equinox.Linear

    def __init__(self, backbone: Nequix, hidden_irreps: str, *, key: jax.Array):
        if len(backbone.layers) < 2:
            raise ValueError("direct-force pre-training requires at least two model layers")
        head_irreps = e3nn.Irreps(hidden_irreps)
        if head_irreps.count("1o") == 0:
            raise ValueError("direct-force pre-training requires 1o hidden features")
        if head_irreps.count("0e") == 0 or head_irreps.count("2e") == 0:
            raise ValueError("direct-stress pre-training requires 0e and 2e hidden features")

        force_key, stress_key = jax.random.split(key)
        self.backbone = backbone
        self.force_readout = e3nn.equinox.Linear(
            irreps_in=head_irreps,
            irreps_out="1o",
            key=force_key,
        )
        self.stress_readout = e3nn.equinox.Linear(
            irreps_in=head_irreps,
            irreps_out="0e + 2e",
            key=stress_key,
        )

    def __call__(self, data: jraph.GraphsTuple, n_inner_edges: Optional[int] = None):
        positions = data.nodes["positions"]
        displacements = positions[data.senders] - positions[data.receivers]
        if data.globals["cell"] is not None:
            displacements = displacements + jnp.einsum(
                "ij,ijk->ik", data.edges["shifts"], _cell_per_edge(data, data.globals["cell"])
            )

        node_energies, features = self.backbone.node_energies_and_penultimate_features(
            displacements,
            data.nodes["species"],
            data.senders,
            data.receivers,
            n_inner_edges,
        )
        forces = self.force_readout(features).array
        forces = jnp.where(jraph.get_node_padding_mask(data)[:, None], forces, 0.0)
        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=data.n_node.shape[0],
            indices_are_sorted=True,
        )

        if data.globals["cell"] is None:
            stress = None
        else:
            graph_virial = jraph.segment_sum(
                self.stress_readout(features).array,
                node_graph_idx(data),
                num_segments=data.n_node.shape[0],
                indices_are_sorted=True,
            )
            # 1o x 1o symmetric change of basis: 0e+2e components to a 3x3 tensor
            basis = e3nn.reduced_symmetric_tensor_product_basis("1o", 2).array
            virial = jnp.einsum("ijk,gk->gij", basis, graph_virial)
            det = jnp.abs(jnp.linalg.det(data.globals["cell"]))[:, None, None]
            det = jnp.where(det > 0.0, det, 1.0)  # padded graphs have det = 0
            graph_mask = jraph.get_graph_padding_mask(data)
            stress = jnp.where(graph_mask[:, None, None], virial / det, 0.0)

        return graph_energies[:, 0], forces, stress


class LoRALinear(eqx.Module):
    """Frozen ``Linear`` plus a trainable low-rank delta on its weight matrix."""

    base: Linear
    lora_a: jax.Array
    lora_b: jax.Array
    scaling: float = eqx.field(static=True)

    def __init__(self, base: Linear, rank: int, alpha: float, *, key: jax.Array):
        n_in, n_out = base.weights.shape
        effective_rank = min(rank, n_in, n_out)
        self.base = base
        self.lora_a = jax.random.normal(key, (n_in, effective_rank)) / math.sqrt(n_in)
        self.lora_b = jnp.zeros((effective_rank, n_out))
        self.scaling = alpha / rank

    def _with_delta(self, base: Linear) -> Linear:
        weights = base.weights + self.scaling * self.lora_a @ self.lora_b
        return eqx.tree_at(lambda module: module.weights, base, weights)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._with_delta(jax.lax.stop_gradient(self.base))(x)

    def merge(self) -> Linear:
        return self._with_delta(self.base)


class LoRAE3nnLinear(eqx.Module):
    """Frozen ``e3nn.equinox.Linear`` plus trainable low-rank deltas per weight block.

    Every block maps one input irrep chunk to one output irrep chunk with a plain
    ``(..., in, out)`` matrix, so a per-block ``(..., in, r) @ (..., r, out)`` delta
    stays within the same equivariant weight space and merges exactly.
    """

    base: e3nn.equinox.Linear
    lora_a: dict[str, jax.Array]
    lora_b: dict[str, jax.Array]
    scaling: float = eqx.field(static=True)

    def __init__(self, base: e3nn.equinox.Linear, rank: int, alpha: float, *, key: jax.Array):
        self.base = base
        lora_a: dict[str, jax.Array] = {}
        lora_b: dict[str, jax.Array] = {}
        for i, name in enumerate(sorted(base._weights)):
            *batch_shape, n_in, n_out = base._weights[name].shape
            effective_rank = min(rank, n_in, n_out)
            lora_a[name] = jax.random.normal(
                jax.random.fold_in(key, i), (*batch_shape, n_in, effective_rank)
            ) / math.sqrt(n_in)
            lora_b[name] = jnp.zeros((*batch_shape, effective_rank, n_out))
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scaling = alpha / rank

    def _with_delta(self, base_weights: dict[str, jax.Array]) -> e3nn.equinox.Linear:
        weights = {
            name: weight + self.scaling * self.lora_a[name] @ self.lora_b[name]
            for name, weight in base_weights.items()
        }
        return eqx.tree_at(lambda module: module._weights, self.base, weights)

    def __call__(self, *args) -> e3nn.IrrepsArray:
        return self._with_delta(jax.lax.stop_gradient(self.base._weights))(*args)

    def merge(self) -> e3nn.equinox.Linear:
        return self._with_delta(self.base._weights)


def _is_linear(x) -> bool:
    return isinstance(x, (Linear, e3nn.equinox.Linear))


def _is_lora(x) -> bool:
    return isinstance(x, (LoRALinear, LoRAE3nnLinear))


def apply_lora(model, rank: int, alpha: float | None = None, *, key: jax.Array):
    """Wrap every linear layer with a frozen-base LoRA adapter (zero delta at init).

    Layer-norm scales and biases stay trainable; the atom-energy prior is already
    behind a ``stop_gradient`` in the forward pass.
    """
    if alpha is None:
        alpha = 2.0 * rank
    counter = itertools.count()

    def wrap(x):
        if not _is_linear(x):
            return x
        wrapper = LoRALinear if isinstance(x, Linear) else LoRAE3nnLinear
        return wrapper(x, rank, alpha, key=jax.random.fold_in(key, next(counter)))

    return jax.tree.map(wrap, model, is_leaf=_is_linear)


def merge_lora(model):
    """Fold LoRA deltas into their base linear layers; identity for plain models."""
    leaves = jax.tree.flatten(model, is_leaf=_is_lora)[0]
    if not any(_is_lora(leaf) for leaf in leaves):
        return model
    return jax.tree.map(lambda x: x.merge() if _is_lora(x) else x, model, is_leaf=_is_lora)


def lora_param_count(model) -> int:
    """Number of trainable adapter parameters in a LoRA-wrapped model."""
    total = 0
    for leaf in jax.tree.flatten(model, is_leaf=_is_lora)[0]:
        if _is_lora(leaf):
            total += sum(p.size for p in jax.tree.leaves((leaf.lora_a, leaf.lora_b)))
    return total


def conservative_backbone(model: Nequix | DirectForceNequix) -> Nequix:
    """Return the inference model, dropping any training-only direct-force head
    and folding LoRA adapters into their base weights."""
    return merge_lora(model.backbone if isinstance(model, DirectForceNequix) else model)


def replace_normalization(
    model: Nequix,
    *,
    atom_energies: Sequence[float],
    shift: float,
    scale: float,
) -> Nequix:
    """Return a model with new energy references while preserving learned weights.

    ``shift`` and ``scale`` are static Equinox fields, so they cannot be changed with
    ``eqx.tree_at``. A shallow copy safely gives the returned model a new PyTree
    definition; the learned parameter arrays remain shared until training updates them.
    """
    if len(atom_energies) != model.n_species:
        raise ValueError(
            f"expected {model.n_species} isolated-atom energies, got {len(atom_energies)}"
        )

    updated = copy.copy(model)
    object.__setattr__(updated, "shift", float(shift))
    object.__setattr__(updated, "scale", float(scale))
    return eqx.tree_at(
        lambda candidate: candidate.atom_energies,
        updated,
        jnp.asarray(atom_energies, dtype=model.atom_energies.dtype),
    )


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    # based on https://github.com/google-deepmind/jraph/blob/51f5990/jraph/_src/models.py#L209-L216
    n_graph = data.n_node.shape[0]
    # equivalent to jnp.sum(n_node), but jittable
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node)
    return node_gr_idx


def weight_decay_mask(model):
    """weight decay mask (only apply decay to linear weights; LoRA bases are frozen)"""

    def is_layer(x):
        return _is_linear(x) or _is_lora(x)

    def set_mask(x):
        if _is_lora(x):
            mask = jax.tree.map(lambda _: False, x)
            return eqx.tree_at(
                lambda m: (m.lora_a, m.lora_b),
                mask,
                jax.tree.map(lambda _: True, (x.lora_a, x.lora_b)),
            )
        elif isinstance(x, Linear):
            mask = jax.tree.map(lambda _: True, x)
            mask = eqx.tree_at(lambda m: m.bias, mask, False)
            return mask
        elif isinstance(x, e3nn.equinox.Linear):
            return jax.tree.map(lambda _: True, x)
        else:
            return jax.tree.map(lambda _: False, x)

    mask = jax.tree.map(set_mask, model, is_leaf=is_layer)
    return mask


def save_model(path: str | Path, model: Nequix, metadata: ModelMetadata) -> None:
    """Save model weights with the current strict metadata schema."""
    # Statics are rebuilt from the header at load time, so a mismatch silently
    # changes the served model's energies.
    statics = {
        "avg_n_neighbors": (metadata.avg_n_neighbors, model.layers[0].avg_n_neighbors),
        "shift": (metadata.shift, model.shift),
        "scale": (metadata.scale, model.scale),
    }
    mismatched = {name: pair for name, pair in statics.items() if pair[0] != pair[1]}
    if mismatched:
        raise ValueError(f"metadata does not match model statics: {mismatched}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write((json.dumps(metadata.to_header()) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def model_from_metadata(
    metadata: ModelMetadata, kernel: bool = False, *, key: jax.Array | None = None
) -> Nequix:
    """Construct an unfitted Nequix with the architecture a metadata header describes."""
    config = metadata.model_config
    return Nequix(
        key=key if key is not None else jax.random.key(0),
        n_species=len(metadata.atomic_numbers),
        hidden_irreps=config.hidden_irreps,
        lmax=config.lmax,
        cutoff=config.cutoff,
        n_layers=config.n_layers,
        radial_basis_size=config.radial_basis_size,
        radial_mlp_size=config.radial_mlp_size,
        radial_mlp_layers=config.radial_mlp_layers,
        radial_polynomial_p=config.radial_polynomial_p,
        mlp_init_scale=config.mlp_init_scale,
        index_weights=config.index_weights,
        layer_norm=config.layer_norm,
        shift=metadata.shift,
        scale=metadata.scale,
        avg_n_neighbors=metadata.avg_n_neighbors,
        atom_energies=metadata.atom_energies,
        kernel=kernel,
        zigzag_radii=config.zigzag_radii,
    )


def load_model(path: str | Path, kernel: bool = False) -> tuple[Nequix, ModelMetadata]:
    """Load weights written with the current Nequix model format."""
    with open(path, "rb") as f:
        try:
            metadata = ModelMetadata.from_header(json.loads(f.readline().decode()))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid Nequix model header") from error
        model = eqx.tree_deserialise_leaves(f, model_from_metadata(metadata, kernel))
        return model, metadata
