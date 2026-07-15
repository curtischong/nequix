import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from nequix.data import DataLoader, ParallelLoader
from nequix.train import _distributed_loss, loss


def _graph(identifier, n_node, n_edge=0):
    senders = np.arange(n_edge, dtype=np.int32) % n_node
    receivers = (senders + 1) % n_node
    feature = float(identifier + 1)
    return jraph.GraphsTuple(
        n_node=np.array([n_node], dtype=np.int32),
        n_edge=np.array([n_edge], dtype=np.int32),
        nodes={
            "positions": np.full((n_node, 3), feature, dtype=np.float32),
            "forces": np.full((n_node, 3), feature / 3, dtype=np.float32),
        },
        edges={"shifts": np.zeros((n_edge, 3), dtype=np.float32)},
        senders=senders,
        receivers=receivers,
        globals={
            "energy": np.array([feature / 2], dtype=np.float32),
            "feature": np.array([feature], dtype=np.float32),
            "identifier": np.array([identifier], dtype=np.int32),
            "stress": None,
        },
    )


def _padded(graphs, n_graph, n_node, n_edge=0):
    return jraph.pad_with_graphs(
        jraph.batch_np(graphs), n_graph=n_graph, n_node=n_node, n_edge=n_edge
    )


def test_data_loader_uses_ordered_jraph_dynamic_batches():
    graphs = [_graph(index, n_node) for index, n_node in enumerate((6, 4, 5, 3, 2))]
    loader = DataLoader(
        graphs,
        max_n_nodes=9,
        max_n_edges=0,
        avg_n_nodes=0,
        avg_n_edges=0,
        batch_size=2,
        num_workers=0,
    )
    loader._start_workers = lambda: None
    loader.make_generator = lambda: iter(graphs)

    identifiers = []
    for batch in loader:
        mask = np.asarray(jraph.get_graph_padding_mask(batch))
        identifiers.append(np.asarray(batch.globals["identifier"])[mask].tolist())

    assert identifiers == [[0], [1, 2], [3, 4]]


def test_parallel_loader_uses_only_complete_device_groups():
    batches = [_padded([_graph(index, 1)], n_graph=2, n_node=2) for index in range(3)]

    class Loader:
        def __iter__(self):
            return iter(batches)

    parallel_batches = list(ParallelLoader(Loader(), 2))
    identifiers = []
    for batch in parallel_batches:
        masks = jax.vmap(jraph.get_graph_padding_mask)(batch)
        identifiers.extend(np.asarray(batch.globals["identifier"])[np.asarray(masks)].tolist())

    assert sorted(identifiers) == [0, 1]


class _ToyModel(eqx.Module):
    scale: jax.Array

    def __call__(self, batch):
        energy = self.scale * batch.globals["feature"]
        forces = self.scale * batch.nodes["positions"]
        return energy, forces, None


def test_distributed_loss_and_gradient_match_equivalent_combined_batch():
    graphs = [_graph(0, 1), _graph(1, 2), _graph(2, 3)]
    device_batches = (
        _padded(graphs[:1], n_graph=4, n_node=8),
        _padded(graphs[1:], n_graph=4, n_node=8),
    )
    stacked = jax.tree.map(lambda *values: jnp.stack(values), *device_batches)
    combined = _padded(graphs, n_graph=5, n_node=9)
    model = _ToyModel(jnp.array(0.25))

    def combined_objective(candidate):
        return loss(candidate, combined, 2.0, 3.0, 0.0, "mse")[0]

    def distributed_objective(candidate):
        local_losses, (_, metrics) = jax.vmap(
            lambda batch: _distributed_loss(candidate, batch, 2.0, 3.0, 0.0, "mse"),
            axis_name="device",
        )(stacked)
        return local_losses.sum(), metrics

    combined_value, combined_grad = eqx.filter_value_and_grad(combined_objective)(model)
    (distributed_value, metrics), distributed_grad = eqx.filter_value_and_grad(
        distributed_objective, has_aux=True
    )(model)

    np.testing.assert_allclose(distributed_value, combined_value, rtol=1e-6)
    np.testing.assert_allclose(distributed_grad.scale, combined_grad.scale, rtol=1e-6)
    assert np.isfinite(np.asarray(metrics["force_mae"])).all()
