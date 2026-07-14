import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest
from dataclasses import replace

from nequix.autobatch import (
    ProbeResult,
    autobatch_cache_key,
    batch_shape,
    cache_tune_result,
    cached_tune_result,
    tune_batch_shape,
    tune_training_batch,
)
from nequix.config import RUNS
from nequix.data import (
    ParallelLoader,
    best_fit_dynamic_batch,
    bounded_best_fit_indices,
)
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


def test_bounded_best_fit_is_deterministic_and_enforces_capacity():
    sizes = [(1, 6, 5), (1, 4, 4), (1, 5, 5), (1, 3, 2), (1, 2, 3)]
    capacity = (2, 9, 9)

    first = list(bounded_best_fit_indices(sizes, capacity, lookahead=3))
    second = list(bounded_best_fit_indices(sizes, capacity, lookahead=3))

    assert first == second
    assert sorted(index for packed in first for index in packed) == list(range(len(sizes)))
    for packed in first:
        used = tuple(sum(sizes[index][axis] for index in packed) for axis in range(3))
        assert all(value <= limit for value, limit in zip(used, capacity))
    with pytest.raises(ValueError, match="exceeds capacity"):
        list(bounded_best_fit_indices([(1, 10, 1)], capacity))


def test_best_fit_graph_batches_cover_every_sample_once():
    graphs = [_graph(index, n_node) for index, n_node in enumerate((6, 4, 5, 3, 2))]
    batches = list(
        best_fit_dynamic_batch(iter(graphs), n_graph=3, n_node=10, n_edge=0, lookahead=3)
    )

    identifiers = []
    for batch in batches:
        graph_mask = np.asarray(jraph.get_graph_padding_mask(batch))
        identifiers.extend(np.asarray(batch.globals["identifier"])[graph_mask].tolist())
        assert np.asarray(batch.n_node)[graph_mask].sum() <= 9
        assert graph_mask.sum() <= 2
    assert sorted(identifiers) == list(range(len(graphs)))


def test_parallel_loader_keeps_incomplete_final_device_group():
    batches = [_padded([_graph(index, 1)], n_graph=2, n_node=2) for index in range(3)]

    class Loader:
        def __iter__(self):
            return iter(batches)

    parallel_batches = list(ParallelLoader(Loader(), 2))
    identifiers = []
    for batch in parallel_batches:
        masks = jax.vmap(jraph.get_graph_padding_mask)(batch)
        identifiers.extend(np.asarray(batch.globals["identifier"])[np.asarray(masks)].tolist())

    assert sorted(identifiers) == [0, 1, 2]
    final_empty_device = jax.tree.map(lambda value: value[1], parallel_batches[-1])
    assert not np.asarray(jraph.get_graph_padding_mask(final_empty_device)).any()


def _hardware(memory_bytes=80 * 1024**3):
    return {
        "gpus": [{"name": "H100", "memory_bytes": memory_bytes, "driver": "1"}],
        "cuda_visible_devices": "0",
    }


def _config():
    return RUNS["nequix-mp-1"]


def _stats():
    return {
        "avg_n_nodes": 4.0,
        "avg_n_edges": 8.0,
        "max_n_nodes": 6,
        "max_n_edges": 12,
        "avg_n_neighbors": 2.0,
    }


def test_cache_key_invalidates_for_hardware_model_and_dataset_changes(tmp_path):
    config = _config()
    key = autobatch_cache_key(config, 100, _hardware())

    assert key != autobatch_cache_key(config, 101, _hardware())
    assert key != autobatch_cache_key(config, 100, _hardware(40 * 1024**3))
    assert key != autobatch_cache_key(replace(config, force_mode="direct"), 100, _hardware())
    assert key != autobatch_cache_key(
        replace(config, autobatch_memory_scaling_factor=2.0), 100, _hardware()
    )
    assert key != autobatch_cache_key(
        replace(config, autobatch_minimum_speedup=0.1), 100, _hardware()
    )

    result = tune_batch_shape(_stats(), 100, 1, lambda candidate: ProbeResult(candidate, "ok"))
    cache_path = tmp_path / "autobatch.json"
    cache_tune_result(cache_path, key, result)
    loaded = cached_tune_result(cache_path, key)
    assert loaded is not None
    assert loaded.shape == result.shape
    assert loaded.cached
    assert cached_tune_result(cache_path, "different") is None


def test_probe_failure_and_oom_fall_back_to_single_example_shape():
    initial = batch_shape(1, _stats())

    failed = tune_batch_shape(
        _stats(),
        dataset_size=100,
        device_count=1,
        probe=lambda shape: ProbeResult(shape, "failed", error="worker died"),
    )
    assert failed.shape == initial
    assert "failed" in failed.warning

    def oom_after_baseline(shape):
        if shape.batch_size == initial.batch_size:
            return ProbeResult(shape, "ok", graphs_per_second=10.0)
        return ProbeResult(shape, "oom", error="RESOURCE_EXHAUSTED")

    oom = tune_batch_shape(_stats(), 100, 1, oom_after_baseline)
    assert oom.shape == initial
    assert any(probe.status == "oom" for probe in oom.probes)
    assert "single-example capacity" in oom.warning


def test_tuner_selects_only_a_measurably_faster_safe_candidate():
    def probe(shape):
        if shape.batch_size > 12:
            return ProbeResult(shape, "oom")
        speed = {1: 5.0, 2: 8.0, 4: 10.0, 8: 14.0, 12: 13.0}.get(
            shape.batch_size, 11.0
        )
        return ProbeResult(shape, "ok", graphs_per_second=speed)

    result = tune_batch_shape(
        _stats(), 100, 1, probe, memory_scaling_factor=2.0
    )
    assert result.shape.batch_size == 8
    assert result.warning is None


def test_training_falls_back_to_one_example_without_detected_gpu(monkeypatch):
    config = _config()
    monkeypatch.setattr("nequix.autobatch.query_gpu_hardware", lambda: {"gpus": []})

    with pytest.warns(RuntimeWarning, match="single-example capacity"):
        result = tune_training_batch(config, list(range(10)))

    assert result.shape == batch_shape(1, config.dataset_stats())
    assert not result.probes


def test_memory_scaling_factor_must_be_greater_than_one():
    with pytest.raises(ValueError, match="greater than one"):
        tune_batch_shape(
            _stats(),
            100,
            1,
            lambda shape: ProbeResult(shape, "ok"),
            memory_scaling_factor=1.0,
        )


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
