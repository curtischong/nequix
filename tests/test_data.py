import jraph
import numpy as np

from nequix.data import DataLoader


def _graph(n_node: int) -> jraph.GraphsTuple:
    # Payload is irrelevant to the loader; only node/graph counts drive batching.
    return jraph.GraphsTuple(
        n_node=np.array([n_node], dtype=np.int32),
        n_edge=np.array([0], dtype=np.int32),
        nodes=np.zeros((n_node, 1), dtype=np.float32),
        edges=None,
        senders=np.zeros(0, dtype=np.int32),
        receivers=np.zeros(0, dtype=np.int32),
        globals=None,
    )


def test_data_loader_shutdown_stops_workers():
    loader = DataLoader(
        [_graph(n) for n in (6, 4, 5, 3, 2)],
        max_n_nodes=9,
        max_n_edges=0,
        avg_n_nodes=0,
        avg_n_edges=0,
        batch_size=2,
        num_workers=2,
    )

    assert sum(1 for _ in loader) == 3
    workers = list(loader.workers)

    loader.shutdown()
    assert all(not w.is_alive() for w in workers)
    assert not loader._started

    # loader is reusable after shutdown
    assert sum(1 for _ in loader) == 3
    loader.shutdown()
