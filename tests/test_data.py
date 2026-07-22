import jraph
import numpy as np

from nequix.data import DataLoader


def _graph(identifier: int, n_node: int, n_edge: int = 0) -> jraph.GraphsTuple:
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
            "identifier": np.array([identifier], dtype=np.int32),
        },
    )


def test_data_loader_shutdown_stops_workers():
    graphs = [_graph(index, n_node) for index, n_node in enumerate((6, 4, 5, 3, 2))]
    loader = DataLoader(
        graphs,
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
    assert all(not worker.is_alive() for worker in workers)
    assert not loader._started
    assert sum(1 for _ in loader) == 3
    loader.shutdown()
