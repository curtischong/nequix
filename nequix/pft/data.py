import jraph
import numpy as np
from nequix.data import AtomPackDataset, dict_to_graphstuple


# adds a single column of the hessian, and the corresponding vector to
# evaluate the hessian at that column as a global feature
def add_hessian_col_to_graph(graph, hessian_ref, col):
    v1d = (np.arange(hessian_ref.shape[1]) == col).astype(graph.nodes["positions"].dtype)
    vs = v1d.reshape(graph.nodes["positions"].shape)
    hessian_col = hessian_ref[:, col].reshape(graph.nodes["positions"].shape)
    return graph._replace(nodes={**graph.nodes, "vs": vs, "hessian_col": hessian_col})


class PhononDataset(AtomPackDataset):
    def __init__(
        self,
        file_path: str,
        atomic_numbers: list[int],
        cutoff: float = 5.0,
        random_col: bool = True,
        seed: int = 42,
        backend: str = "jax",
    ):
        super().__init__(file_path, atomic_numbers, cutoff, backend=backend)
        self.random_col = random_col
        self.rng = np.random.RandomState(seed=seed)
        assert self.backend == "jax", "PhononDataset only supports jax backend for now"

    def _get_graph_dict(self, idx: int):
        molecule = self._get_molecule(idx)
        graph = self._molecule_to_graph_dict(molecule, idx)
        if not molecule.has_property("hessian"):
            raise ValueError(
                f"AtomPack phonon record {idx} in {self.file_path} must contain a hessian"
            )
        graph["hessian"] = np.asarray(molecule.get_property("hessian"), dtype=np.float32)
        return graph

    def __getitem__(self, idx: int) -> jraph.GraphsTuple:
        graph_dict = self._get_graph_dict(idx)
        graph = dict_to_graphstuple(graph_dict)
        # use 0 col for validation, otherwise random col
        col = self.rng.randint(0, graph_dict["hessian"].shape[1]) if self.random_col else 0
        return add_hessian_col_to_graph(graph, graph_dict["hessian"], col)
