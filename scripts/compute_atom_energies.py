import argparse
from pathlib import Path

import numpy as np
import yaml
from atompack import Database
from tqdm import tqdm

from nequix.config import ATOMIC_NUMBERS


# based on https://github.com/ACEsuit/mace/blob/d39cc6b/mace/data/utils.py#L300
def compute_atom_energies(
    dataset_paths: list[str], atomic_numbers: tuple[int, ...], batch_size: int = 4096
) -> dict[int, float]:
    """Least-squares fit of per-species isolated-atom energies over the dataset."""
    atomic_indices = {number: i for i, number in enumerate(atomic_numbers)}
    counts = []
    energies = []
    for path in dataset_paths:
        database = Database.open(path)
        for start in tqdm(range(0, len(database), batch_size), desc=path):
            indices = list(range(start, min(start + batch_size, len(database))))
            for molecule in database.get_molecules(indices):
                species = np.array(
                    [atomic_indices[int(number)] for number in molecule.atomic_numbers]
                )
                counts.append(np.bincount(species, minlength=len(atomic_indices)))
                energies.append(molecule.energy)
    A = np.array(counts, dtype=np.float32)
    B = np.array(energies, dtype=np.float32)
    E0s = np.linalg.lstsq(A, B, rcond=None)[0]
    return {number: float(E0s[i]) for number, i in atomic_indices.items()}


def main():
    parser = argparse.ArgumentParser(description=compute_atom_energies.__doc__)
    parser.add_argument("dataset_paths", nargs="+", help="atompack database file(s)")
    parser.add_argument("--output", type=Path, required=True, help="output .yml path")
    args = parser.parse_args()

    atom_energies = compute_atom_energies(args.dataset_paths, ATOMIC_NUMBERS)
    args.output.write_text(yaml.safe_dump(atom_energies, sort_keys=True))
    print(f"wrote {len(atom_energies)} isolated-atom energies to {args.output}")


if __name__ == "__main__":
    main()
