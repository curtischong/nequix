import argparse
import multiprocessing
from pathlib import Path

import ase.io
from atompack import Database, from_ase
from tqdm import tqdm


def read_molecules(file_path):
    atoms_list = ase.io.read(file_path, index=":")
    molecules = [from_ase(atoms, copy_info=False, copy_arrays=False) for atoms in atoms_list]
    # Magnetic moments are not training targets, and ASE represents them as a
    # scalar for one-atom structures but an array otherwise. AtomPack requires
    # one schema per property, so discard this unused, shape-varying result.
    for molecule in molecules:
        if molecule.has_property("magmoms"):
            molecule.delete_property("magmoms")
    return molecules


def preprocess(file_path, output_path, n_workers=16):
    file_path = Path(file_path)
    output_path = Path(output_path)
    if output_path.suffix != ".atp":
        raise ValueError(f"AtomPack output path must end in .atp: {output_path}")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.is_dir():
        file_paths = sorted(file_path.rglob("*.extxyz"))
    else:
        file_paths = [file_path]
    if not file_paths:
        raise ValueError(f"No extxyz files found in {file_path}")

    database = Database(str(output_path), overwrite=True)
    if n_workers == 1 or len(file_paths) == 1:
        molecule_groups = map(read_molecules, file_paths)
        for molecules in tqdm(molecule_groups, total=len(file_paths)):
            database.add_molecules(molecules)
    else:
        with multiprocessing.Pool(min(n_workers, len(file_paths))) as pool:
            molecule_groups = pool.imap(read_molecules, file_paths)
            for molecules in tqdm(molecule_groups, total=len(file_paths)):
                database.add_molecules(molecules)
    database.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--n_workers", type=int, default=16)
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path, args.n_workers)


if __name__ == "__main__":
    main()
