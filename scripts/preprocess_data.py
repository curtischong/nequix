import argparse

import ase.io
from atompack import from_ase

from nequix.data import write_atompack_database


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
    write_atompack_database(file_path, output_path, "*.extxyz", read_molecules, n_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--n_workers", type=int, default=16)
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path, args.n_workers)


if __name__ == "__main__":
    main()
