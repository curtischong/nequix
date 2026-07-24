import argparse

import ase
import numpy as np
import phonopy
from atompack import from_ase
from ase.calculators.singlepoint import SinglePointCalculator
from phonopy.interface.phonopy_yaml import load_yaml

from nequix.data import write_atompack_database


def read_phonopy_molecules(file_path):
    ph_ref = phonopy.load(file_path)
    # NOTE: the MDR phonon files report unit cell energy, so we need to
    # multiply by the number of repetitions to get supercell energy
    energy_unitcell = load_yaml(file_path)["energy"]
    nrep = round(np.linalg.det(ph_ref.supercell_matrix))
    energy = energy_unitcell * nrep
    atoms = ase.Atoms(
        cell=ph_ref.supercell.cell,
        symbols=ph_ref.supercell.symbols,
        scaled_positions=ph_ref.supercell.scaled_positions,
        pbc=True,
    )
    ph_ref.produce_force_constants()
    hessian = (
        np.array(ph_ref.force_constants, dtype=np.float32)  # (n, n, 3, 3)
        .swapaxes(1, 2)  # (n, 3, n, 3)
        .reshape(3 * len(atoms), 3 * len(atoms))  # (3n, 3n)
    )
    # NOTE: MDR data does not include forces or stress, but the authors
    # report performing a relaxation to a convergence criterion of 1e-8
    # eV/A calculation, so we set forces and stress to zero
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        forces=np.zeros_like(atoms.positions),
        stress=np.zeros_like(atoms.cell),
    )
    molecule = from_ase(atoms, copy_info=False, copy_arrays=False)
    molecule.set_property("hessian", hessian)
    return [molecule]


def preprocess(file_path, output_path, n_workers=16):
    write_atompack_database(file_path, output_path, "*.yaml.bz2", read_phonopy_molecules, n_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--n_workers", type=int, default=16)
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path, args.n_workers)


if __name__ == "__main__":
    main()
