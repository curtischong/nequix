import argparse
import multiprocessing
from pathlib import Path

import ase
import numpy as np
import phonopy
from atompack import Database, from_ase
from ase.calculators.singlepoint import SinglePointCalculator
from phonopy.interface.phonopy_yaml import load_yaml
from tqdm import tqdm


def read_phonopy_molecule(file_path):
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
    return molecule


def preprocess(file_path, output_path, n_workers=16):
    file_path = Path(file_path)
    output_path = Path(output_path)
    if output_path.suffix != ".atp":
        raise ValueError(f"AtomPack output path must end in .atp: {output_path}")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.is_dir():
        file_paths = sorted(file_path.rglob("*.yaml.bz2"))
    else:
        file_paths = [file_path]
    if not file_paths:
        raise ValueError(f"No phonopy YAML files found in {file_path}")

    database = Database(str(output_path), overwrite=True)
    if n_workers == 1 or len(file_paths) == 1:
        molecules = map(read_phonopy_molecule, file_paths)
        for molecule in tqdm(molecules, total=len(file_paths)):
            database.add_molecule(molecule)
    else:
        with multiprocessing.Pool(min(n_workers, len(file_paths))) as pool:
            molecules = pool.imap(read_phonopy_molecule, file_paths)
            for molecule in tqdm(molecules, total=len(file_paths)):
                database.add_molecule(molecule)
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
