import numpy as np
import pytest
from ase.io import write
import torch
import ase.build

from nequix.calculator import NequixCalculator

try:
    import openequivariance  # noqa: F401
    import openequivariance_extjax  # noqa: F401

    OEQ_AVAILABLE = True
except (ImportError, AssertionError):
    OEQ_AVAILABLE = False

skip_no_oeq = pytest.mark.skipif(not OEQ_AVAILABLE, reason="OpenEquivariance not installed")


def si():
    return ase.build.bulk("Si", "diamond", a=5.43)


@pytest.fixture(params=["relaxed", "perturbed"])
def structure(request):
    return request.param


@pytest.fixture
def atoms(structure):
    atoms = si()
    if structure == "perturbed":
        atoms.positions[0] += [0.1, 0.05, -0.05]
    return atoms


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("use_kernel", [True, False])
def test_nequix_calculator_backends_match_fresh_checkpoint(
    atoms, jax_model_path, backend, use_kernel
):
    if use_kernel and not OEQ_AVAILABLE:
        pytest.skip("OpenEquivariance not installed")

    if use_kernel and backend == "torch" and not torch.cuda.is_available():
        pytest.skip("Torch kernel requires CUDA")

    reference_atoms = atoms.copy()
    reference_atoms.calc = NequixCalculator(jax_model_path, backend="jax", use_kernel=False)
    reference = (
        reference_atoms.get_potential_energy(),
        reference_atoms.get_forces(),
        reference_atoms.get_stress(voigt=True),
    )

    atoms.calc = NequixCalculator(jax_model_path, backend=backend, use_kernel=use_kernel)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)

    np.testing.assert_allclose(energy, reference[0], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(forces, reference[1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(stress, reference[2], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "backend, kernel",
    [
        ("torch", False),
        pytest.param("torch", True, marks=skip_no_oeq),
        ("jax", False),
        pytest.param("jax", True, marks=skip_no_oeq),
    ],
)
def test_calculator_without_cell(jax_model_path, backend, kernel, tmp_path):
    atoms = ase.build.molecule("H2O")
    calc = NequixCalculator(jax_model_path, backend=backend, use_kernel=kernel)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
    write(tmp_path / "molecule.extxyz", atoms)


@pytest.mark.parametrize("kernel", [False, pytest.param(True, marks=skip_no_oeq)])
def test_calculator_hessian(jax_model_path, kernel):
    atoms = si()
    calc = NequixCalculator(jax_model_path, backend="jax", use_kernel=kernel)
    hessian = calc.get_hessian(atoms)
    print(hessian)
    assert hessian.shape == (len(atoms), len(atoms), 3, 3)
    assert np.all(np.isfinite(hessian))
