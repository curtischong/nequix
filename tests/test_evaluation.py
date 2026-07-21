import json
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import write

from nequix.config import BenchmarkConfig, LongMDEvalConfig, MLIPArenaConfig, ValidationConfig
from nequix.evaluation import (
    _energy_drift,
    benchmarks_due,
    load_long_md_systems,
    long_md_protocol,
    run_long_md_evaluation,
    validate_benchmark_config,
    validate_validation_config,
)


def test_evaluation_config_and_protocol_defaults():
    assert ValidationConfig().every_steps == 20_000
    assert BenchmarkConfig().every_steps == 20_000
    benchmarks = BenchmarkConfig(every_steps=100, long_md=LongMDEvalConfig())
    validate_validation_config(ValidationConfig())
    validate_benchmark_config(benchmarks)
    assert not benchmarks_due(BenchmarkConfig(), 100)
    assert not benchmarks_due(benchmarks, 99)
    assert benchmarks_due(benchmarks, 100)
    assert benchmarks_due(benchmarks, 200)
    assert not benchmarks_due(replace(benchmarks, every_steps=None), 100)
    assert long_md_protocol(LongMDEvalConfig(dataset="tm23")) == (20_000, 5.0)
    assert long_md_protocol(LongMDEvalConfig(dataset="md22")) == (100_000, 1.0)
    assert long_md_protocol(LongMDEvalConfig(steps=12, time_step_fs=0.5)) == (12, 0.5)

    with pytest.raises(ValueError, match="every_steps"):
        validate_validation_config(ValidationConfig(every_steps=0))
    with pytest.raises(ValueError, match="every_steps"):
        validate_benchmark_config(BenchmarkConfig(every_steps=0))
    with pytest.raises(ValueError, match="max_workers"):
        validate_benchmark_config(BenchmarkConfig(mlip_arena=MLIPArenaConfig(max_workers=0)))


def test_energy_drift_uses_per_ps_slope_after_equilibration():
    times = np.linspace(0.0, 10.0, 101)
    energies = 2.0 + 0.001 * times

    assert _energy_drift(energies, times) == pytest.approx(1.0)


def test_load_long_md_systems_honors_prefix_before_reading(tmp_path):
    path = tmp_path / "tm23" / "Ag_cold_nequip_test.xyz"
    path.parent.mkdir()
    write(path, Atoms("Ag", positions=[[0.0, 0.0, 0.0]], cell=[4.0] * 3, pbc=True))
    config = LongMDEvalConfig(dataset_root=str(tmp_path), max_systems=1)

    systems = load_long_md_systems(config)

    assert len(systems) == 1
    assert systems[0][0] == "Ag-cold"
    assert systems[0][2] == pytest.approx(1235.0 * 0.25)


def test_long_md_evaluation_writes_metrics(tmp_path):
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    config = LongMDEvalConfig(
        output_dir=str(tmp_path),
        steps=25,
        time_step_fs=0.1,
        save_frequency=1,
        relaxation_steps=1,
    )

    metrics = run_long_md_evaluation(
        config,
        LennardJones(),
        systems=[("argon", atoms, 10.0)],
    )

    assert metrics["successful_systems"] == 1
    assert metrics["failed_systems"] == 0
    assert np.isfinite(metrics["drift_mev_per_atom_ps"])
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["systems"][0]["name"] == "argon"
