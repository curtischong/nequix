"""Evaluate a Nequix checkpoint on the full Matbench Discovery WBM test set.

Stage 1 relaxes the ~257k WBM initial structures with the standard Matbench
Discovery force-field protocol (FrechetCellFilter + FIRE). Shard it across GPUs
by launching one process per device; every shard resumes from its own file:

    CUDA_VISIBLE_DEVICES=$i uv run --python 3.12 --extra mbd \
        python scripts/eval_matbench_discovery.py relax checkpoints/model.nqx \
        --shard-index $i --num-shards 8

Stage 2 applies MP2020 energy corrections, computes formation energies and hull
distances, and writes leaderboard metrics for the full test set and the
unique-prototype subset:

    uv run --python 3.12 --extra mbd \
        python scripts/eval_matbench_discovery.py join checkpoints/model.nqx

Benchmark data caches under ~/.cache/matbench-discovery (override with
MATBENCH_DISCOVERY_CACHE_DIR). figshare.com/ndownloader is unreachable from
some machines (empty HTTP 202 responses); if the automatic download fails,
fetch the same file id from https://api.figshare.com/v2/file/download/<id>.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from tqdm import tqdm

E_FORM_PRED = "e_form_per_atom_nequix"
EACH_PRED = "e_above_hull_pred_nequix"
EACH_TRUE = "e_above_hull_mp2020_corrected_ppd_mp"
E_FORM_DFT = "e_form_per_atom_mp2020_corrected"


def shard_path(out_dir: Path, shard_index: int, num_shards: int) -> Path:
    return out_dir / f"relaxations-{shard_index:03d}-of-{num_shards:03d}.jsonl"


def relax_one(atoms: Atoms, calculator: Any, fmax: float, max_steps: int) -> dict[str, Any]:
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms.calc = calculator
    optimizer = FIRE(FrechetCellFilter(atoms), logfile=None)
    try:
        converged = optimizer.run(fmax=fmax, steps=max_steps)
    except Exception as error:  # one bad structure must not kill a multi-day shard
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "energy": atoms.get_potential_energy(),
        "structure": AseAtomsAdaptor.get_structure(atoms).as_dict(),
        "steps": optimizer.nsteps,
        "converged": bool(converged),
    }


def run_relaxations(args: argparse.Namespace) -> None:
    from matbench_discovery.data import DataFiles, ase_atoms_from_zip

    from nequix.calculator import NequixCalculator

    out_path = shard_path(args.out_dir, args.shard_index, args.num_shards)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["material_id"] for line in out_path.read_text().splitlines()}

    atoms_list = ase_atoms_from_zip(DataFiles.wbm_initial_atoms.path, limit=args.limit)
    atoms_list.sort(key=lambda atoms: atoms.info["material_id"])
    shard = atoms_list[args.shard_index :: args.num_shards]
    calculator = NequixCalculator(args.model_path, backend=args.backend)
    with out_path.open("a") as out_file:
        for atoms in tqdm(shard, desc=f"relax shard {args.shard_index + 1}/{args.num_shards}"):
            material_id = atoms.info["material_id"]
            if material_id in done:
                continue
            record = {"material_id": material_id}
            record |= relax_one(atoms, calculator, args.fmax, args.max_steps)
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()


def compute_metrics(args: argparse.Namespace) -> None:
    import pandas as pd
    from matbench_discovery.data import DataFiles, df_wbm
    from matbench_discovery.energy import get_e_form_per_atom
    from matbench_discovery.metrics import stable_metrics
    from pymatgen.core import Structure
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    records: dict[str, dict[str, Any]] = {}
    n_failed = 0
    for path in sorted(args.out_dir.glob("relaxations-*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if "energy" in record:
                records[record["material_id"]] = record
            else:
                n_failed += 1
    print(f"loaded {len(records)} relaxations ({n_failed} failed) from {args.out_dir}")

    cse_frame = pd.read_json(DataFiles.wbm_computed_structure_entries.path)
    cse_frame = cse_frame.set_index("material_id")
    # Swap the model energy and relaxed structure into the WBM entry so MP2020
    # GGA/GGA+U and anion corrections apply exactly as on the leaderboard.
    entries = {}
    for material_id, record in tqdm(records.items(), desc="building entries"):
        cse_dict = cse_frame.loc[material_id, "computed_structure_entry"]
        cse = ComputedStructureEntry.from_dict(cse_dict)
        cse._energy = record["energy"]
        cse._structure = Structure.from_dict(record["structure"])
        entries[material_id] = cse
    MaterialsProject2020Compatibility().process_entries(entries.values(), verbose=True, clean=True)

    e_form_pred = pd.Series(
        {material_id: get_e_form_per_atom(cse) for material_id, cse in entries.items()},
        name=E_FORM_PRED,
    ).reindex(df_wbm.index)
    each_pred = df_wbm[EACH_TRUE] + e_form_pred - df_wbm[E_FORM_DFT]
    unique_prototypes = df_wbm["unique_prototype"]
    # stable_metrics(fillna=True) counts missing/failed relaxations against the model.
    metrics = {
        "full_test_set": stable_metrics(df_wbm[EACH_TRUE], each_pred),
        "unique_prototypes": stable_metrics(
            df_wbm[EACH_TRUE][unique_prototypes], each_pred[unique_prototypes]
        ),
        "n_relaxed": len(records),
        "n_failed": n_failed,
        "n_missing": int(e_form_pred.isna().sum()),
    }
    preds = pd.DataFrame({E_FORM_PRED: e_form_pred, EACH_PRED: each_pred})
    preds.to_csv(args.out_dir / "preds.csv.gz")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=["relax", "join"])
    parser.add_argument("model_path", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, help="default: evaluations/matbench_discovery/<model name>"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--limit", type=int, help="only relax the first N structures (smoke runs)")
    parser.add_argument("--backend", choices=["jax", "torch"], default="jax")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path("evaluations/matbench_discovery") / args.model_path.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Relaxations share GPUs with other jobs (e.g. training), so allocate on
    # demand and reuse the persistent JAX compilation cache across shards.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR", str(Path("evaluations/jax_cache").absolute())
    )
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    if args.stage == "relax":
        run_relaxations(args)
    else:
        compute_metrics(args)


if __name__ == "__main__":
    main()
