"""Evaluate a Nequix checkpoint on the Matbench Discovery leaderboard metrics.

Stage 1 (relax) relaxes the ~257k WBM initial structures with the standard Matbench
Discovery force-field protocol (FrechetCellFilter + FIRE). Shard it across GPUs
by launching one process per device; every shard resumes from its own file:

    CUDA_VISIBLE_DEVICES=$i uv run --python 3.12 --extra mbd \
        python scripts/eval_matbench_discovery.py relax checkpoints/model.nqx \
        --shard-index $i --num-shards 8

Stage 2 (kappa) runs the PhononDB-103 thermal-conductivity benchmark (relax ->
FC2 -> FC3 -> Wigner conductivity) needed for the leaderboard's kappa_SRME and
CPS columns. It shards and resumes the same way:

    CUDA_VISIBLE_DEVICES=$i uv run --python 3.12 --extra mbd \
        python scripts/eval_matbench_discovery.py kappa checkpoints/model.nqx \
        --shard-index $i --num-shards 8

Stage 3 (join) applies MP2020 energy corrections, computes formation energies,
hull distances, and RMSD vs the DFT-relaxed WBM structures, merges kappa records
if complete, and writes metrics.json whose "leaderboard" block mirrors the main
columns on https://matbench-discovery.materialsproject.org (CPS, F1, DAF, Prec,
Acc, MAE, R2, kappa_SRME, RMSD; discovery metrics use the site's default
unique-prototypes subset). CPS is null until the kappa stage has finished:

    uv run --python 3.12 --extra mbd \
        python scripts/eval_matbench_discovery.py join checkpoints/model.nqx

Benchmark data caches under ~/.cache/matbench-discovery (override with
MBD_CACHE_DIR). figshare.com/ndownloader is unreachable from
some machines (empty HTTP 202 responses); if the automatic download fails,
fetch the same file id from https://api.figshare.com/v2/file/download/<id>.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from tqdm import tqdm

E_FORM_PRED = "e_form_per_atom_nequix"
EACH_PRED = "e_above_hull_pred_nequix"
EACH_TRUE = "e_above_hull_mp2020_corrected_ppd_mp"
E_FORM_DFT = "e_form_per_atom_mp2020_corrected"
RMSD_COL = "structure_rmsd_vs_dft"
# keep in sync with the matbench-discovery pin in pyproject.toml
MBD_REV = "21d9b1616442c3f97b30b6b1225d2c3742199112"
# CPS normalization constants from site/src/lib/labels.ts + combined-scores.svelte.ts
RMSD_BASELINE = 0.15


def ensure_mbd_repo_files() -> None:
    """matbench-discovery wheels omit repo files it reads at runtime (data/datasets.yml
    for `import matbench_discovery.data`, models/run_kappa.py for the kappa manifest's
    source hash); fetch them from the pinned rev once."""
    import matbench_discovery

    for rel_path in ("data/datasets.yml", "models/run_kappa.py"):
        path = Path(matbench_discovery.ROOT) / rel_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                f"https://raw.githubusercontent.com/janosh/matbench-discovery/{MBD_REV}/{rel_path}"
            )
            path.write_bytes(urlopen(url).read())


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
    from matbench_discovery.data import ase_atoms_from_zip
    from matbench_discovery.enums import DataFiles

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


def run_kappa(args: argparse.Namespace) -> None:
    from matbench_discovery.enums import DataFiles
    from matbench_discovery.phonons.pipeline import (
        KappaSettings,
        load_phonondb_atoms,
        run_kappa_shard,
    )

    from nequix.calculator import NequixCalculator

    dry_run = args.limit is not None
    atoms_by_id = load_phonondb_atoms(dry_run=dry_run, dry_run_size=args.limit or 1)
    calculator = NequixCalculator(args.model_path, backend=args.backend)
    run_kappa_shard(
        calculator=calculator,
        model_key=args.model_path.stem,
        atoms_by_id=atoms_by_id,
        dataset_path=DataFiles.phonondb_pbe_103_structures.path,
        shard_dir=str(args.out_dir / "kappa"),
        shard_index=args.shard_index,
        n_shards=args.num_shards,
        settings=KappaSettings(),
        dtype="float32",
        device="cuda",
        dry_run=dry_run,
    )


def structure_rmsd(pred_and_ref: tuple[dict[str, Any], dict[str, Any]]) -> float | None:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    pred, ref = pred_and_ref
    # scale=False and stol=1 match the leaderboard's geo-opt protocol
    matcher = StructureMatcher(stol=1.0, scale=False)
    rmsd_and_max_dist = matcher.get_rms_dist(Structure.from_dict(pred), Structure.from_dict(ref))
    return None if rmsd_and_max_dist is None else rmsd_and_max_dist[0]


def kappa_metrics_if_complete(out_dir: Path) -> dict[str, float | None] | None:
    from matbench_discovery.metrics.phonons import evaluate_kappa_predictions
    from matbench_discovery.phonons import read_kappa_json
    from matbench_discovery.phonons.pipeline import (
        KAPPA_MANIFEST_FILE,
        KAPPA_RECORD_DIR,
        PHONONDB_N_STRUCTURES,
        merge_kappa_shards,
        read_kappa_manifest,
        write_kappa_artifacts,
    )

    manifest_path = out_dir / "kappa" / KAPPA_MANIFEST_FILE
    if not manifest_path.exists():
        print("no kappa run found, leaderboard kappa_SRME and CPS will be null")
        return None
    manifest = read_kappa_manifest(str(manifest_path))
    n_records = len(list((out_dir / "kappa" / KAPPA_RECORD_DIR).glob("*.json.gz")))
    if n_records < len(manifest.material_ids) or manifest.dry_run:
        print(
            f"kappa run incomplete ({n_records}/{len(manifest.material_ids)} records, "
            f"dry_run={manifest.dry_run}), leaderboard kappa_SRME and CPS will be null"
        )
        return None
    merged = merge_kappa_shards(str(out_dir / "kappa"), model_key=manifest.model_key)
    pred_path = out_dir / "kappa-preds.json.gz"
    write_kappa_artifacts(merged, pred_file_path=str(pred_path))
    if len(manifest.material_ids) != PHONONDB_N_STRUCTURES:
        print("kappa run does not cover all 103 PhononDB structures, skipping kappa_SRME")
        return None
    return evaluate_kappa_predictions(read_kappa_json(str(pred_path)))


def combined_performance_score(f1: float, rmsd: float, kappa_srme: float | None) -> float | None:
    """CPS as computed by the leaderboard (site/src/lib/combined-scores.svelte.ts):
    weighted mean of F1 (0.5), 1 - kappa_SRME/2 (0.4), and 1 - RMSD/0.15 clamped
    to [0, 1] (0.1); undefined until every component is available."""
    if kappa_srme is None:
        return None
    rmsd_score = max(0.0, min(1.0, 1 - rmsd / RMSD_BASELINE))
    return 0.5 * f1 + 0.4 * (1 - kappa_srme / 2) + 0.1 * rmsd_score


def compute_metrics(args: argparse.Namespace) -> None:
    import pandas as pd
    from matbench_discovery.data import df_wbm
    from matbench_discovery.energy import calc_energy_from_e_refs, mp_elemental_ref_energies
    from matbench_discovery.enums import DataFiles
    from matbench_discovery.metrics.discovery import stable_metrics
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

    cse_frame = pd.read_json(DataFiles.wbm_computed_structure_entries.path, lines=True)
    cse_frame = cse_frame.set_index("material_id")
    # Swap the model energy and relaxed structure into the WBM entry so MP2020
    # GGA/GGA+U and anion corrections apply exactly as on the leaderboard.
    entries = {}
    rmsd_pairs = {}
    for material_id, record in tqdm(records.items(), desc="building entries"):
        cse_dict = cse_frame.loc[material_id, "computed_structure_entry"]
        rmsd_pairs[material_id] = (record["structure"], cse_dict["structure"])
        cse = ComputedStructureEntry.from_dict(cse_dict)
        cse._energy = record["energy"]
        cse._structure = Structure.from_dict(record["structure"])
        entries[material_id] = cse
    MaterialsProject2020Compatibility().process_entries(entries.values(), verbose=True, clean=True)

    with ProcessPoolExecutor(max_workers=min(64, os.cpu_count())) as pool:
        rmsd_values = list(
            tqdm(
                pool.map(structure_rmsd, rmsd_pairs.values(), chunksize=32),
                total=len(rmsd_pairs),
                desc="RMSD vs DFT",
            )
        )
    # missing and unmatched structures count as 1.0 (the stol value), like the leaderboard
    rmsd_series = (
        pd.Series(rmsd_values, index=list(rmsd_pairs), name=RMSD_COL)
        .astype(float)
        .reindex(df_wbm.index)
        .fillna(1.0)
    )

    e_form_pred = pd.Series(
        {
            material_id: calc_energy_from_e_refs(cse, mp_elemental_ref_energies)
            for material_id, cse in entries.items()
        },
        name=E_FORM_PRED,
    ).reindex(df_wbm.index)
    each_pred = df_wbm[EACH_TRUE] + e_form_pred - df_wbm[E_FORM_DFT]
    unique_prototypes = df_wbm["unique_prototype"]
    kappa = kappa_metrics_if_complete(args.out_dir)
    kappa_srme = kappa["srme"] if kappa else None
    # stable_metrics(fillna=True) counts missing/failed relaxations against the model.
    unique_prototype_metrics = stable_metrics(
        df_wbm[EACH_TRUE][unique_prototypes], each_pred[unique_prototypes]
    )
    mean_rmsd = float(rmsd_series.mean())
    metrics = {
        # mirrors the main columns on https://matbench-discovery.materialsproject.org,
        # which reports discovery metrics on the unique-prototypes subset by default
        "leaderboard": {
            "CPS": combined_performance_score(
                unique_prototype_metrics["F1"], mean_rmsd, kappa_srme
            ),
            "F1": unique_prototype_metrics["F1"],
            "DAF": unique_prototype_metrics["DAF"],
            "Prec": unique_prototype_metrics["Precision"],
            "Acc": unique_prototype_metrics["Accuracy"],
            "MAE": unique_prototype_metrics["MAE"],
            "R2": unique_prototype_metrics["R2"],
            "kappa_SRME": kappa_srme,
            "RMSD": mean_rmsd,
        },
        "full_test_set": stable_metrics(df_wbm[EACH_TRUE], each_pred),
        "unique_prototypes": unique_prototype_metrics,
        "kappa": kappa,
        "n_relaxed": len(records),
        "n_failed": n_failed,
        "n_missing": int(e_form_pred.isna().sum()),
    }
    preds = pd.DataFrame({E_FORM_PRED: e_form_pred, EACH_PRED: each_pred, RMSD_COL: rmsd_series})
    preds.to_csv(args.out_dir / "preds.csv.gz")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    csv_text = write_leaderboard_csv(metrics["leaderboard"], args.out_dir)
    print(csv_text)


def write_leaderboard_csv(leaderboard: dict[str, Any], out_dir: Path) -> str:
    header = ",".join(["model", *leaderboard])
    row = ",".join([out_dir.name, *("" if v is None else str(v) for v in leaderboard.values())])
    csv_text = f"{header}\n{row}\n"
    (out_dir / "leaderboard.csv").write_text(csv_text)
    return csv_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=["relax", "kappa", "join"])
    parser.add_argument("model_path", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, help="default: evaluations/matbench_discovery/<model name>"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--limit", type=int, help="only run the first N structures (smoke runs; marks kappa dry)"
    )
    parser.add_argument("--backend", choices=["jax", "torch"], default="jax")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path("evaluations/matbench_discovery") / args.model_path.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # matbench-discovery would otherwise cache benchmark data inside site-packages,
    # which a venv rebuild would wipe
    os.environ.setdefault("MBD_CACHE_DIR", str(Path.home() / ".cache" / "matbench-discovery"))
    ensure_mbd_repo_files()

    # Relaxations share GPUs with other jobs (e.g. training), so allocate on
    # demand and reuse the persistent JAX compilation cache across shards.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR", str(Path("evaluations/jax_cache").absolute())
    )
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    if args.stage == "relax":
        run_relaxations(args)
    elif args.stage == "kappa":
        run_kappa(args)
    else:
        compute_metrics(args)


if __name__ == "__main__":
    main()
