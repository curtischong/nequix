"""Evaluate a Nequix checkpoint on the Matbench Discovery leaderboard metrics.

Stage 1 (relax) relaxes the ~257k WBM initial structures with the standard Matbench
Discovery force-field protocol (FrechetCellFilter to fmax=0.05). The relaxation
runs batched through torch-sim rather than one structure at a time through ASE:
WBM is a quarter of a million cells averaging ten atoms, and driven singly they
leave a GPU at a fraction of its power draw because every optimizer step is
microseconds of arithmetic behind milliseconds of kernel-launch and Python
overhead. Shard it across GPUs by launching one process per device; every shard
resumes from its own file:

    CUDA_VISIBLE_DEVICES=$i uv run --python 3.12 --extra mbd \
        python scripts/eval_matbench_discovery.py relax checkpoints/model.nqx \
        --shard-index $i --num-shards 8

Stage 2 (kappa) runs the PhononDB-103 thermal-conductivity benchmark (relax ->
FC2 -> FC3 -> Wigner conductivity) needed for the leaderboard's kappa_SRME and
CPS columns. It shards and resumes the same way:

    CUDA_VISIBLE_DEVICES=$i uv run --python 3.12 --extra mbd \
        python scripts/eval_matbench_discovery.py kappa checkpoints/model.nqx \
        --shard-index $i --num-shards 8

Stage 3 (join) hydrates the shard files into matbench-discovery's own artifact
pipeline (MP2020 corrections, formation energies, hull distances, RMSD and
symmetry vs the DFT-relaxed WBM structures), merges kappa records if complete,
and writes metrics.json whose "leaderboard" block mirrors the main columns on
https://matbench-discovery.materialsproject.org (CPS, F1, DAF, Prec, Acc, MAE,
R2, kappa_SRME, RMSD; discovery metrics use the site's default unique-prototypes
subset). CPS is null until the kappa stage has finished:

    uv run --python 3.12 --extra mbd \
        python scripts/eval_matbench_discovery.py join checkpoints/model.nqx

Benchmark data caches under ~/.cache/matbench-discovery (override with
MBD_CACHE_DIR). figshare.com/ndownloader is unreachable from
some machines (empty HTTP 202 responses); if the automatic download fails,
fetch the same file id from https://api.figshare.com/v2/file/download/<id>.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
from ase import Atoms
from tqdm import tqdm

RMSD_COL = "structure_rmsd_vs_dft"
# keep in sync with the matbench-discovery pin in pyproject.toml
MBD_REV = "21d9b1616442c3f97b30b6b1225d2c3742199112"
# CPS normalization constants from site/src/lib/labels.ts + combined-scores.svelte.ts
RMSD_BASELINE = 0.15
# Structures per ``optimize`` call. Not the GPU batch size -- torch-sim's
# autobatcher sizes that from free memory and refills it as systems converge --
# but the unit of durability: records are appended after each chunk, so a shard
# killed mid-campaign loses at most this much work.
CHUNK_STRUCTURES = 512
# Ceiling on the autobatcher's memory probe, which grows a trial batch by 1.6x
# until it runs out of memory. Its own default stops at half a million atoms:
# WBM cells average ten atoms, so that is a hundred thousand systems, and the
# last few probes cost far more than the relaxation they are sizing. A chunk is
# at most CHUNK_STRUCTURES of about fifty atoms, so anything above that is
# measuring a batch that will never be run.
MAX_PROBE_ATOMS = 32_768


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


def completed_ids(out_path: Path) -> set[str]:
    """Material ids already relaxed successfully in this shard's file.

    Error records are excluded so a structure that hit a transient failure (a
    neighbor's OOM, say) is retried on resume; the join stage keeps the
    successful record when both exist.
    """
    if not out_path.exists():
        return set()
    return {
        record["material_id"]
        for line in out_path.read_text().splitlines()
        if "energy" in (record := json.loads(line))
    }


def make_batcher(model: Any, shard_atoms: int) -> Any:
    """One in-flight autobatcher for the whole shard, probed once.

    The batcher keeps the GPU full by swapping a converged system out for a
    pending one instead of waiting on the slowest member of a fixed batch. It
    sizes itself by growing a trial batch by 1.6x until it runs out of memory,
    which is worth paying once per shard and not once per chunk --
    ``autobatcher=True`` builds a fresh one, and so re-probes, on every
    ``optimize`` call.

    The probe is bounded by the work that exists: no batch can exceed the
    shard, so neither should the probe.
    """
    from torch_sim.autobatching import InFlightAutoBatcher

    return InFlightAutoBatcher(
        model=model,
        memory_scales_with=model.memory_scales_with,
        max_atoms_to_try=max(1, min(MAX_PROBE_ATOMS, shard_atoms)),
    )


def relax_chunk(
    atoms_list: list[Atoms],
    model: Any,
    fmax: float,
    max_steps: int,
    batcher: Any,
) -> list[dict[str, Any]]:
    """Relax a chunk of structures together, one record per structure."""
    import torch_sim as ts
    from pymatgen.io.ase import AseAtomsAdaptor

    # ASE counts the cell degrees of freedom in fmax when it drives a filter.
    convergence_fn = ts.generate_force_convergence_fn(force_tol=fmax, include_cell_forces=True)
    final = ts.optimize(
        atoms_list,
        model=model,
        optimizer=ts.Optimizer.lbfgs,
        convergence_fn=convergence_fn,
        max_steps=max_steps,
        # alpha + step_size pin the inverse-Hessian guess to ASE's fixed
        # H0 = 1/alpha with undamped steps, rather than torch-sim's default
        # dynamically rescaled H0; max_step and max_history are ASE's maxstep
        # and memory. Together they are ase.optimize.LBFGS at its defaults.
        init_kwargs=dict(cell_filter=ts.CellFilter.frechet, alpha=70.0, step_size=1.0),
        max_step=0.2,
        max_history=100,
        autobatcher=batcher,
    )
    # Report the same criterion that stopped the optimizer, rather than a second
    # reduction that could disagree with it.
    converged = convergence_fn(final).tolist()
    energies = final.energy.detach().double().tolist()
    return [
        {
            "material_id": atoms.info["material_id"],
            "energy": energy,
            "structure": AseAtomsAdaptor.get_structure(relaxed).as_dict(),
            "converged": bool(is_converged),
            # Per-system step counts are not observable through the batched
            # optimizer: systems leave the batch as they converge. Only the
            # converged flag feeds a metric; n_steps is reporting metadata.
            "n_steps": max_steps,
        }
        for atoms, relaxed, energy, is_converged in zip(
            atoms_list, final.to_atoms(), energies, converged, strict=True
        )
    ]


def run_relaxations(args: argparse.Namespace) -> None:
    import torch
    from matbench_discovery.data import ase_atoms_from_zip
    from matbench_discovery.enums import DataFiles

    from nequix.torch_sim import NequixTorchSimModel

    out_path = shard_path(args.out_dir, args.shard_index, args.num_shards)
    done = completed_ids(out_path)
    # float32 matches the precision the JAX ASE path evaluated at, and resolves
    # an absolute DFT total to ~1e-4 eV, well under the three decimals the join
    # rounds to before scoring.
    model = NequixTorchSimModel(
        args.model_path,
        use_kernel=not args.no_kernel,
        device=torch.device(args.device),
        dtype=torch.float32,
    )

    atoms_list = ase_atoms_from_zip(DataFiles.wbm_initial_atoms.path, limit=args.limit)
    atoms_list.sort(key=lambda atoms: atoms.info["material_id"])
    shard = [
        atoms
        for atoms in atoms_list[args.shard_index :: args.num_shards]
        if atoms.info["material_id"] not in done
    ]
    if not shard:
        return
    chunks = [
        shard[start : start + CHUNK_STRUCTURES] for start in range(0, len(shard), CHUNK_STRUCTURES)
    ]
    batcher = make_batcher(model, sum(len(atoms) for atoms in shard))
    with out_path.open("a") as out_file:
        for chunk in tqdm(chunks, desc=f"relax shard {args.shard_index + 1}/{args.num_shards}"):
            try:
                records = relax_chunk(chunk, model, args.fmax, args.max_steps, batcher)
            except Exception as error:  # one bad chunk must not kill a shard
                records = [
                    {
                        "material_id": atoms.info["material_id"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                    for atoms in chunk
                ]
            for record in records:
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

    # A limit means a wiring check, not a benchmark: the manifest is marked dry
    # so the join stage refuses to report kappa_SRME from it.
    dry_run = args.limit is not None
    atoms_by_id = load_phonondb_atoms(dry_run=dry_run, dry_run_size=args.limit or 1)
    calculator = NequixCalculator(
        args.model_path, backend=args.backend, use_kernel=not args.no_kernel
    )
    run_kappa_shard(
        calculator=calculator,
        # The directory is the model, which is what the join stage names the
        # model column after too. Taking the key from the checkpoint stem
        # instead leaves every artifact of a run directed elsewhere keyed by
        # the checkpoint's filename.
        model_key=args.out_dir.name,
        atoms_by_id=atoms_by_id,
        dataset_path=DataFiles.phonondb_pbe_103_structures.path,
        shard_dir=str(args.out_dir / "kappa"),
        shard_index=args.shard_index,
        n_shards=args.num_shards,
        settings=KappaSettings(),
        dtype="float32",
        device=args.device,
        dry_run=dry_run,
    )


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
    # Only records that skipped conductivity (imaginary modes) carry the
    # conductivity_skipped flag; pandas fills NaN — which is truthy — into every
    # other row on read, failing the whole run instead of that one structure.
    for record in merged.records:
        record.result.setdefault("conductivity_skipped", False)
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


def clean_json(value: Any) -> Any:
    """Convert numpy values and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_records(out_dir: Path) -> tuple[list[Any], int, int]:
    """Hydrate every shard file into official records, newest success winning.

    A failed attempt followed by a successful retry counts as a success, so the
    successful record replaces it and only genuinely unrelaxed materials are
    counted as failures.
    """
    from matbench_discovery.discovery import RelaxationRecord

    shard_paths = sorted(out_dir.glob("relaxations-*.jsonl"))
    by_id: dict[str, RelaxationRecord] = {}
    for path in shard_paths:
        for line in path.read_text().splitlines():
            record = RelaxationRecord.from_dict(json.loads(line))
            if record.energy is not None or record.material_id not in by_id:
                by_id[record.material_id] = record
    n_failed = sum(record.energy is None for record in by_id.values())
    print(f"loaded {len(by_id)} relaxations ({n_failed} failed) from {out_dir}")
    return list(by_id.values()), len(shard_paths), n_failed


def geo_opt_metrics(records: list[Any], df_cse: Any, index: Any) -> dict[str, float]:
    """RMSD and symmetry of the relaxed structures against the DFT references.

    Reindexed onto the whole WBM set before scoring, so a material the campaign
    never relaxed counts against the model as the leaderboard's missing-structure
    RMSD of 1.0 rather than being quietly dropped from the mean.
    """
    from matbench_discovery.metrics.geo_opt import calc_geo_opt_metrics
    from matbench_discovery.structure.symmetry import (
        get_sym_info_from_structs,
        pred_vs_ref_struct_symmetry,
    )
    from pymatgen.core import Structure

    relaxed = {
        record.material_id: Structure.from_dict(record.structure)
        for record in records
        if record.structure is not None
    }
    reference = {
        material_id: Structure.from_dict(
            df_cse.loc[material_id, "computed_structure_entry"]["structure"]
        )
        for material_id in relaxed
    }
    df_sym = pred_vs_ref_struct_symmetry(
        get_sym_info_from_structs(relaxed),
        get_sym_info_from_structs(reference),
        relaxed,
        reference,
    )
    return calc_geo_opt_metrics(df_sym.reindex(index))


def compute_metrics(args: argparse.Namespace) -> None:
    """Join the relaxations (and kappa, if complete) into leaderboard columns.

    The scoring itself is matbench-discovery's own: shard files are hydrated
    into ``RelaxationRecord``s and handed to ``write_discovery_artifacts``
    (MP2020 corrections, formation energies, the prediction CSV) and then to
    ``calc_discovery_metrics`` and the geo-opt metrics. Going through the
    official path rather than reimplementing it is what keeps the numbers
    comparable with published ones: predictions far from DFT are masked,
    references and predictions are rounded to three decimals before scoring,
    and DAF is divided by the prevalence computed from unrounded hull
    distances.

    Energy convention: records carry raw DFT-style total energies in eV. Each
    is swapped into the original WBM ComputedStructureEntry, keeping its run
    metadata, so the MP2020 GGA/GGA+U and anion corrections apply exactly as
    they do on the leaderboard.
    """
    import pandas as pd
    from matbench_discovery.data import df_wbm
    from matbench_discovery.discovery import (
        DISCOVERY_PRED_COL,
        MergedDiscoveryRun,
        RelaxationSettings,
        write_discovery_artifacts,
    )
    from matbench_discovery.enums import DataFiles, TestSubset
    from matbench_discovery.metrics.discovery import (
        calc_discovery_metrics,
        prepare_model_predictions,
        wbm_uniq_proto_prevalence,
    )

    out_dir = args.out_dir
    records, n_shards, n_failed = load_records(out_dir)
    df_cse = pd.read_json(DataFiles.wbm_computed_structure_entries.path, lines=True).set_index(
        "material_id"
    )

    merged = MergedDiscoveryRun(
        model_key=out_dir.name,
        settings=RelaxationSettings(),
        records=tuple(records),
        run_metadata={},
        n_shards=max(n_shards, 1),
    )
    artifacts = write_discovery_artifacts(
        merged,
        pred_file_path=str(out_dir / "preds.csv.gz"),
        geo_opt_file_path=str(out_dir / "geo-opt.jsonl.gz"),
        df_wbm_cse=df_cse,
    )

    reference, predictions, subsets = prepare_model_predictions(
        df_wbm, artifacts.predictions[DISCOVERY_PRED_COL]
    )
    by_subset = calc_discovery_metrics(
        reference,
        predictions,
        subset_indices=subsets,
        uniq_proto_prevalence=wbm_uniq_proto_prevalence(),
    )
    # The leaderboard reports discovery metrics on the unique-prototypes subset.
    unique_prototypes = by_subset[TestSubset.uniq_protos]
    geo_opt = geo_opt_metrics(records, df_cse, df_wbm.index)
    kappa = kappa_metrics_if_complete(out_dir)
    kappa_srme = kappa["srme"] if kappa else None
    rmsd = geo_opt[RMSD_COL]

    metrics = {
        # mirrors the main columns on https://matbench-discovery.materialsproject.org
        "leaderboard": {
            "CPS": combined_performance_score(unique_prototypes["F1"], rmsd, kappa_srme),
            "F1": unique_prototypes["F1"],
            "DAF": unique_prototypes["DAF"],
            "Prec": unique_prototypes["Precision"],
            "Acc": unique_prototypes["Accuracy"],
            "MAE": unique_prototypes["MAE"],
            "R2": unique_prototypes["R2"],
            "kappa_SRME": kappa_srme,
            "RMSD": rmsd,
        },
        **{str(subset): subset_metrics for subset, subset_metrics in by_subset.items()},
        "geo_opt": geo_opt,
        "kappa": kappa,
        "n_relaxed": artifacts.n_success,
        "n_failed": n_failed,
        "n_missing": len(df_wbm) - artifacts.n_success,
    }
    # A metric is NaN whenever its inputs are empty -- F1 with no positive
    # prediction at all, which a partial campaign produces -- and a bare NaN
    # literal is not JSON, so a strict reader rejects the whole file rather
    # than the one column that has no value yet.
    metrics = clean_json(metrics)
    payload = json.dumps(metrics, indent=2, allow_nan=False)
    (out_dir / "metrics.json").write_text(payload)
    print(payload)
    print(write_leaderboard_csv(metrics["leaderboard"], out_dir))


def write_leaderboard_csv(leaderboard: dict[str, Any], out_dir: Path) -> str:
    header = ",".join(["model", *leaderboard])
    row = ",".join([out_dir.name, *("" if v is None else str(v) for v in leaderboard.values())])
    csv_text = f"{header}\n{row}\n"
    (out_dir / "leaderboard.csv").write_text(csv_text)
    return csv_text


def main() -> None:
    # A shard runs for days and is usually redirected to a log. Block buffering
    # would hold every line until the process exits, so a run that is killed or
    # still going looks identical to one that produced nothing.
    sys.stdout.reconfigure(line_buffering=True)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--limit",
        type=int,
        help="only run the first N structures, relaxed properly (marks kappa dry)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="cap the relaxation at a couple of steps; a wiring check, not a measurement",
    )
    parser.add_argument("--backend", choices=["jax", "torch"], default="jax", help="kappa only")
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="skip the OpenEquivariance kernels (environments without the oeq extra)",
    )
    args = parser.parse_args()

    if not args.model_path.is_file():
        parser.error(f"model checkpoint does not exist: {args.model_path}")
    if args.out_dir is None:
        args.out_dir = Path("evaluations/matbench_discovery") / args.model_path.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # matbench-discovery would otherwise cache benchmark data inside site-packages,
    # which a venv rebuild would wipe
    os.environ.setdefault("MBD_CACHE_DIR", str(Path.home() / ".cache" / "matbench-discovery"))
    ensure_mbd_repo_files()

    # The kappa stage's jax backend shares GPUs with other jobs (e.g. training),
    # so allocate on demand and reuse the persistent JAX compilation cache.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR", str(Path("evaluations/jax_cache").absolute())
    )
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    if args.stage == "relax":
        from matbench_discovery.discovery import RelaxationSettings, dry_run_settings

        # A dry run caps the steps rather than waiting out a real relaxation. That
        # cap is what keeps a smoke run seconds long: nothing converges under a
        # barely-trained checkpoint, so every structure otherwise rides the
        # 500-step cap, and batched they hold the whole chunk there together.
        #
        # It is deliberately not implied by --limit: relaxing a subset properly
        # is a sample-sized measurement, while a two-step relaxation measures
        # neither the energies nor the convergence criterion.
        settings = RelaxationSettings(max_force=args.fmax, max_steps=args.max_steps)
        if args.dry_run:
            settings = dry_run_settings(settings)
        args.fmax, args.max_steps = settings.max_force, settings.max_steps
        run_relaxations(args)
    elif args.stage == "kappa":
        run_kappa(args)
    else:
        compute_metrics(args)


if __name__ == "__main__":
    main()
