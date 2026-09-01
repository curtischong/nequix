import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase.db import connect
from atompack import Database
from tqdm import tqdm


def source_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        paths = sorted(input_path.rglob("*.aselmdb"))
    else:
        paths = [input_path]
    if not paths:
        raise ValueError(f"No ASE-LMDB files found in {input_path}")
    return paths


def source_counts(paths: list[Path]) -> list[int]:
    counts = []
    for path in paths:
        with connect(str(path), use_lock_file=False) as database:
            counts.append(database.count())
    return counts


def atom_record(atoms) -> dict:
    record = {
        "positions": np.ascontiguousarray(atoms.get_positions(), dtype=np.float32),
        "atomic_numbers": np.ascontiguousarray(atoms.get_atomic_numbers(), dtype=np.uint8),
        "energy": float(atoms.get_potential_energy()),
        "forces": np.ascontiguousarray(atoms.get_forces(), dtype=np.float32),
        "pbc": np.ascontiguousarray(atoms.pbc, dtype=bool),
    }
    record["cell"] = (
        np.ascontiguousarray(atoms.cell.array, dtype=np.float64) if record["pbc"].any() else None
    )
    try:
        record["stress"] = np.ascontiguousarray(atoms.get_stress(voigt=False), dtype=np.float64)
    except Exception:
        record["stress"] = None
    return record


def record_key(record: dict) -> tuple:
    return (
        len(record["atomic_numbers"]),
        record["cell"] is not None,
        record["stress"] is not None,
    )


def add_records(database: Database, records: list[dict]) -> None:
    if not records:
        return
    kwargs = {
        "energy": np.ascontiguousarray([record["energy"] for record in records]),
        "forces": np.ascontiguousarray(
            np.stack([record["forces"] for record in records]), dtype=np.float32
        ),
        "pbc": np.ascontiguousarray(np.stack([record["pbc"] for record in records])),
    }
    if records[0]["cell"] is not None:
        kwargs["cell"] = np.ascontiguousarray(
            np.stack([record["cell"] for record in records]), dtype=np.float64
        )
    if records[0]["stress"] is not None:
        kwargs["stress"] = np.ascontiguousarray(
            np.stack([record["stress"] for record in records]), dtype=np.float64
        )
    database.add_arrays_batch(
        np.ascontiguousarray(
            np.stack([record["positions"] for record in records]), dtype=np.float32
        ),
        np.ascontiguousarray(
            np.stack([record["atomic_numbers"] for record in records]), dtype=np.uint8
        ),
        **kwargs,
    )


def convert_paths(paths: list[Path], output_path: Path, batch_size: int, progress=None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_database = Database(str(output_path), overwrite=True)
    processed = 0
    for path in paths:
        with connect(str(path), use_lock_file=False) as source_database:
            batch: list[dict] = []
            batch_key = None
            for row in source_database.select():
                record = atom_record(row.toatoms())
                key = record_key(record)
                if batch and (key != batch_key or len(batch) == batch_size):
                    add_records(output_database, batch)
                    processed += len(batch)
                    if progress is not None:
                        progress.update(len(batch))
                    batch.clear()
                batch_key = key
                batch.append(record)
            if batch:
                add_records(output_database, batch)
                processed += len(batch)
                if progress is not None:
                    progress.update(len(batch))

    output_database.flush()
    return processed


def partition_paths(paths: list[Path], counts: list[int], n_workers: int) -> list[list[Path]]:
    n_workers = min(n_workers, len(paths))
    groups = []
    group = []
    group_size = 0
    remaining_size = sum(counts)
    remaining_workers = n_workers
    for path, count in zip(paths, counts):
        group.append(path)
        group_size += count
        target_size = remaining_size / remaining_workers
        if len(groups) < n_workers - 1 and group_size >= target_size:
            groups.append(group)
            group = []
            remaining_size -= group_size
            remaining_workers -= 1
            group_size = 0
    if group:
        groups.append(group)
    return groups


def merge_parts(
    part_paths: list[Path], output_path: Path, total: int, batch_size: int = 4096
) -> None:
    output_database = Database(str(output_path), overwrite=True)
    with tqdm(total=total, unit="structures", desc="Merging") as progress:
        for part_path in part_paths:
            part_database = Database.open(str(part_path))
            for offset in range(0, len(part_database), batch_size):
                indices = list(range(offset, min(offset + batch_size, len(part_database))))
                molecules = part_database.get_molecules(indices)
                output_database.add_molecules(molecules)
                progress.update(len(molecules))
    output_database.flush()


def preprocess(
    input_path: str, output_path: str, batch_size: int = 512, n_workers: int = 1
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.suffix != ".atp":
        raise ValueError(f"AtomPack output path must end in .atp: {output_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    paths = source_paths(input_path)
    counts = source_counts(paths)
    total = sum(counts)
    if n_workers == 1 or len(paths) == 1:
        with tqdm(total=total, unit="structures") as progress:
            convert_paths(paths, output_path, batch_size, progress)
        return

    groups = partition_paths(paths, counts, n_workers)
    parts_dir = output_path.with_suffix(output_path.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in parts_dir.glob("part-*.atp"):
        stale_path.unlink()
    part_paths = [parts_dir / f"part-{index:03d}.atp" for index in range(len(groups))]

    with tqdm(total=total, unit="structures", desc="Converting") as progress:
        with ProcessPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(convert_paths, group, part_path, batch_size): part_path
                for group, part_path in zip(groups, part_paths)
            }
            for future in as_completed(futures):
                progress.update(future.result())

    merge_parts(part_paths, output_path, total)
    if len(Database.open(str(output_path))) != total:
        raise RuntimeError(f"Merged AtomPack length does not match source length: {output_path}")
    for part_path in part_paths:
        part_path.unlink()
    parts_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path, args.batch_size, args.n_workers)


if __name__ == "__main__":
    main()
