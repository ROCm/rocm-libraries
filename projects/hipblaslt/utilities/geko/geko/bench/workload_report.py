# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Before/after workload benchmarks around a stage-merged tuned library.

After tuning, hipBLASLt still serves sizes from the stock logic unless the
tuned winners are merged into that logic and rebuilt. This module:

1. Copies the stock asm_full logic for the architecture and merges the tuned
   exact winners into it (a staging copy, not the source tree), then builds
   that library with TensileCreateLibrary.
2. Benchmarks the original workload YAML once on the installed library and
   once on the staged library.
3. Drops any tuned winner that measures slower than the stock kernel, then
   rebuilds and re-benchmarks, so no GEMM regresses.

Merging follows how the runtime selects a solution. Each hardware branch
(architecture, CU count, chip id) holds one row per logic type, and exact
(Equality) rows are tried before Prediction or GridBased rows. A tuned file
therefore goes into every compatible hardware branch that already serves its
problem type: merged into a stock Equality file, or written beside a
Prediction file as a sibling Equality file with that branch's hardware fields.
Sizes not in the tuned table fall through to the unchanged stock logic.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from geko import logger
from geko.bench.bench import run as run_bench
from geko.library.operations import load_library
from geko.library.operations import merge as merge_logic
from geko.utils import ensure_tensile_importable

_ASM_FULL = Path("library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full")
_TOKEN_RE = re.compile(r"Cijk_[^_]+_[^_]+_([^_]+)_")
_MERGEABLE_TYPES = ("Equality",)
_KEY_COLUMNS = ["transA", "transB", "m", "n", "k", "batch_count"]


def datatype_tokens(names: list[str]) -> list[str]:
    """Return the problem-type tokens embedded in tuned logic filenames.

    ``gfx1250_Cijk_Alik_Bljk_BBS_BH_...yaml`` yields ``BBS``.
    """
    tokens: list[str] = []
    for name in names:
        match = _TOKEN_RE.search(name)
        token = match.group(1) if match else None
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _hardware(lib) -> tuple[str | None, int | None, tuple[str, ...]]:
    """Return ``(architecture, cu_count, device_names)`` of a logic library."""
    data = lib.data
    if lib.format == "dict":
        arch = data.get("ArchitectureName")
        cu = data.get("CUCount")
        devices = data.get("DeviceNames") or []
    else:
        arch_field = data[2]
        if isinstance(arch_field, dict):
            arch = arch_field.get("Architecture")
            cu = arch_field.get("CUCount")
        else:
            arch, cu = arch_field, None
        devices = data[3] if len(data) > 3 and data[3] else []
    if isinstance(devices, str):
        devices = [devices]
    return arch, cu, tuple(devices)


def _set_hardware(lib, cu_count: int | None, devices: tuple[str, ...]) -> None:
    data = lib.data
    if lib.format == "dict":
        data["CUCount"] = cu_count
        data["DeviceNames"] = list(devices)
        return
    arch_field = data[2]
    if isinstance(arch_field, dict):
        arch_field["CUCount"] = cu_count
    else:
        data[2] = {"Architecture": arch_field, "CUCount": cu_count}
    data[3] = list(devices)


def _hardware_compatible(tuned_hw: tuple, stock_hw: tuple) -> bool:
    """True when tuned winners may override logic in the stock hardware branch."""
    tuned_arch, tuned_cu, tuned_devices = tuned_hw
    stock_arch, stock_cu, stock_devices = stock_hw
    if tuned_arch != stock_arch:
        return False
    if tuned_cu is not None and stock_cu is not None and tuned_cu != stock_cu:
        return False
    if tuned_devices and stock_devices and not set(tuned_devices) & set(stock_devices):
        return False
    return True


def stage_merge_tuned_library(
    hipblaslt_path: str | Path,
    tuned_dir: str | Path,
    staging_logic: str | Path,
    arch: str | None = None,
    datatype_filter: bool = False,
) -> Path:
    """Merge tuned logic YAML into a copy of the stock asm_full tree.

    For each tuned file, stock files with the same basename are grouped by
    hardware branch (the directory above the logic-type directory). In each
    branch whose hardware is compatible with the tuned file, the tuned winners
    are merged into the stock Equality file with TensileMergeLibrary, or, when
    the branch only has Prediction/GridBased logic for that problem type,
    written as ``<branch>/Equality/<name>`` with the branch's CU count and
    device names. The tuned file is also placed under ``<arch>/Equality`` with
    its own hardware fields unless a branch with that hardware already holds it.

    Args:
        hipblaslt_path: hipBLASLt checkout root.
        tuned_dir: Directory of tuned logic YAML (``final_libs``).
        staging_logic: Destination logic root. Replaced if it already exists.
        arch: TensileCreateLibrary architecture. Read from the first tuned
            file when omitted.
        datatype_filter: Copy only stock files whose name carries a datatype
            token of the tuned files. Faster to build; the default copies the
            full stock logic so the staged library matches the installed one.

    Returns:
        The staging logic directory.

    Raises:
        FileNotFoundError: Stock logic for ``arch`` is missing.
        ValueError: ``tuned_dir`` has no YAML files.
    """
    hipblaslt_path = Path(hipblaslt_path)
    ensure_tensile_importable(hipblaslt_path)
    tuned_dir = Path(tuned_dir)
    tuned_files = sorted(tuned_dir.glob("*.yaml"))
    if not tuned_files:
        raise ValueError(f"No tuned logic yaml in '{tuned_dir}'")

    if arch is None:
        arch = load_library(tuned_files[0]).schedule
    stock_root = hipblaslt_path / _ASM_FULL / arch
    if not stock_root.is_dir():
        raise FileNotFoundError(f"Stock logic directory not found: '{stock_root}'")

    tokens = datatype_tokens([path.name for path in tuned_files]) if datatype_filter else []
    staging_logic = Path(staging_logic)
    if staging_logic.exists():
        shutil.rmtree(staging_logic)
    staging_logic.mkdir(parents=True)

    copied = 0
    for src in stock_root.rglob("*.yaml"):
        if tokens and not any(token in src.name for token in tokens):
            continue
        dest = staging_logic / src.relative_to(stock_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    logger.info(
        f"Staged {copied} stock logic file(s) for {tokens or ['all']} under '{staging_logic}'"
    )

    incremental_root = staging_logic.parent / "_incremental"
    if incremental_root.exists():
        shutil.rmtree(incremental_root)

    for tuned in tuned_files:
        _stage_one(hipblaslt_path, staging_logic, incremental_root, arch, tuned)
    return staging_logic


def _stage_one(
    hipblaslt_path: Path, staging_logic: Path, incremental_root: Path, arch: str, tuned: Path
) -> None:
    tuned_lib = load_library(tuned)
    tuned_hw = _hardware(tuned_lib)

    branches: dict[Path, list[Path]] = {}
    for match in sorted(staging_logic.rglob(tuned.name)):
        branches.setdefault(match.parent.parent, []).append(match)

    covered: set[tuple] = set()
    for branch, matches in branches.items():
        stock_libs = [(match, load_library(match)) for match in matches]
        branch_hw = _hardware(stock_libs[0][1])
        if not _hardware_compatible(tuned_hw, branch_hw):
            logger.info(
                f"Leaving '{branch.relative_to(staging_logic)}' unchanged for '{tuned.name}': "
                f"hardware {branch_hw} does not match tuned hardware {tuned_hw}"
            )
            continue

        exact = [match for match, lib in stock_libs if lib.type in _MERGEABLE_TYPES]
        merged = False
        for match in exact:
            inc_dir = incremental_root / tuned.stem / match.parent.relative_to(staging_logic)
            inc_dir.mkdir(parents=True)
            shutil.copy2(tuned, inc_dir / tuned.name)
            try:
                merge_logic(hipblaslt_path, match.parent, inc_dir, match.parent, eff=False, force=True)
                logger.info(f"Merged '{tuned.name}' into '{match}'")
                merged = True
            except ValueError as exc:
                logger.warning(f"TensileMergeLibrary could not merge into '{match}': {exc}")
        if not merged:
            types = sorted({str(lib.type) for _, lib in stock_libs})
            _place_new_logic(branch / "Equality", tuned_lib, tuned.name, branch_hw)
            logger.info(
                f"Branch '{branch.relative_to(staging_logic)}' serves '{tuned.name}' with "
                f"{types} logic; added tuned exact winners beside it"
            )
        covered.add(branch_hw[1:])

    if tuned_hw[1:] not in covered:
        _place_new_logic(staging_logic / arch / "Equality", tuned_lib, tuned.name, tuned_hw)


def _place_new_logic(dest_dir: Path, tuned_lib, name: str, hardware: tuple) -> None:
    dest = dest_dir / name
    if dest.exists():
        # Never replace stock logic with the tuned subset; that would drop
        # every stock size in the file.
        logger.warning(f"Not overwriting existing logic '{dest}'")
        return
    lib = type(tuned_lib)(copy.deepcopy(tuned_lib.data), tuned_lib.name)
    _set_hardware(lib, hardware[1], hardware[2])
    dest_dir.mkdir(parents=True, exist_ok=True)
    lib.dump(dest_dir, name)
    logger.info(f"Added tuned logic '{name}' at '{dest}' (CUCount={hardware[1]})")


def build_staged_library(
    hipblaslt_path: str | Path,
    logic_dir: str | Path,
    output_dir: str | Path,
    arch: str,
    jobs: int | None = None,
    log_path: str | Path | None = None,
) -> Path:
    """Build a msgpack lazy library from staged logic. Returns ``output_dir``."""
    hipblaslt_path = Path(hipblaslt_path)
    ensure_tensile_importable(hipblaslt_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    n_jobs = jobs if jobs is not None else (os.cpu_count() or 8)
    cmd = [
        str(hipblaslt_path / "tensilelite/tensilelite/bin/TensileCreateLibrary"),
        "--code-object-version",
        "5",
        "--library-format",
        "msgpack",
        "--architecture",
        arch,
        "--jobs",
        str(n_jobs),
        "--logic-filter",
        "*",
        str(Path(logic_dir).resolve()),
        str(output_dir.resolve()),
        "HIP",
    ]
    logger.info(f"Building staged library: {' '.join(cmd)}")
    if log_path is None:
        subprocess.run(cmd, check=True)
    else:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return output_dir


def write_workload_benchmark(
    hipblaslt_path: str | Path,
    workload: str | Path,
    log_path: str | Path,
    devices: list[int],
    custom_lib_dir: str | Path | None = None,
    bench_freq: bool = False,
) -> pd.DataFrame:
    """Run hipblaslt-bench on ``workload`` and write the log plus a CSV.

    Returns the benchmark table. The log is the raw hipblaslt-bench output and
    the CSV is written next to it with a ``.csv`` suffix.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df = run_bench(
        hipblaslt_path,
        workload,
        log_path,
        custom_lib_dir=custom_lib_dir,
        devices=devices,
        cache=False,
        bench_freq=bench_freq,
    )
    csv_path = log_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Benchmark log '{log_path}' and csv '{csv_path}' ({len(df)} rows)")
    return df


def _time_column(df: pd.DataFrame) -> str:
    return "mean_us" if "mean_us" in df.columns and df["mean_us"].notna().all() else "us"


def _with_time(df: pd.DataFrame) -> pd.DataFrame:
    """Copy ``df`` with ``time_us`` set from the adaptive mean, else ``us``."""
    out = df.copy()
    out["time_us"] = out[_time_column(out)]
    return out


def compare_runs(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Join aggregated before/after tables and classify each GEMM.

    ``outcome`` is ``same_kernel`` when the stage-merged library selects the
    stock solution, ``tuned_faster`` when it selects a different solution that
    is not slower, and ``tuned_slower`` otherwise.
    """
    keys = [c for c in _KEY_COLUMNS if c in before.columns]
    left = before[keys + ["time_us", "solution"]].rename(
        columns={"time_us": "before_us", "solution": "before_solution"}
    )
    right = after[keys + ["time_us", "solution"]].rename(
        columns={"time_us": "after_us", "solution": "after_solution"}
    )
    table = left.merge(right, on=keys, how="inner")
    table["ratio"] = table["after_us"] / table["before_us"]
    table["same_solution"] = table["before_solution"] == table["after_solution"]
    table["outcome"] = "same_kernel"
    changed = ~table["same_solution"]
    table.loc[changed & (table["after_us"] <= table["before_us"]), "outcome"] = "tuned_faster"
    table.loc[changed & (table["after_us"] > table["before_us"]), "outcome"] = "tuned_slower"
    return table


def drop_tuned_sizes(tuned_dir: str | Path, output_dir: str | Path, rejected: pd.DataFrame) -> int:
    """Copy tuned logic to ``output_dir`` without the rejected GEMM sizes.

    Exact-logic keys are ``[M, N, Batch, K]``; transposes come from each
    file's ProblemType. Files left with no sizes are not written. Returns the
    number of sizes removed.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    drop = {
        (str(r.transA).upper() == "T", str(r.transB).upper() == "T", int(r.m), int(r.n), int(r.batch_count), int(r.k))
        for r in rejected.itertuples()
    }
    removed = 0
    for path in sorted(Path(tuned_dir).glob("*.yaml")):
        lib = load_library(path)
        problem = lib.problem or {}
        trans = (bool(problem.get("TransposeA")), bool(problem.get("TransposeB")))
        kept = []
        for entry in lib.sizes or []:
            size = tuple(int(v) for v in entry[0][:4])
            if trans + size in drop:
                removed += 1
            else:
                kept.append(entry)
        if not kept:
            logger.info(f"All tuned sizes in '{path.name}' were rejected; leaving it out")
            continue
        lib.sizes = kept
        lib.dump(output_dir, path.name)
    return removed


def _bench_pair(
    hipblaslt_path: Path,
    workload: Path,
    out_dir: Path,
    devices: list[int],
    build_dir: Path,
    bench_freq: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Workload benchmark: installed library")
    before = write_workload_benchmark(
        hipblaslt_path, workload, out_dir / "before_tune.log", devices,
        custom_lib_dir=None, bench_freq=bench_freq,
    )
    logger.info("Workload benchmark: stage-merged library")
    after = write_workload_benchmark(
        hipblaslt_path, workload, out_dir / "after_tune.log", devices,
        custom_lib_dir=build_dir, bench_freq=bench_freq,
    )
    return _with_time(before), _with_time(after)


def benchmark_before_and_after(
    hipblaslt_path: str | Path,
    workload: str | Path,
    tuned_dir: str | Path,
    results_dir: str | Path,
    devices: list[int],
    staging_root: str | Path,
    bench_freq: bool = False,
    jobs: int | None = None,
    datatype_filter: bool = False,
    max_guard_passes: int = 2,
) -> pd.DataFrame | None:
    """Benchmark the workload on the installed and stage-merged libraries.

    Writes into ``results_dir``:

    - ``before_tune.log/.csv`` and ``after_tune.log/.csv``: the final pass.
    - ``before_after.csv``: per-GEMM times, selected solutions and outcome.
    - ``validated_libs/``: the tuned logic that survived the regression guard.
    - ``workload_pass<N>/``: per-round logs and CSVs of every pass.

    A tuned winner that the stage-merged library selects but that measures
    slower than the stock kernel is dropped from the tuned logic, and the
    library is rebuilt and re-benchmarked, up to ``max_guard_passes`` times.

    Returns the final ``before_after`` table, or None when there is no tuned
    logic to merge.
    """
    hipblaslt_path = Path(hipblaslt_path)
    workload = Path(workload)
    tuned_dir = Path(tuned_dir)
    results_dir = Path(results_dir)
    staging_root = Path(staging_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not workload.is_file():
        raise FileNotFoundError(f"Workload YAML not found: '{workload}'")

    tuned_files = list(tuned_dir.glob("*.yaml"))
    if not tuned_files:
        logger.warning(f"No tuned logic in '{tuned_dir}'. Skipping the workload benchmark.")
        return None
    arch = load_library(tuned_files[0]).schedule

    current = tuned_dir
    table = before = after = None
    pass_dir = results_dir
    for attempt in range(max_guard_passes + 1):
        pass_dir = results_dir / f"workload_pass{attempt + 1}"
        logic_dir = stage_merge_tuned_library(
            hipblaslt_path, current, staging_root / "logic", arch=arch, datatype_filter=datatype_filter
        )
        build_dir = build_staged_library(
            hipblaslt_path, logic_dir, staging_root / "build", arch, jobs=jobs,
            log_path=staging_root / "TensileCreateLibrary.log",
        )
        before, after = _bench_pair(
            hipblaslt_path, workload, pass_dir, devices, build_dir, bench_freq
        )
        table = compare_runs(before, after)
        slower = table[table["outcome"] == "tuned_slower"]
        counts = table["outcome"].value_counts().to_dict()
        logger.info(f"Workload pass {attempt + 1}: {counts}")
        if slower.empty:
            break
        if attempt == max_guard_passes:
            logger.warning(
                f"{len(slower)} tuned winner(s) are still slower than stock after "
                f"{max_guard_passes} guard pass(es)"
            )
            break
        next_dir = staging_root / f"tuned_pass{attempt + 2}"
        removed = drop_tuned_sizes(current, next_dir, slower)
        logger.info(f"Dropped {removed} tuned size(s) slower than stock; rebuilding")
        current = next_dir
        if not any(next_dir.glob("*.yaml")):
            logger.warning("Every tuned winner was rejected; nothing left to merge")
            break

    validated = results_dir / "validated_libs"
    if validated.exists():
        shutil.rmtree(validated)
    shutil.copytree(current, validated)

    before.to_csv(results_dir / "before_tune.csv", index=False)
    after.to_csv(results_dir / "after_tune.csv", index=False)
    table.to_csv(results_dir / "before_after.csv", index=False)
    shutil.copy2(pass_dir / "before_tune.log", results_dir / "before_tune.log")
    shutil.copy2(pass_dir / "after_tune.log", results_dir / "after_tune.log")
    logger.info(
        f"Workload benchmark written to '{results_dir}' "
        f"({table['outcome'].value_counts().to_dict()})"
    )
    return table
