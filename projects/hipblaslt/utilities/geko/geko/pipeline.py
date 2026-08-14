# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""High-level pipeline entry points used by the geko CLI, scripts, and tests.

This module owns the configure / optimize / search / bench workflows that
were previously spread across configure.py, optimize.py, and
search.py at the package root. Each public function wraps the lower-level
geko.bench, geko.optim, and geko.library helpers and
maintains the run_state.json in the per-run workdir so phases can be
resumed and audited.

Public entry points:
    run_bench: Summarize a hipBLASLt workload log via standard_benchmark.
    run_search: Dense search workflow (summarize + search + analyze + final libs).
    run_configure: Summarize a workload and emit per-GEMM tuning YAMLs.
    run_optimize: Run optim, merge libs, analyze, and emit final libs.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from geko import bench, library, logger, optim, search, _set_log_level
from geko.config_generator.load_input_config import gemm_configs_from_gemm_dataframe
from geko.bench.utils import update_lib_source
from geko.constants import SUPPORTED_ARCH
from geko.schemas import GemmConfig, RunState
from geko.utils import parse_devices


def _prepare_workflow_context(
    hipblaslt_path: str | Path,
    log_file: str | Path,
    workdir: str | Path,
    verbose: int,
) -> tuple[Path, Path, Path, RunState, Path]:
    """Validate paths, set log level, ensure workdir, and load/create run state."""
    hip_path = Path(hipblaslt_path)
    if not hip_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hip_path}'")

    log_path = Path(log_file)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: '{log_path}'")

    _set_log_level(verbose)

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    state_path = workdir_path / "run_state.json"
    if state_path.exists():
        state = RunState.load(state_path)
        state.verify(log_path)
    else:
        state = RunState.create(log_path)

    return hip_path, log_path, workdir_path, state, state_path


def run_bench(
    hipblaslt_path: str,
    workload_path: str,
    output_dir: str | Path,
    devices: Sequence[int] | None = None,
    benchmark_duration: float = 0.5,
    bench_freq: bool = False,
    device: int | None = None,
) -> int:
    """Summarize a workload log into output_dir (benchmark path only; no tuning).

    Parses the workload, materializes a bench YAML, runs bench.standard_benchmark
    (probe then scaled iterations), and leaves summary artifacts under output_dir.

    Args:
        hipblaslt_path: hipBLASLt checkout root.
        workload_path: GEMM workload YAML or CSV with GEMM_LOG_FIELDS.
        output_dir: Writes summary.csv, gemms.csv, and intermediate YAML here.
        devices: GPU indices for summarizing.
        benchmark_duration: Target seconds per cold and per timed phase for standard_benchmark.
        bench_freq: Passed to bench.standard_benchmark (HIPBLASLT_BENCH_FREQ).
        device: Backward-compatible single-device alias. If set, overrides devices.

    Returns:
        0 on success; 1 if hipBLASLt or workload path is missing (logged).
    """
    hip_path = Path(hipblaslt_path)
    log_path = Path(workload_path)
    out = Path(output_dir)
    if not hip_path.is_dir():
        logger.error(f"hipBLASLt path not found: '{hip_path}'")
        return 1
    if not log_path.is_file():
        logger.error(f"Workload file not found: '{log_path}'")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    if device is not None:
        devices = [device]
    if devices is None:
        devices = [0]  # Default to device 0 if not specified
    devices = parse_devices(devices)

    data = bench.log.update(bench.log.parse(log_path, as_df=False))[0]
    logger.info(f"Working on '{log_path}' with {len(data)} GEMMs...")
    bench_yaml = out / f"{log_path.stem}.yaml"
    bench.log.dump(data, bench_yaml)
    bench_out = bench_yaml.with_suffix(".out")
    _ = bench.standard_benchmark(
        hip_path,
        bench_yaml,
        bench_out,
        devices=devices,
        duration=benchmark_duration,
        bench_freq=bench_freq,
    )
    logger.info(f"Benchmark outputs under '{out.resolve()}'")
    return 0


def run_search(
    hipblaslt_path: str,
    log_file: str,
    *,
    devices: Sequence[int] | None = None,
    keep_thr: float = 0.1,
    up_thr: float = 1.03,
    workdir: str = "workdir",
    verbose: int = 1,
    duration: float = 0.04,
    bench_freq: bool = False,
) -> None:
    """Run the dense search workflow end-to-end.

    Pipeline stages:

        1. Validate paths and load/create run_state.json.
        2. Summarize the workload with bench.log.summarize and filter
           by keep_thr.
        3. Build per-GEMM search configs and run search.run across
           devices.
        4. Extract winners, merge into workdir/libs, and analyze the
           result via optim.analyze.
        5. Emit the remapped final library to workdir/final_libs.

    Args:
        hipblaslt_path (str): hipBLASLt checkout root.
        log_file (str): Workload YAML/CSV path.
        devices (Sequence[int], optional): GPU device IDs used by search and
            Defaults to None, which is interpreted as all 8 devices if not specified.
        keep_thr (float, optional): Minimum grouped % of total time to keep
            a GEMM row when summarizing. Defaults to 0.1.
        up_thr (float, optional): Minimum speedup ratio vs reference kept by
            optim.analyze. Defaults to 1.03.
        workdir (str, optional): Per-run working directory. Created if
            absent. Defaults to "workdir".
        verbose (int, optional): Logging verbosity passed to
            _set_log_level (0 = WARNING, 1 = INFO). Defaults to 1.
        duration (float, optional): Per-config benchmark duration (seconds)
            used by search.configure. Defaults to 0.04.
        bench_freq (bool, optional): Forwarded to bench.log.summarize,
            search.run, and optim.analyze (controls HIPBLASLT_BENCH_FREQ).
            Defaults to False.

    Raises:
        FileNotFoundError: If hipblaslt_path or log_file is missing.
        ValueError: If all benchmarks fail (no winners are produced).
    """
    hipblaslt_path, log_file, workdir, state, state_path = _prepare_workflow_context(
        hipblaslt_path, log_file, workdir, verbose
    )
    if devices is None:
        devices = list(range(8))  # Default to all 8 devices if not specified
    devices = parse_devices(devices)
    final_output_dir = workdir / "final_libs"
    full_output_dir = workdir / "full_libs"

    logger.info("Starting search...")
    logger.info(f"hipBLASLt path: '{hipblaslt_path}'")
    logger.info(f"Log file: '{log_file}'")
    logger.info(f"Working directory: '{workdir}'")
    logger.info(f"Final output directory: '{final_output_dir}'")
    logger.info(f"Full output directory: '{full_output_dir}'")
    logger.info(f"Keep threshold: {keep_thr}")
    logger.info(f"Performance threshold: {up_thr}")

    summary_df, uniq_df = bench.log.summarize(
        hipblaslt_path,
        log_file,
        output_dir=workdir,
        devices=devices,
        keep_thr=keep_thr,
        cache=True,
        bench_freq=bench_freq,
    )
    if uniq_df.empty:
        logger.warning("No GEMM operations found after filtering. Consider lowering keep_thr")
        return

    configs = search.configure(summary_df, duration=duration)
    state.configured = True
    state.optimized = False
    state.dump(state_path)

    lib_dir = workdir / "libs"
    results_dir = workdir / "results"
    benchmarks_dir = workdir / "benchmarks"
    custom_lib_dir = workdir / "build"

    winners = search.run(
        hipblaslt_path,
        configs,
        workdir / "search",
        devices=devices,
        bench_freq=bench_freq,
    )
    if winners.empty:
        raise ValueError("All benchmarks failed!")
  
    winners_file = workdir / "winners.csv"
    cleanup_needed = True
    if winners_file.is_file():
        cached_winners = pd.read_csv(winners_file)
        cached_winners = cached_winners.sort_values(cached_winners.columns.tolist()).reset_index(drop=True)
        if cached_winners.equals(winners.sort_values(winners.columns.tolist()).reset_index(drop=True)):
            cleanup_needed = False
          
    if cleanup_needed:
        if lib_dir.is_dir():
            shutil.rmtree(lib_dir)
        if custom_lib_dir.is_dir():
            shutil.rmtree(custom_lib_dir)
        if benchmarks_dir.is_dir():
            shutil.rmtree(benchmarks_dir)

    winners.to_csv(winners_file, index=False)

    match_table_path = hipblaslt_path / "build/release/device-library/MatchTable.yaml"
    libs = library.operations.extract_solutions(winners, match_table_path)
    libs.dump(lib_dir)
    logger.info(f"Extracted library available in: '{lib_dir}'")

    if results_dir.is_dir():
        shutil.rmtree(results_dir)
    if final_output_dir.is_dir():
        shutil.rmtree(final_output_dir)
    if full_output_dir.is_dir():
        shutil.rmtree(full_output_dir)

    logger.info("Analyzing results...")
    res, full_df = optim.analyze(
        hipblaslt_path,
        lib_dir,
        results_dir,
        benchmark_dir=benchmarks_dir,
        custom_lib_dir=custom_lib_dir,
        devices=devices,
        up_thr=up_thr,
        verify=False,
        log_summary=workdir / "summary.csv",
        bench_freq=bench_freq,
    )

    if res is not None:
        logger.info("Creating remapped libraries...")
        library.from_dataframe(res, lib_dir, type_override="Equality").dump(final_output_dir)
        logger.info(f"Final remapped library available in: '{final_output_dir}'")

    logger.info("Creating full optimized libraries...")
    library.from_full_dataframe(
        full_df, lib_dir, 
        match_table_path,
        type_override="Equality"
    ).dump(full_output_dir)
    logger.info(f"Full optimized library available in: '{full_output_dir}'")

    logger.info("Search workflow completed successfully!")
    state.optimized = True
    state.dump(state_path)


def run_configure(
    hipblaslt_path: str,
    log_file: str,
    devices: Sequence[int] | None = None,
    keep_thr: float = 0,
    arch: str = "gfx950",
    backend: str = "ductile",
    workdir: str = "workdir",
    verbose: int = 1,
    bench_freq: bool = False,
    device: int | None = None,
) -> None:
    """Summarize the workload log, then write tuning YAML under workdir/optimizations.

    Loads or creates run_state.json, runs bench.log.summarize, builds GemmConfig
    rows from the unique-GEMM DataFrame, and calls optim.configure. Sets
    state.configured and rewrites run_state.json (unless no GEMMs remain).

    Args:
        hipblaslt_path: hipBLASLt checkout root (must exist).
        log_file: Workload YAML/CSV path (must exist).
        devices: GPU indices for summarize (must be 0-7 on this validation path).
        keep_thr: Minimum grouped % of total time to keep a GEMM row.
        arch: Target gfx string in constants.SUPPORTED_ARCH.
        backend: "ductile" or "tensile".
        workdir: Created if missing; holds run_state.json and outputs above.
        verbose: Logger level via _set_log_level.
        bench_freq: Forwarded to bench.log.summarize when keep_thr > 0
            (controls HIPBLASLT_BENCH_FREQ); ignored when keep_thr == 0
            because that branch skips benchmarking.

        device: Backward-compatible single-device alias. If set, overrides devices.

    Raises:
        FileNotFoundError: hipBLASLt or log_file missing.
        ValueError: invalid device id, unknown arch, or invalid backend string.
    """
    if device is not None:
        devices = [device]
    if devices is None:
        devices = [0]  # Default to device 0 if not specified
    devices = parse_devices(devices)

    for d in devices:
        if d < 0 or d >= 8:
            raise ValueError(
                f"Device ID must be 0-7, got {d}. Use -d or --device to specify the device ID."
            )

    if arch not in SUPPORTED_ARCH:
        raise ValueError(f"Must be one of {SUPPORTED_ARCH}")

    if backend.lower() not in ("ductile", "tensile"):
        raise ValueError("'backend' must be one of ('ductile', 'tensile')")

    hipblaslt_path, log_file, workdir, state, state_path = _prepare_workflow_context(
        hipblaslt_path, log_file, workdir, verbose
    )

    logger.info("Starting configuration phase...")
    logger.info(f"hipBLASLt path: '{hipblaslt_path}'")
    logger.info(f"Log file: '{log_file}'")
    logger.info(f"Backend: '{backend}'")
    logger.info(f"Working directory: '{workdir}'")
    logger.info(f"Keep threshold: {keep_thr}")

    _, uniq_df = bench.log.summarize(
        hipblaslt_path,
        log_file,
        output_dir=workdir,
        devices=devices,
        keep_thr=keep_thr,
        cache=True,
        bench_freq=bench_freq,
    )

    if uniq_df.empty:
        logger.warning("No GEMM operations found after filtering. Consider lowering keep_thr")
        return

    logger.info("Generating GEMM Kernel Optimization configs...")

    tuning_dir = workdir / "optimizations"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    gemm_configs: List[GemmConfig] = gemm_configs_from_gemm_dataframe(uniq_df)

    optim.configure(
        hipblaslt_path,
        gemm_configs,
        tuning_dir,
        arch=arch,
        backend=backend,
    )
    n_configs = len(gemm_configs)

    logger.info(f"Generated {n_configs} configuration files in '{tuning_dir}'")
    logger.info("Configuration phase completed successfully!")

    state.configured = True
    state.dump(state_path)


def run_optimize(
    hipblaslt_path: str,
    workdir: str = "workdir",
    devices: Sequence[int] | None = None,
    n_slots: int = 4,
    up_thr: float = 1.03,
    err_thr: float = 0.03,
    client_build_dir: str = "build_tmp",
    retry: bool = True,
    verbose: int = 1,
    bench_freq: bool = False,
) -> None:
    """Run optim.run, merge 3_LibraryLogic YAMLs, analyze, and optionally write final_libs.

    Requires run_state.json with configured=True and workdir/optimizations.
    May delete stale libs/, build/, benchmarks/ when retrying. On success sets
    state.optimized and updates run_state.json.

    Args:
        hipblaslt_path: hipBLASLt checkout root.
        workdir: Directory produced by run_configure.
        devices: GPU ids used by optim.run and analyze.
        n_slots: Per-device concurrency for optim.run.
        up_thr: Minimum speedup ratio vs reference kept in optim.analyze.
        err_thr: Maximum relative error for tuned kernels in optim.analyze.
        client_build_dir: Relative tensilelite client build dir for optim.run.
        retry: Passed to optim.run; also triggers cleanup of partial artifacts.
        verbose: Logger level via _set_log_level.
        bench_freq: Forwarded to optim.analyze (controls HIPBLASLT_BENCH_FREQ
            during the post-optim benchmark sweep).

    Raises:
        FileNotFoundError: hipBLASLt, workdir, optimizations/, or state file missing.
        ValueError: configure not completed, no GEMM configs, or none completed after run.
    """
    hipblaslt_path = Path(hipblaslt_path)
    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    workdir = Path(workdir)
    if not workdir.is_dir():
        raise FileNotFoundError(f"Working directory not found: {workdir}")

    state_path = workdir / "run_state.json"
    state = RunState.load(state_path)
    if not state.configured:
        raise ValueError("Configuration step was not completed")

    input_dir = workdir / "optimizations"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Optimizations directory not found: {input_dir}")

    _set_log_level(verbose)
    if devices is None:
        devices = list(range(8))  # Default to all 8 devices if not specified
    devices = parse_devices(devices)
    final_output_dir = workdir / "final_libs"
    full_output_dir = workdir / "full_libs"

    logger.info("Starting optimization phase...")
    logger.info(f"hipBLASLt path: '{hipblaslt_path}'")
    logger.info(f"Working directory: '{workdir}'")
    logger.info(f"Final output directory: '{final_output_dir}'")
    logger.info(f"Full output directory: '{full_output_dir}'")
    logger.info(f"Devices: {devices}")
    logger.info(f"Jobs per device: {n_slots}")
    logger.info(f"Performance threshold: {up_thr}")
    logger.info(f"Error threshold: {err_thr}")

    n_gemms, n_completed, n_failed = optim.utils.check_progress(input_dir)
    logger.debug(
        f"Initial optimization progress: total={n_gemms} completed={n_completed} failed={n_failed}"
    )
    if n_gemms == 0:
        raise ValueError(f"No GEMMs found in '{input_dir}'")

    benchmarks_dir = workdir / "benchmarks"
    custom_lib_dir = workdir / "build"
    lib_dir = workdir / "libs"
    cleanup_needed = (retry and n_gemms > n_completed) or (n_gemms > (n_completed + n_failed))
    logger.debug(
        f"Cleanup decision: retry={retry} total={n_gemms} completed={n_completed} "
        f"failed={n_failed} cleanup_needed={cleanup_needed}"
    )
    if cleanup_needed:
        if lib_dir.is_dir():
            logger.debug(f"Removing stale library directory: {lib_dir}")
            shutil.rmtree(lib_dir)
        if custom_lib_dir.is_dir():
            logger.debug(f"Removing stale custom library directory: {custom_lib_dir}")
            shutil.rmtree(custom_lib_dir)
        if benchmarks_dir.is_dir():
            logger.debug(f"Removing stale benchmark directory: {benchmarks_dir}")
            shutil.rmtree(benchmarks_dir)

    state.optimized = False
    state.dump(state_path)
    logger.debug(
        f"Calling optim.run with input_dir={input_dir} devices={devices} "
        f"client_build_dir={client_build_dir} n_slots={n_slots} retry={retry}"
    )
    optim.run(
        hipblaslt_path,
        input_dir,
        devices=devices,
        client_build_dir=client_build_dir,
        n_slots=n_slots,
        retry=retry,
    )

    _, n_completed, _ = optim.utils.check_progress(input_dir)
    logger.debug(f"Post-run progress: total={n_gemms} completed={n_completed}")

    if n_completed == 0:
        raise ValueError(f"No GEMMs completed out of {n_gemms} in {input_dir}")

    merged_libs = library.merge_solutions(input_dir, epilogues=True)

    if lib_dir.is_dir():
        shutil.rmtree(lib_dir)

    merged_libs.dump(lib_dir)
    logger.info(f"Merged library available in: '{lib_dir}'")

    results_dir = workdir / "results"
    if results_dir.is_dir():
        shutil.rmtree(results_dir)

    if final_output_dir.is_dir():
        shutil.rmtree(final_output_dir)
    if full_output_dir.is_dir():
        shutil.rmtree(full_output_dir)

    log_summary = workdir / "summary.csv"

    logger.info("Analyzing results...")
    res, full_df = optim.analyze(
        hipblaslt_path,
        lib_dir,
        results_dir,
        benchmark_dir=benchmarks_dir,
        custom_lib_dir=custom_lib_dir,
        devices=devices,
        up_thr=up_thr,
        error_thr=err_thr,
        log_summary=log_summary if log_summary.is_file() else None,
        bench_freq=bench_freq,
    )
    logger.debug(
        "Analyze outcome: no improving kernels found"
        if res is None
        else f"Analyze outcome: retained_rows={len(res)}"
    )

    if res is not None:
        logger.info("Creating optimized libraries...")
        library.from_dataframe(res, lib_dir, type_override="Equality").dump(final_output_dir)
        logger.info(f"Final optimized library available in: '{final_output_dir}'")

    logger.info("Creating full optimized libraries...")
    match_table_path = hipblaslt_path / "build/release/device-library/MatchTable.yaml"
    library.from_full_dataframe(
        full_df, 
        lib_dir,
        match_table_path,
        type_override="Equality"
    ).dump(full_output_dir)
    logger.info(f"Full optimized library available in: '{full_output_dir}'")

    logger.info("Optimization workflow completed successfully!")

    state.optimized = True
    state.dump(state_path)
