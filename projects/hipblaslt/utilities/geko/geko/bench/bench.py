################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from __future__ import annotations

"""Core benchmarking functionality for kernel performance evaluation.

Executes hipblaslt-bench tool with custom configurations and parses results
into structured DataFrames. Supports comparison between reference and tuned
solution libraries with accuracy verification.

Key functions:
    run: Execute single benchmark configuration.
    compare: Compare reference vs tuned kernel performance across libraries.
"""

import os
import copy
import math
import yaml
import glob
import subprocess
import logging
import pandas as pd
import numpy as np

from threading import Lock
from tqdm import tqdm
from pathlib import Path
from typing import Sequence

from geko.bench.utils import parse_benchmark_output, update_lib_source
from geko import library
from geko.constants import GEMM_FIELDS
from geko.bench.log import verify_output, dump as dump_bench_yaml, read as read_bench_yaml
from geko.concurrency.runner import Runner, Worker
from geko.utils import parse_devices

logger = logging.getLogger("GEKO")

__all__ = ["run", "standard_benchmark", "compare"]


UNIQ_COLS = (
    "hipblaslt-Gflops", 
    "hipblaslt-GB/s", 
    "us", 
    "error", 
    "kernel", 
    "solution", 
    "lib", 
    "lib_source", 
    "solutionIdx",
    "lowest_avg_freq", 
    "lowest_median_freq", 
    "avg_MCLK", 
    "median_MCLK"
)


def run(
    hipblaslt_path: str | Path,
    bench_file: str | Path,
    output_file: str | Path,
    custom_lib_dir: str | Path = None,
    devices: Sequence[int] = [0],
    cache: bool = False,
    bench_freq: bool = False,
    min_chunk_size: int = 1,
    silent: bool = False,
) -> pd.DataFrame:
    """Execute hipBLASLt benchmark and parse results.

    Runs hipblaslt-bench with specified configuration and parses
    the output into a structured DataFrame.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        bench_file (str | Path): YAML file with benchmark configuration.
        output_file (str | Path): Output file for benchmark results.
        custom_lib_dir (str | Path, optional): Custom library directory.
            Defaults to None.
        devices (Sequence[int], optional): GPU device IDs used by the 
            concurrency runner. Defaults to [0].
        cache (bool, optional): Whether to use cached results if available.
            Defaults to False.
        bench_freq (bool, optional): If True, set HIPBLASLT_BENCH_FREQ=true so
            hipblaslt-bench collects clock frequency telemetry. Default is
            False to avoid the small collection/output overhead; enable when
            you actually want the frequency data. Defaults to False.
        silent (bool, optional): If True, disable progress logging. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with benchmark results including performance metrics.

    Raises:
        ValueError: If hipblaslt-bench execution fails.
        FileNotFoundError: If hipBLASLt path or benchmark file does not exist.

    Note:
        Sets HIPBLASLT_TENSILE_LIBPATH environment variable when using custom library.
    """
    devices = parse_devices(devices)

    hipblaslt_path = Path(hipblaslt_path)
    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    bench_file = Path(bench_file)
    if not bench_file.is_file():
        raise FileNotFoundError(f"Benchmark file does not exist '{bench_file}")

    output_file = Path(output_file)
    if custom_lib_dir:
        custom_lib_dir = Path(custom_lib_dir)

    def hipblaslt_bench(_bench_file: Path, _output_file: Path, _device: int):
        cmd = hipblaslt_path / "build/release/clients/hipblaslt-bench"

        env = {**os.environ}
        if bench_freq:
            env["HIPBLASLT_BENCH_FREQ"] = "true"
        if custom_lib_dir:
            lib_path = custom_lib_dir / "library"
            arch_dirs = list(lib_path.glob("gfx*"))
            if arch_dirs:
                lib_path = arch_dirs[0]
            env["HIPBLASLT_TENSILE_LIBPATH"] = str(lib_path)

        with open(_output_file, "w") as f:
            try:
                subprocess.run(
                    [cmd, "--yaml", _bench_file, "--device", str(_device)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                raise ValueError(f"hipblaslt-bench failed to run, check '{_output_file}' for more information")
    
    prev_level = logger.level
    mute_for_progress = (logger.getEffectiveLevel() > logging.DEBUG) and silent
    if mute_for_progress:
        logger.setLevel(logging.WARNING)

    using_cache = cache and verify_output(output_file, bench_file)
    if using_cache:  # Using an old benchmark
        logger.info(f"Using cached output in '{output_file}'")
        return parse_benchmark_output(output_file)
    
    if len(devices) == 1:
        logger.info(
            f"Running hipblaslt-bench for '{bench_file}' in device {devices[0]}, output will be saved in '{output_file}'"
        )
        hipblaslt_bench(bench_file, output_file, devices[0])
        if mute_for_progress:
            logger.setLevel(prev_level)
        return parse_benchmark_output(output_file)

    workdir = output_file.parent / f"{output_file.stem}_chunks"
    workdir.mkdir(parents=True, exist_ok=True)

    output_file_data = []
    _queue_lock = Lock()

    class BenchWorker(Worker[dict]):
        """Worker that runs hipblaslt-bench for a given benchmark file."""

        def setup(self) -> None:
            self.bench_file = self.item
            
            if not self.bench_file.is_file():
                raise FileNotFoundError(f"Benchmark file does not exist '{self.bench_file}'")
           
            self.output_file = self.bench_file.with_suffix(".out")
            
        def run(self) -> bool:
            using_cache = cache and verify_output(self.output_file, self.bench_file)
            if using_cache:  # Using an old benchmark
                logger.info(f"Using cached output in '{self.output_file}'")
                return True
            logger.info(
                f"Running hipblaslt-bench for '{self.bench_file}' in device {self.device}, output will be saved in '{self.output_file}'"
            )

            hipblaslt_bench(self.bench_file, self.output_file, self.device)
            return verify_output(self.output_file, self.bench_file)

        def teardown(self) -> None:
            with _queue_lock:
                output_file_data.append(self.output_file.read_text())
    
    data = read_bench_yaml(bench_file)

    chunks = np.array_split(data, max(1, min(len(data) // min_chunk_size, len(devices))))
    items = []
    for i in range(len(chunks)):
        chunk_bench_file = workdir / f"chunk_{i}.yaml"
        dump_bench_yaml(chunks[i].tolist(), chunk_bench_file)
        items.append(chunk_bench_file)

    runner = Runner(
        items=items,
        worker_impl=BenchWorker,
        devices=devices,
        n_slots=1,
        estimate_workload_fn=None,
        job_logger_fn=None,
    )

    logger.info(f"Running benchmark with {len(chunks)} chunks across devices {devices[:len(chunks)]}")

    try:
        runner(workdir, silent=silent)
    finally:
        if mute_for_progress:
            logger.setLevel(prev_level)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(output_file_data))

    return parse_benchmark_output(output_file)


def standard_benchmark(
    hipblaslt_path: str | Path,
    bench_file: str | Path,
    output_file: str | Path,
    custom_lib_dir: str | Path = None,
    devices: Sequence[int] = [0],
    duration: float = 0.5,
    initial_iters: int = 10,
    initial_cold_iters: int = 2,
    bench_freq: bool = False,
) -> pd.DataFrame:
    """Run a probe benchmark then scale cold and timed iterations from latency.

    Executes hipblaslt-bench with initial_iters and initial_cold_iters, parses
    per-GEMM latency (us), then sets cold_iters and iters on each row so the cold
    and timed phases each approximate duration seconds, using the same scaling
    rule as bench.log.update.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        bench_file (str | Path): YAML file with benchmark configuration.
        output_file (str | Path): Output file for the final benchmark results.
        custom_lib_dir (str | Path, optional): Custom library directory.
            Defaults to None.
        devices (Sequence[int], optional): GPU device IDs used by the 
            concurrency runner. Defaults to [0].
        duration (float, optional): Target seconds for the cold phase and for the timed phase.
            Defaults to 0.5.
        initial_iters (int, optional): Timed iterations for the probe run.
            Defaults to 10.
        initial_cold_iters (int, optional): Cold iterations for the probe run.
            Defaults to 2.
        bench_freq (bool, optional): Forwarded to run for both probe and
            scaled passes. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame from the scaled benchmark (same columns as run).

    Raises:
        ValueError: If the YAML is not a non-empty list, hipblaslt-bench fails, the device
            ID is invalid, or probe output row count does not match the benchmark file.
        FileNotFoundError: If hipBLASLt path or benchmark file does not exist.

    Note:
        Both benchmark passes call run with cache=False. Writes probe and scaled YAML
        files in the same directory as output_file, using output_file's stem in the filenames.
    """
    bench_file = Path(bench_file)
    output_file = Path(output_file)
    if not bench_file.is_file():
        raise FileNotFoundError(f"Benchmark file does not exist '{bench_file}'")

    with open(bench_file) as f:
        rows = yaml.safe_load(f)
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"Benchmark file must be a non-empty YAML list: '{bench_file}'")

    probe_rows = copy.deepcopy(rows)
    for row in probe_rows:
        row["iters"] = int(initial_iters)
        row["cold_iters"] = int(initial_cold_iters)

    probe_yaml = output_file.parent / f"{output_file.stem}_standard_probe.yaml"
    probe_out = output_file.parent / f"{output_file.stem}_standard_probe.out"
    dump_bench_yaml(probe_rows, probe_yaml)

    logger.info(
        f"standard_benchmark probe: iters={initial_iters} cold_iters={initial_cold_iters} "
        f"yaml='{probe_yaml}'"
    )
    df_probe = run(
        hipblaslt_path,
        probe_yaml,
        probe_out,
        custom_lib_dir=custom_lib_dir,
        devices=devices,
        cache=False,
        bench_freq=bench_freq,
    )

    if len(df_probe) != len(rows):
        raise ValueError(
            f"Probe output rows ({len(df_probe)}) do not match bench rows ({len(rows)}); "
            f"check '{probe_out}'"
        )

    scaled_rows = copy.deepcopy(rows)
    for i, row in enumerate(scaled_rows):
        us = float(df_probe["us"].iloc[i])
        if math.isfinite(us) and us > 0:
            scaled = max(math.ceil(duration * 1e6 / us), 1)
            row["cold_iters"] = scaled
            row["iters"] = scaled
        else:
            row["cold_iters"] = int(initial_cold_iters)
            row["iters"] = int(initial_iters)
            logger.warning(
                f"standard_benchmark: non-positive or invalid us at row {i} ({us!r}); "
                f"keeping probe iteration counts"
            )

    scaled_yaml = output_file.parent / f"{output_file.stem}_standard_scaled.yaml"
    dump_bench_yaml(scaled_rows, scaled_yaml)
    logger.info(
        f"standard_benchmark final run: duration={duration}s per phase, yaml='{scaled_yaml}'"
    )
    return run(
        hipblaslt_path,
        scaled_yaml,
        output_file,
        custom_lib_dir=custom_lib_dir,
        devices=devices,
        cache=False,
        bench_freq=bench_freq,
    )


def compare(
    hipblaslt_path: str | Path,
    lib_dir: str | Path,
    custom_lib_dir: str | Path = "build",
    benchmark_dir: str | Path = "benchmarks",
    verify: bool = True,
    cache: bool = False,
    duration: float = 0.5,
    beta: bool = False,
    devices: Sequence[int] = [0],
    bench_freq: bool = False,
) -> pd.DataFrame:
    """Compare performance between reference and tuned solution libraries.

    Benchmarks all library files against both reference (default) and
    tuned implementations, optionally including accuracy verification.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        lib_dir (str | Path): Directory containing library YAML files to benchmark.
        custom_lib_dir (str | Path, optional): Directory for custom library creation.
            Defaults to "build".
        benchmark_dir (str | Path, optional): Output directory for benchmark files.
            Defaults to "benchmarks".
        verify (bool, optional): Whether to run accuracy verification tests.
            Defaults to True.
        cache (bool, optional): Whether to reuse existing Tensile library and results.
            Defaults to False.
        duration (float, optional): Target benchmark duration in seconds.  
            This is the duration of each benchmark. It will be used to calculate the number of iterations.
            Defaults to 0.5 seconds.
        beta (bool, optional): Whether to use non-zero beta values in benchmarks.
            Defaults to False.
        devices (Sequence[int], optional): GPU device IDs used by the 
            concurrency runner. Defaults to [0].
        bench_freq (bool, optional): Forwarded to run for every reference,
            tuned, and verify pass. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with comparison results including performance ratios
            and error metrics (if verify=True).

    Raises:
        ValueError: If no valid libraries found in lib_dir.

    Note:
        - Creates Tensile library if not cached.
        - Generates benchmark input files for each library.
        - Compares reference vs tuned performance with ratio calculation.
    """
    lib_dir = Path(lib_dir)
    libs = library.load_collection(lib_dir)
    if len(libs) == 0:
        raise ValueError(f"No valid libraries found in '{lib_dir}'")

    custom_lib_dir = Path(custom_lib_dir)
    benchmark_dir = Path(benchmark_dir)

    # Build custom library with TensileCreateLibrary if not found
    if not cache or len(list(custom_lib_dir.glob("library/**/TensileLibrary_lazy_gfx*.dat"))) == 0:
        logger.debug(f"Creating custom library cache={cache} custom_lib_dir={custom_lib_dir}")
        library.operations.create(hipblaslt_path, lib_dir, custom_lib_dir)

    benchmark_dir.mkdir(parents=True, exist_ok=True)

    prev_level = logger.level
    mute_for_progress = logger.getEffectiveLevel() > logging.DEBUG
    if mute_for_progress:
        logger.setLevel(logging.WARNING)
    ref, tuned, error = [], [], []
    for lib in tqdm(libs, desc=f"Benchmarking libraries in '{lib_dir}'"):
        logger.debug(f"Benchmarking library {lib.name}")
        bench_file, verif_file = lib.create_bench_input(
            benchmark_dir,
            duration=duration,
            verify=verify,
            beta=beta,
        )

        # Benchmark default library
        bench_file = Path(bench_file)
        log_file = benchmark_dir / (bench_file.stem + "-reference.out")
        res = run(
            hipblaslt_path, bench_file, log_file,
            custom_lib_dir=None, devices=devices, cache=cache, bench_freq=bench_freq,
            silent=True
        )

        # add lib_source column
        matchtable_path = Path(hipblaslt_path / "build/release/device-library/MatchTable.yaml")
        res = update_lib_source(res, matchtable_path)

        res.rename({c: c + "_reference" for c in UNIQ_COLS}, axis=1, inplace=True)
        res["lib"] = bench_file.stem.split("_bench")[0] + ".yaml"
        ref.append(res)

        # Benchmark optimized library
        log_file = benchmark_dir / (bench_file.stem + "-tuned.out")
        res = run(
            hipblaslt_path, bench_file, log_file,
            custom_lib_dir=custom_lib_dir, devices=devices, cache=cache, bench_freq=bench_freq,
            silent=True
        )
        res.rename({c: c + "_tuned" for c in UNIQ_COLS}, axis=1, inplace=True)
        tuned.append(res)

        # Verify optimized library
        if verif_file is not None:
            verif_file = Path(verif_file)
            log_file = benchmark_dir / (verif_file.stem + "-tuned.out")
            res = run(
                hipblaslt_path, verif_file, log_file,
                custom_lib_dir=custom_lib_dir, devices=devices, cache=cache, bench_freq=bench_freq,
                silent=True
            )
            res.rename({c: c + "_tuned" for c in UNIQ_COLS}, axis=1, inplace=True)
            error.append(res)

        logger.debug(
            f"Library benchmark summary: lib={lib.name} ref_rows={len(ref[-1])} "
            f"tuned_rows={len(tuned[-1])} verify_rows={(len(error[-1]) if len(error) > 0 else 0)}"
        )

    if mute_for_progress:
        logger.setLevel(prev_level)

    dfr = pd.concat(ref, ignore_index=True).reset_index(drop=True)
    dft = pd.concat(tuned, ignore_index=True).reset_index(drop=True)
    if len(error) > 0: # if verify is true, we will have error.
        dfv = pd.concat(error, ignore_index=True).reset_index(drop=True)
        dfv = dfv[list(GEMM_FIELDS) + ["norm_error", "atol", "rtol"]]
        failed_mask = (dfv["atol"] == "failed") | (dfv["rtol"] == "failed")
        if failed_mask.any():
            logger.warning(
                f"{failed_mask.sum()} cases failed verification (atol or rtol); "
                f"setting error_pr to 1e6 for these cases"
            )
            dfv.loc[failed_mask, "norm_error"] = 1e6
        dft["error_tuned"] = dft.merge(dfv, on=[c for c in dfv.columns if c not in ["norm_error", "atol", "rtol"]])["norm_error"]
        dft["error_tuned"] = dft["error_tuned"] / dft["batch_count"] # normalized by batch_count

    df = dfr.merge(dft, on=[c for c in dfr.columns if c.split("_reference")[0] not in UNIQ_COLS])
    df["ratio"] = df["us_reference"] / df["us_tuned"]
    logger.debug(
        f"Benchmark compare result: rows={len(df)} ratio_mean={df['ratio'].mean():.6f} "
        f"ratio_min={df['ratio'].min():.6f} ratio_max={df['ratio'].max():.6f}"
    )

    return df
