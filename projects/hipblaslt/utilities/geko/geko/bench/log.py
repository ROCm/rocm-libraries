# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""hipBLASLt log file parsing and processing utilities.

Handles reading, parsing, and manipulation of hipBLASLt YAML log files.
Provides functions for log summarization, benchmark configuration generation,
and GEMM operation analysis.

Key functions:
    parse: Parse yaml/csv (log file) files into DataFrame or dictionary format.
    summarize: Analyze logs and filter by contribution thresholds.
    benchmark: Generate and run benchmarks from log configurations.
    update: Modify benchmark configurations with timing parameters.
"""

import yaml
import pandas as pd
import numpy as np
import copy
import math
import logging

from pathlib import Path
from typing import Sequence, List, Tuple, Optional

from geko.constants import LOG_FIELDS, GEMM_LOG_FIELDS, GEMM_FIELDS
from geko import bench
from geko.bench.utils import update_lib_source
from geko.utils import parse_devices


logger = logging.getLogger("GEKO")


def update_compute_type(compute_type: str) -> str:
    """Ensure compute type has proper 'c_' prefix format.

    Args:
        compute_type (str): Compute type string to normalize.

    Returns:
        str: Compute type with 'c_' prefix prepended when the input encodes a
            scalar FP size (contains "32" or "64") and is not already prefixed.
            Otherwise the input string is returned unchanged.
    """
    if compute_type.startswith("c_"):
        return compute_type
    if "32" in compute_type or "64" in compute_type:
        return "c_" + compute_type
    return compute_type


def read(log_file: str | Path) -> List[dict]:
    """Read and parse a YAML log file with error handling.

    Args:
        log_file (str | Path): Path to YAML log file.

    Returns:
        List[dict]: List containing parsed YAML data.

    Note:
        Handles malformed YAML by cleaning common formatting issues
        like trailing commas and extra separators.
    """
    try:
        with open(log_file) as f:
            data = yaml.safe_load(f)
    except yaml.parser.ParserError:
        with open(log_file) as f:
            content = f.read()
        content = content.replace(": ,", "")
        content = content.replace(",  ,", ",")
        content = content.replace(", ,", ",")
        data = yaml.safe_load(content)
    return data


def parse(log_file: str | Path, as_df: bool = False, fmt: str = None) -> pd.DataFrame | List[dict]:
    """Parse hipBLASLt log file into DataFrame or dictionary format.

    Args:
        log_file (str | Path): Path to log file (YAML or CSV format).
        as_df (bool, optional): Whether to return DataFrame (True) or dict (False).
            Defaults to False.
        fmt (str, optional): File format override ('yaml' or 'csv'),
            auto-detected if None. Defaults to None.

    Returns:
        pd.DataFrame | List[dict]: DataFrame if as_df=True, otherwise dictionary
            of log records.

    Raises:
        ValueError: If unsupported file format specified.

    Note:
        - Automatically normalizes compute types and column names.
        - Handles both YAML and CSV input formats.
        - Filters columns to include only valid LOG_FIELDS.
    """
    log_file = Path(log_file)

    if fmt is None:
        fmt = "csv" if log_file.suffix.lower() == ".csv" else "yaml"

    if fmt.lower() not in ("csv", "yaml"):
        raise ValueError(f"File format {fmt} is not supported. Must be one of ('csv', 'yaml')")

    if fmt.lower() == "yaml":
        df = pd.DataFrame(read(log_file))
    else:
        df = pd.read_csv(log_file)

    df.rename({"m": "M", "n": "N", "k": "K"}, axis=1, inplace=True)

    if not all(fld in df.columns for fld in GEMM_LOG_FIELDS):
        raise ValueError(f"Log must have all fields: {GEMM_LOG_FIELDS}")

    if df[list(GEMM_LOG_FIELDS)].isnull().values.any():
        df_tmp = df[list(GEMM_LOG_FIELDS)] 
        raise ValueError(f"Log has missing values in fields: {tuple(df_tmp.columns[df_tmp.isnull().any()])}")

    df["compute_type"] = df["compute_type"].apply(update_compute_type)

    df = df[[c for c in df.columns if c in LOG_FIELDS]]
    if df.isnull().values.any():
        logger.warning(f"Dropping fields with missing values {tuple(df.columns[df.isnull().any()])}")
        df = df.dropna(axis=1)

    if "call_count" not in df.columns:
        df["call_count"] = 1
    if "function" not in df.columns:
        df["function"] = "matmul"
    if "scale_type" not in df.columns:
        df["scale_type"] = df["compute_type"].apply(lambda s: s.lstrip("c_").lstrip("x"))

    n_rows = len(df)
    df = df.groupby([c for c in df.columns if c != "call_count"], sort=False)["call_count"].sum().reset_index()
    if len(df) < n_rows:
        logger.warning(f"Duplicate rows found, their call counts have been aggregated")

    df = df.reindex(sorted(df.columns, key=lambda c: LOG_FIELDS.index(c)), axis=1)

    if as_df:
        return df

    return df.to_dict(orient="records")


def dump(data: dict, log_file: str | Path) -> None:
    """Save benchmark data to a YAML file.

    Args:
        data (dict): Dictionary containing benchmark data.
        log_file (str | Path): Output YAML file path.
    """
    with open(log_file, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False, width=5000)


def verify_output(latency_file: str | Path, bench_file: str | Path) -> bool:
    """
    Verifies benchmark output correctness and that benchmark input and output match.

    Args:
        latency_file (str | Path): Path to the hipBLASLt benchmark output file.
        bench_file (str | Path): Path to the hipBLASLt benchmark input file.

    Returns:
        bool: Whether the output format is correct, and benchmark input and output match.
    """

    try:
        dfr = bench.utils.parse_benchmark_output(latency_file)
        dfr["compute_type"] = dfr["compute_type"].apply(update_compute_type)

        dfl = parse(bench_file, as_df=True).rename({"M": "m", "N": "n", "K": "k"}, axis=1)

        if len(dfl) != len(dfr):
            return False

        dfr = dfr[list(GEMM_FIELDS)].drop_duplicates()
        dfl = dfl[list(GEMM_FIELDS)].drop_duplicates()

        if len(dfl.merge(dfr, on=GEMM_FIELDS, how="inner")) == len(dfl):
            return True
    except:
        pass

    return False


def update(
    data: List[dict] | str | Path,
    latency: Sequence[float] | np.ndarray = None,
    duration: float = 0.5,
    iters: int = 100,
    cold_iters: int = 20,
    rotating: int = 512,
    beta: bool = None,
    flush: bool = None,
    aux: bool = False,
    print_kernel_info: bool = True,
) -> Tuple[List[dict], Optional[str]]:
    """Update benchmark configuration.

    Args:
        data (List[dict] | str | Path]): Benchmark data list or file path to update.
        latency (Sequence[float], optional): Expected latencies for duration calculation.
            Defaults to None.
        duration (float, optional): Target benchmark duration in seconds.
            Defaults to 0.5.
        iters (int, optional): Number of benchmark iterations.
            Defaults to 100.
        cold_iters (int, optional): Number of warm-up iterations.
            Defaults to 20.
        rotating (int, optional): Memory rotation parameter.
            Defaults to 512.
        beta (bool, optional): If set, whether to non-zero beta values.
            Defaults to None.
        flush (bool, optional): If set, whether to flush GPU caches.
            Defaults to None.
        aux (bool, optional): Whether to include auxiliary data types.
            Defaults to False.
        print_kernel_info (bool, optional): Whether to print kernel information.
            Defaults to True.

    Returns:
        Tuple[List[dict], Optional[str]] |  Tuple[List[dict]]: A tuple containing:
            - updated_data: The updated benchmark data.
            - output_file_path (optional): File path (if data was provided as list).

    Raises:
        TypeError: If unknown data format or invalid latency sequence.
        ValueError: If latency has different length w.r.t. data.

    Note:
        Dynamically adjusts iteration counts based on expected latencies
        to achieve target benchmark duration.
    """
    output_file = None
    if isinstance(data, (str, Path)):
        log_file = Path(data)
        output_file = str(log_file.with_suffix("")) + "_upd.yaml"
        data = parse(log_file)
    elif isinstance(data, List):
        data = copy.deepcopy(data)
    else:
        raise TypeError(f"Unknown format")

    if latency is not None:
        if not isinstance(latency, (Sequence, np.ndarray)):
            raise TypeError(f"latency must be None or a sequence")
        if len(latency) != len(data):
            raise ValueError(f"input data and latency lengths must match: {len(data)} vs {len(latency)}")

    for i, row in enumerate(data):
        if "function" not in row:
            row["function"] = "matmul"
        if "initialization" not in row:
            row["initialization"] = "trig_float"

        row["compute_type"] = update_compute_type(row["compute_type"])
        if "scale_type" not in row:
            row["scale_type"] = row["compute_type"].lstrip("c_").lstrip("x")

        if "aux_type" in row and not aux:
            del row["aux_type"]
        if "solution_index" in row:
            del row["solution_index"]
        if "algo_method" in row:
            del row["algo_method"]

        if flush is not None:
            row["flush"] = flush
        if beta is not None:
            row["beta"] = float(beta)

        row["rotating"] = rotating
        row["use_gpu_timer"] = True
        row["print_kernel_info"] = print_kernel_info

        if latency is not None and duration > 0 and latency[i] > 0:
            iters = math.ceil(duration * 1e6 / float(latency[i]))
            row["iters"] = iters
            row["cold_iters"] = iters
        else:
            row["iters"] = iters
            row["cold_iters"] = cold_iters

    if output_file is None:
        return (data,)

    logger.info(f"Saved updated benchmark file to '{output_file}'")
    dump(data, output_file)

    return data, output_file


def benchmark(
    hipblaslt_path: str | Path,
    log_file: str | Path,
    benchmark_dir: str | Path,
    latency: Sequence[float] = None,
    duration: float = 0.5,
    iters: int = 100,
    cold_iters: int = 20,
    rotating: int = 512,
    beta: bool = False,
    flush: bool = True,
    aux: bool = False,
    device: int = 0,
    bench_freq: bool = False,
) -> pd.DataFrame:
    """Run benchmark from log file configuration.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        log_file (str | Path): Input log file with GEMM operations.
        benchmark_dir (str | Path): Output directory for benchmark files.
        latency (Sequence[float], optional): Expected latencies for timing adjustment.
            Defaults to None.
        duration (float, optional): Target benchmark duration in seconds.
            Defaults to 0.5.
        iters (int, optional): Number of benchmark iterations.
            Defaults to 100.
        cold_iters (int, optional): Number of warm-up iterations.
            Defaults to 20.
        rotating (int, optional): Memory rotation parameter.
            Defaults to 512.
        beta (bool, optional): Whether to use non-zero beta values.
            Defaults to True.
        flush (bool, optional): Whether to flush GPU caches.
            Defaults to True.
        aux (bool, optional): Whether to include auxiliary data types.
            Defaults to False.
        device (int, optional): GPU device ID for benchmarking.
            Defaults to 0.
        bench_freq (bool, optional): Forwarded to bench.run (controls
            HIPBLASLT_BENCH_FREQ). Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with benchmark results.
    """
    log_file = Path(log_file)
    data = update(
        parse(log_file),
        latency=latency,
        duration=duration,
        iters=iters,
        cold_iters=cold_iters,
        rotating=rotating,
        beta=beta,
        flush=flush,
        aux=aux,
    )[0]

    benchmark_dir = Path(benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    bench_file = benchmark_dir / (log_file.stem + ".yaml")
    dump(data, bench_file)

    output_file = bench_file.with_suffix(".out")
    df = bench.run(
        hipblaslt_path, bench_file, output_file,
        devices=[device], cache=False, bench_freq=bench_freq,
    )
    return df


def summarize(
    hipblaslt_path: str | Path,
    log_file: str | Path,
    output_dir: str | Path = ".",
    devices: Sequence[int] | None = None,
    keep_thr: float = 0.0,
    cache: bool = False,
    use_standard_benchmark: bool = False,
    benchmark_duration: float = 0.5,
    bench_freq: bool = False,
    device: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze and summarize GEMM operations from hipBLASLt log file.

    Processes log files to extract unique GEMM types and optionally
    filters by performance contribution threshold.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        log_file (str | Path): Input hipBLASLt log file (YAML format).
        output_dir (str | Path, optional): Output directory for summary files.
            Defaults to ".".
        devices (Sequence[int], optional): GPU device IDs used by the load
            balancer for benchmarking (if keep_thr > 0). Defaults to range(1).
        keep_thr (float, optional): Minimum percentage contribution threshold
            for filtering. Defaults to 0.0.
        cache (bool, optional): Whether to use cached benchmark results.
            Forwarded to bench.run (ignored when use_standard_benchmark).
            Defaults to False.
        use_standard_benchmark (bool, optional): When True, use
            bench.standard_benchmark (probe + per-row iteration scaling)
            instead of bench.run. Defaults to False.
        benchmark_duration (float, optional): Target seconds for each cold and
            timed phase when use_standard_benchmark=True. Defaults to 0.5.
        bench_freq (bool, optional): Forwarded to bench.run or
            bench.standard_benchmark (controls HIPBLASLT_BENCH_FREQ).
            Ignored when keep_thr == 0 because that branch skips
            benchmarking entirely. Defaults to False.
        device (int, optional): Backward-compatible single-device alias.
            If set, overrides devices.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames with summarized and unique GEMM operations, filtered by threshold
            if specified.

    Note:
        - When keep_thr > 0: runs benchmarks and filters by contribution percentage.
        - Creates CSV files with performance analysis and unique GEMM list.
        - Groups identical GEMM types and aggregates call counts.
    """
    log_file = Path(log_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is not None:
        devices = [device]
    if devices is None:
        devices = [0]
    devices = parse_devices(devices)

    data = update(parse(log_file, as_df=False))[0]
    logger.info(f"Working on '{log_file}' with {len(data)} GEMMs...")

    if keep_thr == 0.0: # no need to benchmark if keep_thr=0 (tune all sizes), build summary from log file
        logger.info("keep_thr=0: skipping benchmarks, building summary from log file")
        tmp_df = pd.DataFrame(data).rename(columns={"M": "m", "N": "n", "K": "k"})
        summary_df = tmp_df[[c for c in tmp_df.columns if c in GEMM_FIELDS]].copy()
        summary_df["call_count"] = tmp_df["call_count"] if "call_count" in tmp_df.columns else 1
    else:
        # benchmark and build summary
        log_file = output_dir / (log_file.stem + ".yaml")
        dump(data, log_file)

        output_file = log_file.with_suffix(".out")
        if use_standard_benchmark:
            summary_df = bench.standard_benchmark(
                hipblaslt_path,
                log_file,
                output_file,
                devices=devices,
                duration=benchmark_duration,
                bench_freq=bench_freq,
            )
        else:
            summary_df = bench.run(
                hipblaslt_path, log_file, output_file,
                devices=devices, cache=cache, bench_freq=bench_freq,
            )

        # add lib_source column
        matchtable_path = Path(hipblaslt_path / "build/release/device-library/MatchTable.yaml")
        summary_df = update_lib_source(summary_df, matchtable_path)

        summary_df["call_count"] = pd.DataFrame(data)["call_count"] if "call_count" in data[0] else 1
        summary_df["total (us)"] = summary_df["call_count"] * summary_df["us"]
        summary_df["% of total"] = 100 * summary_df["total (us)"] / summary_df["total (us)"].sum()
        summary_df = summary_df.sort_values("total (us)", ascending=False, ignore_index=True)

        logger.info(f"Min GEMM contribution percentage: {summary_df['% of total'].min()}")
        logger.info(f"Max GEMM contribution percentage: {summary_df['% of total'].max()}")

        keep = summary_df.groupby(list(GEMM_FIELDS))["% of total"].transform(lambda g: g.sum() >= keep_thr)
        logger.info(f"Removed {len(summary_df) - keep.sum()} out of {len(summary_df)} GEMMs after filtering")
        summary_df = summary_df[keep]

    csv_file = output_dir / "summary.csv"
    summary_df.to_csv(csv_file, index=False)
    logger.info(f"Saved CSV summary in '{csv_file}'")

    uniq_df = summary_df[list(GEMM_FIELDS)].drop_duplicates().reset_index(drop=True)
    logger.info(f"Found {len(uniq_df)} unique GEMMs after filtering")

    csv_file = output_dir / "gemms.csv"
    uniq_df.to_csv(csv_file, index=False)
    logger.info(f"Saved unique GEMMs in '{csv_file}'")

    return summary_df, uniq_df
