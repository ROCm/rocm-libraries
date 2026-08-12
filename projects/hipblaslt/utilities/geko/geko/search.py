# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""GEMM Kernel Search module.

Implements the GEMM kernel search workflow, a.k.a. 'offline tuning'.
Generates benchmarking configurations for specific GEMM types and executes multi-GPU
dense search with automatic load balancing and progress monitoring.

Key functions:
    configure: Generate benchmarking configurations for each input GEMM.
    run: Execute dense search across multiple GPUs.

Integrates with hipBLASLt for benchmarking.
"""

import pandas as pd
import logging
import numpy as np
import time

from threading import Lock
from pathlib import Path
from typing import Sequence, List
from geko import bench
from geko.constants import GEMM_LOG_FIELDS, GEMM_FIELDS
from geko.utils import parse_devices
from geko.concurrency.runner import Runner, Worker

logger = logging.getLogger("GEKO")


def _normalize_compute_type(v: str) -> str:
    v = str(v)
    return v if v.startswith("c_") else f"c_{v}"


def _load_processed_df(processed_file: Path) -> pd.DataFrame:
    """Load processed.csv, deleting it if schema is invalid.

    Returns an empty DataFrame with GEMM_FIELDS columns when file is missing or
    removed due to schema mismatch.
    """
    if not processed_file.is_file():
        return pd.DataFrame(columns=list(GEMM_FIELDS))

    df = pd.read_csv(processed_file)
    if set(df.columns) != set(GEMM_FIELDS):
        logger.warning(
            f"Invalid processed.csv schema in '{processed_file}'. "
            f"Expected columns {list(GEMM_FIELDS)}; deleting file."
        )
        processed_file.unlink(missing_ok=True)
        return pd.DataFrame(columns=list(GEMM_FIELDS))

    df = df[list(GEMM_FIELDS)].copy()
    df["compute_type"] = df["compute_type"].apply(_normalize_compute_type)
    return df


def configure(
    df: pd.DataFrame,
    duration: float = 0.04,
    iters: int = 100,
    cold_iters: int = 20,
    rotating: int = 512,
    beta: bool = False,
    flush: bool = False,
) -> List[dict]:
    """Generate benchmarking configurations for each input GEMM.

    Creates YAML and shell script configuration files for tuning a specific
    GEMM operation type using the config generator framework.

    Args:
        df (DataFrame): DataFrame containing the GEMMs to benchmark.
        duration (float, optional): Target benchmark duration in seconds.
            Defaults to 0.04.
        iters (int, optional): Number of benchmark iterations.
            Defaults to 100.
        cold_iters (int, optional): Number of warm-up iterations.
            Defaults to 20.
        rotating (int, optional): Memory rotation parameter.
            Defaults to 512.
        beta (bool, optional): Whether to use non-zero beta values.
            Defaults to False.
        flush (bool, optional): Whether to flush GPU caches.
            Defaults to False.

    Returns:
        dict: List of dictionaries containing the benchmark configurations.

    Raises:
        ValueError: If input DataFrame is empty.
    """
    if df.empty:
        raise ValueError("No GEMM operations found in input DataFrame")

    df = df.rename({"m": "M", "n": "N", "k": "K"}, axis=1)
    uniq_df = df[list(GEMM_LOG_FIELDS)].drop_duplicates()

    latency = None
    if "us" in df.columns:
        latency = df.loc[uniq_df.index, "us"].values

    configs = bench.log.update(
        uniq_df.to_dict(orient="records"),
        latency=latency,
        duration=duration,
        iters=iters,
        cold_iters=cold_iters,
        rotating=rotating,
        beta=beta,
        flush=flush,
        aux=False,
        print_kernel_info=True,
    )[0]

    for i, config in enumerate(configs):
        config["algo_method"] = 1
        config["requested_solution_num"] = -1

        if latency is not None and latency[i] > 20:
            config["skip_slow_solution_ratio"] = 0.6
        elif latency is not None and latency[i] <= 20:
            config["skip_slow_solution_ratio"] = 0.3
        else: # latency is None
            if config["M"] > 2048 and config["N"] > 2048 and config["K"] > 1024:
                config["skip_slow_solution_ratio"] = 0.6
            else:
                config["skip_slow_solution_ratio"] = 0.3
    return configs


# TODO take also custom library
def run(
    hipblaslt_path: str | Path,
    configs: List[dict],
    output_dir: str | Path,
    devices: Sequence[int] | None = None,
    bench_freq: bool = False,
    max_chunk_size: int = 25,
) -> pd.DataFrame:
    """Run GEMM dense benchmarks across multiple GPU devices.

    Executes hipblaslt-bench for each configuration in ``configs`` using the
    shared concurrency runner with one slot per device.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        configs (List[dict]): List of GEMM benchmarking configurations.
        devices (Sequence[int], optional): List of GPU device IDs to use for
            optimization. Defaults to None, which is interpreted as all available 
            devices [0, 1, 2, 3, 4, 5, 6, 7].
        bench_freq (bool, optional): Forwarded to bench.run for each
            per-config dense-search benchmark (controls HIPBLASLT_BENCH_FREQ).
            Defaults to False.
        max_chunk_size (int, optional): Maximum number of configs per chunk. 
            Defaults to 25.

    Raises:
        FileNotFoundError: If hipBLASLt path does not exist or max_chunk_size is invalid.

    Note:
        - Uses multiprocessing and threading for parallel optimization.
        - Tracks progress with tqdm progress bar.
        - Creates failed_benchmarks.log for any optimization failures.
    """
    if devices is None:
        devices = list(range(8))
    devices = parse_devices(devices)
    logger.debug(
        f"Starting dense search with output_dir={output_dir} devices={devices} "
        f"n_configs={len(configs)}"
    )

    hipblaslt_path = Path(hipblaslt_path)
    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if max_chunk_size <= 0:
        raise ValueError(f"max_chunk_size must be > 0, got {max_chunk_size}")

    if len(configs) == 0:
        return pd.DataFrame()

    def _config_to_gemm_key(cfg: dict) -> tuple:
        return (
            cfg["transA"],
            cfg["transB"],
            cfg["batch_count"],
            cfg["M"],
            cfg["N"],
            cfg["K"],
            cfg["a_type"],
            cfg["b_type"],
            cfg["c_type"],
            cfg["d_type"],
            _normalize_compute_type(cfg["compute_type"]),
        )

    configs = [(cfg, i) for i, cfg in enumerate(configs)]
    processed_file = output_dir / "processed.csv"
    processed_df = _load_processed_df(processed_file)
    if not processed_df.empty:
        processed_keys = set(
            tuple(row[c] for c in GEMM_FIELDS)
            for _, row in processed_df.drop_duplicates().iterrows()
        )
        configs = [ cfg for cfg in configs if _config_to_gemm_key(cfg[0]) not in processed_keys ]
        logger.info(
            f"Skipping {len(processed_keys)} already-processed GEMMs; remaining configs={len(configs)}"
        )

    if len(configs) == 0:
        existing = sorted(data_dir.glob("*.csv"))
        if len(existing) == 0:
            logger.warning(f"No succesful configs found; returning empty DataFrame")
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True).reset_index(drop=True)
        idx = df.groupby(list(GEMM_FIELDS))["us"].idxmin()
        return df.loc[idx].reset_index(drop=True)

    chunk_size = max(min(max_chunk_size, len(configs) // len(devices)), 1)
    chunks = np.array_split(configs, len(configs) // chunk_size)
    items = [ {"configs": chunk.tolist(), "keep_failed_logs": False} for chunk in chunks ]

    io_lock = Lock()
    failed_lock = Lock()
    failed_configs = []

    class SearchWorker(Worker[dict]):
        def setup(self) -> None:
            self.configs = self.item["configs"]
            self.ids = [cfg[1] for cfg in self.configs]
            self.chunk_id = self.ids[0] if len(self.ids) == 1 else f"{self.ids[0]}-{self.ids[-1]}"
            self.bench_file = output_dir / f"gemms_{self.chunk_id}.yaml"
            self.log_file = output_dir / f"gemms_{self.chunk_id}.out"
            self.passed = False
            self.keep_failed_logs = self.item.get("keep_failed_logs", False)
            
        def _winner_count(self) -> int:
            if not self.log_file.is_file():
                return 0
            return self.log_file.read_text().count("Winner")

        def run(self) -> bool:
            bench.log.dump([cfg[0] for cfg in self.configs], self.bench_file)
            try:
                bench.run(
                    hipblaslt_path,
                    self.bench_file,
                    self.log_file,
                    devices=[self.device],
                    cache=False,
                    bench_freq=bench_freq,
                    silent=True,
                )
            except Exception:
                self.passed = False
                return False
            
            self.passed = self._winner_count() == len(self.configs)
            return self.passed

        def teardown(self) -> None:
            if self.log_file.is_file() and self.passed:
                try:
                    res = bench.utils.parse_benchmark_output(self.log_file)
                    idx = res.groupby(list(GEMM_FIELDS))["us"].idxmin()
                    winners = res.loc[idx].reset_index(drop=True)

                    with io_lock:
                        ts = str(time.time_ns())
                        winners.to_csv(data_dir / f"{ts}.csv", index=False)

                        processed_chunk = winners[list(GEMM_FIELDS)].drop_duplicates().reset_index(drop=True)
                        processed_chunk["compute_type"] = processed_chunk["compute_type"].apply(_normalize_compute_type)

                        processed_df = _load_processed_df(processed_file)
                        processed_df = pd.concat([processed_df, processed_chunk], ignore_index=True)
                        processed_df = processed_df[list(GEMM_FIELDS)].drop_duplicates().reset_index(drop=True)
                        processed_df.to_csv(processed_file, index=False)

                except Exception as e:
                    logger.warning(f"Failed to persist gemms_{self.chunk_id} results: {e}")
            else:
                logger.warning(f"GEMMs {self.chunk_id} failed or missing log; skipping result persistence")
                if not self.keep_failed_logs:
                    self.log_file.unlink(missing_ok=True)
                    self.bench_file.unlink(missing_ok=True)
                with failed_lock:
                    failed_configs.extend(self.configs)
    
    prev_level = logger.level
    mute_for_progress = logger.getEffectiveLevel() > logging.DEBUG
    if mute_for_progress:
        logger.setLevel(logging.WARNING)

    runner = Runner(
        items=items,
        worker_impl=SearchWorker,
        devices=devices,
        n_slots=1,
    )

    try:
        runner(output_dir)
    finally:
        if mute_for_progress:
            logger.setLevel(prev_level)

    if failed_configs:
        retry_candidates = list(failed_configs)
        retry_total = len(retry_candidates)
        retry_items = [ {"configs": [cfg], "keep_failed_logs": True} for cfg in retry_candidates ]

        logger.info(f"Retrying {retry_total} failed configs as single-config items")
        failed_configs.clear()

        prev_level = logger.level
        mute_for_progress = logger.getEffectiveLevel() > logging.DEBUG
        if mute_for_progress:
            logger.setLevel(logging.WARNING)

        retry_runner = Runner(
            items=retry_items,
            worker_impl=SearchWorker,
            devices=devices,
            n_slots=1,
        )

        try:
            retry_runner(output_dir)
        finally:
            if mute_for_progress:
                logger.setLevel(prev_level)

        logger.info(
            f"Retry summary: passed {retry_total - len(failed_configs)} out of {retry_total} GEMMs"
        )

    if failed_configs:
        logger.info(
            f"Saving {len(failed_configs)} failed configs to '{output_dir / 'failed_gemms.yaml'}'"
        )
        bench.log.dump([cfg[0] for cfg in failed_configs], output_dir / "failed_gemms.yaml")

    data_files = sorted(data_dir.glob("*.csv"))
    if len(data_files) == 0:
        logger.warning(f"No succesful configs found; returning empty DataFrame")
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(p) for p in data_files], ignore_index=True).reset_index(drop=True)
    idx = df.groupby(list(GEMM_FIELDS))["us"].idxmin()
    return df.loc[idx].reset_index(drop=True)
