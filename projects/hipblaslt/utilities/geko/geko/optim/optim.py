# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""GEMM Kernel Optimization module.

Implements the main optimization workflow.
Generates Tensile tensilelite for specific GEMM types and executes multi-GPU
optimization with automatic load balancing and progress monitoring.

Key functions:
    configure: Generate tensilelite configs for GEMM types.
    run: Execute optimization across multiple GPUs.
    analyze: Benchmark and filter optimized kernels.

Integrates with hipBLASLt for benchmarking and Tensile for kernel compilation.
"""

import os
import subprocess
import re
import shutil
import time
import yaml
import pandas as pd

import logging

logger = logging.getLogger("GEKO")

from pathlib import Path
from typing import List, Sequence, Union
from threading import Lock
from dataclasses import dataclass

from geko import bench
from geko.config_generator import config_generator as cg
from geko.config_generator.load_input_config import apply_input_config_defaults
from geko.constants import GEMM_FIELDS
from geko.schemas import GemmConfig
from geko.concurrency.runner import Runner, Worker
from geko.utils import (
    build_tensilelite_client,
    parse_devices,
)
from geko.concurrency.utils import wait_process_or_stop
from geko.optim.utils import *
from geko.optim import _metrics


__all__ = ["configure", "run", "analyze"]

_THIS_FILE = Path(__file__).resolve()
if len(_THIS_FILE.parents) < 4:
    raise RuntimeError(f"Invalid GEKO layout: expected at least 4 parent levels for '{_THIS_FILE}'")

GEKO_ROOT = _THIS_FILE.parents[1]
if  GEKO_ROOT.name != "geko" or not (GEKO_ROOT / "__init__.py").is_file():
    raise RuntimeError(
        f"Invalid GEKO_ROOT '{GEKO_ROOT}'. Expected root package directory containing "
        f"'__init__.py'."
    )
TD_PATH = _THIS_FILE.parents[3]


@dataclass
class DeviceState:
    free_slots: int
    workload: float


def _log_work(tuning_dir: Path, name: str):
    """Collect last observed GA generation from active build dirs and save a sorted summary log."""

    def _extract_last_generation(content: str) -> int:
        # Ductile logs contain table rows like: "   3   |  1829   | ..."
        matches = re.findall(r"^\s*(\d+)\s*\|", content, flags=re.MULTILINE)
        if len(matches) > 0:
            return int(matches[-1])

        return 0

    jobs = []
    try:
        running_markers = list(tuning_dir.glob("build_*/.running"))
    except (FileNotFoundError, OSError):
        # Directory may be deleted by concurrent worker during glob; return empty
        running_markers = []

    for marker in running_markers:
        build_dir = marker.parent
        try:
            info = dict(
                line.split("=", 1)
                for line in marker.read_text().splitlines()
                if "=" in line
            )
            device_id = info.get("device", "?")
            slot_id = info.get("slot", "?")
        except (FileNotFoundError, OSError, ValueError):
            continue
        try:
            logs = list(build_dir.glob("*-tensilelite.log"))
            if not logs:
                n_gen = 0
                path = marker
            else:
                path = logs[0]
                with open(path, "r") as f:
                    content = f.read()
                n_gen = _extract_last_generation(content)
        except (FileNotFoundError, OSError):
            continue
        jobs.append((device_id, slot_id, n_gen, path))

    jobs.sort(key=lambda x: (x[0], x[1]))
    content = "\n".join([f"device={d} | slot={s} | n_gen={n:02d} | log='{p}'" for d, s, n, p in jobs])
    with open(tuning_dir / f"{name}.log", "w") as f:
        f.write(content)



def configure(
    hipblaslt_path: str | Path,
    gemm_configs: Union[GemmConfig, Sequence[GemmConfig]],
    output_dir: str | Path,
    arch: str = "gfx950",
    backend: str = "ductile"
) -> dict:
    """Generate Tensile optimization configuration for one or more GEMM types.

    Builds a config dict from gemm_configs (each a GemmConfig with its
    GemmType and size list), applies ARCH-specific defaults via
    apply_input_config_defaults, and runs config_generator.run to write
    tensilelite tuning YAML (and side artifacts) under output_dir.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        gemm_configs (GemmConfig | Sequence[GemmConfig]): One GemmConfig
            or a sequence of them.  All entries are placed into
            config["GemmProblems"] and emitted in a single generator pass.
        output_dir (str | Path): Output directory for generated config files.
        arch (str, optional): Target GPU architecture (gfx-style string written
            into config["ARCH"]). Defaults to "gfx950".
        backend (str, optional): tensilelite backend; "ductile" (GA) or
            "tensile" (sets config["GA"] accordingly). Defaults to
            "ductile".

    Returns:
        dict: The fully populated config dict (after defaults and the
            generator pass), including GemmProblems and ARCH-derived
            hardware fields.
    """
    gcs: List[GemmConfig] = (
        [gemm_configs] if isinstance(gemm_configs, GemmConfig) else list(gemm_configs)
    )

    for gc in gcs:
        logger.info(f"{gc.gemm_type} with {len(gc.sizes)} sizes")
        gt = gc.gemm_type
        logger.debug(
            f"Preparing optimization config: gemm_type={gc.gemm_type} "
            f"arch={arch} backend={backend} n_sizes={len(gc.sizes)}"
        )
        logger.debug(
            f"GEMM datatype mapping: DataType={gt.data_type} "
            f"DestDataType={gt.dest_data_type} ComputeDataType={gt.compute_data_type} "
            f"TRANSA={gt.transA} TRANSB={gt.transB} gemm_name={gt.gemm_name}"
        )

    config: dict = {
        "ARCH": arch,
        "GA": backend.lower() == "ductile",
    }
    config["GemmProblems"] = gcs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_input_config_defaults(config)

    cg.run(
        config,
        hipblaslt_path,
        output_dir,
        write_shell_scripts=False,
    )

    names = ", ".join(gc.gemm_type.gemm_name for gc in gcs)
    logger.info(f"Saved GEMM config(s) to '{output_dir}': {names}")
    return config

def run(
    hipblaslt_path: str | Path,
    tuning_dir: str | Path,
    devices: Sequence[int] | None = None,
    client_build_dir: str | Path = Path("build_tmp"),
    n_slots: int = 4,
    retry: bool = True,
) -> None:
    """Run tensilelite optimization across every tuning YAML under tuning_dir.

    Builds the tensilelite client once and dispatches one job per discovered
    config to the shared Runner abstraction. Scheduling matches the previous
    run() implementation (workload-based assignment with LPT ordering);
    progress reporting follows search.run() (queue-driven completion updates).

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation. Used both
            for the Tensile binary and to add tensilelite to PYTHONPATH.
        tuning_dir (str | Path): Directory containing per-GEMM optimization
            YAML configs (see configure).
        devices (Sequence[int], optional): GPU device IDs used by the load
            balancer. 
            Defaults to None, which is interpreted as all 8 devices if not specified.
        client_build_dir (str | Path, optional): Build directory for the
            tensilelite client (see build_tensilelite_client).
            Defaults to Path("build_tmp").
        n_slots (int, optional): Maximum concurrent jobs per device. Defaults
            to 4.
        retry (bool, optional): Worker-level retry behavior for existing
            failed / resumable build dirs. Defaults to True.

    Raises:
        FileNotFoundError: If hipblaslt_path does not exist.
        ValueError: If n_slots is less than 1.

    Note:
        Writes timing.log, periodic running_jobs.log snapshots, optional
        failed_jobs.log, and per-config build_* artifact trees underneath
        tuning_dir.
    """
    if devices is None:
        devices = list(range(8))  # Default to all 8 devices if not specified
    devices = parse_devices(devices)
    logger.debug(
        f"Starting optimization run_new with tuning_dir={tuning_dir} devices={devices} "
        f"n_slots={n_slots} retry={retry}"
    )

    hipblaslt_path = Path(hipblaslt_path)
    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    tuning_dir = Path(tuning_dir)
    client_build_dir = Path(client_build_dir)

    if n_slots < 1:
        raise ValueError("n_slots must be >= 1")

    configs = list_optimization_configs(tuning_dir)

    if len(configs) == 0:
        logger.info("No optimizations to run")
        return

    build_tensilelite_client(hipblaslt_path, build_dir=client_build_dir)

    _timing_lock = Lock()

    class OptimWorker(Worker):
        def setup(self) -> None:
            self.start_time = time.time()
            self.config = Path(self.item)
            self.config_name = self.config.stem
            self.build_dir = tuning_dir / f"build_{self.config_name}"
            self.retry_allowed = retry
            self.should_run = True

            self.state = get_build_state(self.build_dir)

            if self.state == "running":
                (self.build_dir / ".running").unlink(missing_ok=True)
                self.state = get_build_state(self.build_dir)

            if self.stop_event.is_set() or self.state == "completed":
                self.should_run = False
                return
            
            if self.state in ("failed", "resumable"):
                if self.retry_allowed:
                    clean_failed_build(self.build_dir)
                else:
                    # Failed/resumable jobs are reported without mutating existing artifacts.
                    self.should_run = False

        def run(self) -> bool:
            if not self.should_run:
                return self.state == "completed"

            logger.debug(
                f"Worker starting config={self.config_name} device={self.device} slot={self.slot_id}"
            )

            with open(self.config) as f:
                content = re.sub(r"Device:\s*\d+", f"Device: {self.device}", f.read())
            with open(self.config, "w") as f:
                f.write(content)

            self.build_dir.mkdir(parents=True, exist_ok=True)
            (self.build_dir / ".running").write_text(f"device={self.device}\nslot={self.slot_id}\n")

            env = {"PYTHONPATH": str(hipblaslt_path / "tensilelite")}
            with open(self.build_dir / f"{self.config_name}-tensilelite.log", "w") as f:
                proc = subprocess.Popen(
                    [
                        hipblaslt_path / "tensilelite/Tensile/bin/Tensile",
                        self.config,
                        self.build_dir,
                        "--prebuilt-client",
                        client_build_dir / "tensilelite/client/tensilelite-client",
                        "--client-lock",
                        (tuning_dir / f"gpulock_{self.device}").resolve(),
                    ],
                    env=os.environ | env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                wait_process_or_stop(proc, self.stop_event, self.config_name)

            if proc.returncode != 0:
                logger.debug(
                    f"Optimizer subprocess reported non-zero return code for config={self.config_name}; "
                    "artifacts may be incomplete"
                )
                return False

            shutil.copyfile(self.config, self.build_dir / self.config.name)
            return True

        def teardown(self) -> None:
            (self.build_dir / ".running").unlink(missing_ok=True)
            if not self.should_run or get_build_state(self.build_dir) != "completed":
                return
            elapsed = int(time.time() - self.start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            timings_log = tuning_dir / "timing.log"
            with _timing_lock:
                with open(timings_log, "a") as f:
                    f.write(f"Time taken for {self.config_name}: {hours}h {minutes}m {seconds}s\n")

    runner = Runner(
        items=configs,
        worker_impl=OptimWorker,
        devices=devices,
        n_slots=n_slots,
        estimate_workload_fn=estimate_workload,
        job_logger_fn=lambda: _log_work(tuning_dir, "running_jobs"),
    )

    runner(tuning_dir)


def analyze(
    hipblaslt_path: str | Path,
    lib_dir: str | Path,
    output_dir: str | Path,
    benchmark_dir: str | Path = Path("benchmarks"),
    custom_lib_dir: str | Path = Path("build"),
    devices: Sequence[int] | None = None,
    error_thr: float = 0.03,
    up_thr: float = 1.03,
    duration: float = 1.0,
    beta: bool = True,
    log_summary: str | Path = None,
    verify: bool = True,
    bench_freq: bool = False,
    device: int | None = None,
) -> Tuple[pd.DataFrame | None, pd.DataFrame]:
    """Benchmark and analyze optimized kernels against reference libraries.

    Compares performance of tuned kernels vs reference implementation,
    filters results by accuracy and performance thresholds.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        lib_dir (str | Path): Directory containing optimized library files.
        output_dir (str | Path): Output directory for analysis results.
        benchmark_dir (str | Path, optional): Directory for benchmark files.
            Defaults to "benchmarks".
        custom_lib_dir (str | Path, optional): Directory for custom library creation.
            Defaults to "build".
        devices (Sequence[int], optional): GPU device IDs used by the load
            Defaults to None, which is interpreted as [0] if not specified.
        error_thr (float, optional): Maximum acceptable numerical error threshold.
            Defaults to 0.03.
        up_thr (float, optional): Minimum performance improvement ratio threshold.
            Defaults to 1.03.
        duration (float, optional): Benchmark duration in seconds.
            Defaults to 1.0.
        beta (bool, optional): Whether to use non-zero beta values.
            Defaults to True.
        log_summary (str | Path, optional): CSV file with GEMM contribution to
            calculate weighted uplift. Defaults to None.
        bench_freq (bool, optional): Forwarded to bench.compare (controls
            HIPBLASLT_BENCH_FREQ). Defaults to False.
        device (int, optional): Backward-compatible single-device alias.
            If set, overrides devices.

    Returns:
        Tuple[pd.DataFrame | None, pd.DataFrame]: Tuple containing the final filtered DataFrame 
            and the full DataFrame with winner column.

    Note:
        - Creates raw_results.csv with all benchmark data.
        - Creates final_results.csv with filtered results.
        - Reports average and weighted GEMM uplift percentages.
    """
    if device is not None:
        devices = [device]
    if devices is None:
        devices = [0]  # Default to device 0 if not specified
    devices = parse_devices(devices)

    df = bench.compare(
        hipblaslt_path,
        lib_dir,
        custom_lib_dir=custom_lib_dir,
        benchmark_dir=benchmark_dir,
        verify=verify,
        cache=True,
        duration=duration,
        beta=beta,
        devices=devices,
        bench_freq=bench_freq,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "error_tuned" not in df.columns:
        df["error_tuned"] = 0.0

    # Distinguish "verification ran and passed" (verified=True, error<thr)
    # from "verification was skipped" (verified=False, error_tuned silently 0).
    df["verified"] = bool(verify)
    df["valid"] = df["error_tuned"] < error_thr
    logger.debug(
        f"Analyze thresholds: rows={len(df)} valid={int(df['valid'].sum())} "
        f"invalid={int((~df['valid']).sum())} error_thr={error_thr:.6f} up_thr={up_thr:.6f}"
    )

    # Enrich raw results with derived metrics so raw_results.csv is also informative.
    df_enriched = _metrics.enrich(df, summary_csv=log_summary)
    df_enriched.drop("lib", axis=1, errors="ignore").to_csv(
        output_dir / "raw_results.csv", index=False
    )
    logger.info(f"RAW benchmark results saved in '{output_dir}/raw_results.csv'")

    # TODO also minimum diff, e.g. half a us
    mask = (df["kernel_reference"] != df["kernel_tuned"]) * (df["ratio"] >= up_thr) * df["valid"]
    final_df = df[mask]
    final_enriched = df_enriched[mask]

    # Run-level summary stats on the full (raw) and kept (final) sets.
    # Run unconditionally so empty-result runs still produce a metrics.json.
    stats = _metrics.summarize(
        df_raw=df_enriched,
        df_final=final_enriched,
        summary_csv=log_summary,
        up_thr=up_thr,
        error_thr=error_thr,
        verify=verify,
    )
    _metrics.write_metrics_json(stats, output_dir / "metrics.json")
    logger.info(f"Run-level metrics saved in '{output_dir}/metrics.json'")

    # Human-readable log lines (preserve historical log-line shape for log scrapers).
    if stats["uplift_kept"].get("mean_uplift_pct") is not None:
        gm = stats["uplift_kept"].get("geomean_uplift_pct")
        gm_str = f"{gm:.4f}" if gm is not None else "n/a"
        logger.info(
            f"Average GEMM uplift of {stats['uplift_kept']['mean_uplift_pct']:.4f}% "
            f"(geomean {gm_str}%) for a total of {stats['counts']['n_kept']} kept GEMMs"
        )
    if stats["e2e"].get("e2e_uplift_pct") is not None:
        logger.info(f"Weighted GEMM uplift of {stats['e2e']['e2e_uplift_pct']:.4f}%")

    df["winner"] = "reference"
    df.loc[mask, "winner"] = "tuned"
    df = df.drop("valid", axis=1, errors="ignore")
    if mask.sum() == 0:
        logger.info(f"No kernels improve over the base library")
        return None, df

    final_enriched.drop(["valid", "lib"], axis=1, errors="ignore").to_csv(
        output_dir / "final_results.csv", index=False
    )
    logger.info(f"FINAL benchmark results saved in '{output_dir}/final_results.csv'")

    bench.utils.as_dashboard_format(final_df.drop("valid", axis=1, errors="ignore")).to_csv(
        output_dir / "dashboard_data.csv", index=False
    )

    return final_df.drop("valid", axis=1, errors="ignore"), df
