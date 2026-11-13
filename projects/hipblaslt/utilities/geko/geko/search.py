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
import time

from collections import defaultdict
from threading import Lock
from pathlib import Path
from typing import Sequence, List, Dict
from geko import bench
from geko.bench.utils import is_built_custom_library
from geko.constants import GEMM_LOG_FIELDS, GEMM_FIELDS
from geko.utils import parse_devices
from geko.concurrency.runner import Runner, Worker

logger = logging.getLogger("GEKO")

# --------------------------------------------------------------------------- #
# Search profiles: CLI-selectable presets bundling the hipblaslt-bench
# --adaptive timing knobs, replacing the old --duration/--no-adaptive-search
# flags. Values come from the adaptive-timing regret-sweep experiment
# (workspace/offtune/test_cpp/experiments/search_adaptive/run_experiment.py):
#
#   fast     - Convergence-driven single dense pass. Cheapest, used as-is
#              for a quick dense search.
#   balanced - Two-stage search: a cheap "stage_a" screen over the full
#              population, then a strict "stage_b" re-verify restricted to
#              each GEMM's own top-"k" screen candidates. Unlike "fast"/
#              "thorough" (flat adaptive-knob dicts), "balanced" nests two
#              such dicts under "stage_a"/"stage_b" plus a "k" int.
#              configure() detects this nesting and builds off "stage_a"
#              only -- search.run() doesn't implement the stage_b re-verify
#              pass yet, so stage_b/k aren't consumed until that lands.
#   thorough - Tightest convergence/ceiling knobs from the regret sweep.
#              Most expensive; full/accurate search. Default profile.
#
# Each flat (non-two-stage) entry fully resolves the adaptive block:
# measure_time is the measurement floor (ms); warmup_time/max_measure_time/
# sample_time are derived from it via warmup_mult/ceil_mult/min_samples
# (warmup_time = warmup_mult x measure_time, max_measure_time = ceil_mult x
# measure_time, sample_time = measure_time / min_samples).
# --------------------------------------------------------------------------- #
SEARCH_PROFILES: dict[str, dict] = {
    "fast": dict(
        measure_time=5.0, warmup_mult=0.4, ceil_mult=3.0, min_samples=10,
        noise_threshold=0.01, min_iters=3, max_iters=10000,
        stability_threshold=0.0, stability_window=32, stability_interval=16,
    ),
    "thorough": dict(
        measure_time=20.0, warmup_mult=2.0, ceil_mult=16.0, min_samples=10,
        noise_threshold=0.005, min_iters=10, max_iters=10000,
        stability_threshold=0.03, stability_window=6, stability_interval=3,
    ),
}
SEARCH_PROFILES["balanced"] = dict(
    stage_a=SEARCH_PROFILES["fast"],
    stage_b=SEARCH_PROFILES["thorough"],
    k=50,
)


def _normalize_compute_type(v: str) -> str:
    # Normalize legacy values so downstream joins always use c_* compute_type keys.
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
    profile: str = "thorough",
    iters: int = 100,
    cold_iters: int = 20,
    rotating: int = 512,
    beta: bool = False,
    flush: bool = False,
) -> List[dict]:
    """Generate benchmarking configurations for each input GEMM.

    Creates YAML and shell script configuration files for tuning a specific
    GEMM operation type using the config generator framework.

    Every config is emitted in hipblaslt-bench ``--adaptive`` mode: the bench
    warms up, then accumulates samples until the measured mean's relative
    standard error drops below ``noise_threshold`` (converged), the robust
    spread plateaus (stable), or a ceiling is hit. The adaptive timing knobs
    (``measure_time``, ``warmup_time``, ``max_measure_time``, ``sample_time``,
    ``noise_threshold``, ``min_iters``/``max_iters``, and the
    ``stability_*`` knobs) are bundled per ``profile`` -- see
        ``SEARCH_PROFILES`` for the "fast"/"balanced"/"thorough" presets. For
    the two-stage "balanced" profile (a ``{"stage_a", "stage_b", "k"}``
    dict rather than a flat adaptive-knob dict), only the "stage_a" screen
    config is used here.
    Args:
        df (DataFrame): DataFrame containing the GEMMs to benchmark.
        profile (str, optional): Search profile name, a key of
            ``SEARCH_PROFILES`` ("fast", "balanced", or "thorough").
            Defaults to "thorough".
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
        ValueError: If input DataFrame is empty or profile is unknown.
    """
    if df.empty:
        raise ValueError("No GEMM operations found in input DataFrame")

    if profile not in SEARCH_PROFILES:
        raise ValueError(
            f"Unknown search profile '{profile}'; expected one of {sorted(SEARCH_PROFILES)}"
        )
    adaptive_cfg = SEARCH_PROFILES[profile]
    if "stage_a" in adaptive_cfg:
        adaptive_cfg = adaptive_cfg["stage_a"]
    measure_time = float(adaptive_cfg["measure_time"])

    df = df.rename({"m": "M", "n": "N", "k": "K"}, axis=1)
    uniq_df = df[list(GEMM_LOG_FIELDS)].drop_duplicates()

    latency = None
    if "us" in df.columns:
        latency = df.loc[uniq_df.index, "us"].values

    configs = bench.log.update(
        uniq_df.to_dict(orient="records"),
        latency=latency,
        duration=measure_time / 1000.0,  # bench.log.update expects seconds; measure_time is ms
        iters=iters,
        cold_iters=cold_iters,
        rotating=rotating,
        beta=beta,
        flush=flush,
        aux=False,
        print_kernel_info=True,
        adaptive=False,  # the profile's own adaptive fields are applied below
    )[0]

    # Profile keys are named after the bench.log.adaptive_fields parameters.
    adaptive_fields = bench.log.adaptive_fields(**adaptive_cfg)
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

        config.update(adaptive_fields)

    return configs


def run(
    hipblaslt_path: str | Path,
    configs: List[dict],
    output_dir: str | Path,
    custom_lib_dir: str | Path | None = None,
    devices: Sequence[int] | None = None,
    bench_freq: bool = False,
    max_chunk_size: int = 5,
    profile: str | None = None,
) -> pd.DataFrame:
    """Run GEMM dense benchmarks across multiple GPU devices.

    Executes hipblaslt-bench for each configuration in ``configs`` using the
    shared concurrency runner with one slot per device.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation.
        configs (List[dict]): List of GEMM benchmarking configurations.
        custom_lib_dir (str | Path | None, optional): Optional built custom
            library directory used by hipblaslt-bench for dense search.
            Defaults to None.
        devices (Sequence[int], optional): List of GPU device IDs to use for
            optimization. Defaults to None, which is interpreted as all available 
            devices [0, 1, 2, 3, 4, 5, 6, 7].
        bench_freq (bool, optional): Forwarded to bench.run for each
            per-config dense-search benchmark (controls HIPBLASLT_BENCH_FREQ).
            Defaults to False.
        max_chunk_size (int, optional): Maximum number of configs per chunk. 
            Defaults to 5.
        profile (str | None, optional): Search profile name, a key of
            ``SEARCH_PROFILES`` (should match the profile passed to
            ``configure()`` for these same ``configs``). When the resolved
            profile has a ``"stage_b"`` entry (currently only "balanced"),
            each chunk is re-verified after its own stage_a (screen) pass:
            every GEMM's top-``k`` stage_a candidates (ranked by measured
            latency) are re-benchmarked individually with stage_b's own
            (stricter) adaptive config, and that stage_b result becomes the
            GEMM's final winner. Defaults to None (single-stage behavior,
            unchanged from before two-stage support was added).

    Raises:
        FileNotFoundError: If hipBLASLt path does not exist.
        ValueError: If max_chunk_size is invalid, custom_lib_dir is not
            built, profile is unknown, or profile has a "stage_b" entry
            without a positive "k".

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

    stage_b_cfg = None
    top_k = None
    if profile is not None:
        if profile not in SEARCH_PROFILES:
            raise ValueError(
                f"Unknown search profile '{profile}'; expected one of {sorted(SEARCH_PROFILES)}"
            )
        profile_cfg = SEARCH_PROFILES[profile]
        stage_b_cfg = profile_cfg.get("stage_b")
        top_k = profile_cfg.get("k")
        if stage_b_cfg is not None and not top_k:
            raise ValueError(
                f"Search profile '{profile}' has a 'stage_b' entry but no positive 'k'"
            )
    stage_b_adaptive = bench.log.adaptive_fields(**stage_b_cfg) if stage_b_cfg is not None else None

    hipblaslt_path = Path(hipblaslt_path)
    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    if custom_lib_dir is not None:
        custom_lib_dir = Path(custom_lib_dir)
        if not is_built_custom_library(custom_lib_dir):
            raise ValueError(
                f"Custom library in '{custom_lib_dir}' is not built. "
                "Expected files matching 'library/**/TensileLibrary_lazy_gfx*.dat'."
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if max_chunk_size <= 0:
        raise ValueError(f"max_chunk_size must be > 0, got {max_chunk_size}")

    if len(configs) == 0:
        return pd.DataFrame()

    def _config_to_gemm_key(cfg: dict) -> tuple:
        key = [cfg[k] for k in GEMM_LOG_FIELDS]
        index = GEMM_LOG_FIELDS.index("compute_type")
        key[index] = _normalize_compute_type(key[index])
        return tuple(key)

    def _res_row_to_gemm_key(row) -> tuple:
        # GEMM_FIELDS and GEMM_LOG_FIELDS share the same field order (only
        # m/n/k vs M/N/K differ in name case; enforced by an assert in
        # constants.py), so this is comparable to _config_to_gemm_key()'s
        # output for the same GEMM.
        key = [row[c] for c in GEMM_FIELDS]
        index = GEMM_FIELDS.index("compute_type")
        key[index] = _normalize_compute_type(key[index])
        return tuple(key)

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

    def _config_flops(cfg: dict) -> float:
        return float(cfg["M"]) * float(cfg["N"]) * float(cfg["K"]) * float(cfg.get("batch_count", 1))

    chunk_size = max(min(max_chunk_size, len(configs) // len(devices)), 1)
    n_chunks = max(len(configs) // chunk_size, 1)
    # Cap chunk size so all chunks stay equally "full" (not just equally
    # flops-weighted), avoiding idle GPUs from undersized chunks.
    capacity = -(-len(configs) // n_chunks)  # ceil(len(configs) / n_chunks)

    # Greedy LPT bin balancing: assign heaviest (FLOPs = M*N*K*batch_count)
    # configs first, each to the lowest-total non-full chunk.
    ordered = sorted(configs, key=lambda cfg: _config_flops(cfg[0]), reverse=True)
    chunk_bins: List[List[tuple]] = [[] for _ in range(n_chunks)]
    chunk_flops = [0.0] * n_chunks
    for cfg in ordered:
        idx = min(
            (i for i in range(n_chunks) if len(chunk_bins[i]) < capacity),
            key=lambda i: chunk_flops[i],
        )
        chunk_bins[idx].append(cfg)
        chunk_flops[idx] += _config_flops(cfg[0])

    # Chunk membership isn't an id range anymore, so tag each chunk with a
    # stable index for naming (see setup() below).
    items = [
        {"configs": chunk, "chunk_index": i, "keep_failed_logs": False}
        for i, chunk in enumerate(chunk_bins)
        if chunk
    ]

    io_lock = Lock()
    failed_lock = Lock()
    failed_configs = []

    class SearchWorker(Worker[dict]):
        def setup(self) -> None:
            self.configs = self.item["configs"]
            self.ids = [cfg[1] for cfg in self.configs]
            # Multi-config chunks: "chunk_<i>". Single-config items (a
            # 1-item chunk, or a retry) keep "gemms_<id>" so retries never
            # clash with an existing chunk's files.
            if len(self.configs) > 1:
                self.chunk_id = f"chunk_{self.item['chunk_index']}"
            else:
                self.chunk_id = f"gemms_{self.ids[0]}"
            self.bench_file = output_dir / f"{self.chunk_id}.yaml"
            self.log_file = output_dir / f"{self.chunk_id}.out"
            self.passed = False
            self.keep_failed_logs = self.item.get("keep_failed_logs", False)
            
        def _winner_count(self) -> int:
            if not self.log_file.is_file():
                return 0
            return self.log_file.read_text().count("Winner")

        def _stage_b_configs(self, stage_a_log: Path) -> List[dict]:
            # Rank each GEMM's own stage_a candidates by measured latency
            # (ascending; ties keep stage_a's first-seen order) and shortlist
            # the first top_k distinct solution indices per GEMM.
            res = bench.utils.parse_benchmark_output(stage_a_log).sort_values("us")
            shortlist: Dict[tuple, List[int]] = defaultdict(list)
            for _, row in res.iterrows():
                key = _res_row_to_gemm_key(row)
                sols = shortlist[key]
                sol = int(row["solutionIdx"])
                if sol in sols:
                    continue
                if len(sols) < top_k:
                    sols.append(sol)

            rows: List[dict] = []
            for cfg, _ in self.configs:
                key = _config_to_gemm_key(cfg)
                for sol in shortlist.get(key, []):
                    row = dict(cfg)
                    row["algo_method"] = 2
                    row["solution_index"] = sol
                    row.pop("requested_solution_num", None)
                    row.update(stage_b_adaptive)
                    rows.append(row)
            return rows

        def run(self) -> bool:
            bench.log.dump([cfg[0] for cfg in self.configs], self.bench_file)
            try:
                bench.run(
                    hipblaslt_path,
                    self.bench_file,
                    self.log_file,
                    custom_lib_dir=custom_lib_dir,
                    devices=[self.device],
                    cache=False,
                    bench_freq=bench_freq,
                    silent=True,
                )
            except Exception:
                self.passed = False
                return False

            stage_a_passed = self._winner_count() == len(self.configs)
            if stage_b_cfg is None or not stage_a_passed:
                self.passed = stage_a_passed
                return self.passed

            # Two-stage: keep stage_a's own output under its own name, then
            # re-verify each GEMM's own top-k stage_a candidates with
            # stage_b's stricter adaptive config, writing the FINAL result
            # back to self.log_file. teardown() stays unchanged and simply
            # operates on stage_b data instead of stage_a data.
            stage_a_log = self.log_file.with_name(f"{self.log_file.stem}_stage_a.out")
            self.log_file.rename(stage_a_log)

            stage_b_rows = self._stage_b_configs(stage_a_log)
            if not stage_b_rows:
                self.passed = False
                return False

            bench.log.dump(stage_b_rows, self.bench_file)
            try:
                bench.run(
                    hipblaslt_path,
                    self.bench_file,
                    self.log_file,
                    custom_lib_dir=custom_lib_dir,
                    devices=[self.device],
                    cache=False,
                    bench_freq=bench_freq,
                    silent=True,
                )
            except Exception:
                self.passed = False
                return False

            try:
                stage_b_res = bench.utils.parse_benchmark_output(self.log_file)
            except ValueError:
                self.passed = False
                return False

            self.passed = len(stage_b_res) == len(stage_b_rows)
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
                    logger.warning(f"Failed to persist {self.chunk_id} results: {e}")
            else:
                logger.warning(f"{self.chunk_id} failed or missing log; skipping result persistence")
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
