# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run selected CPU checks concurrently in isolated rocjitsu processes."""

import json
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def available_cpu_count() -> int:
    """Respect Linux CPU affinity and cgroup quotas, not host CPU count alone."""
    count = len(os.sched_getaffinity(0))
    for quota_path, period_path in (
        (Path("/sys/fs/cgroup/cpu.max"), None),
        (
            Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        ),
    ):
        try:
            if period_path is None:
                quota, period = quota_path.read_text().split()
                if quota == "max":
                    continue
                quota, period = int(quota), int(period)
            else:
                quota = int(quota_path.read_text())
                period = int(period_path.read_text())
            if quota > 0 and period > 0:
                count = min(count, max(1, quota // period))
        except FileNotFoundError:
            continue
    return max(1, count)


def run_cpu_shards(
    command: list[str],
    rocjitsu: Path,
    config_path: Path,
    env: dict[str, str],
    working_dir: Path,
    reports: Path,
    workers: int,
    timeout: int,
    dispatch_threads: int,
    run_logged: Callable[[list[str], dict[str, str], Path, Path, int], None],
) -> tuple[list[Path], list[str]]:
    """Attempt every shard, returning expected XML paths and actual errors.

    Keep absent reports in the result so aggregation rejects partial runs.
    run_logged owns timeout, process-group cleanup and stdout/stderr logs.
    """
    cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= dispatch_threads <= 4 or workers < 1 or timeout <= 0:
        raise ValueError(
            "CPU shards require workers >= 1, dispatch 1..4 and timeout > 0"
        )
    if workers * dispatch_threads > min(len(cpus), available_cpu_count()):
        raise ValueError("CPU shard masks exceed the available CPU allocation")
    reports = reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    report_paths = [reports / f"cpu-shard-{index}.xml" for index in range(workers)]
    for report in report_paths:
        report.unlink(missing_ok=True)
    config = json.loads(config_path.read_text())
    config["vm"]["gpu"]["num_gpus"] = 1
    config["cpu_dispatch_threads"] = dispatch_threads
    config["num_threads"] = 1
    config["async_helper_threads"] = 0
    cpu_config = reports / "rocjitsu-cpu-config.json"
    cpu_config.write_text(json.dumps(config, indent=2) + "\n")
    launcher = [str(rocjitsu.resolve()), "--config", str(cpu_config), "--"]
    cpu_sets = [
        cpus[index * dispatch_threads : (index + 1) * dispatch_threads]
        for index in range(workers)
    ]
    # CLI flags outrank environment settings: keep selection, but make output,
    # repetition and fail-fast shard-local so no inherited flags truncate it.
    shard_command = execution_command(command)

    def run_shard(index: int) -> None:
        report = report_paths[index]
        with tempfile.TemporaryDirectory(prefix=f"miopen-cpu-shard-{index}-") as temp:
            state = Path(temp)
            shard_env = env.copy()
            shard_env.pop("GTEST_SHARD_STATUS_FILE", None)
            for name, directory in (
                ("HOME", "home"),
                ("MIOPEN_USER_DB_PATH", "db"),
                ("MIOPEN_CUSTOM_CACHE_DIR", "cache"),
                ("MIOPEN_VERIFY_CACHE_PATH", "verify-cache"),
                ("XDG_CACHE_HOME", "xdg-cache"),
                ("XDG_CONFIG_HOME", "xdg-config"),
                ("XDG_DATA_HOME", "xdg-data"),
                ("XDG_STATE_HOME", "xdg-state"),
                ("XDG_RUNTIME_DIR", "xdg-runtime"),
                ("TMPDIR", "tmp"),
            ):
                directory_path = state / directory
                directory_path.mkdir(mode=0o700)
                shard_env[name] = str(directory_path)
            for name in (
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "OPENBLAS_NUM_THREADS",
                "GOTO_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "MIOPEN_COMPILE_PARALLEL_LEVEL",
                "MIOPEN_DBSYNC_MAX_THREADS",
                "MIOPEN_AI_FDEEP_USE_SINGLE_THREAD_PREDICT",
            ):
                shard_env[name] = "1"
            shard_env["GTEST_TOTAL_SHARDS"] = str(workers)
            shard_env["GTEST_SHARD_INDEX"] = str(index)
            shard_env["GTEST_REPEAT"] = "1"
            shard_env["GTEST_FAIL_FAST"] = "0"
            shard_env["GTEST_OUTPUT"] = f"xml:{report}"
            run_logged(
                ["taskset", "-c", ",".join(map(str, cpu_sets[index]))]
                + launcher
                + shard_command,
                shard_env,
                working_dir,
                reports / f"cpu-shard-{index}.log",
                timeout,
            )

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_shard, index) for index in range(workers)]
        for index, future in enumerate(futures):
            try:
                future.result()
            except Exception as error:
                # Failure must not cancel other CPU shards or the GPU phase.
                errors.append(
                    f"CPU shard {index}: {error}; see {reports / f'cpu-shard-{index}.log'}"
                )
    return sorted(report_paths), errors


def execution_command(command: list[str]) -> list[str]:
    """Remove GTest controls that could truncate or replace execution reports."""
    return [
        value
        for value in command
        if not value.startswith(
            ("--gtest_output=", "--gtest_repeat=", "--gtest_fail_fast=")
        )
        and value != "--gtest_fail_fast"
    ]
