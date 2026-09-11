# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Utility functions for optimization progress tracking and device management.

Provides functions for:
- Monitoring optimization progress across configuration files.
- Managing GPU device specifications and parsing.
- Cleaning failed optimization attempts.
- Tracking completed vs failed optimization jobs.
- Estimating optimization job workload.

Functions:
    check_progress: Monitor optimization completion status.
    clean_failed_build: Remove incomplete artifacts for a single build directory.
    list_optimization_configs: Find optimization configuration files.
    clean_failed_builds: Remove incomplete optimization artifacts.
    estimate_workload: Estimates the workload of an optimization job.
"""

import shutil
import re
import math
import yaml

from pathlib import Path
from typing import List, Tuple


try:
    SafeLoader = yaml.CSafeLoader
except (ModuleNotFoundError, AttributeError):
    SafeLoader = yaml.SafeLoader


__all__ = [
    "check_progress",
    "clean_failed_build",
    "clean_failed_builds",
    "list_optimization_configs",
    "get_failed_optimizations",
    "get_build_state",
    "get_checkpoint_file",
    "estimate_workload"
]



def get_checkpoint_file(build_dir: str | Path) -> Path | None:
    """Return checkpoint file found in a build directory.

    Only files ending with the .checkpoint extension are considered.

    Raises:
        ValueError: If more than one checkpoint file exists in a single build directory.
    """
    build_dir = Path(build_dir)
    if not build_dir.is_dir():
        return None

    checkpoints = sorted(
        [f for f in build_dir.glob("*.checkpoint") if f.is_file()],
        key=lambda p: p.name,
    )

    if len(checkpoints) > 1:
        names = ", ".join(c.name for c in checkpoints)
        raise ValueError(
            f"Expected at most 1 checkpoint in '{build_dir}', found {len(checkpoints)}: {names}"
        )

    if len(checkpoints) == 0:
        return None

    return checkpoints[0]


def get_build_state(build_dir: str | Path) -> str:
    """Classify optimization build directory state.

    States:
        - "missing": build directory does not exist.
        - "running": build is actively being processed (has a .running sentinel file).
        - "completed": build has non-empty 3_LibraryLogic directory.
        - "resumable": build has checkpoint files but is not completed.
        - "failed": build exists without outputs or checkpoint files.
    """
    build_dir = Path(build_dir)
    if not build_dir.is_dir():
        return "missing"

    if (build_dir / ".running").is_file():
        return "running"

    lib_dir = build_dir / "3_LibraryLogic"
    if lib_dir.is_dir() and len(list(lib_dir.iterdir())) > 0:
        return "completed"

    checkpoint = get_checkpoint_file(build_dir)
    if checkpoint is not None:
        # Resumable only if the checkpoint is the sole remaining content.
        if all(p == checkpoint for p in build_dir.iterdir()):
            return "resumable"

    return "failed"


def list_optimization_configs(tuning_dir: str | Path) -> List[str]:
    """Get all tuning configuration YAML files from a directory.

    Args:
        tuning_dir (str | Path): Directory containing tuning configuration files.

    Returns:
        List[str]: List of paths to YAML config files, excluding files with 'config' in name.
    """

    def extract_number(file: Path) -> int:
        """Extract the numeric part from the filename before the extension."""
        match = re.search(r"_(\d+)\.yaml$", file.name)
        return int(match.group(1)) if match else -1  # Default -1 if no number found

    tuning_dir = Path(tuning_dir)
    configs = []
    for f in sorted(tuning_dir.glob("*.yaml"), key=extract_number):
        if "config" in f.name.lower():
            continue
        configs.append(str(f))
    return configs


def get_failed_optimizations(tuning_dir: str | Path) -> List[str]:
    """Get list of optimization config names that failed to complete.

    Args:
        tuning_dir (str | Path): Directory containing tuning configs and build outputs.

    Returns:
        List[str]: List of config names that either have empty/missing 3_LibraryLogic
            output directory.
    """
    tuning_dir = Path(tuning_dir)
    failed = []
    for f in list_optimization_configs(tuning_dir):
        config_name = Path(f).stem
        build_dir = tuning_dir / f"build_{config_name}"
        if get_build_state(build_dir) == "failed":
            failed.append(config_name)

    return failed


def clean_failed_build(build_dir: str | Path) -> None:
    """Remove stale artifacts from a single failed or resumable build directory.

    If a checkpoint is present, it is preserved so the build can resume.
    Completed and missing build directories are left unchanged.

    Args:
        build_dir (str | Path): Build directory to clean.
    """
    build_dir = Path(build_dir)
    state = get_build_state(build_dir)

    if state == "running":
        # Stale .running marker left by a previously crashed run; remove it and re-evaluate.
        (build_dir / ".running").unlink(missing_ok=True)
        state = get_build_state(build_dir)

    if state in ("missing", "completed", "resumable"):
        return

    if state != "failed":
        raise ValueError(f"Unsupported build state '{state}' for '{build_dir}'")

    checkpoint = get_checkpoint_file(build_dir)
    if checkpoint is not None:
        for item in list(build_dir.iterdir()):
            if item == checkpoint:
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        return

    shutil.rmtree(build_dir)


def clean_failed_builds(tuning_dir: str | Path) -> None:
    """Remove build directories for failed optimizations.

    Args:
        tuning_dir (str | Path): Directory containing tuning configs and build outputs.

    Note:
        Removes build directories that exist but have no valid library output
        in the 3_LibraryLogic subdirectory.
    """
    tuning_dir = Path(tuning_dir)
    for f in list_optimization_configs(tuning_dir):
        config_name = Path(f).stem
        build_dir = tuning_dir / f"build_{config_name}"
        clean_failed_build(build_dir)


def check_progress(tuning_dir: str | Path) -> Tuple[int, int, int]:
    """Check optimization progress for all configs in a tuning directory.

    Args:
        tuning_dir (str | Path): Directory containing tuning configs and build outputs.

    Returns:
        Tuple[int, int, int]: A tuple containing:
            - total_configs: Total number of YAML config files found.
            - completed_configs: Number with successful library output.
            - failed_configs: Number that failed (have tensilelite logs but no library).
    """
    tuning_dir = Path(tuning_dir)
    n_completed, n_failed = 0, 0
    configs = list_optimization_configs(tuning_dir)
    for f in configs:
        config_name = Path(f).stem
        build_dir = tuning_dir / f"build_{config_name}"
        state = get_build_state(build_dir)
        if state == "completed":
            n_completed += 1
        elif state == "failed":
            n_failed += 1

    return len(configs), n_completed, n_failed


def estimate_workload(conf_fl: str | Path, pop_size: int = 512) -> float:
    """Estimate relative optimization workload for a Tensile config YAML.

    The estimate is used only for scheduling priority (larger means heavier),
    not as an exact runtime prediction.

    Workload is computed as:
        (sum over problem sizes of 2*m*n*k*b / 1e9)
        * (EnqueuesPerSync + NumWarmups)
        * adjusted_pop_size

    where `adjusted_pop_size` is adjusted based on parameter with the largest
    space size, usually the number of MatrixInstuctions.

    Args:
        conf_fl: Path to the job YAML config file.
        pop_size: Baseline population size used in the estimate.

    Returns:
        A positive workload score used to rank jobs.
    
    Raises:
        ValueError: If input YAML format is not correct.
    """
    with open(conf_fl, "r") as f:
        conf = yaml.load(f, Loader=SafeLoader)
    
    # Superficial YAML structure validation
    required_keys = ["GlobalParameters", "BenchmarkProblems"]
    if not all(k in conf for k in required_keys):
        raise ValueError(f"Missing required keys: {required_keys}")

    if not isinstance(conf["BenchmarkProblems"], list) or len(conf["BenchmarkProblems"]) == 0:
        raise ValueError("BenchmarkProblems must be non-empty list")

    bfp = conf["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"]

    sizes = [ps["Exact"] for el in bfp if "ProblemSizes" in el for ps in el["ProblemSizes"]]
    if len(sizes) == 0:
        raise ValueError("No 'ProblemSizes' found")
    
    gflops = sum(2 * math.prod(s) for s in sizes) / 1e9

    iters = conf["GlobalParameters"].get("EnqueuesPerSync", 1) + conf["GlobalParameters"].get("NumWarmups", 0)

    fork_params = conf["BenchmarkProblems"][0][1]["ForkParameters"]

    groups = [g for g in fork_params if "Groups" in g]
    if len(groups) == 0:
        raise ValueError("No 'Groups' found")
    
    max_space_sz = max(len(g) for g in groups[0]["Groups"])
    max_space_sz = max(max_space_sz, *(len(g) for g in fork_params if "Groups" not in g))
    
    if max_space_sz > pop_size:
        pop_size = max_space_sz * 1.05  # account for decay
    else:
        # This is not duplicated code, we reduce pop_size at most twice
        pop_size = pop_size // 2 if max_space_sz < pop_size / 5 else pop_size
        pop_size = pop_size // 2 if max_space_sz < pop_size / 5 else pop_size
   
    return gflops * iters * pop_size
