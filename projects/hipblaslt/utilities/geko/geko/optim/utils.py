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
    _gpu_targets_from_configs: Read ArchitectureName from tuning YAML configs.
"""

import shutil
import logging
import re
import math
import yaml

from pathlib import Path
from typing import List, Sequence, Tuple

logger = logging.getLogger("GEKO")


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
    "estimate_workload",
    "_gpu_targets_from_configs",
    "record_shape_failure",
    "shape_failure_count",
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
        # Resumable only if the checkpoint is the sole remaining content, except
        # for .failcount: clean_failed_build deliberately preserves it alongside
        # the checkpoint, so counting it as content would report every resumable
        # build as failed.
        if all(p == checkpoint or p.name == ".failcount" for p in build_dir.iterdir()):
            return "resumable"

    return "failed"


def shape_failure_count(build_dir: Path) -> int:
    """How many times this shape has been attempted and failed.

    Stored as a plain integer in ``<build_dir>/.failcount`` so it survives reboots
    and separate invocations, which is exactly when it matters.
    """
    try:
        return int((Path(build_dir) / ".failcount").read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def record_shape_failure(build_dir: Path) -> int:
    """Increment and return this shape's persistent failure count."""
    build_dir = Path(build_dir)
    count = shape_failure_count(build_dir) + 1
    try:
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / ".failcount").write_text(str(count))
    except OSError:
        pass
    return count


def list_optimization_configs(
    tuning_dir: str | Path, max_shape_failures: int = 0
) -> List[str]:
    """Get all tuning configuration YAML files from a directory.

    Args:
        tuning_dir (str | Path): Directory containing tuning configuration files.
        max_shape_failures (int): Leave out shapes whose ``.failcount`` has reached
            this; 0 keeps every shape.

    Returns:
        List[str]: Paths of the YAML configs to run, in file-number order. Files with
        'config' in the name and configs without MatrixInstruction groups are skipped.
    """

    def extract_number(file: Path) -> int:
        """Extract the numeric part from the filename before the extension."""
        match = re.search(r"_(\d+)\.yaml$", file.name)
        return int(match.group(1)) if match else -1  # Default -1 if no number found

    tuning_dir = Path(tuning_dir)
    configs = []
    skipped_no_mi = []
    retired = []
    for f in sorted(tuning_dir.glob("*.yaml"), key=extract_number):
        if "config" in f.name.lower():
            continue
        if not _has_matrix_instructions(f):
            # A config with no MatrixInstruction entries generates no kernels, so
            # running it occupies a GPU slot to produce nothing and trips the load
            # balancer ("Could not compute workload ... 'NoneType' object is not
            # iterable", which reads Groups[0]). Tiny shapes hit this: M=N=6 leaves
            # nothing after MI filtering, since every MI tile is at least 16x16.
            skipped_no_mi.append(f.name)
            continue

        if (
            max_shape_failures
            and shape_failure_count(tuning_dir / f"build_{f.stem}") >= max_shape_failures
        ):
            # Retire a shape that has already failed. Opt-in (default 0, off):
            # on a node that degrades over a run, re-attempting a shape that
            # already failed spends a limited healthy window for no result.
            #
            # To re-attempt a retired shape later, delete its .failcount file.
            retired.append(f.name)
            continue
        configs.append(str(f))

    if retired:
        logger.warning(
            f"Retiring {len(retired)} shape(s) that failed >= {max_shape_failures} "
            f"times: {', '.join(retired[:5])}" + (" ..." if len(retired) > 5 else "")
        )
    if skipped_no_mi:
        logger.warning(
            f"Skipping {len(skipped_no_mi)} config(s) with no MatrixInstruction "
            f"groups (cannot generate kernels): {', '.join(skipped_no_mi[:5])}"
            + (" ..." if len(skipped_no_mi) > 5 else "")
        )
    return configs


def _gpu_targets_from_configs(config_paths: Sequence[str | Path]) -> str | None:
    """Read ArchitectureName from the first tuning YAML, if present."""
    for cfg_path in config_paths:
        try:
            with Path(cfg_path).open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            arch = (data.get("LibraryLogic") or {}).get("ArchitectureName")
            if arch:
                return str(arch)
        except (OSError, yaml.YAMLError, TypeError, AttributeError):
            continue
    return None


def _has_matrix_instructions(path: Path) -> bool:
    """True when a tuning config declares at least one MatrixInstruction.

    Parsed textually rather than with yaml.safe_load: these configs are ~100KB
    each and this runs over every config at startup, so a substring scan keeps
    the check cheap. Any read error is reported as True so an unreadable file
    fails loudly downstream rather than being silently dropped here.
    """
    try:
        with open(path, "r") as handle:
            return "MatrixInstruction:" in handle.read()
    except OSError:
        return True


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

    # The failure counter must outlive the build dir it sits in, or shape
    # retirement can never trigger: every retry cleans the dir and resets the
    # count to zero.
    failcount = shape_failure_count(build_dir)

    checkpoint = get_checkpoint_file(build_dir)
    if checkpoint is not None:
        for item in list(build_dir.iterdir()):
            if item == checkpoint or item.name == ".failcount":
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        return

    shutil.rmtree(build_dir)
    if failcount:
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / ".failcount").write_text(str(failcount))


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
