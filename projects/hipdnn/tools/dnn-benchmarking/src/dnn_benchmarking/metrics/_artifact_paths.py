# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Profiling-artifact path utilities shared by the rocprofv3 sources.

Lives in its own module to avoid a circular import between
``profiling_orchestrator`` (which imports each source) and the source
modules (which need these helpers).
"""

import os
from pathlib import Path


def profiling_subprocess_timeout_seconds() -> int:
    """Wall-clock budget for a single profiling subprocess invocation.

    Every external profiler we shell out to (rocprofv3 PMC + trace,
    perf, rocprof-compute, rocpd convert) can wedge — a hung kernel
    under PMC, a perf paranoid race, a rocprof-compute multi-stage
    pipeline that silently stalls. Without a timeout the entire
    benchmark suite blocks on the wedged child indefinitely, which
    breaks the orchestrator's "nothing about profiling is fatal"
    contract.

    Default is 600 s (10 min) — long enough for a heavy graph under
    multi-pass PMC replay on a slow host, short enough that a stuck
    child doesn't burn a CI slot. Override via the
    ``DNN_BENCH_PROFILING_TIMEOUT_S`` env var for known-long workloads;
    set to ``0`` to disable the timeout entirely (not recommended).
    """
    raw = os.environ.get("DNN_BENCH_PROFILING_TIMEOUT_S", "600")
    try:
        value = int(raw)
    except ValueError:
        return 600
    return max(value, 0)


def _strip_rocprofv3_suffix(name: str) -> str:
    """Drop the ``_results`` infix rocprofv3 hardcodes into output filenames.

    rocprofv3 names every output file ``<-o value>_results.<ext>`` —
    the ``_results`` part is appended internally and there is no flag
    to suppress it. We pass ``-o results`` to drop the per-run ``<pid>_``
    prefix, which produces ``results_results.<ext>``. This helper
    collapses that to ``results.<ext>``.

    Returns the input unchanged if it doesn't match the pattern.
    """
    if "_results." not in name:
        return name
    stem, _, ext = name.rpartition(".")
    # stem ends with "_results"? then drop it.
    if stem.endswith("_results"):
        return f"{stem[: -len('_results')]}.{ext}"
    return name


def flatten_hostname_dir(out_dir: Path) -> None:
    """Flatten rocprofv3's output layout into a single directory.

    rocprofv3's output path is ``<-d>/[<hostname>/]<basename>_results.<ext>``:
      * The ``<hostname>/`` segment appears with the default ``-o``
        and disappears when an explicit ``-o`` value is supplied.
      * The ``_results.`` filename infix is hardcoded and there is no
        flag to suppress it.

    Both are dead weight in the artifact path for single-host runs.
    This helper:
      * Moves files out of any first-level subdir up to ``out_dir``
        (the hostname-segment case), stripping the ``_results`` infix
        from the filename in transit (``results_results.db`` →
        ``results.db``).
      * Also strips the ``_results`` infix from any files that are
        already directly under ``out_dir`` (the explicit-``-o`` case).
      * Removes emptied hostname dirs.

    No-op when ``out_dir`` doesn't exist. Non-empty hostname dirs are
    left alone (the rmdir is best-effort) so unexpected nested
    artifacts are never silently dropped.
    """
    if not out_dir.exists():
        return
    # First, top-level files (the explicit-`-o` case where rocprofv3
    # writes <out_dir>/<basename>_results.<ext> with no hostname dir).
    for f in list(out_dir.iterdir()):
        if not f.is_file():
            continue
        new_name = _strip_rocprofv3_suffix(f.name)
        if new_name != f.name:
            f.replace(out_dir / new_name)
    # Then, hostname subdirs (the default-`-o` case).
    for sub in list(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in list(sub.iterdir()):
            if f.is_file():
                target = out_dir / _strip_rocprofv3_suffix(f.name)
                f.replace(target)
        try:
            sub.rmdir()
        except OSError:
            # Non-empty (nested dirs we didn't move) — leave it alone
            # rather than risk losing data.
            pass
