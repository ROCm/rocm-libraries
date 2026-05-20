# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocprof-compute roofline collection.

Wraps the workload in ``rocprof-compute profile --roof-only --`` to
produce a roofline plot. The PDF is the user-facing artifact;
``extra_metrics["roofline"]`` records file paths only (no parsing of
the underlying SQLite).

Datatype selection is intentionally absent here: in current
rocprof-compute (and upstream rocm-systems develop) the
``--roofline-data-type`` flag exists only under
``rocprof-compute analyze``, not ``profile``. The profile run captures
the HBM/compute ceilings using rocprof-compute's default datatype
(FP32). Users who need FP16/BF16/etc. plots run::

    rocprof-compute analyze --path <workload_path> \\
        --roofline-data-type FP16

against the workload directory we record in
``extra_metrics["roofline"]["workload_path"]``.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._artifact_paths import profiling_subprocess_timeout_seconds
from ._diagnostic import warn_once
from ._tool_resolver import resolve_rocm_tool


def _build_argv(
    workload_dir: Path,
    inner_argv: List[str],
    rocprof_compute_binary: str,
) -> List[str]:
    return [
        rocprof_compute_binary,
        "profile",
        "--roof-only",
        "-n",
        workload_dir.name,
        "-p",
        str(workload_dir.parent),
        "--",
        *inner_argv,
    ]


def _find_named(search_dir: Path, name: str) -> Optional[Path]:
    """Return the first match for an exact filename anywhere under
    ``search_dir``, or ``None`` if absent."""
    candidates = sorted(search_dir.rglob(name))
    return candidates[0] if candidates else None


def run(
    inner_argv: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run rocprof-compute --roof-only and record the artefact paths.

    ``profile --roof-only`` emits CSV ceiling data (``roofline.csv``
    plus per-IP ``results_pmc_perf_<n>.csv``) and a sysinfo dump — no
    PDF and no SQLite. The PDF/HTML is rendered later by a separate
    ``rocprof-compute analyze --path <workload_dir> [--roofline-data-type
    DTYPE]`` run, which the user is expected to run themselves against
    the ``workload_path`` we record.
    """
    binary = resolve_rocm_tool("rocprof-compute")
    if binary is None:
        warn_once(
            "roofline",
            "rocprof-compute binary not found; skipping roofline",
        )
        return {"roofline": {"skipped": "rocprof-compute binary not found"}}

    out_dir.mkdir(parents=True, exist_ok=True)
    workload_dir = out_dir / "workload"
    argv = _build_argv(workload_dir, inner_argv, binary)

    timeout_s = profiling_subprocess_timeout_seconds() or None
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        warn_once(
            "roofline",
            f"rocprof-compute timed out after {timeout_s}s — roofline replay "
            "fires the workload ~3 times; raise DNN_BENCH_PROFILING_TIMEOUT_S "
            "for slow workloads",
        )
        return {
            "roofline": {"skipped": f"rocprof-compute timed out after {timeout_s}s"}
        }
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("roofline", f"rocprof-compute invocation failed: {e}")
        return {"roofline": {"skipped": f"rocprof-compute invocation failed: {e}"}}

    result: Dict[str, Any] = {}
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
        warn_once(
            "roofline",
            f"rocprof-compute exited {proc.returncode}; "
            "see extra_metrics['roofline']['error_tail']",
        )
        result["returncode"] = proc.returncode
        result["error_tail"] = tail
        return {"roofline": result}

    # roofline.csv carries the empirical HBM/compute ceilings — the
    # single most useful artifact and what users point `analyze` at.
    #
    # If the workload dir doesn't exist at all, rocprof-compute exited 0
    # without writing anything — usually a tool/version mismatch where
    # --roof-only is silently a no-op. Tighten the diagnostic so users
    # don't chase a phantom missing-CSV bug.
    if not workload_dir.exists():
        warn_once(
            "roofline",
            "rocprof-compute exited 0 but wrote no workload directory; "
            "the installed build may not support `profile --roof-only`",
        )
        result["warnings"] = [
            "rocprof-compute produced no workload directory "
            f"(expected at {workload_dir})"
        ]
        return {"roofline": result}
    roofline_csv = _find_named(out_dir, "roofline.csv")
    sysinfo_csv = _find_named(out_dir, "sysinfo.csv")
    if roofline_csv is None and sysinfo_csv is None:
        warn_once("roofline", "no roofline.csv or sysinfo.csv produced")
        result["warnings"] = ["no roofline.csv or sysinfo.csv produced"]
        return {"roofline": result}
    if roofline_csv is not None:
        result["roofline_csv"] = str(roofline_csv)
        # The workload directory is what `rocprof-compute analyze
        # --path ...` expects. Record it explicitly so the user
        # doesn't have to derive it.
        result["workload_path"] = str(roofline_csv.parent)
    if sysinfo_csv is not None:
        result["sysinfo_csv"] = str(sysinfo_csv)
    return {"roofline": result}
