# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocprof-compute roofline collection.

Wraps the workload in ``rocprof-compute profile --roof-only --`` to
capture HBM/compute ceilings. The profile pass emits CSVs
(``roofline.csv``, ``sysinfo.csv``, per-IP results CSVs) and the
workload directory; no PDF or HTML is produced at profile time.
``extra_metrics["roofline"]`` records those file paths.

Rendering the actual roofline (ASCII / web GUI / TUI) is a post-hoc
``rocprof-compute analyze --path <workload>`` invocation against the
recorded ``workload_path``. See ``docs/troubleshooting.md`` for the
separate-venv setup ``analyze`` needs (its deps would downgrade torch's
numpy if installed into the dnn-benchmarking venv).

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
from typing import Any, Dict, List

from ._artifact_paths import DEFAULT_PROFILING_TIMEOUT_S, find_first
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


def run(
    inner_argv: List[str],
    out_dir: Path,
    timeout_s: int = DEFAULT_PROFILING_TIMEOUT_S,
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

    subprocess_timeout = timeout_s or None
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=subprocess_timeout,
        )
    except subprocess.TimeoutExpired:
        warn_once(
            "roofline",
            f"rocprof-compute timed out after {subprocess_timeout}s — roofline "
            "replay fires the workload ~3 times; raise --profiling-timeout "
            "for slow workloads",
        )
        return {
            "roofline": {
                "skipped": f"rocprof-compute timed out after {subprocess_timeout}s"
            }
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
    # Layout varies across rocprof-compute versions:
    #
    #   * Older builds honoured `-n workload -p out_dir` and put output
    #     under `out_dir/workload/<gpu>/`.
    #   * rocprof-compute 3.6+ writes directly to `-p` (so `out_dir/`)
    #     and ignores `-n`.
    #
    # `find_first` rglobs from `out_dir` so it handles both shapes.
    roofline_csv = find_first(out_dir, "roofline.csv")
    sysinfo_csv = find_first(out_dir, "sysinfo.csv")
    # If neither named artefact is anywhere under out_dir, distinguish
    # "tool ran but produced no recognised output" (some CSVs present)
    # from "tool produced nothing at all" (likely a version mismatch
    # where --roof-only is silently a no-op).
    if roofline_csv is None and sysinfo_csv is None:
        any_csv = next(out_dir.rglob("*.csv"), None)
        if any_csv is None:
            warn_once(
                "roofline",
                "rocprof-compute exited 0 but produced no CSV output; "
                "the installed build may not support `profile --roof-only`",
            )
            result["warnings"] = [
                "rocprof-compute produced no CSV output "
                "(tool/build may not support --roof-only)"
            ]
        else:
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
