# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocprof-compute roofline collection.

Wraps the workload in ``rocprof-compute profile --roof-only --
roofline-data-type <dt> --`` to produce a roofline plot. The PDF is
the user-facing artifact; ``extra_metrics["roofline"]`` records
file paths only (no parsing of the underlying SQLite). Stack-style
multi-type plots are deferred — users can re-run rocprof-compute
against the recorded ``db_path`` if they want one.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._diagnostic import warn_once


def _build_argv(
    data_type: str,
    workload_dir: Path,
    inner_argv: List[str],
) -> List[str]:
    return [
        "rocprof-compute",
        "profile",
        "--roof-only",
        "--roofline-data-type",
        data_type,
        "-n",
        workload_dir.name,
        "-p",
        str(workload_dir.parent),
        "--",
        *inner_argv,
    ]


def _find(search_dir: Path, suffix: str) -> Optional[Path]:
    candidates = sorted(search_dir.rglob(f"*{suffix}"))
    return candidates[0] if candidates else None


def run(
    inner_argv: List[str],
    out_dir: Path,
    data_type: str,
) -> Dict[str, Any]:
    """Run rocprof-compute --roof-only and record the artefact paths."""
    binary = shutil.which("rocprof-compute")
    if binary is None:
        warn_once(
            "roofline",
            "rocprof-compute binary not found on PATH; skipping roofline",
        )
        return {
            "roofline": {
                "data_type": data_type,
                "skipped": "rocprof-compute binary not found on PATH",
            }
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    workload_dir = out_dir / "workload"
    argv = _build_argv(data_type, workload_dir, inner_argv)

    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("roofline", f"rocprof-compute invocation failed: {e}")
        return {
            "roofline": {
                "data_type": data_type,
                "skipped": f"rocprof-compute invocation failed: {e}",
            }
        }

    result: Dict[str, Any] = {"data_type": data_type}
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

    pdf = _find(out_dir, ".pdf")
    db = _find(out_dir, ".db")
    if pdf is None and db is None:
        warn_once("roofline", "no PDF or db produced by rocprof-compute")
        result["warnings"] = ["no PDF or db produced"]
        return {"roofline": result}
    if pdf is not None:
        result["pdf_path"] = str(pdf)
    if db is not None:
        result["db_path"] = str(db)
    return {"roofline": result}
