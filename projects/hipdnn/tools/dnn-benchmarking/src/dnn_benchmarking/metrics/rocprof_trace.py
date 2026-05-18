# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocprofv3 kernel/memcpy trace export.

Wraps the workload in ``rocprofv3 --kernel-trace --memory-copy-trace
--output-format <fmt>`` and records the resulting artifact path.

Supports two formats:

* ``pftrace`` — single ``.pftrace`` file. Opens directly in
  https://ui.perfetto.dev. Default and always available.
* ``kineto`` — emits the rocpd db, then converts via
  ``python3 -m rocpd convert -i <db> --output-format chrome`` for
  PyTorch Kineto / Chrome-trace consumers. Falls back to pftrace when
  the rocpd Python module is not importable.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._diagnostic import warn_once


def _build_argv(
    fmt: str,
    out_dir: Path,
    inner_argv: List[str],
) -> List[str]:
    return [
        "rocprofv3",
        "--kernel-trace",
        "--memory-copy-trace",
        "--output-format",
        fmt,
        "-d",
        str(out_dir),
        "--",
        *inner_argv,
    ]


def _find_artifact(search_dir: Path, suffix: str) -> Optional[Path]:
    candidates = sorted(search_dir.rglob(f"*{suffix}"))
    return candidates[0] if candidates else None


def _convert_to_kineto(db_path: Path) -> Optional[Path]:
    """Convert a rocpd db to Chrome-trace format. Returns the .json path or None.

    Uses ``python -m rocpd convert``. If the rocpd module isn't
    importable in the *current* interpreter, returns None and the
    caller must fall back to pftrace. We deliberately probe importability
    rather than relying on the subprocess to error: the converter has
    historically returned 0 even when missing dependencies bailed out.
    """
    try:
        import rocpd  # noqa: F401
    except ImportError:
        return None
    out_path = db_path.with_suffix(".chrome.json")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "rocpd",
                "convert",
                "-i",
                str(db_path),
                "--output-format",
                "chrome",
                "-o",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("rocprof_trace", f"rocpd convert invocation failed: {e}")
        return None
    if proc.returncode != 0:
        warn_once(
            "rocprof_trace",
            f"rocpd convert exited {proc.returncode}: "
            f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}",
        )
        return None
    return out_path if out_path.exists() else None


def run(
    inner_argv: List[str],
    out_dir: Path,
    fmt: str,
) -> Dict[str, Any]:
    """Run rocprofv3 trace and return the extra_metrics slice.

    Args:
        inner_argv: argv for the ``--internal-profiling-run`` sub-mode.
        out_dir: Per-source output directory.
        fmt: ``pftrace`` or ``kineto``. ``kineto`` triggers a post-run
            ``python -m rocpd convert``.

    Never raises.
    """
    # rocprofv3 nests its own <hostname>/<pid>_results.* under -d. Pass
    # out_dir directly so the path doesn't double the hostname segment.
    out_dir.mkdir(parents=True, exist_ok=True)

    # rocprofv3 always emits at least pftrace when given pftrace, or
    # the rocpd db when given anything kineto-shaped. The simplest
    # cross-version invocation is to ask for pftrace directly when the
    # user wants pftrace, and ask for the db form when they want kineto.
    rocprof_fmt = "pftrace" if fmt == "pftrace" else "rocpd"
    argv = _build_argv(rocprof_fmt, out_dir, inner_argv)

    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("rocprof_trace", f"rocprofv3 invocation failed: {e}")
        return {
            "trace": {
                "format": fmt,
                "skipped": f"rocprofv3 invocation failed: {e}",
            }
        }

    result: Dict[str, Any] = {"format": fmt}
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
        warn_once(
            "rocprof_trace",
            f"rocprofv3 exited {proc.returncode}; see extra_metrics['trace']['error_tail']",
        )
        result["returncode"] = proc.returncode
        result["error_tail"] = tail
        return {"trace": result}

    if fmt == "pftrace":
        path = _find_artifact(out_dir, ".pftrace")
        if path is None:
            warn_once("rocprof_trace", "no .pftrace file produced")
            result["warnings"] = ["no .pftrace artifact found"]
        else:
            result["path"] = str(path)
        return {"trace": result}

    # kineto path: rocpd db is the always-available artifact; chrome
    # JSON is the value-add.
    db_path = _find_artifact(out_dir, ".db")
    if db_path is None:
        warn_once("rocprof_trace", "no rocpd .db file produced")
        result["warnings"] = ["no rocpd .db artifact found"]
        return {"trace": result}
    result["db_path"] = str(db_path)
    chrome_path = _convert_to_kineto(db_path)
    if chrome_path is None:
        # Fall back to pftrace: re-run rocprofv3 with pftrace format so
        # the user still gets a viewable artifact.
        result["kineto_unavailable"] = (
            "rocpd Python module not importable or convert failed"
        )
        warn_once(
            "rocprof_trace",
            "kineto conversion unavailable; falling back to pftrace",
        )
        fallback_argv = _build_argv("pftrace", out_dir, inner_argv)
        try:
            fb_proc = subprocess.run(
                fallback_argv, capture_output=True, text=True, check=False
            )
        except (OSError, subprocess.SubprocessError) as e:
            result["fallback_error"] = f"pftrace fallback invocation failed: {e}"
            return {"trace": result}
        if fb_proc.returncode == 0:
            pftrace_path = _find_artifact(out_dir, ".pftrace")
            if pftrace_path is not None:
                result["fallback_format"] = "pftrace"
                result["path"] = str(pftrace_path)
        else:
            result["fallback_error"] = f"pftrace fallback exited {fb_proc.returncode}"
        return {"trace": result}

    result["path"] = str(chrome_path)
    return {"trace": result}
