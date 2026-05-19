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

from ._artifact_paths import flatten_hostname_dir
from ._diagnostic import warn_once
from ._tool_resolver import resolve_rocm_tool


def _build_argv(
    fmt: str,
    out_dir: Path,
    inner_argv: List[str],
    rocprofv3_binary: str,
) -> List[str]:
    # `-o results` strips rocprofv3's default `<pid>_` filename prefix;
    # ``flatten_hostname_dir`` then strips the `<hostname>/` segment
    # after the subprocess returns.
    return [
        rocprofv3_binary,
        "--kernel-trace",
        "--memory-copy-trace",
        "--output-format",
        fmt,
        "-d",
        str(out_dir),
        "-o",
        "results",
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
    rocprofv3_binary = resolve_rocm_tool("rocprofv3")
    if rocprofv3_binary is None:
        warn_once("rocprof_trace", "rocprofv3 binary not found; skipping trace pass")
        return {
            "trace": {
                "format": fmt,
                "skipped": "rocprofv3 binary not found",
            }
        }

    # rocprofv3 nests its own <hostname>/<pid>_results.* under -d. Pass
    # out_dir directly so the path doesn't double the hostname segment.
    out_dir.mkdir(parents=True, exist_ok=True)

    # rocprofv3 always emits at least pftrace when given pftrace, or
    # the rocpd db when given anything kineto-shaped. The simplest
    # cross-version invocation is to ask for pftrace directly when the
    # user wants pftrace, and ask for the db form when they want kineto.
    rocprof_fmt = "pftrace" if fmt == "pftrace" else "rocpd"
    argv = _build_argv(rocprof_fmt, out_dir, inner_argv, rocprofv3_binary)

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

    # Hoist results.<ext> out of <out_dir>/<hostname>/ to <out_dir>/.
    # Done on both success and error paths so anything rocprofv3
    # produced before exiting is reachable at a stable path.
    flatten_hostname_dir(out_dir)

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
        fallback_argv = _build_argv("pftrace", out_dir, inner_argv, rocprofv3_binary)
        try:
            fb_proc = subprocess.run(
                fallback_argv, capture_output=True, text=True, check=False
            )
        except (OSError, subprocess.SubprocessError) as e:
            result["fallback_error"] = f"pftrace fallback invocation failed: {e}"
            return {"trace": result}
        flatten_hostname_dir(out_dir)
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
