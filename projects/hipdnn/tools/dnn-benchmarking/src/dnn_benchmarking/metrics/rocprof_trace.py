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

from ._artifact_paths import (
    DEFAULT_PROFILING_TIMEOUT_S,
    find_first,
    flatten_hostname_dir,
)
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


def _rocpd_importable() -> bool:
    """True iff `import rocpd` succeeds in the current interpreter.

    The kineto branch needs the rocpd module both at run-time (for
    ``python -m rocpd convert``) and as a pre-condition for asking
    rocprofv3 to emit the rocpd db form. Probing upfront lets us pick
    pftrace before the first rocprofv3 invocation when rocpd is absent,
    avoiding a wasteful re-run of the entire workload.
    """
    try:
        import rocpd  # noqa: F401
    except ImportError:
        return False
    return True


def _convert_to_kineto(db_path: Path, timeout_s: int) -> Optional[Path]:
    """Convert a rocpd db to Chrome-trace format. Returns the .json path or None.

    Caller has already verified rocpd is importable via
    ``_rocpd_importable()``; this function only needs to handle the
    converter exiting non-zero (missing transitive deps, schema drift).
    """
    out_path = db_path.with_suffix(".chrome.json")
    subprocess_timeout = timeout_s or None
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
            timeout=subprocess_timeout,
        )
    except subprocess.TimeoutExpired:
        warn_once(
            "rocprof_trace", f"rocpd convert timed out after {subprocess_timeout}s"
        )
        return None
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
    timeout_s: int = DEFAULT_PROFILING_TIMEOUT_S,
) -> Dict[str, Any]:
    """Run rocprofv3 trace and return the extra_metrics slice.

    Args:
        inner_argv: argv for the ``--internal-profiling-run`` sub-mode.
        out_dir: Per-source output directory.
        fmt: ``pftrace`` or ``kineto``. ``kineto`` triggers a post-run
            ``python -m rocpd convert``.
        timeout_s: Per-subprocess wall-clock budget; ``0`` disables.

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
    #
    # If the user asked for kineto but rocpd isn't importable, the
    # conversion step would fail anyway. Decide upfront and run pftrace
    # directly instead of paying for the full workload re-run on the
    # fallback path.
    rocpd_present = _rocpd_importable()
    kineto_downgraded = fmt == "kineto" and not rocpd_present
    if kineto_downgraded:
        warn_once(
            "rocprof_trace",
            "rocpd Python module not importable; recording pftrace "
            "instead of kineto (one workload run, not two)",
        )
        rocprof_fmt = "pftrace"
    else:
        rocprof_fmt = "pftrace" if fmt == "pftrace" else "rocpd"
    argv = _build_argv(rocprof_fmt, out_dir, inner_argv, rocprofv3_binary)

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
            "rocprof_trace",
            f"rocprofv3 trace pass timed out after {subprocess_timeout}s",
        )
        return {
            "trace": {
                "format": fmt,
                "skipped": f"rocprofv3 trace pass timed out after {subprocess_timeout}s",
            }
        }
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
    if kineto_downgraded:
        # The user asked for kineto; we recorded pftrace. Tell them why
        # in the result so the format mismatch isn't silent.
        result["kineto_unavailable"] = "rocpd Python module not importable"
        result["recorded_format"] = "pftrace"
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
        warn_once(
            "rocprof_trace",
            f"rocprofv3 exited {proc.returncode}; see extra_metrics['trace']['error_tail']",
        )
        result["returncode"] = proc.returncode
        result["error_tail"] = tail
        return {"trace": result}

    if rocprof_fmt == "pftrace":
        path = find_first(out_dir, "*.pftrace")
        if path is None:
            warn_once("rocprof_trace", "no .pftrace file produced")
            result["warnings"] = ["no .pftrace artifact found"]
        else:
            result["path"] = str(path)
        return {"trace": result}

    # kineto path with rocpd present: db is the always-available
    # artifact; chrome JSON is the value-add. If conversion fails we
    # record the db path + kineto_unavailable note rather than re-running
    # the workload — the user already has the db and can convert
    # themselves with `python -m rocpd convert -i <db>`.
    db_path = find_first(out_dir, "*.db")
    if db_path is None:
        warn_once("rocprof_trace", "no rocpd .db file produced")
        result["warnings"] = ["no rocpd .db artifact found"]
        return {"trace": result}
    result["db_path"] = str(db_path)
    chrome_path = _convert_to_kineto(db_path, timeout_s=timeout_s)
    if chrome_path is None:
        result["kineto_unavailable"] = (
            "rocpd convert failed; rerun manually: "
            f"python -m rocpd convert -i {db_path} --output-format chrome"
        )
        return {"trace": result}

    result["path"] = str(chrome_path)
    return {"trace": result}
