# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""GPU device pre-flight check for the GEKO CLI.

Before --tune/--search/--bench start dispatching real work, we want to
confirm the host actually exposes at least as many GPU devices as were
requested via --devices. Rather than calling into HIP (hip-python's
hipInit/hipGetDeviceCount, which is blocking and not a geko dependency
today -- it's test-only scaffolding in tensilelite, see
Tests/unit/occupancy_hip_testutil.py), this shells out to the same
lightweight device-enumerator CLI tools tensilelite's own build pipeline
already relies on (Tensile/GpuArch.py, Toolchain/Validators.py):

- Linux: amdgpu-arch, falling back to rocm_agent_enumerator, then rocminfo.
- Windows: hipinfo.

Each probe is bounded by subprocess.run(timeout=...), so a hung/misbehaving
tool can never block the caller indefinitely.

This module is intentionally named generically (not e.g. "gpu_check.py")
so other device-related helpers can be added here later.
"""

import logging
import os
import re
import shutil
import subprocess
import sys

logger = logging.getLogger("GEKO")

_DEFAULT_TIMEOUT = 15.0

_GFX_LINE_RE = re.compile(r"gfx[0-9a-z]+(?:-[0-9a-z]+)*", re.IGNORECASE)
_PLACEHOLDER_ARCH = "gfx000"

_POSIX_ENUMERATORS = ("amdgpu-arch", "rocm_agent_enumerator", "rocminfo")
_WINDOWS_ENUMERATORS = ("hipinfo",)


def _candidate_enumerators() -> tuple[str, ...]:
    """Return the ordered device-enumerator tool names to try for this OS."""
    return _WINDOWS_ENUMERATORS if os.name == "nt" else _POSIX_ENUMERATORS


def _find_enumerator() -> str | None:
    """Return the first candidate enumerator tool found on PATH, else None."""
    for name in _candidate_enumerators():
        path = shutil.which(name)
        if path:
            return path
    return None


def _count_gpus(output: str) -> int:
    """Count GPU devices referenced in enumerator output.

    Works across amdgpu-arch/rocm_agent_enumerator (plain "gfxNNN" lines),
    rocminfo ("  Name:  gfx1250-strict"), and hipinfo
    ("gcnArchName:                gfx1100") by searching each line for a
    gfx-arch token. Excludes the "gfx000" placeholder amdgpu-arch reports
    on a GPU-less host.
    """
    count = 0
    for line in output.splitlines():
        match = _GFX_LINE_RE.search(line)
        if match and match.group(0).lower() != _PLACEHOLDER_ARCH:
            count += 1
    return count


def _probe_device_count(timeout: float = _DEFAULT_TIMEOUT) -> tuple[int | None, str | None]:
    """Run the best available enumerator tool and count GPU devices.

    Returns (count, error). error is "timeout" if the subprocess exceeded
    `timeout`, another short string if no tool was found or it failed, and
    None on success (count is then a valid int).
    """
    tool = _find_enumerator()
    if tool is None:
        return None, (
            "no GPU device-enumeration tool found on PATH "
            f"(tried: {', '.join(_candidate_enumerators())})"
        )
    try:
        proc = subprocess.run(
            [tool], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError as e:
        return None, f"failed to run '{tool}': {e}"
    if proc.returncode != 0:
        return None, f"'{tool}' exited with code {proc.returncode}"
    return _count_gpus(proc.stdout), None


def main(device_count: int) -> int:
    """Pre-flight GPU device check.

    Returns:
        0: enough devices found, or the check could not be performed (no
           enumerator tool available / it failed) -- permissive skip.
        1: the enumerator subprocess timed out.
        2: fewer devices were found than `device_count` requested.
    """
    count, error = _probe_device_count()
    if error == "timeout":
        logger.warning(
            f"GPU check: device enumeration timed out after {_DEFAULT_TIMEOUT}s; aborting."
        )
        return 1
    if error is not None or count is None:
        logger.warning(f"GPU check: {error}; skipping.")
        return 0
    if count < device_count:
        logger.warning(
            f"GPU check: found {count} GPU device(s), fewer than the {device_count} requested."
        )
        return 2
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m geko.device <device_count>", file=sys.stderr)
        sys.exit(2)
    try:
        _requested = int(sys.argv[1])
    except ValueError:
        print(f"device_count must be an integer, got: {sys.argv[1]!r}", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(_requested))
