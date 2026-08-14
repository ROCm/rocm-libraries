# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU architecture detection with gfx1250 v0/v1 ASIC-revision disambiguation.

This module lives inside the ``Tensile`` package (rather than in the
tensilelite-root ``tasks.py``) so it ships with the installed test artifacts
and can be imported without pulling in ``invoke`` — see
``Tensile/Tests/unit/test_GpuRevisionTarget.py``.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# The probe source lives at <tensilelite_root>/tools/gpu_revision_probe.cpp.
# This file is at <tensilelite_root>/Tensile/GpuRevisionTarget.py, so parents[1]
# is the tensilelite root.
_TENSILELITE_ROOT = pathlib.Path(__file__).resolve().parent.parent
_REVISION_PROBE_SRC = _TENSILELITE_ROOT / "tools" / "gpu_revision_probe.cpp"


def detect_gpu_arch():
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator", "-v"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        if result.returncode == 0:
            target = next(
                (line.strip() for line in result.stdout.splitlines()
                 if line.startswith("gfx") and line.strip() != "gfx000"),
                None,
            )
            if target:
                return target
    except FileNotFoundError:
        print(
            "Error: 'rocm_agent_enumerator' command not found. "
            "Please install ROCm.", file=sys.stderr,
        )
    except subprocess.TimeoutExpired:
        print(
            "Error: GPU detection timed out. Hardware might be unresponsive.",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"An unexpected error occurred during GPU detection: {e}",
            file=sys.stderr,
        )

    print(
        "Failed to detect a valid GPU architecture (gfx target not found).",
        file=sys.stderr,
    )
    return None


def _revision_to_gpu_target(base_arch, asic_revision):
    """Map a detected base arch + ASIC revision to a Tensile --gpu-targets value.

    Only gfx1250 revision 0 is the pre-production v0. Everything else -- the
    shipping v1 (revision 1), an unknown revision (-1 when HIP is too old to
    expose the field), any future/unexpected value, and every non-gfx1250 arch --
    is returned unchanged so tests default to the shipping stepping.
    """
    if base_arch == "gfx1250" and asic_revision == 0:
        return "gfx1250v0"
    return base_arch


def _probe_asic_revision(build_dir=None, device_id=0):
    """Compile (once, cached) and run the HIP revision probe.

    Returns a (arch, revision) tuple on success, or None on any failure (hipcc
    missing, build error, no device, non-zero exit, or unparsable output) so the
    caller can fall back to the shipping default. Never raises.
    """
    hipcc = shutil.which("hipcc")
    if not hipcc:
        print("warning: hipcc not found; cannot probe gfx1250 ASIC revision.",
              file=sys.stderr)
        return None

    out_dir = pathlib.Path(build_dir) if build_dir else _TENSILELITE_ROOT / "build_tmp"
    probe_bin = out_dir / "gpu_revision_probe"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stale = (not probe_bin.exists()
                 or (_REVISION_PROBE_SRC.exists()
                     and probe_bin.stat().st_mtime < _REVISION_PROBE_SRC.stat().st_mtime))
        if stale:
            # Compile to a unique temp file then atomically rename, so concurrent
            # callers (e.g. pytest-xdist workers) can never observe or run a
            # half-written binary (ETXTBSY / truncated exe).
            fd, tmp_bin = tempfile.mkstemp(
                dir=str(out_dir), prefix=".gpu_revision_probe.", suffix=".tmp")
            os.close(fd)
            try:
                subprocess.run(
                    [hipcc, "-O0", str(_REVISION_PROBE_SRC), "-o", tmp_bin],
                    check=True, capture_output=True, text=True, timeout=180,
                )
                os.chmod(tmp_bin, 0o755)
                os.replace(tmp_bin, str(probe_bin))
            finally:
                if os.path.exists(tmp_bin):
                    os.remove(tmp_bin)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"warning: failed to build ASIC revision probe: {e}",
              file=sys.stderr)
        return None

    try:
        result = subprocess.run(
            [str(probe_bin), str(device_id)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"warning: ASIC revision probe failed to run: {e}",
              file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"warning: ASIC revision probe exited {result.returncode}: "
              f"{result.stderr.strip()}", file=sys.stderr)
        return None

    lines = result.stdout.splitlines()  # arch on line 1, revision on line 2
    if len(lines) < 2:
        print(f"warning: unexpected ASIC revision probe output: {result.stdout!r}",
              file=sys.stderr)
        return None
    try:
        revision = int(lines[1].strip())
    except ValueError:
        print(f"warning: could not parse ASIC revision from: {result.stdout!r}",
              file=sys.stderr)
        return None
    return (lines[0].strip(), revision)


def detect_gpu_revision_target(build_dir=None, device_id=0):
    """Detect the Tensile --gpu-targets value, distinguishing gfx1250 v0 from v1.

    Non-gfx1250 arches are returned unchanged without probing. For gfx1250, the
    ASIC revision is probed via HIP: revision 0 -> gfx1250v0, otherwise (and on
    any probe failure or arch mismatch) -> gfx1250 (the shipping v1 default).
    """
    base_arch = detect_gpu_arch()
    if base_arch != "gfx1250":
        return base_arch

    probed = _probe_asic_revision(build_dir=build_dir, device_id=device_id)
    if probed is None:
        print("warning: could not determine gfx1250 ASIC revision; "
              "defaulting to gfx1250 (v1).", file=sys.stderr)
        return "gfx1250"

    probe_arch, revision = probed
    # gcnArchName carries feature suffixes on real hardware
    # (e.g. "gfx1250:sramecc+:xnack-"); compare only the base arch token.
    if probe_arch.split(":")[0] != "gfx1250":
        print(f"warning: revision probe reported arch '{probe_arch}' != detected "
              "'gfx1250'; defaulting to gfx1250 (v1).", file=sys.stderr)
        return "gfx1250"

    return _revision_to_gpu_target("gfx1250", revision)
