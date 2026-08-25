################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
GPU architecture detection for TensileLite unit and common tests.
"""

import os
import re
import subprocess


def get_available_archs() -> list[str]:
    """Get list of available GPU architectures via rocm_agent_enumerator.

    Environment variable priority:
        1. TENSILE_ROCM_PATH (test-specific override)
        2. ROCM_PATH (standard ROCm variable)
        3. /opt/rocm (default)

    Returns:
        List of unique gfx architecture strings (e.g. ["gfx950"]).
        Returns empty list if rocm_agent_enumerator is not found or fails.
    """
    rocmpath = os.environ.get(
        "TENSILE_ROCM_PATH", os.environ.get("ROCM_PATH", "/opt/rocm")
    )
    enumerator = os.path.join(rocmpath, "bin", "rocm_agent_enumerator")
    if not os.path.exists(enumerator):
        return []
    try:
        output = subprocess.check_output([enumerator, "-t", "GPU"])
        return list(dict.fromkeys(
            line.strip()
            for line in output.decode().splitlines()
            if line.strip() and "gfx000" not in line
        ))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def has_arch(target: str) -> bool:
    """Check if a specific GPU architecture is available."""
    return any(target in arch for arch in get_available_archs())


# Filename arch tokens: match gfx1250v0 before gfx\\d+ so *gfx1250v0.yaml is
# not captured as gfx1250.
_ARCH_IN_FILENAME = re.compile(r"(gfx1250v0|gfx\d+)")


def filename_arch_token(filename: str):
    """Architecture token encoded in a YAML filename, or None."""
    if not filename:
        return None
    match = _ARCH_IN_FILENAME.search(filename)
    return match.group(1) if match else None


def resolve_skip_archs(compile_archs, enumerated_archs=None, revision_target=None):
    """Pytest skip/xfail matching set. Compile targets are not skip identity.

    gfx1250v0 expands (gfx1250v0 -> {gfx1250, gfx1250v0}) so
    skip-gfx1250 still matches; --gpu-targets gfx1250v0 must never *replace*
    gfx1250 in this set. When a real gfx1250 GPU is present, asicRevision is
    always probed even if --gpu-targets says gfx1250. GpuRevisionTarget is
    applied only when the enumerator listed gfx1250. Probe failure is
    fail-open (skip set stays {gfx1250} so skip-gfx1250v0 does not fire).

    enumerated_archs / revision_target are injectable for tests.
    """
    from Tensile import GpuRevisionTarget as gpu_rev

    skip = set()
    for arch in compile_archs or ():
        skip |= gpu_rev.expand_revision_skip_archs(arch)

    if enumerated_archs is None:
        enumerated_archs = get_available_archs()

    if gpu_rev.enumerator_reports_gfx1250(enumerated_archs):
        if revision_target is None:
            revision_target = gpu_rev.detect_gpu_revision_target()
        skip |= gpu_rev.skip_archs_for_gfx1250_revision_target(revision_target)

    return sorted(skip)
