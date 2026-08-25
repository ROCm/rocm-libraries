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


def gpu_targets_from_tensile_options(tensile_options):
    """Parse pytest ``--tensile-options=--gpu-targets,gfx1250v0`` (comma-split argv)."""
    if not tensile_options:
        return []
    from Tensile.GpuRevisionTarget import gpu_targets_from_argv
    return gpu_targets_from_argv(str(tensile_options).split(","))


def merge_pytest_compile_archs(gpu_targets=None, tensile_options=None):
    """Union pytest ``--gpu-targets`` with ``--tensile-options`` ``--gpu-targets``.

    Legacy tox passed gfx1250v0 only via tensile-options; that must still
    expand skip-gfx1250v0.
    """
    if gpu_targets:
        archs = [t.strip() for t in gpu_targets.split(";") if t.strip()]
    else:
        archs = list(get_available_archs())
    for extra in gpu_targets_from_tensile_options(tensile_options):
        if extra not in archs:
            archs.append(extra)
    return archs


def resolve_skip_archs(compile_archs, enumerated_archs=None, revision_target=None):
    """Pytest skip/xfail matching set. Compile targets are not skip identity.

    gfx1250v0 expands to {gfx1250, gfx1250v0} (never drop the family name).
    HIP asicRevision is applied only when the enumerator listed gfx1250.
    Probe failure is fail-open: skip set stays {gfx1250}.
    """
    from Tensile import GpuRevisionTarget as gpu_rev

    skip = set()
    for arch in compile_archs or ():
        skip |= gpu_rev.expand_revision_skip_archs(arch)

    if enumerated_archs is None:
        enumerated_archs = get_available_archs()

    if gpu_rev.enumerator_reports_gfx1250(enumerated_archs):
        if revision_target is None:
            revision_target = gpu_rev.gfx1250_revision_skip_target()
        token = gpu_rev.arch_skip_token(revision_target)
        if token == gpu_rev.GFX1250_V0:
            skip |= {gpu_rev.GFX1250_ARCH, gpu_rev.GFX1250_V0}
        else:
            skip |= {gpu_rev.GFX1250_ARCH}

    return sorted(skip)
