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

``get_available_archs``: every arch on the machine, for common-config ``skip-gfxNNNN``.
``detect_gpu_target``: the device HIP gives this process, for the unit-test ``gpu`` mark.
"""

import contextlib
import functools
import os

from Tensile.GpuArch import cmake_gpu_target, detect_gpu_archs


@contextlib.contextmanager
def _rocmPathFromTestOverride():
    """``TENSILE_ROCM_PATH`` standing in for ``ROCM_PATH`` inside the block.

    ``Tensile.GpuArch`` reads ``ROCM_PATH``, the variable the build uses. The
    tests keep their own override so a run can probe a different install without
    moving the one the build points at, so it is applied here and taken back
    out again rather than leaking into anything the tests go on to launch.
    """
    override = os.environ.get("TENSILE_ROCM_PATH")
    if not override:
        yield
        return

    previous = os.environ.get("ROCM_PATH")
    os.environ["ROCM_PATH"] = override
    try:
        yield
    finally:
        if previous is None:
            del os.environ["ROCM_PATH"]
        else:
            os.environ["ROCM_PATH"] = previous


def get_available_archs() -> list[str]:
    """Get list of available GPU architectures, one entry per distinct name.

    Reads through ``Tensile.GpuArch``, the same detection the build uses, so a
    stepping suffix survives. It has to: ``config_helpers.configMarks`` keys its
    ``skip-<arch>`` marks off this list, and the two spellings of a config carry
    mirrored marks -- a strict config says ``skip-gfx1250``, a base config says
    ``skip-gfx1250-strict``. Handing that comparison a truncated ``gfx1250`` for
    an agent named ``gfx1250-strict`` therefore does not narrow the selection,
    it reverses it: every base config runs against a strict client and every
    strict config is skipped. ``rocm_agent_enumerator``, which this used to
    call, truncates in exactly that way.

    The target features go, through the same ``cmake_gpu_target`` the build
    spells ``GPU_TARGETS`` with. A detection tool answers with a configuration,
    so amdgpu-arch names a gfx90a agent ``gfx90a:sramecc+:xnack-``, and a mark
    is written for the architecture rather than for one agent's features:
    ``skip-gfx90a`` never matches ``skip-gfx90a:sramecc+:xnack-``, so keeping
    them would unskip every config pinned off this architecture. Only the
    colon-delimited features come off; the hyphenated stepping stays.

    Environment variable priority:
        1. TENSILE_ROCM_PATH (test-specific override)
        2. ROCM_PATH (standard ROCm variable)
        3. /opt/rocm (default)

    Returns:
        List of unique gfx architecture strings (e.g. ["gfx1250-strict"]).
        Returns empty list if no detection tool is found or all of them fail.
    """
    with _rocmPathFromTestOverride():
        return list(dict.fromkeys(map(cmake_gpu_target, detect_gpu_archs())))


def has_arch(target: str) -> bool:
    """Check if a specific GPU architecture is available."""
    return any(target in arch for arch in get_available_archs())


# ---- Unit-test device (HIP; lazy, so `-m "not gpu"` never initialises it) ----

@functools.cache
def _hip():
    try:
        from hip import hip
    except ImportError:
        return None
    return hip


def hip_available() -> bool:
    """True when hip-python is importable."""
    return _hip() is not None


def _check(result):
    """Unwrap a hip-python ``(err, value)`` result; raise on a HIP error."""
    err, *rest = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"HIP error {int(err)}")
    return rest[0] if len(rest) == 1 else rest


@functools.cache
def detect_gpu_target() -> str | None:
    """Arch of HIP device 0 (e.g. ``gfx1250-strict``), or None.

    ``TENSILE_GPU_TARGET`` overrides. HIP, unlike rocm_agent_enumerator, honours
    ROCR_VISIBLE_DEVICES, sees only devices this process can open, and keeps
    the stepping.
    """
    override = os.environ.get("TENSILE_GPU_TARGET")
    if override:
        return override
    hip = _hip()
    if hip is None:
        return None
    try:
        if int(_check(hip.hipGetDeviceCount())) == 0:
            return None
        try:
            # hip-python 10 (a wrapper over rocm-bindings) returns the struct.
            props = _check(hip.hipGetDeviceProperties(0))
        except TypeError:
            # Earlier releases fill a caller-owned struct instead.
            props = hip.hipDeviceProp_t()
            _check(hip.hipGetDeviceProperties(props, 0))
    except RuntimeError:
        return None
    name = props.gcnArchName
    if isinstance(name, bytes):
        name = name.decode()
    # gcnArchName carries target features (gfx942:sramecc+:xnack-).
    return name.split(":")[0] or None


def arch_matches(found: str, wanted: str) -> bool:
    """``gfx1250`` matches ``gfx1250`` and ``gfx1250-strict``; ``gfx1250-strict`` only itself."""
    return found == wanted or found.startswith(wanted + "-")


def gpu_is(*archs: str) -> bool:
    """True if the GPU matches any of *archs*. For non-gate code; tests use ``gpu(...)``."""
    found = detect_gpu_target()
    return found is not None and any(arch_matches(found, a) for a in archs)
