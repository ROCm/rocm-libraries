# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU architecture detection for Tensile --gpu-targets.

Reports the name the runtime reports, which is also the compiler target its
kernels must be built for. That matters for gfx1250, which ships as two
steppings sharing ISA (12,5,0) but built as separate targets: code objects for
gfx1250 are rejected on gfx1250-strict and vice versa.

Deliberately rocisa-free (no Tensile.Common import) so the invoke build path
never pulls in rocisa just to probe the arch, and it lives inside the packaged
Tensile tree so ROCm test artifacts can exercise it directly; tensilelite's
tasks.py only wraps it in an invoke @task entry point.
"""

import os
import re
import shutil
import subprocess
import sys

# An agent name line from rocminfo, e.g. "  Name:  gfx1250-strict".
_ROCMINFO_AGENT_RE = re.compile(r"^\s+Name:\s+(gfx\S+?)\s*$", re.MULTILINE)
# amdgpu-arch prints one target per line.
_ARCH_LINE_RE = re.compile(r"^(gfx\S+)$", re.MULTILINE)

# A machine with no GPU still enumerates this placeholder.
_PLACEHOLDER_ARCH = "gfx000"

# amdgpu-arch sits under llvm/bin in a stock ROCm install and under lib/llvm/bin
# in the Windows SDK and TheRock layouts.
_AMDGPU_ARCH_RELPATHS = (
    ("llvm", "bin", "amdgpu-arch"),
    ("lib", "llvm", "bin", "amdgpu-arch"),
)


def _rocm_root():
    return os.environ.get("ROCM_PATH", "/opt/rocm")


def _tool(*relative_parts):
    """An executable from the ROCm install, falling back to PATH."""
    candidate = os.path.join(_rocm_root(), *relative_parts)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which(relative_parts[-1])


def _first_tool(relpaths):
    """The first of several candidate locations that yields an executable."""
    return next((found for parts in relpaths if (found := _tool(*parts))), None)


def _run(command):
    """stdout of a successful run, or None on any failure. Never raises."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _first_real_arch(names):
    return next((n for n in names if n and n != _PLACEHOLDER_ARCH), None)


def detect_gpu_arch():
    """The architecture name the runtime reports for the first GPU, or None.

    Deliberately does not use rocm_agent_enumerator: it parses rocminfo with a
    capture group that ends at ``gfx\\d+``, so it truncates a suffix and answers
    "gfx1250" for an agent rocminfo names "gfx1250-strict" -- which would build
    the wrong stepping's kernels without a word. amdgpu-arch and rocminfo both
    report the full name.
    """
    amdgpu_arch = _first_tool(_AMDGPU_ARCH_RELPATHS)
    if amdgpu_arch:
        output = _run([amdgpu_arch])
        if output:
            arch = _first_real_arch(_ARCH_LINE_RE.findall(output))
            if arch:
                return arch

    rocminfo = _tool("bin", "rocminfo")
    if rocminfo:
        output = _run([rocminfo])
        if output:
            arch = _first_real_arch(_ROCMINFO_AGENT_RE.findall(output))
            if arch:
                return arch

    if not amdgpu_arch and not rocminfo:
        print(
            "Error: neither 'amdgpu-arch' nor 'rocminfo' found. Please install ROCm.",
            file=sys.stderr,
        )
        return None

    print(
        "Failed to detect a valid GPU architecture (gfx target not found).",
        file=sys.stderr,
    )
    return None
