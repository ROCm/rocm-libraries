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
