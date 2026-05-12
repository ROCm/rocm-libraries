# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU architecture detection for arch-specific PMC counter sets.

Used by :mod:`rocprof_pmc` to pick the right counter list for the live
device. The fallback chain is torch -> rocminfo -> ``"fallback"``: torch
is preferred because the dnn-benchmarking process already imports it for
GPU timing, so the lookup is cheap; rocminfo is the universal probe for
hosts where torch is unavailable; ``"fallback"`` returns a sparse safe
counter set rather than raising so the orchestrator can still produce
*something* for the user.
"""

import re
import shutil
import subprocess
from typing import Optional

from ._diagnostic import warn_once

_GFX_PATTERN = re.compile(r"\b(gfx[0-9a-f]+)\b", re.IGNORECASE)


def _detect_via_torch() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        # gcnArchName is a ROCm-specific attribute on torch.cuda device
        # properties; on CUDA builds it doesn't exist.
        arch = getattr(props, "gcnArchName", None)
        if not arch:
            return None
        m = _GFX_PATTERN.search(arch)
        return m.group(1).lower() if m else None
    except Exception as e:
        warn_once("arch", f"torch arch lookup failed: {e}")
        return None


def _detect_via_rocminfo() -> Optional[str]:
    binary = shutil.which("rocminfo")
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("arch", f"rocminfo invocation failed: {e}")
        return None
    if proc.returncode != 0:
        return None
    # rocminfo lists GPU agents with a "Name:" line that contains the
    # gfx target; CPUs in the same output have non-gfx names.
    for line in proc.stdout.splitlines():
        m = _GFX_PATTERN.search(line)
        if m:
            return m.group(1).lower()
    return None


def detect_arch() -> str:
    """Return the gfx target of the live GPU.

    Returns ``"fallback"`` when no GPU is detectable. Callers should
    look up the returned key in their per-arch table; the fallback key
    must always exist in those tables.
    """
    arch = _detect_via_torch()
    if arch is not None:
        return arch
    arch = _detect_via_rocminfo()
    if arch is not None:
        return arch
    return "fallback"
