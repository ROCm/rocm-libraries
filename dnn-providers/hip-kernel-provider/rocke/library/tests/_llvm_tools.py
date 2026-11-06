# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Test helper: locate the LLVM tools (``llvm-objdump``, ``llvm-readelf``) the
ISA tests run on a comgr code object.

ROCm's own build is tried first (``$ROCM_PATH/llvm/bin``, ``/opt/rocm`` when
unset): a distro LLVM on ``PATH`` may lack the AMDGPU target or predate the gfx
target under test. A host with neither gives ``None``, and the caller skips.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["find_llvm_tool"]


def find_llvm_tool(name: str) -> str | None:
    """``name`` from ROCm's LLVM, else from ``PATH``; None when absent."""
    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    return shutil.which(str(root / "llvm" / "bin" / name)) or shutil.which(name)
