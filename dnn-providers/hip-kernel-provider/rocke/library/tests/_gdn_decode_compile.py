# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared GDN/KDA decode test helper: kernel -> HSACO -> code-object resources.

Compiles a kernel through comgr (no GPU) and reads its resource metadata
(scratch bytes and the like) with ``llvm-objdump`` / ``llvm-readelf``. It
raises ``unittest.SkipTest`` (pytest reports that as a skip too) when the
toolchain is missing or cannot read the code object. A compile error is a
defect and propagates.

Tool lookup is ``rocke.analysis.isa.analyze_hsaco``'s default: the
``llvm-objdump`` / ``llvm-readelf`` found on ``PATH``, nothing else. Put a
ROCm ``llvm/bin`` first on ``PATH`` to run the checks. A tool that is missing,
or a distro llvm-objdump that cannot read an AMDGPU code object, skips with
the tool's own message; so does a code object whose scratch size cannot be
parsed. A host whose ``PATH`` has no usable tool runs no spill check.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ARCH = "gfx950"


def compiled_resources(kernel, arch: str = ARCH):
    """Compile ``kernel`` through comgr and return its ``ResourceInfo``."""
    try:
        from rocke.analysis.isa import analyze_hsaco
        from rocke.helpers.compile import compile_kernel
    except Exception as e:  # pragma: no cover - env-dependent
        raise unittest.SkipTest(f"comgr toolchain unavailable: {e}")
    try:
        art = compile_kernel(kernel, arch=arch, capture_ir_text=False)
    except ImportError as e:  # pragma: no cover - env-dependent
        raise unittest.SkipTest(f"comgr toolchain unavailable: {e}")
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as fh:
        fh.write(bytes(art.hsaco))
        fh.flush()
        try:
            return analyze_hsaco(Path(fh.name)).resources
        except (FileNotFoundError, RuntimeError) as e:
            raise unittest.SkipTest(f"HSACO introspection unavailable: {e}")


def compiled_scratch_bytes(kernel, arch: str = ARCH) -> int:
    """Compile ``kernel`` and return the scratch bytes it spills to."""
    scratch = compiled_resources(kernel, arch).scratch_bytes
    if scratch is None:  # pragma: no cover - metadata shape drift
        raise unittest.SkipTest("could not parse the scratch size from the HSACO")
    return scratch
