# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: find the LLVM tools (:func:`find_llvm_tool`) and build small
AMDGPU kernels with llc to read the compiler's own occupancy report and code
objects.

Each kernel uses an exact number of VGPRs (and AGPRs), so a test can compare an
occupancy model with what the compiler reports for the same register count.
An llc that does not know a processor gives ``None`` from
:func:`llc_occupancy` and a ``unittest.SkipTest`` from :func:`llc_code_object`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

__all__ = ["find_llvm_tool", "llc_code_object", "llc_occupancy"]


def find_llvm_tool(name: str) -> str | None:
    """The LLVM tool ``name`` (llc, llvm-readelf, llvm-objdump), or None when
    absent.

    ROCm's own LLVM is tried first (``$ROCM_PATH/llvm/bin``, ``/opt/rocm`` when
    unset), then ``PATH``: a distro LLVM may lack the AMDGPU target or predate
    the processor under test.
    """
    rocm_bin = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm" / "bin"
    return shutil.which(str(rocm_bin / name)) or shutil.which(name)


def _vgpr_kernel_ir(vgprs: int, wave_size: int = 32, agprs: int = 0) -> str:
    """An AMDGPU kernel that uses exactly ``vgprs`` VGPRs and ``agprs`` AGPRs
    (it clobbers the top register of each).

    The workgroup is capped at 64 lanes so the workgroup size never limits the
    occupancy the compiler reports; only the register count does. ``wave_size``
    only matters on RDNA: CDNA processors always build wave64 code.
    """
    features = ' "target-features"="+wavefrontsize64"' if wave_size == 64 else ""
    clobbers = f"~{{v{vgprs - 1}}}" + (f",~{{a{agprs - 1}}}" if agprs else "")
    return (
        "define amdgpu_kernel void @k() #0 {\n"
        f'  call void asm sideeffect "", "{clobbers}"()\n'
        "  ret void\n"
        "}\n"
        f'attributes #0 = {{ "amdgpu-flat-work-group-size"="1,64"{features} }}\n'
    )


def _run_llc(llc: str, processor: str, ir: str, *extra_args: str) -> bytes | None:
    """Run llc on ``ir``; None when this llc does not know ``processor``."""
    out = subprocess.run(
        [llc, "-mtriple=amdgcn-amd-amdhsa", f"-mcpu={processor}", *extra_args],
        input=ir.encode(),
        capture_output=True,
        timeout=120,
        check=True,
    )
    if b"not a recognized processor" in out.stderr:
        return None
    return out.stdout


def llc_occupancy(llc: str, processor: str, vgprs: int, agprs: int = 0) -> int | None:
    """The waves per SIMD llc reports for a kernel using ``vgprs`` VGPRs and
    ``agprs`` AGPRs."""
    asm = _run_llc(llc, processor, _vgpr_kernel_ir(vgprs, agprs=agprs), "-o", "-")
    if asm is None:
        return None
    return int(re.search(rb"; Occupancy: (\d+)", asm).group(1))


def llc_code_object(
    llc: str, processor: str, vgprs: int, wave_size: int = 32, agprs: int = 0
) -> bytes:
    """A real relocatable code object (ELF with AMDGPU notes) built by llc."""
    obj = _run_llc(
        llc,
        processor,
        _vgpr_kernel_ir(vgprs, wave_size, agprs),
        "-filetype=obj",
        "-o",
        "-",
    )
    if obj is None:
        raise unittest.SkipTest(f"this llc does not know {processor}")
    return obj
