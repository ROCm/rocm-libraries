################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# SPDX-License-Identifier: MIT
################################################################################
"""
CUOccupancy measurement utilities: occupancy formula and custom-kernel ASM parsing.

This module is PRODUCTION code (part of the Tensile package). It must not
import hip-python or perform any GPU runtime queries. HIP-based occupancy
measurement now lives exclusively in the test layer:
    Tensile/Tests/unit/occupancy_hip_testutil.py

THE CODEGEN-TIME FIX (updateOccupancyFromScan)
----------------------------------------------
updateOccupancyFromScan() runs in kernelBody() AFTER rocIsaPass completes.  It
rescans the instruction body with regex to find the actual max VGPR/AGPR indices,
then:
  1. Calls mkb.setGprs(scanned_vgprs, scanned_agprs) — updating the kernel
     descriptor's totalVgprs = ceil(scanned/8)*8 + scanned_agprs, which becomes
     the .amdhsa_next_free_vgpr directive in the emitted .s file.
  2. Recomputes kernel["CUOccupancy"] from the same scanned values.

Because the assembler takes .amdhsa_next_free_vgpr literally (no compiler-side
VGPR reduction for -x assembler input), the assembled .o's vgpr_count equals
exactly what updateOccupancyFromScan wrote.  Codegen-time occupancy is therefore
correct without any additional build-time readback pass.

CUSTOM KERNELS (compute_occupancy_from_asm_source)
--------------------------------------------------
Custom kernels bypass the normal codegen path and therefore skip
checkResources / updateOccupancyFromScan.  compute_occupancy_from_asm_source()
parses the hand-written .s directives (.amdhsa_next_free_vgpr /
.amdhsa_next_free_sgpr / .amdhsa_group_segment_fixed_size) and calls
compute_occupancy_from_resources() to set CUOccupancy at build time.
Called from KernelWriterAssembly.getSourceFileString() for custom kernels.
"""

from math import ceil
from typing import Optional, Tuple


# ── Occupancy computation formula ─────────────────────────────────────────────

def compute_occupancy_from_resources(
    vgpr_count: int,
    lds_bytes: int,
    sgpr_count: int,
    num_threads: int,
    physical_vgpr: int,
    physical_sgpr: int,
    device_lds: int,
    max_waves_per_simd: int,
) -> int:
    """Compute CUOccupancy from kernel resource counts (no GPU needed).

    This mirrors the Tensile getOccupancy formula applied to actual resource
    counts (e.g. from .amdhsa_next_free_vgpr in a custom .s file, or from
    scanned values after updateOccupancyFromScan).

    For ArchAccUnifiedRegs: the unified physical_vgpr pool (512 on gfx950)
    is divided by vgpr_count (=.amdhsa_next_free_vgpr).

    Args:
        vgpr_count:        Unified VGPR count (from .amdhsa_next_free_vgpr or scan).
        lds_bytes:         Static LDS from group_segment_fixed_size.
        sgpr_count:        SGPR count.
        num_threads:       Block size (from NumThreads or .max_flat_workgroup_size).
        physical_vgpr:     Hardware physical VGPR pool per SIMD
                           (512 for ArchAccUnifiedRegs gfx9; 256 otherwise).
        physical_sgpr:     Hardware physical SGPR pool per SIMD (800 for gfx9).
        device_lds:        Device total LDS per CU (163840 B for gfx950).
        max_waves_per_simd: Hardware wave cap per SIMD (8 for ArchAccUnifiedRegs, 10 otherwise).

    Returns:
        Max active workgroups (blocks) per CU, identical to Tensile's CUOccupancy.
    """
    # Occupancy limits
    vgpr_occ    = physical_vgpr // max(vgpr_count, 1)
    sgpr_occ    = physical_sgpr // max(sgpr_count, 1) if sgpr_count > 0 else max_waves_per_simd

    lds_occ = device_lds // max(((lds_bytes + 255) // 256) * 256, 256) if lds_bytes > 0 else max_waves_per_simd

    # Wave-count limit from block size
    multiplier  = int(ceil(num_threads / 256.0))
    wave_occ    = max_waves_per_simd // multiplier

    return max(1, min(vgpr_occ, sgpr_occ, lds_occ, wave_occ, max_waves_per_simd))


# ── Build-time helpers ────────────────────────────────────────────────────────

def _arch_caps_for_kernel(kernel) -> Tuple[int, int, int, int]:
    """Return (physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd) for a kernel.

    Derived from the kernel's ISA tuple using the same logic as rocisa::getArchCaps.
    We avoid importing rocisa here to keep this module lightweight.
    """
    isa = tuple(kernel.get("ISA", (9, 0, 8)))

    # ArchAccUnifiedRegs: gfx90a (9,0,10), gfx942 (9,4,2), gfx950 (9,5,0)
    arch_acc_unified = isa in {(9, 0, 10), (9, 4, 2), (9, 5, 0)}

    if arch_acc_unified:
        physical_vgpr    = 512
        max_waves_per_simd = 8
    else:
        physical_vgpr    = 256
        max_waves_per_simd = 10

    physical_sgpr = 800   # gfx9 family

    # DeviceLDS: gfx950 has 160 KB; others have 64 KB
    if isa == (9, 5, 0):
        device_lds = 163840  # 160 * 1024
    elif isa in {(9, 0, 10), (9, 4, 2)}:
        device_lds = 65536   # 64 KB (gfx90a/gfx942)
    else:
        device_lds = 65536   # gfx908 and others

    return physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd


def compute_occupancy_from_asm_source(kernel, asm_source: str) -> Optional[int]:
    """Parse .amdhsa_* directives from a hand-written custom kernel .s and return CUOccupancy.

    Used by :func:`KernelWriterAssembly._getCustomKernelSource` to compute occupancy at
    build (codegen) time for custom kernels that bypass the normal ``checkResources`` path.

    For ArchAccUnifiedRegs ISAs (gfx90a/gfx942/gfx950) ``.amdhsa_next_free_vgpr`` is the
    combined unified VGPR total (regularVgprs + accVgprs) and is passed directly to
    :func:`compute_occupancy_from_resources`, which divides ``physical_vgpr`` by that
    total — matching the hardware allocation granularity.

    The regex is deliberately tolerant of inline comments after values, e.g.::

        .amdhsa_next_free_vgpr 256 // vgprs

    Args:
        kernel:     Kernel dict; must contain ``"ISA"`` (list/tuple) and ``"NumThreads"``.
        asm_source: Full text of the .s file (may contain multiple kernel blocks).

    Returns:
        Computed CUOccupancy (``>= 1``) or ``None`` if any required directive is missing
        or unparseable.  The caller must leave ``kernel["CUOccupancy"]`` unchanged on ``None``.
    """
    import re as _re

    def _parse(directive: str, text: str) -> Optional[int]:
        m = _re.search(r'\.' + directive + r'\s+(\d+)', text)
        return int(m.group(1)) if m else None

    vgpr_total = _parse('amdhsa_next_free_vgpr', asm_source)
    sgpr_total = _parse('amdhsa_next_free_sgpr', asm_source)
    lds_size   = _parse('amdhsa_group_segment_fixed_size', asm_source)

    if vgpr_total is None or sgpr_total is None or lds_size is None:
        return None

    num_threads = kernel.get('NumThreads', 256)
    phy_vgpr, phy_sgpr, device_lds, max_waves = _arch_caps_for_kernel(kernel)

    try:
        return compute_occupancy_from_resources(
            vgpr_count=vgpr_total,
            lds_bytes=lds_size,
            sgpr_count=sgpr_total,
            num_threads=num_threads,
            physical_vgpr=phy_vgpr,
            physical_sgpr=phy_sgpr,
            device_lds=device_lds,
            max_waves_per_simd=max_waves,
        )
    except Exception:
        return None


# ── Utility: gfx string from ISA tuple ────────────────────────────────────────

def isa_to_gfx(isa: tuple) -> str:
    """Convert an ISA tuple to gfx string (e.g. (9,5,0) -> 'gfx950')."""
    return "gfx" + "".join(str(x) for x in isa)
