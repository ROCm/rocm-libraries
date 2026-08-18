# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
CUOccupancy measurement utilities: occupancy formula and custom-kernel ASM parsing.

Production code — must not import hip-python or perform GPU runtime queries.
HIP-based occupancy measurement lives in the test layer:
    Tensile/Tests/unit/occupancy_hip_testutil.py
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

    Mirrors getVgprOccupancy in KernelWriterAssembly.py exactly:
    the workgroup multiplier scales the aligned VGPR footprint (not just the
    wave cap), so >256-thread kernels correctly yield lower VGPR occupancy.

    For ArchAccUnifiedRegs ISAs the unified physical_vgpr pool (512 on gfx950)
    is divided by vgpr_count (=.amdhsa_next_free_vgpr).

    Args:
        vgpr_count:         Unified VGPR count (from .amdhsa_next_free_vgpr or scan).
        lds_bytes:          Static LDS from group_segment_fixed_size.
        sgpr_count:         SGPR count.
        num_threads:        Block size (NumThreads or .max_flat_workgroup_size).
        physical_vgpr:      Hardware VGPR pool per SIMD (512 ArchAccUnifiedRegs; 256 otherwise).
        physical_sgpr:      Hardware SGPR pool per SIMD (800 for gfx9).
        device_lds:         Total LDS per CU (163840 B for gfx950).
        max_waves_per_simd: Hardware wave cap per SIMD (8 ArchAccUnifiedRegs; 10 otherwise).

    Returns:
        Max active workgroups per CU (Tensile's CUOccupancy).
    """
    # Workgroup multiplier: ceil(max(numThreads, 256) / 256) — matches getVgprOccupancy.
    # For >256-thread blocks each SIMD slot holds multiplier waves, so the aligned
    # VGPR footprint per slot scales up by the same factor (Fix #1).
    multiplier    = max(int(ceil(num_threads / 256.0)), 1)
    max_occ       = max_waves_per_simd // multiplier
    vgpr_align    = 4   # AMDGPU VGPR allocation granularity (dwords)
    vgprs_aligned = int(ceil(max(vgpr_count, 1) / vgpr_align)) * vgpr_align * multiplier
    vgpr_occ      = physical_vgpr // vgprs_aligned
    sgpr_occ      = physical_sgpr // max(sgpr_count, 1) if sgpr_count > 0 else max_waves_per_simd
    lds_occ       = device_lds // max(((lds_bytes + 255) // 256) * 256, 256) if lds_bytes > 0 else max_waves_per_simd

    return max(1, min(vgpr_occ, sgpr_occ, lds_occ, max_occ))


# ── Build-time helpers ────────────────────────────────────────────────────────

def _arch_caps_for_kernel(kernel) -> Optional[Tuple[int, int, int, int]]:
    """Return (physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd) for a kernel.

    Derived from the kernel's ISA tuple; mirrors rocisa/hardware_caps.hpp constants.
    KernelWriterAssembly passes live rocisa caps directly to
    compute_occupancy_from_asm_source() at runtime; this table is a fallback for
    standalone/test use where rocisa is not yet initialized for the target ISA.
    Must be kept in sync when new ISAs are added to rocisa/hardware_caps.hpp.

    Returns None for ISAs not explicitly listed (Fix #4) — callers must treat
    None as "cannot compute → skip CUOccupancy override" to avoid emitting
    incorrect caps for unsupported architectures.
    """
    isa = tuple(kernel.get("ISA", (9, 0, 8)))

    # ArchAccUnifiedRegs: gfx90a (9,0,10), gfx942 (9,4,2), gfx950 (9,5,0)
    # physical_vgpr = MaxVgpr(256) * 2 = 512 (matches getSourceFileString cap passthrough)
    if isa in {(9, 0, 10), (9, 4, 2), (9, 5, 0)}:
        physical_vgpr      = 512
        max_waves_per_simd = 8
        # DeviceLDS per hardware_caps.hpp: gfx950 = 160 KB; gfx90a/gfx942 = 64 KB
        device_lds = 163840 if isa == (9, 5, 0) else 65536
    elif isa[0] == 9:
        # Other gfx9 (e.g. gfx908/gfx906): non-unified, 64 KB LDS, 10 waves
        physical_vgpr      = 256
        max_waves_per_simd = 10
        device_lds         = 65536
    elif isa == (12, 5, 0):
        # gfx1250: MaxVgpr=1024, ArchAccUnifiedRegs=False, DeviceLDS=320 KB
        # (hardware_caps.hpp: MaxVgpr=1024, deviceLDS=327680, MaxWavesPerSimd=10)
        physical_vgpr      = 1024
        max_waves_per_simd = 10
        device_lds         = 327680  # 320 * 1024
    else:
        # Unknown or unsupported ISA: return None rather than silently emitting
        # garbage caps.  The occupancy formula is gfx9/wave64-specific; callers
        # should skip the CUOccupancy override for unrecognised architectures.
        return None

    physical_sgpr = 800   # gfx9 family and gfx1250 (hardware_caps.hpp PhysicalMaxSgpr)

    return physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd


def compute_occupancy_from_asm_source(
    kernel,
    asm_source: str,
    arch_caps: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[int]:
    """Parse .amdhsa_* directives from a hand-written custom kernel .s and return CUOccupancy.

    Used by KernelWriterAssembly.getSourceFileString for custom kernels that bypass the
    normal checkResources path.  For ArchAccUnifiedRegs ISAs .amdhsa_next_free_vgpr is
    the combined unified VGPR total (regularVgprs + accVgprs).

    The regex tolerates inline comments after values (e.g. ``.amdhsa_next_free_vgpr 256 // vgprs``).

    Args:
        kernel:     Kernel dict with ``"ISA"`` and ``"NumThreads"``.
        asm_source: Full text of the .s file.
        arch_caps:  Optional ``(physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd)``
                    from rocisa; when None, falls back to :func:`_arch_caps_for_kernel`.

    Returns:
        Computed CUOccupancy (>= 1) or None if any required directive is missing.
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
    if arch_caps is not None:
        phy_vgpr, phy_sgpr, device_lds, max_waves = arch_caps
    else:
        _caps = _arch_caps_for_kernel(kernel)
        if _caps is None:
            # Unsupported ISA: skip rather than emit wrong occupancy.
            return None
        phy_vgpr, phy_sgpr, device_lds, max_waves = _caps

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
