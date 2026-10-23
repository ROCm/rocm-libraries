# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction-scheduling helpers for the tiling kernels' steady-state loop.

The AMDGPU ``sched_group_barrier`` intrinsic pins ``count`` instructions of a class at a point so
the compiler interleaves the K-loop's LDS reads, MFMAs, LDS writes, and global prefetch
deterministically. Two things were copy-pasted across the demo kernels: the class-mask hex, and the
per-class instruction-count math -- and that hand-typed count math is exactly what silently
misschedules when the tile config changes. ``InstrClass`` gives the masks one home; ``derive_sched_group_counts``
gives the count math one home.

These are generic instruction classes (DS / VMEM / VALU / MFMA), not MMA-specific, so they live at
the tiling package root (like ``WarpDistributionEncoding``), not under ``mma/``.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["InstrClass", "derive_sched_group_counts"]


class InstrClass(IntEnum):
    """AMDGPU ``sched_group_barrier`` instruction-class mask bits.

    The member value IS the hardware mask, so a member passes straight to
    ``b.sched_group_barrier(mask, count, group)`` -- no ``.mask`` indirection. This is the single home
    for the token -> hex map that was duplicated as ``_SGB_*`` constants across the demos.
    """

    VALU = 0x002
    MFMA = 0x008
    VMEM_READ = 0x020
    VMEM_WRITE = 0x040
    DS_READ = 0x100
    DS_WRITE = 0x200


def derive_sched_group_counts(plan, *, tile_m: int, tile_n: int, n_waves: int, vw: int):
    """Per-class ``sched_group_barrier`` counts for one steady-state trip, per wave.

    The SINGLE source for the count math that was duplicated verbatim across the demo kernels (the
    silent-misschedule-on-config-change bug). The MFMA count and the WARP (LDS-read) tile come from
    the driver's ``plan``; the caller passes the MACRO tile it cooperatively loads (``tile_m``,
    ``tile_n``), the cooperative wave count ``n_waves``, and the per-op vector width ``vw``.

    The divisor is asymmetric by design -- and this is exactly what byte-identity to the hand-math
    depends on:

    * ``DS_READ`` -- the per-wave LDS read of the WARP tile: ``warp_volume / wave_size / vw``.
    * ``VMEM_READ`` / ``DS_WRITE`` -- the COOPERATIVE macro load/store shared across the wave grid:
      ``macro_volume / (wave_size * n_waves) / vw``. ``DS_WRITE`` equals ``VMEM_READ`` -- each
      cooperatively loaded element is stored to LDS exactly once.

    ``vw`` is REQUIRED, never defaulted: it is the kernel's chosen wide-mem access width (elements per
    op -- 8 for the demos' f16 dwordx4, 4 for a b64 load), a per-kernel decision and NOT a property of
    the dtype, so there is no safe default to guess.

    Generality boundary (this is the trimmed dedup helper, not a general costing model): A and B are
    lumped into one volume under a single ``vw``, so it assumes A and B share the access width and the
    macro LDS-store volume equals the macro load volume -- true for the demos' symmetric f16 operands.
    Asymmetric-width operands must split the volumes themselves.

    Returns a ``dict`` keyed by :class:`InstrClass`. ``VMEM_WRITE`` is not part of a GEMM trip and is
    omitted; author it explicitly if a kernel needs it.
    """
    warp_m, warp_n, tile_k = plan.shape
    ws = plan.wave_size
    macro = tile_m * tile_k + tile_n * tile_k
    coop = ws * n_waves
    return {
        InstrClass.MFMA: plan.mfma_count,
        InstrClass.DS_READ: (warp_m * tile_k + warp_n * tile_k) // ws // vw,
        InstrClass.VMEM_READ: macro // coop // vw,
        InstrClass.DS_WRITE: macro // coop // vw,
    }
