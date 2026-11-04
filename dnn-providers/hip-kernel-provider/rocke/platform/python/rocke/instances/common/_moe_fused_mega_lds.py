# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Whole-kernel LDS accounting shared by the fused-MoE mega-kernel family.

Each mega (``moe_fused_mega`` f16/bf16, ``moe_fused_mega_fp8``, and the gfx1250
``fused_moe_mega_wmma``) validates its gate/up and its down GEMM as two
INDEPENDENT :class:`UniversalGemmSpec` s. Neither sub-validation sees
``Hidden_smem`` -- the persistent LDS bridge that is the whole point of the
fusion -- nor the fact that both GEMMs' operand buffers are allocated in the
builder prologue and therefore all coexist. A mega whose two halves each fit the
per-WG budget can still blow it as a whole; without this module that only
surfaces as a kernel-load failure on the device instead of a spec rejection.

Why a plain sum is EXACT here, not conservative
-----------------------------------------------
``core/lower_llvm.py`` packs every ``smem_alloc`` into one ``@smem_pool`` global
with a liveness-driven linear scan, so an allocation *may* be placed on top of a
dead one -- the cshuffle epilogue in ``gemm_universal`` relies on exactly that.
It cannot happen for the megas: a live interval opens at the ``tile.smem_alloc``
op, and every mega buffer is allocated in the prologue, before any of them is
first used. All pairs therefore interfere by construction and the packer gives
each a disjoint range, so the down GEMM's ``Bd_smem`` does **not** alias the
by-then-dead ``Bg_smem`` / ``Bu_smem``. Summing the aligned segments reproduces
``_smem_pool_size`` byte for byte; each mega's ``tests/instances/
test_moe_fused_mega*.py`` asserts that equality against the lowered IR so the
accounting and the packer cannot drift.

Allocations the emitter declares but never references are dead-stripped by the
packer and must be left out of the sequence passed here (the fp8 mega's
``BStage_smem`` under ``use_dtla=False`` is the live example).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple


__all__ = [
    "LdsAlloc",
    "lds_elem_bytes",
    "mega_lds_pool_bytes",
    "validate_mega_lds_budget",
]


# Mirror of ``lower_llvm._Lowerer._compute_smem_layout``'s ``_elem_bytes`` map
# (and its width-2 fallback) so the accounting cannot disagree with the packer.
_ELEM_BYTES = {
    "i8": 1,
    "fp8e4m3": 1,
    "bf8e5m2": 1,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "f32": 4,
    "i64": 8,
}


class LdsAlloc(NamedTuple):
    """One ``smem_alloc`` as the LDS budget sees it.

    ``name`` is the emitter's ``name_hint`` (it names the buffer in the
    rejection message); ``elem_bytes`` and ``elem_count`` give the segment size.
    """

    name: str
    elem_bytes: int
    elem_count: int

    @property
    def nbytes(self) -> int:
        return self.elem_bytes * self.elem_count


def lds_elem_bytes(dtype) -> int:
    """Bytes per element of an IR scalar type, as the smem packer measures it."""
    return _ELEM_BYTES.get(getattr(dtype, "name", dtype), 2)


def _seg_align(elem_bytes: int) -> int:
    """Segment alignment the packer applies: 16 B for byte-element types, else 4."""
    return 16 if elem_bytes == 1 else 4


def mega_lds_pool_bytes(allocs: Sequence[LdsAlloc]) -> int:
    """Bytes the ``@smem_pool`` global occupies for ``allocs``.

    Replays the packer's placement for the all-interfering mega case: each
    segment starts at the next multiple of its alignment past the previous one,
    and the pool is rounded up to 16 B. ``allocs`` must be in emitter
    declaration order.
    """
    end = 0
    for alloc in allocs:
        aln = _seg_align(alloc.elem_bytes)
        end = ((end + aln - 1) & ~(aln - 1)) + alloc.nbytes
    return (end + 15) & ~15


def validate_mega_lds_budget(allocs: Sequence[LdsAlloc], arch: str) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a mega's total LDS against ``arch``'s budget."""
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    total = mega_lds_pool_bytes(allocs)
    if target.fits_lds(total):
        return True, "ok"
    breakdown = ", ".join(f"{a.name}={a.nbytes}" for a in allocs)
    return False, (
        f"LDS budget {total} > {target.lds_capacity_bytes} cap "
        f"({breakdown}) on {arch}"
    )
