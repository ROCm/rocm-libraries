# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-arch capability checks for the attention backward kernels.

Every backward module asks this file, never the raw catalog, whether a lever is
legal on a target. The facts come from :class:`rocke.core.arch.ArchTarget`
(wave size, LDS capacity, MMA atoms, register file split, ``has_ds_read_tr``,
``has_async_lds``) plus a few facts the catalog does not carry yet and that are
recorded here as explicit tables:

* the LDS-DMA (``buffer_load ... lds``) widths in dwords per lane: one dword on
  gfx942, one, three or four dwords on gfx950, none on RDNA;
* the transpose-read width: only the 64-bit ``ds_read_tr16_b64`` is used by the
  backward (gfx950 only); the 128-bit form is never legal here;
* native no-return fp32 global atomic add (``global_atomic_add_f32`` without a
  compare-and-swap loop) per arch.

Every check raises :class:`ValueError` *before* any IR is emitted, so a caller
that validates a geometry first never leaves a half-built kernel behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

from rocke.core.arch import ArchTarget
from rocke.core.arch.target import MmaOp

__all__ = [
    "BWD_ARCHES",
    "BWD_BARRIER_FENCE",
    "BwdArchCaps",
    "bwd_arch_caps",
    "bwd_atoms",
    "check_agpr_alloc",
    "mfma_asm_mnemonic",
    "AGPR_ASM_TAIL_NOP",
    "ASM_MFMA_VALU_LEAD_WAIT_STATES",
    "asm_tail_nop_min",
    "check_asm_tail_nop",
    "mfma_result_wait_states",
    "check_lds_bytes",
    "check_lds_dma_width",
    "check_transpose_read",
    "check_waves",
    "emit_lds_dma",
    "select_bwd_atom",
    "wide_k_atom_available",
]

# Arches the backward knows about (gfx1201 is compile-only for the families).
BWD_ARCHES = ("gfx942", "gfx950", "gfx1151", "gfx1201")

# Fence of every barrier of the backward kernels (``IRBuilder.sync``, set as
# the builder's ``fence_barriers``): a compiler memory barrier, so no LDS
# access moves across a barrier that orders it against other waves (an
# unfenced barrier let machine-level sinking move stage-0 fragment loads past
# the barrier that retires them before the loop overwrites the bytes). The
# workgroup fences ("workgroup") add nothing to that and cost cache waits /
# invalidations on the work-group-processor targets.
BWD_BARRIER_FENCE = "compiler"

# LDS-DMA widths in dwords per lane (``raw_ptr_buffer_load_lds``). The catalog has
# only ``has_async_lds``; the width is a per-arch fact kept here until the core
# catalog grows a field for it.
_LDS_DMA_DWORDS = {
    "gfx942": (1,),
    "gfx950": (1, 3, 4),
    "gfx1151": (),
    "gfx1201": (),
}

# Native no-return fp32 global atomic add (no compare-and-swap expansion).
_NATIVE_FP32_ATOMIC_ADD = {
    "gfx942": True,
    "gfx950": True,
    "gfx1151": True,
    "gfx1201": True,
}

_WAVES = {"mfma": (1, 2, 4, 8), "wmma": (1, 2, 4)}

_DTYPES = {"fp16": "fp16", "f16": "fp16", "bf16": "bf16"}


def _norm_dtype(dtype: str) -> str:
    try:
        return _DTYPES[dtype]
    except KeyError:
        raise ValueError(f"backward operand dtype must be fp16 or bf16, got {dtype!r}")


@dataclass(frozen=True)
class BwdArchCaps:
    """Resolved backward-relevant facts for one gfx target."""

    arch: str
    wave_size: int
    matrix_path: str  # "mfma" | "wmma"
    lds_capacity_bytes: int
    lds_dma_dwords: tuple[int, ...]  # legal LDS-DMA widths; () = no LDS DMA
    has_tr_read_b64: bool  # ds_read_tr16_b64 usable
    has_tr_read_b128: bool  # always False for the backward
    has_agprs: bool
    arch_vgprs: int  # per-lane arch-VGPR budget
    agprs: int  # per-lane AGPR budget (0 on RDNA)
    native_fp32_atomic_add: bool
    legal_waves: tuple[int, ...]

    @property
    def max_lds_dma_dwords(self) -> int:
        return max(self.lds_dma_dwords, default=0)


@cache
def bwd_arch_caps(arch: str) -> BwdArchCaps:
    """Return the backward capability record of ``arch`` (a gfx id)."""
    if arch not in BWD_ARCHES:
        raise ValueError(
            f"attention backward does not target {arch!r}; known: {BWD_ARCHES}"
        )
    t = ArchTarget.from_gfx(arch)
    path = "mfma" if t.has_mfma else "wmma"
    agprs = int(t.limits.agprs)
    dma = _LDS_DMA_DWORDS[arch] if t.memory.has_async_lds else ()
    return BwdArchCaps(
        arch=arch,
        wave_size=int(t.wave_size),
        matrix_path=path,
        lds_capacity_bytes=int(t.lds_capacity_bytes),
        lds_dma_dwords=dma,
        has_tr_read_b64=bool(t.memory.has_ds_read_tr),
        has_tr_read_b128=False,
        has_agprs=agprs > 0,
        arch_vgprs=int(t.limits.vgprs) - agprs,
        agprs=agprs,
        native_fp32_atomic_add=_NATIVE_FP32_ATOMIC_ADD[arch],
        legal_waves=_WAVES[path],
    )


@cache
def bwd_atoms(arch: str, dtype: str) -> tuple[MmaOp, ...]:
    """The 16x16 MMA atoms the backward may use on ``arch``, by increasing K.

    32x32 atoms are excluded on purpose: ``kM0 = 16`` cannot fill them and the
    register C-to-A relabel is proven only for the 16x16 atoms.
    """
    caps = bwd_arch_caps(arch)
    dt = _norm_dtype(dtype)
    family = "mma" if caps.matrix_path == "mfma" else "wmma"
    ops = ArchTarget.from_gfx(arch).mma.enumerate(
        family=family, a_dtype=dt, b_dtype=dt, c_dtype="fp32", m=16, n=16
    )
    out = sorted(
        (op for op in ops if op.wave_size == caps.wave_size and op.a_frag_len > 0),
        key=lambda op: op.k,
    )
    if not out:
        raise ValueError(f"no 16x16 {dt} MMA atom on {arch}")
    return tuple(out)


def select_bwd_atom(arch: str, dtype: str, k: int = 16) -> MmaOp:
    """The 16x16xk atom for ``arch`` and ``dtype``; raises if the arch lacks it.

    ``k = 16`` exists everywhere; ``k = 32`` is native only on gfx950. A gfx942
    K=32 step is issued as two 16x16x16 instructions by the caller instead.
    """
    for op in bwd_atoms(arch, dtype):
        if op.k == k:
            return op
    raise ValueError(
        f"no native 16x16x{k} {_norm_dtype(dtype)} atom on {arch} "
        f"(available K: {[op.k for op in bwd_atoms(arch, dtype)]})"
    )


def wide_k_atom_available(arch: str, dtype: str) -> bool:
    """True when the native 16x16x32 atom exists (gfx950)."""
    return any(op.k == 32 for op in bwd_atoms(arch, dtype))


def check_waves(arch: str, waves: int) -> None:
    caps = bwd_arch_caps(arch)
    if waves not in caps.legal_waves:
        raise ValueError(
            f"waves={waves} is not legal on {arch}; legal: {caps.legal_waves}"
        )


def check_lds_bytes(arch: str, nbytes: int) -> None:
    caps = bwd_arch_caps(arch)
    if nbytes < 0 or nbytes > caps.lds_capacity_bytes:
        raise ValueError(
            f"LDS footprint {nbytes} B exceeds the {arch} capacity "
            f"{caps.lds_capacity_bytes} B"
        )


def check_lds_dma_width(arch: str, dwords: int) -> None:
    """Raise unless an LDS-DMA copy of ``dwords`` per lane is legal on ``arch``."""
    caps = bwd_arch_caps(arch)
    if not caps.lds_dma_dwords:
        raise ValueError(f"{arch} has no global-to-LDS DMA path")
    if dwords not in caps.lds_dma_dwords:
        raise ValueError(
            f"LDS-DMA width of {dwords} dword(s) is not legal on {arch}; "
            f"legal widths: {caps.lds_dma_dwords}"
        )


def check_transpose_read(arch: str, bits: int) -> None:
    """Raise unless an LDS transpose read of ``bits`` per lane is legal.

    Only the 64-bit form (``ds_read_tr16_b64``) on gfx950 is legal. The 128-bit
    form is refused on every arch: the backward never routes a transposed
    operand through it.
    """
    caps = bwd_arch_caps(arch)
    if bits == 128:
        raise ValueError(
            "ds_read_tr16_b128 is not used by the attention backward (only the "
            "64-bit transpose read is legal)"
        )
    if bits != 64:
        raise ValueError(f"unsupported transpose-read width {bits}")
    if not caps.has_tr_read_b64:
        raise ValueError(f"{arch} has no ds_read_tr16_b64")


_AGPR_ALLOC = (None, "backend", (0, 0), (128, 128), (192, 192), (256, 256))

# MFMA mnemonics of the inline-asm G1 / G3 route, per (atom K, dtype); the
# 16x16x32 atom exists on gfx950 only (see the arch atom table).
_MFMA_ASM = {
    (16, "fp16"): "v_mfma_f32_16x16x16_f16",
    (16, "bf16"): "v_mfma_f32_16x16x16_bf16",
    (32, "fp16"): "v_mfma_f32_16x16x32_f16",
    (32, "bf16"): "v_mfma_f32_16x16x32_bf16",
}

# s_nop after each inline-asm MFMA cluster, per atom K. An asm MFMA is opaque
# to the backend's hazard recognizer, so the cluster itself must cover its
# result latency (the next cluster's srcC, the epilogue's AGPR read) and the
# read window of its A / B sources (a later write of those VGPRs). The legal
# range is asm_tail_nop_min..15 (the ISA wait-state table, below); the largest
# s_nop stays the default until a timed sweep picks a value in that range.
AGPR_ASM_TAIL_NOP = MappingProxyType({16: 15, 32: 15})

# Passes of the XDL MFMA atoms of the inline-asm route (the backend's own
# scheduling model): every 16x16 f16 / bf16 atom on gfx942 and gfx950 is a
# 4-pass XDL op.
_XDL_PASSES = MappingProxyType({16: 4, 32: 4})

# Wait states the ISA requires after a VALU write of a VGPR before an MFMA
# reads it (any source); an inline-asm MFMA hides this from the backend too,
# so every cluster of the asm route opens with them (``s_nop 1``).
ASM_MFMA_VALU_LEAD_WAIT_STATES = 2


def mfma_result_wait_states(arch: str, atom_k: int) -> int:
    """Wait states the ISA requires between an XDL MFMA of the asm route and
    a dependent read of its result that is not the next MFMA's full-overlap
    srcC: a VALU / VMEM / LDS / export read or a VALU write of the result, an
    MFMA reading it as srcA / srcB, or a partially overlapping srcC. The
    largest of those is the VALU / memory read, ``passes + 3`` on gfx942 and
    ``passes + 4`` on gfx950 (2-pass atoms excepted; the route has none). A
    full-overlap srcC chain needs none, and the srcC write-after-read window
    (3 for a 4-pass op) is shorter.
    """
    caps = bwd_arch_caps(arch)
    if not caps.has_agprs or arch not in ("gfx942", "gfx950"):
        raise ValueError(f"no asm MFMA hazard table for {arch}")
    passes = _XDL_PASSES[atom_k]
    return passes + 3 + (arch == "gfx950" and passes != 2)


def asm_tail_nop_min(arch: str, dtype: str, atom_k: int) -> int:
    """Smallest legal ``s_nop`` count after an inline-asm MFMA cluster.

    ``s_nop n`` gives ``n + 1`` wait states, and the instruction right after
    the cluster may read the last MFMA's result, so the floor is the ISA's
    result wait states minus one (gfx942: 6, gfx950: 7, both atoms). It does
    not depend on the cluster size (the last MFMA has no slack).
    """
    mfma_asm_mnemonic(arch, dtype, atom_k)
    return mfma_result_wait_states(arch, atom_k) - 1


def check_asm_tail_nop(arch: str, dtype: str, atom_k: int, nop: int) -> None:
    """Raise unless ``s_nop nop`` covers the asm cluster's result hazards."""
    lo = asm_tail_nop_min(arch, dtype, atom_k)
    if not isinstance(nop, int) or isinstance(nop, bool) or not lo <= nop <= 15:
        raise ValueError(
            f"asm tail s_nop {nop!r} is not legal on {arch} for the 16x16x{atom_k} "
            f"{dtype} atom; legal: {lo}..15 (ISA MFMA result wait states)"
        )


def mfma_asm_mnemonic(arch: str, dtype: str, atom_k: int) -> str:
    """The MFMA mnemonic the inline-asm route issues for ``(atom K, dtype)``.

    Raises unless the atom is legal on ``arch`` (an MFMA target with that K).
    """
    caps = bwd_arch_caps(arch)
    if not caps.has_agprs:
        raise ValueError(f"{arch} has no AGPRs; the asm MFMA route is MFMA-only")
    if (atom_k, dtype) not in _MFMA_ASM:
        raise ValueError(f"no asm MFMA for atom K {atom_k} and {dtype}")
    if atom_k == 32 and arch != "gfx950":
        raise ValueError("the 16x16x32 MFMA atom is gfx950-only")
    return _MFMA_ASM[(atom_k, dtype)]


def check_agpr_alloc(arch: str, agpr_alloc: tuple[int, int] | str | None) -> None:
    """Raise unless ``agpr_alloc`` is one of the legal knob values on ``arch``."""
    caps = bwd_arch_caps(arch)
    if agpr_alloc is None:
        return
    if agpr_alloc == "backend":
        if not caps.has_agprs:
            raise ValueError(f"{arch} has no AGPRs; agpr_alloc must be None")
        return
    if not caps.has_agprs:
        raise ValueError(f"{arch} has no AGPRs; agpr_alloc must be None")
    if agpr_alloc not in _AGPR_ALLOC:
        raise ValueError(
            f"agpr_alloc={agpr_alloc!r} is not a legal value; legal: {_AGPR_ALLOC}"
        )
    if agpr_alloc[1] > caps.agprs:
        raise ValueError(f"agpr_alloc={agpr_alloc!r} exceeds {caps.agprs} AGPRs")


def emit_lds_dma(b, *, arch, rsrc, lds_ptr, voffset, soffset, dwords, coherency=0):
    """``async_buffer_load_lds`` behind the per-arch width check.

    The check runs first, so an illegal request raises without emitting any op.
    """
    check_lds_dma_width(arch, dwords)
    b.async_buffer_load_lds(rsrc, lds_ptr, voffset, soffset, dwords, coherency)
