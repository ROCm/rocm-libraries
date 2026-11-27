# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-arch capability checks for the attention backward kernels.

Every backward module asks this file, never the raw catalog, whether a lever is
legal on a target. The hardware facts come from the attention capability table
shared with the forward kernels, :mod:`rocke.helpers.attention_arch_caps`
(LDS-DMA widths, the transpose-read width, the 16x16 MMA atoms, native fp32
global atomic add, and the inline-asm MFMA mnemonics and hazard wait states).
This module adds the backward's policy on top of it:

* the arches the backward targets (:data:`BWD_ARCHES`); every check refuses
  any other arch with the backward's own message;
* fp16 / bf16 operands only;
* the legal wave counts and ``agpr_alloc`` values;
* the 128-bit transpose read is never legal here, though it may exist;
* the barrier fence and the default asm tail ``s_nop``.

Every check raises :class:`ValueError` *before* any IR is emitted, so a caller
that validates a geometry first never leaves a half-built kernel behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

from rocke.core.arch.target import MmaOp
from rocke.helpers import attention_arch_caps as _caps
from rocke.helpers.attention_arch_caps import ASM_MFMA_VALU_LEAD_WAIT_STATES

__all__ = [
    "BWD_ARCHES",
    "BWD_BARRIER_FENCE",
    "BwdArchCaps",
    "bwd_arch_caps",
    "bwd_atoms",
    "atom_knob",
    "atom_knob_for",
    "atom_knob_k",
    "atom_knobs",
    "bwd_atom_knobs",
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

_WAVES = {"mfma": (1, 2, 4, 8), "wmma": (1, 2, 4)}

_DTYPES = {"fp16": "fp16", "f16": "fp16", "bf16": "bf16"}


def _norm_dtype(dtype: str) -> str:
    try:
        return _DTYPES[dtype]
    except KeyError:
        raise ValueError(f"backward operand dtype must be fp16 or bf16, got {dtype!r}")


def _gate(arch: str) -> None:
    if arch not in BWD_ARCHES:
        raise ValueError(
            f"attention backward does not target {arch!r}; known: {BWD_ARCHES}"
        )


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
    _gate(arch)
    c = _caps.attn_arch_caps(arch)
    return BwdArchCaps(
        arch=arch,
        wave_size=c.wave_size,
        matrix_path=c.matrix_path,
        lds_capacity_bytes=c.lds_capacity_bytes,
        lds_dma_dwords=c.lds_dma_dwords,
        has_tr_read_b64=c.has_tr_read_b64,
        has_tr_read_b128=False,
        has_agprs=c.has_agprs,
        arch_vgprs=c.arch_vgprs,
        agprs=c.agprs,
        native_fp32_atomic_add=c.native_fp32_atomic_add,
        legal_waves=_WAVES[c.matrix_path],
    )


@cache
def bwd_atoms(arch: str, dtype: str) -> tuple[MmaOp, ...]:
    """The 16x16 MMA atoms the backward may use on ``arch``, by increasing K.

    32x32 atoms are excluded on purpose: ``kM0 = 16`` cannot fill them and the
    register C-to-A relabel is proven only for the 16x16 atoms.
    """
    _gate(arch)
    return _caps.attn_atoms(arch, _norm_dtype(dtype))


# ---------------------------------------------------------------------------
# atom knobs: the spec spellings (atom_g02 / atom_g13 / atom_g4) of the
# catalog's backward atoms. A knob names a shape; the instruction comes from
# the catalog (select_bwd_atom with the knob's K).
# ---------------------------------------------------------------------------

# Operand dtypes the knobs cover (attention_bwd.BWD_DTYPES; repeated here to
# keep this module free of the kernel module).
_KNOB_DTYPES = ("fp16", "bf16")


def atom_knob(op: MmaOp) -> str:
    """The knob spelling of a catalog atom: ``"<m>x<n>x<k>"``, prefixed
    ``"wmma"`` for a WMMA atom."""
    prefix = "wmma" if op.family == "wmma" else ""
    return f"{prefix}{op.m}x{op.n}x{op.k}"


@cache
def _knob_ks(arch: str) -> dict[str, tuple[bool, int]]:
    """``{knob: (wmma, K)}`` of ``arch``'s backward atoms, by increasing K."""
    knobs: dict[str, tuple[bool, int]] = {}
    for dtype in _KNOB_DTYPES:
        for op in bwd_atoms(arch, dtype):
            knobs.setdefault(atom_knob(op), (op.family == "wmma", op.k))
    return dict(sorted(knobs.items(), key=lambda kv: kv[1][1]))


def bwd_atom_knobs(arch: str) -> tuple[str, ...]:
    """The atom knobs legal on ``arch``, by increasing K."""
    return tuple(_knob_ks(arch))


@cache
def _all_knob_ks() -> dict[str, tuple[bool, int]]:
    """``{knob: (wmma, K)}`` over every backward arch: MFMA by K, then WMMA."""
    knobs: dict[str, tuple[bool, int]] = {}
    for arch in BWD_ARCHES:
        knobs.update(_knob_ks(arch))
    return dict(sorted(knobs.items(), key=lambda kv: kv[1]))


def atom_knobs(*, mfma_only: bool = False) -> tuple[str, ...]:
    """Every atom knob of the backward (MFMA by K, then WMMA), or the MFMA ones."""
    return tuple(
        k for k, (wmma, _k) in _all_knob_ks().items() if not (mfma_only and wmma)
    )


def atom_knob_k(knob: str) -> int:
    """The K of an atom knob; ``ValueError`` for a spelling no backward atom has."""
    try:
        return _all_knob_ks()[knob][1]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown atom {knob!r}; known atoms: {atom_knobs()}"
        ) from None


def atom_knob_for(k: int, *, wmma: bool = False) -> str:
    """The MFMA (or WMMA) atom knob of K ``k``."""
    for knob, (is_wmma, kk) in _all_knob_ks().items():
        if is_wmma == wmma and kk == k:
            return knob
    kind = "WMMA" if wmma else "MFMA"
    raise ValueError(f"no {kind} atom knob with K {k}")


def select_bwd_atom(arch: str, dtype: str, k: int = 16) -> MmaOp:
    """The 16x16xk atom for ``arch`` and ``dtype``; raises if the arch lacks it.

    ``k = 16`` exists everywhere; ``k = 32`` is native only on gfx950. A gfx942
    K=32 step is issued as two 16x16x16 instructions by the caller instead.
    """
    _gate(arch)
    return _caps.select_attn_atom(arch, _norm_dtype(dtype), k)


def wide_k_atom_available(arch: str, dtype: str) -> bool:
    """True when the native 16x16x32 atom exists (gfx950)."""
    _gate(arch)
    return _caps.wide_k_atom_available(arch, _norm_dtype(dtype))


def check_waves(arch: str, waves: int) -> None:
    caps = bwd_arch_caps(arch)
    if waves not in caps.legal_waves:
        raise ValueError(
            f"waves={waves} is not legal on {arch}; legal: {caps.legal_waves}"
        )


def check_lds_bytes(arch: str, nbytes: int) -> None:
    _gate(arch)
    _caps.check_lds_bytes(arch, nbytes)


def check_lds_dma_width(arch: str, dwords: int) -> None:
    """Raise unless an LDS-DMA copy of ``dwords`` per lane is legal on ``arch``."""
    _gate(arch)
    _caps.check_lds_dma_width(arch, dwords)


def check_transpose_read(arch: str, bits: int) -> None:
    """Raise unless an LDS transpose read of ``bits`` per lane is legal.

    Only the 64-bit form (``ds_read_tr16_b64``) on gfx950 is legal. The 128-bit
    form is refused on every arch: the backward never routes a transposed
    operand through it.
    """
    _gate(arch)
    if bits == 128:
        raise ValueError(
            "ds_read_tr16_b128 is not used by the attention backward (only the "
            "64-bit transpose read is legal)"
        )
    _caps.check_transpose_read(arch, bits)


_AGPR_ALLOC = (None, "backend", (0, 0), (128, 128), (192, 192), (256, 256))

# s_nop after each inline-asm MFMA cluster, per atom K. An asm MFMA is opaque
# to the backend's hazard recognizer, so the cluster itself must cover its
# result latency (the next cluster's srcC, the epilogue's AGPR read) and the
# read window of its A / B sources (a later write of those VGPRs). The legal
# range is asm_tail_nop_min..ASM_TAIL_NOP_MAX (the ISA wait-state table, in
# rocke.helpers.attention_arch_caps); the largest s_nop stays the default until
# a timed sweep picks a value in that range.
AGPR_ASM_TAIL_NOP = MappingProxyType({16: 15, 32: 15})


def mfma_result_wait_states(arch: str, atom_k: int) -> int:
    """Wait states the ISA requires before a dependent read of an asm MFMA's
    result; see :func:`rocke.helpers.attention_arch_caps.mfma_result_wait_states`.
    """
    _gate(arch)
    return _caps.mfma_result_wait_states(arch, atom_k)


def asm_tail_nop_min(arch: str, dtype: str, atom_k: int) -> int:
    """Smallest legal ``s_nop`` count after an inline-asm MFMA cluster
    (gfx942: 6, gfx950: 7, both atoms); see
    :func:`rocke.helpers.attention_arch_caps.asm_tail_nop_min`.
    """
    _gate(arch)
    return _caps.asm_tail_nop_min(arch, dtype, atom_k)


def check_asm_tail_nop(arch: str, dtype: str, atom_k: int, nop: int) -> None:
    """Raise unless ``s_nop nop`` covers the asm cluster's result hazards."""
    _gate(arch)
    _caps.check_asm_tail_nop(arch, dtype, atom_k, nop)


def mfma_asm_mnemonic(arch: str, dtype: str, atom_k: int) -> str:
    """The MFMA mnemonic the inline-asm route issues for ``(atom K, dtype)``.

    Raises unless the atom is legal on ``arch`` (an MFMA target with that K).
    """
    _gate(arch)
    return _caps.mfma_asm_mnemonic(arch, dtype, atom_k)


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
