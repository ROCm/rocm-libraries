# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-arch capability table of the attention kernels, forward and backward.

Every attention builder asks this module, not a literal in its own file,
whether a lever is legal on a target:

* the LDS-DMA (``buffer_load ... lds``) widths in dwords per lane, from
  :attr:`rocke.core.arch.ArchTarget.async_lds_dwords` (gfx942 ``(1,)``,
  gfx950 ``(1, 3, 4)``, none on RDNA and gfx1250);
* the LDS transpose-read widths in bits per lane: the 64-bit
  ``ds_read_tr16_b64`` where the catalog sets ``has_ds_read_tr`` (gfx950);
* the 16x16 MMA atoms per dtype, from the arch MMA catalog;
* native no-return fp32 global atomic add (no compare-and-swap loop);
* the inline-asm MFMA facts: mnemonics, the ISA result wait states and the
  smallest legal ``s_nop`` after an asm MFMA cluster.

The architectural facts come from :class:`~rocke.core.arch.ArchTarget`; the
tables below hold only what the catalog does not carry. Every check raises
:class:`ValueError` before any IR is emitted. Policy (which arches a family
targets, which levers it exposes) stays with the family; the attention
backward's lives in ``kernels/common/_attention_bwd_caps.py``.

C++ twin: ``rocke/helper_rocke.helpers.attention_arch_caps.h``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

from rocke.core.arch import ArchTarget
from rocke.core.arch.target import MmaOp

__all__ = [
    "ATTN_CAPS_ARCHES",
    "ASM_MFMA_VALU_LEAD_WAIT_STATES",
    "ASM_TAIL_NOP_MAX",
    "AttnArchCaps",
    "attn_arch_caps",
    "attn_atoms",
    "asm_tail_nop_min",
    "check_asm_tail_nop",
    "check_lds_bytes",
    "check_lds_dma_width",
    "check_transpose_read",
    "mfma_asm_mnemonic",
    "mfma_result_wait_states",
    "select_attn_atom",
    "wide_k_atom_available",
]

# Arches the attention kernels have a capability row for.
ATTN_CAPS_ARCHES = ("gfx942", "gfx950", "gfx1151", "gfx1201", "gfx1250")

# Native no-return fp32 global atomic add (``global_atomic_add_f32`` without a
# compare-and-swap expansion). Not the catalog's ``has_global_float_atomic_add``,
# which gates the WMMA split-K epilogue and is set only where that epilogue has
# been validated.
_NATIVE_FP32_ATOMIC_ADD = MappingProxyType(
    {
        "gfx942": True,
        "gfx950": True,
        "gfx1151": True,
        "gfx1201": True,
        "gfx1250": True,
    }
)

# The one LDS transpose read the attention kernels issue: ``ds_read_tr16_b64``.
_TR_READ_B64_BITS = 64

# MFMA mnemonics of the inline-asm route, per (atom K, dtype); the 16x16x32
# atom exists on gfx950 only (see the arch atom table).
_MFMA_ASM = MappingProxyType(
    {
        (16, "fp16"): "v_mfma_f32_16x16x16_f16",
        (16, "bf16"): "v_mfma_f32_16x16x16_bf16",
        (32, "fp16"): "v_mfma_f32_16x16x32_f16",
        (32, "bf16"): "v_mfma_f32_16x16x32_bf16",
    }
)

# Passes of the XDL MFMA atoms of the inline-asm route (the backend's own
# scheduling model): every 16x16 f16 / bf16 atom on gfx942 and gfx950 is a
# 4-pass XDL op.
_XDL_PASSES = MappingProxyType({16: 4, 32: 4})

# Wait states, beyond the atom's passes, the ISA requires before a VALU or
# memory read of an XDL MFMA result (2-pass atoms excepted on gfx950).
_MFMA_RESULT_EXTRA_WAIT_STATES = MappingProxyType({"gfx942": 3, "gfx950": 4})

# Wait states the ISA requires after a VALU write of a VGPR before an MFMA
# reads it (any source); an inline-asm MFMA hides this from the backend too,
# so every cluster of the asm route opens with them (``s_nop 1``).
ASM_MFMA_VALU_LEAD_WAIT_STATES = 2

# Largest ``s_nop`` count the encoding has.
ASM_TAIL_NOP_MAX = 15


@dataclass(frozen=True)
class AttnArchCaps:
    """Resolved attention-relevant facts for one gfx target."""

    arch: str
    wave_size: int
    matrix_path: str  # "mfma" | "wmma"
    lds_capacity_bytes: int
    lds_dma_dwords: tuple[int, ...]  # legal LDS-DMA widths; () = no LDS DMA
    tr_read_bits: tuple[int, ...]  # legal transpose-read widths; () = none
    has_agprs: bool
    arch_vgprs: int  # per-lane arch-VGPR budget
    agprs: int  # per-lane AGPR budget (0 on RDNA)
    native_fp32_atomic_add: bool

    @property
    def max_lds_dma_dwords(self) -> int:
        return max(self.lds_dma_dwords, default=0)

    @property
    def has_tr_read_b64(self) -> bool:
        return _TR_READ_B64_BITS in self.tr_read_bits


@cache
def attn_arch_caps(arch: str) -> AttnArchCaps:
    """Return the attention capability record of ``arch`` (a gfx id)."""
    if arch not in ATTN_CAPS_ARCHES:
        raise ValueError(
            f"no attention capability row for {arch!r}; known: {ATTN_CAPS_ARCHES}"
        )
    t = ArchTarget.from_gfx(arch)
    agprs = int(t.limits.agprs)
    return AttnArchCaps(
        arch=arch,
        wave_size=int(t.wave_size),
        matrix_path="mfma" if t.has_mfma else "wmma",
        lds_capacity_bytes=int(t.lds_capacity_bytes),
        lds_dma_dwords=t.async_lds_dwords,
        tr_read_bits=(_TR_READ_B64_BITS,) if t.memory.has_ds_read_tr else (),
        has_agprs=agprs > 0,
        arch_vgprs=int(t.limits.vgprs) - agprs,
        agprs=agprs,
        native_fp32_atomic_add=_NATIVE_FP32_ATOMIC_ADD[arch],
    )


@cache
def attn_atoms(arch: str, dtype: str) -> tuple[MmaOp, ...]:
    """The 16x16 MMA atoms with fp32 accumulation on ``arch``, by increasing K.

    ``dtype`` is a catalog dtype name (``"fp16"``, ``"bf16"``, ...).
    """
    caps = attn_arch_caps(arch)
    family = "mma" if caps.matrix_path == "mfma" else "wmma"
    ops = ArchTarget.from_gfx(arch).mma.enumerate(
        family=family, a_dtype=dtype, b_dtype=dtype, c_dtype="fp32", m=16, n=16
    )
    out = sorted(
        (op for op in ops if op.wave_size == caps.wave_size and op.a_frag_len > 0),
        key=lambda op: op.k,
    )
    if not out:
        raise ValueError(f"no 16x16 {dtype} MMA atom on {arch}")
    return tuple(out)


def select_attn_atom(arch: str, dtype: str, k: int = 16) -> MmaOp:
    """The 16x16xk atom for ``arch`` and ``dtype``; raises if the arch lacks it."""
    for op in attn_atoms(arch, dtype):
        if op.k == k:
            return op
    raise ValueError(
        f"no native 16x16x{k} {dtype} atom on {arch} "
        f"(available K: {[op.k for op in attn_atoms(arch, dtype)]})"
    )


def wide_k_atom_available(arch: str, dtype: str) -> bool:
    """True when the native 16x16x32 atom exists (gfx950)."""
    return any(op.k == 32 for op in attn_atoms(arch, dtype))


def check_lds_bytes(arch: str, nbytes: int) -> None:
    caps = attn_arch_caps(arch)
    if nbytes < 0 or nbytes > caps.lds_capacity_bytes:
        raise ValueError(
            f"LDS footprint {nbytes} B exceeds the {arch} capacity "
            f"{caps.lds_capacity_bytes} B"
        )


def check_lds_dma_width(arch: str, dwords: int) -> None:
    """Raise unless an LDS-DMA copy of ``dwords`` per lane is legal on ``arch``."""
    caps = attn_arch_caps(arch)
    if not caps.lds_dma_dwords:
        raise ValueError(f"{arch} has no global-to-LDS DMA path")
    if dwords not in caps.lds_dma_dwords:
        raise ValueError(
            f"LDS-DMA width of {dwords} dword(s) is not legal on {arch}; "
            f"legal widths: {caps.lds_dma_dwords}"
        )


def check_transpose_read(arch: str, bits: int) -> None:
    """Raise unless an LDS transpose read of ``bits`` per lane is legal.

    The attention kernels issue only the 64-bit ``ds_read_tr16_b64``.
    """
    caps = attn_arch_caps(arch)
    if bits != _TR_READ_B64_BITS:
        raise ValueError(f"unsupported transpose-read width {bits}")
    if not caps.has_tr_read_b64:
        raise ValueError(f"{arch} has no ds_read_tr16_b64")


def mfma_asm_mnemonic(arch: str, dtype: str, atom_k: int) -> str:
    """The MFMA mnemonic the inline-asm route issues for ``(atom K, dtype)``.

    Raises unless the atom is legal on ``arch`` (an MFMA target with that K).
    """
    caps = attn_arch_caps(arch)
    if not caps.has_agprs:
        raise ValueError(f"{arch} has no AGPRs; the asm MFMA route is MFMA-only")
    if (atom_k, dtype) not in _MFMA_ASM:
        raise ValueError(f"no asm MFMA for atom K {atom_k} and {dtype}")
    if atom_k == 32 and not wide_k_atom_available(arch, dtype):
        raise ValueError("the 16x16x32 MFMA atom is gfx950-only")
    return _MFMA_ASM[(atom_k, dtype)]


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
    caps = attn_arch_caps(arch)
    if not caps.has_agprs or arch not in _MFMA_RESULT_EXTRA_WAIT_STATES:
        raise ValueError(f"no asm MFMA hazard table for {arch}")
    passes = _XDL_PASSES[atom_k]
    extra = _MFMA_RESULT_EXTRA_WAIT_STATES[arch]
    if passes == 2:
        extra = 3
    return passes + extra


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
    if (
        not isinstance(nop, int)
        or isinstance(nop, bool)
        or not lo <= nop <= ASM_TAIL_NOP_MAX
    ):
        raise ValueError(
            f"asm tail s_nop {nop!r} is not legal on {arch} for the 16x16x{atom_k} "
            f"{dtype} atom; legal: {lo}..{ASM_TAIL_NOP_MAX} "
            f"(ISA MFMA result wait states)"
        )
