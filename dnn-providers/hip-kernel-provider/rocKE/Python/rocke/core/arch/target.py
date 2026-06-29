# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ROCKE architecture target metadata (the polymorphic-core SSOT).

``ArchTarget`` is the single ROCKE-owned description of *what a gfx target
supports* — wave size, LDS capacity, the MMA atom catalog, memory capability
bits, and resource limits. It carries **hardware facts only**: no pipeline or
scheduler vocabulary (those are instance-side policy), and no LLVM intrinsic
text (that is the ``ISABackend``).

The data lives in ``core/arch/data/arch_specs.json`` and is loaded here. Nothing
in this module imports from ``dispatcher/`` — ROCKE is standalone and must be
importable/testable without the dispatcher tree on disk.

See ``dsl_docs/architecture/multi_arch_data_layout.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DATA_FILE = Path(__file__).parent / "data" / "arch_specs.json"

# Canonical dtype spellings used as catalog keys. Instance/spec dtype strings
# are normalised through this map so "f16"/"half" and "fp16" all resolve.
_DTYPE_ALIASES = {
    "f16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f32": "fp32",
    "float": "fp32",
    "fp32": "fp32",
    "fp8": "fp8e4m3",
    "fp8e4m3": "fp8e4m3",
    "bf8": "bf8e5m2",
    "bf8e5m2": "bf8e5m2",
    # Integer WMMA: "iu8"/"iu4" are the RDNA WMMA integer operand families
    # (signedness is an instruction operand, not the dtype); "i32" is the
    # integer accumulator. Scalar int spellings pass through for completeness.
    "iu8": "iu8",
    "iu4": "iu4",
    "i8": "i8",
    "int8": "i8",
    "i4": "i4",
    "int4": "i4",
    "i32": "i32",
    "int32": "i32",
}


def normalize_dtype(name: str) -> str:
    """Map a dtype spelling to its canonical catalog key."""
    key = name.strip().lower()
    return _DTYPE_ALIASES.get(key, key)


# A callable that, given an :class:`~rocke.core.ir.IRBuilder`, a runtime lane
# ``Value`` (0..wave_size-1) and a compile-time fragment slot index, emits the
# index arithmetic for the two tile coordinates of that fragment element and
# returns them as a ``(Value, Value)`` pair. The coordinate *meaning* depends on
# the operand role (see :class:`LayoutMap.role`):
#
#   * accumulator (C/D)  -> ``(row, col)``  in the atom's M x N output tile
#   * A operand          -> ``(row, k)``    in the atom's M x K input tile
#   * B operand          -> ``(k, col)``    in the atom's K x N input tile
#
# The map is *pure* with respect to the IR: it only emits ``arith.*`` ops via the
# builder, so the same map drives both fragment loads (compute the source
# coordinate, gather into the fragment slot) and fragment stores (compute the
# destination coordinate, scatter the slot).
_LaneCoordFn = Any  # Callable[[IRBuilder, Value, int], Tuple[Value, Value]]


@dataclass(frozen=True)
class LayoutMap:
    """Lane/slot -> tile-coordinate map for one MMA fragment role.

    A ``LayoutMap`` is the data-layout contract between an MMA atom and the
    kernel body: it converts a *physical* register location (which lane holds
    the value, and which slot of that lane's fragment vector) into the *logical*
    matrix coordinate that location represents. Because MFMA (wave64) and WMMA
    (wave32) scatter the same logical tile across lanes very differently, the
    kernel cannot hard-code the math — it asks the atom's ``LayoutMap`` for it.

    Attributes
    ----------
    role
        ``"acc"`` (accumulator C/D, coords are ``(row, col)``), ``"a"`` (A
        operand, coords are ``(row, k)``) or ``"b"`` (B operand, coords are
        ``(k, col)``).
    frag_len
        Number of fragment slots per lane for this role (the per-lane vector
        length: e.g. 4 for an MFMA 16x16x16 accumulator, 8 for the WMMA
        accumulator).
    wave_size
        The wave size the map's lane arithmetic assumes (64 for MFMA, 32 for
        WMMA). Surfaced so a kernel can assert the launch geometry matches.
    fn
        The lowering callable ``(builder, lane, slot) -> (coord0, coord1)``.

    Use :meth:`coord` to invoke the map; it validates the slot index and emits
    the arithmetic through the supplied builder.
    """

    role: str
    frag_len: int
    wave_size: int
    fn: _LaneCoordFn = field(repr=False)

    def coord(self, builder: Any, lane: Any, slot: int) -> Tuple[Any, Any]:
        """Emit the index math for ``(lane, slot)`` and return the coord pair.

        ``builder`` is an :class:`~rocke.core.ir.IRBuilder`; ``lane`` is a
        runtime ``i32`` ``Value``; ``slot`` is a compile-time int in
        ``[0, frag_len)``. The returned pair is two ``i32`` ``Value``s whose
        meaning is set by :attr:`role`.
        """
        if not (0 <= slot < self.frag_len):
            raise ValueError(
                f"fragment slot {slot} out of range [0, {self.frag_len}) "
                f"for {self.role!r} layout map"
            )
        return self.fn(builder, lane, slot)


@dataclass(frozen=True)
class MmaOp:
    """A single supported matrix-multiply-accumulate atom on a target.

    ``op_id`` is the opaque handle the backend consumes; today it matches the
    ISA-named ``IRBuilder`` method (e.g. ``mfma_f32_16x16x16_f16``). It is **not**
    LLVM intrinsic text — the backend maps ``op_id`` to an intrinsic.

    The ``*_frag_len`` fields and the layout-map accessors describe the
    *physical* register fragmentation of the atom: how many values of each
    operand a single lane holds, and where (in the logical M/N/K tile) each of
    those values lives. They are the cross-arch data-layout contract that lets
    one kernel body drive both MFMA (wave64) and WMMA (wave32). They are derived
    from ``(op_id, family, shape)`` at load time via :data:`_MMA_FRAGMENT_INFO`;
    op_ids without verified layout maps still carry frag lengths but raise from
    the accessor until a map is added.
    """

    family: str  # "mma" | "wmma"
    a_dtype: str
    b_dtype: str
    c_dtype: str
    m: int
    n: int
    k: int
    op_id: str
    a_frag_len: int = 0
    b_frag_len: int = 0
    c_frag_len: int = 0
    wave_size: int = 64
    # Layout maps are kept off the public field list (repr=False, compare=False)
    # so two MmaOps with the same shape still compare equal regardless of the
    # closure identity of their maps.
    _a_layout: Optional[LayoutMap] = field(default=None, repr=False, compare=False)
    _b_layout: Optional[LayoutMap] = field(default=None, repr=False, compare=False)
    _c_layout: Optional[LayoutMap] = field(default=None, repr=False, compare=False)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.m, self.n, self.k)

    # --- physical-layout accessors ---------------------------------------
    def a_layout(self) -> LayoutMap:
        """The A-operand ``(row, k)`` lane/slot -> coordinate map."""
        return self._require_layout(self._a_layout, "a")

    def b_layout(self) -> LayoutMap:
        """The B-operand ``(k, col)`` lane/slot -> coordinate map."""
        return self._require_layout(self._b_layout, "b")

    def c_layout(self) -> LayoutMap:
        """The accumulator (C/D) ``(row, col)`` lane/slot -> coordinate map."""
        return self._require_layout(self._c_layout, "c")

    # Convenience aliases for the accumulator (the most-used map).
    def acc_layout(self) -> LayoutMap:
        return self.c_layout()

    def _require_layout(self, layout: Optional[LayoutMap], role: str) -> LayoutMap:
        if layout is None:
            raise NotImplementedError(
                f"no verified {role!r} layout map for MMA op_id {self.op_id!r} "
                f"({self.m}x{self.n}x{self.k}); add one to "
                f"_MMA_FRAGMENT_INFO before consuming it"
            )
        return layout


# ---------------------------------------------------------------------------
# Physical fragment layout maps
# ---------------------------------------------------------------------------
#
# The lane arithmetic below is the BYTE-IDENTICAL contract for the MFMA path: it
# reproduces exactly the math the existing MFMA kernels already emit, namely
# ``helpers/atoms.py::MfmaAtom.lane_to_output`` (accumulator) and the operand
# layout documented on the ``MfmaAtom`` factory methods. The WMMA maps encode
# the hardware-verified wave32 gfx1151 layout (see ``instances/gfx1151``).
#
# Each builder returns a ``(builder, lane, slot) -> (coord0, coord1)`` closure.
# ``slot`` is bound at closure-build time? No — it is passed at call time so a
# single LayoutMap covers all slots of the fragment.


def _mfma_acc_16x16(builder, lane, slot):
    """MFMA 16x16 accumulator: slot ``i`` -> ``(m_blk*4 + i, lane % 16)``.

    Mirrors ``MfmaAtom.lane_to_output`` for the (16, 16) case with
    ``c_per_lane == 4`` (``row = m_blk * 4 + i``, ``col = lane % 16``).
    """
    c16 = builder.const_i32(16)
    n_in_atom = builder.mod(lane, c16)
    m_blk = builder.div(lane, c16)
    row = builder.add(builder.mul(m_blk, builder.const_i32(4)), builder.const_i32(slot))
    return row, n_in_atom


def _mfma_acc_32x32(builder, lane, slot):
    """MFMA 32x32 accumulator (``c_per_lane == 16``).

    Mirrors ``MfmaAtom.lane_to_output`` for the (32, 32) case:
    ``row = (i // 4) * 8 + (lane // 32) * 4 + (i % 4)``, ``col = lane % 32``.
    """
    c32 = builder.const_i32(32)
    n_in_atom = builder.mod(lane, c32)
    m_blk = builder.div(lane, c32)
    rb = slot // 4
    ri = slot % 4
    row = builder.add(
        builder.add(
            builder.const_i32(rb * 8), builder.mul(m_blk, builder.const_i32(4))
        ),
        builder.const_i32(ri),
    )
    return row, n_in_atom


def _mfma_a_16x16(builder, lane, slot):
    """MFMA 16x16x16 A operand: lane holds row ``lane % 16``, K ``k_blk*4 + slot``.

    Per ``MfmaAtom.f16_16x16x16``: ``m_in_atom = lane % 16``,
    ``k_blk = lane // 16``, and the lane's ``<4 x half>`` covers
    ``K = [k_blk*4 : k_blk*4 + 4]``. Returns ``(row, k)``.
    """
    c16 = builder.const_i32(16)
    m_in_atom = builder.mod(lane, c16)
    k_blk = builder.div(lane, c16)
    k = builder.add(builder.mul(k_blk, builder.const_i32(4)), builder.const_i32(slot))
    return m_in_atom, k


def _mfma_b_16x16(builder, lane, slot):
    """MFMA 16x16x16 B operand: lane holds col ``lane % 16``, K ``k_blk*4 + slot``.

    Symmetric to :func:`_mfma_a_16x16`; the B fragment is laid out by column.
    Returns ``(k, col)``.
    """
    c16 = builder.const_i32(16)
    n_in_atom = builder.mod(lane, c16)
    k_blk = builder.div(lane, c16)
    k = builder.add(builder.mul(k_blk, builder.const_i32(4)), builder.const_i32(slot))
    return k, n_in_atom


def _mfma_a_16x16x4_f32(builder, lane, slot):
    """MFMA 16x16x4 fp32 A operand: each lane holds one fp32 scalar.

    row = lane % 16, k = lane // 16 (slot is always 0 — a_frag_len == 1).
    Returns ``(row, k)``.
    """
    c16 = builder.const_i32(16)
    m_in_atom = builder.mod(lane, c16)
    k = builder.div(lane, c16)
    return m_in_atom, k


def _mfma_b_16x16x4_f32(builder, lane, slot):
    """MFMA 16x16x4 fp32 B operand: each lane holds one fp32 scalar.

    col = lane % 16, k = lane // 16 (slot is always 0 — b_frag_len == 1).
    Returns ``(k, col)``.
    """
    c16 = builder.const_i32(16)
    n_in_atom = builder.mod(lane, c16)
    k = builder.div(lane, c16)
    return k, n_in_atom


def _mfma_a_32x32x2_f32(builder, lane, slot):
    """MFMA 32x32x2 fp32 A operand: each lane holds one fp32 scalar.

    row = lane % 32, k = lane // 32 (slot is always 0 — a_frag_len == 1).
    Returns ``(row, k)``.
    """
    c32 = builder.const_i32(32)
    m_in_atom = builder.mod(lane, c32)
    k = builder.div(lane, c32)
    return m_in_atom, k


def _mfma_b_32x32x2_f32(builder, lane, slot):
    """MFMA 32x32x2 fp32 B operand: each lane holds one fp32 scalar.

    col = lane % 32, k = lane // 32 (slot is always 0 — b_frag_len == 1).
    Returns ``(k, col)``.
    """
    c32 = builder.const_i32(32)
    n_in_atom = builder.mod(lane, c32)
    k = builder.div(lane, c32)
    return k, n_in_atom


def _mfma_a_32x32x8(builder, lane, slot):
    """MFMA 32x32x8 A operand: row ``lane % 32``, K ``k_blk*4 + slot``.

    Per ``MfmaAtom.f16_32x32x8``: ``m_in_atom = lane % 32``,
    ``k_blk = lane // 32`` (0 or 1), lane ``<4 x half>`` covers
    ``K = [k_blk*4 : k_blk*4 + 4]``. Returns ``(row, k)``.
    """
    c32 = builder.const_i32(32)
    m_in_atom = builder.mod(lane, c32)
    k_blk = builder.div(lane, c32)
    k = builder.add(builder.mul(k_blk, builder.const_i32(4)), builder.const_i32(slot))
    return m_in_atom, k


def _mfma_b_32x32x8(builder, lane, slot):
    """MFMA 32x32x8 B operand: col ``lane % 32``, K ``k_blk*4 + slot``."""
    c32 = builder.const_i32(32)
    n_in_atom = builder.mod(lane, c32)
    k_blk = builder.div(lane, c32)
    k = builder.add(builder.mul(k_blk, builder.const_i32(4)), builder.const_i32(slot))
    return k, n_in_atom


def _mfma_a_16x16x32(builder, lane, slot):
    """MFMA 16x16x32 A operand: row ``lane % 16``, K ``k_blk*8 + slot``.

    Per ``MfmaAtom.f16_16x16x32`` (the K-packed CDNA3 atom, hardware-verified to
    1e-3): ``m_in_atom = lane % 16``, ``k_blk = lane // 16`` (0..3), and the
    lane's ``<8 x half>`` covers the *contiguous* block
    ``K = [k_blk*8 : k_blk*8 + 8]``. The flat-concat alternative
    (``[k_blk*4 : k_blk*4+4] + [k_blk*4+16 : k_blk*4+20]``) compiles and
    validates only to 1e-2 -- this contiguous packing is the correct one.
    Returns ``(row, k)``.
    """
    c16 = builder.const_i32(16)
    m_in_atom = builder.mod(lane, c16)
    k_blk = builder.div(lane, c16)
    k = builder.add(builder.mul(k_blk, builder.const_i32(8)), builder.const_i32(slot))
    return m_in_atom, k


def _mfma_b_16x16x32(builder, lane, slot):
    """MFMA 16x16x32 B operand: col ``lane % 16``, K ``k_blk*8 + slot``.

    Symmetric to :func:`_mfma_a_16x16x32` (B laid out by column). Returns
    ``(k, col)``.
    """
    c16 = builder.const_i32(16)
    n_in_atom = builder.mod(lane, c16)
    k_blk = builder.div(lane, c16)
    k = builder.add(builder.mul(k_blk, builder.const_i32(8)), builder.const_i32(slot))
    return k, n_in_atom


def _mfma_a_32x32x16(builder, lane, slot):
    """MFMA 32x32x16 A operand: row ``lane % 32``, K ``k_blk*8 + slot``.

    The K=16 sibling of :func:`_mfma_a_32x32x8` (per ``MfmaAtom.f16_32x32x16``):
    ``m_in_atom = lane % 32``, ``k_blk = lane // 32`` (0 or 1), and the lane's
    ``<8 x half>`` covers the contiguous block ``K = [k_blk*8 : k_blk*8 + 8]`` --
    the same CDNA3 K-packing rule the verified 16x16x32 atom uses (contiguous,
    not flat-concat). Returns ``(row, k)``.
    """
    c32 = builder.const_i32(32)
    m_in_atom = builder.mod(lane, c32)
    k_blk = builder.div(lane, c32)
    k = builder.add(builder.mul(k_blk, builder.const_i32(8)), builder.const_i32(slot))
    return m_in_atom, k


def _mfma_b_32x32x16(builder, lane, slot):
    """MFMA 32x32x16 B operand: col ``lane % 32``, K ``k_blk*8 + slot``.

    Symmetric to :func:`_mfma_a_32x32x16` (B laid out by column). Returns
    ``(k, col)``.
    """
    c32 = builder.const_i32(32)
    n_in_atom = builder.mod(lane, c32)
    k_blk = builder.div(lane, c32)
    k = builder.add(builder.mul(k_blk, builder.const_i32(8)), builder.const_i32(slot))
    return k, n_in_atom


def _wmma_acc_16x16(builder, lane, slot):
    """WMMA 16x16x16 accumulator (wave32, hardware-verified gfx1151).

    ``<8 x float>`` per lane; slot ``i`` -> ``(row 2*i + lane // 16,
    col lane % 16)``. Returns ``(row, col)``.
    """
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    half = builder.div(lane, c16)
    row = builder.add(builder.const_i32(2 * slot), half)
    return row, col


def _wmma_a_16x16(builder, lane, slot):
    """WMMA 16x16x16 A operand (wave32): lane ``l`` holds row ``l % 16``;
    the ``<16 x half>`` fragment slot ``i`` is K=``i`` (0..15). Returns
    ``(row, k)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    return row, builder.const_i32(slot)


def _wmma_b_16x16(builder, lane, slot):
    """WMMA 16x16x16 B operand (wave32): lane ``l`` holds col ``l % 16``;
    fragment slot ``i`` is K=``i`` (0..15). Returns ``(k, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    return builder.const_i32(slot), col


# --- RDNA3/3.5 integer WMMA (iu8) 16x16x16 lane maps --------------------------
# Same lane geometry as f16 WMMA (lane l holds row/col l%16, cross-half
# duplication), but the K=16 dimension is *packed* into the <4 x i32> A/B
# fragment: slot j (0..3) is one i32 holding the four int8 K-values
# [4j, 4j+1, 4j+2, 4j+3]. The lane map therefore returns the K *base* of the
# slot (4*j); the kernel/staging code packs the four consecutive K bytes from
# there. The accumulator is identical to f16 WMMA (_wmma_acc_16x16), only the
# element type is i32 instead of f32.
def _wmma_a_16x16_iu8(builder, lane, slot):
    """iu8 WMMA A operand (wave32): lane ``l`` holds row ``l % 16``; A fragment
    slot ``j`` is the i32 packing K=[4j..4j+3]. Returns ``(row, k_base=4j)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    return row, builder.const_i32(4 * slot)


def _wmma_b_16x16_iu8(builder, lane, slot):
    """iu8 WMMA B operand (wave32): lane ``l`` holds col ``l % 16``; B fragment
    slot ``j`` is the i32 packing K=[4j..4j+3]. Returns ``(k_base=4j, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    return builder.const_i32(4 * slot), col


# --- RDNA3/3.5 integer WMMA (iu4) 16x16x16 lane maps --------------------------
# Same lane geometry as iu8, but the K=16 dimension is packed into <2 x i32>:
# slot j (0..1) holds eight signed int4 values K=[8j..8j+7].
def _wmma_a_16x16_iu4(builder, lane, slot):
    """iu4 WMMA A operand (wave32): lane ``l`` holds row ``l % 16``; A fragment
    slot ``j`` is the i32 packing K=[8j..8j+7]. Returns ``(row, k_base=8j)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    return row, builder.const_i32(8 * slot)


def _wmma_b_16x16_iu4(builder, lane, slot):
    """iu4 WMMA B operand (wave32): lane ``l`` holds col ``l % 16``; B fragment
    slot ``j`` is the i32 packing K=[8j..8j+7]. Returns ``(k_base=8j, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    return builder.const_i32(8 * slot), col


# --- RDNA4 (gfx12) WMMA 16x16x16 lane maps ------------------------------------
# RDNA4 dropped the RDNA3/3.5 cross-half duplication: A/B fragments are
# ``<8 x half>`` per lane (not <16 x half>), and the K dimension is split across
# the two lane-halves (lanes 0-15 carry K 0..7, lanes 16-31 carry K 8..15). The
# accumulator is column-distributed (CDNA/MFMA-style): lanes index columns,
# registers index rows. These maps are the *hypothesis* verified empirically by
# examples/gfx1201/wmma_probe.py before matmul_nbits is trusted on gfx1201.
def _wmma_gfx12_acc_16x16(builder, lane, slot):
    """RDNA4 WMMA 16x16x16 accumulator (wave32): ``<8 x float>`` per lane;
    slot ``i`` -> ``(row (lane // 16) * 8 + i, col lane % 16)``. Returns
    ``(row, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    half = builder.div(lane, c16)
    row = builder.add(builder.mul(half, builder.const_i32(8)), builder.const_i32(slot))
    return row, col


def _wmma_gfx12_a_16x16(builder, lane, slot):
    """RDNA4 WMMA 16x16x16 A operand (wave32): lane ``l`` holds row ``l % 16``;
    the ``<8 x half>`` fragment slot ``i`` is K=``(l // 16) * 8 + i``. Returns
    ``(row, k)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(builder.mul(k_half, builder.const_i32(8)), builder.const_i32(slot))
    return row, k


def _wmma_gfx12_b_16x16(builder, lane, slot):
    """RDNA4 WMMA 16x16x16 B operand (wave32): lane ``l`` holds col ``l % 16``;
    the ``<8 x half>`` fragment slot ``i`` is K=``(l // 16) * 8 + i``. Returns
    ``(k, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(builder.mul(k_half, builder.const_i32(8)), builder.const_i32(slot))
    return k, col


# --- gfx1250 (gfx1250) WMMA 16x16x32 lane maps (CDNA, GFX12 programming model) --
# gfx1250 is wave32/WMMA like gfx12 RDNA but the primary fp16/bf16 atom is
# 16x16x32 (K=32, not 16). A/B fragments are <16 x half> per lane (16 K-elements
# each); the K dimension is split across the two lane-halves (lanes 0-15 carry
# K 0..15, lanes 16-31 carry K 16..31). The accumulator is the same 16x16
# column-distributed layout as gfx12 (<8 x float>, slot i -> row (l//16)*8 + i,
# col l%16), since the output tile is still 16x16. These maps are the
# *hypothesis* verified empirically by examples/gfx1250/wmma_probe.py.
def _wmma_gfx1250_a_16x16x32(builder, lane, slot):
    """gfx1250 WMMA 16x16x32 A operand (wave32): lane ``l`` holds row ``l % 16``;
    the ``<16 x half>`` fragment slot ``i`` is K=``(l // 16) * 16 + i``. Returns
    ``(row, k)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(builder.mul(k_half, builder.const_i32(16)), builder.const_i32(slot))
    return row, k


def _wmma_gfx1250_b_16x16x32(builder, lane, slot):
    """gfx1250 WMMA 16x16x32 B operand (wave32): lane ``l`` holds col ``l % 16``;
    the ``<16 x half>`` fragment slot ``i`` is K=``(l // 16) * 16 + i``. Returns
    ``(k, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(builder.mul(k_half, builder.const_i32(16)), builder.const_i32(slot))
    return k, col


@dataclass(frozen=True)
class _FragInfo:
    """Per-op_id fragment metadata: per-lane vector lengths, wave size, and the
    (optional) lane/slot coordinate functions for A / B / accumulator."""

    a_frag_len: int
    b_frag_len: int
    c_frag_len: int
    wave_size: int
    a_fn: _LaneCoordFn = None
    b_fn: _LaneCoordFn = None
    c_fn: _LaneCoordFn = None


# op_id -> physical fragment metadata. Frag lengths are populated for every
# IRBuilder MFMA/WMMA atom (so ``IRBuilder.mma`` can size the result vector);
# layout-map functions are populated for the atoms whose lane math is verified.
# Adding a new atom is one row here.
_MMA_FRAGMENT_INFO: Dict[str, _FragInfo] = {
    # --- MFMA fp32 (wave64) -----------------------------------------------
    # A/B are scalar float per lane (a_frag_len=b_frag_len=1); accumulator
    # shares the standard 16x16 / 32x32 layout (c_frag_len=4 / 16).
    "mfma_f32_16x16x4_f32": _FragInfo(
        1, 1, 4, 64, _mfma_a_16x16x4_f32, _mfma_b_16x16x4_f32, _mfma_acc_16x16
    ),
    "mfma_f32_32x32x2_f32": _FragInfo(
        1, 1, 16, 64, _mfma_a_32x32x2_f32, _mfma_b_32x32x2_f32, _mfma_acc_32x32
    ),
    # --- MFMA f16 (wave64) ------------------------------------------------
    "mfma_f32_16x16x16_f16": _FragInfo(
        4, 4, 4, 64, _mfma_a_16x16, _mfma_b_16x16, _mfma_acc_16x16
    ),
    "mfma_f32_16x16x32_f16": _FragInfo(
        8, 8, 4, 64, _mfma_a_16x16x32, _mfma_b_16x16x32, _mfma_acc_16x16
    ),
    "mfma_f32_32x32x8_f16": _FragInfo(
        4, 4, 16, 64, _mfma_a_32x32x8, _mfma_b_32x32x8, _mfma_acc_32x32
    ),
    "mfma_f32_32x32x16_f16": _FragInfo(
        8, 8, 16, 64, _mfma_a_32x32x16, _mfma_b_32x32x16, _mfma_acc_32x32
    ),
    "mfma_f32_4x4x4_f16": _FragInfo(4, 4, 4, 64),
    # --- MFMA bf16 (wave64) ----------------------------------------------
    "mfma_f32_16x16x16_bf16": _FragInfo(
        4, 4, 4, 64, _mfma_a_16x16, _mfma_b_16x16, _mfma_acc_16x16
    ),
    "mfma_f32_16x16x32_bf16": _FragInfo(
        8, 8, 4, 64, _mfma_a_16x16x32, _mfma_b_16x16x32, _mfma_acc_16x16
    ),
    "mfma_f32_32x32x8_bf16": _FragInfo(
        4, 4, 16, 64, _mfma_a_32x32x8, _mfma_b_32x32x8, _mfma_acc_32x32
    ),
    "mfma_f32_32x32x16_bf16": _FragInfo(
        8, 8, 16, 64, _mfma_a_32x32x16, _mfma_b_32x32x16, _mfma_acc_32x32
    ),
    # --- MFMA fp8 / bf8 (wave64) -----------------------------------------
    # fp8/bf8 share the f16 operand lane layout (a_per_lane=8, same K-packing);
    # only the element type / intrinsic mangling differ.
    "mfma_f32_16x16x32_fp8": _FragInfo(
        8, 8, 4, 64, _mfma_a_16x16x32, _mfma_b_16x16x32, _mfma_acc_16x16
    ),
    "mfma_f32_16x16x32_bf8": _FragInfo(
        8, 8, 4, 64, _mfma_a_16x16x32, _mfma_b_16x16x32, _mfma_acc_16x16
    ),
    "mfma_f32_32x32x16_fp8": _FragInfo(
        8, 8, 16, 64, _mfma_a_32x32x16, _mfma_b_32x32x16, _mfma_acc_32x32
    ),
    "mfma_f32_32x32x16_bf8": _FragInfo(
        8, 8, 16, 64, _mfma_a_32x32x16, _mfma_b_32x32x16, _mfma_acc_32x32
    ),
    # --- MFMA MX (wave64), frag lengths only -----------------------------
    "mfma_f32_16x16x128_fp4": _FragInfo(16, 16, 4, 64),
    "mfma_f32_16x16x96_fp6": _FragInfo(12, 12, 4, 64),
    # Unscaled fp8 K=128 hero atom (lowers through the f8f6f4 scale-MFMA
    # intrinsic; there is no dense plain fp8 K=128). A/B are 32 fp8 bytes per
    # lane (<8 x i32> at the intrinsic boundary), accumulator <4 x float> --
    # same fragment widths as the f8f6f4 sibling below. Registered so the op_id
    # (already in ir._MMA_C_FRAG_LEN) resolves to correct fragment lengths if
    # added to the JSON catalog, instead of the zero-length _frag_info fallback.
    "mfma_f32_16x16x128_fp8": _FragInfo(32, 32, 4, 64),
    "mfma_scale_f32_16x16x128_f8f6f4": _FragInfo(32, 32, 4, 64),
    # --- WMMA f16 / bf16 (wave32, RDNA) ----------------------------------
    "wmma_f32_16x16x16_f16": _FragInfo(
        16, 16, 8, 32, _wmma_a_16x16, _wmma_b_16x16, _wmma_acc_16x16
    ),
    # bf16 shares the f16 fragment layout (same 16x16x16 lane math; only the
    # element type / intrinsic mangling differ — operands lower as <16 x i16>).
    "wmma_f32_16x16x16_bf16": _FragInfo(
        16, 16, 8, 32, _wmma_a_16x16, _wmma_b_16x16, _wmma_acc_16x16
    ),
    # --- WMMA iu8 (wave32, RDNA3/3.5) ------------------------------------------
    # A/B fragments are <4 x i32> (16 int8 packed 4-per-i32); accumulator is
    # <8 x i32> with the same lane math as the f16 WMMA accumulator.
    "wmma_i32_16x16x16_iu8": _FragInfo(
        4, 4, 8, 32, _wmma_a_16x16_iu8, _wmma_b_16x16_iu8, _wmma_acc_16x16
    ),
    # --- WMMA iu4 (wave32, RDNA3/3.5) ------------------------------------------
    # A/B fragments are <2 x i32> (16 int4 packed 8-per-i32); accumulator is
    # <8 x i32> with the same lane math as the f16 WMMA accumulator.
    "wmma_i32_16x16x16_iu4": _FragInfo(
        2, 2, 8, 32, _wmma_a_16x16_iu4, _wmma_b_16x16_iu4, _wmma_acc_16x16
    ),
    # --- WMMA f16 / bf16 (wave32, RDNA4 / gfx12) -------------------------------
    # No cross-half duplication: A/B are <8 x half> per lane; column-distributed
    # accumulator. Lane maps verified by examples/gfx1201/wmma_probe.py.
    "wmma_gfx12_f32_16x16x16_f16": _FragInfo(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_acc_16x16
    ),
    "wmma_gfx12_f32_16x16x16_bf16": _FragInfo(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_acc_16x16
    ),
    # --- WMMA f16 / bf16 (wave32, gfx1250, CDNA) ----------------------
    # K=32 atom: A/B are <16 x half> per lane (K split across lane-halves, 16
    # each); accumulator is the same 16x16 column-distributed <8 x float> as
    # gfx12. Lane maps verified by examples/gfx1250/wmma_probe.py.
    "wmma_gfx1250_f32_16x16x32_f16": _FragInfo(
        16,
        16,
        8,
        32,
        _wmma_gfx1250_a_16x16x32,
        _wmma_gfx1250_b_16x16x32,
        _wmma_gfx12_acc_16x16,
    ),
    # gfx1250 FP8/BF8 K=64 WMMA. A/B carry 32 low-bit bytes per lane presented
    # as <8 x i32>; accumulator is the same 16x16 column-distributed <8 x float>
    # as the f16/bf16 K=32 atom. The block-scaled GEMM kernel computes operand
    # offsets directly (no LayoutMap), so only frag lengths + wave size are
    # registered here; the accumulator map is shared with the f16 path.
    "wmma_gfx1250_f32_16x16x64_fp8_fp8": _FragInfo(
        8,
        8,
        8,
        32,
        None,
        None,
        _wmma_gfx12_acc_16x16,
    ),
    "wmma_gfx1250_f32_16x16x64_fp8_bf8": _FragInfo(
        8,
        8,
        8,
        32,
        None,
        None,
        _wmma_gfx12_acc_16x16,
    ),
    "wmma_gfx1250_f32_16x16x64_bf8_fp8": _FragInfo(
        8,
        8,
        8,
        32,
        None,
        None,
        _wmma_gfx12_acc_16x16,
    ),
    "wmma_gfx1250_f32_16x16x64_bf8_bf8": _FragInfo(
        8,
        8,
        8,
        32,
        None,
        None,
        _wmma_gfx12_acc_16x16,
    ),
    "wmma_gfx1250_f32_16x16x32_bf16": _FragInfo(
        16,
        16,
        8,
        32,
        _wmma_gfx1250_a_16x16x32,
        _wmma_gfx1250_b_16x16x32,
        _wmma_gfx12_acc_16x16,
    ),
}


def _frag_info(op_id: str) -> _FragInfo:
    info = _MMA_FRAGMENT_INFO.get(op_id)
    if info is None:
        # Unknown atoms still load (e.g. a future JSON-only op_id); they carry
        # zero frag lengths and no maps until registered here.
        return _FragInfo(0, 0, 0, 64)
    return info


@dataclass(frozen=True)
class MemoryCapabilities:
    has_async_lds: bool
    has_async_global_lds: bool
    has_ds_read_tr: bool
    has_tdm: bool
    buffer_load_max_dwords: int


@dataclass(frozen=True)
class ResourceLimits:
    max_threads_per_block: int
    vgprs: int
    agprs: int
    sgprs: int


class MmaCatalog:
    """The arch-selected set of MMA atoms, with enumeration + best-K selection."""

    def __init__(self, ops: List[MmaOp]) -> None:
        self._ops = tuple(ops)

    @property
    def ops(self) -> Tuple[MmaOp, ...]:
        return self._ops

    def enumerate(
        self,
        *,
        family: str = "mma",
        a_dtype: str,
        b_dtype: str,
        c_dtype: str,
        m: Optional[int] = None,
        n: Optional[int] = None,
    ) -> List[MmaOp]:
        a, b, c = (
            normalize_dtype(a_dtype),
            normalize_dtype(b_dtype),
            normalize_dtype(c_dtype),
        )
        out = []
        for op in self._ops:
            if op.family != family:
                continue
            if (op.a_dtype, op.b_dtype, op.c_dtype) != (a, b, c):
                continue
            if m is not None and op.m != m:
                continue
            if n is not None and op.n != n:
                continue
            out.append(op)
        return out

    def has_shape(
        self,
        *,
        family: str = "mma",
        a_dtype: str,
        b_dtype: str,
        c_dtype: str,
        m: int,
        n: int,
        k: int,
    ) -> bool:
        return any(
            op.shape == (m, n, k)
            for op in self.enumerate(
                family=family,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                c_dtype=c_dtype,
                m=m,
                n=n,
            )
        )

    def select_largest_k(
        self,
        *,
        family: str = "mma",
        a_dtype: str,
        b_dtype: str,
        c_dtype: str,
        m: int,
        n: int,
        k_max: Optional[int] = None,
    ) -> Optional[MmaOp]:
        cands = [
            op
            for op in self.enumerate(
                family=family,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                c_dtype=c_dtype,
                m=m,
                n=n,
            )
            if k_max is None or op.k <= k_max
        ]
        if not cands:
            return None
        return max(cands, key=lambda op: op.k)

    def by_op_id(self, op_id: str) -> Optional[MmaOp]:
        """Look up an atom by its ``op_id`` handle (the backend's MMA key)."""
        for op in self._ops:
            if op.op_id == op_id:
                return op
        return None

    def op_for_shape(
        self,
        *,
        family: str = "mma",
        a_dtype: str,
        b_dtype: str,
        c_dtype: str,
        m: int,
        n: int,
        k: int,
    ) -> Optional[MmaOp]:
        for op in self.enumerate(
            family=family,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            m=m,
            n=n,
        ):
            if op.k == k:
                return op
        return None


@dataclass(frozen=True)
class ArchTarget:
    """Hardware-facts surface for one gfx target. Frozen; cheap to pass around."""

    gfx: str
    family: str
    target_family: str
    wave_size: int
    lds_capacity_bytes: int
    vmcnt_bits: int
    mma: MmaCatalog
    memory: MemoryCapabilities
    limits: ResourceLimits
    stepping: Optional[str] = None
    matrix_path: str = "mfma"
    has_mfma: bool = True
    has_wmma: bool = False
    waitcnt_model: str = "legacy"
    barrier_model: str = "legacy"
    requires_shader_end_padding: bool = False
    virtual_address_bits: int = 48
    wgp_cache_lds_shared: bool = False

    # --- identity ---------------------------------------------------------
    @property
    def isa_triple(self) -> str:
        return f"amdgcn-amd-amdhsa--{self.gfx}"

    # --- hardware predicates (policies compose these) ---------------------
    def fits_lds(self, bytes_in_use: int) -> bool:
        return bytes_in_use <= self.lds_capacity_bytes

    def supports_dtype_combo(
        self, a: str, b: str, c: str, *, family: str = "mma"
    ) -> bool:
        return (
            len(self.mma.enumerate(family=family, a_dtype=a, b_dtype=b, c_dtype=c)) > 0
        )

    def max_vector_load_dwords(self, dtype: str) -> int:
        # Width is gated by the buffer-load path, not the element type today.
        return self.memory.buffer_load_max_dwords

    @property
    def max_threads_per_block(self) -> int:
        return self.limits.max_threads_per_block

    @staticmethod
    def from_gfx(gfx: str) -> "ArchTarget":
        return _build_target(gfx)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_specs() -> Dict[str, dict]:
    # Pin UTF-8: the embedded interpreter defaults to the ASCII codec, and
    # arch_specs.json contains non-ASCII bytes.
    with open(_DATA_FILE, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc["arches"]


def _build_mma_op(o: dict) -> MmaOp:
    """Construct an :class:`MmaOp` from one catalog JSON row, attaching the
    physical fragment lengths and layout maps registered for its op_id."""
    op_id = o["op_id"]
    info = _frag_info(op_id)

    def _mk(role: str, frag_len: int, fn: _LaneCoordFn) -> Optional[LayoutMap]:
        if fn is None or frag_len <= 0:
            return None
        return LayoutMap(role=role, frag_len=frag_len, wave_size=info.wave_size, fn=fn)

    return MmaOp(
        family=o["family"],
        a_dtype=normalize_dtype(o["a"]),
        b_dtype=normalize_dtype(o["b"]),
        c_dtype=normalize_dtype(o["c"]),
        m=o["m"],
        n=o["n"],
        k=o["k"],
        op_id=op_id,
        a_frag_len=info.a_frag_len,
        b_frag_len=info.b_frag_len,
        c_frag_len=info.c_frag_len,
        wave_size=info.wave_size,
        _a_layout=_mk("a", info.a_frag_len, info.a_fn),
        _b_layout=_mk("b", info.b_frag_len, info.b_fn),
        _c_layout=_mk("c", info.c_frag_len, info.c_fn),
    )


@lru_cache(maxsize=None)
def _build_target(gfx: str) -> ArchTarget:
    specs = _load_specs()
    if gfx not in specs:
        raise KeyError(
            f"unknown gfx target {gfx!r}; known: {sorted(specs)}. "
            f"Add a row to {_DATA_FILE.name}."
        )
    row = specs[gfx]
    mma = MmaCatalog([_build_mma_op(o) for o in row["mma"]])
    mem = row["memory"]
    lim = row["limits"]
    has_mfma = any(op.family == "mma" for op in mma.ops)
    has_wmma = any(op.family == "wmma" for op in mma.ops)
    return ArchTarget(
        gfx=gfx,
        family=row["family"],
        target_family=row["target_family"],
        wave_size=row["wave_size"],
        lds_capacity_bytes=row["lds_capacity_bytes"],
        vmcnt_bits=row["vmcnt_bits"],
        mma=mma,
        memory=MemoryCapabilities(
            has_async_lds=mem["has_async_lds"],
            has_async_global_lds=mem.get("has_async_global_lds", mem["has_async_lds"]),
            has_ds_read_tr=mem["has_ds_read_tr"],
            has_tdm=mem.get("has_tdm", False),
            buffer_load_max_dwords=mem["buffer_load_max_dwords"],
        ),
        limits=ResourceLimits(
            max_threads_per_block=lim["max_threads_per_block"],
            vgprs=lim["vgprs"],
            agprs=lim["agprs"],
            sgprs=lim["sgprs"],
        ),
        stepping=row.get("stepping"),
        matrix_path=row.get(
            "matrix_path",
            "wmma" if has_wmma and not has_mfma else "mfma" if has_mfma else "scalar",
        ),
        has_mfma=row.get("has_mfma", has_mfma),
        has_wmma=row.get("has_wmma", has_wmma),
        waitcnt_model=row.get("waitcnt_model", "legacy"),
        barrier_model=row.get("barrier_model", "legacy"),
        requires_shader_end_padding=row.get("requires_shader_end_padding", False),
        virtual_address_bits=row.get("virtual_address_bits", 48),
        wgp_cache_lds_shared=row.get("wgp_cache_lds_shared", False),
    )


def known_arches() -> Tuple[str, ...]:
    return tuple(sorted(_load_specs()))


def arch_from_isa(isa: str) -> str:
    """Extract the gfx token from an isa triple like ``amdgcn-amd-amdhsa--gfx942``."""
    return isa.rsplit("-", 1)[-1] if "-" in isa else isa
