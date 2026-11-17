# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emission helpers shared by the gfx942 and gfx950 forward attention kernels.

Each helper here was a byte-for-byte copy in both arch modules
(``kernels/gfx942/attention_*.py`` and ``kernels/gfx950/attention_*.py``).
Keeping one copy means a fix lands on both arches at once. Helpers that differ
between the arches, even by one literal, stay in the arch modules.

Mirrored by instance_attention_tiled_common_internal.h in the C++ engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rocke.core.ir import IRBuilder, Type, Value
from rocke.helpers.atoms import MfmaAtom, make_c_warp_dstr_encoding
from rocke.helpers.distribution import make_static_tile_distribution

__all__ = [
    "C32_DIST",
    "AccLayout",
    "TiledLaneMap",
    "mfma_32x32_c_col",
    "mfma_32x32_c_row",
]


# CK-Tile C-accumulator warp distribution for the 32x32x16 MFMA atom. The
# hand-rolled ``mfma_32x32x16_c_row/col`` lane arithmetic (in
# ``helpers/attention.py``) is exactly the ``calculate_x`` of CK Tile's
# ``CWarpDstrEncoding`` (``warp_gemm_attribute_mfma.hpp``): for one wavefront
# the lane splits as ``(m_blk, n) = (lane // 32, lane % 32)`` and the per-lane
# ``<16 x f32>`` accumulator slot ``i`` splits row-major over the two M Y dims
# ``(kCM0PerLane, kCM1PerLane) = (4, 4)``. The distribution's ``calculate_x``
# then yields ``row = (i // 4) * 8 + m_blk * 4 + (i % 4)`` and ``col = n`` --
# the same mapping the scalar helpers produce, but emitted by the tile-
# distribution algebra instead of open-coded ``div``/``mul``/``add``. The C
# layout is dtype-independent, so the f16 atom drives it for both fp16 and
# bf16. The 3D tiled kernels still use the scalar helpers from
# ``helpers/attention.py``.
C32_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.f16_32x32x16())
)


def mfma_32x32_c_row(b, lane, elem_idx: int):
    """MFMA-local output row for a ``32x32x16`` C element ``elem_idx`` (0..15).

    Drives CK Tile's C-warp ``TileDistribution.calculate_x`` instead of the
    hand-rolled ``row = (i//4)*8 + (lane//32)*4 + (i%4)`` lane arithmetic.
    """
    if not (0 <= elem_idx < 16):
        raise ValueError(f"mfma_32x32x16 elem_idx must be 0..15, got {elem_idx}")
    m_blk = b.div(lane, b.const_i32(32))
    n = b.mod(lane, b.const_i32(32))
    y0 = b.const_i32(elem_idx // 4)
    y1 = b.const_i32(elem_idx % 4)
    row, _col = C32_DIST.calculate_x(b, ys=[y0, y1], ps=[[m_blk, n]])
    return row


def mfma_32x32_c_col(b, lane, n_tile32: int = 0):
    """MFMA-local output col for ``32x32x16`` C elements in N-tile ``n_tile32``.

    The per-lane column is the N coordinate of CK Tile's C-warp distribution
    (``lane % 32``); ``n_tile32 * 32`` is the 32-column tile base added on top.
    Every per-lane accumulator slot shares this column, so the element index
    does not enter the formula.
    """
    m_blk = b.div(lane, b.const_i32(32))
    n = b.mod(lane, b.const_i32(32))
    _row, col = C32_DIST.calculate_x(
        b, ys=[b.const_i32(0), b.const_i32(0)], ps=[[m_blk, n]]
    )
    if n_tile32 == 0:
        return col
    return b.add(b.const_i32(n_tile32 * 32), col)


@dataclass(frozen=True)
class TiledLaneMap:
    """Per-lane row map and register-P helpers of the tiled 2D kernels.

    Holds the lane decomposition the 2D builder computes once per kernel; the
    methods emit IR only when called, in the same order as the inline
    closures they replace. ``use_mfma_32x32`` and ``transposed_qk_32x32``
    select the softmax-state row layout.
    """

    b: IRBuilder
    lane: Value
    lane_rg: Value
    lane_col_div4: Value
    lane_col_mod4: Value
    lane_rg_is0: Value
    lane_rg_is1: Value
    lane_rg_is2: Value
    dtype: Type
    use_mfma_32x32: bool
    transposed_qk_32x32: bool

    # For ``block_m_per_warp=16`` the lane owns 4 row slots within one
    # 16-row atom (rows ``lane_rg*4 + r`` for ``r in 0..3``). For
    # ``block_m_per_warp=32`` the lane owns 8 row slots across 2
    # stacked atoms: reg ``r`` maps to ``(atom_idx, in_atom) =
    # (r // 4, r % 4)`` and the in-warp row is
    # ``atom_idx * 16 + lane_rg * 4 + in_atom``.
    def in_warp_row(self, r: int) -> Value:
        b = self.b
        atom_idx = r // 4
        in_atom = r % 4
        return b.add(
            b.mul(self.lane_rg, b.const_i32(4)),
            b.const_i32(atom_idx * 16 + in_atom),
        )

    def state_row(self, r: int) -> Value:
        """Row represented by softmax state slot ``r`` in the active layout."""
        if self.use_mfma_32x32 and not self.transposed_qk_32x32:
            return mfma_32x32_c_row(self.b, self.lane, r)
        return self.in_warp_row(r)

    def bit2(self, v: Value, bit: int) -> Value:
        b = self.b
        return b.land(b.lshr(v, b.const_i32(bit)), b.const_i32(1))

    def select_lane_rg(self, v0: Value, v1: Value, v2: Value, v3: Value) -> Value:
        b = self.b
        return b.select(
            self.lane_rg_is0,
            v0,
            b.select(self.lane_rg_is1, v1, b.select(self.lane_rg_is2, v2, v3)),
        )

    def permute_p_c_to_a16(self, p_regs_f32: list[Value]) -> list[Value]:
        """Convert one 16-col P tile from 16x16 MFMA-C regs to PV-A regs.

        Current production geometry (not the 32x32 migration) has QK C
        indexed by lane fields ``(A=lane/16, B=(lane%16)/4, C=lane%4)``
        and per-lane register ``R``:

            P[row=A*4+R, col=B*4+C]

        PV's A operand wants:

            P[row=B*4+C, col=A*4+R]

        so the transform is ``(A,B,C,R) -> (B,A,R,C)``. This is the same
        register-P idea CK Tile uses to avoid an LDS bridge between two
        GEMMs. Two bit-level transposes swap ``C`` with ``R`` inside a
        lane quad; then lane-field swaps exchange ``A`` and ``B``.
        """
        b = self.b
        vals = p_regs_f32
        for bit in (0, 1):
            lane_bit = self.bit2(self.lane_col_mod4, bit)
            reg_bit = 1 << bit  # 1 or 2: intra-quad -> dpp (quad_perm), no ds_swizzle
            old = vals
            vals = []
            for reg in range(4):
                partner = b.warp_shuffle_xor_dpp(old[reg ^ reg_bit], reg_bit)
                same_bit = b.cmp_eq(lane_bit, b.const_i32((reg >> bit) & 1))
                vals.append(b.select(same_bit, old[reg], partner))

        for bit, lane_xor in ((0, 20), (1, 40)):
            a_bit = self.bit2(self.lane_rg, bit)
            b_bit = self.bit2(self.lane_col_div4, bit)
            swap = b.cmp_ne(a_bit, b_bit)
            vals = [b.select(swap, b.warp_shuffle_xor(v, lane_xor), v) for v in vals]
        return vals

    def pack_p_a16(self, p_regs_f32: list[Value]) -> Value:
        b = self.b
        return b.vec_pack(
            [b.cast_f32_to(v, self.dtype) for v in self.permute_p_c_to_a16(p_regs_f32)],
            self.dtype,
        )

    def pack_p_a32(self, p_regs0_f32: list[Value], p_regs1_f32: list[Value]) -> Value:
        b = self.b
        vals0 = self.permute_p_c_to_a16(p_regs0_f32)
        vals1 = self.permute_p_c_to_a16(p_regs1_f32)
        lo: list[Value] = []
        hi: list[Value] = []
        for j in range(4):
            v0 = vals0[j]
            v1 = vals1[j]
            v0_x16 = b.warp_shuffle_xor(v0, 16)
            v0_x32 = b.warp_shuffle_xor(v0, 32)
            v0_x48 = b.warp_shuffle_xor(v0, 48)
            v1_x16 = b.warp_shuffle_xor(v1, 16)
            v1_x32 = b.warp_shuffle_xor(v1, 32)
            v1_x48 = b.warp_shuffle_xor(v1, 48)
            lo.append(self.select_lane_rg(v0, v0_x48, v1_x32, v1_x16))
            hi.append(self.select_lane_rg(v0_x16, v0_x32, v1_x48, v1))
        return b.vec_pack([b.cast_f32_to(v, self.dtype) for v in (lo + hi)], self.dtype)


@dataclass(frozen=True)
class AccLayout:
    """Flat index of the PV accumulators carried through the KV loop.

    Accumulators are stored per ``(N-tile, M-atom)`` at ``n * acc_m_atoms +
    atom``. Pure host-side indexing; emits no IR.
    """

    acc_m_atoms: int

    def idx(self, n: int, atom: int) -> int:
        return n * self.acc_m_atoms + atom

    def getter(self, vals: list[Value]) -> Callable[[int, int], Value]:
        """Return ``get(n, atom)`` reading the accumulator list ``vals``."""
        return lambda n, atom: vals[self.idx(n, atom)]
