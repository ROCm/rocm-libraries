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
from typing import Callable, Optional

from kernels.common._fmha_common import gqa_ratio
from kernels.common._spec_checks import require_bool
from rocke.core.ir import BF16, F16, F32, FP8E4M3, I32, IRBuilder, Type, Value
from rocke.helpers.atoms import MfmaAtom, make_c_warp_dstr_encoding
from rocke.helpers.distribution import make_static_tile_distribution

__all__ = [
    "C16_DIST",
    "C32_DIST",
    "AccLayout",
    "Fp8KvSyncLoader",
    "KvLdsSwizzle",
    "ReduceTiledSpecMixin",
    "Tiled2DSpecMixin",
    "Tiled3DSpecMixin",
    "TiledDtypeIrMixin",
    "TiledGqaMixin",
    "TiledLaneMap",
    "dequant_fp8_lds_to_bf16",
    "fast_paged_kv_blocks",
    "load_page_id_window",
    "mfma_16x16_c_row",
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


# CK-Tile C-accumulator warp distribution for the 16x16x16 MFMA atom. For one
# wavefront the lane splits as ``(m_blk, n) = (lane // 16, lane % 16)`` and the
# per-lane ``<4 x f32>`` accumulator slot ``reg`` is the inner M Y dim
# (``kCM1PerLane = 4``), so ``calculate_x`` returns ``row = m_blk * 4 + reg``
# (and ``col = n``). The C layout is dtype-independent, so the f16 atom drives
# it for fp16 and bf16. Used by the single-warp tiled 3D segment kernels.
C16_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.f16_16x16x16())
)


def mfma_16x16_c_row(b, lane, reg: int):
    """MFMA-local output row for a ``16x16`` C element ``reg`` (0..3).

    Drives CK Tile's C-warp ``TileDistribution.calculate_x`` instead of the
    hand-rolled ``row = (lane // 16) * 4 + reg`` lane arithmetic. The 16x16 C
    layout has a single inner M Y dim of length 4 (``kCM0PerLane = 1``), so the
    outer Y is always 0 and ``reg`` is the inner Y.
    """
    if not (0 <= reg < 4):
        raise ValueError(f"mfma_16x16 reg must be 0..3, got {reg}")
    m_blk = b.div(lane, b.const_i32(16))
    n = b.mod(lane, b.const_i32(16))
    row, _col = C16_DIST.calculate_x(
        b, ys=[b.const_i32(0), b.const_i32(reg)], ps=[[m_blk, n]]
    )
    return row


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


@dataclass(frozen=True)
class Fp8KvSyncLoader:
    """Sync fp8 K/V cache loaders of the tiled 2D kernels.

    ``issue_dequant_loads`` loads fp8 K or V from the paged cache,
    dequantises it and stores the working dtype to LDS;
    ``issue_v_mfma_stripe`` stores raw fp8 V into the striped LDS layout the
    native-fp8 PV path reads. Fields are the builder values the inline
    closures used to capture; only valid when the spec stores K/V as fp8.
    """

    b: IRBuilder
    hd: int
    threads: int
    tid: Value
    key: Value
    value: Value
    k_scale: Value
    v_scale: Value
    k_lds: object
    v_lds: object
    kv_head_idx: Value
    paged_kv_desc: object
    dtype: Type
    i64_kv_addr: bool
    elems_per_chunk: int
    chunks_per_thread: int

    def issue_dequant_loads(
        self, kv_tile_idx: Value, buf_idx: Value, lds_token: str
    ) -> None:
        """Sync per-thread fp8 -> f32 -> *scale -> bf16/fp16 -> LDS.

        ``lds_token`` is either ``"K"`` or ``"V"``; selects the LDS slab
        and the per-tensor scale parameter.

        Per-chunk serial structure: each iteration issues one VMEM load,
        does the dequant chain, then publishes to LDS. The AMDGPU
        compiler's instruction scheduler interleaves the back-to-back
        chunks' loads, dequant, and stores.

        Cvt chain: uses the packed ``cvt_pk_f32_fp8x4`` primitive
        (lowers to packed ``v_cvt_pk_f32_fp8`` instructions) instead of
        the per-element ``cvt_fp8_to_f32``. The compiler does not fuse
        per-element fp8 converts into the packed form on its own, so the
        loader requests the packed form explicitly; it converts two fp8
        values per instruction instead of one. The packed primitive
        matches AITER's ``to_float_fp8x4`` helper in
        ``csrc/include/attention_common.cuh``, which AITER's
        paged-attention FP8 K/V path uses.

        Each 8-fp8 chunk is split into two 4-fp8 halves, and each half
        goes through one packed cvt.
        """
        b = self.b
        scale = self.k_scale if lds_token == "K" else self.v_scale
        lds = self.k_lds if lds_token == "K" else self.v_lds
        src = self.key if lds_token == "K" else self.value
        assert self.elems_per_chunk == 8, (
            f"FP8 dequant loader expects 8-elem chunks (got "
            f"{self.elems_per_chunk}); the packed-cvt path needs the chunk "
            f"to split cleanly into 2× <4 x fp8> sub-chunks."
        )
        for call in range(self.chunks_per_thread):
            chunk_id = b.add(
                b.mul(b.const_i32(call), b.const_i32(self.threads)),
                self.tid,
            )
            row = b.div(chunk_id, b.const_i32(self.hd // self.elems_per_chunk))
            col = b.mul(
                b.mod(chunk_id, b.const_i32(self.hd // self.elems_per_chunk)),
                b.const_i32(self.elems_per_chunk),
            )
            linear_half_first = b.add(
                b.mul(row, b.const_i32(self.hd)),
                col,
            )
            if self.i64_kv_addr:
                # Flat per-lane global load: full i64 byte/element offset so
                # the per-thread physical_block * stride cannot overflow the
                # cache addressing for paged caches > 2 GiB.
                voff, _ = self.paged_kv_desc.offset_i64(
                    b,
                    "physical_block",
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half_first,
                    kv_head=self.kv_head_idx,
                )
            else:
                voff, _ = self.paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half_first,
                    kv_head=self.kv_head_idx,
                )
            fp8_vec = b.global_load_vN(
                src, voff, FP8E4M3, n=self.elems_per_chunk, align=self.elems_per_chunk
            )
            # Split <8 x fp8> into 2x <4 x fp8> via extract+pack so we
            # can apply the packed cvt (the LLVM backend collapses the
            # extract+pack chain back into a bitcast-equivalent shuffle
            # once it sees the cvt.pk.f32.fp8 operand is whole-i32).
            lo_quad = b.vec_pack(
                [b.vec_extract(fp8_vec, i) for i in range(4)],
                FP8E4M3,
            )
            hi_quad = b.vec_pack(
                [b.vec_extract(fp8_vec, i) for i in range(4, 8)],
                FP8E4M3,
            )
            # CORRECTNESS: ``v_cvt_scalef32_pk_f32_fp8`` interprets its
            # ``scale`` operand as an E8M0 microscaling factor (only the
            # f32 exponent bits matter, mantissa is discarded). For
            # arbitrary per-tensor scales like ``k_scale = max_abs/448 ≈
            # 0.011`` the hardware silently rounds to the nearest
            # power-of-two (0.0078 instead of 0.011), giving outputs
            # ~0.71x of expected. Power-of-two scales happen to work
            # bit-correctly by coincidence, so a check that only uses
            # power-of-two scales does not catch the error.
            #
            # The correct sequence is unfused: ``cvt_pk_f32_fp8`` (no
            # scaling) followed by a regular ``v_pk_mul`` against the
            # f32 scale. The extra packed multiplies are the cost of
            # being correct for non-pow2 scales. A future fast path can
            # detect pow2 scales and opt into the fused intrinsic.
            lo_f32x4 = b.cvt_pk_f32_fp8x4(lo_quad)
            hi_f32x4 = b.cvt_pk_f32_fp8x4(hi_quad)
            lo_scaled = []
            hi_scaled = []
            for i in range(4):
                lo_scaled.append(b.fmul(b.vec_extract(lo_f32x4, i), scale))
                hi_scaled.append(b.fmul(b.vec_extract(hi_f32x4, i), scale))
            lo_f32x4 = b.vec_pack(lo_scaled, F32)
            hi_f32x4 = b.vec_pack(hi_scaled, F32)
            dequanted = []
            for i in range(4):
                dequanted.append(b.cast_f32_to(b.vec_extract(lo_f32x4, i), self.dtype))
            for i in range(4):
                dequanted.append(b.cast_f32_to(b.vec_extract(hi_f32x4, i), self.dtype))
            packed = b.vec_pack(dequanted, self.dtype)
            b.smem_store_vN(lds, [buf_idx, row, col], packed, self.elems_per_chunk)
        # Caller is expected to issue a `b.sync()` (LDS visibility) when
        # appropriate. The sync loader has no in-flight async work, so the
        # consumer side only needs the LDS barrier, not VMEM waitcnt.

    def issue_v_mfma_stripe(self, kv_tile_idx: Value) -> None:
        """Load raw fp8 V and store it into V_lds as
        [V_BUFS=1, N_STRIPES=HD/16, T, 16].

        Each thread loads 8 contiguous fp8 values from V[token, col..col+7]
        in HBM, then writes them as one 8-byte LDS vec store into the
        owning stripe at V_lds[0, col/16, token, col%16].

        Because our chunk size is exactly 8 fp8 (= half a 16-byte stripe
        row), col%16 is always 0 or 8 — both 8-byte-aligned, so the LDS
        vec store stays b64 and there is no scalar-byte regression.
        """
        b = self.b
        for call in range(self.chunks_per_thread):
            chunk_id = b.add(
                b.mul(b.const_i32(call), b.const_i32(self.threads)),
                self.tid,
            )
            token = b.div(chunk_id, b.const_i32(self.hd // self.elems_per_chunk))
            col = b.mul(
                b.mod(chunk_id, b.const_i32(self.hd // self.elems_per_chunk)),
                b.const_i32(self.elems_per_chunk),
            )
            linear_first = b.add(b.mul(token, b.const_i32(self.hd)), col)
            voff, _ = self.paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_first,
                kv_head=self.kv_head_idx,
            )
            fp8_vec = b.global_load_vN(
                self.value,
                voff,
                FP8E4M3,
                n=self.elems_per_chunk,
                align=self.elems_per_chunk,
            )
            stripe_idx = b.div(col, b.const_i32(16))
            col_in_stripe = b.mod(col, b.const_i32(16))
            b.smem_store_vN(
                self.v_lds,
                [b.const_i32(0), stripe_idx, token, col_in_stripe],
                fp8_vec,
                self.elems_per_chunk,
            )


@dataclass(frozen=True)
class KvLdsSwizzle:
    """Optional 16-byte-chunk XOR swizzle of the tiled 3D K/V LDS rows.

    With ``swizzle`` off both methods are identities and emit no IR.
    """

    b: IRBuilder
    hd: int
    swizzle: bool
    chunks_per_row: int

    def col(self, row: Value, col: Value) -> Value:
        # LDS column of K/V element (row, col): its 16-byte chunk is permuted
        # by the row under the swizzle.
        if not self.swizzle:
            return col
        b = self.b
        chunk = b.xor(
            b.div(col, b.const_i32(8)), b.mod(row, b.const_i32(self.chunks_per_row))
        )
        return b.add(b.mul(chunk, b.const_i32(8)), b.mod(col, b.const_i32(8)))

    def src_linear(self, linear_half: Value) -> Value:
        # The async copies write LDS in lane order, so under the swizzle each
        # lane fetches the element its LDS position holds (col is its own
        # inverse).
        if not self.swizzle:
            return linear_half
        b = self.b
        row = b.div(linear_half, b.const_i32(self.hd))
        return b.add(
            b.mul(row, b.const_i32(self.hd)),
            self.col(row, b.mod(linear_half, b.const_i32(self.hd))),
        )


class TiledDtypeIrMixin:
    """``dtype_ir`` of the tiled attention specs (2D, 3D segment, reduce)."""

    dtype: str

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16


class TiledGqaMixin(TiledDtypeIrMixin):
    """GQA geometry shared by the tiled 2D and 3D segment specs.

    ``block_q`` reads ``block_m``, which each spec family defines.
    """

    num_query_heads: int
    num_kv_heads: int

    @property
    def num_queries_per_kv(self) -> int:
        return gqa_ratio(self.num_query_heads, self.num_kv_heads)

    @property
    def block_q(self) -> int:
        return self.block_m // self.num_queries_per_kv


class Tiled2DSpecMixin(TiledGqaMixin):
    """Derived tile geometry of the tiled 2D spec."""

    block_m_per_warp: int
    num_warps: int
    tile_size: Optional[int]
    block_size: int

    @property
    def block_m(self) -> int:
        return self.block_m_per_warp * self.num_warps

    @property
    def tile_size_eff(self) -> int:
        """Effective per-iter KV-tile size in tokens."""
        return self.tile_size if self.tile_size is not None else self.block_size

    @property
    def n_blocks_per_tile(self) -> int:
        """How many paged-KV cache blocks one kernel iter consumes."""
        return self.tile_size_eff // self.block_size


class Tiled3DSpecMixin(TiledGqaMixin):
    """Derived tile geometry of the tiled 3D segment spec."""

    @property
    def block_m(self) -> int:
        return 16


class ReduceTiledSpecMixin(TiledDtypeIrMixin):
    """Validation shared by the tiled 3D reduce specs."""

    emit_lse: bool
    num_segments: int

    def __post_init__(self) -> None:
        require_bool("emit_lse", self.emit_lse)
        if self.num_segments < 1:
            raise ValueError("num_segments must be >= 1")


def fast_paged_kv_blocks(
    b: IRBuilder,
    kv_tile_idx: Value,
    *,
    block_tables: Value,
    seq_base: Value,
    block_table_max_idx: Value,
) -> tuple[Value, Value]:
    """Physical page ids of the two paged-KV blocks of KV tile ``kv_tile_idx``.

    The fast paged-KV descriptor covers each tile with two cache blocks. Loads
    past ``block_table_max_idx`` read 0. Returns both ids as SGPR values.
    """
    logical_block0 = b.mul(kv_tile_idx, b.const_i32(2))
    logical_block1 = b.add(logical_block0, b.const_i32(1))
    idx0 = b.add(seq_base, logical_block0)
    idx1 = b.add(seq_base, logical_block1)
    block0 = b.masked_global_load(
        block_tables,
        idx0,
        b.cmp_lt(idx0, block_table_max_idx),
        b.const_i32(0),
        dtype=I32,
        align=4,
    )
    block1 = b.masked_global_load(
        block_tables,
        idx1,
        b.cmp_lt(idx1, block_table_max_idx),
        b.const_i32(0),
        dtype=I32,
        align=4,
    )
    return b.to_sgpr_u32(block0), b.to_sgpr_u32(block1)


def dequant_fp8_lds_to_bf16(
    b: IRBuilder,
    buf_idx: Value,
    scale: Value,
    fp8_lds,
    bf16_lds,
    bf16_buf: Value,
    *,
    tid: Value,
    threads: int,
    dtype: Type,
    elems_per_chunk: int,
    chunks_per_thread: int,
    cols_per_row: int,
) -> None:
    """LDS->LDS dequant: fp8 -> f32 * scale -> bf16.

    `buf_idx` selects the fp8 source buffer (for K, the active
    double-buffer slot; for V, always 0).
    `bf16_buf` selects the bf16 destination buffer (for K, always
    0 since FP8 path single-buffers the bf16 K_lds; for V, 0).
    """
    for c in range(chunks_per_thread):
        chunk_id = b.add(b.mul(b.const_i32(c), b.const_i32(threads)), tid)
        row = b.div(chunk_id, b.const_i32(cols_per_row))
        col = b.mul(
            b.mod(chunk_id, b.const_i32(cols_per_row)),
            b.const_i32(elems_per_chunk),
        )
        fp8_vec = b.smem_load_vN(
            fp8_lds,
            buf_idx,
            row,
            col,
            dtype=FP8E4M3,
            n=elems_per_chunk,
        )
        dequanted = []
        for i in range(elems_per_chunk):
            fp8_v = b.vec_extract(fp8_vec, i)
            f32_v = b.fmul(b.cvt_fp8_to_f32(fp8_v), scale)
            dequanted.append(b.cast_f32_to(f32_v, dtype))
        packed = b.vec_pack(dequanted, dtype)
        b.smem_store_vN(
            bf16_lds,
            [bf16_buf, row, col],
            packed,
            elems_per_chunk,
        )


def load_page_id_window(
    b: IRBuilder,
    rel_page: Value,
    *,
    tid: Value,
    block_tables: Value,
    seq_base: Value,
    seq_pages: Value,
    seg_first_page: Value,
) -> Value:
    """Lane l: the id of the segment's page ``rel_page + l``.

    Past the sequence's last page the lane reads 0, which is never used.
    ``seg_first_page`` is the segment's first page, which each arch's tiled
    3D builder computes for its own grid.
    """
    pg = b.add(b.add(seg_first_page, rel_page), tid)
    return b.masked_global_load(
        block_tables,
        b.add(seq_base, pg),
        b.cmp_lt(pg, seq_pages),
        b.const_i32(0),
        dtype=I32,
        align=4,
    )
