# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-launch fused-MoE MEGA-kernel (FP8 e4m3 block-scale).

The fp8 mega-kernel snapshot for this level. A SINGLE
fused kernel computes, per (inter-slice, sorted-m-block) threadgroup, the full
MoE per-expert path with fp8 e4m3 operands and per-128-block f32 scales:

    GEMM0 (gate) + GEMM1 (up)  -- fp8 X . fp8 W -> f32 acc, per-128-group
        dequant fold (group-accumulator pattern, scale applied POST-MFMA)
      -> SiLU(gate_dq) * up_dq in f32
      -> DYNAMIC-QUANTIZE the f32 Hidden to fp8 (per-128-block-along-inter,
         per-row amax/448 scale), stage into a PERSISTENT LDS Hidden_smem,
         and stash the per-block dynamic scales in an LDS scratch.
      -> GEMM2 (down) -- fp8 Hidden . fp8 W_down -> f32 acc, dequant by
         (hidden_dyn_scale * down_scale) -> weighted atomic reduce into Y.

See ``examples/gfx950/fused_mega_moe/docs/BUILD_SPEC_FP8.md`` for the authoritative
build spec. The dequant ordering follows BUILD_SPEC_FP8 Section 1.2 (the
``block_scale_gemm.py`` group-accumulator pattern): within a 128-wide
contraction block the scales are constant, so
``sum_k (a.sa)(b.sb) = sa.sb . sum_k (a.b)`` -- the scale is applied per
K-group, post-MFMA, NOT in-instruction (which would mean the E8M0 trap of
``cvt_scalef32_pk_f32_fp8x4``).

STAGING STATUS (incremental implementation per BUILD_SPEC_FP8 Phase plan):

* STAGE 1 (THIS FILE, current state):
    - ``FusedMegaKernelSpecFp8`` (fp8 spec / signature / grid).
    - the gate+up fp8 GEMM via ``_emit_fp8_gateup_group_gemm`` (fp8 atom,
      fp8 operand loads, per-128-block scale dequant of BOTH accumulators).
    - SiLU(gate)*up in f32.
    - ``_emit_hidden_dyn_quant_stage``: per-(row, 128-inter-block) amax ->
      dynamic scale -> quantize the f32 Hidden to fp8 and stage into the
      persistent LDS ``Hidden_smem`` + stash the per-block scales in
      ``HiddenScale_smem``.
* STAGE 2 (THIS FILE, current state): the fp8 down GEMM reading ``Hidden_smem``
  as the LDS-resident A operand (``make_transposed``-equivalent implicit reshape
  -- the dynamic-quant write addr == the down MFMA A-read addr, same logical
  ``(m, inter)`` cell, BUILD_SPEC_FP8 Section 3.5), with the per-128-group
  group-accumulator dequant by (``HiddenScale_smem[row, blk]`` * ``down_scale``),
  multiply by the sorted per-token weight, and atomic-add the f32 partial into Y
  (padded rows token id -1 / >= tokens are skipped). Tiled over the H_out output
  in ``tile_n_down`` chunks; grid.x split the inter contraction so each TG
  atomic-adds a PARTIAL Y over the whole H_out.

The fp8 MFMA atom + cvt primitives are 100% reused from ``helpers/atoms.py``
and ``core/ir.py`` (never modified). This file builds the gate+up fp8 GEMM and
the dynamic-quant staging from the fp8-aware ``helpers/mfma_gemm_inner.py``
toolkit (which dispatches the fp8 atom directly) rather than the
``UniversalGemmSpec``-bound f16/bf16 helpers (whose ``io_ir_type`` rejects fp8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import (
    F32,
    FP8E4M3,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ...helpers.atoms import MfmaAtom
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    validate_arch_and_block_size,
    validate_mfma_atom_in_catalog,
)
from ...helpers.quant import quant_max_abs
from ...helpers.tensor_view import TensorDescriptor, TensorView


__all__ = [
    "FusedMegaKernelSpecFp8",
    "build_moe_fused_mega_gemm_fp8",
    "moe_fused_mega_fp8_grid",
    "moe_fused_mega_fp8_signature",
]


# Group block along the contraction axis (= 4 fp8_16x16x32 atoms).
GROUP_K = 128
# fp8e4m3 saturating clamp magnitude.
FP8_MAX = quant_max_abs("fp8e4m3")  # 448.0
# hand-tuned asm dynamic-quant amax floor.
AMAX_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedMegaKernelSpecFp8:
    """Single-launch fused-MoE mega-kernel spec (FP8 e4m3 block-scale).

    Tile geometry is IDENTICAL to the f16 :class:`FusedMegaKernelSpec`
    (BUILD_SPEC_FP8 Section 2.0): the fp8 atom is also K=32 and the per-lane
    fragment width is the same, so only the operand element dtype (2B->1B) and
    the MFMA intrinsic change.

    * ``tile_m`` = sorted tokens per m-block (hand-tuned asm ``sub_x``), default 32.
    * ``tile_n_inter`` = inter columns this TG owns (hand-tuned asm ``sub_gu``); the
      GEMM0/1 N extent AND the GEMM2 contraction extent, default 256.
    * ``tile_k_gu`` = K-loop tile along the hidden contraction H for gate/up.
    * MFMA atom = fp8 ``16x16x32`` (e4m3); the 128-wide scale group = 4 atoms.
    """

    name: str
    tile_m: int = 32
    tile_n_inter: int = 256
    tile_k_gu: int = 32
    warp_m: int = 1
    warp_n: int = 4
    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 32
    tile_n_down: int = 256
    tile_k_down: int = 64
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp8e4m3"

    def __post_init__(self) -> None:
        if self.block_size == 0:
            object.__setattr__(
                self,
                "block_size",
                self.warp_m * self.warp_n * self.wave_size,
            )

    # -- derived geometry helpers ----------------------------------------

    def gate_up_atom(self) -> MfmaAtom:
        return MfmaAtom.fp8_16x16x32()

    def down_atom(self) -> MfmaAtom:
        return MfmaAtom.fp8_16x16x32()

    @property
    def mfmas_m(self) -> int:
        """Atoms along M per warp for the gate/up tile."""
        return (self.tile_m // self.warp_m) // self.warp_tile_m

    @property
    def mfmas_n(self) -> int:
        """Atoms along N per warp for the gate/up tile."""
        return (self.tile_n_inter // self.warp_n) // self.warp_tile_n

    @property
    def mfmas_m_down(self) -> int:
        """Atoms along M per warp for the down tile (M x H_out_slice)."""
        return (self.tile_m // self.warp_m) // self.warp_tile_m

    @property
    def mfmas_n_down(self) -> int:
        """Atoms along N (= H_out output) per warp for the down tile."""
        return (self.tile_n_down // self.warp_n) // self.warp_tile_n

    def kernel_name(self) -> str:
        return (
            f"{self.name}_moe_fused_mega_fp8_"
            f"m{self.tile_m}n{self.tile_n_inter}k{self.tile_k_gu}"
        )


# ---------------------------------------------------------------------------
# Grid + signature (BUILD_SPEC_FP8 Section 2.7 / 2.8)
# ---------------------------------------------------------------------------


def moe_fused_mega_fp8_grid(
    num_m_blocks: int, inter: int, spec: FusedMegaKernelSpecFp8
) -> Tuple[int, int, int]:
    """Mega-kernel launch grid (unchanged from the f16 kernel).

    ``grid = (ceil(inter / tile_n_inter), num_m_blocks, 1)``. Canonical decode
    (I=7168, tile_n_inter=256): grid.x = 28; grid.y = num_m_blocks = 8.
    """
    sub_gu = spec.tile_n_inter
    gx = (inter + sub_gu - 1) // sub_gu
    return (gx, num_m_blocks, 1)


def moe_fused_mega_fp8_signature(spec: FusedMegaKernelSpecFp8):
    from ...helpers.spec import SignatureBuilder

    return (
        SignatureBuilder()
        .ptr(
            "A", "fp8e4m3"
        )  # quantized activation X (hand-tuned asm: fp8 + input_scale)
        .ptr("WGate", "fp8e4m3")
        .ptr("WUp", "fp8e4m3")
        .ptr("WDown", "fp8e4m3")
        .ptr("AScale", "f32")  # input_scale, per (token-block, H-block-of-128)
        .ptr("WGateScale", "f32")  # fc1_scale gate half, per (E, I-block, H-block)
        .ptr("WUpScale", "f32")  # fc1_scale up half
        .ptr("WDownScale", "f32")  # fc2_scale, per (E, H_out-block, I-block)
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("BlockExpertIds", "i32")
        .ptr("Y", "f32")  # f32 to reuse the f16 atomic epilogue unchanged
        .scalar("M", "i32")
        .scalar("N", "i32")  # = I (inter dim)
        .scalar("K", "i32")  # = H (hidden contraction)
        .scalar("H_out", "i32")  # = H (down output)
        .scalar("stride_a", "i32")
        .scalar("stride_b_gate", "i32")
        .scalar("stride_b_up", "i32")
        .scalar("stride_b_down", "i32")
        .scalar("stride_a_scale", "i32")
        .scalar("stride_gate_scale", "i32")
        .scalar("stride_up_scale", "i32")
        .scalar("stride_down_scale", "i32")
        .scalar("stride_gate_scale_e", "i32")
        .scalar("stride_up_scale_e", "i32")
        .scalar("stride_down_scale_e", "i32")
        .scalar("slot_size", "i32")
        .scalar("tokens", "i32")
        .build()
    )


# ---------------------------------------------------------------------------
# STAGE 1a: gate+up fp8 GEMM with per-128-block dequant
# ---------------------------------------------------------------------------


def _emit_fp8_gateup_group_gemm(
    b: IRBuilder,
    *,
    A: Value,
    WGate: Value,
    WUp: Value,
    AScale: Value,
    WGateScale: Value,
    WUpScale: Value,
    atom: MfmaAtom,
    lane_decode,
    m_tile_base: Value,
    n_tile_base: Value,
    K: Value,
    stride_a_scale: Value,
    stride_gate_scale: Value,
    stride_up_scale: Value,
    tag: str,
) -> Tuple[Value, Value]:
    """Gate + up fp8 GEMM, returning ``(gate_dq, up_dq)`` per-lane f32 vectors.

    Group-accumulator pattern (BUILD_SPEC_FP8 Section 1.2): the outer loop walks
    128-wide groups along the hidden contraction K (= H). Per group, 4
    fp8_16x16x32 atoms accumulate into a FRESH ``group_acc`` (one for gate, one
    for up), then the group is folded into the outer accumulator scaled by
    ``a_scale * b_scale`` -- a single ``v_pk_fma_f32`` per accumulator. The
    A read (the quantized activation) is shared across gate and up; only the
    B-side weight scale differs.

    Scale index math (BUILD_SPEC_FP8 Section 1.3), per-128 along the
    contraction and per-128 along the output:

        a_scale_off = (m_row // GROUP_K_M) * k_scale_count + kg
        b_scale_off = kg * n_scale_count + (n_col // GROUP_K)

    Here the activation amax is per-(token-block, H-block); we use the per-row
    granularity available to the lane (``m_row``) with the H-block index ``kg``,
    and the weight scale is per (output-128, contraction-128).
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    atoms_per_group = GROUP_K // atom.k  # 4

    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    n_col = b.add(n_tile_base, lane_decode.n_in_atom)

    # Per-lane scale offsets vary along the K-group index ``kg`` only.
    a_row_scale_base = b.mul(m_row, stride_a_scale)
    n_blk = b.div(n_col, c_group_k)

    zero = atom.zero_acc(b)
    gate_zero = atom.zero_acc(b)
    up_zero = atom.zero_acc(b)

    # Outer loop over 128-wide contraction groups: num_groups = K // GROUP_K.
    num_groups = b.div(K, c_group_k)
    outer = b.scf_for_iter(
        b.const_i32(0),
        num_groups,
        b.const_i32(1),
        [(f"gate_outer_{tag}", gate_zero), (f"up_outer_{tag}", up_zero)],
        iv_name=f"kg_{tag}",
    )
    with outer as (kg, (gate_outer, up_outer)):
        # Per-group scales (one f32 each).
        a_scale_off = b.add(a_row_scale_base, kg)
        a_scale_v = b.global_load_f32(AScale, a_scale_off)
        gate_scale_off = b.add(b.mul(kg, stride_gate_scale), n_blk)
        up_scale_off = b.add(b.mul(kg, stride_up_scale), n_blk)
        gate_scale_v = b.global_load_f32(WGateScale, gate_scale_off)
        up_scale_v = b.global_load_f32(WUpScale, up_scale_off)
        gate_ab = b.fmul(a_scale_v, gate_scale_v)
        up_ab = b.fmul(a_scale_v, up_scale_v)

        k_group_base = b.mul(kg, c_group_k)

        # Fresh per-group accumulators; 4 fp8 atoms cover the 128-wide group.
        ginner = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(atoms_per_group),
            b.const_i32(1),
            [(f"g_acc_{tag}", zero), (f"u_acc_{tag}", zero)],
            iv_name=f"kk_{tag}",
        )
        with ginner as (kk, (g_acc, u_acc)):
            k_tile_base = b.add(k_group_base, b.mul(kk, c_atom_k))
            a_frag = _load_a_fp8(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=k_tile_base,
                K=K,
            )
            gb_frag = _load_b_fp8(
                b,
                B=WGate,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=k_tile_base,
                N=K,
            )
            ub_frag = _load_b_fp8(
                b,
                B=WUp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=k_tile_base,
                N=K,
            )
            g_new = atom.emit(b, a_frag, gb_frag, g_acc)
            u_new = atom.emit(b, a_frag, ub_frag, u_acc)
            b.scf_yield(g_new, u_new)
        group_gate = ginner.results[0]
        group_up = ginner.results[1]

        # Fold (post-MFMA, per-group): outer += group * (a_scale * b_scale).
        gate_scale_vec = b.vector_splat(gate_ab, atom.c_per_lane)
        up_scale_vec = b.vector_splat(up_ab, atom.c_per_lane)
        gate_outer_new = b.vector_fma(group_gate, gate_scale_vec, gate_outer)
        up_outer_new = b.vector_fma(group_up, up_scale_vec, up_outer)
        b.scf_yield(gate_outer_new, up_outer_new)

    return outer.results[0], outer.results[1]


def _load_a_fp8(
    b: IRBuilder, *, A, atom, lane_decode, m_tile_base, k_tile_base, K
) -> Value:
    """Per-lane fp8 A load for row-major (M, K) -- K contiguous; scalar loads."""
    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    a_addr = b.add(b.mul(m_row, K), k_base)
    # The a_per_lane fp8 bytes are CONTIGUOUS along K (addr + j) -> one
    # coalesced vector load (global_load_dwordx2 for n=8) instead of
    # a_per_lane byte-granular global_load_ubyte. Bit-identical values.
    return b.global_load_vN(A, a_addr, FP8E4M3, atom.a_per_lane)


def _load_b_fp8(
    b: IRBuilder, *, B, atom, lane_decode, n_tile_base, k_tile_base, N
) -> Value:
    """Per-lane fp8 B load for row-major (K, N) -- col-strided scalar loads.

    ``N`` is the weight's K extent (the matrix is stored (out, K) row-major;
    B[k, n] address = (n_tile_base + n_in_atom) * K + (k_base + j)).
    """
    n_col = b.add(n_tile_base, lane_decode.n_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    # W is stored (out_rows, K) row-major -> address of W[n_col, k] = n_col*K + k.
    row_base = b.mul(n_col, N)
    # The b_per_lane fp8 weight bytes are CONTIGUOUS along K (k_base + j) ->
    # one coalesced vector load (global_load_dwordx2 for n=8) instead of
    # b_per_lane byte-granular global_load_ubyte. This is the dominant HBM
    # weight traffic; bit-identical values.
    b_addr = b.add(row_base, k_base)
    return b.global_load_vN(B, b_addr, FP8E4M3, atom.b_per_lane)


# ---------------------------------------------------------------------------
# STAGE 2: down fp8 GEMM (LDS-A) + per-128-group dequant + weighted atomic Y
# ---------------------------------------------------------------------------


def _load_a_fp8_lds(
    b: IRBuilder, *, a_view, atom, lane_decode, m_tile_base, k_tile_base
) -> Value:
    """Per-lane fp8 A load for the down GEMM, reading the LDS-resident Hidden.

    The down-GEMM A operand is ``Hidden_smem`` (``[tile_m, tile_n_inter]``,
    logical ``(row=m, col=inter)``). The contraction axis is the inter slice, so
    for output row ``m_row`` (= ``m_tile_base + m_in_atom``) the lane reads its
    ``a_per_lane`` contiguous fp8 elements starting at inter column
    ``k_tile_base + k_blk * a_per_lane``. This is the SAME logical ``(m, inter)``
    cell the dynamic-quant Pass C wrote (BUILD_SPEC_FP8 Section 3.5: the implicit
    reshape -- cshuffle/quant write addr == down MFMA A-read addr).
    """
    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
    k_col = b.add(k_tile_base, k_lane_start)
    return b.smem_load_vN(a_view.base, m_row, k_col, dtype=FP8E4M3, n=atom.a_per_lane)


def _emit_fp8_down_group_gemm(
    b: IRBuilder,
    *,
    a_view,
    WDown: Value,
    WDownScale: Value,
    atom: MfmaAtom,
    lane_decode,
    n_tile_base: Value,
    scale_view,
    inter_slice: int,
    inter_full: Value,
    inter_blk_base: Value,
    stride_down_scale: Value,
    tag: str,
) -> Value:
    """Down fp8 GEMM for one warp-atom output cell -> per-lane f32 vector.

    Group-accumulator pattern (BUILD_SPEC_FP8 Section 1.2 / 2.2): the outer loop
    walks 128-wide groups along the inter CONTRACTION (= this TG's inter slice,
    ``tile_n_inter`` = ``inter_slice``). Per group, 4 fp8_16x16x32 atoms
    accumulate into a FRESH ``group_acc``; the group is folded into the outer
    accumulator scaled by ``hidden_dyn_scale * down_b_scale`` -- a single
    ``v_pk_fma_f32``.

    Index conventions (the two-space split):
    * **A (Hidden, LDS):** read at LOCAL inter columns ``[0, inter_slice)``; its
      per-128-block dynamic scale is ``HiddenScale_smem[m_row, local_blk]`` (the
      dynamic scale is only defined over THIS TG's slice).
    * **B (W_down, global):** stored ``(H_out, I_full)`` row-major; the
      contraction column is the GLOBAL inter position ``inter_blk_base*128 + local``
      and the row stride is the FULL inter dim ``inter_full``. The B-side scale
      ``WDownScale`` is indexed per (GLOBAL inter-block, H_out-block):
      ``off = (inter_blk_base + kg) * stride_down_scale + h_out_blk``.

    The W_down per-expert pointer base is folded by the caller; the inter slice
    base is folded here (column ``inter_blk_base*128 + local``), NOT into the
    pointer, so the row stride math stays the full-inter stride.
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    atoms_per_group = GROUP_K // atom.k  # 4

    n_col = b.add(n_tile_base, lane_decode.n_in_atom)
    h_out_blk = b.div(n_col, c_group_k)
    m_row = b.add(b.const_i32(0), lane_decode.m_in_atom)

    # Global inter column base for this TG's slice (W_down contraction origin).
    inter_col_base = b.mul(inter_blk_base, c_group_k)

    zero = atom.zero_acc(b)
    outer_zero = atom.zero_acc(b)

    # num_groups = inter_slice // GROUP_K (local inter slice / 128).
    num_groups = b.const_i32(inter_slice // GROUP_K)
    outer = b.scf_for_iter(
        b.const_i32(0),
        num_groups,
        b.const_i32(1),
        [(f"down_outer_{tag}", outer_zero)],
        iv_name=f"dg_{tag}",
    )
    with outer as (kg, (down_outer,)):
        # A-side dynamic scale: HiddenScale_smem[m_row, local-inter-block kg].
        a_scale_v = b.vec_extract(
            b.smem_load_vN(scale_view.base, m_row, kg, dtype=F32, n=1), 0
        )
        # B-side W_down scale: per (GLOBAL inter-block, H_out-block).
        global_blk = b.add(inter_blk_base, kg)
        down_scale_off = b.add(b.mul(global_blk, stride_down_scale), h_out_blk)
        down_scale_v = b.global_load_f32(WDownScale, down_scale_off)
        ab_scale = b.fmul(a_scale_v, down_scale_v)

        local_k_group = b.mul(kg, c_group_k)
        global_k_group = b.add(inter_col_base, local_k_group)
        ginner = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(atoms_per_group),
            b.const_i32(1),
            [(f"d_acc_{tag}", zero)],
            iv_name=f"dk_{tag}",
        )
        with ginner as (kk, (d_acc,)):
            kk_off = b.mul(kk, c_atom_k)
            # A reads LOCAL inter columns from Hidden_smem.
            a_frag = _load_a_fp8_lds(
                b,
                a_view=a_view,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=b.const_i32(0),
                k_tile_base=b.add(local_k_group, kk_off),
            )
            # W_down reads GLOBAL inter columns with FULL inter row stride.
            b_frag = _load_b_fp8(
                b,
                B=WDown,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=b.add(global_k_group, kk_off),
                N=inter_full,
            )
            d_new = atom.emit(b, a_frag, b_frag, d_acc)
            b.scf_yield(d_new)
        group_acc = ginner.results[0]

        scale_vec = b.vector_splat(ab_scale, atom.c_per_lane)
        down_outer_new = b.vector_fma(group_acc, scale_vec, down_outer)
        b.scf_yield(down_outer_new)

    return outer.results[0]


def _emit_down_atomic_reduce(
    b: IRBuilder,
    *,
    atom: MfmaAtom,
    down_list,
    warp_m_off: Value,
    warp_n_off: Value,
    lane: Value,
    mfmas_m: int,
    mfmas_n: int,
    block_m_off: Value,
    ho_off: Value,
    H_out: Value,
    SortedTokenIds: Value,
    SortedWeights: Value,
    Y: Value,
    tokens: Value,
) -> None:
    """Weighted, token-validity-masked atomic reduce of the down result into Y.

    ``Y[token, h_out] += weight * down_dq`` (f32 atomic add). Padded rows
    (sorted token id < 0 or >= tokens) are skipped. Mirrors the f16
    ``_emit_down_reduce_epilogue_atomic`` but on the raw-atom lane layout used by
    the rest of this fp8 file (``atom.lane_to_output``). The sorted-token bucket
    index is the GLOBAL output row ``block_m_off + warp/atom row``; the output
    column is ``ho_off + warp/atom col`` along H_out.
    """
    c0 = b.const_i32(0)
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            flat = mi * mfmas_n + ni
            acc = down_list[flat]
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                row = b.add(
                    block_m_off,
                    b.add(warp_m_off, b.add(b.const_i32(mi * atom.m), row_in)),
                )
                col = b.add(
                    ho_off,
                    b.add(warp_n_off, b.add(b.const_i32(ni * atom.n), col_in)),
                )
                bucket = row
                token = b.global_load_i32(SortedTokenIds, bucket)
                valid = b.land(b.cmp_ge(token, c0), b.cmp_lt(token, tokens))
                with b.scf_if(valid):
                    w = b.global_load_f32(SortedWeights, bucket)
                    v = b.vec_extract(acc, i)
                    contrib = b.fmul(w, v)
                    y_off = b.add(b.mul(token, H_out), col)
                    b.global_atomic_add(Y, y_off, contrib)


# ---------------------------------------------------------------------------
# STAGE 1b: SiLU(gate)*up + dynamic-quantize Hidden to fp8 in LDS
# ---------------------------------------------------------------------------


def _silu_mul_f32(
    b: IRBuilder, g: Value, u: Value, *, one_f32: Value, c_neg_log2e: Value
) -> Value:
    """f32 SwiGLU chain ``silu(g) * u`` (sigmoid via exp2), op order matched."""
    sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
    silu = b.fmul(g, sig)
    return b.fmul(silu, u)


def _store_hidden_f32_pass(
    b: IRBuilder,
    *,
    atom: MfmaAtom,
    gate_list,
    up_list,
    f32_view,
    warp_m_off: Value,
    warp_n_off: Value,
    lane: Value,
    mfmas_m: int,
    mfmas_n: int,
    one_f32: Value,
    c_neg_log2e: Value,
) -> None:
    """Pass A: silu(gate)*up -> f32 LDS scratch at each MFMA output cell."""
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            flat = mi * mfmas_n + ni
            g_vec = gate_list[flat]
            u_vec = up_list[flat]
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                row = b.add(warp_m_off, b.add(b.const_i32(mi * atom.m), row_in))
                col = b.add(warp_n_off, b.add(b.const_i32(ni * atom.n), col_in))
                g = b.vec_extract(g_vec, i)
                u = b.vec_extract(u_vec, i)
                h = _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e)
                f32_view_store(b, f32_view, row, col, h)


def f32_view_store(b: IRBuilder, view, row: Value, col: Value, val: Value) -> None:
    b.smem_store_vN(view.base, [row, col], val, 1)


def f32_view_load(b: IRBuilder, view, row: Value, col: Value) -> Value:
    v = b.smem_load_vN(view.base, row, col, dtype=F32, n=1)
    return b.vec_extract(v, 0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_moe_fused_mega_gemm_fp8(
    spec: FusedMegaKernelSpecFp8, arch: str = "gfx950"
) -> KernelDef:
    """Build the STAGE 1 fp8 fused-MoE mega-kernel.

    Current implementation covers STAGE 1 (gate+up fp8 GEMM with per-128-block
    dequant -> SiLU*up f32 -> dynamic-quant to fp8 staged in the persistent LDS
    ``Hidden_smem`` + per-block scales in ``HiddenScale_smem``). The down GEMM +
    weighted atomic reduce (STAGE 2) is a documented stub at the end of the body.
    """
    ok, why, _ = validate_arch_and_block_size(arch, spec.block_size)
    if not ok:
        raise ValueError(f"invalid fp8 fused-mega spec for {arch}: {why}")
    atom = spec.gate_up_atom()
    validate_mfma_atom_in_catalog(atom, arch, where="moe_fused_mega_fp8")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    # ---- params (BUILD_SPEC_FP8 Section 2.7) ---------------------------
    A = b.param("A", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16)
    WGate = b.param(
        "WGate", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    WUp = b.param(
        "WUp", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    WDown = b.param(
        "WDown", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    AScale = b.param("AScale", PtrType(F32, "global"), readonly=True, align=4)
    WGateScale = b.param("WGateScale", PtrType(F32, "global"), readonly=True, align=4)
    WUpScale = b.param("WUpScale", PtrType(F32, "global"), readonly=True, align=4)
    WDownScale = b.param("WDownScale", PtrType(F32, "global"), readonly=True, align=4)
    SortedTokenIds = b.param(
        "SortedTokenIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    SortedWeights = b.param(
        "SortedWeights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    BlockExpertIds = b.param(
        "BlockExpertIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    Y = b.param("Y", PtrType(F32, "global"), align=16)
    M = b.param("M", I32)
    N = b.param("N", I32)  # = I (inter dim)
    K = b.param("K", I32)  # = H (hidden contraction)
    H_out = b.param("H_out", I32)  # = H (down output)
    stride_a = b.param(
        "stride_a", I32
    )  # noqa: F841 -- ABI (A is dense, gather elsewhere)
    stride_b_gate = b.param("stride_b_gate", I32)
    stride_b_up = b.param("stride_b_up", I32)
    stride_b_down = b.param("stride_b_down", I32)  # noqa: F841 -- used in STAGE 2
    stride_a_scale = b.param("stride_a_scale", I32)
    stride_gate_scale = b.param("stride_gate_scale", I32)
    stride_up_scale = b.param("stride_up_scale", I32)
    stride_down_scale = b.param("stride_down_scale", I32)  # noqa: F841 -- STAGE 2
    # Per-expert ELEMENT stride for the weight scale tensors (scales are
    # per-expert, but the scale pointer is NOT folded by _b_base; fold here).
    stride_gate_scale_e = b.param("stride_gate_scale_e", I32)
    stride_up_scale_e = b.param("stride_up_scale_e", I32)
    stride_down_scale_e = b.param("stride_down_scale_e", I32)
    slot_size = b.param("slot_size", I32)  # noqa: F841 -- ABI
    tokens = b.param("tokens", I32)  # noqa: F841 -- used in STAGE 2 epilogue

    tile_m = spec.tile_m
    tile_n = spec.tile_n_inter
    n_blocks = tile_n // GROUP_K

    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(spec.warp_n)
    c_block_m = b.const_i32(tile_m)
    c_block_n = b.const_i32(tile_n)
    c0 = b.const_i32(0)

    # ---- block/thread prelude -----------------------------------------
    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    m_block_idx = b.block_id_y()
    expert_idx = b.global_load_i32(BlockExpertIds, m_block_idx)

    # ---- BYTE-BASE FIX (BUILD_SPEC_FP8 Section 2.4): fp8 weights = 1 byte.
    elem_bytes_b = b.const_i64(1)

    def _b_base(ptr: Value, stride_b: Value) -> Value:
        bytes_off = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_b, I64)), elem_bytes_b
        )
        return b.global_ptr_add(ptr, bytes_off)

    WGate = _b_base(WGate, stride_b_gate)
    WUp = _b_base(WUp, stride_b_up)
    WDown = _b_base(WDown, stride_b_down)

    # Per-expert base for the f32 weight scale tensors (4-byte elements). The
    # scale index math inside the k-loops carries no expert term, so the
    # per-expert slice is selected here on the pointer.
    def _scale_base(ptr: Value, stride_e: Value) -> Value:
        bytes_off = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_e, I64)),
            b.const_i64(4),
        )
        return b.global_ptr_add(ptr, bytes_off)

    WGateScale = _scale_base(WGateScale, stride_gate_scale_e)
    WUpScale = _scale_base(WUpScale, stride_up_scale_e)
    WDownScale = _scale_base(WDownScale, stride_down_scale_e)

    block_m_off = b.mul(m_block_idx, c_block_m)
    gu_n_off = b.mul(b.block_id_x(), c_block_n)

    # ---- LDS allocations ----------------------------------------------
    # Persistent fp8 Hidden buffer (half the f16 bytes): silu(gate)*up quantized
    # here, reused as the down-GEMM LDS-resident A operand in STAGE 2.
    Hidden_smem = b.smem_alloc(FP8E4M3, [tile_m, tile_n], name_hint="Hidden_smem")
    # Per-(row, 128-inter-block) dynamic scales for the down dequant (STAGE 2).
    HiddenScale_smem = b.smem_alloc(
        F32, [tile_m, n_blocks], name_hint="HiddenScale_smem"
    )
    # f32 scratch for the exact per-block amax reduction (STAGE 1 only).
    HiddenF32_smem = b.smem_alloc(F32, [tile_m, tile_n], name_hint="HiddenF32_smem")

    f32_view = TensorView(
        base=HiddenF32_smem,
        desc=TensorDescriptor.packed((tile_m, tile_n), F32),
        addr_space="lds",
    )
    fp8_view = TensorView(
        base=Hidden_smem,
        desc=TensorDescriptor.packed((tile_m, tile_n), FP8E4M3),
        addr_space="lds",
    )
    scale_view = TensorView(
        base=HiddenScale_smem,
        desc=TensorDescriptor.packed((tile_m, n_blocks), F32),
        addr_space="lds",
    )

    lane_decode = decode_mfma_lanes(b, atom, lane)
    mfmas_m = spec.mfmas_m
    mfmas_n = spec.mfmas_n
    mfmas_m_down = spec.mfmas_m_down
    mfmas_n_down = spec.mfmas_n_down
    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * atom.m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * atom.n))

    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)
    c_fp8_max = b.const_f32(FP8_MAX)
    c_floor = b.const_f32(AMAX_FLOOR)

    c_group_k = b.const_i32(GROUP_K)
    c_threads = b.const_i32(spec.block_size)
    _c_n_blocks = b.const_i32(n_blocks)
    c_tile_n = b.const_i32(tile_n)

    def _emit_body() -> None:
        # ---- STAGE 1a: gate + up fp8 GEMM -> f32 (per-128-block dequant) ----
        # Per warp atom (mi, ni) -> one gate_dq / up_dq vector. The contraction
        # group GEMM is run per warp-atom output position; the lane decode and
        # tile bases select the rows/cols this lane owns.
        gate_list = []
        up_list = []
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                m_tile_base = b.add(
                    block_m_off, b.add(warp_m_off, b.const_i32(mi * atom.m))
                )
                n_tile_base = b.add(
                    gu_n_off, b.add(warp_n_off, b.const_i32(ni * atom.n))
                )
                g_dq, u_dq = _emit_fp8_gateup_group_gemm(
                    b,
                    A=A,
                    WGate=WGate,
                    WUp=WUp,
                    AScale=AScale,
                    WGateScale=WGateScale,
                    WUpScale=WUpScale,
                    atom=atom,
                    lane_decode=lane_decode,
                    m_tile_base=m_tile_base,
                    n_tile_base=n_tile_base,
                    K=K,
                    stride_a_scale=stride_a_scale,
                    stride_gate_scale=stride_gate_scale,
                    stride_up_scale=stride_up_scale,
                    tag=f"{mi}_{ni}",
                )
                gate_list.append(g_dq)
                up_list.append(u_dq)

        # ---- STAGE 1b Pass A: SiLU(gate)*up -> f32 LDS scratch -------------
        _store_hidden_f32_pass(
            b,
            atom=atom,
            gate_list=gate_list,
            up_list=up_list,
            f32_view=f32_view,
            warp_m_off=warp_m_off,
            warp_n_off=warp_n_off,
            lane=lane,
            mfmas_m=mfmas_m,
            mfmas_n=mfmas_n,
            one_f32=one_f32,
            c_neg_log2e=c_neg_log2e,
        )
        b.sync()

        # ---- STAGE 1b Pass B: per-(token-block, 128-inter-block) amax ----
        # hand-tuned asm G_dyn_quant granularity = per-token-block (sub_x=32), NOT
        # per-row: ONE dynamic scale per 128-inter-block, reduced over ALL
        # tile_m rows of the block. This row-uniform-within-block scheme is
        # MANDATORY because the down-GEMM fold applies a single per-lane scalar
        # (indexed by the A-input Hidden row ``m_in_atom``) to output slots that
        # span 4 DIFFERENT output rows; only a row-uniform block scale stays
        # correct under that fold (the same constraint as the activation scale,
        # BUILD_SPEC_FP8 OPEN RISK #1). The scale is broadcast to every row of
        # ``scale_view`` so any ``m_in_atom`` read returns the block scale.
        sweep = b.scf_for_iter(
            tid, b.const_i32(n_blocks), c_threads, [], iv_name="cell"
        )
        with sweep as blk:
            col0 = b.mul(blk, c_group_k)
            # amax over ALL rows x 128 cols of this inter-block.
            amax_loop = b.scf_for_iter(
                c0,
                b.const_i32(tile_m * GROUP_K),
                b.const_i32(1),
                [("amax", c_floor)],
                iv_name="j",
            )
            with amax_loop as (j, (amax_acc,)):
                jr = b.div(j, c_group_k)
                jc = b.add(col0, b.mod(j, c_group_k))
                hv = f32_view_load(b, f32_view, jr, jc)
                amax_new = b.fmax(amax_acc, b.fabs(hv))
                b.scf_yield(amax_new)
            amax = amax_loop.results[0]
            scale = b.fmul(amax, b.rcp(c_fp8_max))  # amax / 448
            # Broadcast to every row of this inter-block column.
            row_bc = b.scf_for_iter(c0, c_block_m, b.const_i32(1), [], iv_name="rb")
            with row_bc as rr:
                b.smem_store_vN(scale_view.base, [rr, blk], scale, 1)
                b.scf_yield()
            b.scf_yield()
        b.sync()

        # ---- STAGE 1b Pass C: quantize f32 Hidden -> fp8 LDS --------------
        # Flat thread sweep over all (tile_m * tile_n) cells; each thread reads
        # its f32 value + the block's scale, computes cvt_f32_to_fp8(h * inv),
        # writes fp8 into Hidden_smem at the SAME (row, col) (implicit reshape).
        total_q = tile_m * tile_n
        qsweep = b.scf_for_iter(
            tid, b.const_i32(total_q), c_threads, [], iv_name="qcell"
        )
        with qsweep as qcell:
            row = b.div(qcell, c_tile_n)
            col = b.mod(qcell, c_tile_n)
            blk = b.div(col, c_group_k)
            hv = f32_view_load(b, f32_view, row, col)
            sc = b.smem_load_vN(scale_view.base, row, blk, dtype=F32, n=1)
            sc = b.vec_extract(sc, 0)
            inv = b.rcp(sc)
            q = b.cvt_f32_to_fp8(b.fmul(hv, inv))
            b.smem_store_vN(fp8_view.base, [row, col], q, 1)
            b.scf_yield()
        b.sync()

        # ---- STAGE 2: down fp8 GEMM (LDS-A) -> dequant -> weighted atomic Y --
        # grid.x split the inter contraction; this TG owns the inter slice at
        # gu_n_off and produces a PARTIAL Y over the WHOLE H_out, tiled in
        # tile_n_down chunks. Per output tile, run the LDS-A down group GEMM
        # (contracting this TG's inter slice with the group-accumulator dequant
        # fold by hidden_dyn_scale * down_b_scale) then atomic-add the weighted,
        # token-validity-masked partial into Y.
        #
        # The Hidden A operand (fp8 in Hidden_smem) and its per-128-block dynamic
        # scales (HiddenScale_smem) are already resident from STAGE 1. The
        # contraction extent is the LOCAL inter slice tile_n_inter; A reads local
        # inter columns, W_down reads GLOBAL inter columns (full inter row stride
        # N) at this slice's base. inter_blk_base = gu_n_off // GROUP_K.
        inter_blk_base = b.div(gu_n_off, c_group_k)
        down_for = b.scf_for_iter(
            c0, H_out, b.const_i32(spec.tile_n_down), [], iv_name="ho"
        )
        with down_for as ho:
            down_warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m_down * atom.m))
            down_warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n_down * atom.n))
            down_list = []
            for mi in range(mfmas_m_down):
                for ni in range(mfmas_n_down):
                    # Down output column base (along H_out) for this warp-atom.
                    n_tile_base = b.add(
                        ho,
                        b.add(down_warp_n_off, b.const_i32(ni * atom.n)),
                    )
                    d_dq = _emit_fp8_down_group_gemm(
                        b,
                        a_view=fp8_view,
                        WDown=WDown,
                        WDownScale=WDownScale,
                        atom=atom,
                        lane_decode=lane_decode,
                        n_tile_base=n_tile_base,
                        scale_view=scale_view,
                        inter_slice=tile_n,
                        inter_full=N,
                        inter_blk_base=inter_blk_base,
                        stride_down_scale=stride_down_scale,
                        tag=f"d{mi}_{ni}",
                    )
                    down_list.append(d_dq)
            # Barrier before the next H_out tile reuses Hidden_smem reads
            # (read-only here, but keep the scf.for body well-formed).
            _emit_down_atomic_reduce(
                b,
                atom=atom,
                down_list=down_list,
                warp_m_off=down_warp_m_off,
                warp_n_off=down_warp_n_off,
                lane=lane,
                mfmas_m=mfmas_m_down,
                mfmas_n=mfmas_n_down,
                block_m_off=block_m_off,
                ho_off=ho,
                H_out=H_out,
                SortedTokenIds=SortedTokenIds,
                SortedWeights=SortedWeights,
                Y=Y,
                tokens=tokens,
            )
            b.scf_yield()
        _ = (M, stride_a, slot_size)

    # Empty tail block (BlockExpertIds == -1) skips all work.
    with b.scf_if(b.cmp_ge(expert_idx, c0)):
        _emit_body()

    b.ret()
    return b.kernel
