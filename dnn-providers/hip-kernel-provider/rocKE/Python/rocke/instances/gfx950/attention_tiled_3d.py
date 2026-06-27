# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiled MFMA implementation of AITER's split-KV ``kernel_unified_attention_3d``.

The 3D path runs many CTAs per (q-block, kv-head), each one covering a
segment of the KV sequence. After the segment kernel, ``reduce_segments``
combines the per-segment partial m / l / acc into the final attention
output. AITER selects this path whenever
``use_2d_kernel == False``: i.e. for long sequences with no sliding window
when 2D grid is too small for the device.

This module re-uses every MFMA / async DMA / softmax helper from the
2D tiled kernel and changes only what's structural to the 3D variant:

  - Grid is ``(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)``.
  - ``tile_start..tile_end`` is bounded by the segment range, not the full
    sequence.
  - Outputs are written to a workspace ``segm_output[total_q, num_qh,
    num_segments, head_size]`` plus ``segm_max[..., num_segments]`` and
    ``segm_expsum[..., num_segments]`` (all fp32).
  - The final ``acc /= L`` and fp16 cast happen in the reduce kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ...core.ir import (
    BF16,
    F16,
    F32,
    FP8E4M3,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from ...helpers.atoms import MfmaAtom, make_c_warp_dstr_encoding
from ...helpers.attention import (
    apply_softcap_log2 as _apply_softcap,
    binary_search_seq_idx as _binary_search_seq_idx_helper,
    dequant_fp8x8_to_dtype,
    mfma_16x16x16_for_dtype as _mfma_16x16x16,
    mfma_16x16x32_for_dtype as _mfma_16x16x32,
    warp_xor_reduce_max as _warp_xor_reduce_max,
    warp_xor_reduce_sum as _warp_xor_reduce_sum,
    wave64_reduce_max as _wave64_reduce_max,
    wave64_reduce_sum as _wave64_reduce_sum,
)
from ...helpers.distribution import make_static_tile_distribution
from ...helpers.layouts import TransposeLdsReader
from ...helpers.transforms import TensorDescriptor, indirect, unmerge


MFMA_M = 16
MFMA_N = 16


# CK-Tile C-accumulator warp distribution for the 16x16x16 MFMA atom. This
# segment kernel is single-warp (lane == tid), and every accumulator output
# row decode it does is the same ``row = (lane // 16) * 4 + reg`` mapping that
# CK Tile's ``CWarpDstrEncoding`` (``warp_gemm_attribute_mfma.hpp``) expresses
# as a ``tile_distribution_encoding``: for one wavefront the lane splits as
# ``(m_blk, n) = (lane // 16, lane % 16)`` and the per-lane ``<4 x f32>``
# accumulator slot ``reg`` is the inner M Y dim (``kCM1PerLane = 4``), so
# ``calculate_x`` returns ``row = m_blk * 4 + reg`` (and ``col = n``). Driving
# the row decode through ``calculate_x`` replaces the open-coded ``mul``/``add``
# lane math with the tile-distribution algebra (Phase-C adoption; the C layout
# is dtype-independent so the f16 atom drives it for fp16 and bf16).
_C16_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.f16_16x16x16())
)


def _mfma_16x16_c_row(b, lane, reg: int):
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
    row, _col = _C16_DIST.calculate_x(
        b, ys=[b.const_i32(0), b.const_i32(reg)], ps=[[m_blk, n]]
    )
    return row


@dataclass(frozen=True)
class UnifiedAttention3DTiledSpec:
    """Spec for the split-KV 3D segment kernel.

    Mirrors :class:`UnifiedAttention2DTiledSpec` and adds the segment-count
    knob exactly as AITER's ``select_3d_config`` derives it.
    """

    head_size: int
    block_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    use_sinks: bool
    sliding_window: int
    has_softcap: bool
    num_segments: int
    use_alibi: bool = False
    use_qq_bias: bool = False
    num_seqs: int = 0
    # AMDGPU occupancy hint (``"amdgpu-waves-per-eu"``). Attention is
    # register-pressure-bound; setting this to 2 or 3 tightens the
    # VGPR allocation in exchange for higher occupancy. ``None`` keeps
    # the LLVM heuristic.
    waves_per_eu: Optional[int] = None
    # FP8 K/V cache (mirrors UnifiedAttention2DTiledSpec.kv_storage_dtype).
    # See that spec's docstring for the semantics.
    kv_storage_dtype: Optional[str] = None
    # ``tile_size_override`` / ``use_invariant_hoist`` / ``use_wide_kv_load``
    # are accepted for signature parity with the shared dispatch spec builder
    # (``_tiled_3d_spec_from_problem``) and the gfx942 spec. They select gfx942
    # narrow-atom 3D optimizations; the corresponding ``_gfx942_3d_*`` helpers
    # return None/False on gfx950, so the gfx950 segment kernel does not key on
    # them.
    tile_size_override: Optional[int] = None
    use_invariant_hoist: bool = False
    use_wide_kv_load: bool = False
    # 64-bit paged-KV addressing. When the paged K/V cache exceeds the ~2 GiB
    # i32 buffer-voffset cap, the per-block byte offset
    # (``physical_block * block_stride``) overflows the 32-bit voffset and
    # silently corrupts the load. With this set, the per-block base is folded
    # into a 64-bit buffer base (a wave-uniform ``make_buffer_rsrc`` per block)
    # and only the within-block byte offset stays in the i32 voffset -- exactly
    # the 2D tiled kernel's ``use_i64_kv_addr`` path. The dispatcher only sets
    # this when the cache is actually that large (see ``_enable_i64_kv_addr``),
    # so the default (small-cache) build is byte-identical to before.
    use_i64_kv_addr: bool = False

    def __post_init__(self):
        if self.kv_storage_dtype is not None and self.kv_storage_dtype != "fp8e4m3":
            raise ValueError(
                f"kv_storage_dtype must be None or 'fp8e4m3' (got {self.kv_storage_dtype!r})"
            )

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def block_m(self) -> int:
        return 16

    @property
    def block_q(self) -> int:
        return self.block_m // self.num_queries_per_kv

    @property
    def tile_size(self) -> int:
        return self.block_size

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16

    @property
    def binary_search_iters(self) -> int:
        if self.num_seqs <= 0:
            return 32
        return max(1, int(math.ceil(math.log2(self.num_seqs + 1))))

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_uattn3d_tiled",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            f"seg{self.num_segments}",
            self.dtype,
            f"kv{self.kv_storage_dtype}" if self.kv_storage_dtype else "",
            "i64kv" if self.use_i64_kv_addr else "",
            "sinks" if self.use_sinks else "",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "softcap" if self.has_softcap else "",
            "alibi" if self.use_alibi else "",
            "qqb" if self.use_qq_bias else "",
        )


def supports_tiled_3d(
    *,
    head_size: int,
    block_size: int,
    dtype: str,
    num_queries_per_kv: int,
    use_alibi: bool,
    use_qq_bias: bool,
    use_fp8: bool,
    q_dtype,
    kv_storage_dtype: Optional[str] = None,
    arch: str = "gfx950",
) -> Tuple[bool, str]:
    # The tiled 3D segment kernel uses the same gfx950-only wide-K MFMA +
    # LDS transpose-read primitives as the 2D kernel; reject other targets
    # with a structured reason. See ``instances/common/attention_arch.py``.
    from ..common.attention_arch import validate_tiled_attention_arch

    arch_ok, arch_reason = validate_tiled_attention_arch(arch)
    if not arch_ok:
        return False, arch_reason
    if dtype not in ("fp16", "bf16"):
        return False, f"tiled 3D kernel currently supports fp16/bf16 (got {dtype!r})"
    if head_size not in (64, 128, 256):
        return (
            False,
            f"tiled 3D kernel only supports head_size in {{64,128,256}} (got {head_size})",
        )
    if head_size % 32 != 0:
        return (
            False,
            f"tiled 3D kernel requires head_size divisible by 32 (got {head_size})",
        )
    if block_size not in (16, 32, 64):
        return (
            False,
            f"tiled 3D kernel only supports block_size in {{16,32,64}} (got {block_size})",
        )
    # FP8 K/V cache: enabled via ``kv_storage_dtype="fp8e4m3"``. The
    # ``use_fp8`` flag mirrors the upstream API; both must be set
    # consistently.
    if kv_storage_dtype is not None and kv_storage_dtype != "fp8e4m3":
        return (
            False,
            f"tiled 3D kernel: unsupported kv_storage_dtype {kv_storage_dtype!r}",
        )
    if use_fp8 and kv_storage_dtype is None:
        return (
            False,
            "tiled 3D kernel: use_fp8=True requires kv_storage_dtype='fp8e4m3'",
        )
    if q_dtype is not None and q_dtype not in ("fp16", "bf16"):
        return False, f"tiled 3D kernel: unsupported q_dtype {q_dtype!r}"
    if num_queries_per_kv > 16 or num_queries_per_kv < 1:
        return (
            False,
            f"tiled 3D kernel needs 1<=num_queries_per_kv<=16 (got {num_queries_per_kv})",
        )
    if 16 % num_queries_per_kv != 0:
        return False, "tiled 3D kernel needs num_queries_per_kv to divide BLOCK_M=16"
    # ALiBi and QQ-bias are now supported by the tiled 3D kernel; FP8 K/V
    # cache is gated above via kv_storage_dtype.
    return True, "supported"


def build_unified_attention_3d_tiled(
    spec: UnifiedAttention3DTiledSpec, *, arch: str = "gfx950"
) -> KernelDef:
    """Emit the tiled split-KV 3D segment kernel.

    Each CTA writes its segment's partial state into ``segm_output``,
    ``segm_max``, and ``segm_expsum``. The companion reduce kernel
    (:func:`build_unified_attention_reduce_tiled`) combines those into the
    final output.
    """
    # Same gfx950-only wide-K MFMA + LDS transpose-read dependency as the 2D
    # kernel; reject unsupported targets before emitting IR (comgr would
    # otherwise abort with "LLVM ERROR: Cannot select intrinsic").
    from ..common.attention_arch import require_tiled_attention_arch

    require_tiled_attention_arch(arch)

    if spec.dtype not in ("fp16", "bf16"):
        raise NotImplementedError("tiled 3D kernel supports fp16/bf16")
    dtype = spec.dtype_ir

    HD = spec.head_size
    T = spec.tile_size
    BS = spec.block_size
    BLOCK_M = spec.block_m
    BLOCK_Q = spec.block_q
    NQK = spec.num_queries_per_kv
    NUM_KV = spec.num_kv_heads
    NUM_QH = spec.num_query_heads
    NUM_SEG = spec.num_segments
    SLIDING_WINDOW = spec.sliding_window
    USE_SOFTCAP = spec.has_softcap
    USE_SINKS = spec.use_sinks
    USE_ALIBI = spec.use_alibi
    USE_QQ_BIAS = spec.use_qq_bias
    # 64-bit paged-KV addressing for caches > 2 GiB (see spec docstring).
    I64_KV_ADDR = spec.use_i64_kv_addr
    # FP8 K/V cache: see ``UnifiedAttention2DTiledSpec.kv_storage_dtype``.
    KV_FP8 = spec.kv_storage_dtype == "fp8e4m3"
    KV_BYTES = 1 if KV_FP8 else 2
    kv_io_dtype = FP8E4M3 if KV_FP8 else dtype

    QK_K_STEP = 32
    PV_K_STEP = 32 if T % 32 == 0 else 16
    QK_K_ITERS = HD // QK_K_STEP
    QK_N_TILES = T // MFMA_N
    PV_K_ITERS = T // PV_K_STEP
    PV_N_TILES = HD // MFMA_N

    THREADS = 64

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    # ---------------- parameter declarations ----------------
    # NOTE: the AITER 3D signature distinguishes between segm_* workspace
    # pointers and the regular K/V/cu/seq_lens inputs. We mirror that order.
    segm_output_ptr = b.param(
        "segm_output_ptr",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    segm_max_ptr = b.param(
        "segm_max_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    segm_expsum_ptr = b.param(
        "segm_expsum_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    query = b.param(
        "query_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr",
        PtrType(kv_io_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(kv_io_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sinks = b.param("sink_ptr", PtrType(dtype, "global"), readonly=True, align=16)
    block_tables = b.param(
        "block_tables_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    alibi_slopes_ptr = b.param(
        "alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4
    )
    qq_bias_ptr = b.param("qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale_p = b.param("scale", F32)
    k_scale_p = b.param("k_scale", F32)
    v_scale_p = b.param("v_scale", F32)
    softcap_p = b.param("softcap", F32)
    num_seqs_p = b.param("num_seqs", I32)
    bt_stride_p = b.param("block_table_stride", I32)
    qq_bias_stride0_p = b.param("qq_bias_stride_0", I32)

    q_block_global_idx = b.block_id_x()
    kv_head_idx = b.block_id_y()
    seg_idx = b.block_id_z()
    tid = b.thread_id_x()

    seq_idx = _binary_search_seq_idx_helper(
        b,
        cu_q,
        q_block_global_idx,
        num_seqs_p,
        block_q=BLOCK_Q,
        iterations=spec.binary_search_iters,
    )
    cu_q_start = b.global_load_i32(cu_q, seq_idx)
    cu_q_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    cur_batch_q_len = b.sub(cu_q_stop, cu_q_start)
    q_block_start_idx = b.add(b.div(cu_q_start, b.const_i32(BLOCK_Q)), seq_idx)
    q_block_local_idx = b.sub(q_block_global_idx, q_block_start_idx)
    seq_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(seq_len, cur_batch_q_len)

    qb_start_pos = b.mul(q_block_local_idx, b.const_i32(BLOCK_Q))
    with b.scf_if(b.cmp_ge(qb_start_pos, cur_batch_q_len)):
        b.ret()

    # tiles_per_segment = cdiv(seq_len, NUM_SEG * T)
    tps = b.div(b.add(seq_len, b.const_i32(NUM_SEG * T - 1)), b.const_i32(NUM_SEG * T))

    # If this segment is past seq_len, write a neutral entry to the workspace
    # (m=-inf zeros the reduce contribution; acc=0 and l=0 keep everything
    # finite even when other segments have finite m). AITER's Triton kernel
    # achieves the same effect through `tl.store` masks.
    # Coordinate transforms over the three workspace tensors:
    #   segm_max     [total_q, num_qh, num_segments]
    #   segm_expsum  [total_q, num_qh, num_segments]
    #   segm_output  [total_q, num_qh, num_segments, head_size]
    # plus Q / output which both share ``(token, head, dim)``. ``1<<30``
    # is a compile-time upper bound on ``total_q`` so the row-major
    # stride product matches the kernel's layout assumptions exactly.
    ml_desc = TensorDescriptor.naive(
        "segm_ml",
        lengths=[1 << 30, NUM_QH, NUM_SEG],
        coord_names=("token", "head", "seg"),
    )
    seg_acc_desc = TensorDescriptor.naive(
        "segm_output",
        lengths=[1 << 30, NUM_QH, NUM_SEG, HD],
        coord_names=("token", "head", "seg", "dim"),
    )
    q_desc = TensorDescriptor.naive(
        "Q",
        lengths=[1 << 30, NUM_QH, HD],
        coord_names=("token", "head", "dim"),
    )

    seg_start_tile_pos = b.mul(b.mul(seg_idx, tps), b.const_i32(T))
    with b.scf_if(b.cmp_ge(seg_start_tile_pos, seq_len)):
        neg_inf_local = b.const_f32(float("-inf"))
        zero_local = b.const_f32(0.0)
        lane_writes_ml_e = b.cmp_eq(b.mod(tid, b.const_i32(16)), b.const_i32(0))
        for reg in range(4):
            row = _mfma_16x16_c_row(b, tid, reg)
            qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            row_ok = b.land(
                b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            )
            qp_r_safe = b.select(row_ok, qp_r, b.const_i32(0))
            qh_r_safe = b.select(row_ok, qh_r, b.const_i32(0))
            qtoken = b.add(cu_q_start, qp_r_safe)
            ml_idx, _ = ml_desc.offset(b, token=qtoken, head=qh_r_safe, seg=seg_idx)
            with b.scf_if(lane_writes_ml_e):
                b.global_store(segm_max_ptr, ml_idx, neg_inf_local, align=4)
                b.global_store(segm_expsum_ptr, ml_idx, zero_local, align=4)
        # Zero acc for this segment across all (n, lane_col) entries
        # belonging to this CTA. Each lane writes its slot.
        lane_col_e = b.mod(tid, b.const_i32(16))
        for n in range(PV_N_TILES):
            for reg in range(4):
                row = _mfma_16x16_c_row(b, tid, reg)
                col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col_e)
                qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
                qh_r = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
                )
                row_ok = b.land(
                    b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
                )
                qp_r_safe = b.select(row_ok, qp_r, b.const_i32(0))
                qh_r_safe = b.select(row_ok, qh_r, b.const_i32(0))
                qtoken = b.add(cu_q_start, qp_r_safe)
                seg_acc_idx, _ = seg_acc_desc.offset(
                    b,
                    token=qtoken,
                    head=qh_r_safe,
                    seg=seg_idx,
                    dim=col,
                )
                b.global_store(segm_output_ptr, seg_acc_idx, zero_local, align=4)
        b.ret()

    # ---------------- LDS layout ----------------
    Q_lds = b.smem_alloc(dtype, [BLOCK_M, HD], name_hint="Qlds")
    K_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Vlds")
    P_lds = b.smem_alloc(dtype, [BLOCK_M, T], name_hint="Plds")

    # CK Tile ``TransposeLDSLayout<M=16, K=PV_K_STEP, B=1>`` lane formulas.
    # The 3D segment kernel is single-warp, so ``lane == tid``.
    pv_tr_reader = TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, tid)
    tr_col_lane = pv_tr_reader.col

    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    one_f = b.const_f32(1.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)
    qk_scale = b.fmul(scale_p, rcp_ln2)
    sw_const = b.const_i32(int(SLIDING_WINDOW))
    z8 = b.zero_vec(dtype, 8)

    # ---------------- Q -> LDS ----------------
    Q_VECS_PER_ROW = HD // 8
    Q_VECS_PER_THREAD = (BLOCK_M * Q_VECS_PER_ROW) // THREADS
    for li in range(Q_VECS_PER_THREAD):
        q_vid = b.add(b.mul(b.const_i32(li), b.const_i32(THREADS)), tid)
        Q_row = b.div(q_vid, b.const_i32(Q_VECS_PER_ROW))
        Q_col = b.mul(b.mod(q_vid, b.const_i32(Q_VECS_PER_ROW)), b.const_i32(8))
        q_pos_t = b.add(qb_start_pos, b.div(Q_row, b.const_i32(NQK)))
        qh_t = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)),
            b.mod(Q_row, b.const_i32(NQK)),
        )
        qmask_t = b.land(
            b.cmp_lt(q_pos_t, cur_batch_q_len), b.cmp_lt(qh_t, b.const_i32(NUM_QH))
        )
        q_pos_safe = b.select(qmask_t, q_pos_t, b.const_i32(0))
        qh_safe = b.select(qmask_t, qh_t, b.const_i32(0))
        q_off_base, _ = q_desc.offset(
            b,
            token=b.add(cu_q_start, q_pos_safe),
            head=qh_safe,
            dim=b.const_i32(0),
        )
        v8 = b.global_load_vN(query, b.add(q_off_base, Q_col), dtype, 8, align=16)
        b.smem_store_vN(
            Q_lds,
            [Q_row, Q_col],
            b.vector_select(b.vector_splat(qmask_t, 8), v8, z8),
            8,
        )
    b.sync()

    # ---------------- Per-segment tile range ----------------
    bm1_div_nqk = (BLOCK_M - 1) // NQK
    msp_raw = b.add(b.add(context_len, qb_start_pos), b.const_i32(bm1_div_nqk + 1))
    max_seq_prefix_len = b.select(b.cmp_lt(msp_raw, seq_len), msp_raw, seq_len)
    num_tiles = b.div(b.add(max_seq_prefix_len, b.const_i32(T - 1)), b.const_i32(T))

    # Segment bounds: [seg_idx * tps, min((seg_idx+1)*tps, num_tiles))
    tile_start = b.mul(seg_idx, tps)
    tile_end_raw = b.mul(b.add(seg_idx, b.const_i32(1)), tps)
    tile_end = b.select(b.cmp_lt(tile_end_raw, num_tiles), tile_end_raw, num_tiles)

    # The sliding-window path is not used by the AITER 3D selector
    # (use_2d_kernel returns True whenever sliding_window > 0), but we still
    # mirror the mask code in case a future selector wants it. We never
    # restrict the segment range by sliding window: the per-cell mask below
    # already handles that case.
    _ = sw_const  # documented; only used inside the per-cell mask code below

    # ---------------- online softmax registers ----------------
    lane_rg = b.div(tid, b.const_i32(16))
    lane_col = b.mod(tid, b.const_i32(16))

    if USE_SINKS:
        # Triton's 3D applies sinks only when segm_idx == 0.
        seg0 = b.cmp_eq(seg_idx, b.const_i32(0))
        m_inits = []
        for r in range(4):
            row = _mfma_16x16_c_row(b, tid, r)
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
            sink_h = b.global_load(sinks, qh, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            sink_with_mask = b.select(qh_in, sink_f, neg_inf)
            m_inits.append(b.select(seg0, sink_with_mask, neg_inf))
    else:
        m_inits = [neg_inf, neg_inf, neg_inf, neg_inf]
    l_inits = [one_f, one_f, one_f, one_f]

    acc_zero = b.zero_vec_f32(4)
    acc_inits = [acc_zero for _ in range(PV_N_TILES)]

    iter_args = []
    for r in range(4):
        iter_args.append((f"m{r}", m_inits[r]))
        iter_args.append((f"l{r}", l_inits[r]))
    for n in range(PV_N_TILES):
        iter_args.append((f"acc{n}", acc_inits[n]))

    # ---------------- async K/V infra (identical to 2D) ----------------
    big_bytes = b.const_i32(0x7FFF0000)
    key_rsrc = b.buffer_rsrc(key, big_bytes)
    value_rsrc = b.buffer_rsrc(value, big_bytes)

    KV_HALVES_PER_CALL = THREADS * 8
    assert (T * HD) % KV_HALVES_PER_CALL == 0
    kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL
    bytes_per_call = KV_HALVES_PER_CALL * 2
    # Byte strides for paged KV cache. KV_BYTES is 2 for bf16/fp16, 1 for
    # fp8e4m3. The FP8 sync loader path below dequantises and stores to
    # LDS in the working dtype, so ``bytes_per_buf`` stays at the working
    # dtype slab size.
    kv_stride_blk_b = BS * NUM_KV * HD * KV_BYTES
    kv_stride_tok_b = NUM_KV * HD * KV_BYTES
    kv_stride_h_b = HD * KV_BYTES
    bytes_per_buf = T * HD * 2
    # One-block buffer bound for the i64-addressing path: each per-block
    # buffer descriptor only spans a single physical KV block, so the
    # within-block i32 voffset can never overflow.
    kv_block_bytes_c = b.const_i32(kv_stride_blk_b)

    lane_half_base = b.mul(tid, b.const_i32(8))
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    zero_soff = b.const_i32(0)

    # Paged-KV byte descriptor — same DAG composition the 2D kernel uses:
    # an ``indirect(tile_idx -> physical_block)`` table lookup followed
    # by ``unmerge(linear_half -> (token, dim))`` to split the per-lane
    # half offset, plus the byte-stride 4D base. One ``.offset()`` call
    # produces the final byte address for one async DMA call.
    seq_base = b.mul(seq_idx, bt_stride_p)
    paged_kv_desc = TensorDescriptor.naive(
        "paged_kv_bytes",
        lengths=[1 << 24, T, NUM_KV, HD],
        strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
        coord_names=("physical_block", "token", "kv_head", "dim"),
    ).transform(
        indirect("tile_idx", into="physical_block", table=block_tables, base=seq_base),
        unmerge("linear_half", into=("token", "dim"), dims=(T, HD)),
    )

    def _issue_k_load(kv_tile_idx: Value, buf_idx: Value) -> None:
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        K_buf_base = b.smem_ptr_add(K_lds_addr, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            call_rsrc = key_rsrc
            if I64_KV_ADDR:
                base_i64, voff, _ = paged_kv_desc.offset_i64_split(
                    b,
                    "physical_block",
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half,
                    kv_head=kv_head_idx,
                )
                call_rsrc = b.buffer_rsrc(
                    b.global_ptr_add(key, base_i64), kv_block_bytes_c
                )
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half,
                    kv_head=kv_head_idx,
                )
            k_dst = b.smem_ptr_add(K_buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(call_rsrc, k_dst, voff, zero_soff, 4)

    def _issue_v_load(kv_tile_idx: Value, buf_idx: Value) -> None:
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        V_buf_base = b.smem_ptr_add(V_lds_addr, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            call_rsrc = value_rsrc
            if I64_KV_ADDR:
                base_i64, voff, _ = paged_kv_desc.offset_i64_split(
                    b,
                    "physical_block",
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half,
                    kv_head=kv_head_idx,
                )
                call_rsrc = b.buffer_rsrc(
                    b.global_ptr_add(value, base_i64), kv_block_bytes_c
                )
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half,
                    kv_head=kv_head_idx,
                )
            v_dst = b.smem_ptr_add(V_buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(call_rsrc, v_dst, voff, zero_soff, 4)

    # FP8 K/V cache: sync dequant loader. See attention_tiled_2d.py for the
    # rationale. The FP8 path stores the working dtype (bf16/fp16) into LDS,
    # so the rest of the kernel reads K/V_lds unchanged.
    fp8_elems_per_chunk = 8
    fp8_total_chunks = (T * HD) // fp8_elems_per_chunk
    if KV_FP8:
        assert fp8_total_chunks % THREADS == 0, (
            f"fp8 loader: total chunks {fp8_total_chunks} must be divisible by "
            f"THREADS={THREADS} (T={T}, HD={HD})"
        )
    fp8_chunks_per_thread = fp8_total_chunks // THREADS

    def _issue_fp8_dequant_loads(
        kv_tile_idx: Value, buf_idx: Value, lds_token: str
    ) -> None:
        """Sync per-thread fp8 -> f32 -> *scale -> bf16/fp16 -> LDS.

        Uses gfx950's packed ``v_cvt_pk_f32_fp8`` (via
        :meth:`IRBuilder.cvt_pk_f32_fp8x4`) instead of 8 scalar
        ``v_cvt_f32_fp8`` per chunk. The 2D sync sibling
        ``_issue_fp8_dequant_loads`` (in ``attention_tiled_2d.py``)
        documents the rationale: the compiler does NOT fuse 8 scalar
        cvts into 2 packed cvts on its own (ISA inspection of the
        round-1 sync loader confirmed 8 scalar emits per chunk), and
        AITER's production paged-attention FP8 K/V dequant uses the
        packed primitive too (see
        ``csrc/include/attention_common.cuh::to_float_fp8x4``).

        The scale is applied **unfused** as a plain ``v_pk_mul``
        against an f32 scale, NOT via the gfx950 fused
        ``v_cvt_scalef32_pk_f32_fp8`` intrinsic. The fused form
        interprets ``scale`` as an E8M0 micro-scaling exponent and
        silently truncates arbitrary per-tensor scales like
        ``k_scale = max_abs / 448`` to the nearest power of two,
        giving ~0.71x of expected magnitude on production traces. See
        the comment block in the 2D sync loader (~lines 1600-1630) for
        the full correctness argument.
        """
        scale = k_scale_p if lds_token == "K" else v_scale_p
        lds = K_lds if lds_token == "K" else V_lds
        src = key if lds_token == "K" else value
        assert fp8_elems_per_chunk == 8, (
            f"FP8 dequant loader expects 8-elem chunks (got "
            f"{fp8_elems_per_chunk}); the packed-cvt path needs the chunk "
            f"to split cleanly into 2× <4 x fp8> sub-chunks."
        )
        for call in range(fp8_chunks_per_thread):
            chunk_id = b.add(b.mul(b.const_i32(call), b.const_i32(THREADS)), tid)
            row = b.div(chunk_id, b.const_i32(HD // fp8_elems_per_chunk))
            col = b.mul(
                b.mod(chunk_id, b.const_i32(HD // fp8_elems_per_chunk)),
                b.const_i32(fp8_elems_per_chunk),
            )
            linear_half_first = b.add(b.mul(row, b.const_i32(HD)), col)
            if I64_KV_ADDR:
                # Flat per-lane global load: full i64 byte/element offset so
                # the per-thread physical_block * stride cannot overflow the
                # cache addressing for paged caches > 2 GiB.
                voff, _ = paged_kv_desc.offset_i64(
                    b,
                    "physical_block",
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half_first,
                    kv_head=kv_head_idx,
                )
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half_first,
                    kv_head=kv_head_idx,
                )
            fp8_vec = b.global_load_vN(
                src, voff, FP8E4M3, n=fp8_elems_per_chunk, align=fp8_elems_per_chunk
            )
            # Split <8 x fp8> into 2x <4 x fp8> for the packed cvt; the
            # backend collapses the extract+pack chain back into a
            # bitcast-equivalent shuffle once it sees the
            # cvt.pk.f32.fp8 operand is whole-i32.
            packed = dequant_fp8x8_to_dtype(b, fp8_vec, scale, dtype)
            b.smem_store_vN(lds, [buf_idx, row, col], packed, fp8_elems_per_chunk)

    def _issue_k(tile_idx: Value, buf_idx: Value) -> None:
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "K")
        else:
            _issue_k_load(tile_idx, buf_idx)

    def _issue_v(tile_idx: Value, buf_idx: Value) -> None:
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "V")
        else:
            _issue_v_load(tile_idx, buf_idx)

    _issue_k(tile_start, b.const_i32(0))

    cur_buf_init = b.const_i32(0)
    iter_args.append(("cur_buf", cur_buf_init))

    kvloop = b.scf_for_iter(
        tile_start, tile_end, b.const_i32(1), iter_args, iv_name="kv_tile"
    )
    with kvloop as (kv_tile_iv, carry):
        m_vals = [carry[2 * r] for r in range(4)]
        l_vals = [carry[2 * r + 1] for r in range(4)]
        acc_vals = [carry[8 + n] for n in range(PV_N_TILES)]
        cur_buf = carry[8 + PV_N_TILES]
        nxt_buf = b.sub(b.const_i32(1), cur_buf)
        tile_off = b.mul(kv_tile_iv, b.const_i32(T))

        next_tile_iv_raw = b.add(kv_tile_iv, b.const_i32(1))
        in_range_next = b.cmp_lt(next_tile_iv_raw, tile_end)
        safe_next_tile = b.select(in_range_next, next_tile_iv_raw, kv_tile_iv)

        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.sync()

        # QK
        A_kits = []
        for k in range(QK_K_ITERS):
            q_col_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
            A_kits.append(b.smem_load_vN(Q_lds, lane_col, q_col_off, dtype=dtype, n=8))
        S_n = []
        for n in range(QK_N_TILES):
            acc_v = b.zero_vec_f32(4)
            for k in range(QK_K_ITERS):
                kc_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                k_row = b.add(b.const_i32(n * 16), lane_col)
                B_v = b.smem_load_vN(K_lds, cur_buf, k_row, kc_off, dtype=dtype, n=8)
                acc_v = _mfma_16x16x32(b, dtype, A_kits[k], B_v, acc_v)
            S_n.append(acc_v)

        _issue_v(kv_tile_iv, cur_buf)
        _issue_k(safe_next_tile, nxt_buf)

        # See attention_tiled_2d.py for the rationale on applying ALiBi /
        # QQ-bias before the select-with-(-inf) (equivalent to Triton's
        # post-select add for finite biases via IEEE -inf semantics, with
        # better robustness against compiler reordering).
        if USE_ALIBI:
            alibi_per_row = []
            for reg in range(4):
                row = _mfma_16x16_c_row(b, tid, reg)
                qh_r = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
                )
                qh_ok = b.cmp_lt(qh_r, b.const_i32(NUM_QH))
                slope = b.masked_global_load(
                    alibi_slopes_ptr, qh_r, qh_ok, b.const_f32(0.0), dtype=F32, align=4
                )
                alibi_per_row.append(slope)
        masked = {}
        for reg in range(4):
            row = _mfma_16x16_c_row(b, tid, reg)
            qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            row_ok = b.land(
                b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            )
            for n in range(QK_N_TILES):
                col_abs = b.add(
                    b.add(tile_off, b.mul(b.const_i32(n), b.const_i32(16))), lane_col
                )
                causal_lim = b.add(context_len, qp_r)
                causal_ok = b.cmp_le(col_abs, causal_lim)
                in_prefix = b.cmp_lt(col_abs, max_seq_prefix_len)
                m_ok = b.land(b.land(row_ok, causal_ok), in_prefix)
                if SLIDING_WINDOW > 0:
                    dist = b.sub(causal_lim, col_abs)
                    m_ok = b.land(m_ok, b.cmp_lt(dist, sw_const))
                s_raw = b.vec_extract(S_n[n], reg)
                s_scaled = b.fmul(s_raw, qk_scale)
                if USE_SOFTCAP:
                    s_scaled = b.fmul(_apply_softcap(b, s_scaled, softcap_p), rcp_ln2)
                if USE_ALIBI:
                    pos_off = b.sub(col_abs, context_len)
                    pos_f = b.sitofp_f32(pos_off)
                    add_term = b.fmul(b.fmul(alibi_per_row[reg], pos_f), rcp_ln2)
                    s_scaled = b.fadd(s_scaled, add_term)
                if USE_QQ_BIAS:
                    # See attention_tiled_2d.py for the rationale on row_ok.
                    krp = b.sub(col_abs, context_len)
                    krp_ok = b.land(
                        b.cmp_ge(krp, b.const_i32(0)), b.cmp_lt(krp, qq_bias_stride0_p)
                    )
                    qq_ok = b.land(row_ok, krp_ok)
                    qp_safe = b.select(row_ok, qp_r, b.const_i32(0))
                    qq_idx = b.add(b.mul(qp_safe, qq_bias_stride0_p), krp)
                    qq_v = b.masked_global_load(
                        qq_bias_ptr,
                        qq_idx,
                        qq_ok,
                        b.const_f32(0.0),
                        dtype=F32,
                        align=4,
                    )
                    s_scaled = b.fadd(s_scaled, b.fmul(qq_v, rcp_ln2))
                masked[(n, reg)] = b.select(m_ok, s_scaled, neg_inf)

        m_new = []
        s_local = {}
        for reg in range(4):
            local_max = neg_inf
            for n in range(QK_N_TILES):
                v = masked[(n, reg)]
                s_local[(reg, n)] = v
                local_max = b.fmax(local_max, v)
            full_max_raw = _warp_xor_reduce_max(b, local_max)
            ok = b.fcmp("ogt", full_max_raw, neg_inf)
            m_new.append(b.select(ok, full_max_raw, zero_f))

        l_local = []
        for reg in range(4):
            row = _mfma_16x16_c_row(b, tid, reg)
            sum_p = zero_f
            for n in range(QK_N_TILES):
                p = b.exp2(b.fsub(s_local[(reg, n)], m_new[reg]))
                col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
                b.smem_store_vN(P_lds, [row, col], b.cast_f32_to(p, dtype), 1)
                sum_p = b.fadd(sum_p, p)
            l_local.append(_warp_xor_reduce_sum(b, sum_p))

        alpha_regs = [b.exp2(b.fsub(m_vals[r], m_new[r])) for r in range(4)]
        new_l_vals = [
            b.fadd(b.fmul(l_vals[r], alpha_regs[r]), l_local[r]) for r in range(4)
        ]
        if KV_FP8:
            # FP8 sync loader has no in-flight async work.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sync()
        else:
            b.s_waitcnt(vmcnt=kv_calls_per_tile, lgkmcnt=kv_calls_per_tile)
            b.sync()

        new_acc = []
        for n in range(PV_N_TILES):
            scaled_comps = []
            for reg in range(4):
                e = b.vec_extract(acc_vals[n], reg)
                scaled_comps.append(b.fmul(e, alpha_regs[reg]))
            acc_v = b.vec_pack(scaled_comps, F32)

            n_col_base = b.add(b.mul(b.const_i32(n), b.const_i32(16)), tr_col_lane)

            for k in range(PV_K_ITERS):
                if PV_K_STEP == 32:
                    p_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                    A_p = b.smem_load_vN(P_lds, lane_col, p_off, dtype=dtype, n=8)
                    row_r0 = pv_tr_reader.row(b, k_offset=k * 32, read=0)
                    row_r1 = pv_tr_reader.row(b, k_offset=k * 32, read=1)
                    B_r0 = b.ds_read_tr16_b64(
                        V_lds, cur_buf, row_r0, n_col_base, dtype=dtype
                    )
                    B_r1 = b.ds_read_tr16_b64(
                        V_lds, cur_buf, row_r1, n_col_base, dtype=dtype
                    )
                    B_v = b.vec_concat(B_r0, B_r1)
                    acc_v = _mfma_16x16x32(b, dtype, A_p, B_v, acc_v)
                else:
                    p_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                    A_p = b.smem_load_vN(P_lds, lane_col, p_off, dtype=dtype, n=4)
                    row_lane = pv_tr_reader.row(b, k_offset=k * 16, read=0)
                    B_v = b.ds_read_tr16_b64(
                        V_lds, cur_buf, row_lane, n_col_base, dtype=dtype
                    )
                    acc_v = _mfma_16x16x16(b, dtype, A_p, B_v, acc_v)
            new_acc.append(acc_v)

        yields = []
        for r in range(4):
            yields.append(m_new[r])
            yields.append(new_l_vals[r])
        for n in range(PV_N_TILES):
            yields.append(new_acc[n])
        yields.append(nxt_buf)
        b.scf_yield(*yields)

    # ---------------- write segment workspace ----------------
    final = kvloop.results
    m_final = [final[2 * r] for r in range(4)]
    l_final = [final[2 * r + 1] for r in range(4)]
    acc_final = [final[8 + n] for n in range(PV_N_TILES)]

    # Per-thread: write own (row, col) of acc; only the lane in each
    # 4-lane row group (lane%4 == 0) writes m and l. The workspace
    # layouts ``segm_output[token, head, seg, dim]`` and
    # ``segm_{max,expsum}[token, head, seg]`` are encoded by the
    # ``seg_acc_desc`` / ``ml_desc`` coordinate transforms defined at
    # the top of this function.
    for n in range(PV_N_TILES):
        for reg in range(4):
            row = _mfma_16x16_c_row(b, tid, reg)
            col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
            qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            row_ok = b.land(
                b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            )
            qtoken = b.add(cu_q_start, qp_r)
            seg_acc_idx, _ = seg_acc_desc.offset(
                b,
                token=qtoken,
                head=qh_r,
                seg=seg_idx,
                dim=col,
            )
            v_acc = b.vec_extract(acc_final[n], reg)
            with b.scf_if(row_ok):
                b.global_store(segm_output_ptr, seg_acc_idx, v_acc, align=4)

    lane_writes_ml = b.cmp_eq(b.mod(tid, b.const_i32(16)), b.const_i32(0))
    for reg in range(4):
        row = _mfma_16x16_c_row(b, tid, reg)
        qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
        qh_r = b.add(b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK)))
        row_ok = b.land(
            b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
        )
        qtoken = b.add(cu_q_start, qp_r)
        ml_idx, _ = ml_desc.offset(b, token=qtoken, head=qh_r, seg=seg_idx)
        do_write = b.land(lane_writes_ml, row_ok)
        with b.scf_if(do_write):
            b.global_store(segm_max_ptr, ml_idx, m_final[reg], align=4)
            b.global_store(segm_expsum_ptr, ml_idx, l_final[reg], align=4)

    return b.kernel


# ---------------------------------------------------------------------------
# Reduce kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnifiedAttentionReduceTiledSpec:
    head_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    num_segments: int
    # AMDGPU occupancy hint (``"amdgpu-waves-per-eu"``). ``None`` keeps
    # the LLVM backend's heuristic.
    waves_per_eu: Optional[int] = None

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_uattn_reduce_tiled",
            f"d{self.head_size}",
            f"h{self.num_query_heads}",
            f"seg{self.num_segments}",
            self.dtype,
        )


def build_unified_attention_reduce_tiled(
    spec: UnifiedAttentionReduceTiledSpec,
    *,
    arch: str = "gfx950",
) -> KernelDef:
    """Combine the per-segment partial state into the final fp16/bf16 output.

    The reduce kernel is arch-neutral (pure f32 load / exp2 / store, no MFMA
    or LDS transpose reads), so it builds on both gfx942 and gfx950. The
    ``arch`` parameter is accepted for call-site symmetry with the segment
    builder and is currently unused.

    Grid: ``(total_q, num_query_heads, 1)``.  Each CTA reduces over the
    ``NUM_SEGMENTS`` segments for one (token, head):

      1. overall_max = max(segm_max)
      2. overall_expsum = sum(segm_expsum * exp2(segm_max - overall_max))
      3. acc[d] = sum_s segm_output[s,d] * exp2(segm_max[s] - overall_max)
      4. output[d] = acc[d] / overall_expsum  (with 0/0 -> 0)
    """
    HD = spec.head_size
    NUM_SEG = spec.num_segments
    NUM_QH = spec.num_query_heads
    dtype = spec.dtype_ir

    THREADS = 64
    HALFS_PER_THREAD = HD // THREADS
    assert HALFS_PER_THREAD * THREADS == HD

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    out = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    seg_out = b.param(
        "segm_output_ptr", PtrType(F32, "global"), readonly=True, align=16
    )
    seg_max = b.param("segm_max_ptr", PtrType(F32, "global"), readonly=True, align=4)
    seg_l = b.param("segm_expsum_ptr", PtrType(F32, "global"), readonly=True, align=4)
    _seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)

    q_token = b.block_id_x()
    q_head = b.block_id_y()
    tid = b.thread_id_x()

    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)

    # Workspace layouts (same as the segment kernel's outputs):
    #   segm_max / segm_expsum : [total_q, num_qh, num_segments]
    #   segm_output            : [total_q, num_qh, num_segments, head_size]
    # And the final output uses ``(token, head, dim)``. Encoding them
    # as descriptors makes the per-pass offset math one call instead of
    # an `add(mul(qt,...), mul(qh,...))` ladder.
    ml_desc_red = TensorDescriptor.naive(
        "segm_ml",
        lengths=[1 << 30, NUM_QH, NUM_SEG],
        coord_names=("token", "head", "seg"),
    )
    seg_acc_desc_red = TensorDescriptor.naive(
        "segm_output",
        lengths=[1 << 30, NUM_QH, NUM_SEG, HD],
        coord_names=("token", "head", "seg", "dim"),
    )
    out_desc_red = TensorDescriptor.naive(
        "out",
        lengths=[1 << 30, NUM_QH, HD],
        coord_names=("token", "head", "dim"),
    )

    # Per-segment ml base for this (q_token, q_head); the
    # segment-strided loads below add the segment index, which is the
    # same as ``ml_desc_red.offset(... seg=sv)``.
    base_ml, _ = ml_desc_red.offset(b, token=q_token, head=q_head, seg=b.const_i32(0))

    # The reduce CTA is a single wave64. The original kernel had all 64
    # lanes redundantly walk the full ``NUM_SEG`` segment list twice
    # (pass 1 max, pass 2 expsum) and a third time inside pass 3 — a
    # 64x-redundant, serially-dependent chain of global loads that was
    # the flat ~21 us floor on decode shapes (constant in KV length;
    # see notes/ATTENTION_PARITY_REPORT.md split-KV analysis). Instead
    # each lane now owns the strided segment subset ``{tid, tid+64,
    # ...}``: it loads each ``(seg_max, seg_l)`` pair exactly once, and
    # a 64-lane XOR butterfly folds the per-lane partials. This cuts the
    # serial dependency chain from ``NUM_SEG`` to ``ceil(NUM_SEG/64)``
    # and removes the 2x redundant ml reload.
    #
    # The per-segment NaN-safe factor ``exp2(seg_max - overall_max)`` is
    # then cached in LDS once so pass 3 (the per-dim acc reduce) reuses
    # it instead of reloading ``seg_max`` and recomputing ``exp2`` for
    # every output dim.
    SEG_PER_LANE = (NUM_SEG + THREADS - 1) // THREADS
    factor_lds = b.smem_alloc_f32([NUM_SEG], name_hint="seg_factor")

    # ---- pass 1: per-lane partial max over the owned segments ----
    seg_idx_of = []  # (sv_value, in_range_pred or None) per owned slot
    seg_max_cache = []
    seg_l_cache = []
    local_max = neg_inf
    for j in range(SEG_PER_LANE):
        sv = b.add(b.const_i32(j * THREADS), tid)
        in_rng = (
            None
            if (j * THREADS + THREADS) <= NUM_SEG
            else b.cmp_lt(sv, b.const_i32(NUM_SEG))
        )
        sv_safe = sv if in_rng is None else b.select(in_rng, sv, b.const_i32(0))
        idx = b.add(base_ml, sv_safe)
        ms = b.global_load_f32(seg_max, idx)
        ls = b.global_load_f32(seg_l, idx)
        if in_rng is not None:
            ms = b.select(in_rng, ms, neg_inf)
            ls = b.select(in_rng, ls, zero_f)
        seg_idx_of.append((sv, in_rng, sv_safe))
        seg_max_cache.append(ms)
        seg_l_cache.append(ls)
        local_max = b.fmax(local_max, ms)
    overall_max = _wave64_reduce_max(b, local_max)

    # ---- pass 2: per-lane partial expsum + cache per-segment factor ----
    local_den = zero_f
    for j in range(SEG_PER_LANE):
        sv, in_rng, sv_safe = seg_idx_of[j]
        ms = seg_max_cache[j]
        ls = seg_l_cache[j]
        # NaN-safe factor: when both ms and overall_max are -inf, the
        # difference is NaN; force factor to 0 in that case.
        ms_finite = b.fcmp("ogt", ms, neg_inf)
        factor_raw = b.exp2(b.fsub(ms, overall_max))
        factor = b.select(ms_finite, factor_raw, zero_f)
        local_den = b.fadd(local_den, b.fmul(ls, factor))
        # Stash the factor for this lane's owned segments. Out-of-range
        # lanes (when NUM_SEG < THREADS) must NOT write: their masked
        # ``sv_safe == 0`` would otherwise race lane 0's real write to
        # ``factor_lds[0]``. Guard the store on the in-range predicate.
        if in_rng is None:
            b.smem_store_vN_f32(factor_lds, [sv_safe], factor, 1)
        else:
            with b.scf_if(in_rng):
                b.smem_store_vN_f32(factor_lds, [sv], factor, 1)
    overall_expsum = _wave64_reduce_sum(b, local_den)
    safe_expsum = b.fcmp("oeq", overall_expsum, zero_f)
    inv_l = b.select(safe_expsum, zero_f, b.rcp(overall_expsum))

    b.sync()  # publish factor_lds before the per-dim acc reduce reads it

    # ---- pass 3: per-element reduce + normalize + write ----
    # Each thread handles HALFS_PER_THREAD output dims. The per-segment
    # factor is read from LDS (computed once above) rather than
    # reloading seg_max + recomputing exp2 per dim.
    for li in range(HALFS_PER_THREAD):
        d = b.add(b.mul(b.const_i32(li), b.const_i32(THREADS)), tid)
        acc_loop = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(NUM_SEG),
            b.const_i32(1),
            [(f"ac{li}", zero_f)],
            iv_name=f"s_acc{li}",
        )
        with acc_loop as (sv, (ac,)):
            factor = b.smem_load_vN_f32(factor_lds, sv, n=1)
            factor_s = b.vec_extract(factor, 0)
            idx_acc, _ = seg_acc_desc_red.offset(
                b,
                token=q_token,
                head=q_head,
                seg=sv,
                dim=d,
            )
            ov = b.global_load_f32(seg_out, idx_acc)
            b.scf_yield(b.fadd(ac, b.fmul(ov, factor_s)))
        scalar_out_f32 = b.fmul(acc_loop.results[0], inv_l)
        scalar_out = b.cast_f32_to(scalar_out_f32, dtype)
        out_idx, _ = out_desc_red.offset(b, token=q_token, head=q_head, dim=d)
        b.global_store(out, out_idx, scalar_out, align=2)

    return b.kernel
