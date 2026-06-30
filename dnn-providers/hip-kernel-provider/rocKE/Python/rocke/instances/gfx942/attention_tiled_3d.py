# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 (CDNA3) narrow-atom port of the tiled split-KV 3D attention kernel.

This is the gfx942 sibling of ``instances/gfx950/attention_tiled_3d.py``. The
*structure* (split-KV segment kernel + online-softmax reduce) is identical to
the gfx950 version; only the three gfx950-only ISA spots are swapped for the
gfx942-legal forms that the 2D gfx942 kernel
(``instances/gfx942_mi300/attention_tiled_2d.py``) already validates:

  * **QK MFMA**: gfx942 has no wide-K ``mfma_f32_16x16x32`` f16/bf16 atom, so
    the QK matmul uses the narrow ``16x16x16`` atom with a K-step of 16
    (``QK_K_ITERS = HD // 16``). Both A (Q) and B (K^T) operands are loaded as
    ``<4 x dtype>`` instead of ``<8 x dtype>``.
  * **PV MFMA**: the gfx950 version reads the V B-operand with
    ``ds_read_tr16_b64`` (an ``ds_read_*_tr_*`` transpose read absent on
    gfx942) and runs a wide-K PV. gfx942 builds the V B-operand from ordinary
    strided LDS loads (``_strided_v_b_operand``, reproducing the exact per-lane
    (row, col) the transpose read delivers for a ``16x16x16`` atom) and runs the
    narrow ``16x16x16`` PV with a K-step of 16 (``PV_K_ITERS = T // 16``).
  * **FP8 K/V dequant**: uses the same ``cvt_pk_f32_fp8x4`` packed-cvt the
    gfx942 2D kernel uses (gfx942-legal), via ``dequant_fp8x8_to_dtype``.

The reduce kernel is pure f32 load / exp2 / store (no MFMA, no transpose read),
so it is arch-neutral and is a byte-for-byte port of the gfx950 reduce kernel.

See the gfx950 module docstring for the split-KV grid / workspace layout.
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
    warp_xor_reduce_max as _warp_xor_reduce_max,
    warp_xor_reduce_sum as _warp_xor_reduce_sum,
    wave64_reduce_max as _wave64_reduce_max,
    wave64_reduce_sum as _wave64_reduce_sum,
)
from ...helpers.distribution import make_static_tile_distribution
from ...helpers.transforms import TensorDescriptor, embed, indirect, unmerge


MFMA_M = 16
MFMA_N = 16


# CK-Tile C-accumulator warp distribution for the 16x16x16 MFMA atom (identical
# to the gfx950 file; the C layout is dtype- and arch-independent so the f16
# atom drives it for both fp16 and bf16). This segment kernel is single-warp
# (lane == tid); the per-lane ``<4 x f32>`` accumulator row decode is
# ``row = (lane // 16) * 4 + reg`` expressed through ``calculate_x``.
_C16_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.f16_16x16x16())
)


def _mfma_16x16_c_row(b, lane, reg: int):
    """MFMA-local output row for a ``16x16`` C element ``reg`` (0..3)."""
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
    """Spec for the split-KV 3D segment kernel (gfx942 narrow-atom variant).

    Field-compatible with the gfx950 ``UnifiedAttention3DTiledSpec`` so the
    dispatcher and harnesses can construct it identically.
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
    waves_per_eu: Optional[int] = None
    kv_storage_dtype: Optional[str] = None
    tile_size_override: Optional[int] = None
    use_invariant_hoist: bool = False
    # Wide KV feed: replace the gfx942 1-DWORD (4-byte) async buffer_load_lds DMA
    # with wide 128-bit (8-half) synchronous global->register->LDS loads (the same
    # vehicle the in-kernel fp8 path and the 2D cfvst V feed use). 4x fewer load
    # instructions -> less VMEM issue pressure on the memory-bound decode segment.
    # fp16/bf16 only (the fp8 path already loads wide). Signature-tracked.
    use_wide_kv_load: bool = False
    # 64-bit paged-KV addressing for caches > 2 GiB. Accepted for signature
    # parity with the shared dispatch spec builder and the gfx950 spec; the
    # gfx942 narrow segment loaders do not key on it yet (the gfx942 >2 GiB
    # decode path is a separate follow-on), so a True value here would be a
    # no-op on gfx942 -- guard it so the dispatcher cannot silently build an
    # uncorrected gfx942 kernel for an oversized cache.
    use_i64_kv_addr: bool = False

    def __post_init__(self):
        if self.use_i64_kv_addr:
            raise NotImplementedError(
                "gfx942 tiled 3D kernel does not support use_i64_kv_addr "
                "(paged KV cache > 2 GiB) yet"
            )
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
        return (
            self.tile_size_override
            if self.tile_size_override is not None
            else self.block_size
        )

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
            "rocke_uattn3d_tiled_gfx942",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            f"seg{self.num_segments}",
            self.dtype,
            f"kv{self.kv_storage_dtype}" if self.kv_storage_dtype else "",
            "sinks" if self.use_sinks else "",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "softcap" if self.has_softcap else "",
            "alibi" if self.use_alibi else "",
            "qqb" if self.use_qq_bias else "",
            "hoist" if self.use_invariant_hoist else "",
            "wkv" if self.use_wide_kv_load else "",
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
    arch: str = "gfx942",
) -> Tuple[bool, str]:
    # gfx942 narrow-atom path: the arch gate admits gfx942 via the
    # ``_NARROW_TILED_2D_ARCHES`` branch (16x16x16 f16/bf16 atom, strided-V).
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
    return True, "supported"


def build_unified_attention_3d_tiled(
    spec: UnifiedAttention3DTiledSpec, *, arch: str = "gfx942"
) -> KernelDef:
    """Emit the gfx942 tiled split-KV 3D segment kernel (narrow 16x16x16)."""
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
    USE_INVARIANT_HOIST = spec.use_invariant_hoist
    KV_FP8 = spec.kv_storage_dtype == "fp8e4m3"
    KV_BYTES = 1 if KV_FP8 else 2
    kv_io_dtype = FP8E4M3 if KV_FP8 else dtype

    # gfx942 narrow atom: K-step is 16 for both QK and PV (no wide-K atom).
    QK_K_STEP = 16
    PV_K_STEP = 16
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
    # gfx942 uses the natural row-major V_lds[2, T, HD]; the PV B-operand is
    # built from strided LDS loads (no transpose-read intrinsic).
    Q_lds = b.smem_alloc(dtype, [BLOCK_M, HD], name_hint="Qlds")
    K_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Vlds")
    P_lds = b.smem_alloc(dtype, [BLOCK_M, T], name_hint="Plds")

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

    tile_start = b.mul(seg_idx, tps)
    tile_end_raw = b.mul(b.add(seg_idx, b.const_i32(1)), tps)
    tile_end = b.select(b.cmp_lt(tile_end_raw, num_tiles), tile_end_raw, num_tiles)

    _ = sw_const  # documented; only used inside the per-cell mask code below

    # ---------------- online softmax registers ----------------
    lane_rg = b.div(tid, b.const_i32(16))
    lane_col = b.mod(tid, b.const_i32(16))

    if USE_INVARIANT_HOIST:
        hoist_row = []
        hoist_qp_r = []
        hoist_qh_r = []
        hoist_row_ok = []
        hoist_causal_lim = []
        for reg in range(4):
            row = _mfma_16x16_c_row(b, tid, reg)
            qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            row_ok = b.land(
                b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            )
            hoist_row.append(row)
            hoist_qp_r.append(qp_r)
            hoist_qh_r.append(qh_r)
            hoist_row_ok.append(row_ok)
            hoist_causal_lim.append(b.add(context_len, qp_r))
    else:
        hoist_row = hoist_qp_r = hoist_qh_r = hoist_row_ok = hoist_causal_lim = None

    if USE_SINKS:
        seg0 = b.cmp_eq(seg_idx, b.const_i32(0))
        m_inits = []
        for r in range(4):
            if USE_INVARIANT_HOIST:
                qh = hoist_qh_r[r]
            else:
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

    # ---------------- async K/V infra (gfx942 1-dword DMA) ----------------
    # CDNA3 (gfx942) async global->LDS DMA caps at 1 DWORD (4 bytes = 2 halves)
    # per lane: ``llvm.amdgcn.raw.ptr.buffer.load.lds`` can only legalise a
    # 1-dword load-to-LDS on gfx942. The wider b96/b128 forms (3/4 dwords) are
    # gfx950-only; emitting a 16-byte (4-dword) LDS DMA on gfx942 aborts the
    # backend with "Do not know how to expand this operator's operand!". The
    # gfx950 3D kernel used 8 halves/lane (16 bytes, 4 dwords), so here we clamp
    # to 2 halves/lane (1 dword) and issue 4x as many calls to move the same
    # bytes. The LDS deposit stays lane-contiguous, so the [2, T, HD] K/V_lds
    # layout and the strided-V B-operand math are unchanged. See
    # attention_tiled_2d.py ASYNC_LDS_MAX_DWORDS for the same clamp.
    ASYNC_LDS_DWORDS = 1
    HALVES_PER_LANE = ASYNC_LDS_DWORDS * 2  # 4 bytes / 2 bytes-per-half
    big_bytes = b.const_i32(0x7FFF0000)
    key_rsrc = b.buffer_rsrc(key, big_bytes)
    value_rsrc = b.buffer_rsrc(value, big_bytes)

    KV_HALVES_PER_CALL = THREADS * HALVES_PER_LANE
    assert (T * HD) % KV_HALVES_PER_CALL == 0
    kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL
    bytes_per_call = KV_HALVES_PER_CALL * 2
    kv_stride_blk_b = BS * NUM_KV * HD * KV_BYTES
    kv_stride_tok_b = NUM_KV * HD * KV_BYTES
    kv_stride_h_b = HD * KV_BYTES
    bytes_per_buf = T * HD * 2

    lane_half_base = b.mul(tid, b.const_i32(HALVES_PER_LANE))
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    zero_soff = b.const_i32(0)

    seq_base = b.mul(seq_idx, bt_stride_p)
    _kv_base = TensorDescriptor.naive(
        "paged_kv_bytes",
        lengths=[1 << 24, BS, NUM_KV, HD],
        strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
        coord_names=("physical_block", "token", "kv_head", "dim"),
    )
    if T == BS:
        paged_kv_desc = _kv_base.transform(
            indirect(
                "tile_idx", into="physical_block", table=block_tables, base=seq_base
            ),
            unmerge("linear_half", into=("token", "dim"), dims=(T, HD)),
        )
    else:
        assert BS % T == 0, "3D tile_size_override must divide block_size"
        BLOCKS_PER_CACHE_BLOCK = BS // T
        paged_kv_desc = _kv_base.transform(
            unmerge(
                "tile_idx",
                into=("linear_block_idx", "tile_within_block"),
                dims=(1 << 24, BLOCKS_PER_CACHE_BLOCK),
            ),
            indirect(
                "linear_block_idx",
                into="physical_block",
                table=block_tables,
                base=seq_base,
            ),
            unmerge("linear_half", into=("token_in_tile", "dim"), dims=(T, HD)),
            embed(
                ("tile_within_block", "token_in_tile"),
                into="token",
                strides=(T, 1),
            ),
        )

    def _issue_k_load(kv_tile_idx: Value, buf_idx: Value) -> None:
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        K_buf_base = b.smem_ptr_add(K_lds_addr, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_half,
                kv_head=kv_head_idx,
            )
            k_dst = b.smem_ptr_add(K_buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(
                key_rsrc, k_dst, voff, zero_soff, ASYNC_LDS_DWORDS
            )

    def _issue_v_load(kv_tile_idx: Value, buf_idx: Value) -> None:
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        V_buf_base = b.smem_ptr_add(V_lds_addr, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_half,
                kv_head=kv_head_idx,
            )
            v_dst = b.smem_ptr_add(V_buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(
                value_rsrc, v_dst, voff, zero_soff, ASYNC_LDS_DWORDS
            )

    # Wide KV feed (opt-in): 8-half (16-byte / b128) synchronous global->register
    # ->LDS loads, mirroring the in-kernel fp8 loader and the 2D cfvst V store.
    # Fills K/V_lds identically to the async path (every linear half written once
    # with the correct paged value) -- only the per-lane load WIDTH changes from
    # 1 DWORD to 4 DWORDs, cutting load-instruction count 4x. The dim axis is
    # contiguous in the paged cache (stride = KV_BYTES) so 8 consecutive halves
    # are a legal coalesced b128 load.
    WIDE_ELEMS = 8
    WIDE_OK = (T * HD) % (THREADS * WIDE_ELEMS) == 0
    wide_chunks_per_thread = (T * HD) // (THREADS * WIDE_ELEMS) if WIDE_OK else 0

    def _issue_wide_load(src, lds, kv_tile_idx: Value, buf_idx: Value) -> None:
        for call in range(wide_chunks_per_thread):
            chunk_id = b.add(b.mul(b.const_i32(call), b.const_i32(THREADS)), tid)
            linear_half = b.mul(chunk_id, b.const_i32(WIDE_ELEMS))
            row = b.div(linear_half, b.const_i32(HD))
            col = b.mod(linear_half, b.const_i32(HD))
            voff, _ = paged_kv_desc.offset(
                b, tile_idx=kv_tile_idx, linear_half=linear_half, kv_head=kv_head_idx
            )
            # paged_kv_desc uses BYTE strides; global_load_vN is element-indexed,
            # so convert (exact: every stride is a multiple of KV_BYTES).
            elem_off = b.div(voff, b.const_i32(KV_BYTES))
            vec = b.global_load_vN(src, elem_off, dtype, n=WIDE_ELEMS, align=16)
            b.smem_store_vN(lds, [buf_idx, row, col], vec, WIDE_ELEMS)

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

        Uses ``cvt_pk_f32_fp8x4`` (the gfx942 2D kernel validates this packed
        cvt; see ``dequant_fp8x8_to_dtype``). The scale is applied unfused as a
        plain ``v_pk_mul`` against an f32 scale -- NOT the E8M0 micro-scaling
        fused cvt -- so arbitrary per-tensor scales are exact.
        """
        scale = k_scale_p if lds_token == "K" else v_scale_p
        lds = K_lds if lds_token == "K" else V_lds
        src = key if lds_token == "K" else value
        assert fp8_elems_per_chunk == 8
        for call in range(fp8_chunks_per_thread):
            chunk_id = b.add(b.mul(b.const_i32(call), b.const_i32(THREADS)), tid)
            row = b.div(chunk_id, b.const_i32(HD // fp8_elems_per_chunk))
            col = b.mul(
                b.mod(chunk_id, b.const_i32(HD // fp8_elems_per_chunk)),
                b.const_i32(fp8_elems_per_chunk),
            )
            linear_half_first = b.add(b.mul(row, b.const_i32(HD)), col)
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_half_first,
                kv_head=kv_head_idx,
            )
            fp8_vec = b.global_load_vN(
                src, voff, FP8E4M3, n=fp8_elems_per_chunk, align=fp8_elems_per_chunk
            )
            packed = dequant_fp8x8_to_dtype(b, fp8_vec, scale, dtype)
            b.smem_store_vN(lds, [buf_idx, row, col], packed, fp8_elems_per_chunk)

    WIDE_KV = spec.use_wide_kv_load and not KV_FP8 and WIDE_OK

    def _issue_k(tile_idx: Value, buf_idx: Value) -> None:
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "K")
        elif WIDE_KV:
            _issue_wide_load(key, K_lds, tile_idx, buf_idx)
        else:
            _issue_k_load(tile_idx, buf_idx)

    def _issue_v(tile_idx: Value, buf_idx: Value) -> None:
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "V")
        elif WIDE_KV:
            _issue_wide_load(value, V_lds, tile_idx, buf_idx)
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

        # ---------------- QK (gfx942 narrow 16x16x16, K-step 16) ----------------
        # A (Q) operand per lane: row = lane%16, K = k*16 + lane_rg*4 + 0..3
        # (<4 x dtype>). gfx950 used 8-elem A and _mfma_16x16x32 (K-step 32).
        A_kits = []
        for k in range(QK_K_ITERS):
            q_col_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
            A_kits.append(b.smem_load_vN(Q_lds, lane_col, q_col_off, dtype=dtype, n=4))
        S_n = []
        for n in range(QK_N_TILES):
            acc_v = b.zero_vec_f32(4)
            for k in range(QK_K_ITERS):
                # B (K^T) operand per lane: col = n*16 + lane%16,
                # K = k*16 + lane_rg*4 + 0..3 (<4 x dtype>).
                kc_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                k_row = b.add(b.const_i32(n * 16), lane_col)
                B_v = b.smem_load_vN(K_lds, cur_buf, k_row, kc_off, dtype=dtype, n=4)
                acc_v = _mfma_16x16x16(b, dtype, A_kits[k], B_v, acc_v)
            S_n.append(acc_v)

        _issue_v(kv_tile_iv, cur_buf)
        _issue_k(safe_next_tile, nxt_buf)

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
            if USE_INVARIANT_HOIST:
                qp_r = hoist_qp_r[reg]
                row_ok = hoist_row_ok[reg]
                causal_lim = hoist_causal_lim[reg]
            else:
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
                if not USE_INVARIANT_HOIST:
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
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sync()
        else:
            b.s_waitcnt(vmcnt=kv_calls_per_tile, lgkmcnt=kv_calls_per_tile)
            b.sync()

        # ---------------- PV (gfx942 narrow 16x16x16, strided-V B) ----------
        # gfx950 read the V B-operand via ds_read_tr16_b64 (transpose read,
        # absent on gfx942) and ran a wide-K PV. gfx942 builds the <4 x dtype>
        # B-operand from 4 strided LDS loads that reproduce the EXACT per-lane
        # (row, col) the transpose read delivers for a 16x16x16 atom over
        # V_lds[buf, T, HD] at (row base = k*16, col base = n*16):
        #   lane l = 16*k_chunk + n_col  (k_chunk = lane/16, n_col = lane%16)
        #   B[j] = V_lds[buf, k*16 + k_chunk*4 + j, n*16 + n_col], j in 0..3
        new_acc = []
        for n in range(PV_N_TILES):
            scaled_comps = []
            for reg in range(4):
                e = b.vec_extract(acc_vals[n], reg)
                scaled_comps.append(b.fmul(e, alpha_regs[reg]))
            acc_v = b.vec_pack(scaled_comps, F32)

            v_n_col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
            v_k_chunk_base = b.mul(lane_rg, b.const_i32(4))

            def _strided_v_b_operand(
                k_iter: int, v_n_col=v_n_col, v_k_chunk_base=v_k_chunk_base
            ) -> Value:
                bv = b.zero_vec(dtype, 4)
                for j in range(4):
                    v_row = b.add(b.const_i32(k_iter * 16 + j), v_k_chunk_base)
                    elem = b.vec_extract(
                        b.smem_load_vN(
                            V_lds, cur_buf, v_row, v_n_col, dtype=dtype, n=1
                        ),
                        0,
                    )
                    bv = b.vec_insert(bv, elem, j)
                return bv

            for k in range(PV_K_ITERS):
                p_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                A_p = b.smem_load_vN(P_lds, lane_col, p_off, dtype=dtype, n=4)
                B_v = _strided_v_b_operand(k)
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

    for n in range(PV_N_TILES):
        for reg in range(4):
            if USE_INVARIANT_HOIST:
                row = hoist_row[reg]
                qp_r = hoist_qp_r[reg]
                qh_r = hoist_qh_r[reg]
                row_ok = hoist_row_ok[reg]
            else:
                row = _mfma_16x16_c_row(b, tid, reg)
                qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
                qh_r = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
                )
                row_ok = b.land(
                    b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
                )
            col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
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
        if USE_INVARIANT_HOIST:
            qp_r = hoist_qp_r[reg]
            qh_r = hoist_qh_r[reg]
            row_ok = hoist_row_ok[reg]
        else:
            row = _mfma_16x16_c_row(b, tid, reg)
            qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
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
# Reduce kernel (arch-neutral; byte-for-byte port of the gfx950 reduce kernel)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnifiedAttentionReduceTiledSpec:
    head_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    num_segments: int
    waves_per_eu: Optional[int] = None

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_uattn_reduce_tiled_gfx942",
            f"d{self.head_size}",
            f"h{self.num_query_heads}",
            f"seg{self.num_segments}",
            self.dtype,
        )


def build_unified_attention_reduce_tiled(
    spec: UnifiedAttentionReduceTiledSpec,
    *,
    arch: str = "gfx942",
) -> KernelDef:
    """Combine the per-segment partial state into the final fp16/bf16 output."""
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

    base_ml, _ = ml_desc_red.offset(b, token=q_token, head=q_head, seg=b.const_i32(0))

    SEG_PER_LANE = (NUM_SEG + THREADS - 1) // THREADS
    factor_lds = b.smem_alloc_f32([NUM_SEG], name_hint="seg_factor")

    # ---- pass 1: per-lane partial max over the owned segments ----
    seg_idx_of = []
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
        ms_finite = b.fcmp("ogt", ms, neg_inf)
        factor_raw = b.exp2(b.fsub(ms, overall_max))
        factor = b.select(ms_finite, factor_raw, zero_f)
        local_den = b.fadd(local_den, b.fmul(ls, factor))
        if in_rng is None:
            b.smem_store_vN_f32(factor_lds, [sv_safe], factor, 1)
        else:
            with b.scf_if(in_rng):
                b.smem_store_vN_f32(factor_lds, [sv], factor, 1)
    overall_expsum = _wave64_reduce_sum(b, local_den)
    safe_expsum = b.fcmp("oeq", overall_expsum, zero_f)
    inv_l = b.select(safe_expsum, zero_f, b.rcp(overall_expsum))

    b.sync()

    # ---- pass 3: per-element reduce + normalize + write ----
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
