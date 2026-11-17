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

from dataclasses import dataclass, field
from typing import Optional, Tuple

from kernels.common._attention_strided_kv import (
    declare_strided_kv_params,
    strided_kv_offset,
    strided_kv_resource,
)
from kernels.common._lse_store import lse_store

from rocke.core.ir import (
    BF16,
    CACHE_POLICY_NAMES,
    F16,
    F32,
    FP8E4M3,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    TemporalHint,
    Type,
    Value,
    cache_policy_from_name,
)
from rocke.helpers.atoms import MfmaAtom, make_c_warp_dstr_encoding
from rocke.helpers.attention import (
    apply_softcap_log2 as _apply_softcap,
    binary_search_seq_idx as _binary_search_seq_idx_helper,
    binary_search_seq_idx_iters,
    dequant_fp8x8_to_dtype,
    mfma_16x16x16_for_dtype as _mfma_16x16x16,
    wave64_reduce_max as _wave64_reduce_max,
    wave64_reduce_sum as _wave64_reduce_sum,
)
from rocke.helpers.distribution import make_static_tile_distribution
from rocke.helpers.transforms import TensorDescriptor, embed, indirect, unmerge

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
    # Inert: the kernel computes the query-row invariants once per lane before
    # the KV loop (S^T layout). Kept so saved specs that set it still load.
    use_invariant_hoist: bool = False
    # Wide KV feed: replace the gfx942 1-DWORD (4-byte) async buffer_load_lds DMA
    # with wide 128-bit (8-half) synchronous global->register->LDS loads (the same
    # vehicle the in-kernel fp8 path and the 2D cfvst V feed use). 4x fewer load
    # instructions -> less VMEM issue pressure on the memory-bound decode segment.
    # fp16/bf16 only (the fp8 path already loads wide). Signature-tracked.
    use_wide_kv_load: bool = False
    # 64-bit paged-KV addressing. Accepted for signature parity with the gfx950
    # spec, but this kernel has no i64 path: the builders set it only for a
    # cache past what the K/V loader reads (``i32_kv_cache_bytes``), and it is
    # refused below so an oversized cache cannot build a kernel that misreads it.
    use_i64_kv_addr: bool = False
    # Direct non-paged [B,H,S,D] addressing; D remains contiguous.
    kv_layout: str = "paged"
    # All-decode launch grid: one CTA per sequence along grid x
    # (``seq_idx = block_id_x``, single query block per sequence) instead of
    # the prefill upper bound ``total_q // BLOCK_Q + num_seqs`` with a binary
    # search over ``query_start_len``. Legal only when every sequence has at
    # most one query token (``UnifiedAttentionProblem.all_decode``); the
    # dispatcher derives it from the problem (``_use_decode_grid``) and the
    # launcher sizes grid x to ``num_seqs``. It could later be relaxed to
    # ``max_seqlen_q <= BLOCK_Q`` (one query block per sequence), which needs
    # its own checks of the per-row causal diagonal, sliding window and
    # per-segment KV range.
    use_decode_grid: bool = False
    # Cache policy of the K/V copies, one of ``CACHE_POLICY_NAMES`` (as the
    # gfx950 3D and attention 2D ``kv_cache_policy``). The dispatcher sets
    # "stream" (NT) for all-decode batches, whose K/V bytes are each read once
    # (``_kv_cache_policy_3d``). The wide loader's only cache control is the
    # non-temporal hint, so it takes "all" or "stream"; the fp8 K/V loader keeps
    # "all".
    kv_cache_policy: str = "all"
    # K/V LDS tiles store each row's 16-byte chunks XOR-permuted by the row
    # (chunk q of row r at q ^ (r % (head_size // 8))), which spreads the QK
    # and PV reads over the LDS banks. The wide loader stores the permuted
    # chunks; the async copies fetch them (permuted source chunks, as on
    # gfx950). Either way the tiles stay unpadded. Implemented for fp16/bf16
    # K/V, which can be extended to fp8 K/V.
    use_kv_lds_swizzle: bool = False
    # The wave loads the segment's page-table entries 64 at a time, one per
    # lane, and each tile reads its page id from those registers (v_readlane),
    # so no page-table load sits on the KV loop's critical path. Paged K/V;
    # implemented for fp16/bf16, which can be extended to the fp8 loader.
    use_wave_page_id_load: bool = False

    def __post_init__(self):
        if self.num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        if self.kv_layout not in ("paged", "strided"):
            raise ValueError("kv_layout must be 'paged' or 'strided'")
        if self.kv_layout == "strided" and (
            self.kv_storage_dtype is not None
            or self.use_i64_kv_addr
            or self.use_wide_kv_load
        ):
            raise ValueError(
                "strided KV requires fp16/bf16 async loads without paged i64 addressing"
            )
        if self.use_i64_kv_addr:
            raise NotImplementedError(
                "gfx942 tiled 3D kernel does not support use_i64_kv_addr "
                "(paged KV cache > 2 GiB) yet"
            )
        if self.kv_storage_dtype is not None and self.kv_storage_dtype != "fp8e4m3":
            raise ValueError(
                f"kv_storage_dtype must be None or 'fp8e4m3' (got {self.kv_storage_dtype!r})"
            )
        if self.kv_cache_policy not in CACHE_POLICY_NAMES:
            raise ValueError(
                f"kv_cache_policy must be one of {', '.join(CACHE_POLICY_NAMES)} "
                f"(got {self.kv_cache_policy!r})"
            )
        if self.kv_cache_policy != "all" and self.kv_storage_dtype is not None:
            raise ValueError("kv_cache_policy applies to fp16/bf16 KV storage only")
        if self.use_wide_kv_load and self.kv_cache_policy not in ("all", "stream"):
            raise ValueError(
                "the wide K/V loader takes kv_cache_policy 'all' or 'stream'"
            )
        if self.use_kv_lds_swizzle and self.kv_storage_dtype is not None:
            raise ValueError("use_kv_lds_swizzle needs fp16/bf16 KV")
        if self.use_wave_page_id_load and (
            self.kv_layout != "paged" or self.kv_storage_dtype is not None
        ):
            raise ValueError("use_wave_page_id_load needs paged fp16/bf16 KV")

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
        return binary_search_seq_idx_iters(self.num_seqs)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_uattn3d_tiled_gfx942",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            f"seg{self.num_segments}",
            self.dtype,
            f"kv{self.kv_storage_dtype}" if self.kv_storage_dtype else "",
            "fnuz" if self.kv_storage_dtype else "",
            "sinks" if self.use_sinks else "",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "softcap" if self.has_softcap else "",
            "alibi" if self.use_alibi else "",
            "qqb" if self.use_qq_bias else "",
            "wkv" if self.use_wide_kv_load else "",
            "dg" if self.use_decode_grid else "",
            f"kvcp{self.kv_cache_policy}" if self.kv_cache_policy != "all" else "",
            "kvswz" if self.use_kv_lds_swizzle else "",
            "wpid" if self.use_wave_page_id_load else "",
            "stridedkv" if self.kv_layout == "strided" else "",
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


# The wide K/V loader moves 8 halves per lane over the 64-lane wave; a tile it
# cannot split evenly falls back to the async copies.
_WIDE_KV_ELEMS = 8


def wide_kv_load_fits(tile_size: int, head_size: int) -> bool:
    """True when ``use_wide_kv_load`` takes effect for this tile."""
    return (tile_size * head_size) % (64 * _WIDE_KV_ELEMS) == 0


def i32_kv_cache_bytes(
    *, kv_fp8: bool, use_wide_kv_load: bool, tile_size: int, head_size: int
) -> int:
    """Largest paged K/V cache, in bytes, the selected K/V loader reads.

    This kernel has no i64 KV path. The fp8 dequant and wide loaders are flat
    global loads indexed by an i32 byte offset, correct below 2^31 bytes. The
    async copies read through a buffer resource bounded at 0x7FFF0000 bytes.
    """
    if kv_fp8 or (use_wide_kv_load and wide_kv_load_fits(tile_size, head_size)):
        return 0x8000_0000
    return 0x7FFF_0000


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
    KV_FP8 = spec.kv_storage_dtype == "fp8e4m3"
    KV_CACHE_BITS = cache_policy_from_name(spec.kv_cache_policy)
    KV_TEMPORAL_HINT = (
        TemporalHint.STREAMING
        if spec.kv_cache_policy == "stream"
        else TemporalHint.DEFAULT
    )
    KV_BYTES = 1 if KV_FP8 else 2
    kv_io_dtype = FP8E4M3 if KV_FP8 else dtype
    KV_SWZ = spec.use_kv_lds_swizzle
    KV_CHUNKS_PER_ROW = HD // 8  # 16-byte chunks per K/V LDS row

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
    if spec.kv_layout == "strided":
        k_strides = declare_strided_kv_params(b, "k")
        v_strides = declare_strided_kv_params(b, "v")

    q_block_global_idx = b.block_id_x()
    kv_head_idx = b.block_id_y()
    seg_idx = b.block_id_z()
    tid = b.thread_id_x()

    if spec.use_decode_grid:
        # One CTA per sequence along x, one query block each: no search.
        seq_idx = q_block_global_idx
    else:
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
    if spec.use_decode_grid:
        q_block_local_idx = b.const_i32(0)
    else:
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
    Q_lds = b.smem_alloc(dtype, [BLOCK_M, HD], name_hint="Qlds")
    K_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Klds")
    # One V slot: V(i) is loaded after QK(i) and consumed by PV(i) in the same
    # iteration, and the next iteration's leading wait-all retires PV(i) before
    # V(i+1) is issued, so a second slot would never hold a live tile. V_lds is
    # row-major; the V^T operand of the PV MFMA is built from strided LDS loads
    # (gfx942 has no transpose read).
    V_lds = b.smem_alloc(dtype, [1, T, HD], name_hint="Vlds")

    def _row_reduce(x: Value, op) -> Value:
        # Combine x over the 4 lanes {l, l^16, l^32, l^48} that hold the 4
        # token groups of one query row: ds_swizzle (xor 16), then ds_bpermute
        # (xor 32; gfx942 has no permlane32_swap).
        x = op(x, b.warp_shuffle_xor(x, 16))
        return op(x, b.warp_shuffle_xor(x, 32))

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

    # The QK MFMA computes S^T = K Q^T: lane l owns query row (BLOCK_M row)
    # l % 16 and holds tokens 4 * (l // 16) + 0..3 of each 16-token block, so
    # the online softmax keeps one (m, l) pair per lane.
    if USE_SINKS:
        # Triton's 3D applies sinks only when segm_idx == 0.
        seg0 = b.cmp_eq(seg_idx, b.const_i32(0))
        qh = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(lane_col, b.const_i32(NQK))
        )
        qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
        sink_h = b.global_load(sinks, qh, dtype, align=2)
        sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
        sink_with_mask = b.select(qh_in, sink_f, neg_inf)
        m_init = b.select(seg0, sink_with_mask, neg_inf)
    else:
        m_init = neg_inf

    acc_zero = b.zero_vec_f32(4)

    iter_args = [("m", m_init), ("l", one_f)]
    for n in range(PV_N_TILES):
        iter_args.append((f"acc{n}", acc_zero))

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
    if spec.kv_layout == "strided":
        key_rsrc = strided_kv_resource(b, key, k_strides, seq_idx, kv_head_idx)
        value_rsrc = strided_kv_resource(b, value, v_strides, seq_idx, kv_head_idx)
    else:
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

    seq_base = b.mul(seq_idx, bt_stride_p) if spec.kv_layout == "paged" else None
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

    def _issue_k_load(
        kv_tile_idx: Value, buf_idx: Value, page_id: Value | None
    ) -> None:
        if page_id is not None:
            # The tile base goes into soffset; each copy adds its lane's
            # offset within the page.
            soff = _tile_base(page_id, kv_tile_idx, kv_stride_blk_b, kv_stride_tok_b)
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        K_buf_base = b.smem_ptr_add(K_lds_addr, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            if spec.kv_layout == "strided":
                voff = strided_kv_offset(
                    b,
                    k_strides,
                    kv_tile_idx,
                    _kv_src_linear(linear_half),
                    seq_len,
                    HD,
                    T,
                )
            elif page_id is not None:
                src_half = _kv_src_linear(linear_half)
                voff = b.add(
                    b.add(
                        b.mul(
                            b.div(src_half, b.const_i32(HD)),
                            b.const_i32(kv_stride_tok_b),
                        ),
                        b.mul(kv_head_idx, b.const_i32(kv_stride_h_b)),
                    ),
                    b.mul(b.mod(src_half, b.const_i32(HD)), b.const_i32(KV_BYTES)),
                )
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=_kv_src_linear(linear_half),
                    kv_head=kv_head_idx,
                )
            k_dst = b.smem_ptr_add(K_buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(
                key_rsrc,
                k_dst,
                voff,
                soff if page_id is not None else zero_soff,
                ASYNC_LDS_DWORDS,
                coherency=KV_CACHE_BITS,
            )

    def _issue_v_load(
        kv_tile_idx: Value, buf_idx: Value, page_id: Value | None
    ) -> None:
        if page_id is not None:
            # The tile base goes into soffset; each copy adds its lane's
            # offset within the page.
            soff = _tile_base(page_id, kv_tile_idx, kv_stride_blk_b, kv_stride_tok_b)
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        V_buf_base = b.smem_ptr_add(V_lds_addr, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            if spec.kv_layout == "strided":
                voff = strided_kv_offset(
                    b,
                    v_strides,
                    kv_tile_idx,
                    _kv_src_linear(linear_half),
                    seq_len,
                    HD,
                    T,
                )
            elif page_id is not None:
                src_half = _kv_src_linear(linear_half)
                voff = b.add(
                    b.add(
                        b.mul(
                            b.div(src_half, b.const_i32(HD)),
                            b.const_i32(kv_stride_tok_b),
                        ),
                        b.mul(kv_head_idx, b.const_i32(kv_stride_h_b)),
                    ),
                    b.mul(b.mod(src_half, b.const_i32(HD)), b.const_i32(KV_BYTES)),
                )
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=_kv_src_linear(linear_half),
                    kv_head=kv_head_idx,
                )
            v_dst = b.smem_ptr_add(V_buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(
                value_rsrc,
                v_dst,
                voff,
                soff if page_id is not None else zero_soff,
                ASYNC_LDS_DWORDS,
                coherency=KV_CACHE_BITS,
            )

    # Wide KV feed (opt-in): 8-half (16-byte / b128) synchronous global->register
    # ->LDS loads, mirroring the in-kernel fp8 loader and the 2D cfvst V store.
    # Fills K/V_lds identically to the async path (every linear half written once
    # with the correct paged value) -- only the per-lane load WIDTH changes from
    # 1 DWORD to 4 DWORDs, cutting load-instruction count 4x. The dim axis is
    # contiguous in the paged cache (stride = KV_BYTES) so 8 consecutive halves
    # are a legal coalesced b128 load.
    WIDE_ELEMS = _WIDE_KV_ELEMS
    WIDE_OK = wide_kv_load_fits(T, HD)
    wide_chunks_per_thread = (T * HD) // (THREADS * WIDE_ELEMS) if WIDE_OK else 0

    def _kv_col(row: Value, col: Value) -> Value:
        # LDS column of K/V element (row, col): its 16-byte chunk is permuted
        # by the row under the swizzle.
        if not KV_SWZ:
            return col
        chunk = b.xor(
            b.div(col, b.const_i32(8)), b.mod(row, b.const_i32(KV_CHUNKS_PER_ROW))
        )
        return b.add(b.mul(chunk, b.const_i32(8)), b.mod(col, b.const_i32(8)))

    def _kv_src_linear(linear_half: Value) -> Value:
        # The async copies write LDS in lane order, so under the swizzle each
        # lane fetches the element its LDS position holds (_kv_col is its own
        # inverse).
        if not KV_SWZ:
            return linear_half
        row = b.div(linear_half, b.const_i32(HD))
        return b.add(
            b.mul(row, b.const_i32(HD)),
            _kv_col(row, b.mod(linear_half, b.const_i32(HD))),
        )

    WAVE_PAGE_ID_LOAD = spec.use_wave_page_id_load
    if WAVE_PAGE_ID_LOAD:
        assert not KV_FP8 and spec.kv_layout == "paged"
    TILES_PER_PAGE = BS // T

    def _tile_page(kv_tile_idx: Value) -> Value:
        # The segment-local page index of a tile (tiles never straddle pages).
        if TILES_PER_PAGE == 1:
            return kv_tile_idx
        return b.div(kv_tile_idx, b.const_i32(TILES_PER_PAGE))

    def _tile_base(
        page_id: Value, kv_tile_idx: Value, page_stride: int, token_stride: int
    ) -> Value:
        # Offset of a tile's first token: the page's, plus the tile's rows
        # before it in the page (sub-page tiles).
        base = b.mul(page_id, b.const_i32(page_stride))
        if TILES_PER_PAGE == 1:
            return base
        in_page = b.mod(kv_tile_idx, b.const_i32(TILES_PER_PAGE))
        return b.add(base, b.mul(in_page, b.const_i32(T * token_stride)))

    def _issue_wide_load(
        src, lds, kv_tile_idx: Value, buf_idx: Value, page_id: Value | None
    ) -> None:
        for call in range(wide_chunks_per_thread):
            chunk_id = b.add(b.mul(b.const_i32(call), b.const_i32(THREADS)), tid)
            linear_half = b.mul(chunk_id, b.const_i32(WIDE_ELEMS))
            row = b.div(linear_half, b.const_i32(HD))
            col = b.mod(linear_half, b.const_i32(HD))
            if page_id is not None:
                # The tile base plus the chunk's offset within the page, which
                # is loop-invariant: no page-table lookup per copy.
                elem_off = b.add(
                    _tile_base(page_id, kv_tile_idx, BS * NUM_KV * HD, NUM_KV * HD),
                    b.add(
                        b.add(
                            b.mul(row, b.const_i32(NUM_KV * HD)),
                            b.mul(kv_head_idx, b.const_i32(HD)),
                        ),
                        col,
                    ),
                )
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half,
                    kv_head=kv_head_idx,
                )
                # paged_kv_desc uses BYTE strides; global_load_vN is
                # element-indexed, so convert (exact: every stride is a
                # multiple of KV_BYTES).
                elem_off = b.div(voff, b.const_i32(KV_BYTES))
            vec = b.global_load_vN(
                src,
                elem_off,
                dtype,
                n=WIDE_ELEMS,
                align=16,
                temporal_hint=KV_TEMPORAL_HINT,
            )
            b.smem_store_vN(lds, [buf_idx, row, _kv_col(row, col)], vec, WIDE_ELEMS)

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

    def _issue_k(tile_idx: Value, buf_idx: Value, page_id: Value | None) -> None:
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "K")
        elif WIDE_KV:
            _issue_wide_load(key, K_lds, tile_idx, buf_idx, page_id)
        else:
            _issue_k_load(tile_idx, buf_idx, page_id)

    def _issue_v(tile_idx: Value, buf_idx: Value, page_id: Value | None) -> None:
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "V")
        elif WIDE_KV:
            _issue_wide_load(value, V_lds, tile_idx, buf_idx, page_id)
        else:
            _issue_v_load(tile_idx, buf_idx, page_id)

    first_page_id = None
    if WAVE_PAGE_ID_LOAD:
        # The wave holds a 64-page window of the segment's page table, one id
        # per lane. A tile lies within one page, and it starts below seq_len,
        # so its page is always one of the sequence's.
        seq_pages = b.div(b.add(seq_len, b.const_i32(BS - 1)), b.const_i32(BS))
        seg_first_page = _tile_page(tile_start)

        def _load_page_id_window(rel_page: Value) -> Value:
            # Lane l: the id of the segment's page rel_page + l (0 past the
            # sequence's last page: never used).
            pg = b.add(b.add(seg_first_page, rel_page), tid)
            return b.masked_global_load(
                block_tables,
                b.add(seq_base, pg),
                b.cmp_lt(pg, seq_pages),
                b.const_i32(0),
                dtype=I32,
                align=4,
            )

        def _window_tile_page_id(page_id_window: Value, kv_tile_idx: Value) -> Value:
            lane = b.mod(
                b.sub(_tile_page(kv_tile_idx), seg_first_page), b.const_i32(64)
            )
            return b.readlane(page_id_window, lane)

        def _refill_page_id_window(page_id_window: Value, kv_tile_idx: Value) -> Value:
            # The window for the next iteration: if tile kv_tile_idx + 2 starts
            # a new window, load it now; the loop-top wait completes it before
            # the next iteration reads it.
            t2 = b.add(kv_tile_idx, b.const_i32(2))
            rel2 = b.sub(_tile_page(t2), seg_first_page)
            cross = b.land(
                b.cmp_eq(b.mod(rel2, b.const_i32(64)), b.const_i32(0)),
                b.cmp_lt(t2, tile_end),
            )
            if TILES_PER_PAGE > 1:
                # Only the page's first tile starts the window.
                first_in_page = b.cmp_eq(
                    b.mod(t2, b.const_i32(TILES_PER_PAGE)), b.const_i32(0)
                )
                cross = b.land(cross, first_in_page)
            # A wave-uniform 0/1-trip loop is an if with a result (scf_if has
            # none): 0 trips keep the window, 1 trip loads the next one. It
            # compiles to a scalar branch around the load.
            refill = b.scf_for_iter(
                b.const_i32(0),
                b.select(cross, b.const_i32(1), b.const_i32(0)),
                b.const_i32(1),
                [("pid_window_r", page_id_window)],
                iv_name="pid_window_load",
            )
            with refill as (_iv, _w):
                b.scf_yield(_load_page_id_window(rel2))
            return refill.results[0]

        first_page_id_window = _load_page_id_window(b.const_i32(0))
        first_page_id = _window_tile_page_id(first_page_id_window, tile_start)
    _issue_k(tile_start, b.const_i32(0), first_page_id)

    cur_buf_init = b.const_i32(0)
    iter_args.append(("cur_buf", cur_buf_init))
    if WAVE_PAGE_ID_LOAD:
        iter_args.append(("pid", first_page_id))
        iter_args.append(("pid_window", first_page_id_window))

    # LICM hoist: per-row invariants that are constant across all KV tiles.
    # qp_r, qh_r, row_ok, and causal_lim depend only on CTA-level constants
    # (qb_start_pos, kv_head_idx, context_len, cur_batch_q_len) and the
    # lane's query row — none of which change inside the KV loop.
    # ALiBi slopes are per query-head: also loop-invariant. Hoisting them here
    # mirrors the 2D kernel's hoist_* block (attention_tiled_2d.py:2851-2898)
    # and avoids re-emitting the global loads and arithmetic every tile.
    qp_r = b.add(qb_start_pos, b.div(lane_col, b.const_i32(NQK)))
    qh_r = b.add(
        b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(lane_col, b.const_i32(NQK))
    )
    row_ok = b.land(
        b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
    )
    causal_lim = b.add(context_len, qp_r)

    if USE_ALIBI:
        qh_ok = b.cmp_lt(qh_r, b.const_i32(NUM_QH))
        alibi_slope = b.masked_global_load(
            alibi_slopes_ptr, qh_r, qh_ok, b.const_f32(0.0), dtype=F32, align=4
        )

    # The QK B operand (Q^T) is read from LDS once; every tile's MFMAs take it
    # from registers.
    Q_kits = []
    for k in range(QK_K_ITERS):
        q_col_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
        Q_kits.append(b.smem_load_vN(Q_lds, lane_col, q_col_off, dtype=dtype, n=4))

    kvloop = b.scf_for_iter(
        tile_start, tile_end, b.const_i32(1), iter_args, iv_name="kv_tile"
    )
    with kvloop as (kv_tile_iv, carry):
        m_val = carry[0]
        l_val = carry[1]
        acc_vals = [carry[2 + n] for n in range(PV_N_TILES)]
        cur_buf = carry[2 + PV_N_TILES]
        cur_page_id = carry[3 + PV_N_TILES] if WAVE_PAGE_ID_LOAD else None
        nxt_buf = b.sub(b.const_i32(1), cur_buf)
        v_buf = b.const_i32(0)
        tile_off = b.mul(kv_tile_iv, b.const_i32(T))

        next_tile_iv_raw = b.add(kv_tile_iv, b.const_i32(1))
        in_range_next = b.cmp_lt(next_tile_iv_raw, tile_end)
        safe_next_tile = b.select(in_range_next, next_tile_iv_raw, kv_tile_iv)

        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.sync()
        # Page id of the next tile, for its K copies below.
        nxt_page_id = None
        if WAVE_PAGE_ID_LOAD:
            # Read from the window in registers: no load in the loop.
            page_id_window = carry[4 + PV_N_TILES]
            nxt_page_id = _window_tile_page_id(page_id_window, safe_next_tile)

        # QK
        S_n = []
        for n in range(QK_N_TILES):
            acc_v = b.zero_vec_f32(4)
            for k in range(QK_K_ITERS):
                kc_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                k_row = b.add(b.const_i32(n * 16), lane_col)
                K_v = b.smem_load_vN(
                    K_lds, cur_buf, k_row, _kv_col(k_row, kc_off), dtype=dtype, n=4
                )
                acc_v = _mfma_16x16x16(b, dtype, K_v, Q_kits[k], acc_v)
            S_n.append(acc_v)

        _issue_v(kv_tile_iv, v_buf, cur_page_id)
        _issue_k(safe_next_tile, nxt_buf, nxt_page_id)
        if WAVE_PAGE_ID_LOAD:
            nxt_page_id_window = _refill_page_id_window(page_id_window, kv_tile_iv)

        # See attention_tiled_2d.py for the rationale on applying ALiBi /
        # QQ-bias before the select-with-(-inf) (equivalent to Triton's
        # post-select add for finite biases via IEEE -inf semantics, with
        # better robustness against compiler reordering).
        masked = {}
        for reg in range(4):
            # S^T element reg of block n: token 4 * lane_rg + reg of the block.
            tok_in_blk = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
            for n in range(QK_N_TILES):
                col_abs = b.add(
                    b.add(tile_off, b.mul(b.const_i32(n), b.const_i32(16))),
                    tok_in_blk,
                )
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
                    add_term = b.fmul(b.fmul(alibi_slope, pos_f), rcp_ln2)
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

        # Row max and sum: this lane's 4 * QK_N_TILES tokens, then the row's
        # other 3 lanes. m_new is the running max (m_val included), so
        # alpha = exp2(m_val - m_safe) is at most 1, as exp2_fast requires; a
        # tile-local max would overflow alpha once a tile scores ~128 log2 units
        # below an earlier one. Until the row has an unmasked key (or, in
        # segment 0, a sink), m_new is -inf and the exponents subtract m_safe = 0
        # instead, so P and alpha are 0, not NaN. A row with neither in the
        # segment stores m = -inf, which the reduce kernel weights by 0.
        local_max = neg_inf
        for v in masked.values():
            local_max = b.fmax(local_max, v)
        tile_max = _row_reduce(local_max, b.fmax)
        m_new = b.fmax(m_val, tile_max)
        ok = b.fcmp("ogt", m_new, neg_inf)
        m_safe = b.select(ok, m_new, zero_f)
        p_regs = {}
        sum_p = zero_f
        for key, v in masked.items():
            p = b.exp2_fast(b.fsub(v, m_safe))
            p_regs[key] = p
            sum_p = b.fadd(sum_p, p)
        l_local = _row_reduce(sum_p, b.fadd)

        alpha = b.exp2_fast(b.fsub(m_val, m_safe))
        new_l_val = b.fadd(b.fmul(l_val, alpha), l_local)
        if KV_FP8:
            # FP8 sync loader has no in-flight async work.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sync()
        else:
            b.s_waitcnt(vmcnt=kv_calls_per_tile, lgkmcnt=kv_calls_per_tile)
            b.sync()

        # O^T = V^T P^T, one 16-token block per MFMA: A = V^T (head-dim rows
        # n*16 + 0..15 via 4 strided LDS loads; gfx942 has no transpose read),
        # B = P^T straight from this lane's registers (its query row, tokens
        # 4 * lane_rg + 0..3).
        new_acc = []
        for n in range(PV_N_TILES):
            acc_v = b.vec_pack(
                [b.fmul(b.vec_extract(acc_vals[n], r), alpha) for r in range(4)], F32
            )
            v_col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
            v_row_base = b.mul(lane_rg, b.const_i32(4))
            for kb in range(QK_N_TILES):
                A_v = b.zero_vec(dtype, 4)
                for j in range(4):
                    v_row = b.add(b.const_i32(kb * 16 + j), v_row_base)
                    elem = b.vec_extract(
                        b.smem_load_vN(
                            V_lds, v_buf, v_row, _kv_col(v_row, v_col), dtype=dtype, n=1
                        ),
                        0,
                    )
                    A_v = b.vec_insert(A_v, elem, j)
                B_p = b.vec_pack(
                    [b.cast_f32_to(p_regs[(kb, r)], dtype) for r in range(4)], dtype
                )
                acc_v = _mfma_16x16x16(b, dtype, A_v, B_p, acc_v)
            new_acc.append(acc_v)

        yields = [m_new, new_l_val]
        for n in range(PV_N_TILES):
            yields.append(new_acc[n])
        yields.append(nxt_buf)
        if WAVE_PAGE_ID_LOAD:
            yields.append(nxt_page_id)
            yields.append(nxt_page_id_window)
        b.scf_yield(*yields)

    # ---------------- write segment workspace ----------------
    # The workspace layouts ``segm_output[token, head, seg, dim]`` and
    # ``segm_{max,expsum}[token, head, seg]`` are encoded by the
    # ``seg_acc_desc`` / ``ml_desc`` coordinate transforms defined at the top
    # of this function. O^T tile n: lane l holds head dims
    # n * 16 + 4 * lane_rg + 0..3 of query row lane_col, stored as one vector;
    # the lanes with lane_rg == 0 write the row's m and l.
    final = kvloop.results
    m_final = final[0]
    l_final = final[1]
    acc_final = [final[2 + n] for n in range(PV_N_TILES)]

    qtoken = b.add(cu_q_start, qp_r)
    with b.scf_if(row_ok):
        for n in range(PV_N_TILES):
            d0 = b.add(b.const_i32(n * 16), b.mul(lane_rg, b.const_i32(4)))
            seg_acc_idx, _ = seg_acc_desc.offset(
                b, token=qtoken, head=qh_r, seg=seg_idx, dim=d0
            )
            b.global_store_vN(segm_output_ptr, seg_acc_idx, acc_final[n], 4, align=16)
    ml_idx, _ = ml_desc.offset(b, token=qtoken, head=qh_r, seg=seg_idx)
    with b.scf_if(b.land(b.cmp_eq(lane_rg, b.const_i32(0)), row_ok)):
        b.global_store(segm_max_ptr, ml_idx, m_final, align=4)
        b.global_store(segm_expsum_ptr, ml_idx, l_final, align=4)

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
    # emit_lse: also write the natural-log softmax normalizer (LSE) as FP32
    #   [total_q, num_query_heads, 1] through a trailing ``lse_ptr`` argument.
    #   Off by default; off adds no LSE parameter and no LSE store. (The reduce IR
    #   still differs from before LSE existed: the masked-segment merge fix below
    #   applies with LSE on or off.)
    emit_lse: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.emit_lse) is not bool:
            raise ValueError(
                f"emit_lse must be a bool, got {type(self.emit_lse).__name__}"
            )
        if self.num_segments < 1:
            raise ValueError("num_segments must be >= 1")

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        parts = [
            f"d{self.head_size}",
            f"h{self.num_query_heads}",
            f"seg{self.num_segments}",
            self.dtype,
        ]
        if self.emit_lse:
            parts.append("lse")
        return kernel_name_join("rocke_uattn_reduce_tiled_gfx942", *parts)


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
    lse = (
        b.param(
            "lse_ptr",
            PtrType(F32, "global"),
            noalias=True,
            writeonly=True,
            align=4,
        )
        if spec.emit_lse
        else None
    )

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
        # A segment whose tiles were all masked for this row stores max=0 with
        # sum=0. Its max carries no mass, so it must not set overall_max (it
        # would underflow every real segment's factor) nor get a factor here:
        # treat it as -inf in both passes.
        ms = b.select(b.fcmp("ogt", ls, zero_f), ms, neg_inf)
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
        factor_raw = b.exp2_fast(b.fsub(ms, overall_max))
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

    if spec.emit_lse:
        # Every lane holds the same butterfly-reduced overall_max/expsum, so one
        # lane writes. overall_max is in log2 units (segm_max carries
        # scale*log2e); lse_store converts to natural log and writes -inf for a
        # row with no mass (every segment empty), whose O is already 0.
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            lse_index = b.add(b.mul(q_token, b.const_i32(NUM_QH)), q_head)
            lse_store(
                b,
                lse,
                lse_index,
                overall_max,
                overall_expsum,
                row_has_mass=b.fcmp("ogt", overall_expsum, zero_f),
            )

    return b.kernel


def supports_tiled_3d_spec(
    spec: UnifiedAttention3DTiledSpec, *, arch: str = "gfx942"
) -> Tuple[bool, str]:
    """Spec-shaped wrapper for ``supports_tiled_3d`` (gfx942 variant).

    ``supports_tiled_3d`` takes its parameters as keyword-only arguments, so
    ``hkp_pack``'s ``_check_support_predicate`` cannot call it with a spec instance.
    This wrapper reads each of them off the spec and delegates, making the 3D
    unified builder validatable at pack time.
    """
    use_fp8 = spec.kv_storage_dtype is not None
    return supports_tiled_3d(
        head_size=spec.head_size,
        block_size=spec.block_size,
        dtype=spec.dtype,
        num_queries_per_kv=spec.num_queries_per_kv,
        use_alibi=spec.use_alibi,
        use_qq_bias=spec.use_qq_bias,
        use_fp8=use_fp8,
        q_dtype=spec.dtype,
        kv_storage_dtype=spec.kv_storage_dtype,
        arch=arch,
    )
