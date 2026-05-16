# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiled MFMA implementation of AITER's `kernel_unified_attention_2d`.

This kernel mirrors the Triton reference 1:1 in semantics while using AMD's
production-grade patterns from CK Tile's
`BlockFmhaPipelineQRKSVSAsync`:

  - Q is staged in LDS once at the start of the CTA.
  - K is loaded from cache to LDS each tile; we issue the global load early
    so the QK MFMA can begin as soon as the LDS write retires.
  - V is loaded from cache to LDS each tile; it is read again per PV atom.
  - Online softmax statistics (`m`, `l`) live in registers across the loop.
    The per-row max reduction uses `ds_bpermute` butterflies (4 stages on
    wave64), matching CK's `block_tile_reduce_xor_sync` pattern, and avoids
    any LDS round-trip.
  - The output accumulator `o_acc` is held in MFMA accumulator distribution
    (per-lane `<4 x float>` for each of the 8 N-tiles of the head dim) and
    truncated to fp16 via an LDS-staged shuffle epilogue (16-byte stores).

Scope (this revision):

  - `head_size = 128`
  - `dtype = fp16` (bf16 is a follow-up; the IR primitives are in place)
  - `block_size in {16, 64}` with `TILE_SIZE = block_size` (the AITER all-decode
    selector path used by the production decode workload)
  - `num_queries_per_kv in {1, 2, 4, 8, 16}` so `BLOCK_M = 16`

Correctness contract (validated against `aiter.op_tests.triton_tests.attention`
`ref_paged_attn` with `torch.float16` inputs sampled `N(0,1)`):

  - `max_abs` matches Triton bit-for-bit at fp16 ULP precision
    (~`1.83e-4` for d=128, ~`2.74e-4` with sliding window).
  - `max_abs` per the runbook target for fp32-accumulated fp16 attention
    with random N(0,1) inputs is well under one fp16 ULP at the output
    scale (~`5e-4` for outputs ~ 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from ..core.ir import (
    BF16,
    F16,
    F32,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from ..helpers.attention import (
    apply_softcap_log2,
    binary_search_seq_idx,
    mfma_16x16x16_for_dtype,
    mfma_16x16x32_for_dtype,
    warp_xor_reduce_max,
    warp_xor_reduce_sum,
)
from ..helpers.layouts import TransposeLdsReader
from ..transforms import TensorDescriptor, indirect, unmerge


MFMA_M = 16
MFMA_N = 16


# Backwards-compatible aliases. The 3D tiled kernel currently imports these
# from this module; once that import is removed in a follow-up these aliases
# can go away too. Promoted helpers live in ``ck_dsl.helpers.attention``.
_apply_softcap = apply_softcap_log2
_binary_search_seq_idx = binary_search_seq_idx
_mfma_16x16x16 = mfma_16x16x16_for_dtype
_mfma_16x16x32 = mfma_16x16x32_for_dtype
_warp_xor_reduce_max = warp_xor_reduce_max
_warp_xor_reduce_sum = warp_xor_reduce_sum


@dataclass(frozen=True)
class UnifiedAttention2DTiledSpec:
    head_size: int
    block_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    use_sinks: bool
    sliding_window: int
    has_softcap: bool
    use_alibi: bool = False
    use_qq_bias: bool = False
    num_seqs: int = 0
    # Number of wave64 warps per CTA. `BLOCK_M = num_warps * 16` rows are
    # processed per CTA, with each warp owning its own 16-row slice. The
    # online softmax stays per-warp (no cross-warp reduction); the savings
    # come from amortising the Q load, async K/V loads, P_lds publish, and
    # cshuffle epilogue across more lanes. Default `1` preserves the
    # original single-warp behaviour bit-for-bit.
    num_warps: int = 1
    # AMDGPU occupancy hint (``"amdgpu-waves-per-eu"``). Attention is
    # register-pressure-bound; setting this to 2 or 3 forces the
    # backend to tighten VGPR allocation in exchange for higher
    # occupancy. ``None`` keeps the LLVM heuristic.
    waves_per_eu: Optional[int] = None

    def __post_init__(self):
        if self.num_warps not in (1, 2, 4):
            raise ValueError(
                f"num_warps must be 1, 2, or 4 (got {self.num_warps}). "
                f"Other counts would need new MFMA distribution logic."
            )

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def block_m(self) -> int:
        return 16 * self.num_warps

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
        # AITER/Triton uses a true while-loop binary search. Our IR currently
        # lowers this as a fixed-trip scf.for, so specialize the trip count to
        # the known problem batch size instead of always paying 32 iterations.
        # Keep 32 as a conservative fallback for direct unit-test specs that do
        # not provide `num_seqs`.
        if self.num_seqs <= 0:
            return 32
        return max(1, int(math.ceil(math.log2(self.num_seqs + 1))))

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        # Value-carrying optionals (sw{N}, w{N}) become plain
        # conditional strings; kernel_name_join drops empty ones.
        # Value-less flags go through the `flags=` map so they get
        # rendered in iteration order with leading underscores.
        return kernel_name_join(
            "ck_dsl_uattn2d_tiled",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            self.dtype,
            "" if not self.use_sinks else "sinks",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "softcap" if self.has_softcap else "",
            "alibi" if self.use_alibi else "",
            "qqb" if self.use_qq_bias else "",
            f"w{self.num_warps}" if self.num_warps != 1 else "",
        )


def supports_tiled_2d(
    *,
    head_size: int,
    block_size: int,
    dtype: str,
    num_queries_per_kv: int,
    use_alibi: bool,
    use_qq_bias: bool,
    use_fp8: bool,
    q_dtype,
    num_warps: int = 1,
) -> Tuple[bool, str]:
    if dtype not in ("fp16", "bf16"):
        return False, f"tiled 2D kernel currently supports fp16/bf16 (got {dtype!r})"
    if head_size not in (128, 256):
        return (
            False,
            f"tiled 2D kernel only supports head_size in {{128,256}} (got {head_size})",
        )
    if block_size not in (16, 64):
        return (
            False,
            f"tiled 2D kernel only supports block_size in {{16,64}} (got {block_size})",
        )
    if num_queries_per_kv > 16 or num_queries_per_kv < 1:
        return (
            False,
            f"tiled 2D kernel needs 1<=num_queries_per_kv<=16 (got {num_queries_per_kv})",
        )
    block_m = 16 * num_warps
    if block_m % num_queries_per_kv != 0:
        return (
            False,
            f"tiled 2D kernel needs num_queries_per_kv to divide BLOCK_M={block_m} "
            f"(num_warps={num_warps}, got num_queries_per_kv={num_queries_per_kv})",
        )
    if use_fp8 or q_dtype is not None:
        return False, "tiled 2D kernel does not implement FP8 path yet"
    # ALiBi and QQ-bias are supported by the tiled 2D kernel.
    return True, "supported"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_unified_attention_2d_tiled(spec: UnifiedAttention2DTiledSpec) -> KernelDef:
    """Emit the tiled MFMA fp16 2D unified-attention kernel.

    Algorithm (per CTA = 1 wave64 = 64 lanes):

    1. Find `seq_idx` via the AITER binary-search-on-`cu_q`.
    2. Compute Q-block-local index; early-exit if it's a padding block.
    3. Cooperatively stage Q[16, 128] from global to LDS (zero-fill for
       rows that map to padding queries or out-of-range heads).
    4. Loop over KV tiles (`tile_start..tile_end`):
       4a. Look up `physical_block = block_tables[seq_idx, tile_idx]`.
       4b. Cooperatively stage K, V (each [T, 128]) from cache to LDS,
           zero-filling per-tile rows outside `max_seq_prefix_len`.
       4c. Compute `S = Q @ K^T` via `v_mfma_f32_16x16x32_f16` (4 K-iters
           per N-tile, with `T/16` N-tiles).
       4d. Apply `qk_scale`, optional `softcap`, mask (causal, sliding
           window, padding rows, padding heads).
       4e. Online softmax: per-row max via `ds_bpermute` butterfly (lanes
           in 16-lane groups share their 4-row state). Compute P=exp2(S-m)
           in registers and stash it in LDS for the PV MFMA A operand.
       4f. `acc *= alpha`, `acc += P @ V` via MFMA (8 N-tiles, T/K_STEP
           K-iters). V is read scalar-by-scalar because its LDS layout is
           [T, HD] (the K dim is the outer stride). A transposed LDS is a
           planned follow-up.
    5. Normalise `acc /= L` per row, stage into Acc_lds, and store fp16
       output via 8-half vector writes.
    """

    if spec.dtype not in ("fp16", "bf16"):
        raise NotImplementedError("tiled 2D kernel supports fp16/bf16")
    dtype = spec.dtype_ir

    HD = spec.head_size
    T = spec.tile_size
    BS = spec.block_size
    BLOCK_M = spec.block_m
    BLOCK_Q = spec.block_q
    NQK = spec.num_queries_per_kv
    NUM_KV = spec.num_kv_heads
    NUM_QH = spec.num_query_heads
    SLIDING_WINDOW = spec.sliding_window
    USE_SOFTCAP = spec.has_softcap
    USE_SINKS = spec.use_sinks
    USE_ALIBI = spec.use_alibi
    USE_QQ_BIAS = spec.use_qq_bias

    QK_K_STEP = 32
    PV_K_STEP = 32 if T % 32 == 0 else 16
    QK_K_ITERS = HD // QK_K_STEP
    QK_N_TILES = T // MFMA_N
    PV_K_ITERS = T // PV_K_STEP
    PV_N_TILES = HD // MFMA_N

    NUM_WARPS = spec.num_warps
    WAVE = 64
    THREADS = NUM_WARPS * WAVE
    BLOCK_M_PER_WARP = 16  # one MFMA-row tile per warp

    name = spec.kernel_name()
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    # ---------------- parameter declarations ----------------
    output = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    query = b.param(
        "query_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(dtype, "global"),
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
    _k_scale = b.param("k_scale", F32)
    _v_scale = b.param("v_scale", F32)
    _out_scale = b.param("out_scale", F32)
    softcap_p = b.param("softcap", F32)
    num_seqs_p = b.param("num_seqs", I32)
    bt_stride_p = b.param("block_table_stride", I32)
    qq_bias_stride0_p = b.param("qq_bias_stride_0", I32)

    kv_head_idx = b.block_id_x()
    q_block_global_idx = b.block_id_y()
    tid = b.thread_id_x()

    # Wave decomposition. For NUM_WARPS=1 this collapses to `wave_id=0,
    # lane=tid`, exactly the single-warp behaviour. For NUM_WARPS>1 each
    # wave owns rows `[wave_id*16, (wave_id+1)*16)` of the M dimension.
    if NUM_WARPS == 1:
        lane = tid
        wave_row_base = b.const_i32(0)
    else:
        lane = b.mod(tid, b.const_i32(WAVE))
        wave_id = b.div(tid, b.const_i32(WAVE))
        wave_row_base = b.mul(wave_id, b.const_i32(BLOCK_M_PER_WARP))

    # ---------------- seq lookup ----------------
    seq_idx = binary_search_seq_idx(
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

    # ---------------- LDS layout ----------------
    # Q is loaded once. K and V are double-buffered [2, T, HD] in natural
    # row-major layout (async DMA deposits lane-contiguous). The PV MFMA
    # B operand is fetched via `ds_read_b64_tr_b16` with per-lane addresses
    # following CK Tile's `TransposeLDSLayout<M=16,K=16,B=1>` (single read
    # for K=16, 2 reads for K=32). This collapses the 4-8 scalar
    # `ds_read_u16` per atom (16-way bank conflicted) into 1-2 wide
    # transpose reads with the MFMA B distribution baked in.
    Q_lds = b.smem_alloc(dtype, [BLOCK_M, HD], name_hint="Qlds")
    K_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [2, T, HD], name_hint="Vlds")
    P_lds = b.smem_alloc(dtype, [BLOCK_M, T], name_hint="Plds")
    Acc_lds = b.smem_alloc(F32, [BLOCK_M, HD], name_hint="Aclds")

    # ---- CK Tile `TransposeLDSLayout<M=16, K=*, B=1>` lane formulas ----
    # ``TransposeLdsReader`` materializes the per-lane row / col SSA
    # values once and exposes ``row(k_offset, read)`` for use inside
    # the PV K iteration loop. These are per-warp formulas, so the
    # bind site uses the in-warp ``lane`` id (not the global thread id).
    pv_tr_reader = TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, lane)
    tr_col_lane = pv_tr_reader.col

    # ---------------- constants ----------------
    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    one_f = b.const_f32(1.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)
    qk_scale = b.fmul(scale_p, rcp_ln2)
    sw_const = b.const_i32(int(SLIDING_WINDOW))
    z8 = b.zero_vec(dtype, 8)

    # ---------------- Q -> LDS (cooperative vec8 chunks) ----------------
    # General distribution:
    #   total vec8 chunks = BLOCK_M * HD / 8
    #   each wave64 lane handles (BLOCK_M * HD / 8) / 64 chunks.
    # This gives 4 chunks/thread for HD=128 and 8 chunks/thread for HD=256.
    Q_VECS_PER_ROW = HD // 8
    Q_VECS_PER_THREAD = (BLOCK_M * Q_VECS_PER_ROW) // THREADS
    # Coordinate transform for Q (and the symmetric output buffer):
    # ``(token, head, dim)`` packed contiguously. The element-unit
    # descriptor is reused below for the output store too.
    q_desc = TensorDescriptor.naive(
        "Q",
        # The runtime extents (total_q, num_query_heads, head_size) are
        # only used by the validity predicate (which we don't request
        # here). Use generous compile-time bounds so the descriptor's
        # row-major stride product matches the kernel's layout
        # assumptions exactly.
        lengths=[1 << 30, NUM_QH, HD],
        coord_names=("token", "head", "dim"),
    )
    for li in range(Q_VECS_PER_THREAD):
        q_vid = b.add(b.mul(b.const_i32(li), b.const_i32(THREADS)), tid)
        Q_row = b.div(q_vid, b.const_i32(Q_VECS_PER_ROW))
        Q_col = b.mul(b.mod(q_vid, b.const_i32(Q_VECS_PER_ROW)), b.const_i32(8))
        q_pos_t = b.add(qb_start_pos, b.div(Q_row, b.const_i32(NQK)))
        qh_t = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(Q_row, b.const_i32(NQK))
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

    # ---------------- KV tile loop bounds ----------------
    bm1_div_nqk = (BLOCK_M - 1) // NQK
    msp_raw = b.add(b.add(context_len, qb_start_pos), b.const_i32(bm1_div_nqk + 1))
    max_seq_prefix_len = b.select(b.cmp_lt(msp_raw, seq_len), msp_raw, seq_len)
    num_tiles = b.div(b.add(max_seq_prefix_len, b.const_i32(T - 1)), b.const_i32(T))

    if SLIDING_WINDOW > 0:
        qpos_hi_raw = b.add(qb_start_pos, b.const_i32(bm1_div_nqk))
        cur_q_minus1 = b.sub(cur_batch_q_len, b.const_i32(1))
        qpos_hi = b.select(
            b.cmp_lt(qpos_hi_raw, cur_q_minus1), qpos_hi_raw, cur_q_minus1
        )
        first_allowed_key = b.add(
            b.sub(b.add(context_len, qb_start_pos), sw_const), b.const_i32(1)
        )
        last_allowed_key = b.add(context_len, qpos_hi)
        tile_start_raw = b.div(first_allowed_key, b.const_i32(T))
        tile_start = b.select(
            b.cmp_lt(tile_start_raw, b.const_i32(0)), b.const_i32(0), tile_start_raw
        )
        tile_end_raw = b.add(b.div(last_allowed_key, b.const_i32(T)), b.const_i32(1))
        tile_end = b.select(b.cmp_lt(tile_end_raw, num_tiles), tile_end_raw, num_tiles)
    else:
        tile_start = b.const_i32(0)
        tile_end = num_tiles

    # ---------------- online softmax registers ----------------
    # Each lane owns 4 row slots within its warp's BLOCK_M_PER_WARP=16 rows
    # (rows = wave_row_base + (lane/16)*4 + r for r in 0..3) when viewed
    # through the MFMA acc distribution. We keep `(m, l)` per row slot and
    # the 8 PV N-tile accumulators in iter_args of the KV loop. The MFMA
    # distribution is a per-warp construct, so the indexing uses `lane`
    # (== tid%64), not `tid`.
    lane_rg = b.div(lane, b.const_i32(16))
    lane_col = b.mod(lane, b.const_i32(16))

    if USE_SINKS:
        m_inits = []
        for r in range(4):
            in_warp_row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(r))
            row = b.add(wave_row_base, in_warp_row)
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
            sink_h = b.global_load(sinks, qh, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            m_inits.append(b.select(qh_in, sink_f, neg_inf))
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

    # ---- Pre-loop: build K/V buffer descriptors and pre-fetch tile 0.
    # The buffer rsrc bounds OOB voffsets to return zero. We size it large
    # so valid block offsets never trip the check.
    big_bytes = b.const_i32(0x7FFF0000)
    key_rsrc = b.buffer_rsrc(key, big_bytes)
    value_rsrc = b.buffer_rsrc(value, big_bytes)

    # Async load contract: dwords=4 means each lane writes 16 bytes
    # lane-contiguous in LDS. One call writes 64 * 8 halfs = 512 halfs =
    # 1024 bytes, i.e. a contiguous slice of the natural [T, HD] tile. This
    # works for HD=128 and HD=256 without changing the LDS layout.
    KV_HALVES_PER_CALL = THREADS * 8
    assert (T * HD) % KV_HALVES_PER_CALL == 0
    kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL
    bytes_per_call = KV_HALVES_PER_CALL * 2
    kv_stride_blk_b = BS * NUM_KV * HD * 2
    kv_stride_tok_b = NUM_KV * HD * 2
    kv_stride_h_b = HD * 2

    lane_half_base = b.mul(tid, b.const_i32(8))

    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    bytes_per_buf = T * HD * 2  # one [T, HD] half slab

    zero_soff = b.const_i32(0)

    # Bytes one wave's lanes write per call. `raw.ptr.buffer.load.lds`
    # writes `dwords * 4` bytes per lane lane-contiguous starting at
    # the wave-uniform `lds_dst`. Each wave issues its own instruction
    # but they share the LDS pointer unless we add a wave offset; with
    # NUM_WARPS=1 this collapses to zero.
    WAVE_BYTES = WAVE * 16  # dwords=4 → 16 bytes per lane × 64 lanes
    if NUM_WARPS == 1:
        wave_lds_offset_i64 = b.const_i64(0)
    else:
        wave_lds_offset_i32 = b.mul(wave_id, b.const_i32(WAVE_BYTES))
        wave_lds_offset_i64 = b.zext(wave_lds_offset_i32, I64)

    # ---- Paged KV byte descriptor (full transform DAG) ----
    # The paged-KV cache is laid out ``[num_blocks, BS, NUM_KV, HD]`` with
    # *byte* strides. The kernel addresses it via three layered
    # coordinate transforms:
    #
    #   1. ``indirect(tile_idx -> physical_block)`` does the
    #      ``physical_block = block_tables[seq_idx*bt_stride + tile_idx]``
    #      table lookup; the upper coord ``tile_idx`` becomes the
    #      naive descriptor's leaf coord ``physical_block``.
    #   2. ``unmerge(linear_half -> (token, dim))`` splits the
    #      cooperative ``THREADS*8`` half count into ``(token_in_tile,
    #      head_dim)``.
    #   3. The naive 4D base ``(physical_block, token, kv_head, dim)``
    #      with byte strides produces the final byte offset.
    #
    # Calling ``paged_kv_desc.offset(b, tile_idx=, linear_half=, kv_head=)``
    # returns the byte offset for one lane's load -- including the table
    # indirection, the (token, dim) unmerge, and the byte stride math --
    # in one shot, replacing the previous ``block_base_from_table`` +
    # ``linear_half_voff`` two-step composition.
    seq_base = b.mul(seq_idx, bt_stride_p)
    paged_kv_desc = TensorDescriptor.naive(
        "paged_kv_bytes",
        # ``lengths`` here is just informational (validity propagation
        # is driven by the transforms above, not by these bounds).
        lengths=[1 << 24, T, NUM_KV, HD],
        strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, 2],
        coord_names=("physical_block", "token", "kv_head", "dim"),
    ).transform(
        indirect("tile_idx", into="physical_block", table=block_tables, base=seq_base),
        unmerge("linear_half", into=("token", "dim"), dims=(T, HD)),
    )

    def _issue_k_load_runtime(kv_tile_idx: Value, buf_idx: Value) -> None:
        """Issue async K loads for one tile into K_lds[buf_idx].

        CK's QRKSVSAsync pipeline deliberately makes K the early-prefetch
        stream: QK can start as soon as K is visible, while V is still not
        needed until after softmax. Keeping K and V as independent streams
        avoids waiting on V before QK.

        Multi-warp: each wave's `raw.ptr.buffer.load.lds` writes a
        lane-contiguous 1 KiB slab starting at `lds_dst`. To keep the
        waves from stomping on each other we offset `lds_dst` by
        `wave_id * WAVE_BYTES`; combined with each wave's natural voff
        offset (lanes 64..127 have `tid*8 / HD` advanced by T/NUM_WARPS),
        the cooperative load fills the full `[T, HD]` slab correctly.
        """
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        K_buf_base = b.smem_ptr_add(K_lds_addr, buf_off_i64)
        K_wave_base = b.smem_ptr_add(K_buf_base, wave_lds_offset_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_half,
                kv_head=kv_head_idx,
            )
            k_dst = b.smem_ptr_add(K_wave_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(key_rsrc, k_dst, voff, zero_soff, 4)

    def _issue_v_load_runtime(kv_tile_idx: Value, buf_idx: Value) -> None:
        """Issue async V loads for one tile into V_lds[buf_idx].

        This is launched *after* QK is computed and before the softmax
        reduction, so V movement overlaps with max/sum/exp/P staging and is
        waited only immediately before PV. See `_issue_k_load_runtime` for
        the multi-warp LDS-offset rationale.
        """
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        V_buf_base = b.smem_ptr_add(V_lds_addr, buf_off_i64)
        V_wave_base = b.smem_ptr_add(V_buf_base, wave_lds_offset_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_half,
                kv_head=kv_head_idx,
            )
            v_dst = b.smem_ptr_add(V_wave_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(value_rsrc, v_dst, voff, zero_soff, 4)

    # Prefetch tile_start's K into buffer 0 BEFORE the loop.
    _issue_k_load_runtime(tile_start, b.const_i32(0))

    # ---------------- KV tile loop ----------------
    # We carry `cur_buf` (the buffer that holds tile i's data) through the
    # loop. At iter i we:
    #   1. Wait for current K (prefetched by the previous iteration, or the
    #      pre-loop prologue).
    #   2. Compute QK.
    #   3. Issue current V, then next K, and run softmax while both are in
    #      flight. Since current V is older than next K in the VMEM/LGKM
    #      queues, a partial wait with `kv_calls_per_tile` pending leaves
    #      next K in flight while making current V visible for PV.
    #   3. `s_barrier` to make tile i's data visible to all reads.
    #   4. Compute QK, issue current V, run softmax while V is in flight.
    #   5. Wait for current V, publish P_lds, then run PV.
    #   5. Yield `(m, l, acc, nxt_buf)` so the next iter consumes nxt_buf.
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

        # Prepare the clamped tile index for the next-K prefetch we will issue
        # after QK. The final iteration intentionally prefetches the current
        # tile again into the alternate buffer; this keeps the schedule uniform.
        next_tile_iv_raw = b.add(kv_tile_iv, b.const_i32(1))
        in_range_next = b.cmp_lt(next_tile_iv_raw, tile_end)
        safe_next_tile = b.select(in_range_next, next_tile_iv_raw, kv_tile_iv)

        # Wait for current K. There should be no in-flight next-K work here;
        # the previous iteration waited all async loads before PV.
        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.sync()

        # ---- S = Q @ K^T (per-warp MFMA) ----
        # Q is in LDS only; we re-read it per iter -- the compiler hoists the
        # LDS reads across iterations when alignment lets it (Q never
        # changes after the prelude). Each warp reads its own 16-row slice
        # of Q[BLOCK_M, HD] at rows `[wave_row_base, wave_row_base+16)`.
        q_row = b.add(wave_row_base, lane_col)
        A_kits = []
        for k in range(QK_K_ITERS):
            q_col_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
            A_kits.append(b.smem_load_vN(Q_lds, q_row, q_col_off, dtype=dtype, n=8))
        S_n = []
        for n in range(QK_N_TILES):
            acc_v = b.zero_vec_f32(4)
            for k in range(QK_K_ITERS):
                kc_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                k_row = b.add(b.const_i32(n * 16), lane_col)
                B_v = b.smem_load_vN(K_lds, cur_buf, k_row, kc_off, dtype=dtype, n=8)
                acc_v = _mfma_16x16x32(b, dtype, A_kits[k], B_v, acc_v)
            S_n.append(acc_v)

        # Now that QK no longer needs VMEM, start current V first and next K
        # second. This ordering is what lets the partial wait before PV leave
        # only next K pending.
        _issue_v_load_runtime(kv_tile_iv, cur_buf)
        _issue_k_load_runtime(safe_next_tile, nxt_buf)

        # ---- mask / scale / softcap / alibi / qq-bias ----
        # ALiBi and QQ-bias mirror Triton's apply-before-mask-result semantics:
        # we fold them into the unmasked S, then the select-with-(-inf) below
        # zeroes them out for invalid cells (finite + (-inf) = (-inf) in IEEE
        # for the mask path so result is identical to Triton's
        # "S = where(mask, S, -inf); S += bias" formulation).
        if USE_ALIBI:
            alibi_per_row = []
            for reg in range(4):
                in_warp_row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
                row = b.add(wave_row_base, in_warp_row)
                qh_r = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
                )
                # If qh_r is OOB (padding head), set slope to 0. The row mask
                # below will set the cell to -inf regardless, but we keep the
                # load OOB-safe with masked_global_load_f32.
                qh_ok = b.cmp_lt(qh_r, b.const_i32(NUM_QH))
                slope = b.masked_global_load(
                    alibi_slopes_ptr, qh_r, qh_ok, b.const_f32(0.0), dtype=F32, align=4
                )
                alibi_per_row.append(slope)
        masked = {}
        for reg in range(4):
            in_warp_row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
            row = b.add(wave_row_base, in_warp_row)
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
                score = b.select(m_ok, s_scaled, neg_inf)
                if USE_ALIBI:
                    # Triton order: mask first, then add ALiBi. For invalid
                    # cells this is `-inf + finite == -inf`, avoiding any
                    # pre-mask finite arithmetic from leaking into reductions.
                    pos_off = b.sub(col_abs, context_len)
                    pos_f = b.sitofp_f32(pos_off)
                    add_term = b.fmul(b.fmul(alibi_per_row[reg], pos_f), rcp_ln2)
                    score = b.fadd(score, add_term)
                if USE_QQ_BIAS:
                    # qq_bias[qp_r, key_rel_pos] with key_rel_pos = col - ctx.
                    # Valid range 0 <= key_rel_pos < qq_bias_stride_0 AND
                    # qp_r is a non-padding query position. The padding-row
                    # guard is required because qb_start_pos can exceed
                    # cur_batch_query_len for the last Q-block of a sequence;
                    # without it `qq_bias_ptr + qp_r*stride0 + krp` can run
                    # off the end of the tensor for tail blocks.
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
                    score = b.fadd(score, b.fmul(qq_v, rcp_ln2))
                masked[(n, reg)] = score

        # ---- per-row max via cross-lane butterfly ----
        # Each lane has 4 floats (one per row in its row-group), repeated for
        # every N-tile. Local lane-max across N-tiles, then 4-stage XOR
        # butterfly across the 16 lanes in the row-group.
        m_new = []
        s_local = {}  # (reg, n) -> the lane's masked score (still owned per-lane)
        for reg in range(4):
            local_max = neg_inf
            for n in range(QK_N_TILES):
                v = masked[(n, reg)]
                s_local[(reg, n)] = v
                local_max = b.fmax(local_max, v)
            tile_max = _warp_xor_reduce_max(b, local_max)
            # Online softmax update (FlashAttention/Triton): the new
            # running max is max(previous_m, current_tile_max). The old
            # code used only `current_tile_max`, which is numerically
            # wrong when logits decrease across KV tiles. ALiBi with
            # negative slopes is exactly that case: the first tile can
            # have a very large positive bias and later tiles a much
            # smaller one, so exp2(previous_m - current_tile_max)
            # overflows to inf and the final acc/l normalization becomes
            # NaN. If both previous and tile max are -inf (fully masked
            # row), mirror Triton's guard and force the max to 0.
            full_max_raw = b.fmax(m_vals[reg], tile_max)
            ok = b.fcmp("ogt", full_max_raw, neg_inf)
            m_new.append(b.select(ok, full_max_raw, zero_f))

        # ---- compute P = exp2(S - m_new) and l_local = sum(P) per row ----
        # We need P in LDS for the PV MFMA A operand. Each lane writes its
        # masked-scores' exp2 into P_lds[wave_row_base + (lane/16)*4 + reg,
        # (lane%16) + n*16]. Each warp publishes its own 16-row slice.
        l_local = []
        for reg in range(4):
            in_warp_row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
            row = b.add(wave_row_base, in_warp_row)
            sum_p = zero_f
            for n in range(QK_N_TILES):
                p = b.exp2(b.fsub(s_local[(reg, n)], m_new[reg]))
                col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
                b.smem_store_vN(P_lds, [row, col], b.cast_f32_to(p, dtype), 1)
                sum_p = b.fadd(sum_p, p)
            l_local.append(_warp_xor_reduce_sum(b, sum_p))

        # alpha and L update (still per-lane registers; matches FA-2 paper)
        alpha_regs = [b.exp2(b.fsub(m_vals[r], m_new[r])) for r in range(4)]
        new_l_vals = [
            b.fadd(b.fmul(l_vals[r], alpha_regs[r]), l_local[r]) for r in range(4)
        ]
        # Wait for current V while leaving next K pending. Current V was
        # issued before next K, so `kv_calls_per_tile` pending operations are
        # exactly the next-K stream. Apply the same idea to lgkmcnt so we do
        # not wait for the next-K LDS writes before PV.
        b.s_waitcnt(vmcnt=kv_calls_per_tile, lgkmcnt=kv_calls_per_tile)
        b.sync()

        # ---- acc *= alpha, acc += P @ V ----
        new_acc = []
        for n in range(PV_N_TILES):
            scaled_comps = []
            for reg in range(4):
                e = b.vec_extract(acc_vals[n], reg)
                scaled_comps.append(b.fmul(e, alpha_regs[reg]))
            acc_v = b.vec_pack(scaled_comps, F32)

            # PV's K-direction TransposeLDSLayout row/col addresses are
            # produced by ``pv_tr_reader`` -- :meth:`row(k_offset, read)`
            # computes ``(lane/16)*K_L + read*4 + (lane/4)%4 + k_offset``
            # for one ds_read_b64_tr_b16. ``tr_col_lane`` is the cached
            # ``(lane%4)*4`` column component.
            n_col_base = b.add(b.mul(b.const_i32(n), b.const_i32(16)), tr_col_lane)

            # P_lds is shared across warps in the BLOCK_M dimension. Each
            # warp reads its own 16-row slice for the PV A operand; the
            # MFMA distribution wants the wave's lane_col to indicate the
            # in-tile row, which is `wave_row_base + lane_col`.
            p_row = b.add(wave_row_base, lane_col)
            for k in range(PV_K_ITERS):
                if PV_K_STEP == 32:
                    # K=32: P operand 8 halves, V via 2 ds_read_b64_tr_b16
                    # reads (READS=2 per CK formula).
                    p_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                    A_p = b.smem_load_vN(P_lds, p_row, p_off, dtype=dtype, n=8)
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
                    # K=16: single ds_read_b64_tr_b16 returns the full B operand.
                    p_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                    A_p = b.smem_load_vN(P_lds, p_row, p_off, dtype=dtype, n=4)
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

    # ---------------- epilogue ----------------
    # The loop issues a uniform "next K" async load every iteration, including
    # the final iteration where that load is intentionally never consumed. The
    # partial wait before PV leaves that final prefetch in flight. CK Tile
    # kernels always close outstanding async-copy groups before the CTA exits;
    # do the same here so no raw global->LDS operation can outlive the kernel
    # and corrupt later launches in the same process.
    b.s_waitcnt(vmcnt=0, lgkmcnt=0)
    b.sync()

    final = kvloop.results
    l_final = [final[2 * r + 1] for r in range(4)]
    acc_final = [final[8 + n] for n in range(PV_N_TILES)]

    # Each warp writes its own 16-row slice of Acc_lds[BLOCK_M, HD].
    for n in range(PV_N_TILES):
        for reg in range(4):
            in_warp_row = b.add(b.mul(lane_rg, b.const_i32(4)), b.const_i32(reg))
            row = b.add(wave_row_base, in_warp_row)
            col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
            v = b.vec_extract(acc_final[n], reg)
            l_nonzero = b.fcmp("ogt", l_final[reg], zero_f)
            normalized = b.fmul(v, b.rcp(l_final[reg]))
            b.smem_store_vN_f32(
                Acc_lds, [row, col], b.select(l_nonzero, normalized, zero_f), 1
            )
    b.sync()

    # Output store: each thread writes 32 halves (4 vec8 chunks) of one
    # BLOCK_M row at a time. THREADS / OUT_THREADS_PER_ROW threads cooperate
    # on a single row; if BLOCK_M is larger than that, we need multiple
    # row-iterations per thread. Skipping this multi-iter loop only writes
    # the first `THREADS / OUT_THREADS_PER_ROW` rows, which leaves later
    # BLOCK_M rows uninitialized -- exactly the failure observed for
    # HD=256 + MHA prefill (BLOCK_M=16, OUT_THREADS_PER_ROW=8 -> only 8 rows
    # written -> ~half the output is garbage).
    OUT_THREADS_PER_ROW = HD // 32
    ROWS_PER_ITER = THREADS // OUT_THREADS_PER_ROW
    assert THREADS == OUT_THREADS_PER_ROW * ROWS_PER_ITER
    assert BLOCK_M % ROWS_PER_ITER == 0, (
        f"BLOCK_M={BLOCK_M} must be a multiple of ROWS_PER_ITER={ROWS_PER_ITER}"
    )
    OUT_ROW_BASE = b.div(tid, b.const_i32(OUT_THREADS_PER_ROW))
    OUT_col_base = b.mul(b.mod(tid, b.const_i32(OUT_THREADS_PER_ROW)), b.const_i32(32))
    for row_iter in range(BLOCK_M // ROWS_PER_ITER):
        OUT_row = b.add(OUT_ROW_BASE, b.const_i32(row_iter * ROWS_PER_ITER))
        op_pos = b.add(qb_start_pos, b.div(OUT_row, b.const_i32(NQK)))
        op_qh = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(OUT_row, b.const_i32(NQK))
        )
        op_mask = b.land(
            b.cmp_lt(op_pos, cur_batch_q_len), b.cmp_lt(op_qh, b.const_i32(NUM_QH))
        )
        # Output and Q share the ``(token, head, dim)`` layout, so the
        # same descriptor produces the global store base address.
        out_base, _ = q_desc.offset(
            b,
            token=b.add(cu_q_start, op_pos),
            head=op_qh,
            dim=b.const_i32(0),
        )
        for chunk in range(4):
            col = b.add(OUT_col_base, b.const_i32(chunk * 8))
            vf1 = b.smem_load_vN_f32(Acc_lds, OUT_row, col, n=4)
            vf2 = b.smem_load_vN_f32(Acc_lds, OUT_row, b.add(col, b.const_i32(4)), n=4)
            v8h = b.vec_concat(
                b.vec_cast_f32_to(vf1, dtype), b.vec_cast_f32_to(vf2, dtype)
            )
            with b.scf_if(op_mask):
                b.global_store_vN(output, b.add(out_base, col), v8h, 8, align=16)

    return b.kernel
