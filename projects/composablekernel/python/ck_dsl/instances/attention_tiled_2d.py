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
    CACHE_STREAM,
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
from ..helpers.attention import (
    apply_softcap_log2,
    binary_search_seq_idx,
    mfma_16x16x16_for_dtype,
    mfma_16x16x32_for_dtype,
    warp_xor_reduce_max,
    warp_xor_reduce_sum,
)
from ..helpers.layouts import TransposeLdsReader
from ..transforms import TensorDescriptor, embed, indirect, unmerge


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
    # FP8 K/V cache. When ``"fp8e4m3"``, the kernel takes K/V cache
    # pointers as ``ptr<fp8e4m3, global>`` (1 byte/element), uses a
    # sync per-thread load that emits ``cvt_fp8_to_f32 -> fmul k_scale
    # -> cast_f32_to_bf16`` and writes the working dtype (bf16) into
    # LDS. The async DMA path is disabled on this lane because
    # ``raw_ptr_buffer_load_lds`` cannot intercept the scale step. Q is
    # still passed in the working dtype (``self.dtype``), and the rest
    # of the kernel (MFMA, softmax, epilogue) is unchanged.
    kv_storage_dtype: Optional[str] = None
    # ``T`` (per-CTA-iter KV-tile size in tokens). When ``None``, the
    # kernel uses ``T = block_size`` (one paged-KV cache block per
    # iter, matching the AITER decode path). Setting ``tile_size > block_size``
    # makes each iter walk multiple consecutive ``block_table`` entries
    # and amortizes the outer loop overhead — this is what unlocks
    # Triton-class prefill throughput (Triton 2D uses ``TILE_SIZE=64``
    # with ``BLOCK_M=128``). ``tile_size`` must be a positive multiple
    # of ``block_size`` (so the descriptor's multi-block decomposition
    # is well-defined) and ``T * head_size >= num_warps * 64 * 8``
    # (the async-DMA call carries one wave's lane-contiguous payload).
    tile_size: Optional[int] = None
    # Per-warp M-dimension tile size. Default is one ``MFMA_M`` atom
    # (16 rows) per warp. Setting this to 32 stacks two ``MFMA_M=16``
    # atoms per warp so each warp's QK / PV phase processes twice the
    # rows -- matching Triton's prefill config (``BLOCK_M=128`` with
    # ``num_warps=4`` ⇒ each warp owns 32 rows). The accumulator,
    # ``m`` / ``l`` running stats, and mask/softmax loops then iterate
    # over ``REGS_PER_LANE = block_m_per_warp / 4`` register slots
    # per lane instead of 4. The LDS budget grows with ``BLOCK_M``
    # (``Q_lds``, ``P_lds``, ``Acc_lds``), so ``block_m_per_warp=32``
    # crosses MI355X's 3 → 2 WGs/CU threshold for the prefill workload.
    # Only ``{16, 32}`` are supported; 32 requires ``num_warps``
    # in ``{1, 2, 4}`` so total threads stay within the 1024 CTA cap.
    #
    # **Measured (MI355X, bf16, HD=64, BS=32, T=64)**: ``block_m_per_warp=32``
    # with ``num_warps=4`` (BLOCK_M=128) was 1.6-2.0× SLOWER than the
    # default on every prefill shape we tested, because the doubled
    # ``Q_lds`` + ``P_lds`` + ``Acc_lds`` push the kernel from 3 → 2
    # WGs/CU. The per-CTA throughput gain from bigger BLOCK_M is
    # cancelled by the occupancy loss. The knob is kept exposed for
    # future workloads (e.g. HD=128 or shapes with different LDS
    # budgets) where the trade-off might flip. See
    # ``/workspace/probe_blockm32_perf.py`` for the sweep.
    block_m_per_warp: int = 16

    def __post_init__(self):
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(
                f"num_warps must be 1, 2, 4, or 8 (got {self.num_warps}). "
                f"Other counts would need new MFMA distribution logic. "
                f"num_warps=8 (BLOCK_M=128, THREADS=512) matches the BLOCK_M "
                f"the production Triton 2D kernel uses for high-q prefill "
                f"shapes; both are well within MI355X's 1024-thread CTA cap."
            )
        if self.block_m_per_warp not in (16, 32):
            raise ValueError(
                f"block_m_per_warp must be 16 or 32 (got {self.block_m_per_warp})."
            )
        if self.block_m_per_warp == 32 and self.num_warps not in (1, 2, 4):
            raise ValueError(
                f"block_m_per_warp=32 requires num_warps in {{1,2,4}} "
                f"(got {self.num_warps}); the 8-warp variant would exceed "
                f"the 1024-thread CTA cap with 32 rows per warp."
            )
        if self.kv_storage_dtype is not None and self.kv_storage_dtype != "fp8e4m3":
            raise ValueError(
                f"kv_storage_dtype must be None or 'fp8e4m3' (got {self.kv_storage_dtype!r})"
            )
        if self.tile_size is not None:
            if self.tile_size <= 0 or self.tile_size % self.block_size != 0:
                raise ValueError(
                    f"tile_size must be a positive multiple of block_size "
                    f"(got tile_size={self.tile_size}, block_size={self.block_size})"
                )

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def block_m(self) -> int:
        return self.block_m_per_warp * self.num_warps

    @property
    def regs_per_lane(self) -> int:
        """Number of accumulator register slots per lane per N-tile.

        For ``MFMA_M=16``, the 16x16 MFMA distribution gives each lane
        4 row slots (rows ``lane_rg*4..lane_rg*4+3`` within a 16-row
        atom). With ``block_m_per_warp=32``, we stack two MFMA atoms
        per warp ⇒ 8 row slots per lane.
        """
        return self.block_m_per_warp // 4  # 4 for M=16, 8 for M=32

    @property
    def block_q(self) -> int:
        return self.block_m // self.num_queries_per_kv

    @property
    def tile_size_eff(self) -> int:
        """Effective per-iter KV-tile size in tokens."""
        return self.tile_size if self.tile_size is not None else self.block_size

    @property
    def n_blocks_per_tile(self) -> int:
        """How many paged-KV cache blocks one kernel iter consumes."""
        return self.tile_size_eff // self.block_size

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
            f"t{self.tile_size_eff}" if self.n_blocks_per_tile != 1 else "",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            self.dtype,
            f"kv{self.kv_storage_dtype}" if self.kv_storage_dtype else "",
            "" if not self.use_sinks else "sinks",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "softcap" if self.has_softcap else "",
            "alibi" if self.use_alibi else "",
            "qqb" if self.use_qq_bias else "",
            f"w{self.num_warps}" if self.num_warps != 1 else "",
            f"mw{self.block_m_per_warp}" if self.block_m_per_warp != 16 else "",
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
    kv_storage_dtype: Optional[str] = None,
    tile_size: Optional[int] = None,
) -> Tuple[bool, str]:
    if dtype not in ("fp16", "bf16"):
        return False, f"tiled 2D kernel currently supports fp16/bf16 (got {dtype!r})"
    if head_size not in (64, 128, 256):
        return (
            False,
            f"tiled 2D kernel only supports head_size in {{64,128,256}} (got {head_size})",
        )
    if head_size % 32 != 0:
        return (
            False,
            f"tiled 2D kernel requires head_size divisible by 32 (got {head_size})",
        )
    if block_size not in (16, 32, 64):
        return (
            False,
            f"tiled 2D kernel only supports block_size in {{16,32,64}} (got {block_size})",
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
    # FP8 K/V cache is supported via ``kv_storage_dtype="fp8e4m3"`` plus
    # ``use_fp8=True`` (the latter is what the upstream selector flips on
    # for the FP8 path).
    if kv_storage_dtype is not None and kv_storage_dtype != "fp8e4m3":
        return (
            False,
            f"tiled 2D kernel: unsupported kv_storage_dtype {kv_storage_dtype!r}",
        )
    if use_fp8 and kv_storage_dtype is None:
        return (
            False,
            "tiled 2D kernel: use_fp8=True requires kv_storage_dtype='fp8e4m3'",
        )
    if q_dtype is not None and q_dtype not in ("fp16", "bf16"):
        return False, f"tiled 2D kernel: unsupported q_dtype {q_dtype!r}"
    if tile_size is not None:
        if tile_size <= 0 or tile_size % block_size != 0:
            return (
                False,
                f"tiled 2D kernel: tile_size={tile_size} must be a positive "
                f"multiple of block_size={block_size}",
            )
        # The async DMA call carries THREADS*8 lane-contiguous halves; the
        # per-tile KV slab must hold at least that much, otherwise the wave
        # under-fills the LDS slab and corrupts the partial buffer.
        threads = num_warps * 64
        if tile_size * head_size < threads * 8:
            return (
                False,
                f"tiled 2D kernel: tile_size*head_size={tile_size * head_size} too "
                f"small for num_warps={num_warps} (need >= {threads * 8})",
            )
        # Per-wave window must fit within one block. Each wave (64 lanes)
        # owns ``WAVE * 8 // head_size`` consecutive tokens within a call.
        # If a wave straddles two blocks, the per-lane block_table lookup
        # diverges within the wave -- the multi-block descriptor's
        # ``global_load_i32`` becomes lane-divergent (per-lane VMEM) and
        # the async DMA's lane-contiguous LDS layout no longer matches
        # the physical block, so the wave under-fills the slab. Per-WAVE
        # uniformity (waves land in different blocks but each wave is
        # entirely in one block) is allowed; this is the
        # ``num_warps=8, HD=64, BS=32, T=64`` Triton-class config.
        per_wave_tokens = (64 * 8) // head_size
        if per_wave_tokens > block_size:
            return (
                False,
                f"tiled 2D kernel: per-wave tokens {per_wave_tokens} exceeds "
                f"block_size={block_size}; would need lane-divergent block lookup",
            )
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
    T = spec.tile_size_eff
    BS = spec.block_size
    N_BLOCKS_PER_TILE = spec.n_blocks_per_tile
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
    # FP8 K/V cache: when set, K/V cache pointers are ``ptr<fp8e4m3, global>``
    # (one byte per element), the async DMA path is disabled, and a sync
    # per-thread load + ``cvt_fp8_to_f32 * k_scale -> cast<bf16>`` chain
    # populates the same LDS slabs as the bf16 path. ``KV_BYTES`` flips
    # the byte-stride math for the paged-KV descriptor.
    KV_FP8 = spec.kv_storage_dtype == "fp8e4m3"
    KV_BYTES = 1 if KV_FP8 else 2
    kv_io_dtype = FP8E4M3 if KV_FP8 else dtype

    QK_K_STEP = 32
    PV_K_STEP = 32 if T % 32 == 0 else 16
    QK_K_ITERS = HD // QK_K_STEP
    QK_N_TILES = T // MFMA_N
    PV_K_ITERS = T // PV_K_STEP
    PV_N_TILES = HD // MFMA_N

    NUM_WARPS = spec.num_warps
    WAVE = 64
    THREADS = NUM_WARPS * WAVE
    BLOCK_M_PER_WARP = spec.block_m_per_warp
    # Number of stacked MFMA-M=16 atoms per warp's M dimension. For
    # ``block_m_per_warp=16`` this is 1 (the original kernel); for
    # ``block_m_per_warp=32`` it's 2, so each warp does two stacked
    # ``mfma_f32_16x16x*`` atoms per QK / PV step.
    M_ATOMS_PER_WARP = BLOCK_M_PER_WARP // MFMA_M
    # Per-lane accumulator register slot count per N-tile. The 16x16
    # MFMA distribution gives each lane 4 row slots within one 16-row
    # atom (rows ``lane_rg*4..lane_rg*4+3``); stacking ``M_ATOMS_PER_WARP``
    # atoms gives ``4 * M_ATOMS_PER_WARP`` slots per lane per N-tile.
    REGS_PER_LANE = spec.regs_per_lane  # 4 for M=16, 8 for M=32

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
    # Epilogue staging buffer: ``Acc_lds`` re-uses LDS across multiple
    # ``OUT_STRIPE_COLS``-wide stripes (one stripe = ``OUT_STRIPE_COLS /
    # MFMA_N`` consecutive PV N-tiles). Two regimes:
    #
    # 1. ``HD <= 64``: use 32-col stripes. The big LDS savings (e.g. NW=4
    #    T=64 HD=64 drops 16 KiB → 2 KiB) crosses MI355X's 2 → 3 WGs/CU
    #    threshold and is worth a couple of extra sync barriers per CTA.
    #
    # 2. ``HD >= 128``: use full-HD stripes. The old F32 Acc_lds was
    #    ``BLOCK_M * HD * 4`` bytes; the new dtype-only Acc_lds is
    #    ``BLOCK_M * HD * 2`` (still a 2× LDS reduction, but inside the
    #    same WG/CU class). Splitting into 32-col stripes here would add
    #    3-7 extra sync barriers per CTA without an occupancy gain --
    #    that's a measurable regression for decode workloads (the
    #    HD=128 + BLOCK_M=16 NW=1 path takes only ~6 µs of MFMA + small
    #    amounts of LDS work, so a few hundred extra cycles of barrier
    #    is visible).
    if HD <= 64:
        OUT_STRIPE_COLS = 32
    else:
        OUT_STRIPE_COLS = HD
    OUT_STRIPES = HD // OUT_STRIPE_COLS
    assert HD % OUT_STRIPE_COLS == 0, (
        f"HD={HD} must split evenly into {OUT_STRIPE_COLS}-col stripes"
    )
    # ---- LDS pad swizzle (Q_lds, P_lds) ----
    # Q_lds and P_lds are read with a pattern where 16 lanes (one MFMA
    # 16x16 row-group) hit DIFFERENT rows but the SAME column. With a
    # natural [BLOCK_M, HD] row-major layout and row stride = HD halves
    # = HD*2 bytes, all 16 lanes land on the SAME 32-byte LDS bank cycle
    # → 16-way bank conflict.
    #
    # Padding each row by 8 halves (16 bytes) breaks the bank alignment:
    # row r now occupies bytes ``r * (HD*2 + 16)`` so the bank index
    # ``(r * (HD/2 + 4)) % 32`` cycles every ~8 rows instead of every 1.
    # That converts the worst case from 16-way to 2-way bank conflict.
    # The conv kernel optimization study at
    # ``/workspace/mlse-tools-internal/performance/kernel_optimization/
    # analysis/00_CONSOLIDATED_FINDINGS.md`` measured +43% throughput on
    # MI355X gfx950 from this same trick.
    #
    # We pad by exactly 16 bytes (8 halves) -- not 4 -- to preserve
    # 16-byte alignment for ``ds_write_b128``/``ds_read_b128`` (vec8).
    # A 4-byte pad makes row 1's start at byte ``HD*2 + 4`` which is
    # only 4-byte aligned for HD=64, downgrading the vec8 store to
    # scalar and erasing the LDS savings.
    #
    # K_lds and V_lds cannot be padded the same way because the async
    # DMA (``raw_ptr_buffer_load_lds``) writes lane-contiguous bytes; a
    # padded row stride would corrupt the layout. V_lds reads use
    # ``ds_read_tr16_b64`` which is bank-conflict-free by design. K_lds
    # reads (in QK) still go through regular ``smem_load_vN`` — converting
    # them to ``ds_read_tr16`` is a follow-up.
    # ---- K/V LDS buffering ----
    # **K is double-buffered** (correctness requirement -- see note below).
    # **V is single-buffered** (safe because V[i+1] is issued in iter i+1
    # AFTER the iter-start ``vmcnt=0, lgkmcnt=0`` drain, which guarantees
    # PV[i]'s LDS reads have retired before V[i+1] can write the shared
    # slot. Saves 8 KiB LDS per CTA).
    #
    # **Q is held in per-lane VGPRs across the kvloop** when its size fits
    # in K_lds[0] (``BLOCK_M * HD <= T * HD`` i.e. ``BLOCK_M <= T``). The
    # prologue cooperatively writes Q into ``K_lds[0]`` (treating it as
    # scratch), syncs, then each lane gathers its MFMA A-operand into 16
    # VGPRs and the K[0] prefetch overwrites the K_lds[0] slot. This
    # eliminates Q's permanent LDS allocation entirely (saving another 8
    # KiB on the NW=4 prefill config). The QK MFMA reads Q from registers,
    # eliminating the per-iter 16-way bank-conflicted Q_lds reads.
    #
    # When ``BLOCK_M > T`` (only the NW=8, BLOCK_M=128 config in our
    # supported set) Q doesn't fit in K_lds[0] and we fall back to a
    # dedicated Q_lds allocation.
    #
    # The K single-buffer variant was tried (see git history) and silently
    # produced wrong results on prefill q>=1k: even with ``wait_K[i] →
    # QK[i] → issue V[i] → issue K[i+1] → softmax → wait_V[i] → PV[i]``,
    # the async-DMA write of ``K[i+1]`` to the shared K slot can race with
    # the tail end of QK[i]'s LDS reads through ``raw_ptr_buffer_load_lds``'s
    # lgkmcnt accounting, corrupting the working tile. K must stay
    # double-buffered.
    Q_BYTES = BLOCK_M * HD * 2
    K_BUF_BYTES = T * HD * 2
    K_TOTAL_BYTES = 2 * K_BUF_BYTES  # K_lds has 2 double-buffer slots
    # Q can alias K_lds when it fits in the full K_lds region (both
    # slots). When ``BLOCK_M <= T`` Q fits in one slot (rows 0..BLOCK_M
    # of K_lds[0]); when ``BLOCK_M > T`` Q spills into K_lds[1] (rows
    # T..2T map to K_lds[1, row-T, :]). The gather + store use
    # ``(buf, row, col)`` indexing in both cases.
    Q_ALIAS_K = Q_BYTES <= K_TOTAL_BYTES
    Q_USES_DUAL_SLOT = Q_ALIAS_K and BLOCK_M > T
    # K_lds double-buffered for bf16 path (race-free async DMA writes to
    # one slot while QK reads the other). For FP8 path: single-buf is
    # race-free because the dequant writes K_lds AFTER the async load of
    # K_fp8_lds completes (the fp8 staging slab takes the double-buffer
    # role); the dequant + barrier sequence ensures K_lds reads see the
    # right tile.
    K_lds_bufs = 1 if KV_FP8 else 2
    K_lds = b.smem_alloc(dtype, [K_lds_bufs, T, HD], name_hint="Klds")
    V_BUFS = 1  # single-buffer V (race-free: see comment above)
    V_lds = b.smem_alloc(dtype, [V_BUFS, T, HD], name_hint="Vlds")
    P_lds = b.smem_alloc(dtype, [BLOCK_M, T], name_hint="Plds")
    # ---- FP8 K/V staging (round-2 async-DMA path) ----
    # When KV_FP8, the loader is split into two phases:
    #   1. async-DMA raw fp8 bytes from HBM into the fp8 staging slab
    #      (K_fp8_lds / V_fp8_lds) — same hardware primitive as the bf16
    #      async load, just with fp8 byte-stride descriptors and one
    #      call/wave for T=64/HD=64 (fp8 is half the bytes of bf16 so we
    #      need half the calls).
    #   2. After ``s_waitcnt vmcnt=0; s_barrier``, sync dequant from the
    #      fp8 staging slab into the existing K_lds / V_lds (bf16) — each
    #      thread reads 8 fp8, applies ``cvt_fp8_to_f32 * scale -> bf16``,
    #      writes 8 bf16. The dequant is LDS-to-LDS (no HBM stall) and
    #      runs entirely from the same WG (no cross-WG synchronisation).
    #
    # The bf16 K_lds and V_lds layouts are unchanged so the rest of the
    # kernel (Q gather, QK MFMA, P_lds publish, PV MFMA, epilogue) doesn't
    # know about FP8.
    if KV_FP8:
        K_fp8_lds = b.smem_alloc(FP8E4M3, [2, T, HD], name_hint="Kfp8lds")
        V_fp8_lds = b.smem_alloc(FP8E4M3, [1, T, HD], name_hint="Vfp8lds")
    if Q_ALIAS_K:
        # Reuse K_lds[0] (and K_lds[1] if BLOCK_M > T) as Q-load scratch.
        # Q is gathered to VGPRs after, then the K-prefetch overwrites
        # the slot(s).
        Q_lds = K_lds
    else:
        Q_lds = b.smem_alloc(dtype, [BLOCK_M, HD], name_hint="Qlds")
    Acc_lds = b.smem_alloc(dtype, [BLOCK_M, OUT_STRIPE_COLS], name_hint="Aclds")

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
        # When Q is aliased to K_lds (single shared scratch), the store
        # goes through ``[buf, row_in_buf, col]`` indexing of the K_lds
        # 3D buffer. ``buf = Q_row // T`` selects which K_lds slot,
        # ``row_in_buf = Q_row % T`` is the row within that slot.
        # For single-slot (BLOCK_M <= T), buf is statically 0 and
        # row_in_buf == Q_row, so the div/mod fold to constants.
        if Q_ALIAS_K:
            if Q_USES_DUAL_SLOT:
                q_buf = b.div(Q_row, b.const_i32(T))
                q_row_in_buf = b.mod(Q_row, b.const_i32(T))
            else:
                q_buf = b.const_i32(0)
                q_row_in_buf = Q_row
            q_store_idx = [q_buf, q_row_in_buf, Q_col]
        else:
            q_store_idx = [Q_row, Q_col]
        b.smem_store_vN(
            Q_lds,
            q_store_idx,
            b.vector_select(b.vector_splat(qmask_t, 8), v8, z8),
            8,
        )
    b.sync()
    # The per-lane Q → VGPR gather is deferred until after ``lane_rg`` /
    # ``lane_col`` / ``wave_row_base`` are materialized below; see
    # the ``Q_reg`` block right after the softmax-state allocation.

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

    # ---- Per-lane row map ----
    # For ``block_m_per_warp=16`` the lane owns 4 row slots within one
    # 16-row atom (rows ``lane_rg*4 + r`` for ``r in 0..3``). For
    # ``block_m_per_warp=32`` the lane owns 8 row slots across 2
    # stacked atoms: reg ``r`` maps to ``(atom_idx, in_atom) =
    # (r // 4, r % 4)`` and the in-warp row is
    # ``atom_idx * 16 + lane_rg * 4 + in_atom``.
    def _in_warp_row(r: int) -> Value:
        atom_idx = r // 4
        in_atom = r % 4
        return b.add(
            b.mul(lane_rg, b.const_i32(4)),
            b.const_i32(atom_idx * 16 + in_atom),
        )

    if USE_SINKS:
        m_inits = []
        for r in range(REGS_PER_LANE):
            row = b.add(wave_row_base, _in_warp_row(r))
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
            sink_h = b.global_load(sinks, qh, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            m_inits.append(b.select(qh_in, sink_f, neg_inf))
    else:
        m_inits = [neg_inf for _ in range(REGS_PER_LANE)]
    l_inits = [one_f for _ in range(REGS_PER_LANE)]

    # Acc storage: one vec_f32(4) per (N-tile, M-atom). For
    # ``block_m_per_warp=16`` (M_ATOMS_PER_WARP=1) this collapses to
    # the original ``[vec_f32(4) for n in PV_N_TILES]`` layout.
    acc_zero = b.zero_vec_f32(4)
    acc_inits = [acc_zero for _ in range(PV_N_TILES * M_ATOMS_PER_WARP)]

    def _acc_idx(n: int, atom: int) -> int:
        return n * M_ATOMS_PER_WARP + atom

    iter_args = []
    for r in range(REGS_PER_LANE):
        iter_args.append((f"m{r}", m_inits[r]))
        iter_args.append((f"l{r}", l_inits[r]))
    for n in range(PV_N_TILES):
        for atom in range(M_ATOMS_PER_WARP):
            iter_args.append(
                (
                    f"acc{n}a{atom}" if M_ATOMS_PER_WARP > 1 else f"acc{n}",
                    acc_inits[_acc_idx(n, atom)],
                )
            )

    # ---- Pre-loop: build K/V buffer descriptors and pre-fetch tile 0.
    # The buffer rsrc bounds OOB voffsets to return zero. We size it large
    # so valid block offsets never trip the check.
    big_bytes = b.const_i32(0x7FFF0000)
    key_rsrc = b.buffer_rsrc(key, big_bytes)
    value_rsrc = b.buffer_rsrc(value, big_bytes)

    # Async load contract (bf16 K/V path): dwords=4 means each lane writes
    # 16 bytes lane-contiguous in LDS. One call writes 64 * 8 halfs = 512
    # halfs = 1024 bytes, i.e. a contiguous slice of the natural [T, HD] tile.
    # This works for HD=128 and HD=256 without changing the LDS layout.
    KV_HALVES_PER_CALL = THREADS * 8
    assert (T * HD) % KV_HALVES_PER_CALL == 0
    kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL
    bytes_per_call = KV_HALVES_PER_CALL * 2
    # Byte strides for the paged-KV cache. ``KV_BYTES`` is 2 for bf16, 1
    # for fp8e4m3. The async DMA reads bytes verbatim (no implicit cast),
    # so for the FP8 K/V path the loader switches to a sync per-thread
    # dequant chain below; the byte-stride math here is shared so the
    # paged_kv_desc compiles consistently in both branches.
    kv_stride_blk_b = BS * NUM_KV * HD * KV_BYTES
    kv_stride_tok_b = NUM_KV * HD * KV_BYTES
    kv_stride_h_b = HD * KV_BYTES

    lane_half_base = b.mul(tid, b.const_i32(8))

    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    bytes_per_buf = T * HD * 2  # one [T, HD] *working-dtype* (bf16) slab

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
        # ``wave_lds_offset_i32`` is wave-uniform (it derives from ``wave_id``,
        # which is constant across a wave's lanes). Pin it to SGPR via
        # ``to_sgpr_u32`` so the register allocator doesn't re-materialise
        # it as a per-lane VGPR each time the unrolled K/V-load loops
        # consume it (saves a ``v_readfirstlane_b32`` per use). See
        # ``dsl_docs/primitives/wave_and_cross_lane.md`` ("Wave-uniform
        # LDS base hoist" section).
        wave_lds_offset_i32 = b.to_sgpr_u32(b.mul(wave_id, b.const_i32(WAVE_BYTES)))
        wave_lds_offset_i64 = b.zext(wave_lds_offset_i32, I64)

    # ---- Paged KV byte descriptor (full transform DAG) ----
    # The paged-KV cache is laid out ``[num_blocks, BS, NUM_KV, HD]`` with
    # *byte* strides. The kernel addresses it via a chain of coordinate
    # transforms.
    #
    # **Single-block tile** (``N_BLOCKS_PER_TILE == 1``, ``T == BS``):
    #
    #   1. ``indirect(tile_idx -> physical_block)`` does the
    #      ``physical_block = block_tables[seq_idx*bt_stride + tile_idx]``
    #      table lookup.
    #   2. ``unmerge(linear_half -> (token, dim))`` splits the
    #      cooperative ``THREADS*8`` half count into ``(token_in_tile,
    #      head_dim)`` (token range ``[0, BS)`` here).
    #
    # **Multi-block tile** (``N_BLOCKS_PER_TILE > 1``, ``T == N_B*BS``,
    # used for prefill workloads to match Triton's ``TILE_SIZE=64`` while
    # the paged cache still has ``BS=32``):
    #
    #   1. ``unmerge(linear_half -> (block_within_tile, token, dim))``
    #      where ``block_within_tile in [0, N_B)`` and ``token in [0, BS)``.
    #   2. ``embed((tile_idx, block_within_tile) -> linear_block_idx)``
    #      with strides ``(N_B, 1)`` so
    #      ``linear_block_idx = tile_idx * N_B + block_within_tile``.
    #   3. ``indirect(linear_block_idx -> physical_block)`` looks up
    #      ``block_tables[seq_base + linear_block_idx]`` once per
    #      sub-block (per-wave-uniform when ``per_call_tokens <= BS``;
    #      we enforce this in ``supports_tiled_2d``).
    #
    # In both cases the naive base ``(physical_block, token, kv_head, dim)``
    # with byte strides ``(BS*NUM_KV*HD, NUM_KV*HD, HD, 1) * KV_BYTES``
    # produces the final byte offset. Calling
    # ``paged_kv_desc.offset(b, tile_idx=, linear_half=, kv_head=)``
    # transparently picks up the multi-block lookup; loaders are unchanged.
    # ``seq_base`` indexes the per-sequence offset into the global
    # block_tables; it's CTA-wide-uniform (depends only on ``seq_idx`` and
    # ``bt_stride_p``, both CTA constants). Pin to SGPR so the per-iter
    # ``indirect()`` table lookup inside the paged-KV descriptor doesn't
    # re-materialise the base into a VGPR.
    seq_base = b.to_sgpr_u32(b.mul(seq_idx, bt_stride_p))
    _kv_base = TensorDescriptor.naive(
        "paged_kv_bytes",
        # ``lengths`` here is just informational (validity propagation
        # is driven by the transforms above, not by these bounds).
        lengths=[1 << 24, BS, NUM_KV, HD],
        strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
        coord_names=("physical_block", "token", "kv_head", "dim"),
    )
    if N_BLOCKS_PER_TILE == 1:
        paged_kv_desc = _kv_base.transform(
            indirect(
                "tile_idx",
                into="physical_block",
                table=block_tables,
                base=seq_base,
            ),
            unmerge("linear_half", into=("token", "dim"), dims=(T, HD)),
        )
    else:
        paged_kv_desc = _kv_base.transform(
            unmerge(
                "linear_half",
                into=("block_within_tile", "token", "dim"),
                dims=(N_BLOCKS_PER_TILE, BS, HD),
            ),
            embed(
                ("tile_idx", "block_within_tile"),
                into="linear_block_idx",
                strides=(N_BLOCKS_PER_TILE, 1),
            ),
            indirect(
                "linear_block_idx",
                into="physical_block",
                table=block_tables,
                base=seq_base,
            ),
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
            # CACHE_STREAM (SLC): one-shot streaming load, never re-read
            # within this kernel. Documented in
            # ``dsl_docs/primitives/intrinsics_and_primitives.md`` as the
            # right hint for K-loop streaming tile loads.
            b.async_buffer_load_lds_addr(
                key_rsrc, k_dst, voff, zero_soff, 4, coherency=CACHE_STREAM
            )

    def _issue_v_load_runtime(kv_tile_idx: Value, buf_idx: Value) -> None:
        """Issue async V loads for one tile into V_lds[0] (single-buffered).

        ``buf_idx`` is ignored -- V is single-buffered (safe because PV[i]
        reads retire before V[i+1] is issued in iter i+1, after the
        iter-start full drain). Saves 8 KiB LDS per CTA vs the original
        double-buffer V layout.
        """
        # V is single-buffered; ignore buf_idx, always write slot 0.
        V_wave_base = b.smem_ptr_add(V_lds_addr, wave_lds_offset_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_half,
                kv_head=kv_head_idx,
            )
            v_dst = b.smem_ptr_add(V_wave_base, b.const_i64(call * bytes_per_call))
            # CACHE_STREAM (SLC): V is consumed once per iter and never
            # re-read within this kernel; see _issue_k_load_runtime for
            # the rationale.
            b.async_buffer_load_lds_addr(
                value_rsrc, v_dst, voff, zero_soff, 4, coherency=CACHE_STREAM
            )

    # ---------------- FP8 K/V cache: async DMA loader (round 2) ----------------
    # Two-phase split that mirrors the bf16 path's HW DMA pipeline:
    #   1. `_issue_kv_fp8_async_load` issues `raw.ptr.buffer.load.lds`
    #      writing fp8 bytes directly into K_fp8_lds / V_fp8_lds. Same
    #      dwords=4 (16 bytes/lane) as the bf16 path; one wave covers
    #      16 fp8 elements per lane vs 8 bf16 halves per lane, so fp8
    #      needs half the calls per tile (1 call vs 2 for T=64 HD=64
    #      THREADS=256).
    #   2. `_dequant_fp8_lds_to_bf16` runs in the kv loop, after the
    #      ``s_waitcnt vmcnt=0; s_barrier`` that publishes the fp8
    #      bytes. Each thread reads 8 fp8 from the fp8 slab, applies
    #      ``cvt_fp8_to_f32 * scale -> bf16``, writes 8 bf16 to the
    #      regular K_lds / V_lds slab. The dequant is LDS-to-LDS so it
    #      runs at LDS throughput (no HBM stall).
    #
    # This replaces the round-1 sync path (`_issue_fp8_dequant_loads`)
    # which issued `global_load_vN(FP8, n=8)` per chunk and blocked the
    # wave on every load. For long-prefill no-SW shapes that visit
    # 9-17 kv-tiles per CTA, the sync stall was the dominant cost
    # (5000 µs vs Triton's 50 µs measured on
    # `n254q5880k10999_fp8kv` in round 1).
    if KV_FP8:
        FP8_DWORDS_PER_LANE = 4  # async DMA dwords-per-lane (16 bytes/lane)
        FP8_BYTES_PER_LANE = FP8_DWORDS_PER_LANE * 4
        FP8_ELEMS_PER_LANE = FP8_BYTES_PER_LANE  # 1 byte per fp8 element
        # Per-call element count across all lanes in the WG.
        FP8_ELEMS_PER_CALL = THREADS * FP8_ELEMS_PER_LANE
        assert (T * HD) % FP8_ELEMS_PER_CALL == 0, (
            f"fp8 async loader: T*HD={T * HD} must be divisible by "
            f"THREADS*16={FP8_ELEMS_PER_CALL} (T={T}, HD={HD}, THREADS={THREADS})"
        )
        FP8_CALLS_PER_TILE = (T * HD) // FP8_ELEMS_PER_CALL
        # Wave-uniform LDS offset (same idea as the bf16 path); each
        # wave's lanes write a contiguous WAVE*16-byte slab in LDS.
        FP8_WAVE_BYTES = WAVE * FP8_BYTES_PER_LANE
        FP8_BYTES_PER_CALL = FP8_ELEMS_PER_CALL  # 1 byte per fp8 element
        FP8_BYTES_PER_BUF = T * HD  # 1 byte per fp8 element

        # Lane base in ELEMENTS (== bytes for fp8) for this wave's call.
        lane_fp8_base = b.mul(tid, b.const_i32(FP8_ELEMS_PER_LANE))

        if NUM_WARPS == 1:
            wave_fp8_offset_i64 = b.const_i64(0)
        else:
            wave_fp8_offset_i32 = b.to_sgpr_u32(
                b.mul(wave_id, b.const_i32(FP8_WAVE_BYTES))
            )
            wave_fp8_offset_i64 = b.zext(wave_fp8_offset_i32, I64)

        K_fp8_lds_addr = b.smem_addr_of(K_fp8_lds)
        V_fp8_lds_addr = b.smem_addr_of(V_fp8_lds)

        def _issue_kv_fp8_async_load(
            kv_tile_idx: Value, buf_idx: Value, slot: str
        ) -> None:
            """Issue async DMA of fp8 K or V bytes into the staging slab.

            `slot` is ``"K"`` (uses K_fp8_lds[buf_idx]) or ``"V"`` (uses
            V_fp8_lds[0]; V is single-buffered, buf_idx ignored).
            """
            if slot == "K":
                rsrc = key_rsrc
                buf_off_i32 = b.mul(buf_idx, b.const_i32(FP8_BYTES_PER_BUF))
                buf_off_i64 = b.zext(buf_off_i32, I64)
                buf_base = b.smem_ptr_add(K_fp8_lds_addr, buf_off_i64)
            else:
                rsrc = value_rsrc
                # V is single-buffered; ignore buf_idx
                buf_base = V_fp8_lds_addr
            wave_base = b.smem_ptr_add(buf_base, wave_fp8_offset_i64)
            for call in range(FP8_CALLS_PER_TILE):
                # paged_kv_desc returns a BYTE offset; for fp8, linear_half
                # is interpreted as fp8-ELEMENT index (KV_BYTES=1 so element
                # offset == byte offset within a tile).
                linear_elem = b.add(
                    b.const_i32(call * FP8_ELEMS_PER_CALL), lane_fp8_base
                )
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_elem,
                    kv_head=kv_head_idx,
                )
                lds_dst = b.smem_ptr_add(
                    wave_base, b.const_i64(call * FP8_BYTES_PER_CALL)
                )
                b.async_buffer_load_lds_addr(
                    rsrc,
                    lds_dst,
                    voff,
                    zero_soff,
                    FP8_DWORDS_PER_LANE,
                    coherency=CACHE_STREAM,
                )

        # The dequant step distributes T*HD elements across THREADS in
        # 8-element chunks (matches the existing fp8_chunks_per_thread
        # layout). Each thread reads 8 fp8 from K_fp8_lds, applies the
        # dequant chain, and writes 8 bf16 to K_lds. The K_lds layout is
        # exactly what the bf16 async DMA produces, so the rest of the
        # kernel (Q gather, QK MFMA, PV MFMA) is identical to bf16.
        fp8_dequant_elems_per_chunk = 8
        fp8_dequant_total_chunks = (T * HD) // fp8_dequant_elems_per_chunk
        assert fp8_dequant_total_chunks % THREADS == 0, (
            f"fp8 dequant: total chunks {fp8_dequant_total_chunks} must be "
            f"divisible by THREADS={THREADS}"
        )
        fp8_dequant_chunks_per_thread = fp8_dequant_total_chunks // THREADS
        fp8_cols_per_row = HD // fp8_dequant_elems_per_chunk

        def _dequant_fp8_lds_to_bf16(
            buf_idx: Value, scale: Value, fp8_lds, bf16_lds, bf16_buf: Value
        ) -> None:
            """LDS->LDS dequant: fp8 -> f32 * scale -> bf16.

            `buf_idx` selects the fp8 source buffer (for K, the active
            double-buffer slot; for V, always 0).
            `bf16_buf` selects the bf16 destination buffer (for K, always
            0 since FP8 path single-buffers the bf16 K_lds; for V, 0).
            """
            for c in range(fp8_dequant_chunks_per_thread):
                chunk_id = b.add(b.mul(b.const_i32(c), b.const_i32(THREADS)), tid)
                row = b.div(chunk_id, b.const_i32(fp8_cols_per_row))
                col = b.mul(
                    b.mod(chunk_id, b.const_i32(fp8_cols_per_row)),
                    b.const_i32(fp8_dequant_elems_per_chunk),
                )
                fp8_vec = b.smem_load_vN(
                    fp8_lds,
                    buf_idx,
                    row,
                    col,
                    dtype=FP8E4M3,
                    n=fp8_dequant_elems_per_chunk,
                )
                dequanted = []
                for i in range(fp8_dequant_elems_per_chunk):
                    fp8_v = b.vec_extract(fp8_vec, i)
                    f32_v = b.fmul(b.cvt_fp8_to_f32(fp8_v), scale)
                    dequanted.append(b.cast_f32_to(f32_v, dtype))
                packed = b.vec_pack(dequanted, dtype)
                b.smem_store_vN(
                    bf16_lds,
                    [bf16_buf, row, col],
                    packed,
                    fp8_dequant_elems_per_chunk,
                )

    # ---------------- FP8 K/V cache: sync dequant loader (round 1, kept as fallback) ----------------
    # Each thread loads one byte per fp8 element from HBM, dequantises
    # (cvt_fp8_to_f32 * scale), casts to the working bf16/fp16 dtype, and
    # stores 8 elements at a time to LDS. The total bytes per tile match
    # the async path's working-dtype LDS layout, so the rest of the
    # kernel reads K_lds / V_lds in the working dtype unchanged.
    #
    # Layout: distribute T*HD elements across THREADS threads such that
    # each thread processes a contiguous run of ``elems_per_thread`` fp8
    # bytes from HBM, lane-contiguous, then writes them as bf16 to LDS at
    # the same linear offset. We pick the chunk size to be 8 so the LDS
    # store is one ``smem_store_vN(..., n=8)``.
    fp8_elems_per_chunk = 8
    fp8_total_chunks = (T * HD) // fp8_elems_per_chunk
    assert fp8_total_chunks % THREADS == 0, (
        f"fp8 loader: total chunks {fp8_total_chunks} must be divisible by "
        f"THREADS={THREADS} (T={T}, HD={HD})"
    )
    fp8_chunks_per_thread = fp8_total_chunks // THREADS

    def _issue_fp8_dequant_loads(
        kv_tile_idx: Value, buf_idx: Value, lds_token: str
    ) -> None:
        """Sync per-thread fp8 -> f32 -> *scale -> bf16/fp16 -> LDS.

        ``lds_token`` is either ``"K"`` or ``"V"``; selects the LDS slab
        and the per-tensor scale parameter.
        """
        scale = k_scale_p if lds_token == "K" else v_scale_p
        lds = K_lds if lds_token == "K" else V_lds
        src = key if lds_token == "K" else value
        for call in range(fp8_chunks_per_thread):
            # One thread, one 8-fp8 chunk per call. Across THREADS threads
            # and ``fp8_chunks_per_thread`` calls, we cover all T*HD elements.
            chunk_id = b.add(
                b.mul(b.const_i32(call), b.const_i32(THREADS)),
                tid,
            )
            row = b.div(chunk_id, b.const_i32(HD // fp8_elems_per_chunk))
            col = b.mul(
                b.mod(chunk_id, b.const_i32(HD // fp8_elems_per_chunk)),
                b.const_i32(fp8_elems_per_chunk),
            )
            # Compute the per-element byte offset for the first fp8 in this
            # chunk via the paged-KV descriptor. The descriptor returns a
            # byte offset (KV_BYTES=1 → identical to the element offset).
            linear_half_first = b.add(
                b.mul(row, b.const_i32(HD)),
                col,
            )
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_half_first,
                kv_head=kv_head_idx,
            )
            # Dequant 8 fp8 elements -> 8 bf16 elements. We do a buffer-rsrc
            # OOB-safe scalar load per byte to keep the loader simple; the
            # compiler folds the 8 calls into a single 8-byte VMEM load
            # when the address pattern is contiguous.
            # One vectorised fp8 load (8 bytes -> <8 x fp8e4m3>) replaces
            # 8 scalar byte loads. AMDGPU coalesces this into a single
            # 8-byte VMEM op; the dequant chain then unpacks per-lane.
            fp8_vec = b.global_load_vN(
                src, voff, FP8E4M3, n=fp8_elems_per_chunk, align=fp8_elems_per_chunk
            )
            dequanted = []
            for i in range(fp8_elems_per_chunk):
                fp8_v = b.vec_extract(fp8_vec, i)
                f32_v = b.fmul(b.cvt_fp8_to_f32(fp8_v), scale)
                dequanted.append(b.cast_f32_to(f32_v, dtype))
            packed = b.vec_pack(dequanted, dtype)
            b.smem_store_vN(lds, [buf_idx, row, col], packed, fp8_elems_per_chunk)
        # Caller is expected to issue a `b.sync()` (LDS visibility) when
        # appropriate. The sync loader has no in-flight async work, so the
        # consumer side only needs the LDS barrier, not VMEM waitcnt.

    def _issue_k(tile_idx: Value, buf_idx: Value) -> None:
        """Issue a K load into the appropriate LDS slab.

        For bf16: async DMA to K_lds (bf16 working dtype).

        For FP8: round-2 attempted an async DMA of raw fp8 bytes into
        K_fp8_lds followed by an LDS->LDS dequant, mirroring the bf16
        pipeline. The dequant works (correctness preserved) but the
        explicit phase ordering (async-load → barrier → dequant → barrier
        → MFMA) LOSES the chunk-level instruction pipelining that the
        round-1 sync loader got implicitly from the compiler interleaving
        load/dequant/store across chunks. Result: ~10% REGRESSION on
        long-prefill no-SW FP8 (5494 µs vs 4967 µs round 1, measured on
        `n254q5880k10999_fp8kv` Wed 2026-05-19). Reverted; the async
        infrastructure (K_fp8_lds, V_fp8_lds, _issue_kv_fp8_async_load,
        _dequant_fp8_lds_to_bf16) is kept in the source for future
        iteration but the dispatch routes back to the sync loader.
        """
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "K")
        else:
            _issue_k_load_runtime(tile_idx, buf_idx)

    def _issue_v(tile_idx: Value, buf_idx: Value) -> None:
        """Issue a V load (single-buffered). See `_issue_k` for FP8 notes."""
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "V")
        else:
            _issue_v_load_runtime(tile_idx, buf_idx)

    # ---- Per-lane Q → VGPR gather (eliminates per-iter Q LDS reads) ----
    # Each lane reads its MFMA-A operand slice of Q (16 halves per atom)
    # into VGPRs ONCE per CTA. Subsequent QK iterations use ``Q_reg``
    # directly instead of paying the 16-way bank-conflicted Q_lds read
    # ``num_tiles`` times. Per lane VGPR cost: 8 halves × QK_K_ITERS ×
    # M_ATOMS_PER_WARP = up to 32 halves = 16 VGPRs for
    # ``BLOCK_M_PER_WARP=32``.
    Q_reg = [[None] * QK_K_ITERS for _ in range(M_ATOMS_PER_WARP)]
    for atom in range(M_ATOMS_PER_WARP):
        q_row_atom = b.add(wave_row_base, b.add(b.const_i32(atom * 16), lane_col))
        # Map Q_row → (buf, row_in_buf) for the K_lds-aliased case.
        if Q_ALIAS_K:
            if Q_USES_DUAL_SLOT:
                q_buf_atom = b.div(q_row_atom, b.const_i32(T))
                q_row_in_buf_atom = b.mod(q_row_atom, b.const_i32(T))
            else:
                q_buf_atom = b.const_i32(0)
                q_row_in_buf_atom = q_row_atom
        for k in range(QK_K_ITERS):
            q_col_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
            q_load_idx_args = (
                (q_buf_atom, q_row_in_buf_atom, q_col_off)
                if Q_ALIAS_K
                else (q_row_atom, q_col_off)
            )
            Q_reg[atom][k] = b.smem_load_vN(Q_lds, *q_load_idx_args, dtype=dtype, n=8)

    if Q_ALIAS_K:
        # Drain the per-lane Q-gather LDS reads BEFORE issuing K[0] async
        # write to the same K_lds[0] slot. Without this, the async DMA's
        # LDS write can race with the in-flight ds_read, corrupting Q
        # for the first QK iter. One-time cost at kernel start.
        b.s_waitcnt(lgkmcnt=0)
        b.sync()

    # Prefetch tile_start's K into buffer 0 BEFORE the loop.
    _issue_k(tile_start, b.const_i32(0))

    # ---------------- KV tile loop ----------------
    # Double-buffered: we carry ``cur_buf`` (the buffer that holds tile i's
    # data) through the loop. At iter i we:
    #   1. Wait for current K (prefetched by the previous iteration, or the
    #      pre-loop prologue).
    #   2. Compute QK.
    #   3. Issue current V, then next K, and run softmax while both are in
    #      flight. Since current V is older than next K in the VMEM/LGKM
    #      queues, a partial wait with ``kv_calls_per_tile`` pending leaves
    #      next K in flight while making current V visible for PV.
    #   4. ``s_barrier`` to make tile i's data visible to all reads.
    #   5. Wait for current V, publish P_lds, then run PV.
    #   6. Yield ``(m, l, acc, nxt_buf)`` so the next iter consumes nxt_buf.
    cur_buf_init = b.const_i32(0)
    iter_args.append(("cur_buf", cur_buf_init))

    # ---------------- LICM hoist: per-reg invariants ----------------
    # ``qp_r``, ``qh_r``, ``row_ok``, ``causal_lim``, and ``alibi_per_row``
    # depend only on CTA-level constants (``wave_row_base``, ``lane_rg``,
    # ``qb_start_pos``, ``kv_head_idx``, ``cur_batch_q_len``, ``context_len``,
    # ``NUM_QH``, ``NQK``). Computing them BEFORE the kvloop avoids paying
    # them per kv tile. LLVM LICM should hoist them automatically, but
    # explicit hoisting makes the IR's hot path leaner and eliminates a
    # source of compiler-scheduling variability. The per-reg arrays
    # ``hoist_*[reg]`` for ``reg in range(REGS_PER_LANE)`` are indexed by
    # the per-lane row slot.
    hoist_row = []
    hoist_qp_r = []
    hoist_qh_r = []
    hoist_row_ok = []
    hoist_causal_lim = []
    for reg in range(REGS_PER_LANE):
        row = b.add(wave_row_base, _in_warp_row(reg))
        qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
        qh_r = b.add(b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK)))
        row_ok = b.land(
            b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
        )
        causal_lim = b.add(context_len, qp_r)
        hoist_row.append(row)
        hoist_qp_r.append(qp_r)
        hoist_qh_r.append(qh_r)
        hoist_row_ok.append(row_ok)
        hoist_causal_lim.append(causal_lim)

    if USE_ALIBI:
        hoist_alibi = []
        for reg in range(REGS_PER_LANE):
            qh_r = hoist_qh_r[reg]
            qh_ok = b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            slope = b.masked_global_load(
                alibi_slopes_ptr, qh_r, qh_ok, b.const_f32(0.0), dtype=F32, align=4
            )
            hoist_alibi.append(slope)
    else:
        hoist_alibi = None

    kvloop = b.scf_for_iter(
        tile_start, tile_end, b.const_i32(1), iter_args, iv_name="kv_tile"
    )
    with kvloop as (kv_tile_iv, carry):
        m_vals = [carry[2 * r] for r in range(REGS_PER_LANE)]
        l_vals = [carry[2 * r + 1] for r in range(REGS_PER_LANE)]
        ml_count = 2 * REGS_PER_LANE
        acc_vals = [
            carry[ml_count + n * M_ATOMS_PER_WARP + a]
            for n in range(PV_N_TILES)
            for a in range(M_ATOMS_PER_WARP)
        ]

        # acc_vals is flat indexed by (n * M_ATOMS_PER_WARP + atom).
        def _acc_get(n: int, atom: int) -> Value:
            return acc_vals[n * M_ATOMS_PER_WARP + atom]

        cur_buf = carry[ml_count + PV_N_TILES * M_ATOMS_PER_WARP]
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

        # ---- FP8 path: dequant K_fp8_lds[cur_buf] -> K_lds[0] ----
        # After the wait above, K_fp8_lds[cur_buf] holds the raw fp8 bytes
        # for tile `kv_tile_iv`. Run the cooperative LDS->LDS dequant
        # which populates K_lds[0] (single-buf for the FP8 path) with bf16
        # in the layout the QK MFMA reads expect. A second barrier
        # publishes the bf16 to all warps. The bf16 path skips this
        # entirely. K_fp8_lds keeps double-buffer; K_lds is single-buf so
        # FP8's LDS footprint matches bf16's (no occupancy regression).
        if KV_FP8:
            _dequant_fp8_lds_to_bf16(
                cur_buf, k_scale_p, K_fp8_lds, K_lds, b.const_i32(0)
            )
            # ds_writes need lgkmcnt drain + barrier so QK's per-warp
            # ds_reads see the dequanted bf16.
            b.s_waitcnt(lgkmcnt=0)
            b.sync()

        # ---- S = Q @ K^T (per-warp MFMA) ----
        # Q is in LDS only; we re-read it per iter -- the compiler hoists the
        # LDS reads across iterations when alignment lets it (Q never changes
        # after the prelude). Each warp reads its own ``BLOCK_M_PER_WARP``-row
        # slice of Q[BLOCK_M, HD] at rows
        # ``[wave_row_base, wave_row_base + BLOCK_M_PER_WARP)``.
        # For ``BLOCK_M_PER_WARP=32`` we read Q for both atoms (rows
        # ``wave_row_base + lane_col`` for atom 0 and ``wave_row_base + 16 +
        # lane_col`` for atom 1), then run the inner MFMA loop twice -- once
        # per atom -- sharing the same K read across both atoms.
        # Q comes from pre-loop VGPR gather (``Q_reg[atom][k]``); no per-iter
        # Q_lds reads, no per-iter Q LDS bank conflict.
        A_kits = Q_reg
        # S_n[atom][n] = vec_f32(4) -- per-atom, per-N-tile accumulator.
        # ``sched_group_barrier`` hints were tried here (mirroring the CK
        # Tile ``compv4`` GEMM pattern) but **regressed prefill_q64 by
        # ~50%** -- the hints constrain the scheduler in a way that doesn't
        # fit attention's mask + softmax + PV pattern, where the post-RA
        # scheduler's default heuristics already produce good interleave.
        # Consistent with the conv-kernel optimization study finding that
        # "compiler scheduling hints don't work on gfx950" (see
        # ``/workspace/mlse-tools-internal/performance/kernel_optimization/
        # analysis/00_CONSOLIDATED_FINDINGS.md``). Leaving them out.
        S_n = [[None] * QK_N_TILES for _ in range(M_ATOMS_PER_WARP)]
        for n in range(QK_N_TILES):
            acc_per_atom = [b.zero_vec_f32(4) for _ in range(M_ATOMS_PER_WARP)]
            for k in range(QK_K_ITERS):
                kc_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                k_row = b.add(b.const_i32(n * 16), lane_col)
                # For FP8 path K_lds is single-buf; the dequant always
                # populates slot 0. The bf16 path uses cur_buf as the
                # double-buffer index.
                k_lds_buf = b.const_i32(0) if KV_FP8 else cur_buf
                B_v = b.smem_load_vN(K_lds, k_lds_buf, k_row, kc_off, dtype=dtype, n=8)
                for atom in range(M_ATOMS_PER_WARP):
                    acc_per_atom[atom] = _mfma_16x16x32(
                        b, dtype, A_kits[atom][k], B_v, acc_per_atom[atom]
                    )
            for atom in range(M_ATOMS_PER_WARP):
                S_n[atom][n] = acc_per_atom[atom]

        # Now that QK no longer needs VMEM, start current V first and next K
        # second. This ordering is what lets the partial wait before PV leave
        # only next K pending.
        _issue_v(kv_tile_iv, cur_buf)
        _issue_k(safe_next_tile, nxt_buf)

        # ---- mask / scale / softcap / alibi / qq-bias ----
        # ALiBi and QQ-bias mirror Triton's apply-before-mask-result semantics:
        # we fold them into the unmasked S, then the select-with-(-inf) below
        # zeroes them out for invalid cells (finite + (-inf) = (-inf) in IEEE
        # for the mask path so result is identical to Triton's
        # "S = where(mask, S, -inf); S += bias" formulation).
        # All per-reg loops iterate over REGS_PER_LANE = 4 (block_m_per_warp=16)
        # or 8 (block_m_per_warp=32); reg `r` maps to in-warp-row
        # `(r // 4) * 16 + lane_rg * 4 + (r % 4)` and to the QK acc atom
        # ``S_n[r // 4][n]``.
        # ``alibi_per_row``, ``qp_r``, ``qh_r``, ``row_ok``, ``causal_lim``
        # are all CTA-constants -- pulled in from the pre-loop hoist above.
        alibi_per_row = hoist_alibi if USE_ALIBI else None
        masked = {}
        for reg in range(REGS_PER_LANE):
            atom = reg // 4
            in_atom = reg % 4
            qp_r = hoist_qp_r[reg]
            row_ok = hoist_row_ok[reg]
            causal_lim = hoist_causal_lim[reg]
            for n in range(QK_N_TILES):
                col_abs = b.add(
                    b.add(tile_off, b.mul(b.const_i32(n), b.const_i32(16))), lane_col
                )
                causal_ok = b.cmp_le(col_abs, causal_lim)
                in_prefix = b.cmp_lt(col_abs, max_seq_prefix_len)
                m_ok = b.land(b.land(row_ok, causal_ok), in_prefix)
                if SLIDING_WINDOW > 0:
                    dist = b.sub(causal_lim, col_abs)
                    m_ok = b.land(m_ok, b.cmp_lt(dist, sw_const))
                s_raw = b.vec_extract(S_n[atom][n], in_atom)
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
        # Each lane has REGS_PER_LANE floats (one per row in its row-group(s)),
        # repeated for every N-tile. Local lane-max across N-tiles, then 4-stage
        # XOR butterfly across the 16 lanes in the row-group.
        m_new = []
        s_local = {}  # (reg, n) -> the lane's masked score (still owned per-lane)
        for reg in range(REGS_PER_LANE):
            local_max = neg_inf
            for n in range(QK_N_TILES):
                v = masked[(n, reg)]
                s_local[(reg, n)] = v
                local_max = b.fmax(local_max, v)
            tile_max = _warp_xor_reduce_max(b, local_max)
            # Online softmax update (FlashAttention/Triton): see docstring.
            full_max_raw = b.fmax(m_vals[reg], tile_max)
            ok = b.fcmp("ogt", full_max_raw, neg_inf)
            m_new.append(b.select(ok, full_max_raw, zero_f))

        # ---- compute P = exp2(S - m_new) and l_local = sum(P) per row ----
        # P_lds[row, col] = exp2(S[row, col] - m_new[row]). Each warp publishes
        # its own BLOCK_M_PER_WARP-row slice (16 or 32 rows). Row coords come
        # from the pre-loop hoist.
        l_local = []
        for reg in range(REGS_PER_LANE):
            row = hoist_row[reg]
            sum_p = zero_f
            for n in range(QK_N_TILES):
                p = b.exp2(b.fsub(s_local[(reg, n)], m_new[reg]))
                col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
                b.smem_store_vN(P_lds, [row, col], b.cast_f32_to(p, dtype), 1)
                sum_p = b.fadd(sum_p, p)
            l_local.append(_warp_xor_reduce_sum(b, sum_p))

        # alpha and L update (still per-lane registers; matches FA-2 paper)
        alpha_regs = [b.exp2(b.fsub(m_vals[r], m_new[r])) for r in range(REGS_PER_LANE)]
        new_l_vals = [
            b.fadd(b.fmul(l_vals[r], alpha_regs[r]), l_local[r])
            for r in range(REGS_PER_LANE)
        ]
        if KV_FP8:
            # Round-2 async FP8 path: V was issued async before next-K, so
            # `FP8_CALLS_PER_TILE` pending VMEM ops are exactly the next-K
            # stream. Partial wait lets V's fp8 bytes retire while next-K
            # stays in flight. lgkmcnt mirrored so we don't block on the
            # next-K LDS write tracking either.
            b.s_waitcnt(vmcnt=FP8_CALLS_PER_TILE, lgkmcnt=FP8_CALLS_PER_TILE)
            b.sync()
            # Dequant V_fp8_lds[0] -> V_lds[0] so PV reads the working dtype.
            _dequant_fp8_lds_to_bf16(
                b.const_i32(0), v_scale_p, V_fp8_lds, V_lds, b.const_i32(0)
            )
            b.s_waitcnt(lgkmcnt=0)
            b.sync()
        else:
            # Wait for current V while leaving next K pending. Current V was
            # issued before next K, so `kv_calls_per_tile` pending operations are
            # exactly the next-K stream. Apply the same idea to lgkmcnt so we do
            # not wait for the next-K LDS writes before PV.
            b.s_waitcnt(vmcnt=kv_calls_per_tile, lgkmcnt=kv_calls_per_tile)
            b.sync()

        # ---- acc *= alpha, acc += P @ V ----
        # For ``M_ATOMS_PER_WARP=2`` we stack two MFMAs in M per (n, k) atom:
        # both atoms share the same B (V) operand, but read different P rows
        # (atom 0: rows wave_row_base..wave_row_base+15; atom 1: rows
        # wave_row_base+16..wave_row_base+31). Each atom has its own
        # accumulator + per-reg alpha.
        new_acc = [None] * (PV_N_TILES * M_ATOMS_PER_WARP)
        for n in range(PV_N_TILES):
            # Per-atom: scale the inherited acc by per-row alpha, then add P @ V.
            acc_per_atom: list[Value] = []
            for atom in range(M_ATOMS_PER_WARP):
                scaled_comps = []
                for in_atom in range(4):
                    reg = atom * 4 + in_atom
                    e = b.vec_extract(_acc_get(n, atom), in_atom)
                    scaled_comps.append(b.fmul(e, alpha_regs[reg]))
                acc_per_atom.append(b.vec_pack(scaled_comps, F32))

            # PV's K-direction TransposeLDSLayout row/col addresses are
            # produced by ``pv_tr_reader`` -- :meth:`row(k_offset, read)`
            # computes ``(lane/16)*K_L + read*4 + (lane/4)%4 + k_offset``
            # for one ds_read_b64_tr_b16. ``tr_col_lane`` is the cached
            # ``(lane%4)*4`` column component.
            n_col_base = b.add(b.mul(b.const_i32(n), b.const_i32(16)), tr_col_lane)

            # V is single-buffered; the V_lds buffer index is always 0.
            v_buf = b.const_i32(0)
            for k in range(PV_K_ITERS):
                if PV_K_STEP == 32:
                    # K=32: P operand 8 halves, V via 2 ds_read_b64_tr_b16 reads.
                    p_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                    row_r0 = pv_tr_reader.row(b, k_offset=k * 32, read=0)
                    row_r1 = pv_tr_reader.row(b, k_offset=k * 32, read=1)
                    B_r0 = b.ds_read_tr16_b64(
                        V_lds, v_buf, row_r0, n_col_base, dtype=dtype
                    )
                    B_r1 = b.ds_read_tr16_b64(
                        V_lds, v_buf, row_r1, n_col_base, dtype=dtype
                    )
                    B_v = b.vec_concat(B_r0, B_r1)
                    for atom in range(M_ATOMS_PER_WARP):
                        # P_lds row for this atom: each warp's atom_idx slice
                        # of P_lds[BLOCK_M_PER_WARP, T] -- the in-warp row is
                        # ``atom * 16 + lane_col``.
                        p_row = b.add(
                            wave_row_base, b.add(b.const_i32(atom * 16), lane_col)
                        )
                        A_p = b.smem_load_vN(P_lds, p_row, p_off, dtype=dtype, n=8)
                        acc_per_atom[atom] = _mfma_16x16x32(
                            b, dtype, A_p, B_v, acc_per_atom[atom]
                        )
                else:
                    # K=16: single ds_read_b64_tr_b16 returns the full B operand.
                    p_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                    row_lane = pv_tr_reader.row(b, k_offset=k * 16, read=0)
                    B_v = b.ds_read_tr16_b64(
                        V_lds, v_buf, row_lane, n_col_base, dtype=dtype
                    )
                    for atom in range(M_ATOMS_PER_WARP):
                        p_row = b.add(
                            wave_row_base, b.add(b.const_i32(atom * 16), lane_col)
                        )
                        A_p = b.smem_load_vN(P_lds, p_row, p_off, dtype=dtype, n=4)
                        acc_per_atom[atom] = _mfma_16x16x16(
                            b, dtype, A_p, B_v, acc_per_atom[atom]
                        )
            for atom in range(M_ATOMS_PER_WARP):
                new_acc[n * M_ATOMS_PER_WARP + atom] = acc_per_atom[atom]

        yields = []
        for r in range(REGS_PER_LANE):
            yields.append(m_new[r])
            yields.append(new_l_vals[r])
        for n in range(PV_N_TILES):
            for atom in range(M_ATOMS_PER_WARP):
                yields.append(new_acc[n * M_ATOMS_PER_WARP + atom])
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
    l_final = [final[2 * r + 1] for r in range(REGS_PER_LANE)]
    ml_count_final = 2 * REGS_PER_LANE
    # acc_final indexed by (n * M_ATOMS_PER_WARP + atom)
    acc_final = [
        final[ml_count_final + n * M_ATOMS_PER_WARP + atom]
        for n in range(PV_N_TILES)
        for atom in range(M_ATOMS_PER_WARP)
    ]

    def _acc_final_get(n: int, atom: int) -> Value:
        return acc_final[n * M_ATOMS_PER_WARP + atom]

    # Per-row reciprocal of L (computed once, reused across stripes).
    rcp_l = [b.rcp(l_final[r]) for r in range(REGS_PER_LANE)]
    l_nonzero = [b.fcmp("ogt", l_final[r], zero_f) for r in range(REGS_PER_LANE)]

    # ---------------- striped epilogue ----------------
    # Loop in ``OUT_STRIPES`` stripes, each covering ``OUT_STRIPE_COLS = 32``
    # consecutive output columns (= 2 PV N-tiles). For each stripe we:
    #   1. Cast and normalise each warp's MFMA acc slice (4 floats per
    #      N-tile per lane), store as the working dtype into Acc_lds.
    #   2. Sync so the cooperative output store sees every warp's writes.
    #   3. Cooperative vec8 output store from Acc_lds into the global
    #      output buffer at the right stripe column base.
    #   4. Sync so the next stripe can safely overwrite Acc_lds.
    #
    # ``Acc_lds`` is only [BLOCK_M, OUT_STRIPE_COLS] of the working dtype,
    # so the per-CTA epilogue LDS is ``BLOCK_M * 32 * 2`` bytes -- a 75-87%
    # reduction vs the previous ``BLOCK_M * HD * 4`` F32 buffer. That LDS
    # savings is what gives MI355X room for 3 WGs/CU on prefill workloads
    # (the documented Triton ``num_warps=4`` BLOCK_M=128 configuration runs
    # at 3-4 WGs/CU; we now match that occupancy class).
    N_TILES_PER_STRIPE = OUT_STRIPE_COLS // MFMA_N
    assert PV_N_TILES % N_TILES_PER_STRIPE == 0
    # Cooperative store distribution for one stripe ([BLOCK_M, OUT_STRIPE_COLS]
    # of dtype). Per stripe: total halves = BLOCK_M * OUT_STRIPE_COLS; per
    # thread = BLOCK_M * OUT_STRIPE_COLS / THREADS. We unroll
    # ``OUT_CHUNKS_PER_THREAD`` consecutive 16-byte ``vec8`` stores per thread
    # so each thread always writes one row's slice of the stripe contiguously.
    # For BLOCK_M=16 NW=1 HD=128 (decode) THREADS=64, full-HD stripe:
    #   16*128/64 = 32 halves/thread = 4 vec8 chunks per thread per stripe.
    # For BLOCK_M=64 NW=4 HD=64 (prefill) THREADS=256, 32-col stripe:
    #   64*32/256 = 8 halves/thread = 1 vec8 chunk per thread per stripe.
    OUT_VEC = 8
    OUT_PER_THREAD_HALVES = (BLOCK_M * OUT_STRIPE_COLS) // THREADS
    assert OUT_PER_THREAD_HALVES % OUT_VEC == 0 and OUT_PER_THREAD_HALVES > 0, (
        f"Expected a positive multiple of vec{OUT_VEC} halves per thread per "
        f"stripe (got {OUT_PER_THREAD_HALVES} for BLOCK_M={BLOCK_M} "
        f"STRIPE_COLS={OUT_STRIPE_COLS} THREADS={THREADS})"
    )
    OUT_CHUNKS_PER_THREAD = OUT_PER_THREAD_HALVES // OUT_VEC
    OUT_THREADS_PER_ROW = OUT_STRIPE_COLS // (OUT_CHUNKS_PER_THREAD * OUT_VEC)
    OUT_ROWS_PER_ITER = THREADS // OUT_THREADS_PER_ROW
    assert OUT_ROWS_PER_ITER == BLOCK_M, (
        f"Stripe cooperative-store assumes one row per thread group "
        f"(got OUT_ROWS_PER_ITER={OUT_ROWS_PER_ITER}, BLOCK_M={BLOCK_M})"
    )
    OUT_ROW_BASE = b.div(tid, b.const_i32(OUT_THREADS_PER_ROW))
    OUT_col_base_in_stripe = b.mul(
        b.mod(tid, b.const_i32(OUT_THREADS_PER_ROW)),
        b.const_i32(OUT_CHUNKS_PER_THREAD * OUT_VEC),
    )

    # Compute (op_pos, op_qh, op_mask, out_base) once per CTA -- these
    # depend only on OUT_row, which is loop-invariant across stripes.
    op_pos = b.add(qb_start_pos, b.div(OUT_ROW_BASE, b.const_i32(NQK)))
    op_qh = b.add(
        b.mul(kv_head_idx, b.const_i32(NQK)),
        b.mod(OUT_ROW_BASE, b.const_i32(NQK)),
    )
    op_mask = b.land(
        b.cmp_lt(op_pos, cur_batch_q_len), b.cmp_lt(op_qh, b.const_i32(NUM_QH))
    )
    out_base, _ = q_desc.offset(
        b,
        token=b.add(cu_q_start, op_pos),
        head=op_qh,
        dim=b.const_i32(0),
    )

    for stripe in range(OUT_STRIPES):
        n_start = stripe * N_TILES_PER_STRIPE
        # ---- stage 1: write this stripe's N-tiles into Acc_lds ----
        # For ``M_ATOMS_PER_WARP=2`` each warp writes 2 stacked 16-row tiles
        # per N-tile. The reg loop iterates over all REGS_PER_LANE = 4*M_ATOMS
        # row slots; reg ``r`` decomposes into ``(atom=r//4, in_atom=r%4)``
        # for both the row offset and the per-atom accumulator pick.
        for n_local in range(N_TILES_PER_STRIPE):
            n = n_start + n_local
            for reg in range(REGS_PER_LANE):
                atom = reg // 4
                in_atom = reg % 4
                row = b.add(wave_row_base, _in_warp_row(reg))
                # Column within the stripe = n_local*16 + lane_col
                col_in_stripe = b.add(b.const_i32(n_local * MFMA_N), lane_col)
                v = b.vec_extract(_acc_final_get(n, atom), in_atom)
                normalized = b.fmul(v, rcp_l[reg])
                final_h = b.cast_f32_to(
                    b.select(l_nonzero[reg], normalized, zero_f), dtype
                )
                b.smem_store_vN(Acc_lds, [row, col_in_stripe], final_h, 1)
        b.sync()
        # ---- stage 2: cooperative vec8 store(s) from Acc_lds to global ----
        for chunk in range(OUT_CHUNKS_PER_THREAD):
            col_in_stripe = b.add(OUT_col_base_in_stripe, b.const_i32(chunk * OUT_VEC))
            v8h = b.smem_load_vN(
                Acc_lds, OUT_ROW_BASE, col_in_stripe, dtype=dtype, n=OUT_VEC
            )
            out_col = b.add(b.const_i32(stripe * OUT_STRIPE_COLS), col_in_stripe)
            with b.scf_if(op_mask):
                b.global_store_vN(
                    output, b.add(out_base, out_col), v8h, OUT_VEC, align=16
                )
        # ---- stage 3: sync so the next stripe can overwrite Acc_lds ----
        if stripe + 1 < OUT_STRIPES:
            b.sync()

    return b.kernel
