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
  - QK computes S^T = K Q^T, so each lane owns one query row: the online
    softmax keeps one (m, l) per lane, and P stays in registers as the B
    operand of O^T = V^T P^T (no LDS round trip for P).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from kernels.common._attention_additive_bias import (
    add_additive_bias,
    declare_additive_bias_params,
)
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
    mfma_16x16x32_for_dtype as _mfma_16x16x32,
    wave64_reduce_max as _wave64_reduce_max,
    wave64_reduce_sum as _wave64_reduce_sum,
)
from rocke.helpers.distribution import make_static_tile_distribution
from rocke.helpers.layouts import TransposeLdsReader
from rocke.helpers.transforms import TensorDescriptor, indirect, unmerge

MFMA_M = 16
MFMA_N = 16
# Async K/V feed: the segment kernel is one wave64 CTA, and each lane's
# buffer_load ... lds moves 4 DWORDs (8 halves), so one async call stages
# 64 * 8 halves. Every call is one outstanding vmcnt event; the gfx950
# s_waitcnt vmcnt field is 6 bits.
SEGMENT_THREADS = 64
KV_HALVES_PER_LANE = 8
KV_HALVES_PER_CALL = SEGMENT_THREADS * KV_HALVES_PER_LANE
MAX_VMCNT = (1 << 6) - 1


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
    use_additive_bias: bool = False
    num_seqs: int = 0
    # AMDGPU occupancy hint (``"amdgpu-waves-per-eu"``). Attention is
    # register-pressure-bound; setting this to 2 or 3 tightens the
    # VGPR allocation in exchange for higher occupancy. ``None`` keeps
    # the LLVM heuristic.
    waves_per_eu: Optional[int] = None
    # FP8 K/V cache (mirrors UnifiedAttention2DTiledSpec.kv_storage_dtype).
    # See that spec's docstring for the semantics.
    kv_storage_dtype: Optional[str] = None
    # Multi-page KV tile. ``None`` keeps one KV tile per paged block
    # (T == block_size). A multiple of ``block_size`` makes each loop
    # iteration gather ``T // block_size`` blocks, so every iteration moves
    # more K/V bytes and the loop runs fewer, larger tiles. Unlike gfx942,
    # where the override must divide ``block_size``, gfx950 only accepts
    # multiples.
    tile_size_override: Optional[int] = None
    # Inert on both arches; kept so saved specs that set it still load.
    use_invariant_hoist: bool = False
    # Inert: selects a gfx942 K/V loader. Accepted for signature parity with
    # the gfx942 spec.
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
    # Cache policy of the async K/V copies, one of ``CACHE_POLICY_NAMES`` (as
    # attention 2D's ``kv_cache_policy``). The dispatcher sets "stream" (NT)
    # for all-decode batches, whose K/V bytes are each read once
    # (``_kv_cache_policy_3d``). The fp8 K/V loader is synchronous and keeps
    # "all".
    kv_cache_policy: str = "all"
    # K/V LDS tiles store each row's 16-byte chunks XOR-permuted by the row
    # (chunk q of row r at q ^ (r % (head_size // 8))), which spreads the QK
    # and PV reads over the LDS banks. The DMA lanes fetch permuted source
    # chunks, so the copies stay contiguous and need no padding. Implemented
    # for fp16/bf16 K/V, which can be extended to fp8 K/V.
    use_kv_lds_swizzle: bool = False
    # The wave loads the segment's page-table entries 64 at a time, one per
    # lane, and each tile reads its page ids from those registers
    # (v_readlane), so no page-table load sits on the KV loop's critical
    # path. Paged fp16/bf16 K/V only.
    use_wave_page_id_load: bool = False
    # Top-left causal (unshifted diagonal, ``context_len = 0``); see the 2D spec.
    # At ``q_len == 1`` the single query row attends only key 0 and the cached
    # context is ignored -- correct by definition, not a bug.
    causal_top_left: bool = False
    # Right bound of the attention band; see ``UnifiedAttention2DTiledSpec.right_bound``
    # (0 = causal, R > 0 = R keys of lookahead, -1 = unbounded).
    right_bound: int = 0

    def __post_init__(self):
        if type(self.use_additive_bias) is not bool:
            raise ValueError(
                "use_additive_bias must be a bool, got "
                f"{type(self.use_additive_bias).__name__}"
            )
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
        if self.right_bound < -1:
            raise ValueError(
                f"right_bound must be >= -1 (-1 = unbounded), got {self.right_bound}"
            )
        if self.right_bound != 0 and (self.use_alibi or self.use_qq_bias):
            raise ValueError(
                "right_bound != 0 is incompatible with use_alibi / use_qq_bias "
                "(their offsets are defined on the causal diagonal)"
            )
        if self.kv_layout == "strided" and self.use_additive_bias:
            raise ValueError("strided KV does not support use_additive_bias")
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
        if self.use_kv_lds_swizzle and self.kv_storage_dtype is not None:
            raise ValueError("use_kv_lds_swizzle needs fp16/bf16 KV")
        if self.use_wave_page_id_load and (
            self.kv_layout != "paged" or self.kv_storage_dtype is not None
        ):
            raise ValueError("use_wave_page_id_load needs paged fp16/bf16 KV")
        t = self.tile_size_override
        if t is not None:
            if t < self.block_size or t % self.block_size or t > 128:
                raise ValueError(
                    "tile_size_override must be a multiple of block_size in "
                    f"[block_size, 128] (got {t}, block_size={self.block_size})"
                )
            # Multi-page tiles rely on the async page-id loader; the fp8 KV
            # path loads synchronously (load, dequantize, store to LDS), so it
            # keeps page-sized tiles.
            if t != self.block_size and self.kv_storage_dtype is not None:
                raise ValueError("multi-page tiles require fp16/bf16 KV storage")
            # The loop waits on the async calls of a tile with s_waitcnt
            # vmcnt, which can count at most MAX_VMCNT outstanding calls.
            if (t * self.head_size) // KV_HALVES_PER_CALL > MAX_VMCNT:
                raise ValueError(
                    f"tile_size_override={t} x head_size={self.head_size} needs "
                    "more async calls per tile than s_waitcnt can count"
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
        if self.tile_size_override is not None:
            return self.tile_size_override
        return self.block_size

    @property
    def n_blocks_per_tile(self) -> int:
        return self.tile_size // self.block_size

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16

    @property
    def binary_search_iters(self) -> int:
        return binary_search_seq_idx_iters(self.num_seqs)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_uattn3d_tiled",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"t{self.tile_size}" if self.n_blocks_per_tile != 1 else "",
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
            "addb" if self.use_additive_bias else "",
            "dg" if self.use_decode_grid else "",
            f"kvcp{self.kv_cache_policy}" if self.kv_cache_policy != "all" else "",
            "kvswz" if self.use_kv_lds_swizzle else "",
            "wpid" if self.use_wave_page_id_load else "",
            "stridedkv" if self.kv_layout == "strided" else "",
            "tl" if self.causal_top_left else "",
            (
                ("rbu" if self.right_bound < 0 else f"rb{self.right_bound}")
                if self.right_bound != 0
                else ""
            ),
        )


def supports_tiled_3d(
    *,
    head_size: int,
    block_size: int,
    dtype: str,
    num_queries_per_kv: int,
    use_alibi: bool,
    use_qq_bias: bool,
    use_additive_bias: bool = False,
    use_fp8: bool = False,
    q_dtype=None,
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
    USE_ADDITIVE_BIAS = spec.use_additive_bias
    # 64-bit paged-KV addressing for caches > 2 GiB (see spec docstring).
    I64_KV_ADDR = spec.use_i64_kv_addr
    # FP8 K/V cache: see ``UnifiedAttention2DTiledSpec.kv_storage_dtype``.
    KV_FP8 = spec.kv_storage_dtype == "fp8e4m3"
    KV_CACHE_BITS = cache_policy_from_name(spec.kv_cache_policy)
    KV_SWZ = spec.use_kv_lds_swizzle
    KV_CHUNKS_PER_ROW = HD // 8  # 16-byte chunks per K/V LDS row
    KV_BYTES = 1 if KV_FP8 else 2
    kv_io_dtype = FP8E4M3 if KV_FP8 else dtype

    QK_K_STEP = 32
    QK_K_ITERS = HD // QK_K_STEP
    QK_N_TILES = T // MFMA_N
    PV_N_TILES = HD // MFMA_N

    THREADS = SEGMENT_THREADS

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
    # Additive-bias params are appended only when the feature is on, so the
    # non-bias kernel ABI (and its IR) is unchanged.
    additive_bias = declare_additive_bias_params(b) if USE_ADDITIVE_BIAS else None
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
    if spec.causal_top_left:
        context_len = b.const_i32(0)
    else:
        context_len = b.sub(seq_len, cur_batch_q_len)

    # Right bound of the band; identical to the causal IR when ``RIGHT == 0``.
    RIGHT = spec.right_bound

    def _upper_ok(col_abs, causal_lim):
        if RIGHT < 0:
            return None
        lim = causal_lim if RIGHT == 0 else b.add(causal_lim, b.const_i32(RIGHT))
        return b.cmp_le(col_abs, lim)

    def _band_row_mask(row_ok, upper_ok, in_prefix):
        if upper_ok is None:
            return b.land(row_ok, in_prefix)
        return b.land(b.land(row_ok, upper_ok), in_prefix)

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
    # One V slot: V(i) is loaded after QK(i) and consumed by PV(i) in the same
    # iteration, and the next iteration's leading wait-all retires PV(i) before
    # V(i+1) is issued, so a second slot would never hold a live tile.
    V_lds = b.smem_alloc(dtype, [1, T, HD], name_hint="Vlds")

    # CK Tile ``TransposeLDSLayout<M=16, K=16, B=1>`` lane formulas (the V^T
    # operand of the PV MFMA). The 3D segment kernel is single-warp, so
    # ``lane == tid``.
    pv_tr_reader = TransposeLdsReader(K=16, M=16).bind(b, tid)
    tr_col_lane = pv_tr_reader.col

    def _row_reduce(x: Value, op) -> Value:
        # Combine x over the 4 lanes {l, l^16, l^32, l^48} that hold the 4
        # token groups of one query row.
        sw = b.bitcast(b.ds_swizzle_xor(b.bitcast(x, I32), 16), F32)
        x = op(x, sw)
        lo, hi = b.permlane32_swap(b.bitcast(x, I32), b.bitcast(x, I32))
        return op(b.bitcast(lo, F32), b.bitcast(hi, F32))

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
    if RIGHT < 0:
        max_seq_prefix_len = seq_len  # no upper bound: every key of the sequence
    else:
        msp_raw = b.add(
            b.add(context_len, qb_start_pos), b.const_i32(bm1_div_nqk + 1 + RIGHT)
        )
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

    # ---------------- async K/V infra (identical to 2D) ----------------
    big_bytes = b.const_i32(0x7FFF0000)
    if spec.kv_layout == "strided":
        key_rsrc = strided_kv_resource(b, key, k_strides, seq_idx, kv_head_idx)
        value_rsrc = strided_kv_resource(b, value, v_strides, seq_idx, kv_head_idx)
    else:
        key_rsrc = b.buffer_rsrc(key, big_bytes)
        value_rsrc = b.buffer_rsrc(value, big_bytes)

    KV_HALVES_PER_CALL = THREADS * KV_HALVES_PER_LANE
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
    N_BLOCKS_PER_TILE = spec.n_blocks_per_tile
    seq_base = b.mul(seq_idx, bt_stride_p) if spec.kv_layout == "paged" else None
    # fp16/bf16 tiles look up each page id once per tile, as a wave-uniform
    # value. K and V of a tile share the ids, and the loop fetches the next
    # tile's ids one iteration ahead (the loop-carried ``pid*`` values), so the
    # vmcnt wait on a table read doesn't drain the K/V copies already in
    # flight. The page base goes into soffset; the per-lane part of the address
    # is loop-invariant. The fp8 sync loader (page-sized tiles only) keeps the
    # paged-KV descriptor; strided KV has no page table and keeps per-call
    # strided offsets.
    PAGE_IDS = not KV_FP8 and spec.kv_layout == "paged"
    WAVE_PAGE_ID_LOAD = spec.use_wave_page_id_load
    if not PAGE_IDS:
        paged_kv_desc = TensorDescriptor.naive(
            "paged_kv_bytes",
            lengths=[1 << 24, T, NUM_KV, HD],
            strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
            coord_names=("physical_block", "token", "kv_head", "dim"),
        ).transform(
            indirect(
                "tile_idx", into="physical_block", table=block_tables, base=seq_base
            ),
            unmerge("linear_half", into=("token", "dim"), dims=(T, HD)),
        )
    else:
        if N_BLOCKS_PER_TILE > 1 or WAVE_PAGE_ID_LOAD:
            # A sequence's last tile can reach past its last page; the table
            # entries there are not the sequence's (uninitialized, or the next
            # row). Those pages reuse the tile's first page id, always a live
            # page of the sequence; the in_prefix mask discards their tokens.
            seq_pages = b.div(b.add(seq_len, b.const_i32(BS - 1)), b.const_i32(BS))
        tokens_per_call = KV_HALVES_PER_CALL // HD
        assert (
            BS % tokens_per_call == 0
        ), f"page-id loader needs whole calls per page (BS={BS}, HD={HD})"
        calls_per_block = BS // tokens_per_call
        lane_tok = b.div(lane_half_base, b.const_i32(HD))
        lane_dim = b.mod(lane_half_base, b.const_i32(HD))
        lane_tok_head = b.add(
            b.mul(lane_tok, b.const_i32(kv_stride_tok_b)),
            b.mul(kv_head_idx, b.const_i32(kv_stride_h_b)),
        )
        # Per-lane byte offset within a page, keyed by the call's swizzle
        # phase (always 0 without the swizzle).
        lane_within_b = {}
        if KV_SWZ:
            # A call's LDS rows are ph + lane_tok (mod chunks per row), with
            # ph = its first row; ph is a multiple of tokens_per_call and
            # lane_tok < tokens_per_call, so the sum never wraps. The lane
            # writing LDS chunk c fetches chunk c ^ row.
            lane_chunk = b.div(lane_dim, b.const_i32(8))
            for call in range(kv_calls_per_tile):
                ph = (call * tokens_per_call) % KV_CHUNKS_PER_ROW
                if ph in lane_within_b:
                    continue
                src_chunk = b.xor(lane_chunk, b.add(lane_tok, b.const_i32(ph)))
                lane_within_b[ph] = b.add(
                    lane_tok_head, b.mul(src_chunk, b.const_i32(8 * KV_BYTES))
                )
        else:
            lane_within_b[0] = b.add(
                lane_tok_head, b.mul(lane_dim, b.const_i32(KV_BYTES))
            )

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

    def _tile_page_ids(kv_tile_idx: Value) -> list:
        # A tile starts below seq_len, so its first page is always live.
        first_blk = b.mul(kv_tile_idx, b.const_i32(N_BLOCKS_PER_TILE))
        first_pid = b.global_load_i32(block_tables, b.add(seq_base, first_blk))
        ids = [b.to_sgpr_u32(first_pid)]
        for j in range(1, N_BLOCKS_PER_TILE):
            blk = b.add(first_blk, b.const_i32(j))
            pid = b.masked_global_load(
                block_tables,
                b.add(seq_base, blk),
                b.cmp_lt(blk, seq_pages),
                first_pid,
                dtype=I32,
                align=4,
            )
            ids.append(b.to_sgpr_u32(pid))
        return ids

    def _issue_paged_tile(
        src: Value, src_rsrc: Value, lds_base: Value, buf_idx: Value, page_ids
    ) -> None:
        buf_off_i64 = b.zext(b.mul(buf_idx, b.const_i32(bytes_per_buf)), I64)
        buf_base = b.smem_ptr_add(lds_base, buf_off_i64)
        for call in range(kv_calls_per_tile):
            blk = call // calls_per_block
            tok0 = (call % calls_per_block) * tokens_per_call
            ph = (call * tokens_per_call) % KV_CHUNKS_PER_ROW if KV_SWZ else 0
            voff = b.add(lane_within_b[ph], b.const_i32(tok0 * kv_stride_tok_b))
            if I64_KV_ADDR:
                base_i64 = b.mul(
                    b.zext(page_ids[blk], I64), b.const_i64(kv_stride_blk_b)
                )
                call_rsrc = b.buffer_rsrc(
                    b.global_ptr_add(src, base_i64), kv_block_bytes_c
                )
                soff = zero_soff
            else:
                call_rsrc = src_rsrc
                soff = b.mul(page_ids[blk], b.const_i32(kv_stride_blk_b))
            dst = b.smem_ptr_add(buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(
                call_rsrc, dst, voff, soff, 4, coherency=KV_CACHE_BITS
            )

    def _issue_strided_tile(
        src_rsrc: Value, strides, lds_base: Value, kv_tile_idx: Value, buf_idx: Value
    ) -> None:
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        buf_base = b.smem_ptr_add(lds_base, buf_off_i64)
        for call in range(kv_calls_per_tile):
            linear_half = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            voff = strided_kv_offset(
                b, strides, kv_tile_idx, _kv_src_linear(linear_half), seq_len, HD, T
            )
            dst = b.smem_ptr_add(buf_base, b.const_i64(call * bytes_per_call))
            b.async_buffer_load_lds_addr(
                src_rsrc, dst, voff, zero_soff, 4, coherency=KV_CACHE_BITS
            )

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

    def _issue_k(tile_idx: Value, buf_idx: Value, page_ids=None) -> None:
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "K")
        elif spec.kv_layout == "strided":
            _issue_strided_tile(key_rsrc, k_strides, K_lds_addr, tile_idx, buf_idx)
        else:
            _issue_paged_tile(key, key_rsrc, K_lds_addr, buf_idx, page_ids)

    def _issue_v(tile_idx: Value, buf_idx: Value, page_ids=None) -> None:
        if KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "V")
        elif spec.kv_layout == "strided":
            _issue_strided_tile(value_rsrc, v_strides, V_lds_addr, tile_idx, buf_idx)
        else:
            _issue_paged_tile(value, value_rsrc, V_lds_addr, buf_idx, page_ids)

    first_page_ids = None
    if WAVE_PAGE_ID_LOAD:
        # The wave holds a 64-page window of the segment's page table, one id
        # per lane; a tile never straddles two windows.
        assert 64 % N_BLOCKS_PER_TILE == 0
        seg_first_page = b.mul(tile_start, b.const_i32(N_BLOCKS_PER_TILE))

        def _load_page_id_window(rel_page: Value) -> Value:
            # Lane l: the id of the segment's page rel_page + l (0 past the
            # sequence's last page: never used, see _window_tile_ids).
            pg = b.add(b.add(seg_first_page, rel_page), tid)
            return b.masked_global_load(
                block_tables,
                b.add(seq_base, pg),
                b.cmp_lt(pg, seq_pages),
                b.const_i32(0),
                dtype=I32,
                align=4,
            )

        def _window_tile_ids(page_id_window: Value, kv_tile_idx: Value) -> list:
            first_blk = b.mul(kv_tile_idx, b.const_i32(N_BLOCKS_PER_TILE))
            lane0 = b.mod(b.sub(first_blk, seg_first_page), b.const_i32(64))
            id0 = b.readlane(page_id_window, lane0)
            ids = [id0]
            for j in range(1, N_BLOCKS_PER_TILE):
                idj = b.readlane(page_id_window, b.add(lane0, b.const_i32(j)))
                live = b.cmp_lt(b.add(first_blk, b.const_i32(j)), seq_pages)
                ids.append(b.select(live, idj, id0))
            return ids

        def _refill_page_id_window(page_id_window: Value, kv_tile_idx: Value) -> Value:
            # The window for the next iteration: if tile kv_tile_idx + 2 starts
            # a new window, load it now (after this tile's copies, so the
            # explicit waits still cover V(t)); the loop-top wait completes it
            # before the next iteration reads it.
            t2 = b.add(kv_tile_idx, b.const_i32(2))
            rel2 = b.sub(b.mul(t2, b.const_i32(N_BLOCKS_PER_TILE)), seg_first_page)
            cross = b.land(
                b.cmp_eq(b.mod(rel2, b.const_i32(64)), b.const_i32(0)),
                b.cmp_lt(t2, tile_end),
            )
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
        first_page_ids = _window_tile_ids(first_page_id_window, tile_start)
    elif PAGE_IDS:
        first_page_ids = _tile_page_ids(tile_start)
    _issue_k(tile_start, b.const_i32(0), first_page_ids)

    cur_buf_init = b.const_i32(0)
    iter_args.append(("cur_buf", cur_buf_init))
    if PAGE_IDS:
        for j, pid in enumerate(first_page_ids):
            iter_args.append((f"pid{j}", pid))
    if WAVE_PAGE_ID_LOAD:
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
        q_col_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
        Q_kits.append(b.smem_load_vN(Q_lds, lane_col, q_col_off, dtype=dtype, n=8))

    kvloop = b.scf_for_iter(
        tile_start, tile_end, b.const_i32(1), iter_args, iv_name="kv_tile"
    )
    with kvloop as (kv_tile_iv, carry):
        m_val = carry[0]
        l_val = carry[1]
        acc_vals = [carry[2 + n] for n in range(PV_N_TILES)]
        cur_buf = carry[2 + PV_N_TILES]
        nxt_buf = b.sub(b.const_i32(1), cur_buf)
        v_buf = b.const_i32(0)
        cur_page_ids = (
            [carry[3 + PV_N_TILES + j] for j in range(N_BLOCKS_PER_TILE)]
            if PAGE_IDS
            else None
        )
        tile_off = b.mul(kv_tile_iv, b.const_i32(T))

        next_tile_iv_raw = b.add(kv_tile_iv, b.const_i32(1))
        in_range_next = b.cmp_lt(next_tile_iv_raw, tile_end)
        safe_next_tile = b.select(in_range_next, next_tile_iv_raw, kv_tile_iv)

        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.sync()
        # Page ids of the next tile, for its K copies below.
        nxt_page_ids = None
        if WAVE_PAGE_ID_LOAD:
            # Read from the window in registers: no load in the loop.
            page_id_window = carry[3 + PV_N_TILES + N_BLOCKS_PER_TILE]
            nxt_page_ids = _window_tile_ids(page_id_window, safe_next_tile)
        elif PAGE_IDS:
            # Loaded here, before this tile's V copies, so the load overlaps
            # the QK below; the next K copies still wait for it.
            nxt_page_ids = _tile_page_ids(safe_next_tile)

        # QK
        S_n = []
        for n in range(QK_N_TILES):
            acc_v = b.zero_vec_f32(4)
            for k in range(QK_K_ITERS):
                kc_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                k_row = b.add(b.const_i32(n * 16), lane_col)
                K_v = b.smem_load_vN(
                    K_lds, cur_buf, k_row, _kv_col(k_row, kc_off), dtype=dtype, n=8
                )
                acc_v = _mfma_16x16x32(b, dtype, K_v, Q_kits[k], acc_v)
            S_n.append(acc_v)

        _issue_v(kv_tile_iv, v_buf, cur_page_ids)
        _issue_k(safe_next_tile, nxt_buf, nxt_page_ids)
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
                causal_ok = _upper_ok(col_abs, causal_lim)
                in_prefix = b.cmp_lt(col_abs, max_seq_prefix_len)
                m_ok = _band_row_mask(row_ok, causal_ok, in_prefix)
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
                if USE_ADDITIVE_BIAS:
                    # Per-row query head (GQA-correct). Under ``m_ok``, padding
                    # rows and tail columns (col_abs >= seq_len) load element 0
                    # instead of an out-of-range index.
                    s_scaled = add_additive_bias(
                        b,
                        s_scaled,
                        additive_bias,
                        seq_idx=seq_idx,
                        q_head=qh_r,
                        q_row=qp_r,
                        col=col_abs,
                        mask=m_ok,
                        rcp_ln2=rcp_ln2,
                    )
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

        # O^T = V^T P^T: A = V^T (head-dim rows n*16 + 0..15 via the transpose
        # read), B = P^T straight from this lane's registers (its query row,
        # tokens 4 * lane_rg + 0..3 of each 16-token block). Two blocks share one
        # 16x16x32 MFMA, which costs the same as one 16x16x16 on gfx950: A and B
        # list the two blocks' tokens in the same order, which is all the k-sum
        # needs. An odd last block uses 16x16x16.
        new_acc = []
        for n in range(PV_N_TILES):
            acc_v = b.vec_pack(
                [b.fmul(b.vec_extract(acc_vals[n], r), alpha) for r in range(4)], F32
            )
            n_col_base = b.add(b.mul(b.const_i32(n), b.const_i32(16)), tr_col_lane)
            for kb in range(0, QK_N_TILES, 2):
                blocks = range(kb, min(kb + 2, QK_N_TILES))
                A_parts = []
                for blk in blocks:
                    row_lane = pv_tr_reader.row(b, k_offset=blk * 16, read=0)
                    A_parts.append(
                        b.ds_read_tr16_b64(
                            V_lds,
                            v_buf,
                            row_lane,
                            _kv_col(row_lane, n_col_base),
                            dtype=dtype,
                        )
                    )
                B_p = b.vec_pack(
                    [
                        b.cast_f32_to(p_regs[(blk, r)], dtype)
                        for blk in blocks
                        for r in range(4)
                    ],
                    dtype,
                )
                if len(A_parts) == 2:
                    acc_v = _mfma_16x16x32(b, dtype, b.vec_concat(*A_parts), B_p, acc_v)
                else:
                    acc_v = _mfma_16x16x16(b, dtype, A_parts[0], B_p, acc_v)
            new_acc.append(acc_v)

        yields = [m_new, new_l_val]
        for n in range(PV_N_TILES):
            yields.append(new_acc[n])
        yields.append(nxt_buf)
        if PAGE_IDS:
            yields.extend(nxt_page_ids)
        if WAVE_PAGE_ID_LOAD:
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
        return kernel_name_join("rocke_uattn_reduce_tiled", *parts)


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

      1. overall_max = max(segm_max) over segments with segm_expsum > 0 (a
         segment masked for the row stores max=0, sum=0 and is treated as -inf)
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
        # NaN-safe factor: when both ms and overall_max are -inf, the
        # difference is NaN; force factor to 0 in that case.
        ms_finite = b.fcmp("ogt", ms, neg_inf)
        factor_raw = b.exp2_fast(b.fsub(ms, overall_max))
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
    spec: UnifiedAttention3DTiledSpec, *, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Spec-shaped wrapper for ``supports_tiled_3d``.

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
