# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 WMMA split-KV 3D decode attention (segment + reduce).

The gfx1250 wave32 WMMA counterpart of the gfx950 tiled-3D split-KV
unified-attention path. It targets the decode regime (small ``q_len``, long KV)
where the 2D grid is too small to fill the device: the KV sequence is sliced
into ``num_segments`` segments, each segment CTA computes a partial online
softmax over its KV slice, and a second ``reduce`` kernel merges the per-segment
``(m, l, acc)`` partials into the final normalized output.

It reuses the validated gfx1250 2D WMMA QK / softmax / P*V structure
(``attention_tiled_2d.py``) and changes only what is structural to the 3D
variant:

  * 19-param segment ABI matching the common dispatcher's ``_3d_signature``;
    outputs go to ``segm_output / segm_max / segm_expsum`` (fp32) workspaces.
  * grid is ``(total_num_q_blocks, num_kv_heads, num_segments)``;
  * the KV tile range is bounded to the segment; empty/past segments write a
    neutral ``(m=-inf, l=0, acc=0)`` so the reduce sees every segment;
  * partials are stored UN-normalized; ``acc/L`` + output cast happen in reduce.

Block size 16 and 32 are both supported: a tile is a fixed 32-token window
(two WMMA N-subtiles of 16), which spans one paged block when ``block_size==32``
and two consecutive paged blocks when ``block_size==16`` — the per-N-subtile and
per-token block-table lookups handle both. KV cache may be bf16 (no dequant) or
fp8e4m3 (in-register dequant). bf8e5m2 is gated off pending its shared helper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from ...core.ir import BF16, F32, I32, IRBuilder, KernelDef, PtrType, Type, Value
from ...helpers.attention import (
    PagedKvDescriptor,
    binary_search_seq_idx,
    wave_reduce_max,
    wave_reduce_sum,
)
from ._wmma_attention_common import (
    BLOCK_M as _BLOCK_M,
    HEAD_SIZE as _HEAD_SIZE,
    WAVE as _WAVE,
    WMMA_N as _WMMA_N,
    check_wmma_arch,
    compute_pv,
    compute_pv_dstr,
    compute_pv_from_probs,
    compute_pv_wide,
    compute_qk_scores,
    kv_storage_ir as _kv_storage_ir,
    load_q_frags,
    resolve_wmma,
    softmax_row_update,
    stage_v_tile,
    stage_v_tile_buf,
    stage_v_tile_transposed,
)

__all__ = [
    "UnifiedAttention3DTiledSpec",
    "UnifiedAttentionReduceTiledSpec",
    "build_unified_attention_3d_tiled",
    "build_unified_attention_reduce_tiled",
    "supports_tiled_3d",
]

_T = 32  # tokens per KV tile (two WMMA N-subtiles); spans 1 (bs32) or 2 (bs16) blocks


@dataclass(frozen=True)
class UnifiedAttention3DTiledSpec:
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
    # accepted for signature parity with the shared dispatch spec builder
    tile_size_override: Optional[int] = None
    use_invariant_hoist: bool = False
    use_wide_kv_load: bool = False
    use_register_p: bool = False
    wmma_spacing: int = 0
    # Cooperative multi-wave32 CTA: ``num_waves`` waves split each segment's KV
    # tiles (strided) and merge their partial online-softmax state through LDS
    # (inter-wave reduction) before the single workspace write. ``1`` keeps the
    # validated single-wave path byte-identical.
    num_waves: int = 1
    # --- gap-closing levers (ISA-driven; see gfx1250_mha_optimization_case_study) ---
    # lever 1 (default on): transposed V_lds + wide ds_load reads, replacing the
    #    64 scalar ds_load_u16 PV gathers/tile with wide loads.
    use_wide_lds_reads: bool = True
    # lever 2 (opt-in): DTLA (buffer_load...lds) staging + double-buffer + prefetch.
    use_dtla_prefetch: bool = False
    # hardware transpose-LDS read (ds_load_tr16_b128) for the PV V operand:
    # V staged token-major, read transposed in HW (the gfx950 ds_read_tr
    # equivalent), replacing the 64 scalar ds_load_u16 gathers/tile.
    use_ds_tr_reads: bool = False
    # software-pipeline stack (small-batch latency lever): drives the DTLA
    # async-V double-buffer "shadow" + instruction-scheduling cadence
    # (iglp_opt / sched_group_barrier) so the per-tile serial ALU/LDS
    # dependency chains (s_delay_alu) overlap the in-flight next-tile load.
    # The case studies (skinny-gemm, fused-MoE, deep-conv) show these levers
    # only pay off TOGETHER, hence a single flag. bf16 KV only (uses DTLA).
    use_sw_pipeline: bool = False
    # DEBUG/perf-only: replace the online-softmax wave reductions + exp2 with
    # constants (output is garbage). Used to ablate the softmax critical-path
    # cost vs the rest of the per-tile pipeline. Never ship enabled.
    ablate_softmax: bool = False
    # DEBUG/perf-only: skip V staging (load + fp8 dequant) and the PV-GEMM
    # entirely (output is garbage). The P_lds store + alpha-scaled accs keep the
    # QK+softmax chain live, so (full - ablate_pv) is the PV+V-staging ceiling --
    # the absolute upper bound a native-fp8 PV path could ever recover. Plain
    # (non-DTLA, P-in-LDS) path only. Never ship enabled.
    ablate_pv: bool = False
    # lever 3 (opt-in): fused single-kernel reduce (write final output, no fp32
    #    partials workspace / second launch). Only valid when one CTA covers the
    #    whole KV for a (token, head) (num_segments == 1).
    use_fused_reduce: bool = False
    # online-softmax cross-lane reduction via DPP ``row_xmask`` (VALU) instead
    # of ``ds_swizzle`` (LDS port). gfx1250 (RDNA) only: the wave32 16-lane
    # row-group butterfly moves off the serialized LDS port onto the VALU,
    # cutting the softmax critical-path ``lgkmcnt`` stalls. Emitted as the fused
    # ``v_max_num_f32_dpp`` / ``v_add_f32_dpp`` (one VALU op per stage). Measured
    # 1.4-1.56x seg speedup at low wave count (softmax on the critical path) and
    # ~2-3% at the high-wave shipping configs; exact vs the ds_swizzle baseline.
    # Forces ``wmma_spacing>=1`` (see __post_init__) to re-supply the dependent-WMMA
    # hazard gap ds_swizzle gave for free. Default on (pure win, never slower).
    use_dpp_softmax: bool = True

    def __post_init__(self) -> None:
        ok, why = supports_tiled_3d(
            head_size=self.head_size,
            block_size=self.block_size,
            dtype=self.dtype,
            num_queries_per_kv=self.num_queries_per_kv,
            use_alibi=self.use_alibi,
            use_qq_bias=self.use_qq_bias,
            use_fp8=self.kv_storage_dtype == "fp8e4m3",
            q_dtype=None,
            kv_storage_dtype=self.kv_storage_dtype,
            arch="gfx1250",
        )
        if not ok:
            raise ValueError(why)
        if self.has_softcap:
            raise ValueError("gfx1250 tiled 3D does not support softcap yet")
        if self.num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        if self.wmma_spacing < 0:
            raise ValueError("wmma_spacing must be non-negative")
        if self.num_waves not in (1, 2, 4, 8):
            raise ValueError("num_waves must be one of {1,2,4,8}")
        if self.num_waves > 1 and (self.use_register_p or self.use_wide_kv_load):
            raise ValueError(
                "num_waves>1 uses the LDS-P single-buffer path "
                "(use_register_p / use_wide_kv_load must be False)"
            )
        if self.use_wide_lds_reads and self.use_register_p:
            raise ValueError(
                "use_wide_lds_reads needs P in LDS (set use_register_p=False)"
            )
        if self.use_wide_lds_reads and self.use_wide_kv_load:
            raise ValueError(
                "use_wide_lds_reads (transposed single-buffer V) and "
                "use_wide_kv_load (double-buffer [2,T,HD]) are mutually exclusive; "
                "the prefetched+transposed combine lands with use_dtla_prefetch"
            )
        if self.use_wide_lds_reads and self.num_waves > 1:
            raise ValueError("use_wide_lds_reads not yet wired for num_waves>1")
        if self.use_dtla_prefetch:
            # DTLA stages V async into a token-major double buffer ([2,T,HD]);
            # it owns the V_lds layout + the prefetch/wait pipeline, so it is
            # mutually exclusive with the other V-staging levers. The combined
            # wide-read (lever 1) + DTLA (lever 2) stack needs a transpose-on-read and is
            # handled separately (case study section 7, combined run).
            if self.num_waves > 1:
                raise ValueError(
                    "use_dtla_prefetch requires single-wave (num_waves==1)"
                )
            if self.use_register_p:
                raise ValueError(
                    "use_dtla_prefetch needs P in LDS (use_register_p=False)"
                )
            if self.use_wide_lds_reads:
                raise ValueError(
                    "use_dtla_prefetch (token-major V) and use_wide_lds_reads "
                    "(dim-major V) are mutually exclusive; the combined stack "
                    "lands with a transpose-on-read"
                )
            if self.use_wide_kv_load:
                raise ValueError(
                    "use_dtla_prefetch and use_wide_kv_load both double-buffer V; "
                    "use only one"
                )
            if self.kv_storage_dtype == "fp8e4m3":
                raise ValueError(
                    "use_dtla_prefetch fp8 (raw-in-LDS + dequant-on-read) is not "
                    "yet implemented; bf16 KV only for now"
                )
            if self.block_size not in (16, 32):
                raise ValueError("use_dtla_prefetch supports block_size in {16,32}")
        if self.use_ds_tr_reads:
            if self.use_register_p:
                raise ValueError(
                    "use_ds_tr_reads needs P in LDS (use_register_p=False)"
                )
            if self.use_wide_lds_reads:
                raise ValueError(
                    "use_ds_tr_reads (HW transpose read of token-major V) and "
                    "use_wide_lds_reads (dim-major V) are mutually exclusive"
                )
            if self.use_wide_kv_load:
                raise ValueError(
                    "use_ds_tr_reads uses token-major V; disable use_wide_kv_load"
                )
            # use_dtla_prefetch + use_ds_tr_reads IS the combined latency-hiding
            # stack: DTLA streams V async into a token-major double buffer (wide
            # global read) and ds_load_tr does the transpose-on-read (wide LDS
            # read). Both consume token-major V, so they compose -- the dstr path
            # just takes the DTLA buffer index. (Allowed; the DTLA branch wires it.)
            if self.num_waves > 1:
                raise ValueError("use_ds_tr_reads not yet wired for num_waves>1")
        if self.use_sw_pipeline:
            # sw-pipeline rides the DTLA async-V staging path.
            if self.use_dtla_prefetch:
                raise ValueError(
                    "use_sw_pipeline already implies the DTLA staging path"
                )
            if (
                self.num_waves > 1
                or self.use_register_p
                or self.use_wide_lds_reads
                or self.use_wide_kv_load
                or self.use_ds_tr_reads
            ):
                raise ValueError(
                    "use_sw_pipeline owns the V path (single-wave, P in LDS, no "
                    "wlds/wkv/ds-tr/register-P)"
                )
            if self.kv_storage_dtype == "fp8e4m3":
                raise ValueError("use_sw_pipeline is bf16 KV only (rides DTLA)")
            if self.block_size not in (16, 32):
                raise ValueError("use_sw_pipeline supports block_size in {16,32}")
        if self.use_fused_reduce and self.num_segments != 1:
            raise ValueError(
                "use_fused_reduce requires num_segments==1 (one CTA per token/head)"
            )
        if self.ablate_pv:
            # perf-only ceiling probe: needs the plain non-DTLA P-in-LDS path so
            # the P_lds store keeps QK+softmax live after the PV-GEMM is dropped.
            if self.use_register_p or self.use_dtla_prefetch or self.use_sw_pipeline:
                raise ValueError(
                    "ablate_pv requires the plain P-in-LDS path "
                    "(no register_p / dtla / sw_pipeline)"
                )
        if self.use_dpp_softmax and self.wmma_spacing < 1:
            # The DPP row_xmask reduction runs on the VALU and removes the
            # ds_swizzle LDS serialization that incidentally supplied the
            # gfx1250 dependent-WMMA hazard gap. Without it, high-occupancy
            # configs (bf16 KV + multi-wave) produce a garbage accumulator
            # (NaN, GPU-validated). Re-supply the gap with a single v_nop.
            object.__setattr__(self, "wmma_spacing", 1)

    @property
    def num_queries_per_kv(self) -> int:
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        return self.num_query_heads // self.num_kv_heads

    @property
    def block_q(self) -> int:
        return _BLOCK_M // self.num_queries_per_kv

    @property
    def dtype_ir(self) -> Type:
        return BF16

    @property
    def binary_search_iters(self) -> int:
        if self.num_seqs <= 0:
            return 32
        # Match attention_unified scalar oracle (32 iters): avoids rare
        # mis-resolution when ``ceil(log2(n+1))`` is tight for large batches.
        return max(32, int(math.ceil(math.log2(self.num_seqs + 1))))

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_uattn3d_seg_gfx1250",
            "wmma16x16x32",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            self.dtype,
            f"kv{self.kv_storage_dtype}" if self.kv_storage_dtype else "kvbf16",
            f"seg{self.num_segments}",
            "sinks" if self.use_sinks else "",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "hoist" if self.use_invariant_hoist else "",
            "wkvb" if self.use_wide_kv_load else "",
            "regp" if self.use_register_p else "ldsP",
            f"nop{self.wmma_spacing}" if self.wmma_spacing else "",
            f"mw{self.num_waves}" if self.num_waves > 1 else "",
            "wlds" if self.use_wide_lds_reads else "",
            "dstr" if self.use_ds_tr_reads else "",
            "dtla" if self.use_dtla_prefetch else "",
            "swp" if self.use_sw_pipeline else "",
            "fred" if self.use_fused_reduce else "",
        )


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
        return BF16

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_uattn3d_reduce_gfx1250",
            f"d{self.head_size}",
            f"h{self.num_query_heads}",
            self.dtype,
            f"seg{self.num_segments}",
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
    arch: str = "gfx1250",
    **_ignored,
) -> Tuple[bool, str]:
    if arch != "gfx1250":
        return False, f"gfx1250 tiled 3D only supports arch='gfx1250' (got {arch!r})"
    if dtype != "bf16":
        return (
            False,
            f"gfx1250 tiled 3D currently supports bf16 Q/O only (got {dtype!r})",
        )
    if head_size != _HEAD_SIZE:
        return (
            False,
            f"gfx1250 tiled 3D currently supports head_size=64 (got {head_size})",
        )
    if block_size not in (16, 32):
        return (
            False,
            f"gfx1250 tiled 3D supports block_size in {{16,32}} (got {block_size})",
        )
    if num_queries_per_kv not in (8,):
        return False, (
            "gfx1250 tiled 3D currently supports GQA-8 "
            f"(got num_queries_per_kv={num_queries_per_kv})"
        )
    if use_alibi:
        return False, "gfx1250 tiled 3D does not support ALiBi yet"
    if use_qq_bias:
        return False, "gfx1250 tiled 3D does not support QQ bias yet"
    if q_dtype is not None and q_dtype != "bf16":
        return False, f"gfx1250 tiled 3D: unsupported q_dtype {q_dtype!r}"
    if kv_storage_dtype not in (None, "bf16", "fp8e4m3"):
        return False, (
            "gfx1250 tiled 3D supports bf16/fp8e4m3 KV cache "
            f"(got {kv_storage_dtype!r}; bf8e5m2 pending shared helper)"
        )
    if use_fp8 and kv_storage_dtype != "fp8e4m3":
        return False, "use_fp8 requires kv_storage_dtype='fp8e4m3'"

    ok, why = check_wmma_arch(arch)
    if not ok:
        return False, why
    return True, "supported by gfx1250 WMMA tiled 3D v1"


def _seg_declare_params(b: IRBuilder, kv_dtype: Type):
    segm_output = b.param(
        "segm_output_ptr",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    segm_max = b.param(
        "segm_max_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    segm_expsum = b.param(
        "segm_expsum_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    query = b.param(
        "query_ptr", PtrType(BF16, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr",
        PtrType(kv_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(kv_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sinks = b.param("sink_ptr", PtrType(BF16, "global"), readonly=True, align=16)
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
    scale = b.param("scale", F32)
    k_scale = b.param("k_scale", F32)
    v_scale = b.param("v_scale", F32)
    softcap = b.param("softcap", F32)
    num_seqs = b.param("num_seqs", I32)
    block_table_stride = b.param("block_table_stride", I32)
    qq_bias_stride_0 = b.param("qq_bias_stride_0", I32)
    return locals()


def build_unified_attention_3d_tiled(
    spec: UnifiedAttention3DTiledSpec, arch: str = "gfx1250"
) -> KernelDef:
    """Build the gfx1250 WMMA split-KV 3D decode *segment* kernel."""
    ok, why = supports_tiled_3d(
        head_size=spec.head_size,
        block_size=spec.block_size,
        dtype=spec.dtype,
        num_queries_per_kv=spec.num_queries_per_kv,
        use_alibi=spec.use_alibi,
        use_qq_bias=spec.use_qq_bias,
        use_fp8=spec.kv_storage_dtype == "fp8e4m3",
        q_dtype=None,
        kv_storage_dtype=spec.kv_storage_dtype,
        arch=arch,
    )
    if not ok:
        raise ValueError(why)

    op, a_map, c_map, a_frag, c_frag = resolve_wmma(arch)

    dtype = spec.dtype_ir
    kv_dtype = _kv_storage_ir(spec.kv_storage_dtype)
    HD = spec.head_size
    BS = spec.block_size
    NQK = spec.num_queries_per_kv
    NUM_QH = spec.num_query_heads
    NUM_SEG = spec.num_segments
    SLIDING_WINDOW = spec.sliding_window
    BLOCK_Q = spec.block_q
    NUM_WAVES = spec.num_waves
    # sw-pipeline rides the DTLA async-V double-buffer staging path.
    use_dtla_stage = spec.use_dtla_prefetch or spec.use_sw_pipeline

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _WAVE * NUM_WAVES
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    p = _seg_declare_params(b, kv_dtype)

    segm_output = p["segm_output"]
    segm_max = p["segm_max"]
    segm_expsum = p["segm_expsum"]
    query = p["query"]
    key = p["key"]
    value = p["value"]
    sinks = p["sinks"]
    block_tables = p["block_tables"]
    seq_lens = p["seq_lens"]
    cu_q = p["cu_q"]
    scale = p["scale"]
    k_scale = p["k_scale"]
    v_scale = p["v_scale"]
    num_seqs = p["num_seqs"]
    block_table_stride = p["block_table_stride"]

    q_block_global_idx = b.block_id_x()
    kv_head_idx = b.block_id_y()
    seg_idx = b.block_id_z()
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(_WAVE))
    wave_id = b.div(tid, b.const_i32(_WAVE))

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)
    one_f = b.const_f32(1.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)
    qk_scale = b.fmul(scale, rcp_ln2)

    seq_idx = binary_search_seq_idx(
        b,
        cu_q,
        q_block_global_idx,
        num_seqs,
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

    # Whole q-block past this sequence's queries -> padding block, no real
    # output token; nothing to write (reduce never reads padding tokens).
    with b.scf_if(b.cmp_ge(qb_start_pos, cur_batch_q_len)):
        b.ret()

    lane_row = b.mod(lane, b.const_i32(16))
    half_k = a_map.coord(b, lane, 0)[1]
    col = b.mod(lane, b.const_i32(16))

    bm1_div_nqk = (_BLOCK_M - 1) // NQK
    msp_raw = b.add(b.add(context_len, qb_start_pos), b.const_i32(bm1_div_nqk + 1))
    max_seq_prefix_len = b.select(b.cmp_lt(msp_raw, seq_len), msp_raw, seq_len)
    num_tiles = b.div(b.add(max_seq_prefix_len, b.const_i32(_T - 1)), b.const_i32(_T))
    # tiles per segment = ceil(seq_len / (NUM_SEG * T))
    tps = b.div(
        b.add(seq_len, b.const_i32(NUM_SEG * _T - 1)), b.const_i32(NUM_SEG * _T)
    )
    tile_start = b.mul(seg_idx, tps)
    te_raw = b.mul(b.add(seg_idx, b.const_i32(1)), tps)
    tile_end = b.select(b.cmp_lt(te_raw, num_tiles), te_raw, num_tiles)
    if SLIDING_WINDOW > 0:
        # Clamp the segment's first tile up to the earliest tile any row in this
        # q-block can still attend to under the sliding window. Must happen
        # before the empty-segment check so fully-masked segments write neutral.
        first_allowed = b.add(
            b.sub(b.add(context_len, qb_start_pos), b.const_i32(SLIDING_WINDOW)),
            b.const_i32(1),
        )
        sw_tile_start = b.div(first_allowed, b.const_i32(_T))
        ts2 = b.select(
            b.cmp_lt(sw_tile_start, b.const_i32(0)), b.const_i32(0), sw_tile_start
        )
        tile_start = b.select(b.cmp_lt(tile_start, ts2), ts2, tile_start)

    def _write_partials(ms, ls, accs):
        for r in range(c_frag):
            row_rel, col_n = c_map.coord(b, lane, r)
            q_pos = b.add(qb_start_pos, b.div(row_rel, b.const_i32(NQK)))
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row_rel, b.const_i32(NQK))
            )
            row_valid = b.land(
                b.cmp_lt(q_pos, cur_batch_q_len), b.cmp_lt(qh, b.const_i32(NUM_QH))
            )
            out_token = b.add(cu_q_start, q_pos)
            ml_idx = b.add(
                b.mul(
                    b.add(b.mul(out_token, b.const_i32(NUM_QH)), qh),
                    b.const_i32(NUM_SEG),
                ),
                seg_idx,
            )
            so_base = b.mul(ml_idx, b.const_i32(HD))
            # m/l written once per row by its col-0 lane.
            with b.scf_if(b.land(row_valid, b.cmp_eq(col_n, b.const_i32(0)))):
                b.global_store(segm_max, ml_idx, ms[r], align=4)
                b.global_store(segm_expsum, ml_idx, ls[r], align=4)
            for d in range(HD // _WMMA_N):
                o_col = b.add(b.const_i32(d * _WMMA_N), col_n)
                with b.scf_if(row_valid):
                    b.global_store(
                        segm_output,
                        b.add(so_base, o_col),
                        b.vec_extract(accs[d], r),
                        align=4,
                    )

    # Empty segment (no tiles in range): write neutral partials so the reduce
    # sees a contribution of zero for every (token, head, segment).
    with b.scf_if(b.cmp_ge(tile_start, tile_end)):
        neutral_acc = [b.zero_vec_f32(c_frag) for _ in range(HD // _WMMA_N)]
        _write_partials(
            [neg_inf for _ in range(c_frag)],
            [zero_f for _ in range(c_frag)],
            neutral_acc,
        )
        b.ret()

    # --- Q fragments (A operand), zeroed for invalid rows ---
    q_row_for_a = lane_row
    q_pos_for_a = b.add(qb_start_pos, b.div(q_row_for_a, b.const_i32(NQK)))
    qh_for_a = b.add(
        b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(q_row_for_a, b.const_i32(NQK))
    )
    q_valid_for_a = b.land(
        b.cmp_lt(q_pos_for_a, cur_batch_q_len), b.cmp_lt(qh_for_a, b.const_i32(NUM_QH))
    )
    q_pos_safe = b.select(q_valid_for_a, q_pos_for_a, b.const_i32(0))
    qh_safe = b.select(q_valid_for_a, qh_for_a, b.const_i32(0))
    q_token = b.add(cu_q_start, q_pos_safe)
    q_addr_row_base = b.mul(
        b.add(b.mul(q_token, b.const_i32(NUM_QH)), qh_safe), b.const_i32(HD)
    )
    q_frags = load_q_frags(
        b,
        query,
        q_addr_row_base,
        half_k,
        q_valid_for_a,
        head_size=HD,
        a_frag=a_frag,
        dtype=dtype,
    )

    kv_desc = PagedKvDescriptor(
        block_size=BS,
        stride_0=BS * spec.num_kv_heads * HD,
        stride_1=spec.num_kv_heads * HD,
        stride_2=HD,
        stride_3=1,
    )

    def _phys_block(tok_global):
        logical = b.div(tok_global, b.const_i32(BS))
        return b.global_load_i32(
            block_tables, b.add(b.mul(seq_idx, block_table_stride), logical)
        )

    if NUM_WAVES > 1:
        # ---- cooperative multi-wave32 path (LDS inter-wave reduction) ----
        W = NUM_WAVES
        D_BLK = HD // _WMMA_N
        P_lds = b.smem_alloc(dtype, [W, _BLOCK_M, _T], name_hint="P3d_mw")
        V_lds = b.smem_alloc(dtype, [W, _T, HD], name_hint="V3d_mw")
        m_lds = b.smem_alloc(F32, [W, _BLOCK_M], name_hint="m3d_mw")
        l_lds = b.smem_alloc(F32, [W, _BLOCK_M], name_hint="l3d_mw")
        acc_lds = b.smem_alloc(F32, [W, _BLOCK_M, HD], name_hint="acc3d_mw")

        # Per-wave online-softmax init. The sink mass must be counted exactly
        # once, so only wave 0 (segment 0) seeds ``m`` with the sink; the
        # ``l=1`` start is zeroed by the first valid tile's ``alpha`` and is
        # neutralised by ``f=0`` for any wave that sees no valid tile.
        m_inits = []
        for r in range(c_frag):
            row_rel, _ = c_map.coord(b, lane, r)
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row_rel, b.const_i32(NQK))
            )
            qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
            if spec.use_sinks:
                sink_h = b.global_load(sinks, qh, dtype, align=2)
                sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
                use_sink = b.land(
                    b.land(qh_in, b.cmp_eq(seg_idx, b.const_i32(0))),
                    b.cmp_eq(wave_id, b.const_i32(0)),
                )
                m_inits.append(b.select(use_sink, sink_f, neg_inf))
            else:
                m_inits.append(neg_inf)
        l_inits = [one_f for _ in range(c_frag)]
        acc_inits = [b.zero_vec_f32(c_frag) for _ in range(D_BLK)]

        iter_args = []
        for r in range(c_frag):
            iter_args.append((f"m{r}", m_inits[r]))
            iter_args.append((f"l{r}", l_inits[r]))
        for d in range(D_BLK):
            iter_args.append((f"acc{d}", acc_inits[d]))

        # Uniform iteration count across waves so every CTA barrier is reached
        # the same number of times (mismatched trip counts would deadlock).
        span = b.sub(tile_end, tile_start)
        n_iter = b.div(b.add(span, b.const_i32(W - 1)), b.const_i32(W))

        kloop = b.scf_for_iter(
            b.const_i32(0), n_iter, b.const_i32(1), iter_args=iter_args, iv_name="it"
        )
        with kloop as (it, state):
            ms = [state[2 * r] for r in range(c_frag)]
            ls = [state[2 * r + 1] for r in range(c_frag)]
            accs = list(state[2 * c_frag :])
            tile_raw = b.add(b.add(tile_start, wave_id), b.mul(it, b.const_i32(W)))
            tile_valid = b.cmp_lt(tile_raw, tile_end)
            tile_idx = b.select(tile_valid, tile_raw, tile_start)
            tile_base = b.mul(tile_idx, b.const_i32(_T))

            scores = compute_qk_scores(
                b,
                q_frags,
                key,
                kv_desc,
                tile_base=tile_base,
                lane_row=lane_row,
                half_k=half_k,
                kv_head_idx=kv_head_idx,
                block_size=BS,
                head_size=HD,
                kv_dtype=kv_dtype,
                k_scale=k_scale,
                dtype=dtype,
                c_frag=c_frag,
                phys_block=_phys_block,
                spacing=spec.wmma_spacing,
            )

            new_ms, new_ls, new_accs = [], [], list(accs)
            ps = [[], []]
            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                q_pos = b.add(qb_start_pos, b.div(row_rel, b.const_i32(NQK)))
                qh = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)),
                    b.mod(row_rel, b.const_i32(NQK)),
                )
                row_valid = b.land(
                    b.cmp_lt(q_pos, cur_batch_q_len), b.cmp_lt(qh, b.const_i32(NUM_QH))
                )
                causal_lim = b.add(context_len, q_pos)
                srs = []
                for nsub in range(2):
                    key_pos = b.add(
                        b.add(tile_base, b.const_i32(nsub * _WMMA_N)), col_k
                    )
                    score_log2 = b.fmul(b.vec_extract(scores[nsub], r), qk_scale)
                    causal_keep = b.cmp_le(key_pos, causal_lim)
                    in_seq = b.cmp_lt(key_pos, seq_len)
                    keep = b.land(
                        b.land(row_valid, tile_valid), b.land(in_seq, causal_keep)
                    )
                    if SLIDING_WINDOW > 0:
                        dist = b.sub(causal_lim, key_pos)
                        keep = b.land(keep, b.cmp_lt(dist, b.const_i32(SLIDING_WINDOW)))
                    srs.append(b.select(keep, score_log2, neg_inf))

                m_new, l_new, alpha, p = softmax_row_update(
                    b,
                    ms[r],
                    ls[r],
                    srs,
                    neg_inf=neg_inf,
                    zero_f=zero_f,
                    use_dpp=spec.use_dpp_softmax,
                )
                new_ms.append(m_new)
                new_ls.append(l_new)
                ps[0].append(p[0])
                ps[1].append(p[1])
                for d in range(D_BLK):
                    old = b.vec_extract(new_accs[d], r)
                    new_accs[d] = b.vec_insert(new_accs[d], b.fmul(old, alpha), r)

            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                b.smem_store_vN(
                    P_lds, [wave_id, row_rel, col_k], b.cast_f32_to(ps[0][r], dtype), 1
                )
                b.smem_store_vN(
                    P_lds,
                    [wave_id, row_rel, b.add(col_k, b.const_i32(_WMMA_N))],
                    b.cast_f32_to(ps[1][r], dtype),
                    1,
                )

            stage_v_tile_buf(
                b,
                V_lds,
                wave_id,
                value,
                kv_desc,
                kv_head_idx=kv_head_idx,
                tile_base=tile_base,
                lane=lane,
                block_size=BS,
                head_size=HD,
                kv_dtype=kv_dtype,
                v_scale=v_scale,
                dtype=dtype,
                phys_block=_phys_block,
            )
            b.sync()

            new_accs = compute_pv(
                b,
                P_lds,
                V_lds,
                new_accs,
                a_map=a_map,
                c_map=c_map,
                lane=lane,
                lane_row=lane_row,
                col=col,
                a_frag=a_frag,
                c_frag=c_frag,
                head_size=HD,
                dtype=dtype,
                v_extra_idx=wave_id,
                p_extra_idx=wave_id,
                spacing=spec.wmma_spacing,
            )
            b.sync()

            yields = []
            for r in range(c_frag):
                yields.append(new_ms[r])
                yields.append(new_ls[r])
            yields.extend(new_accs)
            b.scf_yield(*yields)

        final = kloop.results
        ms_final = [final[2 * r] for r in range(c_frag)]
        ls_final = [final[2 * r + 1] for r in range(c_frag)]
        accs_final = list(final[2 * c_frag :])

        # Stage each wave's partials into LDS in [row, col] layout.
        for r in range(c_frag):
            row_rel, col_n = c_map.coord(b, lane, r)
            with b.scf_if(b.cmp_eq(col_n, b.const_i32(0))):
                b.smem_store_vN(m_lds, [wave_id, row_rel], ms_final[r], 1)
                b.smem_store_vN(l_lds, [wave_id, row_rel], ls_final[r], 1)
            for d in range(D_BLK):
                o_col = b.add(b.const_i32(d * _WMMA_N), col_n)
                b.smem_store_vN(
                    acc_lds,
                    [wave_id, row_rel, o_col],
                    b.vec_extract(accs_final[d], r),
                    1,
                )
        b.sync()

        # Wave 0 merges the W partials (online-softmax combine) and writes once.
        with b.scf_if(b.cmp_eq(wave_id, b.const_i32(0))):
            comb_m, comb_l = [], []
            comb_acc = [b.zero_vec_f32(c_frag) for _ in range(D_BLK)]
            for r in range(c_frag):
                row_rel, col_n = c_map.coord(b, lane, r)
                mws = [
                    b.vec_extract(
                        b.smem_load_vN(m_lds, b.const_i32(w), row_rel, dtype=F32, n=1),
                        0,
                    )
                    for w in range(W)
                ]
                om = neg_inf
                for w in range(W):
                    om = b.fmax(om, mws[w])
                fws = []
                ol = zero_f
                for w in range(W):
                    fw = b.select(
                        b.fcmp("ogt", mws[w], neg_inf),
                        b.exp2(b.fsub(mws[w], om)),
                        zero_f,
                    )
                    fws.append(fw)
                    lw = b.vec_extract(
                        b.smem_load_vN(l_lds, b.const_i32(w), row_rel, dtype=F32, n=1),
                        0,
                    )
                    ol = b.fadd(ol, b.fmul(lw, fw))
                comb_m.append(om)
                comb_l.append(ol)
                for d in range(D_BLK):
                    o_col = b.add(b.const_i32(d * _WMMA_N), col_n)
                    a = zero_f
                    for w in range(W):
                        av = b.vec_extract(
                            b.smem_load_vN(
                                acc_lds, b.const_i32(w), row_rel, o_col, dtype=F32, n=1
                            ),
                            0,
                        )
                        a = b.fadd(a, b.fmul(av, fws[w]))
                    comb_acc[d] = b.vec_insert(comb_acc[d], a, r)
            _write_partials(comb_m, comb_l, comb_acc)
        b.ret()
        return b.kernel

    P_lds = (
        None
        if spec.use_register_p
        else b.smem_alloc(dtype, [_BLOCK_M, _T], name_hint="P3d_gfx1250")
    )
    if spec.use_wide_lds_reads:
        # dim-major [HD, T] so each lane's 16-token WMMA-B fragment is one wide read.
        V_lds = b.smem_alloc(dtype, [HD, _T], name_hint="V3dT_gfx1250")
    elif spec.use_wide_kv_load or use_dtla_stage:
        # token-major double buffer; DTLA streams async global->LDS into it.
        V_lds = b.smem_alloc(dtype, [2, _T, HD], name_hint="V3d_gfx1250_dbl")
    else:
        V_lds = b.smem_alloc(dtype, [_T, HD], name_hint="V3d_gfx1250")

    causal_lim_r = []
    row_valid_r = []
    if spec.use_invariant_hoist:
        for r in range(c_frag):
            row_rel, _col_k = c_map.coord(b, lane, r)
            q_pos_r = b.add(qb_start_pos, b.div(row_rel, b.const_i32(NQK)))
            qh_r = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row_rel, b.const_i32(NQK))
            )
            row_valid_r.append(
                b.land(
                    b.cmp_lt(q_pos_r, cur_batch_q_len),
                    b.cmp_lt(qh_r, b.const_i32(NUM_QH)),
                )
            )
            causal_lim_r.append(b.add(context_len, q_pos_r))

    # online-softmax init; sinks only contribute on segment 0.
    m_inits = []
    for r in range(c_frag):
        row_rel, _ = c_map.coord(b, lane, r)
        qh = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row_rel, b.const_i32(NQK))
        )
        qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
        if spec.use_sinks:
            sink_h = b.global_load(sinks, qh, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            use_sink = b.land(qh_in, b.cmp_eq(seg_idx, b.const_i32(0)))
            m_inits.append(b.select(use_sink, sink_f, neg_inf))
        else:
            m_inits.append(neg_inf)
    l_inits = [
        (
            b.select(b.cmp_eq(seg_idx, b.const_i32(0)), one_f, zero_f)
            if spec.use_sinks
            else one_f
        )
        for _ in range(c_frag)
    ]
    acc_inits = [b.zero_vec_f32(c_frag) for _ in range(HD // _WMMA_N)]

    iter_args = []
    for r in range(c_frag):
        iter_args.append((f"m{r}", m_inits[r]))
        iter_args.append((f"l{r}", l_inits[r]))
    for d in range(HD // _WMMA_N):
        iter_args.append((f"acc{d}", acc_inits[d]))

    # ---- DTLA: async global->LDS V staging + double buffer + prefetch ----
    # gfx1250 (GFX12 / gfx1250) does NOT have the gfx9 ``buffer_load_lds`` /
    # ``global_load_lds`` DirectToLDS instructions (they don't select). It has a
    # dedicated async-DMA family: ``global_load_async_to_lds_b128`` (per-lane
    # global->LDS copy of 16 B) tracked by its own ASYNC counter, drained with
    # ``s_wait_asynccnt``. We mirror ``stage_v_tile``'s addressing (lane == one
    # of the 32 tile tokens; ``HD/8`` b128 calls walk the head dim), so the
    # per-token ``_phys_block`` lookup handles block_size 16 (2 blocks/tile) and
    # 32 (1 block/tile) without a separate byte descriptor.
    if use_dtla_stage:
        DTLA_CALLS_PER_TILE = HD // 8  # 8 b128 (16 B = 8 bf16) calls / token
        # CACHE_STREAM (SLC): one-shot streaming KV, not reused.
        DTLA_CPOL = 2

        def _issue_v_load(kt_val: Value, buf_idx: Value) -> None:
            tile_base_l = b.mul(kt_val, b.const_i32(_T))
            v_global = b.add(tile_base_l, lane)  # lane 0..31 -> token in tile
            vpblk = _phys_block(v_global)
            v_tib = b.mod(v_global, b.const_i32(BS))
            # Base (dim 0) global + LDS pointers; the per-call head-dim chunk is
            # carried in the instruction's IMMEDIATE byte offset (applies to both
            # the global source and the LDS dest). Baking the constant chunk
            # offset into the *pointers* instead makes the gfx1250 backend merge
            # the unrolled async calls (only the dim-0 chunk lands) -- verified
            # via a fill readback probe. 8 bf16 (b128 = 16 B) per call.
            base_src = kv_desc.offset(
                b,
                physical_block=vpblk,
                token_in_block=v_tib,
                kv_head=kv_head_idx,
                dim=b.const_i32(0),
            )
            for call in range(DTLA_CALLS_PER_TILE):
                b.global_load_async_to_lds(
                    value,
                    base_src,
                    V_lds,
                    [buf_idx, lane, b.const_i32(0)],
                    width_bytes=16,
                    coherency=DTLA_CPOL,
                    offset_bytes=call * 16,
                )

        # Prologue: kick the first tile's V load so it overlaps the first QK.
        _issue_v_load(tile_start, b.const_i32(0))
        iter_args.append(("cur_buf", b.const_i32(0)))
    elif spec.use_wide_kv_load:
        tile_base_pre = b.mul(tile_start, b.const_i32(_T))
        stage_v_tile_buf(
            b,
            V_lds,
            b.const_i32(0),
            value,
            kv_desc,
            kv_head_idx=kv_head_idx,
            tile_base=tile_base_pre,
            lane=lane,
            block_size=BS,
            head_size=HD,
            kv_dtype=kv_dtype,
            v_scale=v_scale,
            dtype=dtype,
            phys_block=_phys_block,
        )
        b.sync()

    kloop = b.scf_for_iter(
        tile_start, tile_end, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        if spec.use_sw_pipeline:
            # GEMM-style MFMA<->load interleave cadence; only pays off because
            # the DTLA async-V load is in flight (shadow) across the body.
            b.iglp_opt(1)
        ms = [state[2 * r] for r in range(c_frag)]
        ls = [state[2 * r + 1] for r in range(c_frag)]
        if use_dtla_stage:
            cur_buf = state[-1]
            accs = list(state[2 * c_frag : -1])
        else:
            cur_buf = None
            accs = list(state[2 * c_frag :])
        tile_base = b.mul(kt, b.const_i32(_T))

        scores = compute_qk_scores(
            b,
            q_frags,
            key,
            kv_desc,
            tile_base=tile_base,
            lane_row=lane_row,
            half_k=half_k,
            kv_head_idx=kv_head_idx,
            block_size=BS,
            head_size=HD,
            kv_dtype=kv_dtype,
            k_scale=k_scale,
            dtype=dtype,
            c_frag=c_frag,
            phys_block=_phys_block,
            spacing=spec.wmma_spacing,
        )

        new_ms, new_ls, new_accs = [], [], list(accs)
        ps = [[], []]
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)
            if spec.use_invariant_hoist:
                row_valid = row_valid_r[r]
                causal_lim = causal_lim_r[r]
            else:
                q_pos = b.add(qb_start_pos, b.div(row_rel, b.const_i32(NQK)))
                qh = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)),
                    b.mod(row_rel, b.const_i32(NQK)),
                )
                row_valid = b.land(
                    b.cmp_lt(q_pos, cur_batch_q_len), b.cmp_lt(qh, b.const_i32(NUM_QH))
                )
                causal_lim = b.add(context_len, q_pos)
            srs = []
            for nsub in range(2):
                key_pos = b.add(b.add(tile_base, b.const_i32(nsub * _WMMA_N)), col_k)
                score_log2 = b.fmul(b.vec_extract(scores[nsub], r), qk_scale)
                causal_keep = b.cmp_le(key_pos, causal_lim)
                in_seq = b.cmp_lt(key_pos, seq_len)
                keep = b.land(row_valid, b.land(in_seq, causal_keep))
                if SLIDING_WINDOW > 0:
                    dist = b.sub(causal_lim, key_pos)
                    keep = b.land(keep, b.cmp_lt(dist, b.const_i32(SLIDING_WINDOW)))
                srs.append(b.select(keep, score_log2, neg_inf))

            if spec.ablate_softmax:
                # perf-only ablation: drop the wave_reduce (ds_swizzle) + exp2;
                # keep P dependent on the scores so QK isn't DCE'd.
                m_new, l_new, alpha, p = ms[r], ls[r], one_f, [srs[0], srs[1]]
            else:
                m_new, l_new, alpha, p = softmax_row_update(
                    b,
                    ms[r],
                    ls[r],
                    srs,
                    neg_inf=neg_inf,
                    zero_f=zero_f,
                    use_dpp=spec.use_dpp_softmax,
                )
            new_ms.append(m_new)
            new_ls.append(l_new)
            ps[0].append(p[0])
            ps[1].append(p[1])
            for d in range(HD // _WMMA_N):
                old = b.vec_extract(new_accs[d], r)
                new_accs[d] = b.vec_insert(new_accs[d], b.fmul(old, alpha), r)

        if not spec.use_register_p:
            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                b.smem_store_vN(
                    P_lds, [row_rel, col_k], b.cast_f32_to(ps[0][r], dtype), 1
                )
                b.smem_store_vN(
                    P_lds,
                    [row_rel, b.add(col_k, b.const_i32(_WMMA_N))],
                    b.cast_f32_to(ps[1][r], dtype),
                    1,
                )

        v_read = b.mod(b.sub(kt, tile_start), b.const_i32(2))
        nxt_buf = None
        if use_dtla_stage:
            # V[kt] is already streaming (issued by the prologue / previous
            # iter). Fully drain the ASYNC counter so V[kt] has landed, publish
            # the P_lds store (dscnt), *then* kick V[kt+1] so its async copy
            # overlaps the PV below + the next iteration's QK/softmax. Draining
            # to 0 before issuing the next tile keeps correctness independent of
            # async completion ordering; partial-keep (s_wait_asynccnt(n>0)) is
            # a follow-up tuning step once in-order completion is confirmed.
            nxt_buf = b.sub(b.const_i32(1), cur_buf)
            next_kt = b.add(kt, b.const_i32(1))
            # Clamp the prefetch to a valid tile so the pipeline depth (and thus
            # the static wait immediate) stays constant; the extra last-iter
            # copy lands in nxt_buf, which is never consumed.
            safe_next = b.select(b.cmp_lt(next_kt, tile_end), next_kt, kt)
            b.s_wait_asynccnt(0)
            b.sync_lds_only()
            _issue_v_load(safe_next, nxt_buf)
            if spec.use_ds_tr_reads:
                # combined stack: DTLA async double-buffer (wide global read) +
                # ds_load_tr transpose-on-read (wide LDS read) off the cur_buf slab.
                new_accs = compute_pv_dstr(
                    b,
                    P_lds,
                    V_lds,
                    new_accs,
                    a_map=a_map,
                    lane=lane,
                    lane_row=lane_row,
                    a_frag=a_frag,
                    head_size=HD,
                    dtype=dtype,
                    v_extra_idx=cur_buf,
                    spacing=spec.wmma_spacing,
                )
            else:
                new_accs = compute_pv(
                    b,
                    P_lds,
                    V_lds,
                    new_accs,
                    a_map=a_map,
                    c_map=c_map,
                    lane=lane,
                    lane_row=lane_row,
                    col=col,
                    a_frag=a_frag,
                    c_frag=c_frag,
                    head_size=HD,
                    dtype=dtype,
                    v_extra_idx=cur_buf,
                    spacing=spec.wmma_spacing,
                )
            # PV's ds_reads settle before the next iter overwrites P_lds; the
            # in-flight V[kt+1] async copy is NOT drained here (asynccnt), so it
            # keeps streaming across the barrier into the next iteration.
            b.sync_lds_only()
        elif spec.ablate_pv:
            # perf-only ceiling: skip V staging (load + fp8 dequant) + PV-GEMM.
            # The P_lds store above + alpha-scaled accs keep QK+softmax live.
            pass
        else:
            if spec.use_wide_lds_reads:
                stage_v_tile_transposed(
                    b,
                    V_lds,
                    value,
                    kv_desc,
                    kv_head_idx=kv_head_idx,
                    tile_base=tile_base,
                    lane=lane,
                    block_size=BS,
                    head_size=HD,
                    kv_dtype=kv_dtype,
                    v_scale=v_scale,
                    dtype=dtype,
                    phys_block=_phys_block,
                )
            elif spec.use_wide_kv_load:
                next_kt = b.add(kt, b.const_i32(1))
                has_next = b.cmp_lt(next_kt, tile_end)
                v_write = b.mod(b.add(v_read, b.const_i32(1)), b.const_i32(2))
                tile_base_n = b.mul(next_kt, b.const_i32(_T))
                with b.scf_if(has_next):
                    stage_v_tile_buf(
                        b,
                        V_lds,
                        v_write,
                        value,
                        kv_desc,
                        kv_head_idx=kv_head_idx,
                        tile_base=tile_base_n,
                        lane=lane,
                        block_size=BS,
                        head_size=HD,
                        kv_dtype=kv_dtype,
                        v_scale=v_scale,
                        dtype=dtype,
                        phys_block=_phys_block,
                    )
            else:
                stage_v_tile(
                    b,
                    V_lds,
                    value,
                    kv_desc,
                    kv_head_idx=kv_head_idx,
                    tile_base=tile_base,
                    lane=lane,
                    block_size=BS,
                    head_size=HD,
                    kv_dtype=kv_dtype,
                    v_scale=v_scale,
                    dtype=dtype,
                    phys_block=_phys_block,
                )
            b.sync()

            if spec.use_register_p:
                new_accs = compute_pv_from_probs(
                    b,
                    ps[0],
                    ps[1],
                    V_lds,
                    new_accs,
                    a_map=a_map,
                    c_map=c_map,
                    lane=lane,
                    col=col,
                    a_frag=a_frag,
                    c_frag=c_frag,
                    head_size=HD,
                    dtype=dtype,
                    v_extra_idx=v_read if spec.use_wide_kv_load else None,
                    spacing=spec.wmma_spacing,
                )
            elif spec.use_wide_lds_reads:
                new_accs = compute_pv_wide(
                    b,
                    P_lds,
                    V_lds,
                    new_accs,
                    a_map=a_map,
                    lane=lane,
                    lane_row=lane_row,
                    a_frag=a_frag,
                    head_size=HD,
                    dtype=dtype,
                    spacing=spec.wmma_spacing,
                )
            elif spec.use_ds_tr_reads:
                new_accs = compute_pv_dstr(
                    b,
                    P_lds,
                    V_lds,
                    new_accs,
                    a_map=a_map,
                    lane=lane,
                    lane_row=lane_row,
                    a_frag=a_frag,
                    head_size=HD,
                    dtype=dtype,
                    spacing=spec.wmma_spacing,
                )
            else:
                new_accs = compute_pv(
                    b,
                    P_lds,
                    V_lds,
                    new_accs,
                    a_map=a_map,
                    c_map=c_map,
                    lane=lane,
                    lane_row=lane_row,
                    col=col,
                    a_frag=a_frag,
                    c_frag=c_frag,
                    head_size=HD,
                    dtype=dtype,
                    v_extra_idx=v_read if spec.use_wide_kv_load else None,
                    spacing=spec.wmma_spacing,
                )
            b.sync()

        yields = []
        for r in range(c_frag):
            yields.append(new_ms[r])
            yields.append(new_ls[r])
        yields.extend(new_accs)
        if use_dtla_stage:
            yields.append(nxt_buf)
        b.scf_yield(*yields)

    final = kloop.results
    ms_final = [final[2 * r] for r in range(c_frag)]
    ls_final = [final[2 * r + 1] for r in range(c_frag)]
    accs_final = (
        list(final[2 * c_frag : -1]) if use_dtla_stage else list(final[2 * c_frag :])
    )
    _write_partials(ms_final, ls_final, accs_final)
    b.ret()
    return b.kernel


def build_unified_attention_reduce_tiled(
    spec: UnifiedAttentionReduceTiledSpec, arch: str = "gfx1250"
) -> KernelDef:
    """Build the gfx1250 split-KV reduce kernel (merge per-segment partials)."""
    dtype = spec.dtype_ir
    HD = spec.head_size
    NUM_QH = spec.num_query_heads
    NUM_SEG = spec.num_segments
    WAVE = _WAVE

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = WAVE
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    output = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    segm_output = b.param(
        "segm_output_ptr", PtrType(F32, "global"), readonly=True, align=16
    )
    segm_max = b.param("segm_max_ptr", PtrType(F32, "global"), readonly=True, align=4)
    segm_expsum = b.param(
        "segm_expsum_ptr", PtrType(F32, "global"), readonly=True, align=4
    )
    _seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)

    q_token = b.block_id_x()
    q_head = b.block_id_y()
    lane = b.mod(b.thread_id_x(), b.const_i32(WAVE))
    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    ml_base = b.mul(
        b.add(b.mul(q_token, b.const_i32(NUM_QH)), q_head), b.const_i32(NUM_SEG)
    )
    so_base = b.mul(
        ml_base, b.const_i32(HD)
    )  # noqa: F841 -- side-effecting emit; keep for byte-identity

    factor_lds = b.smem_alloc(F32, [NUM_SEG], name_hint="factor3d_gfx1250")
    n_iter = (NUM_SEG + WAVE - 1) // WAVE

    # pass 1: overall max over segments
    local_max = neg_inf
    for i in range(n_iter):
        s = b.add(lane, b.const_i32(i * WAVE))
        valid = b.cmp_lt(s, b.const_i32(NUM_SEG))
        s_safe = b.select(valid, s, b.const_i32(0))
        m = b.global_load(segm_max, b.add(ml_base, s_safe), F32, align=4)
        local_max = b.fmax(local_max, b.select(valid, m, neg_inf))
    overall_max = wave_reduce_max(b, local_max, wave_size=WAVE, lanes_per_row=WAVE)

    # pass 2: factor[s] + overall denom
    local_sum = zero_f
    for i in range(n_iter):
        s = b.add(lane, b.const_i32(i * WAVE))
        valid = b.cmp_lt(s, b.const_i32(NUM_SEG))
        s_safe = b.select(valid, s, b.const_i32(0))
        m = b.global_load(segm_max, b.add(ml_base, s_safe), F32, align=4)
        l = b.global_load(
            segm_expsum, b.add(ml_base, s_safe), F32, align=4
        )  # noqa: E741 -- l = per-segment expsum load
        m_finite = b.land(b.fcmp("oeq", m, m), b.fcmp("ogt", m, neg_inf))
        f = b.select(m_finite, b.exp2(b.fsub(m, overall_max)), zero_f)
        f = b.select(valid, f, zero_f)
        with b.scf_if(valid):
            b.smem_store_vN(factor_lds, [s_safe], f, 1)
        local_sum = b.fadd(local_sum, b.fmul(l, f))
    overall = wave_reduce_sum(b, local_sum, wave_size=WAVE, lanes_per_row=WAVE)
    inv_l = b.select(b.fcmp("oeq", overall, zero_f), zero_f, b.rcp(overall))
    b.sync()

    # pass 3: per-dim acc reduce + normalize + cast
    d_iter = (HD + WAVE - 1) // WAVE
    for i in range(d_iter):
        d = b.add(lane, b.const_i32(i * WAVE))
        d_valid = b.cmp_lt(d, b.const_i32(HD))
        d_safe = b.select(d_valid, d, b.const_i32(0))
        acc = zero_f
        for s in range(NUM_SEG):
            f = b.vec_extract(
                b.smem_load_vN(factor_lds, b.const_i32(s), dtype=F32, n=1), 0
            )
            ov = b.global_load(
                segm_output,
                b.add(b.mul(b.add(ml_base, b.const_i32(s)), b.const_i32(HD)), d_safe),
                F32,
                align=4,
            )
            acc = b.fadd(acc, b.fmul(ov, f))
        out_addr = b.add(
            b.mul(b.add(b.mul(q_token, b.const_i32(NUM_QH)), q_head), b.const_i32(HD)),
            d_safe,
        )
        with b.scf_if(d_valid):
            b.global_store(
                output, out_addr, b.cast_f32_to(b.fmul(acc, inv_l), dtype), align=2
            )
    b.ret()
    return b.kernel
