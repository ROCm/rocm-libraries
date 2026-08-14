# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Spec-creation functions for the tiled unified-attention kernels.

These two functions are the single seam between an ``UnifiedAttentionProblem``
descriptor and the arch-specific ``UnifiedAttention2DTiledSpec`` /
``UnifiedAttention3DTiledSpec`` dataclasses that drive compilation.  They live
here (in the builders layer) so that analysis scripts, tuners, and benchmarks
can instantiate and inspect specs without importing the full kernel-dispatch
machinery.

``kernels.common.attention_unified`` imports both symbols from here;
callers that already import from that module do not need to change.
"""
from __future__ import annotations

from dataclasses import fields, replace

from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _enable_combo_2d,
    _enable_early_v_schedule,
    _enable_fp8_mfma_qk,
    _enable_gfx942_3d_invariant_hoist,
    _enable_gfx942_3d_wide_kv_load,
    _enable_gfx942_bf16_flash,
    _enable_gfx942_flash_k_sliced_ldsseq,
    _enable_gfx942_flash_k_sliced_ring,
    _select_gfx942_flash_ring_depth,
    _select_gfx942_flash_k_slice_hd,
    _enable_gfx942_flash_mask_limit,
    _enable_gfx942_flash_q_direct,
    _enable_gfx942_fp16_flash,
    _enable_i64_kv_addr,
    _enable_k_single_buffer,
    _enable_mfma_32x32,
    _enable_register_pv,
    _enable_sched_barrier,
    _enable_softmax_mfma_interleave,
    _enable_transposed_half_local_pv,
    _enable_transposed_qk_32x32,
    _enable_transposed_subflags,
    _enable_v_double_buffer,
    _gfx942_3d_tile_size_override,
    _gfx942_bf16_wide_geometry,
    _gfx942_bf16_wide_tile_size,
    _gfx942_bf16_wide_use_cfvst,
    _gfx942_flash_kv_cache_policy,
    _gfx942_flash_use_cfvst,
    _gfx942_flash_use_single_buffer,
    _gfx942_flash_wide_setting,
    _kv_storage_dtype,
    _num_segments,
    _resolve_attention_arch,
    _resolve_gfx1250_tiled3d,
    _select_2d_block_m_per_warp,
    _select_2d_num_warps,
    _select_2d_tile_size,
    _select_2d_waves_per_eu,
    _select_3d_waves_per_eu,
    _select_gfx942_flash_num_warps,
    _tiled_2d_impl,
    _tiled_3d_impl,
)

# Imported as a module (not a bound symbol) so tests that
# ``mock.patch.object(attention_unified, "_d256_gfx950_fast", ...)`` still steer
# the builder's fast-route branch below (a bound import would freeze the ref).
from kernels.common import attention_unified as _kau


def _tiled_spec_from_problem(
    problem: UnifiedAttentionProblem,
):
    arch = _resolve_attention_arch()
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl(arch)
    if arch == "gfx1250":
        return UnifiedAttention2DTiledSpec(
            head_size=problem.head_size,
            block_size=problem.block_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            use_sinks=problem.use_sinks,
            sliding_window=problem.sliding_window,
            has_softcap=problem.softcap > 0,
            use_alibi=problem.use_alibi,
            use_qq_bias=problem.use_qq_bias,
            num_seqs=problem.num_seqs,
            num_warps=1,
            waves_per_eu=_select_2d_waves_per_eu(problem),
            kv_storage_dtype=_kv_storage_dtype(problem),
            tile_size=_select_2d_tile_size(problem),
            block_m_per_warp=16,
        )
    # The gfx942 4-warp GQA cohort (D256 + D128 sliding-window, see _gfx942_4warp_fast)
    # is built by build_gfx942_4warp_gqa with its own HD/BS-derived geometry, and needs
    # the default branch's *discriminator* spec (num_warps=1, no mfma_32x32 / transposed_qk
    # / single-buffer), NOT the flash spec. The prior PR opened the flash gate for D128-SW
    # (kept as the fallback for SW edge cases the 4-warp excludes), so guard both flash
    # branches against the 4-warp cohort here -- otherwise the flash fields (num_warps=2,
    # single-buffer) build a spec the __post_init__ validator rejects for fp16 bs16/32.
    if _enable_gfx942_bf16_flash(problem) and not _kau._gfx942_4warp_fast(problem):
        # gfx942 bf16 wide-K (32x32x8) transposed flash path. DEFAULT-ON for
        # eligible shapes (small_q_narrow excluded; see _enable_gfx942_bf16_flash).
        # Uses the CDNA3-legal mfma_f32_32x32x8_bf16 atom (the K=16 bf16 atom is
        # gfx950-only). The transposed orientation consumes V from strided LDS +
        # P^T from registers (no P_lds, no gfx950-only transpose reads).
        #
        # When the sliced-K ring is active (HIPDNN_GFX942_K_SLICED_RING not
        # disabled, prefill), the bf16 path mirrors the fp16 ring geometry:
        #   nw=4 (BLOCK_M=128), 3-slot K ring, cfvst, T=64.
        # Without ring, falls back to the legacy bf16-wide geometry:
        #   D64  -> nw=4, double-buffered K.
        #   D128 -> nw=2 (BLOCK_M=64=T) + K single-buffer: LDS=48 KB.
        use_ring = _enable_gfx942_flash_k_sliced_ring(problem)
        if use_ring:
            nw = _gfx942_flash_wide_setting()
            single_k = False  # ring uses 3-slot staging, not single/double buffer
            use_cfvst = True  # ring requires cfvst (spec validator enforces this)
        else:
            nw, single_k = _gfx942_bf16_wide_geometry(problem)
            use_cfvst = _gfx942_bf16_wide_use_cfvst(problem)
        use_mask_limit = _enable_gfx942_flash_mask_limit(problem)
        return UnifiedAttention2DTiledSpec(
            head_size=problem.head_size,
            block_size=problem.block_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            use_sinks=problem.use_sinks,
            sliding_window=problem.sliding_window,
            has_softcap=problem.softcap > 0,
            use_alibi=problem.use_alibi,
            use_qq_bias=problem.use_qq_bias,
            num_seqs=problem.num_seqs,
            num_warps=nw,
            waves_per_eu=_select_2d_waves_per_eu(problem),
            kv_storage_dtype=_kv_storage_dtype(problem),
            tile_size=64 if use_ring else _gfx942_bf16_wide_tile_size(problem),
            block_m_per_warp=32,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=use_mask_limit,
            use_transposed_invariant_hoist=use_mask_limit,
            use_transposed_mask_once=use_mask_limit,
            use_transposed_mask_limit=use_mask_limit,
            use_conflict_free_v_store=use_cfvst,
            use_k_single_buffer=single_k,
            use_k_sliced_ring=use_ring,
            ring_depth=_select_gfx942_flash_ring_depth(problem),
            k_slice_hd=_select_gfx942_flash_k_slice_hd(problem),
            use_k_sliced_ldsseq=_enable_gfx942_flash_k_sliced_ldsseq(problem),
            use_q_direct_global=_enable_gfx942_flash_q_direct(problem),
            kv_cache_policy=_gfx942_flash_kv_cache_policy(problem),
            use_i64_kv_addr=_enable_i64_kv_addr(problem),
        )
    if _enable_gfx942_fp16_flash(problem) and not _kau._gfx942_4warp_fast(problem):
        num_warps = _select_gfx942_flash_num_warps(problem)
        use_cfvst = _gfx942_flash_use_cfvst(problem)
        use_single = _gfx942_flash_use_single_buffer(problem)
        use_mask_limit = _enable_gfx942_flash_mask_limit(problem)
        return UnifiedAttention2DTiledSpec(
            head_size=problem.head_size,
            block_size=problem.block_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            use_sinks=problem.use_sinks,
            sliding_window=problem.sliding_window,
            has_softcap=problem.softcap > 0,
            use_alibi=problem.use_alibi,
            use_qq_bias=problem.use_qq_bias,
            num_seqs=problem.num_seqs,
            num_warps=num_warps,
            waves_per_eu=_select_2d_waves_per_eu(problem),
            kv_storage_dtype=_kv_storage_dtype(problem),
            tile_size=_select_2d_tile_size(problem),
            block_m_per_warp=_select_2d_block_m_per_warp(problem),
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=use_mask_limit,
            use_transposed_invariant_hoist=use_mask_limit,
            use_transposed_mask_once=use_mask_limit,
            use_transposed_mask_limit=use_mask_limit,
            use_conflict_free_v_store=use_cfvst,
            use_k_single_buffer=use_single,
            use_k_sliced_ring=_enable_gfx942_flash_k_sliced_ring(problem),
            ring_depth=_select_gfx942_flash_ring_depth(problem),
            k_slice_hd=_select_gfx942_flash_k_slice_hd(problem),
            use_k_sliced_ldsseq=_enable_gfx942_flash_k_sliced_ldsseq(problem),
            use_q_direct_global=_enable_gfx942_flash_q_direct(problem),
            kv_cache_policy=_gfx942_flash_kv_cache_policy(problem),
            use_i64_kv_addr=_enable_i64_kv_addr(problem),
        )
    combo = _enable_combo_2d(problem)
    combo_no_sw = combo and problem.sliding_window == 0
    # The transposed-softmax VALU sub-flags now fire for the WHOLE no-SW
    # transposed-32x32 cohort (the narrow _enable_combo_2d family, the
    # single-batch d128/d64 prefill cohort, AND the multi-batch transposed
    # d128/d64 path that previously left them on the table -- the autotuner's
    # ~1.19x multi-batch miss). ``_enable_transposed_subflags`` already
    # excludes sliding window, so OR-ing it with the existing combo gates
    # preserves the SW-combo behaviour byte-for-byte:
    #   * scalar_state / skip_legacy_qreg : old ``combo``  -> ``combo OR sub``
    #     (SW combo: combo=True keeps them True; sub=False under SW.)
    #   * mask_once / mask_limit          : old ``combo_no_sw`` -> ``combo_no_sw OR sub``
    #     (SW combo: both stay False.)
    subflags = _enable_transposed_subflags(problem)
    scalar_state = combo or subflags
    skip_legacy_qreg = combo or subflags
    _bias_active = problem.softcap > 0 or problem.use_alibi or problem.use_qq_bias
    mask_opts = (combo_no_sw and not _bias_active) or subflags
    # gfx950-only schedule fields: the gfx942 2D spec class does not declare
    # ``use_v_double_buffer`` / ``use_sched_barrier``, and the default gfx942
    # forward reaches this shared return (no flash opt-in). Pass them only when
    # the resolved spec class actually declares the field -- gfx950 keeps the
    # exact same construction (byte-identical), while gfx942 no longer raises
    # ``TypeError: unexpected keyword argument`` on the unknown kwarg.
    _spec_field_names = {f.name for f in fields(UnifiedAttention2DTiledSpec)}
    _gfx950_schedule_fields = {}
    if "use_v_double_buffer" in _spec_field_names:
        _gfx950_schedule_fields["use_v_double_buffer"] = _enable_v_double_buffer(
            problem
        )
    if "use_sched_barrier" in _spec_field_names:
        _gfx950_schedule_fields["use_sched_barrier"] = _enable_sched_barrier(problem)
    # gfx950 d128 softmax<->MFMA interleave lever (iglp_opt(1)); paired with the
    # nw=4 widening in _select_2d_num_warps for the same cohort. Field-presence
    # guarded (gfx942/gfx1250 spec classes lack it). Mutually exclusive with
    # use_sched_barrier -- the cohorts do not overlap (sched_barrier is the
    # nw==1 short-prefill cohort; interleave is the wider d128 combo).
    if "use_softmax_mfma_interleave" in _spec_field_names and (
        _enable_softmax_mfma_interleave(problem)
    ):
        _gfx950_schedule_fields["use_softmax_mfma_interleave"] = True
        _gfx950_schedule_fields["softmax_interleave_mode"] = 1
    # d128 long-context lever: K single-buffer lets the larger T=64 tile fit
    # the 2-WG/CU LDS budget at HD=128 (see _select_2d_tile_size). Gated on the
    # same d128 small-tile cohort + opt-in env so default/production routing is
    # byte-identical. Field-presence guarded (gfx942/gfx1250 spec classes lack
    # it). _enable_k_single_buffer also re-asserts the T=64 / V-single-buffer /
    # no-fp8 preconditions so it can never fire on an incompatible spec.
    if "use_k_single_buffer" in _spec_field_names and _enable_k_single_buffer(problem):
        _gfx950_schedule_fields["use_k_single_buffer"] = True
    _spec = UnifiedAttention2DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
        num_warps=_select_2d_num_warps(problem),
        waves_per_eu=_select_2d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size=_select_2d_tile_size(problem),
        block_m_per_warp=_select_2d_block_m_per_warp(problem),
        use_mfma_32x32=_enable_mfma_32x32(problem),
        use_transposed_qk_32x32=_enable_transposed_qk_32x32(problem),
        use_transposed_half_local_pv=_enable_transposed_half_local_pv(problem),
        # Full combo stack (fires for the validated _enable_combo_2d family,
        # the single-batch d128/d64 prefill cohort, and the multi-batch
        # transposed d128/d64 path; a strict superset of the plain transposed
        # path). See the ``subflags`` reconciliation above.
        use_transposed_scalar_state=scalar_state,
        use_transposed_mask_once=mask_opts,
        use_transposed_mask_limit=mask_opts,
        use_mfma32_skip_legacy_qreg=skip_legacy_qreg,
        # Single-batch combo V-prefetch schedule (autotuner winners): short
        # prefill -> V double-buffer; long prefill -> early-V issue. Mutually
        # exclusive; both bit-identical to the no-flag path. Off for the
        # multi-batch combo family (its winners did not stack a V schedule).
        # (``use_v_double_buffer`` is injected via ``_gfx950_schedule_fields``
        # below -- gfx942's spec class does not declare it.)
        use_early_v_schedule=_enable_early_v_schedule(problem),
        # The fast paged-KV descriptor is specialised for bf16 / T=64 /
        # num_warps=4, which only the bf16 no-SW combo geometry uses (SW
        # combo is nw2 / T=32; fp8 combo uses the sync-dequant loader). The
        # gfx950 spec restricts it further to the exact 64-query / 8-kv head
        # cohort it was built for; `_enable_combo_2d` only checks the GQA-8
        # *ratio*, so a tensor-parallel-sharded GQA-8 model (e.g. 16/2) would
        # otherwise enable it and trip the spec validator. Match the validator's
        # absolute head-count restriction so non-64/8 GQA-8 combo shapes keep
        # the rest of the combo stack without the fast descriptor.
        use_fast_paged_kv_desc=(
            combo_no_sw
            and not problem.use_fp8
            and problem.num_query_heads == 64
            and problem.num_kv_heads == 8
            # self-consistency: fast_paged_kv_desc requires T==64. Only enable it
            # when the tile selector actually picks 64 for this shape, so the flag
            # can never be set with an incompatible tile (which trips the spec
            # validator). _select_2d_tile_size forces T=64 for this family.
            and _select_2d_tile_size(problem) == 64
        ),
        use_register_pv=_enable_register_pv(problem),
        use_fp8_mfma_qk=_enable_fp8_mfma_qk(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
        # CK-Tile-derived sched_barrier steering (lever 3 from the CK Tile ISA analysis). Fences the
        # QK MFMA cluster from the post-QK async prefetch VMEM so the LLVM
        # scheduler keeps the MFMAs packed. Additive perf knob (no routing
        # change); enabled only for the single-batch d128 short-prefill cohort
        # (num_warps==1 + V-double-buffer) where the single resident wave cannot
        # otherwise hide the prefetch-in-MFMA-window cost.
        # (``use_sched_barrier`` is injected via ``_gfx950_schedule_fields``
        # below -- gfx942's spec class does not declare it.)
        **_gfx950_schedule_fields,
    )
    if _kau._d256_gfx950_fast(problem):
        # D256 gfx950 bf16 prefill fast route. Authored here in the builder
        # (was a post-build override in kernels.common ``_tiled_spec_from_problem``,
        # so the winning spec is created in the builder, not
        # baked into the dispatch layer). Geometry (num_warps / tile_size /
        # block_m_per_warp) already comes from the gated selectors above; this
        # pins the 32x32 transposed + FA3 softmax<->MFMA-interleave codegen
        # constellation. The cohort is discriminated in ``_tiled_cache_key`` by
        # ``_d256_gfx950_fast`` so the key stays faithful to the built kernel.
        _spec = replace(_spec, **_kau._d256_gfx950_spec_overrides())
    return _spec


def _tiled_3d_spec_from_problem(
    problem: UnifiedAttentionProblem,
):
    arch = _resolve_attention_arch()
    UnifiedAttention3DTiledSpec, *_ = _tiled_3d_impl(arch)
    tile_size_override = _gfx942_3d_tile_size_override(problem)
    if arch == "gfx1250":
        r = _resolve_gfx1250_tiled3d(problem)
        return UnifiedAttention3DTiledSpec(
            head_size=problem.head_size,
            block_size=problem.block_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            use_sinks=problem.use_sinks,
            sliding_window=problem.sliding_window,
            has_softcap=problem.softcap > 0,
            num_segments=r.num_segments,
            use_alibi=problem.use_alibi,
            use_qq_bias=problem.use_qq_bias,
            num_seqs=problem.num_seqs,
            waves_per_eu=r.waves_per_eu,
            kv_storage_dtype=r.kv_storage_dtype,
            tile_size_override=r.tile_size_override,
            use_invariant_hoist=r.use_invariant_hoist,
            use_wide_kv_load=r.use_wide_kv_load,
            use_register_p=r.use_register_p,
            num_waves=r.num_waves,
            use_wide_lds_reads=r.use_wide_lds_reads,
            use_dtla_prefetch=r.use_dtla_prefetch,
            use_ds_tr_reads=r.use_ds_tr_reads,
            use_fused_reduce=r.use_fused_reduce,
            use_dpp_softmax=r.use_dpp_softmax,
        )
    return UnifiedAttention3DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        num_segments=_num_segments(problem),
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
        waves_per_eu=_select_3d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size_override=tile_size_override,
        use_invariant_hoist=_enable_gfx942_3d_invariant_hoist(problem),
        use_wide_kv_load=_enable_gfx942_3d_wide_kv_load(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
    )
