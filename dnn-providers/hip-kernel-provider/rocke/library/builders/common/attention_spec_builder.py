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

from dataclasses import replace

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
#
# ``_resolve_attention_arch`` MUST be reached through this module handle for the
# same reason, and is deliberately absent from the ``from ... import`` list
# above. It used to be bound, which silently defeated
# ``mock.patch.object(au, "_resolve_attention_arch", ...)``: the patch rebinds
# the attribute on the module, but a bound reference captured at import time
# still points at the original. The builder then resolved the REAL device arch,
# ``_tiled_2d_impl`` handed back that arch's spec class, and the gfx950-only
# override fields (e.g. ``use_q_direct_reg``) raised TypeError -- but only when
# this module was already imported before the patch was applied, so the
# breakage looked like flaky test ordering.
from kernels.common import attention_unified as _kau


def _spec_gfx942_fp16_flash(problem: UnifiedAttentionProblem):
    """gfx942 fp16 transposed-x8 flash geometry (the ``gfx942_dense_pipe`` engine).

    Self-contained per-engine spec builder (GEMM ``spec_fn`` pattern). Extracted
    verbatim from the ``_enable_gfx942_fp16_flash`` branch of
    ``_tiled_spec_from_problem`` -- byte-identical, no value change. The
    ``gfx942_dense_pipe`` dispatch candidate owns this geometry; the dispatcher
    itself still only decides ``(path, head_size, block_size)`` (see
    ``dispatch/AGENTS.md``).
    """
    arch = _kau._resolve_attention_arch()
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl(arch)
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


def _spec_gfx942_bf16_flash(problem: UnifiedAttentionProblem):
    """gfx942 bf16 wide-K (32x32x8) transposed flash geometry.

    Self-contained per-engine spec builder (GEMM ``spec_fn`` pattern), extracted
    verbatim from the ``_enable_gfx942_bf16_flash`` branch of
    ``_tiled_spec_from_problem`` -- byte-identical, no value change. Geometry lives
    in the builder layer; the dispatcher's ``(path, head_size, block_size)``
    identity + C++ parity are unchanged. See ``dispatch/AGENTS.md`` ->
    "Per-engine spec_fn".

    DEFAULT-ON for eligible shapes (small_q_narrow excluded; see
    _enable_gfx942_bf16_flash). Uses the CDNA3-legal mfma_f32_32x32x8_bf16 atom
    (the K=16 bf16 atom is gfx950-only). The transposed orientation consumes V
    from strided LDS + P^T from registers (no P_lds, no gfx950-only transpose
    reads). When the sliced-K ring is active (HIPDNN_GFX942_K_SLICED_RING not
    disabled, prefill), the bf16 path mirrors the fp16 ring geometry: nw=4
    (BLOCK_M=128), 3-slot K ring, cfvst, T=64. Without ring, falls back to the
    legacy bf16-wide geometry: D64 -> nw=4, double-buffered K; D128 -> nw=2
    (BLOCK_M=64=T) + K single-buffer (LDS=48 KB).
    """
    arch = _kau._resolve_attention_arch()
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl(arch)
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


def _base_2d_generic_fields(problem: UnifiedAttentionProblem) -> dict:
    """The 2D generic (non-flash fallthrough) spec fields shared by EVERY arch's
    generic builder.

    This is the block that was copy-pasted into both the non-gfx950 and gfx950
    generic builders. Extracting it keeps the two from drifting. The gfx950-only
    schedule fields (``use_v_double_buffer`` / ``use_sched_barrier`` /
    ``use_softmax_mfma_interleave`` / ``softmax_interleave_mode`` /
    ``use_k_single_buffer``) and the D256 fast-route override are NOT here -- they
    are the arch-unique tail that lives in ``_spec_gfx950_generic``. Every field
    below is declared by all the 2D spec classes, so splatting this dict is
    byte-identical to the inline construction it replaces.
    """
    combo = _enable_combo_2d(problem)
    combo_no_sw = combo and problem.sliding_window == 0
    subflags = _enable_transposed_subflags(problem)
    scalar_state = combo or subflags
    skip_legacy_qreg = combo or subflags
    _bias_active = problem.softcap > 0 or problem.use_alibi or problem.use_qq_bias
    mask_opts = (combo_no_sw and not _bias_active) or subflags
    return dict(
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
        use_transposed_scalar_state=scalar_state,
        use_transposed_mask_once=mask_opts,
        use_transposed_mask_limit=mask_opts,
        use_mfma32_skip_legacy_qreg=skip_legacy_qreg,
        use_early_v_schedule=_enable_early_v_schedule(problem),
        use_fast_paged_kv_desc=(
            combo_no_sw
            and not problem.use_fp8
            and problem.num_query_heads == 64
            and problem.num_kv_heads == 8
            and _select_2d_tile_size(problem) == 64
        ),
        use_register_pv=_enable_register_pv(problem),
        use_fp8_mfma_qk=_enable_fp8_mfma_qk(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
    )


def _spec_generic_2d_non_gfx950(problem: UnifiedAttentionProblem):
    """Generic (non-flash) 2D geometry for every NON-gfx950 arch.

    Self-contained per-engine spec builder (GEMM ``spec_fn`` pattern) for the
    ``else`` branch of ``_tiled_spec_from_problem``: it serves gfx942 AND the
    other non-gfx950 arches (gfx1201, gfx1151, ...), which is why it is not named
    for gfx942 alone. It builds only the shared ``_base_2d_generic_fields`` -- the
    gfx950-only schedule fields are omitted deliberately: their helpers hard-gate
    to gfx950 (so they would evaluate off here anyway) AND the fields default to
    off in the spec classes that declare them, so leaving them unset is
    byte-identical to the old ``_spec_field_names``-guarded fallthrough on every
    arch this branch serves. (If a future schedule predicate ever fires off
    gfx950, this branch would need its own handling -- today it does not.)
    Geometry stays in the builder layer; dispatcher identity + C++ parity
    unchanged.
    """
    arch = _kau._resolve_attention_arch()
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl(arch)
    return UnifiedAttention2DTiledSpec(**_base_2d_generic_fields(problem))


def _spec_gfx950_generic(problem: UnifiedAttentionProblem):
    """gfx950 generic 2D geometry -- combo / single-batch schedule + D256 override.

    Self-contained per-engine spec builder (GEMM ``spec_fn`` pattern). Extracted
    from the shared fallthrough of ``_tiled_spec_from_problem`` for gfx950. Because
    this only runs for gfx950 -- whose spec class always declares the schedule
    fields -- the old ``if "<field>" in _spec_field_names`` guards are dropped and
    the fields are set directly (value-conditional helpers preserved verbatim, so
    the result is byte-identical to the guarded fallthrough). The D256 gfx950 fast
    route folds in as a tail ``replace`` -- kept behind the ``_kau.`` module handle
    so ``mock.patch.object(attention_unified, "_d256_gfx950_fast", ...)`` still
    steers it. Geometry stays in the builder layer; dispatcher identity + C++
    parity unchanged.
    """
    arch = _kau._resolve_attention_arch()
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl(arch)
    # gfx950 schedule fields: set directly (no _spec_field_names guard -- the
    # gfx950 spec class always declares them). v_double_buffer / sched_barrier
    # take the helper's value unconditionally; interleave + k_single_buffer stay
    # value-conditional (only set when their helper fires), matching the guarded
    # fallthrough byte-for-byte.
    _schedule_fields = {
        "use_v_double_buffer": _enable_v_double_buffer(problem),
        "use_sched_barrier": _enable_sched_barrier(problem),
    }
    if _enable_softmax_mfma_interleave(problem):
        _schedule_fields["use_softmax_mfma_interleave"] = True
        _schedule_fields["softmax_interleave_mode"] = 1
    if _enable_k_single_buffer(problem):
        _schedule_fields["use_k_single_buffer"] = True
    # Shared base fields + the gfx950-only schedule tail (disjoint keys).
    _spec = UnifiedAttention2DTiledSpec(
        **_base_2d_generic_fields(problem),
        **_schedule_fields,
    )
    if _kau._d256_gfx950_fast(problem):
        # D256 gfx950 bf16 prefill fast route -- pins the 32x32 transposed + FA3
        # softmax<->MFMA-interleave codegen constellation on top of the gated
        # geometry above. Kept behind ``_kau.`` for test-steering (see docstring).
        _spec = replace(_spec, **_kau._d256_gfx950_spec_overrides())
    return _spec


def _tiled_spec_from_problem(
    problem: UnifiedAttentionProblem,
):
    arch = _kau._resolve_attention_arch()
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
        return _spec_gfx942_bf16_flash(problem)
    if _enable_gfx942_fp16_flash(problem) and not _kau._gfx942_4warp_fast(problem):
        return _spec_gfx942_fp16_flash(problem)
    # Generic (non-flash) fallthrough, split by arch. The combo / single-batch
    # schedule / D256 machinery is gfx950-only (its predicates hard-gate to
    # gfx950), so gfx950 gets its own builder and EVERY OTHER arch (gfx942,
    # gfx1201, gfx1151, ...) shares the base-only builder. Both are byte-identical
    # to the prior guarded fallthrough for the arches they serve (see above).
    if arch == "gfx950":
        return _spec_gfx950_generic(problem)
    return _spec_generic_2d_non_gfx950(problem)


def _spec_generic_3d(problem: UnifiedAttentionProblem):
    """gfx942/gfx950 generic 3D split-KV geometry -- the shared fallthrough.

    Self-contained per-engine spec builder (GEMM ``spec_fn`` pattern), extracted
    verbatim from the generic (non-gfx1250) fallthrough of
    ``_tiled_3d_spec_from_problem``. gfx942 and gfx950 share this single path --
    the ``_gfx942_3d_*`` helpers self-gate internally, so no arch split is needed
    (unlike the 2D generic cohort). Geometry stays in the builder layer;
    dispatcher identity + C++ parity unchanged.
    """
    arch = _kau._resolve_attention_arch()
    UnifiedAttention3DTiledSpec, *_ = _tiled_3d_impl(arch)
    tile_size_override = _gfx942_3d_tile_size_override(problem)
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


def _tiled_3d_spec_from_problem(
    problem: UnifiedAttentionProblem,
):
    arch = _kau._resolve_attention_arch()
    UnifiedAttention3DTiledSpec, *_ = _tiled_3d_impl(arch)
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
    # gfx942/gfx950 generic 3D split-KV -- one shared builder (see _spec_generic_3d).
    return _spec_generic_3d(problem)
