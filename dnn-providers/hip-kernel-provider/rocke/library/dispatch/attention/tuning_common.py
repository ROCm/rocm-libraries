# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared machinery for explicit gfx942/gfx950 attention tuning candidates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from builders.common.attention_tuning_builder import (
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
)
from kernels.common.attention_unified import supports_native_unified_attention
from rocke.dispatch.core import Capability, KernelCandidate, OperatorRequest, ShapeRange

from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_FEATURES,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_DTYPES,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
    AttentionTuningSpec,
    FAMILY,
    _problem,
    _request_errors,
)


TUNING_ALGORITHM = "unified_tuning"


@dataclass(frozen=True)
class AttentionGeometryVariant:
    arch: str
    path: str
    codepath: str
    builder_kind: str
    tile_policy: str
    num_warps: int = 1
    block_m_per_warp: int = 16
    num_segments: int = 0
    compile_backend: str = "llvm"

    @property
    def variant_id(self) -> str:
        if self.path == "2d":
            tile = self.tile_policy.replace("x", "xb")
            return (
                f"{self.codepath}_nw{self.num_warps}_mw"
                f"{self.block_m_per_warp}_t{tile}_{self.compile_backend}"
            )
        tile = self.tile_policy.replace("x", "xb")
        return f"{self.codepath}_seg{self.num_segments}_t{tile}"

    @property
    def candidate_name(self) -> str:
        return f"attention_{self.arch}_u{self.path}_{self.variant_id}"

    @property
    def spec_id(self) -> str:
        return f"{self.arch}_u{self.path}_{self.variant_id}"


def _items(values: Mapping[str, object]) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(values.items()))


_GFX950_TRANSPOSED_BASE = {
    "use_mfma_32x32": True,
    "use_transposed_qk_32x32": True,
}
_GFX942_X8_BASE = {"use_mfma_32x32x8": True}
_GFX942_TRANSPOSED_BASE = {
    "use_mfma_32x32x8": True,
    "use_transposed_qk_32x32": True,
}


def _merge(*parts: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for part in parts:
        out.update(part)
    return out


def _gfx950_profile_dicts(codepath: str) -> Iterable[dict[str, object]]:
    """Implemented, non-dead gfx950 schedule profiles.

    Orthogonal value axes (WPE, K-pad and softmax interleave mode) are expanded
    below; dependency-heavy booleans are expressed as named stacks so invalid
    2**N syntax is never materialized.
    """
    if codepath == "narrow":
        profiles = (
            {},
            {"use_register_pv": True},
            {"use_fp8_mfma_qk": True},
            {"use_fp8_mfma_pv": True},
            {"use_fp8_mfma_qk": True, "use_fp8_mfma_pv": True},
            {"use_early_v_schedule": True},
            {"use_v_double_buffer": True},
            {"use_v_double_buffer": True, "use_staggered_iter_wait": True},
        )
        yield from profiles
        return
    if codepath == "wide32":
        base = {"use_mfma_32x32": True}
        profiles = (
            {},
            {"use_early_v_schedule": True},
            {"use_v_double_buffer": True},
            {"use_v_double_buffer": True, "use_staggered_iter_wait": True},
        )
        for profile in profiles:
            yield _merge(base, profile)
        return
    if codepath != "transposed32":
        return

    valu_profiles = (
        {},
        {"use_transposed_scalar_state": True},
        {
            "use_transposed_scalar_state": True,
            "use_transposed_invariant_hoist": True,
            "use_transposed_mask_once": True,
        },
        {
            "use_transposed_scalar_state": True,
            "use_transposed_invariant_hoist": True,
            "use_transposed_mask_once": True,
            "use_transposed_mask_limit": True,
        },
        {
            "use_transposed_scalar_state": True,
            "use_transposed_invariant_hoist": True,
            "use_transposed_mask_once": True,
            "use_transposed_mask_limit": True,
            "use_transposed_half_local_pv": True,
            "use_mfma32_skip_legacy_qreg": True,
        },
    )
    memory_profiles = (
        {},
        {"use_early_v_schedule": True},
        {"use_v_double_buffer": True},
        {"use_v_double_buffer": True, "use_staggered_iter_wait": True},
        {"use_k_single_buffer": True},
        {"use_k_single_buffer": True, "use_q_direct_reg": True},
        {"kv_ring_depth": 3},
        {"use_q_direct_reg": True},
        {"use_grouped_kv2_softmax": True},
        {"use_fast_paged_kv_desc": True},
        {"use_agpr_alloc_zero": True},
    )
    for valu, memory in product(valu_profiles, memory_profiles):
        yield _merge(_GFX950_TRANSPOSED_BASE, valu, memory)


def _gfx942_profile_dicts(codepath: str) -> Iterable[dict[str, object]]:
    if codepath == "narrow":
        yield from (
            {},
            {"use_register_pv": True},
            {"use_early_v_schedule": True},
            {"use_iglp_opt": True},
            {"use_q_major_grid": True},
            {"use_global_load_lds_k": True},
            {"use_fast_paged_kv_desc": True},
            {"use_v_hbm_direct": True},
            {"use_k_hbm_direct": True},
        )
        return
    if codepath == "wide32x8":
        yield from (
            _GFX942_X8_BASE,
            _merge(_GFX942_X8_BASE, {"use_iglp_opt": True}),
            _merge(_GFX942_X8_BASE, {"use_q_major_grid": True}),
        )
        return
    if codepath == "gfx942_4warp":
        yield {}
        return
    if codepath != "transposed_x8":
        return

    valu_profiles = (
        {},
        {"use_transposed_scalar_state": True},
        {
            "use_transposed_scalar_state": True,
            "use_transposed_invariant_hoist": True,
            "use_transposed_mask_once": True,
        },
        {
            "use_transposed_scalar_state": True,
            "use_transposed_invariant_hoist": True,
            "use_transposed_mask_once": True,
            "use_transposed_mask_limit": True,
        },
    )
    memory_profiles = [
        {},
        {"use_conflict_free_v": True},
        {"use_conflict_free_v_store": True},
        {
            "use_conflict_free_v_store": True,
            "use_conflict_free_v_store_split": False,
        },
        {
            "use_conflict_free_v_store": True,
            "use_conflict_free_v_ck_vlds": False,
        },
        {"use_k_single_buffer": True},
        {"use_q_direct_global": True},
        {"use_v_hbm_direct": True},
        {"use_k_hbm_direct": True},
        {"use_global_load_lds_k": True},
        {"use_q_major_grid": True},
        {"use_causal_mask_phase_split": True},
        {"use_agpr_alloc_zero": True},
        {"use_iglp_opt": True},
        {"use_qk_pv_sched_group_barrier": True},
    ]
    for depth in (2, 3):
        for width in (8, 16, 32, 64):
            memory_profiles.append(
                {
                    "use_conflict_free_v_store": True,
                    "use_k_sliced_ring": True,
                    "ring_depth": depth,
                    "k_slice_hd": width,
                }
            )
            if depth == 3:
                memory_profiles.append(
                    {
                        "use_conflict_free_v_store": True,
                        "use_k_sliced_ring": True,
                        "ring_depth": depth,
                        "k_slice_hd": width,
                        "use_k_sliced_ldsseq": True,
                    }
                )
    for cache_policy in ("stream", "default", "last_use"):
        memory_profiles.append({"kv_cache_policy": cache_policy})
    for valu, memory in product(valu_profiles, memory_profiles):
        yield _merge(_GFX942_TRANSPOSED_BASE, valu, memory)


def _explicit_configs(
    problem,
    variant: AttentionGeometryVariant,
) -> Iterable[AttentionTuningSpec]:
    if variant.path == "3d":
        knob_sets: Sequence[Mapping[str, object]]
        if variant.arch == "gfx942":
            knob_sets = (
                {},
                {"use_invariant_hoist": True},
                {"use_wide_kv_load": True},
                {"use_invariant_hoist": True, "use_wide_kv_load": True},
            )
        else:
            knob_sets = ({},)
        for wpe, knobs in product((None, 1, 2, 3, 4), knob_sets):
            config = ExplicitAttention3DConfig(
                num_segments=variant.num_segments,
                tile_policy=variant.tile_policy,
                waves_per_eu=wpe,
                knobs=_items(knobs),
            )
            try:
                segment, reduce = make_explicit_attention_3d_specs(
                    problem, config, arch=variant.arch
                )
            except (ValueError, NotImplementedError):
                continue
            tuning_id = (
                f"{variant.variant_id}_wpe{wpe if wpe is not None else 'none'}"
                f"_{segment.kernel_name()}"
            )
            yield AttentionTuningSpec(
                path="3d",
                arch=variant.arch,
                builder_kind="tiled_3d",
                compile_backend="llvm",
                candidate_name=variant.candidate_name,
                tuning_id=tuning_id,
                kernel_spec=segment,
                reduce_spec=reduce,
            )
        return

    profiles = (
        _gfx950_profile_dicts(variant.codepath)
        if variant.arch == "gfx950"
        else _gfx942_profile_dicts(variant.codepath)
    )
    for profile in profiles:
        pad_options: Sequence[Optional[int]] = (None,)
        if variant.arch == "gfx950" and variant.codepath != "narrow":
            pad_options = (None, 8, 16)
        interleave_options: Sequence[Optional[Tuple[int, int]]] = (None,)
        if variant.arch == "gfx950" and variant.codepath == "transposed32":
            interleave_options = (None, (0, 1), (1, 1), (2, 4))
        mask_phase_options = (
            (False, True)
            if variant.arch == "gfx950" and variant.codepath == "transposed32"
            else (False,)
        )
        for wpe, pad, interleave, mask_phase in product(
            (None, 1, 2, 3, 4),
            pad_options,
            interleave_options,
            mask_phase_options,
        ):
            knobs = dict(profile)
            if pad is not None:
                knobs.update(use_kq_lds_pad=True, kq_lds_pad_halves=pad)
            if interleave is not None:
                mode, groups = interleave
                knobs.update(
                    use_softmax_mfma_interleave=True,
                    softmax_interleave_mode=mode,
                    softmax_interleave_groups=groups,
                )
            if mask_phase:
                knobs["use_mask_phase_split"] = True
            config = ExplicitAttention2DConfig(
                num_warps=variant.num_warps,
                block_m_per_warp=variant.block_m_per_warp,
                tile_policy=variant.tile_policy,
                waves_per_eu=wpe,
                compile_backend=variant.compile_backend,
                builder_kind=variant.builder_kind,
                knobs=_items(knobs),
            )
            try:
                kernel_spec = make_explicit_attention_2d_spec(
                    problem, config, arch=variant.arch
                )
            except (ValueError, NotImplementedError):
                continue
            tuning_id = (
                f"{variant.variant_id}_wpe{wpe if wpe is not None else 'none'}"
                f"_{kernel_spec.kernel_name()}"
            )
            yield AttentionTuningSpec(
                path="2d",
                arch=variant.arch,
                builder_kind=variant.builder_kind,
                compile_backend=variant.compile_backend,
                candidate_name=variant.candidate_name,
                tuning_id=tuning_id,
                kernel_spec=kernel_spec,
            )


def tuning_specs(
    req: AttentionRequest, variant: AttentionGeometryVariant
) -> Tuple[AttentionTuningSpec, ...]:
    problem = _problem(req)
    seen = set()
    specs = []
    for spec in _explicit_configs(problem, variant):
        key = (
            spec.builder_kind,
            spec.compile_backend,
            spec.kernel_name(),
            repr(spec.kernel_spec),
            repr(spec.reduce_spec),
        )
        if key not in seen:
            seen.add(key)
            specs.append(spec)
    return tuple(specs)


def _find_tuning_spec(
    req: AttentionRequest,
    variant: AttentionGeometryVariant,
    tuning_id: str,
) -> Optional[AttentionTuningSpec]:
    """Find one valid point without materializing the whole sweep space."""
    for spec in _explicit_configs(_problem(req), variant):
        if tuning_id == "auto" or spec.tuning_id == tuning_id:
            return spec
    return None


def _opted_in(req: AttentionRequest, variant: AttentionGeometryVariant) -> bool:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", TUNING_ALGORITHM):
        return False
    if spec_id not in ("auto", variant.spec_id):
        return False
    return algorithm == TUNING_ALGORITHM or spec_id == variant.spec_id


def make_tuning_candidate(
    variant: AttentionGeometryVariant,
) -> KernelCandidate:
    def support(req: OperatorRequest):
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        if not _opted_in(req, variant):
            return False, "explicit unified tuning candidate is opt-in"
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem, arch=variant.arch)
        if not ok:
            return False, why
        tuning_id = req.attention_tuning_id.strip()
        if _find_tuning_spec(req, variant, tuning_id) is None:
            if tuning_id != "auto":
                return False, f"unknown or invalid attention_tuning_id {tuning_id!r}"
            return False, "no dependency-valid tuning spec for this request"
        return True, "ok"

    def sweep(req: OperatorRequest):
        ok, _ = candidate.admits(req)
        if not ok:
            return ()
        assert isinstance(req, AttentionRequest)
        return tuning_specs(req, variant)

    def select(req: OperatorRequest):
        assert isinstance(req, AttentionRequest)
        tuning_id = req.attention_tuning_id.strip()
        spec = _find_tuning_spec(req, variant, tuning_id)
        if spec is None:
            raise ValueError(f"{variant.candidate_name} does not support request")
        return spec

    def build(spec, arch):
        from builders.common.attention_tuning_builder import (
            build_explicit_attention_2d,
            build_explicit_attention_3d,
        )

        if spec.path == "3d":
            return build_explicit_attention_3d(
                spec.kernel_spec, spec.reduce_spec, arch=arch
            )
        if spec.builder_kind == "gfx942_4warp_gqa":
            from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

            return build_gfx942_4warp_gqa(spec.kernel_spec, arch=arch)
        return build_explicit_attention_2d(spec.kernel_spec, arch=arch)

    def signature(spec):
        from kernels.common.attention_unified import (
            _3d_signature,
            _attn_signature,
            _kv_storage_dtype,
        )

        if spec.path == "3d":
            return _3d_signature(
                spec.kernel_spec.dtype,
                kv_dtype=spec.kernel_spec.kv_storage_dtype,
            )
        return _attn_signature(
            spec.kernel_spec.dtype,
            include_bt_stride=True,
            include_qq_bias_stride=True,
            kv_dtype=spec.kernel_spec.kv_storage_dtype,
        )

    def grid(spec, req):
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        ks = spec.kernel_spec
        if spec.path == "3d":
            block_q = max(1, 16 // problem.num_queries_per_kv)
            qblocks = problem.total_q // block_q + problem.num_seqs
            return (qblocks, problem.num_kv_heads, ks.num_segments)
        if spec.builder_kind == "gfx942_4warp_gqa":
            qblocks = problem.total_q // 128 + problem.num_seqs
            return (problem.num_query_heads, qblocks, 1)
        block_q = (
            ks.block_m // problem.num_queries_per_kv
            if problem.num_queries_per_kv <= ks.block_m
            else 1
        )
        qblocks = problem.total_q // block_q + problem.num_seqs
        if bool(getattr(ks, "use_q_major_grid", False)):
            return (qblocks, problem.num_kv_heads, 1)
        return (problem.num_kv_heads, qblocks, 1)

    def block(spec):
        if spec.path == "3d":
            return (64, 1, 1)
        if spec.builder_kind == "gfx942_4warp_gqa":
            return (256, 1, 1)
        return (64 * spec.kernel_spec.num_warps, 1, 1)

    candidate = KernelCandidate(
        name=variant.candidate_name,
        family=FAMILY,
        algorithm=TUNING_ALGORITHM,
        spec_id=variant.spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=30,
        capability=Capability(
            arches=(variant.arch,),
            dtypes=UNIFIED_DTYPES,
            shapes=(
                ShapeRange("hdim_q", allowed=UNIFIED_HEAD_SIZES),
                ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
            ),
            supports_features=ATTENTION_FEATURES,
        ),
        _supports=support,
        select_spec=select,
        signature=signature,
        grid=grid,
        block=block,
        sweep_space=sweep,
        build=build,
    )
    return candidate
