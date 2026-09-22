# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared machinery for explicit gfx942/gfx950 attention tuning candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from itertools import product
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .tuning_specs import (
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
)
from kernels.common.attention_unified import supports_native_unified_attention
from rocke.dispatch.core import (
    Capability,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
    stable_json_hash,
)

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
# Geometry axis is always present; 2D micro-expansion uses a reduced WPE set so
# the per-shape space stays in the low thousands rather than ~70K.
_SWEEP_WAVES_2D: Tuple[Optional[int], ...] = (None, 2, 4)
_SWEEP_WAVES_3D: Tuple[Optional[int], ...] = (None, 1, 2, 3, 4)


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
_SCHED_BARRIER_MASKS = (0, 0x008, 0x108)
_GFX942_X8_BASE = {"use_mfma_32x32x8": True}
_GFX942_TRANSPOSED_BASE = {
    "use_mfma_32x32x8": True,
    "use_transposed_qk_32x32": True,
}
_INTERLEAVE_STACKS = frozenset({"baseline", "r4_hlpv", "vdbuf"})


def _merge(*parts: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for part in parts:
        out.update(part)
    return out


def _named(name: str, knobs: Mapping[str, object]) -> dict[str, object]:
    payload = dict(knobs)
    payload["_stack"] = name
    return payload


def _gfx950_tuning_lds_bytes(spec) -> int:
    """Dispatcher-side static LDS model for one explicit gfx950 tuning spec."""
    tile = int(spec.tile_size_eff)
    head = int(spec.head_size)
    block_m = int(spec.block_m)
    kv_fp8 = spec.kv_storage_dtype == "fp8e4m3"
    fp8_qk = kv_fp8 and bool(spec.use_fp8_mfma_qk)
    fp8_pv = kv_fp8 and bool(spec.use_fp8_mfma_pv)
    k_elem_bytes = 1 if fp8_qk else 2
    v_elem_bytes = 1 if fp8_pv else 2
    k_bufs = (
        1
        if spec.use_k_single_buffer
        else int(spec.kv_ring_depth) if int(spec.kv_ring_depth) > 2 else 2
    )
    v_bufs = 2 if spec.use_v_double_buffer else 1

    pad = int(spec.kq_lds_pad_halves) if spec.use_kq_lds_pad else 0
    slab_rows = (512 // head) if pad and 512 % head == 0 else 0
    pad_active = bool(
        pad
        and not fp8_qk
        and slab_rows
        and tile % slab_rows == 0
        and pad % 8 == 0
        and k_bufs == 1
    )
    if pad_active:
        k_bytes = k_bufs * (tile // slab_rows) * (slab_rows * head + pad) * k_elem_bytes
    else:
        k_bytes = k_bufs * tile * head * k_elem_bytes
    v_bytes = v_bufs * tile * head * v_elem_bytes

    transposed_register_p = bool(spec.use_mfma_32x32 and spec.use_transposed_qk_32x32)
    p_bytes = (
        0
        if spec.use_register_pv or transposed_register_p
        else block_m * (tile + (16 if fp8_pv else 8)) * v_elem_bytes
    )

    q_bytes = block_m * head * 2
    q_aliases_k = bool(not fp8_qk and q_bytes <= 2 * tile * head * k_elem_bytes)
    if spec.use_q_reread or spec.use_q_direct_reg:
        q_aliases_k = False
    q_lds_bytes = 0 if spec.use_q_direct_reg or q_aliases_k else q_bytes

    out_stripe_cols = 32 if head <= 64 else head
    acc_bytes = block_m * out_stripe_cols * 2
    fp8_staging_bytes = 3 * tile * head if kv_fp8 and spec.use_fp8_mfma_qk else 0
    return k_bytes + v_bytes + p_bytes + q_lds_bytes + acc_bytes + fp8_staging_bytes


def _supports_tuning_spec(
    variant: AttentionGeometryVariant, kernel_spec
) -> Tuple[bool, str]:
    """Residual per-spec support that cannot be represented by Capability."""
    if variant.arch != "gfx950" or variant.path != "2d":
        return True, "supported"

    if kernel_spec.use_kq_lds_pad:
        head = int(kernel_spec.head_size)
        tile = int(kernel_spec.tile_size_eff)
        pad = int(kernel_spec.kq_lds_pad_halves)
        slab_rows = (512 // head) if 512 % head == 0 else 0
        if kernel_spec.kv_storage_dtype == "fp8e4m3" and kernel_spec.use_fp8_mfma_qk:
            return False, "KQ LDS pad does not support native-FP8 K LDS"
        if not slab_rows or tile % slab_rows != 0 or pad % 8 != 0:
            return (
                False,
                "KQ LDS pad requires an aligned slab layout "
                f"(head_size={head}, tile_size={tile}, pad={pad})",
            )
        k_bufs = (
            1
            if kernel_spec.use_k_single_buffer
            else (
                int(kernel_spec.kv_ring_depth)
                if int(kernel_spec.kv_ring_depth) > 2
                else 2
            )
        )
        if k_bufs != 1:
            return False, "KQ LDS pad requires a single-K schedule"
        q_bytes = int(kernel_spec.block_m) * head * 2
        q_aliases_k = bool(
            q_bytes <= 2 * tile * head * 2
            and not kernel_spec.use_q_reread
            and not kernel_spec.use_q_direct_reg
        )
        if q_aliases_k:
            return False, "padded K LDS does not support aliased Q"

    from rocke.core.arch import ArchTarget

    capacity = ArchTarget.from_gfx(variant.arch).lds_capacity_bytes
    lds_bytes = _gfx950_tuning_lds_bytes(kernel_spec)
    if lds_bytes > capacity:
        return (
            False,
            f"estimated LDS {lds_bytes} B exceeds the {variant.arch} "
            f"{capacity} B LDS budget; hipcc/comgr codegen would fail",
        )
    return True, "supported"


def _gfx950_profile_dicts(codepath: str) -> Iterable[dict[str, object]]:
    """Named schedule stacks. Invalid 2**N syntax is never materialized."""
    if codepath == "narrow":
        profiles = (
            _named("baseline", {}),
            _named("regpv", {"use_register_pv": True}),
            _named("fp8qk", {"use_fp8_mfma_qk": True}),
            _named("fp8pv", {"use_fp8_mfma_pv": True}),
            _named("fp8both", {"use_fp8_mfma_qk": True, "use_fp8_mfma_pv": True}),
            _named("early_v", {"use_early_v_schedule": True}),
            _named("vdbuf", {"use_v_double_buffer": True}),
            _named(
                "vdbuf_stgw",
                {"use_v_double_buffer": True, "use_staggered_iter_wait": True},
            ),
        )
        yield from profiles
        for mask in _SCHED_BARRIER_MASKS:
            yield _named(
                f"schedb_{mask:#x}",
                {"use_sched_barrier": True, "sched_barrier_mask": mask},
            )
            yield _named(
                f"vdbuf_schedb_{mask:#x}",
                {
                    "use_v_double_buffer": True,
                    "use_sched_barrier": True,
                    "sched_barrier_mask": mask,
                },
            )
        return
    if codepath == "wide32":
        base = {"use_mfma_32x32": True}
        for name, extra in (
            ("baseline", {}),
            ("early_v", {"use_early_v_schedule": True}),
            ("vdbuf", {"use_v_double_buffer": True}),
            (
                "vdbuf_stgw",
                {"use_v_double_buffer": True, "use_staggered_iter_wait": True},
            ),
        ):
            yield _named(name, _merge(base, extra))
        return
    if codepath != "transposed32":
        return

    base = dict(_GFX950_TRANSPOSED_BASE)
    r4_s1 = _merge(
        base,
        {
            "use_transposed_scalar_state": True,
            "use_transposed_invariant_hoist": True,
            "use_transposed_mask_once": True,
        },
    )
    r4_mlim = _merge(r4_s1, {"use_transposed_mask_limit": True})
    r4_hlpv = _merge(
        r4_mlim,
        {
            "use_transposed_half_local_pv": True,
            "use_mfma32_skip_legacy_qreg": True,
        },
    )
    stacks = (
        ("baseline", base),
        ("scalar", _merge(base, {"use_transposed_scalar_state": True})),
        ("r4_s1", r4_s1),
        ("r4_s1_mlim", r4_mlim),
        ("r4_hlpv", r4_hlpv),
        ("early_v", _merge(base, {"use_early_v_schedule": True})),
        ("vdbuf", _merge(base, {"use_v_double_buffer": True})),
        (
            "vdbuf_stgw",
            _merge(
                base,
                {"use_v_double_buffer": True, "use_staggered_iter_wait": True},
            ),
        ),
        ("ksb", _merge(base, {"use_k_single_buffer": True})),
        (
            "ksb_qdreg",
            _merge(base, {"use_k_single_buffer": True, "use_q_direct_reg": True}),
        ),
        ("ring3", _merge(base, {"kv_ring_depth": 3})),
        ("qdreg", _merge(base, {"use_q_direct_reg": True})),
        ("qrr", _merge(base, {"use_q_reread": True})),
        ("gkv2", _merge(base, {"use_grouped_kv2_softmax": True})),
        ("fastkv", _merge(base, {"use_fast_paged_kv_desc": True})),
        ("r4_mlim_vdbuf", _merge(r4_mlim, {"use_v_double_buffer": True})),
        ("r4_hlpv_qrr", _merge(r4_hlpv, {"use_q_reread": True})),
        ("r4_mlim_phase", _merge(r4_mlim, {"use_mask_phase_split": True})),
        ("r4_hlpv_agpr0", _merge(r4_hlpv, {"use_agpr_alloc_zero": True})),
    )
    for name, knobs in stacks:
        yield _named(name, knobs)


def _gfx942_profile_dicts(codepath: str) -> Iterable[dict[str, object]]:
    if codepath == "narrow":
        # use_k_hbm_direct is omitted: the 16x16 QK loop always reads K_lds, so
        # khbm (which skips K staging) is numerically wrong on this path.
        for name, knobs in (
            ("baseline", {}),
            ("regpv", {"use_register_pv": True}),
            ("early_v", {"use_early_v_schedule": True}),
            ("iglp", {"use_iglp_opt": True}),
            ("qmajor", {"use_q_major_grid": True}),
            ("gldsk", {"use_global_load_lds_k": True}),
            ("fastkv", {"use_fast_paged_kv_desc": True}),
            ("vhbm", {"use_v_hbm_direct": True}),
        ):
            yield _named(name, knobs)
        return
    if codepath == "wide32x8":
        yield _named("baseline", _GFX942_X8_BASE)
        yield _named("iglp", _merge(_GFX942_X8_BASE, {"use_iglp_opt": True}))
        yield _named("qmajor", _merge(_GFX942_X8_BASE, {"use_q_major_grid": True}))
        return
    if codepath == "gfx942_4warp":
        yield _named("baseline", {})
        return
    if codepath != "transposed_x8":
        return

    base = dict(_GFX942_TRANSPOSED_BASE)
    r4_s1 = _merge(
        base,
        {
            "use_transposed_scalar_state": True,
            "use_transposed_invariant_hoist": True,
            "use_transposed_mask_once": True,
        },
    )
    r4_mlim = _merge(r4_s1, {"use_transposed_mask_limit": True})
    stacks = [
        ("baseline", base),
        ("scalar", _merge(base, {"use_transposed_scalar_state": True})),
        ("r4_s1", r4_s1),
        ("r4_s1_mlim", r4_mlim),
        ("cfv", _merge(base, {"use_conflict_free_v": True})),
        ("cfvst", _merge(base, {"use_conflict_free_v_store": True})),
        (
            "cfvst_nosplit",
            _merge(
                base,
                {
                    "use_conflict_free_v_store": True,
                    "use_conflict_free_v_store_split": False,
                },
            ),
        ),
        ("ksb", _merge(base, {"use_k_single_buffer": True})),
        ("qdglob", _merge(base, {"use_q_direct_global": True})),
        ("vhbm", _merge(base, {"use_v_hbm_direct": True})),
        ("khbm", _merge(base, {"use_k_hbm_direct": True})),
        ("gldsk", _merge(base, {"use_global_load_lds_k": True})),
        ("qmajor", _merge(base, {"use_q_major_grid": True})),
        ("cphase", _merge(base, {"use_causal_mask_phase_split": True})),
        ("agpr0", _merge(base, {"use_agpr_alloc_zero": True})),
        ("iglp", _merge(base, {"use_iglp_opt": True})),
        ("schedg", _merge(base, {"use_qk_pv_sched_group_barrier": True})),
        (
            "ring_d2_w32",
            _merge(
                base,
                {
                    "use_conflict_free_v_store": True,
                    "use_k_sliced_ring": True,
                    "ring_depth": 2,
                    "k_slice_hd": 32,
                },
            ),
        ),
        (
            "ring_d3_w32",
            _merge(
                base,
                {
                    "use_conflict_free_v_store": True,
                    "use_k_sliced_ring": True,
                    "ring_depth": 3,
                    "k_slice_hd": 32,
                },
            ),
        ),
        (
            "ring_d3_w32_ldsseq",
            _merge(
                base,
                {
                    "use_conflict_free_v_store": True,
                    "use_k_sliced_ring": True,
                    "ring_depth": 3,
                    "k_slice_hd": 32,
                    "use_k_sliced_ldsseq": True,
                },
            ),
        ),
        ("kvcp_nt", _merge(base, {"kv_cache_policy": "nt"})),
    ]
    for name, knobs in stacks:
        yield _named(name, knobs)


def _canonical_payload(
    *,
    arch: str,
    path: str,
    builder_kind: str,
    compile_backend: str,
    fp8_fnuz: bool,
    kernel_spec,
    reduce_spec,
) -> dict:
    def _as_payload(value):
        if value is None:
            return None
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        return repr(value)

    return {
        "abi": ATTENTION_ABI_VERSION,
        "arch": arch,
        "path": path,
        "builder_kind": builder_kind,
        "compile_backend": compile_backend,
        "fp8_fnuz": bool(fp8_fnuz),
        "kernel_spec": _as_payload(kernel_spec),
        "reduce_spec": _as_payload(reduce_spec),
    }


def _make_tuning_id(
    variant: AttentionGeometryVariant,
    kernel_spec,
    reduce_spec=None,
    *,
    fp8_fnuz: bool = False,
) -> str:
    wpe = getattr(kernel_spec, "waves_per_eu", None)
    digest = stable_json_hash(
        _canonical_payload(
            arch=variant.arch,
            path=variant.path,
            builder_kind=variant.builder_kind,
            compile_backend=variant.compile_backend,
            fp8_fnuz=fp8_fnuz,
            kernel_spec=kernel_spec,
            reduce_spec=reduce_spec,
        ),
        n=16,
    )
    return f"{variant.variant_id}_wpe{wpe if wpe is not None else 'none'}@{digest}"


def retarget_tuning_spec(
    spec: AttentionTuningSpec, *, num_kv_blocks: int
) -> AttentionTuningSpec:
    """Refresh i32/i64 paged addressing once the physical cache is known."""
    count = int(num_kv_blocks)
    if count < 0:
        raise ValueError("num_kv_blocks must be non-negative")
    kernel_spec = spec.kernel_spec
    if not hasattr(kernel_spec, "use_i64_kv_addr"):
        return replace(spec, num_kv_blocks=count)

    elem_bytes = 1 if kernel_spec.kv_storage_dtype == "fp8e4m3" else 2
    block_stride = (
        int(kernel_spec.block_size)
        * int(kernel_spec.num_kv_heads)
        * int(kernel_spec.head_size)
        * elem_bytes
    )
    use_i64 = count > 0 and count * block_stride > 0x8000_0000
    refreshed_kernel = replace(kernel_spec, use_i64_kv_addr=use_i64)
    if refreshed_kernel == kernel_spec and int(spec.num_kv_blocks) == count:
        return spec

    prefix = spec.tuning_id.rsplit("@", 1)[0]
    digest = stable_json_hash(
        _canonical_payload(
            arch=spec.arch,
            path=spec.path,
            builder_kind=spec.builder_kind,
            compile_backend=spec.compile_backend,
            fp8_fnuz=spec.fp8_fnuz,
            kernel_spec=refreshed_kernel,
            reduce_spec=spec.reduce_spec,
        ),
        n=16,
    )
    return replace(
        spec,
        tuning_id=f"{prefix}@{digest}",
        kernel_spec=refreshed_kernel,
        num_kv_blocks=count,
    )


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
        for wpe, knobs in product(_SWEEP_WAVES_3D, knob_sets):
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
            yield AttentionTuningSpec(
                path="3d",
                arch=variant.arch,
                builder_kind="tiled_3d",
                compile_backend="llvm",
                candidate_name=variant.candidate_name,
                tuning_id=_make_tuning_id(
                    variant,
                    segment,
                    reduce,
                    fp8_fnuz=bool(problem.fp8_fnuz),
                ),
                kernel_spec=segment,
                fp8_fnuz=bool(problem.fp8_fnuz),
                num_kv_blocks=int(problem.num_kv_blocks),
                reduce_spec=reduce,
            )
        return

    profiles = (
        _gfx950_profile_dicts(variant.codepath)
        if variant.arch == "gfx950"
        else _gfx942_profile_dicts(variant.codepath)
    )
    for profile in profiles:
        stack = str(profile.get("_stack", ""))
        knobs_base = {k: v for k, v in profile.items() if k != "_stack"}
        pad_options: Sequence[Optional[int]] = (None,)
        if (
            variant.arch == "gfx950"
            and variant.codepath != "narrow"
            and knobs_base.get("use_k_single_buffer")
            and (knobs_base.get("use_q_direct_reg") or knobs_base.get("use_q_reread"))
        ):
            pad_options = (None, 16)
        interleave_options: Sequence[Optional[Tuple[int, int]]] = (None,)
        if (
            variant.arch == "gfx950"
            and variant.codepath == "transposed32"
            and stack in _INTERLEAVE_STACKS
            and not knobs_base.get("use_sched_barrier")
        ):
            interleave_options = (None, (2, 4))
        for wpe, pad, interleave in product(
            _SWEEP_WAVES_2D, pad_options, interleave_options
        ):
            knobs = dict(knobs_base)
            if pad is not None:
                knobs.update(use_kq_lds_pad=True, kq_lds_pad_halves=pad)
            if interleave is not None:
                mode, groups = interleave
                knobs.update(
                    use_softmax_mfma_interleave=True,
                    softmax_interleave_mode=mode,
                    softmax_interleave_groups=groups,
                )
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
            ok, _why = _supports_tuning_spec(variant, kernel_spec)
            if not ok:
                continue
            yield AttentionTuningSpec(
                path="2d",
                arch=variant.arch,
                builder_kind=variant.builder_kind,
                compile_backend=variant.compile_backend,
                candidate_name=variant.candidate_name,
                tuning_id=_make_tuning_id(
                    variant,
                    kernel_spec,
                    fp8_fnuz=bool(problem.fp8_fnuz),
                ),
                kernel_spec=kernel_spec,
                fp8_fnuz=bool(problem.fp8_fnuz),
                num_kv_blocks=int(problem.num_kv_blocks),
            )


def iter_tuning_specs(
    req: AttentionRequest, variant: AttentionGeometryVariant
) -> Iterable[AttentionTuningSpec]:
    seen: set[str] = set()
    for spec in _explicit_configs(_problem(req), variant):
        if spec.tuning_id not in seen:
            seen.add(spec.tuning_id)
            yield spec


def tuning_specs(
    req: AttentionRequest, variant: AttentionGeometryVariant
) -> Tuple[AttentionTuningSpec, ...]:
    return tuple(iter_tuning_specs(req, variant))


def _find_tuning_spec(
    req: AttentionRequest,
    variant: AttentionGeometryVariant,
    tuning_id: str,
) -> Optional[AttentionTuningSpec]:
    """Find one valid point without materializing the whole sweep space."""
    wanted = tuning_id.strip()
    for spec in _explicit_configs(_problem(req), variant):
        if wanted in ("auto", "") or spec.tuning_id == wanted:
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
        from .tuning_specs import (
            build_explicit_attention_2d,
            build_explicit_attention_3d,
            validate_explicit_fp8_encoding,
        )

        if bool(spec.fp8_fnuz) != bool(getattr(spec.kernel_spec, "fp8_fnuz", False)):
            raise ValueError("tuning wrapper and kernel spec disagree on fp8_fnuz")
        validate_explicit_fp8_encoding(
            arch=arch,
            use_fp8=spec.kernel_spec.kv_storage_dtype == "fp8e4m3",
            fp8_fnuz=spec.fp8_fnuz,
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

    def bind_torch(request, spec, tensors, **kwargs):
        from .bindings import bind_tuning_attention_torch

        payload = dict(tensors)
        if "problem" not in payload:
            payload["problem"] = _problem(request)
        return bind_tuning_attention_torch(
            request,
            spec,
            payload,
            grid=grid(spec, request),
            block=block(spec),
            **kwargs,
        )

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
        bind_torch=bind_torch,
    )
    return candidate
