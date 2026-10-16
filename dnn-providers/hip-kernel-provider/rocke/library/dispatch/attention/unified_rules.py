# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unified tiled-attention tuning: geometry variants and their knob space.

A variant fixes the geometry (codepath, tile, warps, segments, backend); the
knobs on its axes vary the rest. :class:`UnifiedSpace` states the unified
rules as the hooks of the shared :class:`rocke.dispatch.tuning.KnobSpace`,
which makes every spec -- production stacks, the full walk, the sampler, knob
pins and id search all go through its ``canonicalize``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields as _dataclass_fields
from functools import lru_cache
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Tuple

from .axes import (
    DEAD_END_KNOBS,
    KNOWN_WRONG_KNOBS,
    _CODEPATH_KNOBS,
    _NARROW_ONLY_KNOBS,
    _PROD_INTERLEAVE,
    _PROD_INTERLEAVE_STACKS,
    _PROD_PAD,
    _PROD_PAD_STACKS,
    _PRODUCTION_STACKS,
    _TRANSPOSED_ONLY_KNOBS,
    _items,
    tuning_axes,
)
from .common import (
    ATTENTION_ABI_VERSION,
    AttentionRequest,
    AttentionTuningSpec,
    _problem,
)
from .tuning_specs import (
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
)
from .waves import FULL_WAVES, PRODUCTION_WAVES, WavesPerEuSpace

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


def _policy_conflict(arch: str, knobs: Mapping[str, object]) -> Optional[str]:
    if not knobs.get("use_transposed_qk_32x32"):
        for name in _TRANSPOSED_ONLY_KNOBS.get(arch, ()):
            if knobs.get(name):
                return f"{name} is only emitted on the transposed-32x32 path"
    if knobs.get("use_mfma_32x32"):
        for name in _NARROW_ONLY_KNOBS.get(arch, ()):
            if knobs.get(name):
                return f"{name} is only emitted in the 16x16 QK loop"
    return None


def _explicit_3d_specs(problem, variant, knobs, wpe):
    config = ExplicitAttention3DConfig(
        num_segments=variant.num_segments,
        tile_policy=variant.tile_policy,
        waves_per_eu=wpe,
        knobs=_items(knobs),
    )
    return make_explicit_attention_3d_specs(problem, config, arch=variant.arch)


def _explicit_2d_spec(problem, variant, knobs, wpe):
    config = ExplicitAttention2DConfig(
        num_warps=variant.num_warps,
        block_m_per_warp=variant.block_m_per_warp,
        tile_policy=variant.tile_policy,
        waves_per_eu=wpe,
        compile_backend=variant.compile_backend,
        builder_kind=variant.builder_kind,
        knobs=_items(knobs),
    )
    return make_explicit_attention_2d_spec(problem, config, arch=variant.arch)


def _variant_axes(variant: AttentionGeometryVariant):
    if variant.builder_kind == "gfx942_4warp_gqa":
        return ()  # the 4-warp GQA builder reads no tuning knobs
    return tuning_axes(variant.arch, variant.path)


@lru_cache(maxsize=None)
def _field_defaults(spec_type: type) -> Mapping[str, object]:
    """Declared field defaults (``MISSING`` for required fields)."""
    return {f.name: f.default for f in _dataclass_fields(spec_type)}


@lru_cache(maxsize=None)
def _declared_defaults(spec_type: type) -> Tuple[Tuple[str, object], ...]:
    """The fields with a declared default -- the ones knobs are a delta
    against -- for the defaults fingerprint."""
    return tuple(
        (name, value)
        for name, value in _field_defaults(spec_type).items()
        if value is not MISSING
    )


def _other_codepath_knobs(arch: str, knobs: Mapping[str, object]) -> frozenset:
    """The set knobs in ``knobs`` that only another codepath's body emits (see
    :func:`_policy_conflict`); on this one they re-emit the same kernel."""
    inert = set()
    if not knobs.get("use_transposed_qk_32x32"):
        inert.update(k for k in _TRANSPOSED_ONLY_KNOBS.get(arch, ()) if knobs.get(k))
    if knobs.get("use_mfma_32x32"):
        inert.update(k for k in _NARROW_ONLY_KNOBS.get(arch, ()) if knobs.get(k))
    return frozenset(inert)


def _production_knob_sets(variant: AttentionGeometryVariant):
    """Curated stacks for one geometry, plus their pad / interleave variants.

    A stack that turns on a dead-end or known-wrong knob is a data error: the
    production level exists to keep those out.
    """
    stacks = _PRODUCTION_STACKS.get((variant.arch, variant.codepath))
    if not stacks:
        return
    base = dict(_CODEPATH_KNOBS.get((variant.arch, variant.codepath), {}))
    banned = DEAD_END_KNOBS.get(variant.arch, frozenset()) | KNOWN_WRONG_KNOBS.get(
        variant.arch, frozenset()
    )
    for name, overrides in stacks:
        knobs = dict(base)
        knobs.update(overrides)
        if any(knobs.get(knob) for knob in banned):
            raise ValueError(
                f"production stack {variant.arch}/{variant.codepath}/{name} "
                f"sets a dead-end knob"
            )
        yield knobs
        if (
            name in _PROD_PAD_STACKS
            and variant.arch == "gfx950"
            and variant.codepath != "narrow"
        ):
            yield {**knobs, **_PROD_PAD}
        if (
            name in _PROD_INTERLEAVE_STACKS
            and variant.arch == "gfx950"
            and variant.codepath == "transposed32"
        ):
            yield {**knobs, **_PROD_INTERLEAVE}


def _sets_dead_end(spec: AttentionTuningSpec) -> bool:
    banned = DEAD_END_KNOBS.get(spec.arch, frozenset())
    return any(getattr(spec.kernel_spec, knob, False) for knob in banned)


# gfx950 2D LDS rules. The tiled 2D validator does not model LDS, so an
# over-budget spec only fails at codegen, and the KQ pad has cases the kernel
# silently disables or computes wrongly. These mirror attention_tiled_2d.py.
_KQ_PAD_KNOBS = ("use_kq_lds_pad", "kq_lds_pad_halves")


def _k_schedule_bufs(spec) -> int:
    if spec.use_k_single_buffer:
        return 1
    depth = int(spec.kv_ring_depth)
    return depth if depth > 2 else 2


def _fp8_mfma_qk(spec) -> bool:
    return spec.kv_storage_dtype == "fp8e4m3" and bool(spec.use_fp8_mfma_qk)


def _kq_pad_slab_rows(spec) -> int:
    head = int(spec.head_size)
    return 512 // head if head and 512 % head == 0 else 0


def _kq_pad_off_reason(spec) -> Optional[str]:
    """Why the kernel emits a set KQ pad as no pad at all, or None."""
    if _fp8_mfma_qk(spec):
        return "the KQ LDS pad is off under native-FP8 K LDS"
    slab_rows = _kq_pad_slab_rows(spec)
    if (
        not slab_rows
        or int(spec.tile_size_eff) % slab_rows
        or int(spec.kq_lds_pad_halves) % 8
    ):
        return "the KQ LDS pad needs an aligned slab layout"
    if _k_schedule_bufs(spec) != 1:
        return "the KQ LDS pad is only laid out for a single-K schedule"
    return None


def _q_aliases_k(spec) -> bool:
    """Whether Q is staged in K_lds rather than its own slab."""
    if spec.use_q_reread or spec.use_q_direct_reg or _fp8_mfma_qk(spec):
        return False
    tile, head = int(spec.tile_size_eff), int(spec.head_size)
    return int(spec.block_m) * head * 2 <= 2 * tile * head * 2


def _kq_pad_active(spec) -> bool:
    return bool(spec.use_kq_lds_pad) and _kq_pad_off_reason(spec) is None


def _gfx950_2d_lds_bytes(spec) -> int:
    """The LDS pool the LLVM lowering packs for one gfx950 tiled 2D spec."""
    tile = int(spec.tile_size_eff)
    head = int(spec.head_size)
    block_m = int(spec.block_m)
    kv_fp8 = spec.kv_storage_dtype == "fp8e4m3"
    fp8_qk = _fp8_mfma_qk(spec)
    fp8_pv = kv_fp8 and bool(spec.use_fp8_mfma_pv)
    k_elem_bytes = 1 if fp8_qk else 2
    v_elem_bytes = 1 if fp8_pv else 2
    k_bufs = _k_schedule_bufs(spec)
    v_bufs = 2 if spec.use_v_double_buffer else 1

    if _kq_pad_active(spec):
        slab_rows = _kq_pad_slab_rows(spec)
        slab_cols = slab_rows * head + int(spec.kq_lds_pad_halves)
        k_bytes = k_bufs * (tile // slab_rows) * slab_cols * k_elem_bytes
    else:
        k_bytes = k_bufs * tile * head * k_elem_bytes
    v_bytes = v_bufs * tile * head * v_elem_bytes

    transposed = bool(spec.use_mfma_32x32 and spec.use_transposed_qk_32x32)
    p_bytes = (
        0
        if spec.use_register_pv or transposed
        else block_m * (tile + (16 if fp8_pv else 8)) * v_elem_bytes
    )
    q_bytes = 0 if spec.use_q_direct_reg or _q_aliases_k(spec) else block_m * head * 2
    acc_bytes = block_m * (32 if head <= 64 else head) * 2
    fp8_staging_bytes = 3 * tile * head if fp8_qk else 0
    loop_bytes = k_bytes + v_bytes + p_bytes + q_bytes + fp8_staging_bytes
    # The transposed body touches the epilogue slab only after the KV loop, so
    # the lowering's liveness packing overlays it on the loop buffers.
    if transposed:
        return max(loop_bytes, acc_bytes)
    return loop_bytes + acc_bytes


def _gfx950_2d_legality(spec) -> Tuple[bool, str]:
    if _kq_pad_active(spec) and _q_aliases_k(spec):
        return False, "padded K LDS does not support aliased Q (wrong output)"
    from rocke.core.arch import ArchTarget

    capacity = ArchTarget.from_gfx("gfx950").lds_capacity_bytes
    used = _gfx950_2d_lds_bytes(spec)
    if used > capacity:
        return False, (
            f"estimated LDS {used} B exceeds the gfx950 {capacity} B LDS budget; "
            "codegen would fail"
        )
    return True, "ok"


class UnifiedKernels(NamedTuple):
    kernel_spec: Any
    reduce_spec: Any = None


@dataclass(frozen=True)
class UnifiedSpace(WavesPerEuSpace):
    """One geometry variant's space. ``base`` is the request's
    ``UnifiedAttentionProblem``."""

    variant: AttentionGeometryVariant = None

    # A problem whose production stacks are all illegal can still have a
    # legal full-space point; auto takes the first one that is not a dead end.
    default_from_full = True

    def axes(self, base):
        return _variant_axes(self.variant)

    def fixed(self, base):
        return _CODEPATH_KNOBS.get((self.arch, self.variant.codepath), {})

    def refuse(self, base, knobs):
        wrong = sorted(k for k in KNOWN_WRONG_KNOBS.get(self.arch, ()) if knobs.get(k))
        if wrong:
            return f"{wrong} are known to produce wrong output on {self.arch}"
        return None

    def prefilter(self, base, knobs):
        inert = _other_codepath_knobs(self.arch, {**self.fixed(base), **knobs})
        return {k: v for k, v in knobs.items() if k not in inert}

    def build(self, base, knobs):
        knobs = dict(knobs)
        wpe = knobs.pop("waves_per_eu", None)
        if self.path == "3d":
            return UnifiedKernels(*_explicit_3d_specs(base, self.variant, knobs, wpe))
        return UnifiedKernels(_explicit_2d_spec(base, self.variant, knobs, wpe))

    def field_value(self, kernel, name):
        return getattr(kernel.kernel_spec, name, MISSING)

    def defaults(self, base, kernel):
        reduce = kernel.reduce_spec
        return {
            "codepath": dict(self.fixed(base)),
            "kernel": dict(_declared_defaults(type(kernel.kernel_spec))),
            "reduce": (
                None if reduce is None else dict(_declared_defaults(type(reduce)))
            ),
        }

    def base_value(self, base, kernel, name):
        return _field_defaults(type(kernel.kernel_spec)).get(name, MISSING)

    def _gfx950_2d(self) -> bool:
        return self.arch == "gfx950" and self.path == "2d"

    def inert(self, base, kernel):
        spec = kernel.kernel_spec
        if not self._gfx950_2d() or not spec.use_kq_lds_pad:
            return {}
        why = _kq_pad_off_reason(spec)
        return {} if why is None else dict.fromkeys(_KQ_PAD_KNOBS, why)

    def validate(self, base, kernel):
        if not self._gfx950_2d():
            return True, "ok"
        return _gfx950_2d_legality(kernel.kernel_spec)

    def outer_values(self, base, level):
        if level == "production" and self.path == "2d":
            return PRODUCTION_WAVES
        return FULL_WAVES

    def production(self, base, axes, is_valid):
        return _production_knob_sets(self.variant)

    def is_valid(self, base, knobs):
        if _policy_conflict(self.arch, knobs):
            return False
        try:
            kernel = self.build(base, knobs)
        except (ValueError, NotImplementedError):
            return False
        return self.validate(base, kernel)[0]

    def accept_default(self, spec):
        return not _sets_dead_end(spec)

    def wrap(self, base, kernel, knobs, key, tid):
        variant = self.variant
        return AttentionTuningSpec(
            path=variant.path,
            arch=variant.arch,
            builder_kind="tiled_3d" if variant.path == "3d" else variant.builder_kind,
            compile_backend="llvm" if variant.path == "3d" else variant.compile_backend,
            candidate_name=variant.candidate_name,
            tuning_id=tid,
            kernel_spec=kernel.kernel_spec,
            fp8_fnuz=bool(base.fp8_fnuz),
            num_kv_blocks=int(base.num_kv_blocks),
            reduce_spec=kernel.reduce_spec,
            variant_id=variant.variant_id,
            config_key=key,
            knobs=knobs,
        )


@lru_cache(maxsize=None)
def unified_space(variant: AttentionGeometryVariant) -> UnifiedSpace:
    return UnifiedSpace(
        abi=ATTENTION_ABI_VERSION,
        arch=variant.arch,
        path=variant.path,
        variant_id=variant.variant_id,
        candidate_name=variant.candidate_name,
        variant=variant,
    )


def canonicalize_tuning_spec(
    problem, variant: AttentionGeometryVariant, knobs: Mapping[str, object]
) -> Tuple[Optional[AttentionTuningSpec], str]:
    """``variant`` with ``knobs`` applied, in canonical form, or ``(None, why)``."""
    return unified_space(variant).canonicalize(problem, knobs)


def _explicit_configs(
    problem, variant: AttentionGeometryVariant
) -> Iterable[AttentionTuningSpec]:
    return unified_space(variant).stream(problem, "full")


def _production_configs(problem, variant: AttentionGeometryVariant):
    return unified_space(variant).stream(problem, "production")


def iter_tuning_specs(
    req: AttentionRequest,
    variant: AttentionGeometryVariant,
    level: str = "production",
):
    """Specs for one geometry at ``level``: ``production`` walks the curated
    stacks (no dead-end knobs); ``full`` walks every kernel knob -- consume it
    through :func:`sample_tuning_specs` unless the space is known to be small."""
    return unified_space(variant).stream(_problem(req), level)


def sample_tuning_specs(
    req: AttentionRequest,
    variant: AttentionGeometryVariant,
    n: int,
    seed: int = 0,
) -> Iterable[AttentionTuningSpec]:
    """Up to ``n`` distinct random legal specs from the full knob space, dead
    ends included; ``KNOWN_WRONG_KNOBS`` never."""
    return unified_space(variant).sample(_problem(req), n, seed)
