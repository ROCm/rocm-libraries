# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dense-kernel tuning rules and the per-candidate dense space.

The axes are data in :mod:`.axes`. This module owns what is specific to the
dense bodies: which axes a base spec reads (scope), which explicit values
restate a policy, and which knobs are inert on others. :class:`DenseSpace`
states those rules as the hooks of the shared
:class:`rocke.dispatch.tuning.KnobSpace`, which makes every spec.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as _dataclass_fields, replace
from functools import lru_cache
from itertools import product
from typing import Callable, Iterable, Mapping, Optional, Tuple

from rocke.dispatch.tuning.walk import one_knob_at_a_time

from .axes import (
    DENSE_OUTPUT_FIELDS,
    DENSE_PROBLEM_FIELDS,
    KnobAxis,
    axis_knob_names,
    tuning_axes,
)
from .common import AttentionTuningSpec
from .waves import WavesPerEuSpace, waves_per_eu_sweep_values

DenseSupports = Callable[..., Tuple[bool, str]]
# Request-set fields: never recorded as a tuning point's defaults.
_REQUEST_FIELDS = DENSE_PROBLEM_FIELDS.union(*DENSE_OUTPUT_FIELDS.values())


def _gfx950_dense_inert_knobs(spec) -> dict:
    defaults = _dense_field_defaults(type(spec))
    inert = {}
    if (
        not spec.lazy_rescale
        and spec.lazy_rescale_threshold != defaults["lazy_rescale_threshold"]
    ):
        inert["lazy_rescale_threshold"] = "read only with lazy_rescale"
    if (
        not spec.resolved_pv_sched_fence()
        and spec.pv_sched_fence_mask != defaults["pv_sched_fence_mask"]
    ):
        inert["pv_sched_fence_mask"] = "read only while the fence is on"
    if (
        not spec.resolved_pv_sched_group_template()
        and spec.pv_sched_group_ds_read != defaults["pv_sched_group_ds_read"]
    ):
        inert["pv_sched_group_ds_read"] = "read only while the template is on"
    return inert


def _gfx942_dense_inert_knobs(spec) -> dict:
    # Only the conflict-free-V store builds the transposed V_lds these shape.
    if spec.resolved_use_cfvst():
        return {}
    return {
        name: "read only on the cfvst path"
        for name in ("v_row_pad", "use_v_swizzle")
        if getattr(spec, name) is not None
    }


@dataclass(frozen=True)
class _DenseTuningArch:
    """What the shared dense walk needs to know about one arch's dense body.

    ``scope`` drops an axis for a base spec whose problem or variant never
    reads it (the symbol drops it too, so offering it only re-emits the base
    kernel). ``policy_knobs`` are ``None`` fields resolved by a
    ``resolved_<name>()`` policy; an explicit value equal to the policy is a
    duplicate. ``inert`` names the remaining knob-on-knob dependencies.
    ``num_persistent_over_tiles`` resolves the symbolic CTA counts for every
    ``block_m`` the tile axis offers, not only the base's, because a count
    such as ``gqa_pair`` is exact for one tile. ``production_crossed`` axes
    get a full one-knob-at-a-time production pass at each of their values
    instead of being one more single knob.
    """

    scope: Mapping[str, Callable[[object], bool]]
    policy_knobs: Tuple[str, ...]
    inert: Callable[[object], Mapping[str, str]]
    num_persistent_over_tiles: bool = False
    production_crossed: Tuple[str, ...] = ()


_DENSE_ARCH: Mapping[str, _DenseTuningArch] = {
    "gfx950": _DenseTuningArch(
        scope={
            "num_persistent": lambda s: bool(s.persistent),
            "persist_decode": lambda s: bool(s.persistent),
            "interleave": lambda s: bool(s.persistent) and bool(s.causal),
            # Wide DMA locks the K/V slab pads.
            "lds_k_row_pad": lambda s: int(s.head_size) == 128 and not s.wide_lds_dma,
            "lds_v_row_pad": lambda s: int(s.head_size) == 128 and not s.wide_lds_dma,
            "lds_k_group_pad": lambda s: int(s.head_size) < 128,
            # Sliding window keeps its own three-phase band loop.
            "causal_diag_split": lambda s: bool(s.causal)
            and int(s.sliding_window) == 0,
        },
        policy_knobs=(
            "exp_per_pv_step",
            "pv_sched_fence",
            "pv_sched_group_template",
            "iglp_mode",
            "pv_loop_order",
        ),
        inert=_gfx950_dense_inert_knobs,
        num_persistent_over_tiles=True,
        # The query tile changes what every other knob does.
        production_crossed=("block_m",),
    ),
    "gfx942": _DenseTuningArch(
        # persistent is itself an axis here, so the persistent-only knobs stay
        # and the per-spec duplicate check drops them on non-persistent specs.
        scope={
            "interleave": lambda s: bool(s.causal),
            "lds_row_pad": lambda s: int(s.head_size) == 128,
            "lds_k_group_pad": lambda s: int(s.head_size) < 128,
            # Policy turns cfvst (and its swizzle) on only where it is legal;
            # the knobs can only turn it off.
            "use_cfvst": lambda s: s.resolved_use_cfvst(),
            "use_v_swizzle": lambda s: s.resolved_use_cfvst(),
            "v_row_pad": lambda s: s.resolved_use_cfvst(),
            "causal_diag_split": lambda s: bool(s.causal)
            and int(s.sliding_window) == 0,
        },
        policy_knobs=("use_cfvst", "use_exp2_fast", "v_row_pad", "use_v_swizzle"),
        inert=_gfx942_dense_inert_knobs,
    ),
}


def _dense_arch(arch: str) -> _DenseTuningArch:
    try:
        return _DENSE_ARCH[arch]
    except KeyError:
        raise ValueError(f"no dense tuning space for arch {arch!r}") from None


@lru_cache(maxsize=None)
def _dense_field_defaults(spec_type: type) -> Mapping[str, object]:
    return {f.name: f.default for f in _dataclass_fields(spec_type)}


def resolve_dense_num_persistent(
    spec, policy: str, block_m: Optional[int] = None
) -> int:
    """Persistent-CTA count for one symbolic ``num_persistent`` policy, at
    ``block_m`` (the spec's by default).

    ``half`` / ``0.75x`` / ... / ``2x`` scale the base count; ``gqa_pair`` /
    ``gqa_pair_2phase`` are the exact counts those decodes require; ``work`` is
    one CTA per (query block, head, batch) work item.
    """
    nqb = -(-int(spec.seqlen_q) // int(block_m or spec.block_m))
    pairs = nqb * int(spec.num_kv_heads) * int(spec.batch)
    base = int(spec.num_persistent)
    counts = {
        "half": base // 2,
        "0.75x": base * 3 // 4,
        "1.25x": base * 5 // 4,
        "1.5x": base * 3 // 2,
        "1.75x": base * 7 // 4,
        "2x": base * 2,
        "gqa_pair": pairs,
        "gqa_pair_2phase": pairs * int(spec.num_queries_per_kv) // 2,
        "work": nqb * int(spec.num_query_heads) * int(spec.batch),
    }
    try:
        return counts[policy]
    except KeyError:
        raise ValueError(
            f"num_persistent policy must be one of {sorted(counts)}, got {policy!r}"
        ) from None


@lru_cache(maxsize=256)
def _dense_axes_for(base, arch: str) -> Tuple[KnobAxis, ...]:
    """The declared dense axes that apply to ``base``, with concrete values."""
    rules = _dense_arch(arch)
    declared = tuning_axes(arch, "dense")
    block_ms = [int(base.block_m)]
    if rules.num_persistent_over_tiles:
        block_ms += [
            int(value)
            for axis in declared
            if axis.name == "block_m"
            for choice in axis.choices
            for _name, value in choice
            if int(value) != int(base.block_m)
        ]
    axes = []
    for axis in declared:
        scope = rules.scope.get(axis.name)
        if scope is not None and not scope(base):
            continue
        if axis.name == "num_persistent":
            counts: list[int] = []
            for block_m in block_ms:
                for choice in axis.choices:
                    for _name, policy in choice:
                        n = resolve_dense_num_persistent(base, policy, block_m)
                        if n > 0 and n != int(base.num_persistent) and n not in counts:
                            counts.append(n)
            axis = KnobAxis(
                axis.name, ((),) + tuple((("num_persistent", n),) for n in counts)
            )
        axes.append(axis)
    return tuple(axes)


def _dense_policy_value(spec, name: str):
    try:
        return getattr(replace(spec, **{name: None}), f"resolved_{name}")()
    except ValueError:
        return None


def _dense_redundant_knobs(spec, arch: str) -> dict:
    """``{field: why}`` for every field of ``spec`` whose explicit value
    compiles to the same IR as leaving it at its default.

    An explicit value equal to what its policy resolves to, or a knob the body
    does not read for this spec, re-emits an existing kernel under a different
    symbol. The walk treats such a prefix as invalid; every relation here
    points at an earlier axis or at the base spec, so pruning is exact.
    Canonicalization drops these fields instead.
    """
    rules = _dense_arch(arch)
    redundant = {}
    for name in rules.policy_knobs:
        value = getattr(spec, name)
        if value is not None and value == _dense_policy_value(spec, name):
            redundant[name] = f"{name}={value!r} restates its policy"
    if not spec.persistent:
        if spec.persist_decode != "auto":
            redundant["persist_decode"] = "read only by the persistent body"
        if spec.interleave:
            redundant["interleave"] = "read only by the persistent body"
    elif (
        spec.persist_decode != "auto"
        and replace(spec, persist_decode="auto").resolved_persist_decode
        == spec.persist_decode
    ):
        redundant["persist_decode"] = (
            f"persist_decode={spec.persist_decode!r} is what auto resolves to"
        )
    if spec.interleave and "interleave" not in redundant:
        nqb = -(-int(spec.seqlen_q) // int(spec.block_m))
        if not spec.causal or spec.resolved_persist_decode != "qb_major" or nqb < 2:
            redundant["interleave"] = (
                "read only on the causal qb_major decode with NQB > 1"
            )
    for name, why in rules.inert(spec).items():
        redundant.setdefault(name, f"{name} is {why}")
    return redundant


def _dense_redundant_knob(spec, arch: str) -> Optional[str]:
    """Why ``spec`` re-emits a kernel an earlier walk step already emits."""
    return next(iter(_dense_redundant_knobs(spec, arch).values()), None)


@lru_cache(maxsize=256)
def _dense_defaults(base, recorded: frozenset) -> Tuple[Tuple[str, object], ...]:
    """The default spec's values that are neither request fields (problem or
    output) nor resolved per problem: the candidate's body, the base constants
    (the CU-count default, the shipped WPE, the layout pads) and every dataclass
    default."""
    return tuple(
        (f.name, getattr(base, f.name))
        for f in _dataclass_fields(base)
        if f.name not in _REQUEST_FIELDS and f.name not in recorded
    )


@dataclass(frozen=True)
class DenseSpace(WavesPerEuSpace):
    """One dense candidate's space. ``base`` is its default kernel spec for the
    request; ``supports`` is the kernel's own validator."""

    supports: DenseSupports = None
    recorded_fields: frozenset = frozenset()
    # Problem fields that also depend on knobs, recomputed on every build:
    # ``derived(base, knobs) -> {field: value}`` (gfx950: ``ragged`` follows
    # the swept tile).
    derived: Optional[
        Callable[[object, Mapping[str, object]], Mapping[str, object]]
    ] = None

    def recorded(self, base):
        return self.recorded_fields

    def axes(self, base):
        return _dense_axes_for(base, self.arch)

    def production(self, base, axes, is_valid) -> Iterable[dict]:
        crossed_names = _dense_arch(self.arch).production_crossed
        crossed = [a for a in axes if a.name in crossed_names]
        rest = tuple(a for a in axes if a.name not in crossed_names)
        for choices in product(*(a.choices for a in crossed)):
            point = {k: v for choice in choices for k, v in choice}
            if point and not is_valid(point):
                continue
            for knobs in one_knob_at_a_time(
                rest, lambda k, point=point: is_valid({**point, **k})
            ):
                yield {**point, **knobs}

    def known_knobs(self, base):
        return axis_knob_names(tuning_axes(self.arch, "dense"))

    def defaults(self, base, kernel):
        return dict(_dense_defaults(base, self.recorded_fields))

    def build(self, base, knobs):
        extra = self.derived(base, knobs) if self.derived is not None else {}
        return replace(base, **knobs, **extra)

    def inert(self, base, kernel):
        inert = dict(_dense_redundant_knobs(kernel, self.arch))
        # The grid body never reads the persistent-CTA count.
        if not kernel.persistent and kernel.num_persistent != base.num_persistent:
            inert["num_persistent"] = "read only by the persistent body"
        return inert

    def validate(self, base, kernel):
        return self.supports(kernel, arch=self.arch)

    def outer_default(self, base):
        return int(base.waves_per_eu)

    def outer_values(self, base, level):
        return waves_per_eu_sweep_values(int(base.waves_per_eu), level)

    def wrap(self, base, kernel, knobs, key, tid):
        return AttentionTuningSpec(
            path="dense",
            arch=self.arch,
            builder_kind="dense",
            compile_backend="llvm",
            candidate_name=self.candidate_name,
            tuning_id=tid,
            kernel_spec=kernel,
            variant_id=self.variant_id,
            config_key=key,
            knobs=knobs,
        )
