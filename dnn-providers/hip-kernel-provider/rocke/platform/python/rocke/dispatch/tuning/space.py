# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One registered variant's tuning space.

A family subclasses :class:`KnobSpace`, implements :meth:`~KnobSpace.axes`,
:meth:`~KnobSpace.build` and :meth:`~KnobSpace.wrap`, and overrides whichever
hooks its kernel needs. Everything a candidate hands out -- swept, sampled,
pinned by knobs, or found by id -- is made by :meth:`KnobSpace.canonicalize`,
so one configuration has one ``tuning_id`` whichever entry point produced it.

``base`` is whatever the family's variant needs to build a spec for one
request: a default kernel spec (dense attention), or a problem description
the variant turns into one (unified attention). The space never inspects it.
"""

from __future__ import annotations

import random
from dataclasses import MISSING, dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple

from .axes import KnobAxis, Knobs, axis_knob_names, knob_requirements
from .identity import (
    config_key,
    defaults_fingerprint,
    key_of,
    knob_items,
    names_variant,
    tuning_id,
)
from .walk import (
    FULL_WAVES,
    PRODUCTION_WAVES,
    SWEEP_LEVELS,
    iter_knob_sets,
    one_knob_at_a_time,
    random_knob_set,
)

Verdict = Tuple[Optional[Any], str]

# What ``build`` raises to refuse a knob set.
BUILD_ERRORS = (ValueError, TypeError, NotImplementedError, ZeroDivisionError)


@dataclass(frozen=True)
class KnobSpace:
    """Base class for a variant's tuning space; see the module docstring.

    Canonicalization drops knobs that compile to the default -- values equal
    to the base (:meth:`base_value`), :meth:`inert` knobs, known knobs out of
    scope for this problem, and gated sub-knobs whose gate is off -- and
    refuses the rest of what a sweep would never emit: :meth:`refuse`,
    changing a :meth:`fixed` knob, fields that are not knobs of the variant,
    :meth:`build` errors, and :meth:`validate` rejections.
    """

    abi: str
    arch: str
    path: str
    variant_id: str
    candidate_name: str

    # ------------------------------------------------------------ must define
    def axes(self, base) -> Tuple[KnobAxis, ...]:
        """The axes in scope for ``base``, prerequisites first."""
        raise NotImplementedError

    def build(self, base, knobs: Mapping[str, object], waves_per_eu: Optional[int]):
        """The kernel spec for ``knobs`` (fixed knobs included) at
        ``waves_per_eu`` (``None``: the base's). Raise one of
        :data:`BUILD_ERRORS` to refuse."""
        raise NotImplementedError

    def wrap(self, base, kernel, knobs: Knobs, key: str, tid: str):
        """The tuned spec (a :class:`~rocke.dispatch.tuning.spec.TunedSpec`)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ hooks
    def fixed(self, base) -> Mapping[str, object]:
        """Knobs the variant fixes (a codepath's). Restating one is dropped,
        changing one is refused."""
        return {}

    def known_knobs(self, base) -> frozenset:
        """Every knob the variant can take on some problem. One outside
        :meth:`axes` for this ``base`` (out of scope, never read) is dropped;
        a field outside this set is refused."""
        return axis_knob_names(self.axes(base))

    def refuse(self, base, knobs: Mapping[str, object]) -> Optional[str]:
        """Why ``knobs`` must never be built (known-wrong output), or None."""
        return None

    def prefilter(self, base, knobs: Mapping[str, object]) -> dict:
        """Drop knobs that are inert before building (for instance knobs only
        another codepath emits, which its validator may not even accept)."""
        return dict(knobs)

    def field_value(self, kernel, name: str):
        return getattr(kernel, name, MISSING)

    def base_value(self, base, kernel, name: str):
        """The default value of ``name``; ``MISSING`` if unknown."""
        return getattr(base, name, MISSING)

    def inert(self, base, kernel) -> Mapping[str, str]:
        """``{knob: why}`` for fields of ``kernel`` the body does not read, or
        whose explicit value restates a policy. Like :meth:`base_value`, it
        must not depend on ``waves_per_eu``: streams check a knob set once and
        then only rebuild it per WPE value."""
        return {}

    def validate(self, base, kernel) -> Tuple[bool, str]:
        """The kernel's own legality check."""
        return True, "ok"

    def default_waves(self, base) -> Optional[int]:
        """The default spec's ``waves_per_eu``; a pinned equal value is dropped."""
        return None

    def defaults(self, base, kernel) -> Mapping[str, object]:
        """The problem-independent defaults the knobs are a delta against:
        every value a knob-free spec takes that is not a problem field or a
        :meth:`recorded` field. Folded into ``config_key``, so changing one in
        tree changes every key and a stored pin is refused instead of silently
        building a different kernel. Must be the same for every problem."""
        return {}

    def recorded(self, base) -> frozenset:
        """Fields whose default the base resolves per problem (from the work
        size, the dtype...). They are always recorded with their effective
        value, so the same knobs name the same configuration on every problem;
        dropping them when they equal this problem's default would not."""
        return frozenset()

    def waves_per_eu_of(self, kernel) -> Optional[int]:
        """The value shown in the id stem."""
        value = self.field_value(kernel, "waves_per_eu")
        return None if value is MISSING else value

    def waves(self, base, level: str) -> Tuple[Optional[int], ...]:
        return PRODUCTION_WAVES if level == "production" else FULL_WAVES

    def root(self, base) -> Mapping[str, object]:
        """Where the walks start."""
        return self.fixed(base)

    def production(self, base, axes: Tuple[KnobAxis, ...], is_valid) -> Iterable[dict]:
        """The production knob sets: by default every knob on its own."""
        return one_knob_at_a_time(axes, is_valid)

    def is_valid(self, base, knobs: Mapping[str, object]) -> bool:
        """Walk pruning: by default a prefix is valid when it is already
        canonical -- nothing in it would be dropped or refused -- and passes
        :meth:`validate`. Stops at the first reason, since the walk only needs
        a verdict."""
        knobs = dict(knobs)
        if self.refuse(base, knobs):
            return False
        fixed = self.fixed(base)
        if any(k in fixed and v != fixed[k] for k, v in knobs.items()):
            return False
        knobs = {k: v for k, v in knobs.items() if k not in fixed}
        axes = self.axes(base)
        if not set(knobs) <= axis_knob_names(axes):
            return False
        if len(self.prefilter(base, knobs)) != len(knobs):
            return False
        try:
            kernel = self.build(base, {**fixed, **knobs}, None)
        except BUILD_ERRORS:
            return False
        if self._droppable(base, kernel, knobs, knob_requirements(axes)):
            return False
        return self.validate(base, kernel)[0]

    def _droppable(self, base, kernel, knobs: Mapping[str, object], requires) -> set:
        drop = {
            k
            for k, v in knobs.items()
            if (b := self.base_value(base, kernel, k)) is not MISSING and b == v
        }
        drop |= set(self.inert(base, kernel)) & set(knobs)
        drop |= {
            k
            for k in knobs
            if k in requires
            and self.field_value(kernel, requires[k]) in (MISSING, False, None)
        }
        return drop

    def accept_default(self, spec) -> bool:
        return True

    # Search the full space for a default when production has none.
    default_from_full = False

    # -------------------------------------------------------------- the rules
    def _checked(self, base, knobs: Mapping[str, object], waves_per_eu):
        """``(kernel, canonical_knobs, why)``; ``kernel`` is None if refused."""
        knobs = dict(knobs)
        why = self.refuse(base, knobs)
        if why:
            return None, knobs, why
        fixed = self.fixed(base)
        moved = sorted(k for k, v in knobs.items() if k in fixed and v != fixed[k])
        if moved:
            return None, knobs, f"{moved} are fixed by {self.candidate_name}"
        knobs = {k: v for k, v in knobs.items() if k not in fixed}
        axes = self.axes(base)
        in_scope = axis_knob_names(axes)
        stray = sorted(set(knobs) - in_scope - self.known_knobs(base))
        if stray:
            return None, knobs, f"{stray} are not tunable on {self.candidate_name}"
        # A known knob this problem does not read is inert here, not illegal.
        knobs = self.prefilter(base, {k: v for k, v in knobs.items() if k in in_scope})
        requires = knob_requirements(axes)
        kernel = None
        for _ in range(len(knobs) + 1):
            try:
                kernel = self.build(base, {**fixed, **knobs}, waves_per_eu)
            except BUILD_ERRORS as e:
                return None, knobs, str(e) or type(e).__name__
            drop = self._droppable(base, kernel, knobs, requires)
            if not drop:
                break
            knobs = {k: v for k, v in knobs.items() if k not in drop}
        ok, why = self.validate(base, kernel)
        if not ok:
            return None, knobs, why
        return kernel, knobs, "ok"

    def _finish(self, base, kernel, knobs: Mapping[str, object], waves_per_eu):
        canonical = dict(knobs)
        recorded = self.recorded(base)
        for name in recorded - {"waves_per_eu"}:
            canonical[name] = self.field_value(kernel, name)
        if "waves_per_eu" in recorded:
            canonical["waves_per_eu"] = self.waves_per_eu_of(kernel)
        elif waves_per_eu is not None and waves_per_eu != self.default_waves(base):
            canonical["waves_per_eu"] = int(waves_per_eu)
        items = knob_items(canonical)
        key = config_key(
            abi=self.abi,
            arch=self.arch,
            path=self.path,
            variant_id=self.variant_id,
            knobs=items,
            defaults=defaults_fingerprint(self.defaults(base, kernel)),
        )
        tid = tuning_id(self.variant_id, self.waves_per_eu_of(kernel), key)
        return self.wrap(base, kernel, items, key, tid)

    # ------------------------------------------------------------ public API
    def canonicalize(self, base, knobs: Mapping[str, object]) -> Verdict:
        """``base`` with ``knobs`` (``waves_per_eu`` included) applied, in
        canonical form, or ``(None, why)``."""
        knobs = dict(knobs)
        waves_per_eu = knobs.pop("waves_per_eu", None)
        kernel, canonical, why = self._checked(base, knobs, waves_per_eu)
        if kernel is None:
            return None, why
        return self._finish(base, kernel, canonical, waves_per_eu), "ok"

    def stream(self, base, level: str) -> Iterator:
        """Specs at ``level``, production's first spec being the default.

        ``production`` walks :meth:`production`; ``full`` walks the pruned
        product of all axes -- consume it through :meth:`sample` unless the
        space is known to be small.
        """
        if level not in SWEEP_LEVELS:
            raise ValueError(f"sweep level must be one of {SWEEP_LEVELS}, got {level!r}")
        axes = self.axes(base)

        def is_valid(knobs) -> bool:
            return self.is_valid(base, knobs)

        knob_sets = (
            self.production(base, axes, is_valid)
            if level == "production"
            else iter_knob_sets(axes, self.root(base), is_valid)
        )
        seen: Optional[set] = set() if level == "production" else None
        fixed = self.fixed(base)
        for knobs in knob_sets:
            kernel, canonical, _why = self._checked(base, knobs, None)
            if kernel is None:
                continue
            for waves_per_eu in self.waves(base, level):
                spec = self._at_waves(base, kernel, fixed, canonical, waves_per_eu)
                if spec is None:
                    continue
                if seen is not None:
                    if spec.tuning_id in seen:
                        continue
                    seen.add(spec.tuning_id)
                yield spec

    def _at_waves(self, base, kernel, fixed, canonical, waves_per_eu):
        """The checked knob set at one WPE: rebuild and revalidate only, since
        the drop rules never read ``waves_per_eu``."""
        if waves_per_eu is not None:
            try:
                kernel = self.build(base, {**fixed, **canonical}, waves_per_eu)
            except BUILD_ERRORS:
                return None
            ok, _why = self.validate(base, kernel)
            if not ok:
                return None
        return self._finish(base, kernel, canonical, waves_per_eu)

    def sample(self, base, n: int, seed: int) -> Iterator:
        """Up to ``n`` distinct random legal specs from the full space, deduped
        by ``tuning_id``. The candidate name salts the seed so variants sharing
        a seed do not draw in lockstep; stops after ``20 * n`` draws."""
        axes = self.axes(base)

        def is_valid(knobs) -> bool:
            return self.is_valid(base, knobs)

        rng = random.Random(f"{int(seed)}:{self.candidate_name}")
        waves = self.waves(base, "full")
        seen: set[str] = set()
        for _ in range(20 * int(n)):
            if len(seen) >= n:
                return
            knobs = random_knob_set(axes, self.root(base), is_valid, rng)
            if knobs is None:
                continue
            spec, _why = self.canonicalize(base, _with_waves(knobs, rng.choice(waves)))
            if spec is not None and spec.tuning_id not in seen:
                seen.add(spec.tuning_id)
                yield spec

    def default(self, base) -> Verdict:
        """The first production spec :meth:`accept_default` takes, else (with
        :attr:`default_from_full`) the first such full-space spec. When there
        is none, the reason is why the untuned spec was refused."""
        levels = SWEEP_LEVELS if self.default_from_full else ("production",)
        for level in levels:
            spec = next(
                (s for s in self.stream(base, level) if self.accept_default(s)), None
            )
            if spec is not None:
                return spec, "ok"
        untuned, why = self.canonicalize(base, {})
        if untuned is None:
            return None, why
        return None, f"no valid spec on {self.candidate_name} for this problem"

    def find(self, base, wanted: str):
        """Resolve an id without its knobs, within the production set only.

        The full space can hold millions of specs, so a bare id from it is not
        searched for: dispatch must stay bounded. A full-space id replays from
        the knobs recorded next to it, which every sweep row and pinned result
        carries."""
        if not names_variant(wanted, self.variant_id):
            return None
        key = key_of(wanted)
        return next(
            (s for s in self.stream(base, "production") if s.config_key == key), None
        )


def _with_waves(knobs: Mapping[str, object], waves_per_eu) -> dict:
    if waves_per_eu is None:
        return dict(knobs)
    return {**knobs, "waves_per_eu": waves_per_eu}
