# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A dispatcher :class:`~rocke.dispatch.tuning.space.KnobSpace` as a search space.

The parameters come from the variant's axes for this problem
(:func:`.params.params_from_axes`), plus its outer knob when it has one.
Base values the kernel does not report are probed: a choice whose
canonical config equals the default's restates the base, which places the
``()`` level (no two levels of a parameter name one config).
Validity and identity are the space's own: a genome's key is the
``config_key`` of ``KnobSpace.canonicalize(base, knobs)`` -- the same key a
sweep row, a pin and the tuning DB carry -- and ``None`` when
canonicalization refuses the knobs. Two genomes that canonicalize to the
same spec (a knob restating the base, a gated knob with its gate off) share
a key and are measured once.

The tuned spec of every key seen is kept (:meth:`KnobSearchSpace.spec`), so
a backend compiles exactly the spec dispatch would serve for the pin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any

from ..axes import axis_knob_names
from ..space import BUILD_ERRORS, KnobSpace
from ..walk import random_knob_set
from .params import CATEGORICAL, ORDINAL, Param, SearchSpace, _same, params_from_axes

__all__ = ["KnobSearchSpace"]

# Bound on remembered refusal reasons (diagnostics only).
_WHY_LIMIT = 4096


def _numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _probe_base_values(knob_space: KnobSpace, base: Any, axes, known) -> dict:
    """Effective base values the kernel does not report (``None`` fields
    that a policy or the request resolves): a single-knob choice whose
    canonical config is the default's restates the base, so its value is
    the base value. Without this a ``choices`` axis keeps a level equal to
    ``()`` and, when numeric, cannot be ordinal."""
    default, _why = knob_space.canonicalize(base, {})
    out: dict[str, object] = {}
    for axis in axes:
        nonempty = [c for c in axis.choices if c]
        if not nonempty or not all(len(c) == 1 for c in nonempty):
            continue
        names = {c[0][0] for c in nonempty}
        if len(names) != 1:
            continue
        (name,) = names
        if name in out or (
            name in known and any(_same(v, known[name]) for ((_k, v),) in nonempty)
        ):
            continue  # the reported base is one of the choices: placed already
        for ((knob, value),) in nonempty:
            try:
                spec, _w = knob_space.canonicalize(base, {knob: value})
            except BUILD_ERRORS:
                continue
            if spec is not None and spec.config_key == default.config_key:
                out[knob] = value
                break
    return out


class KnobSearchSpace(SearchSpace):
    """Search over ``knob_space`` for one problem ``base``."""

    def __init__(self, knob_space: KnobSpace, base: Any, *, include_outer: bool = True):
        axes = knob_space.axes(base)
        base_values: dict[str, object] = {}
        try:
            kernel = knob_space.build(base, dict(knob_space.fixed(base)))
            for name in sorted(axis_knob_names(axes)):
                v = knob_space.base_value(base, kernel, name)
                if v is not MISSING:
                    base_values[name] = v
        except BUILD_ERRORS:
            pass
        default, why = knob_space.canonicalize(base, {})
        if default is not None:
            base_values.update(_probe_base_values(knob_space, base, axes, base_values))
        params = list(params_from_axes(axes, base_values))
        outer = knob_space.outer_knob
        if include_outer and outer is not None:
            vals = [v for v in knob_space.outer_values(base, "full") if v is not None]
            vals = list(dict.fromkeys(vals))
            if len(vals) > 1:
                odef = knob_space.outer_default(base)
                numeric = all(_numeric(v) for v in vals)
                if numeric:
                    vals.sort()
                if odef in vals:
                    levels = tuple(() if v == odef else ((outer, v),) for v in vals)
                else:
                    levels = ((),) + tuple(((outer, v),) for v in vals)
                kind = (
                    ORDINAL
                    if numeric and odef in vals and len(levels) >= 3
                    else CATEGORICAL
                )
                params.append(Param(outer, levels, kind))
        if not params:
            params.append(Param("_default", ((),), CATEGORICAL))
        self.knob_space = knob_space
        self.base = base
        self._specs: dict[str, Any] = {}
        self._why: dict[str, str] = {}
        outer_vals = (
            tuple(knob_space.outer_values(base, "full")) if outer is not None else ()
        )

        def prefix_valid(config: dict) -> bool:
            knobs = {k: v for k, v in config.items() if k != outer}
            return knob_space.is_valid(base, knobs)

        def sampler(rng):
            knobs = random_knob_set(
                axes, knob_space.root(base), lambda k: knob_space.is_valid(base, k), rng
            )
            if knobs is None:
                return None
            if outer_vals:
                o = rng.choice(outer_vals)
                if o is not None:
                    knobs = {**knobs, outer: o}
            return knobs

        super().__init__(
            params,
            canonical=self._canonical,
            prefix_valid=prefix_valid,
            sampler=sampler,
            name=knob_space.candidate_name,
            fingerprint_extra={
                "abi": knob_space.abi,
                "arch": knob_space.arch,
                "path": knob_space.path,
                "variant": knob_space.variant_id,
                "default": None if default is None else default.config_key,
                "why": "" if default is not None else why,
            },
        )

    def _canonical(self, config: dict) -> str | None:
        spec, why = self.knob_space.canonicalize(self.base, config)
        if spec is None:
            if len(self._why) >= _WHY_LIMIT:
                self._why.pop(next(iter(self._why)))
            self._why[repr(sorted(config.items()))] = why
            return None
        self._specs.setdefault(spec.config_key, spec)
        return spec.config_key

    def spec(self, key_or_genome: str | Sequence[int]) -> Any:
        """The tuned spec of a key (or genome); None if never seen/refused."""
        if isinstance(key_or_genome, str):
            return self._specs.get(key_or_genome)
        key = self.key(key_or_genome)
        return None if key is None else self._specs.get(key)

    def why(self, genome) -> str:
        """Why the space refuses ``genome`` ("" when it is valid)."""
        if self.valid(genome):
            return ""
        return self._why.get(repr(sorted(self.decode(genome).items())), "refused")
