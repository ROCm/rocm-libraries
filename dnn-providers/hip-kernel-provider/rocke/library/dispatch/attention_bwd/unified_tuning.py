# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuning registry and legality of the unified (paged KV) attention backward.

Every tuning field of
:class:`~kernels.common.attention_bwd_unified.AttnBwdUnifiedSpec` and every
runtime value of the unified plan
(:data:`kernels.common.attention_bwd_unified_plan.RUNTIME_KNOBS`) is one
:class:`~.tuning_common.KnobAxis`, with the general registry's conventions
(``choices[0]`` is the default; ``not_built`` values are registered, never
emitted):

* the tile-step knobs the unified kernel shares with the general family
  (:data:`~kernels.common.attention_bwd_unified.UNIFIED_KNOBS`) are the
  general axes, narrowed to what the unified shell builds: the knobs of the
  general main-kernel shell (``_SHELL_KNOBS``) take their default only, the
  LDS DMA path is not built, and a Q / dO prefetch ring needs the barrier
  merge (:func:`unified_axis`);
* ``sub_page`` (the compile-time page of the ``"sub"`` page class) and the
  page-tile configuration (``page_tile``: a named configuration of
  :func:`~kernels.common.attention_bwd_unified_plan.attn_bwd_unified_tile_configs`,
  the values of the tuned page-tile table);
* the runtime values ``g_split``, ``lb_order``, ``scale_placement`` and
  ``page_mode`` (the latter also decides the spec's ``page_mode``).

:func:`unified_spec_legality` is the kernel's own legality
(``validate_attn_bwd_unified_spec``, the general emission check on the
general spec of the same tile step, and the known-wrong values of
:data:`~.tuning_common.BWD_KNOWN_WRONG_KNOBS`). :data:`UNIFIED_TUNED_PAGE_TILES`
is the tuned page-tile table the unified plan's shipped tuning reads
(:data:`~kernels.common.attention_bwd_unified_plan.TUNED_PAGE_TILES`); every
entry is checked at import (:func:`check_page_tile_entry`): a configuration
the arch ships for the head size, legal on the arch and serving the page.
The table's provenance (paired runs of
``benchmarks/common/attention_bwd_unified_bench.py``) is in
``platform/dsl_docs/instances/attention_bwd_unified.md``, section 3.
The registry's completeness (one axis per tuning field and per runtime value)
is checked at import too.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Tuple

from kernels.common.attention_bwd import _not_built
from kernels.common.attention_bwd_unified import (
    PAGE_GRANULE,
    PAGE_MODES,
    UNIFIED_KNOBS,
    AttnBwdUnifiedSpec,
    _SHELL_KNOBS,
    validate_attn_bwd_unified_spec,
)
from kernels.common.attention_bwd_unified_plan import (
    LB_ORDERS,
    RUNTIME_KNOBS,
    SHIPPED_SUB_PAGES,
    TUNED_PAGE_TILES,
    VALIDATED_ARCHS,
    _page_legal,
    attn_bwd_unified_tile_configs,
)
from kernels.common.attention_bwd_plan import SCALE_PLACEMENTS

from .tuning_common import (
    KIND_COMPILE,
    KIND_RUNTIME,
    KnobAxis,
    backward_axis,
    check_known_wrong,
    known_wrong_findings,
)

__all__ = [
    "PAGE_TILE_NAMES",
    "UNIFIED_BWD_KNOB_AXES",
    "UNIFIED_PROBLEM_FIELDS",
    "UNIFIED_TUNED_PAGE_TILES",
    "check_page_tile_entry",
    "unified_axis",
    "unified_spec_legality",
    "unified_tuning_axes",
]

# AttnBwdUnifiedSpec fields that describe the problem, not a tuning decision.
UNIFIED_PROBLEM_FIELDS = frozenset(
    {"head_size", "dtype", "mask_class", "stage_vec", "name"}
)
# Named page-tile configurations (attn_bwd_unified_tile_configs; each arch
# and head size ships the subset that is legal there).
PAGE_TILE_NAMES: Tuple[str, ...] = (
    "default",
    "k64",
    *(f"sub{p}" for p in SHIPPED_SUB_PAGES),
    "fallback",
)
# The tuned page-tile table of the unified plan's shipped tuning:
# {(arch, head size, page): configuration name}.
UNIFIED_TUNED_PAGE_TILES: Mapping[tuple, str] = TUNED_PAGE_TILES

# General knobs the unified shell does not build beyond one value.
_NOT_BUILT_GLOBAL_PATHS = ("dma",)


def _req_ring(value, knobs: Mapping[str, Any]) -> bool:
    # the unified shell builds the prefetch ring only under the barrier merge
    return value in (None, 1) or knobs.get("barrier_merge") == "on"


def _narrowed(ax: KnobAxis) -> KnobAxis:
    """The general axis ``ax`` as the unified shell builds it."""
    if ax.name in _SHELL_KNOBS:
        keep = _SHELL_KNOBS[ax.name]
        if ax.default != keep:
            raise AssertionError(
                f"{ax.name}: the unified shell's value {keep!r} is not the "
                f"general default {ax.default!r}"
            )
        extra = tuple(v for v in ax.choices if v != ax.default)
        return dataclasses.replace(
            ax,
            not_built=extra,
            full_only=tuple(v for v in ax.full_only if v not in extra),
            gated=tuple(v for v in ax.gated if v not in extra),
            note="a knob of the general main-kernel shell: the unified shell "
            "builds its default only",
        )
    if ax.name == "global_path":
        nb = tuple(dict.fromkeys(ax.not_built + _NOT_BUILT_GLOBAL_PATHS))
        return dataclasses.replace(
            ax,
            not_built=tuple(v for v in nb if v in ax.choices),
            note="the LDS DMA path of the general shell is not built here",
        )
    if ax.name == "ring_depth":
        general = ax.requires

        def requires(value, knobs):
            if general is not None and not general(value, knobs):
                return False
            return _req_ring(value, knobs)

        return dataclasses.replace(
            ax,
            requires=requires,
            note="the Q / dO prefetch ring only under barrier_merge='on'",
        )
    return ax


def _page_tiles(arch: str) -> Tuple[str, ...]:
    # the unified family serves the validated targets only
    return PAGE_TILE_NAMES if arch in VALIDATED_ARCHS else ("default",)


UNIFIED_BWD_KNOB_AXES: Tuple[KnobAxis, ...] = tuple(
    _narrowed(backward_axis(name)) for name in UNIFIED_KNOBS
) + (
    KnobAxis(
        "sub_page",
        (None,) + SHIPPED_SUB_PAGES,
        family=1,
        stage="geometry",
        numerics="tile",
        note="the 'sub' page class: the default tile serving one compile-time "
        "page below its kN0 (None: the 'multi' class, runtime page)",
    ),
    KnobAxis(
        "page_tile",
        PAGE_TILE_NAMES,
        family=1,
        stage="geometry",
        numerics="tile",
        arch_values=_page_tiles,
        note="a shipped page-tile configuration (attn_bwd_unified_tile_configs); "
        "the tuned table's values",
    ),
    KnobAxis(
        "page_mode",
        (None,) + PAGE_MODES,
        family=7,
        stage="runtime",
        kind=KIND_RUNTIME,
        numerics="reorders_fp",
        note="None: the plan's choice (direct when block_table_injective and "
        "g_split == 1); sets the spec's page_mode",
    ),
    KnobAxis(
        "g_split",
        (1,),
        family=7,
        stage="runtime",
        kind=KIND_RUNTIME,
        numerics="reorders_fp",
        derived="divisors of the head group (g_split > 1: atomic page mode)",
    ),
    KnobAxis(
        "lb_order", tuple(LB_ORDERS), family=7, stage="runtime", kind=KIND_RUNTIME
    ),
    KnobAxis(
        "scale_placement",
        SCALE_PLACEMENTS,
        family=7,
        stage="runtime",
        kind=KIND_RUNTIME,
        numerics="reorders_fp",
    ),
)
_BY_NAME: Mapping[str, KnobAxis] = {ax.name: ax for ax in UNIFIED_BWD_KNOB_AXES}
# Axes that select among named configurations rather than set a spec field.
_PRESET_AXES = ("page_tile",)


def unified_axis(name: str) -> KnobAxis:
    """The unified axis ``name``."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"no unified backward knob axis {name!r}") from None


def unified_tuning_axes(arch: str, level: str = "full") -> Tuple[KnobAxis, ...]:
    """The unified axes with values on ``arch`` at ``level``."""
    return tuple(
        ax
        for ax in UNIFIED_BWD_KNOB_AXES
        if (level == "full" or ax.level == "production") and ax.values(arch, level)
    )


def unified_spec_legality(spec, arch: str) -> str:
    """``"legal"``, ``"not_built"`` or ``"illegal"`` of a unified spec on
    ``arch``: the unified validator, the known-wrong values, then the general
    emission check on the general spec of the same tile step. A value the
    unified shell does not build (its not-built marks) is ``"not_built"``."""
    for name, ax in _BY_NAME.items():
        if name in _PRESET_AXES or ax.kind != KIND_COMPILE:
            continue
        if hasattr(spec, name) and getattr(spec, name) in ax.not_built:
            return "not_built"
    try:
        resolved = validate_attn_bwd_unified_spec(spec, arch)
    except ValueError:
        return "illegal"
    fields = {f.name: getattr(resolved, f.name) for f in dataclasses.fields(resolved)}
    if known_wrong_findings(arch, fields):
        return "illegal"
    if _not_built(resolved.general_spec().resolved(arch)):
        return "not_built"
    return "legal"


def check_page_tile_entry(key: tuple, name: str) -> None:
    """Raise ``ValueError`` unless ``key -> name`` is a legal tuned page-tile
    entry: ``key = (arch, head size, page)`` on a unified target, ``name`` a
    configuration the arch ships for the head size at ``stage_vec = 8`` (the
    shipped tuning looks the name up at the request's stage_vec and falls
    back to the tile rules where it is not shipped), legal on the arch for
    fp16 and bf16, serving the page, with no known-wrong value."""
    if not (isinstance(key, tuple) and len(key) == 3):
        raise ValueError(f"page-tile key {key!r} is not (arch, head size, page)")
    arch, d, page = key
    if arch not in VALIDATED_ARCHS:
        raise ValueError(f"page-tile key {key!r}: {arch!r} is not a unified target")
    if not (isinstance(page, int) and page > 0 and page % PAGE_GRANULE == 0):
        raise ValueError(f"page-tile key {key!r}: page is not a multiple of 16")
    if name not in unified_axis("page_tile").values(arch):
        raise ValueError(f"page-tile entry {key!r}: {name!r} is not a configuration")
    cfgs = attn_bwd_unified_tile_configs(arch, d, 8)
    if name not in cfgs:
        raise ValueError(f"page-tile entry {key!r}: {name!r} is not shipped at d{d}")
    knobs = dict(cfgs[name][1])
    for dtype in ("fp16", "bf16"):
        if not _page_legal(arch, d, dtype, 8, page, knobs):
            raise ValueError(
                f"page-tile entry {key!r}: {name!r} does not serve page {page} "
                f"({dtype})"
            )
        spec = AttnBwdUnifiedSpec(head_size=d, dtype=dtype, stage_vec=8, **knobs)
        status = unified_spec_legality(spec, arch)
        if status != "legal":
            raise ValueError(f"page-tile entry {key!r}: {name!r} is {status}")


def _check_registry() -> None:
    """One axis per unified tuning field and runtime value; the narrowed
    marks are non-default choices (raised at import)."""
    names = [ax.name for ax in UNIFIED_BWD_KNOB_AXES]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate unified backward knob axis")
    spec_fields = {f.name for f in dataclasses.fields(AttnBwdUnifiedSpec)}
    tuning = spec_fields - UNIFIED_PROBLEM_FIELDS
    covered = set(names) - set(_PRESET_AXES)
    runtime = {ax.name for ax in UNIFIED_BWD_KNOB_AXES if ax.kind == KIND_RUNTIME}
    if tuning - covered:
        raise AssertionError(
            f"unified spec tuning fields without an axis: {sorted(tuning - covered)}"
        )
    if covered - tuning - runtime:
        raise AssertionError(
            f"unified axes without a field: {sorted(covered - tuning - runtime)}"
        )
    if runtime != set(RUNTIME_KNOBS):
        raise AssertionError(
            f"unified runtime axes {sorted(runtime)} != RUNTIME_KNOBS "
            f"{sorted(RUNTIME_KNOBS)}"
        )
    for ax in UNIFIED_BWD_KNOB_AXES:
        extra = set(ax.not_built) | set(ax.full_only) | set(ax.gated)
        if not extra <= set(ax.choices) or ax.default in extra:
            raise AssertionError(
                f"{ax.name}: marked values must be non-default choices"
            )
    check_known_wrong(_BY_NAME)
    for key, name in UNIFIED_TUNED_PAGE_TILES.items():
        try:
            check_page_tile_entry(key, name)
        except ValueError as ex:
            raise AssertionError(f"tuned page-tile table: {ex}") from ex


_check_registry()
