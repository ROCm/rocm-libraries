# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention backward dispatch: tuning registries, family routing, candidates.

* :mod:`.tuning_common`: every backward knob of the layout-general family as
  a :class:`~.tuning_common.KnobAxis`, :func:`backward_tuning_axes`, the
  legality helper :func:`backward_spec_legality` and the known-wrong knob
  values :data:`~.tuning_common.BWD_KNOWN_WRONG_KNOBS` (held out of every
  level, the legality verdicts, the tuned tables and the candidates). The
  sweep driver (``benchmarks/common/attention_bwd_sweep.py``) reads its
  catalog and lever families from it.
* :mod:`.dense_tuning`: the dense family's own knobs (the shared body's
  knobs are the general axes) and the tuned dense table.
* :mod:`.unified_tuning`: the unified (paged KV) family's knobs, its legality
  :func:`unified_spec_legality` and its tuned page-tile table.
* :mod:`.gfx942_tuning`, :mod:`.gfx950_tuning`: the per-arch tuned tables
  (general and dense; an entry is a key whose winner passed the bench's
  confirm stage) and :mod:`.tuned_tables`, the check every entry passes (its
  knobs are registered values and plan on representative requests of its
  key; run on every entry when the package is imported) and that turns the
  sweep's emitted entries into table entries. The performance-status tables
  are empty by policy (information only).
* :mod:`.routing`: the route of an accepted request (dense when its dense key
  is in the tuned pack, else general) and the plan of exactly one family;
  a rocKE unified (paged KV) request has its own entry,
  :func:`.routing.plan_attention_bwd_unified`.
* :mod:`.registry`: the dispatcher over them, in the forward dispatcher's
  shape: one candidate per (family, arch) in
  :data:`ATTENTION_BWD_REGISTRY`, the ranker
  :func:`attention_bwd_ranker` (tuned tables first, then the defaults), and
  the library-level entry points :func:`dispatch_attention_bwd`,
  :func:`dispatch_attention_bwd_all`, :func:`launch_attention_bwd` and
  :func:`run_attention_bwd`.
"""

from __future__ import annotations

from .dense_tuning import (
    DENSE_BWD_KNOB_AXES,
    DENSE_BWD_TUNED,
    dense_axis,
    dense_body_axes,
    dense_runtime_axes,
    dense_spec_legality,
    dense_tuning_axes,
)
from .registry import (
    ATTENTION_BWD_REGISTRY,
    AttentionBwdRequest,
    attention_bwd_candidates,
    attention_bwd_ranker,
    check_attention_bwd_tensors,
    dispatch_attention_bwd,
    dispatch_attention_bwd_all,
    launch_attention_bwd,
    run_attention_bwd,
)
from .routing import (
    BWD_ROUTES,
    attention_bwd_route,
    dense_policy,
    plan_attention_bwd,
    plan_attention_bwd_unified,
    routed_policy,
)
from .tuning_common import (
    BWD_KNOB_AXES,
    BWD_KNOWN_WRONG_KNOBS,
    LEVER_FAMILIES,
    BwdSpecLegality,
    KnobAxis,
    backward_axis,
    backward_spec_legality,
    backward_tuning_axes,
    knob_family,
)
from .unified_tuning import (
    UNIFIED_BWD_KNOB_AXES,
    UNIFIED_TUNED_PAGE_TILES,
    unified_axis,
    unified_spec_legality,
    unified_tuning_axes,
)

__all__ = [
    "ATTENTION_BWD_REGISTRY",
    "BWD_KNOB_AXES",
    "BWD_KNOWN_WRONG_KNOBS",
    "BWD_ROUTES",
    "DENSE_BWD_KNOB_AXES",
    "DENSE_BWD_TUNED",
    "LEVER_FAMILIES",
    "UNIFIED_BWD_KNOB_AXES",
    "UNIFIED_TUNED_PAGE_TILES",
    "AttentionBwdRequest",
    "BwdSpecLegality",
    "KnobAxis",
    "attention_bwd_candidates",
    "attention_bwd_ranker",
    "attention_bwd_route",
    "backward_axis",
    "backward_spec_legality",
    "backward_tuning_axes",
    "check_attention_bwd_tensors",
    "dense_axis",
    "dense_body_axes",
    "dense_policy",
    "dense_runtime_axes",
    "dense_spec_legality",
    "dense_tuning_axes",
    "dispatch_attention_bwd",
    "dispatch_attention_bwd_all",
    "knob_family",
    "launch_attention_bwd",
    "plan_attention_bwd",
    "plan_attention_bwd_unified",
    "routed_policy",
    "run_attention_bwd",
    "unified_axis",
    "unified_spec_legality",
    "unified_tuning_axes",
]
