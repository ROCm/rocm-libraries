# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention backward dispatch: tuning registries and family routing.

* :mod:`.tuning_common`: every backward knob of the layout-general family as
  a :class:`~.tuning_common.KnobAxis`, :func:`backward_tuning_axes` and the
  legality helper :func:`backward_spec_legality`. The sweep driver
  (``benchmarks/common/attention_bwd_sweep.py``) reads its catalog and lever
  families from it.
* :mod:`.dense_tuning`: the dense family's own knobs (the shared body's
  knobs are the general axes) and the tuned dense table.
* :mod:`.routing`: the route of an accepted request (dense when its dense key
  is in the tuned pack, else general) and the plan of exactly one family;
  a rocKE unified (paged KV) request has its own entry,
  :func:`.routing.plan_attention_bwd_unified`.
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
    LEVER_FAMILIES,
    BwdSpecLegality,
    KnobAxis,
    backward_axis,
    backward_spec_legality,
    backward_tuning_axes,
    knob_family,
)

__all__ = [
    "BWD_KNOB_AXES",
    "BWD_ROUTES",
    "DENSE_BWD_KNOB_AXES",
    "DENSE_BWD_TUNED",
    "LEVER_FAMILIES",
    "BwdSpecLegality",
    "KnobAxis",
    "attention_bwd_route",
    "backward_axis",
    "backward_spec_legality",
    "backward_tuning_axes",
    "dense_axis",
    "dense_body_axes",
    "dense_policy",
    "dense_runtime_axes",
    "dense_spec_legality",
    "dense_tuning_axes",
    "knob_family",
    "plan_attention_bwd",
    "plan_attention_bwd_unified",
    "routed_policy",
]
