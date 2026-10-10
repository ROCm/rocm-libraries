# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention backward dispatch: the tuning registry of the backward knobs.

Only the registry exists so far (:mod:`.tuning_common`): every backward knob
as a :class:`~.tuning_common.KnobAxis`, :func:`backward_tuning_axes` and the
legality helper :func:`backward_spec_legality`. The sweep driver
(``benchmarks/common/attention_bwd_sweep.py``) reads its catalog and lever
families from it.
"""

from __future__ import annotations

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
    "LEVER_FAMILIES",
    "BwdSpecLegality",
    "KnobAxis",
    "backward_axis",
    "backward_spec_legality",
    "backward_tuning_axes",
    "knob_family",
]
