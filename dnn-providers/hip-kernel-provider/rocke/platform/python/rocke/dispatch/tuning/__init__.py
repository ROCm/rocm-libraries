# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared machinery for tuned (sweepable) dispatch candidates.

A family with a tuning space supplies its data and rules; this package
supplies everything else:

- :mod:`.axes` -- :class:`KnobAxis` and the helpers to declare axes;
- :mod:`.walk` -- sweep levels, the pruned walk, the sampler;
- :mod:`.identity` -- ``config_key`` / ``tuning_id``;
- :mod:`.space` -- :class:`KnobSpace`, the one place specs are made;
- :mod:`.candidate` -- :func:`make_tuned_candidate` and pin resolution;
- :mod:`.spec` -- the :class:`TunedSpec` / :class:`TunableRequest` protocols;
- :mod:`.testing` -- :func:`assert_tuning_contract`, the conformance kit.

See ``rocke/dispatch/ARCHITECTURE.md`` section 14 for how a family uses it.
"""

from .axes import (
    KnobAxis,
    Knobs,
    axis_knob_names,
    choices,
    flag,
    gated,
    knob_requirements,
    knob_types,
    sorted_items,
    values,
)
from .candidate import explicitly_pinned, make_tuned_candidate, resolve_pinned
from .identity import (
    TUNING_ID_VERSION,
    config_key,
    defaults_fingerprint,
    key_of,
    knob_items,
    normalize_knobs,
    tuning_id,
)
from .space import BUILD_ERRORS, KnobSpace, Verdict
from .spec import TunableRequest, TunedSpec
from .walk import (
    SWEEP_LEVELS,
    configure_sweep,
    current_sweep_level,
    iter_at_level,
    sample_count,
    sweep_level,
)

__all__ = [
    "BUILD_ERRORS",
    "KnobAxis",
    "KnobSpace",
    "Knobs",
    "SWEEP_LEVELS",
    "TUNING_ID_VERSION",
    "TunableRequest",
    "TunedSpec",
    "Verdict",
    "axis_knob_names",
    "choices",
    "config_key",
    "configure_sweep",
    "current_sweep_level",
    "defaults_fingerprint",
    "explicitly_pinned",
    "flag",
    "gated",
    "iter_at_level",
    "key_of",
    "knob_items",
    "knob_requirements",
    "knob_types",
    "make_tuned_candidate",
    "normalize_knobs",
    "resolve_pinned",
    "sample_count",
    "sorted_items",
    "sweep_level",
    "tuning_id",
    "values",
]
