# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Probe: knob constraints, and generating a sweep axis from one.

Needs an engine that declares constrained knobs.
"""

import hipdnn_frontend as hipdnn

from _common import build, emit

graph, handle, tensors = build()
report = {"knobs": [], "sweep": None}
sweep_source = None

for config in graph.get_engine_configs():
    for knob in graph.get_knobs_for_engine(config.engine_id):
        constraint = knob.constraint
        entry = {
            "engine_id": config.engine_id,
            "knob_id": knob.knob_id,
            "description": knob.description,
            "is_deprecated": knob.is_deprecated,
            "value_type": knob.value_type.name,
            "constraint": type(constraint).__name__,
            "repr": repr(constraint),
            "default_ok": knob.validate(
                hipdnn.KnobSetting(knob.knob_id, knob.default_value)
            ).is_good(),
        }
        if isinstance(constraint, hipdnn.IntConstraint):
            entry["min_value"] = constraint.min_value
            entry["max_value"] = constraint.max_value
            entry["step"] = constraint.step
            entry["valid_values"] = sorted(constraint.valid_values)
            entry["over_max_ok"] = knob.validate(
                hipdnn.KnobSetting(knob.knob_id, constraint.max_value + 1000)
            ).is_good()
            if sweep_source is None:
                values = sorted(constraint.valid_values) or list(
                    range(
                        constraint.min_value,
                        constraint.max_value + 1,
                        constraint.step,
                    )
                )
                sweep_source = (config.engine_id, knob.knob_id, values[:3])
        elif isinstance(constraint, hipdnn.FloatConstraint):
            entry["min_value"] = constraint.min_value
            entry["max_value"] = constraint.max_value
            entry["over_max_ok"] = knob.validate(
                hipdnn.KnobSetting(knob.knob_id, constraint.max_value + 1.0)
            ).is_good()
        elif isinstance(constraint, hipdnn.StringConstraint):
            entry["max_length"] = constraint.max_length
            entry["valid_values"] = sorted(constraint.valid_values)
            entry["unlisted_ok"] = knob.validate(
                hipdnn.KnobSetting(knob.knob_id, "no.such.value")
            ).is_good()
        report["knobs"].append(entry)

# The payoff: an axis generated from a constraint is accepted as a sweep.
if sweep_source is not None:
    engine_id, knob_id, values = sweep_source
    axis = hipdnn.KnobSweepAxis()
    axis.knob_id = knob_id
    axis.values = values
    spec = hipdnn.EngineSweepSpec()
    spec.engine_id = engine_id
    spec.axes = [axis]
    error = graph.add_engine_sweep([spec])
    report["sweep"] = {
        "knob_id": knob_id,
        "values": values,
        "accepted": error.is_good(),
        "message": error.get_message(),
    }

emit(report)
