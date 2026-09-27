# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Probe: add_engine_sweep() expansion over a Cartesian product of knob axes."""

import hipdnn_frontend as hipdnn

from _common import WORKSPACE, build, emit, pack, quick_config, settings_of

graph, handle, tensors = build()

# Find an engine with two constrained knobs to cross: an integer axis and a
# string axis, so the product is more than a relabelled single axis.
target = None
for config in graph.get_engine_configs():
    int_axis = None
    string_axis = None
    for knob in graph.get_knobs_for_engine(config.engine_id):
        constraint = knob.constraint
        if int_axis is None and isinstance(constraint, hipdnn.IntConstraint):
            values = sorted(constraint.valid_values) or list(
                range(constraint.min_value, constraint.max_value + 1, constraint.step)
            )
            if len(values) > 1:
                int_axis = (knob.knob_id, values[:4])
        elif string_axis is None and isinstance(constraint, hipdnn.StringConstraint):
            values = sorted(constraint.valid_values)
            if len(values) > 1:
                string_axis = (knob.knob_id, values[:3])
    if int_axis is not None and string_axis is not None:
        target = (config.engine_id, int_axis, string_axis)
        break

report = {"target": target}
if target is None:
    emit(report)
    raise SystemExit(0)

engine_id, (int_knob, int_values), (string_knob, string_values) = target

axis_a = hipdnn.KnobSweepAxis()
axis_a.knob_id = int_knob
axis_a.values = int_values
axis_b = hipdnn.KnobSweepAxis()
axis_b.knob_id = string_knob
axis_b.values = string_values
spec = hipdnn.EngineSweepSpec()
spec.engine_id = engine_id
spec.axes = [axis_a, axis_b]

added = graph.add_engine_sweep([spec])
report["added"] = {"ok": added.is_good(), "message": added.get_message()}
variant_pack, buffers = pack(tensors)
workspace = hipdnn.DeviceBuffer(WORKSPACE)
results = graph.autotune(
    handle,
    variant_pack,
    workspace.ptr(),
    workspace_size=WORKSPACE,
    config=quick_config(),
)
report["results"] = [
    {
        "engine_id": r.engine_id,
        "succeeded": r.succeeded,
        "settings": settings_of(r),
        "error": r.error_message,
    }
    for r in results
]

# A second sweep pins one knob through fixed_settings instead of sweeping it.
fixed_graph, fixed_handle, fixed_tensors = build()
single = hipdnn.KnobSweepAxis()
single.knob_id = int_knob
single.values = int_values
fixed_spec = hipdnn.EngineSweepSpec()
fixed_spec.engine_id = engine_id
fixed_spec.axes = [single]
fixed_spec.fixed_settings = {string_knob: string_values[0]}
fixed_added = fixed_graph.add_engine_sweep([fixed_spec])
report["fixed_added"] = {
    "ok": fixed_added.is_good(),
    "message": fixed_added.get_message(),
}
fixed_pack, fixed_buffers = pack(fixed_tensors)
fixed_results = fixed_graph.autotune(
    fixed_handle,
    fixed_pack,
    workspace.ptr(),
    workspace_size=WORKSPACE,
    config=quick_config(),
)
report["fixed_results"] = [
    {"succeeded": r.succeeded, "settings": settings_of(r)} for r in fixed_results
]

emit(report)
