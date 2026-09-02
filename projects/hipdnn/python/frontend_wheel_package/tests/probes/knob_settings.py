# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Probe: knob settings driven end to end against an engine that declares knobs."""

import hipdnn_frontend as hipdnn

from _common import WORKSPACE, build, emit, pack, quick_config, settings_of

graph, handle, tensors = build()
engines = {
    c.engine_id: [k.knob_id for k in graph.get_knobs_for_engine(c.engine_id)]
    for c in graph.get_engine_configs()
}
# Pick an engine with an integer knob that publishes an explicit value list.
target = None
for engine_id in sorted(engines):
    for knob in graph.get_knobs_for_engine(engine_id):
        constraint = knob.constraint
        if isinstance(constraint, hipdnn.IntConstraint) and constraint.valid_values:
            target = (engine_id, knob.knob_id, sorted(constraint.valid_values))
            break
    if target is not None:
        break

report = {"engines": engines, "target": target}
if target is None:
    emit(report)
    raise SystemExit(0)

engine_id, knob_id, values = target
chosen, other = values[0], values[-1]

accepted = graph.add_engine(engine_id, [hipdnn.KnobSetting(knob_id, chosen)])
report["accepted"] = {"ok": accepted.is_good(), "message": accepted.get_message()}

rejected = graph.add_engine(
    engine_id, [hipdnn.KnobSetting(knob_id, max(values) + 1000)]
)
report["rejected"] = {"ok": rejected.is_good(), "message": rejected.get_message()}

wrong_type = graph.add_engine(engine_id, [hipdnn.KnobSetting(knob_id, "text")])
report["wrong_type"] = {"ok": wrong_type.is_good(), "message": wrong_type.get_message()}

# Two variants of the same engine differing only in the knob value: the
# results must echo back exactly what was asked for.
variants = []
for value in (chosen, other):
    variant = hipdnn.EngineVariant()
    variant.engine_id = engine_id
    variant.knob_settings = {knob_id: value}
    variants.append(variant)

tuned, tuned_handle, tuned_tensors = build()
added = tuned.add_engine_variants(variants)
report["variants_added"] = {"ok": added.is_good(), "message": added.get_message()}
variant_pack, buffers = pack(tuned_tensors)
workspace = hipdnn.DeviceBuffer(WORKSPACE)
results = tuned.autotune(
    tuned_handle,
    variant_pack,
    workspace.ptr(),
    workspace_size=WORKSPACE,
    config=quick_config(),
)
report["variant_results"] = [
    {
        "engine_id": r.engine_id,
        "succeeded": r.succeeded,
        "settings": settings_of(r),
        "error": r.error_message,
    }
    for r in results
]

# The same settings drive a hard-selected plan end to end.
planned, planned_handle, planned_tensors = build()
created = planned.create_execution_plan_ext(
    engine_id, [hipdnn.KnobSetting(knob_id, other)]
)
report["plan_created"] = {"ok": created.is_good(), "message": created.get_message()}

# An illegal value has to be refused here too. If the binding dropped the
# settings vector, as it did while create_execution_plan_ext hardcoded an empty
# one, this would succeed and nothing else in the flow would notice.
plan_rejected = planned.create_execution_plan_ext(
    engine_id, [hipdnn.KnobSetting(knob_id, max(values) + 1000)]
)
report["plan_rejected"] = {
    "ok": plan_rejected.is_good(),
    "message": plan_rejected.get_message(),
}
built_plans = planned.build_plans()
report["plan_built"] = {
    "ok": built_plans.is_good(),
    "message": built_plans.get_message(),
}
planned_pack, planned_buffers = pack(planned_tensors)
plan_workspace_size = planned.get_workspace_size()
plan_workspace = (
    hipdnn.DeviceBuffer(plan_workspace_size) if plan_workspace_size > 0 else None
)
executed = planned.execute(
    planned_handle,
    planned_pack,
    plan_workspace.ptr() if plan_workspace is not None else 0,
)
report["plan_executed"] = {"ok": executed.is_good(), "message": executed.get_message()}
report["plan_engine_id"] = planned.get_execution_plan_engine_id()
report["plan_name"] = planned.get_plan_name()

emit(report)
