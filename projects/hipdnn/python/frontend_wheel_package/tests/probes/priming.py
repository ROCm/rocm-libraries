# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Probe: EXHAUSTIVE priming under both PrimingFailurePolicy branches.

Needs an engine that exposes the ``global.benchmarking`` knob, which the stub
engine the pytest session pins does not.
"""

import hipdnn_frontend as hipdnn

from _common import WORKSPACE, build, emit, pack, quick_config

graph, handle, tensors = build()
capable = [c.engine_id for c in graph.get_engine_configs() if c.supports_exhaustive]
report = {"capable": capable, "runs": {}}

for policy in ("ABORT_ON_PRIMING_FAILURE", "BENCHMARK_UNPRIMED"):
    graph, handle, tensors = build()
    if not capable or not graph.add_engines(capable).is_good():
        break
    variant_pack, buffers = pack(tensors)
    workspace = hipdnn.DeviceBuffer(WORKSPACE)
    config = quick_config()
    config.mode = hipdnn.TuneMode.EXHAUSTIVE
    config.priming_failure_policy = getattr(hipdnn.PrimingFailurePolicy, policy)
    try:
        results = graph.autotune(
            handle,
            variant_pack,
            workspace.ptr(),
            workspace_size=WORKSPACE,
            config=config,
        )
    except RuntimeError as exc:
        report["runs"][policy] = {"error": str(exc)}
        continue
    report["runs"][policy] = {
        "results": [
            {
                "engine_id": r.engine_id,
                "succeeded": r.succeeded,
                "mode_used": r.mode_used.name,
                "supports_exhaustive": r.supports_exhaustive,
                "ran_exhaustive": r.ran_exhaustive,
                "reason": r.exhaustive_not_run_reason,
            }
            for r in results
        ]
    }

emit(report)
