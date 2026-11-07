# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Host overhead per execute.

Measures, for a small rms_norm that is launch-bound:
* host time per torch.ops.hipdnn_vllm.rms_norm call (dispatcher + key + cache
  lookup + execute), and
* the part spent in Runtime::execute (stream bind + variant pack + hipDNN
  Graph::execute), from the C++ counters,
and the same call shape through torch's own rms_norm for scale. Results are
printed and, when HIPDNN_VLLM_BENCH_OUT is set, written there as JSON. The
numbers are host-specific and are recorded with the job, not in the repo.

The steady path is checked exactly: every timed call is one hipDNN execute of
the cached plan, with no plan build and no fallback.
"""

import json
import os
import time

import torch

from hipdnn_vllm import stats

N = 2000
WARMUP = 50


def _host_us_per_call(fn, n=N):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / n * 1e6


def test_host_overhead(C):
    x = torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    torch.ops.hipdnn_vllm.rms_norm(x, w, 1e-6)
    torch.cuda.synchronize()

    stats.reset()
    op_us = _host_us_per_call(lambda: torch.ops.hipdnn_vllm.rms_norm(x, w, 1e-6))
    rt = C.stats_snapshot()["runtime"]
    exec_us = rt["execute_host_ns"] / rt["execute_calls"] / 1e3
    native_us = _host_us_per_call(lambda: torch.nn.functional.rms_norm(x, (4096,), w, 1e-6))

    result = {
        "device": torch.cuda.get_device_properties(0).gcnArchName,
        "op_host_us_per_call": op_us,
        "runtime_execute_host_us_per_call": exec_us,
        "torch_rms_norm_host_us_per_call": native_us,
        "op_vs_torch_host_ratio": op_us / native_us,
        "calls": rt["execute_calls"],
    }
    print(json.dumps(result, indent=2))
    out = os.environ.get("HIPDNN_VLLM_BENCH_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    calls = N + WARMUP
    assert rt["execute_calls"] == calls, rt
    assert rt["plan_builds"] == 0 and rt["plan_build_declines"] == 0, rt
    routes = {(r["route"], r["reason"], r["phase"]): r["count"]
              for r in C.stats_snapshot()["counters"] if r["op"] == "rms_norm"}
    assert routes == {("hipdnn", "", "eager"): calls}, routes
