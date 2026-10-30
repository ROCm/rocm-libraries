# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""torch.ops.hipdnn_vllm.custom_op_prepare / custom_op_exec.

Decline contract: prepare returns the reason and never raises; exec raises only
when strict. The plan key is op id + payload + input shapes, never addresses.
The engine-backed tests need the megakernel engine (ENABLE_MEGAKERNEL_ENGINE)
and skip without it.
"""

import json

import pytest
import torch

from hipdnn_vllm import stats
from hipdnn_vllm.declines import parse_op_error

UNKNOWN = "nosuchplugin.nosuchop"


def _inputs():
    return [torch.zeros(8, dtype=torch.int64, device="cuda"), torch.zeros(4, dtype=torch.int64, device="cuda")]


def _fallbacks(C):
    return {
        r["reason"]: r["count"]
        for r in C.stats_snapshot()["counters"]
        if r["op"] == "custom_op" and r["route"] == "fallback"
    }


@pytest.fixture
def enabled(C):
    C.custom_op_set_enabled(True)
    yield C
    C.custom_op_set_enabled(False)


def test_prepare_returns_reason_for_unknown_op_id(enabled):
    reason = torch.ops.hipdnn_vllm.custom_op_prepare(UNKNOWN, "{}", _inputs())
    assert reason and not reason.startswith("DISABLED"), reason


def test_exec_strict_raises_and_non_strict_counts_fallback(enabled):
    C = enabled
    stats.reset()
    torch.ops.hipdnn_vllm.custom_op_exec(UNKNOWN, '{"x":1}', _inputs(), False)
    counts = _fallbacks(C)
    assert sum(counts.values()) == 1, counts
    with pytest.raises(RuntimeError) as ei:
        torch.ops.hipdnn_vllm.custom_op_exec(UNKNOWN, '{"x":1}', _inputs(), True)
    op, decline = parse_op_error(ei.value)
    assert op == "custom_op" and decline.reason == next(iter(counts)), (ei.value, counts)


def test_plan_cache_keyed_by_shape_and_payload_not_address(enabled):
    C = enabled
    payload = '{"k":2}'
    torch.ops.hipdnn_vllm.custom_op_prepare(UNKNOWN, payload, _inputs())
    rt0 = C.stats_snapshot()["runtime"]
    # New tensors, same shapes: no build.
    torch.ops.hipdnn_vllm.custom_op_exec(UNKNOWN, payload, _inputs(), False)
    rt1 = C.stats_snapshot()["runtime"]
    assert rt1["plan_builds"] + rt1["plan_build_declines"] == rt0["plan_builds"] + rt0["plan_build_declines"]
    assert rt1["plan_cache_hits"] == rt0["plan_cache_hits"] + 1
    # Another payload or another shape is another plan.
    torch.ops.hipdnn_vllm.custom_op_prepare(UNKNOWN, '{"k":3}', _inputs())
    torch.ops.hipdnn_vllm.custom_op_prepare(
        UNKNOWN, payload, [torch.zeros(9, dtype=torch.int64, device="cuda"), _inputs()[1]]
    )
    rt2 = C.stats_snapshot()["runtime"]
    assert rt2["plan_builds"] + rt2["plan_build_declines"] == rt1["plan_builds"] + rt1["plan_build_declines"] + 2


def test_disabled_declines_without_building(C):
    C.custom_op_set_enabled(False)
    rt0 = C.stats_snapshot()["runtime"]
    assert torch.ops.hipdnn_vllm.custom_op_prepare(UNKNOWN, "{}", _inputs()) == "DISABLED"
    with pytest.raises(RuntimeError) as ei:
        torch.ops.hipdnn_vllm.custom_op_exec(UNKNOWN, "{}", _inputs(), True)
    assert parse_op_error(ei.value)[1].reason == "DISABLED"
    rt1 = C.stats_snapshot()["runtime"]
    assert rt1["plan_builds"] + rt1["plan_build_declines"] == rt0["plan_builds"] + rt0["plan_build_declines"]


def test_capture_miss_strict_raises_non_strict_records(enabled):
    C = enabled
    inputs = [torch.zeros(5, dtype=torch.int64, device="cuda"), torch.zeros(3, dtype=torch.int64, device="cuda")]
    torch.cuda.synchronize()
    stats.reset()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.hipdnn_vllm.custom_op_exec(UNKNOWN, '{"miss":1}', inputs, False)
    assert any(r.startswith("CAPTURE_INVARIANT") for r in _fallbacks(C)), _fallbacks(C)
    g2 = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError) as ei:
        with torch.cuda.graph(g2):
            torch.ops.hipdnn_vllm.custom_op_exec(UNKNOWN, '{"miss":2}', inputs, True)
    assert parse_op_error(ei.value)[1].reason.startswith("CAPTURE_INVARIANT")


# --- megakernel engine ------------------------------------------------------


_H6_SCRIPT = r"""
import json, torch
from hipdnn_vllm import _bootstrap, _C
_bootstrap.load()
dev = torch.cuda.current_device()
mk = [e["name"] for e in _C.engines(dev) if "megakernel" in e["name"].lower()]
x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
w = torch.randn(256, dtype=torch.bfloat16, device="cuda")
torch.ops.hipdnn_vllm.rms_norm(x, w, 1e-6)
torch.ops.hipdnn_vllm.layer_norm(x, w, w, 1e-6, True)
_, linear_engine = _C.linear_probe(x, torch.randn(64, 256, dtype=torch.bfloat16, device="cuda"))
tab = torch.zeros(8, dtype=torch.int64, device="cuda")
prep = torch.ops.hipdnn_vllm.custom_op_prepare("megakernel.qwen38-27b-mxfp4-gfx942", "{}", [tab, tab])
torch.cuda.synchronize()
print("H6", json.dumps({"mk": mk, "linear_engine": linear_engine, "prepare": prep}), flush=True)
"""


def test_engine_untouched_unless_opted_in_h6(tmp_path):
    """H6: with the megakernel engine built in and custom_op not opted in (the
    default HIPDNN_VLLM_OPS), the default ops keep their engines and the
    megakernel engine logs no decision (its every match/decline logs at INFO
    as hipkernel:Megakernel)."""
    import os
    import subprocess
    import sys

    log = tmp_path / "hipdnn.log"
    env = {k: v for k, v in os.environ.items() if k != "HIPDNN_VLLM_OPS"}
    env.update(HIPDNN_LOG_LEVEL="info", HIPDNN_LOG_FILE=str(log))
    out = subprocess.run([sys.executable, "-c", _H6_SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    res = json.loads(next(l for l in out.stdout.splitlines() if l.startswith("H6 "))[3:])
    if not res["mk"]:
        pytest.skip("megakernel engine not in this build")
    assert res["prepare"] == "DISABLED", res
    assert "megakernel" not in res["linear_engine"].lower(), res
    text = (log.read_text() if log.exists() else "") + out.stderr
    assert text, "no hipDNN log captured"
    assert "hipkernel:Megakernel" not in text, [l for l in text.splitlines() if "hipkernel:Megakernel" in l][:5]


def _megakernel_engine(C):
    names = [e["name"] for e in C.engines(torch.cuda.current_device())]
    return next((n for n in names if "megakernel" in n.lower()), None)


def _mk_case():
    """A runnable megakernel case: HIPDNN_VLLM_TEST_MK_CASE names a JSON file
    {"op_id", "payload", "tab": [int64...], "glob": [int64...]} whose code
    object is in the engine's catalog."""
    import os

    path = os.environ.get("HIPDNN_VLLM_TEST_MK_CASE")
    if not path:
        pytest.skip("HIPDNN_VLLM_TEST_MK_CASE not set")
    with open(path) as f:
        return json.load(f)


def test_megakernel_capture_replays(enabled):
    C = enabled
    if _megakernel_engine(C) is None:
        pytest.skip("megakernel engine not in this build")
    case = _mk_case()
    tab = torch.tensor(case["tab"], dtype=torch.int64, device="cuda")
    glob = torch.tensor(case["glob"], dtype=torch.int64, device="cuda")
    op_id, payload = case["op_id"], case["payload"]
    assert torch.ops.hipdnn_vllm.custom_op_prepare(op_id, payload, [tab, glob]) == ""
    torch.ops.hipdnn_vllm.custom_op_exec(op_id, payload, [tab, glob], True)
    torch.cuda.synchronize()
    eager = glob.clone()

    stats.reset()
    g = torch.cuda.CUDAGraph()
    glob.copy_(torch.tensor(case["glob"], dtype=torch.int64, device="cuda"))
    with torch.cuda.graph(g):
        torch.ops.hipdnn_vllm.custom_op_exec(op_id, payload, [tab, glob], True)
    captured = [r for r in C.stats_snapshot()["counters"] if r["op"] == "custom_op" and r["phase"] == "capture"]
    assert captured and captured[0]["route"] == "hipdnn", captured
    for _ in range(2):
        glob.copy_(torch.tensor(case["glob"], dtype=torch.int64, device="cuda"))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(glob, eager)
