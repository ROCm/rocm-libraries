# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""torch.ops.hipdnn_vllm.custom_op_prepare / custom_op_exec.

Decline contract: prepare returns the reason and never raises; exec raises only
when strict. The plan key is op id + payload + input shapes, never addresses.
The engine-backed tests need the megakernel engine (ENABLE_MEGAKERNEL_ENGINE)
and skip without it.
"""

import json
import re

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
    assert rt1["plan_builds"] == rt0["plan_builds"]  # counts every build attempt, declined or not
    assert rt1["plan_cache_hits"] == rt0["plan_cache_hits"] + 1
    # Another payload or another shape is another plan.
    torch.ops.hipdnn_vllm.custom_op_prepare(UNKNOWN, '{"k":3}', _inputs())
    torch.ops.hipdnn_vllm.custom_op_prepare(
        UNKNOWN, payload, [torch.zeros(9, dtype=torch.int64, device="cuda"), _inputs()[1]]
    )
    rt2 = C.stats_snapshot()["runtime"]
    assert rt2["plan_builds"] == rt1["plan_builds"] + 2


def test_disabled_declines_without_building(C):
    C.custom_op_set_enabled(False)
    rt0 = C.stats_snapshot()["runtime"]
    assert torch.ops.hipdnn_vllm.custom_op_prepare(UNKNOWN, "{}", _inputs()) == "DISABLED"
    with pytest.raises(RuntimeError) as ei:
        torch.ops.hipdnn_vllm.custom_op_exec(UNKNOWN, "{}", _inputs(), True)
    assert parse_op_error(ei.value)[1].reason == "DISABLED"
    rt1 = C.stats_snapshot()["runtime"]
    assert rt1["plan_builds"] == rt0["plan_builds"]


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
prep = torch.ops.hipdnn_vllm.custom_op_prepare("megakernel.qwen38-27b-mxfp4-gfx942", '{"schema": 1}', [tab, tab])
torch.cuda.synchronize()
print("H6", json.dumps({"mk": mk, "linear_engine": linear_engine, "prepare": prep}), flush=True)
"""

# Every megakernel matcher decision logs at INFO: a decline as
# "hipkernel:Megakernel declined ...", a match as the ingestor's selection line.
_MK_DECISION = re.compile(r"hipkernel:Megakernel declined|'hipkernel:Megakernel'.*selected")


def _run_h6(tmp_path, name, ops):
    import os
    import subprocess
    import sys

    log = tmp_path / f"{name}.log"
    env = {k: v for k, v in os.environ.items() if k != "HIPDNN_VLLM_OPS"}
    if ops is not None:
        env["HIPDNN_VLLM_OPS"] = ops
    env.update(HIPDNN_LOG_LEVEL="info", HIPDNN_LOG_FILE=str(log))
    out = subprocess.run([sys.executable, "-c", _H6_SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    res = json.loads(next(l for l in out.stdout.splitlines() if l.startswith("H6 "))[3:])
    lines = ((log.read_text() if log.exists() else "") + out.stderr).splitlines()
    custom_nodes = [l for l in lines if "CustomOpOperationDescriptor" in l]
    return res, [l for l in lines if _MK_DECISION.search(l)], custom_nodes


def test_engine_untouched_unless_opted_in_h6(tmp_path):
    """H6: with the megakernel engine built in and custom_op not opted in (the
    default HIPDNN_VLLM_OPS), the default ops keep their engines, no CUSTOM_OP
    node reaches hipDNN and the megakernel matcher makes no decision. The
    opted-in run is the control that the same log shows both."""
    res, hits, nodes = _run_h6(tmp_path, "default", None)
    if not res["mk"]:
        pytest.skip("megakernel engine not in this build")
    assert res["prepare"] == "DISABLED", res
    assert "megakernel" not in res["linear_engine"].lower(), res
    assert nodes == [], nodes[:3]
    assert hits == [], hits[:5]

    from hipdnn_vllm.config import DEFAULT_OPS

    res, hits, nodes = _run_h6(tmp_path, "opted_in", ",".join(DEFAULT_OPS + ("custom_op",)))
    assert res["prepare"] not in ("", "DISABLED"), res
    assert "megakernel" not in res["linear_engine"].lower(), res
    assert nodes, "the opted-in control built no CUSTOM_OP node"
    # The catalog is gfx942-only; elsewhere the matcher never runs.
    if torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942"):
        assert hits, "the opted-in control logged no megakernel decision"


# The engine's stand-in kernel (test content, not in the wheel): with
# HIPDNN_DESCRIPTOR_DIR at the provider's packed integration set, op
# megakernel.megakernel-standin-gfx942 writes glob[i] = (0x4D4B << 48) |
# (nlayers << 32) | i for every thread i of the 304 x 256 grid and copies tab[0]
# into glob[304 * 256]. No weights, so it can run anywhere on an MI300X.
STANDIN_OP = "megakernel.megakernel-standin-gfx942"
STANDIN_THREADS = 304 * 256


def _standin_payload(nlayers):
    return json.dumps(
        {
            "schema": 1,
            "model": "megakernel-standin-gfx942",
            "src": "0123456789abcdef",
            "arch": "gfx942",
            "cus": 304,
            "kv": "ROCM_ATTN",
            "block": 784,
            "M": 1,
            "tp": 1,
            "rt": 0,
            "lm_head": False,
            "nlayers": nlayers,
        }
    )


def _standin_expected(tab0, nlayers):
    i = torch.arange(STANDIN_THREADS, dtype=torch.int64)
    return torch.cat([(0x4D4B << 48) | (nlayers << 32) | i, torch.tensor([tab0], dtype=torch.int64)])


def test_megakernel_capture_replays(enabled):
    C = enabled
    if not any("megakernel_standin" in e["name"] for e in C.engines(torch.cuda.current_device())):
        pytest.skip("megakernel stand-in not loaded (set HIPDNN_DESCRIPTOR_DIR to the integration set)")
    tab = torch.zeros(16, dtype=torch.int64, device="cuda")
    glob = torch.zeros(STANDIN_THREADS + 1, dtype=torch.int64, device="cuda")
    payload = _standin_payload(64)
    assert torch.ops.hipdnn_vllm.custom_op_prepare(STANDIN_OP, payload, [tab, glob]) == ""
    tab[0] = 0x0123456789ABCDEF
    torch.ops.hipdnn_vllm.custom_op_exec(STANDIN_OP, payload, [tab, glob], True)
    torch.cuda.synchronize()
    assert torch.equal(glob.cpu(), _standin_expected(0x0123456789ABCDEF, 64))

    # Capture with other buffers of the same shapes: the eager plan is reused
    # (no build under capture) and bound to the new addresses.
    tab2, glob2 = torch.zeros_like(tab), torch.zeros_like(glob)
    torch.cuda.synchronize()
    stats.reset()
    builds = C.stats_snapshot()["runtime"]["plan_builds"]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.hipdnn_vllm.custom_op_exec(STANDIN_OP, payload, [tab2, glob2], True)
    snap = C.stats_snapshot()
    captured = [r for r in snap["counters"] if r["op"] == "custom_op" and r["phase"] == "capture"]
    assert [r["route"] for r in captured] == ["hipdnn"], captured
    assert snap["runtime"]["plan_builds"] == builds
    for tab0 in (7, -3):
        glob2.zero_()
        tab2[0] = tab0
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(glob2.cpu(), _standin_expected(tab0, 64))
    # nlayers is a launch argument, not a plan key field the engine matches on,
    # but a different payload is still a different plan.
    assert torch.ops.hipdnn_vllm.custom_op_prepare(STANDIN_OP, _standin_payload(2), [tab, glob]) == ""
    torch.ops.hipdnn_vllm.custom_op_exec(STANDIN_OP, _standin_payload(2), [tab, glob], True)
    torch.cuda.synchronize()
    assert torch.equal(glob.cpu(), _standin_expected(int(tab[0]), 2))


def test_megakernel_key_mismatch_declines(enabled):
    C = enabled
    if not any("megakernel_standin" in e["name"] for e in C.engines(torch.cuda.current_device())):
        pytest.skip("megakernel stand-in not loaded (set HIPDNN_DESCRIPTOR_DIR to the integration set)")
    tab = torch.zeros(16, dtype=torch.int64, device="cuda")
    glob = torch.zeros(STANDIN_THREADS + 1, dtype=torch.int64, device="cuda")
    bad = json.loads(_standin_payload(64))
    bad["src"] = "fedcba9876543210"
    reason = torch.ops.hipdnn_vllm.custom_op_prepare(STANDIN_OP, json.dumps(bad), [tab, glob])
    assert reason, "a payload whose src matches no code object must decline"
    with pytest.raises(RuntimeError) as ei:
        torch.ops.hipdnn_vllm.custom_op_exec(STANDIN_OP, json.dumps(bad), [tab, glob], True)
    assert parse_op_error(ei.value)[1].reason == reason
