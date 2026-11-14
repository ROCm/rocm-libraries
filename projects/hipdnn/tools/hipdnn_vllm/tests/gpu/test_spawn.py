# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""LLM(attention_backend="HIPDNN") with a spawned EngineCore. The child
unpickles the vLLM config, which holds the dynamic HIPDNN
AttentionBackendEnum member, before it loads the general plugins. vLLM spawns
when VLLM_WORKER_MULTIPROC_METHOD=spawn, and also when HIP is initialized
before LLM(); this test does both.

Needs a small decoder model: HIPDNN_VLLM_TEST_MODEL (HF id or local path)."""

import glob
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("vllm")

_SCRIPT = r"""
import sys

import torch


def main(model):
    torch.cuda.init()
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, attention_backend="HIPDNN", enforce_eager=True,
              gpu_memory_utilization=0.3, max_model_len=256)
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=4, temperature=0.0))
    print("SPAWN_OK", repr(out[0].outputs[0].text), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
"""


def test_llm_with_spawned_engine_core(tmp_path):
    from importlib.metadata import entry_points

    model = os.environ.get("HIPDNN_VLLM_TEST_MODEL")
    if not model:
        pytest.skip("HIPDNN_VLLM_TEST_MODEL not set")
    # vLLM's child processes find the plugin only through its installed entry point.
    if not any(ep.value.startswith("hipdnn_vllm") for ep in entry_points(group="vllm.general_plugins")):
        pytest.skip("hipdnn_vllm is not installed (no vllm.general_plugins entry point)")
    script = tmp_path / "spawn_llm.py"
    script.write_text(_SCRIPT)
    reports = tmp_path / "reports"
    env = dict(os.environ)
    for k in ("HIPDNN_VLLM_DISABLE", "HIPDNN_VLLM_OPS"):
        env.pop(k, None)
    env.update(
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        HIPDNN_VLLM_REPORT_DIR=str(reports),
        HIPDNN_VLLM_REPORT_INTERVAL="1",
    )
    out = subprocess.run([sys.executable, str(script), model], env=env, capture_output=True, text=True,
                         timeout=900)
    log = out.stdout + out.stderr
    assert "is not a valid AttentionBackendEnum" not in log, log[-4000:]
    assert out.returncode == 0 and "SPAWN_OK" in out.stdout, log[-4000:]
    routes = {}
    for path in glob.glob(str(reports / "report-*.json")):
        for r in json.load(open(path))["counters"]:
            if r["op"].startswith("attention"):
                routes[r["route"]] = routes.get(r["route"], 0) + r["count"]
    # The HIPDNN backend served the attention calls (on hipDNN or by fallback).
    assert sum(routes.values()) > 0, (routes, log[-2000:])
