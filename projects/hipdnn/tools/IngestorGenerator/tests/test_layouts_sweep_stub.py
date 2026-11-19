# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The graphs `ground_truth.py layouts` writes are a corpus tools/sweep.py can stage,
time through its stub benchmark, ledger and resume. The stub (test_sweep_tools) is a
control-flow fixture: nothing here times a kernel."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from tests.test_sweep_tools import Sweep, _needs_posix_exec

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import ground_truth as gt  # noqa: E402

_SPEC = dict(
    dtypes=["bf16", "fp16"],
    head_dims=[64, 128],
    mask_types=["NO_MASK", "TOP_LEFT_CAUSAL"],
    layouts=list(gt.LAYOUTS),
    shapes=[
        dict(id="gqa", batch=2, nhead_q=32, nhead_k=8, seqlen_q=2048),
        dict(id="mha", batch=1, nhead_q=32, nhead_k=32, seqlen_q=4096),
        dict(id="tp", batch=2, nhead_q=16, nhead_k=2, seqlen_q=1024),
    ],
)


def _staged(tmp_path, arch):
    """A staged sweep whose corpus is the 120 layouts graphs of ``arch``'s manifest."""
    staged = Sweep(tmp_path, graphs=1)
    shapes, out = tmp_path / "shapes.yaml", tmp_path / "layouts"
    shapes.write_text(yaml.safe_dump(_SPEC))
    manifest = _ROOT / "configs" / f"{arch}_attention_dense.engine.yaml"
    assert (
        gt.main(f"layouts --manifest {manifest} --shapes {shapes} --out {out}".split())
        == 0
    )
    for stale in staged.corpus.glob("*.json"):
        stale.unlink()
    for path in out.glob("*.json"):
        shutil.copy2(path, staged.corpus / path.name)
    config = json.loads(staged.config_path.read_text())
    config["corpora"][0]["expected_graphs"] = 120
    config["min_served"] = 120
    staged.config_path.write_text(json.dumps(config))
    return staged, yaml.safe_load(gt.layouts_map_path(out).read_text())


def _ledger(staged):
    (phase,) = (staged.root / "results" / "attempts").glob("timing__*/phase.json")
    return json.loads(phase.read_text())["ledger"]


@_needs_posix_exec
@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
def test_the_sweep_driver_stages_ledgers_and_resumes_120_layout_graphs(tmp_path, arch):
    staged, mapping = _staged(tmp_path, arch)
    result = staged.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SWEEP_TIMING_ONLY" in result.stdout
    ledger = _ledger(staged)
    assert [e["graph_name"] for e in ledger] == [m["graph_name"] for m in mapping]
    assert all(e["outcome"] == "served" for e in ledger)
    # the join key `ground_truth.py compare` uses survives staging
    assert [e["semantic_identity"] for e in ledger] == [
        m["semantic_identity"] for m in mapping
    ]
    assert staged.resumed(staged.run()), "an unchanged rerun resumes"


@_needs_posix_exec
def test_editing_one_layout_graph_invalidates_the_resume(tmp_path):
    staged, _ = _staged(tmp_path, "gfx950")
    assert staged.run().returncode == 0
    victim = staged.corpus / "layouts_r0002.json"
    graph = json.loads(victim.read_text())
    graph["tensors"][0]["strides"][1] += 8
    victim.write_text(json.dumps(graph, indent=4) + "\n")
    assert not staged.resumed(staged.run())
