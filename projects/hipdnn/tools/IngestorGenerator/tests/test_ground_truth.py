# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tools/ground_truth.py: ingest, graphs and compare on inline data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import ground_truth as gt  # noqa: E402
import registry_enumerator as re_  # noqa: E402

_MANIFEST = _ROOT / "configs" / "gfx950_attention_dense.engine.yaml"
_ROCKE = re_._REPO / "dnn-providers" / "hip-kernel-provider" / "rocke" / "library"
_CSV = """arch,b,hq,hk,sq,sk,d,dtype,mask,time_us,err
gfx950,1,8,8,512,512,128,bf16,TOP_LEFT_CAUSAL,100.0,0.001
gfx942,1,8,8,512,512,128,bf16,NO_MASK,90.0,
gfx950,2,16,2,1024,96,64,fp16,0,200.0,
"""
_MAP = "batch=b nhead_q=hq nhead_k=hk seqlen_q=sq seqlen_k=sk hdim_q=d hdim_v=d"
_MAP += " dtype=dtype mask_type=mask"
_REQUEST = dict(batch=1, nhead_q=8, nhead_k=8, seqlen_q=512, seqlen_k=512)
_REQUEST |= dict(hdim_q=128, hdim_v=128, dtype="bf16", mask_type="NO_MASK")


def _row(number, time_ms=0.1, device="gfx950", **overrides):
    request = {k: v for k, v in (_REQUEST | overrides).items() if v is not None}
    row = dict(request=request, time_ms=time_ms, accuracy={}, device=device)
    return row | {"_provenance": {"source": "ground_truth", "row": number}}


def _graphs(tmp_path, rows):
    gt_path, out = tmp_path / "GT.json", tmp_path / "graphs"
    gt_path.write_text(json.dumps(rows))
    rc = gt.main(f"graphs --gt {gt_path} --manifest {_MANIFEST} --out {out}".split())
    return rc, gt_path, out


def test_ingest_maps_columns_to_request_fields(tmp_path, capsys):
    if not _ROCKE.is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE}")
    source, out = tmp_path / "owner.csv", tmp_path / "GT.json"
    argv = f"ingest --input {source} --manifest {_MANIFEST} --time-col time_us "
    argv += f"--time-unit us --accuracy-col err --filter arch=gfx950 --out {out} "
    argv = (argv + "--device gfx950 --map " + " --map ".join(_MAP.split())).split()
    source.write_text(_CSV)
    assert gt.main(argv) == 0, capsys.readouterr().err
    first, third = json.loads(out.read_text())
    assert first == _row(1, mask_type="TOP_LEFT_CAUSAL") | {"accuracy": {"err": 0.001}}
    assert third["_provenance"]["row"] == 3, "filtered rows keep their numbering"
    assert third["time_ms"] == pytest.approx(0.2)
    shape = dict(batch=2, nhead_q=16, nhead_k=2, seqlen_q=1024, seqlen_k=96)
    shape |= dict(hdim_q=64, hdim_v=64, dtype="fp16")
    assert third["request"] == _REQUEST | shape, "mask 0 resolves by value"

    out.unlink()
    source.write_text(_CSV.replace("TOP_LEFT_CAUSAL", "causal"))
    assert gt.main(argv) == 2
    err = capsys.readouterr().err
    assert "row 1: mask_type <- mask='causal'" in err and "row 3" not in err
    assert not out.exists()


def _bshd(dims):
    """Gfx950AttentionDenseNative.cpp hasBshdStrides: [B,H,S,D] -> [S*H*D, D, H*D, 1]."""
    _, heads, seq, hdim = dims
    return [seq * heads * hdim, hdim, heads * hdim, 1]


def test_graphs_fill_placeholders_patches_and_bshd_strides(tmp_path):
    shape = dict(nhead_k=2, seqlen_k=96, hdim_v=32)
    rows = [_row(1, dtype="fp16", mask_type="TOP_LEFT_CAUSAL", **shape), _row(2)]
    rc, gt_path, out = _graphs(tmp_path, rows)
    assert rc == 0
    causal = json.loads((out / "ground_truth_r0001.json").read_text())
    plain = json.loads((out / "ground_truth_r0002.json").read_text())
    assert causal["_provenance"] == rows[0]["_provenance"]
    assert causal["io_data_type"] == "half" and plain["io_data_type"] == "bfloat16"
    assert causal["nodes"][0]["attributes"]["causal_mask"] is True
    assert plain["nodes"][0]["attributes"]["causal_mask"] is False
    for graph in (causal, plain):
        for tensor in graph["tensors"]:
            assert tensor["strides"] == _bshd(tensor["dims"]), tensor["name"]
            assert "stride_order" not in tensor
    dims = {t["name"]: t["dims"] for t in causal["tensors"]}
    assert dims == {
        "query": [1, 8, 512, 128],
        "key": [1, 2, 96, 128],
        "value": [1, 2, 96, 32],
        "output": [1, 8, 512, 32],
    }
    assert plain["tensors"][0]["strides"] == [524288, 128, 1024, 1]
    assert json.loads(gt.graphs_map_path(gt_path).read_text()) == [
        {"row": 1, "graph_name": "ground_truth_r0001"},
        {"row": 2, "graph_name": "ground_truth_r0002"},
    ]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"hdim_v": None}, "['hdim_v']"),
        ({"mask_type": "BOTTOM_RIGHT_CAUSAL"}, "lists no 'BOTTOM_RIGHT_CAUSAL'"),
        ({"sliding_window": 128}, "sliding_window is consumed by no template"),
    ],
)
def test_graphs_refuse_unusable_rows(tmp_path, capsys, overrides, message):
    rc, _, out = _graphs(tmp_path, [_row(1, **overrides)])
    assert rc == 2 and message in capsys.readouterr().err
    assert not out.exists()


def _phase(sweep, kind, *ledger):
    path = sweep / "attempts" / kind / "phase.json"
    path.parent.mkdir(parents=True)
    phase = {"kind": kind, "corpus": "ground_truth", "arm": "full", "round": 1}
    path.write_text(json.dumps({"status": "success", "phase": phase, "ledger": ledger}))


def _entry(row, outcome="served", **fields):
    return {"graph_name": f"ground_truth_r{row:04d}", "outcome": outcome, **fields}


def test_compare_reports_margin_correctness_and_missing(tmp_path, capsys):
    rows = [_row(n, time_ms=0.1 * n) for n in (1, 2, 3, 4)]
    assert _graphs(tmp_path, rows)[0] == 0
    sweep, report = tmp_path / "sweep", tmp_path / "report.md"
    declined = _entry(3, "declined", reason="no kernel")
    timed = _entry(1, mean_ms=0.104), _entry(2, mean_ms=0.26), _entry(4, mean_ms=0.4)
    _phase(sweep, "timing", *timed, declined)
    ok = {"passed": True, "execution_success": True, "tolerance_match": True}
    bad = ok | {"tolerance_match": False}
    ref, no_ref = {"reference": {"usable": True}}, {"reference": {"usable": False}}
    skipped = _entry(4, correctness=ok | {"tolerance_match": None}, **no_ref)
    checks = _entry(1, correctness=ok, **ref), _entry(2, correctness=bad, **ref)
    _phase(sweep, "correctness", *checks, skipped)
    identities = sweep / "input-identities.json"
    identities.write_text(json.dumps({"device": {"arch": "gfx950"}}))
    argv = f"compare --gt {tmp_path / 'GT.json'} --sweep-output {sweep} "
    argv = (argv + f"--margin 0.05 --report {report}").split()
    assert gt.main(argv) == 0
    summary = "4 rows, 2 within margin, 1 outside, 1 failing correctness, 1 missing"
    assert summary in capsys.readouterr().out
    # Only row 1 is within the margin with a passing verdict. Row 2 is outside and
    # fails, 3 is missing, and 4 is on time but has no usable reference.
    assert gt.main(argv + ["--strict"]) == 1
    assert "rows [2, 3, 4] are refused" in capsys.readouterr().err
    text = report.read_text()
    assert "declined: no kernel" in text
    assert "## Rows with no usable reference\n\n- row 4 " in text

    report.unlink()
    identities.write_text(json.dumps({"device": {"arch": "gfx942"}}))
    assert gt.main(argv) == 2
    assert "['gfx950']" in capsys.readouterr().err and not report.exists()


@pytest.mark.parametrize("checked, rc", [(True, 0), (False, 1)])
def test_strict_compare_wants_a_passing_correctness_verdict(tmp_path, checked, rc):
    assert _graphs(tmp_path, [_row(1, time_ms=0.1)])[0] == 0
    sweep = tmp_path / "sweep"
    _phase(sweep, "timing", _entry(1, mean_ms=0.104))
    if checked:
        ok = {"passed": True, "execution_success": True, "tolerance_match": True}
        _phase(
            sweep, "correctness", _entry(1, correctness=ok, reference={"usable": True})
        )
    (sweep / "input-identities.json").write_text(
        json.dumps({"device": {"arch": "gfx950"}})
    )
    argv = f"compare --gt {tmp_path / 'GT.json'} --sweep-output {sweep} --margin 0.05 "
    argv += f"--report {tmp_path / 'report.md'}"
    assert gt.main(argv.split()) == 0
    assert gt.main(argv.split() + ["--strict"]) == rc


def test_patch_pointers_name_their_node_not_an_index():
    graph = {"nodes": [{"name": "a", "x": 0}, {"name": "b", "x": 0}]}
    container, key = gt._pointer_parent(graph, "/nodes/b/x")
    container[key] = 1
    assert graph["nodes"] == [{"name": "a", "x": 0}, {"name": "b", "x": 1}]
    for pointer in ("/nodes/0/x", "/nodes/c/x"):
        with pytest.raises(gt.GroundTruthError, match="names no template location"):
            gt._pointer_parent(graph, pointer)
