# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tools/ground_truth.py: ingest, graphs and compare on inline data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import ground_truth as gt  # noqa: E402
import sweep as sweep_tool  # noqa: E402
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
    mapping = json.loads(gt.graphs_map_path(gt_path).read_text())
    assert [(m["row"], m["graph_name"]) for m in mapping] == [
        (1, "ground_truth_r0001"),
        (2, "ground_truth_r0002"),
    ]
    graphs = {g["name"]: g for g in (causal, plain)}
    for entry in mapping:
        assert entry["semantic_identity"] == sweep_tool.semantic_identity(
            graphs[entry["graph_name"]]
        )


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


def _value_map_manifest(tmp_path, field, edit):
    """A copy of the shipped manifest whose value_map for ``field`` is ``edit``ed."""
    manifest = yaml.safe_load(_MANIFEST.read_text())
    block = manifest["ground_truth"]
    block["graph_template"] = str(_ROOT / "configs" / block["graph_template"])
    edit(block["value_map"][field])
    path = tmp_path / "m.engine.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


@pytest.mark.parametrize("command", ["ingest", "graphs"])
@pytest.mark.parametrize(
    "field, edit",
    [
        ("mask_type", lambda m: m.pop("NO_MASK")),
        ("mask_type", lambda m: m.update(BOTTOM_RIGHT_CAUSAL=True)),
        ("dtype", lambda m: m.pop("fp16")),
    ],
    ids=["missing-key", "extra-key", "missing-dtype"],
)
def test_a_value_map_must_cover_exactly_the_declared_cell_values(
    tmp_path, capsys, command, field, edit
):
    """The offending row's category need not occur in the owner rows."""
    if not _ROCKE.is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE}")
    manifest = _value_map_manifest(tmp_path, field, edit)
    out = tmp_path / "out"
    if command == "graphs":
        rows = tmp_path / "GT.json"
        rows.write_text(
            json.dumps([_row(1, dtype="bf16", mask_type="TOP_LEFT_CAUSAL")])
        )
        argv = f"graphs --gt {rows} --manifest {manifest} --out {out}"
    else:
        source = tmp_path / "owner.csv"
        source.write_text(_CSV)
        argv = f"ingest --input {source} --manifest {manifest} --time-col time_us "
        argv += f"--time-unit us --filter arch=gfx950 --out {out} --device gfx950 "
        argv += "--map " + " --map ".join(_MAP.split())
    assert gt.main(argv.split()) == 2
    assert field in capsys.readouterr().err
    assert not out.exists()


def _recorded(sweep) -> dict:
    """``{graph name: entry}`` of the row map `graphs` wrote beside GT.json."""
    mapping = json.loads(gt.graphs_map_path(sweep.parent / "GT.json").read_text())
    return {m["graph_name"]: m for m in mapping}


def _phase(sweep, kind, *ledger, number=1, status="success"):
    """A phase record whose ledger entries carry the identity `graphs` recorded for
    their graph, unless the entry names its own."""
    path = sweep / "attempts" / f"{kind}-r{number}" / "phase.json"
    path.parent.mkdir(parents=True)
    phase = {"kind": kind, "corpus": "ground_truth", "arm": "full", "round": number}
    recorded = _recorded(sweep)
    ledger = [
        {"semantic_identity": recorded[e["graph_name"]].get("semantic_identity"), **e}
        for e in ledger
    ]
    path.write_text(json.dumps({"status": status, "phase": phase, "ledger": ledger}))


def _entry(row, outcome="served", **fields):
    """A ledger entry; ``semantic_identity`` is filled by `_phase`."""
    return {"graph_name": f"ground_truth_r{row:04d}", "outcome": outcome, **fields}


def test_a_non_finite_accuracy_cell_goes_through_graphs_and_compare(tmp_path):
    """`ingest` keeps a NaN accuracy as the owner reported it; the row digest `graphs`
    records and `compare` rechecks must not refuse it."""
    row = _row(1, time_ms=0.1) | {"accuracy": {"err": float("nan")}}
    assert _graphs(tmp_path, [row])[0] == 0
    sweep, report = tmp_path / "sweep", tmp_path / "report.md"
    _phase(sweep, "timing", _entry(1, mean_ms=0.1))
    (sweep / "input-identities.json").write_text(
        json.dumps({"device": {"arch": "gfx950"}})
    )
    argv = f"compare --gt {tmp_path / 'GT.json'} --sweep-output {sweep} "
    argv = (argv + f"--margin 0.05 --report {report}").split()
    assert gt.main(argv) == 0


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


@pytest.mark.parametrize("time_ms", [0, -1.0, "fast"])
def test_graphs_refuse_a_ground_truth_time_that_is_not_positive(
    tmp_path, capsys, time_ms
):
    """`compare` divides the served time by it."""
    rc, _, out = _graphs(tmp_path, [_row(1, time_ms=time_ms)])
    assert rc == 2 and "time_ms" in capsys.readouterr().err
    assert not out.exists()


def test_graphs_refuse_two_rows_with_one_provenance_row(tmp_path, capsys):
    rc, _, out = _graphs(tmp_path, [_row(1), _row(1, nhead_q=16, nhead_k=16)])
    assert rc == 2 and "share a provenance row number" in capsys.readouterr().err
    assert not out.exists()


def test_graphs_refuse_a_directory_holding_other_graphs(tmp_path, capsys):
    (tmp_path / "graphs").mkdir()
    stale = tmp_path / "graphs" / "stale.json"
    stale.write_text("{}")
    rc, _, out = _graphs(tmp_path, [_row(1)])
    assert rc == 2
    assert "already holds other graphs: ['stale.json']" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["stale.json"]


@pytest.mark.parametrize("device", ["gfx9", "gfx950-dcgpu", ""])
def test_ingest_refuses_a_device_that_is_not_one_exact_arch(device):
    args = argparse.Namespace(map=[], filter=[], device=device)
    with pytest.raises(gt.GroundTruthError, match="not an exact gfx arch"):
        gt.ingest([], None, args)


def _compare(tmp_path, rows, *phases):
    """Graph the rows, lay out the timing `phases` ((entries, number, status)) and
    compare at margin 0.05 unless a test passes its own."""
    assert _graphs(tmp_path, rows)[0] == 0
    sweep = tmp_path / "sweep"
    for entries, number, status in phases:
        _phase(sweep, "timing", *entries, number=number, status=status)
    (sweep / "input-identities.json").write_text(
        json.dumps({"device": {"arch": "gfx950"}})
    )
    return tmp_path / "GT.json", sweep


def test_compare_takes_the_median_over_the_successful_served_rounds(tmp_path):
    """Rounds 1-3 serve 0.1, 0.1 and 0.4 ms: the median is 0.1 where the mean is 0.2.
    A failed round 4 and a declined entry in round 5 are not served times."""
    phases = [
        ([_entry(1, mean_ms=0.1)], 1, "success"),
        ([_entry(1, mean_ms=0.1)], 2, "success"),
        ([_entry(1, mean_ms=0.4)], 3, "success"),
        ([_entry(1, mean_ms=5.0)], 4, "failed"),
        ([_entry(1, "declined", mean_ms=9.0, reason="no kernel")], 5, "success"),
    ]
    gt_path, sweep = _compare(tmp_path, [_row(1, time_ms=0.1)], *phases)
    comparison = gt.compare(gt_path, sweep, None, 0.05)
    (result,) = comparison["results"]
    assert (result["rounds"], result["hipdnn_ms"]) == (3, 0.1)
    assert result["ratio"] == 1.0 and result["within_margin"] is True
    assert comparison["timing_rounds"] == [1, 2, 3, 5]


def test_a_ratio_equal_to_one_plus_margin_is_within_it(tmp_path):
    """0.75 / 0.5 is exactly 1.5, so margin 0.5 admits it and not 0.8 / 0.5."""
    rows = [_row(1, time_ms=0.5), _row(2, time_ms=0.5, nhead_q=16, nhead_k=16)]
    served = [_entry(1, mean_ms=0.75), _entry(2, mean_ms=0.8)]
    gt_path, sweep = _compare(tmp_path, rows, (served, 1, "success"))
    results = gt.compare(gt_path, sweep, None, 0.5)["results"]
    assert [r["ratio"] for r in results] == [1.5, 1.6]
    assert [r["within_margin"] for r in results] == [True, False]


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


_OK = {"passed": True, "execution_success": True, "tolerance_match": True}


def _compare_argv(tmp_path, sweep):
    argv = f"compare --gt {tmp_path / 'GT.json'} --sweep-output {sweep} --margin 0.05 "
    return (argv + f"--report {tmp_path / 'report.md'}").split()


def test_the_recorded_identity_is_the_one_a_sweep_computes_for_the_graph(tmp_path):
    rows = [_row(1), _row(2, nhead_q=16, nhead_k=16)]
    _, _, out = _graphs(tmp_path, rows)
    inventory = sweep_tool.corpus_inventory({"path": out, "expected_graphs": 2}, "none")
    recorded = _recorded(tmp_path / "sweep")
    assert {g["graph_name"]: g["semantic_identity"] for g in inventory} == {
        name: entry["semantic_identity"] for name, entry in recorded.items()
    }


def test_compare_refuses_a_ground_truth_row_edited_after_graphs(tmp_path, capsys):
    rows = [_row(1), _row(2, nhead_q=16, nhead_k=16)]
    served = [_entry(1, mean_ms=0.1), _entry(2, mean_ms=0.1)]
    gt_path, sweep = _compare(tmp_path, rows, (served, 1, "success"))
    rows = json.loads(gt_path.read_text())
    rows[1]["time_ms"] = 0.5
    gt_path.write_text(json.dumps(rows))
    assert gt.main(_compare_argv(tmp_path, sweep)) == 2
    assert "row 2" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


@pytest.mark.parametrize("identity", ["not-the-recorded-one", None])
def test_compare_refuses_a_ledger_graph_whose_identity_is_not_the_recorded_one(
    tmp_path, capsys, identity
):
    """A different or absent identity means the sweep timed some other graph under
    this graph's name."""
    phases = [([_entry(1, mean_ms=0.1, semantic_identity=identity)], 1, "success")]
    _, sweep = _compare(tmp_path, [_row(1)], *phases)
    assert gt.main(_compare_argv(tmp_path, sweep)) == 2
    assert "row 1" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


def test_strict_compare_refuses_a_graph_declined_in_any_timing_round(tmp_path):
    phases = [
        ([_entry(1, "declined", reason="no kernel")], 1, "success"),
        ([_entry(1, mean_ms=0.1)], 2, "success"),
    ]
    _, sweep = _compare(tmp_path, [_row(1, time_ms=0.1)], *phases)
    checked = _entry(1, correctness=_OK, reference={"usable": True})
    _phase(sweep, "correctness", checked)
    argv = _compare_argv(tmp_path, sweep)
    assert gt.main(argv) == 0
    assert gt.main(argv + ["--strict"]) == 1


@pytest.mark.parametrize("status, rc", [("success", 0), ("failed", 1)])
def test_strict_compare_refuses_a_failed_correctness_phase(tmp_path, status, rc):
    """The row's own verdict passes; its phase failed another gate."""
    timing = ([_entry(1, mean_ms=0.1)], 1, "success")
    _, sweep = _compare(tmp_path, [_row(1, time_ms=0.1)], timing)
    checked = _entry(1, correctness=_OK, reference={"usable": True})
    _phase(sweep, "correctness", checked, status=status)
    argv = _compare_argv(tmp_path, sweep)
    assert gt.main(argv) == 0
    assert gt.main(argv + ["--strict"]) == rc


def test_patch_pointers_accept_a_node_name_or_a_legacy_index():
    graph = {"nodes": [{"name": "a", "x": 0}, {"name": "b", "x": 0}]}
    for pointer, expected in (("/nodes/b/x", "b"), ("/nodes/0/x", "a")):
        container, key = gt._pointer_parent(graph, pointer)
        container[key] = 1
        assert [n["x"] for n in graph["nodes"] if n["name"] == expected] == [1]
    for pointer in ("/nodes/c/x", "/nodes/2/x"):
        with pytest.raises(gt.GroundTruthError, match="names no template location"):
            gt._pointer_parent(graph, pointer)
