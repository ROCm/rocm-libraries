# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Route-facts producer: ship a producer that ordinary users can run and consume.

Pass: read both log masks, reconcile weighted counts, and emit evidence-backed
facts consumed by the real specialization CLI. Break: leave the producer only
in private tooling, assume an absent API witness, or silently accept a mismatch.
The API fixture is synthetic producer-shaped text; it is not live route proof.
"""
import json
from pathlib import Path

import pytest

from tests.specialize_test_support import build_fixture, input_fixture, invoke, outputs, row, tree_hash
from tests.test_specialize_recipe_unit import command


def api_trace(root, count, *, amax=False, bias_gradient=False):
    root.mkdir(parents=True)
    prefix = "[2026-11-19 00:00:00][HIPBLASLT][101][Api]"
    lines = [prefix + "[rocblaslt_create] handle[out]=0x1234"]
    if amax:
        lines.append(prefix + "[rocblaslt_matmul_desc_set_attribute] matmulDesc=0x1234 attr=MATMUL_DESC_AMAX_D_POINTER buf=0x100000000 sizeInBytes=8 bufData=0x0")
    if bias_gradient:
        lines.append(prefix + "[rocblaslt_matmul_desc_set_attribute] matmulDesc=0x1234 attr=MATMUL_DESC_EPILOGUE buf=0x1000 sizeInBytes=4 bufData=BGRADB")
    for _ in range(count):
        lines.append(prefix + "[rocblaslt_matmul_algo_get_heuristic] returnAlgoCount=1")
        lines.append(prefix + "[rocblaslt_matmul] A=0x1000 Adesc=0x2000 B=0x3000 Bdesc=0x4000 C=0x5000 Cdesc=0x6000 D=0x5000 Ddesc=0x6000 computeDesc=0x1234 algo=0x7000 workSpace=0 workSpaceSizeInBytes=0 stream=0")
    (root / "hipblaslt_101.log").write_text("\n".join(lines) + "\n")
    library = root.parent / "baseline/Tensile/library/gfx950"  # the library input_fixture binds the trace to
    (root / "run.json").write_text(json.dumps({"status": "complete", "mode": "analysis-api", "run_configuration": "fixture-on",
                                               "model": {"slug": "fixture-model"}, "library_fp": tree_hash(library),
                                               "env_values": {}, "process_inventory": [{"pid": 101, "role": "engine"}]}))
    return root


def produce(logs, api, out, monkeypatch):
    return invoke(["route-facts", str(logs), "--api-log", str(api), "--out", str(out)], monkeypatch)


# Pass: the producer's per-call facts include real file/line/hash evidence, and
# those facts actually allow generation. Break: return an empty guessed table,
# omit identity/evidence, or use strings the downstream consumer does not accept.
def test_public_route_facts_producer_output_drives_real_specialization(tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, _ = input_fixture(tmp_path, [row(call_count=5, bias_vector=True)])
    api = api_trace(tmp_path / "api", 5)
    facts = tmp_path / "produced.json"
    code = produce(logs, api, facts, monkeypatch)
    assert code == 0, "geko must ship the public route-facts producer, not require private project tooling"
    assert facts.is_file(), "successful producer must write consumable route facts"
    document = json.loads(facts.read_text())
    assert document["mask128_call_count"] == 5
    assert document["matmul_count"] == 5
    assert len(document["calls"]) == 1
    entry = document["calls"][0]
    assert entry["call_route"] == "ordinary matmul API"
    assert entry["bias_axis"] == "M"
    assert entry["output_amax"] is False
    for fact in ("call_route", "bias_axis", "output_amax"):
        witnesses = entry["evidence"][fact]
        assert witnesses, f"producer guessed {fact} without evidence"
        assert all(witness.get("file") and len(witness.get("sha256", "")) == 64 for witness in witnesses)
        assert all(witness.get("line", witness.get("line_start", 0)) > 0 for witness in witnesses)
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs, facts, out, extra=["--keep-beta"]), monkeypatch))
    assert result["code"] == 0
    assert len(result["logic"]) == 1
    assert result["manifest"]["calls"][0]["call_count"] == 5


# Pass: raw mask128 weights must reconcile with every API matmul record.
# Break: compare row count rather than call_count, or accept missing API calls.
def test_public_route_facts_producer_refuses_unreconciled_counts(tmp_path, monkeypatch, capsys):
    build_fixture(tmp_path / "baseline")
    logs, _ = input_fixture(tmp_path, [row(call_count=5)])
    api = api_trace(tmp_path / "api", 4)
    facts = tmp_path / "produced.json"
    code = produce(logs, api, facts, monkeypatch)
    assert code == 1, "route producer must refuse a 5-call trace with only 4 API matmuls"
    diagnostic = capsys.readouterr()
    text = diagnostic.out + diagnostic.err + (facts.read_text() if facts.is_file() else "")
    assert "5" in text and "4" in text
    if facts.is_file():
        assert not json.loads(facts.read_text()).get("reconciliation", {}).get("matched", False)


# Pass: seeing an AMAX pointer attribute is not proof of absence, even when the
# printed low pointer bits are zero. Break: emit output_amax=false from bufData=0.
def test_public_route_facts_producer_never_proves_amax_absent_after_attribute_set(tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, _ = input_fixture(tmp_path, [row(call_count=5)])
    api = api_trace(tmp_path / "api", 5, amax=True)
    facts = tmp_path / "produced.json"
    code = produce(logs, api, facts, monkeypatch)
    assert code in (0, 1), "valid producer inputs must not be an unrecognized CLI invocation"
    if code == 0:
        assert facts.is_file()
        assert all(entry.get("output_amax") is not False for entry in json.loads(facts.read_text())["calls"])
        out = tmp_path / "out"
        result = outputs(out, invoke(command(tmp_path, logs, facts, out), monkeypatch))
        assert result["code"] == 1
        assert result["logic"] == {}


# Pass: every unsafe trace/API observation is a reasoned stop, consumed as a
# refusal by specialization. Break: omit the named whole-trace safety check.
@pytest.mark.parametrize("record_changes,api_line,missing_heuristic,stop", [
    ({"function": "extension_matmul"}, "", False, "nonordinary-trace"),
    ({"function": None}, "", False, "nonordinary-trace"),
    ({"bias_source": "a"}, "", False, "bias-axis"),
    ({"gradient": True}, "", False, "bias-axis"),
    ({}, "getAlgosFromIndex index=17", False, "pinned-index"),
    ({}, "OverrideAlgoCount=1", False, "tuning-override"),
    ({}, "TunableOp enabled", False, "tunableop"),
    ({}, "[rocblaslt_matmul_desc_set_attribute] attr=MATMUL_DESC_EPILOGUE bufData=256", False, "bias-axis"),
    ({}, "[rocblaslt_matmul_desc_set_attribute] attr=MATMUL_DESC_EPILOGUE bufData=0x200", False, "bias-axis"),
    ({}, "[rocblaslt_matmul_desc_set_attribute] attr=MATMUL_DESC_EPILOGUE bufData=BGRADB", False, "bias-axis"),
    ({}, "[rocblaslt_matmul_desc_set_attribute] attr=MATMUL_DESC_EPILOGUE", False, "bias-axis"),
    ({}, "[rocblaslt_matmul_get_all_algos] count=1", False, "nonordinary-api"),
    ({}, "", True, "missing-heuristic"),
], ids=["nonordinary-trace", "unknown-function", "bias-source", "gradient", "pinned-index",
        "override", "tunableop", "bgrada", "bgradb", "unknown-epilogue", "missing-epilogue",
        "nonordinary-api", "missing-heuristic"])
def test_public_route_facts_stops_are_consumed_as_counted_refusals(
        record_changes, api_line, missing_heuristic, stop, tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    record = row(call_count=5, **{key: value for key, value in record_changes.items() if value is not None})
    for key, value in record_changes.items():
        if value is None:
            del record[key]
    logs, _ = input_fixture(tmp_path, [record])
    api = api_trace(tmp_path / "api", 5)
    api_file = api / "hipblaslt_101.log"
    lines = api_file.read_text().splitlines()
    if missing_heuristic:
        lines = [line for line in lines if "[rocblaslt_matmul_algo_get_heuristic]" not in line]
    if api_line:
        lines.append("[2026-11-19 00:00:00][HIPBLASLT][101][Api]" + api_line)
    api_file.write_text("\n".join(lines) + "\n")
    facts = tmp_path / "produced.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    matching = [entry for entry in document["stops"] if entry["code"] == stop]
    assert matching and all(entry["evidence"] for entry in matching)
    assert document["calls"] == [] and document["mask128_call_count"] == 5
    output = tmp_path / "out"
    result = outputs(output, invoke(command(tmp_path, logs, facts, output), monkeypatch))
    assert result["code"] == 1 and result["logic"] == {}
    assert len(result["refusals"]) == 1 and result["refusals"][0]["call_count"] == 5
    assert result["refusals"][0]["code"] == "route-facts-stopped"
    assert stop in result["refusals"][0]["reason"]


# Pass: numeric non-gradient epilogues remain ordinary and usable. Break: stop
# on every epilogue attribute rather than only unknown/gradient values.
def test_public_route_facts_accepts_numeric_forward_epilogue(tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, _ = input_fixture(tmp_path, [row(call_count=5)])
    api = api_trace(tmp_path / "api", 5)
    api_file = api / "hipblaslt_101.log"
    with api_file.open("a") as stream:
        stream.write("[2026-11-19 00:00:00][HIPBLASLT][101][Api]"
                     "[rocblaslt_matmul_desc_set_attribute] attr=MATMUL_DESC_EPILOGUE bufData=0x0\n")
    facts = tmp_path / "produced.json"
    assert produce(logs / "hipblaslt_101.log", api / "hipblaslt_101.log", facts, monkeypatch) == 0
    document = json.loads(facts.read_text())
    assert document["stops"] == [] and document["api_epilogues"][0]["value"] == "0x0"
    output = tmp_path / "out"
    result = outputs(output, invoke(command(tmp_path, logs, facts, output), monkeypatch))
    assert result["code"] == 0 and len(result["logic"]) == 1
    assert result["manifest"]["calls"][0]["call_count"] == 5


# Pass: malformed producer input is a usage refusal with a diagnostic and no
# claimed route file. Break: accept invalid JSON/non-object metadata or traces.
@pytest.mark.parametrize("defect,reason", [
    ("trace", "mask-128"), ("run-json", "JSON"), ("run-array", "JSON object"),
    ("empty-directory", "hipblaslt_"),
])
def test_public_route_facts_input_errors_explain_the_refusal(defect, reason, tmp_path, monkeypatch, capsys):
    build_fixture(tmp_path / "baseline")
    logs, _ = input_fixture(tmp_path, [row(call_count=5)])
    api = api_trace(tmp_path / "api", 5)
    if defect == "trace":
        (logs / "hipblaslt_101.log").write_text("- { incomplete\n")
    elif defect.startswith("run-"):
        (logs / "run.json").write_text("{" if defect == "run-json" else "[]")
    else:
        logs = tmp_path / "empty"
        logs.mkdir()
    facts = tmp_path / "produced.json"
    assert produce(logs, api, facts, monkeypatch) == 2
    assert reason in capsys.readouterr().err
    assert not facts.exists()
