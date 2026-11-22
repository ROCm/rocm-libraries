# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generic trace ingestion through the CLI: per-call versus whole-input refusal.

Unsupported observations are counted refusals that keep good calls; zero-call
configurations need the route facts' positive proof; malformed or mixed-mask input
refuses the whole run and replaces stale outputs.
"""
import json

import pytest
import yaml

from tests.specialize_test_support import (
    CALL_DATA, configured, generated, one_solution, refusal, row, specialize,
)


@pytest.mark.parametrize("missing", ["solution_Name", "kernel_name", "both"])
def test_roccroller_missing_names_refuses_only_that_call(missing, tmp_path, monkeypatch):
    good, unsupported = row(call_count=3), row(call_count=7, N=619)
    for field in (["solution_Name", "kernel_name"] if missing == "both" else [missing]):
        del unsupported[field]
    logs, facts = configured(tmp_path, [good, unsupported])
    doc = json.loads(facts.read_text())
    doc["calls"][1]["call_route"] = "RocRoller"
    facts.write_text(json.dumps(doc))
    result = generated(tmp_path, logs, facts, monkeypatch)
    refusal(result, 7, "RocRoller")
    assert len(result["logic"]) == 1
    assert [c["call_count"] for c in result["manifest"]["calls"]] == [3]


# Break: upgrade an old-profile record to host mode while writing CSV/manifest.
def test_record_without_extended_profile_generates_without_pointer_narrowing(tmp_path, monkeypatch):
    record = dict(yaml.load((CALL_DATA / "mask128_without_extended_fields.yaml").read_text(), Loader=yaml.BaseLoader)[0])
    # Re-bind the old observation to this fixture build's recipe explicitly.
    matched = row("f32")
    for field in ("solution_index", "solution_Name", "kernel_name"):
        record[field] = matched[field]
    result = specialize(tmp_path, [record], monkeypatch)
    assert result["code"] == 0
    assert len(result["calls"]) == 1
    logic = one_solution(result)
    assert logic["ProblemType"]["UseBeta"] is True
    assert logic["ProblemType"]["UseScaleAlphaVec"] == 1
    entry = result["manifest"]["calls"][0]
    assert "pointer_mode" not in entry["identity"]
    assert {"UseBeta", "UseScaleAlphaVec"} <= {n["fact"] for n in entry["facts_not_applied"]}


# A configuration with zero parsed calls (empty/missing log, [] or comments only) is
# reported only with route facts' zero_verified proof. Break: accept it from run.json
# metadata alone, reject a proven [] / comment-only log, or drop it from the manifest.
ZERO_CASES = [("empty", "verified"), ("missing", "verified"), ("sequence", "verified"), ("comments", "verified"),
              ("sequence", "zero_unverified"), ("comments", "zero_unverified"), ("empty", "zero_unverified"),
              ("empty", "missing_file"), ("empty", "nonzero_api")]


@pytest.mark.parametrize("log_form,proof", ZERO_CASES, ids=[f"{log}-{proof}" for log, proof in ZERO_CASES])
def test_zero_call_configuration_needs_route_zero_proof(log_form, proof, tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [], cfg="thin")
    log = logs / "hipblaslt_101.log"
    if log_form == "missing":
        log.unlink()
    elif log_form != "empty":
        log.write_text("[]\n" if log_form == "sequence" else "# producer logged zero calls\n")
    if proof == "missing_file":
        facts.unlink()
    elif proof != "verified":
        doc = json.loads(facts.read_text())
        if proof == "zero_unverified":
            doc["zero_verified"] = False
        else:
            doc["matmul_count"] = 1
            doc["per_process"]["101"]["rocblaslt_matmul"] = 1
        facts.write_text(json.dumps(doc))
    result = generated(tmp_path, logs, facts, monkeypatch)
    assert result["logic"] == {}
    assert result["calls"] == []
    if proof == "verified":
        assert result["code"] == 0
        assert result["refusals"] == []
        assert [c["run_configuration"] for c in result["manifest"]["zero_call_configurations"]] == ["thin"]
    else:
        assert result["code"] != 0, "zero calls require the route facts' positive proof"
        if log_form != "empty":
            assert len(result["refusals"]) == 1, "an unproven parsed-zero input needs a current refusal report"


# Break: skip bad records for a shrunken "successful" corpus, or return early and
# leave a previous run's corpus/manifest/logic in a reused workdir. User files stay.
@pytest.mark.parametrize("defect", ["grouped", "mask64", "unparseable", "malformed_run_json"])
def test_whole_input_failure_replaces_stale_outputs_in_reused_workdir(defect, tmp_path, monkeypatch, capsys):
    logs, facts = configured(tmp_path, [row()])
    before = generated(tmp_path, logs, facts, monkeypatch)
    assert before["code"] == 0
    assert len(before["calls"]) == len(before["manifest"]["calls"]) == len(before["logic"]) == 1
    assert before["refusals"] == []
    out = tmp_path / "out"
    owned_elsewhere = out / "user-owned.txt"
    owned_elsewhere.write_text("preserve this user artifact\n")
    trace = logs / "hipblaslt_101.log"
    bad_file = trace
    if defect == "grouped":
        # Producer-shaped unsupported grouped syntax (unquoted commas).
        trace.write_text(trace.read_text() + "- { function: grouped_matmul, M: 128,256, N: 128,256, call_count: 7 }\n")
    elif defect == "mask64":
        trace.write_text(trace.read_text() + (CALL_DATA / "producer_mask64.yaml").read_text())
    elif defect == "unparseable":
        trace.write_text(trace.read_text() + "- { broken\n")
    else:
        bad_file = logs / "run.json"
        bad_file.write_text("{ not JSON\n")
    capsys.readouterr()
    after = generated(tmp_path, logs, facts, monkeypatch)
    assert after["code"] != 0
    assert not (out / "calls.csv").exists(), "whole-input failure must remove the stale corpus"
    assert not (out / "specialize_manifest.json").exists()
    assert not (out / "logic").exists()
    assert len(after["refusals"]) == 1
    entry = after["refusals"][0]
    assert entry["file"] == str(bad_file)
    assert entry["reason"]
    assert "call_count" not in entry, "malformed input has no trustworthy call count"
    assert owned_elsewhere.read_text() == "preserve this user artifact\n"
    if defect == "grouped":
        text = capsys.readouterr()
        assert "grouped" in (text.out + text.err).lower()
