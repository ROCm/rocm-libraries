# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""geko --specialize resilience: call aggregation, complete inputs, isolated derivation errors.

Injected failures sit at the external TensileLite parser boundary; successful
calls still use real recipe confirmation, derivation, naming and output writes.
"""
import json
from pathlib import Path

import pytest
import yaml

from geko.specialize import calls
from tests.specialize_test_support import (
    DATA, RECIPES, build_fixture, input_fixture, invoke,
    outputs, row, specialize, write_rows,
)
from tests.test_specialize_recipe_unit import command


# Pass: all printed NaN signs canonicalize for activation identity only, while
# signed infinities and unrelated finite facts stay distinct. Break: text-only
# NaN keys or float NaN equality inserts one row for each observation.
@pytest.mark.parametrize("argument", ["activation_arg0", "activation_arg1"])
def test_nonfinite_activation_arguments_aggregate_without_merging_infinity_signs(argument, tmp_path):
    values = [("nan", 2), ("-nan", 3), ("nan", 5),
              ("inf", 7), ("inf", 11), ("-inf", 13), ("-inf", 17)]
    trace = write_rows(tmp_path / "hipblaslt_101.log", [row(**{argument: v, "call_count": n}) for v, n in values])
    table = calls.load_calls(trace)
    assert len(table) == 3, "NaN is one activation-argument identity; +inf and -inf remain separate"
    assert sorted(entry.call_count for entry in table.values()) == [10, 18, 30]
    assert all(sum(entry.observations.values()) == entry.call_count for entry in table.values())
    assert all(sum(entry.sources.values()) == entry.call_count for entry in table.values())


# Pass: validate metadata before specializing any named input path, including a
# single log whose run.json is beside it. Break: bind only library_fp and ignore
# status/mode, so a SIGTERM dump or startup log appears to be a complete corpus.
@pytest.mark.parametrize("form", ["file", "configuration", "parent"])
@pytest.mark.parametrize("metadata", [
    {"status": "running"}, {"status": "failed"}, {"status": None},
    {"mode": "startup"}, {"mode": "analysis-api"}, {"mode": "performance"},
    {"mode": "greedy"}, {"mode": None},
], ids=["running", "failed", "missing-status", "startup", "mask16-mode", "graphs", "greedy", "missing-mode"])
def test_model_run_inputs_require_complete_analysis_metadata(form, metadata, tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row(call_count=7)])
    run = logs / "run.json"
    doc = json.loads(run.read_text())
    for key, value in metadata.items():
        if value is None:
            doc.pop(key)
        else:
            doc[key] = value
    run.write_text(json.dumps(doc))
    workload = {"file": logs / "hipblaslt_101.log", "configuration": logs, "parent": logs.parent}[form]
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, workload, facts.parent, out), monkeypatch))
    assert result["code"] == 1, "non-complete or non-analysis model_run traces must be refused"
    assert result["logic"] == {}
    assert result["refusals"], "invalid run metadata must produce a recorded refusal"
    reason = json.dumps(result["refusals"]).lower()
    assert "run.json" in reason
    assert next(iter(metadata)) in reason


# Pass: one bad configuration cannot be hidden by another complete one.
# Break: check only the first run.json, or mix the bad run's weight into the good.
def test_every_configuration_is_checked_before_writing_bespoke_logic(tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    good, _ = input_fixture(tmp_path, [row(call_count=3)], cfg="a-complete")
    bad, _ = input_fixture(tmp_path, [row(N=619, call_count=7)], cfg="z-incomplete")
    doc = json.loads((bad / "run.json").read_text())
    doc["status"] = "running"
    (bad / "run.json").write_text(json.dumps(doc))
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, good.parent, tmp_path / "route_facts", out), monkeypatch))
    assert result["code"] == 1
    assert result["refusals"]
    assert all(call["run_configuration"] != "z-incomplete" for call in result["manifest"].get("calls", []))


# Pass: convert any target RuntimeError into a per-call refusal with its original
# text, preserving successful real calls. Break: re-raise all errors except the
# old assertion-contradiction special case, or catch once around the whole run.
@pytest.mark.parametrize("stage", ["stock", "bespoke"])
def test_tensile_runtime_error_refuses_only_the_affected_call(stage, tmp_path, monkeypatch):
    import tensilelite.LibraryIO as library_io

    records = [row(call_count=3), row("f32", call_count=7), row(N=620, call_count=5)]
    control = specialize(tmp_path / "control", records, monkeypatch)
    assert control["code"] == 0
    assert len(control["manifest"]["calls"]) == 3
    target = yaml.safe_load((DATA / "stock" / RECIPES["f32"]["logic_file"]).read_text())["Solutions"][0]["SolutionIndex"]
    original = library_io.parseLibraryLogicFile

    def parse(path, *args, **kwargs):
        logic = yaml.safe_load(Path(path).read_text())
        solution = logic["Solutions"][0]
        is_bespoke = solution.get("BespokeNarrowing", False)
        if solution["SolutionIndex"] == target and is_bespoke == (stage == "bespoke"):
            raise RuntimeError("unsupported synthetic derivation feature")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(library_io, "parseLibraryLogicFile", parse)
    escaped = None
    result = None
    try:
        result = specialize(tmp_path / "failure", records, monkeypatch)
    except RuntimeError as exc:
        escaped = str(exc)
    assert escaped is None, f"TensileLite RuntimeError escaped instead of refusing one call: {escaped}"
    assert result["code"] == 1
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["call_count"] == 7
    assert "unsupported synthetic derivation feature" in result["refusals"][0]["reason"]
    assert sorted(call["call_count"] for call in result["manifest"]["calls"]) == [3, 5]
    assert len(result["logic"]) == 2


