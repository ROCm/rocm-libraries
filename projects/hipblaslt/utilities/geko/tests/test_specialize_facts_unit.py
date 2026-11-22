# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Proven request facts, conservative narrowing of unknowns, and the call domain.

facts.py separates generation requirements from narrowing and applies only sourced
facts; writer.py emits logic matching the independently checked-in oracle.
Break: default UNKNOWN to host/none/false, ignore route/domain facts, or change fixed
stock parameters instead of only the expressible narrowing controls.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

from geko.specialize import facts as facts_api
from geko.specialize import recipe
from tests.specialize_test_support import (
    DATA, ORACLE, RECIPES, configured, effective_solution, generated, identity, input_fixture,
    one_solution, oracle_logic, case_record, refusal, row, specialize, write_rows,
)


def checked_stock_error(tmp_path, problem_type, keep_beta=False):
    from geko.specialize.calls import load_calls
    call = next(iter(load_calls(write_rows(tmp_path / "hipblaslt_101.log", [row()]))))
    try:
        facts_api.check_stock(call, problem_type, batched=True, keep_beta=keep_beta)
    except facts_api.Refused as caught:
        return caught
    return None


def stock_problem_type(**changes):
    problem_type = yaml.safe_load((DATA / "stock" / RECIPES["dp"]["logic_file"]).read_text())["ProblemType"]
    problem_type.update(changes)
    return problem_type


# Break: omit a bespoke key, use another candidate, mishandle clamp/SAV/beta, or
# over-narrow a stock constraint (independently recorded expected changes).
@pytest.mark.parametrize("case", list(ORACLE["host_calls"]))
@pytest.mark.parametrize("keep_beta", [False, True], ids=["B", "keep_beta"])
def test_host_and_boundary_logic_equals_checked_in_oracle(case, keep_beta, tmp_path, monkeypatch):
    record = case_record("host_calls", case)
    result = specialize(tmp_path, [record], monkeypatch, arch=RECIPES[ORACLE["host_calls"][case]["recipe"]]["arch"],
                        extra=["--keep-beta"] if keep_beta else [])
    assert result["code"] == 0
    got = one_solution(result)
    assert got == oracle_logic(f"{case}_{'A' if keep_beta else 'B'}"), "bespoke logic must equal the reference"
    assert got["Solutions"][0]["BespokeNarrowing"] is True
    assert result["refusals"] == []
    assert len(result["manifest"]["calls"]) == 1
    assert result["manifest"]["calls"][0]["identity"] == identity(record)
    assert result["manifest"]["calls"][0]["keep_beta"] is keep_beta
    assert result["manifest"]["stage"] == ("keep-beta" if keep_beta else "B")


# Beta-off only for typed exact zero and raw host pointer 0. Break: truthiness or
# string-equality mistakes, dropping negative zero, or narrowing with --keep-beta.
@pytest.mark.parametrize("beta,want_B", [("-0.000000", False), ("0.000000", False), ("1.000000", True)])
@pytest.mark.parametrize("keep_beta", [False, True], ids=["B", "keep_beta"])
def test_beta_policy_preserves_logged_scalar_text(beta, want_B, keep_beta, tmp_path, monkeypatch):
    result = specialize(tmp_path, [row(beta=beta)], monkeypatch, extra=["--keep-beta"] if keep_beta else [])
    assert one_solution(result)["ProblemType"]["UseBeta"] is (True if keep_beta else want_B)
    assert result["manifest"]["calls"][0]["identity"]["beta"] == beta
    assert result["calls"][0]["beta"] == beta


# Withhold beta and SAV only for pointer modes 1/4/unknown, not the whole call.
@pytest.mark.parametrize("mode", ["1", "4", "unknown"])
@pytest.mark.parametrize("keep_beta", [False, True], ids=["B", "keep_beta"])
def test_nonhost_and_unknown_pointer_modes_generate_conservatively(mode, keep_beta, tmp_path, monkeypatch):
    result = specialize(tmp_path, [case_record("generation_only", mode)], monkeypatch,
                        extra=["--keep-beta"] if keep_beta else [])
    assert result["code"] == 0
    actual = one_solution(result)
    assert result["refusals"] == []
    assert actual["ProblemType"]["UseBeta"] is True
    assert actual["ProblemType"]["UseScaleAlphaVec"] == 1
    expected = oracle_logic("dp_beta0_B")
    expected["ProblemType"].update(UseBeta=True, UseScaleAlphaVec=1)
    assert actual == expected, "all non-pointer facts must match the host solution"
    not_applied = {e["fact"]: e["reason"].lower() for e in result["manifest"]["calls"][0]["facts_not_applied"]}
    for fact in ("UseBeta", "UseScaleAlphaVec"):
        assert "pointer" in not_applied[fact] and mode in not_applied[fact]
    assert result["calls"][0]["pointer_mode"] == ("device" if mode == "1" else "unknown")
    assert result["calls"][0]["pointer_mode_raw"] == mode


# A non-host call on a recipe without SAV is incompatible (counted refusal), while
# the SAV-capable companion stays generated and conservative.
@pytest.mark.parametrize("mode", ["1", "4"])
def test_nonhost_non_sav_recipe_is_incompatible_not_unknown(mode, tmp_path, monkeypatch):
    bad = row(pointer_mode=mode, call_count=7)
    good = row("duplicate", pointer_mode=mode, beta="-0.000000", call_count=3)
    logs, facts = configured(tmp_path, [bad, good])
    stock_file = tmp_path / "baseline/source-logic" / RECIPES["dp"]["logic_file"]
    stock = yaml.safe_load(stock_file.read_text())
    stock["ProblemType"]["UseScaleAlphaVec"] = 0
    stock_file.write_text(yaml.safe_dump(stock, sort_keys=False))
    # Rebind the observation to the genuinely derived non-SAV recipe names, so the
    # refusal under test is compatibility, not a name mismatch.
    source_solution = recipe.deriver("gfx950").derive(stock, tmp_path, "non_sav_fixture.yaml")
    bad["solution_Name"], bad["kernel_name"] = recipe.names(source_solution)
    write_rows(logs / "hipblaslt_101.log", [bad, good])
    result = generated(tmp_path, logs, facts, monkeypatch)
    refusal(result, 7, "recipe cannot serve device pointer mode (no SAV)")
    assert len(result["logic"]) == 1
    assert [c["call_count"] for c in result["manifest"]["calls"]] == [3]
    logic = one_solution(result)
    assert logic["ProblemType"]["UseBeta"] is True
    assert logic["ProblemType"]["UseScaleAlphaVec"] == 1
    assert {"UseBeta", "UseScaleAlphaVec"} <= {n["fact"] for n in result["manifest"]["calls"][0]["facts_not_applied"]}


# Break: default a missing dimension/type/route, or drop a column globally.
@pytest.mark.parametrize("field", [
    "a_type", "b_type", "c_type", "d_type", "compute_type", "M", "N", "K", "batch_count",
    "transA", "transB", "lda", "ldb", "ldc", "ldd", "stride_a", "stride_b", "stride_c", "stride_d", "route",
])
def test_missing_generation_fact_refuses_only_incomplete_call(field, tmp_path, monkeypatch):
    good, bad = row(call_count=2), row(call_count=3, N=619)
    if field != "route":
        del bad[field]
    logs, facts = configured(tmp_path, [good, bad])
    if field == "route":
        document = json.loads(facts.read_text())
        del document["calls"][1]["call_route"]
        facts.write_text(json.dumps(document))
    result = generated(tmp_path, logs, facts, monkeypatch)
    refusal(result, 3, f"missing generation fact: {field}")
    assert len(result["logic"]) == 1
    assert [c["call_count"] for c in result["manifest"]["calls"]] == [2]


# No source means no narrowing; generation needs a proven route.
@pytest.mark.parametrize("fact", ["call_route", "bias_axis", "output_amax"])
def test_route_fact_without_evidence_is_not_treated_as_proven(fact, tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row(bias_vector=True)])
    document = json.loads(facts.read_text())
    document["calls"][0]["evidence"][fact] = []
    facts.write_text(json.dumps(document))
    result = generated(tmp_path, logs, facts, monkeypatch)
    if fact == "bias_axis":
        assert result["code"] == 0
        assert one_solution(result)["ProblemType"]["BiasDataTypeList"] == [0, 7]
        notes = result["manifest"]["calls"][0]["facts_not_applied"]
        assert any("bias" in n["fact"].lower() and n["reason"] for n in notes)
    else:
        refusal(result, 2, "missing generation fact: route" if fact == "call_route" else "amax")
        assert result["logic"] == {}


# Break: coerce missing beta to 0, missing bias to none, or missing SAV to false.
@pytest.mark.parametrize("field,key,want", [
    ("beta", "UseBeta", True), ("pointer_mode", "UseBeta", True),
    ("bias_type", "BiasDataTypeList", [0, 7]),
    ("scaleAlpha_vector", "UseScaleAlphaVec", 1), ("activation_type", "ActivationType", "hipblaslt_all"),
])
def test_unknown_narrowing_fact_keeps_stock_capability(field, key, want, tmp_path, monkeypatch):
    record = row(bias_vector=True) if field == "bias_type" else row()
    del record[field]
    result = specialize(tmp_path, [record], monkeypatch)
    assert result["code"] == 0
    assert one_solution(result)["ProblemType"][key] == want
    notes = result["manifest"]["calls"][0]["facts_not_applied"]
    assert any(n["fact"] == key and n["reason"] for n in notes)
    assert field not in result["manifest"]["calls"][0]["identity"]


# Pass: an unreachable bias-capable request is refused without losing its companion.
# Break: keep unknown bias reachable, discard counts/identity/reason, or abort the whole run.
def assert_bias_refused_and_companion_generated(result, uncertain, companion):
    assert result["code"] == 1, "unknown bias presence on a bias-capable request must be refused"
    assert len(result["logic"]) == len(result["manifest"]["calls"]) == len(result["refusals"]) == 1
    refused = result["refusals"][0]
    assert refused["identity"] == identity(uncertain) and refused["call_count"] == 7
    assert "bias" in refused["reason"].lower()
    assert result["manifest"].get("refusals") == result["refusals"]
    accepted = result["manifest"]["calls"][0]
    assert accepted["identity"] == identity(companion) and accepted["call_count"] == 3
    assert accepted["status"] == "generated" and accepted["kernel_name"] and accepted["logic_file"]
    csv_refused, = [c for c in result["calls"] if c["status"] == "refused"]
    csv_accepted, = [c for c in result["calls"] if c["status"] == "generated"]
    assert csv_refused["bias_vector"] == "unknown" and csv_refused["call_count"] == "7"
    assert "bias" in csv_refused["refusal"].lower()
    assert not csv_refused["expected_kernel_name"] and not csv_refused["bespoke_logic_file"]
    assert csv_accepted["call_count"] == "3" and csv_accepted["activation_type"] == "relu"
    logic = one_solution(result)
    assert {key: logic["ProblemType"][key] for key in
            ("UseBias", "ActivationType", "UseBeta", "UseScaleAlphaVec")} == {
                "UseBias": 0, "ActivationType": "relu", "UseBeta": False, "UseScaleAlphaVec": 0}
    effective = effective_solution(logic)
    assert effective["BespokeNarrowing"] is True
    assert effective["AssertSizeEqual"] == {0: 32, 1: 618, 2: 1, 3: 120}


def test_unknown_bias_presence_refuses_bias_capable_recipe_and_generates_companion(tmp_path, monkeypatch):
    uncertain = row(call_count=7)
    del uncertain["bias_vector"]
    companion = row(activation_type="relu", call_count=3)
    result = specialize(tmp_path, [uncertain, companion], monkeypatch)
    assert_bias_refused_and_companion_generated(result, uncertain, companion)


def test_unknown_bias_sigmoid_refuses_bias_and_generates_companion(tmp_path, monkeypatch):
    uncertain = row(activation_type="sigmoid", call_count=7)
    del uncertain["bias_vector"]
    companion = row(activation_type="relu", call_count=3)
    result = specialize(tmp_path, [uncertain, companion], monkeypatch)
    assert_bias_refused_and_companion_generated(result, uncertain, companion)


# Clamp needs both arguments; other facts still apply.
@pytest.mark.parametrize("missing", ["activation_arg0", "activation_arg1", "both"])
def test_unknown_clamp_arguments_withhold_single_activation(missing, tmp_path, monkeypatch):
    record = case_record("host_calls", "sk_clamp")
    for key in (["activation_arg0", "activation_arg1"] if missing == "both" else [missing]):
        del record[key]
    result = specialize(tmp_path, [record], monkeypatch)
    assert result["code"] == 0
    logic = one_solution(result)
    assert logic["ProblemType"]["ActivationType"] == "hipblaslt_all"
    assert logic["Solutions"][0]["AssertSizeEqual"] == {0: 4096, 1: 4096, 2: 1, 3: 4096}
    assert any(n["fact"] == "ActivationType" and "argument" in n["reason"].lower()
               for n in result["manifest"]["calls"][0]["facts_not_applied"])


# Narrowing evidence is met across every configuration of one logged identity.
@pytest.mark.parametrize("proven_cfg", ["first", "second"])
def test_repeated_identity_uses_conservative_evidence_meet(proven_cfg, tmp_path, monkeypatch):
    record = row(bias_vector=True, call_count=2)
    logs, facts = configured(tmp_path, [record], cfg="first")
    _, second = input_fixture(tmp_path, [dict(record, call_count="7")], cfg="second")
    missing_file = second if proven_cfg == "first" else facts
    document = json.loads(missing_file.read_text())
    document["calls"][0]["evidence"]["bias_axis"] = []
    missing_file.write_text(json.dumps(document))
    result = generated(tmp_path, logs, facts, monkeypatch)
    assert result["code"] == 0
    assert len(result["logic"]) == 1, "one identity needs one shared conservatively supported solution"
    assert {c["run_configuration"]: c["call_count"] for c in result["manifest"]["calls"]} == {"first": 2, "second": 7}
    assert len({c["kernel_name"] for c in result["manifest"]["calls"]}) == 1
    logic = one_solution(result)
    assert logic["ProblemType"]["UseBias"] == 1
    assert logic["ProblemType"]["BiasDataTypeList"] == [0, 7]
    assert logic["ProblemType"]["UseBeta"] is False
    assert logic["ProblemType"]["UseScaleAlphaVec"] == 0
    assert effective_solution(logic)["AssertSizeEqual"] == {0: 32, 1: 618, 2: 1, 3: 120}
    for call in result["manifest"]["calls"]:
        notes = {n["fact"]: n["reason"] for n in call["facts_not_applied"]}
        assert notes.get("BiasDataTypeList"), "withheld evidence must be reported for both configurations"
        assert not any(n["fact"] == "BiasDataTypeList" for n in call["facts_applied"])


def mixed_route_fixture(root, defect):
    """One identity: two eligible configurations (2+3 calls) and one ineligible (7)."""
    record = row(bias_vector=True, call_count=2)
    logs, facts = configured(root, [record], cfg="healthy-first")
    input_fixture(root, [dict(record, call_count="3")], cfg="healthy-second")
    _, bad_file = input_fixture(root, [dict(record, call_count="7")], cfg="bad-route")
    document = json.loads(bad_file.read_text())
    # The ineligible configuration's missing bias evidence must not weaken the meet.
    document["calls"][0]["evidence"]["bias_axis"] = []
    if defect == "missing_entry":
        document["calls"] = []
    elif defect == "missing_call_route":
        del document["calls"][0]["call_route"]
    elif defect == "unproven_call_route":
        document["calls"][0]["evidence"]["call_route"] = []
    else:
        document["calls"][0]["call_route"] = "extension matmul API"
    bad_file.write_text(json.dumps(document))
    return record, logs, facts


# Break: merge call_route before eligibility, or let an ineligible configuration's
# witness into the narrowing meet.
@pytest.mark.parametrize("defect,code", [
    ("missing_entry", "missing-generation-fact"), ("missing_call_route", "missing-generation-fact"),
    ("unproven_call_route", "missing-generation-fact"), ("nonordinary_route", "route"),
])
def test_route_refusal_is_local_and_only_eligible_configurations_narrow(defect, code, tmp_path, monkeypatch):
    record, logs, facts = mixed_route_fixture(tmp_path, defect)
    result = generated(tmp_path, logs, facts, monkeypatch)
    assert result["code"] == 1
    assert {c["run_configuration"]: c["call_count"] for c in result["manifest"]["calls"]} == {
        "healthy-first": 2, "healthy-second": 3,
    }, "one configuration's route refusal must not suppress the healthy configurations"
    assert {c["status"] for c in result["manifest"]["calls"]} == {"generated"}
    assert all(c["identity"] == identity(record) for c in result["manifest"]["calls"])
    assert len({c["kernel_name"] for c in result["manifest"]["calls"]}) == 1
    assert len({c["logic_file"] for c in result["manifest"]["calls"]}) == 1
    logic = one_solution(result)
    solution = effective_solution(logic)
    assert solution["BespokeNarrowing"] is True
    assert solution["AssertSizeEqual"] == {0: 32, 1: 618, 2: 1, 3: 120}
    assert logic["ProblemType"]["BiasDataTypeList"] == [7], "ineligible configurations are outside the narrowing meet"
    assert [(c["run_configuration"], c["call_count"], c["status"]) for c in sorted(
        result["calls"], key=lambda c: c["run_configuration"],
    )] == [("bad-route", "7", "refused"), ("healthy-first", "2", "generated"), ("healthy-second", "3", "generated")]
    assert len(result["refusals"]) == 1
    entry = result["refusals"][0]
    assert entry["run_configurations"] == ["bad-route"]
    assert entry["call_count"] == 7
    assert entry["identity"] == identity(record)
    assert entry["code"] == code
    assert "route" in entry["reason"].lower()


# Out-of-domain routes and epilogues are counted refusals, never unsafe logic.
@pytest.mark.parametrize("changes,route_changes,reason", [
    ({"gradient": True}, {}, "gradient"), ({"activation_type": "dgelu"}, {}, "gradient"),
    ({"use_e": True}, {}, "aux"), ({"scaleC": 1}, {}, "scaled"), ({"scaleD": 1}, {}, "scaled"),
    ({"scaleA": 1}, {}, "scaleA"), ({"scaleB": 1}, {}, "scaleB"),
    ({"bias_vector": True, "bias_source": "a"}, {}, "bias source"),
    ({"bias_vector": True, "bias_source": "b"}, {}, "bias source"),
    ({"activation_type": "tanh"}, {}, "activation"),
    ({"activation_type": "sigmoid", "bias_vector": True}, {}, "sigmoid"),
    ({}, {"output_amax": True}, "amax"), ({}, {"output_amax": None}, "amax"),
    ({"bias_vector": True}, {"bias_axis": "N"}, "bias"),
    ({}, {"call_route": "grouped"}, "grouped"),
    ({}, {"call_route": "device_args"}, "device"),
    ({}, {"call_route": "hipblaslt-ext"}, "route"),
    ({}, {"call_route": "RocRoller"}, "RocRoller"),
], ids=["gradient", "gradient_activation", "aux", "scaleC", "scaleD", "scaleA", "scaleB", "bias_source_a",
        "bias_source_b", "tanh", "sigmoid_bias", "amax", "unknown_amax", "bias_N", "grouped", "device_args",
        "ext", "RocRoller"])
def test_out_of_domain_calls_are_refused(changes, route_changes, reason, tmp_path, monkeypatch):
    result = specialize(tmp_path, [row(call_count=7, **changes)], monkeypatch, route_changes=route_changes)
    refusal(result, 7, reason)
    assert result["logic"] == {}


# Break: compare DataTypeA/B only, or default an explicit MAC type to the call's type.
@pytest.mark.parametrize("pt_change", [{"DataType": 15}, {"MacDataTypeA": 15}, {"MacDataTypeB": 15}],
                         ids=["DataType_only", "MAC_A", "MAC_B"])
def test_recipe_type_mismatch_refuses_even_with_matching_io_types(pt_change, tmp_path):
    error = checked_stock_error(tmp_path, stock_problem_type(**pt_change))
    assert error is not None, "the type check must refuse independently mismatched recipe DataType/MAC"
    assert error.code == "type-mismatch"
    assert "MacDataTypeA/B" in error.reason
    assert not re.search(r"\b[A-Z][A-Z0-9]{0,2}(?:-(?:[A-Z]\d+|\d{2,}))+\b", error.reason, re.I), \
        "refusal text must not carry a tracker-style tag"


# Break: resurrect beta capability on a beta-off stock recipe with --keep-beta.
def test_keep_beta_on_beta_off_recipe_has_oracle_refusal_code(tmp_path):
    error = checked_stock_error(tmp_path, stock_problem_type(UseBeta=False), keep_beta=True)
    assert error is not None, "--keep-beta must refuse a beta-off stock recipe"
    assert error.code == "keep-beta"
    assert "interim" not in error.reason.lower()


# Break: stamp bespoke logic for another input type or silently change the recipe.
def test_trace_type_mismatch_is_refused_in_cli_output(tmp_path, monkeypatch):
    result = specialize(tmp_path, [row(a_type="f16_r", b_type="f16_r")], monkeypatch)
    refusal(result, 2, "type-mismatch")
    assert result["logic"] == {}


# Custom-kernel recipes are reported in CSV/manifest but never generated.
def test_custom_kernel_is_pending_and_retained_in_corpus_outputs(tmp_path, monkeypatch):
    record = row("custom", transA="T", M=16, N=1, K=32, lda=32, ldb=32, ldc=16, ldd=16, beta="1.000000")
    result = specialize(tmp_path, [record], monkeypatch)
    refusal(result, 2, "custom-kernel recipe unsupported")
    assert result["logic"] == {}
    assert len(result["calls"]) == 1
    assert result["calls"][0]["stock_solution_index"] == str(RECIPES["custom"]["baseline_index"])
    assert len(result["manifest"]["calls"]) == 1
    assert result["manifest"]["calls"][0]["identity"] == identity(record)
    assert result["manifest"]["calls"][0]["status"] == "custom-kernel-recipe-unsupported"
    assert result["manifest"]["calls"][0]["kernel_name"] is None, "pending calls have no expected bespoke kernel"


# A TensileLite fatal (SystemExit) in derivation is a counted per-call refusal and the
# healthy companion is still written. Only the target-tree parse is interposed.
def test_tensilelite_fatal_is_a_counted_per_call_derivation_refusal(tmp_path, monkeypatch):
    import tensilelite.LibraryIO as library_io
    real_parse = library_io.parseLibraryLogicFile
    def parse(path, *args, **kwargs):
        solution = effective_solution(yaml.safe_load(Path(path).read_text()))
        if solution.get("BespokeNarrowing") and solution.get("AssertSizeEqual", {}).get(1) == 619:
            raise SystemExit(255)
        return real_parse(path, *args, **kwargs)
    monkeypatch.setattr(library_io, "parseLibraryLogicFile", parse)
    result = specialize(tmp_path, [row(call_count=2), row(N=619, call_count=7)], monkeypatch)
    assert result["code"] == 1, "a target-tree fatal must become a controlled per-call refusal"
    refusal(result, 7, "derivation")
    assert len(result["logic"]) == 1
    assert [c["call_count"] for c in result["manifest"]["calls"]] == [2]
    assert sorted(int(c["call_count"]) for c in result["calls"]) == [2, 7]
    assert {c["status"]: c["call_count"] for c in result["calls"]} == {"generated": "2", "refused": "7"}


# A derivation that does not honour a mandatory key is refused, never reverted.
@pytest.mark.parametrize("key", ["BespokeNarrowing", "AssertSizeEqual"])
def test_unhonoured_mandatory_keys_refuse_only_the_affected_call(key, tmp_path, monkeypatch):
    real_derive = recipe.Deriver.derive
    def derive(self, logic, scratch, name):
        solution = real_derive(self, logic, scratch, name)
        state = effective_solution(logic)
        if state.get("BespokeNarrowing") and state.get("AssertSizeEqual", {}).get(1) == 619:
            # Real derived solution with one mandatory request not honoured.
            solution._state[key] = False if key == "BespokeNarrowing" else {}
        return solution
    monkeypatch.setattr(recipe.Deriver, "derive", derive)
    result = specialize(tmp_path, [row(call_count=2), row(N=619, call_count=7)], monkeypatch)
    assert result["code"] == 1, "unhonoured mandatory keys require a per-call derivation refusal"
    entry = refusal(result, 7, "derivation")
    assert key in entry["reason"]
    assert len(result["logic"]) == 1
    assert [c["call_count"] for c in result["manifest"]["calls"]] == [2]
    solution = effective_solution(one_solution(result))
    assert solution["BespokeNarrowing"] is True
    assert solution["AssertSizeEqual"] == {0: 32, 1: 618, 2: 1, 3: 120}
