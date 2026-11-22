# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Safety refusals and derivation rollback through the real CPU producer/validator.

Pass: incompatible inputs retain their reason and weight without deployable logic.
Break: bypass a compatibility check, trust changed derivation facts, or lose the
healthy companion. Derivation interposition changes inputs to real TensileLite;
it never substitutes a fake solution, validator, name, or written result.
"""
import copy

import pytest
import yaml

from geko.specialize import facts, recipe
from geko.specialize.calls import load_calls
from tests.specialize_test_support import (
    DATA, RECIPES, command, configured, effective_solution, generated, identity,
    invoke, recipe_key, refusal, route_document, row, specialize, write_rows,
)


def stock_logic():
    return yaml.safe_load((DATA / "stock" / RECIPES["dp"]["logic_file"]).read_text())


# Pass: a compatible call validates first, then the changed recipe is refused
# with the failing fact. Break: omit the named fixed-stock compatibility check.
@pytest.mark.parametrize("call_changes,stock_changes,batched,code,reason", [
    ({}, {"TransposeA": True}, True, "stock-mismatch", "TransposeA"),
    ({}, {"TransposeB": True}, True, "stock-mismatch", "TransposeB"),
    ({"batch_count": 2}, {}, False, "stock-mismatch", "batch"),
    ({"beta": "1.000000"}, {"UseBeta": False}, True, "stock-mismatch", "beta"),
    ({"activation_type": "relu"}, {"Activation": False, "ActivationType": "none"},
     True, "stock-mismatch", "activation"),
    ({"bias_vector": True}, {"UseBias": 0}, True, "stock-mismatch", "UseBias"),
    ({"bias_vector": True}, {"BiasDataTypeList": [0]}, True, "stock-mismatch", "bias_type"),
    ({"scaleAlpha_vector": True}, {"UseScaleAlphaVec": 0}, True, "stock-mismatch", "scale-alpha"),
    ({}, {"Sparse": 1}, True, "stock-out-of-domain", "Sparse"),
    ({}, {"Gradient": True}, True, "stock-out-of-domain", "Gradient"),
    ({}, {"UseE": True}, True, "stock-out-of-domain", "UseE"),
    ({}, {"BiasSrc": "A"}, True, "stock-out-of-domain", "BiasSrc"),
], ids=["transpose-a", "transpose-b", "batch", "beta-off", "activation", "bias",
        "bias-type", "sav", "sparse", "gradient", "aux", "bias-source"])
def test_stock_validator_refuses_each_incompatible_recipe(
        call_changes, stock_changes, batched, code, reason, tmp_path):
    record = row(**call_changes)
    call_identity = next(iter(load_calls(write_rows(tmp_path / "hipblaslt_101.log", [record]))))
    call = facts.call_facts(call_identity, route_document([record])["calls"][0])
    compatible = stock_logic()["ProblemType"]
    facts.check_stock(call, compatible)
    incompatible = {**compatible, **stock_changes}
    with pytest.raises(recipe.Refused) as caught:
        facts.check_stock(call, incompatible, batched=batched)
    assert caught.value.code == code
    assert reason.lower() in caught.value.reason.lower()


# Pass: raw invalid identity facts fail the real fact producer with their reason.
# Break: coerce an unknown transpose/type, or permit unlike C/D or unsupported bias.
@pytest.mark.parametrize("changes,removed,code,reason", [
    ({"transA": "X"}, None, "transpose", "transA"),
    ({"a_type": "i8_r"}, None, "dtype", "a_type"),
    ({"c_type": "f16_r"}, None, "dtype", "c_type"),
    ({"bias_vector": True, "bias_type": "i8_r"}, None, "dtype", "bias_type"),
    ({}, "M", "missing-generation-fact", "M"),
], ids=["transpose", "input-type", "different-c-d", "bias-type", "missing-size"])
def test_identity_fact_validator_refuses_invalid_generation_facts(
        changes, removed, code, reason, tmp_path):
    record = row(**changes)
    if removed:
        del record[removed]
    call_identity = next(iter(load_calls(write_rows(tmp_path / "hipblaslt_101.log", [record]))))
    with pytest.raises(recipe.Refused) as caught:
        facts.identity_facts(call_identity, bias_axis="M")
    assert caught.value.code == code
    assert reason in caught.value.reason


# Pass: activation-none exclusion preserves general activation and explains why;
# an unexcluded relu companion still narrows. Break: ignore none-call exclusions.
def test_activation_none_exclusion_is_scoped_and_records_why_it_stays_general(tmp_path, monkeypatch):
    none = row(call_count=5, activation_type="none")
    relu = row(call_count=3, activation_type="relu")
    exclusion = tmp_path / "exclude.yaml"
    exclusion.write_text(yaml.safe_dump({"exclusions": [{
        "recipe": recipe_key(), "identity": identity(none), "facts": ["ActivationType"],
    }]}))
    result = specialize(tmp_path, [none, relu], monkeypatch,
                        extra=["--exclude-controls", str(exclusion)])
    assert result["code"] == 0 and result["refusals"] == []
    assert len(result["logic"]) == 2
    entries = {entry["identity"]["activation_type"]: entry for entry in result["manifest"]["calls"]}
    general = result["logic"][entries["none"]["logic_file"].split("/")[-1]]["ProblemType"]
    narrowed = result["logic"][entries["relu"]["logic_file"].split("/")[-1]]["ProblemType"]
    assert general["ActivationType"] == "hipblaslt_all" and general["Activation"] is True
    assert narrowed["ActivationType"] == "relu"
    assert entries["none"]["call_count"] == 5 and entries["relu"]["call_count"] == 3
    notes = [note for note in entries["none"]["facts_not_applied"] if note["fact"] == "ActivationType"]
    assert len(notes) == 1 and "exclude" in notes[0]["reason"].lower()


# Pass: actual derived stock assertions reject the call before bespoke generation.
# Break: omit multiple, size, stride or unevaluated-AI checks in the pre-filter.
@pytest.mark.parametrize("constraint", [
    {"AssertSummationElementMultiple": 2},
    {"AssertSizeEqual": {0: 64}},
    {"AssertSizeGreaterThan": {0: 32}},
    {"AssertSizeLessThan": {1: 619}},
    {"AssertStrideAEqual": {1: 64}},
    {"AssertAIGreaterThanEqual": 1},
], ids=["multiple", "equal", "greater", "less", "stride", "ai-bound"])
def test_writer_refuses_call_rejected_by_real_derived_stock(constraint, tmp_path, monkeypatch):
    accepted = row("duplicate", call_count=3)
    rejected = row(N=619, K=119, ldb=119, call_count=7)
    logs, routes = configured(tmp_path, [accepted, rejected])
    source = tmp_path / "baseline/source-logic" / RECIPES["dp"]["logic_file"]
    logic = yaml.safe_load(source.read_text())
    logic["Solutions"][0].update(constraint)
    source.write_text(yaml.safe_dump(logic, sort_keys=False))
    restricted = recipe.deriver("gfx950").derive(logic, tmp_path, "restricted-stock.yaml")
    rejected["solution_Name"], rejected["kernel_name"] = recipe.names(restricted)
    write_rows(logs / "hipblaslt_101.log", [accepted, rejected])
    result = generated(tmp_path, logs, routes, monkeypatch)
    entry = refusal(result, 7, next(iter(constraint)))
    assert entry["code"] == "stock-rejects-call" and entry["identity"] == identity(rejected)
    assert len(result["logic"]) == 1
    assert [call["call_count"] for call in result["manifest"]["calls"]] == [3]
    refused = next(call for call in result["calls"] if call["status"] == "refused")
    assert not refused["bespoke_logic_file"] and not refused["expected_kernel_name"]


# Pass: when real derivation keeps a general optional fact, the writer rolls the
# request back with a reason; retained bias for a no-bias call is then unreachable.
# Break: write the discarded request, lose its reason, or skip self-reachability.
@pytest.mark.parametrize("field,changes,derived_value,section", [
    ("ActivationType", {"activation_type": "relu"}, "hipblaslt_all", "ProblemType"),
    ("UseBeta", {}, True, "ProblemType"),
    ("BiasDataTypeList", {"bias_vector": True}, [0, 7], "ProblemType"),
    ("UseScaleAlphaVec", {}, 1, "ProblemType"),
    ("UseBias", {}, 1, "ProblemType"),
    ("AssertFree0ElementMultiple", {}, 1, "Solution"),
])
def test_writer_rolls_back_unhonoured_optional_derivation_facts(
        field, changes, derived_value, section, tmp_path, monkeypatch):
    accepted = row(call_count=3)
    rejected = row(N=619, call_count=7, **changes)
    real_derive = recipe.Deriver.derive

    def derive(self, logic, scratch, name):
        effective = effective_solution(logic)
        if name == "candidate.yaml" and effective.get("AssertSizeEqual", {}).get(1) == 619:
            logic = copy.deepcopy(logic)
            target = logic["ProblemType"] if section == "ProblemType" else logic["Solutions"][0]
            target[field] = derived_value
        return real_derive(self, logic, scratch, name)

    monkeypatch.setattr(recipe.Deriver, "derive", derive)
    result = specialize(tmp_path, [accepted, rejected], monkeypatch)
    assert len(result["manifest"]["calls"]) >= 1
    assert next(call for call in result["manifest"]["calls"]
                if call["identity"] == identity(accepted))["call_count"] == 3
    if field == "UseBias":
        entry = refusal(result, 7, "bias")
        assert entry["code"] == "unreachable" and len(result["logic"]) == 1
        return
    assert result["code"] == 0 and result["refusals"] == []
    entry = next(call for call in result["manifest"]["calls"] if call["identity"] == identity(rejected))
    notes = [note for note in entry["facts_not_applied"] if note["fact"] == field]
    assert len(notes) == 1 and field in notes[0]["reason"] and "derive" in notes[0]["reason"]
    logic = result["logic"][entry["logic_file"].split("/")[-1]]
    written = logic["ProblemType"] if section == "ProblemType" else effective_solution(logic)
    assert written[field] == derived_value and entry["call_count"] == 7


# Pass: re-derivation cannot silently change a fact written into deployable logic.
# Break: omit the writer's final derived-versus-written assertion check.
def test_writer_refuses_changed_final_bespoke_derivation(tmp_path, monkeypatch):
    accepted, rejected = row(call_count=3), row(N=619, call_count=7)
    real_derive = recipe.Deriver.derive

    def derive(self, logic, scratch, name):
        if name == "bespoke.yaml" and effective_solution(logic).get("AssertSizeEqual", {}).get(1) == 619:
            logic = copy.deepcopy(logic)
            logic["Solutions"][0]["AssertFree0ElementMultiple"] = 1
        return real_derive(self, logic, scratch, name)

    monkeypatch.setattr(recipe.Deriver, "derive", derive)
    result = specialize(tmp_path, [accepted, rejected], monkeypatch)
    entry = refusal(result, 7, "AssertFree0ElementMultiple")
    assert entry["code"] == "bespoke-derivation" and entry["identity"] == identity(rejected)
    assert len(result["logic"]) == 1
    assert [call["call_count"] for call in result["manifest"]["calls"]] == [3]


# Pass: coherent empty or invalid real target derivations are controlled refusals,
# not usable recipes. Break: accept zero/invalid solutions or leak iterator errors.
@pytest.mark.parametrize("defect", ["empty", "invalid-geometry"])
def test_recipe_deriver_refuses_no_valid_derived_solution(defect, tmp_path):
    logic = stock_logic()
    derive = recipe.deriver("gfx950")
    assert derive.derive(logic, tmp_path, "valid-stock.yaml")["Valid"] is True
    if defect == "empty":
        logic["Solutions"] = []
        logic["ExactLogic"] = []
    else:
        logic["Solutions"][0]["DepthU"] = 2
    with pytest.raises(recipe.Refused) as caught:
        derive.derive(logic, tmp_path, "invalid-stock.yaml")
    assert caught.value.code == "derivation"
    assert "no valid solution" in caught.value.reason


# Pass: MatchTable resolution refuses another architecture while keeping its
# healthy companion. Break: infer architecture from an index/name or skip it.
def test_recipe_architecture_mismatch_refuses_only_its_call(tmp_path, monkeypatch):
    rejected = row("g942", call_count=7)
    result = specialize(tmp_path, [row(call_count=3), rejected], monkeypatch)
    entry = refusal(result, 7, "gfx942")
    assert entry["code"] == "arch" and "gfx950" in entry["reason"]
    assert entry["identity"] == identity(rejected)
    assert len(result["logic"]) == 1
    assert [call["call_count"] for call in result["manifest"]["calls"]] == [3]


# Pass: argument failures report the missing/unsupported specialization input,
# exit 2 before writing output and never enter GPU enumeration. Break: default
# missing inputs or accept an unsupported architecture at this CLI boundary.
@pytest.mark.parametrize("defect,reason", [
    ("--workload-log", "--workload-log"), ("--hipblaslt", "--hipblaslt"),
    ("--arch", "--arch"), ("--workdir", "--workdir"), ("architecture", "gfx950"),
    ("devices", "--devices"),
])
def test_specialize_cli_argument_errors_explain_the_refusal(defect, reason, tmp_path, monkeypatch, capsys):
    logs, routes = configured(tmp_path, [row()])
    output = tmp_path / "out"
    arguments = command(tmp_path, logs, routes, output)
    if defect == "architecture":
        arguments[arguments.index("--arch") + 1] = "gfx1250"
    elif defect == "devices":
        arguments = ["--tune", "--workload-log", str(logs / "hipblaslt_101.log")]
    else:
        index = arguments.index(defect)
        del arguments[index:index + 2]
        if defect == "--workload-log":
            arguments += ["--list", str(logs / "hipblaslt_101.log")]
    assert invoke(arguments, monkeypatch) == 2
    assert reason in capsys.readouterr().err
    assert not output.exists()
