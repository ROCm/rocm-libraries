# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A generated bespoke solution must be selectable for every identity it owns."""
import pytest
import yaml

from geko.specialize import facts
from tests.specialize_test_support import identity, one_solution, recipe_key, row, specialize


def assert_refused(result, record):
    assert result["code"] == 1, "unreachable bespoke call must be refused, not generated"
    assert result["logic"] == {}, "a refused call must not leave deployable logic"
    assert len(result["calls"]) == len(result["refusals"]) == 1
    assert result["manifest"]["calls"] == []
    csv_call = result["calls"][0]
    refusal = result["refusals"][0]
    assert csv_call["status"] == "refused"
    assert csv_call["call_count"] == str(record["call_count"])
    assert refusal["call_count"] == int(record["call_count"])
    assert refusal["identity"] == identity(record)
    assert "bias" in csv_call["refusal"].lower() and "bias" in refusal["reason"].lower()
    assert result["manifest"].get("refusals") == result["refusals"]
    assert not csv_call["expected_kernel_name"] and not csv_call["bespoke_logic_file"]


# Pass: refuse retained bias capability unless bias presence was logged true.
# Break: treat unknown bias as safe, or let an exclusion bypass self-reachability.
@pytest.mark.parametrize("trigger", ["unknown-bias", "excluded-bias"])
def test_retained_use_bias_refuses_without_proven_bias_presence(trigger, tmp_path, monkeypatch):
    record = row(call_count=7, bias_vector=False)
    extra = []
    if trigger == "unknown-bias":
        del record["bias_vector"]
    else:
        exclude = tmp_path / "exclude.yaml"
        exclude.write_text(yaml.safe_dump({"exclusions": [{
            "recipe": recipe_key(), "identity": identity(record), "facts": ["UseBias"],
        }]}))
        extra = ["--exclude-controls", str(exclude)]
    result = specialize(tmp_path, [record], monkeypatch, extra=extra)
    assert_refused(result, record)


# Pass: the writer checks selection facts independently of the narrowing producer.
# Break: trust a successfully derived single activation without matching its call.
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_writer_refuses_single_activation_rejecting_one_own_identity(reverse, tmp_path, monkeypatch):
    accepted = row(activation_type="relu", call_count=3)
    rejected = row(activation_type="gelu", call_count=11)
    real_narrowing = facts.narrowing

    def incompatible_activation(call, stock_pt, keep_beta, excluded=frozenset()):
        request, notes = real_narrowing(call, stock_pt, keep_beta, excluded)
        request.update(Activation=True, ActivationType="relu")
        return request, notes

    monkeypatch.setattr(facts, "narrowing", incompatible_activation)
    records = [accepted, rejected]
    if reverse:
        records.reverse()
    result = specialize(tmp_path, records, monkeypatch)
    calls = {c["identity"]["activation_type"]: c for c in result["manifest"]["calls"]}
    assert calls["relu"]["status"] == "generated", "matching identity is the positive control"
    assert result["code"] == 1, "writer emitted a single-relu solution for its own gelu identity"
    assert set(calls) == {"relu"}
    assert len(result["logic"]) == len(result["refusals"]) == 1
    assert one_solution(result)["ProblemType"]["ActivationType"] == "relu"
    refusal = result["refusals"][0]
    assert refusal["identity"] == identity(rejected) and refusal["call_count"] == 11
    assert "activation" in refusal["reason"].lower()
    assert result["manifest"].get("refusals") == result["refusals"]
    csv_calls = {c["activation_type"]: c for c in result["calls"]}
    assert csv_calls["gelu"]["status"] == "refused" and csv_calls["gelu"]["refusal"]
    assert not csv_calls["gelu"]["bespoke_logic_file"]


# Pass: ordinary no-bias calls narrow UseBias to zero; proven bias remains reachable,
# even with bias narrowing excluded. Break: retain bias for no-bias calls, or refuse all bias calls.
@pytest.mark.parametrize("bias_on,excluded", [(False, False), (True, False), (True, True)])
def test_reachable_bias_calls_keep_their_kernel(bias_on, excluded, tmp_path, monkeypatch):
    record = row(bias_vector=bias_on, call_count=5)
    extra = []
    if excluded:
        path = tmp_path / "exclude.yaml"
        path.write_text(yaml.safe_dump({"exclusions": [{
            "recipe": recipe_key(), "identity": identity(record), "facts": ["UseBias"],
        }]}))
        extra = ["--exclude-controls", str(path)]
    result = specialize(tmp_path, [record], monkeypatch, extra=extra)
    assert result["code"] == 0 and result["refusals"] == []
    assert len(result["logic"]) == len(result["calls"]) == len(result["manifest"]["calls"]) == 1
    assert one_solution(result)["ProblemType"]["UseBias"] == (1 if bias_on else 0)
    call = result["manifest"]["calls"][0]
    assert call["identity"] == identity(record) and call["call_count"] == 5
    assert call["kernel_name"] and call["logic_file"]


# Pass: independently evaluate every narrowed host selection fact for each owned identity.
# Break: trust a successfully derived BetaZero, bias-off, SAV-off or bias-type request.
@pytest.mark.parametrize("fact,control_changes,rejected_changes,narrowed_request,cause", [
    ("beta", {"beta": "0.000000"}, {"beta": "1.000000"}, {"UseBeta": False}, "beta"),
    ("bias-off", {"bias_vector": False}, {"bias_vector": True},
     {"UseBias": 0, "BiasDataTypeList": []}, "bias"),
    ("sav-off", {"scaleAlpha_vector": False, "pointer_mode": 0},
     {"scaleAlpha_vector": True, "pointer_mode": 4}, {"UseScaleAlphaVec": 0}, "scale"),
    ("bias-type", {"bias_vector": True, "bias_type": "bf16_r"},
     {"bias_vector": True, "bias_type": "f32_r"}, {"UseBias": 1, "BiasDataTypeList": [7]}, "bias"),
], ids=["beta-zero", "bias-off", "sav-off", "bias-type-whitelist"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_writer_refuses_narrowed_fact_rejecting_one_own_identity(
        fact, control_changes, rejected_changes, narrowed_request, cause, reverse, tmp_path, monkeypatch):
    accepted = row(call_count=3, **control_changes)
    rejected = row(call_count=11, **rejected_changes)
    real_narrowing = facts.narrowing

    def incompatible_request(call, stock_pt, keep_beta, excluded=frozenset()):
        changes, notes = real_narrowing(call, stock_pt, keep_beta, excluded)
        changes.update(narrowed_request)
        return changes, notes

    monkeypatch.setattr(facts, "narrowing", incompatible_request)
    records = [accepted, rejected]
    if reverse:
        records.reverse()
    result = specialize(tmp_path, records, monkeypatch, extra=["--keep-beta"])
    generated_call = next(c for c in result["manifest"]["calls"] if c["identity"] == identity(accepted))
    assert generated_call["status"] == "generated" and generated_call["call_count"] == 3
    assert generated_call["kernel_name"] and generated_call["logic_file"], "matching identity is the positive control"
    assert result["code"] == 1, f"writer emitted a {fact} solution rejecting its own identity"
    assert len(result["manifest"]["calls"]) == len(result["logic"]) == len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["identity"] == identity(rejected) and refusal["call_count"] == 11
    assert refusal["code"] == "unreachable"
    assert cause in refusal["reason"].lower()
    assert result["manifest"].get("refusals") == result["refusals"]
    csv_refused, = [call for call in result["calls"] if call["status"] == "refused"]
    assert csv_refused["call_count"] == "11" and cause in csv_refused["refusal"].lower()
    assert not csv_refused["expected_kernel_name"] and not csv_refused["bespoke_logic_file"]
    assert {key: one_solution(result)["ProblemType"][key] for key in narrowed_request} == narrowed_request
