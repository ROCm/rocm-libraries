# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deterministic corpus/logic/manifest output, exclusions, and workdir ownership.

writer.py canonicalizes bespoke solution contents, writes consumer-ready corpus rows
and honest stage/arch metadata, applies per-identity exclusions, and only ever
removes artifacts it owns. Break: share by shape only, let insertion order rename a
solution, discard configuration weights, or delete user files in the workdir.
"""
import json
from pathlib import Path

import pytest
import yaml

from tests.specialize_test_support import (
    MANIFEST_COLUMNS, RECIPES, assert_directories_preserved, command, configured, effective_solution,
    generated, held_directories, identity, input_fixture, invoke, one_solution, outputs, readme_specialize_examples,
    recipe_key, row, specialize, write_rows,
)


def mapping(result):
    return {json.dumps(c["identity"], sort_keys=True): c["kernel_name"] for c in result["manifest"]["calls"]}


# Break: argument/beta spellings create redundant identical solutions, IDs depend on
# first occurrence, or different epilogues share one solution.
def test_identical_solutions_share_one_file_in_both_insertion_orders(tmp_path, monkeypatch):
    first = row(beta="0.000000", activation_arg0="0", call_count=2)
    second = row(beta="-0.000000", activation_arg0="2.25", call_count=3)
    changed = row(activation_type="relu", call_count=5)
    a = specialize(tmp_path / "forward", [first, second, changed], monkeypatch)
    b = specialize(tmp_path / "reverse", [changed, second, first], monkeypatch)
    assert len(a["manifest"]["calls"]) == len(b["manifest"]["calls"]) == 3
    assert len(a["logic"]) == len(b["logic"]) == 2
    assert a["logic"] == b["logic"]
    assert mapping(a) == mapping(b)
    amap = mapping(a)
    assert amap[json.dumps(identity(first), sort_keys=True)] == amap[json.dumps(identity(second), sort_keys=True)]
    assert amap[json.dumps(identity(first), sort_keys=True)] != amap[json.dumps(identity(changed), sort_keys=True)]
    assert sorted(c["call_count"] for c in a["manifest"]["calls"]) == [2, 3, 5]


# Break: add the source file stem to an otherwise identical logic-content key.
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_equal_bespoke_contents_share_across_source_stems(reverse, tmp_path, monkeypatch):
    records = [row(call_count=2), row("duplicate", beta="-0.000000", call_count=7)]
    if reverse:
        records.reverse()
    result = specialize(tmp_path, records, monkeypatch)
    assert result["code"] == 0
    assert len(result["logic"]) == 1, "equal bespoke contents must not be duplicated by source stem"
    assert len(result["manifest"]["calls"]) == 2
    assert len({c["kernel_name"] for c in result["manifest"]["calls"]}) == 1
    assert len({c["logic_file"] for c in result["manifest"]["calls"]}) == 1
    sources = {RECIPES["dp"]["logic_file"], RECIPES["duplicate"]["logic_file"]}
    assert {c["recipe"]["source_file"] for c in result["manifest"]["calls"]} == sources
    assert sorted(int(c["call_count"]) for c in result["calls"]) == [2, 7]
    assert {c["stock_logic_file"] for c in result["calls"]} == sources


# Break: merge corpus counts across configurations or bake the configuration into code IDs.
def test_same_identity_across_configurations_keeps_rows_and_shares_solution(tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row(call_count=2)], cfg="aiter-on")
    input_fixture(tmp_path, [row(call_count=5)], cfg="aiter-off")
    result = generated(tmp_path, logs, facts, monkeypatch)
    assert len(result["calls"]) == 2
    assert {r["run_configuration"]: int(r["call_count"]) for r in result["calls"]} == {"aiter-on": 2, "aiter-off": 5}
    assert len(result["logic"]) == 1
    assert len({c["kernel_name"] for c in result["manifest"]["calls"]}) == 1


# Break: coerce 6-decimal scalars, lose arguments/strides, or write wrong type names.
def test_calls_csv_has_raw_identity_and_replay_columns(tmp_path, monkeypatch):
    record = row(activation_type="clamp", activation_arg0="-1.5", activation_arg1="2.25", beta="-0.000000")
    result = specialize(tmp_path, [record], monkeypatch)
    assert len(result["calls"]) == 1
    got = result["calls"][0]
    assert set(MANIFEST_COLUMNS) <= got.keys()
    collisions = {"pointer_mode", "a_type", "b_type", "c_type", "d_type", "compute_type", "bias_type"}
    reconstructed = {field: got[field + "_raw"] if field in collisions else got[field]
                     for field in identity(record)}
    assert reconstructed == identity(record), "CSV must round-trip every logged identity fact"
    assert reconstructed == result["manifest"]["calls"][0]["identity"]
    assert got["pointer_mode_raw"] == "0"
    assert got["pointer_mode"] == "host"
    assert (got["a_type"], got["b_type"], got["c_type"], got["d_type"], got["compute_type"]) == ("bf16", "bf16", "bf16", "bf16", "f32")
    assert (got["batch"], got["bias"], got["activation"], got["sav"], got["api_path"]) == ("1", "none", "clamp", "0", "gemm")
    assert got["act_args"] == "-1.5;2.25"
    assert got["model"] == "fixture-model"
    assert got["run_configuration"] == "fixture-on"
    assert got["call_count"] == "2"
    assert got["problem_id"]
    assert result["manifest"]["calls"][0]["identity"] == identity(record)
    assert result["manifest"]["calls"][0]["facts_applied"]


# Break: ingest startup/analysis-2/greedy/performance/acceptance or non-process .log files.
def test_mixed_directory_uses_only_designated_configuration_trace_files(tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row(call_count=2)])
    write_rows(logs / "hipblaslt_202.log", [row(call_count=3)])
    for name in ("startup", "analysis-2", "analysis-api", "greedy", "performance", "acceptance"):
        write_rows(logs / name / "hipblaslt_999.log", [row(call_count=101, N=619)])
    write_rows(logs / "not-a-process.log", [row(call_count=101, N=619)])
    (logs / "api.log").write_text("rocblaslt_create\nrocblaslt_matmul\n")
    result = generated(tmp_path, logs, facts, monkeypatch)
    assert len(result["calls"]) == 1
    assert result["calls"][0]["call_count"] == "5"
    assert result["manifest"]["calls"][0]["call_count"] == 5
    assert len(result["logic"]) == 1


def exclusions(path, *entries):
    path.write_text(yaml.safe_dump({"exclusions": [
        {"recipe": recipe_key(), "identity": ident, "facts": facts} for ident, facts in entries]}))
    return ["--exclude-controls", str(path)]


# Break: exclude by shape globally or drop a required exact-size assertion.
@pytest.mark.parametrize("fact", ["UseBeta", "ActivationType", "UseBias", "UseScaleAlphaVec",
                                  "AssertFree0ElementMultiple", "AssertFree1ElementMultiple",
                                  "AssertSummationElementMultiple"])
def test_exclusions_are_scoped_to_recipe_and_call_identity(fact, tmp_path, monkeypatch):
    shape = dict(activation_type="relu", M=64, N=64, K=128, lda=64, ldb=128, ldc=64, ldd=64)
    first = row(**shape, call_count=2)
    second = row(**shape, beta="-0.000000", call_count=3)
    extra = exclusions(tmp_path / "exclude.yaml", (identity(first), [fact]))
    result = specialize(tmp_path, [first, second], monkeypatch, extra=extra)
    if fact == "UseBias":
        assert result["code"] == 1, "retained bias must refuse only the excluded no-bias identity"
        assert len(result["logic"]) == len(result["manifest"]["calls"]) == len(result["refusals"]) == 1
        refused = result["refusals"][0]
        assert refused["identity"] == identity(first) and refused["call_count"] == 2
        assert "bias" in refused["reason"].lower()
        assert result["manifest"].get("refusals") == result["refusals"]
        accepted = result["manifest"]["calls"][0]
        assert accepted["identity"] == identity(second) and accepted["call_count"] == 3
        assert accepted["status"] == "generated" and accepted["kernel_name"] and accepted["logic_file"]
        assert not any(n["fact"] == fact and "exclu" in n["reason"].lower()
                       for n in accepted["facts_not_applied"])
        csv_calls = {c["beta"]: c for c in result["calls"]}
        assert csv_calls["0.000000"]["status"] == "refused"
        assert csv_calls["0.000000"]["call_count"] == "2"
        assert "bias" in csv_calls["0.000000"]["refusal"].lower()
        assert not csv_calls["0.000000"]["expected_kernel_name"]
        assert not csv_calls["0.000000"]["bespoke_logic_file"]
        assert csv_calls["-0.000000"]["status"] == "generated"
        assert csv_calls["-0.000000"]["call_count"] == "3"
        logic = one_solution(result)
        assert {key: logic["ProblemType"][key] for key in
                ("UseBias", "ActivationType", "UseBeta", "UseScaleAlphaVec")} == {
                    "UseBias": 0, "ActivationType": "relu", "UseBeta": False, "UseScaleAlphaVec": 0}
        effective = effective_solution(logic)
        assert effective["BespokeNarrowing"] is True
        assert effective["AssertSizeEqual"] == {0: 64, 1: 64, 2: 1, 3: 128}
        return
    assert result["code"] == 0
    assert len(result["manifest"]["calls"]) == 2
    assert len(result["logic"]) == 2
    # Independently known stock settings: beta general, all activations, bias/SAV on M.
    stock_values = {"UseBeta": True, "ActivationType": "hipblaslt_all", "UseBias": 1, "UseScaleAlphaVec": 1,
                    "AssertFree0ElementMultiple": 1, "AssertFree1ElementMultiple": 1, "AssertSummationElementMultiple": 1}
    values = []
    for logic in result["logic"].values():
        effective = effective_solution(logic)
        assert effective["AssertSizeEqual"] == {0: 64, 1: 64, 2: 1, 3: 128}
        assert effective["BespokeNarrowing"] is True
        values.append(logic["ProblemType"][fact] if fact in logic["ProblemType"] else effective[fact])
    assert stock_values[fact] in values
    assert len({str(v) for v in values}) == 2
    scoped = {c["identity"]["beta"]: c for c in result["manifest"]["calls"]}
    assert any(n["fact"] == fact and "exclu" in n["reason"].lower() for n in scoped["0.000000"]["facts_not_applied"])
    assert not any(n["fact"] == fact and "exclu" in n["reason"].lower() for n in scoped["-0.000000"]["facts_not_applied"])


def test_assert_size_equal_cannot_be_excluded(tmp_path, monkeypatch):
    extra = exclusions(tmp_path / "exclude.yaml", (identity(row()), ["AssertSizeEqual"]))
    result = specialize(tmp_path, [row()], monkeypatch, extra=extra)
    assert result["code"] == 2, "mandatory exact sizes cannot be excluded"
    assert result["logic"] == {}


# Break: str() a non-string identity value so it silently fails (or happens) to match.
@pytest.mark.parametrize("field,value", [("beta", 0.0), ("alpha", 1.0), ("pointer_mode", 0), ("bias_vector", False), ("M", 32)])
def test_exclusions_require_exact_logged_string_values(field, value, tmp_path, monkeypatch, capsys):
    record = row()
    control = specialize(tmp_path / "quoted", [record], monkeypatch,
                         extra=exclusions(tmp_path / "quoted.yaml", (identity(record), ["UseBeta"])))
    assert control["code"] == 0
    assert one_solution(control)["ProblemType"]["UseBeta"] is True
    unquoted = dict(identity(record), **{field: value})
    result = specialize(tmp_path / "plain", [record], monkeypatch,
                        extra=exclusions(tmp_path / "plain.yaml", (unquoted, ["UseBeta"])))
    assert result["code"] == 2, "non-string exclusion identity values must not silently fail to match"
    captured = capsys.readouterr()
    diagnostic = (captured.out + captured.err).lower()
    assert "string" in diagnostic or "quot" in diagnostic
    assert result["logic"] == {}


# Break: silently discard an exclusion that matched no call.
def test_unused_exclusions_are_reported_without_discarding_valid_matches(tmp_path, monkeypatch):
    record = row()
    unused = identity(row(N=619))
    result = specialize(tmp_path, [record], monkeypatch, extra=exclusions(
        tmp_path / "exclusions.yaml", (identity(record), ["UseBeta"]), (unused, ["UseBeta"])))
    assert result["code"] == 0
    assert one_solution(result)["ProblemType"]["UseBeta"] is True
    warnings = result["manifest"].get("unused_exclusions", [])
    assert len(warnings) == 1, "unmatched exclusions must remain visible in the manifest"
    assert warnings[0]["identity"] == unused
    assert warnings[0]["recipe"] == recipe_key()
    assert warnings[0]["facts"] == ["UseBeta"]


# Break: hardcode gfx950 inputs or omit the untested label on any output.
def test_gfx942_selects_its_inputs_and_labels_every_output(tmp_path, monkeypatch):
    result = specialize(tmp_path, [row("g942")], monkeypatch, arch="gfx942")
    assert result["code"] == 0
    logic = one_solution(result)
    assert logic["ArchitectureName"] == "gfx942"
    assert len(result["manifest"]["calls"]) == 1
    assert result["manifest"]["arch"] == "gfx942"
    assert result["manifest"]["calls"][0]["recipe"]["source_file"] == RECIPES["g942"]["logic_file"]
    for filename in ("calls.csv", "specialize_manifest.json", "refusals.json"):
        assert "untested-gfx942" in (result["work"] / filename).read_text()
    assert "untested-gfx942" in json.dumps(logic) or "untested-gfx942" in next(iter(result["logic"]))
    assert not (result["work"] / "logic/gfx950").exists()


# The README's specialize example runs on the portable fixture and writes the corpus.
def test_readme_specialize_example_runs_on_portable_fixture(tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row()])
    out = tmp_path / "out"
    examples = readme_specialize_examples((Path(__file__).parents[1] / "README.md").read_text())
    assert examples, "README must document a runnable --specialize example"
    args = examples[0]
    replacements = {"--workload-log": str(logs.parent), "--route-facts": str(facts.parent),
                    "--hipblaslt": str(tmp_path / "baseline"), "--arch": "gfx950", "--workdir": str(out)}
    for flag, value in replacements.items():
        if flag in args:
            args[args.index(flag) + 1] = value
    result = outputs(out, invoke(args, monkeypatch))
    assert result["code"] == 0
    assert len(result["calls"]) == 1, "README example must produce one fixture call"
    assert len(result["logic"]) == 1
    assert result["refusals"] == []


def seed_user_files(work):
    files = {
        work / "logic/mine/keep.yaml": b"user: keep this logic\n",
        work / "logic/gfx950/handwritten.yaml": b"user: same architecture\n",
        work / "logic/gfx942/stock.yaml": b"user: other architecture\n",
        work / "logic/gfx950/notes.txt": b"user notes inside logic\n",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return files


def previous_run(root, monkeypatch):
    logs, facts = configured(root, [row()])
    before = generated(root, logs, facts, monkeypatch)
    assert before["code"] == 0
    assert len(before["calls"]) == len(before["manifest"]["calls"]) == len(before["logic"]) == 1
    assert before["refusals"] == []
    return logs, facts, before


# Break: invalidate before a failed path/jobs check, deleting user or previous geko files.
@pytest.mark.parametrize("usage", ["missing_trace", "missing_routes", "jobs"])
def test_usage_error_preserves_user_logic_and_previous_specialize_outputs(usage, tmp_path, monkeypatch):
    logs, facts, _ = previous_run(tmp_path, monkeypatch)
    work = tmp_path / "out"
    snapshots = seed_user_files(work)
    written = [work / name for name in ("calls.csv", "specialize_manifest.json", "refusals.json")]
    written += list((work / "logic/gfx950").glob("gfx950_bespoke__*.yaml"))
    snapshots.update({path: path.read_bytes() for path in written})
    args = command(tmp_path, logs.parent, facts.parent, work)
    if usage == "jobs":
        args += ["--jobs", "65"]
    else:
        flag = "--workload-log" if usage == "missing_trace" else "--route-facts"
        args[args.index(flag) + 1] = str(tmp_path / "nonexistent-input")
    assert invoke(args, monkeypatch) == 2
    for path, content in snapshots.items():
        assert path.is_file(), f"usage error deleted a previously existing file: {path.relative_to(work)}"
        assert path.read_bytes() == content


# Break: delete logic wholesale, or leave stale owned logic/current outputs.
@pytest.mark.parametrize("outcome", ["success", "whole_input"])
def test_invalidation_removes_only_owned_artifacts_under_user_logic(outcome, tmp_path, monkeypatch):
    logs, facts, before = previous_run(tmp_path, monkeypatch)
    work = tmp_path / "out"
    old_logic = work / "logic/gfx950" / next(iter(before["logic"]))
    user_files = seed_user_files(work)
    trace = logs / "hipblaslt_101.log"
    if outcome == "success":
        input_fixture(tmp_path, [row(N=620)], cfg=logs.name)
    else:
        trace.write_text("- { broken\n")
    code = invoke(command(tmp_path, logs.parent, facts.parent, work), monkeypatch)
    assert code == (0 if outcome == "success" else 1)
    for path, content in user_files.items():
        assert path.is_file(), f"specialization deleted user logic: {path.relative_to(work)}"
        assert path.read_bytes() == content
    assert not old_logic.exists(), "an obsolete generated solution must not survive invalidation"
    logic = list((work / "logic/gfx950").glob("gfx950_bespoke__*.yaml"))
    refusals = json.loads((work / "refusals.json").read_text())["refusals"]
    if outcome == "success":
        assert len(logic) == 1
        manifest = json.loads((work / "specialize_manifest.json").read_text())
        assert [c["identity"]["N"] for c in manifest["calls"]] == ["620"]
        assert (work / "calls.csv").is_file()
        assert refusals == []
    else:
        assert logic == []
        assert not (work / "calls.csv").exists()
        assert not (work / "specialize_manifest.json").exists()
        assert len(refusals) == 1
        assert refusals[0]["code"] == "whole-input"
        assert refusals[0]["file"] == str(trace)
        assert "call_count" not in refusals[0]


def empty_user_workspace(root, monkeypatch, user_arch):
    logs, facts = configured(root, [row()])
    # Positive control in a distinct workdir; the tested out/ never held a geko artifact.
    control = generated(root, logs, facts, monkeypatch, out=root / "control-out")
    assert control["code"] == 0
    assert len(control["calls"]) == len(control["manifest"]["calls"]) == len(control["logic"]) == 1
    assert control["refusals"] == []
    work = root / "out"
    (work / "logic" / user_arch).mkdir(parents=True)
    return logs, facts, work, [work / "logic" / user_arch, work / "logic"]


# Break: prune every empty arch directory, or remove/recreate the selected one, when
# this run removed no owned file from it.
@pytest.mark.parametrize("user_arch", ["gfx942", "gfx950"], ids=["unrelated_arch", "selected_arch"])
@pytest.mark.parametrize("outcome", ["success", "whole_input"])
def test_unowned_empty_logic_directories_survive_valid_usage(outcome, user_arch, tmp_path, monkeypatch):
    logs, facts, work, directories = empty_user_workspace(tmp_path, monkeypatch, user_arch)
    trace = logs / "hipblaslt_101.log"
    if outcome == "whole_input":
        trace.write_text("- { broken\n")
    with held_directories(directories) as handles:
        code = invoke(command(tmp_path, logs.parent, facts.parent, work), monkeypatch)
        assert code == (0 if outcome == "success" else 1)
        assert_directories_preserved(handles)
    if outcome == "success":
        result = outputs(work, code)
        assert len(result["calls"]) == len(result["manifest"]["calls"]) == len(result["logic"]) == 1
        assert result["calls"][0]["call_count"] == "2"
        assert result["refusals"] == []
    else:
        assert not (work / "calls.csv").exists()
        assert not (work / "specialize_manifest.json").exists()
        assert not list((work / "logic").rglob("*_bespoke__*.yaml"))
        refusals = json.loads((work / "refusals.json").read_text())["refusals"]
        assert len(refusals) == 1
        assert refusals[0]["code"] == "whole-input"
        assert refusals[0]["file"] == str(trace)
        assert "call_count" not in refusals[0]


# Break: invalidate before discovering a missing route-facts input.
@pytest.mark.parametrize("user_arch", ["gfx942", "gfx950"], ids=["unrelated_arch", "selected_arch"])
def test_usage_error_preserves_unowned_empty_logic_directories(user_arch, tmp_path, monkeypatch):
    logs, facts, work, directories = empty_user_workspace(tmp_path, monkeypatch, user_arch)
    args = command(tmp_path, logs.parent, tmp_path / "nonexistent-route-facts", work)
    with held_directories(directories) as handles:
        assert invoke(args, monkeypatch) == 2
        assert_directories_preserved(handles)
    assert not (work / "calls.csv").exists()
    assert not (work / "specialize_manifest.json").exists()
    assert not (work / "refusals.json").exists()
