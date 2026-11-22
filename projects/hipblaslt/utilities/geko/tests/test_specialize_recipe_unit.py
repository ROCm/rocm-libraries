# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""--specialize CLI, baseline build contract and recipe resolution.

recipe.py validates the baseline layout, resolves MatchTable indices, confirms
TensileLite-derived names and enforces per-configuration build fingerprints.
Break: select by kernel-name search, reuse an index across builds, or silently
choose the last observation. All assertions read the CLI's written outputs.
"""
import json
import shutil
import subprocess
import sys

import pytest
import yaml

from tests.specialize_test_support import (
    RECIPES, build_fixture, command, configured, generated, identity, input_fixture, invoke,
    one_solution, outputs, refusal, row, sha, specialize, tree_hash,
)


# Break: keep devices globally required, reject directories, or send specialization to tune.
@pytest.mark.parametrize("form", ["file", "configuration_directories", "stock_only"])
def test_specialize_cli_needs_no_devices(form, tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row()])
    if form == "file":
        logs = logs / "hipblaslt_101.log"
    else:
        logs, facts = logs.parent, facts.parent
    out = tmp_path / "out"
    extra = ["--stock-only"] if form == "stock_only" else []
    code = invoke(command(tmp_path, logs, facts, out, extra=extra), monkeypatch)
    assert code == 0, "valid CPU --specialize must not require --devices"
    assert len(outputs(out, code)["calls"]) == 1


# Break: reinterpret a source checkout as a build or ignore a missing map.
@pytest.mark.parametrize("component", ["device-library/MatchTable.yaml", "Tensile/library/gfx950", "params.txt"])
def test_missing_baseline_component_is_usage_error(component, tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row()])
    missing = tmp_path / "baseline" / component
    if missing.is_dir():
        shutil.rmtree(missing)
    else:
        missing.unlink()
    assert generated(tmp_path, logs, facts, monkeypatch)["code"] == 2, "missing baseline layout must be a usage error"
    assert not list((tmp_path / "out/logic").rglob("*.yaml"))


# Break: let a missing source root/file or an incomplete map/params entry escape as an
# exception instead of exit 2 before any output. A subprocess shows the real exit code.
@pytest.mark.parametrize("missing", ["source_root", "source_file", "matchtable_position", "matchtable_source", "params_logic"])
def test_missing_baseline_source_entries_are_usage_errors(missing, tmp_path):
    logs, facts = configured(tmp_path, [row()])
    baseline = tmp_path / "baseline"
    stock = baseline / "source-logic" / RECIPES["dp"]["logic_file"]
    table_file = baseline / "device-library/MatchTable.yaml"
    if missing == "source_root":
        shutil.rmtree(baseline / "source-logic")
    elif missing == "source_file":
        stock.unlink()
    elif missing.startswith("matchtable"):
        table = yaml.safe_load(table_file.read_text())
        table[RECIPES["dp"]["baseline_index"]] = [str(stock)] if missing == "matchtable_position" else [None, 0]
        table_file.write_text(yaml.safe_dump(table))
    else:
        (baseline / "params.txt").write_text("src=fixture jobs=1\n")
    out = tmp_path / "out"
    process = subprocess.run([sys.executable, "-m", "geko.cli", *command(tmp_path, logs.parent, facts.parent, out)],
                             capture_output=True, text=True, timeout=90)
    assert process.returncode == 2, "missing baseline source/map/params prerequisites need controlled usage exit 2"
    assert "logic" in process.stderr.lower() or "matchtable" in process.stderr.lower()
    assert not any((out / name).exists() for name in ("calls.csv", "specialize_manifest.json", "refusals.json", "logic"))


# Break: persist only a global index/name or change the stock input as a side effect.
def test_matchtable_recipe_provenance_is_persisted_without_modifying_stock(tmp_path, monkeypatch):
    result = specialize(tmp_path, [row()], monkeypatch)
    assert result["code"] == 0
    assert len(result["manifest"]["calls"]) == 1
    recipe = result["manifest"]["calls"][0]["recipe"]
    metadata = RECIPES["dp"]
    assert recipe["source_file"] == metadata["logic_file"]
    assert recipe["sha256"] == metadata["source_sha256"]
    assert recipe["in_file_index"] == 0
    assert result["calls"][0]["stock_logic_file"] == metadata["logic_file"]
    assert result["calls"][0]["stock_logic_sha256"] == metadata["source_sha256"]
    assert result["calls"][0]["stock_logic_index"] == "0"
    assert result["calls"][0]["stock_solution_index"] == str(metadata["baseline_index"])
    assert sha(tmp_path / "baseline/source-logic" / metadata["logic_file"]) == metadata["source_sha256"]
    assert one_solution(result)["Solutions"][0]["BespokeNarrowing"] is True


# Break: trust one name, omit map bounds checks, or abort all calls on one refusal.
@pytest.mark.parametrize("change,reason", [
    ({"solution_index": "999999"}, "index not in baseline MatchTable"),
    ({"solution_index": str(RECIPES["dp_other"]["baseline_index"])}, "recipe name mismatch"),
    ({"solution_Name": "wrong_solution"}, "recipe name mismatch"),
    ({"kernel_name": "wrong_kernel"}, "recipe name mismatch"),
], ids=["unmapped_index", "index_resolves_other", "solution_name", "kernel_name"])
def test_bad_recipe_observation_refuses_only_its_call(change, reason, tmp_path, monkeypatch):
    result = specialize(tmp_path, [row(call_count=2), row(N=619, call_count=3, **change)], monkeypatch)
    refusal(result, 3, reason)
    assert len(result["logic"]) == 1
    assert len(result["manifest"]["calls"]) == 1
    assert result["manifest"]["calls"][0]["call_count"] == 2
    assert result["manifest"]["calls"][0]["identity"]["N"] == "618"


# Break: require global uniqueness of a kernel name although the MatchTable is unambiguous.
def test_shared_kernel_name_does_not_make_two_resolved_recipes_ambiguous(tmp_path, monkeypatch):
    result = specialize(tmp_path, [row(call_count=2), row("duplicate", N=619, call_count=3)], monkeypatch)
    assert result["code"] == 0
    assert len(result["manifest"]["calls"]) == 2
    assert result["refusals"] == []
    assert {c["recipe"]["source_file"] for c in result["manifest"]["calls"]} == {
        RECIPES["dp"]["logic_file"], RECIPES["duplicate"]["logic_file"]}


# Break: last recipe wins, or recipe fields split the call identity.
def test_one_identity_with_two_recipes_retains_counts_but_generates_no_logic(tmp_path, monkeypatch):
    first, second = row(call_count=2), row("duplicate", call_count=3)
    assert identity(first) == identity(second)
    result = specialize(tmp_path, [first, second], monkeypatch)
    refusal(result, 5, "two recipes for one call identity")
    assert result["logic"] == {}
    assert sorted(int(c["call_count"]) for c in result["calls"]) == [2, 3]
    assert {c["stock_solution_index"] for c in result["calls"]} == {
        str(RECIPES["dp"]["baseline_index"]), str(RECIPES["duplicate"]["baseline_index"])}


# Each configuration is checked against the selected arch's content tree; run.json
# without library_fp is not the no-run.json (external, unverified) allowance.
# Break: compare path/mtime alone, trust a changed build, or treat a missing key as external.
@pytest.mark.parametrize("defect,reason", [("mismatch", "trace/build fingerprint mismatch"),
                                           ("missing", "trace fingerprint missing in run.json")],
                         ids=["mismatch", "missing"])
def test_bad_fingerprint_refuses_only_its_configuration(defect, reason, tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row(call_count=2)], cfg="verified")
    bad, _ = input_fixture(tmp_path, [row(call_count=7)], cfg="bad")
    input_fixture(tmp_path, [row(call_count=3)], cfg="external", run=False)
    metadata = json.loads((bad / "run.json").read_text())
    if defect == "mismatch":
        metadata["library_fp"] = "f" * 64
    else:
        del metadata["library_fp"]
    (bad / "run.json").write_text(json.dumps(metadata))
    result = generated(tmp_path, logs, facts, monkeypatch)
    entry = refusal(result, 7, reason)
    assert entry["run_configurations"] == ["bad"]
    assert len(result["logic"]) == 1
    assert {c["run_configuration"]: c["call_count"] for c in result["manifest"]["calls"]} == {"verified": 2, "external": 3}
    build = result["manifest"]["build"]
    assert build["trace_fingerprints"]["verified"] == "verified"
    assert build["trace_fingerprints"]["external"] == "unverified"
    assert build["fingerprint"] == tree_hash(tmp_path / "baseline/Tensile/library/gfx950")


# Break: perform the proven-zero early return before the fingerprint decision.
@pytest.mark.parametrize("fingerprint", ["missing", "mismatch"])
@pytest.mark.parametrize("text", ["", "[]\n", "# no matmul records\n"], ids=["empty", "sequence", "comments"])
def test_proven_zero_configuration_still_requires_its_existing_fingerprint(fingerprint, text, tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row(call_count=3)], cfg="healthy")
    for cfg, run in (("verified-zero", True), ("external-zero", False), ("bad-zero", True)):
        zero_logs, _ = input_fixture(tmp_path, [], cfg=cfg, run=run)
        (zero_logs / "hipblaslt_101.log").write_text(text)
    metadata_file = tmp_path / "traces/bad-zero/run.json"
    metadata = json.loads(metadata_file.read_text())
    if fingerprint == "missing":
        del metadata["library_fp"]
    else:
        metadata["library_fp"] = "f" * 64
    metadata_file.write_text(json.dumps(metadata))
    result = generated(tmp_path, logs, facts, monkeypatch)
    # Positive controls in the same run: a real call, a verified zero, an external zero.
    assert {c["run_configuration"]: c["call_count"] for c in result["manifest"]["calls"]} == {"healthy": 3}
    assert len(result["logic"]) == 1
    fingerprints = result["manifest"]["build"]["trace_fingerprints"]
    assert fingerprints["healthy"] == fingerprints["verified-zero"] == "verified"
    assert fingerprints["external-zero"] == "unverified"
    assert fingerprints["bad-zero"] == fingerprint
    assert result["code"] == 1, "proven zero calls do not authorize a missing or mismatched build fingerprint"
    assert {c["run_configuration"] for c in result["manifest"]["zero_call_configurations"]} == {"verified-zero", "external-zero"}
    assert [(c["run_configuration"], c["call_count"], c["status"]) for c in result["calls"]] == [("healthy", "3", "generated")]
    assert len(result["refusals"]) == 1
    entry = result["refusals"][0]
    assert entry["run_configuration"] == "bad-zero"
    assert entry["call_count"] == 0
    assert entry["identity"] is None
    assert entry["code"] == ("fingerprint-missing" if fingerprint == "missing" else "fingerprint")
    assert entry["reason"]
    assert [c["run_configuration"] for c in result["manifest"]["configuration_errors"]] == ["bad-zero"]


# Break: fabricate a verified fingerprint or demand run.json for an external log.
def test_single_external_log_without_run_json_records_unverified(tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row()], run=False)
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs / "hipblaslt_101.log", facts, out), monkeypatch))
    assert result["code"] == 0
    assert len(result["calls"]) == 1
    assert "unverified" in json.dumps(result["manifest"]["build"]).lower()
    assert len(result["logic"]) == 1


# Break: overwrite the second refusal, lose counts, or drop configuration names.
def test_refused_identity_across_configurations_retains_total_and_attribution(tmp_path, monkeypatch):
    logs, facts = configured(tmp_path, [row(solution_index=999999, call_count=2)], cfg="first")
    input_fixture(tmp_path, [row(solution_index=999999, call_count=3)], cfg="second")
    result = generated(tmp_path, logs, facts, monkeypatch)
    entry = refusal(result, 5, "index not in baseline MatchTable")
    assert "first" in json.dumps(entry)
    assert "second" in json.dumps(entry)
    assert result["logic"] == {}
