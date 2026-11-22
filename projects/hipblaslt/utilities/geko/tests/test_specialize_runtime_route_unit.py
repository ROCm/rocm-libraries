# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""geko --specialize runtime-route proof: bound model runs and API witnesses, per-process
capture completeness, inventory roles, recorded settings and zero-call configurations.

Pass/break pairs are recorded beside each case. Specialization and derivation are
real; only GPU enumeration is forbidden by the shared helper.
"""

import json

import pytest

from tests.specialize_test_support import build_fixture, input_fixture, invoke, outputs, row, write_rows
from tests.test_route_facts_producer_unit import api_trace, produce
from tests.test_specialize_recipe_unit import command


HEURISTIC = "hipBLASLt heuristic"


MATMUL = ("[rocblaslt_matmul] A=0x1000 Adesc=0x2000 B=0x3000 Bdesc=0x4000 C=0x5000 Cdesc=0x6000 D=0x5000 "
          "Ddesc=0x6000 computeDesc=0x1234 algo=0x7000 workSpace=0 workSpaceSizeInBytes=0 stream=0")


def bound_runs(tmp_path, records, *, api_count=None):
    """Trace and API run directories with complete, mutually bound model-run metadata."""
    build_fixture(tmp_path / "baseline")
    logs, _ = input_fixture(tmp_path, records)
    count = sum(int(r["call_count"]) for r in records)
    api = api_trace(tmp_path / "api", count if api_count is None else api_count)
    trace_meta = json.loads((logs / "run.json").read_text())
    cache = {"path": "/cache/model", "sha256": "1" * 64, "manifest_sha256": "2" * 64, "image_id": "sha256:" + "3" * 64}
    common = {"model": trace_meta["model"], "library_fp": trace_meta["library_fp"], "cache": cache,
              "fingerprints": {"tool": "4" * 64}, "engine_exit": 0, "late_signals": [], "reasons": [],
              "env_values": {"HIPBLASLT_LOG_MASK": "x"}, "env_after_import": {}, "aiter_switches": {}}
    trace_meta.update(common, hipblaslt_log_mask=128)
    (logs / "run.json").write_text(json.dumps(trace_meta))
    api_meta = json.loads((api / "run.json").read_text())
    api_meta.update(common, hipblaslt_log_mask=16, mode="analysis-api")
    (api / "run.json").write_text(json.dumps(api_meta))
    for directory in (logs, api):
        (directory / "engine.log").write_text("INFO engine started\nINFO engine stopped\n")
    return logs, api


def edit(path, **changes):
    meta = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
    path.write_text(json.dumps(meta))


def codes(document):
    return {stop["code"] for stop in document["stops"]}


# Pass: complete, bound runs prove the runtime route from run.json and engine.log, and the
# facts drive real generation. Break: drop the runtime_route fact or its evidence.
def test_bound_complete_runs_prove_runtime_route_and_generate(tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 0
    document = json.loads(facts.read_text())
    assert document["stops"] == []
    entry = document["calls"][0]
    assert entry["runtime_route"] == HEURISTIC
    witnesses = entry["evidence"]["runtime_route"]
    assert {w["file"] for w in witnesses} >= {str(logs / "run.json"), str(api / "run.json")}
    assert all(len(w["sha256"]) == 64 for w in witnesses)
    result = outputs(tmp_path / "out", invoke(command(tmp_path, logs, facts, tmp_path / "out", extra=["--keep-beta"]),
                                              monkeypatch))
    assert result["code"] == 0
    assert len(result["logic"]) == 1


# Pass: a failed, unclean or wrong-mask/mode API run is a stop, never a proof.
# Break: ignore the API run.json (only count the matmul lines).
@pytest.mark.parametrize("change,code", [
    ({"status": "failed"}, "run-incomplete"),
    ({"mode": "analysis"}, "run-mode"),
    ({"mode": None}, "run-mode"),
    ({"hipblaslt_log_mask": 128}, "run-mask"),
    ({"engine_exit": 1}, "run-unclean"),
    ({"late_signals": ["SIGTERM"]}, "run-unclean"),
    ({"reasons": ["killed"]}, "run-unclean"),
], ids=["failed", "trace-mode", "no-mode", "mask128", "engine-exit", "late-signal", "reasons"])
def test_failed_or_wrong_api_run_is_a_stop(change, code, tmp_path, monkeypatch, capsys):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit(api / "run.json", **change)
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    assert code in codes(document)
    assert document["calls"] == []
    assert code in capsys.readouterr().err


# Pass: the mask-128 run's own metadata is gated the same way.
# Break: check only the API run.
def test_incomplete_trace_run_is_a_stop(tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit(logs / "run.json", status="running")
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    assert "run-incomplete" in codes(json.loads(facts.read_text()))


# Pass: an API run of another configuration/library/model/cache is a binding stop.
# Break: accept any API log whose matmul count happens to match.
@pytest.mark.parametrize("key,value", [
    ("library_fp", "f" * 64), ("run_configuration", "other-cfg"), ("model", {"slug": "other-model"}),
    ("cache", {"path": "/cache/other", "sha256": "1" * 64, "manifest_sha256": "2" * 64, "image_id": "x"}),
    ("fingerprints", {"tool": "5" * 64}),
])
def test_unbound_api_run_is_a_stop(key, value, tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit(api / "run.json", **{key: value})
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    assert "run-binding" in codes(document)
    assert document["calls"] == []


# Pass: runtime route settings recorded in run.json or engine.log stop the proof, and
# their values are never written or printed. Break: scan only the API log lines.
@pytest.mark.parametrize("side", ["api", "trace"])
@pytest.mark.parametrize("section,key,value,code", [
    ("env_values", "PYTORCH_TUNABLEOP_ENABLED", "1", "tunableop"),
    ("env_after_import", "HIPBLASLT_TUNING_OVERRIDE_FILE", "/private/override-value.txt", "tuning-override"),
    ("env_values", "HIPBLASLT_TUNING_MODE", "private-mode-value", "tuning-override"),
    ("env_values", "TENSILE_SOLUTION_INDEX", "private-index-value", "pinned-index"),
    ("aiter_switches", "AITER_CONFIG_GEMM_BF16", "/private/aiter-gemm-value.csv", "pinned-index"),
    ("engine.log", None, "INFO TunableOp enabled for GEMM", "tunableop"),
], ids=["tunableop-env", "override-file", "tuning-mode", "solution-index", "aiter-gemm-config", "engine-log"])
def test_runtime_route_settings_are_stops_without_leaking_values(side, section, key, value, code, tmp_path,
                                                                 monkeypatch, capsys):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    directory = api if side == "api" else logs
    if section == "engine.log":
        (directory / "engine.log").write_text(f"INFO start\n{value}\n")
    else:
        meta = json.loads((directory / "run.json").read_text())
        meta.setdefault(section, {})[key] = value
        (directory / "run.json").write_text(json.dumps(meta))
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = facts.read_text()
    assert code in codes(json.loads(document))
    assert json.loads(document)["calls"] == []
    captured = capsys.readouterr()
    if "private" in value:  # distinctive values only; "1" occurs in any hash
        assert value not in document + captured.out + captured.err, "environment values must never be emitted"


# Pass: without run metadata the runtime route and the trace/API binding are unproven;
# the producer says so and the consumer refuses those calls. Break: treat absent metadata as clean.
@pytest.mark.parametrize("missing", ["api", "trace", "both"])
def test_absent_run_metadata_leaves_runtime_route_unproven(missing, tmp_path, monkeypatch, capsys):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    if missing in ("api", "both"):
        (api / "run.json").unlink()
    if missing in ("trace", "both"):
        (logs / "run.json").unlink()
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    assert document["runtime_route"]["proven"] is False
    assert len(document["calls"]) == 1
    assert document["calls"][0]["runtime_route"] is None
    assert not document["calls"][0]["evidence"].get("runtime_route")
    assert "runtime route" in capsys.readouterr().err.lower()
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs, facts, out), monkeypatch))
    assert result["code"] == 1
    assert result["logic"] == {}
    assert [r["code"] for r in result["refusals"]] == ["runtime-route-unproven"]
    assert result["refusals"][0]["call_count"] == 5


# Pass: a route-facts document without the runtime_route fact is unknown, never trusted.
# Break: require runtime_route only when the field is present.
def test_route_document_without_runtime_route_is_refused(tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row(call_count=7)])
    document = json.loads(facts.read_text())
    for call in document["calls"]:
        del call["runtime_route"]
        del call["evidence"]["runtime_route"]
    facts.write_text(json.dumps(document))
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs.parent, facts.parent, out), monkeypatch))
    assert result["code"] == 1
    assert result["logic"] == {}
    assert result["refusals"][0]["code"] == "runtime-route-unproven"
    assert "hipBLASLt heuristic" in result["refusals"][0]["reason"]


# Pass: the count mismatch is reported as structured numbers and a named stop.
# Break: report the mismatch without both counts (an assertion on hashes alone can match).
def test_unreconciled_counts_report_both_numbers(tmp_path, monkeypatch, capsys):
    logs, api = bound_runs(tmp_path, [row(call_count=5)], api_count=4)
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    stop = next(s for s in document["stops"] if s["code"] == "api-count-mismatch")
    assert (stop["mask16_matmul_count"], stop["mask128_call_count"]) == (4, 5)
    err = capsys.readouterr().err
    assert "api-count-mismatch" in err and "mask16_matmul_count" in err and "4" in err and "5" in err


# Pass: a stopped producer document makes --specialize name the stop codes.
# Break: report only the generic "no route-facts entry".
def test_specialize_names_producer_stops(tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)], api_count=4)
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs, facts, out), monkeypatch))
    assert result["code"] == 1
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["code"] == "route-facts-stopped"
    assert "api-count-mismatch" in result["refusals"][0]["reason"]


# Pass: an uncounted configuration reports an unknown (null) count and says so.
# Break: write 0, which reads as a proven zero.
def test_run_incomplete_count_is_unknown_not_zero(tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row(call_count=7)])
    edit(logs / "run.json", status="running")
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs.parent, facts.parent, out), monkeypatch))
    assert result["code"] == 1
    entry = result["refusals"][0]
    assert entry["code"] == "run-incomplete"
    assert entry["call_count"] is None
    assert "not counted" in entry["reason"]
    assert result["manifest"]["configuration_errors"][0]["call_count"] is None


def unproven(tmp_path, logs, api, monkeypatch, word):
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    assert document["runtime_route"]["proven"] is False
    assert word in document["runtime_route"]["reason"]
    assert all(call["runtime_route"] is None and not call["evidence"]["runtime_route"] for call in document["calls"])
    return document


def stopped(tmp_path, logs, api, monkeypatch, code):
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    assert code in {s["code"] for s in document["stops"]}
    assert document["calls"] == []
    return document


# Pass: an environment is evidence only when the section is recorded as a mapping
# (empty = recorded, none set). Break: read a missing section as an empty environment.
@pytest.mark.parametrize("side", ["trace", "api"])
@pytest.mark.parametrize("value", [None, ["PYTORCH_TUNABLEOP_ENABLED"], "unset"], ids=["absent", "list", "string"])
def test_unrecorded_environment_leaves_runtime_route_unproven(side, value, tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit((logs if side == "trace" else api) / "run.json", env_values=value)
    unproven(tmp_path, logs, api, monkeypatch, "env_values")


# Pass: the binding is proven only when both runs record run_configuration and every
# binding key the trace records. Break: skip keys missing on one side.
@pytest.mark.parametrize("side,key", [
    ("api", "library_fp"), ("api", "model"), ("api", "cache"), ("api", "fingerprints"),
    ("api", "run_configuration"), ("trace", "run_configuration"),
])
def test_unrecorded_binding_leaves_runtime_route_unproven(side, key, tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit((logs if side == "trace" else api) / "run.json", **{key: None})
    unproven(tmp_path, logs, api, monkeypatch, key)


# Pass: the process inventory is required and bounds the supplied logs.
# Break: read whichever logs happen to be present.
@pytest.mark.parametrize("side", ["trace", "api"])
def test_missing_process_inventory_leaves_runtime_route_unproven(side, tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit((logs if side == "trace" else api) / "run.json", process_inventory=None)
    unproven(tmp_path, logs, api, monkeypatch, "process_inventory")


@pytest.mark.parametrize("side", ["trace", "api"])
def test_log_outside_the_inventory_is_a_stop(side, tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit((logs if side == "trace" else api) / "run.json", process_inventory=[{"pid": 7, "role": "supervisor"}])
    stopped(tmp_path, logs, api, monkeypatch, "inventory-stray-log")


@pytest.mark.parametrize("side", ["trace", "api"])
def test_model_work_process_without_log_is_a_stop(side, tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit((logs if side == "trace" else api) / "run.json",
         process_inventory=[{"pid": 101, "role": "engine"}, {"pid": 202, "role": "worker"}])
    stopped(tmp_path, logs, api, monkeypatch, "inventory-missing-log")


# Pass: a zero-call configuration also needs the proven runtime route.
# Break: accept zero_verified from a document without runtime-route proof.
@pytest.mark.parametrize("proof", [True, False], ids=["proven", "unproven"])
def test_zero_call_configuration_requires_runtime_route(proof, tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [])
    document = json.loads(facts.read_text())
    assert document["zero_verified"] is True and document["thin"] is True
    if not proof:
        document["runtime_route"] = {"proven": False, "evidence": [], "reason": "no run.json"}
        facts.write_text(json.dumps(document))
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs.parent, facts.parent, out), monkeypatch))
    if proof:
        assert result["code"] == 0
        assert [z["run_configuration"] for z in result["manifest"]["zero_call_configurations"]] == ["fixture-on"]
    else:
        assert result["code"] == 1
        assert result["manifest"]["zero_call_configurations"] == []
        assert [e["code"] for e in result["manifest"]["configuration_errors"]] == ["runtime-route-unproven"]


def api_lines(pid, count):
    prefix = f"[2026-11-19 00:00:00][HIPBLASLT][{pid}][Api]"
    lines = [prefix + "[rocblaslt_create] handle[out]=0x1234"]
    for _ in range(count):
        lines += [prefix + "[rocblaslt_matmul_algo_get_heuristic] returnAlgoCount=1", prefix + MATMUL]
    return lines


def inventory(logs, api, entries):
    for directory in (logs, api):
        edit(directory / "run.json", process_inventory=entries)


def two_processes(tmp_path, trace_202, api_202):
    """Process 101 (engine) with 5 calls; process 202 (worker) with the given mask-128 and API calls."""
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    write_rows(logs / "hipblaslt_202.log", [row(N=700, call_count=trace_202)] if trace_202 else [])
    if trace_202 and not api_202:  # keep the totals reconciled: the API calls land in process 101
        with open(api / "hipblaslt_101.log", "a") as f:
            f.write("\n".join(api_lines(101, trace_202)[1:]) + "\n")
    (api / "hipblaslt_202.log").write_text("\n".join(api_lines(202, api_202)) + "\n" if api_202 else "")
    inventory(logs, api, [{"pid": 101, "role": "engine"}, {"pid": 202, "role": "worker"}])
    return logs, api


# Pass: every model-work process must have captured calls when the run has calls
# (lost-log gate of the reference producer). Break: check only that the file exists.
@pytest.mark.parametrize("trace_202,api_202,side", [(0, 0, "trace"), (2, 0, "API")], ids=["trace-empty", "api-empty"])
def test_empty_model_work_log_is_a_stop(trace_202, api_202, side, tmp_path, monkeypatch, capsys):
    logs, api = two_processes(tmp_path, trace_202, api_202)
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    stops = [s for s in document["stops"] if s["code"] == "inventory-empty-log"]
    assert stops and any(s["pid"] == "202" and s["run"] == side for s in stops)
    assert document["calls"] == []


# Positive control: an explicit worker role with its own captured calls proves the route.
def test_worker_with_captured_calls_proves_runtime_route(tmp_path, monkeypatch):
    logs, api = two_processes(tmp_path, 2, 2)
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 0
    document = json.loads(facts.read_text())
    assert document["runtime_route"]["proven"] is True
    assert len(document["calls"]) == 2


# Pass: a process whose role is unrecorded or unknown cannot be classified, so the
# completeness of the logs is unproven. Break: treat it as a process that needs no log.
@pytest.mark.parametrize("side", ["trace", "api"])
@pytest.mark.parametrize("entry", [{"pid": 202}, {"pid": 202, "role": "mystery"}, {"pid": 202, "role": "unattributed"},
                                   {"pid": 202, "role": None}],
                         ids=["no-role", "unknown-role", "unattributed", "null-role"])
def test_unclassified_inventory_process_leaves_runtime_route_unproven(side, entry, tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    edit((logs if side == "trace" else api) / "run.json", process_inventory=[{"pid": 101, "role": "engine"}, entry])
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 1
    document = json.loads(facts.read_text())
    assert document["runtime_route"]["proven"] is False
    assert "role" in document["runtime_route"]["reason"] and "202" in document["runtime_route"]["reason"]


# Positive control: known non-model roles (supervisor, helper) need no log.
def test_known_non_model_roles_need_no_log(tmp_path, monkeypatch):
    logs, api = bound_runs(tmp_path, [row(call_count=5)])
    inventory(logs, api, [{"pid": 7, "role": "supervisor"}, {"pid": 101, "role": "worker"},
                          {"pid": 300, "role": "helper"}])
    facts = tmp_path / "facts.json"
    assert produce(logs, api, facts, monkeypatch) == 0
    assert json.loads(facts.read_text())["runtime_route"]["proven"] is True
