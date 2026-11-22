# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``geko route-facts``: derive ``--specialize`` route facts for one run configuration.

Inputs are two logs of the same workload: the mask-128 trace (``HIPBLASLT_LOG_MASK=128``,
the calls and their weights) and the mask-16 API trace (``HIPBLASLT_LOG_MASK=16``, every
hipBLASLt API call). The API trace proves the facts the mask-128 trace cannot: that every
call went through the ordinary matmul API (no grouped GEMM, ext or device-args route, no
pinned algorithm index or tuning override), that no amax output pointer was set and that no
bias-gradient epilogue was used. The proof is per configuration: the mask-16 log does not
identify full mask-128 call tuples, so every call of the configuration shares it, and its
weighted mask-128 ``call_count`` total must equal the API ``rocblaslt_matmul`` count.

Run metadata (``run.json``, optionally ``engine.log``, beside each trace, as ``model_run``
writes them; a minimal hand-written ``run.json`` is documented in the README) proves the
**runtime route**: both runs complete and clean, the right log masks and modes, the API run
bound to the trace run (same configuration, model, library fingerprint, tool fingerprints and
model cache), and no runtime setting that bypasses or overrides the hipBLASLt heuristic
(TunableOp, a tuning override file or mode, a pinned TENSILE_SOLUTION_INDEX, an AITER GEMM
config); a TunableOp message in ``engine.log`` counts too. Settings are reported by name only,
never by value. Without ``run.json`` beside both traces the runtime route is unproven: the
calls are written with ``runtime_route: null`` and ``--specialize`` refuses them.

Any observation that breaks the proof is a stop: the document is still written (with its
stops and no calls, so ``--specialize`` refuses every call of the configuration) and the
exit status is 1 (also when the runtime route is unproven). Nothing is assumed: a fact is
written only with its evidence (file, line and sha256 of the log line, scanned range or
run.json). Schema: ``docs/route_facts.schema.json``. The rules are those of the reference
producer the corpus route facts were made with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from geko.specialize.calls import UNKNOWN, CallIdentity, _records, configuration_name
from geko.specialize.facts import HEURISTIC_ROUTE

ORDINARY_ROUTE = "ordinary matmul API"
LOG_NAME = re.compile(r"^hipblaslt_(\d+)\.log$")
API = re.compile(r"\]\[Api\]\[(\w+)\]")
ORDINARY_API = frozenset((
    "rocblaslt_create", "rocblaslt_matrix_layout_create", "rocblaslt_matrix_layout_destroy",
    "rocblaslt_matmul_desc_create", "rocblaslt_matmul_desc_destroy", "rocblaslt_matmul_desc_set_attribute",
    "rocblaslt_matmul_preference_create", "rocblaslt_matmul_preference_destroy",
    "rocblaslt_matmul_preference_set_attribute", "rocblaslt_matmul_algo_get_heuristic", "rocblaslt_matmul"))
BIAS_GRADIENT_EPILOGUES = (256, 512)  # HIPBLASLT_EPILOGUE_BGRADA / BGRADB
ENV_SECTIONS = ("env_values", "env_after_import", "aiter_switches")  # model_run run.json environment records
RUNTIME_SETTINGS = {"HIPBLASLT_TUNING_OVERRIDE_FILE": "tuning-override", "HIPBLASLT_TUNING_MODE": "tuning-override",
                    "TENSILE_SOLUTION_INDEX": "pinned-index", "AITER_CONFIG_GEMM_BF16": "pinned-index"}
TUNABLEOP_LOG = re.compile(r"TunableOp.*(?:enabled|tuning|using)|using.*TunableOp", re.I)
BINDING = ("run_configuration", "model", "library_fp", "fingerprints")
CACHE_BINDING = ("path", "sha256", "manifest_sha256", "image_id")
MODEL_WORK = ("engine", "worker")  # process_inventory roles that run the model, so must have a log
# Roles model_run records: the model-work roles above, plus processes that run no GEMM.
KNOWN_ROLES = MODEL_WORK + ("supervisor", "helper")


class UsageError(Exception):
    """Invalid producer input; exit 2, nothing written."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _logs(path: Path, what: str) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        logs = sorted(p for p in path.iterdir() if p.is_file() and LOG_NAME.match(p.name))
        if logs:
            return logs
    raise UsageError(f"{what} {path}: not a hipblaslt_<pid>.log file or a directory holding one")


def _pid(path: Path) -> str:
    match = LOG_NAME.match(path.name)
    return match.group(1) if match else path.name


def _stop(code: str, evidence: Any, **details: Any) -> Dict[str, Any]:
    return {"code": code, "evidence": evidence, **details}


def _trace_calls(logs: List[Path]) -> Tuple[Dict[CallIdentity, Dict[str, Any]], Counter, List[Dict[str, Any]]]:
    """Per call identity its weight and line evidence, per process the weighted calls, and stops."""
    calls: Dict[CallIdentity, Dict[str, Any]] = {}
    per_pid: Counter = Counter()
    stops: List[Dict[str, Any]] = []
    for path in logs:
        data = path.read_bytes()
        digest = _sha(data)
        try:
            text = data.decode()
            records = _records(path, text)
            node = yaml.compose(text) if records else None
        except (UnicodeDecodeError, ValueError, yaml.YAMLError) as e:
            raise UsageError(f"{path}: not a mask-128 trace ({e})") from None
        for record, item in zip(records, node.value if node is not None else ()):
            identity = CallIdentity.from_facts(record)
            evidence = {"file": str(path), "line": item.start_mark.line + 1, "sha256": digest}
            count = record["call_count"][1]
            entry = calls.setdefault(identity, {"call_count": 0, "evidence": []})
            entry["call_count"] += count
            entry["evidence"].append(evidence)
            per_pid[_pid(path)] += count
            if identity["function"] is UNKNOWN or str(identity["function"]) != "matmul":
                stops.append(_stop("nonordinary-trace", evidence, value=str(identity["function"])))
            if str(identity["bias_source"]) != "d" or str(identity["gradient"]) != "false":
                stops.append(_stop("bias-axis", evidence, bias_source=str(identity["bias_source"]),
                                   gradient=str(identity["gradient"])))
    return calls, per_pid, stops


def _api_scan(logs: List[Path]) -> Dict[str, Any]:
    """Whole-file scan of the mask-16 API logs: functions per process, witnesses and stops."""
    scan, stops, matmuls, epilogues, per_process = [], [], [], [], {}
    for path in logs:
        data = path.read_bytes()
        digest = _sha(data)
        lines = data.decode(errors="replace").splitlines()
        scan.append({"file": str(path), "line_start": 1, "line_end": max(len(lines), 1), "sha256": digest,
                     "observation": "entire process log scanned; no AMAX_D pointer attribute set"})
        functions: Counter = Counter()
        for number, text in enumerate(lines, 1):
            evidence = {"file": str(path), "line": number, "sha256": digest}
            match = API.search(text)
            fn = match.group(1) if match else ""
            if fn:
                functions[fn] += 1
            if re.search(r"getAlgosFromIndex|get_algos_from_index", text, re.I):
                stops.append(_stop("pinned-index", evidence))
            if re.search(r"OverrideAlgoCount[= :]+[1-9]", text):
                stops.append(_stop("tuning-override", evidence))
            if re.search(r"TunableOp.*(?:enabled|tuning|using)|using.*TunableOp", text, re.I):
                stops.append(_stop("tunableop", evidence))
            if fn.endswith("desc_set_attribute") and "attr=MATMUL_DESC_EPILOGUE" in text.split():
                value = re.search(r"bufData=(0x[0-9a-fA-F]+|[0-9]+)(?:\s|$)", text)
                epilogues.append({"evidence": evidence, "value": value.group(1) if value else None})
                if value is None or int(value.group(1), 0) in BIAS_GRADIENT_EPILOGUES:
                    stops.append(_stop("bias-axis", evidence, epilogue=value.group(1) if value else None))
            if fn.endswith("desc_set_attribute") and "AMAX_D" in text:
                # Setting the attribute at all is an amax request; the printed bufData is not proof of null.
                stops.append(_stop("output-amax", evidence))
            if fn and fn not in ORDINARY_API:
                stops.append(_stop("nonordinary-api", evidence, function=fn))
            if fn == "rocblaslt_matmul":
                matmuls.append(evidence)
        if functions["rocblaslt_matmul_algo_get_heuristic"] < functions["rocblaslt_matmul"]:
            stops.append(_stop("missing-heuristic", scan[-1], pid=_pid(path),
                               heuristics=functions["rocblaslt_matmul_algo_get_heuristic"],
                               matmuls=functions["rocblaslt_matmul"]))
        per_process[_pid(path)] = dict(sorted(functions.items()))
    return {"scan": scan, "stops": stops, "matmuls": matmuls, "epilogues": epilogues, "per_process": per_process}


def _run_metadata(directory: Path, mode: str, mask: int, role: str, logs: List[Path], captured: Dict[str, int]):
    """(run.json or None, its evidence records, stops, gaps) of the run beside one trace.

    A stop is recorded evidence against the proof; a gap is evidence the run.json does not
    record (environment, run_configuration, process inventory), which leaves the runtime
    route unproven: unknown stays unknown, a missing section never reads as empty."""
    path = directory / "run.json"
    if not path.is_file():
        return None, [], [], []
    raw = path.read_bytes()
    try:
        meta = json.loads(raw)
    except ValueError as e:
        raise UsageError(f"{path}: not JSON ({e})") from None
    if not isinstance(meta, dict):
        raise UsageError(f"{path}: not a JSON object")
    evidence = {"file": str(path), "sha256": _sha(raw)}
    stops: List[Dict[str, Any]] = []
    gaps: List[str] = []

    def stop(code: str, pointer: str, **details: Any) -> None:
        stops.append(_stop(code, dict(evidence, json_pointer=pointer), run=role, **details))

    if meta.get("status") != "complete":
        stop("run-incomplete", "/status", status=meta.get("status"))
    if meta.get("mode") != mode:
        stop("run-mode", "/mode", mode=meta.get("mode"), expected=mode)
    if "hipblaslt_log_mask" in meta and meta["hipblaslt_log_mask"] != mask:
        stop("run-mask", "/hipblaslt_log_mask", mask=meta["hipblaslt_log_mask"], expected=mask)
    if "engine_exit" in meta and meta["engine_exit"] != 0:
        stop("run-unclean", "/engine_exit", engine_exit=meta["engine_exit"])
    for key in ("late_signals", "reasons"):
        if meta.get(key):
            stop("run-unclean", f"/{key}")
    if not isinstance(meta.get("env_values"), dict):
        gaps.append(f"the {role} run.json records no env_values mapping (the environment the run used)")
    for section in ENV_SECTIONS:
        values = meta.get(section)
        if section in meta and not isinstance(values, dict):
            gaps.append(f"the {role} run.json {section} is not a mapping")
            continue
        for key, value in (values or {}).items():
            code = RUNTIME_SETTINGS.get(key)
            if key == "PYTORCH_TUNABLEOP_ENABLED" and str(value) == "1":
                code = "tunableop"
            if code:  # the setting is named, its value never written
                stop(code, f"/{section}/{key}")
    if not meta.get("run_configuration"):
        gaps.append(f"the {role} run.json records no run_configuration")
    inventory = meta.get("process_inventory")
    if not isinstance(inventory, list) or not all(isinstance(p, dict) and "pid" in p for p in inventory):
        gaps.append(f"the {role} run.json records no process_inventory (the processes whose logs make the trace)")
    else:
        roles = {str(p["pid"]): p.get("role") for p in inventory}
        unknown = sorted(p for p, r in roles.items() if r not in KNOWN_ROLES)
        if unknown:  # a process of unknown role may be one whose log is missing
            gaps.append(f"the {role} run.json process_inventory records no known role for pid {', '.join(unknown)} "
                        f"(known roles: {', '.join(KNOWN_ROLES)})")
        supplied = {_pid(log) for log in logs}
        for pid in sorted(supplied - set(roles)):
            stop("inventory-stray-log", "/process_inventory", pid=pid)
        for pid in sorted(p for p, r in roles.items() if r in MODEL_WORK and p not in supplied):
            stop("inventory-missing-log", "/process_inventory", pid=pid, role=roles[pid])
        if sum(captured.values()):  # a model-work process that captured nothing while others did: lost log
            for pid in sorted(p for p, r in roles.items() if r in MODEL_WORK and p in supplied and not captured.get(p)):
                stop("inventory-empty-log", "/process_inventory", pid=pid, role=roles[pid])
    records = [evidence]
    engine = directory / "engine.log"
    if engine.is_file():
        data = engine.read_bytes()
        digest = _sha(data)
        lines = data.decode(errors="replace").splitlines()
        for number, text in enumerate(lines, 1):
            if TUNABLEOP_LOG.search(text):
                stops.append(_stop("tunableop", {"file": str(engine), "line": number, "sha256": digest}, run=role))
        records.append({"file": str(engine), "line_start": 1, "line_end": max(len(lines), 1), "sha256": digest,
                        "observation": "engine log scanned; no TunableOp use"})
    return meta, records, stops, gaps


def _binding(trace: Dict[str, Any], api: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(binding keys both runs record with different values, keys the trace records but the
    API run does not). run_configuration must be recorded by both (a gap otherwise)."""
    def keys(meta):
        cache = meta.get("cache") if isinstance(meta.get("cache"), dict) else {}
        return {**{k: meta.get(k) for k in BINDING}, **{f"cache.{k}": cache.get(k) for k in CACHE_BINDING}}

    a, b = keys(trace), keys(api)
    differ = sorted(k for k in a if a[k] is not None and b.get(k) is not None and a[k] != b[k])
    unrecorded = sorted(k for k in a if a[k] is not None and b.get(k) is None)
    return differ, unrecorded


def produce(trace: Path, api_log: Path) -> Dict[str, Any]:
    """The route-facts document of one run configuration (see the module docstring)."""
    trace_logs = _logs(trace, "trace")
    calls, per_pid, stops = _trace_calls(trace_logs)
    api_logs = _logs(api_log, "--api-log")
    api = _api_scan(api_logs)
    stops += api["stops"]
    count = sum(entry["call_count"] for entry in calls.values())
    total = sum(p.get("rocblaslt_matmul", 0) for p in api["per_process"].values())
    if total != count:
        stops.append(_stop("api-count-mismatch", api["scan"], mask16_matmul_count=total, mask128_call_count=count))
    directory = trace if trace.is_dir() else trace.parent
    trace_meta, trace_records, trace_stops, gaps = _run_metadata(directory, "analysis", 128, "trace", trace_logs,
                                                                 dict(per_pid))
    api_dir = api_log if api_log.is_dir() else api_log.parent
    api_meta, api_records, api_stops, api_gaps = _run_metadata(
        api_dir, "analysis-api", 16, "API", api_logs,
        {pid: functions.get("rocblaslt_matmul", 0) for pid, functions in api["per_process"].items()})
    stops += trace_stops + api_stops
    gaps += api_gaps
    if trace_meta is not None and api_meta is not None:
        differ, unrecorded = _binding(trace_meta, api_meta)
        if differ:
            stops.append(_stop("run-binding", trace_records[:1] + api_records[:1], keys=differ))
        gaps += [f"the API run.json does not record {key}, which binds the trace run" for key in unrecorded]
    gaps = [f"no run.json beside the {role} trace" for role, meta in (("trace", trace_meta), ("API", api_meta))
            if meta is None] + gaps
    runtime = {"proven": not gaps, "evidence": trace_records + api_records if not gaps else [],
               "reason": None if not gaps else
               "; ".join(gaps) + ": the runtime route (TunableOp, tuning override, pinned solution index) and "
               "the trace/API binding are not evidenced"}
    rows = []
    if not stops:
        for identity, entry in sorted(calls.items(), key=lambda kv: json.dumps(sorted(_logged(kv[0]).items()))):
            rows.append({
                "identity": _logged(identity), "call_route": ORDINARY_ROUTE, "bias_axis": "M", "output_amax": False,
                "runtime_route": HEURISTIC_ROUTE if runtime["proven"] else None,
                "evidence": {"call_route": api["matmuls"][:1],
                             "bias_axis": entry["evidence"] + [e["evidence"] for e in api["epilogues"]],
                             "output_amax": api["scan"], "runtime_route": runtime["evidence"],
                             "identity": entry["evidence"]},
                "attribution": "whole-configuration ordinary-route proof; the mask-16 log does not identify "
                               "full mask-128 call tuples"})
    zero = count == 0 and not stops and runtime["proven"] and all(
        p.get("rocblaslt_create", 0) >= 1 and not p.get("rocblaslt_matmul") for p in api["per_process"].values())
    return {
        "schema": 1, "run_configuration": configuration_name(directory),
        "matmul_count": total, "mask128_call_count": count, "call_count": count,
        "thin": len(calls) < 2, "distinct_identities": len(calls),
        "reconciliation": {"matched": total == count, "mask16_matmul_count": total, "mask128_call_count": count,
                           "evidence": api["scan"]},
        "per_process": api["per_process"], "stops": stops, "calls": rows, "zero_verified": zero,
        "runtime_route": runtime,
        "scan": api["scan"], "api_epilogues": api["epilogues"], "ordinary_api_allowlist": sorted(ORDINARY_API),
        "run_json": trace_records[0] if trace_records else None,
        "api_run_json": api_records[0] if api_records else None,
        "traces": [{"file": str(p), "sha256": _sha(p.read_bytes())} for p in trace_logs],
        "producer": {"tool": "geko route-facts"},
    }


def _logged(identity: CallIdentity) -> Dict[str, str]:
    return {k: v for k, v in identity.items() if v is not UNKNOWN}


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="geko route-facts",
        description="Write the --specialize route facts of one run configuration from its mask-128 trace "
                    "and its mask-16 API trace, with each run's run.json (and engine.log) beside its logs "
                    "(exit 0: facts proven; 1: stops or unproven runtime route; 2: usage).")
    parser.add_argument("trace", help="mask-128 trace: a hipblaslt_<pid>.log or the configuration directory "
                                      "holding them; the run configuration is the run_configuration of the "
                                      "run.json beside it, else the directory name (the rule --specialize uses)")
    parser.add_argument("--api-log", required=True, metavar="PATH",
                        help="mask-16 API trace of the same workload: a hipblaslt_<pid>.log or its directory")
    parser.add_argument("--out", required=True, metavar="JSON", help="route-facts file to write")
    args = parser.parse_args(argv)
    try:
        document = produce(Path(args.trace), Path(args.api_log))
    except UsageError as e:
        print(f"geko route-facts: usage: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    for stop in document["stops"]:
        details = {k: v for k, v in stop.items() if k not in ("code", "evidence")}
        print(f"geko route-facts: stop {stop['code']} {details}: no call facts written", file=sys.stderr)
    if not document["runtime_route"]["proven"]:
        print(f"geko route-facts: runtime route unproven: {document['runtime_route']['reason']}; "
              "--specialize refuses these calls", file=sys.stderr)
    return 1 if document["stops"] or not document["runtime_route"]["proven"] else 0
