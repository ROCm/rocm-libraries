#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run a range of packaged solution indices under rocJITsu."""

import argparse
import copy
import csv
import html
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from rocjitsu_known_failures import apply_known_failures
from rocjitsu_sweep_plan import (
    POLICY,
    REGULAR_POLICY,
    make_batches,
    make_plan,
    sha256,
    validate_profile,
)
from tensilelite.Utilities.ClientConfig import problemTypeOptions

DEFAULT_BATCH_SIZES = {"bench": 32, "tensile": 1}


def write_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2) + "\n")


def skipped_cases(job):
    return [
        {
            "case_id": c["id"],
            "index": job["solutions"][0]["index"],
            "shape": c["shape"],
            "status": "SKIPPED",
            "target_dispatches": 0,
            "known_failure": c["skip"],
        }
        for c in job["cases"]
        if c.get("skip")
    ]


def solution_report(job, result, backend):
    """One row of evidence; numerical success alone never means PASS."""
    planned = len(job["cases"])
    observed = sum(c["status"] == "PASSED" for c in result["cases"])
    skipped = [c for c in result["cases"] if c["status"] == "SKIPPED"]
    state = result.get("outcome", "FAIL" if result["failed"] else "PASS")
    if result.get("batch_errors") and "outcome" not in result:
        state = "BATCH FAILED"
    notes = "; ".join(result["errors"] + result.get("batch_errors", []))
    if skipped:
        if not result["failed"]:
            state = "SKIPPED" if len(skipped) == planned else "PASS WITH SKIPS"
        reasons = {
            c["known_failure"]["id"]: c["known_failure"] for c in skipped if "known_failure" in c
        }
        notes += ("; " if notes else "") + f"{len(skipped)}/{planned} cases skipped"
        for key, reason in reasons.items():
            notes += f"; {key}: {reason['reason']} ({', '.join(reason['tracking'])})"
    return [
        str(job["solutions"][0]["index"]),
        state,
        f"{observed}/{planned}",
        f"{result.get('target_dispatches', '?')}/{(planned - len(skipped)) * job.get('target_dispatches_per_case', 2 if backend == 'bench' else 1)}",
        f"{result['batch_seconds']:.2f}" if "batch_seconds" in result else "—",
        notes or "Identity, numerical and race checks passed",
    ]


def failed_batch_ids(results):
    return sorted(
        {
            r.get("batch_id", r["id"])
            for r in results
            if r["failed"] and not r.get("planning_outcome")
        }
    )


def render_report(manifest, summary):
    def cell(value):
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")

    backend = manifest["backend"]
    client = "hipblaslt-bench" if backend == "bench" else "tensilelite-client"
    inventory = manifest.get("inventory", {})
    results = {r["id"]: r for r in summary["results"]}
    lines = [
        f"### {client} solution-index race sweep",
        "",
        f"Original index range: `{cell(manifest['solution_index_range']['begin'])}` "
        f"to `{cell(manifest['solution_index_range']['end'])}` (exclusive; None means catalogue end); "
        f"policy: `{cell(manifest['policy'])}`; "
        f"inventory/preparation: {manifest['preparation_seconds']:.2f} s.",
        f"Artifact fingerprint: `{manifest['artifact_fingerprint']}`.",
        f"Selected solutions: {len(manifest['jobs'])}; eligible/inventory solutions: "
        f"{inventory.get('eligible_solutions', '?')}/{inventory.get('solutions', '?')}; "
        f"eligible/inventory kernel names: "
        f"{inventory.get('eligible_kernel_names', '?')}/{inventory.get('unique_kernel_names', '?')}.",
        "",
        "PASS requires all planned cases, matching identities/dispatches, no races or warnings, and a successful exit. "
        "SKIPPED cases did not run; PASS WITH SKIPS validates only the remaining cases.",
        "VALIDATED_IN_FAILED_PROCESS retains completed per-index checks alongside a process failure. "
        "UNEXECUTED and LAUNCHED_NO_RESULT distinguish absent launches from absent numerical results.",
        f"Failed batches: {len(failed_batch_ids(summary['results']))}/"
        f"{len({j.get('batch_id', j['id']) for j in manifest['jobs']})}.",
        "Full shapes, identities, commands and logs are in this run's reports directory.",
        "",
        "| Solution index | Result | Numerical passes | Target dispatches | Batch seconds | Notes |",
        "|---:|:---|---:|---:|---:|:---|",
    ]
    for job in manifest["jobs"]:
        if job["id"] in results:
            row = solution_report(job, results[job["id"]], backend)
        else:
            final = "unstarted" in summary
            assigned = job["id"] in summary.get("assigned_job_ids", [])
            state = "INCOMPLETE" if assigned else "PENDING"
            if final or (summary.get("stop") and not assigned):
                state = "NOT RUN"
            row = [str(job["solutions"][0]["index"]), state, "—", "—", "—", ""]
        row[-1] += f" (batch-{job.get('batch_id', job['id']):03})"
        lines.append("| " + " | ".join(map(cell, row)) + " |")
    lines += [
        "",
        (
            f"Run status: {'PASS WITH SKIPS' if summary.get('counts', {}).get('SKIPPED') else 'PASS'}."
            if summary.get("passed")
            else (
                "Run status: FAIL."
                if "passed" in summary
                else "Run incomplete; this table is updated after each result."
            )
        ),
        "",
    ]
    if "counts" in summary:
        lines += [
            f"Cases: {summary['counts'].get('PASSED', 0)} numerical passes, "
            f"{summary['counts'].get('SKIPPED', 0)} skipped, "
            f"{len(summary['missing'])} missing, "
            f"{len(summary['unstarted_cases'])} unstarted; "
            f"{summary['planned_cases']} planned.",
            "",
        ]
    if summary.get("stop"):
        lines += [f"Stopped: {cell(summary['stop']['reason'])}.", ""]
    return "\n".join(lines)


def prepare(args):
    begin = time.monotonic()
    explicit_profiles = getattr(args, "hardware_profiles", None)
    configs, profiles = {}, {}
    for number, path in enumerate(explicit_profiles or [args.config]):
        name = f"profile-{number:03}" if explicit_profiles else "default"
        base = json.loads(path.read_text(encoding="utf-8"))
        if explicit_profiles:
            validate_profile(base, args.target)
        base.update(
            max_ticks=0,
            num_threads=1,
            cpu_dispatch_threads=1,
            async_helper_threads=0,
            cpu_thread_budget=1,
        )
        base["plugins"] = {"race": {}}
        base["sinks"] = {"types": ["stderr"]}
        config = args.reports / (f"{name}.json" if explicit_profiles else "rocjitsu.json")
        write_json(config, base)
        configs[name] = config
        profiles[name] = {
            "source": str(path),
            "sha256": sha256(path),
            "config": str(config),
            "device": base["vm"]["gpu"]["device"],
        }
    manifest = make_plan(
        args.library_dir,
        args.target,
        base["vm"]["gpu"]["device"],
        args.solution_index_range_begin,
        args.solution_index_range_end,
        getattr(args, "policy", POLICY),
        profiles={name: profile["device"] for name, profile in profiles.items()},
    )
    manifest.update(
        backend=args.backend,
        workers=args.workers,
        profiles=profiles,
        tools={str(p): sha256(p) for p in (args.rocjitsu, args.client)},
    )
    if args.backend == "bench":
        manifest["tools"].update(
            (str(path), sha256(path))
            for path in (
                Path(sys.executable),
                *(
                    args.client.parent / name
                    for name in (
                        "hipblaslt_gentest.py",
                        "hipblaslt_template.yaml",
                        "hipblaslt_common.yaml",
                    )
                ),
            )
        )
    failures = args.known_failures.read_text(encoding="utf-8")
    apply_known_failures(manifest, failures, args.rocjitsu_revision, args.run_known_failures)
    write_text(args.reports / "known_failures.yaml", failures)
    manifest["preparation_seconds"] = time.monotonic() - begin
    write_json(args.reports / "manifest.json", manifest)
    return manifest, configs


def client_options(job, results):
    problem = job["problem_type"]
    options = problemTypeOptions(problem)
    options.update(
        {
            "library-file": job["library"],
            "results-file": str(results),
            "solution-start-idx": job["solutions"][0]["index"],
            "num-solutions": len(job["solutions"]),
            "bias-type-args": (problem["biasDataTypeWhiteList"] or [problem["computeType"]])[0],
            "activation-enum-args": "None",
            "activation-additional-args": "2.0,2.0",
            "device-idx": 0,
            "init-seed": 20260929,
            "init-a": "Random",
            "init-b": "Random",
            "init-c": "Zero",
            "init-d": "Zero",
            "init-alpha": "One",
            "init-beta": "Zero",
            "init-bias": "Random",
            "init-scaleA": "Two",
            "init-scaleB": "Two",
            "init-scaleC": "Two",
            "init-scaleD": "Two",
            "init-scaleAlphaVec": "One",
            "c-equal-d": False,
            "num-elements-to-validate": -1,
            "num-benchmarks": 1,
            "num-warmups": 0,
            "num-enqueues-per-sync": 1,
            "max-enqueues-per-sync": 1,
            "num-syncs-per-benchmark": 0,
            "use-gpu-timer": False,
            "hardware-monitor": False,
            "sleep-percent": 0,
            "bounds-check": "Disable",
            "print-valids": False,
            "print-max": 4,
            "log-level": "Debug",
            "max-workspace-size": 134217728,
            "PrintWinnersOnly": False,
            "granularity-threshold": 0.0,
            "pristine-on-gpu": True,
            "use-user-args": False,
        }
    )
    return options


def target_dispatch_line(job):
    kernel = job["solutions"][0]["kernel"]
    return f'[rocjitsu] Kernel dispatch: "{kernel}" symbol="{kernel}"'


def classify_tensile(job, text, returncode):
    expected = {(s["index"], tuple(c["shape"])): s for s in job["solutions"] for c in job["cases"]}
    cases = {}
    errors = []
    launches = 0
    for line in text.splitlines():
        launches += line == target_dispatch_line(job)
        if not re.match(r"^0,\d+/\d+,\d+/\d+,", line):
            continue
        row = next(csv.reader([line]))
        key = (
            int(row[2].split("/")[0]),
            tuple(map(int, row[4].strip("()").split(","))),
        )
        if key not in expected:
            errors.append(f"Unplanned case: {key}")
            continue
        if key in cases:
            errors.append(f"Duplicate case: {key}")
        if row[8] != expected[key]["name"]:
            errors.append(f"Wrong solution name: {key}")
        cases[key] = {"status": row[9], "target_dispatches": launches}
        if launches != (1 if row[9] == "PASSED" else 0):
            errors.append(f"Case {key}: unexpected target dispatches ({launches})")
        launches = 0
        if row[9] != "PASSED":
            errors.append(f"Case {key}: {row[9]}")
    missing = sorted(set(expected) - set(cases))
    if missing:
        errors.append(f"{len(missing)} missing case records")
    result = {
        "id": job["id"],
        "cases": [
            {
                "index": i,
                "shape": shape,
                **evidence,
                "case_id": next(c["id"] for c in job["cases"] if tuple(c["shape"]) == shape),
            }
            for (i, shape), evidence in sorted(cases.items())
        ],
    }
    return finish_result(job, text, returncode, result, errors, launches_per_case=1)


def finish_result(job, text, returncode, result, errors, launches_per_case):
    dispatches = Counter(re.findall(r'^\[rocjitsu\] Kernel dispatch: "([^"]+)"', text, re.M))
    kernel = job["solutions"][0]["kernel"]
    wanted = sum(c["status"] == "PASSED" for c in result["cases"]) * launches_per_case
    if dispatches[kernel] != wanted:
        errors.append(
            f"Target dispatch count mismatch: expected {wanted}, observed {dispatches[kernel]}"
        )
    races = re.findall(r"^RACE .*", text, re.M)
    warnings = Counter(line for line in text.splitlines() if "[rj warn]" in line)
    if races:
        errors.append(f"{len(races)} race reports (none suppressed)")
    if warnings:
        errors.append(f"{sum(warnings.values())} emulator warnings")
    if returncode == 124:
        errors.append("Client timed out (exit 124)")
    elif returncode:
        errors.append(f"Client exited {returncode}")
    result.update(
        failed=bool(errors),
        errors=errors,
        returncode=returncode,
        race_headers=races,
        warnings=dict(warnings),
        target_dispatches=dispatches[kernel],
        auxiliary_dispatches={k: v for k, v in dispatches.items() if k != kernel},
    )
    return result


BENCH_TYPES = {
    "Float": "f32_r",
    "Half": "f16_r",
    "BFloat16": "bf16_r",
    "Float8": "f8_r",
    "BFloat8": "bf8_r",
    "Float8_fnuz": "f8_fnuz_r",
    "BFloat8_fnuz": "bf8_fnuz_r",
    "Int8": "i8_r",
    "Int32": "i32_r",
}


def bench_options(job):
    p = job["problem_type"]
    row = {
        "function": "matmul",
        "algo_method": 2,
        "solution_index": job["solutions"][0]["index"],
        "requested_solution_num": 1,
        "transA": "T" if p["transA"] else "N",
        "transB": "T" if p["transB"] else "N",
        **{f"{x}_type": BENCH_TYPES[p[f"{x}Type"]] for x in "abcd"},
        "compute_type": "c_i32_r" if p["computeType"] == "Int32" else "c_f32_r",
        "scale_type": BENCH_TYPES[p["computeType"]],
        "alpha": 1,
        "beta": 0,
        "initialization": "rand_int",
        "bias_vector": bool(p.get("useBias")),
        "bias_type": BENCH_TYPES[(p["biasDataTypeWhiteList"] or [p["computeType"]])[0]],
        "bias_source": {0: "a", 1: "b", 3: "d"}[p["biasSrcWhiteList"][0]],
        "scaleA": {"": "none", "Scalar": "Scalar", "Vector": "Vector"}[p.get("useScaleAB", "")],
        "scaleB": {"": "none", "Scalar": "Scalar", "Vector": "Vector"}[p.get("useScaleAB", "")],
        "scaleC": bool(p.get("useScaleCD")),
        "scaleD": bool(p.get("useScaleCD")),
        "scaleAlpha_vector": bool(p.get("useScaleAlphaVec")),
        "amaxD": bool(p.get("outputAmaxD")),
        "activation_type": "none",
        "norm_check": 1,
        "norm_check_assert": True,
        "allclose_check": 1,
        "iters": 1,
        "cold_iters": 0,
        "use_gpu_timer": False,
        "print_kernel_info": True,
        "user_allocated_workspace": 134217728,
        "skip_slow_solution_ratio": 0.0,
        "adaptive": False,
    }
    # Bench's compute-input overrides support floating-point types only.
    # Integer GEMMs derive their input computation from compute_type.
    if p["computeType"] != "Int32":
        for tensor in "AB":
            row[f"compute_input_type{tensor}"] = BENCH_TYPES[
                p.get(
                    f"computeInputType{tensor}",
                    p.get("computeInputType", p[f"{tensor.lower()}Type"]),
                )
            ]
    if p.get("f32XdlMathOp") == "XFloat32":
        row["compute_type"] = "c_xf32_r"
    return [
        dict(
            row,
            M=c["shape"][0],
            N=c["shape"][1],
            batch_count=c["shape"][2],
            K=c["shape"][3],
        )
        for c in job["cases"]
    ]


def classify_bench(job, text, returncode):
    """Require a numeric result and explicit solution identity for every YAML row."""
    records, headers, pending = [], None, None
    errors = []
    launches = 0
    for line in text.splitlines():
        launches += line == target_dispatch_line(job)
        header = re.match(r"^\[\d+\]:(.*)", line)
        if header:
            headers = next(csv.reader([header[1]]))
            continue
        if headers is not None:
            # Bench prints complex alpha/beta as unquoted (real,imag) scalars.
            # Protect those commas before parsing its otherwise scalar CSV row.
            row = next(
                csv.reader(
                    [
                        re.sub(
                            r"\([^()]*,[^()]*\)",
                            lambda m: '"' + m[0] + '"',
                            line.strip(),
                        )
                    ]
                )
            )
            if len(row) == len(headers):
                pending = dict(zip(headers, row))
                pending["target_dispatches"] = launches
                launches = 0
                records.append(pending)
                headers = None
                continue
        identity = re.match(r"\s*--(Solution index|Solution name|kernel name):\s*(.*)", line)
        if identity and pending is not None:
            pending[identity[1]] = identity[2]
    expected = {tuple(c["shape"]): c for c in job["cases"]}
    cases = {}
    solution = job["solutions"][0]
    for row in records:
        try:
            shape = tuple(int(row[k]) for k in ("m", "n", "batch_count", "k"))
            if shape not in expected:
                raise ValueError(f"Unplanned bench case: {shape}")
            if shape in cases:
                raise ValueError(f"Duplicate bench case: {shape}")
            if (
                int(row["Solution index"]) != solution["index"]
                or row["Solution name"] != solution["name"]
                or row["kernel name"] != solution["kernel"]
            ):
                raise ValueError(f"Wrong bench solution/kernel identity: {shape}")
            # norm_check_assert handles datatype-dependent tolerance in the native
            # client. Require its measurement too; an exit of zero is insufficient.
            numeric = [float(row[k]) for k in ("norm_error", "atol", "rtol")]
            passed = all(math.isfinite(x) and x >= 0 for x in numeric)
            cases[shape] = {
                "case_id": expected[shape]["id"],
                "index": solution["index"],
                "shape": shape,
                "status": "PASSED" if passed else "FAILED",
                "target_dispatches": row["target_dispatches"],
                "numerical": dict(zip(("norm_error", "atol", "rtol"), numeric)),
                "solution_name": row["Solution name"],
                "kernel": row["kernel name"],
            }
            if row["target_dispatches"] != job.get("target_dispatches_per_case", 2):
                errors.append(
                    f"Case {shape}: unexpected target dispatches ({row['target_dispatches']})"
                )
            if not passed:
                errors.append(f"Non-finite bench validation: {shape}")
        except (KeyError, ValueError) as error:
            errors.append(f"Invalid bench record: {error}")
    if len(cases) != len(expected):
        errors.append(f"{len(expected) - len(cases)} missing bench case records")
    return finish_result(
        job,
        text,
        returncode,
        {"id": job["id"], "cases": list(cases.values())},
        errors,
        launches_per_case=job.get("target_dispatches_per_case", 2),
    )


def classify_bench_batch(jobs, text, returncode):
    """Keep record identities, launch evidence and process diagnostics separate."""
    expected_order = [j["solutions"][0]["index"] for j in jobs for c in j["cases"]]
    chunks, record_order, start = {}, [], 0
    batch_errors = []
    for end in re.finditer(r"^\s*--kernel name:[^\n]*", text, re.M):
        chunk = text[start : end.end()]
        start = end.end()
        identity = re.findall(r"^\s*--Solution index:\s*(\d+)\s*$", chunk, re.M)
        if len(identity) != 1:
            batch_errors.append("Bench record has missing or ambiguous solution index")
            continue
        index = int(identity[0])
        record_order.append(index)
        chunks.setdefault(index, []).append(chunk)
    if record_order != expected_order:
        batch_errors.append(
            f"Batch expected {len(expected_order)} ordered records, observed {len(record_order)}"
        )
    if returncode:
        batch_errors.append(f"Batch client exited {returncode}")
    races = re.findall(r"^RACE .*", text, re.M)
    warnings = Counter(line for line in text.splitlines() if "[rj warn]" in line)
    if races:
        batch_errors.append(f"Batch reported {len(races)} races (none suppressed)")
    if warnings:
        batch_errors.append(f"Batch reported {sum(warnings.values())} emulator warnings")

    # Repeated names can identify different indices. Only a verified ordered
    # prefix establishes which cases an aborted process left unexecuted.
    expected_launches = [
        (j["id"], c["id"], target_dispatch_line(j))
        for j in jobs
        for c in j["cases"]
        for _ in range(j.get("target_dispatches_per_case", 2))
    ]
    target_kernels = {j["solutions"][0]["kernel"] for j in jobs}
    observed_launches = []
    for line in text.splitlines():
        dispatch = re.match(r'^\[rocjitsu\] Kernel dispatch: "([^"]+)"', line)
        if dispatch and dispatch[1] in target_kernels:
            observed_launches.append(line)
    prefix = (
        observed_launches == [line for _, _, line in expected_launches[: len(observed_launches)]]
        and len(observed_launches) <= len(expected_launches)
        and record_order == expected_order[: len(record_order)]
    )
    if len(observed_launches) != len(expected_launches) or not prefix:
        batch_errors.append(
            f"Batch target dispatch order/count mismatch: expected {len(expected_launches)}, "
            f"observed {len(observed_launches)}"
        )
    launches = (
        Counter(
            (job_id, case_id) for job_id, case_id, _ in expected_launches[: len(observed_launches)]
        )
        if prefix
        else Counter()
    )
    rejected_case = None
    rejected = "error: NO solution found!" in text
    if rejected:
        batch_errors.append("Native client found no solution for a requested input")
    if prefix and rejected:
        ordered_cases = [(j["id"], c["id"]) for j in jobs for c in j["cases"]]
        if len(record_order) < len(ordered_cases):
            candidate = ordered_cases[len(record_order)]
            if not launches[candidate]:
                rejected_case = candidate
    kernel_owners = Counter(j["solutions"][0]["kernel"] for j in jobs)
    results = []
    for job in jobs:
        index = job["solutions"][0]["index"]
        # Process warnings and exit codes have no per-record identity. They stay
        # in batch_errors; a completed record retains its own numerical evidence.
        records = "\n".join(chunks.get(index, []))
        records = "\n".join(
            line
            for line in records.splitlines()
            if not line.startswith("RACE ") and "[rj warn]" not in line
        )
        result = classify_bench(job, records, 0)
        observed = {c["case_id"]: c for c in result["cases"]}
        for case in job["cases"]:
            if case["id"] in observed:
                continue
            count = launches[job["id"], case["id"]] if prefix else None
            status = (
                "NATIVE_REJECTED"
                if (job["id"], case["id"]) == rejected_case
                else (
                    "LAUNCHED_NO_RESULT"
                    if count
                    else ("UNEXECUTED" if prefix and (returncode or rejected) else "MISSING_RECORD")
                )
            )
            result["cases"].append(
                {
                    "case_id": case["id"],
                    "index": index,
                    "shape": case["shape"],
                    "status": status,
                    "target_dispatches": count,
                }
            )
        if prefix:
            result["target_dispatches"] = sum(launches[job["id"], c["id"]] for c in job["cases"])
        else:
            result["target_dispatches"] = None
        kernel = job["solutions"][0]["kernel"]
        result["race_headers"] = [
            r
            for r in races
            if kernel_owners[kernel] == 1
            and re.search(r"(?:^| )kernel=" + re.escape(kernel) + r"(?: |$)", r)
        ]
        if result["race_headers"]:
            result["errors"].append(f"{len(result['race_headers'])} races naming this kernel")
        result["index_failed"] = bool(result["errors"]) and any(
            c["status"] != "UNEXECUTED" for c in result["cases"]
        )
        result["batch_errors"] = batch_errors.copy()
        result["returncode"] = returncode
        result["process_diagnostics"] = {
            "race_headers": races,
            "warnings": dict(warnings),
        }
        result["failed"] = result["index_failed"] or bool(batch_errors)
        result["outcome"] = index_outcome(result)
        results.append(result)
    return results


def index_outcome(result):
    statuses = {c["status"] for c in result["cases"]}
    for status in (
        "EXCLUDED_HARDWARE",
        "UNSUPPORTED_FEATURE",
        "INVALID_INPUT",
        "MISSING_CODE_OBJECT",
        "NATIVE_REJECTED",
        "LAUNCHED_NO_RESULT",
        "UNEXECUTED",
        "MISSING_RECORD",
    ):
        if status in statuses:
            return status
    if result.get("index_failed", result["failed"] and not result.get("batch_errors")):
        return "FAIL"
    if result.get("batch_errors"):
        return "VALIDATED_IN_FAILED_PROCESS"
    if statuses == {"SKIPPED"}:
        return "SKIPPED"
    if "SKIPPED" in statuses:
        return "PASS_WITH_SKIPS"
    return "PASS"


def convert_bench_input(client, stem, timeout, env):
    """Use the converter/template installed with the tested client."""
    helper = client.parent / "hipblaslt_gentest.py"
    command = [
        sys.executable,
        str(helper),
        "--template",
        str(helper.with_name("hipblaslt_template.yaml")),
        "-o",
        str(stem.with_suffix(".data")),
        str(stem.with_suffix(".yaml")),
    ]
    write_json(stem.with_suffix(".conversion.command.json"), command)
    return execute_command(command, stem.with_suffix(".conversion.log"), timeout, env)


def execute_command(command, log, timeout, env):
    deadline = time.monotonic() + timeout
    with log.open("x", encoding="utf-8") as output:
        if time.monotonic() >= deadline:
            return 124
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def run_queue(batches, workers, execute, progress, deadline=None, continue_on_failure=False):
    lock = threading.Lock()
    results = []
    next_batch = 0
    assigned = []
    active = 0
    maximum = 0
    stop_snapshot = None

    def stop(reason, trigger_job=None):
        nonlocal stop_snapshot
        if stop_snapshot is None:
            stop_snapshot = {
                "reason": reason,
                "trigger_job": trigger_job,
                "assigned_jobs": len(assigned),
                "assigned_job_ids": assigned.copy(),
                "in_flight": active,
            }

    def expired():
        return deadline is not None and time.monotonic() >= deadline

    def worker(slot):
        nonlocal next_batch, active, maximum
        while True:
            with lock:
                if stop_snapshot is not None or next_batch == len(batches):
                    return
                if expired():
                    stop("suite-timeout")
                    return
                batch = batches[next_batch]
                next_batch += 1
                assigned.extend(j["id"] for j in batch)
                active += 1
                maximum = max(maximum, active)
            try:
                completed = execute(batch, slot)
            except Exception as error:
                completed = [
                    {
                        "id": job["id"],
                        "failed": True,
                        "errors": [repr(error)],
                        "cases": [],
                    }
                    for job in batch
                ]
            with lock:
                results.extend(completed)
                active -= 1
                if expired():
                    stop("suite-timeout", batch[0]["id"])
                elif not continue_on_failure:
                    failed = next((r for r in completed if r["failed"]), None)
                    if failed is not None:
                        stop("job-failure", failed["id"])
                try:
                    progress(
                        {
                            "results": results,
                            "completed": completed,
                            "assigned_jobs": len(assigned),
                            "assigned_job_ids": assigned.copy(),
                            "stop": stop_snapshot,
                        }
                    )
                except Exception as error:
                    for result in completed:
                        result["failed"] = True
                        result.setdefault("errors", []).append(f"Cannot write progress: {error!r}")
                    stop("progress-error", batch[0]["id"])

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return {
        "results": sorted(results, key=lambda r: r["id"]),
        "unstarted": [j["id"] for batch in batches[next_batch:] for j in batch],
        "assigned_job_ids": assigned,
        "max_active": maximum,
        "stop": stop_snapshot,
    }


def physical_cpus(count):
    cpus = []
    seen = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (
            (topology / "physical_package_id").read_text(),
            (topology / "core_id").read_text(),
        )
        if key not in seen:
            seen.add(key)
            cpus.append(cpu)
        if len(cpus) == count:
            return cpus
    raise ValueError(f"Need {count} distinct available physical cores; found {len(cpus)}")


def run(args):
    suite_start = time.monotonic()
    deadline = math.inf if args.suite_timeout is None else suite_start + args.suite_timeout
    args.reports.mkdir(parents=True, exist_ok=False)
    try:
        manifest, configs = prepare(args)
        batch_size = getattr(args, "batch_size", None)
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZES[args.backend]
        if args.backend == "tensile" and batch_size != 1:
            raise ValueError("Tensile backend requires batch size 1")
        batches = make_batches(manifest["jobs"], batch_size, group_by_shard=args.backend == "bench")
        for batch_id, batch in enumerate(batches):
            for job in batch:
                job["batch_id"] = batch_id
        manifest.update(
            batch_size=batch_size,
            batches=[[j["id"] for j in batch] for batch in batches],
        )
        write_json(args.reports / "manifest.json", manifest)
        print(
            f"{args.backend}: selected {len(manifest['jobs'])} solutions, "
            f"{manifest['planned_cases']} cases; preparation {manifest['preparation_seconds']:.2f}s. "
            f"Reports: {args.reports}",
            flush=True,
        )
        write_text(args.reports / "summary.md", render_report(manifest, {"results": []}))
        cpus = physical_cpus(args.workers)
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("RJ_", "TENSILE_", "HIPBLASLT_"))
        }
        env.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            HSA_OVERRIDE_CPU_AFFINITY_DEBUG="0",
            HSA_ENABLE_SDMA="1",
        )
        if args.backend == "bench":
            env["HIPBLASLT_TENSILE_LIBPATH"] = manifest["library_dir"]
        write_json(
            args.reports / "settings.json",
            {
                "cpus": cpus,
                "timeout": args.timeout,
                "suite_timeout": args.suite_timeout,
                "backend": args.backend,
                "batch_size": batch_size,
                "continue_on_failure": getattr(args, "continue_on_failure", False),
                "recover_unexecuted": getattr(args, "recover_unexecuted", False),
                "library_dir": manifest["library_dir"],
                "controlled_environment": {
                    k: env[k]
                    for k in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "HSA_OVERRIDE_CPU_AFFINITY_DEBUG",
                        "HSA_ENABLE_SDMA",
                    )
                },
            },
        )

        def execute(batch, slot, recovery=False):
            begin = time.monotonic()
            batch_id = batch[0]["batch_id"]
            stem = args.reports / f"batch-{batch_id:03}"
            if recovery:
                stem = args.reports / f"batch-{batch_id:03}-recovery-{batch[0]['id']:03}"
            completed, runnable = [], []
            for job in batch:
                if "planning_error" in job:
                    completed.append(
                        {
                            "id": job["id"],
                            "failed": True,
                            "errors": [job["planning_error"]],
                            "planning_outcome": job.get("planning_outcome", "INVALID_INPUT"),
                            "outcome": job.get("planning_outcome", "INVALID_INPUT"),
                            "target_dispatches": 0,
                            "cases": [
                                {
                                    "case_id": c["id"],
                                    "index": job["solutions"][0]["index"],
                                    "shape": c["shape"],
                                    "status": job.get("planning_outcome", "INVALID_INPUT"),
                                }
                                for c in job["cases"]
                            ],
                        }
                    )
                    continue
                active_job = {
                    **job,
                    "cases": [c for c in job["cases"] if not c.get("skip")],
                }
                if active_job["cases"]:
                    runnable.append(active_job)
                else:
                    completed.append(
                        {
                            "id": job["id"],
                            "failed": False,
                            "errors": [],
                            "cases": [],
                            "target_dispatches": 0,
                            "auxiliary_dispatches": {},
                            "race_headers": [],
                            "warnings": {},
                            "returncode": None,
                        }
                    )
            # Charge conversion and native execution to one shared batch deadline.
            batch_deadline = min(deadline, begin + args.timeout * len(runnable))
            conversion_seconds = native_seconds = 0
            native_executed = False
            if runnable:
                if args.backend == "tensile":
                    job = runnable[0]
                    options = client_options(job, stem.with_suffix(".csv"))
                    ini = [f"{k}={v}" for k, v in options.items()]
                    ini += ["problem-size=" + ",".join(map(str, c["shape"])) for c in job["cases"]]
                    ini += ["code-object=" + path for path in job["code_objects"]]
                    write_text(stem.with_suffix(".ini"), "\n".join(ini) + "\n")
                    client_args = ["--config-file", str(stem.with_suffix(".ini"))]
                    rc = 0
                else:
                    import yaml

                    write_text(
                        stem.with_suffix(".yaml"),
                        "Tests:\n"
                        + "".join(
                            (
                                job["bench_yaml"]
                                if "bench_yaml" in job
                                else yaml.safe_dump(bench_options(job), sort_keys=False)
                            )
                            for job in runnable
                        ),
                    )
                    conversion_begin = time.monotonic()
                    rc = convert_bench_input(
                        args.client, stem, batch_deadline - conversion_begin, env
                    )
                    conversion_seconds = time.monotonic() - conversion_begin
                    client_args = [
                        "--host_side_fill_kernel",
                        "--data",
                        str(stem.with_suffix(".data")),
                    ]
                if rc:
                    completed += [
                        finish_result(
                            job,
                            "",
                            rc,
                            {"id": job["id"], "cases": []},
                            ["Bench YAML conversion failed; see batch conversion.log"],
                            job.get("target_dispatches_per_case", 2),
                        )
                        for job in runnable
                    ]
                else:
                    command = [
                        "taskset",
                        "-c",
                        str(cpus[slot]),
                        str(args.rocjitsu),
                        "--config",
                        str(configs[runnable[0]["profile"]]),
                        "--",
                        str(args.client),
                        *client_args,
                    ]
                    write_json(stem.with_suffix(".command.json"), command)
                    launch_begin = time.monotonic()
                    native_executed = True
                    rc = execute_command(
                        command,
                        stem.with_suffix(".log"),
                        batch_deadline - launch_begin,
                        env,
                    )
                    native_seconds = time.monotonic() - launch_begin
                    text = stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
                    completed += (
                        [classify_tensile(runnable[0], text, rc)]
                        if args.backend == "tensile"
                        else classify_bench_batch(runnable, text, rc)
                    )
                if rc == 124:
                    for result in completed:
                        result["errors"].append(
                            "Suite time budget exhausted"
                            if time.monotonic() >= deadline
                            else "Batch timeout"
                        )
                        result["failed"] = True
            seconds = time.monotonic() - begin
            for result in completed:
                result.update(
                    batch_id=batch_id,
                    attempt_id=stem.name,
                    profile=batch[0].get("profile"),
                    cpu=cpus[slot],
                    batch_seconds=seconds,
                )
            write_json(
                stem.with_suffix(".result.json"),
                {
                    "id": batch_id,
                    "jobs": [j["id"] for j in batch],
                    "seconds": seconds,
                    "conversion_seconds": conversion_seconds,
                    "native_seconds": native_seconds,
                    "native_executed": native_executed,
                    "profile": batch[0].get("profile"),
                    "results": completed,
                },
            )
            if (
                not recovery
                and args.backend == "bench"
                and getattr(args, "recover_unexecuted", False)
            ):
                originals = {job["id"]: job for job in runnable}
                for result in completed:
                    if not result["cases"] or any(
                        c["status"] != "UNEXECUTED" for c in result["cases"]
                    ):
                        continue
                    if time.monotonic() >= deadline:
                        break
                    original = copy.deepcopy(result)
                    try:
                        retried = execute([originals[result["id"]]], slot, recovery=True)[0]
                    except Exception as error:
                        retried = {
                            "id": result["id"],
                            "batch_id": batch_id,
                            "profile": result["profile"],
                            "attempt_id": f"batch-{batch_id:03}-recovery-{result['id']:03}",
                            "failed": True,
                            "index_failed": True,
                            "outcome": "RECOVERY_ERROR",
                            "errors": [repr(error)],
                            "cases": [
                                {
                                    **case,
                                    "status": "RECOVERY_ERROR",
                                    "target_dispatches": None,
                                }
                                for case in result["cases"]
                            ],
                            "target_dispatches": None,
                            "returncode": None,
                            "batch_errors": [],
                            "process_diagnostics": {},
                        }
                    result.update(retried)
                    result["attempts"] = [original, copy.deepcopy(retried)]
                    result["failed"] = original["failed"] or retried["failed"]
                    if retried.get("outcome") == "PASS":
                        result["outcome"] = "PASS_RECOVERED"
            return completed

        def progress(value):
            for result in value["completed"]:
                job = manifest["jobs"][result["id"]]
                observed = {c["case_id"] for c in result["cases"]}
                result["cases"] += [c for c in skipped_cases(job) if c["case_id"] not in observed]
                if result.get("outcome") in (None, "PASS"):
                    result["outcome"] = index_outcome(result)
                result["batch_id"] = job["batch_id"]
                write_json(args.reports / f"solution-{job['id']:03}.result.json", result)
            write_json(args.reports / "progress.json", value)
            write_text(args.reports / "summary.md", render_report(manifest, value))
            for result in value["completed"]:
                row = solution_report(manifest["jobs"][result["id"]], result, args.backend)
                print(
                    f"{args.backend} solution {row[0]}: {row[1]}; numerical={row[2]}; dispatches={row[3]}; batch_seconds={row[4]}; {row[5]}",
                    flush=True,
                )

        start = time.monotonic()
        summary = run_queue(
            batches,
            args.workers,
            execute,
            progress,
            deadline=deadline,
            continue_on_failure=getattr(args, "continue_on_failure", False),
        )
        by_id = {j["id"]: j for j in manifest["jobs"]}
        counts = Counter()
        missing = []
        for result in summary["results"]:
            job = by_id[result["id"]]
            observed = {c["case_id"] for c in result["cases"]}
            # A worker exception can interrupt native setup/parsing. These cases
            # were still excluded from its input; retain their skip evidence.
            skipped = [c for c in skipped_cases(job) if c["case_id"] not in observed]
            if skipped:
                result["cases"] += skipped
                observed.update(c["case_id"] for c in skipped)
            missing += [
                {
                    "job": job["id"],
                    "index": job["solutions"][0]["index"],
                    "case_id": c["id"],
                    "shape": c["shape"],
                }
                for c in job["cases"]
                if c["id"] not in observed
            ]
            counts.update(c["status"] for c in result["cases"])
            result["batch_id"] = job["batch_id"]
            write_json(args.reports / f"solution-{job['id']:03}.result.json", result)
        unstarted = [
            {
                "job": j,
                "index": by_id[j]["solutions"][0]["index"],
                "case_id": c["id"],
                "shape": c["shape"],
            }
            for j in summary["unstarted"]
            for c in by_id[j]["cases"]
        ]
        summary.update(
            backend=args.backend,
            solution_index_range=manifest["solution_index_range"],
            artifact_fingerprint=manifest["artifact_fingerprint"],
            seconds=time.monotonic() - start,
            suite_seconds=time.monotonic() - suite_start,
            suite_timeout=args.suite_timeout,
            planned_cases=manifest["planned_cases"],
            counts=dict(counts),
            missing=missing,
            unstarted_cases=unstarted,
            failed_batch_ids=failed_batch_ids(summary["results"]),
            outcome_counts=dict(
                Counter(r.get("outcome") or index_outcome(r) for r in summary["results"])
            ),
            recovered_indices=[
                by_id[r["id"]]["solutions"][0]["index"]
                for r in summary["results"]
                if r.get("attempts")
            ],
        )
        processes = []
        for path in sorted(args.reports.glob("batch-*.result.json")):
            process = json.loads(path.read_text(encoding="utf-8"))
            if process["native_executed"]:
                processes.append(
                    {
                        "id": path.name.removesuffix(".result.json"),
                        "jobs": process["jobs"],
                        "profile": process["profile"],
                        "native_seconds": process["native_seconds"],
                        "failed": any(r["failed"] for r in process["results"]),
                        "diagnostics": next(
                            (
                                r.get("process_diagnostics", {})
                                for r in process["results"]
                                if "process_diagnostics" in r
                            ),
                            {},
                        ),
                    }
                )
        summary["processes"] = processes
        summary["failed_process_ids"] = [p["id"] for p in processes if p["failed"]]
        completed_by_id = {r["id"]: r for r in summary["results"]}
        summary["index_outcomes"] = []
        for job in manifest["jobs"]:
            result = completed_by_id.get(job["id"])
            if result is None:
                result = {
                    "id": job["id"],
                    "outcome": "UNSTARTED",
                    "cases": [
                        {"case_id": c["id"], "shape": c["shape"], "status": "UNSTARTED"}
                        for c in job["cases"]
                    ],
                }
            summary["index_outcomes"].append(
                {
                    **result,
                    "index": job["solutions"][0]["index"],
                    "profile": job.get("profile"),
                }
            )
        summary["outcome_counts"] = dict(
            Counter(r.get("outcome") or index_outcome(r) for r in summary["index_outcomes"])
        )
        summary["accounted_cases"] = sum(counts.values()) + len(missing) + len(unstarted)
        assert summary["accounted_cases"] == summary["planned_cases"]
        summary["passed"] = (
            summary["stop"] is None
            and not unstarted
            and not missing
            and all(not r["failed"] for r in summary["results"])
            and counts["PASSED"] + counts["SKIPPED"] == summary["planned_cases"]
        )
        summary["fully_validated"] = summary["passed"] and not counts["SKIPPED"]
        write_json(args.reports / "summary.json", summary)
        write_text(args.reports / "summary.md", render_report(manifest, summary))
        print(
            f"{counts['PASSED']}/{summary['planned_cases']} numerical passes; {counts['SKIPPED']} skipped, {len(missing)} missing, {len(unstarted)} unstarted. Reports: {args.reports}"
        )
        return 0 if summary["passed"] else 1
    except Exception as error:
        write_json(args.reports / "setup-or-run-error.json", {"error": repr(error)})
        report = args.reports / "summary.md"
        prior = (
            report.read_text(encoding="utf-8")
            if report.exists()
            else f"### {args.backend} solution-index race sweep\n"
        )
        write_text(
            report,
            prior + f"\nRun status: FAIL. Setup/run error: {html.escape(repr(error))}\n",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rocjitsu", "client", "library-dir", "reports"):
        parser.add_argument("--" + name, type=Path, required=True)
    configuration = parser.add_mutually_exclusive_group(required=True)
    configuration.add_argument("--config", type=Path, help="Single emulator profile")
    configuration.add_argument(
        "--hardware-profile",
        dest="hardware_profiles",
        type=Path,
        action="append",
        help="Complete emulator profile; repeat to select among profiles instead of --config",
    )
    parser.add_argument(
        "--recover-unexecuted",
        action="store_true",
        help="Replay wholly unexecuted indices once, individually, retaining original process failures",
    )
    parser.add_argument("--target", choices=("gfx942", "gfx950"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Maximum solution indices per native process; amortizes client startup and library loading "
            f"(default bench: {DEFAULT_BATCH_SIZES['bench']}; tensile: {DEFAULT_BATCH_SIZES['tensile']})"
        ),
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Finish the selected workload after failures; still exit unsuccessfully",
    )
    parser.add_argument(
        "--solution-index-range-begin",
        type=int,
        default=0,
        help="First original catalogue index to include (default: 0)",
    )
    parser.add_argument(
        "--solution-index-range-end",
        type=int,
        help="Exclusive original catalogue index limit (default: catalogue end)",
    )
    parser.add_argument("--backend", choices=("tensile", "bench"), required=True)
    parser.add_argument(
        "--policy",
        choices=(POLICY, REGULAR_POLICY),
        help="GEMM construction policy (bench: regular-v1; tensile: tile-depth-v1)",
    )
    parser.add_argument(
        "--known-failures",
        type=Path,
        default=None,
        help="Versioned list of kernel names and cases to skip across their solutions",
    )
    parser.add_argument(
        "--rocjitsu-revision",
        required=True,
        help="Full source revision of the detector; must match the list unless rechecking",
    )
    parser.add_argument(
        "--run-known-failures",
        action="store_true",
        help="Execute every selected case, ignoring skips and revision/policy guards",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Seconds per solution; multiplied by the number of runnable solutions in a batch",
    )
    parser.add_argument(
        "--suite-timeout",
        type=float,
        help="Optional overall time budget in seconds, including inventory preparation",
    )
    args = parser.parse_args()
    if args.policy is None:
        args.policy = REGULAR_POLICY if args.backend == "bench" else POLICY
    if args.known_failures is None:
        args.known_failures = Path(__file__).with_name(
            "known_failures_regular.yaml"
            if args.policy == REGULAR_POLICY
            else "known_failures.yaml"
        )
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZES[args.backend]
    if args.batch_size < 1 or (args.backend == "tensile" and args.batch_size != 1):
        parser.error("Batch size must be positive; tensile requires batch size 1")
    if args.policy == REGULAR_POLICY and args.backend != "bench":
        parser.error("regular-v1 requires the bench backend")
    if (
        not 1 <= args.workers <= 16
        or not 0 <= args.solution_index_range_begin < (1 << 31)
        or (
            args.solution_index_range_end is not None
            and not args.solution_index_range_begin < args.solution_index_range_end <= (1 << 31)
        )
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or (
            args.suite_timeout is not None
            and (not math.isfinite(args.suite_timeout) or args.suite_timeout <= 0)
        )
    ):
        parser.error(
            "Require 1–16 workers, 0 <= range begin < range end <= 2**31, "
            "and finite positive timeouts"
        )
    for key in ("rocjitsu", "client", "config", "library_dir", "reports"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
