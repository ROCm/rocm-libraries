#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run a range of packaged solution indices under rocJITsu."""

import argparse
import bisect
import copy
import csv
import functools
import html
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

from rocjitsu_known_failures import apply_known_failures
from rocjitsu_sweep_plan import (
    POLICY,
    REGULAR_POLICY,
    catalogue_end,
    derive_devices,
    make_batches,
    make_plan,
    sha256,
    validate_profile,
)
from tensilelite.Utilities.ClientConfig import problemTypeOptions

DEFAULT_BATCH_SIZES = {"bench": 32, "tensile": 1}
# Progress checkpoints are at least this many seconds apart, and further apart
# when writing one takes longer: at most 1/CHECKPOINT_COST_RATIO of the
# coordinator's time goes to rewriting them as results accumulate.
CHECKPOINT_SECONDS = 5
CHECKPOINT_COST_RATIO = 20
# Peak resident memory of one emulated native process with rocjitsu's race
# plugin, fitted to whole-catalogue gfx942 and gfx950 sweeps (12,196 processes):
# about 0.35 GiB plus 6.0 times the uncompressed size of the Tensile code
# objects it loads. Decompression, loader staging and the emulated device copy
# each hold the code. With this factor, the base covers every measured process
# with at least 0.1 GiB to spare.
MEMORY_BASE_GIB = 0.6
MEMORY_PER_CODE_OBJECT_GIB = 6.0


@functools.cache
def code_object_bytes(path):
    """Uncompressed size of a code object, reading a compressed bundle's header."""
    with open(path, "rb") as stream:
        header = stream.read(24)
    if header[:4] != b"CCOB":
        return os.path.getsize(path)
    version = int.from_bytes(header[4:6], "little")
    if version == 1:
        return int.from_bytes(header[8:12], "little")
    if version == 2:
        return int.from_bytes(header[12:16], "little")
    if version == 3:
        return int.from_bytes(header[16:24], "little")
    raise ValueError(f"{path}: unsupported compressed bundle version {version}")


def batch_memory_gib(batch):
    """Estimated peak memory of the native process that runs a batch."""
    paths = {path for job in batch for path in job.get("code_objects", ()) if path.endswith(".co")}
    return MEMORY_BASE_GIB + MEMORY_PER_CODE_OBJECT_GIB * sum(
        code_object_bytes(path) for path in paths
    ) / (1 << 30)


def checkpoint_after(started, finished):
    """When the next progress checkpoint is due, after one written in [started, finished]."""
    return finished + max(CHECKPOINT_SECONDS, CHECKPOINT_COST_RATIO * (finished - started))


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
    if result.get("attempts"):
        notes += ("; " if notes else "") + "Earlier attempts and failures retained"
    if result.get("latest_attempt_errors"):
        notes += "; latest infrastructure error: " + "; ".join(result["latest_attempt_errors"])
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
        notes or "Identity, numerical and race checks passed",
    ]


def merge_attempts(previous, current):
    """Retain history without letting a reporting/setup error erase qualified work."""
    history = copy.deepcopy(previous.get("attempts", [previous]))
    history.extend(copy.deepcopy(current.get("attempts", [current])))
    preserve = current.get("outcome") == "INFRASTRUCTURE_ERROR" and previous.get("outcome") in {
        "PASS",
        "PASS_RECOVERED",
        "PASS_RETAINED",
    }
    result = copy.deepcopy(previous if preserve else current)
    result.update(
        attempts=history,
        failed=previous["failed"] or current["failed"],
        current_attempt_failed=current["failed"],
    )
    if preserve:
        result.update(outcome="PASS_RETAINED", latest_attempt_errors=current["errors"])
    elif result.get("outcome") == "PASS":
        result["outcome"] = "PASS_RECOVERED"
    return result


def numerical_index_counts(results):
    counts = Counter()
    for result in results:
        statuses = {c["status"] for c in result["cases"]}
        status = (
            "FAILED"
            if "FAILED" in statuses
            else "PASSED" if statuses == {"PASSED"} else "INCOMPLETE"
        )
        counts[status] += 1
    return dict(counts)


def numerical_failure_indices(results):
    failures = set()
    for result in results:
        pending = [result]
        while pending:
            attempt = pending.pop()
            if attempt.get("attempts"):
                pending.extend(attempt["attempts"])
            elif any(c["status"] == "FAILED" for c in attempt["cases"]):
                failures.add(result["index"])
    return sorted(failures)


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
        f"policy: `{cell(manifest['policy'])}`.",
        *(
            [
                "Selected from an index list with SHA-256 "
                f"`{manifest['solution_index_list_sha256']}`."
            ]
            if "solution_index_list_sha256" in manifest
            else []
        ),
        *(
            [
                "Case parameters: "
                f"reduction blocks `{cell(manifest['case_parameters']['reduction_blocks'])}`, "
                f"alpha `{cell(manifest['case_parameters']['alpha'])}`, "
                f"beta `{cell(manifest['case_parameters']['beta'])}`."
            ]
            if manifest.get("case_parameters") is not None
            else []
        ),
        f"Artifact fingerprint: `{manifest['artifact_fingerprint']}`.",
        f"Selected solutions: {len(manifest['jobs'])}; eligible/inventory solutions: "
        f"{inventory.get('eligible_solutions', '?')}/{inventory.get('solutions', '?')}; "
        f"eligible/inventory kernel names: "
        f"{inventory.get('eligible_kernel_names', '?')}/{inventory.get('unique_kernel_names', '?')}.",
        "",
        "PASS requires all planned cases, matching identities/dispatches, no races or warnings, and completed native execution. "
        "SKIPPED cases did not run; PASS WITH SKIPS validates only the remaining cases.",
        "VALIDATED_IN_FAILED_PROCESS retains completed per-index checks alongside a process failure. "
        "UNEXECUTED and LAUNCHED_NO_RESULT distinguish absent launches from absent numerical results.",
        "Numerical totals describe the retained result for each index. Earlier numerical failures remain "
        "listed separately, including after a successful replay.",
        "Full shapes, identities, commands and logs are in this run's reports directory.",
        "",
        "| Solution index | Result | Numerical passes | Target dispatches | Notes |",
        "|---:|:---|---:|---:|:---|",
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
            row = [str(job["solutions"][0]["index"]), state, "—", "—", ""]
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
    if "numerical_index_counts" in summary:
        lines += [
            f"Indices: {len(manifest['jobs'])} selected; numerical validation "
            f"{summary['numerical_index_counts'].get('PASSED', 0)} passed, "
            f"{summary['numerical_index_counts'].get('FAILED', 0)} failed, "
            f"{summary['numerical_index_counts'].get('INCOMPLETE', 0)} incomplete.",
            f"Indices with a numerical failure in any retained attempt: "
            f"{len(summary.get('numerical_failure_indices', []))}.",
            "",
        ]
    if summary.get("stop"):
        lines += [f"Stopped: {cell(summary['stop']['reason'])}.", ""]
    return "\n".join(lines)


def prepare(args):
    begin = time.monotonic()
    explicit_profiles = getattr(args, "hardware_profiles", None)
    template = getattr(args, "hardware_profile_template", None)
    if template is not None:
        template_config = json.loads(template.read_text(encoding="utf-8"))
        validate_profile(template_config, args.target)
        devices = derive_devices(
            args.library_dir, args.target, template_config["vm"]["gpu"]["device"], args.workers
        )
        sources = [(template, device) for device in devices]
    else:
        sources = [(path, None) for path in explicit_profiles or [args.config]]
    configs, profiles = {}, {}
    for number, (path, device) in enumerate(sources):
        named = explicit_profiles or template is not None
        name = f"profile-{number:03}" if named else "default"
        base = json.loads(path.read_text(encoding="utf-8"))
        if device is not None:
            base["vm"]["gpu"]["device"] = device
        if named:
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
        config = args.reports / (f"{name}.json" if named else "rocjitsu.json")
        write_json(config, base)
        configs[name] = config
        profiles[name] = {
            "source": str(path),
            # A derived profile is identified by the configuration written from the template.
            "sha256": sha256(config if device is not None else path),
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
        reduction_blocks=getattr(args, "reduction_blocks", 8),
        alpha=getattr(args, "alpha", 1),
        beta=getattr(args, "beta", 0),
        workers=args.workers,
        indices=getattr(args, "solution_indices", None),
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


def finish_result(
    job,
    text,
    returncode,
    result,
    errors,
    launches_per_case,
    counted_statuses=frozenset({"PASSED"}),
):
    dispatches = Counter(re.findall(r'^\[rocjitsu\] Kernel dispatch: "([^"]+)"', text, re.M))
    kernel = job["solutions"][0]["kernel"]
    wanted = sum(c["status"] in counted_statuses for c in result["cases"]) * launches_per_case
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
    verdict = None
    explicit_verdicts = "[hipblaslt-case-v1]" in text
    for line in text.splitlines():
        launches += line == target_dispatch_line(job)
        validation = re.fullmatch(r"\[hipblaslt-case-v1\] RESULT (-?\d+) (\w+)", line)
        if validation:
            verdict = (int(validation[1]), validation[2])
            continue
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
                pending["native_verdict"] = verdict
                verdict = None
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
            native = row["native_verdict"]
            if explicit_verdicts and (
                native is None
                or native[0] != solution["index"]
                or native[1] not in {"PASSED", "FAILED", "NOT_CHECKED"}
            ):
                raise ValueError(f"Missing or invalid native numerical verdict: {shape}")
            # The native norm criterion determines pass/fail. On failure, bench
            # prints "failed" for allclose atol/rtol instead of floating values.
            # Keep those measurements verbatim; never infer a pass from them.
            measurements = {}
            finite = True
            for key in ("norm_error", "atol", "rtol"):
                raw = row[key]
                if raw == "failed" and key != "norm_error" and native and native[1] == "FAILED":
                    measurements[key] = raw
                    finite = False
                    continue
                value = float(raw)
                valid = math.isfinite(value) and value >= 0
                measurements[key] = value if valid else raw
                finite &= valid
            status = (
                "NOT_VALIDATED"
                if native and native[1] == "NOT_CHECKED"
                else ("FAILED" if not finite or (native and native[1] == "FAILED") else "PASSED")
            )
            cases[shape] = {
                "case_id": expected[shape]["id"],
                "index": solution["index"],
                "shape": shape,
                "status": status,
                "target_dispatches": row["target_dispatches"],
                "numerical": measurements,
                "solution_name": row["Solution name"],
                "kernel": row["kernel name"],
            }
            if row["target_dispatches"] != job.get("target_dispatches_per_case", 2):
                errors.append(
                    f"Case {shape}: unexpected target dispatches ({row['target_dispatches']})"
                )
            if status != "PASSED":
                errors.append(f"Native numerical validation {status}: {shape}")
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
        counted_statuses=frozenset({"PASSED", "FAILED", "NOT_VALIDATED"}),
    )


def classify_bench_records(jobs, text, returncode):
    """Use native case boundaries to attribute results even after a failed case."""
    planned = [(j, c) for j in jobs for c in j["cases"]]
    begun, active, done = {}, None, None
    protocol_errors = []
    target_kernels = {j["solutions"][0]["kernel"] for j in jobs}
    by_case = {(j["id"], c["id"]): j for j, c in planned}
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("[hipblaslt-case-v1]"):
            fields = line.split()
            try:
                event = fields[1]
                if event == "BEGIN":
                    if active is not None or done is not None or len(fields) != 8:
                        raise ValueError("Unexpected case begin")
                    ordinal, index, *shape = map(int, fields[2:])
                    if ordinal != len(begun) or ordinal >= len(planned):
                        raise ValueError("Case ordinal is not the next selected input")
                    job, case = planned[ordinal]
                    if index != job["solutions"][0]["index"] or shape != list(case["shape"]):
                        raise ValueError("Case begin does not match the selected index/shape")
                    active = (job["id"], case["id"])
                    begun[active] = {
                        "start": offset,
                        "ordinal": ordinal,
                        "index": index,
                        "result_count": 0,
                    }
                elif event == "END":
                    if active is None or len(fields) != 5:
                        raise ValueError("Unexpected case end")
                    record = begun[active]
                    if (int(fields[2]), int(fields[3])) != (
                        record["ordinal"],
                        record["index"],
                    ):
                        raise ValueError("Case end has the wrong identity")
                    if fields[4] not in {
                        "PASSED",
                        "FAILED",
                        "NOT_CHECKED",
                        "INCOMPLETE",
                    }:
                        raise ValueError("Unknown case status")
                    record.update(end=offset + len(line), status=fields[4])
                    active = None
                    if fields[4] != "INCOMPLETE" and record.get("result_status") != fields[4]:
                        record["conflicting_verdict"] = True
                        raise ValueError("Case end disagrees with its numerical result")
                elif event == "RESULT":
                    if active is None or len(fields) != 4:
                        raise ValueError("Numerical result outside a case")
                    record = begun[active]
                    record["result_count"] += 1
                    if (
                        int(fields[2]) != record["index"]
                        or fields[3] not in {"PASSED", "FAILED", "NOT_CHECKED"}
                        or record["result_count"] != 1
                    ):
                        record["conflicting_verdict"] = True
                        raise ValueError("Invalid or duplicate numerical result")
                    record["result_status"] = fields[3]
                elif event == "DONE":
                    if active is not None or done is not None or len(fields) != 4:
                        raise ValueError("Unexpected native completion record")
                    done = tuple(map(int, fields[2:]))
                    failures = sum(
                        r.get("status") in {"FAILED", "INCOMPLETE"} for r in begun.values()
                    )
                    if done != (len(begun), failures):
                        raise ValueError("Native completion counts do not match case records")
                else:
                    raise ValueError("Unknown native case event")
            except (ValueError, IndexError) as error:
                protocol_errors.append(str(error))
        dispatch = re.match(r'^\[rocjitsu\] Kernel dispatch: "([^"]+)"', line)
        if dispatch and dispatch[1] in target_kernels:
            if active is None:
                protocol_errors.append("Selected target dispatch outside a case")
            elif line.rstrip("\r\n") != target_dispatch_line(by_case[active]):
                protocol_errors.append("Selected target dispatch belongs to another case")
        offset += len(line)
    if active is not None:
        begun[active]["end"] = len(text)
    prefix_valid = not protocol_errors
    batch_errors = protocol_errors.copy()
    if done is None:
        batch_errors.append("Missing native completion record")
    expected_validation_exit = (
        prefix_valid
        and done is not None
        and done[0] == len(planned)
        and done[1] > 0
        and all(r.get("status") != "INCOMPLETE" for r in begun.values())
        and returncode == 1
    )
    if returncode and not expected_validation_exit:
        batch_errors.append(f"Native process exited {returncode}")
    if done is not None and len(begun) != len(planned):
        batch_errors.append("Native process did not start every selected input")
    if done is not None and done[1] and not returncode:
        batch_errors.append("Native process returned success despite failed cases")
    races = re.findall(r"^RACE .*", text, re.M)
    warnings = Counter(line for line in text.splitlines() if "[rj warn]" in line)
    if races:
        batch_errors.append(f"Process reported {len(races)} races (none suppressed)")
    if warnings:
        batch_errors.append(f"Process reported {sum(warnings.values())} emulator warnings")
    kernel_owners = Counter(j["solutions"][0]["kernel"] for j in jobs)
    results = []
    for job in jobs:
        parsed_cases, errors, records = [], [], []
        for case in job["cases"]:
            record = begun.get((job["id"], case["id"]))
            if record is None:
                continue
            span = text[record["start"] : record.get("end", len(text))]
            cleaned = "\n".join(
                line
                for line in span.splitlines()
                if not line.startswith("RACE ") and "[rj warn]" not in line
            )
            records.append(cleaned)
            parsed = classify_bench({**job, "cases": [case]}, cleaned, 0)
            parsed_cases.extend(parsed["cases"])
            errors.extend(parsed["errors"])
        result = finish_result(
            job,
            "\n".join(records),
            0,
            {"id": job["id"], "cases": parsed_cases},
            errors,
            job.get("target_dispatches_per_case", 2),
            counted_statuses=frozenset({"PASSED", "FAILED", "NOT_VALIDATED"}),
        )
        observed = {c["case_id"]: c for c in result["cases"]}
        dispatches = 0
        for case in job["cases"]:
            record = begun.get((job["id"], case["id"]))
            span = text[record["start"] : record.get("end", len(text))] if record else ""
            count = sum(line == target_dispatch_line(job) for line in span.splitlines())
            dispatches += count
            if case["id"] in observed:
                actual = observed[case["id"]]
                actual.update(native_started=True, native_completed="status" in record)
                if record.get("conflicting_verdict") or record.get("status") != actual["status"]:
                    actual["reported_numerical_status"] = actual["status"]
                    actual["status"] = "INCOMPLETE"
                    result["errors"].append("Case result lacks a matching completion status")
                continue
            status = (
                "UNEXECUTED"
                if record is None and prefix_valid
                else (
                    "MISSING_RECORD"
                    if not prefix_valid
                    else (
                        "LAUNCHED_NO_RESULT"
                        if count
                        else (
                            "NATIVE_REJECTED"
                            if "error: NO solution found!" in span
                            else "INCOMPLETE"
                        )
                    )
                )
            )
            result["cases"].append(
                {
                    "case_id": case["id"],
                    "index": job["solutions"][0]["index"],
                    "shape": case["shape"],
                    "status": status,
                    "target_dispatches": count,
                    "native_started": record is not None,
                    "native_completed": record is not None and "status" in record,
                }
            )
        kernel = job["solutions"][0]["kernel"]
        result["race_headers"] = [
            r
            for r in races
            if kernel_owners[kernel] == 1
            and re.search(r"(?:^| )kernel=" + re.escape(kernel) + r"(?: |$)", r)
        ]
        if result["race_headers"]:
            result["errors"].append(f"{len(result['race_headers'])} races naming this kernel")
        result.update(
            target_dispatches=dispatches if prefix_valid else None,
            index_failed=bool(result["errors"])
            and any(c["status"] != "UNEXECUTED" for c in result["cases"]),
            batch_errors=batch_errors.copy(),
            returncode=returncode,
            process_diagnostics={
                "race_headers": races,
                "warnings": dict(warnings),
                "completion": done,
            },
        )
        result["failed"] = result["index_failed"] or bool(batch_errors)
        result["outcome"] = index_outcome(result)
        results.append(result)
    return results


def classify_bench_batch(jobs, text, returncode):
    """Keep record identities, launch evidence and process diagnostics separate."""
    if "[hipblaslt-case-v1]" in text:
        return classify_bench_records(jobs, text, returncode)
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
        "INCOMPLETE",
        "NOT_VALIDATED",
        "INFRASTRUCTURE_ERROR",
    ):
        if status in statuses:
            return status
    if "FAILED" in statuses:
        return "NUMERICAL_FAILED"
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


# Native process groups run in their own sessions, so a signal to the runner
# does not reach them. The runner's stop handler kills the groups listed here.
_native_groups = set()


def stop_native_processes(signum, frame):
    """Kill running native process groups and exit; retained results allow resuming."""
    for group in list(_native_groups):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os._exit(128 + signum)


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
        _native_groups.add(process.pid)
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
        finally:
            _native_groups.discard(process.pid)


def run_queue(
    batches,
    workers,
    execute,
    progress,
    deadline=None,
    continue_on_failure=False,
    recover=None,
    memory=None,
):
    """The calling thread owns scheduling, results, retries and progress writes.

    Each worker receives a private batch and transfers its completed results
    through a Queue. After that transfer it never accesses those results again.

    ``memory`` is an optional ``(estimate, budget)`` pair. A batch starts only
    while the estimates of running batches, including its own, fit the budget;
    the first pending batch that fits is chosen. A batch always starts when
    nothing else is running, so an oversized batch runs alone.
    """
    pending = deque(copy.deepcopy(batches))
    original_ids = [job["id"] for batch in batches for job in batch]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError("A selected job appears in more than one initial batch")
    work = [queue.Queue() for _ in range(workers)]
    finished = queue.Queue()
    results, active = {}, {}
    reserved = {}
    costs = {}
    assigned = []
    assigned_ids = set()
    maximum = 0
    maximum_reserved = 0.0
    stop_snapshot = None

    def stop(reason, trigger_job=None):
        nonlocal stop_snapshot
        if stop_snapshot is None:
            stop_snapshot = {
                "reason": reason,
                "trigger_job": trigger_job,
                "assigned_jobs": len(assigned),
                "assigned_job_ids": assigned.copy(),
                "in_flight": len(active),
            }

    def expired():
        return deadline is not None and time.monotonic() >= deadline

    def cost_of(batch):
        key = tuple(job["id"] for job in batch)
        if key not in costs:
            costs[key] = memory[0](batch)
        return costs[key]

    def worker(slot):
        while (batch := work[slot].get()) is not None:
            try:
                completed = execute(batch, slot)
                if sorted(r["id"] for r in completed) != sorted(j["id"] for j in batch):
                    raise ValueError("Worker did not return one result per assigned job")
            # SystemExit in a worker must still produce a completion for the coordinator.
            except BaseException as error:
                completed = [
                    {
                        "id": job["id"],
                        "failed": True,
                        "outcome": "INFRASTRUCTURE_ERROR",
                        "errors": [repr(error)],
                        "cases": [
                            {
                                "case_id": c["id"],
                                "shape": c["shape"],
                                "status": "INFRASTRUCTURE_ERROR",
                                "target_dispatches": None,
                            }
                            for c in job.get("cases", [])
                            if not c.get("skip")
                        ],
                    }
                    for job in batch
                ]
            finished.put((slot, completed))

    def dispatch():
        nonlocal maximum, maximum_reserved
        for slot in range(workers):
            if not pending or stop_snapshot is not None:
                break
            if slot in active:
                continue
            if expired():
                stop("suite-timeout")
                break
            position, cost = 0, 0.0
            if memory is not None:
                budget = memory[1]
                in_use = sum(reserved.values())
                position = next(
                    (
                        i
                        for i, candidate in enumerate(pending)
                        if not active or in_use + cost_of(candidate) <= budget
                    ),
                    None,
                )
                if position is None:
                    break
                cost = cost_of(pending[position])
            batch = pending[position]
            del pending[position]
            for job in batch:
                if job["id"] not in assigned_ids:
                    assigned.append(job["id"])
                    assigned_ids.add(job["id"])
            active[slot] = batch
            reserved[slot] = cost
            maximum = max(maximum, len(active))
            maximum_reserved = max(maximum_reserved, sum(reserved.values()))
            work[slot].put(batch)

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(workers)]
    for thread in threads:
        thread.start()
    try:
        dispatch()
        while active:
            slot, completed = finished.get()
            batch = active.pop(slot)
            reserved.pop(slot, None)
            recovery = recover(batch, completed) if recover and not expired() else []
            for result in completed:
                previous = results.get(result["id"])
                if previous is not None:
                    result = merge_attempts(previous, result)
                results[result["id"]] = result
            completed = [results[r["id"]] for r in completed]
            if expired():
                stop("suite-timeout", batch[0]["id"])
            elif not continue_on_failure and any(r["failed"] for r in completed):
                stop("job-failure", next(r["id"] for r in completed if r["failed"]))
            if stop_snapshot is None:
                pending.extend(recovery)
            try:
                progress(
                    {
                        "results": list(results.values()),
                        "completed": completed,
                        "assigned_jobs": len(assigned),
                        "assigned_job_ids": assigned.copy(),
                        "stop": stop_snapshot,
                    }
                )
            except Exception as error:
                for result in completed:
                    message = f"Cannot write progress: {error!r}"
                    result["failed"] = True
                    result.setdefault("errors", []).append(message)
                    if result.get("outcome") in {
                        "PASS",
                        "PASS_RECOVERED",
                        "PASS_RETAINED",
                    }:
                        result["outcome"] = "PASS_RETAINED"
                        result["latest_attempt_errors"] = [message]
                    elif result.get("outcome") is None:
                        result["outcome"] = "INFRASTRUCTURE_ERROR"
                stop("progress-error", batch[0]["id"])
            dispatch()
    finally:
        for channel in work:
            channel.put(None)
        for thread in threads:
            thread.join()
    return {
        "results": sorted(results.values(), key=lambda r: r["id"]),
        "unstarted": [job_id for job_id in original_ids if job_id not in assigned_ids],
        "assigned_job_ids": assigned,
        "max_active": maximum,
        "max_reserved_memory_gib": round(maximum_reserved, 3),
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


def resume_results(directory, manifest):
    previous = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for key in (
        "artifact_fingerprint",
        "target",
        "policy",
        "backend",
        "tools",
        "rocjitsu_revision",
        "known_failures",
        "solution_index_range",
    ):
        if previous[key] != manifest[key]:
            raise ValueError(f"Cannot resume: {key} changed")
    for key in ("selection", "solution_index_list_sha256"):
        if previous.get(key) != manifest.get(key):
            raise ValueError(f"Cannot resume: {key} changed")
    prior_parameters = previous.get("case_parameters")
    if "case_parameters" not in previous and previous["policy"] == REGULAR_POLICY:
        prior_parameters = {"reduction_blocks": 8, "alpha": 1, "beta": 0}
    if prior_parameters != manifest["case_parameters"]:
        raise ValueError("Cannot resume: case_parameters changed")
    profile_hashes = lambda m: {k: p["sha256"] for k, p in m["profiles"].items()}
    if profile_hashes(previous) != profile_hashes(manifest):
        raise ValueError("Cannot resume: hardware profiles changed")
    jobs = lambda m: [{k: v for k, v in j.items() if k != "batch_id"} for j in m["jobs"]]
    if jobs(previous) != jobs(manifest):
        raise ValueError("Cannot resume: selected jobs or inputs changed")

    evidence, files = {}, {}
    failed_processes = []
    for path in sorted(directory.glob("prior-solution-*.result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        files[path.name] = sha256(path)
        evidence[result["id"]] = [result]
    for path in sorted(
        directory.glob("batch-*.result.json"), key=lambda p: ("-recovery-" in p.name, p.name)
    ):
        process = json.loads(path.read_text(encoding="utf-8"))
        files[path.name] = sha256(path)
        if any(r["failed"] for r in process["results"]):
            failed_processes.append(path.name.removesuffix(".result.json"))
        for result in process["results"]:
            evidence.setdefault(result["id"], []).append(result)
    for path in sorted(directory.glob("solution-*.result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        files[path.name] = sha256(path)
        evidence[result["id"]] = [result]
    by_id = {j["id"]: j for j in manifest["jobs"]}
    if evidence.keys() - by_id.keys():
        raise ValueError("Cannot resume: evidence contains unknown jobs")
    carried, retry = [], {}
    for job in manifest["jobs"]:
        attempts = evidence.get(job["id"], [])
        if not attempts:
            continue
        result = copy.deepcopy(attempts[0])
        for attempt in attempts[1:]:
            result = merge_attempts(result, attempt)
        result["resumed_from"] = str(directory)
        observed = {c["case_id"] for c in result["cases"]}
        expected = {c["id"] for c in job["cases"]}
        if observed - expected or len(observed) != len(result["cases"]):
            raise ValueError(f"Cannot resume: invalid case evidence for job {job['id']}")
        shapes = {c["id"]: c["shape"] for c in job["cases"]}
        if any(c["shape"] != shapes[c["case_id"]] for c in result["cases"]):
            raise ValueError(f"Cannot resume: case shapes changed for job {job['id']}")
        result["cases"] += [c for c in skipped_cases(job) if c["case_id"] not in observed]
        observed = {c["case_id"] for c in result["cases"]}
        incomplete = (
            observed != expected
            or result.get("returncode", 0) == 124
            or (result.get("returncode") is not None and result["returncode"] < 0)
            or any(
                c["status"] in ("UNEXECUTED", "UNSTARTED", "RECOVERY_ERROR")
                for c in result["cases"]
            )
        )
        if incomplete:
            retry[job["id"]] = result
        else:
            result["outcome"] = result.get("outcome") or index_outcome(result)
            carried.append(result)
    manifest["resume"] = {
        "reports": str(directory),
        "manifest_sha256": sha256(directory / "manifest.json"),
        "result_sha256": files,
        "carried_job_ids": [r["id"] for r in carried],
        "retry_job_ids": sorted(retry),
        "prior_failed_process_ids": failed_processes,
    }
    return carried, retry


def run(args):
    suite_start = time.monotonic()
    deadline = math.inf if args.suite_timeout is None else suite_start + args.suite_timeout
    continuation = getattr(args, "continue_on_failure", False) or getattr(
        args, "recover_unexecuted", False
    )
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
        carried, retry = [], {}
        if getattr(args, "resume_from", None) is not None:
            carried, retry = resume_results(args.resume_from, manifest)
        carried_ids = {r["id"] for r in carried}
        execution_batches = [[j for j in batch if j["id"] not in carried_ids] for batch in batches]
        execution_batches = [batch for batch in execution_batches if batch]
        write_json(args.reports / "manifest.json", manifest)
        for result in carried:
            write_json(args.reports / f"solution-{result['id']:03}.result.json", result)
        for job_id, result in retry.items():
            write_json(args.reports / f"prior-solution-{job_id:03}.result.json", result)
        print(
            f"{args.backend}: selected {len(manifest['jobs'])} solutions, "
            f"{manifest['planned_cases']} cases; preparation {manifest['preparation_seconds']:.2f}s. "
            f"Reports: {args.reports}",
            flush=True,
        )
        write_text(args.reports / "summary.md", render_report(manifest, {"results": carried}))
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
                "continue_on_failure": continuation,
                "recover_unexecuted": getattr(args, "recover_unexecuted", False),
                "library_dir": manifest["library_dir"],
                "memory_budget_gib": getattr(args, "memory_budget", None),
                "memory_model_gib": {
                    "base": MEMORY_BASE_GIB,
                    "per_uncompressed_code_object": MEMORY_PER_CODE_OBJECT_GIB,
                },
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

        def execute(batch, slot):
            begin = time.monotonic()
            batch_id = batch[0]["batch_id"]
            stem = args.reports / batch[0].get("attempt_id", f"batch-{batch_id:03}")
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
                    if continuation:
                        client_args.insert(0, "--continue-on-failure")
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
            return completed

        recovery_attempts = Counter()

        def recover(batch, completed):
            remaining = {
                r["id"]
                for r in completed
                if r["cases"] and all(c["status"] == "UNEXECUTED" for c in r["cases"])
            }
            if not remaining:
                return []
            untouched = [copy.deepcopy(j) for j in batch if j["id"] in remaining]
            no_progress = len(untouched) == len(batch)
            retries = batch[0].get("no_progress_retries", 0) + 1 if no_progress else 0
            if retries > 1:
                return []
            batch_id = batch[0]["batch_id"]
            recovery_attempts[batch_id] += 1
            attempt = f"batch-{batch_id:03}-recovery-{recovery_attempts[batch_id]:03}"
            for job in untouched:
                job.update(attempt_id=attempt, no_progress_retries=retries)
            return [untouched]

        next_checkpoint = 0

        def retain_attempt(result):
            original = retry.pop(result["id"], None)
            if original is not None:
                merged = merge_attempts(original, result)
                result.clear()
                result.update(merged)

        def progress(value):
            nonlocal next_checkpoint
            for result in value["completed"]:
                job = manifest["jobs"][result["id"]]
                observed = {c["case_id"] for c in result["cases"]}
                result["cases"] += [c for c in skipped_cases(job) if c["case_id"] not in observed]
                if result.get("outcome") in (None, "PASS"):
                    result["outcome"] = index_outcome(result)
                retain_attempt(result)
                result["batch_id"] = job["batch_id"]
                write_json(args.reports / f"solution-{job['id']:03}.result.json", result)
            if time.monotonic() >= next_checkpoint or value["stop"]:
                started = time.monotonic()
                checkpoint = {
                    **value,
                    "results": sorted(carried + value["results"], key=lambda r: r["id"]),
                }
                write_json(args.reports / "progress.json", checkpoint)
                write_text(args.reports / "summary.md", render_report(manifest, checkpoint))
                next_checkpoint = checkpoint_after(started, time.monotonic())
            for result in value["completed"]:
                row = solution_report(manifest["jobs"][result["id"]], result, args.backend)
                print(
                    f"{args.backend} solution {row[0]}: {row[1]}; numerical={row[2]}; dispatches={row[3]}; {row[4]}",
                    flush=True,
                )

        start = time.monotonic()
        summary = run_queue(
            execution_batches,
            args.workers,
            execute,
            progress,
            deadline=deadline,
            continue_on_failure=continuation,
            recover=recover if args.backend == "bench" and continuation else None,
            memory=(
                None
                if getattr(args, "memory_budget", None) is None
                else (batch_memory_gib, args.memory_budget)
            ),
        )
        if not execution_batches and time.monotonic() >= deadline:
            summary["stop"] = {
                "reason": "suite-timeout",
                "trigger_job": None,
                "assigned_jobs": 0,
                "assigned_job_ids": [],
                "in_flight": 0,
            }
        summary["results"] = sorted(carried + summary["results"], key=lambda r: r["id"])
        by_id = {j["id"]: j for j in manifest["jobs"]}
        counts = Counter()
        missing = []
        for result in summary["results"]:
            retain_attempt(result)
            job = by_id[result["id"]]
            observed = {c["case_id"] for c in result["cases"]}
            if len(observed) != len(result["cases"]) or observed - {c["id"] for c in job["cases"]}:
                raise ValueError(f"Invalid case accounting for job {job['id']}")
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
            if not result.get("resumed_from"):
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
            solution_index_list_sha256=manifest.get("solution_index_list_sha256"),
            artifact_fingerprint=manifest["artifact_fingerprint"],
            seconds=time.monotonic() - start,
            suite_seconds=time.monotonic() - suite_start,
            suite_timeout=args.suite_timeout,
            planned_cases=manifest["planned_cases"],
            counts=dict(counts),
            missing=missing,
            unstarted_cases=unstarted,
            outcome_counts=dict(
                Counter(r.get("outcome") or index_outcome(r) for r in summary["results"])
            ),
            recovered_indices=[
                by_id[r["id"]]["solutions"][0]["index"]
                for r in summary["results"]
                if r.get("attempts")
            ],
            resume=manifest.get("resume"),
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
        summary["failed_batch_ids"] = sorted(
            {int(p["id"].split("-")[1]) for p in processes if p["failed"]}
        )
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
            else:
                result = copy.deepcopy(result)
                observed = {c["case_id"] for c in result["cases"]}
                result["cases"] += [
                    {
                        "case_id": c["id"],
                        "shape": c["shape"],
                        "status": "MISSING_RECORD",
                    }
                    for c in job["cases"]
                    if c["id"] not in observed
                ]
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
        summary["numerical_index_counts"] = numerical_index_counts(summary["index_outcomes"])
        summary["numerical_failure_indices"] = numerical_failure_indices(summary["index_outcomes"])
        assert sum(summary["outcome_counts"].values()) == len(manifest["jobs"])
        assert sum(summary["numerical_index_counts"].values()) == len(manifest["jobs"])
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
            f"Indices: {len(manifest['jobs'])} selected; numerical validation "
            f"{summary['numerical_index_counts'].get('PASSED', 0)} passed, "
            f"{summary['numerical_index_counts'].get('FAILED', 0)} failed, "
            f"{summary['numerical_index_counts'].get('INCOMPLETE', 0)} incomplete; "
            f"{len(summary['numerical_failure_indices'])} with failures in retained history. Reports: {args.reports}"
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


def chunk_complete(directory):
    """Whether a chunk's run finished its whole selection, so a campaign reuses it."""
    path = directory / "summary.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8"))["stop"] is None


def run_campaign(args):
    """Run the selection as consecutive chunks of original indices, then summarize them.

    Each chunk is an ordinary run in its own report directory under --reports.
    Rerunning the same command reuses finished chunks and resumes an unfinished
    one from its retained results, which move to a numbered sibling directory.
    A chunk that stops early (a failure without --continue-on-failure, or the
    suite timeout) ends the campaign.
    """
    root = args.reports
    root.mkdir(parents=True, exist_ok=True)
    begin = args.solution_index_range_begin
    end = args.solution_index_range_end
    if end is None:
        end = catalogue_end(args.library_dir, args.target, args.workers)
    listed = None if args.solution_indices is None else sorted(set(args.solution_indices))
    deadline = math.inf if args.suite_timeout is None else time.monotonic() + args.suite_timeout
    ranges = [(b, min(b + args.chunk_size, end)) for b in range(begin, end, args.chunk_size)]
    if listed is not None:
        # A chunk without listed indices has nothing to select.
        ranges = [
            (b, e)
            for b, e in ranges
            if bisect.bisect_left(listed, b) < len(listed)
            and listed[bisect.bisect_left(listed, b)] < e
        ]
    status, chunks = 0, []
    for chunk_begin, chunk_end in ranges:
        directory = root / f"chunk-{chunk_begin:07}-{chunk_end:07}"
        if not chunk_complete(directory):
            if time.monotonic() >= deadline:
                status = 1
                break
            chunk = copy.copy(args)
            chunk.chunk_size = None
            chunk.reports = directory
            chunk.solution_index_range_begin = chunk_begin
            chunk.solution_index_range_end = chunk_end
            chunk.resume_from = None
            if args.suite_timeout is not None:
                chunk.suite_timeout = deadline - time.monotonic()
            if directory.exists():
                attempt = 1
                while directory.with_name(f"{directory.name}.attempt-{attempt}").exists():
                    attempt += 1
                previous = directory.rename(
                    directory.with_name(f"{directory.name}.attempt-{attempt}")
                )
                if (previous / "manifest.json").is_file():
                    chunk.resume_from = previous
            print(f"campaign: running {directory.name}", flush=True)
            run(chunk)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        outcomes = summary["index_outcomes"]
        chunks.append(
            {
                "name": directory.name,
                "begin": chunk_begin,
                "end": chunk_end,
                "passed": summary["passed"],
                "stop": summary["stop"],
                "outcome_counts": summary["outcome_counts"],
                "not_passed": [
                    {"index": r["index"], "outcome": r["outcome"]}
                    for r in outcomes
                    if not r["outcome"].startswith("PASS")
                ],
            }
        )
        del summary, outcomes
        if chunks[-1]["stop"] is not None:
            status = 1
            break
        status |= 0 if chunks[-1]["passed"] else 1
    totals = Counter()
    for chunk in chunks:
        totals.update(chunk["outcome_counts"])
    complete = len(chunks) == len(ranges) and all(c["stop"] is None for c in chunks)
    campaign = {
        "target": args.target,
        "solution_index_range": {"begin": begin, "end": end},
        "chunk_size": args.chunk_size,
        "complete": complete,
        "passed": status == 0,
        "outcome_counts": dict(totals),
        "chunks": chunks,
    }
    write_json(root / "campaign.json", campaign)
    attention = [
        r
        for chunk in chunks
        for r in chunk["not_passed"]
        if r["outcome"] not in ("UNSUPPORTED_FEATURE", "EXCLUDED_HARDWARE")
    ]
    lines = [
        f"### Solution-index campaign: {args.target}, indices {begin} to {end} (exclusive)",
        "",
        f"Status: {'PASS' if status == 0 else 'FAIL'}; "
        f"{'all chunks ran' if complete else 'incomplete; rerun the same command to continue'}.",
        "Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(totals.items())) + ".",
        "",
        "| Chunk | Result | Outcomes |",
        "|:---|:---|:---|",
        *(
            f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | "
            + ", ".join(f"{k} {v}" for k, v in sorted(c["outcome_counts"].items()))
            + " |"
            for c in chunks
        ),
        "",
        f"Indices that neither passed nor were outside the adapter scope: {len(attention)}.",
        "",
        *(f"- {r['index']}: {r['outcome']}" for r in attention),
        "",
    ]
    write_text(root / "campaign.md", "\n".join(lines))
    print(
        f"campaign: {len(chunks)} chunks; "
        + ", ".join(f"{k} {v}" for k, v in sorted(totals.items()))
        + f". Summary: {root / 'campaign.md'}",
        flush=True,
    )
    return status


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
    configuration.add_argument(
        "--hardware-profile-template",
        type=Path,
        help=(
            "Complete emulator profile from which to derive one profile per PCI chip ID and "
            "CU count that the catalogue's hardware predicates need"
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Continue the identical selection from retained reports, preserving prior failures; use a new --reports directory",
    )
    parser.add_argument(
        "--recover-unexecuted",
        action="store_true",
        help="Alias for --continue-on-failure, including batched recovery of unexecuted indices",
    )
    parser.add_argument("--target", choices=("gfx942", "gfx950"), required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent native processes, each on its own available physical CPU core (default: 4)",
    )
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
    parser.add_argument(
        "--solution-indices",
        type=Path,
        help=(
            "File of original catalogue indices, one per line ('#' starts a comment); "
            "selects only these within the index range"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help=(
            "Run the selection as consecutive chunks of this many original indices, each "
            "in its own directory under --reports; rerunning continues an interrupted campaign"
        ),
    )
    parser.add_argument("--backend", choices=("tensile", "bench"), required=True)
    parser.add_argument(
        "--policy",
        choices=(POLICY, REGULAR_POLICY),
        help="GEMM construction policy (bench: regular-v1; tensile: tile-depth-v1)",
    )
    parser.add_argument(
        "--reduction-blocks",
        type=int,
        default=8,
        help="Preferred full DepthU blocks per split for regular-v1 (default: 8)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="Real alpha scalar for regular-v1; complex imaginary part remains zero (default: 1)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0,
        help="Real beta scalar for regular-v1; complex imaginary part remains zero (default: 0)",
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
    parser.add_argument(
        "--memory-budget",
        type=float,
        help=(
            "Optional GiB limit on the estimated memory of concurrently running native "
            "processes; batches wait until their estimate fits"
        ),
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
    if args.policy != REGULAR_POLICY and (args.reduction_blocks, args.alpha, args.beta) != (
        8,
        1,
        0,
    ):
        parser.error("Reduction blocks and scalar overrides require regular-v1")
    if (
        args.workers < 1
        or not 0 <= args.solution_index_range_begin < (1 << 31)
        or (
            args.solution_index_range_end is not None
            and not args.solution_index_range_begin < args.solution_index_range_end <= (1 << 31)
        )
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or args.reduction_blocks <= 0
        or not math.isfinite(args.alpha)
        or not math.isfinite(args.beta)
        or (
            args.suite_timeout is not None
            and (not math.isfinite(args.suite_timeout) or args.suite_timeout <= 0)
        )
        or (
            args.memory_budget is not None
            and (not math.isfinite(args.memory_budget) or args.memory_budget <= 0)
        )
    ):
        parser.error(
            "Require at least one worker, 0 <= range begin < range end <= 2**31, "
            "finite positive timeouts and memory budget, positive reduction blocks, "
            "and finite scalars"
        )
    if args.chunk_size is not None and (args.chunk_size < 1 or args.resume_from is not None):
        parser.error("Chunk size must be positive; a campaign resumes its own chunks")
    if args.solution_indices is not None:
        try:
            args.solution_indices = [
                int(line.split("#", 1)[0])
                for line in args.solution_indices.read_text(encoding="utf-8").splitlines()
                if line.split("#", 1)[0].strip()
            ]
        except (OSError, ValueError) as error:
            parser.error(f"Cannot read solution indices: {error}")
        if not args.solution_indices or not all(0 <= i < (1 << 31) for i in args.solution_indices):
            parser.error("Solution indices must be a nonempty list of native indices")
    for key in (
        "rocjitsu",
        "client",
        "config",
        "hardware_profile_template",
        "library_dir",
        "reports",
        "resume_from",
    ):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    stops = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous = {signum: signal.signal(signum, stop_native_processes) for signum in stops}
    try:
        return run(args) if args.chunk_size is None else run_campaign(args)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
