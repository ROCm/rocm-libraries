#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run packaged solution entries for the advisory rocJITsu race-check job."""

import argparse
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import math

from rocjitsu_sweep_plan import make_batches, make_plan, sha256, POLICY, REGULAR_POLICY
from rocjitsu_known_failures import apply_known_failures

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
    state = "FAIL" if result["failed"] else "PASS"
    if result.get("batch_errors"):
        state = "BATCH FAILED"
    notes = "; ".join(result["errors"])
    if skipped:
        if not result["failed"]:
            state = "SKIPPED" if len(skipped) == planned else "PASS WITH SKIPS"
        reasons = {
            c["known_failure"]["id"]: c["known_failure"]
            for c in skipped
            if "known_failure" in c
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
    return sorted({r.get("batch_id", r["id"]) for r in results if r["failed"]})


def render_report(manifest, summary):
    def cell(value):
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")

    backend = manifest["backend"]
    client = "hipblaslt-bench" if backend == "bench" else "tensilelite-client"
    inventory = manifest.get("inventory", {})
    results = {r["id"]: r for r in summary["results"]}
    lines = [
        f"### {client} sampled race sweep",
        "",
        f"Seed: `{cell(manifest['seed'])}`; policy: `{cell(manifest['policy'])}`; "
        f"inventory/preparation: {manifest['preparation_seconds']:.2f} s.",
        f"Artifact fingerprint: `{manifest['artifact_fingerprint']}`.",
        f"Selected solutions: {len(manifest['jobs'])}; eligible/inventory solutions: "
        f"{inventory.get('eligible_solutions', '?')}/{inventory.get('solutions', '?')}; "
        f"eligible/inventory kernel names: "
        f"{inventory.get('eligible_kernel_names', '?')}/{inventory.get('unique_kernel_names', '?')}.",
        "",
        "PASS requires all planned cases, matching identities/dispatches, no races or warnings, and a successful exit. "
        "SKIPPED cases did not run; PASS WITH SKIPS validates only the remaining cases.",
        "BATCH FAILED means the process had a diagnostic or incomplete evidence; "
        "it does not identify every member index as a source of failure.",
        f"Failed batches: {len(failed_batch_ids(summary['results']))}/"
        f"{len({j.get('batch_id', j['id']) for j in manifest['jobs']})}.",
        f"Full shapes, identities, commands and logs are in the uploaded `sweep-{backend}/` artifacts.",
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
    base = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = make_plan(
        args.library_dir,
        args.target,
        base["vm"]["gpu"]["device"],
        args.solutions,
        args.seed,
        args.solution_indices,
        getattr(args, "policy", POLICY),
    )
    manifest.update(
        backend=args.backend,
        workers=args.workers,
        tools={str(p): sha256(p) for p in (args.rocjitsu, args.client, args.config)},
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
    apply_known_failures(
        manifest, failures, args.rocjitsu_revision, args.run_known_failures
    )
    write_text(args.reports / "known_failures.yaml", failures)
    manifest["preparation_seconds"] = time.monotonic() - begin
    base.update(
        max_ticks=0,
        num_threads=1,
        cpu_dispatch_threads=1,
        async_helper_threads=0,
        cpu_thread_budget=1,
    )
    base["plugins"] = {"race": {}}
    base["sinks"] = {"types": ["stderr"]}
    config = args.reports / "rocjitsu.json"
    write_json(config, base)
    write_json(args.reports / "manifest.json", manifest)
    return manifest, config


def client_options(job, results):
    problem = job["problem_type"]
    options = problemTypeOptions(problem)
    options.update(
        {
            "library-file": job["library"],
            "results-file": str(results),
            "solution-start-idx": job["solutions"][0]["index"],
            "num-solutions": len(job["solutions"]),
            "bias-type-args": (
                problem["biasDataTypeWhiteList"] or [problem["computeType"]]
            )[0],
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
    expected = {
        (s["index"], tuple(c["shape"])): s
        for s in job["solutions"]
        for c in job["cases"]
    }
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
                "case_id": next(
                    c["id"] for c in job["cases"] if tuple(c["shape"]) == shape
                ),
            }
            for (i, shape), evidence in sorted(cases.items())
        ],
    }
    return finish_result(job, text, returncode, result, errors, launches_per_case=1)


def finish_result(job, text, returncode, result, errors, launches_per_case):
    dispatches = Counter(
        re.findall(r'^\[rocjitsu\] Kernel dispatch: "([^"]+)"', text, re.M)
    )
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
        "scaleA": {"": "none", "Scalar": "Scalar", "Vector": "Vector"}[
            p.get("useScaleAB", "")
        ],
        "scaleB": {"": "none", "Scalar": "Scalar", "Vector": "Vector"}[
            p.get("useScaleAB", "")
        ],
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
        identity = re.match(
            r"\s*--(Solution index|Solution name|kernel name):\s*(.*)", line
        )
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
    """Check ordered records individually and retain diagnostics for the whole process."""
    ends = [m.end() for m in re.finditer(r"^\s*--kernel name:[^\n]*", text, re.M)]
    expected_records = sum(len(j["cases"]) for j in jobs)
    batch_errors = []
    if returncode:
        batch_errors.append(f"Batch client exited {returncode}")
    if len(ends) != expected_records:
        batch_errors.append(
            f"Batch expected {expected_records} record boundaries, observed {len(ends)}"
        )
    races = re.findall(r"^RACE .*", text, re.M)
    warnings = Counter(line for line in text.splitlines() if "[rj warn]" in line)
    if races:
        batch_errors.append(f"Batch reported {len(races)} races (none suppressed)")
    if warnings:
        batch_errors.append(
            f"Batch reported {sum(warnings.values())} emulator warnings"
        )
    # A repeated kernel name may belong to several indices. Count every occurrence.
    expected_dispatches = Counter()
    for job in jobs:
        expected_dispatches[job["solutions"][0]["kernel"]] += len(
            job["cases"]
        ) * job.get("target_dispatches_per_case", 2)
    dispatches = Counter(
        re.findall(r'^\[rocjitsu\] Kernel dispatch: "([^"]+)"', text, re.M)
    )
    for kernel, wanted in expected_dispatches.items():
        if dispatches[kernel] != wanted:
            batch_errors.append(
                f"Batch target dispatch mismatch for {kernel}: expected {wanted}, observed {dispatches[kernel]}"
            )
    results, start, record = [], 0, 0
    for job in jobs:
        record += len(job["cases"])
        end = ends[record - 1] if record < len(ends) else len(text)
        result = classify_bench(job, text[start:end], returncode)
        result["batch_errors"] = batch_errors
        result["errors"] += batch_errors
        result["failed"] = bool(result["errors"])
        results.append(result)
        start = end
    return results


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


def run_queue(
    batches, workers, execute, progress, deadline=None, continue_on_failure=False
):
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
                        result.setdefault("errors", []).append(
                            f"Cannot write progress: {error!r}"
                        )
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
    raise ValueError(
        f"Need {count} distinct available physical cores; found {len(cpus)}"
    )


def run(args):
    suite_start = time.monotonic()
    deadline = suite_start + args.suite_timeout
    args.reports.mkdir(parents=True, exist_ok=False)
    try:
        manifest, config = prepare(args)
        batch_size = getattr(args, "batch_size", None)
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZES[args.backend]
        if args.backend == "tensile" and batch_size != 1:
            raise ValueError("Tensile backend requires batch size 1")
        batches = make_batches(
            manifest["jobs"], batch_size, group_by_shard=args.backend == "bench"
        )
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
        write_text(
            args.reports / "summary.md", render_report(manifest, {"results": []})
        )
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

        def execute(batch, slot):
            begin = time.monotonic()
            batch_id = batch[0]["batch_id"]
            stem = args.reports / f"batch-{batch_id:03}"
            completed, runnable = [], []
            for job in batch:
                if "planning_error" in job:
                    completed.append(
                        {
                            "id": job["id"],
                            "failed": True,
                            "errors": [job["planning_error"]],
                            "target_dispatches": 0,
                            "cases": [
                                {
                                    "case_id": c["id"],
                                    "index": job["solutions"][0]["index"],
                                    "shape": c["shape"],
                                    "status": "UNSUPPORTED_CASE",
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
            if runnable:
                if args.backend == "tensile":
                    job = runnable[0]
                    options = client_options(job, stem.with_suffix(".csv"))
                    ini = [f"{k}={v}" for k, v in options.items()]
                    ini += [
                        "problem-size=" + ",".join(map(str, c["shape"]))
                        for c in job["cases"]
                    ]
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
                        str(config),
                        "--",
                        str(args.client),
                        *client_args,
                    ]
                    write_json(stem.with_suffix(".command.json"), command)
                    launch_begin = time.monotonic()
                    rc = execute_command(
                        command,
                        stem.with_suffix(".log"),
                        batch_deadline - launch_begin,
                        env,
                    )
                    native_seconds = time.monotonic() - launch_begin
                    text = stem.with_suffix(".log").read_text(
                        encoding="utf-8", errors="replace"
                    )
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
                result.update(batch_id=batch_id, cpu=cpus[slot], batch_seconds=seconds)
            write_json(
                stem.with_suffix(".result.json"),
                {
                    "id": batch_id,
                    "jobs": [j["id"] for j in batch],
                    "seconds": seconds,
                    "conversion_seconds": conversion_seconds,
                    "native_seconds": native_seconds,
                    "results": completed,
                },
            )
            return completed

        def progress(value):
            for result in value["completed"]:
                job = manifest["jobs"][result["id"]]
                observed = {c["case_id"] for c in result["cases"]}
                result["cases"] += [
                    c for c in skipped_cases(job) if c["case_id"] not in observed
                ]
                result["batch_id"] = job["batch_id"]
                write_json(
                    args.reports / f"solution-{job['id']:03}.result.json", result
                )
            write_json(args.reports / "progress.json", value)
            write_text(args.reports / "summary.md", render_report(manifest, value))
            for result in value["completed"]:
                row = solution_report(
                    manifest["jobs"][result["id"]], result, args.backend
                )
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
            seed=args.seed,
            artifact_fingerprint=manifest["artifact_fingerprint"],
            seconds=time.monotonic() - start,
            suite_seconds=time.monotonic() - suite_start,
            suite_timeout=args.suite_timeout,
            planned_cases=manifest["planned_cases"],
            counts=dict(counts),
            missing=missing,
            unstarted_cases=unstarted,
            failed_batch_ids=failed_batch_ids(summary["results"]),
        )
        summary["accounted_cases"] = (
            sum(counts.values()) + len(missing) + len(unstarted)
        )
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
            else f"### {args.backend} sampled race sweep\n"
        )
        write_text(
            report,
            prior
            + f"\nRun status: FAIL. Setup/run error: {html.escape(repr(error))}\n",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rocjitsu", "client", "config", "library-dir", "reports"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target", choices=("gfx942", "gfx950"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Solution indices per native process "
            f"(bench: {DEFAULT_BATCH_SIZES['bench']}; tensile: {DEFAULT_BATCH_SIZES['tensile']})"
        ),
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Finish the selected workload after failures; still exit unsuccessfully",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--solutions",
        type=int,
        default=100,
        help="Select this many eligible solution entries (default: 100)",
    )
    selection.add_argument(
        "--all-solutions",
        action="store_true",
        help="Select every eligible solution index in stable priority order",
    )
    selection.add_argument(
        "--solution-index",
        dest="solution_indices",
        action="append",
        type=int,
        help="Run this artifact-specific index; repeat for multiple solutions (including aliases)",
    )
    parser.add_argument("--backend", choices=("tensile", "bench"), required=True)
    parser.add_argument("--policy", choices=(POLICY, REGULAR_POLICY), default=POLICY)
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
        "--seed",
        required=True,
        help="Seed for deterministic solution-name ranking; CI always uses 0",
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
        default=1500,
        help="Per-backend time budget in seconds, including inventory preparation",
    )
    args = parser.parse_args()
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
        not 1 <= args.workers <= 8
        or args.solutions < 1
        or not 0 < args.timeout <= 120
        or not 0 < args.suite_timeout <= 1500
    ):
        parser.error(
            "Limits: 1–8 workers, a positive solution count, process timeout at most "
            "120 seconds, suite timeout at most 1500 seconds"
        )
    if args.all_solutions:
        args.solutions = None
    for key in ("rocjitsu", "client", "config", "library_dir", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
