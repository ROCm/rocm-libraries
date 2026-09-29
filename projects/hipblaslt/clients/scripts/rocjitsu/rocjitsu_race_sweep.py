#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bounded packaged-kernel experiment for the advisory rocJITsu race-check job."""

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
import threading
import time
import math

from rocjitsu_sweep_plan import make_plan, sha256

from Tensile.Utilities.ClientConfig import problemTypeOptions


def write_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2) + "\n")


def solution_report(job, result, backend):
    """One row of evidence; numerical success alone never means PASS."""
    planned = len(job["cases"])
    observed = sum(c["status"] == "PASSED" for c in result["cases"])
    return [
        str(job["solutions"][0]["index"]),
        "FAIL" if result["failed"] else "PASS",
        f"{observed}/{planned}",
        f"{result.get('target_dispatches', '?')}/{planned * (2 if backend == 'bench' else 1)}",
        f"{result['seconds']:.2f}" if "seconds" in result else "—",
        "; ".join(result["errors"]) or "Identity, numerical and race checks passed",
    ]


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
        f"Selected solutions: {len(manifest['jobs'])}; eligible/inventory kernel names: "
        f"{inventory.get('eligible_kernel_names', '?')}/{inventory.get('unique_kernel_names', '?')}.",
        "",
        "PASS requires all planned cases, matching identities/dispatches, no races or warnings, and a successful exit.",
        f"Full shapes, identities, commands and logs are in the uploaded `sweep-{backend}/` artifacts.",
        "",
        "| Solution index | Result | Numerical passes | Target dispatches | Seconds | Notes |",
        "|---:|:---|---:|---:|---:|:---|",
    ]
    for job in manifest["jobs"]:
        if job["id"] in results:
            row = solution_report(job, results[job["id"]], backend)
        else:
            final = "unstarted" in summary
            assigned = job["id"] < summary.get("assigned_jobs", 0)
            state = "INCOMPLETE" if assigned else "PENDING"
            if final or (summary.get("stop") and not assigned):
                state = "NOT RUN"
            row = [str(job["solutions"][0]["index"]), state, "—", "—", "—", ""]
        row[-1] += f" (batch-{job['id']:03})"
        lines.append("| " + " | ".join(map(cell, row)) + " |")
    lines += [
        "",
        (
            f"Run status: {'PASS' if summary['passed'] else 'FAIL'}."
            if "passed" in summary
            else "Run incomplete; this table is updated after each result."
        ),
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
        args.kernels,
        args.seed,
        args.solution_indices,
    )
    manifest.update(
        backend=args.backend,
        workers=args.workers,
        tools={str(p): sha256(p) for p in (args.rocjitsu, args.client, args.config)},
        preparation_seconds=time.monotonic() - begin,
    )
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
        "compute_input_typeA": BENCH_TYPES[
            p.get("computeInputTypeA", p.get("computeInputType", p["aType"]))
        ],
        "compute_input_typeB": BENCH_TYPES[
            p.get("computeInputTypeB", p.get("computeInputType", p["bType"]))
        ],
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
            row = next(csv.reader([line.strip()]))
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
            if row["target_dispatches"] != 2:
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
        launches_per_case=2,
    )


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


def run_queue(jobs, workers, execute, progress, deadline=None):
    lock = threading.Lock()
    results = []
    next_job = 0
    active = 0
    maximum = 0
    stop_snapshot = None

    def stop(reason, trigger_job=None):
        nonlocal stop_snapshot
        if stop_snapshot is None:
            stop_snapshot = {
                "reason": reason,
                "trigger_job": trigger_job,
                "assigned_jobs": next_job,
                "in_flight": active,
            }

    def expired():
        return deadline is not None and time.monotonic() >= deadline

    def worker(slot):
        nonlocal next_job, active, maximum
        while True:
            with lock:
                if stop_snapshot is not None or next_job == len(jobs):
                    return
                if expired():
                    stop("suite-timeout")
                    return
                job = jobs[next_job]
                next_job += 1
                active += 1
                maximum = max(maximum, active)
            try:
                result = execute(job, slot)
            except Exception as error:
                result = {
                    "id": job["id"],
                    "failed": True,
                    "errors": [repr(error)],
                    "cases": [],
                }
            with lock:
                results.append(result)
                active -= 1
                if expired():
                    stop("suite-timeout", job["id"])
                elif result["failed"]:
                    stop("job-failure", job["id"])
                try:
                    progress(
                        {
                            "results": results,
                            "assigned_jobs": next_job,
                            "stop": stop_snapshot,
                        }
                    )
                except Exception as error:
                    result["failed"] = True
                    result.setdefault("errors", []).append(
                        f"Cannot write progress: {error!r}"
                    )
                    stop("progress-error", job["id"])

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return {
        "results": sorted(results, key=lambda r: r["id"]),
        "unstarted": [j["id"] for j in jobs[next_job:]],
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

        def execute(job, slot):
            stem = args.reports / f"batch-{job['id']:03}"
            if "planning_error" in job:
                result = {
                    "id": job["id"],
                    "failed": True,
                    "target_dispatches": 0,
                    "errors": [job["planning_error"]],
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
                write_json(stem.with_suffix(".result.json"), result)
                return result
            if args.backend == "tensile":
                options = client_options(job, stem.with_suffix(".csv"))
                ini = [f"{k}={v}" for k, v in options.items()]
                ini += [
                    "problem-size=" + ",".join(map(str, c["shape"]))
                    for c in job["cases"]
                ]
                ini += ["code-object=" + path for path in job["code_objects"]]
                stem.with_suffix(".ini").write_text(
                    "\n".join(ini) + "\n", encoding="utf-8"
                )
                client_args = ["--config-file", str(stem.with_suffix(".ini"))]
            else:
                import yaml

                stem.with_suffix(".yaml").write_text(
                    yaml.safe_dump(bench_options(job), sort_keys=False),
                    encoding="utf-8",
                )
                client_args = ["--yaml", str(stem.with_suffix(".yaml"))]
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
            begin = time.monotonic()
            rc = execute_command(
                command,
                stem.with_suffix(".log"),
                min(args.timeout, deadline - begin),
                env,
            )
            classifier = (
                classify_tensile if args.backend == "tensile" else classify_bench
            )
            result = classifier(
                job,
                stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace"),
                rc,
            )
            if rc == 124:
                result["errors"].append(
                    "Suite time budget exhausted"
                    if time.monotonic() >= deadline
                    else "Process timeout"
                )
            result.update(cpu=cpus[slot], seconds=time.monotonic() - begin)
            write_json(stem.with_suffix(".result.json"), result)
            return result

        def progress(value):
            write_json(args.reports / "progress.json", value)
            write_text(args.reports / "summary.md", render_report(manifest, value))
            result = value["results"][-1]
            row = solution_report(manifest["jobs"][result["id"]], result, args.backend)
            print(
                f"{args.backend} solution {row[0]}: {row[1]}; numerical={row[2]}; dispatches={row[3]}; seconds={row[4]}; {row[5]}",
                flush=True,
            )

        start = time.monotonic()
        summary = run_queue(
            manifest["jobs"],
            args.workers,
            execute,
            progress,
            deadline=deadline,
        )
        by_id = {j["id"]: j for j in manifest["jobs"]}
        counts = Counter()
        missing = []
        for result in summary["results"]:
            job = by_id[result["id"]]
            observed = {c["case_id"] for c in result["cases"]}
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
            and counts["PASSED"] == summary["planned_cases"]
        )
        write_json(args.reports / "summary.json", summary)
        write_text(args.reports / "summary.md", render_report(manifest, summary))
        print(
            f"{counts['PASSED']}/{summary['planned_cases']} numerical passes; {len(missing)} missing, {len(unstarted)} unstarted. Reports: {args.reports}"
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--kernels", type=int, default=100)
    selection.add_argument(
        "--solution-index",
        dest="solution_indices",
        action="append",
        type=int,
        help="Run this artifact-specific index; repeat for multiple solutions (including aliases)",
    )
    parser.add_argument("--backend", choices=("tensile", "bench"), required=True)
    parser.add_argument(
        "--seed",
        required=True,
        help="Recorded PR revision or explicit seed for reproducible sampling",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--suite-timeout",
        type=float,
        default=1500,
        help="Per-backend time budget in seconds, including inventory preparation",
    )
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 4
        or not 1 <= args.kernels <= 100
        or not 0 < args.timeout <= 120
        or not 0 < args.suite_timeout <= 1500
    ):
        parser.error(
            "Prototype limits: 1–4 workers, 1–100 kernels, process timeout at most "
            "120 seconds, suite timeout at most 1500 seconds"
        )
    for key in ("rocjitsu", "client", "config", "library_dir", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
