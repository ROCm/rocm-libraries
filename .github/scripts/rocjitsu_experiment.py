# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run bounded reference/shape trials against one frozen artifact workload."""

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time

from rocjitsu_compare import audit, compare
from rocjitsu_check_suite import memory_lines
from rocjitsu_evidence import digest, source_name
from rocjitsu_pytest import write_json


def commands(strategy, root, plan, env, workers, seconds, suite):
    scripts = Path(__file__).parent
    tensile = Path(env["TENSILELITE_ROOT"])
    original = tensile / "tensilelite/Tests/common/test_config.py"
    options = [
        "--gpu-targets=" + env["ROCJITSU_TARGET"],
        "-v",
        "--capture=fd",
        "--client-lock-scope=worker",
        "--timeout=" + str(seconds),
        "--rocjitsu-stall-dump-after=" + str(min(120, seconds / 2)),
        "--prebuilt-client=" + env["TENSILELITE_CLIENT"],
        "--global-parameters=LibraryFormat='msgpack'",
    ]
    globals_ = ["PrintWinnersOnly=False", "ClientLogLevel=3", "DataInitSeed=261025"]
    if strategy == "shape":
        globals_.append("CpuThreads=1")
    options.append(
        "--tensile-options="
        + ",".join(
            [
                "--cxx-compiler",
                env["ROCM_PATH"] + "/bin/amdclang++",
                "--gpu-targets",
                env["ROCJITSU_TARGET"],
                "--global-parameters",
                *globals_,
            ]
        )
    )
    mirage = [
        "mirage",
        "run",
        "--profile",
        env["MIRAGE_PROFILE"],
        "--in-process",
        "--config",
        str(root / "rocjitsu-config.json"),
        "--",
    ]
    if strategy == "shape":
        test = [
            sys.executable,
            "-m",
            "pytest",
            str(scripts / "rocjitsu_shape_test.py"),
            "-c",
            str(tensile / "pytest.ini"),
            "-p",
            "rocjitsu_shape_case",
            "-p",
            "rocjitsu_pytest",
            "-p",
            "rocjitsu_evidence",
            "-p",
            "no:cacheprovider",
            "-n",
            "0",
            *options,
        ]
        workload = [
            sys.executable,
            str(scripts / "rocjitsu_shape_queue.py"),
            "--plan",
            str(root / "shape-plan.json"),
            "--report-dir",
            str(root),
            "--workers",
            str(workers),
            "--per-job-seconds",
            str(seconds),
            "--",
            *mirage,
            *test,
        ]
    else:
        if plan["excluded"]:
            raise ValueError(
                "reference trials require a cohort entirely below the ceiling"
            )
        workload = [
            *mirage,
            sys.executable,
            "-m",
            "pytest",
            str(original),
            "-c",
            str(tensile / "pytest.ini"),
            "-p",
            "rocjitsu_pytest",
            "-p",
            "rocjitsu_evidence",
            "-p",
            "no:cacheprovider",
            "--rocjitsu-report-dir",
            str(root),
            "--rocjitsu-evidence-dir",
            str(root / "evidence"),
            "--rocjitsu-durations",
            str(scripts / "rocjitsu_tensilelite_durations.json"),
            "--basetemp",
            str(root / "tmp"),
            "--junit-xml",
            str(root / "junit/tensilelite.xml"),
            "-m",
            env["ROCJITSU_TARGET"],
            "-k",
            env["CASE_FILTER"],
            "-n",
            str(workers),
            *options,
        ]
    return [
        sys.executable,
        str(scripts / "rocjitsu_pytest.py"),
        "--report-dir",
        str(root),
        "--timeout",
        str(suite),
        "--",
        *workload,
    ]


def trial(
    strategy, parent, env, plan, workers, seconds, suite, label=None, dispatch_threads=1
):
    root = parent / (label or strategy)
    root.mkdir()
    shutil.copytree(parent / "shape-yamls", root / "shape-yamls")
    # Paths inside the plan are immutable original inputs. Retained copies in
    # each report allow later offline hash verification without the runner.
    write_json(root / "shape-plan.json", plan)
    config = json.loads((parent / "artifact-config.json").read_text())
    budget = (
        dispatch_threads
        if strategy == "shape"
        else len(os.sched_getaffinity(0)) // workers
    )
    reference_threads = 1 if strategy == "shape" else budget
    config.update(cpu_thread_budget=budget, num_threads=1)
    if strategy == "shape":
        config.update(cpu_dispatch_threads=dispatch_threads, async_helper_threads=0)
    write_json(root / "rocjitsu-config.json", config)
    runtime = dict(
        strategy=strategy,
        workers=workers,
        available_cpus=len(os.sched_getaffinity(0)),
        runner_name=env.get("RUNNER_NAME"),
        harness_revision=env.get("GITHUB_SHA"),
        cpu_thread_budget=budget,
        reference_threads=reference_threads,
        seed=261025,
        target=env["ROCJITSU_TARGET"],
        profile=env["MIRAGE_PROFILE"],
        artifact_run_id=env["ARTIFACT_RUN_ID"],
        config=config,
        config_sha256=digest(root / "rocjitsu-config.json"),
        client_sha256=digest(Path(env["TENSILELITE_CLIENT"])),
        rocjitsu_sha256=digest(Path(env["ROCJITSU_EFFECTIVE_LIB"])),
        mirage_sha256=digest(Path(env["ROCM_PATH"]) / "bin/mirage"),
        compiler_sha256=digest(Path(env["ROCM_PATH"]) / "bin/amdclang++"),
        per_job_seconds=seconds,
        suite_seconds=suite,
        instrumentation=dict(
            PrintWinnersOnly=False,
            ClientLogLevel=3,
            DataInitSeed=261025,
            OPENBLAS_NUM_THREADS=1,
            MKL_NUM_THREADS=1,
            TOKIO_WORKER_THREADS=1,
        ),
    )
    command = commands(strategy, root, plan, env, workers, seconds, suite)
    runtime["command"] = command
    write_json(root / "runtime.json", runtime)
    child_env = env | dict(
        OMP_NUM_THREADS=str(reference_threads),
        OMP_THREAD_LIMIT=str(reference_threads),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKIO_WORKER_THREADS="1",
    )
    (root / "container-memory-before.txt").write_text(
        "\n".join(memory_lines(Path("/sys/fs/cgroup"))) + "\n"
    )
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (root / "tensilelite-test.log").open("w") as log:
        result = subprocess.run(
            command, env=child_env, cwd=root, stdout=log, stderr=subprocess.STDOUT
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (root / "container-memory-after.txt").write_text(
        "\n".join(memory_lines(Path("/sys/fs/cgroup"))) + "\n"
    )
    write_json(
        root / "resource-time.json",
        dict(
            wall_seconds=time.monotonic() - start,
            user_seconds=after.ru_utime - before.ru_utime,
            system_seconds=after.ru_stime - before.ru_stime,
            exit_code=result.returncode,
        ),
    )
    report = audit(root)
    print(
        json.dumps(
            dict(
                strategy=strategy,
                execution=report["execution"],
                counts=report["counts"],
                client_verdicts=report["client_verdicts"],
                timing_seconds=report["timing_seconds"],
                audit_errors=report["audit_errors"],
            )
        ),
        flush=True,
    )
    # Baseline success output also contains code objects. Evidence was copied
    # by the plugin before any cleanup; failures retain their output locally.
    if result.returncode == 0:
        shutil.rmtree(root / "tmp", ignore_errors=True)
    return report


def trial_specifications(
    strategy,
    workers,
    reference_workers,
    dispatch_threads=1,
    allocations="48:1,32:2,16:4",
):
    if strategy == "allocation":
        result = []
        for entry in allocations.split(","):
            fields = entry.split(":")
            if len(fields) != 2 or any(not field.isdecimal() for field in fields):
                raise ValueError(
                    "allocations must be comma-separated workers:dispatch pairs"
                )
            count, width = map(int, fields)
            if count < 1 or width not in (1, 2, 4):
                raise ValueError(
                    "allocation workers must be positive and dispatch must be 1, 2 or 4"
                )
            result.append((f"shape-{count}-dispatch{width}", "shape", count, width))
        if len(result) < 2 or len({s[0] for s in result}) != len(result):
            raise ValueError(
                "allocation comparison requires at least two distinct allocations"
            )
        return result
    if strategy == "matrix":
        if workers < 2:
            raise ValueError("matrix requires at least two requested shape workers")
        return [
            ("reference", "reference", reference_workers, dispatch_threads),
            (f"shape-{workers // 2}", "shape", workers // 2, dispatch_threads),
            (f"shape-{workers}", "shape", workers, dispatch_threads),
        ]
    if strategy == "paired":
        return [
            ("reference", "reference", reference_workers, dispatch_threads),
            ("shape", "shape", workers, dispatch_threads),
        ]
    if strategy == "safety":
        return [
            ("serial", "shape", 1, dispatch_threads),
            ("shape", "shape", workers, dispatch_threads),
        ]
    return [
        (
            strategy,
            strategy,
            reference_workers if strategy == "reference" else workers,
            dispatch_threads,
        )
    ]


def benchmark_report(parent, reports, comparisons):
    rows = []
    for name, report in reports.items():
        usage = report.get("resource_time") or {}
        rows.append(
            dict(
                strategy=name,
                workers=report["runtime"]["workers"],
                cpu_thread_budget=report["runtime"]["cpu_thread_budget"],
                dispatch_threads=report["runtime"]
                .get("config", {})
                .get("cpu_dispatch_threads"),
                reference_threads=report["runtime"].get("reference_threads"),
                workload_seconds=usage.get(
                    "wall_seconds", report["execution"].get("wall_seconds")
                ),
                mean_cpu_cores=usage.get("mean_cpu_cores"),
                cpu_capacity_fraction=usage.get("cpu_capacity_fraction"),
                numerical_passed=report["client_verdicts"].get("PASSED", 0),
                applicability_rejections=report["client_verdicts"].get(
                    "DID_NOT_SATISFY_ASSERTS", 0
                ),
                timeouts=report["counts"].get("job_timeout", 0),
                incomplete=sum(
                    report["counts"].get(field, 0)
                    for field in (
                        "claimed_without_result",
                        "unstarted",
                        "incomplete",
                        "missing_result",
                    )
                ),
                audit_errors=len(report["audit_errors"]),
                counts=json.dumps(report["counts"], sort_keys=True),
                verdicts=json.dumps(report["client_verdicts"], sort_keys=True),
            )
        )
    coverage = next(iter(reports.values()))["coverage"]
    excluded_fraction = coverage["excluded_declared_work_fraction"]
    excluded_share = (
        f" ({100 * excluded_fraction:.3f}% of declared work)"
        if excluded_fraction is not None
        else ""
    )
    result = dict(
        schema=1,
        coverage=coverage,
        strategies=rows,
        comparisons=comparisons,
        independent_blocks=1,
        statistical_claim=False,
    )
    write_json(parent / "benchmark.json", result)
    with (parent / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def display(value):
        return f"{value:.3f}" if value is not None else "unavailable"

    text = [
        "RocJITsu strategy calibration. One randomized block does not establish a repeatable speedup.",
        "",
        f"Selected {coverage['selected_yaml_files']} YAML files; admitted {coverage['admitted_queue_jobs']} queue jobs ({coverage['admitted_problem_descriptors']} normalized problem descriptors).",
        f"Threshold excluded {coverage['threshold_excluded_queue_jobs']} jobs across {coverage['threshold_affected_yaml_files']} YAML files; {coverage['fully_excluded_yaml_files']} whole YAML files were excluded.",
        f"Declared work admitted: {coverage['admitted_declared_work']}; excluded: {coverage['excluded_declared_work']}{excluded_share}. {coverage['work_definition']}",
        "",
        "Workload elapsed time includes job startup, compilation, validation and cleanup. Shared artifact/environment setup and initial workload planning are outside each trial. Average CPU use includes child processes and may include host waiters; it is not a measure of useful emulator arithmetic. Container memory peaks span the shared job and are not per-strategy peaks.",
        "Whole-YAML reference arms retain the original generator settings and use one engine, automatic dispatch and a reference thread budget based on available CPUs divided by workers. Shape arms use serial generation and references with the reported dispatch width. These are comparisons of complete strategies in a common harness; shape arms with the same dispatch width isolate outer concurrency. Allocation comparisons vary workers and dispatch together. All arms use the same artifacts, fixed seed, verdict reporting and Mirage startup-pool limit.",
        "",
        "| Strategy | Workers | Dispatch width | Workload(s) | Average CPU cores | CPU capacity fraction | Numerical PASSED | Applicability rejections | Timeouts/incomplete | Audit errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['strategy']} | {row['workers']} | {row['dispatch_threads'] or 'auto'} | {display(row['workload_seconds'])} | {display(row['mean_cpu_cores'])} | {display(row['cpu_capacity_fraction'])} | {row['numerical_passed']} | {row['applicability_rejections']} | {row['timeouts']}/{row['incomplete']} | {row['audit_errors']} |"
        )
    text += [
        "",
        "Elapsed ratios are available in benchmark.json only when the pair's runtime, requested work, accepted solutions and verdicts pass the equivalence audit. Failures and incomplete work remain in the report.",
    ]
    (parent / "benchmark.md").write_text("\n".join(text) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("shape", "reference", "paired", "safety", "matrix", "allocation"),
        default="shape",
    )
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--reference-workers", type=int)
    parser.add_argument("--dispatch-threads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--allocations", default="48:1,32:2,16:4")
    parser.add_argument("--per-job-seconds", type=int, required=True)
    parser.add_argument("--suite-seconds", type=int, required=True)
    args = parser.parse_args()
    cores = len(os.sched_getaffinity(0))
    if not 1 <= args.workers <= cores:
        parser.error("workers must be 1..available CPUs")
    if not 0 < args.per_job_seconds <= args.suite_seconds:
        parser.error("per-job-seconds must be 1..suite-seconds")
    reference_workers = (
        args.reference_workers if args.reference_workers is not None else min(cores, 16)
    )
    if not 1 <= reference_workers <= cores:
        parser.error("reference-workers must be 1..available CPUs")
    try:
        specifications = trial_specifications(
            args.strategy,
            args.workers,
            reference_workers,
            args.dispatch_threads,
            args.allocations,
        )
    except ValueError as error:
        parser.error(str(error))
    for _, strategy, count, width in specifications:
        if count > cores or (strategy == "shape" and count * width > cores):
            parser.error("each allocation must fit within the available CPU count")
    if args.suite_seconds * len(specifications) > 1500:
        parser.error("combined trial execution limits must total at most 1500 seconds")
    parent = args.report_dir.resolve()
    plan = json.loads((parent / "shape-plan.json").read_text())
    manifest = os.environ.get("COHORT_MANIFEST")
    if manifest:
        cohort = json.loads(Path(manifest).read_text())
        actual = {source_name(Path(s["path"])): s["sha256"] for s in plan["sources"]}
        expected = {s["path"]: s["sha256"] for s in cohort["sources"]}
        if (
            actual != expected
            or len(plan["jobs"]) != cohort["admitted_jobs"]
            or len(plan["excluded"]) != cohort["excluded_jobs"]
            or plan["ceiling"] != cohort["ceiling"]
            or os.environ["ARTIFACT_RUN_ID"] != cohort["artifact_run_id"]
        ):
            parser.error("frozen calibration workload differs from its manifest")
        shutil.copyfile(manifest, parent / "cohort.json")
    if any(s[1] == "reference" for s in specifications) and plan["excluded"]:
        parser.error(
            "whole-YAML comparisons require a cohort with zero gate exclusions"
        )
    random.SystemRandom().shuffle(specifications)
    strategies = [s[0] for s in specifications]
    write_json(
        parent / "trial-order.json",
        dict(order=strategies, independent_blocks=1, statistical_claim=False),
    )
    reports = {}
    for label, strategy, workers, dispatch_threads in specifications:
        reports[label] = trial(
            strategy,
            parent,
            dict(os.environ),
            plan,
            workers,
            args.per_job_seconds,
            args.suite_seconds,
            label=label,
            dispatch_threads=dispatch_threads,
        )
    failed = any(r["audit_errors"] for r in reports.values())
    comparisons = []
    for left, right in itertools.combinations(sorted(reports), 2):
        comparison = dict(
            left=left, right=right, **compare(reports[left], reports[right])
        )
        comparisons.append(comparison)
        print(
            json.dumps({k: v for k, v in comparison.items() if k != "differences"}),
            flush=True,
        )
        failed |= bool(comparison["audit_errors"])
    if len(comparisons) == 1:
        write_json(parent / "comparison.json", comparisons[0])
    elif comparisons:
        write_json(parent / "comparisons.json", comparisons)
    benchmark_report(parent, reports, comparisons)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write("\n" + (parent / "benchmark.md").read_text())
    write_json(
        parent / "experiment.json",
        dict(
            complete=not failed,
            exit_code=int(failed),
            strategies=strategies,
            independent_blocks=1,
        ),
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
