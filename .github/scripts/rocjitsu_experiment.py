# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run bounded reference/shape trials against one frozen artifact workload."""

import argparse
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


def trial(strategy, parent, env, plan, workers, seconds, suite, label=None):
    root = parent / (label or strategy)
    root.mkdir()
    shutil.copytree(parent / "shape-yamls", root / "shape-yamls")
    # Paths inside the plan are immutable original inputs. Retained copies in
    # each report allow later offline hash verification without the runner.
    write_json(root / "shape-plan.json", plan)
    config = json.loads((parent / "artifact-config.json").read_text())
    budget = 1 if strategy == "shape" else len(os.sched_getaffinity(0)) // workers
    config.update(cpu_thread_budget=budget, num_threads=1)
    if strategy == "shape":
        config.update(cpu_dispatch_threads=1, async_helper_threads=0)
    write_json(root / "rocjitsu-config.json", config)
    runtime = dict(
        strategy=strategy,
        workers=workers,
        cpu_thread_budget=budget,
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
        ),
    )
    command = commands(strategy, root, plan, env, workers, seconds, suite)
    runtime["command"] = command
    write_json(root / "runtime.json", runtime)
    child_env = env | dict(
        OMP_NUM_THREADS=str(budget),
        OMP_THREAD_LIMIT=str(budget),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("shape", "reference", "paired", "safety"),
        default="shape",
    )
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--per-job-seconds", type=int, required=True)
    parser.add_argument("--suite-seconds", type=int, required=True)
    args = parser.parse_args()
    cores = len(os.sched_getaffinity(0))
    if not 1 <= args.workers <= cores:
        parser.error("workers must be 1..available CPUs")
    if not 0 < args.per_job_seconds <= args.suite_seconds:
        parser.error("per-job-seconds must be 1..suite-seconds")
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
    strategies = (
        ["reference", "shape"] if args.strategy == "paired" else [args.strategy]
    )
    if args.strategy == "safety":
        strategies = ["serial", "shape"]
    if "reference" in strategies and plan["excluded"]:
        parser.error("reference/paired requires a cohort with zero gate exclusions")
    random.SystemRandom().shuffle(strategies)
    write_json(
        parent / "trial-order.json",
        dict(order=strategies, independent_blocks=1, statistical_claim=False),
    )
    reports = {}
    for strategy in strategies:
        reports[strategy] = trial(
            "shape" if strategy == "serial" else strategy,
            parent,
            dict(os.environ),
            plan,
            (
                args.workers
                if strategy == "shape"
                else 1 if strategy == "serial" else min(cores, 16)
            ),
            args.per_job_seconds,
            args.suite_seconds,
            label=strategy,
        )
    failed = any(r["audit_errors"] for r in reports.values())
    if len(reports) == 2:
        comparison = compare(
            reports["reference" if "reference" in reports else "serial"],
            reports["shape"],
        )
        write_json(parent / "comparison.json", comparison)
        print(
            json.dumps({k: v for k, v in comparison.items() if k != "differences"}),
            flush=True,
        )
        failed |= bool(comparison["audit_errors"])
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write(
                "\nRocJITsu audited calibration; one block does not establish a speedup.\n\n"
            )
            stream.write(
                "| Strategy | Wall(s) | Pytest pass | Numerical PASSED rows | Applicability rejections | Timeouts | Audit errors |\n|---|---:|---:|---:|---:|---:|---:|\n"
            )
            for name, report in reports.items():
                stream.write(
                    f"| {name} | {report['execution'].get('wall_seconds', 0):.3f} | {report['counts'].get('passed', 0)} | {report['client_verdicts'].get('PASSED', 0)} | {report['client_verdicts'].get('DID_NOT_SATISFY_ASSERTS', 0)} | {report['counts'].get('job_timeout', 0)} | {len(report['audit_errors'])} |\n"
                )
    write_json(
        parent / "experiment.json",
        dict(
            complete=True,
            exit_code=int(failed),
            strategies=strategies,
            independent_blocks=1,
        ),
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
