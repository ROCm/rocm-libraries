#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bounded packaged-kernel experiment for the advisory rocJITsu race-check job."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import zlib

SHAPES = [(256, 256, 1, k) for k in (64, 128, 256, 320)]
TYPE_OPTIONS = {
    "problem-identifier": "operationIdentifier",
    "a-type": "aType",
    "b-type": "bType",
    "c-type": "cType",
    "d-type": "dType",
    "e-type": "eType",
    "alpha-type": "computeType",
    "beta-type": "computeType",
    "activation-compute-type": "activationComputeDataType",
    "f32-xdl-math-op": "f32XdlMathOp",
    "use-gradient": "useGradient",
    "use-bias": "useBias",
    "use-e": "useE",
    "output-amaxD": "outputAmaxD",
    "use-scaleAB": "useScaleAB",
    "use-scaleCD": "useScaleCD",
    "use-scaleAlphaVec": "useScaleAlphaVec",
    "swizzle-tensor-a": "swizzleTensorA",
    "swizzle-tensor-b": "swizzleTensorB",
    "sparse": "sparse",
    "high-precision-accumulate": "highPrecisionAccumulate",
    "strided-batched": "stridedBatched",
    "grouped-gemm": "groupedGemm",
    "activation-type": "activationType",
    "activation-no-guard": "activationNoGuard",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_library(path):
    import msgpack

    payload = path.read_bytes()
    if path.name.endswith(".zlib"):
        payload = zlib.decompress(payload)
    data = msgpack.unpackb(payload, raw=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a library mapping: {path}")
    return data


def matches_hardware(predicate, target, device):
    kind = predicate["type"]
    value = predicate.get("value")
    if kind == "AMDGPU":
        return matches_hardware(value, target, device)
    if kind in {"And", "Or"}:
        results = [matches_hardware(p, target, device) for p in value]
        return all(results) if kind == "And" else any(results)
    if kind == "Processor":
        return value == target
    if kind == "PciChipId":
        return value == device["device_id"]
    if kind == "CUCount":
        return value == device["simd_count"] // device["simd_per_cu"]
    # Unknown constraints are outside the prototype's selection scope, never bypassed.
    return False


def ordinary_problem(problem):
    if any(
        problem.get(key, False)
        for key in (
            "groupedGemm",
            "sparse",
            "useGradient",
            "useE",
            "mxBlockA",
            "mxBlockB",
            "useInitialStridesAB",
            "useInitialStridesCD",
        )
    ):
        return False
    allowed = {"Half", "BFloat16", "Float", "Float8", "BFloat8", "Int8", "Int32"}
    return all(problem[key] in allowed for key in ("aType", "bType", "cType", "dType"))


def select_batches(groups, kernel_count, batch_size):
    """Select distinct names in contiguous index ranges without trimming native shards."""
    batches = []
    names = set()
    remaining = kernel_count
    # Round-robin across shards before selecting another batch from a large shard.
    candidates = []
    for key, solutions in sorted(groups.items()):
        solutions.sort(key=lambda s: s["index"])
        candidates.append([key, solutions, 0])
    while remaining:
        made_progress = False
        for candidate in candidates:
            key, solutions, start = candidate
            size = min(batch_size, remaining)
            while start + size <= len(solutions):
                block = solutions[start : start + size]
                start += 1
                kernels = {s["kernelName"] for s in block}
                indices = [s["index"] for s in block]
                if (
                    indices != list(range(indices[0], indices[0] + size))
                    or len(kernels) != size
                    or names & kernels
                ):
                    continue
                names.update(kernels)
                batches.append(
                    {
                        "id": len(batches),
                        "library": key[0],
                        "problem_type": block[0]["problemType"],
                        "hardware_predicate": block[0]["hardwarePredicate"],
                        "solutions": [
                            {
                                "index": s["index"],
                                "name": s["name"],
                                "kernel": s["kernelName"],
                            }
                            for s in block
                        ],
                    }
                )
                remaining -= size
                start += size - 1
                made_progress = True
                break
            candidate[2] = start
            if not remaining:
                break
        if not made_progress:
            raise ValueError(
                f"Requested {kernel_count} distinct kernels; selected only {kernel_count - remaining}"
            )
    return batches


def prepare(args):
    base = json.loads(args.config.read_text(encoding="utf-8"))
    device = base["vm"]["gpu"]["device"]
    library_dir = (
        args.library_dir / args.target
        if (args.library_dir / args.target).is_dir()
        else args.library_dir
    )
    libraries = sorted(
        set(library_dir.glob(f"*{args.target}.dat"))
        | set(library_dir.glob(f"*{args.target}.dat.zlib"))
    )
    if not libraries:
        raise ValueError(
            f"No {args.target} packaged solution metadata under {library_dir}"
        )
    groups = defaultdict(list)
    counts = Counter()
    for library in libraries:
        for solution in read_library(library).get("solutions", []):
            counts["inventory"] += 1
            if not matches_hardware(solution["hardwarePredicate"], args.target, device):
                counts["other_or_unknown_hardware"] += 1
                continue
            if not ordinary_problem(solution["problemType"]):
                counts["outside_problem_scope"] += 1
                continue
            key = (
                str(library),
                json.dumps(solution["problemType"], sort_keys=True),
                json.dumps(solution["hardwarePredicate"], sort_keys=True),
            )
            group = groups[key]
            # Keep only selection fields, and share the type/predicate within a
            # group. Large size mappings and problem predicates stay in the
            # packaged shard, where the native client will enforce them.
            group.append(
                {
                    **{k: solution[k] for k in ("index", "name", "kernelName")},
                    "problemType": (
                        group[0]["problemType"] if group else solution["problemType"]
                    ),
                    "hardwarePredicate": (
                        group[0]["hardwarePredicate"]
                        if group
                        else solution["hardwarePredicate"]
                    ),
                }
            )
            counts["eligible"] += 1
    jobs = select_batches(groups, args.kernels, args.batch_size)
    for job in jobs:
        metadata = Path(job["library"])
        main_object = metadata.with_name(
            metadata.name.removesuffix(".zlib").removesuffix(".dat") + ".co"
        )
        if not main_object.is_file():
            raise ValueError(f"Missing code object for selected shard: {main_object}")
        helpers = [
            library_dir / f"Kernels.so-000-{args.target}.hsaco",
            library_dir / f"hipblasltTransform_{args.target}.hsaco",
        ]
        job["code_objects"] = [str(main_object)] + [
            str(p) for p in helpers if p.is_file()
        ]
        job["sha256"] = {
            str(p): sha256(p) for p in [metadata, *map(Path, job["code_objects"])]
        }
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
    manifest = {
        "target": args.target,
        "workers": args.workers,
        "kernels": args.kernels,
        "shapes": SHAPES,
        "planned_cases": args.kernels * len(SHAPES),
        "inventory": dict(counts),
        "jobs": jobs,
        "tools": {str(p): sha256(p) for p in (args.rocjitsu, args.client, args.config)},
        "note": "Bounded deterministic ordinary-GEMM selection, not full library/path coverage.",
    }
    write_json(args.reports / "manifest.json", manifest)
    return manifest, config


def client_options(job, results):
    problem = job["problem_type"]
    options = {
        key: problem[field] for key, field in TYPE_OPTIONS.items() if field in problem
    }
    options.update(
        {
            "library-file": job["library"],
            "results-file": str(results),
            "solution-start-idx": job["solutions"][0]["index"],
            "num-solutions": len(job["solutions"]),
            "compute-input-type-A": problem.get(
                "computeInputTypeA", problem.get("computeInputType", problem["aType"])
            ),
            "compute-input-type-B": problem.get(
                "computeInputTypeB", problem.get("computeInputType", problem["bType"])
            ),
            "bias-source": problem["biasSrcWhiteList"][0],
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


def classify(job, text, returncode):
    expected = {(s["index"], shape): s for s in job["solutions"] for shape in SHAPES}
    cases = {}
    errors = []
    for line in text.splitlines():
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
        cases[key] = row[9]
        if row[9] != "PASSED":
            errors.append(f"Case {key}: {row[9]}")
    dispatches = Counter(
        re.findall(r'^\[rocjitsu\] Kernel dispatch: "([^"]+)"', text, re.M)
    )
    wanted = Counter(
        expected[key]["kernel"] for key, status in cases.items() if status == "PASSED"
    )
    for name in {s["kernel"] for s in job["solutions"]}:
        if dispatches[name] != wanted[name]:
            errors.append(f"Target dispatch count mismatch: {name}")
    # Retain auxiliary dispatch identities for diagnosis; helpers vary by datatype/GSU.
    auxiliary = {
        k: v
        for k, v in dispatches.items()
        if k not in {s["kernel"] for s in job["solutions"]}
    }
    races = re.findall(r"^RACE .*", text, re.M)
    warnings = Counter(line for line in text.splitlines() if "[rj warn]" in line)
    if races:
        errors.append(
            f"{len(races)} race reports (including runtime/kernel reports; none suppressed)"
        )
    if warnings:
        errors.append(
            f"{sum(warnings.values())} emulator warnings; coverage needs investigation"
        )
    missing = sorted(set(expected) - set(cases))
    if missing:
        errors.append(f"{len(missing)} missing case records")
    if returncode:
        errors.append(f"Client exited {returncode}")
    return {
        "id": job["id"],
        "failed": bool(errors),
        "errors": errors,
        "returncode": returncode,
        "cases": [
            {"index": i, "shape": s, "status": v} for (i, s), v in sorted(cases.items())
        ],
        "race_headers": races,
        "warnings": dict(warnings),
        "auxiliary_dispatches": auxiliary,
    }


def execute_command(command, log, timeout, env):
    with log.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def run_queue(jobs, workers, execute, progress):
    lock = threading.Lock()
    results = []
    next_job = 0
    stopped = False
    active = 0
    maximum = 0
    stop_snapshot = None

    def worker(slot):
        nonlocal next_job, stopped, active, maximum, stop_snapshot
        while True:
            with lock:
                if stopped or next_job == len(jobs):
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
                if result["failed"] and not stopped:
                    stopped = True
                    stop_snapshot = {
                        "trigger_job": job["id"],
                        "assigned_jobs": next_job,
                        "in_flight": active,
                    }
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
                    if not stopped:
                        stopped = True
                        stop_snapshot = {
                            "trigger_job": job["id"],
                            "assigned_jobs": next_job,
                            "in_flight": active,
                        }

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
    args.reports.mkdir(parents=True, exist_ok=False)
    try:
        manifest, config = prepare(args)
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
        write_json(
            args.reports / "settings.json",
            {
                "cpus": cpus,
                "timeout": args.timeout,
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
            options = client_options(job, stem.with_suffix(".csv"))
            ini = [f"{k}={v}" for k, v in options.items()]
            ini += ["problem-size=" + ",".join(map(str, shape)) for shape in SHAPES]
            ini += ["code-object=" + path for path in job["code_objects"]]
            stem.with_suffix(".ini").write_text("\n".join(ini) + "\n", encoding="utf-8")
            command = [
                "taskset",
                "-c",
                str(cpus[slot]),
                str(args.rocjitsu),
                "--config",
                str(config),
                "--",
                str(args.client),
                "--config-file",
                str(stem.with_suffix(".ini")),
            ]
            write_json(stem.with_suffix(".command.json"), command)
            begin = time.monotonic()
            rc = execute_command(command, stem.with_suffix(".log"), args.timeout, env)
            result = classify(
                job,
                stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace"),
                rc,
            )
            result.update(cpu=cpus[slot], seconds=time.monotonic() - begin)
            write_json(stem.with_suffix(".result.json"), result)
            print(
                f"batch {job['id']}: {'FAILED' if result['failed'] else 'PASSED'} ({result['seconds']:.2f}s)",
                flush=True,
            )
            return result

        start = time.monotonic()
        summary = run_queue(
            manifest["jobs"],
            args.workers,
            execute,
            lambda value: write_json(args.reports / "progress.json", value),
        )
        by_id = {j["id"]: j for j in manifest["jobs"]}
        counts = Counter()
        missing = []
        for result in summary["results"]:
            expected = {
                (s["index"], shape)
                for s in by_id[result["id"]]["solutions"]
                for shape in SHAPES
            }
            observed = {(c["index"], tuple(c["shape"])) for c in result["cases"]}
            missing += [
                {"index": i, "shape": shape} for i, shape in sorted(expected - observed)
            ]
            counts.update(c["status"] for c in result["cases"])
        unstarted = [
            {"index": s["index"], "shape": shape}
            for j in summary["unstarted"]
            for s in by_id[j]["solutions"]
            for shape in SHAPES
        ]
        summary.update(
            seconds=time.monotonic() - start,
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
            not unstarted
            and not missing
            and all(not r["failed"] for r in summary["results"])
            and counts["PASSED"] == summary["planned_cases"]
        )
        write_json(args.reports / "summary.json", summary)
        print(
            f"{counts['PASSED']}/{summary['planned_cases']} numerical passes; {len(missing)} missing, {len(unstarted)} unstarted. Reports: {args.reports}"
        )
        return 0 if summary["passed"] else 1
    except Exception as error:
        write_json(args.reports / "setup-or-run-error.json", {"error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rocjitsu", "client", "config", "library-dir", "reports"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target", choices=("gfx942", "gfx950"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kernels", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 4
        or not 1 <= args.kernels <= 100
        or not 1 <= args.batch_size <= 10
        or not 0 < args.timeout <= 120
    ):
        parser.error(
            "Prototype limits: 1–4 workers, 1–100 kernels, 1–10 kernels/batch, timeout at most 120 seconds"
        )
    for key in ("rocjitsu", "client", "config", "library_dir", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
