#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run repeatable GEMMs through rocJITsu on CPUs and count their outcomes."""

import argparse
import csv
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sample_hipblaslt_gemms import SAMPLER_VERSION, sample_gemm


def bench_input(problem):
    return {
        **{key: value for key, value in problem.items() if key != "family"},
        "function": "matmul",
        "algo_method": 0,
        "requested_solution_num": 1,
        "scale_type": "f32_r",
        "compute_input_typeA": problem["a_type"],
        "compute_input_typeB": problem["b_type"],
        "initialization": problem.get("initialization", "rand_int"),
        "norm_check": 1,
        "norm_check_assert": True,
        "allclose_check": 1,
        "iters": 1,
        "cold_iters": 0,
        "adaptive": False,
        "use_gpu_timer": False,
        "print_kernel_info": True,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_input(path, problem):
    # The bench template ends in 'Tests:', so its input must be a block sequence.
    lines = [
        f"  {key}: {json.dumps(value)}" for key, value in bench_input(problem).items()
    ]
    lines[0] = "-" + lines[0][1:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_explicit_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8")) if path else []
    if not isinstance(cases, list):
        raise ValueError("explicit cases must be a JSON list")
    required = set(sample_gemm(random.Random(0)))
    allowed = required | {
        "initialization",
        "lda",
        "ldb",
        "ldc",
        "ldd",
        "stride_a",
        "stride_b",
        "stride_c",
        "stride_d",
    }
    names = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id",
            "description",
            "problem",
        }:
            raise ValueError("each explicit case needs id, description and problem")
        name = case["id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name)
            or name in names
        ):
            raise ValueError("explicit case IDs must be unique lowercase names")
        names.add(name)
        problem = case["problem"]
        if not isinstance(problem, dict) or not required <= set(problem) <= allowed:
            raise ValueError(f"{name}: missing or unknown problem fields")
        for key in ("M", "N", "K", "batch_count"):
            if type(problem[key]) is not int or problem[key] <= 0:
                raise ValueError(f"{name}: {key} must be positive")
        for key in ("alpha", "beta"):
            if not isinstance(problem[key], (int, float)) or not math.isfinite(
                problem[key]
            ):
                raise ValueError(f"{name}: {key} must be finite")
        if problem["alpha"] == 0 or any(
            problem[key] not in ("N", "T") for key in ("transA", "transB")
        ):
            raise ValueError(f"{name}: requires nonzero alpha and N/T transposes")
        if not isinstance(case["description"], str) or not case["description"]:
            raise ValueError(f"{name}: description required")
    return cases


def write_case_report(directory, records):
    records = sorted(records, key=lambda record: record["id"])
    write_json(directory / "cases.json", records)
    lines = [
        "# GEMM inputs and outcomes",
        "",
        "| Case | Source | Input | Outcome | Observation | Purpose |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        result = record["result"]
        problem = json.dumps(record["problem"], sort_keys=True)
        reason = (result.get("reason") or "numerics and race detection passed").replace(
            "|", "\\|"
        )
        purpose = record.get("description", "Fixed-seed random sample").replace(
            "|", "\\|"
        )
        lines.append(
            f"| [{record['id']}]({record['id']}/input.yaml) | {record['source']} | `{problem}` | {result['outcome']} | {reason} | {purpose} |"
        )
    (directory / "cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command, directory, timeout, log_name):
    """Terminate the entire case process group on timeout, including bench."""
    with (directory / log_name).open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        ) as process:
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return 124


def check_result(problem, text, race_log):
    """Require the requested numerical row and GEMM dispatch in the race sink."""
    if re.search(r"^RACE\s", text + "\n" + race_log, re.MULTILINE):
        return "race reported"
    if "[rj warn]" in text or "[rj error]" in text:
        return "rocJITsu diagnostic; inspect bench.log"
    if "error: NO solution found!" in text:
        return "no heuristic solution"
    rows, header = [], None
    for line in text.splitlines():
        if re.match(r"^\[\d+\]:", line):
            header = next(csv.reader([line.split(":", 1)[1]]))
        elif header:
            values = next(csv.reader([line.strip()]))
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))
                header = None
    if len(rows) != 1:
        return "expected one completed bench result"
    row = rows[0]
    try:
        for key in ("M", "N", "K", "batch_count"):
            if int(row[key.lower()]) != problem[key]:
                return "bench result dimensions differ from input"
        for key in (
            "lda",
            "ldb",
            "ldc",
            "ldd",
            "stride_a",
            "stride_b",
            "stride_c",
            "stride_d",
        ):
            if key in problem and int(row[key]) != problem[key]:
                return "bench result layout differs from input"
        for key in ("a_type", "b_type", "c_type", "d_type", "transA", "transB"):
            if row[key] != problem[key]:
                return "bench result types or transposes differ from input"
        if row["compute_type"] != problem["compute_type"][2:]:
            return "bench result compute type differs from input"
        for key in ("alpha", "beta", "scaleA", "scaleB", "bias_vector"):
            if float(row[key]) != problem[key]:
                return "bench result scalars, scaling or epilogue differ from input"
        if row["activation_type"] != problem["activation_type"] or (
            problem["bias_vector"] and row["bias_type"] != problem["bias_type"]
        ):
            return "bench result epilogue differs from input"
        for key in ("norm_error", "atol", "rtol"):
            value = float(row[key])
            if not math.isfinite(value) or value < 0:
                return "invalid numerical result"
    except (KeyError, ValueError):
        return "missing or failed numerical checks"
    kernels = re.findall(r"^\s*--kernel name:\s*(\S+)\s*$", text, re.MULTILINE)
    if len(kernels) != 1 or not any(
        line.startswith("[rocjitsu] Kernel dispatch:")
        and f'symbol="{kernels[0]}"' in line
        for line in race_log.splitlines()
    ):
        return "missing GEMM dispatch in race-detector output"
    return None


def run_case(args, directory, problem, config):
    # Use this interpreter for YAML conversion instead of bench's build-time Python.
    bench_dir = Path(args.bench).parent
    convert = [
        sys.executable,
        str(bench_dir / "hipblaslt_gentest.py"),
        "--template",
        str(bench_dir / "hipblaslt_template.yaml"),
        "-o",
        str(directory / "input.data"),
        str(directory / "input.yaml"),
    ]
    command = [
        args.rocjitsu,
        "--config",
        str(directory / "rocjitsu.json"),
        "--",
        args.bench,
        "--data",
        str(directory / "input.data"),
    ]
    (directory / "command.txt").write_text(
        shlex.join(convert) + "\n" + shlex.join(command) + "\n", encoding="utf-8"
    )
    case_config = {**config, "plugins": {"race": {}, "logging": {}}}
    # Parallelism comes from independent GEMMs, each with one emulator worker.
    case_config.update(
        num_threads=1,
        cpu_dispatch_threads=1,
        async_helper_threads=0,
        cpu_thread_budget=1,
    )
    case_config["sinks"] = {"types": ["stderr", "file"], "dir": str(directory)}
    write_json(directory / "rocjitsu.json", case_config)
    status = run_command(convert, directory, args.timeout, "conversion.log")
    if status:
        return {
            "outcome": "failed",
            "reason": "input conversion failed",
            "exit_code": status,
        }
    status = run_command(command, directory, args.timeout, "bench.log")
    text = (directory / "bench.log").read_text(encoding="utf-8", errors="replace")
    race_path = directory / "race.log"
    race_log = (
        race_path.read_text(encoding="utf-8", errors="replace")
        if race_path.exists()
        else ""
    )
    reason = (
        f"bench exited {status}" if status else check_result(problem, text, race_log)
    )
    if re.search(r"^RACE\s", text + "\n" + race_log, re.MULTILINE):
        reason = "race reported"
    return {
        "outcome": (
            "unsupported"
            if reason == "no heuristic solution"
            else "failed" if reason else "passed"
        ),
        "reason": reason,
        "exit_code": status,
        "kernels": sorted(
            set(re.findall(r"^\s*--kernel name:\s*(\S+)\s*$", text, re.MULTILINE))
        ),
    }


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="number of random samples; zero runs explicit cases only",
    )
    parser.add_argument(
        "--explicit-cases",
        type=Path,
        help="JSON list of required GEMMs, additional to --count",
    )
    parser.add_argument(
        "--explicit-case", help="run only this ID from --explicit-cases"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads", type=positive_int, default=1, help="number of concurrent GEMMs"
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="stop starting new cases after a failure",
    )
    parser.add_argument(
        "--case", type=positive_int, help="run just this 1-based iteration"
    )
    parser.add_argument("--bench", default="hipblaslt-bench")
    parser.add_argument("--rocjitsu", default="rocjitsu")
    parser.add_argument(
        "--config", type=Path, help="CPU-emulation rocJITsu JSON config"
    )
    parser.add_argument(
        "--timeout", type=positive_int, default=180, help="seconds per command"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.count < 0:
        parser.error("--count must be nonnegative")
    try:
        explicit_cases = load_explicit_cases(args.explicit_cases)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.explicit_case:
        explicit_cases = [
            case for case in explicit_cases if case["id"] == args.explicit_case
        ]
        if not explicit_cases or args.case:
            parser.error(
                "--explicit-case requires a matching ID and cannot combine with --case"
            )
    elif args.case:
        explicit_cases = []
    if not args.count and not explicit_cases:
        parser.error("at least one random or explicit case is required")
    if args.case and args.case > args.count:
        parser.error("--case must be within --count")
    config = None
    if not args.dry_run:
        if not args.config:
            parser.error("--config is required for execution")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            parser.error("--config must contain a JSON object")
        if (
            config.get("vm", {})
            .get("gpu", {})
            .get("device", {})
            .get("gfx_target_version")
            != 90500
        ):
            parser.error("--config must emulate gfx950")
        for name in ("bench", "rocjitsu"):
            executable = shutil.which(getattr(args, name))
            if not executable:
                parser.error(f"{name} executable not found")
            setattr(args, name, str(Path(executable).resolve()))
    # A new directory prevents stale detector files from affecting a rerun.
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "sampler": SAMPLER_VERSION,
        "architecture": "gfx950",
        "seed": args.seed,
        "requested": (0 if args.explicit_case else 1 if args.case else args.count)
        + len(explicit_cases),
        "sampled": 0,
        "passed": 0,
        "failed": 0,
        "unsupported": 0,
        "known_failures": 0,
        "coverage": {key: {} for key in ("family", "transpose", "epilogue")},
        "threads": args.threads,
        "dry_run": args.dry_run,
        "complete": False,
        "sources": {
            source: {key: 0 for key in ("started", "passed", "failed", "unsupported")}
            for source in ("random", "explicit")
        },
    }
    rng = random.Random(args.seed)
    lock = threading.Lock()
    stop = threading.Event()
    iterations = iter(range(1, (0 if args.explicit_case else args.count) + 1))
    pending_explicit = iter(explicit_cases)
    records = []
    started = time.perf_counter()

    def worker():
        while True:
            # Sample in iteration order so scheduling cannot change the inputs.
            with lock:
                if stop.is_set():
                    return
                case = next(pending_explicit, None)
                if case:
                    problem = case["problem"]
                    name = "explicit-" + case["id"]
                    source = "explicit"
                else:
                    for iteration in iterations:
                        problem = sample_gemm(rng)
                        if not args.case or iteration == args.case:
                            break
                    else:
                        return
                    name = f"case-{iteration:04d}"
                    source = "random"
                    summary["sampled"] += 1
                summary["sources"][source]["started"] += 1
                categories = {
                    "family": problem["family"],
                    "transpose": problem["transA"] + problem["transB"],
                    "epilogue": ("bias+" if problem["bias_vector"] else "")
                    + problem["activation_type"],
                }
                for key, category in categories.items():
                    counts = summary["coverage"][key]
                    counts[category] = counts.get(category, 0) + 1
            directory = args.output / name
            directory.mkdir()
            write_input(directory / "input.yaml", problem)
            write_json(directory / "problem.json", problem)
            print(
                f"{name}: {problem['M']}x{problem['N']}x{problem['K']} {problem['family']}",
                flush=True,
            )
            result = {"outcome": "dry-run", "reason": "execution not requested"}
            if not args.dry_run:
                try:
                    result = run_case(args, directory, problem, config)
                except OSError as error:
                    result = {
                        "outcome": "failed",
                        "reason": f"execution error: {error}",
                    }
                write_json(directory / "result.json", result)
                with lock:
                    summary[result["outcome"]] += 1
                    summary["sources"][source][result["outcome"]] += 1
                    if args.stop_on_failure and result["outcome"] != "passed":
                        stop.set()
                print(
                    f"{name}: {result['outcome']}: {result['reason'] or 'numerics and races checked'}",
                    flush=True,
                )
            with lock:
                records.append(
                    {
                        "id": name,
                        "source": source,
                        "problem": problem,
                        "bench_input": bench_input(problem),
                        "result": result,
                        **({"description": case["description"]} if case else {}),
                    }
                )

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(worker) for _ in range(args.threads)]
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException:
                    stop.set()
                    raise
        summary["complete"] = (
            sum(counts["started"] for counts in summary["sources"].values())
            == summary["requested"]
        )
    finally:
        random_counts = summary["sources"]["random"]
        completed_random = sum(
            random_counts[key] for key in ("passed", "failed", "unsupported")
        )
        summary["random_no_solution_rate"] = (
            random_counts["unsupported"] / completed_random
            if completed_random
            else None
        )
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", summary)
        write_case_report(args.output, records)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return int(summary["failed"] + summary["unsupported"] != 0)


if __name__ == "__main__":
    sys.exit(main())
