#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run every installed ffm-quick CTest entry under one rocJITsu process tree.

The nightly's CTestTestfile.cmake files already carry the ffm-quick label.
This walks those files, in ascending test-count order, and launches each
selected command as ``rocjitsu --config <gfx942> -- <command>``. A failing
suite is recorded and the walk continues. Each test is capped at one hour
so one suite cannot consume the whole six-hour job.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

LABEL_PATTERN = re.compile(r"^ffm-quick$")
EXCLUDE_PATTERN = re.compile(r"ex_gpu")
DEFAULT_TIMEOUT = 1800
TIMEOUT_CAP = 3600


def tokenize_cmake_args(text: str) -> list[str]:
    """Split one CMake argument list, including quotes and [[brackets]]."""
    args = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        if text.startswith("[[", index):
            end = text.find("]]", index + 2)
            if end < 0:
                raise ValueError(f"unclosed bracket argument in: {text}")
            args.append(text[index + 2 : end])
            index = end + 2
            continue
        if text[index] == '"':
            index += 1
            chars = []
            while index < length and text[index] != '"':
                if text[index] == "\\" and index + 1 < length:
                    chars.append(text[index + 1])
                    index += 2
                    continue
                chars.append(text[index])
                index += 1
            index += 1
            args.append("".join(chars))
            continue
        start = index
        while index < length and not text[index].isspace():
            index += 1
        args.append(text[start:index])
    return args


def _call(line: str):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    for name in ("add_test", "set_tests_properties"):
        prefix = name + "("
        if stripped.startswith(prefix) and stripped.endswith(")"):
            return name, stripped[len(prefix) : -1]
    return None


def parse_ctest_file(text: str) -> dict[str, dict]:
    """Return test name -> command, labels, timeout, skip regex, workdir."""
    tests = {}
    order = []
    for line in text.splitlines():
        parsed = _call(line)
        if parsed is None:
            continue
        name, body = parsed
        args = tokenize_cmake_args(body)
        if name == "add_test":
            if len(args) < 2:
                continue
            if args[0] == "NAME" and "COMMAND" in args:
                test_name = args[1]
                command = args[args.index("COMMAND") + 1 :]
            else:
                test_name = args[0]
                command = args[1:]
            if test_name not in tests:
                order.append(test_name)
                tests[test_name] = {
                    "name": test_name,
                    "command": command,
                    "labels": [],
                    "timeout": None,
                    "skip_regex": "",
                    "working_directory": "",
                }
            else:
                tests[test_name]["command"] = command
            continue
        if "PROPERTIES" not in args:
            continue
        split = args.index("PROPERTIES")
        names = args[:split]
        props = args[split + 1 :]
        updates = {}
        cursor = 0
        while cursor + 1 < len(props):
            key = props[cursor]
            value = props[cursor + 1]
            updates[key] = value
            cursor += 2
        for test_name in names:
            test = tests.setdefault(
                test_name,
                {
                    "name": test_name,
                    "command": [],
                    "labels": [],
                    "timeout": None,
                    "skip_regex": "",
                    "working_directory": "",
                },
            )
            if test_name not in order:
                order.append(test_name)
            if "LABELS" in updates:
                test["labels"] = updates["LABELS"].split(";")
            if "TIMEOUT" in updates:
                test["timeout"] = int(updates["TIMEOUT"])
            if "SKIP_REGULAR_EXPRESSION" in updates:
                test["skip_regex"] = updates["SKIP_REGULAR_EXPRESSION"]
            if "WORKING_DIRECTORY" in updates:
                test["working_directory"] = updates["WORKING_DIRECTORY"]
    return {name: tests[name] for name in order if tests[name]["command"]}


def selects_ffm_quick(test: dict) -> bool:
    labels = test["labels"]
    if any(EXCLUDE_PATTERN.search(label) for label in labels):
        return False
    return any(LABEL_PATTERN.search(label) for label in labels)


def timeout_seconds(test: dict) -> int:
    raw = test.get("timeout")
    if raw is None:
        return DEFAULT_TIMEOUT
    return max(1, min(int(raw), TIMEOUT_CAP))


def discover(rocm_root: Path) -> list[dict]:
    """Installed ffm-quick tests, fewest tests in a directory first."""
    groups = []
    bin_dir = rocm_root / "bin"
    for ctest_file in sorted(bin_dir.rglob("CTestTestfile.cmake")):
        tests = [
            test
            for test in parse_ctest_file(
                ctest_file.read_text(encoding="utf-8", errors="replace")
            ).values()
            if selects_ffm_quick(test)
        ]
        if not tests:
            continue
        relative = ctest_file.parent.relative_to(bin_dir).as_posix()
        groups.append(
            {
                "component": relative,
                "directory": ctest_file.parent,
                "tests": tests,
            }
        )
    groups.sort(key=lambda group: (len(group["tests"]), group["component"]))
    return groups


def output_skips(text: str, skip_regex: str) -> bool:
    if not skip_regex:
        return False
    for pattern in skip_regex.split(";"):
        if not pattern:
            continue
        try:
            if re.search(pattern, text):
                return True
        except re.error:
            continue
    return False


def build_env(rocm_root: Path) -> dict:
    env = os.environ.copy()
    lib = str(rocm_root / "lib")
    bin_dir = str(rocm_root / "bin")
    env["ROCM_PATH"] = str(rocm_root)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = lib if not previous else lib + os.pathsep + previous
    return env


def run_one(rocjitsu: Path, config: Path, test: dict, cwd: Path, env: dict, log_path: Path) -> dict:
    timeout = timeout_seconds(test)
    command = [str(rocjitsu), "--config", str(config), "--", *test["command"]]
    work = Path(test["working_directory"]) if test["working_directory"] else cwd
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(
        f"START {test['name']} timeout={timeout}s cmd={shlex.join(test['command'])}",
        flush=True,
    )
    try:
        proc = subprocess.run(
            command,
            cwd=work,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = proc.stdout or ""
        status = "passed" if proc.returncode == 0 else "failed"
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        status = "timeout"
        returncode = None
    if status != "passed" and output_skips(output, test["skip_regex"]):
        status = "skipped"
    elapsed = time.monotonic() - started
    log_path.write_text(output, encoding="utf-8")
    if status not in ("passed", "skipped"):
        tail = "\n".join(output.splitlines()[-20:])
        if tail:
            print(tail, flush=True)
    print(f"DONE {test['name']} {status} {elapsed:.1f}s", flush=True)
    return {
        "name": test["name"],
        "status": status,
        "seconds": round(elapsed, 1),
        "returncode": returncode,
        "timeout": timeout,
    }


def render_summary(rows: list[dict]) -> str:
    lines = [
        "# rocJITsu ffm-quick",
        "",
        "| component | tests | passed | failed | timeout | skipped | not started | seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        counts = row["counts"]
        lines.append(
            "| {component} | {total} | {passed} | {failed} | {timeout} | "
            "{skipped} | {not_started} | {seconds:.0f} |".format(
                component=row["component"],
                total=row["total"],
                seconds=row["seconds"],
                **counts,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocjitsu", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rocm-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--deadline-seconds", type=int, default=19800)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    rocm_root = Path(args.rocm_root)
    groups = discover(rocm_root)
    total = sum(len(group["tests"]) for group in groups)
    print(f"ffm-quick directories: {len(groups)} tests: {total}", flush=True)
    for group in groups:
        print(f"  {len(group['tests']):4d}  {group['component']}", flush=True)
    if args.list_only:
        return 0 if groups else 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(rocm_root)
    rocjitsu = Path(args.rocjitsu).resolve()
    config = Path(args.config).resolve()
    deadline = time.monotonic() + args.deadline_seconds
    rows = []
    any_problem = False
    for group in groups:
        started = time.monotonic()
        results = []
        print(f"COMPONENT {group['component']} tests={len(group['tests'])}", flush=True)
        for test in group["tests"]:
            if time.monotonic() >= deadline:
                results.append(
                    {
                        "name": test["name"],
                        "status": "not_started",
                        "seconds": 0,
                        "returncode": None,
                        "timeout": timeout_seconds(test),
                    }
                )
                any_problem = True
                continue
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", test["name"])[:120]
            log_path = out_dir / "logs" / group["component"] / f"{safe}.log"
            result = run_one(rocjitsu, config, test, group["directory"], env, log_path)
            results.append(result)
            if result["status"] in ("failed", "timeout"):
                any_problem = True
        counts = {
            status: sum(1 for result in results if result["status"] == status)
            for status in ("passed", "failed", "timeout", "skipped", "not_started")
        }
        row = {
            "component": group["component"],
            "total": len(results),
            "seconds": time.monotonic() - started,
            "counts": counts,
            "tests": results,
        }
        rows.append(row)
        summary = render_summary(rows)
        (out_dir / "summary.md").write_text(summary, encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        print(summary, flush=True)
    return 1 if any_problem or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
