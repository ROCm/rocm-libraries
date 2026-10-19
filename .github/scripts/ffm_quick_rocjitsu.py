#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run installed CTest entries matching a label under rocJITsu.

Each invocation handles one component so workflow matrices can fan out. Tests
have no per-test timeout; the GitHub Actions job limit remains the outer cap.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
                    "properties": {},
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
                    "properties": {},
                },
            )
            if test_name not in order:
                order.append(test_name)
            test["properties"].update(updates)
            if "LABELS" in updates:
                test["labels"] = updates["LABELS"].split(";")
            if "TIMEOUT" in updates:
                test["timeout"] = int(updates["TIMEOUT"])
            if "SKIP_REGULAR_EXPRESSION" in updates:
                test["skip_regex"] = updates["SKIP_REGULAR_EXPRESSION"]
            if "WORKING_DIRECTORY" in updates:
                test["working_directory"] = updates["WORKING_DIRECTORY"]
    return {name: tests[name] for name in order if tests[name]["command"]}


def selects_label(test: dict, label_pattern: re.Pattern) -> bool:
    return any(label_pattern.search(label) for label in test["labels"])


def discover(rocm_root: Path, label_pattern: re.Pattern) -> list[dict]:
    """Installed matching tests, fewest tests in a directory first."""
    groups = []
    bin_dir = rocm_root / "bin"
    for ctest_file in sorted(bin_dir.rglob("CTestTestfile.cmake")):
        tests = [
            test
            for test in parse_ctest_file(
                ctest_file.read_text(encoding="utf-8", errors="replace")
            ).values()
            if selects_label(test, label_pattern)
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


def cmake_true(value: str) -> bool:
    return value.upper() not in ("", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND") and not value.upper().endswith("-NOTFOUND")


def test_command(rocjitsu, config, rocjitsu_args, test):
    if rocjitsu is None:
        return list(test["command"])
    return [str(rocjitsu), "--config", str(config), *rocjitsu_args, "--", *test["command"]]


def scheduling_policy(tests: list[dict], requested: int) -> tuple[int, list[str]]:
    reasons = []
    for test in tests:
        props = test.get("properties", {})
        if cmake_true(props.get("DISABLED", "")):
            continue
        if cmake_true(props.get("RUN_SERIAL", "")):
            reasons.append(f"{test['name']}: RUN_SERIAL")
        for key in ("RESOURCE_LOCK", "DEPENDS", "RESOURCE_GROUPS", "FIXTURES_SETUP", "FIXTURES_REQUIRED", "FIXTURES_CLEANUP"):
            if props.get(key):
                reasons.append(f"{test['name']}: {key}={props[key]}")
    return (1 if reasons else requested), reasons


def run_one(rocjitsu: Path | None, config: Path | None, rocjitsu_args: list[str], test: dict, cwd: Path, env: dict, log_path: Path) -> dict:
    command = test_command(rocjitsu, config, rocjitsu_args, test)
    work = Path(test["working_directory"]) if test["working_directory"] else cwd
    if test["working_directory"] and not work.is_absolute():
        work = cwd / work
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"command": command, "working_directory": str(work), "log": str(log_path)}
    if cmake_true(test.get("properties", {}).get("DISABLED", "")):
        log_path.write_text("Not executed: CTest DISABLED property is true.\n", encoding="utf-8")
        return {"name": test["name"], "status": "skipped", "seconds": 0.0,
                "returncode": None, "timeout": None, "reason": "CTest DISABLED", **metadata}
    started = time.monotonic()
    print(
        f"START {test['name']} timeout=disabled cmd={shlex.join(test['command'])}",
        flush=True,
    )
    try:
        proc = subprocess.run(
            command, cwd=work, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        output = proc.stdout or ""
        returncode = proc.returncode
    except OSError as error:
        output = f"Could not launch test: {error}\n"
        returncode = 127
    status = "passed" if returncode == 0 else "failed"
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
        "timeout": None,
        **metadata,
    }


def render_summary(rows: list[dict], label: str) -> str:
    lines = [
        f"# rocJITsu {label}",
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
    parser.add_argument("--rocjitsu")
    parser.add_argument("--config")
    parser.add_argument("--rocm-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-regex", default="^ffm-quick$", help="CTest label regex; for example ^quick$ or ^ffm-quick$")
    parser.add_argument("--component", help="Exact component name; omit with --list-components")
    parser.add_argument("--rocjitsu-arg", action="append", default=[], help="Extra rocJITsu argument; repeat for multiple flags")
    parser.add_argument("--direct", action="store_true", help="Run test commands directly on hardware without rocJITsu")
    parser.add_argument("--parallel", type=int, default=1, help="Maximum concurrent test commands per component")
    parser.add_argument("--component-filter", help="Comma-separated exact component paths to include")
    parser.add_argument("--list-components", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    if not args.direct and (not args.rocjitsu or not args.config):
        parser.error("--rocjitsu and --config are required unless --direct is set")

    rocm_root = Path(args.rocm_root).resolve()
    label_pattern = re.compile(args.label_regex)
    groups = discover(rocm_root, label_pattern)
    if args.component_filter:
        selected = {item.strip() for item in args.component_filter.split(",") if item.strip()}
        groups = [group for group in groups if group["component"] in selected]
    if args.list_components:
        print(json.dumps([
            {
                "component": group["component"],
                "id": re.sub(r"[^A-Za-z0-9_.-]+", "_", group["component"]),
            }
            for group in groups
        ]))
        return 0
    if args.component:
        groups = [group for group in groups if group["component"] == args.component]
    elif not args.list_only:
        raise SystemExit("--component is required unless --list-components or --list-only is used")
    if not groups:
        raise SystemExit(f"No installed tests match component={args.component!r}, label={args.label_regex!r}")
    total = sum(len(group["tests"]) for group in groups)
    print(f"{args.label_regex} directories: {len(groups)} tests: {total}", flush=True)
    for group in groups:
        print(f"  {len(group['tests']):4d}  {group['component']}", flush=True)
    if args.list_only:
        return 0 if groups else 1

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(rocm_root)
    rocjitsu = None if args.direct else Path(args.rocjitsu).resolve()
    config = None if args.direct else Path(args.config).resolve()
    inventory = [
        {"component": group["component"], "directory": str(group["directory"]),
         "tests": [dict(test, launch_command=test_command(rocjitsu, config, args.rocjitsu_arg, test))
                   for test in group["tests"]]}
        for group in groups
    ]
    (out_dir / "selected-tests.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    (out_dir / "execution-config.json").write_text(json.dumps({
        "direct": args.direct, "rocjitsu": str(rocjitsu) if rocjitsu else None,
        "config": str(config) if config else None,
        "config_contents": json.loads(config.read_text()) if config else None,
        "rocjitsu_args": args.rocjitsu_arg, "requested_parallel": args.parallel,
        "label_regex": args.label_regex, "rocm_root": str(rocm_root),
    }, indent=2) + "\n", encoding="utf-8")
    rows = []
    any_problem = False
    for group in groups:
        started = time.monotonic()
        parallel, sequential_reasons = scheduling_policy(group["tests"], args.parallel)
        print(f"COMPONENT {group['component']} tests={len(group['tests'])}", flush=True)
        if sequential_reasons:
            print("Using sequential execution: " + "; ".join(sequential_reasons), flush=True)
        def execute(index_and_test):
            index, test = index_and_test
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", test["name"])[:120]
            log_path = out_dir / "logs" / group["component"] / f"{index:05d}-{safe}.log"
            return run_one(rocjitsu, config, args.rocjitsu_arg, test, group["directory"], env, log_path)
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            results = list(executor.map(execute, enumerate(group["tests"])))
        any_problem |= any(result["status"] in ("failed", "timeout") for result in results)
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
            "requested_parallel": args.parallel,
            "effective_parallel": parallel,
            "sequential_reasons": sequential_reasons,
        }
        rows.append(row)
        summary = render_summary(rows, args.label_regex)
        (out_dir / "summary.md").write_text(summary, encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        print(summary, flush=True)
    return 1 if any_problem or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
