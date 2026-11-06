#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Developer entrypoint for the rocKE library tests. Ordinary pytest discovery
# is the fast unit tier. Resource-bound exhaustive checks are non-discovered
# modules invoked as explicit stages here and as separate installed CTests.
#
# Usage:
#   python library/tests/run_all.py                 # units + explicit host gates
#   python library/tests/run_all.py --unit-only     # affordable discovered pytest
#   python library/tests/run_all.py --exhaustive    # include large structural matrices
#   python library/tests/run_all.py --gate counted-ir
#   python library/tests/run_all.py --gate counted-isa --require-compiler-gate
#   python library/tests/run_all.py --gate counted-isa --gate-arch gfx942 \
#       --allow-partial-gate                 # focused diagnostic, never GREEN
#   python library/tests/run_all.py --gpu           # units + GPU tests + host gates
#   python library/tests/run_all.py --gpu-only      # only the GPU-marked unit lane
#   python library/tests/run_all.py --only dense    # focused units; gates omitted
#   python library/tests/run_all.py --allow-incomplete  # INCOMPLETE exits 0
#
# The verdict and exit code come from rocke.tests.runner_verdict: GREEN and
# PARTIAL (any narrowing flag) exit 0, RED 1, INCOMPLETE (a skipped gate, a
# stage that executed no tests, or stages not run after a failure) 3.

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# This script runs from source and is not a package module: put the platform
# Python root on sys.path the way library/tests/conftest.py does (library ->
# platform is the allowed direction).
_PYROOT = Path(__file__).resolve().parents[2] / "platform" / "python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))

from rocke.tests.runner_verdict import (
    Outcome,
    RunRecord,
    junit_counts,
    line_buffered_stdout,
)

TESTS = Path(__file__).resolve().parent
EXPLICIT_GATES = {
    "counted-ir": TESTS / "gates" / "attention_bwd_counted_ir_gate.py",
    "counted-isa": TESTS / "gates" / "attention_bwd_counted_isa_gate.py",
}
_OPTIONAL_ISA_ENV = "ROCKE_COUNTED_ISA_OPTIONAL"
_PREFLIGHT_BYPASS_ARGS = ("--noconftest", "--confcutdir")
_SELECTION_ARGS = (
    "-k",
    "-m",
    "--collect-only",
    "--co",
    "--ignore",
    "--ignore-glob",
    "--deselect",
    "--lf",
    "--last-failed",
    "--ff",
    "--failed-first",
    "--nf",
    "--new-first",
    "--setup-only",
    "--setup-plan",
    "--stepwise",
    "--sw",
    "--stepwise-skip",
)
_SHORT_SELECTION_ARGS = ("-k", "-m")
_OPTIONS_WITH_SEPARATE_VALUES = (
    "-k",
    "-m",
    "--ignore",
    "--ignore-glob",
    "--deselect",
    "--basetemp",
    "--log-file",
    "--junitxml",
    "--junit-xml",
    "--json-report-file",
    "--html",
    "--cov",
    "--cov-report",
    "--cov-config",
    "--cov-fail-under",
    "--cov-context",
    "--maxfail",
    "-n",
    "--numprocesses",
    "--dist",
)
_SINGLE_PROCESS_ARGS = (
    "--basetemp",
    "--log-file",
    "--junitxml",
    "--junit-xml",
    "--json-report-file",
    "--html",
    "--cov",
    "--cov-report",
    "--cov-config",
    "--cov-append",
    "--maxfail",
    "--exitfirst",
    "-x",
)


def _positional_pytest_selectors(arguments: list[str]) -> list[str]:
    """Positional selectors that cannot be replayed for every file worker.

    Unknown option-value forms are deliberately conservative: use ``--jobs 0``
    when forwarding an option whose separate value is not listed above.
    """
    selectors = []
    consumes_value = False
    for argument in arguments:
        if consumes_value:
            consumes_value = False
            continue
        if argument in _OPTIONS_WITH_SEPARATE_VALUES:
            consumes_value = True
        elif not argument.startswith("-"):
            selectors.append(argument)
    return selectors


def _forwarded_gpu_marker(arguments: list[str]) -> bool:
    """Whether a forwarded pytest marker expression can select GPU tests."""
    for index, argument in enumerate(arguments):
        if argument in ("-m", "--markexpr") and index + 1 < len(arguments):
            if "gpu" in arguments[index + 1].lower():
                return True
        elif argument.startswith(("-m", "--markexpr=")) and "gpu" in argument.lower():
            return True
    return False


def _selection_is_partial(arguments: list[str]) -> bool:
    """Whether pytest arguments can suppress part or all of an explicit gate."""
    return bool(_positional_pytest_selectors(arguments)) or any(
        argument == option
        or argument.startswith(option + "=")
        or (
            option in _SHORT_SELECTION_ARGS
            and argument.startswith(option)
            and argument != option
        )
        for argument in arguments
        for option in _SELECTION_ARGS
    )


def _requires_single_pytest_process(arguments: list[str]) -> bool:
    """Whether pytest arguments have suite-wide output or stop semantics."""
    return any(
        argument == option
        or argument.startswith(option + "=")
        or (option == "-x" and argument.startswith("-x") and argument != "-x")
        for argument in arguments
        for option in _SINGLE_PROCESS_ARGS
    )


def _preflight_is_disabled(arguments: list[str]) -> bool:
    """Whether pytest arguments suppress an explicit gate's conftest."""
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in arguments
        for option in _PREFLIGHT_BYPASS_ARGS
    )


def _gate_environment(
    name: str,
    *,
    require_compiler: bool,
    gate_arch: str | None = None,
    partial: bool = False,
) -> dict[str, str]:
    """Return an isolated gate environment with explicit compiler policy."""
    environment = dict(os.environ)
    if name in EXPLICIT_GATES:
        environment.pop("PYTEST_ADDOPTS", None)
        environment.pop("ROCKE_EXPLICIT_GATE_PREFLIGHT", None)
        environment["ROCKE_EXPLICIT_GATE_RUN"] = "1"
        environment["ROCKE_EXPLICIT_GATE_PARTIAL"] = "1" if partial else "0"
    if name == "counted-isa":
        # This gate validates the caller-selected ROCm/COMGR toolchain rather
        # than replacing it with torch's bundled compiler.
        environment["ROCKE_BWD_NO_COMGR_PIN"] = "1"
        environment.pop("ROCKE_BWD_ISA_ARCH", None)
        environment.pop("ROCKE_COUNTED_ISA_PARTIAL", None)
        if gate_arch is not None:
            environment["ROCKE_BWD_ISA_ARCH"] = gate_arch
            environment["ROCKE_COUNTED_ISA_PARTIAL"] = "1"
        if require_compiler:
            environment.pop(_OPTIONAL_ISA_ENV, None)
        else:
            environment[_OPTIONAL_ISA_ENV] = "1"
    return environment


_UNIT_FILES_STAGE = "rocke library affordable pytest files"
_UNIT_STAGE = "rocke library unit pytest"


@dataclass(frozen=True)
class _FileResult:
    path: Path
    code: int
    elapsed: float
    log: Path
    timed_out: bool = False
    # Tests the file's JUnit report says ran; ``None`` when unknown.
    executed: int | None = None


def _unit_marker(*, gpu: bool, gpu_only: bool, exhaustive: bool) -> str | None:
    if gpu_only:
        return "gpu"
    if gpu:
        return None if exhaustive else "gpu or not exhaustive"
    return "not gpu" if exhaustive else "not gpu and not exhaustive"


def _collect_unit_files(
    marker: str | None,
    only: str,
    extra: list[str],
    manifest: Path,
) -> tuple[int, list[Path], str]:
    """Collect once and return files owning selected items, in pytest order."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(TESTS),
        "--collect-only",
        "-q",
        "-p",
        "tests._file_manifest_plugin",
        f"--rocke-selected-file-manifest={manifest}",
    ]
    if marker is not None:
        command += ["-m", marker]
    if only:
        command += ["-k", only]
    command += extra
    result = subprocess.run(
        command,
        cwd=str(TESTS.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return result.returncode, [], result.stdout + result.stderr
    files = [Path(line) for line in manifest.read_text().splitlines() if line]
    if not files:
        return 5, [], "collection selected no test files"
    return 0, files, ""


def _forwarded_junit(arguments: list[str]) -> Path | None:
    """The JUnit report path a caller forwarded to pytest, if any (last wins)."""
    report = None
    for index, argument in enumerate(arguments):
        for option in ("--junitxml", "--junit-xml"):
            if argument == option and index + 1 < len(arguments):
                report = arguments[index + 1]
            elif argument.startswith(option + "="):
                report = argument.split("=", 1)[1]
    if report is None:
        return None
    # pytest resolves a relative report path against its working directory.
    return TESTS.parent / report


def _junit_report(arguments: list[str], default: Path) -> tuple[list[str], Path]:
    """Arguments that make pytest write a JUnit report, and that report's path.

    A report the caller already forwarded is read instead of adding another.
    """
    forwarded = _forwarded_junit(arguments)
    if forwarded is not None:
        return [], forwarded
    return [f"--junitxml={default}"], default


def _executed(report: Path | None) -> int | None:
    """Tests a JUnit report says ran, or ``None`` when unknown."""
    if report is None:
        return None
    counts = junit_counts(report)
    return None if counts is None else counts[0]


def _run_unit_file(
    path: Path,
    marker: str | None,
    only: str,
    extra: list[str],
    timeout: float,
    log: Path,
) -> _FileResult:
    # The runner's own per-file report, beside the log. The caller's
    # --junitxml is rejected for file workers before this lane starts.
    report = log.with_suffix(".xml")
    command = [sys.executable, "-m", "pytest", str(path), f"--junitxml={report}"]
    if marker is not None:
        command += ["-m", marker]
    if only:
        command += ["-k", only]
    command += extra
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        try:
            result = subprocess.run(
                command,
                cwd=str(TESTS.parent),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            return _FileResult(
                path,
                result.returncode,
                time.monotonic() - start,
                log,
                executed=_executed(report),
            )
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMEOUT: exceeded {timeout:g}s\n")
            return _FileResult(path, 124, time.monotonic() - start, log, True)


def _run_unit_file_stage(
    record: RunRecord,
    *,
    marker: str | None,
    only: str,
    extra: list[str],
    jobs: int,
    timeout: float,
) -> Outcome:
    """Run each selected test file in a fresh process with bounded concurrency."""
    print(f"\n== {_UNIT_FILES_STAGE} ==", flush=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rocke-library-tests-") as directory:
        root = Path(directory)
        code, files, diagnostic = _collect_unit_files(
            marker, only, extra, root / "manifest.txt"
        )
        if code:
            print(diagnostic, flush=True)
            return record.returncode(_UNIT_FILES_STAGE, code)
        print(
            f" selected {len(files)} files; jobs={jobs}; file-timeout={timeout:g}s",
            flush=True,
        )
        results: list[_FileResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    _run_unit_file,
                    path,
                    marker,
                    only,
                    extra,
                    timeout,
                    root / f"{index:04d}.log",
                ): index
                for index, path in enumerate(files)
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                state = (
                    "TIMEOUT"
                    if result.timed_out
                    else ("PASS" if result.code == 0 else f"FAIL({result.code})")
                )
                print(
                    f" [{completed:>3}/{len(files)}] {state:<9} "
                    f"{result.elapsed:7.2f}s {result.path.name}",
                    flush=True,
                )

        failed = sorted(
            (result for result in results if result.code),
            key=lambda result: str(result.path),
        )
        for result in failed:
            print(
                f"\n== failure: {result.path} ({result.elapsed:.2f}s) ==",
                flush=True,
            )
            print(result.log.read_text(errors="replace"), flush=True)
        counts = [result.executed for result in results]
        executed = None if None in counts else sum(counts)
        tests = "" if executed is None else f"; {executed} tests executed"
        print(
            f"unit-file summary: {len(files) - len(failed)} passed, "
            f"{len(failed)} failed{tests} in {time.monotonic() - started:.2f}s",
            flush=True,
        )
        return record.returncode(
            _UNIT_FILES_STAGE, 1 if failed else 0, executed=executed
        )


def _run(
    label: str,
    command: list[str],
    record: RunRecord,
    *,
    env: dict[str, str] | None = None,
    allow_skip: bool = False,
    junit: Path | None = None,
) -> Outcome:
    """Run one pytest process as a stage; ``junit`` is its report, if it writes one."""
    print(f"\n== {label} ==", flush=True)
    print(" ", " ".join(command), flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=str(TESTS.parent), env=env).returncode
    elapsed = time.monotonic() - started
    outcome = record.returncode(
        label, result, skip_allowed=allow_skip, executed=_executed(junit)
    )
    state = {
        Outcome.PASSED: "PASS",
        Outcome.SKIPPED: "SKIPPED",
        Outcome.EMPTY: "EMPTY (executed no tests)",
    }.get(outcome, f"FAIL({result})")
    print(f"{label}: {state} in {elapsed:.2f}s", flush=True)
    return outcome


def main() -> int:
    line_buffered_stdout()
    parser = argparse.ArgumentParser(description="rocKE library validation runner")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="include the `gpu`-marked numeric unit lane",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="run only the `gpu`-marked unit lane",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue into resource-bound gates after an earlier stage fails",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="fresh test-file processes to run concurrently (default: up to 8 for "
        "host-only runs and 1 when GPU tests are included; 0 uses one traditional "
        "pytest process for focused debugging)",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=None,
        help="seconds allowed for one file in the fresh-process unit lane "
        "(default: 60, or 600 with GPU tests or --exhaustive)",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="include exhaustive structural matrices in discovered pytest; these "
        "are excluded from the affordable default developer lane",
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="run discovered pytest only; omit explicit gates",
    )
    parser.add_argument(
        "--gate",
        action="append",
        choices=tuple(EXPLICIT_GATES),
        help="run only this explicit gate; repeat to select several",
    )
    parser.add_argument(
        "--require-compiler-gate",
        action="store_true",
        help="fail when the counted ISA gate's qualified toolchain is unavailable; "
        "the default local run reports that gate as skipped",
    )
    parser.add_argument(
        "--allow-partial-gate",
        action="store_true",
        help="allow pytest selection arguments for focused gate debugging; the result "
        "is not a complete gate verdict",
    )
    parser.add_argument(
        "--gate-arch",
        choices=("gfx942", "gfx950"),
        help="select one counted-isa architecture for an explicitly partial gate run",
    )
    parser.add_argument(
        "--only",
        default="",
        help="pytest -k expression for focused unit iteration; explicit gates omitted",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit 0 when the verdict is INCOMPLETE (still printed as INCOMPLETE); "
        "never hides RED",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra pytest arguments, after --, applied to every selected stage",
    )
    args = parser.parse_args()
    run_started = time.monotonic()
    extra = args.pytest_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    try:
        inherited = shlex.split(os.environ.get("PYTEST_ADDOPTS", ""))
    except ValueError as exc:
        parser.error(f"invalid PYTEST_ADDOPTS: {exc}")
    # Explicit gates clear inherited addopts before launch. Unit subprocesses do
    # inherit them, so resource and shared-output policy must inspect both.
    worker_arguments = ([] if args.gate else inherited) + extra
    gpu_selected = args.gpu or args.gpu_only or _forwarded_gpu_marker(worker_arguments)
    jobs = (
        args.jobs
        if args.jobs is not None
        else (1 if gpu_selected else min(8, os.cpu_count() or 1))
    )
    file_timeout = (
        args.file_timeout
        if args.file_timeout is not None
        else (600.0 if gpu_selected or args.exhaustive else 60.0)
    )

    if jobs < 0:
        parser.error("--jobs must be non-negative")
    if file_timeout <= 0:
        parser.error("--file-timeout must be positive")
    if args.unit_only and args.gate:
        parser.error("--unit-only and --gate are mutually exclusive")
    if args.gpu_only and args.gate:
        parser.error("--gpu-only and --gate are mutually exclusive")
    if args.gate_arch and (
        not args.allow_partial_gate or not args.gate or "counted-isa" not in args.gate
    ):
        parser.error("--gate-arch needs --gate counted-isa --allow-partial-gate")
    if args.require_compiler_gate and (
        args.unit_only
        or args.gpu_only
        or args.only
        or (args.gate and "counted-isa" not in args.gate)
    ):
        parser.error("--require-compiler-gate needs the counted-isa gate")

    if jobs and any(
        argument in ("-n", "--numprocesses", "--dist")
        or argument.startswith(("-n=", "--numprocesses=", "--dist="))
        or (argument.startswith("-n") and argument != "-n")
        for argument in worker_arguments
    ):
        parser.error(
            "xdist options require --jobs 0; --jobs already runs files concurrently"
        )
    if jobs and _positional_pytest_selectors(worker_arguments):
        parser.error(
            "positional pytest paths/node IDs require --jobs 0; use --only for "
            "file-isolated focused runs"
        )
    if jobs and _requires_single_pytest_process(worker_arguments):
        parser.error(
            "suite-wide pytest output/stop options require --jobs 0 so one process "
            "owns their result"
        )
    filtered = _selection_is_partial(extra)
    preflight_disabled = _preflight_is_disabled(extra)
    gates_selected = bool(args.gate) or not (
        args.unit_only or args.gpu_only or args.only
    )
    if gates_selected and preflight_disabled:
        parser.error("explicit gates do not allow --noconftest or --confcutdir")
    if gates_selected and filtered and not args.allow_partial_gate:
        parser.error(
            "pytest selection arguments need --allow-partial-gate; a partial run is "
            "not a gate verdict"
        )

    record = RunRecord(allow_incomplete=args.allow_incomplete)
    run_units = not args.gate
    for flag, narrowed in (
        ("--unit-only", args.unit_only),
        ("--gpu-only", args.gpu_only),
        ("--only", bool(args.only)),
        # Explicit gates omit the unit lane and every gate not named.
        ("--gate", bool(args.gate)),
        ("pytest selection arguments", filtered),
        ("PYTEST_ADDOPTS selection", run_units and _selection_is_partial(inherited)),
        ("--gate-arch", args.gate_arch is not None),
    ):
        if narrowed:
            record.partial(flag)

    if args.gate:
        gates = args.gate
    elif args.unit_only or args.gpu_only or args.only:
        gates = []
    else:
        gates = list(EXPLICIT_GATES)

    with tempfile.TemporaryDirectory(prefix="rocke-library-reports-") as directory:
        reports = Path(directory)
        stages: list[tuple[str, Callable[[], Outcome]]] = []
        if run_units:
            marker = _unit_marker(
                gpu=args.gpu,
                gpu_only=args.gpu_only,
                exhaustive=args.exhaustive,
            )
            if jobs:
                stages.append(
                    (
                        _UNIT_FILES_STAGE,
                        functools.partial(
                            _run_unit_file_stage,
                            record,
                            marker=marker,
                            only=args.only,
                            extra=extra,
                            jobs=jobs,
                            timeout=file_timeout,
                        ),
                    )
                )
            else:
                junit_args, junit = _junit_report(
                    worker_arguments, reports / "unit.xml"
                )
                command = [sys.executable, "-m", "pytest", str(TESTS), *junit_args]
                if marker is not None:
                    command += ["-m", marker]
                if args.only:
                    command += ["-k", args.only]
                command += extra
                stages.append(
                    (
                        _UNIT_STAGE,
                        functools.partial(
                            _run, _UNIT_STAGE, command, record, junit=junit
                        ),
                    )
                )

        for name in gates:
            label = f"rocke library gate: {name}"
            environment = _gate_environment(
                name,
                require_compiler=args.require_compiler_gate,
                gate_arch=args.gate_arch if name == "counted-isa" else None,
                partial=filtered or args.gate_arch is not None,
            )
            junit_args, junit = _junit_report(extra, reports / f"{name}.xml")
            command = [sys.executable, "-m", "pytest", str(EXPLICIT_GATES[name])]
            command += junit_args + extra
            stages.append(
                (
                    label,
                    functools.partial(
                        _run,
                        label,
                        command,
                        record,
                        env=environment,
                        allow_skip=name == "counted-isa"
                        and not args.require_compiler_gate,
                        junit=junit,
                    ),
                )
            )

        for index, (_, run) in enumerate(stages):
            if run() is Outcome.FAILED and not args.keep_going:
                for later, _ in stages[index + 1 :]:
                    record.not_run(later, "an earlier stage failed")
                break

    print(
        f"\nlibrary run_all finished in {time.monotonic() - run_started:.2f}s",
        flush=True,
    )
    return record.report("library run_all")


if __name__ == "__main__":
    raise SystemExit(main())
