#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Developer entrypoint for the rocKE library tests. Ordinary pytest discovery
# is the fast unit tier. Resource-bound exhaustive checks are non-discovered
# modules invoked as explicit stages here and as separate installed CTests.
#
# Usage:
#   python library/tests/run_all.py                 # affordable pytest + host gates
#   python library/tests/run_all.py --pytest-only   # affordable discovered pytest
#   python library/tests/run_all.py --exhaustive    # include large structural matrices
#   python library/tests/run_all.py --gate counted-structural
#   python library/tests/run_all.py --gate counted-compiler --require-compiler-gate
#   python library/tests/run_all.py --gate counted-compiler --gate-arch gfx942 \
#       --allow-partial-gate                 # focused diagnostic, never GREEN
#   python library/tests/run_all.py --gpu           # pytest + GPU tests + host gates
#   python library/tests/run_all.py --gpu-only      # only GPU-marked discovered pytest
#   python library/tests/run_all.py --only dense    # focused pytest; gates omitted

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from ._explicit_gate_policy import (
        preflight_is_disabled as _preflight_is_disabled,
    )
    from ._explicit_gate_policy import selection_is_partial as _selection_is_partial
    from ._worker_process import available_cpus, run_bounded_to_stream, stop_all
else:
    from _explicit_gate_policy import preflight_is_disabled as _preflight_is_disabled
    from _explicit_gate_policy import selection_is_partial as _selection_is_partial
    from _worker_process import available_cpus, run_bounded_to_stream, stop_all

TESTS = Path(__file__).resolve().parent
_SOURCE_PLATFORM_PYTHON = TESTS.parents[1] / "platform" / "python"
if _SOURCE_PLATFORM_PYTHON.is_dir() and str(_SOURCE_PLATFORM_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SOURCE_PLATFORM_PYTHON))

from rocke.tests.subprocess_launcher import (  # noqa: E402 - source bootstrap above
    rocke_python_command,
    rocke_subprocess_env,
)

EXPLICIT_GATES = {
    "counted-structural": TESTS / "gates" / "attention_bwd_counted_structural_gate.py",
    "counted-compiler": TESTS / "gates" / "attention_bwd_counted_compiler_gate.py",
}
_OPTIONAL_ISA_ENV = "ROCKE_COUNTED_ISA_OPTIONAL"
_STAGE_OUTPUT_ARGS = (
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


def _has_forwarded_marker(arguments: list[str]) -> bool:
    """Whether arguments would replace the runner-owned lane expression."""
    return any(
        argument in ("-m", "--markexpr")
        or argument.startswith("--markexpr=")
        or (argument.startswith("-m") and argument != "-m")
        for argument in arguments
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


def _has_stage_output(arguments: list[str]) -> bool:
    """Whether arguments name output that one pytest stage must own."""
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in arguments
        for option in _STAGE_OUTPUT_ARGS
    )


def _gate_environment(
    name: str,
    *,
    require_compiler: bool,
    gate_arch: str | None = None,
    partial: bool = False,
) -> dict[str, str]:
    """Return an isolated gate environment with explicit compiler policy."""
    partial = partial or gate_arch is not None
    environment = dict(os.environ)
    if name in EXPLICIT_GATES:
        environment.pop("PYTEST_ADDOPTS", None)
        environment.pop("PYTEST_PLUGINS", None)
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        environment.pop("ROCKE_EXPLICIT_GATE_PREFLIGHT", None)
        environment["ROCKE_EXPLICIT_GATE_RUN"] = "1"
        environment["ROCKE_EXPLICIT_GATE_PARTIAL"] = "1" if partial else "0"
        if not partial:
            environment.pop("ROCKE_TEST_SHARD", None)
    if name == "counted-compiler":
        # This gate validates the caller-selected ROCm/comgr toolchain rather
        # than replacing it with torch's bundled compiler.
        environment["ROCKE_BWD_NO_COMGR_PIN"] = "1"
        environment.pop("ROCKE_BWD_ISA_ARCH", None)
        if gate_arch is not None:
            environment["ROCKE_BWD_ISA_ARCH"] = gate_arch
        if require_compiler:
            environment.pop(_OPTIONAL_ISA_ENV, None)
        else:
            environment[_OPTIONAL_ISA_ENV] = "1"
    return environment


@dataclass(frozen=True)
class _StageResult:
    code: int
    skipped: bool = False


@dataclass(frozen=True)
class _FileResult:
    path: Path
    code: int
    elapsed: float
    log: Path
    timed_out: bool = False


def _lane_marker(*, gpu: bool, gpu_only: bool, exhaustive: bool) -> str | None:
    if gpu_only:
        return "gpu"
    if gpu:
        return (
            None if exhaustive else "gpu or (not exhaustive and not worker_integration)"
        )
    return (
        "not gpu"
        if exhaustive
        else "not gpu and not exhaustive and not worker_integration"
    )


def _pytest_environment() -> dict[str, str]:
    """Select this rocke package and clear addopts made explicit on commands."""
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    return rocke_subprocess_env(environment, python_paths=(TESTS.parent,))


def _collect_pytest_files(
    marker: str | None,
    only: str,
    extra: list[str],
    manifest: Path,
    timeout: float,
) -> tuple[int, list[Path], str]:
    """Collect once and return files owning selected items, in pytest order."""
    command = rocke_python_command(
        "-m",
        "pytest",
        str(TESTS),
        "--collect-only",
        "-q",
        "-p",
        "tests._file_manifest_plugin",
        f"--rocke-selected-file-manifest={manifest}",
        f"--rocke-expected-test-root={TESTS}",
    )
    if marker is not None:
        command += ["-m", marker]
    if only:
        command += ["-k", only]
    command += extra
    log = manifest.with_suffix(".collection.log")
    with log.open("w+", encoding="utf-8") as stream:
        try:
            result = run_bounded_to_stream(
                command,
                cwd=TESTS.parent,
                stream=stream,
                timeout=timeout,
                env=_pytest_environment(),
            )
        except subprocess.TimeoutExpired:
            return (
                124,
                [],
                f"collection exceeded the affordable-lane budget ({timeout:g}s)",
            )
        stream.seek(0)
        diagnostic = stream.read()
    if result.returncode:
        return result.returncode, [], diagnostic
    files = [Path(line) for line in manifest.read_text().splitlines() if line]
    if not files:
        return 5, [], "collection selected no test files"
    return 0, files, ""


def _run_pytest_file(
    path: Path,
    marker: str | None,
    only: str,
    extra: list[str],
    timeout: float,
    log: Path,
) -> _FileResult:
    command = rocke_python_command("-m", "pytest", str(path))
    if marker is not None:
        command += ["-m", marker]
    if only:
        command += ["-k", only]
    command += extra
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        try:
            result = run_bounded_to_stream(
                command,
                cwd=TESTS.parent,
                stream=stream,
                timeout=timeout,
                env=_pytest_environment(),
            )
            return _FileResult(path, result.returncode, time.monotonic() - start, log)
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMEOUT: exceeded {timeout:g}s\n")
            return _FileResult(path, 124, time.monotonic() - start, log, True)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(TESTS))
    except ValueError:
        return str(path)


def _run_affordable_file_stage(
    *,
    marker: str | None,
    only: str,
    extra: list[str],
    jobs: int,
    timeout: float,
    lane_timeout: float,
) -> _StageResult:
    """Run selected files concurrently within file and whole-lane budgets."""
    print("\n== rocke library affordable pytest files ==", flush=True)
    started = time.monotonic()
    cleanup_grace = min(5.0, lane_timeout / 2)
    deadline = started + lane_timeout - cleanup_grace
    with tempfile.TemporaryDirectory(prefix="rocke-library-tests-") as directory:
        root = Path(directory)
        code, files, diagnostic = _collect_pytest_files(
            marker,
            only,
            extra,
            root / "manifest.txt",
            max(0.001, deadline - time.monotonic()),
        )
        if code:
            print(diagnostic, flush=True)
            return _StageResult(code)
        if time.monotonic() >= deadline:
            print(f"affordable-lane timeout: exceeded {lane_timeout:g}s in collection")
            return _StageResult(124)
        print(
            f" selected {len(files)} files; jobs={jobs}; file-timeout={timeout:g}s; "
            f"lane-timeout={lane_timeout:g}s",
            flush=True,
        )

        results: list[_FileResult] = []
        next_file = iter(enumerate(files))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
        futures: dict[concurrent.futures.Future[_FileResult], int] = {}

        def submit_one() -> bool:
            try:
                index, path = next(next_file)
            except StopIteration:
                return False
            remaining = max(0.001, deadline - time.monotonic())
            future = pool.submit(
                _run_pytest_file,
                path,
                marker,
                only,
                extra,
                min(timeout, remaining),
                root / f"{index:04d}.log",
            )
            futures[future] = index
            return True

        for _ in range(min(jobs, len(files))):
            submit_one()

        lane_timed_out = False
        try:
            while futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    lane_timed_out = True
                    break
                done, _pending = concurrent.futures.wait(
                    futures,
                    timeout=remaining,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    lane_timed_out = True
                    break
                for future in done:
                    futures.pop(future)
                    result = future.result()
                    results.append(result)
                    state = (
                        "TIMEOUT"
                        if result.timed_out
                        else ("PASS" if result.code == 0 else f"FAIL({result.code})")
                    )
                    print(
                        f" [{len(results):>3}/{len(files)}] {state:<9} "
                        f"{result.elapsed:7.2f}s {_display_path(result.path)}",
                        flush=True,
                    )
                    submit_one()
        except KeyboardInterrupt:
            stop_all()
            raise
        finally:
            if lane_timed_out:
                stop_all()
                for future in futures:
                    future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

        if lane_timed_out:
            unfinished = len(files) - len(results)
            print(
                f"affordable-lane timeout: exceeded {lane_timeout:g}s with "
                f"{unfinished} files unfinished",
                flush=True,
            )
            return _StageResult(124)

        failed = sorted(
            (result for result in results if result.code),
            key=lambda result: str(result.path),
        )
        for result in failed:
            print(
                f"\n== failure: {_display_path(result.path)} ({result.elapsed:.2f}s) ==",
                flush=True,
            )
            print(result.log.read_text(errors="replace"), flush=True)
        print(
            f"file summary: {len(files) - len(failed)} passed, "
            f"{len(failed)} failed in {time.monotonic() - started:.2f}s",
            flush=True,
        )
        return _StageResult(1 if failed else 0)


def _run(
    label: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    allow_skip: bool = False,
) -> _StageResult:
    print(f"\n== {label} ==", flush=True)
    print(" ", " ".join(command), flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=str(TESTS.parent), env=env).returncode
    elapsed = time.monotonic() - started
    if result == 77 and allow_skip:
        print(f"{label}: SKIPPED in {elapsed:.2f}s", flush=True)
        return _StageResult(0, skipped=True)
    state = "PASS" if result == 0 else f"FAIL({result})"
    print(f"{label}: {state} in {elapsed:.2f}s", flush=True)
    return _StageResult(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="rocKE library validation runner")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="include the `gpu`-marked numeric pytest lane",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="run only the `gpu`-marked discovered pytest lane",
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
        help="seconds allowed for one file in the fresh-process pytest lane "
        "(default: 60, or 600 with GPU tests or --exhaustive)",
    )
    parser.add_argument(
        "--lane-timeout",
        type=float,
        default=300.0,
        help="seconds allowed for collection plus all affordable file workers "
        "(default: 300)",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="include exhaustive structural matrices in discovered pytest; these "
        "are excluded from the affordable default developer lane",
    )
    parser.add_argument(
        "--pytest-only",
        action="store_true",
        help="run affordable discovered pytest only; omit explicit gates",
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
        help="select one counted-compiler architecture for an explicitly partial gate run",
    )
    parser.add_argument(
        "--only",
        default="",
        help="pytest -k expression for focused iteration; explicit gates omitted",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra pytest arguments, after --, applied to every selected stage",
    )
    args = parser.parse_args()
    run_started = time.monotonic()
    run_discovered_pytest = not args.gate
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
    inherited_shard = os.environ.get("ROCKE_TEST_SHARD", "").strip()
    gpu_selected = args.gpu or args.gpu_only
    jobs = (
        args.jobs
        if args.jobs is not None
        else (1 if gpu_selected else min(8, available_cpus()))
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
    if args.lane_timeout <= 0:
        parser.error("--lane-timeout must be positive")
    if args.pytest_only and args.gate:
        parser.error("--pytest-only and --gate are mutually exclusive")
    if args.gpu_only and args.gate:
        parser.error("--gpu-only and --gate are mutually exclusive")
    if args.gate_arch and (
        not args.allow_partial_gate
        or not args.gate
        or "counted-compiler" not in args.gate
    ):
        parser.error("--gate-arch needs --gate counted-compiler --allow-partial-gate")
    if args.require_compiler_gate and (
        args.pytest_only
        or args.gpu_only
        or args.only
        or (args.gate and "counted-compiler" not in args.gate)
    ):
        parser.error("--require-compiler-gate needs the counted-compiler gate")

    if run_discovered_pytest and jobs and inherited_shard:
        parser.error(
            "ROCKE_TEST_SHARD requires --jobs 0 so collection and execution use "
            "one stable shard; a sharded run reports PARTIAL"
        )
    if run_discovered_pytest and _has_forwarded_marker(worker_arguments):
        parser.error(
            "forwarded pytest marker expressions would replace the affordable-lane "
            "selection; use --gpu, --gpu-only, or --exhaustive"
        )
    if (
        run_discovered_pytest
        and jobs
        and any(
            argument in ("-n", "--numprocesses", "--dist")
            or argument.startswith(("-n=", "--numprocesses=", "--dist="))
            or (argument.startswith("-n") and argument != "-n")
            for argument in worker_arguments
        )
    ):
        parser.error(
            "xdist options require --jobs 0; --jobs already runs files concurrently"
        )
    if (
        run_discovered_pytest
        and jobs
        and _requires_single_pytest_process(worker_arguments)
    ):
        parser.error(
            "suite-wide pytest output/stop options require --jobs 0 so one process "
            "owns their result"
        )
    pytest_filtered = bool(inherited_shard) or _selection_is_partial(worker_arguments)
    gate_filtered = _selection_is_partial(extra)
    preflight_disabled = _preflight_is_disabled(extra)
    gates_selected = bool(args.gate) or not (
        args.pytest_only or args.gpu_only or args.only
    )
    if gates_selected and preflight_disabled:
        parser.error("explicit gates do not allow --noconftest or --confcutdir")
    if gates_selected and gate_filtered and not args.allow_partial_gate:
        parser.error(
            "pytest selection arguments need --allow-partial-gate; a partial run is "
            "not a gate verdict"
        )
    selected_stage_count = (1 if run_discovered_pytest else 0) + (
        len(args.gate) if args.gate else (2 if gates_selected else 0)
    )
    if selected_stage_count > 1 and _has_stage_output(extra):
        parser.error(
            "JUnit output needs a single selected stage; use --pytest-only or one --gate"
        )

    status = 0
    skipped = False
    if run_discovered_pytest:
        marker = _lane_marker(
            gpu=args.gpu,
            gpu_only=args.gpu_only,
            exhaustive=args.exhaustive,
        )
        if jobs:
            pytest_result = _run_affordable_file_stage(
                marker=marker,
                only=args.only,
                extra=worker_arguments,
                jobs=jobs,
                timeout=file_timeout,
                lane_timeout=args.lane_timeout,
            )
        else:
            command = rocke_python_command("-m", "pytest", str(TESTS))
            if marker is not None:
                command += ["-m", marker]
            if args.only:
                command += ["-k", args.only]
            command += worker_arguments
            pytest_result = _run(
                "rocke library discovered pytest",
                command,
                env=_pytest_environment(),
            )
        status = status or pytest_result.code
        skipped |= pytest_result.skipped
        if status and not args.keep_going:
            print(
                "\nRESULT: RED (resource gates not run after discovered-pytest failure) "
                f"in {time.monotonic() - run_started:.2f}s",
                flush=True,
            )
            return status

    if args.gate:
        gates = args.gate
    elif args.pytest_only or args.gpu_only or args.only:
        gates = []
    else:
        gates = list(EXPLICIT_GATES)

    for name in gates:
        environment = _gate_environment(
            name,
            require_compiler=args.require_compiler_gate,
            gate_arch=args.gate_arch if name == "counted-compiler" else None,
            partial=gate_filtered or args.gate_arch is not None,
        )
        command = rocke_python_command("-m", "pytest", str(EXPLICIT_GATES[name]))
        command += extra
        with tempfile.TemporaryDirectory(prefix=f"rocke-{name}-") as directory:
            sentinel = Path(directory) / "session.json"
            environment["ROCKE_EXPLICIT_GATE_SENTINEL"] = str(sentinel)
            gate_result = _run(
                f"rocke library gate: {name}",
                command,
                env=environment,
                allow_skip=(
                    name == "counted-compiler" and not args.require_compiler_gate
                ),
            )
            if gate_result.code == 0 and not gate_result.skipped:
                try:
                    attestation = json.loads(sentinel.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    attestation = {}
                if attestation.get("collected", 0) <= 0:
                    print(
                        f"rocke library gate: {name}: FAIL(no executed pytest session)",
                        flush=True,
                    )
                    gate_result = _StageResult(1)
        status = status or gate_result.code
        skipped |= gate_result.skipped
        if status and not args.keep_going:
            break

    if status:
        verdict = "RED"
    elif pytest_filtered or gate_filtered or args.gate_arch or args.only:
        verdict = "PARTIAL"
    elif skipped:
        verdict = "INCOMPLETE"
    else:
        verdict = "GREEN"
    print(
        f"\nRESULT: {verdict} in {time.monotonic() - run_started:.2f}s",
        flush=True,
    )
    return status


def _terminate_workers(signum, _frame) -> None:
    """Ensure an outer timeout cannot orphan detached worker process groups."""
    if stop_all():
        return
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate_workers)
    raise SystemExit(main())
