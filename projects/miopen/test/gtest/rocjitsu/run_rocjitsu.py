#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the installed rocjitsu-quick MIOpen category on a requested architecture.

Consumes an installed prefix (OUTPUT_ARTIFACTS_DIR, THEROCK_BIN_DIR) and a
prepared emulator. Never fetches, installs, configures or builds. TEST_TYPE
never widens the packaged selection. Every GPU operation runs under rocjitsu.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from rocjitsu_cpu import available_cpu_count, execution_command, run_cpu_shards
from rocjitsu_reports import merge_reports
from rocjitsu_selection import build_filter, listed_tests


CATEGORY = "rocjitsu-quick"
SUITE = f"miopen_gtest_{CATEGORY}_suite"
REPORT_NAME = f"miopen_gtest_{CATEGORY}"

# Probe the actual packaged HSA runtime under the same emulator/config as the
# tests. Configuration text alone cannot prove the runtime's exposed target.
RUNTIME_PROBE = r"""
import ctypes
import sys

class Agent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]

hsa = ctypes.CDLL(sys.argv[1])
expected = sys.argv[2]
callback_type = ctypes.CFUNCTYPE(ctypes.c_uint32, Agent, ctypes.c_void_p)
hsa.hsa_init.restype = ctypes.c_uint32
hsa.hsa_shut_down.restype = ctypes.c_uint32
hsa.hsa_agent_get_info.argtypes = [Agent, ctypes.c_int, ctypes.c_void_p]
hsa.hsa_agent_get_info.restype = ctypes.c_uint32
hsa.hsa_iterate_agents.argtypes = [callback_type, ctypes.c_void_p]
hsa.hsa_iterate_agents.restype = ctypes.c_uint32

def check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed: HSA status {status}")

gpus = []
errors = []

@callback_type
def visit(agent, _):
    device = ctypes.c_int()
    name = ctypes.create_string_buffer(64)
    status = hsa.hsa_agent_get_info(agent, 17, ctypes.byref(device))
    if status == 0:
        status = hsa.hsa_agent_get_info(agent, 0, name)
    if status != 0:
        errors.append(f"hsa_agent_get_info failed: HSA status {status}")
        return status
    decoded = name.value.decode("utf-8")
    print(f"HSA agent: {decoded}, device type: {device.value}", flush=True)
    if device.value == 1:
        gpus.append(decoded)
    return 0

check(hsa.hsa_init(), "hsa_init")
try:
    check(hsa.hsa_iterate_agents(visit, None), "hsa_iterate_agents")
    if errors or gpus != [expected]:
        raise RuntimeError(f"Expected exactly one {expected} GPU agent, observed {gpus}; {errors}")
    print(f"Runtime {expected} probe passed", flush=True)
finally:
    check(hsa.hsa_shut_down(), "hsa_shut_down")
"""


def target_version(arch: str) -> int:
    """Decode AMD's gfx major/minor/revision name into its KFD version number."""
    match = re.fullmatch(r"gfx([1-9][0-9]?)([0-9a-f])([0-9a-f])", arch)
    if not match:
        raise ValueError(f"Invalid AMDGPU architecture: {arch!r}")
    major, minor, revision = match.groups()
    return int(major) * 10000 + int(minor, 16) * 100 + int(revision, 16)


def prepared_paths(arch: str) -> tuple[Path, Path, Path]:
    target_version(arch)
    work = Path("rocjitsu-work") / arch
    return (
        work.resolve(),
        (work / "build/tools/rocjitsu/rocjitsu").resolve(),
        (work / "prepared-config.json").resolve(),
    )


def validated_config(config: Path, arch: str, dispatch_threads: int) -> dict:
    data = json.loads(config.read_text())
    vm = data.get("vm", {})
    device = vm.get("gpu", {}).get("device", {})
    # Pinned profiles express the target via gfx_target_version, not vm.target.
    # An explicit target must agree; an absent version may never default to arch.
    if (
        ("target" in vm and vm["target"] != arch)
        or type(device.get("gfx_target_version")) is not int
        or device["gfx_target_version"] != target_version(arch)
    ):
        raise ValueError(f"rocjitsu configuration does not target {arch}: {config}")
    data["vm"]["gpu"]["num_gpus"] = 1
    data["num_threads"] = 1
    data["async_helper_threads"] = 0
    data["cpu_dispatch_threads"] = dispatch_threads
    return data


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    if policy.get("category") != CATEGORY or not isinstance(
        policy.get("representative_cases"), dict
    ):
        raise ValueError(
            f"Expected exported {CATEGORY} representative-case policy: {path}"
        )
    return policy["representative_cases"]


def suite_metadata(test_dir: Path, reports: Path):
    command = [
        "ctest",
        "--test-dir",
        str(test_dir),
        "--show-only=json-v1",
        "-R",
        f"^{SUITE}$",
        "-L",
        f"^{CATEGORY}$",
    ]
    print(f"+ {' '.join(command)}", flush=True)
    result = subprocess.run(
        command, check=False, text=True, capture_output=True, timeout=60
    )
    (reports / "ctest-metadata.json").write_text(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="", flush=True)
    if result.returncode:
        print(result.stdout, end="", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)
    tests = json.loads(result.stdout)["tests"]
    if len(tests) != 1 or tests[0].get("name") != SUITE:
        raise RuntimeError(
            f"Expected exactly one installed {SUITE}, found {len(tests)}"
        )
    test = tests[0]
    command = test.get("command", [])
    if not command:
        raise RuntimeError(f"Installed {SUITE} has no executable command")
    properties = {prop["name"]: prop["value"] for prop in test.get("properties", [])}
    if CATEGORY not in properties.get("LABELS", []):
        raise RuntimeError(f"Installed {SUITE} lacks the {CATEGORY} label")
    if properties.get("DISABLED"):
        raise RuntimeError(f"Installed {SUITE} is disabled")
    working_dir = Path(properties.get("WORKING_DIRECTORY", str(test_dir)))
    if not working_dir.is_absolute():
        working_dir = test_dir / working_dir
    if not working_dir.is_dir():
        raise RuntimeError(f"Installed suite working directory missing: {working_dir}")
    return command, properties, working_dir.resolve()


def curate_command(command, launcher, env, working_dir, reports, representative_cases):
    indices = [
        i for i, value in enumerate(command) if value.startswith("--gtest_filter=")
    ]
    if len(indices) != 1:
        raise RuntimeError(f"Expected one packaged GTest filter: {command}")
    index = indices[0]
    base_filter = command[index].split("=", 1)[1]
    # Listing XML is not a verdict. GTest discovery must not write execution XML.
    discovery_env = env.copy()
    discovery_env.pop("GTEST_OUTPUT", None)
    discovery_command = execution_command(command)
    result = subprocess.run(
        launcher + discovery_command + ["--gtest_list_tests"],
        env=discovery_env,
        cwd=working_dir,
        text=True,
        capture_output=True,
        timeout=120,
    )
    (reports / "category-test-list.log").write_text(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    names = listed_tests(result.stdout)
    filtered, selected, exclusions = build_filter(
        base_filter, names, representative_cases
    )
    if not selected:
        raise RuntimeError(f"{CATEGORY} filter selects no listed tests")
    original_suites = {name.split(".", 1)[0] for name in names}
    retained_suites = {name.split(".", 1)[0] for name in selected}
    selection = {
        "base_suite": SUITE,
        "base_filter": base_filter,
        "filter": filtered,
        "category_listed": len(names),
        "selected_listed": len(selected),
        "retained_suites": len(retained_suites),
        "removed_suites": sorted(original_suites - retained_suites),
        "excluded_patterns": exclusions,
        "selected_tests": selected,
    }
    (reports / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (reports / "gtest-filter.txt").write_text(filtered + "\n")
    print(
        f"{CATEGORY} selection: {len(selected)}/{len(names)} listed tests, "
        f"{len(retained_suites)}/{len(original_suites)} suites (including disabled)",
        flush=True,
    )
    command = execution_command(command)
    index = next(
        i for i, value in enumerate(command) if value.startswith("--gtest_filter=")
    )
    command[index] = f"--gtest_filter={filtered}"
    return command, selected


def is_cpu_test(name: str) -> bool:
    suite = name.split(".", 1)[0]
    return suite.startswith("CPU_") or "/CPU_" in suite


def split_execution_commands(command, selected_names):
    """Partition without widening; do not expand thousands of GPU identities."""
    index = next(
        i for i, value in enumerate(command) if value.startswith("--gtest_filter=")
    )
    positive, _, negative = command[index].split("=", 1)[1].partition("-")
    cpu_names = [name for name in selected_names if is_cpu_test(name)]
    gpu_names = [name for name in selected_names if not is_cpu_test(name)]
    cpu_command = command.copy()
    gpu_command = command.copy()
    # Exact CPU positives intersect the base selection even when it contains
    # only a narrow CPU sub-suite.
    cpu_filter = f"{':'.join(cpu_names)}-{negative}" if cpu_names else "-*"
    cpu_command[index] = f"--gtest_filter={cpu_filter}"
    # Keep the compact curated GPU filter rather than expanding every identity.
    # Adding CPU negatives only narrows the selected category.
    gpu_negative = ":".join(part for part in (negative, "CPU_*:*/CPU_*") if part)
    gpu_command[index] = f"--gtest_filter={positive}-{gpu_negative}"
    return cpu_command, gpu_command, cpu_names, gpu_names


def command_with_filter_file(command, path):
    """Use GoogleTest's native flag file instead of an oversized argv value."""
    index = next(
        i for i, value in enumerate(command) if value.startswith("--gtest_filter=")
    )
    path.write_text(command[index] + "\n", encoding="utf-8")
    return command[:index] + command[index + 1 :] + [f"--gtest_flagfile={path}"]


def suite_environment(properties):
    """Apply CTest environment metadata before imposing run-local isolation."""
    env = os.environ.copy()
    for entry in properties.get("ENVIRONMENT", []):
        name, value = entry.split("=", 1)
        env[name] = value
    original = env.copy()
    for entry in properties.get("ENVIRONMENT_MODIFICATION", []):
        name, modification = entry.split("=", 1)
        operation, value = modification.split(":", 1)
        if operation == "reset":
            if name in original:
                env[name] = original[name]
            else:
                env.pop(name, None)
        elif operation == "set":
            env[name] = value
        elif operation == "unset":
            env.pop(name, None)
        elif operation in ("string_append", "string_prepend"):
            old = env.get(name, "")
            env[name] = old + value if operation == "string_append" else value + old
        elif operation in (
            "path_list_append",
            "path_list_prepend",
            "cmake_list_append",
            "cmake_list_prepend",
        ):
            separator = os.pathsep if operation.startswith("path_") else ";"
            old = env.get(name, "")
            parts = [old, value] if operation.endswith("append") else [value, old]
            env[name] = separator.join(part for part in parts if part)
        else:
            raise RuntimeError(f"Unsupported CTest environment modification: {entry}")
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    env.pop("HSA_DISABLE_GFX12_STRICT", None)
    return env


def run_logged(command, env, working_dir, log_path, timeout):
    """Stream/save output and bound the launcher plus its child process group."""
    print(f"+ {' '.join(map(str, command))}", flush=True)
    print(f"Output: {log_path} (timeout {timeout}s)", flush=True)
    deadline = time.monotonic() + timeout
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=working_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        def stream():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        try:
            status = process.wait(timeout=timeout)
            reader.join(timeout=max(deadline - time.monotonic(), 0))
            if reader.is_alive():
                raise subprocess.TimeoutExpired(command, timeout)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            reader.join()
            if isinstance(error, subprocess.TimeoutExpired):
                raise RuntimeError(
                    f"Command timed out after {timeout}s; see {log_path}"
                ) from error
            raise
        finally:
            process.stdout.close()
    if status:
        raise subprocess.CalledProcessError(status, command)


def summarize_report(report: Path, arch: str):
    try:
        root = ET.parse(report).getroot()
        total = int(root.attrib["tests"])
        failed = int(root.attrib["failures"])
        skipped = int(root.attrib["skipped"])
        disabled = int(root.attrib["disabled"])
        passed = int(root.attrib["passed"])
        summary = f"{REPORT_NAME}: {passed} passed, {failed} failed, {skipped} skipped"
        if disabled:
            summary += f", {disabled} disabled/not run"
        if not total:
            summary += " (FAILED: empty report)"
    except (OSError, ET.ParseError, KeyError, ValueError):
        summary = f"{REPORT_NAME}: failed to complete with a valid XML report"
    print(summary, flush=True)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as output:
            output.write(f"### MIOpen {arch} {CATEGORY} suite\n\n{summary}\n\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", required=True, help="Exact AMDGPU target, e.g. gfx950"
    )
    parser.add_argument(
        "--rocjitsu-bin", type=Path, help="Real prebuilt rocjitsu launcher"
    )
    parser.add_argument(
        "--rocjitsu-config", type=Path, help="Matching emulator configuration"
    )
    parser.add_argument(
        "--dispatch-threads",
        type=int,
        default=None,
        help="GPU dispatch workers (default: min(16, available CPU resources))",
    )
    parser.add_argument(
        "--cpu-timeout", type=int, default=600, help="Seconds per CPU shard"
    )
    parser.add_argument(
        "--gpu-timeout", type=int, default=2400, help="Seconds for the GPU phase"
    )
    args = parser.parse_args()
    try:
        workdir, default_bin, default_config = prepared_paths(args.arch)
    except ValueError as error:
        parser.error(str(error))
    cpu_count = available_cpu_count()
    dispatch_threads = (
        args.dispatch_threads
        if args.dispatch_threads is not None
        else min(16, cpu_count)
    )
    if not 1 <= dispatch_threads <= min(16, cpu_count):
        parser.error(f"--dispatch-threads must be between 1 and {min(16, cpu_count)}")
    if args.cpu_timeout <= 0 or args.gpu_timeout <= 0:
        parser.error("--cpu-timeout and --gpu-timeout must be positive")
    if bool(args.rocjitsu_bin) != bool(args.rocjitsu_config):
        parser.error("--rocjitsu-bin and --rocjitsu-config must be supplied together")
    rocjitsu = args.rocjitsu_bin.resolve() if args.rocjitsu_bin else default_bin
    config = args.rocjitsu_config.resolve() if args.rocjitsu_config else default_config
    # Configuration/artifact/launch failures must not reuse an earlier verdict.
    reports = workdir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / f"{REPORT_NAME}.xml"
    report.unlink(missing_ok=True)
    prepare_hint = (
        f"Prepare rocjitsu for {args.arch} first, or supply both prebuilt overrides."
    )
    if not rocjitsu.is_file() or not os.access(rocjitsu, os.X_OK):
        raise RuntimeError(
            f"rocjitsu launcher missing or not executable: {rocjitsu}. {prepare_hint}"
        )
    if not config.is_file():
        raise RuntimeError(f"rocjitsu config missing: {config}. {prepare_hint}")
    config_data = validated_config(config, args.arch, dispatch_threads)
    representative_cases = load_policy(Path(__file__).with_name("rocjitsu_policy.json"))

    dist = Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "./build")).resolve()
    bin_dir = Path(os.environ.get("THEROCK_BIN_DIR", str(dist / "bin"))).resolve()
    gtest = bin_dir / "miopen_gtest"
    test_dir = bin_dir / "MIOpen"
    if not gtest.is_file() or not os.access(gtest, os.X_OK):
        raise RuntimeError(f"Installed miopen_gtest missing or not executable: {gtest}")
    if not (test_dir / "CTestTestfile.cmake").is_file():
        raise RuntimeError(
            f"Installed CTest metadata missing: {test_dir / 'CTestTestfile.cmake'}"
        )
    plugin_name = f"libMIOpenCKGroupedConv_{args.arch}.so"
    plugins = list(dist.rglob(plugin_name))
    if not any(path.is_file() and path.stat().st_size for path in plugins):
        raise RuntimeError(f"{plugin_name} missing or empty in {dist}")
    print(f"{args.arch} CK plugins: {[str(path) for path in plugins]}", flush=True)
    hsa = dist / "lib/libhsa-runtime64.so.1"
    if not hsa.is_file():
        raise RuntimeError(f"Packaged HSA runtime missing: {hsa}")

    command, properties, working_dir = suite_metadata(test_dir, reports)
    executable = Path(command[0])
    if not executable.is_absolute():
        executable = working_dir / executable
    if executable.resolve() != gtest.resolve():
        raise RuntimeError(
            f"Installed {SUITE} does not invoke fetched miopen_gtest: {command[0]}"
        )

    with tempfile.TemporaryDirectory(prefix=f"miopen-{args.arch}-rocjitsu-") as temp:
        work = Path(temp)
        print(
            f"rocjitsu GPU dispatch workers: {dispatch_threads} (CPU limit {cpu_count})",
            flush=True,
        )
        used_config = reports / "rocjitsu-config.json"
        used_config.write_text(json.dumps(config_data, indent=2) + "\n")
        env = suite_environment(properties)
        env["ROCM_PATH"] = str(dist)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(dist / "lib"), str(dist / "lib64"), env.get("LD_LIBRARY_PATH", "")]
        )
        env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
        env.pop("GTEST_SHARD_STATUS_FILE", None)
        for name, directory in (
            ("MIOPEN_USER_DB_PATH", "db"),
            ("MIOPEN_CUSTOM_CACHE_DIR", "cache"),
            ("MIOPEN_VERIFY_CACHE_PATH", "verify-cache"),
            ("XDG_CACHE_HOME", "xdg-cache"),
            ("HOME", "home"),
        ):
            path = work / directory
            path.mkdir()
            env[name] = str(path)
        for name in (
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "OPENBLAS_NUM_THREADS",
            "GOTO_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MIOPEN_COMPILE_PARALLEL_LEVEL",
            "MIOPEN_DBSYNC_MAX_THREADS",
        ):
            env[name] = str(dispatch_threads)
        env["MIOPEN_AI_FDEEP_USE_SINGLE_THREAD_PREDICT"] = "1"
        env.update(
            {
                "GTEST_TOTAL_SHARDS": "1",
                "GTEST_SHARD_INDEX": "0",
                "GTEST_REPEAT": "1",
                "GTEST_FAIL_FAST": "0",
                "GTEST_COLOR": "no",
                "MIOPEN_INVOKED_FROM_CTEST": "1",
            }
        )
        launcher = [str(rocjitsu), "--config", str(used_config), "--"]
        run_logged(
            launcher + [sys.executable, "-c", RUNTIME_PROBE, str(hsa), args.arch],
            env,
            working_dir,
            reports / "runtime-probe.log",
            120,
        )
        command, expected_names = curate_command(
            command,
            launcher,
            env,
            working_dir,
            reports,
            representative_cases,
        )
        cpu_command, gpu_command, cpu_names, gpu_names = split_execution_commands(
            command, expected_names
        )
        cpu_command = command_with_filter_file(cpu_command, reports / "cpu.gtest-flags")
        gpu_command = command_with_filter_file(gpu_command, reports / "gpu.gtest-flags")
        cpu_workers = min(8, max(1, cpu_count // 4))
        cpu_dispatch_threads = min(4, cpu_count // cpu_workers)
        gpu_report = reports / "gpu.xml"
        cpu_report = reports / "cpu.xml"
        gpu_phase_report = reports / "gpu-phase.xml"
        for path in (gpu_report, cpu_report, gpu_phase_report):
            path.unlink(missing_ok=True)
        execution_start = time.monotonic()
        cpu_reports, errors = (
            run_cpu_shards(
                cpu_command,
                rocjitsu,
                used_config,
                env,
                working_dir,
                reports,
                cpu_workers,
                args.cpu_timeout,
                cpu_dispatch_threads,
                run_logged,
            )
            if cpu_names
            else ([], [])
        )
        cpu_seconds = time.monotonic() - execution_start
        print(
            f"CPU phase: {cpu_workers if cpu_names else 0} shards, {cpu_seconds:.3f}s",
            flush=True,
        )
        phase_reports = []
        if cpu_names:
            phase_reports.append(cpu_report)
            try:
                merge_reports(cpu_reports, cpu_names, cpu_report, cpu_seconds)
            except ValueError as error:
                errors.append(f"CPU report: {error}")
        env["GTEST_OUTPUT"] = f"xml:{gpu_report}"
        gpu_start = time.monotonic()
        if gpu_names:
            try:
                run_logged(
                    launcher + gpu_command,
                    env,
                    working_dir,
                    reports / "gpu.log",
                    args.gpu_timeout,
                )
            except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
                errors.append(f"GPU phase: {error}")
        gpu_seconds = time.monotonic() - gpu_start
        if gpu_names:
            phase_reports.append(gpu_phase_report)
            try:
                merge_reports([gpu_report], gpu_names, gpu_phase_report, gpu_seconds)
            except ValueError as error:
                errors.append(f"GPU report: {error}")
        elapsed = time.monotonic() - execution_start
        (reports / "execution-phases.json").write_text(
            json.dumps(
                {
                    "cpu_workers": cpu_workers if cpu_names else 0,
                    "cpu_selected": len(cpu_names),
                    "gpu_selected": len(gpu_names),
                    "cpu_dispatch_threads": cpu_dispatch_threads,
                    "gpu_dispatch_threads": dispatch_threads,
                    "cpu_seconds": cpu_seconds,
                    "gpu_seconds": gpu_seconds,
                    "execution_seconds": elapsed,
                    "errors": errors,
                },
                indent=2,
            )
            + "\n"
        )
        try:
            counts = merge_reports(phase_reports, expected_names, report, elapsed)
        except ValueError as error:
            raise RuntimeError(
                f"Incomplete {CATEGORY} execution: {error}; {'; '.join(errors)}"
            ) from error
        finally:
            summarize_report(report, args.arch)
        if errors or counts["failed"]:
            raise RuntimeError(
                f"{CATEGORY} failed: {counts['failed']} tests; {'; '.join(errors)}"
            )
        print(f"Complete merged execution report: {report}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, ET.ParseError) as error:
        sys.exit(f"::error::{error}")
    except subprocess.CalledProcessError as error:
        print(
            f"::error::Command failed with exit status {error.returncode}",
            file=sys.stderr,
        )
        sys.exit(error.returncode if error.returncode > 0 else 1)
    except subprocess.TimeoutExpired as error:
        sys.exit(f"::error::Command timed out after {error.timeout}s")
