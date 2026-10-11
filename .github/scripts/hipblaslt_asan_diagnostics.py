#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collect runner evidence and inspect a timed-out ASAN process before killing it."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


# Do not save the complete environment: CI credentials may be present in it.
ENVIRONMENT_KEYS = (
    "RUNNER_NAME",
    "ARTIFACT_GROUP",
    "HSA_XNACK",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "PARENT_GPU_DEVICES",
    "TELEMETRY_GPU_BDF_MAP",
    "HSA_CU_MASK",
    "OMP_NUM_THREADS",
    "ASAN_OPTIONS",
    "ASAN_SYMBOLIZER_PATH",
    "LSAN_OPTIONS",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
)


def selected_environment(env):
    return {key: env[key] for key in ENVIRONMENT_KEYS if key in env}


def read_file(path):
    try:
        return {"text": Path(path).read_text(errors="replace")}
    except OSError as error:
        return {"error": str(error)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def host_snapshot():
    files = ["/proc/loadavg", "/proc/meminfo", "/proc/sys/kernel/yama/ptrace_scope"]
    files += [
        f"/sys/module/amdgpu/{name}"
        for name in (
            "version",
            "parameters/cwsr_enable",
            "parameters/noretry",
            "parameters/sched_policy",
            "parameters/hws_max_conc_proc",
            "parameters/queue_preemption_timeout_ms",
        )
    ]
    files += sorted(
        str(path)
        for path in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties")
    )
    devices = {}
    for path in sorted(Path("/sys/class/drm").glob("renderD*/device")):
        # The resolved PCI path pairs render nodes with KFD drm_render_minor.
        devices[path.parent.name] = str(path.resolve())
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "uname": list(platform.uname()),
        "environment": selected_environment(os.environ),
        "render_devices": devices,
        "files": {path: read_file(path) for path in files},
    }


def signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def stop_process(process, grace=2):
    signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Also clean up children if their parent exited without waiting for them.
    signal_group(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        print(f"PID {process.pid} did not exit after SIGKILL", flush=True)


def probe(command, output, timeout, env=None):
    """Keep failed-command output and bound the command and its children."""
    result = {"command": [str(arg) for arg in command], "timed_out": False}
    with output.open("w") as stream:
        try:
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        except OSError as error:
            result["error"] = str(error)
            stream.write(str(error) + "\n")
        else:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
            finally:
                stop_process(process)
            result["returncode"] = process.returncode
    print(f"=== {output.name}: {json.dumps(result)} ===", flush=True)
    # Full output is retained in the uploaded file, including on failure.
    text = output.read_text(errors="replace")
    print(text[-65536:], flush=True)
    if len(text) > 65536:
        print(f"Showing the last 64 KiB; full output is in {output}", flush=True)
    return result


def sanity(output, bin_dir, timeout):
    snapshot = host_snapshot()
    write_json(output / "host-before.json", snapshot)
    print(json.dumps(snapshot, indent=2), flush=True)
    smi_env = os.environ.copy()
    runtimes = sorted(
        (bin_dir.parent / "lib/llvm/lib/clang").glob("**/libclang_rt.asan.so")
    )
    if runtimes:
        # amd-smi is Python loading an ASAN-instrumented library. Preload before
        # Python starts; leak checking of this diagnostic is not the smoke test.
        smi_env["LD_PRELOAD"] = " ".join(
            filter(None, (str(runtimes[0]), smi_env.get("LD_PRELOAD")))
        )
        smi_env["ASAN_OPTIONS"] = smi_env.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
    results = {}
    for name, command, env in (
        ("amd-smi", [bin_dir / "amd-smi", "static"], smi_env),
        ("rocminfo", [bin_dir / "rocminfo"], None),
    ):
        results[name] = probe(command, output / f"{name}.log", timeout, env)
    write_json(output / "sanity-results.json", results)


def process_group_members(group):
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            # comm may contain spaces or parentheses. Fields after its final ')'
            # start with state, ppid, pgrp; no PID outside this group is inspected.
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                members.append(path)
        except (OSError, ValueError, IndexError):
            continue
    return sorted(members, key=lambda path: int(path.name))


def process_snapshot(path):
    data = {"pid": int(path.name)}
    for name in ("status", "wchan", "syscall", "maps"):
        data[name] = read_file(path / name)
    try:
        data["exe"] = str((path / "exe").readlink())
        data["command"] = (
            (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
        )
        env = dict(
            entry.split("=", 1)
            for entry in (path / "environ")
            .read_bytes()
            .decode(errors="replace")
            .split("\0")
            if "=" in entry
        )
        data["environment"] = selected_environment(env)
    except OSError as error:
        data["identity_error"] = str(error)
    data["gpu_fds"] = {}
    try:
        for fd in (path / "fd").iterdir():
            try:
                target = str(fd.readlink())
                if target == "/dev/kfd" or target.startswith("/dev/dri/"):
                    data["gpu_fds"][fd.name] = target
            except OSError:
                continue
    except OSError as error:
        data["gpu_fds_error"] = str(error)
    data["threads"] = {
        task.name: {
            name: read_file(task / name)
            for name in ("comm", "wchan", "syscall", "stack", "schedstat")
        }
        for task in (path / "task").glob("[0-9]*")
    }
    return data


def capture_timeout(output, group, budget=30):
    deadline = time.monotonic() + budget
    write_json(output / "host-timeout.json", host_snapshot())
    processes = [process_snapshot(path) for path in process_group_members(group)]
    write_json(output / "processes.json", processes)
    print(f"Live processes at timeout: {json.dumps(processes, indent=2)}", flush=True)
    debugger = shutil.which("gdb")
    results = {}
    if not debugger:
        print(
            "gdb unavailable; process files are retained, but no userspace backtrace",
            flush=True,
        )
        results["gdb"] = {"error": "gdb unavailable"}
    else:
        env = os.environ.copy()
        # The debugger uses system libraries, not the instrumented artifact.
        env.pop("LD_PRELOAD", None)
        env.pop("LD_LIBRARY_PATH", None)
        env["DEBUGINFOD_URLS"] = ""
        for process in processes:
            if not Path(process.get("exe", "")).name.startswith("hipblaslt-test"):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                results["budget"] = {"error": "backtrace capture budget exhausted"}
                break
            pid = process["pid"]
            results[str(pid)] = probe(
                [
                    debugger,
                    "-nx",
                    "-batch",
                    "-iex",
                    "set auto-load off",
                    "-iex",
                    "set debuginfod enabled off",
                    "-p",
                    str(pid),
                    "-ex",
                    "set pagination off",
                    "-ex",
                    "thread apply all bt 20",
                    "-ex",
                    "info sharedlibrary",
                    "-ex",
                    "detach",
                ],
                output / f"backtrace-{pid}.log",
                min(15, remaining),
                env,
            )
    # Permission denial is useful evidence too; no privileged retry or reset.
    results["dmesg"] = probe(["dmesg", "--level=err,warn"], output / "dmesg.log", 5)
    write_json(output / "capture-results.json", results)


def run_test(command, output, timeout):
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    timed_out = False
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(
                f"::error::hipBLASLt ASAN command exceeded {timeout:g}s on "
                f"{os.getenv('RUNNER_NAME', platform.node())}; capturing live "
                f"processes before termination in {output}",
                flush=True,
            )
            try:
                capture_timeout(output, process.pid)
            except Exception as error:
                # Diagnostics must not replace the actual timeout result.
                print(f"Diagnostic capture failed: {error}", flush=True)
    finally:
        stop_process(process, grace=10)
    returncode = 124 if timed_out else process.returncode
    if returncode < 0:
        returncode = 128 - returncode
    result = {
        "command": command,
        "timed_out": timed_out,
        "returncode": returncode,
        "process_returncode": process.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "timeout_seconds": timeout,
    }
    write_json(output / "test-result.json", result)
    print(f"ASAN command result: {json.dumps(result)}", flush=True)
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sanity", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, default=Path("build/bin"))
    parser.add_argument("--timeout", type=float, default=None)
    # Keep the test command separate so its flags are never parsed as ours.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1 :]
    timeout = (
        args.timeout
        if args.timeout is not None
        else (30 if args.mode == "sanity" else 1200)
    )
    if timeout <= 0 or (args.mode == "run" and not command):
        parser.error("timeout must be positive and run requires a command after --")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def cancelled(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancelled)
    if args.mode == "sanity":
        sanity(args.output_dir, args.bin_dir.resolve(), timeout)
        return 0
    return run_test(command, args.output_dir, timeout)


if __name__ == "__main__":
    raise SystemExit(main())
