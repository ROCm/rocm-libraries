#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the unchanged smoke binary, then a separate marked startup probe."""

import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


REPORTS = Path("diagnostics")
SCRIPT_DIR = Path(__file__).resolve().parent
VISIBILITY_KEYS = (
    "DIAGNOSTIC_POOL",
    "DIAGNOSTIC_RUNNER",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_XNACK",
    "OMP_NUM_THREADS",
    "ASAN_OPTIONS",
    "LSAN_OPTIONS",
    "ASAN_SYMBOLIZER_PATH",
    "LD_LIBRARY_PATH",
    "GTEST_SHARD_INDEX",
    "GTEST_TOTAL_SHARDS",
)


def read_file(path):
    try:
        return path.read_text(errors="replace")
    except OSError as error:
        return str(error)


def host_state(label):
    paths = [Path("/proc/loadavg"), Path("/proc/meminfo")]
    paths += list(Path("/sys/module/amdgpu").glob("version"))
    for pattern in ("*/properties", "*/gpu_id"):
        paths += list(Path("/sys/class/kfd/kfd/topology/nodes").glob(pattern))
    for pattern in ("card*/device/gpu_busy_percent", "card*/device/mem_info_vram_used"):
        paths += list(Path("/sys/class/drm").glob(pattern))
    state = {
        "uname": list(os.uname()),
        "environment": {key: os.getenv(key) for key in VISIBILITY_KEYS},
        "files": {str(path): read_file(path) for path in paths},
    }
    (REPORTS / f"host-{label}.json").write_text(json.dumps(state, indent=2))
    print(json.dumps({"phase": label, "environment": state["environment"]}), flush=True)


def permit_parent_debugger():
    # Only this process grants its parent and the parent's debugger permission
    # to inspect it. No host ptrace policy, container capabilities, or ASLR change.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(0x59616D61, os.getppid(), 0, 0, 0) != 0:  # PR_SET_PTRACER
        os.write(2, f"PR_SET_PTRACER failed: {ctypes.get_errno()}\n".encode())


def debugger_environment():
    env = os.environ.copy()
    root = Path("diagnostic-tools/root").resolve()
    env["LD_LIBRARY_PATH"] = (
        f"{root}/usr/lib/x86_64-linux-gnu:{root}/lib/x86_64-linux-gnu"
    )
    env["DEBUGINFOD_URLS"] = ""
    if (root / "usr/lib/python3.12").is_dir():
        env["PYTHONHOME"] = str(root / "usr")
    env.pop("LD_PRELOAD", None)
    return env


def snapshot(name, process):
    if process.poll() is not None:
        return
    root = REPORTS / name
    root.mkdir(exist_ok=True)
    proc = Path(f"/proc/{process.pid}")
    for filename in ("status", "wchan", "syscall", "maps", "smaps_rollup"):
        (root / filename).write_text(read_file(proc / filename))
    threads = {}
    for thread in (proc / "task").glob("*"):
        threads[thread.name] = {
            filename: read_file(thread / filename)
            for filename in ("comm", "wchan", "syscall", "stack")
        }
    (root / "threads.json").write_text(json.dumps(threads, indent=2))
    path_file = REPORTS / "debugger-path.txt"
    debugger = (
        path_file.read_text().strip() if path_file.exists() else shutil.which("gdb")
    )
    if debugger:
        command = [
            debugger,
            "-nx",
            "-nh",
            "-batch",
            "-iex",
            "set auto-load off",
            "-ex",
            "set pagination off",
            "-ex",
            "set debuginfod enabled off",
            "-ex",
            f"attach {process.pid}",
            "-ex",
            "info threads",
            "-ex",
            "thread apply all bt 32",
            "-ex",
            "info sharedlibrary",
            "-ex",
            "detach",
        ]
        data_dir = Path("diagnostic-tools/root/usr/share/gdb").resolve()
        if data_dir.is_dir():
            command.insert(1, f"--data-directory={data_dir}")
        with (root / "backtrace.txt").open("w") as output:
            try:
                result = subprocess.run(
                    command,
                    env=debugger_environment(),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=45,
                    check=False,
                )
                output.write(f"\ngdb returncode: {result.returncode}\n")
            except (OSError, subprocess.TimeoutExpired) as error:
                output.write(f"Debugger error: {error}\n")
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGCONT)
        print(read_file(root / "backtrace.txt")[:120000], flush=True)
    else:
        (root / "backtrace.txt").write_text("No debugger available\n")
    host_state(name)


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Also terminate descendants if the immediate child exited first.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_case(name, command, timeout_seconds, capture_after):
    print(f"Starting {name}: {command}", flush=True)
    log = REPORTS / f"{name}.log"
    started = time.monotonic()
    captured = False
    banner = False
    timed_out = False
    with log.open("wb") as output, log.open(errors="replace") as tail:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=permit_parent_debugger,
        )
        try:
            while process.poll() is None:
                chunk = tail.read()
                if chunk:
                    print(chunk, end="", flush=True)
                    banner |= "hipBLASLt version:" in chunk
                elapsed = time.monotonic() - started
                if not captured and elapsed >= capture_after and not banner:
                    print(
                        f"Capturing {name} PID {process.pid} after {elapsed:.1f}s",
                        flush=True,
                    )
                    snapshot(f"{name}-startup", process)
                    captured = True
                if elapsed >= timeout_seconds:
                    snapshot(f"{name}-timeout", process)
                    timed_out = True
                    stop_process_group(process)
                    break
                time.sleep(0.25)
        finally:
            stop_process_group(process)
        print(tail.read(), end="", flush=True)
    banner = "hipBLASLt version:" in log.read_text(errors="replace")
    result = {
        "command": command,
        "pid": process.pid,
        "returncode": 124 if timed_out else process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "startup_backtrace_attempted": captured,
        "version_banner_observed": banner,
    }
    (REPORTS / f"{name}-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def main():
    def cancelled(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    REPORTS.mkdir(exist_ok=True)
    host_state("before")
    # Nothing initializes the GPU before this original, unmodified smoke run.
    run_case("smoke", ["./build/bin/hipblaslt-test", "--gtest_filter=*smoke*"], 600, 60)
    host_state("after-smoke")

    # This separate process runs after the baseline, so it cannot warm up that
    # run. It uses the artifact's ASAN runtime and libraries, with no new GPU code.
    compiler = shutil.which("clang") or shutil.which("cc")
    if not compiler:
        (REPORTS / "probe-build.log").write_text("No C compiler available\n")
        return
    probe = (REPORTS / "startup-probe").resolve()
    command = [
        compiler,
        "-g",
        "-O0",
        str(SCRIPT_DIR / "startup-probe.c"),
        "-o",
        str(probe),
        "-Wl,--no-as-needed",
        os.environ["DIAGNOSTIC_ASAN_RUNTIME"],
        "-Wl,--as-needed",
        "-ldl",
    ]
    with (REPORTS / "probe-build.log").open("w") as output:
        result = subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, timeout=30, check=False
        )
    if result.returncode == 0:
        run_case("probe", [str(probe)], 120, 15)
    host_state("after-probe")


if __name__ == "__main__":
    main()
