# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect host-ASAN diagnostics without changing the test command's result."""

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def command_log(path, command, timeout_seconds=20, mode="w"):
    # Keep stderr even on failure. A broken GPU probe must not prevent others.
    with path.open(mode) as output:
        print(f"UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", file=output)
        print(f"Command: {command!r}", file=output, flush=True)
        try:
            result = subprocess.run(
                ["timeout", "-k", "5", str(timeout_seconds), *command],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
            print(f"Exit status: {result.returncode}", file=output)
        except OSError as error:
            print(f"Unable to run diagnostic: {error}", file=output)


def descendants(pid):
    # Walk only this command's descendants, including children of worker threads.
    found: set[int] = set()
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        for children in Path(f"/proc/{current}/task").glob("*/children"):
            try:
                pending.extend(int(value) for value in children.read_text().split())
            except OSError:
                pass  # A child can exit while we inspect it.
    return sorted(found)


def smi_command(bin_dir):
    command = [str(bin_dir / "amd-smi")]
    asan_runtime = next(
        (bin_dir.parent / "lib/llvm/lib/clang").rglob("libclang_rt.asan*.so"), None
    )
    if asan_runtime:
        # Also works when sudo clears the caller's runtime environment.
        command = [
            "env",
            f"LD_PRELOAD={asan_runtime.resolve()}",
            f"LD_LIBRARY_PATH={bin_dir.parent.resolve()}/lib/rocm_sysdeps/lib:{bin_dir.parent.resolve()}/lib:{asan_runtime.parent.resolve()}:{os.environ.get('LD_LIBRARY_PATH', '')}",
            f"ASAN_OPTIONS={os.environ.get('ASAN_OPTIONS', '')}:detect_leaks=0",
            *command,
        ]
    return command


def gpu_sample(
    directory,
    bin_dir,
    dumps=Path("/sys/class/devcoredump"),
    debug=Path("/sys/kernel/debug/dri"),
):
    directory.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    # Preserve dumps before querying a possibly wedged GPU. Linux normally
    # expires devcoredumps after five minutes. Never write to their sysfs files.
    saved: list[dict[str, object]] = []
    try:
        available = sorted(dumps.iterdir())
        record["devcoredump_entries"] = [path.name for path in available]
    except OSError as error:
        available = []
        record["devcoredump_error"] = str(error)
    for dump in available:
        entry: dict[str, object] = {"source": str(dump)}
        try:
            device = (dump / "failing_device").resolve(strict=True)
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
            destination = (
                directory / f"{dump.name}-{device.name.replace(':', '_')}.data"
            )
            if destination.exists():
                continue
            partial = destination.with_suffix(".partial")
            with (dump / "data").open("rb") as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            partial.replace(destination)
            entry.update(
                device=str(device),
                file=destination.name,
                bytes=destination.stat().st_size,
            )
            destination.with_suffix(".json").write_text(
                json.dumps({"utc": record["utc"], **entry}, indent=2) + "\n"
            )
        except OSError as error:
            entry["error"] = str(error)
        saved.append(entry)
    record["coredumps"] = saved

    # These read-only driver files can expose process names across PID
    # namespaces, if debugfs is already visible. Never mount it or use host PIDs
    # as indices into this container's /proc: those PIDs mean different things.
    driver_processes: dict[str, object] = {}
    try:
        entries = sorted(
            path
            for path in debug.iterdir()
            if path.name.isdigit() and not path.is_symlink()
        )
        record["debugfs_entries"] = [path.name for path in entries]
    except OSError as error:
        entries = []
        record["debugfs_error"] = str(error)
    for device in entries:
        try:
            result = subprocess.run(
                ["cat", str(device / "name"), str(device / "amdgpu_vm_info")],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            driver_processes[device.name] = {
                "returncode": result.returncode,
                "stderr": result.stderr,
                "identity": result.stdout.splitlines()[:1],
                "processes": [
                    line
                    for line in result.stdout.splitlines()
                    if line.startswith("pid:")
                ],
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            driver_processes[device.name] = {"error": str(error)}
    record["driver_processes"] = driver_processes

    # Keep per-device activity even when AMD-SMI fails. Enumerate PCI devices
    # directly so card/render aliases do not duplicate the same GPU.
    devices: dict[str, object] = {}
    for device in sorted(Path("/sys/bus/pci/drivers/amdgpu").glob("????:??:??.?")):
        fields: dict[str, object] = {}
        for name in ("gpu_busy_percent", "mem_info_vram_used", "mem_info_gtt_used"):
            try:
                fields[name] = (device / name).read_text().strip()
            except OSError as error:
                fields[name] = {"error": str(error)}
        devices[device.name] = fields
    record["devices"] = devices

    gpu_processes: object = []
    try:
        result = subprocess.run(
            [*smi_command(bin_dir), "process", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        record["smi_returncode"] = result.returncode
        record["smi_stderr"] = result.stderr
        try:
            gpu_processes = json.loads(result.stdout)
        except ValueError:
            record["smi_stdout"] = result.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        record["smi_error"] = str(error)
    record["gpu_processes"] = gpu_processes
    with (directory / "timeline.jsonl").open("a") as output:
        print(json.dumps(record), file=output, flush=True)


def snapshot(directory, root_pid=None, executable=None):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        name: os.environ.get(name)
        for name in (
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "GITHUB_SHA",
            "ARTIFACT_RUN_ID",
            "SOURCE_SHA",
            "THEROCK_REF",
            "RUNNER_NAME",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "HSA_XNACK",
        )
    }
    metadata["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (directory / "context.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if root_pid is not None:
        traced = 0
        for pid in descendants(root_pid):
            proc = Path(f"/proc/{pid}")
            try:
                # Never attach to unrelated jobs or to the Python/timeout wrapper.
                if (proc / "exe").resolve(strict=True) != executable:
                    continue
                if traced == 4:
                    (directory / "backtrace-limit.txt").write_text(
                        "Limited capture to four test processes.\n"
                    )
                    break
                traced += 1
                (directory / f"{pid}-maps.txt").write_text((proc / "maps").read_text())
            except OSError:
                continue
            command_log(
                directory / f"{pid}-threads.txt",
                ["ps", "-L", "-p", str(pid), "-o", "pid,tid,stat,wchan:32,comm"],
            )
            command_log(
                directory / f"{pid}-backtrace.txt",
                [
                    "sudo",
                    "-n",
                    "/usr/bin/gdb",
                    "--batch",
                    "--nx",
                    "--nh",
                    "-iex",
                    "set auto-load off",
                    "-iex",
                    "set debuginfod enabled off",
                    "-ex",
                    "set pagination off",
                    "--pid",
                    str(pid),
                    "-ex",
                    "thread apply all bt 40",
                    "-ex",
                    "detach",
                ],
            )

    paths = [
        Path("/proc/sys/kernel/random/boot_id"),
        Path("/proc/uptime"),
        Path("/sys/module/amdgpu/version"),
        *Path("/sys/class/kfd/kfd/topology/nodes").glob("*/gpu_id"),
        *Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"),
    ]
    with (directory / "driver-topology.txt").open("w") as output:
        for path in paths:
            print(f"\n{path}:", file=output)
            try:
                print(path.read_text(), file=output)
            except OSError as error:
                print(error, file=output)
    command_log(directory / "kernel.txt", ["uname", "-a"])
    command_log(
        directory / "dmesg.txt", ["sudo", "-n", "dmesg", "--time-format", "iso"]
    )
    bin_dir = Path(os.environ.get("THEROCK_BIN_DIR", "build/bin"))
    for mode in ("static", "metric", "process"):
        command_log(directory / f"amd-smi-{mode}.txt", [*smi_command(bin_dir), mode])
    command_log(directory / "rocminfo.txt", [str(bin_dir / "rocminfo")])


def safe_snapshot(directory, **kwargs):
    try:
        snapshot(directory, **kwargs)
    except Exception as error:
        # Even filesystem/procfs failures must not turn a passing test red.
        print(f"::warning::Unable to finish ASAN diagnostics: {error}", flush=True)


def run_monitored(
    command, log_file, directory, executable, quiet_seconds, gpu_samples=False
):
    process = subprocess.Popen(command, start_new_session=True)

    def forward(sig, _frame=None):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    previous = {
        sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    last_size = 0
    last_progress = time.monotonic()
    captured = False
    next_sample = 0.0
    try:
        while process.poll() is None:
            if gpu_samples and time.monotonic() >= next_sample:
                directory.mkdir(parents=True, exist_ok=True)
                command_log(
                    directory / "gpu-sample-status.txt",
                    [
                        "sudo",
                        "-n",
                        "env",
                        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--output-dir",
                        str(directory / "gpu-timeline"),
                        "--sample-gpu",
                        "--bin-dir",
                        str(
                            Path(
                                os.environ.get("THEROCK_BIN_DIR", "build/bin")
                            ).resolve()
                        ),
                    ],
                    timeout_seconds=20,
                    mode="a",
                )
                next_sample = time.monotonic() + 5
            try:
                size = log_file.stat().st_size
            except OSError:
                size = 0
            if size != last_size:
                last_size = size
                last_progress = time.monotonic()
            if not captured and time.monotonic() - last_progress >= quiet_seconds:
                if output := os.environ.get("GITHUB_OUTPUT"):
                    try:
                        with open(output, "a") as stream:
                            print("hang_diagnostics=true", file=stream)
                    except OSError as error:
                        print(
                            f"::warning::Unable to record diagnostic output: {error}",
                            flush=True,
                        )
                safe_snapshot(
                    directory / "stall",
                    root_pid=process.pid,
                    executable=executable,
                )
                captured = True
            time.sleep(1)
        returncode = process.wait()
        if returncode:
            safe_snapshot(directory / "after-failure")
        return returncode if returncode >= 0 else 128 - returncode
    finally:
        # Includes descendants left behind when timeout kills the command.
        forward(signal.SIGKILL)
        process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--quiet-seconds", type=float, default=300)
    parser.add_argument("--gpu-samples", action="store_true")
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--bin-dir", type=Path, default=Path("build/bin"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.sample_gpu:
        gpu_sample(args.output_dir, args.bin_dir)
        return 0
    if not args.command:
        safe_snapshot(args.output_dir)
        return 0
    if (
        args.command[0] != "--"
        or not args.log_file
        or not args.executable
        or args.quiet_seconds <= 0
    ):
        parser.error(
            "monitoring requires --log-file, --executable, a positive --quiet-seconds, and -- COMMAND"
        )
    return run_monitored(
        args.command[1:],
        args.log_file,
        args.output_dir,
        args.executable.resolve(),
        args.quiet_seconds,
        args.gpu_samples,
    )


if __name__ == "__main__":
    raise SystemExit(main())
