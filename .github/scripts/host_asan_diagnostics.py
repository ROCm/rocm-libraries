# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect host-ASAN diagnostics without changing the test command's result."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def command_log(path, command, timeout_seconds=20):
    # Keep stderr even on failure. A broken GPU probe must not prevent others.
    with path.open("w") as output:
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
        command_log(directory / f"amd-smi-{mode}.txt", [str(bin_dir / "amd-smi"), mode])
    command_log(directory / "rocminfo.txt", [str(bin_dir / "rocminfo")])


def safe_snapshot(directory, **kwargs):
    try:
        snapshot(directory, **kwargs)
    except Exception as error:
        # Even filesystem/procfs failures must not turn a passing test red.
        print(f"::warning::Unable to finish ASAN diagnostics: {error}", flush=True)


def run_monitored(command, log_file, directory, executable, quiet_seconds):
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
    try:
        while process.poll() is None:
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
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
