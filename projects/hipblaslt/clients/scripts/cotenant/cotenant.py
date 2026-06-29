#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run a command (typically hipblaslt-bench) under CU contention.

Launches a persistent HIP "busy" kernel pinned to a fixed number of CUs
(one workgroup per CU), waits until all its workgroups are resident, then runs
the given command while that contention is in effect. The cotenant is always
killed when the command finishes or this script is interrupted.

    cotenant.py --cus 64 -- hipblaslt-bench -m 4096 -n 4096 -k 4096

The cotenant is built on first use with hipcc (override with HIPCC=...).
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "busy_cotenant.hip"
BIN = HERE / "busy_cotenant"
ARCH_STAMP = HERE / ".busy_cotenant.arch"


def probe_device(env):
    """Return (gfx_arch, total_cus) for the first GPU via rocminfo, or (None, None)."""
    try:
        out = subprocess.check_output(
            ["rocminfo"], text=True, env=env, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    arch, cus, in_gpu = None, None, False
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "Name:" and fields[1].startswith("gfx"):
            arch, in_gpu = fields[1], True
        elif in_gpu and line.strip().startswith("Compute Unit:"):
            cus = int(fields[-1])
            break
    return arch, cus


def ensure_built(arch, env):
    """Compile busy_cotenant if the binary is missing, stale, or built for another arch."""
    if not SRC.is_file():
        sys.exit(f"ERROR: cotenant source not found: {SRC}")
    fresh = (
        BIN.exists()
        and BIN.stat().st_mtime >= SRC.stat().st_mtime
        and ARCH_STAMP.exists()
        and ARCH_STAMP.read_text().strip() == arch
    )
    if fresh:
        return
    hipcc = os.environ.get("HIPCC", "hipcc")
    print(f"building {BIN.name} for {arch} ...")
    try:
        subprocess.run(
            [
                hipcc,
                "-O2",
                "-std=c++17",
                f"--offload-arch={arch}",
                str(SRC),
                "-o",
                str(BIN),
            ],
            check=True,
            env=env,
        )
    except FileNotFoundError:
        sys.exit(f"ERROR: '{hipcc}' not found; install ROCm or set HIPCC to its path.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ERROR: building {BIN.name} failed (hipcc exited {e.returncode}).")
    except OSError as e:
        sys.exit(
            f"ERROR: could not write {BIN} ({e.strerror}); is the directory writable?"
        )
    ARCH_STAMP.write_text(arch)


def run_command(command, env):
    """Run the user's command, returning its exit code; fail cleanly if not found."""
    try:
        return subprocess.run(command, env=env).returncode
    except FileNotFoundError:
        sys.exit(f"ERROR: command not found: {command[0]}")


# Printed by busy_cotenant once every workgroup is confirmed resident on the CUs.
READY_MARKER = "READY"


def wait_ready(proc, log_handle, log_path, wait_s):
    """Block until the cotenant logs READY (all workgroups resident); abort if it exits or times out."""
    deadline = time.monotonic() + wait_s
    while True:
        # Re-read the whole log each poll: the cotenant writes concurrently, so a
        # line-at-a-time read could split the marker across two reads and miss it.
        log_handle.seek(0)
        if READY_MARKER in log_handle.read():
            return
        if proc.poll() is not None:
            sys.exit(
                f"ERROR: cotenant exited (rc={proc.returncode}) before becoming resident; see {log_path}."
            )
        if time.monotonic() >= deadline:
            sys.exit(
                f"ERROR: cotenant not resident after {wait_s}s; aborting. See {log_path} "
                "(raise --wait if the device is slow to schedule the grid)."
            )
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser(
        description="Run a command under CU contention from a busy cotenant kernel.",
        epilog="Pass the command to run after `--`, e.g. -- hipblaslt-bench -m 1024 -n 1024 -k 1024",
    )
    ap.add_argument(
        "--cus",
        type=int,
        required=True,
        metavar="N",
        help="number of CUs the cotenant occupies (0 = uncontended baseline, no cotenant)",
    )
    ap.add_argument(
        "--device", metavar="N", help="set HIP_VISIBLE_DEVICES for cotenant and command"
    )
    ap.add_argument(
        "--arch",
        metavar="GFX",
        help="build target arch (default: rocminfo auto-detect)",
    )
    ap.add_argument(
        "--wait",
        type=float,
        default=30.0,
        metavar="S",
        help="max seconds to wait for residency (default: 30)",
    )
    ap.add_argument(
        "--grace",
        type=float,
        default=0.0,
        metavar="S",
        help="extra seconds to wait after residency is confirmed (default: 0)",
    )
    ap.add_argument(
        "command", nargs=argparse.REMAINDER, help="command to run after `--`"
    )
    args = ap.parse_args()

    if args.cus < 0:
        sys.exit("ERROR: --cus must be >= 0.")
    if not args.wait >= 0:
        sys.exit("ERROR: --wait must be >= 0.")
    if not args.grace >= 0:
        sys.exit("ERROR: --grace must be >= 0.")
    command = (
        args.command[1:] if args.command and args.command[0] == "--" else args.command
    )
    if not command:
        sys.exit("ERROR: no command given; pass it after `--`.")

    env = os.environ.copy()
    if args.device is not None:
        env["HIP_VISIBLE_DEVICES"] = args.device

    # --cus 0 is the uncontended baseline: run the command with no cotenant.
    if args.cus == 0:
        print(f"running uncontended (no cotenant): {' '.join(command)}")
        return run_command(command, env)

    # rocminfo honors ROCR_VISIBLE_DEVICES for device selection.
    probe_env = env.copy()
    if args.device is not None:
        probe_env["ROCR_VISIBLE_DEVICES"] = args.device
    arch, total_cus = probe_device(probe_env)
    arch = args.arch or arch
    if arch is None:
        sys.exit("ERROR: could not detect GPU arch via rocminfo; pass --arch gfxNNN.")

    if total_cus is None:
        print(
            "WARNING: could not read CU count from rocminfo; skipping --cus bounds check."
        )
    elif args.cus > total_cus:
        sys.exit(f"ERROR: --cus {args.cus} exceeds the device CU count ({total_cus}).")
    elif args.cus == total_cus:
        sys.exit(
            f"ERROR: --cus {args.cus} would occupy all {total_cus} CUs and starve the benchmark; use fewer."
        )

    ensure_built(arch, env)

    log_path = HERE / "cotenant.log"
    print(f"launching cotenant on {args.cus} CUs (log: {log_path})")
    with open(log_path, "w") as log, open(log_path, "r") as log_reader:
        cotenant = subprocess.Popen(
            [str(BIN), str(args.cus)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            wait_ready(cotenant, log_reader, log_path, args.wait)
            if args.grace:
                time.sleep(args.grace)
            print(f"running: {' '.join(command)}")
            return run_command(command, env)
        finally:
            if cotenant.poll() is None:
                try:
                    os.killpg(cotenant.pid, signal.SIGTERM)
                    try:
                        cotenant.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(cotenant.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # cotenant exited between the poll and the signal


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # main()'s finally has already torn down the cotenant during unwinding.
        sys.exit(130)
