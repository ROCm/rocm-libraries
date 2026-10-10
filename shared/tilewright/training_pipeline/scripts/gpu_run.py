#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run a command on one GPU under that GPU's lease (lib/gpu_lease.py).

Usage:
  gpu_run.py --gpu N [--purpose TEXT] [--wait S] -- CMD [ARG ...]
  gpu_run.py --status

Takes the lease of physical GPU N (the index ROCR_VISIBLE_DEVICES takes) and
runs CMD with ROCR_VISIBLE_DEVICES=N and without HIP_VISIBLE_DEVICES and
CUDA_VISIBLE_DEVICES, so that GPU N is the only device it sees (its device
0). The lease is held until CMD exits; CMD inherits the lock, so the lease
lasts as long as CMD even when this wrapper dies first. A lease held by
another process tree is refused at once, naming the holder, or after
waiting up to --wait seconds. Processes that already hold memory on GPU N
are named on stderr.

SIGTERM and SIGHUP are passed on to CMD; SIGINT from the terminal reaches it
directly. Exit status: CMD's (128 + N when signal N ended it), 127 when CMD
cannot be started, 2 for a refused lease or bad arguments.

--status lists the lease files: GPU, held or free, and the recorded holder.
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from lib import gpu_lease  # noqa: E402
from lib import kfd  # noqa: E402


def command_env(gpu: int, environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    for var in kfd.HIP_VARS:
        env.pop(var, None)
    env[kfd.ROCR_VAR] = str(int(gpu))
    return env


def _print_status() -> int:
    rows = gpu_lease.status()
    if not rows:
        print(f"no leases in {gpu_lease.lease_dir()}")
    for row in rows:
        holder = row["holder"]
        who = (
            gpu_lease.Holder(**holder).describe()
            if holder
            else "a process that has not recorded itself"
        )
        if row["held"] is False:
            state = "free"
        elif row["held"]:
            state = f"held by {who}"
        else:
            state = f"unknown (/proc/locks unreadable); last holder {who}"
        print(f"GPU {row['gpu']}: {state}")
    return 0


def _warn_foreign(gpu: int) -> None:
    for g in kfd.gpus():
        if g.index == gpu:
            for f in kfd.foreign_now(g):
                print(
                    f"gpu_run: warning: GPU {gpu} is in use: {f.describe()}",
                    file=sys.stderr,
                )


def _run(cmd: Sequence[str], env: Dict[str, str], fds: List[int]) -> int:
    for fd in fds:
        os.set_inheritable(fd, True)
    procs: List[subprocess.Popen] = []
    pending: List[int] = []

    def forward(signum: int, _frame: object) -> None:
        if not procs:
            pending.append(signum)
            return
        try:
            procs[0].send_signal(signum)
        except ProcessLookupError:
            pass

    previous = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGHUP)}
    previous[signal.SIGINT] = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            procs.append(subprocess.Popen(list(cmd), env=env, pass_fds=fds))
        except OSError as e:
            print(f"gpu_run: cannot run {cmd[0]}: {e}", file=sys.stderr)
            return 127
        for signum in pending:
            procs[0].send_signal(signum)
        rc = procs[0].wait()
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
    return 128 - rc if rc < 0 else rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    opts, cmd = argv[:cut], argv[cut + 1 :]
    ap = argparse.ArgumentParser(
        prog="gpu_run.py",
        description="run a command on one GPU under its lease",
        epilog="usage: gpu_run.py --gpu N [--purpose TEXT] [--wait S] -- CMD ...",
    )
    ap.add_argument("--gpu", type=int, help="physical GPU index")
    ap.add_argument("--purpose", default=None, help="recorded with the lease")
    ap.add_argument(
        "--wait",
        type=float,
        default=0.0,
        metavar="S",
        help="wait up to S seconds for a held lease (default: refuse at once)",
    )
    ap.add_argument("--status", action="store_true", help="list the leases")
    args = ap.parse_args(opts)
    if args.status:
        return _print_status()
    if args.gpu is None or args.gpu < 0 or not cmd:
        ap.error("needs --gpu N (N >= 0) and a command after --")
    known = kfd.gpus()
    if known and args.gpu >= len(known):
        print(
            f"gpu_run: there is no GPU {args.gpu}; the KFD topology lists "
            f"{len(known)}",
            file=sys.stderr,
        )
        return 2
    purpose = args.purpose or "gpu_run: " + shlex.join(cmd)
    try:
        with gpu_lease.hold_gpus(
            [args.gpu],
            purpose,
            run_dir=os.getcwd(),
            wait_s=args.wait,
            report=lambda msg: print(f"gpu_run: {msg}", file=sys.stderr, flush=True),
        ) as held:
            _warn_foreign(args.gpu)
            return _run(cmd, command_env(args.gpu), held.fds)
    except gpu_lease.LeaseError as e:
        print(f"gpu_run: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
