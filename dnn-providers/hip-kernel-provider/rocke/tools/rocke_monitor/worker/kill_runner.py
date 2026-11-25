#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Gentle teardown of a WEDGED runner: stop the model, then kill ONLY the runner subprocess. Run as root (it runs
`ollama stop` and `ollama ps` as the worker's user through su).
NEVER systemctl restart ollama on a wedge (it can escalate to a host bugcheck).
The model is config worker_model; the user is <WORKER_USER>, the owner of the working folder (RKMON_* via rkenv.py).
"""

import os
import pwd
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import rkenv  # noqa: E402
from steps import PROG, echo, last_lines, run, write_raw  # noqa: E402


def owner(path: str) -> str:
    """`stat -c %U PATH`: the owner's name (UNKNOWN without one), "" when PATH cannot be read."""
    try:
        uid = os.stat(path).st_uid
    except OSError as exc:
        print(
            f"stat: cannot statx '{path}': {exc.strerror}", file=sys.stderr, flush=True
        )
        return ""
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return "UNKNOWN"


def main() -> int:
    rkenv.load()
    model = os.environ.get("RKMON_WORKER_MODEL", "")
    user = os.environ.get("RKMON_WORKER_USER") or owner(os.environ["RKMON_WORKING_DIR"])
    rc = run(
        ["timeout", "10", "su", user, "-c", f"ollama stop {model}"],
        stderr=subprocess.DEVNULL,
    )
    echo("ollama stop ok" if rc == 0 else "ollama stop hung")
    echo(
        "runner killed"
        if run(["pkill", "-f", "ollama runner|llama-server"]) == 0
        else "no runner process"
    )
    time.sleep(1)
    sys.stdout.flush()
    try:
        out = subprocess.run(
            ["su", user, "-c", "ollama ps"], stdout=subprocess.PIPE
        ).stdout
    except OSError:
        print(f"{PROG}: su: command not found", file=sys.stderr, flush=True)
        out = b""
    write_raw(last_lines(out, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
