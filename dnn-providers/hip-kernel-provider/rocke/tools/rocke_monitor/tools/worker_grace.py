#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Machine limit: no heavy CPU (builds, big test runs) in the local review loop's first 5 minutes. Run this right
before each CPU-heavy step: it returns at once when none of its processes is that young, else sleeps until the
youngest is 5 minutes old. Prints what it saw. A reviewer model on another machine loads this one only lightly: then
only a local model runner counts, not the review loop itself. Where it runs is RKMON_WORKER_REMOTE ("1": another
machine, "0": this one), which the monitor's jobs set from its settings (rkmon_lib/config.py env) and PR code's
environment keeps (worker/pr_env.py), where this runs in a PR job; without it, RKMON_WORKER_URL's host (not on the
loopback address: another machine).

Usage: python3 -B tools/worker_grace.py [max_wait_s]   (default 900)
Exit 0: no local review loop runs, or its youngest process is 5 minutes old; 1: it gave up after max_wait_s.
"""

import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rkmon_lib import container_layout  # noqa: E402

HOST = re.compile(r"^[A-Za-z]+://(\[[^]]*\]|[^/:]*).*")


def say(text: str) -> None:
    print(text, flush=True)


def url_host(url: str) -> str:
    """The URL's host, as `sed -E 's#^[A-Za-z]+://(\\[[^]]*\\]|[^/:]*).*#\\1#'` gives it (line by line)."""
    return "\n".join(HOST.sub(r"\1", x) for x in url.split("\n")).rstrip("\n")


def youngest(pat: str) -> int | None:
    """The smallest elapsed time (s) of a process whose command line matches `pat` (never an awk), else None."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "etimes=,args="],
            stdout=subprocess.PIPE,
            text=True,
            errors="replace",
        ).stdout
    except OSError:
        return None
    rx = re.compile(pat)
    ages = []
    for ln in out.splitlines():
        if rx.search(ln) and "awk" not in ln:
            first = ln.split()[:1]
            if first and first[0].isdigit():
                ages.append(int(first[0]))
    return min(ages) if ages else None


def remote(environ=None) -> bool:
    """The reviewer model runs on another machine (the module's docstring says how that is read)."""
    e = os.environ if environ is None else environ
    flag = e.get("RKMON_WORKER_REMOTE", "")
    if flag in ("0", "1"):
        return flag == "1"
    host = url_host(e.get("RKMON_WORKER_URL", ""))
    return bool(host) and host not in container_layout.LOCAL_HOSTS


def as_int(value: str):
    try:
        return int(value.strip())
    except ValueError:
        return None


def main(argv: list[str]) -> int:
    limit = argv[1] if len(argv) > 1 and argv[1] != "" else "900"
    waited = 0
    pat = (
        "llama-server|ollama_llama_server"
        if remote()
        else "review_agent[.]py|llama-server|ollama_llama_server"
    )
    while True:
        age = youngest(pat)
        if age is None:
            say("worker_grace: no local review loop running")
            return 0
        if age >= 300:
            say(f"worker_grace: youngest worker process is {age}s old: go")
            return 0
        lim = as_int(limit)
        if (
            lim is None
        ):  # as `[ N -ge x ]` with a non-number: an error, and no giving up
            sys.stderr.write(f"{argv[0]}: line 19: [: {limit}: integer expected\n")
        elif waited >= lim:
            say(f"worker_grace: gave up after {waited}s (youngest {age}s old)")
            return 1
        s = 300 - age + 5
        say(f"worker_grace: youngest worker process is {age}s old: waiting {s}s")
        time.sleep(s)
        waited += s


if __name__ == "__main__":
    sys.exit(main(sys.argv))
