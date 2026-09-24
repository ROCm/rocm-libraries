#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe the intended execution host; early mode makes no installation claim."""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


#: An architecture token, not a family prefix. Every shipping gfx name carries
#: at least three characters after `gfx` (gfx90a, gfx942, gfx1100), so `gfx9`
#: is a family the caller must resolve before a sweep can claim it measured one.
ARCH_TOKEN = r"gfx[0-9a-f]{3,}"


class ProbeUnavailable(Exception):
    """The inspection utility could not be run, so nothing was observed.

    Distinct from a negative observation: a missing `rocminfo` is a packaging
    or platform difference, not a statement about the host's GPUs.
    """


#: Inspection utilities in probe order. `rocminfo` reports the architecture as
#: `Name:`/`gfx...`; `hipInfo` reports it as `gcnArchName:` and is what the
#: Windows ROCm wheels ship instead. Trying both turns an unobservable
#: condition into an observed one wherever it can.
DEVICE_TOOLS = ("rocminfo", "hipInfo")


def device_info(arch: str, *, cwd=None, env=None) -> str:
    """Return successful device evidence containing the exact requested arch.

    Raises ValueError when a utility ran and contradicted the request, and
    ProbeUnavailable only when none of them could be run at all.
    """
    unrunnable = []
    for tool in DEVICE_TOOLS:
        try:
            result = subprocess.run(
                [tool], cwd=cwd, env=env, capture_output=True, text=True
            )
        except OSError as exc:
            unrunnable.append(f"cannot run {tool}: {exc}")
            continue
        if result.returncode:
            raise ValueError(
                f"{tool} exited {result.returncode}: {result.stderr.strip()}"
            )
        found = set(
            re.findall(rf"(?<![A-Za-z0-9_]){ARCH_TOKEN}(?![A-Za-z0-9_])", result.stdout)
        )
        if arch not in found:
            raise ValueError(
                f"wanted {arch}, found: {', '.join(sorted(found)) or 'no GPU agents'}"
            )
        return result.stdout
    raise ProbeUnavailable("; ".join(unrunnable))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("early", "installed"))
    parser.add_argument("--arch", required=True, help="Exact architecture, e.g. gfx942")
    parser.add_argument(
        "--sweep-root",
        type=Path,
        required=True,
        help="Existing execution-host-visible directory",
    )
    parser.add_argument(
        "--install", type=Path, help="Existing install prefix; installed mode only"
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(ARCH_TOKEN, args.arch):
        parser.error("--arch must be an exact gfx architecture token")
    if args.mode == "early" and args.install is not None:
        parser.error("early mode rejects --install; installation is a later gate")
    if args.mode == "installed" and args.install is None:
        parser.error("installed mode requires --install")
    print(f"host: {socket.gethostname()}")
    failures = []
    unobserved = []
    try:
        device_info(args.arch)
        print(f"OK device {args.arch} present")
    except ProbeUnavailable as exc:
        unobserved.append(str(exc))
    except (OSError, ValueError) as exc:
        failures.append(str(exc))
    if args.install is not None:
        if not args.install.is_dir():
            failures.append(f"install tree not visible: {args.install}")
        else:
            print(f"OK install tree visible: {args.install.resolve()}")
    try:
        if not args.sweep_root.is_dir():
            raise ValueError(f"sweep root does not exist: {args.sweep_root}")
        with tempfile.TemporaryFile(dir=args.sweep_root) as probe:
            probe.write(b"device probe\n")
            probe.flush()
        print(f"OK sweep root writable: {args.sweep_root.resolve()}")
    except (OSError, ValueError) as exc:
        failures.append(str(exc))
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    for note in unobserved:
        print(f"UNOBSERVED {note}", file=sys.stderr)
    if failures:
        return 1
    if unobserved:
        print(
            "UNOBSERVED a required condition was not observed; this is not a negative "
            "result. Establish it by other means and record the substitution.",
            file=sys.stderr,
        )
        return 3
    print(
        f"{args.mode} feasibility satisfied; no plugin loading or numerical correctness claim"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
