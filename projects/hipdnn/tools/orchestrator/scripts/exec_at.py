#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Run an executable the flow itself produced, at a path that is only valid partway
through the run.

`configs/tools.yaml` resolves every registered tool before the first step launches,
which is wrong for a binary a build step of this same flow has not built yet --
`hipdnn_validate_descriptors` does not exist until the flow has configured with
`HIPDNN_ENABLE_KERNEL_INGESTOR=ON` and built it. A step that runs that binary cannot
name it as a tool; it has to be told the path as a step argument and check for
existence itself, at the moment it actually needs to run, with a diagnostic that
says which step was supposed to have produced it.

    exec_at.py --exe BUILD/hipdnn_validate_descriptors \\
        --path-prepend ROCM/bin --produced-by "the configure+build step" \\
        -- --graph graph.json

`--path-prepend` exists because on Windows a HIP binary that dynamically loads
`amdhip64_7.dll` (or any other ROCm DLL not on the default search path) does not fail
with a linker error or a Python-style traceback -- the OS terminates it before `main`
runs, with exit code 0xC0000135 ("a required DLL was not found"). That looks exactly
like a crash, and the actual cause -- ROCm's `bin` directory missing from `PATH` --
is invisible unless something prepends it first. This script does that prepending for
the one process it launches, rather than requiring the whole orchestrator to run with
ROCm on its PATH.

Exit codes: the child's own exit code, forwarded unchanged, so pass/fail of the
binary under test is indistinguishable from running it directly. Except: 127 if
`--exe` does not exist -- deliberately outside the range a real process ordinarily
returns, so "the binary was never found" is distinguishable in a run report from
"the binary ran and returned 127 itself".
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, help="executable to run")
    parser.add_argument(
        "--path-prepend",
        action="append",
        default=[],
        metavar="DIR",
        help="directory to prepend to the child's PATH; repeatable, first wins",
    )
    parser.add_argument(
        "--produced-by",
        default=None,
        metavar="TEXT",
        help="name of the step that should have produced --exe, for the diagnostic "
        "when it is missing",
    )
    parser.add_argument("--cwd", default=None, help="working directory for the child")
    parser.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="'-- ARG...': the child's argv (everything after --)",
    )
    args = parser.parse_args()

    argv = args.argv
    if argv and argv[0] == "--":
        argv = argv[1:]

    exe = Path(args.exe)
    if not exe.exists():
        print(f"exec_at: {exe} does not exist", file=sys.stderr)
        if args.produced_by:
            print(
                f"it is produced by {args.produced_by}; that step did not run, or "
                "it ran and produced nothing",
                file=sys.stderr,
            )
        return 127

    env = os.environ.copy()
    if args.path_prepend:
        prefix = os.pathsep.join(args.path_prepend)
        env["PATH"] = prefix + os.pathsep + env.get("PATH", "")

    cmd = [str(exe), *argv]
    print(f"exec_at: running {exe} {argv}", file=sys.stderr)

    result = subprocess.run(cmd, cwd=args.cwd, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
