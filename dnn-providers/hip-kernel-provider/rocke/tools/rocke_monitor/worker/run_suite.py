#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Usage: run_suite.py <worktree folder name in the working folder> plat|lib [outdir]

plat: platform pytest suite. lib: develop's library test files run against <worktree>.
ROCKE_LLVM_FLAVOR defaults to llvm22 (override in the environment). Paths come from RKMON_* (rkenv.py).
The suite's output goes to <outdir>/out.txt (default <scratch>/<kind>_<worktree>), which ends with its `exit=` line;
the last two lines are printed. The tests are PR code: they run in a PR runner's slot where there are PR runners (the
container: pr_user.py), in its home; this script, not they, writes <outdir>.
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import pr_user  # noqa: E402
import rkenv  # noqa: E402
from steps import PROG, make_dirs, tail, words  # noqa: E402

PYTEST_ARGS = [
    "-q",
    "-p",
    "no:cacheprovider",
    "--no-header",
    "-rfE",
    "--continue-on-collection-errors",
]


def main(argv: list) -> int:
    rkenv.load()
    D = os.environ["RKMON_WORKING_DIR"]
    PY = f"{os.environ.get('RKMON_VENV', '')}/bin/python"
    W = argv[1] if len(argv) > 1 else ""
    KIND = argv[2] if len(argv) > 2 else ""
    OUT = (
        argv[3] if len(argv) > 3 else ""
    ) or f"{os.environ.get('RKMON_SCRATCH', '')}/{KIND}_{W}"
    K = os.environ["RKMON_ROCKE_PATH"]
    make_dirs(OUT)
    try:
        os.chdir(OUT)
    except OSError as exc:
        print(f"{PROG}: cd: {OUT}: {exc.strerror}", file=sys.stderr, flush=True)
        return 1
    os.environ["ROCKE_LLVM_FLAVOR"] = os.environ.get("ROCKE_LLVM_FLAVOR") or "llvm22"
    env = dict(os.environ)
    if KIND == "plat":
        files = [
            f"{D}/{W}/{K}/platform/tests",
            f"{D}/{W}/{K}/platform/python/rocke/tests",
        ]
    else:
        sys.stdout.flush()
        listed = subprocess.run(
            ["git", "-C", f"{D}/wtdev", "ls-files", f"{K}/library/tests"],
            stdout=subprocess.PIPE,
        ).stdout.decode("utf-8", "surrogateescape")
        tests = [x for x in listed.split("\n") if _is_test(x)]
        files = words(" ".join(f"{D}/{W}/{x}" for x in tests))
        env["ROCKE_CPP_QUIET_FALLBACK"] = "1"
    with open("out.txt", "wb") as out:
        try:
            argv, data = pr_user.wrap(
                [PY, "-I", "-B", "-m", "pytest", *files, *PYTEST_ARGS], env
            )
            rc = subprocess.run(
                argv,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                input=data,
            ).returncode
        except FileNotFoundError:
            out.write(f"{PROG}: {PY}: No such file or directory\n".encode())
            rc = 127
        except PermissionError:
            out.write(f"{PROG}: {PY}: Permission denied\n".encode())
            rc = 126
    with open("out.txt", "a", encoding="utf-8") as out:
        out.write(f"exit={rc}\n")
    tail("out.txt", 2)
    return 0


def _is_test(path: str) -> bool:
    """A library test file: grep -E '/test_[^/]*\\.py$'."""
    return re.search(r"/test_[^/]*\.py$", path) is not None


if __name__ == "__main__":
    sys.exit(main(sys.argv))
