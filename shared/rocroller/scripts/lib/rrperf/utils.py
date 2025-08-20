################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import datetime
import functools
import os
import subprocess

from pathlib import Path

import rrperf


def empty():
    yield from ()


def sjoin(xs):
    return " ".join([str(x) for x in xs])


def load_suite(suite: str):
    """Load performance suite from rrsuites.py."""
    return getattr(rrperf.rrsuites, suite)()


def first_problem_from_suite(suite: str):
    for problem in load_suite(suite):
        return problem
    raise RuntimeError(f"Suite {suite} has no problems.")


def try_getting_commit(repo):
    if repo is not None:
        try:
            return rrperf.git.short_hash(repo)
        except Exception:
            pass
    return None


def get_commit(rundir: str = None, build_dir: Path = None) -> str:
    commit = try_getting_commit(build_dir)
    if commit is None:
        commit = try_getting_commit(rundir)
    if commit is None:
        commit = try_getting_commit(".")
    if commit is None:
        commit = try_getting_commit(Path(__file__).resolve().parent)
    if commit is None:
        commit = "NO_COMMIT"
    return commit


def get_work_dir(rundir: str = None, build_dir: Path = None) -> Path:
    """Return a new work directory path."""

    date = datetime.date.today().strftime("%Y-%m-%d")
    root = "."
    commit = get_commit(rundir, build_dir)

    if rundir is not None:
        root = Path(rundir)

    serial = len(list(Path(root).glob(f"{date}-{commit}-*")))
    return root / Path(f"{date}-{commit}-{serial:03d}")


def get_build_dir() -> Path:
    varname = "ROCROLLER_BUILD_DIR"
    if varname in os.environ:
        return Path(os.environ[varname])
    default = rrperf.git.top() / "build"
    if default.is_dir():
        return default

    raise RuntimeError(f"Build directory not found.  Set {varname} to override.")


@functools.cache
def rocm_gfx():
    """Return GPU architecture (gfxXXXX) for local GPU device."""
    output = None
    try:
        output = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return None

    for line in output.splitlines():
        if line.startswith("  Name:"):
            _, arch, *_ = list(map(lambda x: x.strip(), line.split()))
            if arch.startswith("gfx"):
                return arch

    return None
