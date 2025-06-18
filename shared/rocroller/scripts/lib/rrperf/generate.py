################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
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

"""Generate all kernels from a benchmark suite."""

import argparse
import os
import subprocess

from itertools import chain
from pathlib import Path
from typing import Dict

import rrperf

from rrperf.utils import sjoin


def generate_kernels(
    generator, architecture: str, build_dir: Path, work_dir: Path, env: Dict[str, str]
) -> bool:

    already_run = set()
    failed = []

    for i, problem in enumerate(generator):
        if problem in already_run:
            continue

        cmd = problem.command(architecture=architecture, generate_only=True)
        log = (work_dir / f"{problem.group}-{i:06d}.log").resolve()
        rr_env = {k: str(v) for k, v in env.items() if k.startswith("ROC")}
        rr_env_str = sjoin([f"{k}={v}" for k, v in rr_env.items()])

        with log.open("w") as f:
            print(f"# env: {rr_env_str}", file=f, flush=True)
            print(f"# command: {sjoin(cmd)}", file=f, flush=True)
            print(f"# token: {repr(problem)}", file=f, flush=True)
            print("running:")
            print(f"  command: {sjoin(cmd)}")
            print(f"  wrkdir:  {work_dir.resolve()}")
            print(f"  log:     {log.resolve()}")
            p = subprocess.run(cmd, stdout=f, cwd=build_dir, env=env, check=False)
            status = None
            if p.returncode == 0:
                status = "ok"
            else:
                status = "error"
                failed.append((i, problem))
            print(f"  status:  {status}", flush=True)

        already_run.add(problem)

    if len(failed) > 0:
        print(f"Failed {len(failed)} problems:")
        for i, problem in failed:
            cmd = list(map(str, problem.command()))
            print(f"{i}: {' '.join(cmd)}")

    return len(failed) == 0


def generate(
    architecture: str = None,
    suite: str = None,
    rundir: str = None,
    build_dir: str = None,
    **kwargs,
):
    """Generate kernels!"""

    generator = rrperf.utils.empty()
    if suite is not None:
        generator = chain(generator, rrperf.utils.load_suite(suite))

    if build_dir is None:
        build_dir = rrperf.utils.get_build_dir()
    else:
        build_dir = Path(build_dir)

    env = dict(os.environ)
    env["ROCROLLER_ENFORCE_GRAPH_CONSTRAINTS"] = "1"

    run_dir = rrperf.utils.get_work_dir(rundir, build_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    git_commit = run_dir / "git-commit.txt"
    try:
        hash = rrperf.git.full_hash(build_dir)
        git_commit.write_text(f"{hash}\n")
    except Exception:
        git_commit.write_text("NO_COMMIT\n")

    success = generate_kernels(generator, architecture, build_dir, run_dir, env)

    if not success:
        raise RuntimeError("Some jobs failed.")


def get_args(parser: argparse.ArgumentParser):
    common_args = [
        rrperf.args.rundir,
        rrperf.args.suite,
    ]
    for arg in common_args:
        arg(parser)

    parser.add_argument(
        "--arch",
        help="Architecture to generate for (eg, gfx950).",
        dest="architecture",
    )


def run(args):
    """Generate kernels!"""
    generate(**args.__dict__)
