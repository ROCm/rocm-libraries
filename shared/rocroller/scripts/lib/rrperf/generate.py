# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate all kernels from a benchmark suite."""

import argparse
import os
import subprocess
from itertools import chain
from pathlib import Path

import rrperf
from rrperf.utils import sjoin


def generate_kernels(
    generator,
    architecture: str,
    build_dir: Path,
    work_dir: Path,
    env: dict[str, str],
    id_filter: list[str],
) -> bool:

    already_run = set()
    failed = []

    for i, problem in enumerate(generator):
        if id_filter is not None and not any(
            problem.id.startswith(filt) for filt in id_filter
        ):
            continue

        if problem in already_run:
            continue

        id = getattr(problem, "id", None)
        cmd = problem.command(architecture=architecture, generate_only=True)
        log = (work_dir / f"{problem.group}-{i:06d}.log").resolve()
        rr_env = {k: str(v) for k, v in env.items() if k.startswith("ROC")}
        rr_env_str = sjoin([f"{k}={v}" for k, v in rr_env.items()])

        with log.open("w") as f:
            print(f"# env: {rr_env_str}", file=f, flush=True)
            print(f"# command: {sjoin(cmd)}", file=f, flush=True)
            print(f"# token: {repr(problem)}", file=f, flush=True)
            print("running:")
            print(f"  id: {id}")
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
        ids = [getattr(problem, "id", None) for i, problem in failed]
        print("")
        print(f"Failed {len(failed)} problems ids:")
        print(" ".join([str(id) for id in ids]))
        print("")
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
    id_filter: list[str] = None,
    **kwargs,
):
    """Generate kernels!"""

    generator = rrperf.utils.empty()
    if suite is not None:
        generator = chain(generator, rrperf.utils.load_suite(suite))
    else:
        print("No suite specified.")
        return

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

    success = generate_kernels(
        generator, architecture, build_dir, run_dir, env, id_filter
    )

    if not success:
        raise RuntimeError("Some jobs failed.")


def get_args(parser: argparse.ArgumentParser):
    common_args = [
        rrperf.args.rundir,
        rrperf.args.suite,
        rrperf.args.id_filter,
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
