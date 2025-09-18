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

"""
Run multiple performance tests against multiple commits and/or
the current workspace.
HEAD is tested if commits are not provided. Tags (like HEAD) can be specified.
The output is in {clonedir}/doc_{datetime}.
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from rrperf import compare, git
from rrperf import run as suite_run
import rrperf.args as args


def build_rocroller(
    repo: Path,
    checkout_dir: Path,
    project_dir: Path,
    commit: str,
    threads: int = 32,
) -> Path:
    """
    Build the rocRoller GEMM client in the project directory.

    The build directory path is returned.
    """
    if not checkout_dir.is_dir():
        git.clone(repo, checkout_dir)
        git.checkout(checkout_dir, commit)

    if git.is_dirty(checkout_dir) and commit != "current":
        print(f"Warning: {commit} is dirty")

    build_dir = project_dir / "build_perf"
    build_dir.mkdir(parents=True, exist_ok=True)

    mx_datagen_git_url_env_var = "ROCROLLER_MXDATAGENERATOR_GIT_URL"
    mx_datagen_git_tag_env_var = "ROCROLLER_MXDATAGENERATOR_GIT_TAG"
    mx_datagen_git_url = os.environ.get(mx_datagen_git_url_env_var)
    mx_datagen_git_tag = os.environ.get(mx_datagen_git_tag_env_var)

    if not mx_datagen_git_url:
        print(
            f"Warning: {mx_datagen_git_url_env_var} not defined. Using mxDataGeneator Git URL in CMakeLists.txt."
        )
    if not mx_datagen_git_tag:
        print(
            f"Warning: {mx_datagen_git_tag_env_var} not defined. Using mxDataGeneator Git tag in CMakeLists.txt."
        )

    mx_datagen_git_url_flag = (
        "-DMXDATAGENERATOR_GIT_URL=" + mx_datagen_git_url if mx_datagen_git_url else ""
    )
    mx_datagen_git_tag_flag = (
        "-DMXDATAGENERATOR_GIT_TAG=" + mx_datagen_git_tag if mx_datagen_git_tag else ""
    )

    subprocess.run(
        [
            "cmake",
            f"-B {build_dir}",
            f"-S {project_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DROCROLLER_ENABLE_TIMERS=ON",
            "-DROCROLLER_ENABLE_FETCH=ON",
            "-DCMAKE_PREFIX_PATH='/opt/rocm;/opt/rocm/llvm'",
            "-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DROCROLLER_ENABLE_CPPCHECK=OFF",
            mx_datagen_git_url_flag,
            mx_datagen_git_tag_flag,
            "../",
        ],
        cwd=str(project_dir),
        check=True,
    )

    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            str(threads),
            "--target",
            "all_clients",
        ],
        cwd=str(project_dir),
        check=True,
    )

    return build_dir


def ancestral_targets(targets: List[str]):
    orig_project_dir = git.top()
    targets = git.rev_list(orig_project_dir, targets[0], targets[-1])
    targets.reverse()
    targets = [git.short_hash(orig_project_dir, x) for x in targets]
    if len(targets) == 0:
        raise RuntimeError(
            """No targets. Check `git rev-list` and make sure
            commits are listed from oldest to newest."""
        )
    return targets


def get_args(parser: argparse.ArgumentParser):
    common_args = [
        args.x_value,
        args.normalize,
        args.y_zero,
        args.plot_median,
        args.plot_min,
        args.exclude_boxplot,
        args.group_results,
        args.rundir,
        args.suite,
    ]
    for arg in common_args:
        arg(parser)

    parser.add_argument(
        "--clonedir",
        type=str,
        help="Base directory for repo clone destinations.",
        default=".",
    )
    parser.add_argument(
        "--ancestral",
        action="store_true",
        help="Test every commit between first and last commits.  Off by default.",
        required=False,
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Test the repository in its current state.  Off by default.",
        required=False,
    )
    parser.add_argument(
        "commits", type=str, nargs="*", help="Commits/tags/branches to test."
    )

    parser.add_argument(
        "--no-fail",
        action="append",
        default=[],
        help="Commits/tags/branches where a failure does not cause the"
        " overall command to fail.",
    )


def run(args):
    """Run performance tests against multiple commits"""
    autoperf(**args.__dict__)


def autoperf(
    commits: List[str],
    clonedir: str,
    rundir: str,
    no_fail: List[str] = None,
    current: bool = False,
    ancestral: bool = False,
    suite: str = None,
    filter=None,
    normalize=False,
    y_zero=False,
    plot_median=False,
    plot_min=False,
    exclude_boxplot=False,
    x_value: str = "timestamp",
    **kwargs,
):
    if no_fail is None:
        no_fail = []

    monorepo_dir = git.top()

    targets = [git.short_hash(monorepo_dir, x) for x in commits]
    no_fail_targets = frozenset(
        [git.short_hash(monorepo_dir, x) for x in no_fail if x != "current"]
    )

    if len(targets) + (current) <= 1:
        targets.append(git.short_hash(monorepo_dir))  # HEAD

    if ancestral:
        targets = ancestral_targets(targets)

    if current:
        targets.append("current")

    top = Path(clonedir).resolve()
    top.mkdir(parents=True, exist_ok=True)
    os.chdir(str(top))

    results = []
    success = True
    success_no_fail = True

    for target in targets:
        checkout_dir = top / f"build_{target}"
        if target == "current":
            checkout_dir = monorepo_dir

        build_dir: Path = build_rocroller(
            monorepo_dir,
            checkout_dir,
            checkout_dir / "shared" / "rocroller",
            target,
        )
        target_success, result_dir = suite_run.run_cli(
            build_dir=build_dir, rundir=rundir, suite=suite, filter=filter, recast=True
        )
        if target in no_fail_targets:
            success_no_fail &= target_success
        else:
            success &= target_success
        results.append(result_dir)

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = top / f"doc_{date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    compare.compare(
        results,
        format="console",
        output=sys.stdout,
        normalize=normalize,
    )

    output_file = output_dir / f"comparison_{date}.html"
    with open(output_file, "w+") as f:
        compare.compare(
            results,
            format="html",
            output=f,
            normalize=normalize,
            y_zero=y_zero,
            plot_median=plot_median,
            plot_min=plot_min,
            plot_box=not exclude_boxplot,
            x_value=x_value,
        )
    print(f"Wrote {output_file}")

    if not success_no_fail:
        print("Failures occurred in branches marked no-fail.")
    if not success:
        raise RuntimeError("Some jobs failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_args(parser)

    parsed_args = parser.parse_args()
    run(parsed_args)
