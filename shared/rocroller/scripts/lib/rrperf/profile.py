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
Run Omniperf against a RocRoller kernel or a Tensile guidepost.
Specify a YAML config file to invoke Tensile to build a kernel to be profiled.
Alternatively, specify an rrperf suite to profile a RocRoller kernel.
These kernels are profiled with Omniperf.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import rrperf.args as args
from rrperf.problems import GEMMRun
from rrperf.run import get_build_dir, load_suite


def has_omniperf() -> bool:
    return shutil.which("omniperf")


def run_omniperf(
    working_dir: Path,
    executable: List[str],
    output: Path,
    omniperf_workload_dir: Path = "profiling",
    cwd: Path = ".",
    env: Dict[str, str] = None,
):
    cmd = [
        "omniperf",
        "profile",
        "-n",
        str(omniperf_workload_dir),
        "-p",
        str(working_dir),
        "--",
    ] + executable
    print(" ".join(cmd))
    subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
    )
    omniperf_workload_dir = next((Path(working_dir) / omniperf_workload_dir).glob("*"))
    cmd = [
        "omniperf",
        "analyze",
        "-p",
        str(omniperf_workload_dir),
        "-o",
        str(output),
    ]
    print(" ".join(cmd))
    subprocess.run(
        cmd,
        env=env,
    )


def profile_tensile(config: Path, output_dir: Path, tensile_repo: Path):
    tensile_path: Path = tensile_repo / "Tensile/bin/Tensile"
    omniperf_workload: Path = Path("profile_tensile")

    if not tensile_path.exists():
        raise FileNotFoundError(f"Tensile not found at {tensile_path}")

    with tempfile.TemporaryDirectory() as working_dir_name:
        print("Created temporary directory", working_dir_name)
        subprocess.run(
            [str(tensile_path), str(config), working_dir_name],
            check=True,
        )
        run_scripts = list(Path(working_dir_name).rglob("*/00_Final/build/run.sh"))
        if len(run_scripts) != 1:
            print(f"Bad config: found {len(run_scripts)} run.sh files", run_scripts)
            return

        output: Path = output_dir / "results_tensile.txt"
        run_omniperf(working_dir_name, [str(run_scripts[0])], output, omniperf_workload)


def profile_rr(
    problem: GEMMRun, name: str, output_dir: Path, build_dir: Path, env: Dict[str, str]
):
    i = 0
    output = output_dir / f"results_{name}.txt"
    while True:
        working_dir = output_dir / f"data_{i:02}"
        try:
            working_dir.mkdir()
            break
        except FileExistsError:
            i += 1
    print("Created working directory", working_dir)
    run_omniperf(
        working_dir,
        problem.command(),
        output,
        "profile_" + name,
        build_dir,
        env,
    )


def get_args(parser: argparse.ArgumentParser):
    common_args = [
        args.suite,
    ]
    for arg in common_args:
        arg(parser)

    parser.add_argument("--config", help="Location of Tensile YAML config file.")
    parser.add_argument(
        "--output_dir",
        help="Directory where the Omniperf results are written.",
        default=".",
    )
    parser.add_argument(
        "--tensile_repo",
        help="Directory where Tensile repository is located.",
        default="/home/tensile",
    )


def run(args):
    """Run Omniperf against a RocRoller kernel or a Tensile guidepost."""
    profile(**args.__dict__)


def profile(
    output_dir: str,
    tensile_repo: str,
    build_dir: str = None,
    suite: str = None,
    config: Path = None,
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensile_repo = Path(tensile_repo)

    if not has_omniperf():
        raise FileNotFoundError("Could not find omniperf")

    if config is not None:
        profile_tensile(config, output_dir, tensile_repo)

    if suite is not None:
        if build_dir is None:
            build_dir = get_build_dir()
        else:
            build_dir = Path(build_dir)

        for i, problem in enumerate(load_suite(suite)):
            profile_rr(
                problem,
                f"{i:02}",
                output_dir,
                build_dir,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_args(parser)

    parsed_args = parser.parse_args()
    run(parsed_args)
