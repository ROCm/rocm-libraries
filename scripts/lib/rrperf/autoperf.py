""" autoperf routines """

import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from rrperf import compare, git
from rrperf import run as suite_run


def build_rocroller(
    repo: Path,
    project_dir: Path,
    commit: str,
    threads: int = 32,
) -> Path:
    """
    Build the rocRoller GEMM client in the project directory.

    The build directory path is returned.
    """
    if not project_dir.is_dir():
        git.clone(repo, project_dir)
        git.checkout(project_dir, commit)

    if git.is_dirty(project_dir) and commit != "current":
        print(f"Warning: {commit} is dirty")

    build_dir = project_dir / "build_perf"
    build_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DROCROLLER_ENABLE_TIMERS=ON",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DSKIP_CPPCHECK=On",
            "../",
        ],
        cwd=str(build_dir),
        check=True,
    )

    subprocess.run(
        [
            "make",
            "-j",
            str(threads),
            "all_clients",
        ],
        cwd=str(build_dir),
        check=True,
    )

    return build_dir


def run(
    commits: List[str],
    clonedir: str,
    rundir: str,
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
    orig_project_dir = git.top()

    targets = [git.short_hash(orig_project_dir, x) for x in commits]

    if len(targets) <= 1:
        targets.append(git.short_hash(orig_project_dir))  # HEAD

    if ancestral:
        targets = git.rev_list(orig_project_dir, targets[0], targets[-1])
        targets.reverse()
        targets = [git.short_hash(orig_project_dir, x) for x in targets]
        if len(targets) == 0:
            raise RuntimeError(
                """No targets. Check `git rev-list` and make sure
                commits are listed from oldest to newest."""
            )

    if current:
        targets.append("current")

    top = Path(clonedir).resolve()
    top.mkdir(parents=True, exist_ok=True)
    os.chdir(str(top))

    results = []
    success = True
    for target in targets:
        project_dir = top / f"build_{target}"
        if target == "current":
            project_dir = orig_project_dir

        build_dir: Path = build_rocroller(
            orig_project_dir,
            project_dir,
            target,
        )
        target_success, result_dir = suite_run.run(
            build_dir=build_dir, rundir=rundir, suite=suite, filter=filter, recast=True
        )
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

    if not success:
        raise RuntimeError("Some jobs failed.")
