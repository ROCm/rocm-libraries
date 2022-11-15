""" autoperf routines """

import datetime
import os
import subprocess
from pathlib import Path

from rrperf import compare, git
from rrperf import run as suite_run


def build_rocroller(
    orig_project_dir: Path,
    project_dir: Path,
    commit: str,
    threads: int = 32,
) -> Path:
    """
    Builds the RocRoller GEMM client in the project directory.
    The original project directory is used to clone the project.
    The build_perf directory path is returned.
    """
    if not project_dir.is_dir():
        git.clone(
            orig_project_dir,
            project_dir,
        )
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
            "gemm",
        ],
        cwd=str(build_dir),
        check=True,
    )

    return build_dir


def run(
    commits: str,
    workdir: str,
    current: bool = False,
    ancestral: bool = False,
    filter=None,
    **kwargs,
):
    orig_project_dir = git.top()

    targets = commits.split(",")
    targets = [git.short_hash(orig_project_dir, x) for x in targets]

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

    print(targets)

    top = Path(workdir).resolve()

    top.mkdir(parents=True, exist_ok=True)
    os.chdir(str(top))

    suite_results = []
    for target in targets:
        project_dir = top / f"build_{target}"
        if target == "current":
            project_dir = orig_project_dir

        build_dir: Path = build_rocroller(
            orig_project_dir,
            project_dir,
            target,
        )
        (result, result_dir) = suite_run.run(
            build_dir=build_dir, working_dir=project_dir, filter=filter
        )
        suite_results.append(result_dir)

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = top / f"doc_{date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"comparison_{date}.html", "w+") as f:
        compare.compare(
            suite_results,
            format="html",
            output=f,
        )
