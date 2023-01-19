import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from rrperf.problems import GEMMRun
from rrperf.run import get_build_dir, get_build_env, load_suite


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
    subprocess.run(
        [
            "omniperf",
            "profile",
            "-n",
            str(omniperf_workload_dir),
            "-p",
            str(working_dir),
            "--",
        ]
        + executable,
        cwd=cwd,
        env=env,
    )
    omniperf_workload_dir = next((Path(working_dir) / omniperf_workload_dir).glob("*"))
    subprocess.run(
        [
            "omniperf",
            "analyze",
            "-p",
            str(omniperf_workload_dir),
            "-o",
            str(output),
        ],
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
    problem: GEMMRun, name: str, output: Path, build_dir: Path, env: Dict[str, str]
):
    with tempfile.TemporaryDirectory() as working_dir_name:
        print("Created temporary directory", working_dir_name)
        run_omniperf(
            working_dir_name,
            problem.command(),
            output,
            "profile_" + name,
            build_dir,
            env,
        )


def run(
    output_dir: str,
    tensile_repo: str,
    build_dir: str = None,
    suite: str = None,
    config: Path = None,
    **kwargs,
):
    output_dir = Path(output_dir)
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

        env = get_build_env(build_dir)

        for i, problem in enumerate(load_suite(suite)):
            profile_rr(
                problem,
                str(i),
                output_dir / f"results_{i:02}.txt",
                build_dir,
                env,
            )
