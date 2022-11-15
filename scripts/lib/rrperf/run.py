"""Run routines."""

import datetime
import os
import subprocess
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import rrperf
import yaml


def empty():
    yield from ()


def load_suite(suite: str):
    """Load performance suite from rrsuites.py."""

    tdef = rrperf.git.top() / "scripts" / "rrsuites.py"
    code, ns = compile(tdef.read_text(), str(tdef), "exec"), {}
    exec(code, ns)
    return ns[suite]()


def get_work_dir(working_dir: Path = None) -> Path:
    """Return a new work directory path."""

    date = datetime.date.today().strftime("%Y-%m-%d")
    root = "."
    if working_dir:
        try:
            commit = rrperf.git.short_hash(working_dir)
        except OSError:
            commit = rrperf.git.short_hash(root)
        finally:
            root = working_dir
    else:
        commit = rrperf.git.short_hash(root)
    serial = len(list(Path(root).glob(f"{date}-{commit}-*")))
    return Path(root) / Path(f"{date}-{commit}-{serial:03d}")


def submit_directory(suite: str, wrkdir: Path, ptsdir: Path) -> None:
    """Consolidate performance data and submit it to SOMEWHERE.

    Performance data is read from .yaml files in the work directory
    for the given suite.  Consolidated data is written to a SOMEWHERE
    directory and submitted.
    """
    results = []
    for jpath in wrkdir.glob(f"{suite}-*.yaml"):
        results.extend(yaml.load(jpath.read_text()))
    df = pd.DataFrame(results)
    df.to_csv(f"{str(ptsdir)}/{suite}-benchmark.csv", index=False)
    # TODO: add call to SOMEWHERE to submit


def from_token(token: str):
    yield rrperf.problems.upcast_to_run(eval(token, rrperf.problems.__dict__))


def get_build_dir() -> Path:
    varname = "ROCROLLER_BUILD_DIR"
    if varname in os.environ:
        return Path(os.environ[varname])
    default = rrperf.git.top() / "build"
    if default.is_dir():
        return default

    raise RuntimeError(f"Build directory not found.  Set {varname} to override.")


def run_problems(
    generator, build_dir: Path, work_dir: Path, env: Dict[str, str]
) -> bool:
    already_run = set()
    result = True

    for i, problem in enumerate(generator):
        if filter is not None:
            pass

        if problem in already_run:
            continue

        yaml = (work_dir / f"{problem.group}-{i:06d}.yaml").resolve()
        problem.set_output(yaml)
        cmd = list(map(str, problem.command()))
        scmd = " ".join(cmd)
        log = yaml.with_suffix(".log")
        with log.open("w") as f:
            print(f"# command: {scmd}", file=f, flush=True)
            print(f"# token: {repr(problem)}", file=f, flush=True)
            print("running:")
            print(f"  command: {scmd}")
            print(f"  log:     {log.resolve()}")
            print(f"  token:   {problem.token}", flush=True)
            p = subprocess.run(cmd, stdout=f, cwd=build_dir, env=env, check=False)
            result &= p.returncode == 0
            if p.returncode == 0:
                print("  status:  ok", flush=True)
            else:
                print("  status:  error", flush=True)

        already_run.add(problem)

    return result


def run(
    token: str = None,
    suite: str = None,
    submit: bool = False,
    filter: str = None,
    working_dir: Path = None,
    build_dir: Path = None,
    rocm_smi: str = "rocm-smi",
    pin_clocks: bool = False,
    **kwargs,
) -> Tuple[bool, Path]:
    """Run benchmarks!

    Implements the CLI 'run' subcommand.
    """

    if pin_clocks:
        rrperf.rocm_control.pin_clocks(rocm_smi)

    if suite is None and token is None:
        suite = "all"

    generator = empty()
    if suite is not None:
        generator = chain(generator, load_suite(suite))
    if token is not None:
        generator = chain(generator, from_token(token))

    top = rrperf.git.top()
    if build_dir is None:
        build_dir = get_build_dir()

    env = dict(os.environ)
    if "ROCROLLER_ARCHITECTURE_FILE" not in env:
        arch = build_dir / "source" / "rocRoller" / "GPUArchitecture_def.msgpack"
        env["ROCROLLER_ARCHITECTURE_FILE"] = arch

    working_dir = get_work_dir(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # pts.create_git_info(str(wrkdir / "git-commit.txt"))
    git_commit = working_dir / "git-commit.txt"
    git_commit.write_text(rrperf.git.full_hash(top) + "\n")
    # pts.create_specs_info(str(wrkdir / "machine-specs.txt"))
    machine_specs = working_dir / "machine-specs.txt"
    machine_specs.write_text(str(rrperf.specs.get_machine_specs(0, rocm_smi)) + "\n")

    timestamp = working_dir / "timestamp.txt"
    timestamp.write_text(str(datetime.datetime.now().timestamp()) + "\n")

    print(f"rrperf: {' '.join(sys.argv)}")
    print(f"work directory: {working_dir.resolve()}")

    result = run_problems(generator, build_dir, working_dir, env)

    if submit:
        ptsdir = working_dir / "rocRoller"
        ptsdir.mkdir(parents=True)
        # XXX if running single token, suite might be None
        submit_directory(suite, working_dir, ptsdir)

    return (result, working_dir)
