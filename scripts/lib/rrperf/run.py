"""Run routines."""

import os
import pandas as pd
import pathlib
import subprocess
import tempfile
import datetime
import yaml

from dataclasses import dataclass

from itertools import chain

import rrperf

import sys


def empty():
    yield from ()


def load_suite(suite):
    """Load performance suite from rrsuites.py."""

    tdef = rrperf.git.top() / "scripts" / "rrsuites.py"
    code, ns = compile(tdef.read_text(), str(tdef), "exec"), {}
    exec(code, ns)
    return ns[suite]()


def work_directory():
    """Return a new work directory path."""

    commit = rrperf.git.short_hash(".")
    date = datetime.date.today().strftime("%Y-%m-%d")
    serial = len(list(pathlib.Path(".").glob(f"{date}-{commit}-*")))
    return pathlib.Path(f"{date}-{commit}-{serial:03d}")


def submit_directory(suite: str, wrkdir: pathlib.Path, ptsdir: pathlib.Path):
    """Consolidate performance data and submit it to SOMEWHERE.

    Performance data is read from .yaml files in the work directory
    for the given suite.  Consolidated data is written to a SOMEWHERE
    directory and submitted.
    """
    results = []
    for jpath in wrkdir.glob(f"{suite}-*.yaml"):
        results.extend(yaml.load(jpath.read_text()))
    df = pandas.DataFrame(results)
    df.to_csv(f"{str(ptsdir)}/{suite}-benchmark.csv", index=False)
    # TODO: add call to SOMEWHERE to submit


def from_token(token: str):
    yield rrperf.problems.upcast_to_run(eval(token, rrperf.problems.__dict__))


def run(token=None, suite=None, submit=False, filter=None, **kwargs):
    """Run benchmarks!

    Implements the CLI 'run' subcommand.
    """

    if suite is None and token is None:
        suite = "all"

    generator = empty()
    if suite is not None:
        generator = chain(generator, load_suite(suite))
    if token is not None:
        generator = chain(generator, from_token(token))

    top = rrperf.git.top()

    env = dict(os.environ)
    if "ROCROLLER_ARCHITECTURE_FILE" not in env:
        arch = top / "build" / "source" / "rocRoller" / "GPUArchitecture_def.msgpack"
        env["ROCROLLER_ARCHITECTURE_FILE"] = arch

    wrkdir = work_directory()
    wrkdir.mkdir(parents=True)

    # pts.create_git_info(str(wrkdir / "git-commit.txt"))
    git_commit = wrkdir / "git-commit.txt"
    git_commit.write_text(rrperf.git.full_hash(top) + "\n")
    # pts.create_specs_info(str(wrkdir / "machine-specs.txt"))
    machine_specs = wrkdir / "machine-specs.txt"
    machine_specs.write_text(str(rrperf.specs.get_machine_specs(0)) + "\n")

    print(f"rrperf: {' '.join(sys.argv)}")
    print(f"work directory: {wrkdir.resolve()}")

    already_run = set()
    result = True

    for i, problem in enumerate(generator):
        if filter is not None:
            pass

        if problem in already_run:
            continue

        yaml = (wrkdir / f"{problem.group}-{i:06d}.yaml").resolve()
        problem.set_output(yaml)
        cmd = list(map(str, problem.command()))
        scmd = " ".join(cmd)
        log = yaml.with_suffix(".log")
        with log.open("w") as f:
            print(f"# command: {scmd}", file=f, flush=True)
            print(f"# token: {repr(problem)}", file=f, flush=True)
            print(f"running:")
            print(f"  command: {scmd}")
            print(f"  log:     {log.resolve()}")
            print(f"  token:   {problem.token}", flush=True)
            p = subprocess.run(cmd, stdout=f, cwd="build", env=env, check=False)
            result &= p.returncode == 0
            if p.returncode == 0:
                print("  status:  ok", flush=True)
            else:
                print("  status:  error", flush=True)

        already_run.add(problem)

    if submit:
        ptsdir = wrkdir / "rocRoller"
        ptsdir.mkdir(parents=True)
        # XXX if running single token, suite might be None
        submit_directory(suite, wrkdir, ptsdir)

    return result
