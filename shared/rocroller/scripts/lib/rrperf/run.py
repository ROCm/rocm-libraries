# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run a benchmark suite."""

import argparse
import datetime
import importlib.util
import os
import subprocess
from dataclasses import fields
from itertools import chain
from pathlib import Path

import pandas as pd
import rrperf
import rrperf.dump_csv
import yaml


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


def run_problems(
    generator,
    build_dir: Path,
    work_dir: Path,
    env: dict[str, str],
    id_filter: list[str],
    l2: bool,
) -> bool:

    SOLUTION_NOT_SUPPORTED_ON_ARCH = 3

    already_run = set()
    result = True
    failed = []

    for i, problem in enumerate(generator):
        if id_filter is not None and not any(
            problem.id.startswith(filt) for filt in id_filter
        ):
            continue

        if problem in already_run:
            continue

        id = getattr(problem, "id", None)
        yaml = (work_dir / f"{problem.group}-{i:06d}.yaml").resolve()
        problem.set_output(yaml)
        cmd = problem.command()
        log = yaml.with_suffix(".log")
        rr_env = {k: str(v) for k, v in env.items() if k.startswith("ROC")}
        rr_env_str = " ".join([f"{k}={v}" for k, v in rr_env.items()])

        if l2:
            counters = str(yaml.resolve().parent / yaml.stem)
            cmd = [
                "rocprofv3",
                "--pmc=TCC_HIT,TCC_MISS",
                "--output-file=" + counters,
                "--output-format=json",
                "--",
            ] + cmd

        with log.open("w") as f:
            scmd = " ".join(cmd)
            print(f"# env: {rr_env_str}", file=f, flush=True)
            print(f"# command: {scmd}", file=f, flush=True)
            print(f"# token: {repr(problem)}", file=f, flush=True)
            print("running:")
            print(f"  id: {id}")
            print(f"  command: {scmd}")
            print(f"  wrkdir:  {work_dir.resolve()}")
            print(f"  log:     {log.resolve()}")
            p = subprocess.run(cmd, stdout=f, cwd=build_dir, env=env, check=False)
            status = None
            if p.returncode == 0:
                status = "ok"
            elif p.returncode == SOLUTION_NOT_SUPPORTED_ON_ARCH:
                status = "skipped (not supported on " + rrperf.utils.rocm_gfx() + ")"
            else:
                status = "error"
                result = False
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

    return result


def generate_missing_attr_value(run, attr):
    """Generate value for an option missing in previous rrperf version."""
    match attr:
        case "workgroupMapping":
            wgm_dim = getattr(run, "workgroupMappingDim")
            wgm_value = getattr(run, "workgroupMappingValue")
            return (wgm_dim, wgm_value)
        case "matchMemoryAccess":
            return True
        case "unroll_x" | "unroll_y":
            return 0
        case "storeLDS_D":
            store = getattr(run, "store")
            return "LDS" in store
        case "streamK":
            # Old version used bool, new version uses str
            streamK = getattr(run, "streamK", "None")
            if isinstance(streamK, str):
                return streamK != "None"
            return streamK
        case "streamKTwoTile":
            streamK = getattr(run, "streamK", "None")
            return streamK == "TwoTile"
        case "streamKTwoTileDPFirst":
            streamK = getattr(run, "streamK", "None")
            return streamK == "TwoTileDPFirst"
        case _:
            raise RuntimeError(
                f"Cannot handle attribute missing in previous rrperf version: {attr}"
            )


def backcast(generator, build_dir):
    """Reconstruct run objects from `generator` into run objects from previous rrperf version."""
    pdef = build_dir.parent / "scripts" / "lib" / "rrperf" / "problems.py"
    spec = importlib.util.spec_from_file_location("problems", str(pdef))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for run in generator:
        className = run.__class__.__name__
        backClass = getattr(module, className, None)
        if backClass is not None:
            backObj = backClass(
                **{
                    f.name: (
                        getattr(run, f.name)
                        if hasattr(run, f.name)
                        else generate_missing_attr_value(run, f.name)
                    )
                    for f in fields(backClass)
                }
            )
            yield backObj


def get_args(parser: argparse.ArgumentParser):
    common_args = [
        rrperf.args.rundir,
        rrperf.args.suite,
        rrperf.args.id_filter,
    ]
    for arg in common_args:
        arg(parser)

    parser.add_argument(
        "--submit",
        help="Submit results to SOMEWHERE.",
        action="store_true",
        default=False,
    )
    parser.add_argument("--token", help="Benchmark token to run.")
    parser.add_argument(
        "--rocm_smi",
        default="rocm-smi",
        help="Location of rocm-smi.",
    )
    parser.add_argument(
        "--pin_clocks",
        action="store_true",
        help="Pin clocks before launching benchmark clients.",
    )
    parser.add_argument(
        "--l2",
        action="store_true",
        help="Collect L2 performance counters (TCC_HIT and TCC_MISS).",
    )
    parser.add_argument(
        "--dump_csv",
        help="Dump benchmark CSV with included headers.",
        action="store_true",
        default=False,
    )


def run(args):
    """Run benchmarks!"""
    run_cli(**args.__dict__)


def run_cli(  # noqa: C901
    token: str = None,
    suite: str = None,
    submit: bool = False,
    id_filter: list[str] = None,
    rundir: str = None,
    build_dir: str = None,
    rocm_smi: str = "rocm-smi",
    pin_clocks: bool = False,
    recast: bool = False,
    l2: bool = False,
    **kwargs,
) -> tuple[bool, Path]:
    """Run benchmarks!

    Implements the CLI 'run' subcommand.
    """

    if pin_clocks:
        rrperf.rocm_control.pin_clocks(rocm_smi)

    if suite is None and token is None:
        if rrperf.utils.rocm_gfx().startswith("gfx120"):
            suite = "all_gfx120X"
        else:
            suite = "all"

    generator = rrperf.utils.empty()
    if suite is not None:
        generator = chain(generator, rrperf.utils.load_suite(suite))
    if token is not None:
        generator = chain(generator, from_token(token))
    if recast:
        generator = backcast(generator, build_dir)

    if build_dir is None:
        build_dir = rrperf.utils.get_build_dir()
    else:
        build_dir = Path(build_dir)

    env = dict(os.environ)
    env["ROCROLLER_ENFORCE_GRAPH_CONSTRAINTS"] = "1"
    env["ROCROLLER_AUDIT_CONTROL_TRACERS"] = "1"

    rundir = rrperf.utils.get_work_dir(rundir, build_dir)
    rundir.mkdir(parents=True, exist_ok=True)

    # pts.create_git_info(str(wrkdir / "git-commit.txt"))
    git_commit = rundir / "git-commit.txt"
    try:
        hash = rrperf.git.full_hash(build_dir)
        git_commit.write_text(f"{hash}\n")
    except Exception:
        git_commit.write_text("NO_COMMIT\n")
    # pts.create_specs_info(str(wrkdir / "machine-specs.txt"))
    machine_specs = rundir / "machine-specs.txt"
    machine_specs.write_text(str(rrperf.specs.get_machine_specs(0, rocm_smi)) + "\n")

    timestamp = rundir / "timestamp.txt"
    timestamp.write_text(str(datetime.datetime.now().timestamp()) + "\n")

    result = run_problems(generator, build_dir, rundir, env, id_filter, l2)

    if submit:
        ptsdir = rundir / "rocRoller"
        ptsdir.mkdir(parents=True)
        # XXX if running single token, suite might be None
        submit_directory(suite, rundir, ptsdir)

    if kwargs.get("dump_csv", False):
        rrperf.dump_csv.dump_csv(suite, rundir)

    return result, rundir
