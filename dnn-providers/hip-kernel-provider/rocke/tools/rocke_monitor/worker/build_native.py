#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Usage: build_native.py <worktree folder name in the working folder, e.g. wt101m>

Builds librocke_core + native test binaries (<build>/<wt>/core) and the rocke_engine pybind module
(<build>/<wt>/pybind). Use with: PYTHONPATH=<platform>/python:<build>/<wt>/pybind
Paths come from the job's RKMON_* environment (rkenv.py): the working folder, rocKE's path, the build folder and the
checks' venv (cmake, ninja and pybind11 come from it). The builds run steps.build_jobs() jobs: the CPUs this process
may run on, within the cgroup's CPU quota and its memory (about 1.5 GB a job), or $RKMON_JOBS.
Isolation: always a from-scratch build of exactly this tree; nothing from an earlier build is reused.
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import rkenv  # noqa: E402
from steps import (
    capture,
    build_jobs,
    echo,
    last_lines,
    link,
    make_dirs,
    must,
    remove,
    run,
    tail,
    write_raw,
)  # noqa: E402


def step(build: str, name: str, *argv) -> None:
    """One build step, its output in <build>/<name>.log; a failure prints the log's end and ends the script."""
    echo(f"-- {name}")
    try:
        with open(f"{build}/{name}.log", "wb") as log:
            rc = run(list(argv), stdout=log, stderr=log)
    except OSError as exc:
        print(f"{build}/{name}.log: {exc.strerror}", file=sys.stderr, flush=True)
        rc = 1
    if rc:
        echo(f"FAILED: {name}")
        tail(f"{build}/{name}.log", 25)
        sys.exit(1)


def main(argv: list) -> int:
    rkenv.load()
    D = os.environ["RKMON_WORKING_DIR"]
    W = argv[1] if len(argv) > 1 else ""
    P = f"{D}/{W}/{os.environ['RKMON_ROCKE_PATH']}/platform"
    if not W:
        echo("usage: build_native.py <worktree folder name>")
        return 2
    build_root = os.environ.get("RKMON_BUILD", "")
    B = f"{build_root}/{W}"
    venv = os.environ.get("RKMON_VENV", "")
    PY = f"{venv}/bin/python"
    ramdisk = os.environ.get("RKMON_RAMDISK", "")
    if ramdisk and re.fullmatch(r"wt[0-9]+m?", W):
        # On a host a PR's build lives on the RAM disk (config ramdisk; worker/pr_env.py gives RKMON_RAMDISK to PR code
        # there only), and $RKMON_BUILD/<wt> links to it, so every path that names it works. In the container
        # $RKMON_BUILD is the job's slot's build folder, on the RAM disk already.
        remove(B)
        RB = f"{ramdisk}/build/{W}"
        remove(RB)
        if make_dirs(RB, build_root):
            return 1
        link(RB, B)
        B = RB
    else:
        remove(B)
        if make_dirs(B):
            return 1
    os.environ["PATH"] = f"{venv}/bin:{os.environ.get('PATH', '')}"
    jobs = f"-j{build_jobs()}"  # CPU affinity, the cgroup's quota and memory, or $RKMON_JOBS
    step(
        B,
        "core-configure",
        "cmake",
        "-S",
        P,
        "-B",
        f"{B}/core",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",  # tools/tidy_check.py reads each TU's command
    )
    step(B, "core-build", "cmake", "--build", f"{B}/core", jobs)
    pybind_dir = capture([PY, "-m", "pybind11", "--cmakedir"])[1]
    step(
        B,
        "pybind-configure",
        "cmake",
        "-S",
        f"{P}/cpp/bindings",
        "-B",
        f"{B}/pybind",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DROCKE_ENGINE_ARCHIVE={B}/core/librocke_core.a",
        f"-Dpybind11_DIR={pybind_dir}",
        f"-DPYTHON_EXECUTABLE={PY}",
    )
    step(B, "pybind-build", "cmake", "--build", f"{B}/pybind", jobs)
    must(
        [
            PY,
            "-c",
            "import rocke_engine; print('rocke_engine import OK:', rocke_engine.__file__)",
        ],
        env=dict(os.environ, PYTHONPATH=f"{P}/python:{B}/pybind"),
    )
    # The native tests' count, then the quick ones.
    core = f"{B}/core"
    if not os.path.isdir(core):
        print(
            f"{os.path.basename(argv[0])}: cd: {core}: No such file or directory",
            file=sys.stderr,
            flush=True,
        )
        return 1
    write_raw(last_lines(piped(["ctest", "-N"], core), 1))
    write_raw(
        last_lines(
            piped(
                [
                    "ctest",
                    "-R",
                    "rocke_(mma_frag_ssot|arch_target_identity|future_intrinsic_lowering|"
                    "ir_serialize_roundtrip)$",
                    "--output-on-failure",
                ],
                core,
                both=True,
            ),
            4,
        )
    )
    return 0


def piped(argv: list, cwd: str, both: bool = False) -> bytes:
    """argv's stdout (and stderr with `both`) as a pipe into `tail` reads it: nothing when it cannot start."""
    sys.stdout.flush()
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if both else None,
        ).stdout
    except OSError:
        msg = (
            f"{os.path.basename(sys.argv[0])}: {argv[0]}: command not found\n".encode()
        )
        if both:
            return msg
        sys.stderr.buffer.write(msg)
        sys.stderr.flush()
        return b""


if __name__ == "__main__":
    sys.exit(main(sys.argv))
