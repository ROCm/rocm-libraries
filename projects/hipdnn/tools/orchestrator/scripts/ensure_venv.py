#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Create the IngestorGenerator's virtualenv before the tool registry ever sees it.

The orchestrator resolves every tool named in `configs/tools.yaml` before the first
step launches, and hard-fails the whole run if one of them is missing. That check is
correct for tools the machine already owns and wrong for this one: the
IngestorGenerator's `.venv` interpreter does not exist on a fresh checkout, is not
supposed to be checked in, and is only ever created by a step of the flow itself.
Registering it as a tool would make the run refuse to start because a step which has
not run yet has not run yet.

So the venv is not a tool, it is a gate: a step that runs first, creates the
interpreter if it is missing, installs the pinned requirements, and then proves the
result actually works by importing the two packages every later step depends on
(`yaml` for config loading, `jinja2` for template rendering) *through that
interpreter*, not by trusting pip's own exit code. A `pip install` can exit 0 while
resolving to a broken or partial environment (a stale wheel cache, a half-finished
compile, a version pinned out of existence for the platform); the only fact worth
reporting is whether the venv can actually do the imports the rest of the flow will
ask of it.

    ensure_venv.py --dir GEN/.venv --requirements GEN/requirements.txt --out venv.json

Re-running is safe and cheap: an existing interpreter is never recreated, so a flow
retry after a later step fails does not pay for `python -m venv` again. `--upgrade`
forces `pip install -r` to run again over an existing venv (picking up a requirements
change) without touching the interpreter itself.

Exit codes: 0 the report was written (this includes `ready: 0` -- the flow asserts on
that field, so a broken venv is evidence, not a crash), 1 usage or I/O failure (the
venv directory could not be created, the requirements file does not exist, or the
report could not be written).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _interpreter(venv_dir: Path) -> Path:
    """Where `python -m venv` puts the interpreter, which differs by platform and is
    not something callers should have to hard-code per OS."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _tail(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", required=True, help="venv directory to create or reuse"
    )
    parser.add_argument(
        "--requirements", required=True, help="requirements.txt to install"
    )
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="re-run pip install even if the interpreter already exists",
    )
    args = parser.parse_args()

    venv_dir = Path(args.dir)
    requirements = Path(args.requirements)
    out_path = Path(args.out)

    if not requirements.is_file():
        print(f"error: --requirements not found: {requirements}", file=sys.stderr)
        return 1

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"error: could not create --out directory: {error}", file=sys.stderr)
        return 1

    python = _interpreter(venv_dir)
    created = False
    installed = False
    pip_output = ""

    if not python.exists():
        try:
            venv_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"error: could not create venv parent: {error}", file=sys.stderr)
            return 1
        create = _run([sys.executable, "-m", "venv", str(venv_dir)])
        pip_output += create.stdout + create.stderr
        if create.returncode != 0 or not python.exists():
            feedback = (
                f"`python -m venv {venv_dir}` failed (exit {create.returncode}) or "
                f"did not produce {python}. Last output:\n\n"
                f"{_tail(pip_output, 40)}"
            )
            report = {
                "venv_dir": venv_dir.as_posix(),
                "python": python.as_posix(),
                "created": 0,
                "installed": 0,
                "ready": 0,
                "packages": [],
                "feedback": feedback,
            }
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"venv creation failed: {venv_dir}")
            return 0
        created = True

    if created or args.upgrade:
        install = _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(requirements),
            ]
        )
        pip_output += install.stdout + install.stderr
        installed = install.returncode == 0
    else:
        installed = False

    ready_check = _run([str(python), "-c", "import yaml, jinja2"])
    ready = python.exists() and ready_check.returncode == 0

    freeze = _run([str(python), "-m", "pip", "freeze"])
    packages = (
        [line for line in freeze.stdout.splitlines() if line.strip()]
        if freeze.returncode == 0
        else []
    )

    if ready:
        feedback = ""
    elif not python.exists():
        feedback = f"venv interpreter does not exist at {python}."
    elif (created or args.upgrade) and not installed:
        feedback = (
            f"`pip install -r {requirements}` failed (exit {install.returncode}). "
            f"Last output:\n\n{_tail(pip_output, 40)}"
        )
    else:
        feedback = (
            f"venv interpreter exists at {python} but `import yaml, jinja2` failed "
            f"through it (exit {ready_check.returncode}). Last output:\n\n"
            f"{_tail(ready_check.stdout + ready_check.stderr, 40)}"
        )

    report = {
        "venv_dir": venv_dir.as_posix(),
        "python": python.as_posix(),
        "created": 1 if created else 0,
        "installed": 1 if installed else 0,
        "ready": 1 if ready else 0,
        "packages": packages,
        "feedback": feedback,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"venv {venv_dir}: created={created} installed={installed} ready={ready} "
        f"({len(packages)} package(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
