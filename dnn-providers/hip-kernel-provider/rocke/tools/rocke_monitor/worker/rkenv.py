# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The monitor's settings for a worker tool: the job's RKMON_* environment (rkmon_lib/config.py env), or, for a tool
run by hand, the toolkit's `rk env` ($RKMON_TOOLKIT, or monitor/ beside this folder: the default layout,
<WORKING_DIR>/{tools,monitor}).

- `get(name)`: one setting, for the Python tools (fetch_pr.py, review_agent.py, ...), from `rk env --json` when the
  job gave none.
- `load()`: the worker scripts' start (mkwt.py, review_agent_run.py, build_native.py, run_suite.py, kill_runner.py).
  Without RKMON_WORKING_DIR it reads the toolkit's `rk env` into this process's environment, so every tool the script
  runs inherits the same RKMON_* values; then it requires the working folder, the clone, the repository and rocKE's
  path, and exits 2 with the fix when one is missing.
- `clone_since(repo)`: the oldest commit date the shallow clone keeps (config clone_since), else the clone's own
  shallow boundary; "" for a full clone (never a failure).
- `toolkit()`: the toolkit's folder: $RKMON_TOOLKIT, else monitor/ beside this folder.
- `container_layout()`: the package's fixed values (rkmon_lib/container_layout.py, standard library only): the module
  this process imports, else the toolkit's. A worker tool installed apart from the package reads them so.
"""

import json
import os
import pathlib
import shlex
import subprocess
import sys

_loaded = None
REQUIRED = ("RKMON_WORKING_DIR", "RKMON_REPO_DIR", "RKMON_REPO", "RKMON_ROCKE_PATH")


def toolkit() -> pathlib.Path:
    """The toolkit's folder: $RKMON_TOOLKIT, else monitor/ beside this folder (the default layout)."""
    return pathlib.Path(
        os.environ.get("RKMON_TOOLKIT")
        or pathlib.Path(__file__).resolve().parent.parent / "monitor"
    )


def container_layout():
    """rkmon_lib/container_layout.py: the module this process imports, else the toolkit's."""
    try:
        from rkmon_lib import container_layout as layout
    except ImportError:
        sys.path.append(str(toolkit()))
        from rkmon_lib import container_layout as layout
    return layout


def _from_rk() -> dict:
    rk = toolkit() / "rk"
    if not rk.exists():
        return {}
    try:
        out = subprocess.run(
            [str(rk), "env", "--json"], capture_output=True, text=True, timeout=60
        ).stdout
        return json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def get(name: str, required: bool = True) -> str:
    """One setting; exits with the fix when a required one is missing."""
    global _loaded
    value = os.environ.get(name)
    if not value:
        if _loaded is None:
            _loaded = _from_rk()
        value = _loaded.get(name, "")
    if required and not value:
        raise SystemExit(
            f"{name} is not set: run this from a monitor job, or check config.json with `rk doctor`"
        )
    return value


def _fail(message: str) -> None:
    sys.stdout.flush()
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(2)


def _export(text: str) -> None:
    """`rk env`'s lines (`export NAME=VALUE ...`, the values shell-quoted) into os.environ, as the shell's eval did."""
    for line in text.splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] == "export":
            words = words[1:]
        for word in words:
            name, eq, value = word.partition("=")
            if (
                not eq
                or not name
                or name[0].isdigit()
                or not name.replace("_", "a").isalnum()
                or not name.isascii()
            ):
                raise ValueError(word)
            os.environ[name] = value


def load() -> None:
    """The worker scripts' settings: the job's RKMON_* environment, else the toolkit's `rk env`; then the required
    settings, or exit 2 with the fix."""
    if not os.environ.get("RKMON_WORKING_DIR"):
        rk = str(toolkit() / "rk")
        if not os.access(rk, os.X_OK):
            _fail(
                f"rkenv: no RKMON_* settings and no toolkit at {rk}: run this from a monitor job, or set RKMON_TOOLKIT"
            )
        sys.stdout.flush()
        try:
            out = subprocess.run([rk, "env"], stdout=subprocess.PIPE).stdout.decode(
                "utf-8", "surrogateescape"
            )
            _export(out)
        except (OSError, ValueError):
            _fail("rkenv: `rk env` failed")
    for name in REQUIRED:
        if not os.environ.get(name):
            _fail(f"rkenv: {name} is not set (check config.json with `rk doctor`)")


def clone_since(repo: str) -> str:
    """The oldest commit date the shallow clone keeps (config clone_since), else the clone's own shallow boundary;
    "" for a full clone."""
    if os.environ.get("RKMON_CLONE_SINCE"):
        return os.environ["RKMON_CLONE_SINCE"]
    shallow = pathlib.Path(repo) / ".git" / "shallow"
    try:
        if shallow.stat().st_size == 0:
            return ""
        first = (
            shallow.read_bytes().split(b"\n", 1)[0].decode("utf-8", "surrogateescape")
        )
        p = subprocess.run(
            ["git", "-C", repo, "log", "-1", "--format=%cs", first],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    return p.stdout.decode("utf-8", "surrogateescape").rstrip("\n")
