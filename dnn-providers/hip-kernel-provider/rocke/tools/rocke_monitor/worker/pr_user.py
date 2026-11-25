# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pull requests' code for a worker tool: the command line that runs a command as PR code, the one home
of that call for the worker tools (rkmon_lib/prcode.py runs PR code for the package's own Python; the launcher:
docker/rk_prcode.py). Standard library only: the worker tools import it from their installed copy, worker/mkwt.py and
rkmon_lib/prcode.py from the package's worker/.

- `wrap(cmd, env, cwd)`: the command line that runs `cmd` in a PR runner's slot of its own, and the stdin that hands
  it `env`: `rk pr-run --env /dev/stdin -C DIR -- CMD...`, which takes the slot, runs the command there in DIR
  (default: the slot's home) with only what worker/pr_env.py lists of `env`, and frees the slot. Without the launcher
  (RKMON_PR_RUNNER unset: a host, the tests) `cmd` through the pr_env.py beside this file, and None: a host runs PR
  code as the session user, with the same list of the environment, and pr_env.HOST's variables as they are.
- `launched(argv)`: whether a command line is one `wrap` made here (its stdin is then the environment: `env_data`):
  its shape, around this package's rk.
- `values()`: the PR runners' group, a slot's home folder (a folder of the slot's, as `-C` names it) and the package's
  rk, from the container's layout (rkmon_lib/container_layout.py) of the package these tools are part of. No file found
  at run time is run for them: an installed copy (paths.worker_tools) reads the values its install wrote beside it
  (VALUES_FILE) and nothing else; the package's own worker/, where no install wrote, imports its package's
  rkmon_lib.container_layout. `group()`: the runners' group, which the trees PR code reads are given (worker/mkwt.py).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

RUNNER_VAR = "RKMON_PR_RUNNER"  # the image's launcher (docker/Dockerfile sets it)
HERE = pathlib.Path(__file__).resolve().parent
# What an install writes beside the copies it makes (rkmon_lib/deploy.py, install_worker_tools): the values (KEYS:
# `values_text`), and its record of each copy's digest. A copy with either beside it is an installed one.
VALUES_FILE = "pr_user.json"
MANIFEST = ".installed.json"
KEYS = ("group", "home", "rk")
ENV_FILE = "/dev/stdin"  # where `rk pr-run --env` reads the environment `wrap` hands it


def values_text(group_name: str, home: str, rk: str) -> str:
    """VALUES_FILE's text: the PR runners' group, a slot's home folder and the package's rk."""
    return (
        json.dumps(
            {"group": group_name, "home": home, "rk": rk}, indent=1, sort_keys=True
        )
        + "\n"
    )


def values() -> dict:
    """{"group", "home", "rk"}. An installed copy reads its install's VALUES_FILE and nothing else; the package's own
    worker/ imports rkmon_lib.container_layout (the one the process has, else its package's) and names its package's
    rk."""
    f = HERE / VALUES_FILE
    if not (os.path.lexists(f) or os.path.lexists(HERE / MANIFEST)):
        if "rkmon_lib" not in sys.modules:
            # A tool of the package's worker/ run on its own: its package.
            sys.path.insert(0, str(HERE.parent))
        from rkmon_lib import container_layout

        return {
            "group": container_layout.PR_GROUP,
            "home": container_layout.SLOT_HOME,
            "rk": str(HERE.parent / "rk"),
        }
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        got = {k: d[k] for k in KEYS}
    except (OSError, ValueError, KeyError, TypeError):
        got = {}
    if len(got) == len(KEYS) and all(isinstance(v, str) and v for v in got.values()):
        return got
    raise RuntimeError(
        f"{f} does not name the PR runners' group, a slot's home and rk: `rk worker-tools --install` writes it again"
    )


def group() -> str:
    """The PR runners' group: the trees PR code reads are this group's."""
    return values()["group"]


def runner(environ=None) -> str:
    """The launcher's path when PR code runs in the PR runners' slots here (the image), else ""."""
    e = os.environ if environ is None else environ
    return e.get(RUNNER_VAR, "")


def env_data(env: dict) -> bytes:
    """`env` as NUL-separated NAME=VALUE: what `rk pr-run --env` reads (and reduces to worker/pr_env.py's list)."""
    return b"".join(
        f"{k}={v}".encode("utf-8", "surrogateescape") + b"\0" for k, v in env.items()
    )


def wrap(cmd: list, env: dict, cwd=None, environ=None) -> tuple[list, bytes | None]:
    """(the command line, its stdin): `cmd` in a slot of its own, in `cwd` (absolute, or a folder of the slot's;
    default: its home), with `env` on stdin; without the launcher `cmd` through pr_env.py (its environment reduced to
    the list there), and None.
    """
    if not runner(environ):
        return [
            sys.executable,
            "-I",
            str(HERE / "pr_env.py"),
            *[str(c) for c in cmd],
        ], None
    v = values()
    return [
        v["rk"],
        "pr-run",
        "--env",
        ENV_FILE,
        "-C",
        str(cwd or v["home"]),
        "--",
        *[str(c) for c in cmd],
    ], env_data(env)


def launched(argv: list, environ=None) -> bool:
    """Whether `argv` is a command line `wrap` made here: this package's rk running a command in a slot, its
    environment on stdin."""
    a = [str(x) for x in argv]
    return (
        bool(runner(environ))
        and len(a) > 7
        and a[1:5] == ["pr-run", "--env", ENV_FILE, "-C"]
        and a[6] == "--"
        and a[0] == values()["rk"]
    )
