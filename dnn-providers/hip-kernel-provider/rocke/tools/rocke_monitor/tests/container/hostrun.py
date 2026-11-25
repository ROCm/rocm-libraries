# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What the container tests' drivers (image.py, canary.py, tarball.py) share on the host: folder names as the shell
names them, the container engine (docker or podman), a command's combined output, the indented copy of it, the
PASS/FAIL lines, and a build settings file of your values that only you read (write_settings). Python starts the engine directly, so a Windows host (Docker Desktop) gets every argument as written
and every host path in its C:/... form."""

import json
import os
import pathlib
import subprocess
import sys


def setup_output() -> None:
    """Each line out at once, and any byte a command printed passed through as it is."""
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(line_buffering=True, errors="surrogateescape")


def logical_pwd() -> str:
    """The folder as `pwd` names it: $PWD when it is this folder, else the physical path."""
    p = os.environ.get("PWD", "")
    try:
        if os.path.isabs(p) and os.path.samefile(p, "."):
            return p
    except OSError:
        pass
    return os.getcwd()


def folder(path: str) -> str:
    """`$(cd PATH && pwd)`: the folder's absolute path (its C:/... form on Windows); "" with cd's message when there is
    none."""
    if not os.path.isdir(path):
        print(
            f"{os.path.basename(sys.argv[0])}: cd: can't cd to {path}", file=sys.stderr
        )
        return ""
    return host_path(os.path.normpath(os.path.join(logical_pwd(), path)))


def host_path(path: str) -> str:
    """A host path as the engine takes it: as is on Linux and macOS, C:/... on Windows."""
    return pathlib.Path(path).as_posix() if os.name == "nt" else path


def engine_answers(name: str) -> bool:
    try:
        return (
            subprocess.run(
                [name, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).returncode
            == 0
        )
    except OSError:
        return False


def pick_engine(given: str) -> str:
    """CONTAINER_ENGINE or --engine, else docker or podman, whichever answers; "" when neither does (said)."""
    if given:
        return given
    for name in ("docker", "podman"):
        if engine_answers(name):
            return name
    print("neither docker nor podman answers: start one, or name it with --engine")
    return ""


def quiet(argv: list) -> int:
    """A command with its standard output dropped (its errors still shown): its exit code."""
    sys.stdout.flush()
    try:
        return subprocess.run(argv, stdout=subprocess.DEVNULL).returncode
    except OSError as exc:
        print(f"{argv[0]}: {exc.strerror}", file=sys.stderr)
        return 127


def capture(argv: list, both: bool = True) -> tuple[int, str]:
    """`out=$(COMMAND 2>&1)`: its exit code and its output without the trailing newlines (with both=False, its errors
    go to this terminal instead)."""
    sys.stdout.flush()
    try:
        p = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT if both else None
        )
    except OSError as exc:
        return 127, f"{argv[0]}: {exc.strerror}"
    return p.returncode, p.stdout.decode("utf-8", "surrogateescape").rstrip("\n")


def indented(out: str) -> None:
    """`printf '%s\\n' "$out" | sed 's/^/   | /'`."""
    for line in out.split("\n"):
        print("   | " + line)


def run_opts(default: str) -> list:
    """SMOKE_RUN_OPTS (default DEFAULT), split into words."""
    value = os.environ.get("SMOKE_RUN_OPTS")
    return (default if value is None or value == "" else value).split()


class Results:
    """The PASS/FAIL lines, and whether one failed."""

    def __init__(self):
        self.failed = 0

    def result(self, status: str, text: str) -> None:
        print(f"{status}  {text}")
        if status != "PASS":
            self.failed = 1


def need_value(args: list, option: str) -> str:
    """The value after OPTION, or exit 2 as the shell's `set -u` did on a missing one."""
    if len(args) < 2:
        print(
            f"{os.path.basename(sys.argv[0])}: {option} needs a value", file=sys.stderr
        )
        sys.exit(2)
    return args[1]


def write_settings(folder: str, values: dict) -> str:
    """VALUES, a build settings document's values, as a build settings file in the layout of the package whose
    rkmon_lib is first on the path (rkmon_lib/file_types.py BUILD), in FOLDER, readable by you only: its path.
    """
    from rkmon_lib import file_types

    path = os.path.join(folder, "build_settings.json")
    build = file_types.BUILD
    data = {"kind": build.kind, "schema_version": build.version, **values}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return path


def help_lines(doc: str, n: int) -> str:
    """The first N lines of a usage block as `--help` prints them: each line as a comment."""
    lines = doc.strip("\n").splitlines()[:n]
    return "".join(("# " + line if line else "#") + "\n" for line in lines)
