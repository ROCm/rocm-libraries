# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The sandbox the reviewer model's own code runs in (the offload passes' probes and mutants, reviewer_tools.py): no
network, nothing writable but the review folder, a time limit, and the whole process tree killed with the run.

- bubblewrap (`bwrap`), when it can make its namespaces: the whole file system read-only, the review folder
  writable, no network, its own process ids, and it dies with its parent;
- else unshare (util-linux) with an unprivileged user namespace: no network, and the folders that matter (the
  working folder, the home, the monitor's home) bound read-only, the review folder writable again on top;
- else none: `kind()` is "" and the tools that run the model's own code refuse, saying what to install. A saved probe
  (written by the verifier) still runs as the checks run it, outside any sandbox.

Every path in the wrapper is the harness's own, never the model's.
"""

import os
import pathlib
import shlex
import shutil
import subprocess

_KIND = None  # probed once per process: "bwrap", "unshare" or ""
KILL_AFTER_S = 10


def _works(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def kind() -> str:
    """The sandbox this machine allows: "bwrap", "unshare" or "" (none). RKMON_SANDBOX=none|unshare|bwrap caps it (a
    test, or a host whose namespaces misbehave)."""
    global _KIND
    if _KIND is None:
        want = os.environ.get("RKMON_SANDBOX", "").strip().lower()
        _KIND = ""
        if want != "none":
            if (
                want in ("", "bwrap")
                and shutil.which("bwrap")
                and _works(
                    [
                        "bwrap",
                        "--ro-bind",
                        "/",
                        "/",
                        "--dev",
                        "/dev",
                        "--unshare-net",
                        "--",
                        "true",
                    ]
                )
            ):
                _KIND = "bwrap"
            elif (
                want in ("", "bwrap", "unshare")
                and shutil.which("unshare")
                and _works(
                    [
                        "unshare",
                        "--user",
                        "--map-root-user",
                        "--net",
                        "--mount",
                        "--",
                        "true",
                    ]
                )
            ):
                _KIND = "unshare"
    return _KIND


def describe() -> str:
    return {
        "bwrap": "bubblewrap: read-only but the review folder, no network",
        "unshare": "unshare: no network; the working folder and the homes read-only",
        "": "none (install bubblewrap, or allow unprivileged user namespaces)",
    }[kind()]


def wrap(
    argv: list[str],
    writable: pathlib.Path,
    cwd: pathlib.Path,
    readonly=(),
    timeout_s: int = 600,
) -> list[str]:
    """`argv` inside the sandbox, under `timeout`. ValueError when there is none (kind() is "")."""
    w, c = str(pathlib.Path(writable).resolve()), str(pathlib.Path(cwd).resolve())
    limit = ["timeout", "-k", str(KILL_AFTER_S), str(int(timeout_s))]
    k = kind()
    if k == "bwrap":
        return limit + [
            "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--bind",
            w,
            w,
            "--unshare-net",
            "--unshare-pid",
            "--die-with-parent",
            "--new-session",
            "--chdir",
            c,
            "--",
            *argv,
        ]
    if k == "unshare":
        ro = [
            str(pathlib.Path(p).resolve())
            for p in readonly
            if p and pathlib.Path(p).is_dir()
        ]
        # The read-only folders first; the review folder, which may sit inside one of them, writable again on top.
        steps = [
            f"mount --bind {shlex.quote(d)} {shlex.quote(d)} && mount -o remount,bind,ro {shlex.quote(d)}"
            for d in dict.fromkeys(ro)
            if d != w
        ]
        steps += [
            f"mount --bind {shlex.quote(w)} {shlex.quote(w)} && mount -o remount,bind,rw {shlex.quote(w)}",
            f"cd {shlex.quote(c)}",
            'exec "$@"',
        ]
        return limit + [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--mount",
            "--",
            "sh",
            "-ec",
            "\n".join(steps),
            "sh",
            *argv,
        ]
    raise ValueError("no sandbox on this machine: " + describe())


def env(base: dict, writable: pathlib.Path) -> dict:
    """The environment of a sandboxed run: `base` without credentials, its home, temporary and cache folders in the
    review folder, no bytecode written."""
    w = pathlib.Path(writable)
    e = {
        k: v
        for k, v in base.items()
        if not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
    }
    for name, sub in (("HOME", "home"), ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache")):
        (w / sub).mkdir(parents=True, exist_ok=True)
        e[name] = str(w / sub)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    return e
