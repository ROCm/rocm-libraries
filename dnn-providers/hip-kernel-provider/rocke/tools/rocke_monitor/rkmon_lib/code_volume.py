# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The code volume's names and the facts of a code tree, with the standard library only: docker/rk_root.py loads this
file by its path, as root, to install a release, to go back to the code kept and to keep the volume at each start,
and the package imports it.

The code volume (rkmon_lib/container_layout.py CODE_DIR) keeps each release in RELEASES/<id>: a package's code with its
record (RELEASE_FILE); CURRENT, a link to the release that runs; KEPT, a link to the code CURRENT named before the last
install but a start's (a release, whatever image it was installed over, or the image's own code), which a rollback
goes back to; and MOVED, the one-time move's marks: a file for each copy of a secret the move has dealt with, which it
never makes again (docker/rk_root.py `copies`). Root owns all of it, and no other user writes any of it: root installs
a release (`rkmon apply`, and a start: LEAVE_KEPT) and puts the code kept back in CURRENT's place (`rkmon rollback`),
one change at a time (LOCK), a release unpacked in a STAGE folder first. A release is read-only to every user, as the
image's own code is (docker/Dockerfile: a-w, a+rX): its folders DIR_MODE, its files FILE_MODE, and RUN_MODE for a file
its package runs (an execute bit). The launcher (rk, at the package's top) runs CURRENT's release in place of the
image's own code only while it was installed over this very image (its record's base_image is the image's digest) and
is whole (WHOLE); the writer's broker (docker/rk_write.py) picks the code by the same rule, and runs it only once it
has checked that it is root's and read-only all the way. Root removes a release only once nothing uses it (`in_use`).

- IMAGE_BOUND: the package's files the image keeps its own copies of, whatever release runs: what it is built from
  (IMAGE_BUILT_FROM: the Dockerfile, the scripts its build copies, the Python requirements), and what it runs from its
  own folder (IMAGE_RUNS): the launcher, which picks the code each process tree runs, the entry point, the scripts it
  runs as root (the start's id, layout and root steps) and through sudo (the writer's broker and the PR runner's
  launcher), the GitHub request cap's counter, the first gh on every PATH (docker/rk_gh.py), and the few files of the
  package those load by their paths (tests/test_releases.py follows them, and holds this list to them). Each uses the
  standard library only. Code whose IMAGE_BOUND files differ from an image's runs only on a new image
  (QUICK_BUILD_GUIDE.md); a package update carries the rest. NEW_IMAGE: the exit code of an install refused for
  that (docker/rk_root.py code-install), whose fix `rkmon apply` says (rkmon_lib/releases.py new_image_fix).
- LEAVE_KEPT: code-install's option for a start's install (docker/rkmon start_code), which leaves KEPT as it is: a
  rollback still goes back to the code from before the last apply, whatever a start installed since.
- `runs(mode) -> bool`: whether a file of the mode MODE is one its package runs: any execute bit, as the image's build
  reads it (docker/Dockerfile: a+rX gives the file all three) and as an install does (RUN_MODE: docker/rk_root.py).
- `fingerprints(tree, entries=("",)) -> {path: str}`: the SHA-256 of each file's bytes and whether it runs (`runs`),
  for each file of the code tree TREE that ENTRIES name (a file, or a folder by its trailing /; "" names the whole
  tree), its record aside. It loads the standard library only, so that its source may run inside an image to
  fingerprint the image's own copies (rkmon_lib/confirm.py CODE_PROBE): it runs alone there, so it spells out `runs`'
  rule, which tests/test_container.py holds it to.
- `image_changes(a, b) -> [path]`: the IMAGE_BOUND files that differ between the code trees A and B, by their bytes or
  their execute bit, sorted: none when the code B runs on an image built from A.
- `digest(tree, entries=("",)) -> str`: the digest of the code tree TREE, a SHA-256 in hex: of the path, bytes and
  execute bit of each file that ENTRIES name, as `fingerprints` finds them, its record aside.
- `missing(tree) -> [path]`: the files of WHOLE that TREE lacks or holds empty, in WHOLE's order: none for a whole one.
- `running(folder, image_id) -> path | None`: the release that runs in the code volume's folder FOLDER, resolved:
  CURRENT's while it was installed over the image whose code digest is IMAGE_ID and is whole; None when the image's own
  code runs (the launcher's rule).
- `release_info(root) -> dict`: the record of the release in the folder ROOT: its id, its package's version, the digest
  of its files, when root installed it, and base_image, the digest of the image it was installed over; {} when there
  is none, or none that reads.
- `version(tree) -> str`: the VERSION of the package in TREE, its last line that is neither blank nor a comment; ""
  when there is none.
- PIN: the variable that names the code a process tree runs: the folder the rk that started the tree picked, which it
  exports. PROC: the folder of the processes.
- WORKER_VALUES: the worker tools' values file, by its path in the working folder (rkmon_lib/paths.py UNDER_WORKING
  worker_tools, and worker/pr_user.py VALUES_FILE: tests/test_code_volume.py holds it to those): it names the rk each
  PR run starts.
- `in_use(folder, values=None, proc=PROC) -> {name}`: the names of the releases in the code volume's folder FOLDER
  that are in use: each that a live process in PROC works in (its working folder: rk runs in its code's folder) or
  runs (its PIN), of the processes this user may read; and the one whose rk the worker tools' values file VALUES
  names, as each PR run starts that rk. OSError when PROC cannot be listed. Root may not read another user's
  processes, so docker/rk_root.py runs it as each user who may run a release (`main`).
- `main(argv) -> int`: `code_volume.py in-use FOLDER [VALUES]` prints the names `in_use` finds, a JSON list.
- `listed(names, most=MOST) -> str`: NAMES joined by commas: the first MOST of them, then how many more there are.
- `remove(path) -> bool`: a file or a whole folder removed, its read-only folders too; True once it is gone. A user
  other than root first makes each folder of it writable to itself. Root changes no mode on the way: it needs no write
  bit (it keeps DAC_OVERRIDE), and a folder another user swapped for a link would carry a chmod to the link's target
  (shutil.rmtree follows no link).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import stat
import sys

# What the image is built from: its recipe, the scripts its build copies, and the requirements of its venvs.
IMAGE_BUILT_FROM = (
    "docker/Dockerfile",
    "docker/pkgs.sh",
    "docker/param.sh",
    "docker/fetch.sh",
    "docker/masked.sh",
    "docker/install_rocm.sh",
    "docker/install_atlassian.sh",
    "docker/install_sdk.sh",
    "docker/claude_cli.py",
    "requirements/",
)
# What the image runs from its own folder: the launcher, the entry point, its scripts (the GitHub request cap's counter
# among them), and the package's files those load by their paths (tests/test_releases.py follows them, and holds this
# list to them).
IMAGE_RUNS = (
    "rk",
    "docker/entrypoint.sh",
    "docker/rk_ids.py",
    "docker/rk_layout.py",
    "docker/rk_root.py",
    "docker/rk_write.py",
    "docker/rk_prcode.py",
    "docker/rk_gh.py",
    "rkmon_lib/claude_env_shape.py",
    "rkmon_lib/code_volume.py",
    "rkmon_lib/container_layout.py",
    "rkmon_lib/custom_headers.py",
    "rkmon_lib/gh_cap.py",
    "rkmon_lib/paths.py",
    "rkmon_lib/whole_file.py",
    "worker/perms.py",
    "worker/pr_env.py",
    "worker/pr_results.py",
)
IMAGE_BOUND = IMAGE_BUILT_FROM + IMAGE_RUNS
NEW_IMAGE = 6
LEAVE_KEPT = "--leave-kept"
RELEASE_FILE = ".rkmon_release.json"
RELEASES = "releases"
CURRENT = "current"
KEPT = "previous"
LOCK = ".lock"
STAGE = ".install-"
MOVED = "moved"
WHOLE = ("rk", "rkmon.py", "rkmon_lib/store.py")
DIR_MODE, FILE_MODE, RUN_MODE = 0o555, 0o444, 0o555
MOST = 10
PIN = "RKMON_CODE"
PROC = "/proc"
WORKER_VALUES = "tools/pr_user.json"
# The most of the values file `in_use` reads: it holds three short values.
VALUES_MOST = 1 << 16


def runs(mode: int) -> bool:
    """Whether a file of the mode MODE is one its package runs: see the module's docstring."""
    return bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def fingerprints(tree, entries=("",)) -> dict:
    """{path: the SHA-256 of its bytes and whether it runs} for each file of the code tree TREE: see the module's
    docstring."""
    tree, out = pathlib.Path(tree), {}
    # `runs`' rule, spelled out: this source runs alone in the image's probe (the module's docstring)
    run_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    for entry in entries:
        top = tree / entry
        for p in sorted(top.rglob("*")) if not entry or entry.endswith("/") else [top]:
            rel = p.relative_to(tree).as_posix()
            if p.is_file() and rel != RELEASE_FILE:
                h = hashlib.sha256(p.read_bytes())
                h.update(b"x" if p.stat().st_mode & run_bits else b"-")
                out[rel] = h.hexdigest()
    return out


def digest(tree, entries=("",)) -> str:
    """The digest of the code tree TREE: see the module's docstring."""
    found = fingerprints(tree, entries)
    lines = "".join(f"{k} {v}\n" for k, v in sorted(found.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def image_changes(a, b) -> list:
    """The IMAGE_BOUND files that differ between the code trees A and B: see the module's docstring."""
    old, new = fingerprints(a, IMAGE_BOUND), fingerprints(b, IMAGE_BOUND)
    return sorted(f for f in old.keys() | new.keys() if old.get(f) != new.get(f))


def missing(tree) -> list:
    """The files of WHOLE that the code tree TREE lacks or holds empty: see the module's docstring."""
    out = []
    for rel in WHOLE:
        f = pathlib.Path(tree) / rel
        try:
            if f.is_file() and f.stat().st_size > 0:
                continue
        except OSError:
            pass
        out.append(rel)
    return out


def running(folder, image_id: str) -> pathlib.Path | None:
    """The release that runs in the code volume's folder FOLDER: see the module's docstring."""
    current = pathlib.Path(folder) / CURRENT
    base = release_info(current).get("base_image")
    if not image_id or base != image_id or missing(current):
        return None
    return current.resolve()


def release_info(root) -> dict:
    """The record of the release in the folder ROOT: see the module's docstring."""
    try:
        info = json.loads((pathlib.Path(root) / RELEASE_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def version(tree) -> str:
    """The VERSION of the package in the code tree TREE: see the module's docstring."""
    try:
        lines = (
            (pathlib.Path(tree) / "VERSION").read_text(encoding="utf-8").splitlines()
        )
    except OSError:
        return ""
    return next(
        (x.strip() for x in reversed(lines) if x.strip() and not x.startswith("#")), ""
    )


def in_use(folder, values=None, proc=PROC) -> set:
    """The names of the releases in the code volume's folder FOLDER that are in use: see the module's docstring."""
    top = os.path.realpath(os.path.join(folder, RELEASES))
    names = set()

    def note(path) -> None:
        real = os.path.realpath(path) if os.path.isabs(path) else ""
        if real.startswith(top + os.sep):
            names.add(real[len(top) + 1 :].split(os.sep)[0])

    pin = PIN.encode() + b"="
    for pid in sorted(p for p in os.listdir(proc) if p.isdigit()):
        with contextlib.suppress(OSError):
            note(os.readlink(os.path.join(proc, pid, "cwd")))
        with contextlib.suppress(OSError):
            with open(os.path.join(proc, pid, "environ"), "rb") as f:
                for var in f.read().split(b"\0"):
                    if var.startswith(pin):
                        note(os.fsdecode(var[len(pin) :]))
    if values:
        # A regular file only, and only so much of it: no other kind of file holds the values.
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            with open(os.open(values, os.O_RDONLY | os.O_NONBLOCK), "rb") as f:
                if stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    note(os.path.dirname(json.loads(f.read(VALUES_MOST))["rk"]))
    return names


def main(argv: list) -> int:
    """`code_volume.py in-use FOLDER [VALUES]`: see the module's docstring."""
    if len(argv) not in (2, 3) or argv[0] != "in-use":
        print("usage: code_volume.py in-use FOLDER [VALUES]", file=sys.stderr)
        return 2
    print(json.dumps(sorted(in_use(*argv[1:]))))
    return 0


def listed(names, most: int = MOST) -> str:
    """NAMES joined by commas, at most MOST of them: see the module's docstring."""
    names = list(names)
    shown = ", ".join(names[:most])
    return shown + (f" and {len(names) - most} more" if len(names) > most else "")


def remove(path) -> bool:
    """A file or a whole folder removed, its read-only folders too: see the module's docstring."""
    p = pathlib.Path(path)
    if p.is_dir() and not p.is_symlink():
        if os.geteuid() != 0:
            for d, dirs, _files in os.walk(p):
                for x in [d, *(os.path.join(d, n) for n in dirs)]:
                    try:
                        if not os.path.islink(x):
                            os.chmod(x, os.lstat(x).st_mode | 0o700)
                    except OSError:
                        pass
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except OSError:
            pass
    return not os.path.lexists(p)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
