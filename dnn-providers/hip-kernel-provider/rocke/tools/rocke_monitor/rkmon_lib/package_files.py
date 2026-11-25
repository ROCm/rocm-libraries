# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a package includes: the files the package build takes from a folder of the monitor's code, the copy it makes
of them, and a package's tarball unpacked. The build (tools/make_package.py), `rkmon package update`
(rkmon_lib/package_update.py), the docs' lints, the drift check, the install of the worker tools (rkmon_lib/deploy.py)
and the tests' copy of the package read this one rule.

- `tracked(root, every_file=True, env=None) -> [path]`: the files of the folder ROOT that a package includes, sorted.
  Where git tracks files of ROOT (a clone of the package's repository): those, and the untracked ones it does not
  ignore, never one it ignores (a settings file, a secret or a root certificate left in the folder: .gitignore).
  Where it tracks none (outside a repository, or in one that tracks none of ROOT: the image's copy, an unpacked
  tarball, a package folder in use, a build's folder in an ignored dist/): with EVERY_FILE, as the readers of a
  built folder take them (the docs lints, the drift check, the tests' copy of the package in tests/sandbox.py, a
  package update), every file the folder's own ignore rules keep, read through a git folder of the call's own, with
  no global ignore file; else none, and the build refuses. Never a file of a Python environment: a folder below
  ROOT, of any name, that has a VENV_MARK git lists, whatever Python made it (one before 3.13 gives it no
  .gitignore of its own). Never a path `excluded` names. A link to a file is one of them, a link to a folder is not.
  git runs with ENV (None: this process's environment). OSError when git cannot run; subprocess.CalledProcessError
  when it fails in a folder of the call's own.
- `excluded(rel)`: whether REL, a path relative to a package's top, is one no package includes, wherever it lies: one
  in a folder SKIP_PARTS names, or a file SKIP_FILES matches (a compiled file, a backup a write keeps).
- `copied(root, files=None) -> {rel: path}`: the files `copy` copies, by their paths in ROOT: each of FILES
  (`tracked(root)` when None) but CONFIG, which it writes blank.
- `copy(root, dest, files=None)`: the package a build makes of the folder ROOT, in the new folder DEST: each of
  `copied(root, files)` at its place, a link's file copied, each its owner's to write (a read-only source's too, as
  the image's is), and CONFIG the blank one (BLANK_CONFIG, 644), whatever ROOT holds there.
- `unpack(fileobj, dest)`: a tarball, read from the file FILEOBJ (one it may seek in), unpacked into the folder DEST
  member by member. A member whose path is absolute or holds `..`, a device, or a member or link that would land
  outside DEST (through a link unpacked before it too) is refused (Unsafe), before it is unpacked; tarfile's data
  filter refuses the same besides where this Python has it. What came before it stays in DEST. FILEOBJ is read to
  its end, where a compressed tarball's own check runs. UNWHOLE: what it raises when FILEOBJ is not a tarball, or
  not a whole one (cut short, or damaged).
- `package_in(folder) -> path | None`: the package in the folder FOLDER, resolved: FOLDER itself, or its one folder
  when that is a package (paths.is_package), as the package build's tarball unpacked holds it; None when there is
  none.
- `package_folders(names) -> [folder]`: the folders of a tree with a package in them, found by the paths of its
  files NAMES (relative to its top, with forward slashes, as git lists them): each with every file of a whole code
  tree (code_volume.WHOLE) in it, sorted, its top as "". By it the update finds the package in a fetched commit, and
  the build the package in a sandbox checkout: neither names a folder of the repository the package sits in.
- `git_env() -> dict`: this process's environment for git, but for the variables that point git at a repository
  (git's own list), and with no password asked on the terminal: the ENV to give `tracked` and a fetch of a package,
  so that git reads only the folder it is given.
- `within(path, top)`: whether PATH, every link in it followed, lies in the folder TOP (resolved).
- SKIP_PARTS: the folders no package includes, wherever they lie: git's data, the folders a developer's copy makes
  at its top (paths.COPY_OWN_FOLDERS: a monitor's state, a build's output, the tests' sandbox) and the caches it leaves
  wherever it runs (the folders among paths.COPY_OWN_LEFTOVERS). SKIP_FILES: the files no package includes, as
  .gitignore takes them (the files among paths.COPY_OWN_LEFTOVERS). VENV_MARK: the file at the top of every Python
  environment, whose folder no package includes either. CONFIG and BLANK_CONFIG: a package's config.json, and its
  text.
- `Unsafe(member, why)`: a member of a tarball that is not unpacked, and why (a clause: "is a device").

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import fnmatch
import gzip
import os
import pathlib
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zlib

from . import code_volume, paths, private_files

SKIP_PARTS = frozenset(
    (
        ".git",
        *paths.COPY_OWN_FOLDERS,
        *(p.rstrip("/") for p in paths.COPY_OWN_LEFTOVERS if p.endswith("/")),
    )
)
SKIP_FILES = tuple(p for p in paths.COPY_OWN_LEFTOVERS if not p.endswith("/"))
VENV_MARK = "pyvenv.cfg"  # at the top of every Python environment, whatever its name or the Python that made it
CONFIG = "config.json"
BLANK_CONFIG = (
    '{\n  "_comment": "Blank on purpose: a deployment\'s configuration lives outside the package, in its deploy '
    "settings file, which step 1 of QUICK_DEPLOY_GUIDE.md makes. README.md's Settings reference describes each of its "
    'keys, and docs/internal_settings.md every other key."\n}\n'
)
_LS = ("ls-files", "-z")
# tarfile's data filter, where this Python has it (3.12, and the releases it was brought back to), and the error it
# raises for a member it refuses; unpack's own checks stand alone where it has none.
_FILTER = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
_FILTER_ERROR = getattr(tarfile, "FilterError", ())
# What unpack raises for a file that is not a tarball, or not a whole one: tarfile's errors, and those of the
# decompression a tarball cut short or damaged meets.
UNWHOLE = (tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error)


class Unsafe(Exception):
    """A member of a tarball that is not unpacked: see the module's docstring."""

    def __init__(self, member: str, why: str):
        super().__init__(f"{member} {why}")
        self.member, self.why = member, why


def _names(p: subprocess.CompletedProcess) -> set:
    """The paths git listed, NUL-separated, in P's output."""
    return set(p.stdout.split("\0")) - {""}


def excluded(rel) -> bool:
    """Whether REL, a path relative to a package's top, is one no package includes: see the module's docstring."""
    parts = pathlib.PurePath(rel).parts
    if not parts:
        return False
    return bool(set(parts) & SKIP_PARTS) or any(
        fnmatch.fnmatchcase(parts[-1], pattern) for pattern in SKIP_FILES
    )


def tracked(root, every_file: bool = True, env=None) -> list[pathlib.Path]:
    """The files of ROOT that a package includes: see the module's docstring."""
    root = pathlib.Path(root)

    def listed(*args) -> set:
        git = ["git", "-C", str(root), *_LS, *args]
        p = subprocess.run(git, capture_output=True, text=True, env=env)
        return _names(p) if p.returncode == 0 else set()

    names = listed("-c")
    if names:
        names |= listed("-o", "--exclude-standard")
    elif every_file:
        with tempfile.TemporaryDirectory() as own:
            init = ["git", "init", "-q", "--bare", own]
            subprocess.run(init, capture_output=True, text=True, check=True, env=env)
            # The folder's own ignore rules alone: no global ignore file (core.excludesFile)
            here = ["-c", "core.excludesFile=", "-C", str(root), f"--git-dir={own}"]
            git = [
                "git",
                *here,
                f"--work-tree={root}",
                *_LS,
                "-co",
                "--exclude-standard",
            ]
            p = subprocess.run(git, capture_output=True, text=True, check=True, env=env)
        names = _names(p)
    # The folders of the Python environments, by the mark git lists at their top: "env/" for env/pyvenv.cfg
    venvs = tuple(n[: -len(VENV_MARK)] for n in names if n.endswith("/" + VENV_MARK))
    out = []
    for n in sorted(names):
        f = root / n
        if f.is_file() and not n.startswith(venvs) and not excluded(n):
            out.append(f)
    return out


def copied(root, files=None) -> dict:
    """The files `copy` copies, by their paths in ROOT: see the module's docstring."""
    root = pathlib.Path(root)
    out = {}
    for f in tracked(root) if files is None else files:
        rel = f.relative_to(root).as_posix()
        if rel != CONFIG:
            out[rel] = f
    return out


def copy(root, dest, files=None) -> None:
    """The package a build makes of ROOT, in the new folder DEST: see the module's docstring."""
    root, dest = pathlib.Path(root), pathlib.Path(dest)
    dest.mkdir(parents=True)
    for rel, f in copied(root, files).items():
        to = dest / rel
        to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, to)
        to.chmod(to.stat().st_mode | stat.S_IWUSR)
    config = dest / CONFIG
    config.write_text(BLANK_CONFIG, encoding="utf-8")
    config.chmod(0o644)


def package_in(folder) -> pathlib.Path | None:
    """The package in FOLDER, or None: see the module's docstring."""
    top = private_files.real(folder)
    if paths.is_package(top):
        return top
    try:
        inner = [p for p in top.iterdir() if p.is_dir()]
    except OSError:
        return None
    return inner[0] if len(inner) == 1 and paths.is_package(inner[0]) else None


def package_folders(names) -> list[str]:
    """The folders with a package in them, by the paths of a tree's files NAMES: see the module's docstring."""
    names = set(names)
    first, *rest = code_volume.WHOLE
    tops = [n[: -len(first)] for n in names if n == first or n.endswith("/" + first)]
    return sorted(t.rstrip("/") for t in tops if all(t + rel in names for rel in rest))


def git_env() -> dict:
    """This process's environment for git: see the module's docstring."""
    names = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.split()
    env = {k: v for k, v in os.environ.items() if k not in names}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def within(path, top: str) -> bool:
    """Whether PATH lies in the folder TOP: see the module's docstring."""
    real = os.path.realpath(path)
    return real == top or real.startswith(top + os.sep)


def _unsafe(m: tarfile.TarInfo, top: str) -> str:
    """Why the member M is not unpacked into the folder TOP (resolved), a clause, or "": see `unpack`."""
    path = pathlib.PurePosixPath(m.name)
    if path.is_absolute() or ".." in path.parts:
        return "has a path that leaves the folder"
    if m.isdev():
        return "is a device"
    if not within(os.path.join(top, m.name), top):
        return "would land outside the folder, through a link before it"
    if m.issym() or m.islnk():
        # A hard link names its target from the tarball's top, a symbolic link from its own folder
        base = top if m.islnk() else os.path.join(top, os.path.dirname(m.name))
        if not within(os.path.join(base, m.linkname), top):
            return f"is a link that leaves the folder, to {m.linkname}"
    return ""


def unpack(fileobj, dest) -> None:
    """A tarball, from FILEOBJ, unpacked into DEST: see the module's docstring."""
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    top = os.path.realpath(dest)
    with tarfile.open(fileobj=fileobj, mode="r:*") as tf:
        for m in tf:
            why = _unsafe(m, top)
            if why:
                raise Unsafe(m.name, why)
            try:
                tf.extract(m, top, **_FILTER)
            except _FILTER_ERROR as exc:
                why = f"is refused by tarfile's data filter: {exc}"
                raise Unsafe(m.name, why) from None
        # The rest, to its end: there a compressed tarball's own check (gzip's CRC) finds one damaged
        while tf.fileobj.read(1 << 16):
            pass
