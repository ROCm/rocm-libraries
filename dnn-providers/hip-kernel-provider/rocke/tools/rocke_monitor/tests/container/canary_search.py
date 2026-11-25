# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Canary values, and the search for them (the canary test: tests/container/canary.py, and the image test's canary
stage, stage_canary.py). A canary is a made-up private value, rkmon-canary-<kind>-<32 hex digits>, put where a real one
would go; a search that finds one says where, never the text around it.

- make(KIND): a fresh canary of KIND (1 to 16 lowercase letters).
- shown(TEXT): TEXT with each canary in it masked as <canary KIND>, so that a check can show output that holds one,
  and still tell that it did (MASK).
- Finder(NEEDLES, KINDS): what a buffer or a stream holds: each canary, by its shape whatever its digits (of KINDS
  only, when given), and each of NEEDLES, a map of words to the bytes they name (such as the builder's home folder),
  found as they are and named by their words only.
- sweep(ROOTS, FINDER): every file and folder name under ROOTS, the target of every link, and the content of every
  regular file the caller can read (with CHANGED_SINCE, only those changed since then: a canary planted after it can
  be in no older file); a root that is a file, that file. With / among the roots, also the folders of shared memory
  and message queues; with / or /proc, each process's command line and environment (/proc/<pid>/cmdline, environ).
  The kernel's trees (/proc, /sys, /dev) are not walked. What it cannot list or read, it counts.

As a command, in a container (the canary stage runs it as each user it proves; standard library only, so it runs as a
PR runner, on the system's Python):

  python3 -I canary_search.py sweep [--root PATH]... [--kinds KIND,...] [--changed-since SECONDS]

  --root           a folder or a file to search (default /), any number of times
  --kinds          only canaries of these kinds (default: every kind)
  --changed-since  read only the files changed since this time (seconds since the epoch); every name is still checked
It prints a FOUND line for each find, a SWEPT line of counts, then `CANARIES: none` (exit 0) or `CANARIES: found in N
place(s)` (exit 1).
"""

import argparse
import functools
import os
import re
import secrets
import stat
import sys
import time

PREFIX = "rkmon-canary-"
KIND = re.compile(r"[a-z]{1,16}")
HEX_DIGITS = 32
PATTERN = re.compile(
    re.escape(PREFIX).encode() + rb"([a-z]{1,16})-[0-9a-f]{%d}" % HEX_DIGITS
)
LONGEST_BYTES = len(PREFIX) + 16 + 1 + HEX_DIGITS  # the longest canary
MASK = "<canary "  # how a masked canary begins (shown)
CHUNK_BYTES = 1 << 20  # a stream is searched this much at a time
KERNEL_TREES = ("/proc", "/sys", "/dev")  # never walked
USER_DEV = (
    "/dev/shm",
    "/dev/mqueue",
)  # the kernel trees' folders a user's files go in: walked with /
PROCESS_FILES = ("cmdline", "environ")


def make(kind: str) -> str:
    """A fresh canary of KIND (1 to 16 lowercase letters): rkmon-canary-KIND-<32 random hex digits>."""
    if not KIND.fullmatch(kind):
        raise ValueError(f"a canary's kind is 1 to 16 lowercase letters, not {kind!r}")
    return f"{PREFIX}{kind}-{secrets.token_hex(HEX_DIGITS // 2)}"


def shown(text) -> str:
    """TEXT (str or bytes) with each canary in it masked as <canary KIND>."""
    data = (
        text
        if isinstance(text, bytes)
        else str(text).encode("utf-8", "surrogateescape")
    )
    return PATTERN.sub(rb"<canary \1>", data).decode("utf-8", "replace")


class Finder:
    """What a buffer or a stream holds (the module's docstring): a set of words, `canary KIND` for each kind of canary
    found, and the words of each needle found."""

    def __init__(self, needles: dict | None = None, kinds=()):
        self.needles = {words: n for words, n in (needles or {}).items() if n}
        self.kinds = frozenset(kinds)
        longest = max([LONGEST_BYTES, *(len(n) for n in self.needles.values())])
        self.overlap = (
            longest - 1
        )  # a find that spans two chunks is whole in this much of the first and the second
        self.prefix = PREFIX.encode()

    def in_bytes(self, data: bytes) -> set:
        found = {words for words, n in self.needles.items() if n in data}
        if self.prefix in data:
            for m in PATTERN.finditer(data):
                kind = m.group(1).decode()
                if not self.kinds or kind in self.kinds:
                    found.add(f"canary {kind}")
        return found

    def in_stream(self, read) -> tuple[set, int]:
        """What READ's chunks hold (READ() gives b"" at the end), and how many bytes they were."""
        found, tail, total = set(), b"", 0
        while True:
            chunk = read()
            if not chunk:
                return found, total
            total += len(chunk)
            window = tail + chunk
            found |= self.in_bytes(window)
            tail = window[-self.overlap :]

    def hides(self, data: bytes) -> bool:
        """Whether DATA holds a needle: text that is never shown, even masked."""
        return any(n in data for n in self.needles.values())


def _under(path: str, trees) -> bool:
    return any(path == t or path.startswith(t + "/") for t in trees)


def sweep(roots, finder: Finder, changed_since=None, out=print) -> int:
    """Search ROOTS with FINDER (the module's docstring): print each find, the counts and the verdict; return the
    number of places that held a find."""
    start = time.monotonic()
    n = dict.fromkeys(
        (
            "files",
            "bytes",
            "older",
            "unreadable",
            "unlisted",
            "processes",
            "environments",
        ),
        0,
    )
    places = 0

    def where(path) -> str:
        raw = os.fsencode(path)
        return "(a path that holds a find)" if finder.hides(raw) else shown(raw)

    def found(what: set, place: str) -> None:
        nonlocal places
        for words in sorted(what):
            out(f"FOUND {words} in {place}")
        places += bool(what)

    def read(path: str):
        """The finds in the file at PATH, or None when it cannot be read."""
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(path, flags)
        except OSError:
            return None
        try:
            what, size = finder.in_stream(lambda: os.read(fd, CHUNK_BYTES))
        except OSError:
            return None
        finally:
            os.close(fd)
        n["bytes"] += size
        return what

    def visit(path: str) -> None:
        """One entry: its name, a link's target, a regular file's content."""
        what = finder.in_bytes(os.fsencode(os.path.basename(path)))
        if what:
            found(what, f"{where(path)} (its name)")
        try:
            st = os.lstat(path)
        except OSError:
            n["unreadable"] += 1
            return
        if stat.S_ISLNK(st.st_mode):
            try:
                what = finder.in_bytes(os.fsencode(os.readlink(path)))
            except OSError:
                n["unreadable"] += 1
                return
            if what:
                found(what, f"{where(path)} (where the link points)")
            return
        if not stat.S_ISREG(st.st_mode):
            return
        if changed_since is not None and max(st.st_mtime, st.st_ctime) < changed_since:
            n["older"] += 1
            return
        what = read(path)
        if what is None:
            n["unreadable"] += 1
            return
        n["files"] += 1
        if what:
            found(what, where(path))

    def unlisted(_exc) -> None:
        n["unlisted"] += 1

    walked = [r for r in roots if not _under(r, KERNEL_TREES) or _under(r, USER_DEV)]
    if "/" in roots:
        walked += [d for d in USER_DEV if os.path.isdir(d)]
    for root in dict.fromkeys(walked):
        try:
            is_folder = stat.S_ISDIR(os.lstat(root).st_mode)
        except OSError:
            is_folder = False
        if not is_folder:
            visit(root)
            continue
        for folder, folders, files in os.walk(root, onerror=unlisted):
            folders[:] = [
                f for f in folders if os.path.join(folder, f) not in KERNEL_TREES
            ]
            for name in folders + files:
                visit(os.path.join(folder, name))
    if "/" in roots or "/proc" in roots:
        try:
            pids = sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int)
        except OSError:
            pids = []
        for pid in pids:
            n["processes"] += 1
            try:
                with open(f"/proc/{pid}/comm", "rb") as f:
                    comm = shown(f.read().strip()) or "?"
            except OSError:
                comm = "?"
            for part in PROCESS_FILES:
                path = f"/proc/{pid}/{part}"
                what = read(path)
                if what is None:
                    n["unreadable"] += part == "environ"
                    continue
                n["environments"] += part == "environ"
                if what:
                    found(what, f"{path} ({comm})")
    counts = " ".join(f"{k}={v}" for k, v in n.items())
    out(f"SWEPT {counts} seconds={time.monotonic() - start:.0f}")
    out(f"CANARIES: found in {places} place(s)" if places else "CANARIES: none")
    return places


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sweep", help="search folders, files and processes")
    s.add_argument("--root", action="append", dest="roots", metavar="PATH")
    s.add_argument("--kinds", default="", metavar="KIND,...")
    s.add_argument("--changed-since", type=float, metavar="SECONDS")
    a = ap.parse_args(argv)
    kinds = [k for k in a.kinds.split(",") if k]
    bad = [k for k in kinds if not KIND.fullmatch(k)]
    if bad:
        ap.error(f"--kinds: {bad[0]!r} is no kind (1 to 16 lowercase letters)")
    roots = [os.path.normpath(r) if r != "/" else r for r in (a.roots or ["/"])]
    out = functools.partial(print, flush=True)
    return 1 if sweep(roots, Finder(kinds=kinds), a.changed_since, out) else 0


if __name__ == "__main__":
    sys.exit(main())
