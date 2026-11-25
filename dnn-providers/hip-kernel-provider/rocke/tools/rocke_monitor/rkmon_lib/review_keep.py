# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The review directories on the RAM disk, kept on the disk too: a restart empties the RAM disk, and with it the
verification a review already paid for.

With the RAM disk on, `paths.tmp` is on it, and each review directory (`v<N><round>`, prep.review_dir) holds the
verification's inputs and what the verifier and the code quality reviewer wrote there (notes.md, the post, quality.md).
Every advance (`sync`, from review.advance):
- copies the files of each review directory of a review in flight (a pipe, a draft awaiting verification, an
  interrupted review) that changed since the last copy to `<WORKING_DIR>/kept_reviews/<the same name>`: regular files
  up to MAX_FILE_MB, at most MAX_DEPTH folders deep and MAX_FILES a directory (the agents' large scratch stays out);
- puts a kept directory back when its review directory is gone from the RAM disk while its review is still in flight
  (REVIEW_RESTORED: when the verifier had written its notes and post, the verification is finished: the session posts
  it instead of verifying again);
- removes the copy of a review no longer in flight once nothing in it changed for `retention.kept_reviews_h`
  hours.
Without the RAM disk the review directories are on the disk already, and nothing is copied.
"""

import pathlib
import shutil
import time

from . import paths, ramdisk, registry

KEEP = "kept_reviews"  # in <WORKING_DIR>
MAX_FILE_MB = 5
MAX_DEPTH = 2
MAX_FILES = 400
VERIFIED = (
    "notes.md",
)  # the verifier wrote its notes: its verification is done, or nearly


def quiet_hours(cfg: dict) -> float:
    """`retention.kept_reviews_h`: the hours a finished review's copy stays once nothing in it changed (its default
    when blank, or not a number 0 or more)."""
    h = registry.number(cfg, "retention.kept_reviews_h", float)
    return h if h >= 0 else float(registry.default("retention.kept_reviews_h"))


def keep_dir(cfg: dict) -> pathlib.Path:
    return pathlib.Path(paths.get(cfg, "working_dir")) / KEEP


def _reviews(st: dict) -> list[dict]:
    """The work in flight, each {"pr", "head", ...}: the parallel reviews, the drafts awaiting verification, the reviews
    a stop interrupted, in that order."""
    from .review import interrupted, pipes

    return [*pipes(st), *(st.get("review_ready") or []), *interrupted(st)]


def in_flight(st: dict) -> set[int]:
    """The PRs whose review is in flight: a pipe, a draft awaiting verification, a review a stop interrupted."""
    return {r["pr"] for r in _reviews(st)}


def heads_in_flight(st: dict) -> set[tuple[str, str]]:
    """(PR, head) of each review in flight, the PR as text, as the PR runners' slots name it (rkmon_lib/slots.py)."""
    return {(str(r["pr"]), str(r.get("head") or "")) for r in _reviews(st)}


def head_in_flight(st: dict, n: int) -> str:
    """The head of PR n's review in flight: its pipe's, else its draft's, else the interrupted one's; "" when none
    is."""
    return next((str(r.get("head") or "") for r in _reviews(st) if r["pr"] == n), "")


def _files(d: pathlib.Path):
    """(relative path, the file) for the files to keep: regular, small, shallow; at most MAX_FILES."""
    n = 0
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if len(rel.parts) > MAX_DEPTH + 1 or p.is_symlink() or not p.is_file():
            continue
        try:
            if p.stat().st_size > MAX_FILE_MB * 2**20:
                continue
        except OSError:
            continue
        yield rel, p
        n += 1
        if n >= MAX_FILES:
            return


def copy_changed(src: pathlib.Path, dest: pathlib.Path) -> int:
    """Copy the files of `src` that `dest` lacks or holds another version of (size or time); how many."""
    copied = 0
    for rel, p in _files(src):
        q = dest / rel
        try:
            s = p.stat()
            if q.exists():
                t = q.stat()
                if t.st_size == s.st_size and int(t.st_mtime) == int(s.st_mtime):
                    continue
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            copied += 1
        except OSError:
            continue
    return copied


def _newest(d: pathlib.Path) -> float:
    newest = d.stat().st_mtime
    for p in d.rglob("*"):
        try:
            newest = max(newest, p.lstat().st_mtime)
        except OSError:
            continue
    return newest


def sync(cfg: dict, st: dict, store, clock=time.time) -> list[str]:
    """Keep, restore and prune (the module's docstring); the lines to print."""
    from . import prep  # a call-time import: prep loads most of the package

    try:
        tmp = ramdisk.tmp_dir(cfg)
        keep = keep_dir(cfg)
    except paths.PathsError:
        return []
    if tmp is None:
        return []
    live, out = in_flight(st), []
    if tmp.is_dir():
        for d in sorted(tmp.iterdir()):
            n = prep.review_dir_pr(d.name)
            if n in live and d.is_dir():
                copy_changed(d, keep / d.name)
    if not keep.is_dir():
        return out
    for k in sorted(keep.iterdir()):
        n = prep.review_dir_pr(k.name)
        if n is None or not k.is_dir():
            continue
        if n in live:
            if tmp.is_dir() and not (tmp / k.name).exists():
                shutil.copytree(k, tmp / k.name, symlinks=False)
                done = all((k / f).is_file() for f in VERIFIED) and any(
                    p.name.startswith(f"pr{n}_") and p.suffix == ".md"
                    for p in k.iterdir()
                )
                store.journal(
                    "review",
                    f"#{n}: review directory {k.name} restored from {keep} (the RAM disk was emptied)",
                )
                out.append(
                    f"REVIEW_RESTORED #{n}: its review directory {tmp / k.name} came back from the disk copy (the "
                    "RAM disk was emptied by a restart)"
                    + (
                        ": the verifier's notes and post are there, so its verification is done; check and post it "
                        f"(rk spec-post {n} --round {k.name[len(f'v{n}'):]}) instead of verifying again"
                        if done
                        else "; its verification goes on from what is in it"
                    )
                )
        elif _newest(k) < clock() - quiet_hours(cfg) * 3600:
            shutil.rmtree(k, ignore_errors=True)
    return out
