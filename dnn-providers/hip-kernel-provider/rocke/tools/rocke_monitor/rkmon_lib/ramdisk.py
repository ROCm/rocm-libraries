# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The RAM disk (config `ramdisk`): a tmpfs for each review's short-lived files, so the worktrees, the builds and the
temporary files of the work in flight live in memory instead of on the disk.

On it, when `ramdisk.enabled`:
- each PR's worktrees `wt<N>` and `wt<N>m` (worker/mkwt.py makes them at `<ramdisk>/wt<N>`; `<WORKING_DIR>/wt<N>` is a
  link to each, so every path that names `<WORKING_DIR>/wt<N>` works as before);
- on a host, their builds `build/wt<N>` and `build/wt<N>m` (worker/build_native.py builds in
  `<ramdisk>/build/wt<N>m`; `<build>/wt<N>m` is a link to it);
- `paths.tmp`, unless the configuration names another folder for it: `<ramdisk>/tmp`, which holds the review directories
  (`v<N><round>`, the verification's inputs and its agents' scratch) and the TMPDIR of the checks and the review tools;
- in the container, with the RAM disk at its default folder (container_layout.RAMDISK), the PR runners' slots
  (container_layout.SLOTS_DIR, rkmon_lib/slots.py): each job's results, builds, temporary files and home, and the build
  a slot keeps for its review.
Never on it: the monitor's clone, `state/`, the logs, the saved probes, the posts, the caches (`paths.cache`), the
workers' folders and the shared trees (`wtdev`, `wtbase`, `wtsanity`, `wtbatch`, `wtbisect`): what must survive a
restart stays on the disk.

The RAM disk is the deployment's to mount (in the container, docker/compose.local.example.yaml shows the compose
override; on a host, a tmpfs mount of the operator's). Nothing here mounts anything; when it is not mounted, nothing
that would go on it starts (`room`), so nothing spills to the disk.

Footprints are learned (rkmon_lib/scaler.py): each review's and each heavy job's RAM disk use is sampled while it runs
(with the slots its PR holds: `slot_folders`), and its peak raises a running high estimate (SEED_GB before the first).
The scaler starts a pipe or a heavy job only when the RAM disk has room for its estimate and the memory headroom holds
with the RAM disk counted as used.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

from . import container_layout as L
from . import deployment, paths, slots

# The estimates before the first measured review and heavy job: the package's (deployment.PIPE_GB, HEAVY_GB).
SEED_GB = {"pipe": deployment.PIPE_GB, "heavy": deployment.HEAVY_GB}
FULL_SIZE_SHARE = 0.95  # a mount this share of ramdisk.size_gb or more counts as that size (tmpfs rounds to pages)
PR_TREE = re.compile(r"wt(\d+)m?")  # a PR's worktree or build folder name
REVIEW_DIR_QUIET_H = 2  # a review directory of a PR no longer in use goes once nothing in it changed for this long
GB = 2**30


def settings(cfg: dict) -> dict:
    """`ramdisk` in config: {"enabled", "path", "size_gb"} (path blank, size 0 when not set)."""
    r = cfg.get("ramdisk") or {}
    try:
        size = max(0.0, float(r.get("size_gb") or 0))
    except (TypeError, ValueError):
        size = 0.0
    return {
        "enabled": r.get("enabled") is True,
        "path": paths._clean(r.get("path")),
        "size_gb": size,
    }


def path(cfg: dict) -> str:
    """The RAM disk's folder while it is on, else ""."""
    s = settings(cfg)
    return s["path"] if s["enabled"] and s["path"] else ""


def problems(cfg: dict) -> list[str]:
    """What keeps the RAM disk settings from running (config.problems): a size that is no number of GiB, 0 or more,
    named as the deploy settings file names it. The switch is the size's own (on above 0: deploy_tree's join) and the
    folder the container's (derived.ramdisk_path), so neither has a row."""
    size = (cfg.get("ramdisk") or {}).get("size_gb")
    if size in (None, "") or (
        not isinstance(size, bool) and isinstance(size, (int, float)) and size >= 0
    ):
        return []
    return [
        f"{deployment.N('ramdisk.size_gb')} is {size!r}; it must be a number of GiB, 0 or more"
    ]


def _mount(folder: str, mounts: str = "/proc/mounts") -> tuple[str, str]:
    """(the mount point that holds `folder`, its file system type); ("", "") when unknown."""
    best, fs = "", ""
    try:
        lines = pathlib.Path(mounts).read_text(encoding="utf-8").splitlines()
    except OSError:
        return "", ""
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        point = parts[1].replace("\\040", " ")
        if (folder == point or folder.startswith(point.rstrip("/") + "/")) and len(
            point
        ) >= len(best):
            best, fs = point, parts[2]
    return best, fs


def usage(cfg: dict, mounts: str = "/proc/mounts", statvfs=os.statvfs) -> dict:
    """The RAM disk as it is now: {"enabled", "path", "size_gb" (the setting), "mounted" (a tmpfs is mounted at the
    path), "fs", "total_gb", "used_gb", "free_gb"}. Read-only; the numbers are None when it is not mounted.
    """
    s = settings(cfg)
    out = {
        "enabled": s["enabled"],
        "path": s["path"],
        "size_gb": s["size_gb"],
        "mounted": False,
        "fs": "",
        "total_gb": None,
        "used_gb": None,
        "free_gb": None,
    }
    if not s["enabled"] or not s["path"]:
        return out
    point, fs = _mount(s["path"], mounts)
    out["fs"] = fs
    out["mounted"] = fs == "tmpfs" and point == s["path"]
    if not out["mounted"]:
        return out
    try:
        v = statvfs(s["path"])
    except OSError:
        out["mounted"] = False
        return out
    total, free = v.f_blocks * v.f_frsize, v.f_bavail * v.f_frsize
    out.update(
        total_gb=round(total / GB, 2),
        used_gb=round((total - free) / GB, 2),
        free_gb=round(free / GB, 2),
    )
    return out


def gb_of(folders: list[pathlib.Path], run=subprocess.run) -> float:
    """The GB the folders hold (du; a missing folder holds 0)."""
    there = [str(f) for f in folders if f.exists()]
    if not there:
        return 0.0
    try:
        p = run(["du", "-sk", "--", *there], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 0.0
    kb = sum(
        int(ln.split()[0])
        for ln in p.stdout.splitlines()
        if ln.split() and ln.split()[0].isdigit()
    )
    return round(kb / 2**20, 3)


def tmp_dir(cfg: dict) -> pathlib.Path | None:
    """paths.tmp when it is on the RAM disk, else None."""
    root, tmp = path(cfg), paths.resolve(cfg)["tmp"]
    return (
        pathlib.Path(tmp)
        if root and tmp and (tmp == root or tmp.startswith(root + "/"))
        else None
    )


def pipe_folders(cfg: dict, n: int) -> list[pathlib.Path]:
    """What a review of PR n keeps on the RAM disk: its worktrees, the review tools' build of its head, its review
    directories (prep.review_dir_pr)."""
    from . import prep  # a call-time import: prep loads most of the package

    root = pathlib.Path(path(cfg))
    out = [root / f"wt{n}", root / f"wt{n}m", root / "build" / f"wt{n}"]
    tmp = tmp_dir(cfg)
    if tmp and tmp.is_dir():
        out += sorted(
            p for p in tmp.iterdir() if p.is_dir() and prep.review_dir_pr(p.name) == n
        )
    return out


def heavy_folders(cfg: dict, n: int) -> list[pathlib.Path]:
    """What a review's heavy job (its checks) keeps on the RAM disk: the build of its merged tree."""
    return [pathlib.Path(path(cfg)) / "build" / f"wt{n}m"]


def slot_folders(
    cfg: dict, records=None, environ=None
) -> dict[int, list[pathlib.Path]]:
    """PR n -> the folders of the slots that hold its work (slots.by_pr), while the slots are on the RAM disk: in the
    container (a launcher), with the RAM disk on at the folder that holds them. {} otherwise.
    """
    root = path(cfg)
    if (
        not root
        or not slots.launcher(environ)
        or not L.SLOTS_DIR.startswith(root.rstrip("/") + "/")
    ):
        return {}
    return {
        n: [pathlib.Path(slots.folder(r)) for r in runners]
        for n, runners in slots.by_pr(records).items()
    }


def on_ramdisk(cfg: dict) -> dict[int, list[pathlib.Path]]:
    """PR n -> its worktrees and builds on the RAM disk."""
    root = path(cfg)
    out: dict[int, list[pathlib.Path]] = {}
    if not root:
        return out
    for base in (pathlib.Path(root), pathlib.Path(root) / "build"):
        if base.is_dir():
            for p in base.iterdir():
                m = PR_TREE.fullmatch(p.name)
                if m and p.is_dir() and not p.is_symlink():
                    out.setdefault(int(m.group(1)), []).append(p)
    return out


def _rmtree(folder: pathlib.Path) -> str:
    """Remove a folder whole; its read-only folders that are this user's are made writable first (a test may leave
    one). "" once it is gone, else the first error (what is left stays)."""
    for d, dirs, _files in os.walk(folder):
        for x in [d, *(os.path.join(d, n) for n in dirs)]:
            try:
                if not os.path.islink(x):
                    os.chmod(x, os.lstat(x).st_mode | 0o700)
            except OSError:
                pass  # another user's: its removal below says so
    errors: list[str] = []

    def note(_func, name, exc):
        exc = exc[1] if isinstance(exc, tuple) else exc
        if not errors:
            errors.append(f"{name}: {getattr(exc, 'strerror', None) or exc}"[:200])

    if sys.version_info >= (3, 12):
        shutil.rmtree(folder, onexc=note)
    else:
        shutil.rmtree(folder, onerror=note)
    if not os.path.lexists(folder):
        return ""
    return errors[0] if errors else f"{folder}: still there"


def remove_tree(
    p: pathlib.Path, repo: str, run=subprocess.run, environ=None, pr_run=None
) -> str:
    """Remove a PR's worktree or build folder, wherever it is: a link to the RAM disk (both go), a worktree
    (unregistered from the clone first) or a plain folder. What PR code wrote there as a PR runner and this user cannot
    remove (an install step's folders are 755) is removed in a PR runner's slot (`pr_run`: prcode.run). "" once all of
    it is gone, else why not.
    """
    real = p.resolve() if p.is_symlink() else p
    if (real / ".git").exists() and repo:
        run(
            ["git", "-C", repo, "worktree", "remove", "--force", str(real)],
            capture_output=True,
        )
    why = ""
    if real.exists() and not real.is_symlink():
        why = _rmtree(real)
        if why:
            from . import prcode

            if slots.launcher(environ):
                (pr_run or prcode.run)(
                    ["rm", "-rf", "--", str(real)],
                    cwd=str(real.parent),
                    timeout=300,
                    environ=environ,
                )
                why = _rmtree(real) if os.path.lexists(real) else ""
    if p.is_symlink() and not os.path.lexists(real):
        p.unlink(missing_ok=True)
    return why


STUCK = "ramdisk_stuck"  # state: PR -> what of its finished review could not leave the RAM disk (journaled once)


def sweep(
    cfg: dict,
    st: dict,
    store,
    used: set[int],
    run=subprocess.run,
    clock=time.time,
    environ=None,
) -> list[str]:
    """Prompt cleanup: a review that is over (its PR in no pipe, no draft awaiting verification, no gate, pool or
    shadow draft: `used`) leaves the RAM disk at once: its worktrees and builds, and the links to them. Its review
    directories go once nothing in them changed for REVIEW_DIR_QUIET_H hours. Each line says what is really gone: one
    journal line per PR once its folders are gone; folders that would not go are one journal line (state
    `ramdisk_stuck`, again only when what is left changes), and each sweep tries them again.
    """
    root = path(cfg)
    if not root:
        return []
    p = paths.resolve(cfg)
    wd, build, repo = (
        pathlib.Path(p["working_dir"]),
        pathlib.Path(p["build"]),
        p["repo"],
    )
    stuck = st.get(STUCK) if isinstance(st.get(STUCK), dict) else {}
    out = []
    there = on_ramdisk(cfg)
    for n, dirs in sorted(there.items()):
        if n in used:
            continue
        links = [wd / f"wt{n}", wd / f"wt{n}m", build / f"wt{n}", build / f"wt{n}m"]
        whys = []
        for link in links:
            if link.is_symlink():
                whys.append(remove_tree(link, repo, run, environ))
        for d in dirs:
            whys.append(remove_tree(d, repo, run, environ))
        run(["git", "-C", repo, "worktree", "prune"], capture_output=True)
        left = sorted(str(x) for x in dirs + links if os.path.lexists(x))
        if not left:
            stuck.pop(str(n), None)
            store.journal(
                "cleanup",
                f"#{n}: its review is over; its worktrees and builds left the RAM disk",
            )
            out.append(
                f"RAMDISK #{n}: worktrees and builds removed (the review is over)"
            )
            continue
        if stuck.get(str(n)) != left:
            stuck[str(n)] = left
            why = next((w for w in whys if w), "")
            store.journal(
                "cleanup",
                f"#{n}: its review is over, but {', '.join(left)[:200]} did not leave the RAM disk"
                + (f" ({why})" if why else "")
                + "; each sweep tries again",
            )
            out.append(
                f"RAMDISK #{n}: {len(left)} folder(s) of a finished review would not go; tried again each sweep"
            )
    for k in [
        k for k in stuck if not k.isdigit() or int(k) in used or int(k) not in there
    ]:
        stuck.pop(
            k
        )  # reviewed again (its folders are in use), or its folders are gone otherwise (a restart)
    if stuck:
        st[STUCK] = stuck
    else:
        st.pop(STUCK, None)
    tmp = tmp_dir(cfg)
    if tmp and tmp.is_dir():
        from . import prep  # a call-time import, as in pipe_folders

        cutoff = clock() - REVIEW_DIR_QUIET_H * 3600
        for d in sorted(tmp.iterdir()):
            n = prep.review_dir_pr(d.name)
            if n is None or n in used or not d.is_dir():
                continue
            if _newest(d) < cutoff and not _rmtree(d):
                out.append(
                    f"RAMDISK {d.name}: review directory removed (PR #{n} is in no review)"
                )
    return out


def _newest(d: pathlib.Path) -> float:
    """The latest modification time in a folder tree."""
    newest = d.stat().st_mtime
    for p in d.rglob("*"):
        try:
            newest = max(newest, p.lstat().st_mtime)
        except OSError:
            continue
    return newest
