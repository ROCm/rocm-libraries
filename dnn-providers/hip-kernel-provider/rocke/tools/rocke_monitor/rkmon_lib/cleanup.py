# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cleanup: every tick, remove what reviews of merged or closed PRs left on disk.

Per PR N, the artifacts are the worktrees `wt<N>` / `wt<N>m`, the builds `build/wt<N>` / `build/wt<N>m`, and the worker
outputs `workers/pr<N>`. They are removed only when the PR is merged or closed (from the monitor's records, or a GitHub
read for PRs it does not track) and no job or pipeline uses it. The shared trees (`wtbase`, `wtsanity`, `wtbatch`,
`wtdev`) and the repository itself are never touched. Finished job directories and develop-sanity outputs older than a
week go too. Everything removed can be regenerated from git.

With the RAM disk on (rkmon_lib/ramdisk.py), a PR's worktrees and builds live there, behind links of the same names, and
leave it as soon as its review is over (`ramdisk.sweep`, at every review advance), whatever the PR's state; the sweep
here removes the links and what they name.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import time

from . import jobs, paths, ramdisk, retention

PER_TICK = 8
LOOKUPS_PER_TICK = 10


def in_use(st: dict) -> set[int]:
    """The PRs whose trees a review, a draft or a gate uses now: the pipes, the drafts awaiting verification and their
    checks re-runs, the pool's and the shadow drafter's drafts, the gates."""
    from .review import pipes

    used = {p["pr"] for p in pipes(st)}
    used |= {r["pr"] for r in st.get("review_ready", [])}
    used |= {int(k) for k in (st.get("rechecks") or {})}
    used |= {int(k) for k in (st.get("draft_pool") or {})} | {
        int(k) for k in (st.get("shadow_pool") or {})
    }
    qs = st.get("queue_state") or {}
    for key in ("gate", "armed"):
        if qs.get(key):
            used.add(qs[key]["pr"])
    if qs.get("batch_gate"):
        used |= set(qs["batch_gate"]["members"])
    return used


def _where(cfg: dict, root: pathlib.Path) -> dict:
    """The paths a sweep works in, with `root` as the working folder (the worktrees live there)."""
    return paths.resolve(
        dict(cfg, paths=dict(cfg.get("paths") or {}, working_dir=str(root)))
    )


def artifacts(
    root: pathlib.Path, cfg: dict | None = None
) -> dict[int, list[pathlib.Path]]:
    p = _where(cfg or {}, root)
    out: dict[int, list[pathlib.Path]] = {}
    for base, pat in (
        (root, r"wt(\d+)m?"),
        (pathlib.Path(p["build"]), r"wt(\d+)m?"),
        (pathlib.Path(p["workers"]), r"pr(\d+)"),
    ):
        if base.is_dir():
            for p in base.iterdir():
                m = re.fullmatch(pat, p.name)
                if m and (
                    p.is_dir() or p.is_symlink()
                ):  # a link to the RAM disk too, even one whose tree is gone
                    out.setdefault(int(m.group(1)), []).append(p)
    return out


def _state(cfg: dict, st: dict, n: int, budget: list[int]) -> str:
    rec = st["prs"].get(str(n))
    if rec:
        return rec.get("status") or "unknown"
    cache = st.setdefault("cleanup_cache", {})
    if str(n) in cache:
        return cache[str(n)]
    if budget[0] <= 0:
        return "unknown"
    budget[0] -= 1
    from . import gh

    try:
        p = gh.rest(
            f"repos/{cfg['repo']}/pulls/{n}", jq="{state, merged: (.merged_at != null)}"
        )
    except Exception:  # noqa: BLE001 - an unreadable PR is simply kept
        return "unknown"
    s = "merged" if p.get("merged") else p.get("state", "unknown")
    if s in ("merged", "closed"):
        cache[str(n)] = s
    return s


def _rel(p: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(p.relative_to(root))
    except (
        ValueError
    ):  # a workers or build folder configured outside the working folder
        return str(p)


def _mb(dirs: list[pathlib.Path]) -> int:
    p = subprocess.run(["du", "-sm", *map(str, dirs)], capture_output=True, text=True)
    return sum(
        int(line.split()[0])
        for line in p.stdout.splitlines()
        if line.split() and line.split()[0].isdigit()
    )


def _old(base: pathlib.Path, days: int, keep=lambda p: False) -> list[pathlib.Path]:
    if not base.is_dir():
        return []
    cutoff = time.time() - days * 86400
    return [
        p
        for p in base.iterdir()
        if p.is_dir() and p.stat().st_mtime < cutoff and not keep(p)
    ]


def sweep(
    cfg: dict,
    st: dict,
    store,
    root: pathlib.Path,
    limit: int | None = PER_TICK,
    dry: bool = False,
) -> list[str]:
    where = _where(cfg, root)
    repo = where["repo"]
    used, budget, lines, done = (
        in_use(st),
        [LOOKUPS_PER_TICK if limit else 10**6],
        [],
        0,
    )
    for n, dirs in sorted(artifacts(root, cfg).items()):
        if n in used:
            continue
        s = _state(cfg, st, n, budget)
        if s not in ("merged", "closed"):
            continue
        mb = _mb(dirs)
        if dry:
            lines.append(
                f"would remove #{n} ({s}): {', '.join(_rel(p, root) for p in dirs)} ({mb} MB)"
            )
            continue
        for p in dirs:
            if (
                p.is_symlink()
            ):  # a link to the RAM disk (rkmon_lib/ramdisk.py): the tree or build it names, then the link
                ramdisk.remove_tree(p, str(repo))
                continue
            if (
                p.parent == root
            ):  # a git worktree: unregister it, then make sure the directory is gone
                subprocess.run(
                    ["git", "-C", str(repo), "worktree", "remove", "--force", str(p)],
                    capture_output=True,
                )
            shutil.rmtree(p, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(repo), "update-ref", "-d", f"refs/remotes/origin/pr{n}"],
            capture_output=True,
        )
        store.journal("cleanup", f"#{n} ({s}): removed {len(dirs)} dirs, {mb} MB")
        lines.append(f"CLEANUP #{n} ({s}): removed {len(dirs)} dirs, {mb} MB")
        done += 1
        if limit and done >= limit:
            break
    if not dry:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "prune"], capture_output=True
        )
    days = int(retention.settings(cfg)["job_days"])
    known = jobs.referenced(st)
    old = _old(store.dir / "jobs", days, keep=lambda p: not job_done(p, known)) + _old(
        pathlib.Path(where["scratch"]) / "sanity", days
    )
    if old:
        mb = _mb(old)
        if dry:
            lines.append(
                f"would remove {len(old)} finished job / sanity dirs older than {days} days ({mb} MB)"
            )
        else:
            for p in old:
                shutil.rmtree(p, ignore_errors=True)
            lines.append(
                f"CLEANUP old runs: {len(old)} job / sanity dirs older than {days} days, {mb} MB"
            )
    return lines


def job_done(d: pathlib.Path, known: dict) -> bool:
    """A job folder that may go once old: its job ended (an rc), or nothing names it and its leader is gone (a job
    killed or lost never writes its rc)."""
    import json

    if (d / "rc").exists():
        return True
    if d.name in known:
        return False
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        return True  # no job to speak of: a folder a start never finished
    try:
        return not jobs._leader(meta)
    except (KeyError, TypeError, ValueError):
        return True
