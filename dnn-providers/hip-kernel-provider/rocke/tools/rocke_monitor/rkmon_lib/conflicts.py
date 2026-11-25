# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Merge-conflict hot files: the files PRs conflict on most, for the weekly insights and the dashboard.

Every tick (`collect`) records into `state/conflicts.jsonl`, from two sources, since the module shipped:
- live: each open, non-draft PR into develop that GitHub reports CONFLICTING, merged with develop in the mirror as G4
  merges (`git merge-tree`): the paths that conflict. A row when a PR head first conflicts, when its paths change as
  develop moves, and every `refresh_h` hours while it keeps conflicting, so the PR shows in every window; a row per
  develop move would be about 20 a day per PR with nothing new in them;
- resolved: a conflict the author has already resolved never shows live. Each develop merge on a tracked PR's branch (a
  merge commit with a parent on develop and one off it, committed in the last `scan_days` days) is merged again from
  its parents: the paths its automatic merge conflicts on, the "remerge CONFLICT" paths of `git show --remerge-diff`
  (the same merge, without the diff text). One row per merge commit, clean ones included. A rebase leaves no merge
  commit, so its conflicts are seen only while they last.
Each tick is bounded (`conflicts` in config): merge checks, PR fetches, seconds, and a timeout per check; what is left
waits for the next tick. A failure is reported and never fails the tick.

`hot_files` ranks a window's files by the PRs that conflicted on them, then by episodes (a PR head conflicting on the
file, or a develop merge resolving it; a merge on a head the live source saw conflicting on that file is the same
episode), and rolls them up by directory. `offenders` names the develop PRs that last changed each file. A file in
the folder the monitor reviews (paths.repository_folder, which every caller passes in) shows from that folder's own
name on (`short`, `area`), or from below it (`within`).
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import time

from . import gitops, idle, paths, registry
from .store import minutes_since, now

FILE = "conflicts.jsonl"
DEVELOP = "refs/remotes/origin/develop"
UTC = dt.timezone.utc
DEFAULTS = registry.package_section("conflicts")
MERGED_PR = re.compile(r"\(#(\d+)\)\s*$")
GIT_ERRORS = (gitops.GitError, OSError, subprocess.SubprocessError)
DAYS = 7  # the dashboard's window


def settings(cfg: dict) -> dict:
    return dict(DEFAULTS, **(cfg.get("conflicts") or {}))


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _z(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def conflicting(rec: dict) -> bool:
    """An open, non-draft PR into develop that GitHub reports CONFLICTING (while it reads UNKNOWN, its last answer, as
    for the CONFLICT event). A stacked PR's conflicts are with its base branch, not develop.
    """
    return (
        rec.get("status") == "open"
        and not rec.get("draft")
        and rec.get("base", "develop") == "develop"
        and "CONFLICTING" in (rec.get("mergeable"), rec.get("mergeable_known"))
    )


def _recent(rec: dict, days: int) -> bool:
    return (
        rec.get("status") == "open"
        or bool(rec.get("gone_at"))
        and minutes_since(rec["gone_at"]) <= days * 1440
    )


def develop_merges(
    repo: str, head: str, develop: str, since: float
) -> list[tuple[str, str, str, str]]:
    """(merge, PR-side parent, develop-side parent, committed at) for the develop merges on a PR branch: merge commits
    reachable from `head` and not from develop, committed since `since` (epoch seconds), with one parent on develop and
    one off it (a merge with no parent on develop brings in another branch)."""
    stamp = dt.datetime.fromtimestamp(since, UTC).strftime("%Y-%m-%d %H:%M:%S +0000")
    out = []
    for line in gitops.git(
        repo,
        "log",
        "--merges",
        "--format=%H %ct %P",
        f"--since={stamp}",
        head,
        "--not",
        develop,
        timeout=60,
    ).splitlines():
        sha, ct, *parents = line.split()
        if int(ct) < since:
            continue
        on = [p for p in parents if gitops.is_ancestor(repo, p, develop)]
        off = [p for p in parents if p not in on]
        if on and off:
            out.append(
                (sha, off[0], on[0], _z(dt.datetime.fromtimestamp(int(ct), UTC)))
            )
    return out


def collect(cfg: dict, st: dict, store) -> dict:
    """One tick's recording (see the module docstring). Returns the counts; `errors` lists the PRs that failed."""
    c = settings(cfg)
    repo, dev = (cfg.get("paths") or {}).get("repo"), st["develop"].get("sha")
    out = {"live": 0, "resolved": 0, "checks": 0, "deferred": 0, "errors": []}
    if not repo or not dev:
        return out
    mem = st.setdefault("conflicts", {})
    if not mem.get("since"):  # where the history starts, for "tracked since"
        mem["since"] = now()
        store.append_jsonl(
            FILE,
            [
                {
                    "ts": mem["since"],
                    "source": "start",
                    "scan_from": _z(
                        _ts(mem["since"]) - dt.timedelta(days=c["scan_days"])
                    ),
                }
            ],
        )
    left = {"checks": c["max_checks"], "fetches": c["max_fetches"]}
    stop = time.monotonic() + c["budget_s"]

    def room() -> bool:
        return left["checks"] > 0 and time.monotonic() < stop

    def check(a: str, b: str) -> list[str] | None:
        left["checks"] -= 1
        out["checks"] += 1
        files = gitops.merge_conflicts(repo, a, b, timeout=c["timeout_s"])
        return None if files is None else sorted(set(files))

    # Live: the PRs GitHub reports conflicting, once per (PR, head, develop).
    live = mem.setdefault("live", {})
    want = {k: r for k, r in st["prs"].items() if conflicting(r) and r.get("head")}
    for k in [k for k in live if k not in want]:
        del live[
            k
        ]  # no longer conflicting: a new conflict at the same head is news again
    for k, rec in sorted(want.items(), key=lambda kv: int(kv[0])):
        head, prev = rec["head"], live.get(k) or {}
        if prev.get("head") == head[:12] and prev.get("develop") == dev[:12]:
            continue
        if not room():
            out["deferred"] += 1
            continue
        try:
            if not gitops.has_commit(repo, head):
                if left["fetches"] <= 0:
                    out["deferred"] += 1
                    continue
                left["fetches"] -= 1
                if gitops.fetch_pr(repo, rec["n"]) != head:
                    continue  # pushed since the poll: the next tick has the new head
            files = check(dev, head)
        except GIT_ERRORS as exc:
            out["errors"].append(f"#{k}: {str(exc)[:120]}")
            continue
        if files is None:
            continue
        same = prev.get("head") == head[:12]
        logged = prev.get("logged") if same else None
        if files and (
            not same
            or prev.get("files") != files
            or not logged
            or minutes_since(logged) >= c["refresh_h"] * 60
        ):
            logged = now()
            store.append_jsonl(
                FILE,
                [
                    {
                        "ts": logged,
                        "source": "live",
                        "pr": rec["n"],
                        "head": head[:12],
                        "develop": dev[:12],
                        "files": files,
                    }
                ],
            )
            out["live"] += 1
        live[k] = {
            "head": head[:12],
            "develop": dev[:12],
            "files": files,
            "logged": logged,
        }

    # Resolved: the develop merges on each tracked PR's branch, once per merge commit; a PR's head is scanned once.
    scanned, known = mem.setdefault("scanned", {}), None
    for k in [
        k
        for k in scanned
        if k not in st["prs"] or not _recent(st["prs"][k], c["scan_days"])
    ]:
        del scanned[k]
    since = time.time() - c["scan_days"] * 86400
    for k, rec in sorted(st["prs"].items(), key=lambda kv: -int(kv[0])):
        head = rec.get("head")
        if not head or scanned.get(k) == head[:12] or not _recent(rec, c["scan_days"]):
            continue
        if not room():
            out["deferred"] += 1
            continue
        done = True
        try:
            if not gitops.has_commit(
                repo, head
            ):  # the queue has fetched every open head by now
                scanned[k] = head[
                    :12
                ]  # not tried again every tick: its merges wait for its next head
                continue
            merges = develop_merges(repo, head, dev, since)
            if merges and known is None:
                known = {
                    r.get("merge")
                    for r in store.read_jsonl(FILE, lenient=True)
                    if r.get("source") == "resolved"
                }
            for sha, mine, theirs, at in merges:
                if sha[:12] in known:
                    continue
                if not room():
                    done = False
                    out["deferred"] += 1
                    break
                files = check(mine, theirs)
                if files is None:
                    continue
                known.add(sha[:12])
                store.append_jsonl(
                    FILE,
                    [
                        {
                            "ts": at,
                            "source": "resolved",
                            "pr": rec["n"],
                            "merge": sha[:12],
                            "parent": mine[:12],
                            "develop": theirs[:12],
                            "files": files,
                        }
                    ],
                )
                out["resolved"] += 1
        except GIT_ERRORS as exc:
            out["errors"].append(f"#{k}: {str(exc)[:120]}")
            continue
        if done:
            scanned[k] = head[:12]

    # The dashboard's view names the develop PRs that last changed each top file: one `git log` when either changes.
    try:
        end = dt.datetime.now(UTC)
        start = end - dt.timedelta(days=DAYS)
        top = [
            f["path"]
            for f in hot_files(
                store, start, end, rocke_path=paths.repository_folder(cfg)
            )["files"]
        ]
        o = mem.get("offenders") or {}
        if o.get("develop") != dev[:12] or o.get("files") != top:
            mem["offenders"] = {
                "develop": dev[:12],
                "files": top,
                "by": offenders(repo, top, start, end),
            }
    except GIT_ERRORS as exc:
        out["errors"].append(f"offenders: {str(exc)[:120]}")
    return out


def _name(rocke_path: str) -> str:
    """The name of the folder ROCKE_PATH, its last part: what a path in it shows from."""
    return rocke_path.rsplit("/", 1)[-1]


def within(path: str, rocke_path: str) -> str:
    """A path in the folder ROCKE_PATH from below it (platform/x.py for <provider>/rocke/platform/x.py); others in
    full."""
    return path[len(rocke_path) + 1 :] if path.startswith(rocke_path + "/") else path


def short(path: str, rocke_path: str) -> str:
    """A path in the folder ROCKE_PATH from that folder's name on (rocke/platform/x.py for
    <provider>/rocke/platform/x.py); others in full."""
    return (
        f"{_name(rocke_path)}/{within(path, rocke_path)}"
        if path.startswith(rocke_path + "/")
        else path
    )


def area(path: str, rocke_path: str, depth: int = 4) -> str:
    """The directory a file rolls up to: up to `depth` directories below the folder ROCKE_PATH, from its name on
    (rocke/platform/tests/golden), or below the repository root for a file outside it.
    """
    if path.startswith(rocke_path + "/"):
        return "/".join(
            [_name(rocke_path)] + path[len(rocke_path) + 1 :].split("/")[:-1][:depth]
        )
    return "/".join(path.split("/")[:-1][:depth]) or "(root)"


def hot_files(
    store, start: dt.datetime, end: dt.datetime, top: int = 10, *, rocke_path: str
) -> dict:
    """The files PRs conflicted on in [start, end), most PRs first, then most episodes, with the PRs; the same rolled up
    by directory; and the window's develop merges on PR branches (how many resolved a conflict).
    """
    allrows = store.read_jsonl(FILE, lenient=True)
    mark = next((r for r in allrows if r.get("source") == "start"), None)
    rows = [
        r
        for r in allrows
        if r.get("files") and r.get("ts") and start <= _ts(r["ts"]) < end
    ]
    seen = {
        (r["pr"], r["head"], f)
        for r in rows
        if r["source"] == "live"
        for f in r["files"]
    }
    eps: dict[str, set] = {}
    prs: dict[str, set] = {}
    for r in rows:
        for f in r["files"]:
            if r["source"] == "live":
                key = (r["pr"], r["head"])
            elif (r["pr"], r.get("parent"), f) in seen:
                key = (
                    r["pr"],
                    r["parent"],
                )  # the conflict the live source saw, resolved by this merge
            else:
                key = (r["pr"], r.get("merge"))
            eps.setdefault(f, set()).add(key)
            prs.setdefault(f, set()).add(r["pr"])
    order = sorted(eps, key=lambda f: (-len(prs[f]), -len(eps[f]), f))
    dirs: dict[str, dict] = {}
    for f in eps:
        d = dirs.setdefault(
            area(f, rocke_path), {"prs": set(), "files": set(), "eps": set()}
        )
        d["prs"] |= prs[f]
        d["files"].add(f)
        d["eps"] |= eps[f]
    merges = [
        r
        for r in allrows
        if r.get("source") == "resolved" and r.get("ts") and start <= _ts(r["ts"]) < end
    ]
    return {
        "files": [
            {
                "path": f,
                "short": short(f, rocke_path),
                "prs": sorted(prs[f]),
                "episodes": len(eps[f]),
            }
            for f in order[:top]
        ],
        "dirs": [
            {
                "dir": k,
                "prs": len(d["prs"]),
                "files": len(d["files"]),
                "episodes": len(d["eps"]),
            }
            for k, d in sorted(
                dirs.items(),
                key=lambda kv: (-len(kv[1]["prs"]), -len(kv[1]["eps"]), kv[0]),
            )
        ][:8],
        "n_files": len(eps),
        "prs": sorted(set().union(*prs.values())) if prs else [],
        "merges": len(merges),
        "merges_conflicted": sum(1 for r in merges if r.get("files")),
        "since": (mark or {}).get("ts") or (allrows[0].get("ts") if allrows else None),
        "scan_from": (mark or {}).get("scan_from"),
    }


def offenders(
    repo: str,
    files: list[str],
    start: dt.datetime,
    end: dt.datetime,
    ref: str = DEVELOP,
    limit: int = 3,
) -> dict[str, list[int]]:
    """For each file, the PRs merged into develop in [start, end) that last changed it, newest first (one `git log`)."""
    if not files:
        return {}
    fmt = "%Y-%m-%d %H:%M:%S +0000"
    log = gitops.git(
        repo,
        "-c",
        "core.quotepath=false",
        "log",
        "--first-parent",
        "--format=%x01%s",
        "--name-only",
        f"--since={start.astimezone(UTC).strftime(fmt)}",
        f"--until={end.astimezone(UTC).strftime(fmt)}",
        ref,
        "--",
        *files,
        timeout=60,
    )
    out: dict[str, list[int]] = {}
    pr = None
    for line in log.splitlines():
        if line.startswith("\x01"):
            m = MERGED_PR.search(line[1:])
            pr = int(m.group(1)) if m else None
        elif line.strip() and pr:
            got = out.setdefault(line.strip(), [])
            if pr not in got and len(got) < limit:
                got.append(pr)
    return out


def weekly(cfg: dict, st: dict, store, start: dt.datetime, end: dt.datetime) -> dict:
    """The weekly report's view: the week's hot files, and the develop PRs that last changed each. On the first run the
    live conflicts are recorded first, so the first report is not empty."""
    if not (store.dir / FILE).exists():
        try:
            collect(cfg, st, store)
        except GIT_ERRORS:
            pass
    hot = hot_files(store, start, end, rocke_path=paths.repository_folder(cfg))
    try:
        hot["offenders"] = offenders(
            (cfg.get("paths") or {})["repo"],
            [f["path"] for f in hot["files"]],
            start,
            end,
        )
    except (KeyError, *GIT_ERRORS):
        hot["offenders"] = {}
    return hot


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _prs(ns: list[int], limit: int = 8) -> str:
    return ", ".join(f"#{n}" for n in ns[:limit]) + (
        f" (+{len(ns) - limit})" if len(ns) > limit else ""
    )


def tracked(hot: dict) -> str:
    since = hot.get("since")
    if not since:
        return "not tracked yet"
    scan = hot.get("scan_from")
    return f"tracked since {since[:16].replace('T', ' ')}Z" + (
        f"; develop merges on PR branches since {scan[:10]}" if scan else ""
    )


def top_line(hot: dict) -> str:
    if not hot.get("files"):
        return f"— (none recorded; {tracked(hot)})"
    f, n = hot["files"][0], len(hot["files"][0]["prs"])
    return (
        f"`{_cell(f['short'])}` ({n} PR{'' if n == 1 else 's'}, {f['episodes']} "
        f"episode{'' if f['episodes'] == 1 else 's'})"
    )


def render_md(hot: dict, title: str = "## Merge-conflict hot files") -> list[str]:
    """The markdown section: the top files, their PRs and the develop PRs that last changed each; then directories."""
    by = hot.get("offenders") or {}
    lines = [
        title,
        "",
        f"Files PRs conflicted on with develop, most PRs first ({tracked(hot)}). Live: open PRs GitHub "
        "reports conflicting, merged with develop here. Resolved: develop merges in which the author resolved a "
        "conflict; a rebase is seen only while it conflicts. An episode is one PR head conflicting on the file, "
        "or one merge resolving it.",
        "",
    ]
    if not hot.get("files"):
        return lines + ["None recorded."]
    lines += [
        "| # | File | PRs | Episodes | Which PRs | Last changed on develop by |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {i} | `{_cell(f['short'])}` | {len(f['prs'])} | {f['episodes']} | {_prs(f['prs'])} | "
        f"{_prs(by.get(f['path']) or []) or '—'} |"
        for i, f in enumerate(hot["files"], 1)
    ]
    lines += ["", "| Directory | PRs | Files | Episodes |", "|---|---|---|---|"]
    lines += [
        f"| `{_cell(d['dir'])}/` | {d['prs']} | {d['files']} | {d['episodes']} |"
        for d in hot["dirs"]
    ]
    lines += [
        "",
        f"{hot['n_files']} files on {len(hot['prs'])} PRs in all. Develop merges on PR branches: "
        f"{hot['merges']}, {hot['merges_conflicted']} of them resolving a conflict.",
    ]
    return lines


@idle.timer("conflict history")
def _relog_due(cfg: dict, st: dict, store):
    """When each PR that still conflicts with develop is logged again (`refresh_h`)."""
    hours = settings(cfg)["refresh_h"]
    for k, e in ((st.get("conflicts") or {}).get("live") or {}).items():
        if e.get("logged"):
            yield f"conflict log #{k}", idle.plus(e["logged"], hours * 60)
