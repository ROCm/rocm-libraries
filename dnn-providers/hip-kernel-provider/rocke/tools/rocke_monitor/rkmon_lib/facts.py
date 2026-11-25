# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk facts N`: the facts a verifier brief needs, read when the command runs and printed verbatim (lessons L-201,
L-536).

Facts an orchestrator writes into briefs from memory go wrong (a stale develop SHA, a missed force-push, a wrong post
time, ABI history lines that do not exist), so they are read here, under test. GitHub is read through gh.py (GETs only);
our posts come from the ledger (in a mirror, the live monitor's, up to the feed line: rkmon_lib/mirror.py), our
reviews kept unposted from the outputs folder, and our last review's text, posted or kept, from where
post.our_last_review reads it; the time of our last review, posted or kept, after which the force-pushes are marked and
other people's comments listed, by the follow-up gap's rule (plan.last_review_time; in a mirror, the live monitor's
last post); the ABI lanes from the last queue computation, as `rk abi` prints them; the open develop issues are read
fresh and listed as `rk breakages` lists them, without saving any state. A section that cannot be read says so, and the
others still print.

Also `rk heads N...`: one line per PR, read live.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re

from . import abi, breakage, gh, guard, reviewer_name
from . import config
from .plan import last_review_posts, last_review_time
from .snapshot import worker_meta_head
from .store import now

COMMITS_SHOWN = 6
ISSUES_SHOWN = 16
SHA = re.compile(r"[0-9a-f]{40}")

# Every force-push, not only the last 12 (a brief must never miss the author's latest force-push). With an item type
# filter, totalCount counts the whole timeline; filteredCount counts these.
_FORCE_PUSHES = """query($n: Int!) { repository(owner: "%s", name: "%s") { pullRequest(number: $n) {
  timelineItems(itemTypes: [HEAD_REF_FORCE_PUSHED_EVENT], last: 100) { filteredCount nodes {
    ... on HeadRefForcePushedEvent { createdAt actor { login } beforeCommit { oid } afterCommit { oid } } } } } } }"""
_THREADS = """query($n: Int!) { repository(owner: "%s", name: "%s") { pullRequest(number: $n) {
  reviewThreads(first: 100) { totalCount nodes { isResolved path line
    comments(first: 2) { nodes { author { login } createdAt body } } } } } } }"""


def _repo(cfg: dict) -> tuple[str, str]:
    owner, name = cfg["repo"].split("/")
    return owner, name


def _short(sha: str | None) -> str:
    return (sha or "-")[:10]


def _head(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    p = gh.rest(
        f"repos/{cfg['repo']}/pulls/{n}",
        jq="{sha: .head.sha, state, draft, mergeable, mergeable_state, "
        "user: .user.login, base: .base.ref, updated_at, merged_at, title}",
    )
    ctx["head"] = p["sha"]
    lines = [
        f"{p['sha']} {p['state']}"
        + (f" merged {p['merged_at']}" if p.get("merged_at") else "")
        + f" draft={json.dumps(p['draft'])} mergeable={json.dumps(p['mergeable'])} {p['mergeable_state']} by "
        f"{p['user']} into {p['base']} updated {p['updated_at']} | {p['title']}"
    ]
    seen = (st["prs"].get(str(n)) or {}).get("head")
    if seen and seen != p["sha"]:
        lines.append(f"(the last tick saw head {_short(seen)}: the PR moved since)")
    return lines


def _force_pushes(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    t = gh.graphql(_FORCE_PUSHES % _repo(cfg), n=n)["repository"]["pullRequest"][
        "timelineItems"
    ]
    last = ctx.get("last_review")
    lines = (
        []
        if len(t["nodes"]) >= t["filteredCount"]
        else [f"(the last {len(t['nodes'])} of {t['filteredCount']})"]
    )
    for e in t["nodes"]:
        lines.append(
            f"{e['createdAt']} {(e.get('actor') or {}).get('login') or '-'} "
            f"{_short((e.get('beforeCommit') or {}).get('oid'))} -> "
            f"{_short((e.get('afterCommit') or {}).get('oid'))}"
            + (" (after our last review)" if last and e["createdAt"] > last else "")
        )
    return lines or ["none"]


def _commits(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    rows = gh.rest(
        f"repos/{cfg['repo']}/pulls/{n}/commits?per_page=100",
        paginate=True,
        jq='[.[] | {sha, at: .commit.committer.date, subject: (.commit.message | split("\\n")[0])}]',
    )
    ctx["head_at"] = next((r["at"] for r in rows if r["sha"] == ctx.get("head")), None)
    lines = (
        [f"(the last {COMMITS_SHOWN} of {len(rows)})"]
        if len(rows) > COMMITS_SHOWN
        else []
    )
    return lines + [
        f"{_short(r['sha'])} {r['at']} {r['subject']}" for r in rows[-COMMITS_SHOWN:]
    ]


def _ours(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    """Our reviews of PR n: our posts as the ledger has them, then our review files in the outputs folder, each a
    post's copy or a review kept unposted; in a mirror the live monitor's last post in place of the files.
    """
    lines = [
        f"{r['ts']} {r['kind']} at {_short(r['head']) if SHA.fullmatch(r.get('head') or '') else r.get('head')} "
        f"{r.get('ref') or '-'}" + (f" ({r['note']})" if r.get("note") else "")
        for r in ctx["posts"]
    ]
    from . import mirror

    if mirror.on(cfg):
        ours = (st["prs"].get(str(n)) or {}).get("ours") or {}
        lines += (
            [f"our last post: the live monitor's, under our login: {ours['url']}"]
            if ours.get("url")
            else []
        )
    else:
        from . import post

        lines += [
            f"{'kept unposted' if post.is_kept_file(p) else 'copy'}: {p} ({_mtime(p)})"
            for p in post.copies(cfg, n)
        ]
    return lines or ["none"]


_FINDING = re.compile(r"\| ([NQ]\d+[a-z]?) \| (\S+) \| .*?\*\*(.+?)\*\*")
# A "Highly recommended follow-up (not blocking)" row: | F<k> | 🟠/🟡 | file:line | **item** ... |
_FOLLOWUP = re.compile(r"^\| (F\d+) \| (🟠|🟡) \| [^|]*\| (?:\*\*)?([^|*<]+)")
_FLAGGED = re.compile(r"\| ([^|]+?) \| ((?:🔴|🟠|🟡)[^|<]*)")


def _last_findings(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    """Our last review's header, its flagged status rows and its finding rows, read where post.our_last_review reads
    it, so a brief names our open items from the review itself (a brief written from memory can ask about another PR's
    finding). In a mirror it is the live monitor's post as GitHub holds it under our login, the one the live review
    follows up on.
    """
    from . import mirror, post

    ours = ((st.get("prs") or {}).get(str(n)) or {}).get("ours")
    text, src = post.our_last_review(cfg, n, ours)
    if not text:
        return ["none"]
    lines = [
        f"from {src}"
        + (" (the live monitor's post, under our login)" if mirror.on(cfg) else "")
    ]
    for line in text.splitlines():
        if line.startswith("## PR #"):
            lines.append(line)
        elif m := _FINDING.match(line):
            lines.append(f"{m[1]} {m[2]} {m[3]}")
        elif m := _FOLLOWUP.match(line):
            lines.append(f"{m[1]} {m[2]} follow-up (not blocking): {m[3]}")
        elif (m := _FLAGGED.match(line)) and m[1] not in ("#", "Prior item"):
            lines.append(f"{m[1]}: {m[2].strip()}")
    return lines


def _abi(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    lines = abi.lane_lines(st, n)
    if all(line.startswith(f"#{n}:") for line in lines):
        lines.insert(0, f"#{n} is in no ABI lane")
    return lines + [f"(lanes as of the last tick, {st.get('last_tick') or '-'})"]


def _breakages(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    issues = breakage.fetch_open(cfg)
    for i in issues:
        i["rocke"] = breakage.relevant(cfg, i)
    lines = breakage.issue_lines(issues, guard.incidents(store))
    more = len(lines) - ISSUES_SHOWN
    return (
        lines[:ISSUES_SHOWN] + ([f"(+{more} more: `rk breakages`)"] if more > 0 else [])
    ) or ["none open"]


def _reviews(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    rows = gh.rest(
        f"repos/{cfg['repo']}/pulls/{n}/reviews?per_page=100",
        paginate=True,
        jq='[.[] | {login: (.user.login // "ghost"), state, at: .submitted_at, commit: .commit_id}]',
    )
    latest, standing = {}, {}
    for r in rows:  # oldest first
        latest[r["login"]] = r
        if r["state"] in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            standing[r["login"]] = r

    def line(r: dict) -> str:
        at_head = bool(ctx.get("head")) and r["commit"] == ctx["head"]
        # GitHub moves a review onto a head that only merged develop (an approval listed at a later merge commit), so
        # the reviewer saw the head only if they submitted after it was committed.
        moved = (
            at_head and bool(ctx.get("head_at")) and (r["at"] or "") < ctx["head_at"]
        )
        return f"{r['login']} {r['state']} {r['at'] or '-'} {_short(r['commit'])}" + (
            " (GitHub moved it to the head; submitted before the head was committed)"
            if moved
            else " (at head)" if at_head else ""
        )

    out = []
    for r in sorted(latest.values(), key=lambda r: r["at"] or ""):
        s = standing.get(r["login"])
        # GitHub keeps a reviewer's approval or change request through their later comments: the brief shows the
        # standing review, not the later comment alone.
        out.append(
            line(s) + f"; later {r['state']} {r['at'] or '-'} {_short(r['commit'])}"
            if s and s is not r
            else line(r)
        )
    return out or ["none"]


def _threads(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    t = gh.graphql(_THREADS % _repo(cfg), n=n)["repository"]["pullRequest"][
        "reviewThreads"
    ]
    left = [x for x in t["nodes"] if not x["isResolved"]]
    lines = [
        f"total {t['totalCount']}, unresolved {len(left)}"
        + (
            f" (of the first {len(t['nodes'])})"
            if len(t["nodes"]) < t["totalCount"]
            else ""
        )
    ]
    for x in left:
        said = " / ".join(
            f"{(c.get('author') or {}).get('login') or '?'}: "
            + " ".join((c.get("body") or "").split())[:200]
            for c in x["comments"]["nodes"]
        )
        lines.append(f"* {x.get('path') or '-'}:{x.get('line') or 0} {said}")
    return lines


def _user_comments(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    # In full: a one-line summary can drop an ask ("still open before this can merge"). A reply in a review thread
    # ("will follow up") is no conversation comment; both are read.
    # Our automated posts share the account. Each is known by its line 1, as followups knows it: it starts "Automated"
    # or is an early review's header, read as every reader reads it, quoted or bold too (reviewer_name.automated).
    me = config.operator_login(cfg)
    conversation = gh.rest(
        f"repos/{cfg['repo']}/issues/{n}/comments?per_page=100",
        paginate=True,
        jq=f'[.[] | select(.user.login == "{me}") | {{at: .created_at, url: .html_url, '
        'body: (.body // "")}]',
    )
    threads = gh.rest(
        f"repos/{cfg['repo']}/pulls/{n}/comments?per_page=100",
        paginate=True,
        jq=f'[.[] | select(.user.login == "{me}") | {{at: .created_at, url: .html_url, '
        'body: (.body // ""), path: (.path // ""), line: (.line // .original_line // 0)}]',
    )
    rows = [*conversation, *threads]
    lines = []
    for r in sorted(rows, key=lambda r: r["at"]):
        if not reviewer_name.automated(r["body"]):
            where = (
                f" (review thread, {r['path']}:{r['line']})" if r.get("path") else ""
            )
            lines += [f"--- {r['at']} {r['url']}{where}", r["body"].rstrip()]
    return lines or ["none"]


OTHERS_SHOWN = 8


def _others_comments(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    """Other people's conversation comments since our last review, posted or kept unposted, in full (a reviewer's
    "Blocking" comment or a reminder belongs in the brief beside the operator's)."""
    me, since = config.operator_login(cfg), ctx.get("last_review") or ""
    rows = gh.rest(
        f"repos/{cfg['repo']}/issues/{n}/comments?per_page=100",
        paginate=True,
        jq=f'[.[] | select(.user.login != "{me}" and .user.type != "Bot") | '
        '{login: .user.login, at: .created_at, url: .html_url, body: (.body // "")}]',
    )
    new = [r for r in rows if r["at"] > since]
    lines = (
        [f"({len(new) - OTHERS_SHOWN} earlier ones not shown: read them on the PR)"]
        if len(new) > OTHERS_SHOWN
        else []
    )
    for r in new[-OTHERS_SHOWN:]:
        lines += [f"--- {r['login']} {r['at']} {r['url']}", r["body"].rstrip()]
    return lines or ["none" + (f" since our last review ({since})" if since else "")]


def _worker(cfg: dict, st: dict, store, n: int, ctx: dict) -> list[str]:
    sha, mtime = worker_meta_head(cfg, n)
    lines = [
        (
            f"pr/meta.md: head {_short(sha)}, written {mtime}"
            + (
                ""
                if not ctx.get("head") or (sha and ctx["head"].startswith(sha))
                else " (not the head)"
            )
            if sha
            else "pr/meta.md: none"
        )
    ]
    d = pathlib.Path(cfg["paths"]["workers"]) / f"pr{n}" / "checks"
    files = sorted(d.iterdir()) if d.is_dir() else []
    return lines + [
        f"checks/{p.name} {p.stat().st_size} bytes {_mtime(p)}" for p in files[:30]
    ]


def _mtime(p: pathlib.Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


SECTIONS = (
    ("head and state", _head),
    ("force-pushes (every head the PR had, newest last)", _force_pushes),
    ("commits", _commits),
    (
        "our reviews (the posts in the ledger, then the files in the outputs folder)",
        _ours,
    ),
    (
        "our last review: header, flagged rows, findings (copy them into the brief; never recall them)",
        _last_findings,
    ),
    ("rk abi", _abi),
    ("rk breakages", _breakages),
    ("reviews (latest per reviewer)", _reviews),
    ("unresolved threads", _threads),
    (
        "{me}'s hand-written comments, in full (our automated posts share the account)",
        _user_comments,
    ),
    (
        "other people's comments since our last review, in full (bots left out)",
        _others_comments,
    ),
    ("worker inputs", _worker),
)


def render(cfg: dict, st: dict, store, n: int) -> tuple[list[str], int]:
    """The facts for PR n, section by section, and how many sections could not be read."""
    from . import mirror

    # In a mirror: the live monitor's ledger, up to the feed line, whose last post dates our last review.
    live = mirror.live_posts(cfg, st, n)
    if live is not None:
        posts = live
        last = max((r["ts"] for r in live if r["kind"] == "post"), default=None)
    else:
        posts = [r for r in store.ledger() if r.get("pr") == n]
        ours = ((st.get("prs") or {}).get(str(n)) or {}).get("ours")
        last = last_review_time(n, ours, last_review_posts(posts))
    ctx = {"posts": posts, "last_review": last}
    out, failed = [f"# Facts for #{n}, read {now()}"], 0
    for title, section in SECTIONS:
        out.append("== " + title.format(me=config.operator_login(cfg)))
        try:
            out += section(cfg, st, store, n, ctx)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one section that cannot be read must not hide the others
            failed += 1
            out.append(f"(not read: {str(exc)[:300]})")
    return out, failed


def _jq(value) -> str:
    """A value as jq's string interpolation prints it: a string bare, anything else as JSON (true, false, null)."""
    return value if isinstance(value, str) else json.dumps(value)


def heads(cfg: dict, ns: list[int]) -> tuple[list[str], bool]:
    """`rk heads N...`: per PR, `<n> <head, 10> <state> merged=<bool> mergeable=<mergeable state> updated=<time>`, and
    whether every PR was read."""
    lines, ok = [], True
    for n in ns:
        try:
            p = gh.rest(
                f"repos/{cfg['repo']}/pulls/{n}",
                jq="{number, sha: .head.sha, state, merged, mergeable_state, updated_at}",
            )
        except gh.GhError as exc:
            lines.append(f"{n} (not read: {exc})")
            ok = False
            continue
        lines.append(
            f"{p['number']} {p['sha'][:10]} {p['state']} merged={_jq(p['merged'])} "
            f"mergeable={_jq(p['mergeable_state'])} updated={_jq(p['updated_at'])}"
        )
    return lines, ok
