# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk followup-scan N`: the open items in every reviewer's feedback on a PR, for the follow-up tickets: resolved
comments too, when they defer work ("follow up") or got no response at all.

MERGED_OVER_RED looks only at our own rows, and only when one is open. This reads all the feedback through gh.py (GETs
only): every review thread, resolved or not, with all its comments (GraphQL), every review body, every issue comment,
and the PR's commits. It writes candidates, never verdicts: the orchestrator judges each one and files by rules.md,
Follow-up tickets. Three kinds:
- deferred: a reply that defers: a later comment in a review thread (resolved or not), a PR owner's comment in the
  conversation, or a reviewer's approval, matching the `followup_scan.defer_patterns` regexes, case-insensitive
  ("follow up", "later", "next PR", "TODO", a ticket). A reviewer's own words (a thread's first comment, a comment, a
  review that does not approve) are requests, never deferrals. The ticket keys it names (`ticket_pattern`) are
  recorded;
- unanswered: a reviewer's request (`request_patterns`) with no later reply from the PR author and no later commit
  referencing it. A thread whose code a later commit changed (GitHub marks it outdated) or that its reviewer resolved
  counts as answered; so does a commit message naming the comment or @-mentioning its reviewer;
- ours: our latest post's open 🔴/🟠 rows, its findings and the prior items it still calls OPEN (post.open_rows).
Bots are skipped, but Copilot's review threads count (not its review bodies). Our automated posts are only read for
ours. The tick scans each in-scope PR once, when it merges (`tick`); `rk followup-scan N` writes only its file.
"""

from __future__ import annotations

import pathlib
import re

from . import (
    config,
    gh,
    idle,
    paths,
    post,
    registry,
    reviewer_name,
    snapshot,
    tickets,
)
from .store import now

# Phrases that name a follow-up review or reply rather than defer work, removed before the deferral patterns run:
# "Thanks for the follow-up. N1: fixed in ...", "Follow-up on the review of <date>: ..." fix replies, a review pointing
# at the "author follow-up", "the narrow follow-ups are pushed in <sha>", "follow-up cleanup is pushed", "Follow-up to
# [the November 5 review](…)", "requires ABI 11 or later".
NOT_DEFER = re.compile(
    r"\b(?:thanks|thank you) for (?:the|your) follow[ -]?up(?: review)?\b|\bfollow[ -]?up review\b"
    r"|\bfollow[ -]?up (?:on|to) \[?(?:the|our|your|their)\b[ \w'-]{0,40}?\breviews?\b"
    r"|\bauthor(?:'s)? follow[ -]?up\b"
    r"|\bfollow[ -]?ups?(?: [\w-]+)? (?:are|is|were|was|have been|has been) (?:now )?"
    r"(?:pushed|done|landed|fixed)\b|\bor later\b",
    re.I,
)
MENTION = re.compile(r"@([\w-]+(?:/[\w-]+)?)")  # a login or a team (@acme/team)
TRIES = 3  # a failed scan is retried by the next ticks: this many tries in all
CUT = 300  # characters quoted per comment
KINDS = ("deferred", "unanswered", "ours")

_THREADS = """query($n: Int!, $after: String) { repository(owner: "%s", name: "%s") { pullRequest(number: $n) {
  title mergedAt author { login } mergedBy { login }
  reviewThreads(first: 100, after: $after) { pageInfo { hasNextPage endCursor } nodes {
    isResolved isOutdated path line originalLine resolvedBy { login }
    comments(first: 50) { nodes { databaseId url createdAt body author { __typename login } } } } } } } }"""


def read(cfg: dict, n: int) -> dict:
    """PR n's feedback, read from GitHub (GETs only)."""
    owner, name = cfg["repo"].split("/")
    threads, after = [], None
    while True:
        pr = gh.graphql(_THREADS % (owner, name), n=n, after=after)["repository"][
            "pullRequest"
        ]
        page = pr["reviewThreads"]
        threads += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    base, keep = (
        f"repos/{cfg['repo']}",
        'id, login: .user.login, type: .user.type, url: .html_url, body: (.body // "")',
    )
    return {
        "n": n,
        "title": pr["title"],
        "author": (pr.get("author") or {}).get("login") or "ghost",
        "merged_at": pr.get("mergedAt"),
        "merged_by": (pr.get("mergedBy") or {}).get("login"),
        "threads": threads,
        "comments": gh.rest(
            f"{base}/issues/{n}/comments?per_page=100",
            paginate=True,
            jq=f"[.[] | {{{keep}, at: .created_at}}]",
        ),
        "reviews": gh.rest(
            f"{base}/pulls/{n}/reviews?per_page=100",
            paginate=True,
            jq=f"[.[] | select(.submitted_at != null) | {{{keep}, state, at: .submitted_at}}]",
        ),
        "commits": gh.rest(
            f"{base}/pulls/{n}/commits?per_page=100",
            paginate=True,
            jq="[.[] | {sha, login: .author.login, at: .commit.committer.date, "
            "msg: .commit.message}]",
        ),
    }


def _patterns(cfg: dict):
    """The deferring, request and ticket patterns: config followup_scan's, each the package's
    (config.defaults.json) while it is empty; a ticket of the JIRA project (jira.project, tracker_keys) defers
    too."""
    defer = registry.value(cfg, "followup_scan.defer_patterns", registry.FALSY)
    request = registry.value(cfg, "followup_scan.request_patterns", registry.FALSY)
    keys = config.tracker_keys(cfg)
    if keys:
        defer = [*defer, rf"\b(?:{'|'.join(map(re.escape, keys))})-\d+"]
    return (
        [re.compile(p, re.I | re.M) for p in defer],
        [re.compile(p, re.I | re.M) for p in request],
        re.compile(registry.value(cfg, "followup_scan.ticket_pattern", registry.BLANK)),
    )


def _login(c: dict) -> str:
    return (c.get("author") or {}).get("login") or "ghost"


def _bot(login: str, typ: str | None) -> bool:
    return typ == "Bot" or login.endswith("[bot]")


def _copilot(login: str) -> bool:
    return (
        "copilot" in login.lower()
    )  # GraphQL names it copilot-pull-request-reviewer, REST Copilot


def _plain(body: str) -> str:
    """A comment's own words: no quoted lines (a reply quoting the ask), no HTML comments."""
    body = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    return "\n".join(
        x for x in body.splitlines() if not x.lstrip().startswith(">")
    ).strip()


def _cut(text: str) -> str:
    one = " ".join(_plain(text).replace("<br>", " ").split())
    return one if len(one) <= CUT else one[: CUT - 1] + "…"


def defers_text(text: str, pats: list) -> bool:
    """The text defers work: a deferral pattern matches once the phrases naming a follow-up review are gone."""
    text = NOT_DEFER.sub(" ", text)
    return any(p.search(text) for p in pats)


def _gist(body: str, pats: list) -> str:
    """A deferring comment, cut around its first three deferring words when it is long (a co-author's "addressed the
    first-round review" table, for example)."""
    one = " ".join(_plain(body).replace("<br>", " ").split())
    if len(one) <= CUT:
        return one
    out, end = [], 0
    for a, b in sorted(m.span() for p in pats for m in p.finditer(one)):
        if a >= end and len(out) < 3:
            lo, hi = max(end, a - 90), min(len(one), b + 90)
            out.append(one[lo:hi])
            end = hi
    return "…" + " … ".join(out) + "…" if out else _cut(one)


def _referenced(commits: list[dict], at: str, cid, login: str) -> bool:
    """A commit after `at` whose message names comment `cid` (its id, or a url holding it) or @-mentions `login`."""
    keys = ([str(cid)] if cid else []) + [f"@{login}"]
    return any(
        (c.get("at") or "") > at and any(k in (c.get("msg") or "") for k in keys)
        for c in commits
    )


def _where(cfg: dict, t: dict) -> str:
    path, prefix = t.get("path") or "-", paths.repository_folder(cfg) + "/"
    path = path[len(prefix) :] if prefix != "/" and path.startswith(prefix) else path
    line = t.get("line") or t.get("originalLine")
    return f"{path}:{line}" if line else path


def candidates(
    cfg: dict, data: dict, ours_rows: list[tuple[str, str, str]], ours_url: str | None
):
    """(candidates, read counts) for PR data from `read`; pure. Each candidate: kind, src, reviewer, by, url, quotes
    [(label, text)], resolved / resolved_by / outdated (threads), tickets, where.

    The PR's owners, its author and its commits' authors, answer for it (a co-author may answer the threads "Fixed in
    <sha>"); their own comments are no requests."""
    defer, request, ticket = _patterns(cfg)
    defers = lambda text: defers_text(text, defer)  # noqa: E731
    asks = lambda text: any(p.search(text) for p in request)  # noqa: E731
    keys = lambda texts: sorted(
        {m.group(0) for t in texts for m in ticket.finditer(t or "")}
    )  # noqa: E731
    me, commits = config.operator_login(cfg), data["commits"]
    owners = {data["author"]} | {c["login"] for c in commits if c.get("login")}
    out, bots = [], 0
    for t in data["threads"]:
        cs = sorted(t["comments"]["nodes"], key=lambda c: c["createdAt"])
        if not cs:
            continue
        first, who = cs[0], _login(cs[0])
        if _bot(who, (first.get("author") or {}).get("__typename")) and not _copilot(
            who
        ):
            bots += 1
            continue
        base = {
            "src": "thread",
            "reviewer": who,
            "resolved": t["isResolved"],
            "outdated": t["isOutdated"],
            "resolved_by": (t.get("resolvedBy") or {}).get("login"),
            "where": _where(cfg, t),
        }
        # The latest reply that defers: the ask's own words are no deferral (a review's "use the bare ticket ID"), and
        # Copilot never defers ("used later" is no promise).
        said = [
            c for c in cs[1:] if not _copilot(_login(c)) and defers(_plain(c["body"]))
        ]
        if said:
            d = said[-1]
            out.append(
                dict(
                    base,
                    kind="deferred",
                    by=_login(d),
                    url=d["url"],
                    quotes=[
                        ("ask", _cut(first["body"])),
                        ("reply", _gist(d["body"], defer)),
                    ],
                    tickets=keys(c["body"] for c in said),
                )
            )
            continue
        if who in owners or not asks(_plain(first["body"])):
            continue
        if (
            any(_login(c) in owners for c in cs[1:])
            or base["resolved_by"] == who
            or t["isOutdated"]
            or _referenced(commits, first["createdAt"], first.get("databaseId"), who)
        ):
            continue
        quotes = [("ask", _cut(first["body"]))] + (
            [(f"last of {len(cs)}", _cut(cs[-1]["body"]))] if cs[1:] else []
        )
        out.append(
            dict(
                base,
                kind="unanswered",
                by=None,
                url=first["url"],
                quotes=quotes,
                tickets=[],
            )
        )
    flat = sorted(
        [dict(c, src="comment") for c in data["comments"]]
        + [dict(r, src="review") for r in data["reviews"]],
        key=lambda x: x["at"],
    )
    for x in flat:
        if _bot(
            x["login"], x.get("type")
        ):  # Copilot's review body is an overview: only its threads count
            bots += 1
            continue
        text = _plain(x["body"])
        if not text or (x["login"] == me and reviewer_name.automated(x["body"])):
            # The account's automated posts, the early reviews without line 1 too, are no requests: this deployment's
            # own reviews give their open rows as the ours kind (`ours`, its own alone), and a review another
            # deployment posted under the same account is that deployment's to follow up, never this one's. Each is
            # known by its line 1, read as every reader reads it (reviewer_name.automated): quoted or bold, too.
            continue
        base = {
            "src": x["src"],
            "url": x["url"],
            "resolved": None,
            "resolved_by": None,
            "outdated": None,
            "where": "-",
        }
        own = (
            x["login"] not in owners
        )  # a reviewer's own words defer only in an approval (long reviews mention much)
        if (not own or x.get("state") == "APPROVED") and defers(text):
            to = x["login"] if own else ", ".join(sorted(set(MENTION.findall(text))))
            out.append(
                dict(
                    base,
                    kind="deferred",
                    reviewer=to or "-",
                    by=x["login"],
                    quotes=[("said", _gist(x["body"], defer))],
                    tickets=keys([x["body"]]),
                )
            )
        elif (
            own
            and x.get("state") != "APPROVED"
            and (x.get("state") == "CHANGES_REQUESTED" or asks(text))
            and not any(y["at"] > x["at"] and y["login"] in owners for y in flat)
            and not _referenced(commits, x["at"], x["id"], x["login"])
        ):
            out.append(
                dict(
                    base,
                    kind="unanswered",
                    reviewer=x["login"],
                    by=None,
                    quotes=[("ask", _cut(x["body"]))],
                    tickets=[],
                )
            )
    for k, where, text in ours_rows:
        out.append(
            {
                "kind": "ours",
                "src": "post",
                "reviewer": me,
                "by": None,
                "url": ours_url,
                "resolved": None,
                "resolved_by": None,
                "outdated": None,
                "where": where.strip("` ") or "-",
                "quotes": [(k, _cut(text))],
                "tickets": keys([text]),
            }
        )
    read_counts = {
        "threads": len(data["threads"]),
        "resolved": sum(bool(t["isResolved"]) for t in data["threads"]),
        "comments": len(data["comments"]),
        "reviews": len(data["reviews"]),
        "commits": len(data["commits"]),
        "bots": bots,
    }
    return out, read_counts


def _yes(v) -> str:
    return "-" if v is None else ("yes" if v else "no")


def render(
    data: dict, cands: list[dict], read_counts: dict, ts: str, project: str = ""
) -> str:
    n, r = data["n"], read_counts
    merged = (
        f"merged {data['merged_at']} by {data.get('merged_by') or '?'}"
        if data.get("merged_at")
        else "not merged"
    )
    lines = [
        f"# Follow-up scan: PR #{n} {data['title']}",
        "",
        f"Author {data['author']}; {merged}. Scanned {ts} (`rk followup-scan {n}`, GitHub reads only).",
        f"Read: {r['threads']} review threads ({r['resolved']} resolved), {r['comments']} issue comments, "
        f"{r['reviews']} reviews, {r['commits']} commits; {r['bots']} bot items skipped (Copilot's threads kept).",
        f"Candidates, not verdicts: dedup each against {project or 'the JIRA project'} and the open develop "
        "issues, then judge it (rules.md, Follow-up tickets).",
        "",
    ]
    for kind in KINDS:
        group = [c for c in cands if c["kind"] == kind]
        lines += [f"## {kind.capitalize()} ({len(group)})", ""] + (
            [] if group else ["none", ""]
        )
        for i, c in enumerate(group, 1):
            head = [
                f"- **{kind[0].upper()}{i}** {c['src']}",
                f"reviewer {c['reviewer']}",
            ]
            head += (
                [f"by {c['by']}"] if c.get("by") and c["by"] != c["reviewer"] else []
            )
            if c["src"] == "thread":
                head += [
                    f"resolved {_yes(c['resolved'])}"
                    + (f" by {c['resolved_by']}" if c["resolved_by"] else ""),
                    f"outdated {_yes(c['outdated'])}",
                ]
            head += [f"ticket {', '.join(c['tickets'])}"] if c["tickets"] else []
            head += [f"`{c['where']}`"] if c["where"] != "-" else []
            lines += [" · ".join(head), f"  {c['url'] or '-'}"]
            lines += [f"  > {label}: {text}" for label, text in c["quotes"]]
        lines += [""] if group else []
    return "\n".join(lines).rstrip() + "\n"


def out_path(cfg: dict, n: int) -> pathlib.Path:
    """PR N's candidates file, in the scans' folder: followup_scan.dir, a derived key (rkmon_lib/derived.py)."""
    d = (cfg.get("followup_scan") or {}).get("dir")
    if not d:
        raise ValueError(
            "followup_scan.dir is blank: the configuration has no folder for the scans' files"
        )
    return pathlib.Path(d).expanduser() / f"followups_{n}.md"


def scan(
    cfg: dict, st: dict, n: int, out: pathlib.Path | None = None
) -> tuple[pathlib.Path, dict]:
    """Read PR n's feedback and write its candidates file: (path, counts per kind). Writes nothing else."""
    data = read(cfg, n)
    ours = ((st.get("prs") or {}).get(str(n)) or {}).get(
        "ours"
    ) or snapshot.our_latest_review(cfg, n)
    rows = post.open_rows(post.our_post_body(cfg, ours), prior=True)
    cands, read_counts = candidates(cfg, data, rows, (ours or {}).get("url"))
    path = out or out_path(cfg, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render(
            data,
            cands,
            read_counts,
            now(),
            tickets.jira_project(cfg),
        ),
        encoding="utf-8",
    )
    return path, {k: sum(c["kind"] == k for c in cands) for k in KINDS}


def line(n: int, counts: dict, path) -> str:
    return (
        f"FOLLOWUP_SCAN #{n}: {sum(counts.values())} candidates ({counts['deferred']} deferred, "
        f"{counts['unanswered']} unanswered, {counts['ours']} ours): {path}"
    )


def tick(cfg: dict, st: dict, store, events: list[dict]) -> list[str]:
    """Scan each PR that merged this tick, once: st["followup_scans"] records each scan. A scan that fails is retried by
    the next ticks, TRIES tries in all. The FOLLOWUP_SCAN lines, for the orchestrator (SKILL.md, tick step 4).
    """
    done = st.setdefault("followup_scans", {})
    due = [str(e["pr"]) for e in events if e["kind"] == "MERGED" and e.get("pr")]
    due += [
        k
        for k, r in done.items()
        if "file" not in r and r.get("tries", 0) < TRIES and k not in due
    ]
    lines = []
    for k in due:
        rec = done.get(k) or {}
        if "file" in rec:
            continue  # scanned already: once per merge
        try:
            path, counts = scan(cfg, st, int(k))
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a scan never fails the tick; it says so and is retried
            tries = rec.get("tries", 0) + 1
            done[k] = {
                "at": now(),
                "tries": tries,
                "error": " ".join(str(exc).split())[:200],
            }
            lines.append(
                f"(followup scan #{k} failed, try {tries} of {TRIES}: {done[k]['error']})"
                if tries < TRIES
                else f"!! FOLLOWUP_SCAN #{k} failed {TRIES} times ({done[k]['error']}): run `rk followup-scan "
                f"{k}` by hand"
            )
            continue
        done[k] = {"at": now(), "file": str(path), **counts}
        lines.append(line(int(k), counts, path))
    return lines


@idle.timer("follow-up scan retries", again=True)
def _retries_due(cfg: dict, st: dict, store):
    """A scan that failed is due again at the next tick, TRIES tries in all."""
    for k, r in (st.get("followup_scans") or {}).items():
        if "file" not in r and r.get("tries", 0) < TRIES:
            yield f"follow-up scan #{k}", r.get("at") or now()
