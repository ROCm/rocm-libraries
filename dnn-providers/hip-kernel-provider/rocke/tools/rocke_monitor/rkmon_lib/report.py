# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Human-facing output: the compact tick summary for the orchestrator, the status board, and the dashboard file for the
user. The dashboard follows the HTML page's layout: the status board with one approval
list per code-owner team ("Needs platform approval", then "Needs rocke-lib approval", then "Needs any approval"), the
Mergeable Candidates (the PRs that passed CI and can merge now) and their gates, then the merge plan with its ABI
conflicts and the rebase advice collapsed."""

from __future__ import annotations

import hashlib
import pathlib
import re
import statistics
from collections import Counter

from . import abi, idle, mergeplan
from . import config
from .breakage import RED, STABILITY, entry_kind, entry_label, kind_in_state, kind_label
from .plan import gap_label
from .queue import active_holds, fixes_develop, github_conflicting
from .snapshot import ci_verdict, same_commit
from .store import minutes_since, now

# --- the work in flight: one home for the pipeline's In review box, the Reviews card, the board's Reviews row and
# `rk review-status` --------------------------------------------------------------------------------------------------

QUEUED, STAGING, DRAFTING = "queued to start", "staging", "drafting"
CHECKS_QUEUED, CHECKS_RUNNING, CHECKS_DONE = (
    "checks queued",
    "checks running",
    "checks done",
)
AWAITING, VERIFYING = "awaiting verification", "verifying"
PARTS = (
    QUEUED,
    STAGING,
    DRAFTING,
    CHECKS_QUEUED,
    CHECKS_RUNNING,
    CHECKS_DONE,
    AWAITING,
    VERIFYING,
)  # pipeline order


def pipe_part(pipe: dict, store) -> str:
    """A review in flight's part, as the pipeline runs it. "drafting" only while a drafter drafts (the local worker or a
    drafter binding: the stage the review started with); with no first-pass drafter the job stages the PR only. Then the
    checks wait for the heavy slot (or develop sanity), run, and end; the next advance hands the review to verification.
    """
    from . import draft_stage, jobs

    if pipe.get("stage") == "verify":
        return AWAITING
    ws = jobs.status(store, pipe.get("worker"))
    cs = jobs.status(store, pipe.get("checks"))
    if ws == "running" and draft_stage.of_pipe(pipe) is not draft_stage.NONE:
        return DRAFTING
    if cs == "running":
        return CHECKS_RUNNING
    if ws == "running":
        return STAGING
    return CHECKS_QUEUED if not pipe.get("checks") else CHECKS_DONE


def verifying(cfg: dict, r: dict) -> bool:
    """A review that waits on a verifier (review.waiting_on_verifier) whose verification has started:
    `rk dispatch-pack` (`rk review-prep`) wrote the facts of one of the PR's review directories (prep.review_dirs)
    after REVIEW_READY."""
    from . import prep
    from .store import parse_ts

    if not r.get("at"):
        return False
    since = parse_ts(r["at"]).timestamp()
    for d in prep.review_dirs(cfg, r["pr"]):
        try:
            if (d / "facts.txt").stat().st_mtime >= since:
                return True
        except OSError:
            continue
    return False


def in_review(cfg: dict, st: dict, store) -> list[dict]:
    """Every review in flight, {"pr", "part"}, in pipeline order: the reviews a stop or restart interrupted (they start
    again first) and the review queue's entries that may start now (`rk review-queue`): queued to start; the drafter
    pool's drafts running (a drafter binding only); each pipe in flight (pipe_part); then the reviews that wait on a
    verifier (review.waiting_on_verifier: awaiting verification, or verifying). Shadow drafts are no review of their
    own, and a finished review (review.review_finished: posted, or kept unposted) is in flight no more.
    """
    from . import draft_stage, jobs, review
    from .plan import last_review_posts

    flying = list(review.pipes(st))
    busy = {p["pr"] for p in flying} | {r["pr"] for r in st.get("review_ready") or []}
    out, last = [], None
    for r in review.interrupted(st):
        if r["pr"] not in busy:
            out.append({"pr": r["pr"], "part": QUEUED})
            busy.add(r["pr"])
    for e in st.get(review.FIRST) or []:
        if e["pr"] in busy:
            continue
        last = last_review_posts(store.ledger()) if last is None else last
        if not review.first_wait(cfg, st, store, e, last):
            out.append({"pr": e["pr"], "part": QUEUED})
            busy.add(e["pr"])
    for k, e in sorted(draft_stage.pool(st).items(), key=lambda kv: int(kv[0])):
        if int(k) not in busy and jobs.status(store, e.get("job")) == "running":
            out.append({"pr": int(k), "part": DRAFTING})
    out += [{"pr": p["pr"], "part": pipe_part(p, store)} for p in flying]
    out += [
        {"pr": r["pr"], "part": VERIFYING if verifying(cfg, r) else AWAITING}
        for r in review.waiting_on_verifier(st)
    ]
    return out


def in_review_counts(rows: list[dict]) -> str:
    """The parts in pipeline order with their counts, e.g. "1 checks running · 2 verifying"; "none" when empty."""
    c = Counter(r["part"] for r in rows)
    return " · ".join(f"{c[p]} {p}" for p in PARTS if c[p]) or "none"


def in_review_hint(cfg: dict) -> str:
    """The In review box's hint: the parts the configured pipeline runs (a drafter stage only with a drafter)."""
    from . import draft_stage

    first = (
        "drafting and checks"
        if draft_stage.for_config(cfg) is not draft_stage.NONE
        else "staging, then checks"
    )
    return f"{first}, then my verification"


LABEL = {
    "finding": "🔴/🟠 findings posted",
    "fixed": "our findings fixed before merge",
    "gate": "merges blocked by a gate",
    "develop": "develop breaks caught",
    "verify": "the reviewer model's claims corrected",
    "other": "other",
}
REVIEW_KINDS = ("FOLLOW_UP", "FIRST_ROUND")


def save_label(cfg: dict, kind: str) -> str:
    """A kind of save, in words; the verify saves name the configured reviewer model (config.reviewer_name)."""
    if kind == "verify":
        from . import config

        return f"{config.reviewer_name(cfg)}'s claims corrected"
    return LABEL.get(kind, kind)


MARK = {"pass": "✓", "fail": "✗", "pending": "…"}


def clip(text: str, n: int = 70) -> str:
    """Shorten text to n characters, marking a cut with an ellipsis."""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _develop_line(st: dict) -> str:
    d = st["develop"]
    sha = (d.get("sha") or "?")[:10]
    v = ci_verdict(d["ci"].get(d.get("sha"), {})) if d.get("sha") else "unknown"
    lg = d.get("last_green")
    lg_s = f"{lg['sha'][:10]} ({lg['how']})" if lg else "none yet"
    return f"develop {sha} CI {v}; last green {lg_s}"


# --- develop's status: one word from the latest COMPLETED develop state (the dashboards' Develop tile and card, the
# board's Develop guard row, the md dashboard) ------------------------------------------------------------------------

HEALTHY, CI_FAILURES, BROKEN, UNKNOWN = "Healthy", "CI-failures", "Broken", "Unknown"
DEVELOP_LIGHT = {
    HEALTHY: ("good", "🟢"),
    CI_FAILURES: ("warning", "🟡"),
    BROKEN: ("critical", "🔴"),
    UNKNOWN: ("muted", "⚪"),
}
# ci_verdict's answers once a SHA's CI has completed
DONE_VERDICTS = ("green", "red", "red-other", "infra")


def time_limit_labels(cfg: dict) -> set[str]:
    """The develop issue labels that say a CI time limit (a Test step or a per-test limit), not a code break: config
    incident_issue.time_limit_labels. An open issue or an incident with no culprit whose issue carries one is a CI
    failure, not a break."""
    return set((cfg.get("incident_issue") or {}).get("time_limit_labels") or ())


def _develop_window(cfg: dict, d: dict) -> list[tuple[str, int | None]]:
    """Develop's CI window (and its last green), newest first, with each commit's time (epoch seconds): one read of the
    mirror (`git log --no-walk=sorted`). Without the mirror: the window as stored, develop's head first, no times.
    """
    shas = list(
        dict.fromkeys(
            [
                *([d["sha"]] if d.get("sha") else []),
                *(d.get("ci") or {}),
                *([d["last_green"]["sha"]] if d.get("last_green") else []),
            ]
        )
    )
    repo = (cfg.get("paths") or {}).get("repo")
    if repo and shas:
        try:
            from . import gitops

            out = gitops.git(
                repo,
                "log",
                "--no-walk=sorted",
                "--format=%H %ct",
                *shas,
                timeout=30,
                lazy=False,
            )
            rows = [line.split() for line in out.splitlines() if len(line.split()) == 2]
            if {r[0] for r in rows} == set(shas):
                return [(sha, int(t)) for sha, t in rows]
        except (
            Exception
        ):  # noqa: BLE001 - the mirror is advisory here: the stored order stands
            pass
    return [(sha, None) for sha in shas]


def develop_status(cfg: dict, st: dict, store) -> dict:
    """Develop's state in one word, from the latest COMPLETED develop CI (develop's head usually still runs: develop
    moves several times an hour and its Multi-Arch CI queues for hours), the develop guard's sanity runs, the open
    incidents and the open rocKE develop issues:
    - Broken: a rocKE break on develop: the last completed run fails rocKE jobs, an open incident that is a break (it
      names a culprit, or its issue is not a CI time limit; a failing sanity run opens one), or an open rocKE CI-red
      issue that is not a CI time limit;
    - CI-failures: rocKE itself green, but red jobs outside rocKE, infrastructure only, or a CI time limit (an issue
      labelled with one of time_limit_labels, or an incident with no culprit whose issue is one);
    - Healthy: the last completed run is green and nothing above is open;
    - Unknown: no completed run to judge (and no last green).
    {"word", "level", "light", "sha", "at" (commit time, epoch s, or None), "newer" (develop has moved past it), "why",
    "verdict"}. Stability issues (breaks CI does not see) are counted in `why` only.
    """
    from . import guard

    d = st.get("develop") or {}
    window = d.get("ci") or {}
    order = _develop_window(cfg, d)
    done = [
        (sha, t)
        for sha, t in order
        if ci_verdict(window.get(sha) or {}) in DONE_VERDICTS
    ]
    lg = d.get("last_green")
    if done:
        sha, at = done[0]
        verdict = ci_verdict(window[sha])
    elif lg:
        sha, at, verdict = lg["sha"], dict(order).get(lg["sha"]), "green"
    else:
        sha, at, verdict = None, None, None
    limits = time_limit_labels(cfg)
    labels = {
        i["n"]: set(i.get("labels") or [])
        for i in (st.get("post_merge") or {}).get("open") or []
    }
    timed = lambda n: bool(labels.get(n, set()) & limits)  # noqa: E731
    broken, failing = [], []
    if verdict == "red":
        jobs = sorted(
            {
                j
                for r in (window[sha].get("runs") or {}).values()
                for j in r.get("rocke_failed") or []
            }
        )
        broken.append(f"rocKE jobs fail at {sha[:10]}: {', '.join(jobs[:3])}")
    elif verdict == "red-other":
        failing.append(f"jobs outside rocKE fail at {sha[:10]}")
    elif verdict == "infra":
        failing.append(f"CI infrastructure failed at {sha[:10]} (tests never ran)")
    for x in guard.incidents(store):
        if x.get("status") != "open" or x.get("rehearsal"):
            continue
        if not x.get("culprit") and x.get("issue") and timed(x["issue"]):
            failing.append(
                f"{x.get('id')}: no culprit, its issue #{x['issue']} is a CI time limit"
            )
        else:
            broken.append(
                f"{x.get('id')} open"
                + (f" (culprit #{x['culprit']})" if x.get("culprit") else "")
            )
    stable = 0
    for i in (st.get("post_merge") or {}).get("open") or []:
        if not i.get("rocke"):
            continue
        if entry_kind(cfg, i) != RED:
            stable += 1
        elif timed(i["n"]):
            failing.append(f"#{i['n']} ({entry_label(i)}) is a CI time limit")
        else:
            broken.append(f"#{i['n']} ({entry_label(i)}, CI red)")
    word = (
        BROKEN
        if broken
        else UNKNOWN if verdict is None else CI_FAILURES if failing else HEALTHY
    )
    level, light = DEVELOP_LIGHT[word]
    why = (
        broken
        + failing
        + (
            [
                f"{stable} {kind_label(cfg, STABILITY)} issue{'s' if stable != 1 else ''} open (CI does not see "
                "them)"
            ]
            if stable
            else []
        )
    )
    return {
        "word": word,
        "level": level,
        "light": light,
        "sha": sha,
        "at": at,
        "newer": bool(sha and d.get("sha") and not same_commit(sha, d["sha"])),
        "why": why,
        "verdict": verdict,
    }


def develop_status_line(s: dict) -> str:
    """The line under the word: the commit the verdict is for, its age, and "newer commit running" when develop has
    moved past it."""
    if not s["sha"]:
        return "no completed develop CI run yet"
    import time

    age = f" · {_ago_min((time.time() - s['at']) / 60)} old" if s.get("at") else ""
    return f"{s['sha'][:10]}{age}" + (" · newer commit running" if s["newer"] else "")


def modes(cfg: dict, sep: str = " ") -> str:
    """The features' modes in a few words: reviews=prototype merge-queue=live auto-merge=hints_only (off for a feature
    that is off)."""
    from . import features

    def m(f: str) -> str:
        return features.mode(cfg, f) if features.enabled(cfg, f) else "off"

    return sep.join(
        [
            f"reviews={m('review_automation')}",
            f"merge-queue={m('merge_queue')}",
            f"auto-merge={features.switch(cfg, 'merge_queue.auto_merge')}",
        ]
    )


def tick_summary(
    cfg: dict, st: dict, events: list[dict], plan: list[dict], queue: list[dict]
) -> str:
    state = "ON" if st["enabled"] else "OFF"
    fr = f" FROZEN({st['frozen'].get('reason', '')})" if st.get("frozen") else ""
    lines = [
        f"TICK {now()} monitor {state} {modes(cfg)}{fr}",
        _develop_line(st),
    ]
    if events:
        ev = "; ".join(
            f"{e['kind']}"
            + (f" #{e['pr']}" if e.get("pr") else "")
            + (f" {e['state']}" if e.get("state") else "")
            for e in events[:25]
        )
        lines.append(
            f"EVENTS ({len(events)}): {ev}" + (" ..." if len(events) > 25 else "")
        )
    else:
        lines.append("EVENTS: none")
    ready = [r for r in queue if r.get("position")]
    if ready:
        lines.append(
            "QUEUE: "
            + "  ".join(
                f"{r['position']}.#{r['n']}"
                + ("*" if r["cleared"] else "")
                + f"(d={r['disruption']})"
                for r in ready[:8]
            )
        )
    else:
        lines.append("QUEUE: empty (no PR passes G1-G4, G8, G9)")
    holds = active_holds(st)
    if holds:
        lines.append(
            "HOLDS: "
            + "  ".join(
                f"#{k}({h['status']})"
                for k, h in sorted(holds.items(), key=lambda kv: int(kv[0]))
            )
        )
    for lane, L in (st.get("abi_lanes") or {}).items():
        if L["groups"]:
            lines.append(
                f"ABI {lane} (develop {L['develop']}): "
                + "  ".join(
                    f"{i + 1}.{'+'.join('#' + str(n) for n in g['members'])}->{g['reserve']}"
                    + ("" if g["ready"] else "(wait)")
                    + (f"[claims {abi.claim_text(g)}]" if g.get("taken") else "")
                    for i, g in enumerate(L["groups"])
                )
            )
    bs = [b for b in st.get("batches") or [] if len(b["members"]) > 1]
    if bs:
        lines.append(
            "BATCHES: "
            + "  ".join(
                "[" + " ".join(f"#{n}" for n in b["members"]) + f"]({b['status']})"
                for b in bs[:4]
            )
        )
    if st.get("merge_plan"):
        lines.append("MERGE PLAN: " + mergeplan.summary(st["merge_plan"]))
    crit = [p for p in plan if p["critical"]]
    for p in crit:
        lines.append(f"!! CRITICAL {p['kind']}: {p['text']}")
    todo = [p for p in plan if not p["critical"]]
    if todo:
        # A follow-up waiting for the gap after our last review says until when, and one of a merge-only push says
        # so (its checks decide whether anything is posted).
        lines.append(
            f"PLAN ({len(todo)}): "
            + " | ".join(
                f"P{p['prio']} {p['kind']}"
                + (f" #{p['pr']}" if p["pr"] else "")
                + (" (merge-only)" if p.get("merge_only") else "")
                + (
                    f" (gap: after {p['gap_until'][11:16]}Z)"
                    if p.get("gap_until")
                    else ""
                )
                for p in todo[:10]
            )
        )
    return "\n".join(lines)


def gap_waiting(plan: list[dict]) -> list[dict]:
    """The plan's follow-ups that wait for the gap after our last review of their PR (plan.gap_until), as of now:
    one count for the board and the page."""
    t = now()
    return [
        p for p in plan if p["kind"] == "FOLLOW_UP" and (p.get("gap_until") or "") > t
    ]


def _ago(ts: str | None) -> str:
    return _ago_min(minutes_since(ts)) if ts else "?"


def issues_read(cfg: dict, st: dict, issues: list[dict]) -> str:
    """develop_issues once the tick has read them; before (they are read while the develop guard is on), says so,
    never "none"."""
    if (st.get("post_merge") or {}).get("at"):
        return develop_issues(cfg, issues)
    from . import features

    return (
        "not read yet"
        if features.on(cfg, "develop-checks")
        else "not read (features.merge_queue.develop_guard is off)"
    )


def develop_issues(cfg: dict, issues: list[dict]) -> str:
    """Open develop issues by kind, each by its label (breakage.entry_label), e.g. "<CI-red label> (CI red) #101;
    <stability label> #102, #103"."""
    by = {k: [i for i in issues if entry_kind(cfg, i) == k] for k in (RED, STABILITY)}
    return (
        "; ".join(
            entry_label(v[0])
            + (" (CI red) " if k == RED else " ")
            + ", ".join("#" + str(i["n"]) for i in v)
            for k, v in by.items()
            if v
        )
        or "none"
    )


EQUIV = re.compile(r"#(\d+)\b.*?(?:-> ([0-9a-f]+)|: ([0-9a-f]+) recorded)")


MERGE_ONLY = "merge-only"


def equivalent_clears(store) -> dict[tuple[str, str], tuple[str, str]]:
    """Clears without a post: the heads our review was carried to, one per (PR, new head), with the first time each was
    recorded and how: "merge-only" (a push of merges only, its checks clean), else "rebase" (rebase-only, or rebase plus
    a golden re-record). A journal line repeats when a pass is retried, so distinct pairs count, never lines.
    """
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for ln in store.journal_tail(50000):
        if ("recorded as equivalent" in ln or "our review stands" in ln) and ln[
            :20
        ].endswith("Z"):
            m = EQUIV.search(ln)
            if m:
                out.setdefault(
                    (m.group(1), (m.group(2) or m.group(3))[:8]),
                    (ln[:20], MERGE_ONLY if MERGE_ONLY in ln else "rebase"),
                )
    return out


def rebase_clears(store) -> dict[tuple[str, str], str]:
    """Every clear without a post (equivalent_clears, rebase- and merge-only alike), with the first time each was
    recorded."""
    return {k: ts for k, (ts, _) in equivalent_clears(store).items()}


# A backlog this big and growing is the user's call (more capacity). Below it, growth is a push or two arriving into a
# near-empty queue, which is no loss ("+1 in 6 h" into an empty queue).
LOSING_BACKLOG = 20


def throughput(st: dict, store, plan: list[dict], hours: int = 24) -> dict:
    """Are reviews keeping up with arrivals? Done = posts + clears without a post (rebase- or merge-only; `merge_only`
    counts the latter alone); arrived = new PRs + pushes."""
    win = hours * 60
    posts = sum(
        1
        for r in store.ledger()
        if r["kind"] == "post" and minutes_since(r["ts"]) <= win
    )
    clears = [
        how for ts, how in equivalent_clears(store).values() if minutes_since(ts) <= win
    ]
    equiv, merge_only = len(clears), clears.count(MERGE_ONLY)
    ev = [
        e
        for e in store.read_jsonl("events.jsonl")
        if e.get("ts") and minutes_since(e["ts"]) <= win
    ]
    new = sum(1 for e in ev if e.get("kind") == "NEW_PR")
    pushes = sum(1 for e in ev if e.get("kind") == "PUSH")
    items = [p for p in plan if p["kind"] in REVIEW_KINDS and p.get("pr")]
    waits = sorted(
        minutes_since(st["prs"][str(p["pr"])].get("head_seen_at") or now())
        for p in items
        if str(p["pr"]) in st["prs"]
    )
    backlog = len(items)
    trend, delta = "collecting (needs 6 h of history)", None
    old = [
        m
        for m in store.read_jsonl("metrics.jsonl")
        if "review_backlog" in m and 330 <= minutes_since(m["ts"])
    ]
    if old:
        ref = min(old, key=lambda m: abs(minutes_since(m["ts"]) - 360))
        delta = backlog - ref["review_backlog"]
        span = minutes_since(ref["ts"]) / 60
        trend = (
            f"+{delta} in {span:.0f} h"
            + (": LOSING" if backlog > LOSING_BACKLOG else "")
            if delta > 0
            else (
                f"{delta} in {span:.0f} h: winning"
                if delta < 0
                else f"flat over {span:.0f} h"
            )
        )
    return {
        "backlog": backlog,
        "posts": posts,
        "equiv": equiv,
        "merge_only": merge_only,
        "new": new,
        "pushes": pushes,
        "trend": trend,
        "delta": delta,
        "oldest_min": waits[-1] if waits else 0,
        "median_min": statistics.median(waits) if waits else 0,
    }


CI_WORD = {
    "SUCCESS": "green",
    "PENDING": "running",
    "EXPECTED": "running",
    "NONE": "not started",
}


def merge_blockers(cfg: dict, st: dict, rec: dict) -> list[dict]:
    """What still stands between a PR and its merge, each with who acts. Items are {"kind", "what", "who"}, in the
    order they usually get resolved. An
    empty list means nothing we can see blocks it.

    Kinds: label, stack, hold, review, changes, threads, approval, ci, conflict, analysis (pending, our job),
    analysis_fail, github (GitHub blocks for a reason we cannot see)."""
    from .queue import (
        FAIL,
        PENDING,
        active_holds,
        approvals,
        gate_review,
        gates_dynamic,
        hold_labels,
    )

    out = []
    for lab in hold_labels(cfg, rec):
        out.append(
            {"kind": "label", "what": f"label '{lab}'", "who": "author (remove it)"}
        )
    if rec.get("base") not in (None, "develop"):
        out.append(
            {
                "kind": "stack",
                "what": f"stacked on {rec['base']}",
                "who": "its base PR lands first",
            }
        )
    h = active_holds(st).get(str(rec["n"]))
    if h:
        out.append(
            {
                "kind": "hold",
                "what": f"merge gate failed: {h['reason'][:90]}",
                "who": "us (triage)" if h["status"] == "triage" else "author",
            }
        )
    g1, ours = gate_review(cfg, rec), rec.get("ours") or {}
    if g1[0] == FAIL and g1[1].startswith("our verdict"):
        out.append(
            {
                "kind": "review",
                "what": f"our {ours.get('verdict')} review findings",
                "who": "author",
            }
        )
    elif g1[0] != "pass":
        out.append(
            {
                "kind": "review",
                "what": (
                    "our first-round review"
                    if not ours
                    else f"our follow-up review (changed since our {ours.get('verdict')})"
                ),
                "who": "us",
            }
        )
    a = approvals(cfg, st, rec)
    me, hr = config.operator_login(cfg), (st.get("hold_reviews") or {}).get(
        str(rec["n"])
    )
    for login in a["changes"]:
        if login == me and hr:  # our hold's change request
            if h:
                continue  # the hold item says what to fix; the change request keeps GitHub from merging it meanwhile
            if me in (rec.get("review_requests") or []):
                out.append(
                    {
                        "kind": "changes",
                        "what": f"your change request from the hold ({clip(hr['reason'])}): "
                        f"{rec['author']} re-requested your review",
                        "who": "you (re-approve explicitly)",
                    }
                )
            else:
                out.append(
                    {
                        "kind": "rereview",
                        "what": f"held earlier ({clip(hr['reason'])}): re-request a review "
                        f"from {me} once the fix is in",
                        "who": "author",
                    }
                )
            continue
        out.append(
            {
                "kind": "changes",
                "what": f"CHANGES_REQUESTED by {login}",
                "who": "you" if login == me else f"{login}, after the author's fixes",
            }
        )
    if hr and hr.get("dismissed") and me not in a["changes"] and not h:
        out.append(
            {
                "kind": "changes",
                "what": f"held earlier ({clip(hr['reason'])}); its change request was dismissed "
                "without your re-approval",
                "who": "you (re-approve explicitly)",
            }
        )
    threads = (rec.get("threads") or {}).get("open") or []
    if threads:
        logins = sorted({t["login"] for t in threads})
        out.append(
            {
                "kind": "threads",
                "what": f"{len(threads)} unresolved review thread{'s' if len(threads) > 1 else ''}"
                f" ({', '.join(logins)})",
                "who": "author and " + ", ".join(logins),
            }
        )
    # GitHub never counts an author's approval of their own PR: there it takes another member of the team.
    you = (
        "another {} member" if rec.get("author") == me else "you (or another {} member)"
    )
    # An approval item's `teams`: the code-owner teams any member of which can give it (the approval boxes).
    for area in a["missing"]:
        owners = cfg.get("codeowner_teams", {}).get(area, [])
        teams = "/".join(owners) or "code owners"
        out.append(
            {
                "kind": "approval",
                "what": f"a {area} code-owner approval ({teams})",
                "who": you.format(teams) if area in a["mine"] else teams,
                "teams": list(owners),
            }
        )
    if a["other"]:
        # GitHub's pending review requests name the owners still needed (a PR whose areas are not computed yet would
        # otherwise blame code owners outside the rocKE tree).
        mine = {
            t
            for t, logins in ((st.get("codeowners") or {}).get("teams") or {}).items()
            if me in logins
        }
        req = [x for x in rec.get("review_requests") or [] if x != me]
        out.append(
            {
                "kind": "approval",
                "what": "another required approval (GitHub: review required"
                + (f"; requested: {', '.join(req)})" if req else ")"),
                "who": ", ".join(you.format(x) if x in mine else x for x in req)
                or "code owners outside the rocKE tree",
                "teams": [x for x in req if x in approval_teams(cfg)],
            }
        )
    from .queue import required_missing

    ci = rec.get("ci_rollup")
    left = required_missing(rec)
    if ci in CI_WORD and ci != "SUCCESS":
        out.append({"kind": "ci", "what": f"CI {CI_WORD[ci]}", "who": "wait"})
    elif ci != "SUCCESS":
        out.append(
            {
                "kind": "ci",
                "what": f"CI {(ci or '?').lower()}",
                "who": "author (or develop: see our review)",
            }
        )
    elif (
        left
    ):  # the rollup is green, but a required check never reported, still runs, or failed
        missing = [k for k, v in left.items() if v == "missing"]
        out.append(
            {
                "kind": "ci",
                "what": "required CI not passed: "
                + ", ".join(f"{k} ({v})" for k, v in left.items()),
                "who": (
                    "author or a maintainer (re-run CI or push)" if missing else "wait"
                ),
            }
        )
    if github_conflicting(rec):
        out.append(
            {
                "kind": "conflict",
                "what": "conflicts with develop",
                "who": "author (rebase)",
            }
        )
    # Its tests fail on today's develop (drift.py): shown before any approval, so approvers see it (a PR whose tests
    # fail on develop is never closest to ready).
    from .drift import fresh_fail

    dfail = fresh_fail(st, rec)
    if dfail:
        out.append(
            {
                "kind": "analysis_fail",
                "what": f"its tests fail on today's develop: {dfail}"[:130],
                "who": "author (merge develop, then fix the tests)",
            }
        )
    if (a["approved"] or not a["missing"]) and not dfail:
        dyn = gates_dynamic(
            rec, st["develop"].get("sha") or "", st["develop"].get("rocke_tree")
        )
        bad = [f"{k} {v[1]}" for k, v in dyn.items() if v[0] == FAIL]
        if bad:
            out.append(
                {
                    "kind": "analysis_fail",
                    "what": "merge analysis fails: " + "; ".join(bad)[:110],
                    "who": "author",
                }
            )
        elif any(v[0] == PENDING for v in dyn.values()) and not out:
            out.append(
                {
                    "kind": "analysis",
                    "what": "merge analysis (gate job) pending",
                    "who": "us",
                }
            )
    if (
        hr
        and not h
        and not any(b["kind"] in ("analysis", "analysis_fail") for b in out)
    ):
        # A held PR's fix goes to the user only once our merge analysis passes at its head.
        dyn = gates_dynamic(
            rec, st["develop"].get("sha") or "", st["develop"].get("rocke_tree")
        )
        bad = [f"{k} {v[1]}" for k, v in dyn.items() if v[0] == FAIL]
        if bad:
            out.append(
                {
                    "kind": "analysis_fail",
                    "what": "merge analysis fails: " + "; ".join(bad)[:110],
                    "who": "author",
                }
            )
        elif any(v[0] == PENDING for v in dyn.values()):
            out.append(
                {
                    "kind": "regate",
                    "what": "our merge analysis (G5–G7) of the fix for the hold",
                    "who": "us",
                }
            )
    if (
        rec.get("merge_state") == "BLOCKED"
        and ci == "SUCCESS"
        and "threads" in rec
        and not threads
        and a["approved"]
        and not left
        and (rec.get("required") or {}).get("head") == rec.get("head")
    ):
        out.append(
            {
                "kind": "github",
                "what": "GitHub still blocks the merge with CI green and the approvals in (an "
                "extra approval for changes not attributed to the author?)",
                "who": "an approver",
            }
        )
    return out


def approval_candidates(cfg: dict, st: dict) -> list[dict]:
    """PRs ready for the user's approval (no more reviews needed and no open comment left unaddressed):
    - our review is current at head and clean, with no review thread unresolved, nobody else's CHANGES_REQUESTED, no hold
      label, gate failure or conflict (merge_blockers: only approvals, CI and pending analysis may remain);
    - the user's approval is what is missing: a code-owner approval the user can give, a review requested from the user,
      or the user's own CHANGES_REQUESTED to clear. Approvals at older heads still count (the develop ruleset does not
      dismiss stale reviews), so a PR the user already approved is never listed.
    A held PR (our change request from the hold stands) is listed only once the author
    has pushed the fix and re-requested the user's review, our review is current at the new head and our merge analysis
    passes there: it needs the user's explicit re-approval.
    Not the user's own PRs. Order: develop fixes, then by merge-plan wave (approve what lands next first), then the
    user's own CHANGES_REQUESTED, CI green next, then the oldest verdict."""
    from .queue import approvals

    me = config.operator_login(cfg)
    out = []
    for rec in st["prs"].values():
        if (
            rec.get("status") != "open"
            or rec.get("draft")
            or rec.get("author") == me
            or "threads" not in rec
        ):
            continue  # threads are read for every PR our review clears; unknown means not proven addressed
        a = approvals(cfg, st, rec)
        mine_cr = a["my_state"] == "CHANGES_REQUESTED"
        dismissed = bool(
            ((st.get("hold_reviews") or {}).get(str(rec["n"])) or {}).get("dismissed")
        )
        if not (
            mine_cr or dismissed or a["mine"] or (a["requested"] and not a["approved"])
        ):
            continue
        blockers = _approvable(cfg, st, rec)
        if blockers is None:
            continue
        hr = (
            (st.get("hold_reviews") or {}).get(str(rec["n"]))
            if mine_cr or dismissed
            else None
        )
        why = (
            (
                [
                    f"held for: {hr['reason'][:90]}; the fix at {rec['head'][:10]} passes our merge analysis and "
                    f"{rec['author']} re-requested your review: it needs your explicit re-approval"
                ]
                if hr
                else (
                    ["your CHANGES_REQUESTED is the only review blocking it"]
                    if mine_cr
                    else []
                )
            )
            + (
                [f"needs a {', '.join(a['mine'])} code-owner approval: yours would do"]
                if a["mine"] and not hr
                else []
            )
            + (
                [
                    f"your {'re-' if a['my_state'] == 'APPROVED' else ''}review is requested"
                ]
                if a["requested"] and not hr
                else []
            )
        )
        out.append(
            _approval_row(
                st,
                rec,
                blockers,
                mine_cr,
                why,
                [
                    b["what"]
                    for b in blockers
                    if b["kind"] == "approval" and not b["who"].startswith("you")
                ],
            )
        )
    out.sort(key=_approval_order)
    return out


def _approvable(cfg: dict, st: dict, rec: dict) -> list[dict] | None:
    """A PR's merge blockers when only approvals, CI, our pending analysis and the user's own change request are left
    (the "ready for approval" rules); None while anything else is open. Unread threads are not proven addressed.
    """
    if rec.get("status") != "open" or rec.get("draft") or "threads" not in rec:
        return None
    blockers = merge_blockers(cfg, st, rec)
    if any(
        b["kind"] not in ("approval", "ci", "analysis")
        and not (b["kind"] == "changes" and b["who"].startswith("you"))
        for b in blockers
    ):
        return None
    return blockers


def _approval_row(
    st: dict,
    rec: dict,
    blockers: list[dict],
    mine_cr: bool,
    why: list[str],
    also: list[str],
) -> dict:
    """One row of an approval list: the PR, why it is listed, CI in words, its place in the merge plan, what else it
    needs."""
    ours = rec.get("ours") or {}
    fix = (st.get("develop_fixes") or {}).get(str(rec["n"]))
    ci = CI_WORD.get(rec.get("ci_rollup"), (rec.get("ci_rollup") or "?").lower())
    req = next((b["what"] for b in blockers if b["kind"] == "ci"), None)
    if ci == "green" and req:
        # Every reported check passed, but a required one never reported (a fork whose workflows await a
        # maintainer's approval has only bot checks): CI is incomplete, not green.
        ci = "incomplete: " + req.removeprefix("required CI not passed: ")
    return {
        "n": rec["n"],
        "title": rec["title"],
        "author": rec["author"],
        "verdict": ours.get("verdict") or "?",
        "since": ours.get("at"),
        "url": ours.get("url"),
        "head": rec["head"],
        "fix": fix,
        "mine_cr": mine_cr,
        "why": "; ".join(
            (
                [
                    (
                        f"fixes develop ({', '.join(fix)})"
                        if fixes_develop(fix)
                        else ", ".join(fix)
                    )
                ]
                if fix
                else []
            )
            + why
        ),
        "ci": ci,
        "wave": mergeplan.wave_index(st, rec["n"]),
        "place": mergeplan.place(st, rec["n"]),
        "also": also,
    }


def _approval_order(x: dict) -> tuple:
    return (
        not x["fix"],
        x["wave"],
        not x["mine_cr"],
        x["ci"] != "green",
        x["since"] or "",
    )


# The approval boxes' names ("Needs platform approval", "Needs rocke-lib approval", then "Needs any approval"); any
# other code-owner team shows its slug.
TEAM_LABEL = {"rocke-platform": "platform"}
ANY = "any"  # the last approval box's label: a general approval, any approver will do


def approval_teams(cfg: dict) -> list[str]:
    """The code-owner teams in config order (codeowner_teams): one approval box each."""
    return list(
        dict.fromkeys(
            t for ts in (cfg.get("codeowner_teams") or {}).values() for t in ts
        )
    )


def team_label(team: str) -> str:
    return TEAM_LABEL.get(team, team)


def team_approvals(cfg: dict, st: dict) -> dict[str, list[dict]]:
    """PRs ready for each code-owner team's approval, {team: rows}, teams in config order: one box per team ("Needs
    platform approval": anyone from the platform team can approve; "Needs rocke-lib approval" below it, the same rules
    for the rocke-lib team), since several teams read the dashboard. "Needs any approval" follows (any_approvals).
    The rules of approval_candidates, with the team's approval as what is missing: a code-owner area any member of the
    team can approve, or GitHub asking for the team's review. The user's own items stay visible: their re-approval after
    a hold or their standing change request (only they can clear it) lists the PR in the user's team's box too and shows
    under "also" in the others; a review requested from them is said in every box the PR is in (it is no requirement).
    The user's own PRs are listed: another team member approves. A PR missing two teams' approvals is in both boxes, each
    naming the other team under "also". Order as approval_candidates."""
    from .queue import approvals

    me = config.operator_login(cfg)
    owners = (st.get("codeowners") or {}).get("teams") or {}
    mine = {t for t, logins in owners.items() if me in logins}
    out = {t: [] for t in approval_teams(cfg)}
    for rec in st["prs"].values():
        blockers = _approvable(cfg, st, rec)
        if blockers is None:
            continue
        a = approvals(cfg, st, rec)
        mine_cr = a["my_state"] == "CHANGES_REQUESTED"
        hr = (st.get("hold_reviews") or {}).get(str(rec["n"]))
        hr = hr if hr and (mine_cr or hr.get("dismissed")) else None
        yours = (
            [
                f"{me}'s explicit re-approval (held for: {clip(hr['reason'], 90)}; the fix at {rec['head'][:10]} "
                "passes our merge analysis)"
            ]
            if hr
            else (
                [f"{me}'s change request stands (the only review blocking it)"]
                if mine_cr
                else []
            )
        )
        reqs = rec.get("review_requests") or []
        asked = (
            [
                f"{me}'s {'re-' if a['my_state'] == 'APPROVED' else ''}review is requested"
            ]
            if me in reqs and not a["approved"] and not hr
            else []
        )
        items = [b for b in blockers if b["kind"] == "approval"]
        for team, rows in out.items():
            need = [b["what"] for b in items if team in b.get("teams", [])]
            team_asked = not need and not a["approved"] and team in reqs
            if not (need or team_asked or (team in mine and yours)):
                continue
            why = (
                (
                    [f"needs {'; '.join(need)}"]
                    if need
                    else [f"review requested from {team}"] if team_asked else []
                )
                + (yours if team in mine else [])
                + asked
            )
            also = [b["what"] for b in items if team not in b.get("teams", [])] + (
                [] if team in mine else yours
            )
            rows.append(_approval_row(st, rec, blockers, mine_cr, why, also))
    for rows in out.values():
        rows.sort(key=_approval_order)
    return out


def any_approvals(cfg: dict, st: dict, teams: dict[str, list[dict]]) -> list[dict]:
    """PRs whose only missing approval is a general one, any approver will do ("Needs any approval", after the team
    boxes): the develop ruleset's extra approval that GitHub still asks for with CI green and the code-owner approvals
    in (merge_blockers' "github" item), or a required review GitHub asks for that no code-owner team box gives (an
    approval item naming no team: no request, or requests outside the code-owner teams). The entry rules of the team
    boxes (_approvable: our review current at head and clean, every review thread resolved, nothing but approvals, CI
    and our pending analysis left). A PR in a team box (`teams`, team_approvals) is never here, so each PR is in the
    team boxes it needs or in this one. Order as approval_candidates."""
    from .queue import approvals

    placed = {a["n"] for rows in teams.values() for a in rows}
    out = []
    for rec in st["prs"].values():
        if (
            rec.get("n") in placed
            or rec.get("status") != "open"
            or rec.get("draft")
            or "threads" not in rec
        ):
            continue
        blockers = merge_blockers(cfg, st, rec)
        if any(
            b["kind"] not in ("approval", "ci", "analysis", "github")
            and not (b["kind"] == "changes" and b["who"].startswith("you"))
            for b in blockers
        ):
            continue
        if any(b["kind"] == "approval" and b.get("teams") for b in blockers):
            continue  # a code-owner team's approval: its own box
        general = [b["what"] for b in blockers if b["kind"] in ("approval", "github")]
        if not general:
            continue
        mine_cr = approvals(cfg, st, rec)["my_state"] == "CHANGES_REQUESTED"
        out.append(
            _approval_row(
                st,
                rec,
                blockers,
                mine_cr,
                [f"needs {'; '.join(general)}: any approver will do"],
                (
                    [f"{config.operator_login(cfg)}'s change request stands"]
                    if mine_cr
                    else []
                ),
            )
        )
    out.sort(key=_approval_order)
    return out


def approval_line(a: dict) -> str:
    return f"#{a['n']} {a['verdict']} {a['author']}: {clip(a['title'])} · {a['why']} · our review is current, every " f"thread resolved · CI {a['ci']}" + (
        f" · also needs {'; '.join(a['also'])}" if a["also"] else ""
    ) + (
        f" · merge plan: {a['place']}" if a.get("place") else ""
    )


def close_to_merge(cfg: dict, st: dict) -> list[dict]:
    """Every PR heading for a merge (our latest verdict is 🟢/🟡, at head or before a later push), with what is left:
    develop fixes first, whatever their state, then PRs back from a hold (the user's change request still stands, so
    one must never sort past `rk approvals`' cut), then fewest items first. The dashboard's "what's left" table.
    """
    from .queue import GOOD_VERDICTS

    fixes, hr = st.get("develop_fixes") or {}, st.get("hold_reviews") or {}
    rows = []
    for rec in st["prs"].values():
        fix = fixes.get(str(rec["n"]))
        if (
            rec.get("status") != "open"
            or rec.get("draft")
            or (not fix and (rec.get("ours") or {}).get("verdict") not in GOOD_VERDICTS)
        ):
            continue
        rows.append(
            {
                "n": rec["n"],
                "title": rec["title"],
                "author": rec["author"],
                "verdict": (rec.get("ours") or {}).get("verdict") or "—",
                "fix": fix,
                "blockers": merge_blockers(cfg, st, rec),
            }
        )
    return sorted(
        rows,
        key=lambda r: (not r["fix"], str(r["n"]) not in hr, len(r["blockers"]), r["n"]),
    )


def _actor(who: str, me: str) -> str:
    """Who acts, from a merge blocker's `who`, as one name: the user's login, a team, "authors", "us" or "CI"."""
    if who.startswith("you"):
        return me
    if who.startswith("another ") and who.endswith(
        " member"
    ):  # on the user's own PR, another team member approves
        return who.removeprefix("another ").removesuffix(" member")
    if who.startswith("author"):
        return "authors"
    if who.startswith("us"):
        return "us"
    if who == "wait":
        return "CI"
    return who.split(", ")[0].split(" (")[0]


def plan_needs(cfg: dict, st: dict, ns: list[int]) -> str:
    """Who acts on PRs ns, from their merge blockers, each counted once per PR: the operator (config `operator.github_login`) first, then most PRs,
    e.g. "<me> ×3, rocke-lib ×6, authors ×2"."""
    me, cnt = config.operator_login(cfg) or "you", Counter()
    for n in ns:
        rec = st["prs"].get(str(n))
        if rec:
            cnt.update({_actor(b["who"], me) for b in merge_blockers(cfg, st, rec)})
    return ", ".join(
        f"{a} ×{c}"
        for a, c in sorted(cnt.items(), key=lambda kv: (kv[0] != me, -kv[1], kv[0]))[:4]
    )


def plan_row(cfg: dict, st: dict, mp: dict) -> tuple[str, str]:
    """The board's Merge plan row: "wave 1: #a #b …, then #x after its rebase (needs: <me> ×3,
    rocke-lib ×6)" | "hubs last: #y; 4 waves; ABI in turn: binary ×2; 1 rebase first". No list of the
    priorities: the plan's table has them."""
    if mp.get("unavailable"):
        return (
            f"not computed this tick: {clip(mp['unavailable'], 90)}",
            "the queue orders and batches as before",
        )
    w = mergeplan.wave_rows(mp)
    w1, rb = (w[0][1], w[0][2]) if w else ([], [])
    ids = " ".join("#%d" % n for n in w1[:8]) + (" …" if len(w1) > 8 else "")
    if rb:
        tail = " ".join("#%d" % n for n in rb) + (
            " after its rebase" if len(rb) == 1 else " after their rebases"
        )
        ids = f"{ids}, then {tail}" if ids else tail
    now_s = f"wave 1: {ids}" if ids else "no PR in a wave"
    needs = plan_needs(cfg, st, w1 + rb)
    now_s += f" (needs: {needs})" if needs else ""
    nxt = (
        (
            [f"hubs last: {' '.join('#%d' % n for n in mp['hubs'])}"]
            if mp.get("hubs")
            else []
        )
        + [f"{len(w)} wave{'s' if len(w) != 1 else ''}"]
        + ([mergeplan.abi_turns(mp)] if mergeplan.abi_turns(mp) else [])
        + (
            [f"{len(mp['dev_conflicts'])} rebase first"]
            if mp.get("dev_conflicts")
            else []
        )
        + (
            [
                f"partial: {mp['partial']['pairs']} pairs, {mp['partial']['prs']} PRs not checked yet"
                + (f", {mergeplan.unchecked(mp)}" if mergeplan.unchecked(mp) else "")
            ]
            if mp.get("partial")
            else []
        )
        + (["made by the planner"] if mp.get("made_by") == "planner" else [])
    )
    return now_s, "; ".join(nxt)


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def plan_counts(mp: dict) -> tuple[int, int, int]:
    """The plan's pairs between clean PRs: textual conflicts, ABI conflicts, and pairs with care only."""
    es = mp["edges"].values()
    return (
        sum(1 for e in es if "x" in e),
        sum(1 for e in es if e.get("abi")),
        sum(1 for e in es if "care" in e and "x" not in e and not e.get("abi")),
    )


def abi_lane_lines(st: dict) -> list[str]:
    """The binary and wire lanes, their claims in landing order (the lanes follow the merge plan), e.g. "binary,
    ROCKE_ABI_VERSION (develop 7): #1 → #2+#3" (a stack, or PRs with an identical hunk, is one claim).
    """
    return [
        f"{lane}, {abi.DEFINES[lane]} (develop {L['develop']}): "
        + " → ".join("+".join(f"#{m}" for m in g["members"]) for g in L["groups"])
        for lane, L in (st.get("abi_lanes") or {}).items()
        if lane in abi.DEFINES and L["groups"]
    ]


def requests_md(cfg: dict, st: dict) -> list[str]:
    """The priority requests, first in the md dashboard's merge plan as in the HTML one (other teams may request
    priority; their clicks wait for the user's OK): the pending ones with how to decide, then
    those answered or approved in the last day. Requesters' names and reasons are untrusted: one line each, cells
    escaped."""
    from . import priority, webserve
    from .doctor import rk_command

    items = priority.shown(st)
    if not items:
        return []
    out = [
        f"**Priority requests** ({config.operator_login(cfg) or 'the monitor owner'} approves or declines; other teams ask on the "
        "team dashboard):",
        "",
        "| PR | Requested by | Why | When | Status |",
        "|---|---|---|---|---|",
    ]
    for it in items:
        n = it["pr"]
        status = (
            f"pending: approve on the admin dashboard ({webserve.home_url(cfg)}) or "
            f"`{rk_command('prioritize', str(n))}`; decline there or `{rk_command('priority-decline', str(n))}`"
            if it["status"] == "pending"
            else priority.status_text(it)
        )
        why = " / ".join([it["why"]] + [a["why"] for a in it.get("also") or []])
        out.append(
            f"| #{n} | {_cell(priority.requesters(it))} | {_cell(why)} | {_ago(it['at'])} ago | {_cell(status)} |"
        )
    return out + [""]


def plan_md(cfg: dict, st: dict, details: bool = False) -> list[str]:
    """The merge plan for the md dashboard and `rk merge-plan`: the priority requests (requests_md), the order, who each
    landing makes rebase, who acts, the auto-merge notes, the ABI lanes in plan order and every planned PR's rebase
    advice (with `details`, the dashboard's collapsed <details> block, as on the HTML page).
    """
    mp = st.get("merge_plan")
    head = ["## Merge plan (conflict-free waves on today's develop)", ""]
    req = requests_md(cfg, st)
    if not mp:
        return (
            head + req + ["Not computed yet: the next tick computes it."] if req else []
        )
    out = head + req
    if mp.get("unavailable"):
        return out + [
            f"Not computed this tick: {mp['unavailable']}. The queue orders and batches as before "
            "(head-to-head pairs)."
        ]
    nodes, dc = mp["nodes"], mp["dev_conflicts"]
    nx, nabi, ncare = plan_counts(mp)
    out += [
        f"On develop `{mp['develop'][:10]}`: {len(mp['scope'])} near-merge PRs; {len(nodes)} merge cleanly with "
        f"develop, {len(dc)} conflict with it (they rebase first), {len(mp['unknown'])} not checked yet. Between the "
        f"clean ones: {nx} textual conflicts, {nabi} ABI conflicts, {ncare} more pairs to take care with"
        + (
            f"; {len(mp['unions'])} union conflicts (PRs that conflict only together)"
            if mp.get("unions")
            else ""
        )
        + "."
        + (
            f" Partial: {mp['partial']['pairs']} pairs and {mp['partial']['prs']} PRs not checked yet (they count as "
            "conflicts)"
            + (f"; {mergeplan.unchecked(mp)}" if mergeplan.unchecked(mp) else "")
            + "."
            if mp.get("partial")
            else ""
        )
        + f" {mp['merges']} merges this tick ({mp['secs']} s).",
        "",
        *([mergeplan.made_by_line(mp), ""] if mergeplan.made_by_line(mp) else []),
        f"Order: {mergeplan.RULE} ✓ nothing ahead of it conflicts: it lands as soon as it is ready (only these are "
        "batched). Textual (✗) and ABI (Ⓐ) conflicts order the plan; only textual ones hold a PR; care is shown.",
        "",
        "| Order | PRs | Landing them makes these rebase | Needs |",
        "|---|---|---|---|",
    ]

    def tag(n: int) -> str:
        node = nodes.get(str(n)) or {}
        return f"#{n} {node.get('cls', '?')}" + (" ✓" if node.get("go") else "")

    def forced(
        ns: list[int],
    ) -> str:  # a top priority that rebases first can be among them: it is in no node
        later = sorted(
            {x for n in ns for x in nodes[str(n)]["forces"] if x not in ns},
            key=lambda x: mergeplan.plan_rank(mp, x),
        )
        return (
            mergeplan.some([f"#{x} ({mergeplan.plan_cls(mp, x)})" for x in later], 8)
            or "none"
        )

    # A top priority that conflicts with develop ends its wave, "after its rebase", instead of sinking into "after a
    # rebase" with everyone else.
    for label, ns, rb in mergeplan.wave_rows(mp):
        cell = ", ".join(
            [tag(n) for n in ns]
            + [f"#{n} {mergeplan.plan_cls(mp, n)}, after its rebase" for n in rb]
        )
        out.append(
            f"| {label} | {cell} | {forced(ns) if ns else '—'} | {plan_needs(cfg, st, ns + rb) or '—'} |"
        )
    rest = mergeplan.rebase_last(mp)
    if rest:
        later = [f"#{n} {mergeplan.plan_cls(mp, n)}" for n in rest]
        out.append(
            f"| after a rebase | {', '.join(later)} | — | {plan_needs(cfg, st, rest) or '—'} |"
        )
    am = mp.get("automerge") or {}
    if am:
        out += ["", "**Auto-merge on:**"] + [
            f"- {'⚠ ' if a.get('alert') else ''}#{k}: {a['note']}"
            for k, a in sorted(am.items(), key=lambda kv: int(kv[0]))
        ]
    lanes = abi_lane_lines(st)
    if lanes:
        out += [
            "",
            "**ABI lanes, in plan order** (Ⓐ: two claims on one counter conflict; each later claim renumbers to "
            "develop + 1 once the one before it has landed):",
        ] + [f"- {x}" for x in lanes]
    order = sorted(
        mp["advice"],
        key=lambda k: ((nodes.get(k) or dc.get(k) or {}).get("rank", 10**6), int(k)),
    )
    out += (
        [""]
        + (
            [f"<details><summary>Rebase advice ({len(order)} PRs)</summary>", ""]
            if details
            else []
        )
        + [
            "| PR | Class | Place | Rebase advice | Care (no conflict) |",
            "|---|---|---|---|---|",
        ]
    )
    for k in order:
        n, care = int(k), []
        for x in (nodes.get(k) or {}).get("care", []):
            why = mp["edges"].get(f"{min(n, x)}-{max(n, x)}", {}).get("care", [])
            care.append(f"#{x} ({'; '.join(why)})")
        out.append(
            f"| #{k} | {mergeplan.plan_cls(mp, n)} | {mergeplan.place(st, n) or '—'} "
            f"| {_cell(mp['advice'][k])} | {_cell(mergeplan.some(care, 3)) or '—'} |"
        )
    out += (["", "</details>"] if details else []) + [
        "",
        "_Classes: T0 top priority, R0 ready, R1 only approvals or threads (or a CI run) left, R3 approved with "
        "fixes or our follow-up left, R4 further out. Ⓐ ABI conflict: both bump ROCKE_ABI_VERSION (or ROCKE_RECIPE_ABI): "
        "the later renumbers to develop + 1. Care: the same golden shard, DG family, ABI family string, CMake list or "
        "test-category file, or new parity indices the first to land keeps._",
    ]
    return out


MERGED_SHOWN_H = 24  # a merge while held or over our open 🔴/🟠 stays on the Needs-you list this long
IN_DEPTH = "IN_DEPTH_REVIEW"  # an in-depth review waiting for the user's approval (review.approval_need)
# The Needs-you decisions the admin dashboard takes on the line itself, by kind: the admin service's routes for its
# Approve and Decline are /<route>-approve and /<route>-decline (webserve.REVIEW_ROUTES).
DECIDED_ON_THE_PAGE = {IN_DEPTH: "review"}


def need_ask(
    kind: str, pr: int | str | None, fact: str, text: str, team: str | None = None
) -> dict:
    """A decision on the Needs-you list: its kind (the tick line's name), the PR it is about, its line, the line as the
    team sees it (the team dashboard, the Teams copy: `team`, when it is another), and a key that stays the same while
    it is the same decision. The key is the kind, the PR and the fact that makes the decision new: a new head, wave,
    request or incident state is a new decision."""
    n = int(pr) if str(pr).isdigit() else None
    return {
        "kind": kind,
        "pr": n,
        "key": f"{kind}:{n if n is not None else '-'}:{fact}",
        "text": text,
        "team": team or text,
    }


def board(cfg: dict, st: dict, store, plan: list[dict], q: list[dict]) -> dict:
    """The user's status board: one row per part of the monitor, plus what needs the user. `fp` changes only when
    something the user cares about changes, not when a running job merely gets older. `needs` is the Needs-you list's
    lines; `asks` the decisions among them (need_ask), one per PR for the approval line, which the Teams copy sends
    (needs_teams.py). Each command a line names is the one the operator types on the host (doctor.rk_command).
    """
    from . import guard
    from .doctor import rk_command

    rows, needs, key, asks = [], [], [], []

    def need(
        text: str,
        kind: str | None = None,
        pr: int | str | None = None,
        fact: str = "",
        team: str | None = None,
    ) -> None:
        """One Needs-you line; with `kind`, a decision for the user, in `asks` too (`team`: need_ask's)."""
        needs.append(text)
        if kind:
            asks.append(need_ask(kind, pr, fact, text, team))

    cleared = st.get("cleared") or []
    fz = f"; FROZEN ({st['frozen'].get('reason')})" if st.get("frozen") else ""
    mon = (
        f"{'ON' if st['enabled'] else 'OFF'}, {modes(cfg, ', ')}{fz}",
        "auto-merge only for PRs you clear: "
        + (", ".join(f"#{n}" for n in cleared) or "none cleared"),
    )
    key.append(
        mon[0] + mon[1]
    )  # the tick counts below change every tick: not part of the board's fingerprint
    ticks = idle.stats_text(st)
    rows.append(("Monitor", mon[0], mon[1] + (f"; {ticks}" if ticks else "")))
    from .review import pipes, pipes_in_effect, waiting_on_verifier

    flying = []
    for pipe in pipes(st):
        flying.append(
            f"#{pipe['pr']} {pipe['kind']}: {pipe_part(pipe, store)} (started "
            f"{_ago((pipe.get('worker') or {}).get('started'))} ago)"
        )
        key.append(f"review {pipe['pr']} {pipe.get('stage')}")
    cur = (
        (
            f"{len(flying)} of {pipes_in_effect(cfg)} parallel reviews: "
            + "; ".join(flying)
        )
        if flying
        else "idle"
    )
    if not flying:
        key.append("review idle")
    waiting = waiting_on_verifier(st)
    if waiting:
        cur += "; handed to my verification: " + ", ".join(
            f"#{r['pr']} ({VERIFYING if verifying(cfg, r) else 'waiting'}, {_ago(r.get('at'))})"
            for r in waiting
        )
        key.append("ready " + ",".join(str(r["pr"]) for r in waiting))
    fu = sum(1 for p in plan if p["kind"] == "FOLLOW_UP")
    first = sum(1 for p in plan if p["kind"] == "FIRST_ROUND")
    from . import review

    big = review.oversized(cfg, st, plan, store)
    by_hand = (
        f"; too large for the autopilot (I start them by hand): {', '.join('#%d' % b['pr'] for b in big)}"
        if (big)
        else ""
    )
    tp = throughput(st, store, plan)
    # The row counts the follow-ups waiting for the gap after our last review of their PR, and the merge-only
    # heads carried with no post.
    rows.append(
        (
            "Reviews",
            cur,
            f"waiting: {fu} follow-ups, {first} first rounds{by_hand}; {len(gap_waiting(plan))} "
            f"follow-ups wait for the {gap_label(cfg)} gap; {tp['merge_only']} heads carried as "
            "merge-only (24 h)",
        )
    )
    rows.append(
        (
            "Review throughput",
            f"oldest waiting {_ago_min(tp['oldest_min'])}, median {_ago_min(tp['median_min'])}",
            f"last 24 h: {tp['posts'] + tp['equiv']} cleared ({tp['posts']} posted, "
            f"{tp['equiv'] - tp['merge_only']} rebase-only, {tp['merge_only']} merge-only) vs "
            f"{tp['new'] + tp['pushes']} arrived ({tp['new']} new PRs, {tp['pushes']} pushes); trend {tp['trend']}",
        )
    )
    # The waiting counts and whether the trend says LOSING, not the rolling 24 h counts or the trend's size: a post
    # leaving the window is no change (it would re-show the whole board on most ticks), nor is a backlog going from
    # flat to +1; our own posts still show through the saves and stage keys.
    # Nor the gap and merge-only counts: a gap ends with time alone, and a carry shows in the waiting counts.
    key.append(f"{fu}/{first}/{'LOSING' if tp['trend'].endswith(': LOSING') else '-'}")
    from . import gpu

    g = gpu.board_row(cfg, st)  # remote GPU tests, while that feature is on
    if g:
        rows.append(g[0])
        key.append(g[1])
        needs += g[2]
    from . import daylog

    sv = daylog.saves_since(store, 7)
    rows.append(
        (
            "Issues avoided (7 d)",
            f"{sum(sv.values())} total",
            ", ".join(f"{save_label(cfg, k)} {v}" for k, v in sorted(sv.items()))
            or "none recorded yet",
        )
    )
    key.append(f"saves {sorted(sv.items())}")
    qs = st.get("queue_state") or {}
    holds = active_holds(st)
    ready = [r for r in q if r.get("position")]
    b = (st.get("batches") or [None])[0]
    order = " → ".join(f"#{n}" for n in b["members"]) if b else ""
    if qs.get("armed"):
        cur = f"#{qs['armed']['pr']} armed for auto-merge"
    elif qs.get("gate"):
        cur = f"gate job on #{qs['gate']['pr']}"
    elif qs.get("batch_gate"):
        cur = "batch gate on " + " → ".join(
            f"#{n}" for n in qs["batch_gate"]["members"]
        )
    elif b:
        cur = f"head: {order} ({b.get('status', 'candidate')})"
    elif ready:  # every ready PR waits for a PR ahead of it in the merge plan
        cur = f"no batch: {', '.join('#%d' % r['n'] for r in ready[:4])} wait for the PRs ahead of them in the merge plan"
    else:
        cur = "empty: no PR passes the readiness gates"
    near = [r for r in q if not r.get("position") and not r.get("blockers")][:3]
    near_s = ", ".join(
        f"#{r['n']}"
        + (
            f" ({', '.join(g + ' pending' for g in _pending(r))})"
            if _pending(r)
            else ""
        )
        for r in near
    )
    # PRs that pass everything but wait on a held PR (the "no impact" rule): the board shows them, not only the
    # candidates behind them.
    waiting = [
        r
        for r in q
        if any(b.startswith("waits:") for b in r.get("blockers") or [])
        and all(b.startswith(("waits:", "G9:")) for b in r["blockers"])
        and all(
            ((r.get("gates") or {}).get(g) or [(r.get("gs") or {}).get(g)])[0] == "pass"
            for g in ("G1", "G2", "G3", "G4", "G8")
        )
    ]
    held_by = sorted(
        {
            m
            for r in waiting
            for b in r["blockers"]
            for m in re.findall(r"held (#\d+)", b)
        }
    )
    wait_s = (
        (
            f"; waiting on held {', '.join(held_by)}: "
            + ", ".join(f"#{r['n']}" for r in waiting)
        )
        if waiting
        else ""
    )
    stages = stage_counts(q, st)
    from . import merges

    landed = merges.board_note(st, store)  # rocKE PRs merged per day
    rows.append(
        (
            "Merge queue",
            cur,
            f"{len(ready)} ready, {stages['candidate']} candidates ({stages['analysed']} analysed), "
            f"{len(holds)} held, {stages['label']} on a hold label"
            + wait_s
            + (f"; closest: {near_s}" if near_s else "")
            + f"; {landed}",
        )
    )
    key.append(
        f"{cur} {len(ready)}/{len(holds)} {[r['n'] for r in near]} {[r['n'] for r in waiting]} "
        f"{sorted(stages.items())} {landed}"
    )
    mp = st.get("merge_plan")
    if mp:  # which PRs land together, and in what order; it shows when the plan changes
        rows.append(("Merge plan", *plan_row(cfg, st, mp)))
        key.append(f"plan {mp.get('fp') or 'unavailable'}")
    # A hold ends with a push, but its change request stands until the user re-approves, so the board follows those PRs
    # too (they stay on it after their fix lands).
    back = {
        k: r
        for k, r in (st.get("hold_reviews") or {}).items()
        if k not in holds and k in st["prs"]
    }
    if holds or back:
        parts = [
            f"#{k} held ({clip(h['reason'])}) → {'us (triage)' if h['status'] == 'triage' else 'author'}"
            for k, h in sorted(holds.items(), key=lambda kv: int(kv[0]))
        ]
        for k, r in sorted(back.items(), key=lambda kv: int(kv[0])):
            pr = st["prs"][k]
            since = (
                f"pushed {pr['head'][:10]} since the hold"
                if pr["head"] != r["head"]
                else "no push since the hold"
            )
            left = "; ".join(
                f"{b['what']} → {b['who']}" for b in merge_blockers(cfg, st, pr)[:4]
            )
            parts.append(
                f"#{k} ({pr['author']}, {since}): {left or 'nothing left that we can see'}"
            )
        rows.append(
            (
                "Holds",
                f"{len(holds)} held"
                + (
                    f"; {len(back)} back from a hold, "
                    f"{'it needs' if len(back) == 1 else 'they need'} your explicit "
                    "re-approval"
                    if back
                    else ""
                ),
                " · ".join(parts),
            )
        )
        key.append("holds " + " · ".join(parts))
    d = st["develop"]
    ci = ci_verdict(d["ci"].get(d.get("sha"), {})) if d.get("sha") else "unknown"
    g = st.get("guard") or {}
    run = g.get("running")
    incs = [
        x
        for x in guard.incidents(store)
        if x.get("status") == "open" and not x.get("rehearsal")
    ]
    lsg = g.get("last_sanity_green")
    pm = [i for i in (st.get("post_merge") or {}).get("open") or [] if i.get("rocke")]
    ds = develop_status(cfg, st, store)
    rows.append(
        (
            "Develop guard",
            f"{ds['light']} {ds['word']}: {develop_status_line(ds)}"
            + (f"; sanity running on {run['sha'][:10]}" if run else ""),
            f"last sanity green: {lsg[:10] if lsg else 'none yet'} (sanity runs after each rocKE landing); "
            "incidents open: "
            + (
                ", ".join(
                    f"{x['id']} (culprit {'#' + str(x['culprit']) if x.get('culprit') else 'unknown'}"
                    + (f", issue #{x['issue']}" if x.get("issue") else "")
                    + ")"
                    for x in incs
                )
                or "none"
            )
            + "; rocKE develop issues open: "
            + issues_read(cfg, st, pm),
        )
    )
    key.append(
        f"{(d.get('sha') or '')[:10]} {ci} {ds['word']} {bool(run)} {lsg} {[x.get('id') for x in incs]} "
        f"{[(i['n'], i.get('kind')) for i in pm]}"
    )
    fixes = st.get("develop_fixes") or {}
    if fixes:  # develop fixes climb to the top of the review and merge queues
        fx = []
        for k, why in sorted(fixes.items(), key=lambda kv: int(kv[0])):
            left = merge_blockers(cfg, st, st["prs"][k]) if k in st["prs"] else []
            fx.append(
                f"#{k} ({', '.join(why)}): "
                + (
                    "; ".join(f"{b['what']} → {b['who']}" for b in left[:3])
                    or "nothing left that we can see"
                )
            )
        nfix = sum(fixes_develop(w) for w in fixes.values())
        nyou = len(fixes) - nfix
        rows.append(
            (
                "Develop fixes" if not nyou else "Top priority",
                ", ".join(
                    x
                    for x in (
                        (
                            f"{nfix} PR{'s' if nfix > 1 else ''} fixing develop"
                            if nfix
                            else ""
                        ),
                        f"{nyou} you prioritized" if nyou else "",
                    )
                    if x
                )
                + ": reviewed and merged first",
                " · ".join(fx),
            )
        )
        key.append("fixes " + " ".join(fx))
    if (
        b
        and b.get("status") == "validated"
        and not qs.get("armed")
        and not all(n in cleared for n in b["members"])
    ):
        heads = b.get("heads") or {}
        clear = ", ".join(
            f"`{rk_command('clear', str(n))}`" for n in b["members"] if n not in cleared
        )
        need(
            f"Merge {order}: every gate is green on develop {(d.get('sha') or '?')[:10]}. Merge it, or clear "
            f"it for auto-merge: {clear}.",
            "PROPOSE",
            b["members"][0],
            ",".join(
                f"{n}@{(heads.get(str(n)) or (st['prs'].get(str(n)) or {}).get('head') or '')[:12]}"
                for n in b["members"]
            ),
        )
    for k, risk in sorted(
        (st.get("automerge_risks") or {}).items(), key=lambda kv: int(kv[0])
    ):
        from .merge import automerge_guard_acts

        acts = automerge_guard_acts(
            cfg, st["prs"].get(k) or {"author": config.operator_login(cfg)}
        )
        off = [
            r
            for r in store.ledger()
            if r["kind"] == "automerge-off"
            and r["pr"] == int(k)
            and r["head"] == risk["head"]
        ]
        if (
            off
        ):  # the auto-merge guard acts once per head (post.automerge_off): someone turned it back on
            need(
                f"Auto-merge is on again for #{k}: {risk.get('by') or '?'} turned it back on after I turned it off "
                f"at {off[-1]['ts'][11:16]}Z, though {'; '.join(risk['reasons'])[:140]}. I don't turn it off "
                "twice: your call.",
                "AUTOMERGE_RISK",
                k,
                f"{risk['head'][:12]}:on-again",
            )
            continue
        # When the auto-merge guard acts, the line is news, not a decision: the monitor turns auto-merge off itself.
        need(
            f"Auto-merge is on for #{k} (enabled by {risk.get('by') or '?'}), but {'; '.join(risk['reasons'])[:160]}"
            + (
                ": I turn it off and explain on the PR."
                if acts
                else ": turn it off until that is fixed (the auto-merge guard only alerts here)."
            ),
            None if acts else "AUTOMERGE_RISK",
            k,
            risk["head"][:12],
        )
    for k, a in sorted(
        ((mp or {}).get("automerge") or {}).items(), key=lambda kv: int(kv[0])
    ):
        if a.get("alert"):  # auto-merge would land it ahead of the merge plan
            need(
                f"#{k} would land ahead of the merge plan ({a['note'][:220]}). Your call: ask its author to "
                "wait, or let it land.",
                "AUTOMERGE_JUMP",
                k,
                a.get("key") or "",
            )
    # A top priority ready now behind waves that wait on others, while it holds.
    for k, a in sorted(
        ((mp or {}).get("ready_early") or {}).items(), key=lambda kv: int(kv[0])
    ):
        need(mergeplan.ready_need(k, a), "PRIORITY_READY_EARLY", k, a.get("key") or "")
    for k, h in sorted(holds.items(), key=lambda kv: int(kv[0])):
        if h.get("stale_told"):
            need(
                f"#{k} has been held {_ago(h['since'])} without a push ({clip(h['reason'])}): keep its place, "
                "or release it?",
                "HOLD_STALE",
                k,
                f"{(h.get('head') or '')[:12]}:{h.get('since')}",
            )
    # A hold's change request dismissed without the user's re-approval: GitHub lets the PR merge (merge.hold_reviews).
    for k, r in sorted(
        (st.get("hold_reviews") or {}).items(), key=lambda kv: int(kv[0])
    ):
        pr = st["prs"].get(k) or {}
        if r.get("dismissed") and pr.get("status") == "open":
            need(
                f"#{k}'s change request from the hold ({clip(r.get('reason') or '')}) was removed without your "
                "re-approval, so GitHub lets it merge: re-approve it, or request changes again?",
                "HOLD_DISMISSED",
                k,
                r["dismissed"],
            )
    # A PR merged while held, or over our open 🔴/🟠 (merge.advance records them for MERGED_SHOWN_H hours).
    for k, m in sorted(
        (st.get("merged_over") or {}).items(), key=lambda kv: int(kv[0])
    ):
        if minutes_since(m["at"]) < MERGED_SHOWN_H * 60:
            need(
                f"#{k} merged {_ago(m['at'])} ago, by {m.get('by') or 'someone'}, {m['why']}. I check on develop "
                "what was open and file the develop issue if it fails there; anything more is your call.",
                m["kind"],
                k,
                (m.get("head") or "")[:12],
            )
    for x in incs:
        if minutes_since(x["start"]) >= guard.revert_offer_min(cfg):
            where = (
                f"{kind_label(cfg, kind_in_state(cfg, st, x['issue']))} issue #{x['issue']}"
                if x.get("issue")
                else "no develop issue yet: approve my draft"
            )
            fact = f"{x['id']}:{('issue-' + str(x['issue'])) if x.get('issue') else 'draft'}"
            if x.get("culprit"):
                need(
                    f"Develop has been broken {_ago(x['start'])} ({x['id']}, culprit #{x['culprit']}; "
                    f"{where}). The develop-green team force-reverts where necessary; post our revert "
                    "evidence on the issue?",
                    "REVERT_OFFER",
                    x["culprit"],
                    fact,
                )
            else:  # nothing to revert (e.g. a test past its timeout)
                need(
                    f"Develop has been red {_ago(x['start'])} ({x['id']}, no culprit to revert; {where}): "
                    "the way out is that issue's fix. Nudge its owner, or re-alert the team?",
                    "INCIDENT_AGED",
                    None,
                    fact,
                )
    for p in plan:
        if p[
            "critical"
        ]:  # the orchestrator acts on it at once (the tick's critical lines): news, not a decision
            need(f"Critical: {p['text'][:110]}")
    # Reviews that have waited on verification for hours: one line under one key while any has stalled, so its Teams
    # copy is one message when reviews start to stall (rkmon_lib/verify_stale.py).
    from . import verify_stale

    stall = verify_stale.stalled_text(cfg, st)
    if stall:
        need(stall, verify_stale.STALLED)
    if tp["delta"] is not None and tp["delta"] > 0 and tp["backlog"] > LOSING_BACKLOG:
        need(
            f"The review backlog is growing ({tp['trend']}, {tp['backlog']} waiting): decide on more capacity "
            "(another model, or parallel review agents).",
            "BACKLOG_GROWING",
            None,
            "losing",
        )
    appr = approval_candidates(cfg, st)
    if (
        appr
    ):  # your approval would do: in the platform list below, or wherever the missing approval is
        need(
            f"{len(appr)} PR{'s are' if len(appr) > 1 else ' is'} ready for your approval: no more reviews "
            f"needed, every review thread resolved ({', '.join('#%d' % a['n'] for a in appr[:8])}"
            f"{' …' if len(appr) > 8 else ''}; in the Needs approval lists below)."
        )
        asks.extend(
            need_ask("READY_FOR_APPROVAL", a["n"], a["head"][:12], approval_line(a))
            for a in appr
        )
    key.append("appr " + ",".join(f"{a['n']}:{a['why']}:{a['ci']}" for a in appr))
    # Priority requests from other teams wait for the user's OK.
    from . import priority

    for it in priority.shown(st):
        if it["status"] == "pending":
            n = it["pr"]
            need(
                f"Priority request for #{n} by {priority.requesters(it)}: {clip(it['why'], 140)}. Approve it on "
                f"the admin dashboard or with `{rk_command('prioritize', str(n))}`, or decline it there or with "
                f"`{rk_command('priority-decline', str(n))}`.",
                "PRIORITY_REQUEST",
                n,
                it.get("id") or it.get("at") or "",
            )
    # In-depth reviews wait for the user's OK, one line each, with Approve and Decline on the admin dashboard; the team
    # sees the line without its cost and commands.
    for n in review.awaiting_approval(cfg, st, plan):
        need(
            review.approval_need(cfg, st, store, n),
            IN_DEPTH,
            n,
            team=review.approval_need(cfg, st, store, n, team=True),
        )
    # Proposed tickets wait for the user's OK on the admin dashboard. Only their number here: this list is on the
    # team dashboard too, and a proposed ticket's text stays on the admin dashboard.
    from . import tickets

    waiting = tickets.waiting(st)
    if waiting:
        need(
            f"{len(waiting)} proposed ticket{'s wait' if len(waiting) > 1 else ' waits'} for your OK: approve or "
            f"decline on the admin dashboard's Tickets card (or `{rk_command('tickets')}`, "
            f"`{rk_command('ticket-file', 'ID')}`, `{rk_command('ticket-decline', 'ID')}`).",
            "TICKET_DRAFTS",
            None,
            ",".join(sorted(str(it.get("id") or it.get("at") or "") for it in waiting)),
        )
    # With no Claude session (rk rules-tick), each tick line that needs judgement waits here for a person.
    from . import rules_tick

    for e in rules_tick.entries(st):
        need(
            f"Needs a person (no Claude session here): {clip(e['line'], 220)}"
            + (f" ({clip(e['why'], 160)})" if e.get("why") else ""),
            "NEEDS_PERSON",
            e.get("pr"),
            rules_tick.key(e["line"])[:80],
        )
    # The approval boxes, one per code-owner team, then a general approval's.
    teams = team_approvals(cfg, st)
    anyone = any_approvals(cfg, st, teams)
    key.append(
        "teams "
        + " ".join(
            f"{t}:" + ",".join(f"{a['n']}:{a['why']}:{a['ci']}" for a in ts)
            for t, ts in [*teams.items(), (ANY, anyone)]
        )
    )
    # A Needs-you line's age ("red 4.0 h", "held 26 h") is no change (it would re-show the whole board every tick), nor
    # is the backlog trend's count while it stays LOSING ("+20 in 6 h" -> "+19 in 6 h"; the key above already counts
    # the reviews waiting), nor an in-depth review's cost ("about $88" -> "about $92": the spend card's mean moves with
    # each review).
    stable = [
        re.sub(
            r"\b\d+(?:\.\d+)? (?:min|h|d)\b",
            "N",
            re.sub(r"[+-]\d+ in ", "±N in ", re.sub(r"about \$[\d,]+", "about $N", t)),
        )
        for t in needs
    ]
    fp = hashlib.sha1(("|".join(key) + "||" + "|".join(stable)).encode()).hexdigest()[
        :12
    ]
    md = ["| Part | Now | Next / counts |", "|---|---|---|"] + [
        f"| {a} | {n_} | {c} |" for a, n_, c in rows
    ]
    md += ["", "**Needs you:** " + ("nothing right now." if not needs else "")] + [
        f"{i}. {t}" for i, t in enumerate(needs, 1)
    ]
    for team, ts in [*teams.items(), (ANY, anyone)]:
        if ts:
            md += [
                "",
                f"**Needs {team_label(team)} approval** ("
                + (
                    "any approver will do"
                    if team == ANY
                    else f"anyone in {team} can approve"
                )
                + "):",
            ]
            md += [f"- {approval_line(a)}" for a in ts[:10]]
            if len(ts) > 10:
                md.append(f"- …and {len(ts) - 10} more on the dashboard")
    return {
        "fp": fp,
        "md": "\n".join(md),
        "needs": needs,
        "asks": asks,
        "throughput": tp,
        "approvals": appr,
        "teams": teams,
        "any": anyone,
    }


def stage_counts(q: list[dict], st: dict | None = None) -> dict:
    """Two-tier counts: candidates (approvals + clean review at head) and how many have a fresh analysis (G5-G7). With
    `st`, freshness comes from the PRs' recorded gates (a gate job may have finished after `q` was built).
    """
    from .queue import gates_dynamic

    cands = [r for r in q if r.get("stage") == "candidate"]

    def fresh(r: dict) -> bool:
        rec = (st or {}).get("prs", {}).get(str(r["n"]))
        if rec is None:
            return not set(_dyn_pending(r))
        d = st["develop"]
        return all(
            v[0] != "pending"
            for v in gates_dynamic(
                rec, d.get("sha") or "", d.get("rocke_tree")
            ).values()
        )

    return {
        "candidate": len(cands),
        "analysed": sum(1 for r in cands if fresh(r)),
        "label": sum(
            1
            for r in q
            if any(
                "out of the queue until removed" in b for b in r.get("blockers") or []
            )
        ),
    }


def _dyn_pending(r: dict) -> list[str]:
    if "gates" in r:
        return [
            k
            for k in ("G5", "G6", "G7")
            if r["gates"].get(k, ["pending"])[0] == "pending"
        ]
    return [k for k in r.get("dyn_pending", ["G5", "G6", "G7"])]


STATIC = ("G1", "G2", "G3", "G4", "G8", "G9")


def _pending(r: dict) -> list[str]:
    """Static gates still pending for a queue row (full rows carry `gates`; the stored compact rows carry `pending`)."""
    if "pending" in r:
        return r["pending"]
    return [
        k
        for k, v in sorted(r.get("gates", {}).items())
        if k in STATIC and v[0] == "pending"
    ]


def _ago_min(m: float) -> str:
    return (
        f"{m:.0f} min"
        if m < 120
        else (f"{m / 60:.1f} h" if m < 48 * 60 else f"{m / 1440:.1f} d")
    )


GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")


def ci_cause(st: dict, rec: dict) -> tuple[str, str]:
    """A PR's CI as the dashboards sort it: ("green", ""); ("waiting", why) while it runs, when every job failing at
    its head is known infrastructure (rec ci_infra: the review's known_infra classification of that head), or when
    develop's own CI fails too; else ("pr", why): a failure of the PR's own."""
    from .queue import PASS, PENDING, gate_ci

    if "ci_rollup" not in rec:
        return "waiting", "no CI yet"
    g = gate_ci(rec)
    if g[0] == PASS:
        return "green", ""
    if g[0] == PENDING:
        return "waiting", g[1]
    infra = rec.get("ci_infra") or {}
    if infra.get("jobs") and same_commit(infra.get("head"), rec.get("head")):
        return "waiting", "known CI infrastructure: " + ", ".join(sorted(infra["jobs"]))
    d = st.get("develop") or {}
    dev = (
        ci_verdict(d["ci"].get(d["sha"], {}))
        if d.get("sha") and d.get("ci")
        else "unknown"
    )
    if dev in ("red", "red-other", "infra"):
        return "waiting", f"{g[1]}, and develop's CI fails too ({dev})"
    return "pr", g[1]


def mergeable(st: dict, q: list[dict]) -> dict:
    """What the dashboards call Mergeable Candidates: the ready tier, PRs that passed CI and can merge now (every
    readiness gate passes: G1–G4, G8, G9), in landing order. The other candidates (approvals and a clean review at head;
    the merge analysis runs on them early, queue.py) are not mergeable yet: "ci", (PR, why) waiting on CI (running,
    known infrastructure, develop failing too); "work", a CI failure of the PR's own (Needs Work); "other", (PR, first
    blocker) green CI but another gate open."""
    rows = sorted(
        (r for r in q if r.get("position") or r.get("stage") == "ready"),
        key=lambda r: (r.get("position") is None, r.get("position") or 0),
    )
    out = {"rows": rows, "ci": [], "work": [], "other": []}
    for r in q:
        if r.get("stage") != "candidate" or r.get("position"):
            continue
        state, why = ci_cause(st, st["prs"].get(str(r["n"])) or {})
        if state == "pr":
            out["work"].append(r["n"])
        elif state == "waiting":
            out["ci"].append((r["n"], why))
        else:
            first = (r.get("blockers") or [""])[0] or "pending: " + ", ".join(
                _pending(r)
            )
            out["other"].append((r["n"], first))
    return out


def not_mergeable_line(m: dict) -> str:
    """The Mergeable Candidates count line for the candidates not mergeable yet, e.g. "+2 waiting on CI"; "" for
    none."""
    parts = [
        f"+{len(m['ci'])} waiting on CI" if m["ci"] else "",
        f"+{len(m['other'])} waiting on another gate" if m["other"] else "",
    ]
    return " · ".join(p for p in parts if p)


def not_mergeable_detail(m: dict) -> str:
    """Each not-mergeable candidate and why, e.g. "#7 CI running; #8 known CI infrastructure: X"."""
    return "; ".join(f"#{n} {clip(why, 90)}" for n, why in m["ci"] + m["other"])


def gate_state(r: dict, k: str) -> str:
    """A gate's state from a full queue row (`gates`) or a stored compact row (`gs`)."""
    if "gates" in r:
        return (r["gates"].get(k) or ["pending"])[0]
    return (r.get("gs") or {}).get(k, "pending")


def stage_text(st: dict, r: dict) -> str:
    """A merge-queue row's stage, with its merge batch when it is in one, e.g. "ready · batch 1: validated"."""
    b = next(
        (
            (i, b)
            for i, b in enumerate(st.get("batches") or [], 1)
            if r["n"] in b["members"]
        ),
        None,
    )
    return (r.get("stage") or "") + (f" · batch {b[0]}: {b[1]['status']}" if b else "")


def automerge_state(st: dict, n: int) -> tuple[str, str] | None:
    """Auto-merge on PR n, (status level, words): armed by us, on with a recorded risk, or on; None when off."""
    by = ((st["prs"].get(str(n)) or {}).get("auto_merge") or {}).get("by")
    if ((st.get("queue_state") or {}).get("armed") or {}).get("pr") == n:
        return "good", "armed by us"
    if str(n) in (st.get("automerge_risks") or {}):
        return "critical", f"on ({by}): risk"
    if (st["prs"].get(str(n)) or {}).get("auto_merge"):
        return "warning", f"on ({by})"
    return None


def dashboard(
    cfg: dict,
    st: dict,
    plan: list[dict],
    queue: list[dict],
    journal: list[str],
    board_md: str | None = None,
    spend_md: list[str] | None = None,
    store=None,
) -> str:
    """The md dashboard, in the HTML page's order: develop's status word (with `store`: develop_status), the status
    board with its approval lists, the Mergeable Candidates (the PRs that passed CI and can merge now, and their gates),
    the merge plan (rebase advice collapsed); last, the review spend card (spend.card_md).
    """
    t = now()
    out = [
        f"# rocKE merge queue — {t}",
        "",
        checked_line(t),
        "",
        f"Monitor **{'ON' if st['enabled'] else 'OFF'}** · {modes(cfg, ' · ')} · "
        f"{'**FROZEN**: ' + st['frozen'].get('reason', '') if st.get('frozen') else 'not frozen'}",
        "",
    ]
    if store is not None:
        ds = develop_status(cfg, st, store)
        out += [
            f"**Develop: {ds['light']} {ds['word']}** ({develop_status_line(ds)})"
            + (f": {'; '.join(ds['why'][:4])}" if ds["why"] else ""),
            "",
        ]
    else:
        out += [f"**{_develop_line(st)}**", ""]
    if board_md:
        out += ["## Status board", "", board_md, ""]
    incidents = [
        i for i in st.get("incidents", {}).values() if i.get("status") == "open"
    ]
    if incidents:
        out += (
            ["## Open incidents", ""]
            + [f"- {i['id']}: {i.get('summary', '')}" for i in incidents]
            + [""]
        )
    m = mergeable(st, queue)
    rows = m["rows"]
    out += [
        "## Mergeable Candidates (and their gates)",
        "",
        "Passed CI and can merge now: every readiness gate passes (G1–G4, G8, G9), in landing order. G5–G7 come "
        "from gate jobs on today's develop.",
        "",
    ]
    if rows:
        out += [
            "| # | Stage | PR | Author | Lines | "
            + " | ".join(GATES)
            + " | Auto-merge | First blocker |",
            "|---" * (len(GATES) + 7) + "|",
        ]
        for r in rows:
            rec = st["prs"].get(str(r["n"])) or {}
            am = automerge_state(st, r["n"])
            out.append(
                f"| {r.get('position') or ''} | {stage_text(st, r)} | #{r['n']} | {rec.get('author', '')} | "
                f"{rec.get('size', '')} | "
                + " | ".join(MARK[gate_state(r, k)] for k in GATES)
                + f" | {am[1] if am else '—'} | {_cell(clip((r.get('blockers') or ['—'])[0], 110))} |"
            )
    else:
        out.append("No PR has passed CI and can merge now.")
    more = not_mergeable_line(m)
    if more:
        out += ["", f"{more}: {_cell(not_mergeable_detail(m))}"]
    out.append("")
    md = plan_md(cfg, st, details=True)
    out += md + [""] if md else []
    out += [
        "## Not ready (first blocker)",
        "",
        "| PR | Author | Size | Blocker | Hot spots |",
        "|---|---|---|---|---|",
    ]
    for r in queue:
        if r in rows:  # a Mergeable Candidate, above
            continue
        rec = st["prs"].get(str(r["n"])) or {}
        first = (
            r["blockers"][0]
            if r.get("blockers")
            else "pending: " + ", ".join(_pending(r))
        )
        out.append(
            f"| #{r['n']} | {r.get('author', rec.get('author', ''))} | {r.get('size', rec.get('size', ''))} | "
            f"{_cell(first[:140])} | {', '.join(r.get('hot') or [])} |"
        )
    out += ["", "## Next actions", ""] + [
        f"- P{p['prio']} **{p['kind']}** {p['text']}" for p in plan[:30]
    ]
    out += ["", "## Journal (latest)", "", "```"] + journal + ["```", ""]
    out += spend_md + [""] if spend_md else []
    out += [
        "_Legend: ✓ pass · ✗ fail · … pending. Generated by rkmon; GitHub reads only._"
    ]
    return "\n".join(out) + "\n"


# --- the md dashboard's file, its "checked" stamp, and the board's hourly showing (idle.py) ---------------------------

BOARD_EVERY_MIN = (
    60  # the tick prints the board when it changes, and at least this often
)
CHECKED = re.compile(r"^Checked \S+;", re.M)


def dashboard_path(cfg: dict) -> pathlib.Path:
    return pathlib.Path(cfg["paths"]["dashboard"])


def checked_line(t: str) -> str:
    """The md dashboard's stamp: every tick, idle ones too, writes its time here (idle.py)."""
    return f"Checked {t}; unchanged since the time above."


@idle.stamp("md dashboard")
def _md_checked(cfg: dict, store, t: str) -> list[str]:
    path = dashboard_path(cfg)
    if path.exists():
        path.write_text(
            CHECKED.sub(f"Checked {t};", path.read_text(encoding="utf-8"), count=1),
            encoding="utf-8",
        )
    return []


@idle.timer("board")
def _board_due(cfg: dict, st: dict, store):
    """When the board is printed again although it did not change (BOARD_EVERY_MIN)."""
    shown = (st.get("board") or {}).get("shown")
    if shown:
        yield "board", idle.plus(shown, BOARD_EVERY_MIN)
