# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The stuck-verification alarms. A review handed to verification (the review driver's REVIEW_READY: state
review_ready) waits there until `rk post` posts it. One still waiting past its limit gets, from the full tick, its
VERIFY_STALE line (`lines`: how long ago it was handed to verification, and its head), at most once an hour per PR and
head (EVERY_MIN), until it is posted or dropped. The session then checks the PR's verifier (the skill's tick
procedure); when our account posted a review at that head already that stands for ours (an older monitor's under the
same account, or one a person posted by hand: reviewer_name.stands_for_ours), the line's `rk review-close N` closes
the review without a post. A mirror's line leaves that pointer out: `rk review-close` closes nothing there
(mirror.NO_CLOSE).
The orchestrator's hourly re-hand of a review no verifier holds (orchestrator.rehand) only hands it again, which a
session that believes its verifier is still running sets aside.

The limit is LIMIT_MIN for a review's verification. A split review (`rk review-split`: its plan in the review
directory, with more than one part) verifies its parts in waves, each about one verification long, then runs the
cross pass: LIMIT_MIN for each wave and one for the cross pass.

Not waited for: a review that waits on no verifier (review.waiting_on_verifier), such as one the operator skipped, or a
finished one (review.review_finished): posted, or kept unposted while posting is off.

The stall, the operator's alarm: a review still waiting on verification an hour past its own limit has stalled: 4 hours
for a review that is not split (STALL_MIN). The session has the review's VERIFY_STALE line an hour before. While the
monitor watches (`watched`: it is on, and its review pipeline verifies, in standby too; a pause or a stop says why
nothing is verified), each stall is told:
- the tick's critical line (`stalled_lines`: STALLED_LINE, the stalled reviews and the stall in words), once for each
  review that stalls;
- one line on the Needs-you list (report.board: `stalled_text`, how many reviews and the oldest's wait), under one key:
  where the monitor may send (needs_teams), its Teams copy is one message when reviews start to stall, then quiet while
  any review is still stalled;
- a line of `rk status` (control.status_lines).

st[KEY] = {"<pr>": {"head": the review's head, "said": when its line was last printed}}; a record whose review no
longer waits, or waits at another head, goes at the next full tick.
st[STALLED_KEY] = [the PRs of the stalled reviews the tick's line told]; one that no longer stalls goes at the next full
tick. A review at a new head is handed to verification again, as an entry of its own: the next full tick finds it not
stalled (its VERIFY_STALE line brings one an hour before its stall), and its PR leaves the record.
"""

from __future__ import annotations

import json

from . import features, idle, mirror, prep, report, review, split
from .snapshot import same_commit
from .store import minutes_since, now, parse_ts, stopped

KEY = "verify_stale"
LINE = "!! VERIFY_STALE"
# One verification's limit, in minutes: the monitor's own. A verifier aims at about 40 tool calls, each of which
# re-reads its context, and runs a build or test of over 4 minutes in the background
# (agentic/team_members/rocke_review_verifier.md, Running things on the shared host). Before it hands back, it waits up
# to about 20 minutes for the code quality pass (agentic/skills/rocke-monitor/review_procedure.md, On REVIEW_READY:
# verify, check, post, step 2); its context reaches 300-450K (step 4 there). A review not posted this long after its
# hand-off counts as stuck.
LIMIT_MIN = 180
EVERY_MIN = 60  # the line again for the same PR and head, at most this often
# A review stalls an hour past its own limit (stall_at): 4 hours for a review that is not split. STALL_MIN is that wait,
# in minutes: the monitor's own. The session has the review's VERIFY_STALE line an hour before.
STALL_MIN = 4 * 60
STALLED = "VERIFY_STALLED"  # the stall's kind: its tick line's, and its Needs-you line's (report.board)
STALLED_LINE = f"!! CRITICAL {STALLED}"
STALLED_KEY = "verify_stalled"


def waiting(st: dict) -> dict[int, dict]:
    """The reviews that wait on a verifier (review.waiting_on_verifier), by PR, each with the time it was handed to
    verification."""
    return {r["pr"]: r for r in review.waiting_on_verifier(st) if r.get("at")}


def split_waves(cfg: dict, r: dict) -> int:
    """The waves of part verifiers of review r when it is split: the plan `rk review-split` wrote in one of the PR's
    review directories since the review was handed to verification, with more than one part. 0 for a review not
    split."""
    since, waves = parse_ts(r["at"]).timestamp(), 0
    for d in prep.review_dirs(cfg, r["pr"]):
        f = d / split.DIR / split.PLAN
        try:
            if f.stat().st_mtime < since:
                continue
            plan = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(plan, dict) and len(plan.get("parts") or []) > 1:
            waves = max(waves, len(plan.get("waves") or []))
    return waves


def limit_min(cfg: dict, r: dict) -> int:
    """How long review r may wait for its post: LIMIT_MIN for its verification; a split review's, one for each wave of
    part verifiers and one for the cross pass."""
    return LIMIT_MIN * (1 + split_waves(cfg, r))


def _next(cfg: dict, said: dict, r: dict) -> str:
    """When review r's next line falls due: its limit after it was handed to verification, and no sooner than
    EVERY_MIN after the PR's last line. A line said for an earlier head came before r was handed to verification, so
    r's limit ends past that line's hour."""
    due = idle.plus(r["at"], limit_min(cfg, r))
    rec = said.get(str(r["pr"])) or {}
    if rec.get("said"):
        due = max(due, idle.plus(rec["said"], EVERY_MIN))
    return due


def lines(cfg: dict, st: dict, at: str | None = None) -> list[str]:
    """The VERIFY_STALE lines as of `at` (default: now), each recorded in st[KEY]; the other records go. The full tick
    runs it after the review pipeline's advance, which drops the reviews whose PR closed or moved on.
    """
    t = at or now()
    said, out, kept = st.get(KEY) or {}, [], {}
    for n, r in sorted(waiting(st).items()):
        rec = said.get(str(n))
        if rec and same_commit(rec.get("head"), r.get("head")):
            kept[str(n)] = rec
        if _next(cfg, said, r) <= t:
            age = report._ago_min(minutes_since(r["at"], t))
            close = (
                ""
                if mirror.on(cfg)
                else f"; if our account posted at that head already, run `rk review-close {n}`"
            )
            out.append(
                f"{LINE} #{n}: handed to verification {age} ago, nothing posted (head {(r.get('head') or '-')[:10]})"
                + close
            )
            kept[str(n)] = {"head": r.get("head"), "said": t}
    if kept:
        st[KEY] = kept
    else:
        st.pop(KEY, None)
    return out


def watched(cfg: dict, st: dict) -> bool:
    """Whether a stall is told: the monitor is on (not paused, nor stopped by rk stop or rk kill), and its review
    pipeline verifies (review-drafts), in standby too."""
    return (
        bool(st.get("enabled"))
        and not stopped(st)
        and features.on(cfg, "review-drafts")
    )


def stall_at(cfg: dict, r: dict) -> str:
    """When review r stalls, counted from its hand-off to verification: an hour past its own limit (limit_min),
    STALL_MIN for a review that is not split."""
    return idle.plus(r["at"], limit_min(cfg, r) + STALL_MIN - LIMIT_MIN)


def stalled(cfg: dict, st: dict, at: str | None = None) -> list[dict]:
    """The reviews stalled as of `at` (default: now), the longest waiting first; none while the monitor does not watch
    (watched)."""
    if not watched(cfg, st):
        return []
    t = at or now()
    out = [r for r in waiting(st).values() if stall_at(cfg, r) <= t]
    return sorted(out, key=lambda r: (parse_ts(r["at"]), r["pr"]))


def _stall_words(rs: list[dict], t: str) -> str:
    """The stall of the reviews rs (stalled's, as of `t`) in words: how many, and how long the oldest has waited."""
    n, oldest = len(rs), rs[0]
    return (
        f"{n} review{'s have' if n > 1 else ' has'} waited on verification for over {STALL_MIN / 60:g} hours "
        f"({'the oldest, ' if n > 1 else ''}#{oldest['pr']}, for "
        f"{report._ago_min(minutes_since(oldest['at'], t))})"
    )


def stalled_text(cfg: dict, st: dict, at: str | None = None) -> str:
    """The stall as of `at` (default: now) in one sentence: how many reviews have stalled, and how long the oldest has
    waited; "" when none has."""
    t = at or now()
    rs = stalled(cfg, st, t)
    return _stall_words(rs, t) if rs else ""


def stalled_lines(cfg: dict, st: dict, at: str | None = None) -> list[str]:
    """The tick's critical line as of `at` (default: now), when a review has stalled that the last one did not tell:
    every stalled review, and the stall in words. The stalled reviews' PRs are recorded in st[STALLED_KEY]; one that no
    longer stalls leaves the record, so that it is told again if it stalls again. The full tick runs it after the
    VERIFY_STALE lines."""
    t = at or now()
    rs = stalled(cfg, st, t)
    told = st.get(STALLED_KEY) or []
    new = [r for r in rs if r["pr"] not in told]
    if rs:
        st[STALLED_KEY] = [r["pr"] for r in rs]
    else:
        st.pop(STALLED_KEY, None)
    if not new:
        return []
    prs = " ".join(f"#{r['pr']}" for r in rs)
    return [f"{STALLED_LINE} {prs}: {_stall_words(rs, t)}"]


@idle.timer("verify stale", again=True)
def _due(cfg: dict, st: dict, store):
    """When each review that waits on a verifier gets its next VERIFY_STALE line. A line that falls due after the full
    tick's review step stays due, so the next tick is full and says it. Nothing is due while the review pipeline is off
    (review-drafts): its step, which says the lines, does not run."""
    if not features.on(cfg, "review-drafts"):
        return
    said = st.get(KEY) or {}
    for n, r in waiting(st).items():
        yield f"verify stale #{n}", _next(cfg, said, r)


@idle.timer("verify stalled", again=True)
def _stall_due(cfg: dict, st: dict, store):
    """When each review that waits on a verifier stalls, while the monitor watches (watched), until the tick's line has
    told it. A stall that falls due after the full tick's review step stays due, so the next tick is full and tells
    it."""
    if not watched(cfg, st):
        return
    told = st.get(STALLED_KEY) or []
    for n, r in waiting(st).items():
        if n not in told:
            yield f"verify stalled #{n}", stall_at(cfg, r)
