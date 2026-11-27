# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The stuck-verification alarm. A review handed to verification (the review driver's REVIEW_READY: state
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

st[KEY] = {"<pr>": {"head": the review's head, "said": when its line was last printed}}; a record whose review no
longer waits, or waits at another head, goes at the next full tick.
"""

import json

from . import features, idle, mirror, prep, report, review, split
from .snapshot import same_commit
from .store import minutes_since, now, parse_ts

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
