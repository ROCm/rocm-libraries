# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk tick-filter`: the lines of a tick's output that the orchestrator must act on.

KINDS is the one table of tick line kinds. Each row names a kind by how its line starts (a regex), says what the
orchestrator does with it, names the step that prints it, and says what it means or what the orchestrator does (the
skill's tick procedure has the details). Rows are tried in order and the first match wins, so a narrower row comes
before a broader one of the same kind. A step that prints a new kind adds its row here.

The orchestrator acts on two classes of line:
- ACT: every time the line appears;
- STANDING: a line the tick prints again on every tick while its condition holds (a proposal waiting for the user, a
  notice not posted yet). It acts once; with `--seen FILE` (the tick loop's), a standing line already reported stays
  quiet until it goes away and comes back. Any other tick output (an idle tick, a tick that failed) leaves FILE as it
  is.
ROUTINE lines need nothing.

Fail closed: a line no row knows needs the orchestrator, and so does a tick that printed nothing or exited with an error
(`--rc`). The board a tick prints after `BOARD (changed)` or `BOARD (hourly)`, up to the `time:` line, is the user's:
the dashboard shows it, and none of it needs the orchestrator.
"""

from __future__ import annotations

import pathlib
import re

from . import paths, whole_file

ACT, STANDING, ROUTINE = "act", "standing", "routine"

# What rk tick prints when it finds another tick running (rkmon.main): KINDS's row below, the orchestrator's session
# and tools/tick_loop.py know a skipped tick by it. The tick loop reads no rkmon_lib module, so a test ties its copy
# of the text to this one.
TICK_RUNNING = "another tick is running"

KINDS = (
    # (the line starts with, class, printed by, what it means or what the orchestrator does)
    # -- any step
    (
        r"!! GH_LOGIN ",
        STANDING,
        "ghlogins.tick_lines",
        "the GitHub read/write split does not hold: tell the user; it stands until the logins are fixed",
    ),
    (
        r"!! ",
        ACT,
        "any step",
        "critical or urgent: report it first, then do what the tick procedure says",
    ),
    (
        r"\(",
        ROUTINE,
        "any step",
        "an advisory note; it needs nothing unless the same one repeats for an hour",
    ),
    # -- the tick itself
    (
        r"GH_LOGINS ",
        ROUTINE,
        "ghlogins.tick_lines",
        "which GitHub login each side uses (probed hourly)",
    ),
    (
        r"GH_CAP_WAIT ",
        ROUTINE,
        "rkmon.cmd_tick",
        "gh calls waited for the GitHub request cap while the tick ran; the day log and the admin dashboard say so too",
    ),
    (
        r"GH_CAP_OFF: ",
        STANDING,
        "gh_cap.set_cap",
        "the GitHub request cap's record cannot be written, so no gh call is counted: tell the user, with the fix the "
        "line names",
    ),
    (
        r"gh: waiting until \S+ for the GitHub request cap ",
        ROUTINE,
        "gh_cap.on_stderr",
        "a gh call of the tick waits for the GitHub request cap; GH_CAP_WAIT says how long",
    ),
    (
        r"TICK idle: ",
        ROUTINE,
        "idle.idle_line",
        "nothing changed since the last full tick",
    ),
    (
        r"TICK \S+ monitor OFF",
        ACT,
        "rkmon.cmd_tick",
        "the monitor is off: say so in one line; no loop until it is on",
    ),
    (
        r"TICK \S+ " + re.escape(TICK_RUNNING),
        ROUTINE,
        "rkmon.main",
        "a tick was running already; the next one runs",
    ),
    (r"TICK ", ROUTINE, "report.tick_summary", "the tick's header"),
    (r"develop \S+ CI ", ROUTINE, "report.tick_summary", "develop's head and CI"),
    (r"EVENTS", ROUTINE, "report.tick_summary", "what the poll saw"),
    (r"QUEUE:", ROUTINE, "report.tick_summary", "the merge queue"),
    (r"HOLDS:", ROUTINE, "report.tick_summary", "the holds"),
    (r"ABI ", ROUTINE, "report.tick_summary", "the ABI lanes"),
    (r"BATCHES:", ROUTINE, "report.tick_summary", "the merge batches"),
    (r"MERGE PLAN:", ROUTINE, "report.tick_summary", "the merge plan"),
    (r"PLAN \(", ROUTINE, "report.tick_summary", "the review and merge plan"),
    (
        r"BOARD ",
        ROUTINE,
        "rkmon.full_tick",
        "the status board: the user reads it on the dashboard",
    ),
    (r"time: ", ROUTINE, "rkmon.full_tick", "each step's time"),
    (
        r"check your internet connection",
        ROUTINE,
        "gh._run",
        "the second line of gh's network error, a fragment: the line before it says what failed",
    ),
    (
        r"REVIEW_RESUME ",
        ROUTINE,
        "review._fill",
        "a review a stop or restart interrupted starts again at its head",
    ),
    (
        r"REVIEW_RESTORED ",
        ACT,
        "review_keep.sync",
        "a review directory came back from its disk copy: finish the verification already written there",
    ),
    (
        r"DRAINING ",
        ROUTINE,
        "rkmon.tick_checks",
        "an `rk drain` or `rk stop --drain` drains: nothing new starts until it ends",
    ),
    (
        r"JOB_ORPHAN ",
        ROUTINE,
        "jobs.reap_orphans",
        "a job a tick or command started but never saved was killed; its step starts the work again",
    ),
    (
        r"JOB_STRAY ",
        ROUTINE,
        "jobs.free_tree",
        "an earlier job's stray processes were killed before its tree is reused",
    ),
    # -- the merge queue
    (r"PROPOSE ", STANDING, "merge.advance", "tell the user the validated order"),
    (r"ARM_READY ", STANDING, "merge.advance", "a cleared batch: `rk arm N`"),
    (
        r"QUEUE_NOTICE ",
        STANDING,
        "merge.advance",
        "`rk queue-notice N` (only while queue notices are on)",
    ),
    (
        r"TRIAGE ",
        ACT,
        "merge.advance",
        "triage the held failure on develop, then `rk hold` or `rk release`",
    ),
    (
        r"HOLD_STALE ",
        ACT,
        "merge.advance",
        "ask the user whether to keep the PR's place",
    ),
    (
        r"HOLD_FIXED ",
        ACT,
        "merge.hold_reviews",
        "the fix gets our follow-up review and a re-gate; tell the user",
    ),
    (
        r"HOLD_REAPPROVED ",
        ROUTINE,
        "merge.hold_reviews",
        "the user re-approved a held PR",
    ),
    (
        r"HOLD_CLEARED ",
        ROUTINE,
        "merge.advance",
        "a held PR pushed or left; it is re-gated",
    ),
    (
        r"LANDED ",
        ROUTINE,
        "merge.advance",
        "the armed PR merged; develop sanity follows",
    ),
    (
        r"ANALYSIS_START ",
        ROUTINE,
        "merge.analyse_candidates",
        "a candidate's early gate job",
    ),
    (
        r"GATE_(START|SKIP|DONE|INCOMPLETE) ",
        ROUTINE,
        "merge.start_gate, merge.advance",
        "gate job progress (one cut short records nothing and is gated again)",
    ),
    (
        r"BATCH_(GATE_START|VALIDATED|FAILED|BLOCKED) ",
        ROUTINE,
        "merge.advance",
        "batch gate progress (a failure has a TRIAGE line of its own)",
    ),
    (
        r"CARRIED #",
        ROUTINE,
        "queue.carry_heads",
        "a rebase-only or merge-only push keeps the review or approval at its old head (G1)",
    ),
    # -- the develop guard and develop issues
    (r"SANITY", ROUTINE, "guard.advance", "develop sanity progress"),
    (r"BISECT_START ", ROUTINE, "guard.start_bisect", "a bisect started"),
    (r"BISECT_NEEDED ", ACT, "guard.advance", "several candidates: `rk bisect`"),
    (
        r"BISECT_DONE ",
        ACT,
        "guard.advance",
        "no culprit per merge: check for a flake or an interaction",
    ),
    (
        r"INCIDENT_CLOSED ",
        ACT,
        "guard.advance",
        "comment the fix and the green run on our develop issue",
    ),
    (
        r"INCIDENT_AGED ",
        ACT,
        "guard.advance",
        "no culprit to revert: the way out is the develop issue's fix",
    ),
    (
        r"INCIDENT_FIX_MERGED ",
        ACT,
        "guard.advance",
        "close the incident once develop is green after the fix",
    ),
    (
        r"DEVELOP_RED_STALE ",
        ROUTINE,
        "guard.advance",
        "a red from before an incident's merged fix",
    ),
    (
        r"BREAKAGE_OPEN ",
        ACT,
        "breakage.poll",
        "link it to our incident if it is the same break",
    ),
    (r"BREAKAGE_CLOSED ", ROUTINE, "breakage.poll", "a develop issue closed"),
    (
        r"FIX_CANDIDATE ",
        STANDING,
        "breakage.fix_sweep",
        "verify it, then `rk fix-link I P` or `rk fix-dismiss I P`",
    ),
    (
        r"FIX_LISTED ",
        ROUTINE,
        "breakage.listed_only",
        "fix links are off: a fix candidate, listed for the user only",
    ),
    # -- reviews
    (
        r"REVIEW_READY ",
        ACT,
        "review.advance",
        "verify and post it (review_procedure.md)",
    ),
    (
        r"REVIEW_MANUAL ",
        STANDING,
        "review.manual_lines",
        "too large for the autopilot: `rk review-start N`, then `rk review-split N` at its REVIEW_READY",
    ),
    (
        r"REVIEW_FAIL #\d+: its staging met another fetch into the shared clone",
        ROUTINE,
        "review.advance",
        "its staging met another git in the shared clone, not the PR's fault: the PR is not skipped, and its review "
        "starts again (from the review queue when it came from there)",
    ),
    (
        r"REVIEW_FAIL #\d+: .*(will retry|back in the review queue)",
        ROUTINE,
        "review.advance",
        "the worker is tried again, or the review starts again from the review queue",
    ),
    (
        r"REVIEW_FAIL ",
        ACT,
        "review.advance",
        "the PR is skipped: find out why, then `rk review-unskip N`",
    ),
    (
        r"RECHECK_DONE ",
        ACT,
        "review.finish_rechecks",
        "a draft's checks ran again: verify it",
    ),
    (r"RECHECK ", ROUTINE, "review.recheck", "a draft's checks run again"),
    (
        r"REVIEW_(START|WAIT|SETTLING|CLOSED|DROP|SKIP)\b",
        ROUTINE,
        "review.advance",
        "review progress",
    ),
    (r"REVIEW #", ROUTINE, "review.advance", "review progress"),
    (
        r"review #\d+: state head",
        ROUTINE,
        "review.take_live_head",
        "a review took the PR's live head",
    ),
    (
        r"#\d+ (changed substantively|is merge-only since our review)",
        ROUTINE,
        "review.start_review",
        "what a follow-up reviews",
    ),
    (
        r"#\d+: the (draft|review in flight)",
        ROUTINE,
        "review.close_review",
        "a review closed without a post",
    ),
    (
        r"MERGE_ONLY_REOPENED ",
        ROUTINE,
        "review.mark_merges",
        "a merge-only head's checks run",
    ),
    (
        r"CI_INFRA #",
        ROUTINE,
        "review.advance",
        "a merge-only head recorded as covered: its only flag was CI failures that are known infrastructure",
    ),
    (
        r"DRAFT_(START|WAIT) ",
        ROUTINE,
        "draft_stage.pool_advance",
        "a pool draft started, or waits until the limits on parallel work let it start",
    ),
    (
        r"DRAFT_DROP ",
        ROUTINE,
        "draft_stage.drop",
        "a pool draft stopped: its PR closed or moved on",
    ),
    (
        r"SHADOW_READY ",
        ACT,
        "draft_stage.shadow_collect",
        "a shadow draft is done for a review awaiting verification: give its path to that review's verifier, for the "
        "shadow's assessment lines once the post is final",
    ),
    (
        r"SHADOW_(START|WAIT|DONE|DROP) ",
        ROUTINE,
        "draft_stage.shadow_start",
        "the shadow drafter's progress",
    ),
    (
        r"FOLLOWUP_SCAN #\d+: 0 candidates",
        ROUTINE,
        "followups.tick",
        "nothing to follow up",
    ),
    (
        r"FOLLOWUP_SCAN ",
        ACT,
        "followups.tick",
        "dedup, judge and file the follow-up candidates",
    ),
    (
        r"READY_FOR_APPROVAL ",
        ACT,
        "rkmon.approval_news",
        "tell the user, one line per PR",
    ),
    (
        r"OVERLAP_NOTICE_DUE ",
        STANDING,
        "overlaps.notices_due",
        "draft, check and post the overlap notice",
    ),
    (
        r"DRIFT_FAIL ",
        ACT,
        "drift.collect",
        "check the cause, then tell the author: `rk notice N FILE --kind drift`",
    ),
    (r"DRIFT_START ", ROUTINE, "drift.start", "a drift check started"),
    (r"DRIFT ok ", ROUTINE, "drift.collect", "tests pass on today's develop"),
    (
        r"PRIORITY_REQUEST ",
        ACT,
        "priority.intake",
        "tell the user; it waits for their OK",
    ),
    (
        r"PRIORITY_CLEARED ",
        ROUTINE,
        "priority.prune",
        "a priority's PR merged or closed",
    ),
    (
        r"NEEDS_YOU_TEAMS ",
        ROUTINE,
        "needs_teams.send",
        "the Needs-you decisions went to the Teams channel, or wait for the gap between messages",
    ),
    # -- the remote GPU tests (remote-gpu-tests)
    (
        r"GPU_(DONE|FAIL) ",
        ACT,
        "gpu.tick",
        "`rk gpu-row N` (and for GPU_FAIL, grade and post the drafted notice): the skill's GPU lines",
    ),
    (
        r"GPU_(NOT_QUEUED|PENDING_LONG|NODE_BROKEN|UNREACHABLE)\b",
        STANDING,
        "gpu.tick",
        "tell the user (a node for the remote GPU tests' excluded nodes, which the line names, a job that waits "
        "long, a cluster that does not answer)",
    ),
    (
        r"GPU_(STAGING|QUEUED|RUNNING|CANCELLED|CLEANUP|WAIT)\b",
        ROUTINE,
        "gpu.tick",
        "a GPU job's progress; it never keeps a review waiting",
    ),
    # -- setup, upkeep and the optional model roles
    (
        r"FEATURE_SKIPPED ",
        STANDING,
        "features.tick_reports",
        "a feature is on but not set up: tell the user once, with the fix it names",
    ),
    (
        r"RETENTION ",
        ROUTINE,
        "retention.rotate",
        "an old log or state file moved aside",
    ),
    (
        r"WORKTREES #\d+: .* made again",
        ROUTINE,
        "review.ensure_merge_tree",
        "a review's worktrees, gone (a RAM disk emptied by a restart), made again",
    ),
    (
        r"WORKTREES #",
        ACT,
        "review.ensure_merge_tree",
        "a review's worktrees are gone and could not be made again: read workers/pr<N>/mkwt.log",
    ),
    (
        r"ROLE_STARTED ",
        ROUTINE,
        "roles.start",
        "an optional model role started on the tick's facts",
    ),
    (r"CLEANUP ", ROUTINE, "cleanup.sweep", "disk cleanup"),
    (
        r"SCALER ",
        ROUTINE,
        "scaler.adjust",
        "the parallel reviews or the parallel builds moved within their range",
    ),
    (
        r"RAMDISK ",
        ROUTINE,
        "ramdisk.sweep",
        "a finished review's worktrees and builds left the RAM disk",
    ),
    (
        r"WEEKLY REPORT ",
        ACT,
        "rkmon.maybe_weekly_report",
        "present the weekly report (weekly.md)",
    ),
    (
        r"LESSONS REVIEW ",
        ACT,
        "rkmon.maybe_weekly_report",
        "run the lessons review (weekly.md)",
    ),
)

_ROWS = [(re.compile(p), cls, producer, what) for p, cls, producer, what in KINDS]
BOARD = re.compile(r"BOARD \((changed|hourly)\)")
TIME = "time: "  # a full tick's last line
EXIT_WHY = {
    2: "a tool the tick needs (GitHub, git) failed",
    3: "the tick has critical items",
}  # `rk tick`'s codes


def classify(line: str) -> tuple[str, str] | None:
    """(class, what it means) for a tick line, from the first row of KINDS that matches it; None when none does."""
    for pat, cls, _producer, what in _ROWS:
        if pat.match(line):
            return cls, what
    return None


def needed(text: str, rc: int | None = None) -> list[tuple[str, str]]:
    """The lines of a tick's output (`text`) the orchestrator acts on, in order, each with its class (ACT for a line
    no row knows). `rc`: the tick's exit code; 0 is OK, 3 means critical plan items (their `!!` lines act), any other
    code is a failure."""
    out = []
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    board = False
    for line in lines:
        if board:
            if not line.startswith(TIME):
                continue
            board = False
        if not line.strip():
            continue
        if BOARD.match(line):
            board = True
            continue
        c = classify(line)
        if c is None or c[0] != ROUTINE:
            out.append((line, c[0] if c else ACT))
    if not any(ln.strip() for ln in lines):
        out.append(("TICK_EMPTY: the tick printed nothing", ACT))
    if rc not in (None, 0) and not (
        rc == 3 and any(ln.startswith("!! ") for ln, _ in out)
    ):
        out.insert(
            0,
            (
                f"TICK_EXIT {rc}: {EXIT_WHY.get(rc, 'the tick ended with an error')}",
                ACT,
            ),
        )
    return out


def actionable(
    text: str, rc: int | None = None, seen: pathlib.Path | None = None
) -> list[str]:
    """The lines of a tick's output the orchestrator must act on. With `seen`, a standing line that file already holds
    is left out, and a full tick's standing lines replace the file's: written whole, at the state's shared mode
    (worker/perms.py)."""
    lines = needed(text, rc)
    if seen is None:
        return [ln for ln, _ in lines]
    before = (
        {_standing_key(ln) for ln in seen.read_text(encoding="utf-8").splitlines()}
        if seen.exists()
        else set()
    )
    standing = [ln for ln, cls in lines if cls == STANDING]
    if any(ln.startswith(TIME) for ln in (text or "").splitlines()):
        whole_file.write(
            seen,
            "".join(f"{ln}\n" for ln in standing),
            paths.worker_module("perms").SHARED_FILE_MODE,
        )
    return [
        ln for ln, cls in lines if cls != STANDING or _standing_key(ln) not in before
    ]


# A standing line that differs only in the develop commit it was checked on (a PROPOSE re-validated after develop
# moved), or only in which heavy job a manual review start waits for (REVIEW_MANUAL), is the same ask, so it wakes
# the orchestrator once.
_DEVELOP_SHA = re.compile(r"develop [0-9a-f]{7,40}")
_MANUAL_WHEN = re.compile(
    r"start it by hand (once \S+ is past its first 10 minutes|now)"
)


def _standing_key(line: str) -> str:
    return _MANUAL_WHEN.sub(
        "start it by hand WHEN", _DEVELOP_SHA.sub("develop SHA", line)
    )


# The review driver's exit (tools/review_driver.py): REVIEW_DROP is routine in a tick, but when the driver prints it a
# verification of that PR may be running, whose lanes the orchestrator stops.
DRIVER_ACT = re.compile(r"REVIEW_DROP ")
READY = re.compile(r"(REVIEW_READY|RECHECK_DONE) ")


def driver_lines(text: str) -> list[str]:
    """`rk tick-filter --driver`: the lines of the review driver's last `rk review-advance` the orchestrator acts on,
    then a one-line summary of them, then the REVIEW_READY lines last, so the output's tail holds them all.
    """
    acts = [ln for ln, _ in needed(text)] + [
        ln.rstrip() for ln in (text or "").splitlines() if DRIVER_ACT.match(ln)
    ]
    acts = [ln for ln in acts if not ln.startswith("TICK_EMPTY")]
    if not acts:
        return []
    ready = [ln for ln in acts if READY.match(ln)]
    rest = [ln for ln in acts if not READY.match(ln)]
    heads = ", ".join(" ".join(ln.split()[:2]).rstrip(":") for ln in rest + ready)
    return (
        [
            f"DRIVER_EXIT: {len(acts)} line{'s' if len(acts) != 1 else ''} for the orchestrator: {heads[:300]}"
        ]
        + rest
        + ready
    )


def standing(seen: pathlib.Path) -> list[str]:
    """The standing lines already reported (`seen`): what the tick loop's heartbeat repeats."""
    return seen.read_text(encoding="utf-8").splitlines() if seen.exists() else []
