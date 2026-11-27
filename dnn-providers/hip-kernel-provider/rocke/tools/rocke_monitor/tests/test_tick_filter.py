# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk tick-filter` (rkmon_lib/tickfilter.py): which tick lines need the orchestrator, standing lines, the board, the
exit codes. The tick outputs here are made up in the tick's own format, with the words the tick prints today.
"""

import errno
import importlib
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import gh_cap, gpu, paths, review, tickfilter  # noqa: E402
from rkmon_lib.tickfilter import ACT, ROUTINE, STANDING  # noqa: E402

BOARD = """BOARD (changed): show it to the user
| Part | Now | Next / counts |
|---|---|---|
| Monitor | ON, phase queue | auto-merge only for PRs you clear: none cleared |

**Needs you:**
1. Merge #11 → #12: every gate is green on develop abcdef0123. Merge it, or clear it for auto-merge: `rkmon rk clear 11`, `rkmon rk clear 12`.

**Needs platform approval** (anyone in team-a can approve):
- #13 🟢 someone: a title · needs a code-owner approval"""

FULL = f"""(overlaps: develop: 1 blobs not local)
TICK 2026-11-06T18:20:03Z monitor ON reviews=live merge-queue=live auto-merge=hints_only
develop abcdef0123 CI pending; last green 0123456789 (ci)
EVENTS (2): PUSH #11; CI #12 SUCCESS
QUEUE: 1.#11(d=0)  2.#12(d=1)
ABI binary (develop 3): 1.#11->4
MERGE PLAN: first #11 #12; 2 waves
PLAN (3): P1 FOLLOW_UP #12 | P3 FIRST_ROUND #14 | P4 FOLLOW_UP #15 (gap: after 19:40Z)
PROPOSE #11 → #12: validated on develop abcdef0123; tell the user to merge in this order
REVIEW #14: worker running, checks none
CLEANUP #9 (merged): removed 3 dirs, 120 MB
{BOARD}
time: fingerprint 3s search 31s develop 20s"""

SAMPLES = {  # one line of each kind the tick prints, and its class
    "!! CRITICAL INCIDENT_OPEN INC-1: develop CI red at abcdef0123 (develop CI: unit)": ACT,
    "!! DISARM_NEEDED #5: its head moved. Run `rk disarm 5`": ACT,
    "!! AUTOMERGE_RISK #5 (auto-merge enabled by x): our latest review found problems. run `rk automerge-off 5`": ACT,
    "!! PRIORITY_READY_EARLY #5 (wave 2, …)": ACT,
    "!! FOLLOWUP_SCAN #5 failed 3 times (x): run `rk followup-scan 5` by hand": ACT,
    (
        "!! VERIFY_STALE #5: handed to verification 3.2 h ago, nothing posted (head abcdef0123); if our account posted "
        "at that head already, run `rk review-close 5`"
    ): ACT,
    "!! CRITICAL TOOL failure: gh api graphql: HTTP 502": ACT,
    "(priority requests not read: Broken pipe)": ROUTINE,
    "(followup scan #5 failed, try 1 of 3: x)": ROUTINE,
    "TICK idle: no change since 18:20Z (85 PRs, develop abcdef0123)": ROUTINE,
    (
        "GH_CAP_WAIT while this tick ran, 2 gh calls waited 12 minutes in all for the cap of 100 GitHub requests an "
        "hour (integrations.github_cli.max_requests_per_hour)"
    ): ROUTINE,
    (
        "gh: waiting until 17:40:00Z for the GitHub request cap (integrations.github_cli.max_requests_per_hour)"
    ): ROUTINE,
    "GH_CAP_OFF: "  # the counter's own words (gh_cap.unusable), as the tick prints them
    + gh_cap.unusable(
        pathlib.Path("/work/monitor/state"),
        PermissionError(errno.EACCES, "Permission denied"),
    ): STANDING,
    "TICK 2026-11-06T18:20:03Z monitor OFF: nothing done (paused until `rk resume`)": ACT,
    "TICK 2026-11-06T18:20:03Z another tick is running; skipped": ROUTINE,
    "HOLDS: #5(triage)": ROUTINE,
    "BATCHES: [#5 #6](candidate)": ROUTINE,
    "BOARD unchanged since 2026-11-06T18:00:00Z": ROUTINE,
    "PROPOSE #5: validated on develop abcdef0123; tell the user to merge in this order": STANDING,
    "ARM_READY #5: batch #5 validated on develop abcdef0123; run `rk arm 5`": STANDING,
    "QUEUE_NOTICE #5: next (abcdef0123); run `rk queue-notice 5`": STANDING,
    "TRIAGE #5: gate failed, held. Run the failing check on develop": ACT,
    "HOLD_STALE #5: held 25h without a push (G5). Ask the user": ACT,
    "HOLD_FIXED #5: someone pushed 0123456789 since the hold (G5).": ACT,
    "HOLD_REAPPROVED #5: you re-approved it": ROUTINE,
    "HOLD_CLEARED #5: pushed; it is re-gated": ROUTINE,
    "LANDED #5 (armed by us); develop sanity follows": ROUTINE,
    "ANALYSIS_START #5 (candidate, waited 0 min): GATE_START #5 against develop abcdef0123": ROUTINE,
    "GATE_DONE #5: G5 pass; G6 pass; G7 pass": ROUTINE,
    "BATCH_VALIDATED #5 → #6 on develop abcdef0123": ROUTINE,
    "SANITY_START abcdef0123: 1 rocKE landing(s) since 0123456789 (#5)": ROUTINE,
    "SANITY abcdef0123: running": ROUTINE,
    "SANITY_GREEN abcdef0123: abi, build": ROUTINE,
    "BISECT_NEEDED INC-1: 2 candidates [5, 6] (run `rk bisect`)": ACT,
    "BISECT_DONE INC-1: none (the failure may not reproduce per merge)": ACT,
    "INCIDENT_CLOSED INC-1: develop green again at abcdef0123": ACT,
    "INCIDENT_AGED INC-1: open 130 min with no culprit to revert": ACT,
    "INCIDENT_FIX_MERGED INC-1: #5 merged as abcdef0123: no revert": ACT,
    "DEVELOP_RED_STALE abcdef0123: its commit predates #5; not an incident": ROUTINE,
    "BREAKAGE_OPEN #5 red-develop (x): a test fails on develop": ACT,
    "BREAKAGE_CLOSED #5 red-develop: a test fails on develop": ROUTINE,
    "FIX_CANDIDATE #6 -> #5 (red-develop): it names test_x. Verify it, then `rk fix-link 5 6`": STANDING,
    "REVIEW_READY #5 (first-round): worker done, checks done. Inputs: …": ACT,
    "REVIEW_MANUAL #5 (follow-up, 40000 lines > 30000): too large for the autopilot": STANDING,
    "REVIEW_FAIL #5: worker failed(rc=1), no draft; will retry": ROUTINE,
    f"REVIEW_FAIL #5: worker failed(rc=1), no draft; {review.REQUEUED}": ROUTINE,
    (
        "REVIEW_FAIL #5: its staging met another fetch into the shared clone (shallow file changed); "
        + review.REQUEUED
    ): ROUTINE,
    "REVIEW_FAIL #5: its staging met another fetch into the shared clone (shallow file changed); it starts again, "
    "not skipped": ROUTINE,
    "REVIEW_FAIL #5: worker failed(rc=1) again, no draft; skipped until `rk review-unskip 5`": ACT,
    "RECHECK_DONE #5: checks done; see checks.md": ACT,
    "REVIEW_START #5 (follow-up): qwen worker started at abcdef0123": ROUTINE,
    "REVIEW_WAIT: the next worker starts once gate-pr5 is past its first 10 minutes": ROUTINE,
    "REVIEW_SKIP #5: rebase-only since our review; recorded abcdef0123 as equivalent": ROUTINE,
    "REVIEW_DROP #5: closed or head moved; will re-pick": ROUTINE,
    "review #5: state head 0123456789 was out of date; using live head abcdef0123": ROUTINE,
    "#5 changed substantively: 3 files, e.g. a.py": ROUTINE,
    "#5 is merge-only since our review (1 merge): nothing is posted if its checks come back clean": ROUTINE,
    "#5: the draft awaiting verification at abcdef0123 is closed without a post (merge-only)": ROUTINE,
    "MERGE_ONLY_REOPENED #5: abcdef0123 was recorded as covered before its checks ran; they run now": ROUTINE,
    "DRAFT_START #5 (follow-up): pool draft at abcdef0123": ROUTINE,
    "DRAFT_WAIT #5: the last 5 min used 85% of the budget (under 80% starts another)": ROUTINE,
    "DRAFT_DROP #5: its head moved": ROUTINE,
    "DRAFT_DROP #5: the live stage is local-worker: no pool draft feeds it": ROUTINE,
    "SHADOW_START #5 (follow-up): glm at abcdef0123": ROUTINE,
    "SHADOW_WAIT #5: 6 drafts running: the ceiling is 6": ROUTINE,
    "SHADOW_DONE #5: ready after our review (late)": ROUTINE,
    "SHADOW_DROP #5: its head moved": ROUTINE,
    "SHADOW_READY #5: /w/shadow_glm/pr5/review_agent.md (PASS: 9 claims); give it to the review's verifier": ACT,
    "REVIEW_WAIT #5: its draft waits: 2 drafts running: the ceiling is 2": ROUTINE,
    "FOLLOWUP_SCAN #5: 0 candidates (0 deferred, 0 unanswered, 0 ours): f.md": ROUTINE,
    "FOLLOWUP_SCAN #5: 2 candidates (1 deferred, 1 unanswered, 0 ours): f.md": ACT,
    "READY_FOR_APPROVAL #5 🟢 x: a title · needs a code-owner approval": ACT,
    "OVERLAP_NOTICE_DUE #5: key (owner #6): post the notice": STANDING,
    "DRIFT_FAIL #5 at develop abcdef0123: 2 failing": ACT,
    "DRIFT_START on develop abcdef0123: #5 #6": ROUTINE,
    "DRIFT ok on develop abcdef0123: #5": ROUTINE,
    "PRIORITY_REQUEST #5 by x: needed for a release": ACT,
    "PRIORITY_CLEARED #5: merged; it leaves the board and the dashboard": ROUTINE,
    "WEEKLY REPORT 2026-W45 ready: r.md (present it to the user)": ACT,
    "LESSONS REVIEW 2026-W45 due: l.md": ACT,
    "REVIEW_RESUME #5: the follow-up a stop interrupted starts again at abcdef0123": ROUTINE,
    "REVIEW_RESTORED #5: its review directory came back from the disk copy": ACT,
    "check your internet connection or the GitHub status page": ROUTINE,
    "DRAINING since 2026-11-11T10:00:00Z (rk drain): nothing new starts until it ends": ROUTINE,
    "!! DRAIN_CLEARED (rk drain, since 2026-11-11T10:00:00Z): the process that set it (5) is gone": ACT,
    "JOB_ORPHAN worker-pr5-20261111100000: started by a tick or command that never saved; killed": ROUTINE,
    "JOB_STRAY sanity: killed 2 stray process(es) of an earlier job still in its tree": ROUTINE,
    "GATE_INCOMPLETE #5: job timeout after build: nothing recorded; it is gated again": ROUTINE,
    "SANITY_INCOMPLETE abcdef0123: job lost (build only): not green; it runs again": ROUTINE,
    "!! SANITY_INCOMPLETE abcdef0123: job lost again (no result): not green, not run a third time": ACT,
    # Informational lines a rules-acted tick (no Claude) would otherwise keep for a person.
    "CARRIED #5: rebase-only from abcdef01 to 12345678: the review or approval there stands": ROUTINE,
    "RETENTION ledger: 12.0 MB moved to actions.1.jsonl": ROUTINE,
    "ROLE_STARTED planner: model-x on the facts 0a1b2c": ROUTINE,
    "WORKTREES #5: wt_v5 made again with mkwt.py (gone, e.g. the RAM disk emptied by a restart): the head merged": ROUTINE,
    "WORKTREES #5: wt_v5 gone, and mkwt.py did not make wt_v5 again (see workers/pr5/mkwt.log)": ACT,
    "GPU_STAGING #5 gfx942: 3 GPU-only case(s) staged": ROUTINE,
    "GPU_QUEUED #5 gfx942 at abcdef0123: 3 GPU-only case(s), job 77": ROUTINE,
    "GPU_RUNNING #5 gfx942: job 77 on node n1": ROUTINE,
    "GPU_WAIT #5: no node free; no GPU job is queued": ROUTINE,
    "GPU_DONE #5 gfx942 at abcdef0123 on node n1: passed": ACT,
    "GPU_FAIL #5 gfx942 at abcdef0123: 2 GPU-only case(s) fail": ACT,
    f"GPU_NODE_BROKEN n1 (gfx942): no GPU: add it to {gpu.key_name('exclude')}; "
    "the job goes again without it": STANDING,
    "FEATURE_SKIPPED develop break alerts: on but needs teams_webhook (fix: ./rkmon secret teams_webhook)": STANDING,
}


def run_cli(text: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", str(ROOT / "rkmon.py"), "tick-filter", *args],
        input=text,
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestTable(unittest.TestCase):
    def test_every_kind_the_tick_prints_has_its_class(self):
        for line, cls in SAMPLES.items():
            with self.subTest(line=line):
                self.assertEqual((tickfilter.classify(line) or ("unknown",))[0], cls)

    def test_a_staging_lost_to_the_clone_race_says_it_starts_again(self):
        """A staging that met another fetch into the shared clone is not the PR's fault, whichever way its line ends:
        routine, and its row says the PR is not skipped and its review starts again."""
        race = "REVIEW_FAIL #5: its staging met another fetch into the shared clone (shallow file changed); "
        for end in (review.REQUEUED, "it starts again, not skipped"):
            with self.subTest(end=end):
                cls, what = tickfilter.classify(race + end)
                self.assertEqual(cls, ROUTINE)
                self.assertTrue(what.startswith("its staging met another git"), what)
                self.assertIn("not skipped", what)

    def test_a_line_no_row_knows_needs_the_orchestrator(self):
        self.assertIsNone(tickfilter.classify("NEW_KIND #5: something"))
        self.assertEqual(
            tickfilter.actionable("TICK 2026 monitor ON\nNEW_KIND #5: x\n", 0),
            ["NEW_KIND #5: x"],
        )

    def test_every_producer_the_table_names_exists(self):
        for _, _, producer, _ in tickfilter.KINDS:
            for name in producer.split(", "):
                if name == "any step":
                    continue
                mod, func = name.rsplit(".", 1)
                module = importlib.import_module(
                    mod if mod == "rkmon" else f"rkmon_lib.{mod}"
                )
                self.assertTrue(hasattr(module, func), name)


class TestFilter(unittest.TestCase):
    def test_a_full_tick_keeps_its_actionable_lines_and_drops_the_board(self):
        self.assertEqual(
            tickfilter.actionable(FULL, 0),
            [
                "PROPOSE #11 → #12: validated on develop abcdef0123; tell the user to merge in this order"
            ],
        )
        inside = FULL.replace("time: ", "READY_FOR_APPROVAL #13 🟢 x: y\ntime: ")
        self.assertNotIn(
            "READY_FOR_APPROVAL #13 🟢 x: y", tickfilter.actionable(inside, 0)
        )  # the board's own line
        after = FULL + "\nREADY_FOR_APPROVAL #13 🟢 x: y"
        self.assertIn(
            "READY_FOR_APPROVAL #13 🟢 x: y", tickfilter.actionable(after, 0)
        )  # the board ends at time:

    def test_an_in_depth_review_waiting_for_approval_wakes_nothing(self):
        """Its line is on the board's Needs you list alone, which never wakes the orchestrator."""
        st = {"prs": {"14": {"size": 80000, "author": "someone"}}}
        line = review.approval_need({}, st, None, 14)
        waiting = FULL.replace(
            "\n\n**Needs platform", f"\n2. {line}\n\n**Needs platform"
        )
        self.assertIn(f"2. {line}", waiting)
        self.assertEqual(
            tickfilter.actionable(waiting, 0), tickfilter.actionable(FULL, 0)
        )

    def test_exit_codes_and_an_empty_output(self):
        self.assertEqual(
            tickfilter.actionable(
                "TICK idle: no change since 18:20Z (1 PRs, develop x)", 0
            ),
            [],
        )
        out = tickfilter.actionable(
            "!! CRITICAL TOOL failure: gh api graphql: HTTP 502", 2
        )
        self.assertEqual(
            out,
            [
                "TICK_EXIT 2: a tool the tick needs (GitHub, git) failed",
                "!! CRITICAL TOOL failure: gh api graphql: HTTP 502",
            ],
        )
        self.assertEqual(
            tickfilter.actionable("!! CRITICAL ALERT: develop red", 3),
            ["!! CRITICAL ALERT: develop red"],
        )
        self.assertEqual(
            tickfilter.actionable("TICK x monitor ON\ntime: a 1s", 3)[0],
            "TICK_EXIT 3: the tick has critical items",
        )
        self.assertEqual(
            tickfilter.actionable("Traceback (most recent call last):", 1)[0],
            "TICK_EXIT 1: the tick ended with an error",
        )
        self.assertEqual(
            tickfilter.actionable("", 0), ["TICK_EMPTY: the tick printed nothing"]
        )

    def test_a_standing_line_acts_once_while_it_stands(self):
        with tempfile.TemporaryDirectory() as d:
            seen = pathlib.Path(d) / "seen"
            propose = "PROPOSE #11 → #12: validated on develop abcdef0123; tell the user to merge in this order"
            self.assertEqual(tickfilter.actionable(FULL, 0, seen), [propose])
            self.assertEqual(tickfilter.standing(seen), [propose])
            self.assertEqual(
                tickfilter.actionable(FULL, 0, seen), []
            )  # the same proposal: quiet
            idle = "TICK idle: no change since 18:20Z (1 PRs, develop abcdef0123)"
            self.assertEqual(tickfilter.actionable(idle, 0, seen), [])
            self.assertEqual(
                tickfilter.standing(seen), [propose]
            )  # an idle tick leaves the file as it is
            failed = "TICK 2026-11-06T18:30:00Z monitor ON phase=queue\n!! CRITICAL TOOL failure: x"
            self.assertEqual(
                tickfilter.actionable(failed, 2, seen)[0][:11], "TICK_EXIT 2"
            )
            self.assertEqual(
                tickfilter.standing(seen), [propose]
            )  # so does a tick that failed
            gone = FULL.replace(propose + "\n", "")
            self.assertEqual(tickfilter.actionable(gone, 0, seen), [])
            self.assertEqual(tickfilter.standing(seen), [])  # it went away ...
            self.assertEqual(
                tickfilter.actionable(FULL, 0, seen), [propose]
            )  # ... and came back: it acts again
            other = FULL.replace("develop abcdef0123; tell", "develop 9999999999; tell")
            self.assertEqual(
                tickfilter.actionable(other, 0, seen), []
            )  # validated again on a new develop: the same ask
            new_order = other.replace("#11 → #12", "#11 → #13")
            self.assertEqual(
                len(tickfilter.actionable(new_order, 0, seen)), 1
            )  # a new order acts

    def test_the_seen_file_is_written_whole_at_the_states_shared_mode(self):
        """A full tick's standing lines replace the seen file whole, at the state's shared mode whatever the umask,
        with nothing left beside it."""
        umask = os.umask(0o077)
        self.addCleanup(os.umask, umask)
        with tempfile.TemporaryDirectory() as d:
            seen = pathlib.Path(d) / "seen"
            tickfilter.actionable(FULL, 0, seen)
            self.assertEqual(
                stat.S_IMODE(seen.stat().st_mode),
                paths.worker_module("perms").SHARED_FILE_MODE,
            )
            self.assertEqual([p.name for p in pathlib.Path(d).iterdir()], ["seen"])

    def test_a_manual_review_line_waiting_on_another_job_is_the_same_ask(self):
        with tempfile.TemporaryDirectory() as d:
            seen = pathlib.Path(d) / "seen"
            line = (
                "REVIEW_MANUAL #7 (follow-up, 40000 lines > 30000): too large for the autopilot; start it by hand "
                "once gate-pr8-1 is past its first 10 minutes with `rk review-start 7`"
            )
            tick = line + "\n" + tickfilter.TIME + "1s\n"
            self.assertEqual(tickfilter.actionable(tick, 0, seen), [line])
            other_job = tick.replace("gate-pr8-1", "sanity-abc-2")
            self.assertEqual(tickfilter.actionable(other_job, 0, seen), [])
            now = tick.replace("once gate-pr8-1 is past its first 10 minutes", "now")
            self.assertEqual(tickfilter.actionable(now, 0, seen), [])

    def test_the_command(self):
        r = run_cli(FULL, "--rc", "0")
        self.assertEqual((r.returncode, r.stdout.count("\n")), (0, 1))
        r = run_cli(
            "TICK idle: no change since 18:20Z (1 PRs, develop x)\n", "--rc", "0"
        )
        self.assertEqual((r.returncode, r.stdout), (1, ""))
        with tempfile.TemporaryDirectory() as d:
            seen = str(pathlib.Path(d) / "seen")
            self.assertEqual(run_cli(FULL, "--rc", "0", "--seen", seen).returncode, 0)
            self.assertEqual(run_cli(FULL, "--rc", "0", "--seen", seen).returncode, 1)


if __name__ == "__main__":
    unittest.main()
