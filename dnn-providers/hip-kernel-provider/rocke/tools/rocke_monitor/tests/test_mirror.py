# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A mirror (rkmon_lib/mirror.py: a monitor beside the live one that sends nothing): every outward write command is
refused before any request, one test each, and gh, a team chat message and a ticket's filing refuse below them; the
review driver starts reviews from the live monitor's feed alone, at each line's head, up to mirror.pairs; our last post
is held while a mirrored review runs; `rk post` writes the would-be post with its inputs and cost.json to the mirror's
folder. GitHub, jobs and the model are faked: no request leaves the test."""

import datetime as dt
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rkmon  # noqa: E402
from rkmon_lib import gh, mirror, outward, post, premises, review, tickets  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

H7, H8, H9, OLD = "7" * 40, "8" * 40, "9" * 40, "a" * 40
# Each command that sends (rkmon_lib/outward.py), post and spec-post aside, with arguments its parser takes ({f}: a
# file); the writer's sends with a send their gate weighs (an atlassian command that writes).
ARGV = {
    "notice": ["notice", "7", "{f}", "--kind", "queue", "--key", "k"],
    "arm": ["arm", "7"],
    "disarm": ["disarm", "7", "a", "reason"],
    "queue-notice": ["queue-notice", "7"],
    "automerge-off": ["automerge-off", "7"],
    "issue-file": ["issue-file", "{f}"],
    "issue-comment": ["issue-comment", "7", "{f}", "--key", "k"],
    "issue-relabel": ["issue-relabel", "7", "--to", "stability"],
    "issue-assign": ["issue-assign", "7", "dev-a"],
    "label-create": ["label-create", "x", "--description", "d", "--color", "ffffff"],
    "request-changes": ["request-changes", "7", "{f}"],
    "fix-link": ["fix-link", "1", "7"],
    "teams-breakage": ["teams-breakage", "7"],
    "teams-test": ["teams-test"],
    "needs-you-teams": ["needs-you-teams"],
    "gpu-row": ["gpu-row", "7"],
    "ticket-file": ["ticket-file", "T1"],
    "teams-send": ["teams-send", "--type", post.KIND_BREAK],
    "web-push": ["web-push", "{f}"],
    "atlassian": ["atlassian", "jira", "issue", "create", "--project", "TK"],
}
# The commands a mirror refuses: each that sends, but those it diverts to its folder.
REFUSED = [r.name for r in outward.TABLE if r.dest != outward.NONE and not r.drafts]


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def nothing_leaves():
    """Patches that fail the test if anything would leave it: gh, a subprocess of gh, a URL."""

    def boom(*a, **k):
        raise AssertionError(f"an outward call: {a[:2]}")

    stack = ExitStack()
    stack.enter_context(mock.patch.object(gh.subprocess, "run", boom))
    stack.enter_context(mock.patch("urllib.request.urlopen", boom))
    return stack


class Case(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = pathlib.Path(self.d.name)
        self.feed = self.root / "live_state" / "mirror_feed.jsonl"
        self.feed.parent.mkdir()
        self.cfg = {
            "repo": "ROCm/r",
            "operator": {"github_login": "dev-op"},
            "rocke_path": "rk",
            "deployment": {"phase": "review"},
            "merge_authority": "propose",
            "review_first_line": "Automated review: Model - qwen, effort: max",
            "settle_min": 30,
            "hot_spots": {},
            "review": {"pipes": 2},
            "paths": {
                "working_dir": str(self.root),
                "root": str(self.root),
                "workers": str(self.root / "workers"),
            },
            "mirror": {"enabled": True, "feed": str(self.feed), "pairs": 2},
        }
        self.store = Store(self.root / "state")
        self.st = empty_state()
        self.st["last_tick"] = ago(1)
        common = {
            "status": "open",
            "draft": False,
            "size": 10,
            "head_seen_at": ago(300),
            "head_committed_at": ago(300),
        }
        self.st["prs"] = {
            "7": dict(
                common, n=7, head=H7, ours={"head": OLD, "at": ago(200), "count": 1}
            ),
            "8": dict(common, n=8, head=H8, ours=None),
            "9": dict(common, n=9, head=H9, ours=None),
        }

    def lines(self, *recs):
        with open(self.feed, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


class TestWrites(Case):
    """One test per command that sends (rkmon_lib/outward.py): refused, and nothing leaves."""

    def test_every_write_command_has_its_test(self):
        # post and spec-post are diverted to the mirror folder, never refused (mirror.guard)
        self.assertEqual(set(ARGV), set(REFUSED))
        self.assertEqual(set(outward.names(drafts=True)), {"post", "spec-post"})

    def run_rk(self, argv):
        out = []
        f = self.root / "x.md"
        f.write_text("text\n")
        argv = [a.replace("{f}", str(f)) for a in argv]
        with nothing_leaves(), mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            code = rkmon.main(argv)
        return code, out

    def test_the_backstops_below_the_commands(self):
        gh.configure(self.cfg)
        try:
            with nothing_leaves(), self.assertRaises(gh.GhError) as cm:
                gh.write(["pr", "comment", "7", "--body-file", "x"])
            self.assertIn(
                "a mirror (config mirror.enabled) writes nothing to GitHub",
                str(cm.exception),
            )
        finally:
            gh.configure({})
        with nothing_leaves(), self.assertRaises(post.PostRefused) as cm:
            post.teams_post(self.cfg, {"text": "x"}, post.KIND_BREAK)
        self.assertEqual(
            str(cm.exception), mirror.REFUSAL.format(cmd=outward.TEAMS_SEND)
        )  # in the words of the writer's gate, which refuses rk teams-send
        with mock.patch.object(
            tickets, "find", lambda st, tid: {"id": tid, "status": "waiting"}
        ):
            self.assertEqual(
                tickets.approve_refusal(self.cfg, self.st, "T1"),
                "this monitor is a mirror (config mirror.enabled): nothing is filed; the proposed ticket waits",
            )
        self.assertIsNone(
            mirror.guard(dict(self.cfg, mirror={"enabled": False}), "arm")
        )
        self.assertIsNone(mirror.guard(self.cfg, "post"))  # diverted, not refused
        self.assertIsNone(mirror.guard(self.cfg, "spec-post"))


def _write_test(cmd: str):
    def test(self):
        self.store.save(self.st)
        code, out = self.run_rk(ARGV[cmd])
        self.assertEqual(code, 4, out)
        self.assertEqual(out, [f"REFUSED: {mirror.REFUSAL.format(cmd=cmd)}"])

    test.__doc__ = f"rk {cmd} is refused in a mirror before any request."
    return test


for _cmd in REFUSED:
    setattr(TestWrites, f"test_{_cmd.replace('-', '_')}_is_refused", _write_test(_cmd))


class TestPicks(Case):
    def test_feed_lines_since_the_start_oldest_first_within_the_pairs(self):
        self.lines(
            {
                "pr": 9,
                "head": H9,
                "round": "fr",
                "kind": "first-round",
                "at": "2000-01-01T00:00:00Z",
            }
        )
        m = mirror.state(self.st)
        m["since"] = ago(60)
        self.lines(
            {
                "pr": 8,
                "head": H8,
                "round": "fr",
                "kind": "first-round",
                "again": False,
                "at": ago(30),
            },
            {"pr": 7, "head": OLD, "round": "fu1", "kind": "follow-up", "at": ago(20)},
            {"pr": 7, "head": H7, "round": "fu1", "kind": "follow-up", "at": ago(10)},
            "not json",
        )
        got = mirror.items(self.cfg, self.st, self.store)
        self.assertEqual(
            [(i["pr"], i["head"][:1], i["rkind"], i["round"]) for i in got],
            [(8, "8", "first-round", "fr"), (7, "a", "follow-up", "fu1")],
        )  # pairs 2: two at most
        m["done"]["8:" + H8] = {}
        m["started"]["7:" + OLD] = {"pr": 7, "head": OLD, "at": ago(0)}
        self.st["review_pipes"] = [{"pr": 7, "head": OLD}]
        self.assertEqual(
            mirror.items(self.cfg, self.st, self.store), []
        )  # 1 done + 1 started: the pairs are taken
        self.st["mirror"]["started"] = {}
        self.assertEqual(
            mirror.items(self.cfg, self.st, self.store), []
        )  # #7 is in a pipe
        self.st["review_pipes"] = []
        self.assertEqual(
            [i["head"][:1] for i in mirror.items(self.cfg, self.st, self.store)], ["a"]
        )
        old = dict(self.cfg, mirror=dict(self.cfg["mirror"], max_age_min=5))
        self.assertEqual(mirror.items(old, self.st, self.store), [])
        self.assertIn("over 5 min old", m["dropped"]["7:" + H7])

    def test_an_in_depth_review_waits_here_for_its_approval(self):
        """A feed line of an in-depth review, a first round of a PR over autopilot_max_lines (its first review, or one
        forced on a reviewed PR), waits here for the operator's approval, as on the driver's own path; the lines after
        it go on."""
        mirror.state(self.st)["since"] = ago(60)
        for n in (7, 8):
            self.st["prs"][str(n)]["size"] = 40000
        self.lines(
            {"pr": 8, "head": H8, "round": "fr", "kind": "first-round", "at": ago(30)},
            {"pr": 7, "head": H7, "round": "fr", "kind": "first-round", "at": ago(20)},
            {"pr": 9, "head": H9, "round": "fr", "kind": "first-round", "at": ago(10)},
        )
        self.assertEqual(
            [i["pr"] for i in mirror.items(self.cfg, self.st, self.store)], [9]
        )
        self.st[review.APPROVED] = [review.approval(8), review.approval(7)]
        self.assertEqual(
            [i["pr"] for i in mirror.items(self.cfg, self.st, self.store)], [8, 7]
        )  # pairs 2: two at most

    def test_a_line_starts_at_its_head_or_is_dropped(self):
        live = {
            7: {"state": "open", "head": H7},
            8: {"state": "open", "head": "c" * 40},
            9: {"state": "closed", "head": H9},
        }
        calls = []

        def start_review(cfg, st, store, n, kind, force=False, again=False):
            calls.append((n, kind, force, again))
            return [f"REVIEW_START #{n} ({kind}): draft started"]

        item = {
            "kind": mirror.KIND,
            "round": "fr",
            "rkind": "first-round",
            "again": False,
            "at": ago(5),
        }
        with mock.patch.object(
            gh,
            "rest",
            lambda path, jq=None, **k: dict(live[int(path.rsplit("/", 1)[1])]),
        ), mock.patch.object(review, "start_review", start_review):
            self.assertEqual(
                mirror.start(self.cfg, self.st, self.store, dict(item, pr=8, head=H8)),
                [
                    "MIRROR_DROP #8: the live monitor reviewed 8888888888; GitHub has cccccccccc: not "
                    "mirrored"
                ],
            )
            self.assertEqual(
                mirror.start(self.cfg, self.st, self.store, dict(item, pr=9, head=H9)),
                ["MIRROR_DROP #9: it is closed on GitHub: not mirrored"],
            )
            # The live monitor posted at this head already (our login): a re-review, past the settle.
            self.st["prs"]["7"]["ours"] = {"head": H7, "at": ago(1)}
            lines = mirror.start(
                self.cfg,
                self.st,
                self.store,
                dict(item, pr=7, head=H7, rkind="follow-up"),
            )
        self.assertEqual(lines, ["REVIEW_START #7 (follow-up): draft started"])
        self.assertEqual(calls, [(7, "follow-up", True, True)])
        m = mirror.state(self.st)
        self.assertEqual(set(m["dropped"]), {"8:" + H8, "9:" + H9})
        self.assertEqual(m["started"]["7:" + H7]["kind"], "follow-up")

    def test_the_driver_takes_the_feed_alone_and_runs_no_pool(self):
        self.lines(
            {"pr": 8, "head": H8, "round": "fr", "kind": "first-round", "at": ago(1)}
        )
        mirror.state(self.st)["since"] = ago(60)
        plan = [
            {
                "prio": 1,
                "kind": "FIRST_ROUND",
                "pr": 9,
                "text": "first-round review of #9",
            }
        ]
        started = []
        with mock.patch.object(
            mirror,
            "start",
            lambda cfg, st, store, item: started.append(item["pr"])
            or [f"REVIEW_START #{item['pr']} (first-round): draft started"],
        ), mock.patch.object(
            review.draft_stage, "pool_advance", side_effect=AssertionError("no pool")
        ):
            lines = review.advance(self.cfg, self.st, self.store, plan)
        self.assertEqual(
            (started, lines), ([8], ["REVIEW_START #8 (first-round): draft started"])
        )

    def test_mirror_done_once_the_pairs_are_written(self):
        m = mirror.state(self.st)
        m["done"] = {"7:" + H7: {}, "8:" + H8: {}}
        self.assertEqual(
            review._fill(self.cfg, self.st, self.store, [])[0][:46],
            "MIRROR_DONE: 2 of 2 reviews written: no more f",
        )
        self.assertEqual(
            review._fill(self.cfg, self.st, self.store, []), []
        )  # said once


class TestLiveChecks(Case):
    """mirror.checks: a mirrored review's checks are the live review's for the same head, copied once whole."""

    def test_the_driver_starts_the_copy_not_a_build(self):
        import subprocess

        live = self.root / "live_workers"
        cfg = dict(
            self.cfg,
            roles={"drafter": "cand"},
            model_bindings={"cand": {"url": "http://127.0.0.1:9", "model": "Cand-1"}},
            mirror=dict(self.cfg["mirror"], checks=str(live), checks_wait_min=0),
        )
        ws = self.root / "workers" / "pr8"
        ws.mkdir(parents=True)
        (ws / "staged").write_text("now\n")
        self.st["review_pipes"] = [
            {
                "pr": 8,
                "kind": "first-round",
                "stage": "worker",
                "head": H8,
                "since": None,
                "started": ago(5),
                "worker": {"id": "d8", "started": ago(5)},
                "checks": None,
                "stage_name": "tool-loop",
            }
        ]
        started = []

        def start(store, kind, name, script, timeout_min, env=None):
            started.append((kind, script, timeout_min))
            return {"id": f"{kind}-{name}", "started": ago(0), "pid": 0}

        with mock.patch.object(review.jobs, "start", start), mock.patch.object(
            review.jobs, "status", lambda store, job: "running" if job else "none"
        ), mock.patch.object(review, "_changed_files", lambda cfg, rec: ["rk/x.py"]):
            lines = review._advance_pipe(
                cfg, self.st, self.store, self.st["review_pipes"][0]
            )
        self.assertEqual(
            lines,
            [
                f"REVIEW #8: worker running; checks: the live review's for {H8[:10]}, copied once "
                "whole (checks-pr8)"
            ],
        )
        kind, script, timeout = started[0]
        self.assertEqual((kind, timeout), ("checks", 5))
        self.assertEqual(
            review.jobs.heavy_holders(self.st), []
        )  # no build: no heavy slot

        def run():
            return subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=60,
                env={"PATH": "/usr/bin:/bin", "RKMON_MIRROR_POLL_S": "0"},
            )

        p = run()
        self.assertEqual(
            (p.returncode, p.stdout.strip()),
            (1, f"the live review's checks for {H8[:10]} never came (0 min)"),
        )
        src = live / "pr8" / "checks"
        src.mkdir(parents=True)
        (src / "status").write_text("merge pass clean\n")
        (src / "checks.md").write_text(
            f"# Checks for #8 at {H8[:10]} (first-round, review), generated by rkmon t\n\n"
            "## Merge with develop\n"
        )
        self.assertEqual(
            run().returncode, 1
        )  # not whole yet: the last section is missing
        (src / "checks.md").write_text(
            f"# Checks for #8 at {'c' * 10} (follow-up, review), generated by rkmon t\n\n"
            "## Merge with develop\n\n## Auto-merge (an armed squash ...)\n"
        )
        self.assertEqual(run().returncode, 1)  # another head's
        whole = (
            f"# Checks for #8 at {H8[:10]} (first-round, review), generated by rkmon t\n\n"
            "## Merge with develop\n\n"
            "## Auto-merge (an armed squash ...)\n"
        )
        (src / "checks.md").write_text(whole)
        p = run()
        self.assertEqual(
            (p.returncode, p.stdout.strip()),
            (0, f"the live review's checks for {H8[:10]} copied"),
        )
        self.assertEqual((ws / "checks" / "checks.md").read_text(), whole)
        self.assertEqual((ws / "checks" / "status").read_text(), "merge pass clean\n")


class TestBlind(Case):
    def test_our_last_post_is_held_while_a_mirrored_review_runs(self):
        m = mirror.state(self.st)
        m["started"]["7:" + H7] = {
            "pr": 7,
            "head": H7,
            "feed_at": ago(30),
            "at": ago(29),
        }
        self.assertTrue(
            mirror.hold_ours(self.cfg, self.st, 7, {"at": ago(5)})
        )  # the live monitor's post of the round
        self.assertFalse(mirror.hold_ours(self.cfg, self.st, 7, {"at": ago(60)}))
        self.assertFalse(mirror.hold_ours(self.cfg, self.st, 8, {"at": ago(5)}))
        self.assertFalse(
            mirror.hold_ours(
                dict(self.cfg, mirror={"enabled": False}), self.st, 7, {"at": ago(5)}
            )
        )
        self.st["prs"]["7"]["ours"] = {"id": "old", "head": OLD, "at": ago(200)}
        with mock.patch.object(
            rkmon.snapshot,
            "our_latest_review",
            lambda cfg, n: {"id": "new", "at": ago(5)},
        ):
            rkmon.refresh_ours(self.cfg, self.store, self.st, {"7"})
        self.assertEqual(self.st["prs"]["7"]["ours"]["id"], "old")

    def test_the_premises_leave_our_login_out_and_ready_says_why(self):
        self.assertEqual(
            mirror.not_ours(self.cfg), ' | select(.user.login != "dev-op")'
        )
        self.assertEqual(mirror.not_ours(dict(self.cfg, mirror={})), "")
        mirror.state(self.st)["started"]["7:" + H7] = {
            "pr": 7,
            "head": H7,
            "feed_at": "2026-11-07T01:00:00Z",
        }
        note = review.mirror_note(self.cfg, self.st, {"pr": 7, "head": H7})
        self.assertIn(
            "A mirror's review, arm A: run the code quality pass as usual; nothing is posted: `rk post` "
            "writes this review to ",
            note,
        )
        self.assertIn(
            "no post or comment of ours on the PR newer than 2026-11-07T01:00:00Z", note
        )
        seen = []
        with mock.patch.object(
            premises,
            "_newer",
            lambda slug, path, jq, since: seen.append((path, jq)) or 1,
        ):
            self.assertEqual(
                premises.new_activity(self.cfg, "ROCm/r", 7, "t"), (1, 1, 1)
            )
            premises.new_activity(dict(self.cfg, mirror={}), "ROCm/r", 7, "t")
        self.assertEqual(
            seen,
            [
                (
                    "pulls/7/reviews",
                    '[.[] | select(.user.login != "dev-op") | .submitted_at]',
                ),
                (
                    "pulls/7/comments",
                    '[.[] | select(.user.login != "dev-op") | .updated_at]',
                ),
                (
                    "issues/7/comments",
                    '[.[] | select(.user.type != "Bot") | select(.user.login != "dev-op") | '
                    ".updated_at]",
                ),
                ("pulls/7/reviews", "[.[] | .submitted_at]"),
                ("pulls/7/comments", "[.[] | .updated_at]"),
                (
                    "issues/7/comments",
                    '[.[] | select(.user.type != "Bot") | .updated_at]',
                ),
            ],
        )


class TestArmsAndDrops(Case):
    def test_arms_alternate_until_arm_a_has_its_reviews(self):
        cfg = dict(
            self.cfg,
            roles={"quality": "cand"},
            model_bindings={"cand": {"url": "http://127.0.0.1:9", "model": "Cand-1"}},
            mirror=dict(self.cfg["mirror"], arm_a=2),
        )
        m, arms = mirror.state(self.st), []
        for k in range(5):
            arm = mirror.arm_for(cfg, self.st)
            arms.append(arm)
            m["done"][f"{k}:{H7}"] = {"arm": arm, "dir": "x"}
        self.assertEqual(arms, ["A", "B", "A", "B", "B"])
        self.assertEqual(
            mirror.arm_for(self.cfg, self.st), "A"
        )  # no quality binding: the Claude lane, always
        m["started"]["8:" + H8] = {"pr": 8, "head": H8, "arm": "A"}
        self.assertEqual(
            (mirror.arm_of(cfg, self.st, 8, H8), mirror.arm_of(cfg, self.st, 8, H7)),
            ("A", ""),
        )
        pipe = {"pr": 8, "head": H8, "stage_name": "tool-loop"}
        self.assertEqual(
            review.quality_note(cfg, self.st, self.store, pipe), ""
        )  # arm A: the Claude lane
        self.assertEqual(review.quality_start(cfg, self.st, self.store, pipe), [])

    def test_arm_b_s_note_names_the_binding_s_pass_only_when_it_runs(self):
        """A merge-only head and a review too large for the autopilot get no pass on the binding (review.quality_runs):
        their arm B note does not say it runs."""
        cfg = dict(
            self.cfg,
            roles={"quality": "cand"},
            model_bindings={"cand": {"url": "http://127.0.0.1:9", "model": "Cand-1"}},
        )
        mirror.state(self.st)["started"]["8:" + H8] = {"pr": 8, "head": H8, "arm": "B"}
        pipe = {"pr": 8, "head": H8, "stage_name": "tool-loop"}
        self.assertTrue(
            review.mirror_note(cfg, self.st, pipe).startswith(
                " A mirror's review, arm B: the quality binding runs the code quality pass; nothing is posted: "
            )
        )
        for case, over, size in (
            ("a merge-only head", {"merge_only": {"old": OLD, "merges": 1}}, 10),
            ("over the autopilot's limit", {}, 40000),
        ):
            with self.subTest(case):
                self.st["prs"]["8"]["size"] = size
                note = review.mirror_note(cfg, self.st, dict(pipe, **over))
                self.assertTrue(
                    note.startswith(" A mirror's review, arm B; nothing is posted: "),
                    note,
                )

    def test_every_drop_is_logged_beside_the_reviews(self):
        m = mirror.state(self.st)
        m["started"]["8:" + H8] = {
            "pr": 8,
            "head": H8,
            "round": "fr",
            "arm": "B",
            "at": ago(10),
        }
        mirror.tidy(
            self.cfg, self.st, self.store
        )  # no pipe, nothing awaiting verification: it ended
        rec = json.loads(
            (self.root / "mirror" / mirror.DROPS).read_text().splitlines()[-1]
        )
        self.assertEqual(
            (rec["pr"], rec["head"], rec["round"], rec["arm"]), (8, H8, "fr", "B")
        )
        self.assertIn("ended without a post", rec["why"])
        self.assertEqual(list(m["dropped"]), ["8:" + H8])

    def test_a_written_review_whose_live_review_was_dropped_is_dropped_too(self):
        m = mirror.state(self.st)
        self.feed.with_name("actions.jsonl").write_text(
            json.dumps(
                {
                    "ts": ago(5),
                    "kind": "post",
                    "pr": 8,
                    "head": H8,
                    "ref": "u8",
                    "note": "first",
                }
            )
            + "\n"
        )
        m["done"]["8:" + H8] = {
            "pr": 8,
            "head": H8,
            "round": "fr",
            "arm": "A",
            "dir": "d8",
        }
        m["done"]["9:" + H9] = {
            "pr": 9,
            "head": H9,
            "round": "fr",
            "arm": "B",
            "dir": "d9",
        }
        fed = ago(90)
        m["done"]["7:" + OLD] = {
            "pr": 7,
            "head": OLD,
            "round": "fu1",
            "arm": "A",
            "dir": "d7",
            "feed_at": fed,
        }
        m["done_said"] = ago(1)
        mirror.tidy(self.cfg, self.st, self.store)
        self.assertTrue(
            m["done"]["8:" + H8]["paired"]
        )  # the live post at that head is in the live ledger
        self.assertNotIn(
            "paired", m["done"]["9:" + H9]
        )  # no live post yet, the head unmoved: it waits
        self.assertNotIn(
            "7:" + OLD, m["done"]
        )  # #7 moved to H7 with no live post at OLD: no pair will come
        self.assertTrue(
            m["dropped"]["7:" + OLD].startswith(
                f"the live review of {OLD[:10]} was dropped"
            )
        )
        rec = json.loads(
            (self.root / "mirror" / mirror.DROPS).read_text().splitlines()[-1]
        )
        self.assertEqual(
            (rec["pr"], rec["head"], rec["arm"], rec["dir"], rec["feed_at"]),
            (7, OLD, "A", "d7", fed),
        )
        self.assertNotIn(
            "done_said", m
        )  # below mirror.pairs again: the next feed line takes its place
        self.assertIn(
            "2 of 2 reviews written (1 with the live post; arm A 1, arm B 1)",
            mirror.status_lines(self.cfg, self.st)[0],
        )

    def test_our_last_post_is_the_live_side_s(self):
        from rkmon_lib import facts, prep

        ledger = self.feed.with_name("actions.jsonl")
        ledger.write_text(
            "".join(
                json.dumps(r) + "\n"
                for r in (
                    {
                        "ts": ago(300),
                        "kind": "post",
                        "pr": 7,
                        "head": OLD,
                        "ref": "u1",
                        "note": "followup",
                    },
                    {
                        "ts": ago(5),
                        "kind": "post",
                        "pr": 7,
                        "head": H7,
                        "ref": "u2",
                        "note": "followup",
                    },  # this round's
                    {"ts": ago(200), "kind": "post", "pr": 8, "head": H8, "ref": "u3"},
                )
            )
        )
        mirror.state(self.st)["started"]["7:" + H7] = {
            "pr": 7,
            "head": H7,
            "feed_at": ago(30),
            "at": ago(29),
        }
        self.assertEqual(
            [r["ref"] for r in mirror.live_posts(self.cfg, self.st, 7)], ["u1"]
        )
        self.assertIsNone(mirror.live_posts(dict(self.cfg, mirror={}), self.st, 7))
        self.assertEqual(len(prep._our_posts(self.cfg, self.st, self.store, 7)), 1)
        self.st["prs"]["7"]["ours"] = {
            "head": OLD,
            "url": "https://github.com/ROCm/r/pull/7#issuecomment-1",
        }
        body = (
            "Automated review: Model - qwen, effort: max\n\n## PR #7 — t (dev-a) — 🟠 Follow-up: one open\n\n"
            "| # | Sev | file:line | Finding |\n|---|---|---|---|\n| N1 | 🟠 | `x.py:2` | **off by one** |\n"
        )
        with mock.patch.object(post, "our_post_body", lambda cfg, ours: body):
            got = facts._last_findings(self.cfg, self.st, self.store, 7, {})
            ours = facts._ours(
                self.cfg,
                self.st,
                self.store,
                7,
                {"posts": mirror.live_posts(self.cfg, self.st, 7)},
            )
        self.assertEqual(
            got,
            [
                "from GitHub (the live monitor's post, under our login)",
                "## PR #7 — t (dev-a) — 🟠 Follow-up: one open",
                "N1 🟠 off by one",
            ],
        )
        self.assertTrue(
            ours[-1].startswith(
                "our last post: the live monitor's, under our login: https://"
            ),
            ours,
        )


class TestDivert(Case):
    def setUp(self):
        super().setUp()
        # Arm B, with the quality binding: the review gets its code quality pass (review.quality_runs).
        self.cfg = dict(
            self.cfg,
            roles={"quality": "glm"},
            model_bindings={"glm": {"url": "http://127.0.0.1:9", "model": "M-1"}},
        )
        ws = self.root / "workers" / "pr7"
        (ws / "quality").mkdir(parents=True)
        (ws / "review_agent.md").write_text("the draft\n")
        (ws / "draft_run.json").write_text(
            json.dumps(
                {
                    "head": H7,
                    "binding": "glm",
                    "model": "M-1",
                    "method": "phased",
                    "draft": True,
                    "passes": {
                        "hunt": {
                            "status": "ok",
                            "wall_s": 900.0,
                            "requests": 20,
                            "tokens": 150000,
                        }
                    },
                }
            )
        )
        (ws / "tokens_total.json").write_text(
            json.dumps(
                {
                    "total": {
                        "requests": 30,
                        "tokens": 210000,
                        "cached_tokens": 50000,
                        "wall_s": 1500,
                    }
                }
            )
        )
        (ws / "quality" / "quality.md").write_text(
            "| Code quality | 🟢 Clean<br>• x (read) |\n"
        )
        (ws / "quality" / "quality_run.json").write_text(
            json.dumps(
                {
                    "head": H7,
                    "binding": "glm",
                    "model": "M-1",
                    "findings": 0,
                    "light": "🟢",
                    "passes": {
                        "quality": {
                            "status": "ok",
                            "requests": 9,
                            "tokens": 80000,
                            "wall_s": 600.0,
                        }
                    },
                }
            )
        )
        notes = self.root / "notes.md"
        notes.write_text("glm G1 match N1\n")
        self.spec = {
            "N": 7,
            "ROUND": "fu1",
            "NOTES": str(notes),
            "SRC": str(self.root / "post.md"),
        }
        m = mirror.state(self.st)
        m["started"]["7:" + H7] = {
            "pr": 7,
            "head": H7,
            "round": "fu2",
            "kind": "follow-up",
            "arm": "B",
            "feed_at": ago(70),
            "at": ago(60),
        }
        self.st["review_ready"] = [
            {
                "pr": 7,
                "kind": "follow-up",
                "head": H7,
                "at": ago(20),
                "started": ago(60),
                "jobs": {"checks": {"id": "c7", "started": ago(55)}},
                "quality_on_binding": True,
            }
        ]
        (ws / "quality" / "tokens_total.json").write_text(
            json.dumps(
                {
                    "total": {
                        "requests": 9,
                        "tokens": 80000,
                        "prompt_tokens": 70000,
                        "completion_tokens": 10000,
                        "reasoning_tokens": 6000,
                    }
                }
            )
        )
        # The container's transcripts (CLAUDE_CONFIG_DIR): the monitor session and the verifier's run for #7.
        claude = self.root / "claude"
        proj = claude / "projects" / "work"

        def call(mid, t, read, out=1, effort="high"):
            return {
                "type": "assistant",
                "timestamp": t,
                "perTurnEffort": effort,
                "message": {
                    "id": mid,
                    "usage": {
                        "input_tokens": 1,
                        "cache_read_input_tokens": read,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": out,
                    },
                },
            }

        def write(path, recs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in recs))

        write(
            proj / "s.jsonl",
            [
                call("o0", ago(90), 5),
                call("o1", ago(30), 100),
                call("o2", ago(10), 200),
            ],
        )
        sub = proj / "s" / "subagents"
        write(
            sub / "agent-v1.jsonl",
            [
                {
                    "type": "user",
                    "timestamp": ago(19),
                    "message": {"content": "Verify PR #7 follow-up 2"},
                },
                call("v1", ago(19), 4000, 10),
                call("v2", ago(6), 6000, 10),
            ],
        )
        (sub / "agent-v1.meta.json").write_text(
            json.dumps(
                {"agentType": "rocke-review-verifier", "description": "Verify #7 fu2"}
            )
        )
        write(
            sub / "agent-v0.jsonl", [call("v0", ago(95), 9)]
        )  # an earlier round's run: before the start
        (sub / "agent-v0.meta.json").write_text(
            json.dumps(
                {"agentType": "rocke-review-verifier", "description": "Verify #7 fu1"}
            )
        )
        env = mock.patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(claude)})
        env.start()
        self.addCleanup(env.stop)

    def test_the_post_its_inputs_and_cost_json(self):
        with nothing_leaves(), mock.patch.object(
            post, "preflight", lambda *a, **k: None
        ):
            lines = mirror.divert(
                self.cfg,
                self.st,
                self.store,
                7,
                {"text": "the post\n", "sha": H7},
                self.spec,
            )
        out = self.root / "mirror" / f"pr7_fu2_{H7[:10]}"
        self.assertEqual(
            lines[0],
            f"MIRRORED #7: the would-be post, its notes, draft, code quality pass and cost.json in "
            f"{out} (a mirror, arm B: nothing posted); 1 of 2 reviews written",
        )
        self.assertEqual((out / "post.md").read_text(), "the post\n")
        self.assertEqual((out / "notes.md").read_text(), "glm G1 match N1\n")
        self.assertEqual(json.loads((out / "spec.json").read_text())["ROUND"], "fu1")
        self.assertEqual((out / "draft" / "review_agent.md").read_text(), "the draft\n")
        self.assertTrue((out / "quality" / "quality.md").is_file())
        c = json.loads((out / "cost.json").read_text())
        self.assertEqual(
            (c["pr"], c["head"], c["round"], c["kind"], c["arm"]),
            (7, H7, "fu2", "follow-up", "B"),
        )
        self.assertIs(c["quality_on_binding"], True)
        self.assertEqual(
            (
                c["draft"]["by"],
                c["draft"]["tokens"],
                c["draft"]["requests"],
                c["draft"]["wall_min"],
            ),
            ("glm", 210000, 30, 25.0),
        )
        self.assertEqual(c["draft"]["phases"]["hunt"]["tokens"], 150000)
        q = c["quality"]
        self.assertEqual(
            (
                q["by"],
                q["tokens"],
                q["prompt_tokens"],
                q["completion_tokens"],
                q["reasoning_tokens"],
                q["requests"],
                q["wall_min"],
            ),
            ("glm", 80000, 70000, 10000, 6000, 9, 10.0),
        )
        v = c["verifier"]
        self.assertEqual(
            (v["by"], v["runs"], v["processed"], v["output"], v["effort"]),
            ("claude", 1, 10002, 20, "high"),
        )
        self.assertEqual(v["ieq"], round(2 + 10000 * 0.05 + 20 * 5))
        self.assertEqual(
            v["usd"], round((2 * 4 + 10000 * 0.2 + 20 * 20) / 1e6, 4)
        )  # Opus 5.5's prices
        self.assertIsNone(c["notice"])
        self.assertEqual(
            (c["claude"]["session"]["calls"], c["claude"]["session"]["processed"]),
            (2, 302),
        )
        self.assertEqual(c["claude"]["session_overlap"], 0)
        self.assertGreaterEqual(c["minutes"]["review"], 59.0)
        self.assertGreaterEqual(c["minutes"]["verify"], 19.0)
        self.assertEqual(c["minutes"]["draft"], 25.0)
        m = mirror.state(self.st)
        self.assertEqual((m["started"], list(m["done"])), ({}, ["7:" + H7]))
        self.assertEqual(self.st["review_ready"], [])

    def test_a_review_with_no_pass_on_the_binding_writes_none_of_it(self):
        """A review that gets no code quality pass on the binding by design (here one too large for the autopilot, as
        its REVIEW_READY found): the quality folder holds an earlier review's pass, which is neither copied nor counted.
        """
        self.st["prs"]["7"]["size"] = 40000
        self.st["review_ready"][0]["quality_on_binding"] = False
        with nothing_leaves(), mock.patch.object(
            post, "preflight", lambda *a, **k: None
        ):
            lines = mirror.divert(
                self.cfg,
                self.st,
                self.store,
                7,
                {"text": "the post\n", "sha": H7},
                self.spec,
            )
        out = self.root / "mirror" / f"pr7_fu2_{H7[:10]}"
        self.assertEqual(
            lines[0],
            f"MIRRORED #7: the would-be post, its notes, draft and cost.json in {out} (a mirror, arm B: nothing "
            "posted); 1 of 2 reviews written",
        )
        self.assertFalse((out / "quality").exists())
        c = json.loads((out / "cost.json").read_text())
        self.assertEqual(
            (c["quality_on_binding"], c["quality"], c["minutes"]["quality"]),
            (False, None, None),
        )

    def test_rk_post_s_guards_still_refuse(self):
        def refuse(*a, **k):
            raise post.PostRefused("line 1 is not the configured one")

        with nothing_leaves(), mock.patch.object(
            post, "preflight", refuse
        ), self.assertRaises(post.PostRefused):
            mirror.divert(
                self.cfg, self.st, self.store, 7, {"text": "x\n", "sha": H7}, self.spec
            )
        self.assertIn(
            "7:" + H7, mirror.state(self.st)["started"]
        )  # nothing done: the orchestrator fixes and posts

    def test_rk_post_writes_here_and_nothing_leaves(self):
        self.store.save(self.st)
        f = self.root / "post.md"
        f.write_text("the post\n")
        out = []
        with nothing_leaves(), mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch.object(
            post, "preflight", lambda *a, **k: None
        ), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            code = rkmon.main(["post", "7", str(f)])
        self.assertEqual(code, 0, out)
        self.assertTrue(out[0].startswith("MIRRORED #7: "), out)
        self.assertEqual(len(out), 1, out)  # 1 of 2 written: no MIRROR_DONE yet
        self.assertEqual(list(mirror.state(self.store.load())["done"]), ["7:" + H7])


if __name__ == "__main__":
    unittest.main()
