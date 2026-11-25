# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every outward write command runs only for its capability, in a mirror or not, and sends only while its
feature sends: with the features in prototype (the defaults), each one sends nothing and nothing leaves the
test (one test per command): one whose capability is off says so, and one that runs in prototype is refused, naming the
mode, since a person typed it. `rk post`, by either path (plain, or with its premises), keeps the verified review
unposted while review posting is off (PR_UNPOSTED_<N>.md, its line `UNPOSTED: ...`): the monitor's own as an unsent
item too, while one you type is refused. A write you type is refused naming the mode, then standby; the monitor's own
is kept as an unsent item, one journal line each that names its type and what it was for, never its content.
"""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import rkmon  # noqa: E402
import test_mirror  # noqa: E402
from rkmon_lib import (  # noqa: E402
    config,
    deploy_tree,
    features,
    outward,
    post,
    premises,
    state_store,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402

# The commands that send and gate their own sends (rkmon_lib/outward.py: gate OWN), each with its example in
# test_mirror.ARGV; post and spec-post keep the review unposted while review posting is off
# (test_post_keeps_an_unposted_review_by_either_path).
COMMANDS = outward.names(gate=outward.OWN, drafts=False)
# The capability an example's arguments pick, where a command's arguments pick it: `--kind queue`, `--to stability`.
# An issue command's example names no issue of a type, so both types' features are weighed, the first refusing.
PICKED = {
    "notice": post.NOTICE_FEATURE["queue"][0],
    "issue-relabel": "stability-issues",
}


def feature(cmd: str) -> str:
    """The capability rk CMD's example is for, as its refusal names it."""
    return PICKED.get(cmd) or sorted(outward.row(cmd).features)[0]


class Gates(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = pathlib.Path(self.d.name)
        self.cfg = {
            "repo": "ROCm/r",
            "operator": {"github_login": "dev-op"},
            "rocke_path": "rk",
            "review_first_line": "Automated review: Model - qwen, effort: max",
            "features": {},
            "paths": {"working_dir": str(self.root)},
        }
        self.store = Store(self.root / "state")
        st = empty_state()
        st["prs"] = {
            "7": {
                "n": 7,
                "head": "7" * 40,
                "status": "open",
                "draft": False,
                "size": 10,
                "ours": None,
            }
        }
        self.store.save(st)
        self.file = self.root / "x.md"
        self.file.write_text(
            "Automated review: Model - qwen, effort: max\n\n"
            "## PR #7 — t (dev-a) — 🟢 First-round: ok\n"
        )

    def own(self):
        """This process's writes as the monitor's own (outward.OWN_VAR, as the Claude session's are)."""
        return mock.patch.dict(os.environ, {outward.OWN_VAR: "1"})

    def reaching(self, cmd: str) -> tuple:
        """The patches that take rk CMD's example as far as its write: rk queue-notice builds the queue first, and here
        #7 is due a notice."""
        if cmd != "queue-notice":
            return ()
        return (
            mock.patch.object(rkmon.queue_mod, "build_queue", lambda *a: []),
            mock.patch.object(
                rkmon.merge, "desired_notices", lambda cfg, st, q: {7: ("next", "")}
            ),
            mock.patch.object(
                rkmon.merge,
                "notice_text",
                lambda cfg, st, n, state, detail: f"{post.notice_line(cfg, 'queue')}\n\n#7 is next.\n",
            ),
        )

    def rk(self, argv, *patches):
        out = []
        argv = [a.replace("{f}", str(self.file)) for a in argv]
        with test_mirror.nothing_leaves(), mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            for p in patches:
                p.start()
            try:
                code = rkmon.main(argv)
            finally:
                for p in reversed(patches):
                    p.stop()
        return code, out

    def test_every_write_command_is_covered(self):
        for cmd in COMMANDS:
            with self.subTest(cmd=cmd):
                self.assertIn(cmd, test_mirror.ARGV)
                self.assertIn(feature(cmd), features.CAPABILITIES)

    def test_post_keeps_an_unposted_review_by_either_path(self):
        """In prototype the monitor's own `rk post` (the session's), by either path, keeps the verified review unposted,
        PR_UNPOSTED_<N>.md, and as an unsent item of its own; the same post typed is refused, naming the mode, and keeps
        nothing.
        """
        held = features.held_words(self.cfg, "review_automation")
        code, out = self.rk(["post", "7", "{f}"])
        self.assertEqual(
            (code, out), (4, [f"REFUSED: a review post is not sent: {held}"])
        )
        self.assertEqual(post.copies(self.cfg, 7), [])
        code, out = self.rk(["post", "7", "{f}"], self.own())
        self.assertEqual(code, 0, out)
        kept = f"UNPOSTED: {held}, so the verified review stays unposted, kept as an unsent item: "
        self.assertTrue(out[0].startswith(kept), out)
        first = out[0][len(kept) : -1]
        self.assertEqual(pathlib.Path(first).name, "PR_UNPOSTED_7.md")
        ready = {
            "text": self.file.read_text() + "4 ✓\n",
            "sha": "7" * 40,
            "copy": self.root / "copy.md",
        }
        finished = []

        def finish(cfg, store, n, spec, ready, kept=False):
            finished.append((ready["copy"], kept))
            return [], True

        code, out = self.rk(
            ["post", "7", "{f}", "--premises", str(self.root / "spec.json")],
            self.own(),
            mock.patch.object(premises, "load", lambda *a, **k: {"ROUND": "fr"}),
            mock.patch.object(
                premises, "check", lambda *a, **k: (["premises hold"], ready)
            ),
            mock.patch.object(premises, "finish", finish),
        )
        out = "\n".join(out).splitlines()
        self.assertEqual(code, 0, out)
        self.assertEqual(out[0], "premises hold")
        self.assertTrue(out[1].startswith(kept), out)
        draft = pathlib.Path(out[1][len(kept) : -1])
        self.assertTrue(
            draft.read_text().endswith("4 ✓\n")
        )  # the filled-in post, not the verifier's file
        # The round's copy is the kept post too: the keep used its ROUND, as a post does.
        self.assertEqual((self.root / "copy.md").read_text(), draft.read_text())
        # The after-post work of this machine runs for the kept review, the kept file its copy.
        self.assertEqual(finished, [(draft, True)])
        self.assertEqual(
            [
                (i["type"], i["for"], i["file"], i["feature"], i["why"])
                for i in state_store.unsent_items(self.store.dir)
            ],
            [
                (
                    state_store.REVIEW,
                    "pull request 7",
                    f,
                    "review_automation",
                    "prototype",
                )
                for f in (first, str(draft))
            ],
        )

    def test_a_second_keep_at_the_head_of_our_last_review_is_refused(self):
        """The monitor's own plain `rk post` keeps no second review at the head our last review (rec["ours"], kept or
        posted) is at, as it posts no second one there; --again keeps it (a re-review).
        """
        st = self.store.load()
        st["prs"]["7"]["ours"] = {
            "kind": post.UNPOSTED,
            "head": "7" * 40,
            "count": 1,
            "file": "PR_UNPOSTED_7.md",
        }
        self.store.save(st)
        code, out = self.rk(["post", "7", "{f}"], self.own())
        self.assertEqual(
            (code, out),
            (
                4,
                [
                    "REFUSED: our last review of #7 is at head 7777777777 already (kept unposted: PR_UNPOSTED_7.md) "
                    "(use --again if intended)"
                ],
            ),
        )
        self.assertEqual(post.copies(self.cfg, 7), [])
        code, out = self.rk(["post", "7", "{f}", "--again"], self.own())
        self.assertEqual(code, 0, out)
        self.assertEqual(
            [p.name for p in post.copies(self.cfg, 7)], ["PR_UNPOSTED_7.md"]
        )

    def unposted(self, review_automation: dict, **over) -> tuple:
        """The configuration with review automation at REVIEW_AUTOMATION (OVER: other settings), under which a
        verified review is kept unposted; the file it is kept in, and the line that says so.
        """
        cfg = dict(self.cfg, features={"review_automation": review_automation}, **over)
        saved = str(self.root / "PR_UNPOSTED_7.md")
        self.assertTrue(post.keeps(cfg))
        return cfg, saved, post.kept_line(cfg, saved)

    def test_the_unposted_review_s_line_names_its_file_and_no_post_by_hand(self):
        """The line that says a verified review was kept (post.kept_line) starts `UNPOSTED: ` and names the file, while
        review automation holds it (in prototype), while it is off and while review posting is not set up; it never
        asks for a post by hand: the monitor records only what it posts."""
        for case, review_automation, over, held in (
            ("prototype", {"enabled": True, "mode": "prototype"}, {}, True),
            ("off", {"enabled": False}, {}, False),
            ("not set up", {"enabled": True, "mode": "live"}, {"operator": {}}, False),
        ):
            _, saved, line = self.unposted(review_automation, **over)
            with self.subTest(case=case):
                self.assertTrue(line.startswith("UNPOSTED: "), line)
                self.assertTrue(post.is_kept_line(line), line)
                self.assertTrue(line.endswith(f": {saved}."), line)
                self.assertEqual("unsent item" in line, held, line)
                self.assertNotIn("by hand", line)

    def test_the_unposted_line_in_prototype_names_the_mode(self):
        """Review automation in prototype: the line names its mode and what ends it, and the review's unsent item."""
        _, saved, line = self.unposted({"enabled": True, "mode": "prototype"})
        mode = deploy_tree.name("features.review_automation.mode")
        self.assertEqual(
            line,
            f"UNPOSTED: {mode} is prototype: nothing goes out until it is live, so the verified review stays unposted, "
            f"kept as an unsent item: {saved}.",
        )

    def test_the_unposted_line_with_review_automation_off_names_enabled_and_the_mode(
        self,
    ):
        """Review automation off: the line names both settings review posting runs while, `enabled` and the mode, as
        the outward gate refuses a post typed then (features.why_off): the mode alone does not turn it on.
        """
        _, saved, line = self.unposted({"enabled": False})
        self.assertEqual(
            line,
            "UNPOSTED: review posting is off (it runs while features.review_automation is enabled and live), so the "
            f"verified review stays unposted: {saved}.",
        )

    def test_the_unposted_line_while_posting_is_not_set_up_names_what_it_lacks_and_the_fix(
        self,
    ):
        """Review automation live with no GitHub login: review posting is on but not set up, and the line names what
        it lacks and the fix in features' words (missing, fix), the GitHub login among them, and no mode to set.
        """
        cfg, saved, line = self.unposted({"enabled": True, "mode": "live"}, operator={})
        lacks = features.missing(cfg, post.POSTING)
        self.assertIn(deploy_tree.name("operator.github_login"), lacks[0])
        fix = features.fix(cfg, post.POSTING)
        self.assertEqual(
            line,
            f"UNPOSTED: review posting is on but needs {'; '.join(lacks)} (fix: {fix}), so the verified review stays "
            f"unposted: {saved}.",
        )
        self.assertNotIn(deploy_tree.name("features.review_automation.mode"), line)

    def test_a_write_you_type_is_refused_naming_the_mode_then_standby(self):
        """`rk notice` typed is refused while the merge queue is in prototype, naming its mode, and, with
        the merge queue live, while the monitor is in standby, naming standby; the monitor's own is kept as an unsent
        item instead, and the journal's line for it names its type and what it was for, never its content. Nothing
        leaves, and nothing is ledgered."""
        home = self.root / "home"
        body = self.root / "queue.md"
        body.write_text(
            f"{post.notice_line(self.cfg, 'queue')}\n\nThe plan for #7 is in a quiet note.\n"
        )
        argv = ["notice", "7", str(body), "--kind", "queue", "--key", "k"]
        said, want = [], []
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}):
            for standby in (False, True):
                if standby:
                    self.cfg["features"] = {
                        "merge_queue": {"enabled": True, "mode": "live"}
                    }
                    state_store.set(home, "standby.on", True)
                held = features.held_words(self.cfg, "merge_queue")
                self.assertIn("standby" if standby else "prototype", held)
                code, out = self.rk(argv)
                self.assertEqual(
                    (code, out), (4, [f"REFUSED: a queue notice is not sent: {held}"])
                )
                code, out = self.rk(argv, self.own())
                self.assertEqual(code, 0, out)
                said += out
                want.append(
                    "UNSENT: a queue notice for pull request 7, kept as an unsent item: "
                    + held
                )
        self.assertEqual(said, want)
        self.assertEqual(
            [
                (i["type"], i["for"], i["feature"], i["why"], i["content"])
                for i in state_store.unsent_items(self.store.dir)
            ],
            [
                (
                    state_store.COMMENT,
                    "pull request 7",
                    "merge_queue",
                    why,
                    body.read_text(),
                )
                for why in (state_store.PROTOTYPE, state_store.STANDBY)
            ],
        )
        journal = [ln for ln in self.store.journal_lines() if " unsent " in ln]
        self.assertEqual(
            [ln.split(" unsent ", 1)[1].strip() for ln in journal],
            [
                "comment for pull request 7: not sent (the merge queue in prototype)",
                "comment for pull request 7: not sent (the monitor in standby)",
            ],
        )
        self.assertNotIn("quiet note", "\n".join(self.store.journal_lines()))
        self.assertEqual(self.store.ledger(), [])

    def test_turning_off_an_auto_merge_the_monitor_armed_waits_in_standby_and_runs_in_prototype(
        self,
    ):
        """The protective write waits in standby, the monitor's own as what you type: nothing made and
        nothing kept, the PR still armed, so the tick asks for it again; once standby ends it goes, in prototype too.
        """
        home = self.root / "home"
        st = self.store.load()
        st["queue_state"] = {
            "armed": {"pr": 7, "head": "7" * 40, "develop": "d", "at": "t"}
        }
        self.store.save(st)
        writes = []

        def gh_write(args, **_kw):
            writes.append(args)
            return mock.Mock(returncode=0, stdout="", stderr="")

        argv = ["disarm", "7", "its", "head", "moved"]
        patches = (
            mock.patch.object(post, "_gh_write", gh_write),
            mock.patch.object(rkmon.queue_mod, "build_queue", lambda *a: []),
        )
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}):
            state_store.set(home, "standby.on", True)
            held = features.held_words(self.cfg, "merge_queue")
            code, out = self.rk(argv, self.own(), *patches)
            self.assertEqual(
                (code, out),
                (
                    0,
                    [
                        f"WAITING: turning auto-merge off on #7 waits, nothing was sent: {held}"
                    ],
                ),
            )
            code, out = self.rk(argv, *patches)
            self.assertEqual(
                (code, out),
                (4, [f"REFUSED: turning auto-merge off on #7 is not sent: {held}"]),
            )
            self.assertEqual(writes, [])
            self.assertEqual(self.store.load()["queue_state"]["armed"]["pr"], 7)
            state_store.set(home, "standby.on", False)
            self.assertFalse(features.live(self.cfg, "merge_queue"))
            code, out = self.rk(argv, *patches)
        self.assertEqual((code, out), (0, ["auto-merge disabled"]))
        self.assertEqual([w[:3] for w in writes], [["pr", "merge", "7"]])
        self.assertIsNone(self.store.load()["queue_state"]["armed"])
        self.assertEqual(state_store.unsent_items(self.store.dir), [])

    def test_the_guard_s_off_line_names_the_mode_then_standby(self):
        """rk automerge-off typed while the auto-merge guard only alerts leaves auto-merge on, and its OFF line says
        why: the merge queue's mode, or, with the merge queue live, standby."""
        home = self.root / "home"
        self.cfg["features"] = {"merge_queue": {"enabled": True, "mode": "prototype"}}
        said, why = [], []
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}), mock.patch.object(
            features, "_gh", lambda cfg: []
        ):
            for standby in (False, True):
                if standby:
                    self.cfg["features"]["merge_queue"]["mode"] = "live"
                    state_store.set(home, "standby.on", True)
                said.append(self.rk(test_mirror.ARGV["automerge-off"]))
                why.append(features.held_words(self.cfg, "merge_queue"))
        self.assertIn("features.merge_queue.mode is prototype", why[0])
        self.assertIn("the monitor is in standby", why[1])
        self.assertEqual(
            said,
            [
                (
                    0,
                    [
                        "OFF: the auto-merge guard only alerts while the merge queue sends nothing "
                        f"({w}): auto-merge was left on."
                    ],
                )
                for w in why
            ],
        )

    def test_disarm_runs_with_the_feature_off_while_the_pr_is_armed(self):
        """Turning auto-merge off is protective: a PR the state still records as armed is disarmed even after the
        auto-merge feature was switched off (the feature gate refuses only when nothing is armed).
        """
        st = self.store.load()
        st["queue_state"] = {
            "armed": {"pr": 7, "head": "7" * 40, "develop": "d", "at": "t"}
        }
        self.store.save(st)
        done = []
        code, out = self.rk(
            ["disarm", "7", "feature", "off"],
            mock.patch.object(
                rkmon.post_mod,
                "disarm",
                lambda cfg, st, store, n, why: done.append(n) or "auto-merge disabled",
            ),
            mock.patch.object(rkmon.queue_mod, "build_queue", lambda *a: []),
        )
        self.assertEqual((code, done), (0, [7]), out)
        code, out = self.rk(["disarm", "8"])  # not armed: the feature's refusal
        self.assertTrue(out[0].startswith("OFF: auto-merge is off"), out)

    def test_the_teams_test_card_runs_while_either_alert_is_on(self):
        """rk teams-test's card checks the channel both alerts send to: with both off it does nothing, and either one
        on lets it run, here as far as the webhook, which the test has none of."""
        # the webhook's place, as the load derives it (config.normalize)
        self.cfg["teams"] = config.normalize(self.cfg)["teams"]
        code, out = self.rk([outward.TEAMS_TEST])
        self.assertEqual(code, features.EXIT_OFF, out)
        self.assertTrue(
            out and out[0].startswith("OFF: develop break alerts is off, so nothing"),
            out,
        )
        for alert in ("develop_breaks", "needs_you"):
            with self.subTest(alert=alert):
                self.cfg["features"] = {
                    "alerts": {
                        "enabled": True,
                        "develop_breaks": alert == "develop_breaks",
                        "needs_you": alert == "needs_you",
                    }
                }
                code, out = self.rk([outward.TEAMS_TEST])
                self.assertEqual(code, 4, out)
                self.assertTrue(
                    out and out[0].startswith("REFUSED: no Teams webhook configured"),
                    out,
                )


def _gate_test(cmd: str):
    def test(self):
        code, out = self.rk(test_mirror.ARGV[cmd], *self.reaching(cmd))
        name = feature(cmd)
        title = features.lowered(features.CAPABILITIES[name].title)
        sends = features.CAPABILITIES[name].feature
        if cmd == "automerge-off":
            want = (
                0,
                "OFF: the auto-merge guard only alerts while the merge queue sends nothing "
                "(features.merge_queue.mode is prototype",
            )
        elif features.on(
            self.cfg, name
        ):  # it runs in prototype, and what you type is refused there
            want = (4, "REFUSED: ")
            self.assertTrue(
                out
                and out[0].endswith(
                    f" is not sent: {features.held_words(self.cfg, sends)}"
                ),
                out,
            )
        else:
            want = (0, f"OFF: {title} is off, so nothing was done: it runs while ")
        self.assertEqual(code, want[0], out)
        self.assertTrue(out and out[0].startswith(want[1]), out)

    test.__doc__ = f"rk {cmd} sends nothing while {feature(cmd)} is off or its feature in prototype."
    return test


for _cmd in COMMANDS:
    setattr(
        Gates, f"test_{_cmd.replace('-', '_')}_waits_for_its_feature", _gate_test(_cmd)
    )


if __name__ == "__main__":
    unittest.main()
