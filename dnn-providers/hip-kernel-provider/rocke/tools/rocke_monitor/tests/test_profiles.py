# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Starting templates (the deploy schema's profiles, x-profiles: `rkmon template deploy --profile NAME FILE`) and what a
deployment's features start: each template's features, that no profile is a setting or changes the image, the logins
nothing uses (NOT NEEDED), the parts that do not start, and the rule-acted tick of a deployment with no Claude session
(rkmon_lib/rules_tick.py)."""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    config,
    deploy_tree,
    features,
    file_types,
    logins,
    merge,
    orchestrator,
    outward,
    post,
    report,
    rules_tick,
    settings_file,
    state_store,
    webserve,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402

BASE = {
    "repo": "acme/widgets",
    "rocke_path": "a/b",
    "operator": {"github_login": "op-login"},
    "scope": {"labels": ["x"]},
}


# The kinds of deployment the tests of what starts run: each one's features over the defaults, as the deploy
# settings file's keys. The starting templates are the deploy schema's profiles (TestTemplates).
KINDS = {
    "defaults": {},
    "reviews only": {
        "features.merge_queue.enabled": False,
        "features.merge_queue.develop_guard.enabled": False,
        "features.dashboards.admin.enabled": False,
    },
    "no reviews": {"features.review_automation.enabled": False},
}


def cfg(kind: str = "defaults", **feats) -> dict:
    """A configuration of a kind of deployment (KINDS), with `feats` (the deploy settings file's keys under
    features., dotted) over it, through deploy_tree.to_config."""
    tree = {}
    for k, v in KINDS[kind].items():
        deploy_tree.put(tree, k, v)
    for k, v in feats.items():
        deploy_tree.put(tree, "features." + k.replace("__", "."), v)
    return dict(BASE, **deploy_tree.to_config(tree))


class TestTemplates(unittest.TestCase):
    """Each profile's template (settings_file.template); test_settings_schemas checks that each one passes its own
    checks and holds its profile's values."""

    def template(self, name: str) -> dict:
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "settings.yaml"
            path.write_text(
                settings_file.template(file_types.DEPLOY, name), encoding="utf-8"
            )
            return settings_file.read(path, file_types.DEPLOY).values

    def test_each_profile_s_template_has_its_features(self):
        # Review automation, the merge queue, its develop guard and the admin dashboard. A feature that starts off is
        # written as its switch alone, so review-only has no develop guard key (None).
        want = {
            "full": (True, True, True, True),
            "review-only": (True, False, None, True),
            "merge-queue": (False, True, True, True),
        }
        for name, (ra, mq, guard, admin) in want.items():
            with self.subTest(profile=name):
                t = self.template(name)
                f = t["features"]
                self.assertEqual(
                    (
                        f["review_automation"]["enabled"],
                        f["merge_queue"]["enabled"],
                        f["merge_queue"].get("develop_guard", {}).get("enabled"),
                        f["dashboards"]["admin"]["enabled"],
                    ),
                    (ra, mq, guard, admin),
                )
                # Every template starts in prototype: nothing goes out until a mode is set to live.
                for feature in ("review_automation", "merge_queue"):
                    if f[feature]["enabled"]:
                        self.assertEqual(f[feature]["mode"], "prototype", feature)
                self.assertNotIn("profile", t)

    def test_a_profile_is_never_a_setting(self):
        schema = file_types.DEPLOY.load()
        self.assertNotIn("profile", schema["properties"])
        self.assertNotIn("profile", schema["properties"]["container"]["properties"])
        for p in settings_file.profiles(file_types.DEPLOY):
            self.assertNotIn(
                "image", " ".join(p.values)
            )  # a profile never names the image


class TestClaude(unittest.TestCase):
    def test_only_a_deployment_without_reviews_ticks_by_rules(self):
        c = cfg()
        self.assertTrue(features.uses_claude(c))
        self.assertFalse(features.rules_only(c))
        mq = cfg("no reviews")
        self.assertFalse(features.uses_claude(mq))
        self.assertTrue(features.rules_only(mq))
        ro = cfg("reviews only")
        self.assertTrue(features.uses_claude(ro))
        self.assertFalse(features.rules_only(ro))


class TestLogins(unittest.TestCase):
    def rows(self, c):
        with tempfile.TemporaryDirectory() as d:
            c = dict(c, paths={"working_dir": d})
            with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"}):
                return {
                    r["name"]: r for r in logins.rows(c, probe=lambda _c: ({}, None))
                }

    def test_a_login_nothing_uses_is_not_needed(self):
        ro = self.rows(cfg("reviews only", review_automation__mode="live"))
        for name in (
            "teams_webhook",
            "web_host_ssh_key",
            "atlassian",
            "admin_access_token",  # reviews only: no admin dashboard
            "gpu_ssh_key",
        ):
            self.assertEqual(ro[name]["state"], "NOT NEEDED", name)
        self.assertIn("review posting", ro["github_write"]["needed_by"])
        self.assertIn("Claude session", ro["claude"]["needed_by"])
        mq = self.rows(cfg("no reviews"))
        for name in (
            "github_write",
            "claude",
            "teams_webhook",
            "web_host_ssh_key",
            "atlassian",
            "gpu_ssh_key",
        ):
            self.assertEqual(mq[name]["state"], "NOT NEEDED", name)
        armed = self.rows(
            cfg(
                "no reviews",
                merge_queue__mode="live",
                merge_queue__auto_merge__enabled=True,
                merge_queue__auto_hold__enabled=False,
                merge_queue__develop_guard__enabled=False,
            )
        )
        self.assertIn("auto-merge", armed["github_write"]["needed_by"])
        for r in list(ro.values()) + list(mq.values()):
            self.assertNotEqual(r["state"], "FAIL")


class TestWhatStarts(unittest.TestCase):
    def test_no_web_server_and_no_review_driver(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            from io import StringIO
            from contextlib import redirect_stdout

            buf = StringIO()
            with redirect_stdout(buf):
                rc = webserve.serve(cfg("reviews only"), store)
            self.assertEqual(rc, 0)
            self.assertIn(
                "RKWEB not started: features.dashboards.admin is off", buf.getvalue()
            )
            self.assertFalse((pathlib.Path(d) / "web.lock").exists())
        self.assertFalse(orchestrator.driver_ok(cfg("no reviews")))

    def test_the_container_starts_no_claude_session_and_ticks_by_rules(self):
        from test_entrypoint import fake_rk, function

        with tempfile.TemporaryDirectory() as d:
            rk = fake_rk(
                pathlib.Path(d),
                '[ "$1 $2 $3" = "features --runs claude" ] && exit 1\nexit 0\n',
            )
            script = (
                f'RK={rk}\nORCH_PY=/nonexistent\nsay() {{ echo "$*"; }}\ncode() {{ echo /code; }}\n'
                f'{function("runs")}{function("unreadable")}{function("orch_ready")}'
                f'{function("orchestrator")}orchestrator\n'
            )
            p = subprocess.Popen(
                ["sh", "-c", script],
                stdout=subprocess.PIPE,
                text=True,
                env={"PATH": "/usr/bin:/bin", "RKMON_ORCH_WAIT_S": "0.2"},
            )
            time.sleep(1)
            self.assertIsNone(p.poll())  # waits; never runs the session
            p.kill()
            out, _ = p.communicate(timeout=10)
        self.assertIn("no Claude session runs", out)
        ticks = function("ticks_without_orchestrator")
        self.assertIn('"$RK" features --runs rules', ticks)
        self.assertIn('"$RK" rules-tick', ticks)
        self.assertIn("if runs web; then", function("dashboard"))


TICK = """TICK 12:00Z
ARM_READY #5: batch [5] validated on develop dddddddddd; run `rk arm 5`
PROPOSE #7 next: every gate is green against develop dddddddddd
SOMETHING_NEW #9 the rules do not know
!! CRITICAL INCIDENT_OPEN INC-1: sanity red at dddddddddd: build
BISECT_NEEDED INC-1: 2 candidates [8, 9] (run `rk bisect`)
LANDED #4 (armed by us); develop sanity follows
time: poll 1s
"""


class TestRulesTick(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.store.save(empty_state())
        self.calls = []

    def tearDown(self):
        self.d.cleanup()

    def runner(self, out: str, refuse=()):
        def run(argv, **_kw):
            args = argv[1:]
            self.calls.append(args)
            if args == ["tick"]:
                return subprocess.CompletedProcess(argv, 0, out, "")
            if args[0] in refuse:
                return subprocess.CompletedProcess(
                    argv, 4, f"REFUSED: {args[0]} said no\n", ""
                )
            return subprocess.CompletedProcess(argv, 0, f"{args[0]} done\n", "")

        return run

    def test_fixed_actions_run_and_the_rest_waits_for_a_person(self):
        said = []
        rc = rules_tick.run(
            cfg("no reviews"), self.store, runner=self.runner(TICK), say=said.append
        )
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, [["tick"], ["arm", "5"], ["bisect"]])
        lines = [e["line"] for e in rules_tick.entries(self.store.load())]
        self.assertEqual(
            sorted(lines),
            sorted(
                [
                    "PROPOSE #7 next: every gate is green against develop dddddddddd",
                    "SOMETHING_NEW #9 the rules do not know",
                    "!! CRITICAL INCIDENT_OPEN INC-1: sanity red at dddddddddd: build",
                ]
            ),
        )
        self.assertTrue(any(s.startswith("NEEDS_PERSON: 3 line(s)") for s in said))
        # On the admin dashboard's Needs-you list and in rk status.
        bd = report.board(
            cfg("no reviews"),
            self.store.load(),
            self.store,
            [],
            [],
        )
        self.assertTrue(
            any(
                "Needs a person (no Claude session here): PROPOSE #7" in n
                for n in bd["needs"]
            )
        )
        status = rules_tick.status_lines(self.store.load())
        self.assertTrue(
            status[0].startswith("Needs a person (no Claude session here): 3")
        )

    def test_a_refused_rule_waits_for_a_person_and_a_standing_line_leaves_when_it_is_gone(
        self,
    ):
        rules_tick.run(
            cfg("no reviews"),
            self.store,
            runner=self.runner(TICK, refuse=("arm",)),
            say=lambda _l: None,
        )
        e = {x["line"][:12]: x for x in rules_tick.entries(self.store.load())}
        self.assertIn("ARM_READY #5", e)
        self.assertIn(
            "rk arm 5` did not go through: REFUSED: arm said no",
            e["ARM_READY #5"]["why"],
        )
        # The next full tick prints neither standing line: they leave; the one-off lines stay until they expire.
        self.calls.clear()
        rules_tick.run(
            cfg("no reviews"),
            self.store,
            runner=self.runner("TICK 12:10Z\ntime: poll 1s\n"),
            say=lambda _l: None,
        )
        left = {x["line"][:12] for x in rules_tick.entries(self.store.load())}
        self.assertNotIn("ARM_READY #5", left)
        self.assertNotIn("PROPOSE #7 n", left)
        self.assertIn("SOMETHING_NE", left)
        self.assertEqual(rules_tick.clear(self.store), 2)
        self.assertEqual(rules_tick.entries(self.store.load()), [])

    def test_rules(self):
        self.assertEqual(
            rules_tick.rule("!! DISARM_NEEDED #3: its head moved. Run `rk disarm 3`"),
            ["disarm", "3"],
        )
        self.assertEqual(
            rules_tick.rule(
                "!! AUTOMERGE_RISK #6 (auto-merge enabled by dev-a): G3 red. run `rk automerge-off 6`"
            ),
            ["automerge-off", "6"],
        )
        self.assertIsNone(
            rules_tick.rule(
                "!! AUTOMERGE_RISK #6 (auto-merge enabled by dev-a): G3 red. alert the user"
            )
        )
        self.assertIsNone(rules_tick.rule("!! AUTOMERGE_JUMP #6: ahead of the plan"))
        self.assertEqual(
            rules_tick.rule(
                "QUEUE_NOTICE #8: next (dddddddddd); run `rk queue-notice 8`"
            ),
            ["queue-notice", "8"],
        )


QUEUE_TICK = """TICK 12:00Z
QUEUE_NOTICE #5: next (dddddddddd); run `rk queue-notice 5`
time: poll 1s
"""
URL = "https://github.com/ROCm/r/pull/5#issuecomment-1001"


class TestQueueNoticeByRule(unittest.TestCase):
    """A queue notice with no Claude session: the rules tick runs `rk queue-notice N` for the tick's
    QUEUE_NOTICE line as a fixed rule, and the command's own guards decide. Each rk command the rules tick starts runs
    here, in this process, in the environment the tick gives it (the mark of the monitor's own writes among it), with
    GitHub's writes faked and the queue's own computation left out, so the state's validated batch stands.
    """

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.n = 0

    def tearDown(self):
        self.d.cleanup()

    def deployment(self) -> None:
        """A merge-queue-only deployment, live, in a home of its own, whose state has PR 5 next in line."""
        self.n += 1
        root = pathlib.Path(self.d.name) / str(self.n)
        self.home, self.writes = root / "home", []
        self.store = Store(self.home / "state")
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["batches"] = [{"status": "validated", "members": [5]}]
        self.store.save(st)
        self.cfg = config.load_defaults()
        self.cfg.update(BASE)
        self.cfg["features"] = {
            "review_automation": {"enabled": False},
            "merge_queue": {"enabled": True, "mode": "live"},
        }
        self.cfg["paths"] = dict(
            self.cfg["paths"], working_dir=str(root / "work"), repo=str(root / "none")
        )

    def state(self, **values) -> None:
        st = self.store.load()
        st.update(values)
        self.store.save(st)

    def rk(self, argv, env=None, **_kw):
        """The rules tick's runner: `rk tick` prints QUEUE_TICK; any other rk command runs here, its writes the
        monitor's own as the tick's environment ENV says (outward.OWN_VAR)."""
        args = argv[1:]
        if args == ["tick"]:
            return subprocess.CompletedProcess(argv, 0, QUEUE_TICK, "")
        env = {
            "RKMON_HOME": str(self.home),
            "RKMON_CONTAINER": "",
            outward.OWN_VAR: (env or {}).get(outward.OWN_VAR, ""),
        }
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out):
            rc = rkmon.main(args)
        return subprocess.CompletedProcess(argv, rc, out.getvalue(), "")

    def gh_write(self, args, **_kw):
        self.writes.append(args)
        return subprocess.CompletedProcess(args, 0, URL + "\n", "")

    def tick(self, gh=lambda _cfg: []) -> tuple[list, dict]:
        """One rules tick, with GH as what acting on GitHub lacks (features._gh): what it said, and the state after."""
        (self.home / "config.json").write_text(json.dumps(self.cfg), encoding="utf-8")
        said = []
        with mock.patch.object(post, "_gh_write", self.gh_write), mock.patch.object(
            features, "_gh", gh
        ), mock.patch.object(rkmon.queue_mod, "build_queue", lambda *_a: []):
            rules_tick.run(self.cfg, self.store, runner=self.rk, say=said.append)
        return said, self.store.load()

    def test_a_queue_notice_goes_out_with_no_session(self):
        self.deployment()
        self.assertFalse(features.uses_claude(self.cfg))
        said, st = self.tick()
        self.assertEqual(
            [w[:5] for w in self.writes],
            [["pr", "comment", "5", "--repo", "acme/widgets"]],
        )
        self.assertIn(f"RULE queue-notice 5: notice: {URL}", said)
        self.assertEqual(st["notices"]["5"]["url"], URL)
        self.assertEqual(
            [(r["kind"], r["pr"]) for r in self.store.ledger()],
            [("notice-queue-sticky", 5)],
        )
        self.assertEqual(rules_tick.entries(st), [])

    def test_each_guard_that_stops_one(self):
        """Nothing goes out while a guard of rk queue-notice says no. Off (the switch), with nothing to say, or with the
        merge queue in prototype (the notice kept as an unsent item: the rule's command is the monitor's own), the rule
        is done; a refusal leaves the line for a person, with the guard's words.
        """
        real_gh = features._gh

        def no_gh(cfg):
            with mock.patch.dict(os.environ, {"PATH": "/nonexistent"}):
                return real_gh(cfg)

        def checked(cfg, *_a):
            return f"{post.notice_line(cfg, 'queue')}\n\nThe gate's log: https://ci.example.com/job/1\n"

        def mq(**values):
            return lambda: self.cfg["features"]["merge_queue"].update(values)

        def busy():
            for k in range(10):
                self.store.ledger_add("post", 100 + k, "h", "")

        stopped = {"at": "2026-11-18T12:00:00Z", "how": "stop", "boot": ""}
        rk_off = "OFF: queue notices is off, so nothing was done: it runs while features.merge_queue is enabled and its "
        kept = (  # the held line alone, no success word before it
            "RULE queue-notice 5: UNSENT: a queue notice for pull request 5, kept as an unsent item: "
            "features.merge_queue.mode is prototype: nothing goes out until it is live"
        )
        for guard, arrange, words, refused in (
            ("the switch off", mq(queue_notices=False), rk_off, False),
            ("the merge queue in prototype", mq(mode="prototype"), kept, False),
            (
                "no notice needed",
                lambda: self.state(batches=[]),
                "#5 needs no queue notice right now",
                False,
            ),
            (
                "the monitor stopped",
                lambda: self.state(stopped=stopped),
                "REFUSED: the monitor is stopped",
                True,
            ),
            ("gh not set up", None, "NOT SET UP: queue notices is on but needs", True),
            ("the public-repo check", None, "REFUSED: link to a non-public host", True),
            ("the rate limit", busy, "REFUSED: rate limit: 10 outward actions", True),
        ):
            with self.subTest(guard=guard):
                self.deployment()
                if arrange:
                    arrange()
                with (
                    mock.patch.object(merge, "notice_text", checked)
                    if guard.startswith("the public")
                    else contextlib.nullcontext()
                ):
                    said, st = (
                        self.tick(no_gh) if guard == "gh not set up" else self.tick()
                    )
                self.assertEqual(self.writes, [])
                self.assertNotIn("5", st.get("notices") or {})
                rule = [s for s in said if s.startswith("RULE queue-notice 5: ")]
                self.assertEqual(len(rule), 1, said)
                self.assertIn(words, rule[0])
                waiting = rules_tick.entries(st)
                self.assertEqual(bool(waiting), refused)
                if refused:
                    self.assertIn(
                        "rk queue-notice 5` did not go through: " + words,
                        waiting[0]["why"],
                    )
                self.assertEqual(
                    [
                        (i["type"], i["for"])
                        for i in state_store.unsent_items(self.store.dir)
                    ],
                    [(state_store.COMMENT, "pull request 5")] if words == kept else [],
                )


class TestDocs(unittest.TestCase):
    def test_the_starting_templates_table_and_the_guide_s_profiles_are_the_schema_s(
        self,
    ):
        """The README's table of the starting templates (its generated section, profiles), and the deploy guide's
        "Pick a profile", which names each profile with what it runs, in the schema's order, and the
        command that writes one.
        """
        import re

        text = (ROOT / "README.md").read_text(encoding="utf-8")
        table = text.split("Starting templates\n", 1)[1].split("\n#", 1)[0]
        for p in settings_file.profiles(file_types.DEPLOY):
            with self.subTest(profile=p.name):
                row = next(
                    ln for ln in table.splitlines() if ln.startswith(f"| `{p.name}` | ")
                )
                self.assertTrue(row.startswith(f"| `{p.name}` | {p.what}"), row)
                named = dict(re.findall(r"`([\w.]+): (true|false)`", row))
                self.assertEqual(
                    named, {k: str(v).lower() for k, v in p.values.items()}
                )
        guide = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        pick = guide.split("\n### Pick a profile\n", 1)[1].split("\n#", 1)[0]
        self.assertEqual(
            re.findall(r"^- `([\w-]+)`: (.+)\.$", pick, re.M),
            [(p.name, p.what) for p in settings_file.profiles(file_types.DEPLOY)],
        )
        self.assertIn("`./rkmon template deploy --profile PROFILE FILE`", pick)


if __name__ == "__main__":
    unittest.main()
