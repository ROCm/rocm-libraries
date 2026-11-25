# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The features (rkmon_lib/features.py): review automation, the merge queue, the dashboards, the alerts and ticket
creation decide every capability, each of the five that send in prototype or live; in prototype nothing goes out; a
capability that is off never acts and never fails; one that is on but lacks what it needs is skipped with a plain
report; the combinations that cannot run are refused; auto-merge only ever turns on GitHub's own auto-merge; the setup
summary never shows a secret.
"""

import contextlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    atlassian,
    breakage,
    config,
    deploy_tree,
    features,
    gh,
    outward,
    post,
    publish,
    reach,
    registry,
    secrets_registry,
    state_store,
    summary,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402

import guard_labels  # noqa: E402

LIVE = {
    "review_automation": {"enabled": True, "mode": "live"},
    "merge_queue": {
        "enabled": True,
        "mode": "live",
        "auto_merge": "enabled",
        "auto_hold": True,
        "develop_guard": True,
    },
    "dashboards": {
        "admin": {"enabled": True},
        "team": {"enabled": True, "mode": "live"},
    },
    "alerts": {
        "enabled": True,
        "mode": "live",
        "develop_breaks": True,
        "needs_you": True,
    },
    "ticket_creation": {
        "enabled": True,
        "mode": "live",
        "triggers": {"review_follow_up": True, "out_of_scope_issue": True},
    },
}


def merged(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = (
            merged(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


def base_cfg(d: pathlib.Path, feats: dict | None = None, **top) -> dict:
    c = config.load_defaults()
    c.update(
        repo="ROCm/r",
        rocke_path="a/b/rocke",
        operator={"github_login": "op-login"},
        scope={"labels": ["x"], "path_labels": []},
    )
    c["paths"] = dict(c["paths"], working_dir=str(d / "work"))
    c["features"] = merged(c["features"], feats or {})
    for k, v in top.items():
        c[k] = merged(c.get(k) or {}, v) if isinstance(v, dict) else v
    return c


class _NoNetwork:
    """Fails the test on any GitHub read, or gh or rsync call."""

    def __enter__(self):
        boom = mock.Mock(
            side_effect=AssertionError(
                "a disabled capability called GitHub, gh or rsync"
            )
        )
        self.stack = contextlib.ExitStack()
        for target in (
            mock.patch.object(gh, "rest", boom),
            mock.patch.object(gh, "graphql", boom),
            mock.patch.object(post.subprocess, "run", boom),
            mock.patch.object(publish.subprocess, "run", boom),
        ):
            self.stack.enter_context(target)
        return self

    def __exit__(self, *exc):
        self.stack.close()


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.d.name)
        (self.root / "home").mkdir()
        self.file = self.root / "note.md"
        self.file.write_text("Automated note\n\nbody\n", encoding="utf-8")
        self.issue = self.root / "issue.md"

    def tearDown(self):
        self.d.cleanup()

    def rk(self, cfg: dict, *args, path_without_gh=False, own=False):
        """rk ARGS with configuration CFG, typed (a person's), or the monitor's own (`own`: outward.OWN_VAR): its exit
        code, what it printed, and the ledger after."""
        (self.root / "home" / "config.json").write_text(
            json.dumps(cfg), encoding="utf-8"
        )
        env = {
            "RKMON_HOME": str(self.root / "home"),
            "RKMON_CONTAINER": "",
            outward.OWN_VAR: "1" if own else "",
        }
        if path_without_gh:
            env["PATH"] = "/nonexistent"
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(
            out
        ), _NoNetwork():
            rc = rkmon.main(list(args))
        ledger = (
            Store(self.root / "home" / "state").ledger()
            if (self.root / "home" / "state").exists()
            else []
        )
        return rc, out.getvalue(), ledger


# capability -> (the switches that turn it on, the command line that acts for it: {file} filled in)
COMMANDS = {
    "auto-merge": ({"merge_queue": LIVE["merge_queue"]}, {}, ["arm", "7"]),
    "auto-merge-guard": (
        {"merge_queue": LIVE["merge_queue"]},
        {},
        ["automerge-off", "7"],
    ),
    "holds": (
        {"merge_queue": LIVE["merge_queue"]},
        {"_off": {"merge_queue": {"auto_hold": False}}},
        ["request-changes", "7", "{file}"],
    ),
    "overlap-management": (
        {"review_automation": LIVE["review_automation"]},
        {"_off": {"review_automation": {"enabled": False}}},
        ["overlap-decide", "k", "7"],
    ),
    "stability-issues": (
        {"merge_queue": LIVE["merge_queue"]},
        {"_off": {"merge_queue": {"develop_guard": False}}},
        ["issue-relabel", "7", "--to", "stability"],
    ),
    "incident-issues": (
        {"merge_queue": LIVE["merge_queue"]},
        {"_off": {"merge_queue": {"develop_guard": False}}},
        ["notice", "7", "{file}", "--kind", "incident", "--key", "k"],
    ),
    "chat-alerts": ({"alerts": LIVE["alerts"]}, {}, ["teams-breakage", "7"]),
    "queue-notices": (
        {"merge_queue": LIVE["merge_queue"]},
        {"_off": {"merge_queue": {"queue_notices": False}}},
        ["queue-notice", "7"],
    ),
    "drift-notices": (
        {"merge_queue": LIVE["merge_queue"]},
        {"_off": {"merge_queue": {"drift_notices": False}}},
        ["notice", "7", "{file}", "--kind", "drift", "--key", "k"],
    ),
    "fix-links": (
        {"merge_queue": LIVE["merge_queue"]},
        {"_off": {"merge_queue": {"develop_guard": False}}},
        ["fix-link", "1", "7"],
    ),
    "remote-gpu-tests": ({}, {"gpu": {"enabled": True}}, ["gpu-run", "7"]),
}


class TestModes(unittest.TestCase):
    """prototype computes and keeps everything here, each capability on by its switches as it is live; live acts on
    GitHub (what prototype would send is an unsent item)."""

    def cfg(self, feats=None, **top):
        return base_cfg(pathlib.Path("/w"), feats, **top)

    def test_the_defaults_are_prototype_and_send_nothing(self):
        c = self.cfg()
        on = {n for n in features.CAPABILITIES if features.on(c, n)}
        self.assertEqual(
            on,
            {
                "review-drafts",
                "merge-plan",
                "priority-requests",
                "develop-checks",
                "admin-dashboard",
                "auto-merge-guard",
                "overlap-management",
                # the merge queue's own switches, on by default: each runs, and what it would send is kept
                "holds",
                "queue-notices",
                "drift-notices",
                "stability-issues",
                "incident-issues",
                "fix-links",
            },
        )
        self.assertEqual(
            {f: features.held(c, f) for f in features.SENDS},
            dict.fromkeys(features.SENDS, state_store.PROTOTYPE),
        )
        self.assertFalse(any(features.sends(c, f) for f in features.SENDS))
        self.assertEqual(features.value(c, "auto-merge-guard"), "alert")
        self.assertEqual(features.guard_mode(c), "alert")
        self.assertFalse(features.arms(c))

    def test_the_features_that_are_on_in_one_line(self):
        """on_line, deploy step 8's done line: each feature that is on, with its mode where it has one; and
        on_words, the same words, which rkmon status's first line joins."""
        self.assertEqual(
            features.on_line(self.cfg()),
            "review automation prototype, merge queue prototype, the admin dashboard",
        )
        some = self.cfg(
            {
                "review_automation": {"mode": "live"},
                "merge_queue": {"enabled": False},
                "dashboards": {"team": {"enabled": True}},
                "alerts": LIVE["alerts"],
                "ticket_creation": {"enabled": True},
            }
        )
        self.assertEqual(
            features.on_line(some),
            "review automation live, the admin dashboard, the team dashboard prototype, alerts live, ticket creation "
            "prototype",
        )
        none = self.cfg(
            {
                "review_automation": {"enabled": False},
                "merge_queue": {"enabled": False},
                "dashboards": {"admin": {"enabled": False}},
            }
        )
        self.assertEqual(features.on_line(none), "no feature is on")
        self.assertEqual(
            features.on_words(some),
            [
                "review automation live",
                "the admin dashboard",
                "the team dashboard prototype",
                "alerts live",
                "ticket creation prototype",
            ],
        )
        self.assertEqual(features.on_words(none), [])

    def test_each_feature_s_state_in_one_line(self):
        """headline, deploy step 2's features row and the confirm's headline: each feature's state joined by
        " · ", the merge queue's auto_merge switch after it while it is on."""
        self.assertEqual(
            features.headline(self.cfg({"merge_queue": {"enabled": False}})),
            "review automation prototype · merge queue off · admin dashboard on · team dashboard off · alerts off"
            " · ticket creation off",
        )
        self.assertEqual(
            features.headline(self.cfg()),
            "review automation prototype · merge queue prototype · auto_merge hints_only · admin dashboard on"
            " · team dashboard off · alerts off · ticket creation off",
        )
        every = self.cfg(
            {
                "review_automation": {"mode": "live"},
                "merge_queue": LIVE["merge_queue"],
                "dashboards": {"team": {"enabled": True}},
                "alerts": LIVE["alerts"],
                "ticket_creation": {"enabled": True},
            }
        )
        self.assertEqual(
            features.headline_parts(every),
            [
                "review automation live",
                "merge queue live",
                "auto_merge enabled",
                "admin dashboard on",
                "team dashboard prototype",
                "alerts live",
                "ticket creation prototype",
            ],
        )

    def test_review_automation_live_posts_and_prototype_keeps(self):
        proto = self.cfg()
        live = self.cfg({"review_automation": {"mode": "live"}})
        self.assertFalse(features.on(proto, "review-posting"))
        self.assertTrue(features.on(live, "review-posting"))
        self.assertTrue(
            features.on(proto, "overlap-management")
        )  # found and decided here; in prototype each notice is kept as an unsent item
        self.assertEqual(
            features.filter_lines(proto, ["OVERLAP_NOTICE_DUE #5: k1 (owner #6)"]),
            ["OVERLAP_NOTICE_DUE #5: k1 (owner #6)"],
        )
        self.assertFalse(features.sends(proto, "review_automation"))
        self.assertEqual(
            features.filter_lines(live, ["OVERLAP_NOTICE_DUE #5: k1 (owner #6)"]),
            ["OVERLAP_NOTICE_DUE #5: k1 (owner #6)"],
        )
        self.assertIn(
            "features.review_automation", features.why_off(proto, "review-posting")
        )
        off = self.cfg({"review_automation": {"enabled": False, "mode": "live"}})
        for name in ("review-drafts", "review-posting", "overlap-management"):
            self.assertFalse(features.on(off, name), name)

    def test_merge_queue_live_acts_and_the_develop_guard_follows_its_mode(self):
        proto = self.cfg({"merge_queue": {"auto_merge": "enabled"}})
        live = self.cfg({"merge_queue": {"mode": "live", "auto_merge": "enabled"}})
        for name in (
            "auto-merge",
            "holds",
            "queue-notices",
            "drift-notices",
            "stability-issues",
            "incident-issues",
            "fix-links",
        ):
            # on in both modes: in prototype each runs, and what it would send is kept (it sends for the merge queue)
            self.assertTrue(features.on(proto, name), name)
            self.assertTrue(features.on(live, name), name)
            self.assertEqual(features.CAPABILITIES[name].feature, "merge_queue")
        self.assertEqual(
            (features.sends(proto, "merge_queue"), features.sends(live, "merge_queue")),
            (False, True),
        )
        self.assertEqual(features.value(proto, "auto-merge-guard"), "alert")
        self.assertTrue(
            features.on(proto, "develop-checks")
        )  # computed in prototype too
        self.assertEqual(features.value(live, "auto-merge-guard"), "act")
        hints = self.cfg({"merge_queue": {"mode": "live", "auto_merge": "hints_only"}})
        self.assertFalse(features.on(hints, "auto-merge"))
        no_hold = self.cfg({"merge_queue": {"mode": "live", "auto_hold": False}})
        self.assertFalse(features.on(no_hold, "holds"))
        no_drift = self.cfg({"merge_queue": {"mode": "live", "drift_notices": False}})
        self.assertFalse(features.on(no_drift, "drift-notices"))
        self.assertTrue(features.on(no_drift, "queue-notices"))
        no_guard = self.cfg({"merge_queue": {"mode": "live", "develop_guard": False}})
        for name in (
            "develop-checks",
            "stability-issues",
            "incident-issues",
            "fix-links",
        ):
            self.assertFalse(features.on(no_guard, name), name)
        off = self.cfg({"merge_queue": {"enabled": False}})
        for name in ("merge-plan", "priority-requests", "develop-checks"):
            self.assertFalse(features.on(off, name), name)
        self.assertEqual(features.value(off, "auto-merge-guard"), "off")

    def test_dashboards_alerts_and_tickets_are_their_switches(self):
        c = self.cfg(
            {
                "dashboards": {"team": {"enabled": True}},
                "alerts": LIVE["alerts"],
                "ticket_creation": {
                    "enabled": True,
                    "triggers": {"review_follow_up": False, "out_of_scope_issue": True},
                },
            }
        )
        for name in (
            "team-dashboard",
            "chat-alerts",
            "needs-you-alerts",
            "found-issue-tickets",
        ):
            self.assertTrue(features.on(c, name), name)
        self.assertFalse(features.on(c, "follow-up-tickets"))
        self.assertFalse(
            features.on(
                self.cfg({"dashboards": {"admin": {"enabled": False}}}),
                "admin-dashboard",
            )
        )

    def test_each_kind_of_alert_goes_out_while_its_own_switch_and_the_alerts_are_on(
        self,
    ):
        for breaks, needs_you in ((True, False), (False, True)):
            with self.subTest(develop_breaks=breaks, needs_you=needs_you):
                alerts = {
                    "enabled": True,
                    "develop_breaks": breaks,
                    "needs_you": needs_you,
                }
                c = self.cfg({"alerts": alerts})
                self.assertEqual(
                    [features.on(c, n) for n in ("chat-alerts", "needs-you-alerts")],
                    [breaks, needs_you],
                )
                self.assertIn(
                    f"- `features.alerts`: on, prototype, develop_breaks {str(breaks).lower()}, "
                    f"needs_you {str(needs_you).lower()}",
                    features.feature_lines(c),
                )
        off = self.cfg({"alerts": dict(LIVE["alerts"], enabled=False)})
        self.assertEqual(
            [features.on(off, n) for n in ("chat-alerts", "needs-you-alerts")],
            [False, False],
        )
        self.assertIn("- `features.alerts`: off", features.feature_lines(off))
        self.assertEqual(
            features.why_off(off, "needs-you-alerts"),
            "it runs while features.alerts.needs_you.enabled is true",
        )

    def test_the_team_dashboards_value_is_where_it_goes(self):
        on = {"dashboards": {"team": {"enabled": True}}}
        published = self.cfg(on)  # the default delivery
        served = self.cfg(on, team_dashboard={"mode": "serve"})
        off = self.cfg(
            {"dashboards": {"team": {"enabled": False}}},
            team_dashboard={"mode": "serve"},
        )
        self.assertEqual(
            [features.value(c, "team-dashboard") for c in (published, served, off)],
            ["publish", "serve", "off"],
        )
        self.assertEqual(
            [features.on(c, "team-dashboard") for c in (published, served, off)],
            [True, True, False],
        )
        self.assertEqual(
            features.CAPABILITIES["team-dashboard"].values, ("off", "publish", "serve")
        )

    def test_the_advanced_switches(self):
        self.assertTrue(
            features.on(self.cfg(gpu={"enabled": True}), "remote-gpu-tests")
        )
        self.assertFalse(
            features.on(
                self.cfg(
                    {"review_automation": {"enabled": False}}, gpu={"enabled": True}
                ),
                "remote-gpu-tests",
            )
        )
        self.assertTrue(
            features.on(self.cfg(weekly_report={"enabled": True}), "weekly-report")
        )
        self.assertNotIn(
            "jira-mcp", features.CAPABILITIES
        )  # the JIRA MCP server is an unused feature

    def test_the_claude_session_runs_with_review_automation_in_the_container(self):
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"}):
            self.assertTrue(features.on(self.cfg(), "headless-claude"))
            off = self.cfg({"review_automation": {"enabled": False}})
            self.assertFalse(features.on(off, "headless-claude"))
            self.assertEqual(
                features.components(off),
                {
                    "web": True,
                    "claude": False,
                    "reviews": False,
                    "rules": True,
                    "merge-queue": True,
                    "develop-checks": True,
                },
            )
        # Outside the container no Claude session runs, review automation on or off.
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": ""}):
            for enabled in (True, False):
                cfg = self.cfg({"review_automation": {"enabled": enabled}})
                self.assertFalse(features.on(cfg, "headless-claude"))

    def test_a_switch_outside_its_choices_and_an_unknown_one_are_refused(self):
        bad = features.problems({"features": {"merge_queue": {"mode": "livee"}}})
        self.assertTrue(
            any("features.merge_queue.mode is 'livee'" in p for p in bad), bad
        )
        self.assertTrue(
            any(
                "is not a feature switch" in p
                for p in features.problems({"features": {"teleport": True}})
            )
        )
        self.assertTrue(
            any(
                "is not a feature switch" in p
                for p in features.problems({"features": {"review-posting": True}})
            )
        )


class TestStandby(unittest.TestCase):
    """Standby is the whole monitor's: `standby.on` in the internal state, read again whenever the file
    changes; while it is on no feature sends, whatever its mode; a state that cannot be read counts as standby.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = pathlib.Path(tmp.name)
        env = mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)

    def test_held_says_standby_first_then_prototype(self):
        from rkmon_lib import state_store

        live = base_cfg(self.home, LIVE)
        proto = base_cfg(self.home, {"merge_queue": {"mode": "prototype"}})
        self.assertEqual(features.held(live, "merge_queue"), "")
        self.assertTrue(features.sends(live, "merge_queue"))
        self.assertEqual(features.held(proto, "merge_queue"), "prototype")
        self.assertFalse(features.sends(proto, "merge_queue"))
        state_store.set(self.home, "standby.on", True)
        self.assertTrue(features.standby())
        self.assertEqual(features.held(live, "merge_queue"), "standby")
        self.assertEqual(features.held(proto, "merge_queue"), "standby")
        self.assertFalse(features.sends(live, "merge_queue"))
        state_store.set(self.home, "standby.on", False)
        self.assertEqual(features.held(live, "merge_queue"), "")

    def test_a_state_that_cannot_be_read_counts_as_standby(self):
        from rkmon_lib import state_store

        self.assertFalse(features.standby())  # no internal state yet: none set it
        p = state_store.path(self.home)
        p.parent.mkdir(parents=True)
        p.write_text("{not json", encoding="utf-8")
        self.assertTrue(features.standby())
        self.assertEqual(features.held(base_cfg(self.home, LIVE), "alerts"), "standby")
        p.unlink()
        self.assertFalse(features.standby())

    def test_the_mark_of_an_adoption_is_standby_whatever_the_internal_state_says(self):
        """From before rkmon adopt moves anything until standby off removes it, the mark of the adoption in the
        monitor's home (adopt.marked) is standby: with the internal state's standby.on false, and with none.
        """
        from rkmon_lib import adopt, state_store

        state_store.set(self.home, "standby.on", False)
        mark = {"kind": adopt.MARK_KIND, "applied_at": ""}
        state_store.write_file(adopt.mark_file(self.home), mark)
        self.assertTrue(features.standby())
        self.assertEqual(features.held(base_cfg(self.home, LIVE), "alerts"), "standby")
        state_store.path(self.home).unlink()
        self.assertTrue(features.standby())
        adopt.unmark(self.home)
        self.assertFalse(features.standby())


class TestCoherence(unittest.TestCase):
    def rules(self, feats=None, **top) -> list:
        return features.coherence(base_cfg(pathlib.Path("/w"), feats, **top))

    def whats(self, status, feats=None, **top) -> list:
        return [w for s, w, _f, _st in self.rules(feats, **top) if s == status]

    def test_the_defaults_are_coherent(self):
        # The starting templates are the deploy schema's profiles: test_settings_schemas checks that each one's file
        # passes its own checks.
        self.assertEqual(self.whats("FAIL"), [])

    def test_each_combination_that_cannot_run(self):
        cases = (
            (
                {
                    "dashboards": {
                        "admin": {"enabled": False},
                        "team": {"enabled": True},
                    }
                },
                {},
                "team dashboard",
            ),
            (
                {
                    "dashboards": {"admin": {"enabled": False}},
                    "ticket_creation": LIVE["ticket_creation"],
                },
                {"jira": {"project": "P"}, "atlassian": {"site": "s"}},
                deploy_tree.name(
                    "features.ticket_creation.triggers.out_of_scope_issue"
                ),
            ),
            (
                {
                    "review_automation": {"enabled": False},
                    "ticket_creation": LIVE["ticket_creation"],
                },
                {"jira": {"project": "P"}, "atlassian": {"site": "s"}},
                deploy_tree.name("features.ticket_creation.triggers.review_follow_up"),
            ),
            (
                {"review_automation": {"enabled": False}},
                {"gpu": {"enabled": True}},
                deploy_tree.name("gpu.enabled"),
            ),
            (
                {"ticket_creation": LIVE["ticket_creation"]},
                {"jira": {"project": "P"}},
                deploy_tree.name("atlassian.site"),
            ),
            (
                {"ticket_creation": LIVE["ticket_creation"]},
                {"atlassian": {"site": "s"}},
                deploy_tree.name("jira.project"),
            ),
            (
                {"dashboards": {"team": {"enabled": True}}},
                {},
                deploy_tree.name("publish.html.host"),
            ),
        )
        for feats, top, word in cases:
            with self.subTest(word=word):
                fails = self.whats("FAIL", feats, **top)
                self.assertTrue(any(word in w for w in fails), fails)

    def test_a_published_team_page_needs_its_web_host_its_login_and_its_site_folder(
        self,
    ):
        """Published to the web host, the team dashboard needs the host, the login and the site folder, which the
        page's place there is derived from: the FAIL names the blank ones, and its fix sets each, with where its value
        comes from, serves the page, or turns the dashboard off."""
        team = {"dashboards": {"team": {"enabled": True}}}
        web = {"publish": {"html": {"host": "web.example.com", "user": "op"}}}

        def fails(top: dict) -> list:
            return [
                (w, str(f))
                for s, w, f, _st in self.rules(team, **top)
                if s == "FAIL" and "(delivery publish)" in w
            ]

        self.assertEqual(
            fails(web),
            [
                (
                    "features.dashboards.team is enabled (delivery publish) but these are blank: "
                    "features.dashboards.team.publish.site_folder",
                    "set features.dashboards.team.publish.site_folder in your deploy settings file to the folder the "
                    'web host serves the page from, relative to that user\'s home folder, such as "public_html/my-team"'
                    ', or set features.dashboards.team.delivery in your deploy settings file to "serve", or set '
                    "features.dashboards.team.enabled in your deploy settings file to false",
                )
            ],
        )
        self.assertEqual(
            features.missing(
                base_cfg(pathlib.Path("/w"), team, **web), "team-dashboard"
            )[:1],
            [
                "features.dashboards.team.publish.site_folder (the folder the web host serves the page from)"
            ],
        )  # what the capability lacks names the deploy settings' keys too
        web["publish"]["html"]["site"] = "public_html/my-team"
        self.assertEqual(fails(web), [])

    def test_the_structural_ones_are_configuration_problems(self):
        cfg = base_cfg(
            pathlib.Path("/w"),
            {"dashboards": {"admin": {"enabled": False}, "team": {"enabled": True}}},
        )
        self.assertTrue(
            any("features.dashboards.team" in p for p in features.problems(cfg))
        )
        settings_only = base_cfg(pathlib.Path("/w"), {"alerts": LIVE["alerts"]})
        self.assertFalse(
            any("teams_channel" in p for p in features.problems(settings_only))
        )

    def test_a_blank_message_label_fails_nothing(self):
        """The message label only labels each message, and the webhook decides the channel: alerts on with a
        blank label meet no rule, lack their webhook alone, and say the label is none.
        """
        cfg = base_cfg(pathlib.Path("/w"), {"alerts": LIVE["alerts"]})
        self.assertEqual(features.coherence(cfg), [])
        lacks = features.missing(cfg, "chat-alerts")
        self.assertEqual(len(lacks), 1, lacks)
        self.assertIn("the channel's incoming-webhook URL", lacks[0])
        self.assertIn(
            "message label none", features.configured_with(cfg, "chat-alerts")
        )

    def test_a_blank_develop_label_is_not_set(self):
        """The develop guard's labels as the feature lines show them: as the settings name them, and "not set" while
        one is blank."""
        cfg = base_cfg(pathlib.Path("/w"))
        cfg["incident_issue"] = {
            "labels": ["", guard_labels.EXTRA],
            "stability_label": "",
        }
        for name in ("incident-issues", "stability-issues"):
            self.assertEqual(features.configured_with(cfg, name), "label not set")
        cfg["incident_issue"] = guard_labels.INCIDENT_ISSUE
        self.assertEqual(
            features.configured_with(cfg, "incident-issues"),
            f"label {guard_labels.RED}",
        )
        self.assertEqual(
            features.configured_with(cfg, "stability-issues"),
            f"label {guard_labels.STABILITY}",
        )

    def test_priority_requests_name_the_command_as_the_operator_types_it(self):
        """Where priority requests come from names rk prioritize as the operator types it on the host, through the
        container's wrapper (doctor.rk_command)."""
        cfg = base_cfg(pathlib.Path("/w"))
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            self.assertEqual(
                features.configured_with(cfg, "priority-requests"),
                "the admin dashboard and ./rkmon rk prioritize",
            )

    def test_the_warnings(self):
        warns = self.whats("WARN", {"merge_queue": {"auto_merge": "enabled"}})
        self.assertTrue(
            any("arms nothing" in w or "nothing is armed" in w for w in warns), warns
        )
        warns = self.whats(
            "WARN",
            {
                "review_automation": {"enabled": False},
                "merge_queue": {"enabled": False},
            },
        )
        self.assertTrue(any("only reads GitHub" in w for w in warns), warns)

    def test_a_live_merge_queue_beside_reviews_in_prototype_is_a_warning(self):
        """The merge queue live while review automation is in prototype gates on unposted reviews their authors never saw:
        a WARN, whose fix sets review automation live or keeps the merge queue in prototype, among the rows rkmon
        check and rkmon confirm show (deployment.settings_checks). Both live, both in prototype, the merge queue off
        though its mode is live, or review automation off: no such WARN."""
        from rkmon_lib import checks, deployment

        def warns(feats: dict) -> list:
            return [(w, str(f)) for s, w, f, _st in self.rules(feats) if s == "WARN"]

        warn = (
            "features.merge_queue is live, but review automation is in prototype: the queue gates on unposted "
            "reviews their authors never saw"
        )
        fix = (
            'set features.review_automation.mode in your deploy settings file to "live", or set '
            'features.merge_queue.mode in your deploy settings file to "prototype"'
        )
        self.assertEqual(warns({"merge_queue": {"mode": "live"}}), [(warn, fix)])
        cfg = base_cfg(pathlib.Path("/w"), {"merge_queue": {"mode": "live"}})
        rows = [c for c in deployment.settings_checks(cfg) if c.name == "features"]
        self.assertEqual(
            [(r.status, r.found, str(r.fix)) for r in rows],
            [(checks.Status.WARN, warn, fix)],
        )
        for feats in (
            {"merge_queue": {"mode": "live"}, "review_automation": {"mode": "live"}},
            {"merge_queue": {"mode": "prototype"}},
            {"merge_queue": {"enabled": False, "mode": "live"}},
            {"merge_queue": {"mode": "live"}, "review_automation": {"enabled": False}},
        ):
            with self.subTest(feats=feats):
                self.assertFalse(
                    [w for w, _f in warns(feats) if "never saw" in w], warns(feats)
                )

    def test_each_fix_edits_a_key_of_the_deploy_settings_file(self):
        """Every rule's fix is a checks fix (rkmon_lib/checks.py). Each edit in it names the deploy settings file (in
        words, deploy_tree.FILE, unless a path is given) and a key that file holds, never a section: one of its
        schema's leaves. A switch a rule makes of several keys, such as alerts of each type's, is turned off at each.
        """
        from rkmon_lib import checks, deploy_tree, file_types, schema

        keys = dict(schema.leaves(file_types.DEPLOY.load()))
        self.assertNotIn("features.alerts", keys)
        self.assertIn("features.alerts.needs_you.enabled", keys)
        jira = {"jira": {"project": "P"}, "atlassian": {"site": "s"}}
        tickets = LIVE["ticket_creation"]
        cases = (
            (
                {
                    "dashboards": {
                        "admin": {"enabled": False},
                        "team": {"enabled": True},
                    }
                },
                {},
            ),
            (
                {
                    "dashboards": {"admin": {"enabled": False}},
                    "ticket_creation": tickets,
                },
                jira,
            ),
            (
                {
                    "review_automation": {"enabled": False},
                    "ticket_creation": tickets,
                },
                jira,
            ),
            ({"review_automation": {"enabled": False}}, {"gpu": {"enabled": True}}),
            ({"ticket_creation": tickets}, {"jira": {"project": "P"}}),
            ({"ticket_creation": tickets}, {"atlassian": {"site": "s"}}),
            ({"dashboards": {"team": {"enabled": True}}}, {}),
            ({"merge_queue": {"auto_merge": "enabled"}}, {}),
            ({"merge_queue": {"mode": "live"}}, {}),
            ({"ticket_creation": tickets}, jira),
            (
                {
                    "review_automation": {"enabled": False},
                    "merge_queue": {"enabled": False},
                },
                {},
            ),
            (
                {
                    "alerts": LIVE["alerts"],
                    "merge_queue": {"develop_guard": False},
                    "dashboards": {"admin": {"enabled": False}},
                },
                {"teams": {"channel": "c"}},
            ),
        )
        rules = set()
        for feats, top in cases:
            found = self.rules(feats, **top)
            self.assertTrue(found, feats)
            for _status, what, fix, _structural in found:
                rules.add(what)
                with self.subTest(what=what):
                    self.assertIsInstance(
                        fix, (checks.Command, checks.Edit, checks.Steps)
                    )
                    parts = fix.parts if isinstance(fix, checks.Steps) else (fix,)
                    for e in [p for p in parts if isinstance(p, checks.Edit)]:
                        self.assertEqual(e.file, deploy_tree.FILE)
                        self.assertIn(e.key, keys)
        self.assertGreaterEqual(len(rules), 11, sorted(rules))
        [(_s, _w, fix, _st)] = features.coherence(
            base_cfg(
                pathlib.Path("/w"),
                {
                    "alerts": LIVE["alerts"],
                    "merge_queue": {"develop_guard": False},
                    "dashboards": {"admin": {"enabled": False}},
                },
            ),
            "/srv/d.yaml",
        )
        self.assertIn(
            "set features.dashboards.admin.enabled in /srv/d.yaml to true", str(fix)
        )
        self.assertIn(
            "set features.alerts.develop_breaks.enabled in /srv/d.yaml to false, and set "
            "features.alerts.needs_you.enabled in /srv/d.yaml to false",
            str(fix),
        )


class TestOffNeverActs(Base):
    def test_each_command_of_a_capability_that_is_off_does_nothing_and_exits_0(self):
        for name, (_feats, _top, argv) in COMMANDS.items():
            with self.subTest(capability=name):
                args = [a.format(file=self.file) for a in argv]
                off = (_top or {}).get(
                    "_off"
                )  # a capability on by default: its own way off
                rc, out, ledger = self.rk(base_cfg(self.root, off), *args)
                self.assertEqual(rc, features.EXIT_OFF, out)
                self.assertIn("OFF:", out)
                self.assertEqual(ledger, [])
                self.assertNotIn("Traceback", out)

    def test_issue_relabel_takes_the_kinds_as_breakage_names_them(self):
        """rk issue-relabel's --to takes the two kinds by the names rkmon_lib/breakage.py gives them, and each is gated
        by its own capability: the kinds' names live there alone."""
        with mock.patch.multiple(breakage, RED="kind-a", STABILITY="kind-b"):
            for kind, name in (
                ("kind-a", "incident-issues"),
                ("kind-b", "stability-issues"),
            ):
                with self.subTest(kind=kind):
                    rc, out, ledger = self.rk(
                        base_cfg(self.root, {"merge_queue": {"develop_guard": False}}),
                        "issue-relabel",
                        "7",
                        "--to",
                        kind,
                    )
                    self.assertEqual(rc, features.EXIT_OFF, out)
                    title = features.CAPABILITIES[name].title
                    self.assertIn(f"OFF: {features.lowered(title)} is off", out)
                    self.assertEqual(ledger, [])

    def test_with_the_queue_notices_switch_off_no_queue_notice_goes_out(self):
        """The queue notices' switch (the deploy file's features.merge_queue.queue_notices.enabled) decides while the
        merge queue is live: off, the tick prints no queue notice line, and no queue notice goes out, whoever asks, the
        tick's rule or the session: not through rk queue-notice, rk notice --kind queue, nor the write itself.
        """
        queue = dict(LIVE["merge_queue"], queue_notices=False)
        cfg = base_cfg(self.root, {"merge_queue": queue})
        on = base_cfg(self.root, {"merge_queue": dict(queue, queue_notices=True)})
        self.assertEqual(
            (features.on(on, "queue-notices"), features.on(cfg, "queue-notices")),
            (True, False),
        )
        self.assertIn(
            "its queue_notices.enabled is true", features.why_off(cfg, "queue-notices")
        )
        line = "QUEUE_NOTICE #7: next (abcdef0123); run `rk queue-notice 7`"
        self.assertEqual(features.filter_lines(cfg, [line]), [])
        for argv in (
            ["queue-notice", "7"],
            ["notice", "7", str(self.file), "--kind", "queue", "--key", "k"],
        ):
            with self.subTest(command=argv[0]):
                rc, out, ledger = self.rk(cfg, *argv)
                self.assertEqual(rc, features.EXIT_OFF, out)
                self.assertIn("OFF: queue notices", out)
                self.assertIn("its queue_notices.enabled is true", out)
                self.assertEqual(ledger, [])
        store = Store(self.root / "state")
        with _NoNetwork(), self.assertRaises(post.PostRefused) as cm:
            post.queue_notice(cfg, empty_state(), store, 7, "text", "k")
        self.assertIn("its queue_notices.enabled is true", str(cm.exception))
        self.assertEqual(store.ledger(), [])

    def test_posting_in_prototype_keeps_an_unposted_review_and_nothing_else(self):
        """The monitor's own post in prototype keeps an unposted review, an unsent item too; one you type is refused,
        naming the mode. Nothing else."""
        held = "features.review_automation.mode is prototype: nothing goes out until it is live"
        rc, out, ledger = self.rk(base_cfg(self.root), "post", "7", str(self.file))
        self.assertEqual(
            (rc, out, ledger), (4, f"REFUSED: a review post is not sent: {held}\n", [])
        )
        self.assertEqual(list((self.root / "work").glob("PR_UNPOSTED_7*.md")), [])
        rc, out, ledger = self.rk(
            base_cfg(self.root), "post", "7", str(self.file), own=True
        )
        self.assertEqual(rc, 0, out)
        self.assertIn(
            f"UNPOSTED: {held}, so the verified review stays unposted, kept as an unsent item",
            out,
        )
        unposted = list((self.root / "work").glob("PR_UNPOSTED_7*.md"))
        self.assertEqual(len(unposted), 1)
        self.assertEqual(ledger, [])
        self.assertEqual(
            [
                (i["type"], i["file"])
                for i in state_store.unsent_items(self.root / "home" / "state")
            ],
            [(state_store.REVIEW, str(unposted[0]))],
        )

    def test_the_team_dashboard_is_not_pushed_when_off(self):
        cfg = base_cfg(self.root)
        cfg["publish"]["html"].update(
            host="web.example", user="u", dest="d/index.html", key=str(self.file)
        )
        with _NoNetwork():
            self.assertIsNone(publish.html(cfg, str(self.file)))
            self.assertIsNone(
                publish.pull_requests(
                    dict(
                        cfg,
                        publish={
                            "html": dict(cfg["publish"]["html"], requests="r.jsonl")
                        },
                    ),
                    "/w/x",
                )
            )

    def test_tick_lines_of_capabilities_that_are_off_are_not_printed(self):
        cfg = base_cfg(
            self.root,
            {
                "review_automation": {"enabled": False},
                "merge_queue": {
                    "auto_hold": False,
                    "queue_notices": False,
                    "drift_notices": False,
                    "develop_guard": False,
                },
            },
        )
        lines = [
            "QUEUE_NOTICE #7: x",
            "ARM_READY #7: z",
            "TRIAGE #7",
            "DRIFT_FAIL #7",
            "FIX_CANDIDATE #7 -> #1",
            "OVERLAP_NOTICE_DUE #7",
            "WEEKLY REPORT W ready",
            "REVERT_OFFER x",
            "REVIEW_READY #7 (first-round)",
            "PRIORITY_REQUEST #7 by someone: why",
            "INCIDENT_OPEN x",
        ]
        self.assertEqual(
            features.filter_lines(cfg, lines),
            [
                "REVIEW_READY #7 (first-round)",
                "PRIORITY_REQUEST #7 by someone: why",
                "INCIDENT_OPEN x",
            ],
        )


class TestPrototypeRunsAndSendsNothing(Base):
    """In prototype each capability that is on runs; what it would send waits as an unsent item."""

    def test_the_tick_prints_the_lines_of_what_runs_in_prototype(self):
        lines = [
            "QUEUE_NOTICE #7: x",
            "ARM_READY #7: z",
            "TRIAGE #7",
            "DRIFT_FAIL #7",
            "FIX_CANDIDATE #7 -> #1",
            "OVERLAP_NOTICE_DUE #7",
            "WEEKLY REPORT W ready",
            "REVERT_OFFER x",
        ]
        self.assertEqual(
            features.filter_lines(base_cfg(self.root), lines),
            [ln for ln in lines if not ln.startswith(("ARM_READY", "WEEKLY"))],
        )  # auto_merge is hints_only and the weekly report off by default

    def test_the_monitors_own_arm_keeps_an_unsent_item_and_arms_nothing(self):
        """The merge queue in prototype runs its arming to its last check; the monitor's own arm then keeps an unsent
        item of type auto-merge: nothing is armed, ledgered or sent, and the journal's line names no content.
        """
        st = empty_state()
        st["prs"]["7"] = {"n": 7, "head": "a" * 40}
        st["cleared"] = [7]
        st["batches"] = [{"members": [7], "status": "validated"}]
        cfg = {
            "features": {"merge_queue": {"mode": "prototype", "auto_merge": "enabled"}},
            "repo": "ROCm/r",
        }
        live = {
            "state": "open",
            "head": "a" * 40,
            "draft": False,
            "mergeable": True,
            "mergeable_state": "clean",
        }
        store = Store(self.root / "state")
        with mock.patch.object(gh, "rest", return_value=live), mock.patch.object(
            post.subprocess, "run", side_effect=AssertionError("armed")
        ), mock.patch.dict(os.environ, {outward.OWN_VAR: "1"}):
            out = post.arm(cfg, st, store, 7)
        self.assertEqual(
            out,
            "UNSENT: auto-merge for pull request 7, kept as an unsent item: "
            "features.merge_queue.mode is prototype: nothing goes out until it is live",
        )
        self.assertIsNone((st.get("queue_state") or {}).get("armed"))
        self.assertEqual(store.ledger(), [])
        (item,) = state_store.unsent_items(store.dir)
        self.assertEqual(
            (item["type"], item["for"], item["content"]),
            (
                state_store.AUTO_MERGE,
                "pull request 7",
                f"auto-merge (squash) at head {'a' * 40}",
            ),
        )
        self.assertEqual(
            [ln.split(" unsent ", 1)[1].strip() for ln in store.journal_lines()],
            ["auto-merge for pull request 7: not sent (the merge queue in prototype)"],
        )


class TestOnButNotSetUp(Base):
    def test_each_command_skips_cleanly_and_says_what_is_missing(self):
        for name, (feats, top, argv) in COMMANDS.items():
            with self.subTest(capability=name):
                args = [a.format(file=self.file) for a in argv]
                rc, out, ledger = self.rk(
                    base_cfg(self.root, feats, **top), *args, path_without_gh=True
                )
                self.assertEqual(rc, features.EXIT_NOT_SET_UP, out)
                self.assertIn("NOT SET UP:", out)
                self.assertEqual(ledger, [])
                self.assertNotIn("Traceback", out)

    def test_the_tick_reports_a_missing_dependency_once(self):
        cfg = base_cfg(self.root, {"alerts": LIVE["alerts"]})
        st = empty_state()
        first = [
            x for x in features.tick_reports(cfg, st) if "develop break alerts" in x
        ]
        self.assertEqual(len(first), 1)
        self.assertIn("the channel's incoming-webhook URL", first[0])
        self.assertFalse(
            [x for x in features.tick_reports(cfg, st) if "develop break alerts" in x]
        )  # once

    def test_the_fix_is_the_same_in_every_report(self):
        """A capability's fix is a checks fix: what sets up what it lacks (features.needs_fix: for the team chat, its
        webhook's secret command and what to give it), as the operator types it on the host wherever this runs, the
        same in the refusal, the tick's report, rk features and rk doctor's row, which checks.render prints.
        """
        from rkmon_lib import checks, doctor

        cfg = base_cfg(self.root, {"alerts": LIVE["alerts"]})
        give = ", and give an https:// address on one line"
        for env, want in (
            (
                {"RKMON_CONTAINER": "", "RKMON_WRAPPER": ""},
                "rkmon secret teams_webhook" + give,
            ),
            (
                {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"},
                "./rkmon secret teams_webhook" + give,
            ),
        ):
            with self.subTest(want), mock.patch.dict(os.environ, env):
                self.assertEqual(str(features.fix(cfg, "chat-alerts")), want)
                self.assertTrue(
                    features.refusal(cfg, "chat-alerts")[1].endswith(
                        f"Nothing was done; fix: {want}"
                    )
                )
                said = [
                    x
                    for x in features.tick_reports(cfg, empty_state())
                    if "develop break alerts" in x
                ]
                self.assertTrue(said and said[0].endswith(f"(fix: {want})"), said)
                self.assertIn(f"(fix: {want})", features.summary(cfg))
                row = next(
                    c
                    for c in doctor.feature_checks(cfg)
                    if c.name == "develop break alerts"
                )
                self.assertEqual(row.status, "FAIL")
                self.assertTrue(
                    checks.render([row]).endswith(f"\n        fix: {want}"), row
                )

    def test_the_develop_guard_lacks_the_image_s_parts(self):
        """The checks venv and ROCm's LLVM tools are the image's (rkmon_lib/derived.py): what the develop guard lacks
        of them is the image's, and its fix checks the image, the same in the refusal and rk doctor's row; one check
        names every part the image lacks, wherever this runs.
        """
        from rkmon_lib import checks, doctor

        venv = self.root / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        cfg = base_cfg(
            self.root,
            {"merge_queue": LIVE["merge_queue"]},
            paths={"venv": str(venv), "rocm_llvm_bin": ""},
        )
        name = "develop-checks"
        box = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        want = "./rkmon image check (the image lacks ROCm's LLVM tools)"
        with mock.patch.dict(os.environ, box), mock.patch(
            "shutil.which", return_value=None
        ):
            self.assertEqual(
                features.missing(cfg, name), ["ROCm's LLVM tools (from the image)"]
            )
            self.assertEqual(str(features.fix(cfg, name)), want)
            self.assertTrue(
                features.refusal(cfg, name)[1].endswith(
                    f"Nothing was done; fix: {want}"
                )
            )
            row = next(
                c for c in doctor.feature_checks(cfg) if c.name == "develop guard"
            )
            self.assertEqual(row.status, "FAIL")
            self.assertTrue(
                checks.render([row]).endswith(f"\n        fix: {want}"), row
            )
            cfg["paths"]["venv"] = str(self.root / "no-venv")
            self.assertEqual(
                features.missing(cfg, name),
                ["the checks venv and ROCm's LLVM tools (from the image)"],
            )
            self.assertEqual(
                str(features.fix(cfg, name)),
                "./rkmon image check (the image lacks the checks venv and ROCm's LLVM tools)",
            )
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "", "RKMON_WRAPPER": ""}):
            self.assertEqual(
                features.missing(cfg, name),
                ["the checks venv and ROCm's LLVM tools (from the image)"],
            )
            self.assertEqual(
                str(features.fix(cfg, name)),
                "rkmon image check (the image lacks the checks venv and ROCm's LLVM tools)",
            )

    def test_what_acts_on_github_is_fixed_by_what_it_lacks(self):
        """The capabilities that act on GitHub have no settings section of their own (Capability.section): their fix
        comes from what they lack (missing), as the operator types it on the host wherever this runs, never words
        alone. The account is a
        setting: an edit that names its key in the deploy settings file, then that file confirmed. The gh CLI is the
        image's: the image's check."""
        from rkmon_lib import checks, doctor

        cfg = base_cfg(
            self.root,
            {
                "review_automation": {"enabled": True, "mode": "live"},
                "merge_queue": {
                    "enabled": True,
                    "mode": "live",
                    "auto_merge": "enabled",
                    "auto_hold": True,
                    "develop_guard": True,
                },
            },
            operator={"github_login": ""},
        )
        names = [n for n, c in features.CAPABILITIES.items() if c.check == "_gh"]
        self.assertEqual(len(names), 10)
        box = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        login = "the GitHub login the monitor posts as"
        with mock.patch("shutil.which", return_value=None):
            for name in names:
                with self.subTest(name):
                    self.assertEqual(features.CAPABILITIES[name].section, "")
                    self.assertEqual(
                        features.missing(cfg, name),
                        [
                            f"integrations.github_cli.username ({login})",
                            "the gh CLI (from the image)",
                        ],
                    )
                    for env, w in (({}, "rkmon"), (box, "./rkmon")):
                        self.assertEqual(
                            str(features.fix(cfg, name, environ=env)),
                            f"set integrations.github_cli.username in your deploy settings file to {login}, then "
                            f"{w} confirm and {w} apply; then {w} image check (the image lacks the gh CLI)",
                        )
            with mock.patch.dict(os.environ, box):
                row = next(
                    c for c in doctor.feature_checks(cfg) if c.name == "auto-merge"
                )
            self.assertIn(
                "fix: set integrations.github_cli.username in your deploy settings file",
                checks.render([row]),
            )
            cfg["operator"]["github_login"] = "op-login"
            self.assertEqual(
                str(features.fix(cfg, "holds", environ={})),
                "rkmon image check (the image lacks the gh CLI)",
            )

    def test_a_login_it_lacks_is_set_by_its_command_as_typed_where_the_fix_is_shown(
        self,
    ):
        """A login or a secret a capability lacks (Need.secret) is set by its own command, named as the operator types
        it in the place the fix is made for (features.fix's ENVIRON: in the container, the wrapper that runs it
        there), as each other part of a fix is, whatever this process's own environment says.
        """
        cfg = base_cfg(self.root, {"review_automation": {"enabled": True}})
        box = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        unset = {"set": False, "where": "the monitor's secrets"}
        with mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": ""}
        ), mock.patch(
            "shutil.which", return_value="/usr/bin/claude"
        ), mock.patch.object(
            features, "_secret_found", return_value=unset
        ):
            self.assertEqual(
                [str(n) for n in features.missing(cfg, "headless-claude")],
                ["the Claude session's key (the login claude: rkmon login claude)"],
            )
            fix = str(features.fix(cfg, "headless-claude", environ=box))
        self.assertTrue(fix.startswith("./rkmon login claude, and give "), fix)

    def test_no_need_names_a_key_the_deploy_settings_file_has_not(self):
        """What a capability lacks is never a key with no deploy key: the Claude session's workspace is the container's
        own (derived.claude_workspace), and the admin dashboard is published on the host's loopback alone
        (deploy_plan.env_values: RKMON_ADMIN_HOST 127.0.0.1), so neither claude_workspace nor admin_dashboard.url is
        lacking."""
        from rkmon_lib import container_layout, webserve

        cfg = base_cfg(self.root, {"review_automation": {"enabled": True}})
        cfg.pop("claude_workspace", None)
        found = {"set": True, "where": "the monitor's secrets"}
        deployment = {
            "RKMON_CONTAINER": "1",
            "RKMON_ADMIN_HOST": container_layout.LOOPBACK,
        }
        with mock.patch.dict(os.environ, deployment), mock.patch(
            "shutil.which", return_value="/usr/bin/claude"
        ), mock.patch.object(
            features, "_secret_found", return_value=found
        ), mock.patch.object(
            webserve, "access_token", return_value="a-token"
        ):
            self.assertIsNone(webserve.exposure(cfg))
            self.assertEqual(features.missing(cfg, "headless-claude"), [])
            self.assertEqual(features.missing(cfg, "admin-dashboard"), [])

    def lacks_only_its_key(self, cfg: dict, name: str, want: str) -> None:
        """Capability NAME, on with configuration CFG, lacks one thing, a secret: its fix, and the NOT SET UP refusal
        that ends with it, are WANT, that secret's command as the operator types it on the host, and no settings edit.
        """
        box = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        with mock.patch.dict(os.environ, box):
            self.assertEqual(len(features.missing(cfg, name)), 1)
            self.assertEqual(str(features.fix(cfg, name)), want)
            code, msg = features.refusal(cfg, name)
        self.assertEqual(code, features.EXIT_NOT_SET_UP)
        self.assertTrue(msg.endswith(f"Nothing was done; fix: {want}"), msg)

    def test_a_published_team_page_that_lacks_only_its_key(self):
        cfg = base_cfg(
            self.root,
            {"dashboards": {"team": {"enabled": True, "mode": "live"}}},
            publish={
                "html": {
                    "host": "web.example.com",
                    "user": "op",
                    "site": "public_html/my-team",
                }
            },
            paths={"secrets_dir": str(self.root / "secrets")},
        )
        self.lacks_only_its_key(
            cfg,
            "team-dashboard",
            "./rkmon secret web_host_ssh_key, and give an SSH private key: the whole key file, from its BEGIN line to "
            "its END line",
        )

    def gateway(self) -> dict:
        """Reviews whose first pass is a gateway binding, every setting given; its key, the secret gw_api_key, is not
        set."""
        return base_cfg(
            self.root,
            {"review_automation": LIVE["review_automation"]},
            reviewer="gw",
            model_bindings={
                "gw": {
                    "provider": "openai-compatible",
                    "url": "https://gw.example.com/v1",
                    "model": "m1",
                    "api_key_secret": "gw_api_key",
                }
            },
            paths={"secrets_dir": str(self.root / "secrets")},
        )

    def test_review_drafts_whose_model_lacks_only_its_key(self):
        self.lacks_only_its_key(
            self.gateway(),
            "review-drafts",
            "./rkmon secret gw_api_key, and give one line of text",
        )

    def test_the_doctor_s_reviewer_row_when_the_model_lacks_only_its_key(self):
        """rk doctor's reviewer row fixes what the first pass's model lacks as the capability does (features.needs_fix):
        a key, by its secret's command."""
        from rkmon_lib import doctor

        with mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        ):
            row = doctor.reviewer_check(self.gateway())
        self.assertEqual(
            (row.status, str(row.fix)),
            ("FAIL", "./rkmon secret gw_api_key, and give one line of text"),
        )

    def test_tickets_need_a_jira_project(self):
        for name in ("follow-up-tickets", "found-issue-tickets"):
            cfg = base_cfg(self.root, {"ticket_creation": LIVE["ticket_creation"]})
            code, msg = features.refusal(cfg, name)
            self.assertEqual(code, features.EXIT_NOT_SET_UP)
            self.assertIn(deploy_tree.name("jira.project"), msg)
            self.assertEqual(
                features.refusal(base_cfg(self.root), name)[0], features.EXIT_OFF
            )

    def test_the_ticket_rows_name_the_deploy_file_s_keys(self):
        """Wherever ticket creation says what it lacks, the JIRA project and the Atlassian site are named by the deploy
        settings file's keys (deploy_tree.name): the features' rows and their table, the atlassian CLI's rows, and
        rk doctor's JIRA row, which names the one key that gives both the site and JIRA's address once.
        """
        project, site = deploy_tree.name("jira.project"), deploy_tree.name(
            "atlassian.site"
        )
        self.assertEqual(
            (project, site, deploy_tree.name("jira.url")),
            (
                "features.ticket_creation.project_key",
                "integrations.atlassian_cli.jira_url",
                "integrations.atlassian_cli.jira_url",
            ),
        )
        cfg = base_cfg(self.root, {"ticket_creation": LIVE["ticket_creation"]})
        self.assertEqual(
            features.missing(cfg, "follow-up-tickets"),
            [
                f"{project} (the JIRA project new tickets go to)",
                f"{site} (the JIRA site's address: the atlassian CLI files the tickets there)",
            ],
        )
        self.assertEqual(
            atlassian.check_line(cfg).found, f"not set up ({site} is blank)"
        )
        self.assertEqual(atlassian.summary_row(cfg), f"off ({site} is blank)")
        self.assertEqual(
            reach.jira(cfg)[0].found, f"not set up ({project} and {site} are blank)"
        )
        for name in ("follow-up-tickets", "found-issue-tickets"):
            self.assertTrue(
                features.CAPABILITIES[name].needs.startswith(
                    f"{project} and the atlassian CLI ({site}; in live, logged in)"
                ),
                name,
            )

    def test_in_prototype_a_feature_needs_no_login_of_its_sends(self):
        """A feature that sends needs the login or secret its sends go out with only while it is live:
        in prototype the alerts lack no webhook, the published team page no web host key and ticket creation no
        atlassian login, so nothing refuses or reports them; live, each lacks its own again.
        """
        b = self.root / "bin"
        b.mkdir()
        (b / "atlassian").write_text("#!/bin/sh\nexit 0\n")
        (b / "atlassian").chmod(0o755)
        web = {
            "html": {
                "host": "web.example",
                "user": "u",
                "dest": "d/index.html",
                "url": "http://web.example/x/",
            }
        }
        jira = {"jira": {"project": "TK"}, "atlassian": {"site": "example-site"}}
        for feature, top, name, lacks in (
            ("alerts", {}, "chat-alerts", "the channel's incoming-webhook URL"),
            ("alerts", {}, "needs-you-alerts", "the channel's incoming-webhook URL"),
            (
                "dashboards.team",
                {"publish": web},
                "team-dashboard",
                "web host's ssh key",
            ),
            ("ticket_creation", jira, "follow-up-tickets", "the atlassian CLI's login"),
            (
                "ticket_creation",
                jira,
                "found-issue-tickets",
                "the atlassian CLI's login",
            ),
        ):
            with self.subTest(name=name), mock.patch.dict(os.environ, {"PATH": str(b)}):
                live = base_cfg(self.root, LIVE, **top)
                proto = base_cfg(self.root, LIVE, **top)
                node = proto["features"]
                for part in feature.split("."):
                    node = node[part]
                node["mode"] = "prototype"
                self.assertIn(lacks, " ".join(features.missing(live, name)))
                self.assertEqual(
                    features.refusal(live, name)[0], features.EXIT_NOT_SET_UP
                )
                self.assertEqual(features.missing(proto, name), [])
                self.assertIsNone(features.refusal(proto, name))
                self.assertTrue(features.active(proto, name))

    def test_tickets_need_the_atlassian_cli_here_and_logged_in(self):
        """The JIRA project is not enough: the CLI the tickets are filed through must be on PATH and hold a login,
        and `rk features` (the setup summary) says what is missing."""
        b = self.root / "bin"
        b.mkdir()
        creds = self.root / "work" / ".config" / "atlassian" / "credentials"
        for name in ("follow-up-tickets", "found-issue-tickets"):
            with self.subTest(name=name):
                cfg = base_cfg(self.root, {"ticket_creation": LIVE["ticket_creation"]})
                cfg["jira"] = dict(cfg["jira"], project="TK")
                cfg["atlassian"] = dict(cfg["atlassian"], site="example-site")
                with mock.patch.dict(os.environ, {"PATH": str(b)}):
                    self.assertEqual(
                        features.missing(cfg, name),
                        [
                            "the atlassian CLI (every image built by this package carries it)"
                        ],
                    )
                    self.assertIn(
                        "the atlassian CLI (every image built by this package carries it",
                        summary.render(cfg),
                    )
                    (b / "atlassian").write_text("#!/bin/sh\nexit 0\n")
                    (b / "atlassian").chmod(0o755)
                    self.assertEqual(len(features.missing(cfg, name)), 1)
                    self.assertIn(
                        "the atlassian CLI's login", features.missing(cfg, name)[0]
                    )
                    self.assertEqual(
                        features.refusal(cfg, name)[0], features.EXIT_NOT_SET_UP
                    )
                    creds.parent.mkdir(parents=True, exist_ok=True)
                    creds.write_text("{}")
                    self.assertEqual(features.missing(cfg, name), [])
                    self.assertTrue(features.active(cfg, name))
                    self.assertIsNone(features.refusal(cfg, name))
                    creds.unlink()
                    (b / "atlassian").unlink()


class TestAutoMergeRespectsGitHub(unittest.TestCase):
    # every gh call goes through rkmon_lib/gh.py's run (or, in a tool, the RKMON_GH binary): ["pr", "merge", ...]
    MERGE = re.compile(
        r"\[(?:[^\[\]]*?\"gh\",\s*)?\"pr\",\s*\"merge\"(.{0,300}?)capture_output", re.S
    )

    def test_the_only_merge_action_is_githubs_own_auto_merge(self):
        files = [
            p
            for d in ("rkmon_lib", "tools", "worker")
            for p in (ROOT / d).rglob("*")
            if p.suffix in (".py", ".sh")
        ]
        files.append(ROOT / "rkmon.py")
        calls = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            for m in self.MERGE.finditer(text):
                calls.append((f.name, m.group(1)))
            # no direct merge through the API, no admin bypass, no force, no push of any kind
            for bad in (
                r"pulls/\{?[\w\]\['\"]*\}?/merge",
                r"mergePullRequest",
                r"--admin\b",
                r"git push",
                r"--force-with-lease",
                r"push\s+-f\b",
            ):
                self.assertIsNone(re.search(bad, text), f"{f.name}: {bad}")
        self.assertTrue(calls)
        for name, args in calls:
            self.assertTrue("--auto" in args or "--disable-auto" in args, (name, args))
            self.assertNotIn("--admin", args)

    def test_arm_needs_the_merge_queue_live_with_auto_merge_enabled(self):
        st = empty_state()
        st["prs"]["7"] = {"n": 7, "head": "a" * 40}
        st["cleared"] = [7]
        st["batches"] = [{"members": [7], "status": "validated"}]
        for feats, why in (
            (
                {"merge_queue": {"mode": "prototype", "auto_merge": "enabled"}},
                "auto-merge is not sent: features.merge_queue.mode is prototype",
            ),  # one you type, refused before any read
            (
                {"merge_queue": {"mode": "live", "auto_merge": "hints_only"}},
                "auto-merge is off",
            ),
        ):
            cfg = {"features": feats, "repo": "ROCm/r"}
            with mock.patch.object(
                gh, "rest", side_effect=AssertionError("read")
            ), mock.patch.object(
                post.subprocess, "run", side_effect=AssertionError("armed")
            ):
                with self.assertRaises(post.PostRefused) as cm:
                    post.arm(cfg, st, mock.Mock(), 7)
                self.assertIn(why, str(cm.exception))

    def test_arm_refuses_a_pr_github_calls_unmergeable(self):
        st = empty_state()
        st["prs"]["7"] = {"n": 7, "head": "a" * 40}
        st["cleared"] = [7]
        st["batches"] = [{"members": [7], "status": "validated"}]
        cfg = {
            "features": {"merge_queue": {"mode": "live", "auto_merge": "enabled"}},
            "repo": "ROCm/r",
        }
        for live, why in (
            (
                {
                    "state": "open",
                    "head": "a" * 40,
                    "draft": False,
                    "mergeable": False,
                    "mergeable_state": "dirty",
                },
                "not mergeable",
            ),
            (
                {
                    "state": "open",
                    "head": "a" * 40,
                    "draft": False,
                    "mergeable": None,
                    "mergeable_state": "unknown",
                },
                "has not computed",
            ),
        ):
            with mock.patch.object(gh, "rest", return_value=live), mock.patch.object(
                post.subprocess, "run", side_effect=AssertionError("armed")
            ):
                with self.assertRaises(post.PostRefused) as cm:
                    post.arm(cfg, st, mock.Mock(), 7)
                self.assertIn(why, str(cm.exception))


class TestSummary(Base):
    def test_no_capability_and_no_table_is_for_services(self):
        """The monitor has no services of its own: no capability is one, and rk features
        shows no table of them, in the container or outside it."""
        self.assertNotIn("services", features.CAPABILITIES)
        for box in ("1", ""):
            with self.subTest(container=box), mock.patch.dict(
                os.environ, {"RKMON_CONTAINER": box}
            ):
                text = summary.render(config.normalize(base_cfg(self.root)))
            self.assertIn("### Dashboards", text)
            self.assertNotIn("### Services", text)
            self.assertNotIn("| Units |", text)

    def test_a_row_names_the_deploy_key_that_sets_it(self):
        """A row of the setup summary that names a setting names the deploy key that sets it: a package setting no
        deploy key sets (the session's auto-compaction, the tick triage's and the optional passes' models, the review
        offload's settings) has no row, in the container or outside it.
        """
        owners = {k.name: k.owner for k in registry.keys()}
        cfg = base_cfg(self.root, LIVE, offload={"mode": "shadow"})
        for box in ("1", ""):
            with self.subTest(container=box), mock.patch.dict(
                os.environ, {"RKMON_CONTAINER": box}
            ):
                text = summary.render(config.normalize(cfg))
            named = {
                word
                for line in text.splitlines()
                if line.startswith("| ")
                for word in re.findall(
                    r"[a-z_]+(?:\.[a-z0-9_]+)+", line.split(" | ")[0]
                )
            }
            self.assertIn(deploy_tree.name("teams.channel"), named)  # a row's name
            self.assertEqual(
                sorted(
                    w
                    for w in named
                    if owners.get(w, registry.DEPLOY) != registry.DEPLOY
                ),
                [],
            )

    def test_no_secret_value_is_ever_shown(self):
        cfg = base_cfg(
            self.root,
            {
                "alerts": LIVE["alerts"],
                "dashboards": {"team": {"enabled": True, "mode": "live"}},
            },
        )
        sec = self.root / "work" / ".config"
        sec.mkdir(parents=True)
        values = {
            "admin_access_token": "adm-" + "s3cr3t-1",
            "teams_webhook": "https" + "://hook.example/s3cr3t-2",
            "web_host_ssh_key": "-----BEGIN" + " OPENSSH PRIVATE KEY-----\ns3cr3t-3",
            "first_review_pass_api_key": "s3cr3t-4",
        }
        for name, v in values.items():
            (sec / name).write_text(v + "\n", encoding="utf-8")
        cfg["teams"]["channel"] = "Chan"
        cfg["publish"]["html"].update(
            host="web.example",
            user="u",
            dest="d/index.html",
            url="http://web.example/x/",
        )
        cfg["jira"].update(url="https://jira.example", project="PRJ")
        cfg["worker"].update(
            url="http://127.0.0.1:9",
            model="m:1",
            api_key_secret="first_review_pass_api_key",
        )
        text = summary.render(config.normalize(cfg), ("1.0 (here)", "logged in"))
        for v in values.values():
            for part in v.splitlines():
                self.assertNotIn(part.split("://")[-1], text)
        for word in (
            "admin dashboard access token",
            "team chat webhook",
            "team web host's ssh key",
            "API key",
        ):
            self.assertIn(word, text)
        self.assertIn("set (", text)
        self.assertIn("https://jira.example/PRJ", text)
        self.assertIn("http://127.0.0.1:18613/", text)
        self.assertIn("`features.review_automation`: on, prototype", text)
        self.assertIn("**Configured and on**", text)
        self.assertIn("**Not configured or off**", text)

    def test_the_reviewer_model_rows_read_no_removed_key(self):
        """The local worker's rows, its key's row and its capability's line read what its binding sets, never a key no
        settings file holds (rkmon_lib/removed_keys.py): where the setup interview found the model, a key or a header
        from an environment variable, a file of extra headers. One left in a configuration shows nowhere.
        """
        cfg = base_cfg(self.root, reviewer="worker")
        cfg["worker"].update(
            backend="openai",
            url="https://gw.example.com",
            model="m",
            source="omp",
            omp_provider="a-provider",
            api_key_env="A_KEY_VARIABLE",
            header_env={"X-Org": "AN_ORG_VARIABLE"},
            headers_file=str(self.root / "headers"),
        )
        cfg = config.normalize(cfg)
        rows = dict(summary._worker_rows(cfg, config.worker(cfg), ""))
        self.assertEqual(
            rows[f"Reviewer model: provider ({deploy_tree.name('worker.backend')})"],
            "an OpenAI-compatible endpoint",
        )
        self.assertEqual(rows["Reviewer model: key sent as"], "no key")
        secrets = dict(summary.secrets(cfg))
        self.assertEqual(
            secrets["reviewer model's API key"],
            f"not used ({deploy_tree.name('worker.api_key_secret')} is blank)",
        )
        line = features.configured_with(cfg, "review-drafts")
        self.assertIn("first pass worker: the local review loop", line)
        said = "\n".join([*rows, *rows.values(), *secrets, *secrets.values(), line])
        for word in (
            "Reviewer model: source",
            "from omp",
            "oh-my-pi",
            "a-provider",
            "A_KEY_VARIABLE",
            "AN_ORG_VARIABLE",
            "X-Org",
            "extra headers",
            "API key set",
        ):
            with self.subTest(word=word):
                self.assertNotIn(word, said)

    def test_a_secret_that_waits_for_live_names_the_mode(self):
        """The team chat's webhook and the web host's key are used once their feature is live: in prototype each row
        says it is not used and names the mode; live, it is set or missing, with its place.
        """

        def rows(mode: str) -> dict:
            cfg = base_cfg(
                self.root,
                {
                    "alerts": dict(LIVE["alerts"], mode=mode),
                    "dashboards": {"team": {"enabled": True, "mode": mode}},
                },
                team_dashboard={"mode": "publish"},
            )
            return dict(summary.secrets(cfg))

        waiting = rows("prototype")
        self.assertEqual(
            waiting["team chat webhook"],
            "not used (features.alerts.mode is prototype: nothing goes out until it is live)",
        )
        self.assertEqual(
            waiting["team web host's ssh key"],
            f"not used ({features.prototype_words('dashboards.team')})",
        )
        live = rows("live")
        for row in ("team chat webhook", "team web host's ssh key"):
            with self.subTest(row=row):
                self.assertRegex(live[row], r"^missing \(.+\)$")


class TestReviewsAndTheClaudeSession(unittest.TestCase):
    def test_each_first_pass_needs_its_own_and_says_it(self):
        """Reviews check the first pass's model only: nothing for the default, the local review loop's endpoint and
        model, a binding's address, model and key (never a value)."""
        with tempfile.TemporaryDirectory() as d:
            cfg = base_cfg(pathlib.Path(d))
            cfg["worker"] = dict(cfg.get("worker") or {}, url="", model="")
            cfg["reviewer"] = "worker"
            self.assertEqual(
                features.missing(cfg, "review-drafts"),
                [
                    "model_bindings.worker.server_url (the local review loop's endpoint)",
                    "model_bindings.worker.model_id (the local review loop's model)",
                ],
            )
            cfg["reviewer"] = "default"
            self.assertEqual(features.missing(cfg, "review-drafts"), [])
            said = features.configured_with(cfg, "review-drafts")
            self.assertTrue(
                said.startswith(
                    "first pass default: the verifier (Claude Opus) reviews alone"
                ),
                said,
            )
            self.assertIn(
                "; line 1: Automated review by rocke-monitor: Model - Claude Opus, effort: high",
                said,
            )
            cfg.update(
                reviewer="claude-x",
                model_bindings={
                    "claude-x": {
                        "provider": "claude",
                        "url": "$RKMON_TEST_UNSET_URL",
                        "api_key_secret": "claude_x_api_key",
                        "model": "m-c",
                        "name": "Model C",
                    }
                },
            )
            # its key, where the secrets registry keeps the secret it names (bindings.binding_by_name)
            key = pathlib.Path(
                secrets_registry.binding_key_file(cfg, "claude_x_api_key")
            )
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_text("secret-value-x\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"RKMON_TEST_UNSET_URL": ""}):
                gone = features.missing(cfg, "review-drafts")
                self.assertEqual(len(gone), 1)
                self.assertIn("$RKMON_TEST_UNSET_URL", gone[0])
            with mock.patch.dict(
                os.environ, {"RKMON_TEST_UNSET_URL": "https://gw.example.invalid"}
            ):
                self.assertEqual(features.missing(cfg, "review-drafts"), [])
                said = features.configured_with(cfg, "review-drafts")
                self.assertIn(
                    "first pass claude-x: Model C via claude at $RKMON_TEST_UNSET_URL",
                    said,
                )
                self.assertNotIn("secret-value-x", said)

    def test_the_claude_session_in_the_container_is_the_orchestrator(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = base_cfg(pathlib.Path(d))
            with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"}):
                self.assertIn(
                    "the container's orchestrator",
                    features.configured_with(cfg, "headless-claude"),
                )
            # No timer runs Claude's tick, outside the container either.
            with mock.patch.dict(os.environ, {"RKMON_CONTAINER": ""}):
                self.assertNotIn(
                    "timer", features.configured_with(cfg, "headless-claude")
                )


if __name__ == "__main__":
    unittest.main()
