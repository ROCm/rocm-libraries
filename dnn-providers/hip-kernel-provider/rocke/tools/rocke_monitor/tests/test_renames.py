# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Configuration keys with a new name (rkmon_lib/renames.py): a configuration with an older key is moved to the new
one, with one plain line; the code reads the new keys only; the defaults and the docs name only the new keys.
"""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import (
    config,
    deploy_tree,
    features,
    file_types,
    removed_keys,
    renames,
    schema,
)  # noqa: E402

OLD = {
    "me": "op-login",
    "github_jira_users": {"dev-a": "dev-a@example.com"},
    "worker_tag": "local",
    "min_free_gb": 24,
    "scoreboard": {"lookback_d": 3, "ahead_h": 12},
    "mirror": {"on": True, "pairs": 2},
    "jira": {
        "project": "KEY",
        "mcp_enabled": True,
        "mcp_server": "jira",
        "mcp_url": "https://mcp.example.com/mcp",
    },
    "claude": {"quality_effort": "low", "effort": "high"},
    "profile": "merge-queue",
    "phase": "queue",
    "container": {
        "project": "rkmon-x",
        "image": "rocke-monitor:t1",
        "registry": "registry.example.com",
        "uid": 4242,
    },
    "rocm": {"mode": "host", "host_dir": "/opt/rocm/core-9.9"},
}
# The line of a feature that is on with no mode set (renames.SENT_WHENEVER_ON), by its section.
SENT = (
    "{0}.enabled in config.json is true and {0}.mode is not set, which an older configuration sent whenever it "
    "was on: {0}.mode is now live"
)


class TestMigrate(unittest.TestCase):
    def test_every_older_key_moves_with_one_line(self):
        new, lines = renames.migrate(OLD, "config.json")
        self.assertEqual(new["operator"], {"github_login": "op-login"})
        self.assertEqual(
            new["jira"],
            {"project": "KEY", "accounts": {"dev-a": "dev-a@example.com"}},
        )
        self.assertEqual(new["worker"], {"tag": "local"})
        self.assertEqual(new["checks"], {"min_free_memory_gb": 24})
        self.assertEqual(new["scoreboard"], {"ahead_h": 12, "lookback_days": 3})
        self.assertEqual(new["mirror"], {"pairs": 2, "enabled": True})
        self.assertEqual(
            (new["claude"], new["effort"]), ({"effort": "high"}, {"quality": "low"})
        )
        self.assertEqual(
            config.role_effort(new, "quality"),
            ("low", "features.review_automation.passes.code_quality_pass.effort"),
        )
        self.assertIn(
            "claude.quality_effort in config.json is now effort.quality: moved", lines
        )
        for old in renames.RENAMED:
            self.assertIsNone(config.lookup(new, old), old)
        self.assertEqual(
            new["deployment"],
            {
                "project": "rkmon-x",
                "image": "rocke-monitor:t1",
                "pull_registry": "registry.example.com",
            },
        )
        # The older profile and phase are the features' switches and modes now, one line each.
        self.assertEqual(
            (
                new["features"]["review_automation"]["enabled"],
                new["features"]["merge_queue"]["mode"],
            ),
            (False, "live"),
        )
        self.assertIn(
            "phase queue in config.json is now two keys: features.review_automation.mode prototype, "
            "features.merge_queue.mode live",
            lines,
        )
        self.assertEqual(new["container"], {"uid": 4242})
        self.assertNotIn("rocm", new)  # every image carries its own ROCm
        self.assertIn(
            "rocm.mode in config.json dropped: every image carries its own ROCm", lines
        )
        self.assertIn(
            "jira.mcp_server in config.json dropped: an unused feature, the JIRA MCP server",
            lines,
        )
        self.assertIn(
            "container.registry in config.json is now deployment.pull_registry: moved",
            lines,
        )
        self.assertEqual(
            len(lines), len(renames.RENAMED) - 1 + len(renames.DROPPED) + 2
        )  # every renamed key but teams.needs_you_every_min, every dropped one, the profile and the phase
        self.assertIn("me in config.json is now operator.github_login: moved", lines)
        self.assertIn("me", OLD)  # the input is not changed

    def test_the_new_key_wins(self):
        new, lines = renames.migrate({"me": "old", "operator": {"github_login": "new"}})
        self.assertEqual(new, {"operator": {"github_login": "new"}})
        self.assertEqual(
            lines, ["me is now operator.github_login, which is set already: me dropped"]
        )

    def test_a_configuration_with_the_new_keys_is_left_as_it_is(self):
        cfg = {"operator": {"github_login": "x"}}
        self.assertIs(renames.migrate(cfg)[0], cfg)
        self.assertEqual(renames.migrate(cfg)[1], [])

    def test_each_older_alert_switch_becomes_its_kind_s_own(self):
        """chat-alerts and needs-you-alerts become each kind's own switch, beside the alerts' one, on while either is;
        with neither on, both kinds are on, so the one switch, once on, sends both. Each kind goes out as before: the
        alerts are live while either older switch is on (they had no mode, and were sent whenever on).
        """
        for chat, needs, want in (
            (True, False, (True, True, False)),
            (False, True, (True, False, True)),
            (True, True, (True, True, True)),
            (False, False, (False, True, True)),
        ):
            with self.subTest(chat_alerts=chat, needs_you_alerts=needs):
                raw = {"features": {"chat-alerts": chat, "needs-you-alerts": needs}}
                new, lines = renames.migrate(raw, "config.json")
                alerts = new["features"]["alerts"]
                self.assertEqual(
                    (alerts["enabled"], alerts["develop_breaks"], alerts["needs_you"]),
                    want,
                )
                self.assertEqual(
                    [features.on(new, n) for n in ("chat-alerts", "needs-you-alerts")],
                    [chat, needs],
                )
                self.assertEqual(alerts.get("mode"), "live" if want[0] else None)
                self.assertEqual(SENT.format("features.alerts") in lines, want[0])
                self.assertIn(
                    "features.chat-alerts in config.json is now features.alerts.develop_breaks "
                    f"{str(want[1]).lower()} (each kind of alert is sent while features.alerts.enabled is true)",
                    lines,
                )

    def test_a_feature_that_had_no_mode_is_live_while_it_is_on(self):
        """Alerts, the team dashboard and ticket creation had no mode, and each was sent whenever it was on: each that
        is on with no mode, whether an older switch or its own turns it on, is live, so it sends as before, and a line
        says so; one that is off takes no mode, and starts in prototype; one with a mode keeps it.
        """
        sections = ("alerts", "dashboards.team", "ticket_creation")
        for on in (True, False):
            for shape, raw in (
                (
                    "older switches",
                    {
                        "features": {
                            "chat-alerts": on,
                            "team-dashboard": on,
                            "found-issue-tickets": on,
                        }
                    },
                ),
                (
                    "their own, with no mode",
                    {
                        "features": {
                            "alerts": {"enabled": on, "develop_breaks": True},
                            "dashboards": {"team": {"enabled": on}},
                            "ticket_creation": {"enabled": on},
                        }
                    },
                ),
            ):
                with self.subTest(shape=shape, on=on):
                    new, lines = renames.migrate(raw, "config.json")
                    self.assertEqual(
                        [config.lookup(new, f"features.{s}.mode") for s in sections],
                        ["live" if on else None] * len(sections),
                    )
                    self.assertEqual(
                        [features.live(new, s) for s in sections], [on] * len(sections)
                    )
                    self.assertEqual(
                        [SENT.format(f"features.{s}") in lines for s in sections],
                        [on] * len(sections),
                    )
        given = {
            "features": {
                "alerts": {
                    "enabled": True,
                    "mode": "prototype",
                    "develop_breaks": True,
                },
                "dashboards": {"team": {"enabled": True, "mode": "prototype"}},
                "ticket_creation": {"enabled": True, "mode": "prototype"},
            }
        }
        self.assertIs(renames.migrate(given)[0], given)

    def test_the_jira_mcp_server_goes_with_its_keys_and_its_older_switch(self):
        """The JIRA MCP server is an unused feature: its keys and its older switch jira-mcp are dropped,
        each with one line, and no key takes their place."""
        raw = {
            "features": {"jira-mcp": True},
            "jira": {"project": "KEY", "mcp_enabled": True, "mcp_server": "jira"},
        }
        new, lines = renames.migrate(raw, "config.json")
        self.assertEqual(new["jira"], {"project": "KEY"})
        self.assertNotIn("jira-mcp", new["features"])
        why = "an unused feature, the JIRA MCP server"
        self.assertEqual(
            lines,
            [
                f"jira.mcp_enabled in config.json dropped: {why}",
                f"jira.mcp_server in config.json dropped: {why}",
                f"features.jira-mcp in config.json dropped: {why}",
            ],
        )

    def test_the_install_without_a_container_s_older_switches_go(self):
        """The install without a container goes: its older switches, services and
        headless-claude, are dropped, each with one line, and no key takes their place.
        """
        raw = {"features": {"services": True, "headless-claude": True}}
        new, lines = renames.migrate(raw, "config.json")
        self.assertNotIn("services", new)
        for name in ("services", "headless-claude"):
            self.assertNotIn(name, new["features"])
        self.assertEqual(
            lines,
            [
                "features.services in config.json dropped: an install without a container runs no services",
                "features.headless-claude in config.json dropped: the Claude session runs while "
                "features.review_automation.enabled is true",
            ],
        )

    def test_why_the_services_of_an_install_without_a_container_go_has_one_home(self):
        """Why the services of an install without a container go is said in one place, removed_keys.NO_HOST_SERVICES:
        its keys' why, and the older services switch's (OLD_FEATURES), which the translation of production's settings
        says too, are made from it."""
        why = removed_keys.NO_HOST_SERVICES
        homes = [
            p.relative_to(ROOT).as_posix()
            for p in sorted([*ROOT.glob("rkmon_lib/*.py"), *ROOT.glob("tools/*.py")])
            if why in p.read_text()
        ]
        self.assertEqual(homes, ["rkmon_lib/removed_keys.py"])
        self.assertEqual(renames.OLD_FEATURES["services"], (None, why))
        for key in ("services", "wsl_distro"):
            self.assertEqual(removed_keys.REMOVED_KEYS[key], why)

    def test_the_alerts_one_switch_alone_has_both_kinds_set_on(self):
        """A configuration with the alerts' one switch and neither kind's, as an earlier package wrote it, has both
        kinds set on, so it sends what it sent: on, the alerts are live (they had no mode, and were sent whenever on);
        one with a kind's own is left as it is.
        """
        for on in (True, False):
            with self.subTest(enabled=on):
                raw = {"features": {"alerts": {"enabled": on}}}
                new, lines = renames.migrate(raw, "config.json")
                self.assertEqual(
                    new["features"]["alerts"],
                    {
                        "enabled": on,
                        "develop_breaks": True,
                        "needs_you": True,
                        **({"mode": "live"} if on else {}),
                    },
                )
                self.assertEqual(
                    lines,
                    [
                        "features.alerts.enabled in config.json is the alerts' one switch, and each kind of "
                        "alert has its own now: features.alerts.develop_breaks true, "
                        "features.alerts.needs_you true"
                    ]
                    + ([SENT.format("features.alerts")] if on else []),
                )
                self.assertEqual(
                    raw, {"features": {"alerts": {"enabled": on}}}
                )  # the input is not changed
        kept = {
            "features": {
                "alerts": {"enabled": True, "mode": "live", "needs_you": False}
            }
        }
        self.assertIs(renames.migrate(kept)[0], kept)

    def test_the_older_switches_a_configuration_has_and_whether_each_is_on(self):
        """old_features: each features.<name> that OLD_FEATURES names, and nothing else; is_on: any value but false,
        "off", 0, blank or null turns a switch on."""
        raw = {
            "features": {
                "review-posting": True,
                "auto-merge-guard": "act",
                "queue-notices": "off",
                "review_automation": {"enabled": True},
            }
        }
        self.assertEqual(
            renames.old_features(raw),
            {"review-posting": True, "auto-merge-guard": "act", "queue-notices": "off"},
        )
        self.assertEqual(renames.old_features({"features": "on"}), {})
        self.assertEqual(renames.old_features({}), {})
        for value in (True, "act", "alert", "on", 1):
            self.assertTrue(renames.is_on(value), value)
        for value in (False, "off", 0, "", None):
            self.assertFalse(renames.is_on(value), value)

    def test_every_renamed_key_is_in_the_defaults_and_no_older_one_is(self):
        d = config.load_defaults()
        doc = config.doc()
        sch = file_types.DEPLOY.load()
        for old, new in renames.RENAMED.items():
            self.assertIsNotNone(config.lookup(d, new), new)
            if deploy_tree.name(new) == new:
                self.assertIn(new, doc)  # an advanced key: described in the registry
            else:
                self.assertIsNotNone(schema.node(sch, deploy_tree.name(new)), new)
                # a tree key: described by the schema alone, after the file key that feeds it
                if new in doc:
                    self.assertTrue(
                        doc[new].startswith(f"{deploy_tree.name(new)}: "), new
                    )
            self.assertIsNone(config.lookup(d, old), old)
            self.assertNotIn(old, doc)
        for old in renames.DROPPED:
            self.assertIsNone(config.lookup(d, old), old)
            self.assertNotIn(old, doc)


class TestReaders(unittest.TestCase):
    def test_load_and_normalize_read_the_new_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "config.json"
            raw = dict(
                OLD,
                repo="acme/widgets",
                rocke_path="a/b",
                scope={"labels": ["x"]},
                paths={"working_dir": d},
                phase="observe",
                merge_authority="none",
            )
            p.write_text(json.dumps(raw), encoding="utf-8")
            cfg = config.load(p)
        self.assertEqual(config.operator_login(cfg), "op-login")
        self.assertEqual(config.jira_accounts(cfg), {"dev-a": "dev-a@example.com"})
        self.assertEqual(config.min_free_memory_gb(cfg), 24)
        self.assertEqual(
            cfg["features"]["review_automation"]["enabled"], False
        )  # the profile merge-queue's
        self.assertEqual(
            cfg["features"]["merge_queue"]["mode"], "prototype"
        )  # phase observe
        for old in ("me", "phase", "profile", "merge_authority"):
            self.assertNotIn(
                old, list(cfg)
            )  # the strict mapping refuses even `old in cfg`

    def test_the_migration_moves_the_older_keys(self):
        """renames.migrate (load's one migration, and every other reader of a file's) moves the older keys to their new
        names: top-level and nested ones, and into a new key that is there but blank.
        """
        cfg = config.normalize(
            renames.migrate(
                dict(
                    OLD,
                    repo="acme/widgets",
                    rocke_path="a/b",
                    operator={"github_login": ""},  # blank: the older `me` fills it
                    mirror={"on": True, "enabled": None, "pairs": 2},
                )
            )[0]
        )
        self.assertEqual(config.operator_login(cfg), "op-login")
        self.assertEqual(cfg["jira"]["accounts"], {"dev-a": "dev-a@example.com"})
        self.assertEqual(cfg["worker"]["tag"], "local")
        self.assertEqual(config.min_free_memory_gb(cfg), 24)
        self.assertEqual(cfg["scoreboard"], {"ahead_h": 12, "lookback_days": 3})
        self.assertEqual(cfg["mirror"], {"enabled": True, "pairs": 2})
        for old in renames.RENAMED:
            self.assertIsNone(config.lookup(cfg, old), old)
        # A new key that holds a value wins; the older one is dropped.
        cfg = config.normalize(
            renames.migrate(
                {"me": "old", "operator": {"github_login": "new"}, "repo": "a/b"}
            )[0]
        )
        self.assertEqual((config.operator_login(cfg), "me" in cfg), ("new", False))


if __name__ == "__main__":
    unittest.main()
