# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container deployment's checks (docker/, rkmon_lib/doctor.py): the Claude session's key (`rkmon login
claude`), its settings file and the rows about the session, the atlassian CLI's login line, sites, login folder and
status, the ROCm release the image recorded and the clang the gates were tuned on, and the docker files' own promises
(nothing of a site in them, a non-root user, secrets never baked). Everything runs outside any container, with
`in_container` patched."""

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKES = ROOT / "tests" / "fakes"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from rkmon_lib import (
    atlassian,
    build_plan,
    code_volume,
    commands,
    config,
    container_layout,
    control,
    doctor,
    orchestrator,
    procs,
    registry,
    summary,
)  # noqa: E402
import test_code_volume as code_tests  # noqa: E402  (its code volume; its own tests run in their module)
import test_entrypoint as entry_tests  # noqa: E402  (the entry point's functions; its own tests run in their module)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        env = {
            "RKMON_POINTER": str(self.d / "pointer"),
            "RKMON_HOME": "",
            "XDG_CONFIG_HOME": str(self.d / "x"),
            "RKMON_WORKING_DIR": str(self.d / "work"),
            "RKMON_SECRETS_DIR": str(self.d / "secrets"),
            "RKMON_PORT": "18614",
            "RKMON_CONTAINER": "1",
        }
        self.write_secrets = self.d / "secrets-write"  # the writer's volume
        for p in (
            mock.patch.dict(os.environ, env),
            mock.patch.object(
                container_layout, "WRITE_SECRETS_DIR", str(self.write_secrets)
            ),
            # The image's ROCm record: none, unless a test writes one (here, and inside the image alike).
            mock.patch.object(
                container_layout, "ROCM_RECORD", str(self.d / "rocm-record.json")
            ),
            mock.patch.object(doctor, "in_container", return_value=True),
            mock.patch.object(doctor, "worker_running", return_value=False),
        ):
            p.start()
            self.addCleanup(p.stop)

    def cfg(self) -> dict:
        """A configuration in the container, its folders this test's."""
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
            claude_workspace=str(self.d / "work" / "claude-workspace"),
        )
        c["paths"] = dict(
            c["paths"],
            working_dir=str(self.d / "work"),
            secrets_dir=str(self.d / "secrets"),
        )
        return config.normalize(c)


class TestClaudeSession(Base):
    """The Claude session in the container: its key (the claude login), its settings file (rk orchestrate writes it as
    the session starts: orchestrator.write_settings) and the doctor's rows about it."""

    def test_the_sessions_key_is_the_claude_login_never_shown(self):
        """The session's key is the claude login (`rkmon login claude`). The setup summary says where it is and which
        variables the session gets, never the key."""
        cfg = self.cfg()
        key = self.d / "secrets" / "claude_key"
        self.assertIn(f"the claude login: not set ({key})", summary.render(cfg))
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text("Invented-Key-Q\n", encoding="utf-8")
        key.chmod(0o600)
        page = summary.render(cfg)
        self.assertIn(
            f"the claude login: set ({key}; not shown): the session gets ANTHROPIC_API_KEY",
            page,
        )
        self.assertNotIn("Invented-Key-Q", page)

    def test_the_sessions_key_is_the_claude_login_s_row_with_its_test_call(self):
        """The session's row (rk orchestrate) names what the session lacks but its key: the key is the claude login's
        row, with the login's test call (rkmon_lib/reach.py claude_login), a FAIL while review automation is on, whose
        fix is the claude login's command. The key goes in the call as the session sends it, and in no
        row."""
        from rkmon_lib import reach

        skill = self.d / "work" / "claude-workspace" / ".claude" / "skills"
        (skill / "rocke-monitor").mkdir(parents=True)
        (skill / "rocke-monitor" / "SKILL.md").write_text("the skill\n")
        cfg = self.cfg()
        on = dict(
            cfg,
            features=dict(
                cfg["features"],
                review_automation=dict(
                    cfg["features"]["review_automation"], enabled=True
                ),
            ),
        )
        session = {c.name: c for c in doctor.container_checks(on)}[
            "Claude session (rk orchestrate)"
        ]
        self.assertEqual((session.status, session.fix), ("PASS", None), session.found)
        login = str(doctor.claude_login_fix(on))
        self.assertIn(" login claude", login)
        (row,) = reach.claude_login(on, post=lambda *a: self.fail("no key: no call"))
        self.assertEqual(
            (row.name, row.status, str(row.fix)), ("claude", "FAIL", login)
        )
        self.assertTrue(row.found.startswith("no Claude key; it blocks "), row.found)
        key = self.d / "secrets" / "claude_key"
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text("Invented-Key-Q\n", encoding="utf-8")
        key.chmod(0o600)
        sent = []
        (row,) = reach.claude_login(
            on,
            post=lambda url, headers, body: sent.append(headers)
            or (401, "authentication_error"),
        )
        self.assertEqual((row.status, str(row.fix)), ("FAIL", login))
        self.assertIn(
            "the key is refused, at Anthropic's own API (HTTP 401, authentication_error)",
            row.found,
        )
        self.assertIn("Invented-Key-Q", sent[0].values())
        self.assertNotIn("Invented-Key-Q", f"{row.found} {row.fix}")
        session = {c.name: c for c in doctor.container_checks(on)}[
            "Claude session (rk orchestrate)"
        ]
        self.assertEqual(session.status, "PASS", session.found)

    def test_rk_orchestrate_writes_the_sessions_compaction_window(self):
        """rk orchestrate writes the session's settings as the session starts. While no session runs, the row is
        skipped: its start writes them. A session that runs without them warns, and the fix starts it again.
        """
        ws = self.d / "work" / "claude-workspace"
        (ws / ".claude").mkdir(parents=True)
        (ws / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"allow": ["Read"]}})
        )
        cfg = self.cfg()
        self.assertEqual(
            cfg["claude"]["auto_compact_window"], 300000
        )  # the package default
        restart = (
            "rk orchestrate writes them, into a file that reads as a JSON object, as the session starts: "
            "rkmon stop --drain, then rkmon start"
        )
        c = {c.name: c for c in doctor.container_checks(cfg)}["Claude session settings"]
        self.assertEqual(
            (c.status, c.found),
            (
                "SKIP",
                "no Claude session runs here, and rk orchestrate writes them, into a file that reads as a JSON "
                f"object, as the session starts ({orchestrator.settings_gap(cfg)})",
            ),
        )  # the session has not started
        # A session runs (this process stands in for it) without them.
        run = control.run_dir(cfg)
        procs.register(run, control.ORCHESTRATOR)
        self.addCleanup(procs.unregister, run)
        c = {c.name: c for c in doctor.container_checks(cfg)}["Claude session settings"]
        self.assertEqual((c.status, str(c.fix)), ("WARN", restart))
        said = orchestrator.write_settings(cfg)
        written = json.loads((ws / ".claude" / "settings.json").read_text())
        self.assertEqual(
            written,
            {
                "permissions": {"allow": ["Read"]},
                "autoCompactWindow": 300000,
                "promptCacheTtl": "1h",
                "subagentPromptCacheTtl": "5m",
            },
        )
        self.assertEqual(
            len(list((ws / ".claude").glob("settings.json.bak-*"))), 1
        )  # the file it changed
        self.assertEqual(
            said[-1],
            f"the session's settings: {ws / '.claude' / 'settings.json'} sets autoCompactWindow 300000, "
            "promptCacheTtl 1h, subagentPromptCacheTtl 5m",
        )
        c = {c.name: c for c in doctor.container_checks(cfg)}["Claude session settings"]
        self.assertEqual(c.status, "PASS")
        # Again: nothing changes. Another value in the configuration: the checks see a file that lags behind it, and
        # the session's next start writes it over the old one.
        self.assertIn(
            "already sets autoCompactWindow 300000", orchestrator.write_settings(cfg)[0]
        )
        self.assertEqual(len(list((ws / ".claude").glob("settings.json.bak-*"))), 1)
        lag = dict(cfg, claude=dict(cfg["claude"], auto_compact_window=450000))
        c = {c.name: c for c in doctor.container_checks(lag)}["Claude session settings"]
        self.assertEqual((c.status, str(c.fix)), ("WARN", restart))
        procs.unregister(run)  # the session ended: its next start writes them
        c = {c.name: c for c in doctor.container_checks(lag)}["Claude session settings"]
        self.assertEqual(c.status, "SKIP", c.found)
        self.assertEqual(
            orchestrator.write_settings(lag)[-1],
            f"the session's settings: {ws / '.claude' / 'settings.json'} sets autoCompactWindow 450000, "
            "promptCacheTtl 1h, subagentPromptCacheTtl 5m",
        )
        self.assertEqual(
            json.loads((ws / ".claude" / "settings.json").read_text())[
                "autoCompactWindow"
            ],
            450000,
        )

    def test_a_settings_file_that_is_not_json_is_left_alone(self):
        ws = self.d / "ws"
        (ws / ".claude").mkdir(parents=True)
        (ws / ".claude" / "settings.json").write_text("{not json")
        cfg = {"claude_workspace": str(ws), "claude": {"auto_compact_window": 300000}}
        self.assertIn("is not JSON", orchestrator.write_settings(cfg)[0])
        self.assertEqual((ws / ".claude" / "settings.json").read_text(), "{not json")
        self.assertEqual(
            orchestrator.write_settings({"claude_workspace": ""}),
            ["the session's settings: not written (claude_workspace is blank)"],
        )

    def test_the_window_is_checked(self):
        for bad in (5000, 2000000, "300000", True):
            self.assertIn(
                f"claude.auto_compact_window is {bad!r}",
                " ".join(config.problems({"claude": {"auto_compact_window": bad}})),
            )
            self.assertEqual(
                orchestrator.compact_window({"claude": {"auto_compact_window": bad}}),
                300000,
            )
        self.assertFalse(
            [
                p
                for p in config.problems({"claude": {"auto_compact_window": 500000}})
                if "auto_compact_window" in p
            ]
        )
        self.assertEqual(orchestrator.compact_window({}), 300000)
        self.assertEqual(
            config.load_defaults()["claude"]["auto_compact_window"],
            orchestrator.compact_window({}),
        )

    def test_the_prompt_cache_lifetimes_and_the_tick_minutes(self):
        d = config.load_defaults()["claude"]
        self.assertEqual(
            (d["prompt_cache_ttl"], d["subagent_prompt_cache_ttl"]), ("1h", "5m")
        )
        cfg = {"claude": {"prompt_cache_ttl": "5m"}}
        self.assertEqual(orchestrator.session_settings(cfg)["promptCacheTtl"], "5m")
        self.assertEqual(
            orchestrator.session_settings(cfg)["subagentPromptCacheTtl"], "5m"
        )
        for bad in ("2h", 60):
            self.assertIn(
                f"claude.prompt_cache_ttl is {bad!r}",
                " ".join(config.problems({"claude": {"prompt_cache_ttl": bad}})),
            )
            self.assertEqual(
                orchestrator.cache_ttl(
                    {"claude": {"prompt_cache_ttl": bad}}, "prompt_cache_ttl"
                ),
                "1h",
            )
        self.assertNotIn(
            "CLAUDE_CODE_PROMPT_CACHE_TTL", orchestrator.SESSION_ENV
        )  # one source: the settings file
        # Minutes between ticks: claude.tick_min (your deploy settings' monitor.tick_interval_minutes), its default
        # when unset; no variable of the environment changes it.
        with mock.patch.dict(os.environ, {"RKMON_TICK_MIN": "3"}):
            self.assertEqual(orchestrator.tick_min({"claude": {"tick_min": 7}}), 7)
            self.assertEqual(
                orchestrator.tick_min({}), registry.default("claude.tick_min")
            )
        self.assertIn(
            "monitor.tick_interval_minutes is 0",
            " ".join(config.problems({"claude": {"tick_min": 0}})),
        )

    def test_the_tick_triage(self):
        """The rule filter reads each tick first unless the configuration names a triage model (models.tick, an
        opt-in); one that is not at hand leaves the rule filter deciding, a warning."""
        cfg = self.cfg()
        self.assertEqual(cfg["models"]["tick"], "")
        self.assertEqual(config.problems(cfg), [])
        c = {c.name: c for c in doctor.container_checks(cfg)}["tick triage"]
        self.assertEqual(
            (c.status, c.found),
            (
                "PASS",
                "the rule filter (rk tick-filter, no model) reads each tick first",
            ),
        )
        on = dict(cfg, models=dict(cfg["models"], tick="worker"))
        c = {c.name: c for c in doctor.container_checks(on)}["tick triage"]
        self.assertEqual(
            (c.status, str(c.fix)),
            (
                "WARN",
                "give the triage model what it lacks in your deploy settings file, then rkmon confirm and rkmon "
                "apply",
            ),
        )
        self.assertIn(
            "the rule filter (rk tick-filter, no model); the triage model waits",
            c.found,
        )
        on["worker"] = {
            "backend": "openai",
            "url": "https://models.example.invalid/v1",
            "model": "m-1",
        }
        c = {c.name: c for c in doctor.container_checks(on)}["tick triage"]
        self.assertEqual(
            (c.status, c.found),
            ("PASS", "the local review loop's model m-1 reads each tick first"),
        )
        # The reviewer never drives the triage: an Anthropic reviewer leaves it on the rule filter.
        claude = dict(
            cfg,
            reviewer="claude",
            model_bindings={
                "claude": {"provider": "anthropic", "url": "$X_URL", "model": "m-c"}
            },
        )
        self.assertEqual(config.model_for(claude, "tick")["by"], "none")


# The sites one login reaches, as `atlassian site list` prints them (a container's login: eight
# sites, the active one marked `* `, and the JIRA project's site not the active one), the company's name
# replaced by "example".
SITES = [
    "example-dev-sandbox",
    "example",
    "example-support-sandbox",
    "example-uat",
    "example-hub-sandbox-c2c-20261028",
    "example-sandbox-2025oct30postphase2-swdevuat",
    "example-hub",
    "example-hub-sandbox-20240925",
]
LISTING = (
    "  example-dev-sandbox\n* example\n  example-support-sandbox\n  example-uat\n"
    "  example-hub-sandbox-c2c-20261028\n  example-sandbox-2025oct30postphase2-swdevuat\n  example-hub\n"
    "  example-hub-sandbox-20240925\n\n* = active site\n\nTo switch sites:  atlassian site select <name>\n"
    "A new version (9.9.9) is available. Run: atlassian update\n"
)


class TestAtlassian(Base):
    """The atlassian CLI in the container (here tests/fakes/fake_atlassian on the PATH): its login is the writer's, in
    the writer's volume, and its fix the command that makes it (`rkmon login atlassian`, which the wrapper takes as
    typed); the sites the CLI lists, from its listing or its sites file; the CLI's login folder, named whether or not a
    site is set; and its status rows.
    """

    def setUp(self):
        super().setUp()
        self.bin = self.d / "bin"
        self.bin.mkdir()
        shutil.copy(FAKES / "fake_atlassian", self.bin / "atlassian")
        (self.bin / "atlassian").chmod(0o755)
        self.log = self.d / "atlassian.log"
        p = mock.patch.dict(
            os.environ,
            {
                "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "FAKE_ATLASSIAN_LOG": str(self.log),
            },
        )
        p.start()
        self.addCleanup(p.stop)

    def calls(self) -> list:
        return (
            [json.loads(x) for x in self.log.read_text(encoding="utf-8").splitlines()]
            if self.log.is_file()
            else []
        )

    def test_the_container_s_login_line_is_one_the_wrapper_takes(self):
        """The fix line that logs the CLI in works as typed, wherever the doctor runs: the wrapper's parser
        (commands.table_parser, which docker/rkmon reads each command line with) reads it as its `login atlassian`.
        """
        for box in ("1", ""):
            with mock.patch.dict(os.environ, {"RKMON_CONTAINER": box}):
                (line,) = atlassian.login_fix().commands
            words = shlex.split(line)
            self.assertEqual(words[0], doctor.wrapper_name(), line)
            with contextlib.redirect_stderr(io.StringIO()) as said:
                try:
                    ns = commands.table_parser().parse_args(words[1:])
                except SystemExit:
                    ns = None
            self.assertIsNotNone(ns, f"the wrapper refuses `{line}`: {said.getvalue()}")
            self.assertEqual((ns.row.name, ns.name), ("login", "atlassian"), line)

    def test_the_cli_s_listing_as_it_prints_it(self):
        self.assertEqual([s["name"] for s in atlassian.parse_sites(LISTING)], SITES)
        loose = "\n" + LISTING.replace(
            "  example-uat\n", "  example-uat\n\n"
        )  # blank lines anywhere
        self.assertEqual([s["name"] for s in atlassian.parse_sites(loose)], SITES)
        listed = json.dumps(
            [
                {
                    "id": "1",
                    "name": "example-hub",
                    "url": "https://example-hub.example.invalid",
                }
            ],
            separators=(",", ":"),
        )
        self.assertEqual(
            atlassian.parse_sites(listed + "\nA new version (9.9.9) is available.\n"),
            [
                {
                    "name": "example-hub",
                    "url": "https://example-hub.example.invalid",
                    "id": "1",
                }
            ],
        )
        self.assertEqual(atlassian.parse_sites("No accessible sites found.\n"), [])

    def test_the_cli_s_sites_file_when_its_list_cannot_be_asked(self):
        folder = self.write_secrets / "atlassian"  # the writer's, in the container
        folder.mkdir(parents=True)
        (folder / "sites").write_text(
            json.dumps(
                {
                    "DEFAULT": {
                        "sites": [
                            {
                                "id": "1",
                                "name": "example-a",
                                "url": "https://example-a.example.invalid",
                            }
                        ],
                        "updated_at": 1,
                    }
                }
            )
        )
        cred = folder / "credentials"
        cred.write_text("[DEFAULT]\nrefresh_token = never-read\n", encoding="utf-8")
        cred.chmod(0)  # reading it would fail: it is never read
        self.addCleanup(cred.chmod, 0o600)
        cfg = {
            "paths": {
                "working_dir": str(self.d / "work"),
                "secrets_dir": str(self.d / "secrets"),
            }
        }
        with mock.patch.dict(os.environ, {"FAKE_ATLASSIAN_SITES_FAIL": "1"}):
            found, why = atlassian.sites(cfg)
        self.assertEqual(([s["name"] for s in found], why), (["example-a"], ""))

    def test_the_login_folder_is_named_before_any_site(self):
        raw = config.load_defaults()
        raw.pop("_doc", None)
        raw["paths"] = dict(raw["paths"], working_dir=str(self.d / "work"))
        cfg = config.normalize(raw)
        self.assertEqual(cfg["atlassian"]["site"], "")
        writers = str(
            self.write_secrets / "atlassian"
        )  # in the container, whatever the folders
        self.assertEqual(atlassian.env(cfg), {"ATLASSIAN_CONFIG_DIR": writers})
        self.assertEqual(
            atlassian.env({"paths": {}}), {"ATLASSIAN_CONFIG_DIR": writers}
        )
        # rk's tool environment (the jobs', the worker tools') never names the atlassian CLI's login
        self.assertNotIn("ATLASSIAN_CONFIG_DIR", config.env(cfg))
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": ""}):  # a host
            folder = str(self.d / "work" / ".config" / "atlassian")
            self.assertEqual(atlassian.env(cfg), {"ATLASSIAN_CONFIG_DIR": folder})
            self.assertEqual(
                atlassian.env({"paths": {}}), {}
            )  # no secrets folder known: nothing to name
        # the configuration is the monitor's: a proxy it names never carries the writer's login, wherever the CLI runs
        cfg["atlassian"]["proxy_url"] = "https://proxy.example.invalid"
        self.assertEqual(atlassian.env(cfg), {"ATLASSIAN_CONFIG_DIR": writers})
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": ""}):
            self.assertEqual(atlassian.env(cfg), {"ATLASSIAN_CONFIG_DIR": folder})

    def test_status_reads_version_login_and_site(self):
        cfg = {
            "atlassian": {"site": "example-site"},
            "paths": {"working_dir": str(self.d / "work")},
        }
        answers = {
            ("version",): (
                0,
                "A new version (9.9.9) is available.\natlassian version 1.5.2",
            ),
            ("auth", "status"): (0, "Logged in as op"),
            ("--site", "example-site", "site", "active"): (0, "ok"),
        }

        def run(cmd):
            return answers[tuple(cmd[1:])]

        with mock.patch.object(
            atlassian.shutil, "which", return_value="/usr/local/bin/atlassian"
        ):
            st = atlassian.status(cfg, run=run)
            self.assertEqual(
                (st["version"], st["logged_in"], st["site_ok"]), ("1.5.2", True, True)
            )
            self.assertEqual(atlassian.check_line(cfg, st).status, "PASS")
            # The hour's access token expired: `auth status` exits 1, but the CLI holds the login and refreshes the
            # token at its next call, so it is logged in.
            answers[("auth", "status")] = (
                1,
                "Logged in to Atlassian Cloud\n  Profile:  DEFAULT\n  Token:    expired",
            )
            st = atlassian.status(cfg, run=run)
            self.assertEqual(
                (st["logged_in"], st["expired"], st["minutes_left"]), (True, True, 0)
            )
            row = atlassian.check_line(cfg, st)
            self.assertEqual(row.status, "PASS")
            self.assertIn("the CLI refreshes it at its next call", row.found)
            answers[("auth", "status")] = (
                0,
                "Logged in to Atlassian Cloud\n  Token:    valid (38m remaining)",
            )
            st = atlassian.status(cfg, run=run)
            self.assertEqual((st["expired"], st["minutes_left"]), (False, 38))
            answers[("auth", "status")] = (1, "Not logged in")
            st = atlassian.status(cfg, run=run)
            row = atlassian.check_line(cfg, st)
            self.assertEqual(row.status, "WARN")
            self.assertEqual(str(row.fix), "rkmon login atlassian")
        self.assertEqual(atlassian.check_line({"atlassian": {}}).status, "SKIP")


class TestRocm(Base):
    def test_the_image_record_and_the_reference_clang(self):
        rec = self.d / "rocm.json"
        rec.write_text(
            json.dumps(
                {
                    "installer": "rocm-installer-9.9.0-1.run",
                    "version": "9.9.0-1",
                    "clang": "Example clang version 99.0.0",
                    "llvm_bin": "/opt/rocm/core-9.9/lib/llvm/bin",
                    "gfx": "gfx942",
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(doctor.rocm_image(rec)["version"], "9.9.0-1")
        self.assertEqual(doctor.rocm_image(self.d / "none.json"), {})
        # The gate compiler is the package's build default: a clang line that names its commit passes.
        cfg = {"paths": {"working_dir": str(self.d)}}
        commit = build_plan.ROCM_CLANG_COMMIT
        gate = (
            f"Example clang version 99.0.0 (https://example.com/llvm-project {commit})"
        )
        self.assertEqual(doctor.clang_parity(cfg, gate).status, "PASS")
        row = doctor.clang_parity(cfg, "Example clang version 100.0.0 (abc123)")
        self.assertEqual(row.status, "WARN")
        self.assertIn(
            f"the gates were tuned on the clang of llvm-project {commit}, ROCm "
            f"{build_plan.ROCM_RELEASE}'s",
            row.found,
        )
        self.assertIn(
            f"set rocm.version in your build settings file to {build_plan.ROCM_RELEASE}, the release the "
            "gates were tuned on",
            str(row.fix),
        )
        self.assertIn("QUICK_BUILD_GUIDE.md", str(row.fix))
        # The LLVM tools are the image's, which no setting gives: the fix is that release's image, wherever the
        # doctor runs.
        self.assertTrue(
            str(row.fix).endswith(
                ", then build that image and run it (QUICK_BUILD_GUIDE.md)"
            ),
            row.fix,
        )
        self.assertNotIn("paths.rocm_llvm_bin", str(row.fix))
        for box in ("1", ""):
            with self.subTest(container=box), mock.patch.dict(
                os.environ, {"RKMON_CONTAINER": box}
            ):
                again = doctor.clang_parity(
                    cfg, "Example clang version 100.0.0 (abc123)"
                )
                self.assertEqual(str(again.fix), str(row.fix))
        # No clang here: nothing to compare.
        none = doctor.clang_parity(cfg, "")
        self.assertEqual(none.status, "SKIP")
        self.assertIn(commit, none.found)


class TestDockerFiles(unittest.TestCase):
    """The recipe's promises, read from the files: no address (each download location is given to the build as a
    build secret), no secret baked, no user id of the build machine, the dashboard published on the host's loopback
    only, the deploy machine never building.
    """

    def test_the_recipe(self):
        docker = ROOT / "docker"
        dockerfile = (docker / "Dockerfile").read_text(encoding="utf-8")
        # The users' ids are the deployment's (docker/rk_ids.py at each start), never a build argument.
        self.assertNotRegex(dockerfile, r"(?m)^ARG (UID|GID|WRITE_UID|PR_UID)\b")
        self.assertNotRegex(
            dockerfile, r"(?m)^USER "
        )  # the entry point starts as root, then drops
        # No address: each download location, the public defaults among them, is rkmon_lib/build_plan.py's.
        self.assertEqual(
            re.findall(r"https?://[^\s\"']+", re.sub(r"(?m)^#.*$", "", dockerfile)), []
        )
        for secret in ("ANTHROPIC_API_KEY", "claude_env", "GH_TOKEN"):
            self.assertNotIn(f"ENV {secret}", dockerfile)
        compose = (docker / "compose.yaml").read_text(encoding="utf-8")
        self.assertNotRegex(
            compose, r"(?m)^\s+build:"
        )  # the deploy machine never builds
        self.assertIn("pull_policy: never", compose)
        for var in ("RKMON_UID", "RKMON_GID"):
            self.assertIn(
                f"      {var}: ${{{var}:-}}", compose
            )  # the entry point maps them
        for var in ("RKMON_WRITE_UID", "RKMON_PR_UID"):
            self.assertNotIn(
                var, compose
            )  # the writer's and the runners' ids are the package's
        self.assertIn(
            '"${RKMON_ADMIN_HOST:?see ./rkmon next}:${RKMON_PORT:?see ./rkmon next}:18613"',
            compose,
        )  # docker/.env gives both, and compose keeps no default of its own (tests/test_one_home.py)
        self.assertNotIn(
            "18615", compose
        )  # the team dashboard's port: compose.override.yaml publishes it while the container serves it
        for var, form in (
            ("RKMON_ADMIN_HOST", "?"),
            ("RKMON_PORT", "?"),
            ("RKMON_TEAM_HOST", "-"),
            ("RKMON_TEAM_PORT", "-"),
        ):
            self.assertIn(
                f"      {var}: ${{{var}:{form}", compose
            )  # the container sees what compose published
        self.assertIn("restart: unless-stopped", compose)
        self.assertNotRegex(
            compose, r"(?m)^\s*-\s*\"?\d+:\d+"
        )  # never published on every host address
        install = (docker / "install_rocm.sh").read_text(encoding="utf-8")
        self.assertIn(
            "deps=install gfx=$gfx rocm compo=$compo gpu-access=user assumeyes", install
        )
        # It names no address: the installer's location is the build secret rocm_package_url, and AMD's public
        # folder, the default, has its one home in rkmon_lib/build_plan.py.
        self.assertEqual(re.findall(r"https?://[^\s}\"']+", install), [])
        entry = (docker / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn(
            'RKMON_CODE=$c "$c/tools/rkweb.py" >/dev/null', entry
        )  # the dashboard service, watched, from the code that runs, which it runs from then on (RKMON_CODE)
        self.assertIn(
            'code() { "$RK" --which-code', entry
        )  # the image's code, or the release installed over it (rk code install)
        self.assertIn(
            'exec "$ORCH_PY" -B "$RKMON_CODE/rkmon.py" orchestrate', entry
        )  # from the code that runs, which it runs from then on, once it is set up (tests/test_entrypoint.py)
        self.assertIn(
            'case "$envline" in "export "*) eval', entry
        )  # never an eval of an error message
        self.assertIn('"$RK" stop', entry)
        # gh's read login goes to the secrets volume, also before any setup (a login made before the site is set
        # would land in a one-off container's home, which --rm throws away); the atlassian CLI's login to the
        # writer's volume, which only the writer reads.
        self.assertIn("GH_CONFIG_DIR=${GH_CONFIG_DIR:-$secrets/gh-read}", entry)
        self.assertIn("ATLASSIAN_LOGIN=/secrets-write/atlassian", entry)
        self.assertIn("ATLASSIAN_CONFIG_DIR=$ATLASSIAN_LOGIN", entry)
        self.assertNotIn("$secrets/atlassian", entry)
        # What the container says to run on the host while the work volume is not set up: rk provision --check's line,
        # which names the first start (rkmon start). The entry point has no setup mode, and no line of it names setup:
        # the deploy path sets a deployment up.
        self.assertIn('RKMON_WRAPPER=rkmon "$RK" provision --check', entry)
        self.assertNotRegex(entry, r"\bsetup\b")
        # After rk stop or rk kill, the restart loops, the loop that restarts the admin dashboard and the tick loop that
        # runs without the orchestrator stay down.
        self.assertIn('is_stopped() { "$RK" stopped', entry)
        self.assertIn('is_stopped || RKMON_CODE=$c "$c/tools/rkweb.py" status', entry)
        self.assertIn(
            'TICK_MIN=$("$RK" tick-min', entry
        )  # one source: claude.tick_min, your deploy settings' monitor.tick_interval_minutes
        self.assertNotIn("RKMON_TICK_MIN", entry + compose)
        self.assertIn(
            'tail -n 0 -F "$LOGS/orchestrator.log"', entry
        )  # the orchestrator's output: the container log
        grace = re.search(r"stop_grace_period: (\d+)m", compose)
        self.assertTrue(
            grace and int(grace.group(1)) >= 10,
            "stop_grace_period under rk stop's worst case",
        )

    def test_the_code_is_the_last_layer(self):
        """The tools' layers come first, each built from only the files it needs, and the code last, so a code change
        rebuilds no tool layer; the code's digest is taken after the code is copied."""
        text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        before, sep, after = text.partition("\nCOPY . /opt/rocke_monitor\n")
        self.assertTrue(sep, "the code is copied whole, once")
        self.assertNotIn(
            "/opt/rocke_monitor", re.sub(r"(?m)^#.*$", "", before)
        )  # no tool layer reads the code
        for need in (
            "COPY requirements/ ",
            "COPY docker/install_rocm.sh ",
            "COPY docker/install_atlassian.sh ",
            "COPY docker/claude_cli.py ",
            "venv/bin/pip install",
            "install_sdk.sh",
            "gh_install",
            "useradd",
        ):
            self.assertIn(need, before)
        for tool_step in (
            "pip install",
            "apt-get",
            "install_rocm.sh",
            "install_atlassian.sh",
            "install_sdk.sh",
            "gh_install",
        ):
            self.assertNotIn(tool_step, after)
        self.assertIn("> /etc/rkmon/image_code_id", after)
        self.assertNotRegex(after, r"(?m)^COPY ")

    def test_the_gh_layer_holds_what_its_step_imports(self):
        """The gh layer copies only the files its step needs, so a code change rebuilds no gh: rkmon_lib/deploy.py and
        what it imports when it loads. Its gh_install loads from those files alone."""
        text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        stage = text[text.index("\nFROM base AS gh\n") + 1 :]
        stage = stage[: stage.index("\nFROM ")]
        copy = re.search(r"(?m)^COPY ((?:rkmon_lib/\S+ )+)\S+$", stage)
        self.assertTrue(copy, "the gh layer copies its rkmon_lib files")
        self.assertIn("deploy.gh_install(", stage)
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "rkmon_lib").mkdir()
            for f in copy.group(1).split():
                shutil.copy(ROOT / f, pathlib.Path(d) / f)
            r = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); from rkmon_lib import deploy; deploy.gh_install",
                    d,
                ],
                capture_output=True,
                text=True,
                cwd=d,
            )
        self.assertEqual((r.returncode, r.stderr), (0, ""))

    def test_the_build_skips_the_check_on_a_base_image_with_no_default(self):
        """BASE_IMAGE has no default, as no build argument has, so a build not given one stops at the base stage's FROM;
        Docker's build check InvalidDefaultArgInFrom, which warns of that at every build, is skipped by a parser
        directive, which Docker reads only above every comment and instruction."""
        text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[0], "# check=skip=InvalidDefaultArgInFrom")
        self.assertIn("FROM ${BASE_IMAGE} AS base\n", text)

    def test_keep_stays_down_while_stopped(self):
        entry = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        funcs = entry[
            entry.index("is_stopped() {") : entry.index("\n# rk start undoes")
        ]
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            rk = d / "rk"
            rk.write_text(f'#!/bin/sh\n[ "$1" = stopped ] && [ -f {d}/stopped ]\n')
            rk.chmod(0o755)
            script = (
                f'RK={rk}\nsay() {{ echo "$*"; }}\npids=""\n{funcs}\n'
                f'keep "job" {d}/log sh -c "echo ran >> {d}/runs; exec sleep 30"\nsleep 2\n'
                'for p in $pids; do kill -TERM "$p"; done; wait\n'
            )
            (d / "stopped").write_text("")
            subprocess.run(["sh", "-c", script], timeout=60, capture_output=True)
            self.assertFalse((d / "runs").exists())  # stopped: never started
            self.assertIn(
                "job stays down: the monitor is stopped", (d / "log").read_text()
            )
            (d / "stopped").unlink()
            subprocess.run(["sh", "-c", script], timeout=60, capture_output=True)
            self.assertEqual((d / "runs").read_text(), "ran\n")  # not stopped: it runs

    def test_what_rk_start_says_goes_to_the_container_log_too(self):
        """At each start of `run`, what rk start says, the one-time move's report with it, goes to the start's log as
        it is, and to the container's log (docker/rkmon logs) a line each, the empty ones left out, each as written:
        a backslash in it is no escape.
        """
        entry = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        say = entry_tests.function("say")
        start = entry[
            entry.index("\n# rk start undoes") : entry.index("\n# The worker tools")
        ]
        said = (
            "STARTED: the monitor runs again",
            "the one-time move: config.json split by owner",
            r"a line with backslashes: C:\tmp\new, \\ and \c, as written",
        )
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            rk = d / "rk"
            rk.write_text(
                f'#!/bin/sh\n[ "$1" = start ] || exit 2\nprintf "%s\\n\\n" {shlex.quote(said[0])}\n'
                f'printf "%s\\n" {shlex.quote(said[1])} {shlex.quote(said[2])} >&2\n'
            )
            rk.chmod(0o755)
            r = subprocess.run(
                ["sh", "-c", f"RK={rk}\nLOGS={d}\n{say}{start}\n"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                (d / "start.log").read_text(), f"{said[0]}\n\n{said[1]}\n{said[2]}\n"
            )
        self.assertEqual(
            [re.sub(r"^\S+ ", "", line) for line in r.stdout.splitlines()],
            [f"rkmon-container: {line}" for line in said],
        )


class TestTheScriptsNames(unittest.TestCase):
    """A name the container's scripts share has one home, which each reads, or which a test holds its copy to: the
    logs' folder is the layout's (container_layout.LOGS_FOLDER), the file of a login gh's (gh.HOSTS_FILE), and the
    words before each line the entry point says in the container's log control's (control.ENTRY_POINT_SAYS).
    """

    def test_the_entry_point_says_each_line_after_controls_words(self):
        """Each line the entry point says in the container's log (docker/entrypoint.sh, say) comes after its time and
        control.ENTRY_POINT_SAYS, by which the host's start finds them there (docker/rkmon, start_said): the entry
        point, shell, keeps a copy of those words."""
        say = entry_tests.function("say")
        r = subprocess.run(
            ["sh", "-c", f"{say}\nsay 'a line'\n"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertRegex(
            r.stdout, rf"\A\S+ {re.escape(control.ENTRY_POINT_SAYS)}a line\n\Z"
        )

    def test_a_logins_file_is_ghs(self):
        """The root step's move of the write login (docker/rk_root.py LOGIN) and the entry point's word while there is
        no read login look for gh's file of a login, the latter in the read login's folder in the secrets volume (the
        secrets registry's github_read)."""
        from rkmon_lib import gh, secrets_registry

        read = secrets_registry.entry({}, "github_read")
        self.assertEqual(code_tests.rk_root.LOGIN, gh.HOSTS_FILE)
        entry = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn(
            f'[ ! -f "${{RKMON_SECRETS_DIR:-{container_layout.SECRETS_DIR}}}/{read.place}/{gh.HOSTS_FILE}" ]',
            entry,
        )

    def test_the_entry_points_logs_folder_is_the_layouts(self):
        """The entry point's logs' folder, unless a test names another (RKMON_LOGS), is the layout's."""
        entry = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn(
            f"\nLOGS=${{RKMON_LOGS:-{container_layout.WORK_DIR}/{container_layout.LOGS_FOLDER}}}",
            entry,
        )

    def test_the_writers_log_is_in_the_layouts_logs_folder(self):
        """The write broker (docker/rk_write.py) and the writer's run (rkmon_lib/writer_run.py) log to the logs' folder
        the layout names."""
        from test_container_layout import script_with_layout

        want = f"{container_layout.WORK_DIR}/logs9/rk-write.log"
        with tempfile.TemporaryDirectory() as d:
            broker = script_with_layout(
                "rk_write.py", pathlib.Path(d), '\nLOGS_FOLDER = "logs9"\n'
            )
        with mock.patch.object(container_layout, "LOGS_FOLDER", "logs9"):
            spec = importlib.util.spec_from_file_location(
                "writer_run_of_another_layout", ROOT / "rkmon_lib" / "writer_run.py"
            )
            run = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run)
        self.assertEqual((broker.LOG, run.LOG), (want, want))


class TestTheRootPhase(unittest.TestCase):
    """The entry point as root, its root steps stand-ins that note their names and arguments in order, and setpriv one
    that notes the layout's LEAVE_VOLUMES_VARIABLE as the command it drops to has it."""

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.d = pathlib.Path(t.name)
        docker, self.stubs = self.d / "pkg" / "docker", self.d / "bin"
        docker.mkdir(parents=True)
        self.stubs.mkdir()
        self.order, self.kept = self.d / "order", self.d / "kept"
        for step in ("ids", "layout", "root"):
            (docker / f"rk_{step}.py").write_text(
                f"import sys\nopen({str(self.order)!r}, 'a').write(' '.join([{step!r}, *sys.argv[1:]]) + ' ')\n"
            )
        name = container_layout.LEAVE_VOLUMES_VARIABLE
        for stub, body in (
            (
                "id",
                '[ "$*" = -u ] && { echo 0; exit 0; }\n'
                '[ "$1" = -g ] && { echo 1000; exit 0; }\nexec /usr/bin/id "$@"\n',
            ),
            ("setpriv", f'echo "${{{name}-unset}}" > {self.kept}\n'),
        ):
            (self.stubs / stub).write_text("#!/bin/sh\n" + body)
            (self.stubs / stub).chmod(0o755)
        self.script = self.d / "entrypoint.sh"
        entry = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        self.script.write_text(
            entry.replace("PKG=/opt/rocke_monitor", f"PKG={docker.parent}", 1)
        )

    def phase(self, *args: str, **env: str) -> tuple:
        """The root steps the entry point ran with ARGS, and the variable as the command it dropped to has it (None
        when it dropped to none)."""
        for f in (self.order, self.kept):
            f.unlink(missing_ok=True)
        subprocess.run(
            ["sh", str(self.script), *args],
            capture_output=True,
            timeout=30,
            env={"PATH": f"{self.stubs}:/usr/bin:/bin", "HOME": str(self.d), **env},
        )
        kept = self.kept.read_text().strip() if self.kept.exists() else None
        return self.order.read_text(), kept

    def test_a_one_off_that_only_reads_walks_no_file_of_the_volumes(self):
        """A command but `run` with the layout's LEAVE_VOLUMES_VARIABLE set (a one-off container whose command only
        reads) gets the logins' walk's first two steps alone, the users' ids (docker/rk_ids.py --users) and the
        container's own temporary folders (docker/rk_layout.py --system-tmp), so it walks no file of the volumes and
        writes no ids record; the command does not inherit the variable. Without it, and for `run` whatever it says,
        the root phase is whole (tests/test_entrypoint.py)."""
        leave = {container_layout.LEAVE_VOLUMES_VARIABLE: "1"}
        for args in (("rk", "status"), ("shell",), ("claude", "--version")):
            with self.subTest(args=args):
                self.assertEqual(
                    self.phase(*args, **leave),
                    ("ids --users layout --system-tmp ", "unset"),
                )
                self.assertEqual(self.phase(*args), ("ids layout ", "unset"))
        for args in (("run",), ()):
            with self.subTest(args=args):
                self.assertEqual(
                    self.phase(*args, **leave), ("ids layout root start ", "unset")
                )

    def test_code_install_passes_its_option_on(self):
        """code-install, as root, hands the root step its option (code_volume.LEAVE_KEPT: a start's install)."""
        for args in (("code-install",), ("code-install", code_volume.LEAVE_KEPT)):
            with self.subTest(args=args):
                self.assertEqual(
                    self.phase(*args, RKMON_AS="root"),
                    (" ".join(("root", *args)) + " ", None),
                )


class TestTheCodeKept(code_tests.Base):
    """The code volume (docker/rk_root.py), over an image of the tests' own, with a fake root (tests/test_code_volume.py
    Base)."""

    def test_a_starts_install_leaves_the_code_kept_as_it_is(self):
        """A start's install (code-install with code_volume.LEAVE_KEPT: docker/rkmon start_code) leaves KEPT as it is:
        at a first start, none; over an image the apply did not run on, its tag moved by hand, the code from before the
        apply, which a rollback still goes back to. The option is code-install's alone.
        """
        rk_root = code_tests.rk_root
        said = self.volume.install(self.pkg("2026.11.19"), leave_kept=True)
        before = self.current()
        self.assertEqual(
            said,
            [
                f"INSTALLED: release {before.name} (2026.11.19) over this image, in place of the image's own code; no "
                "code is kept"
            ],
        )
        self.assertFalse(os.path.lexists(self.code / code_volume.KEPT))
        applied = self.install("2026.11.20")
        self.assertEqual(self.kept(), f"releases/{before.name}")
        self.idf.write_text("img2\n")  # the image's tag moved by hand
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(
            rk_root.os, "geteuid", lambda: 0
        ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rk_root.main(
                ["code-install", code_volume.LEAVE_KEPT],
                self.pkg("2026.11.20"),
                volume=self.volume,
            )
            refused = rk_root.main(["code-back", code_volume.LEAVE_KEPT])
        started = self.current()
        self.assertEqual((code, refused), (rk_root.DONE, rk_root.USAGE))
        self.assertIn(f"code-install [{code_volume.LEAVE_KEPT}] |", err.getvalue())
        self.assertNotEqual(started, applied)
        self.assertEqual(
            out.getvalue().splitlines(),
            [
                f"INSTALLED: release {started.name} (2026.11.20) over this image, in place of release {applied.name} "
                f"(installed over another image); the code kept stays release {before.name} (installed over another "
                "image): rkmon rollback goes back to it"
            ],
        )
        self.assertEqual(self.kept(), f"releases/{before.name}")
        self.assertEqual(
            self.volume.back(),
            [
                f"RESTORED: release {before.name} (installed over another image) is back, in place of release "
                f"{started.name}"
            ],
        )

    def test_a_file_runs_by_any_execute_bit(self):
        """A file with an execute bit for its owner, its group or the others alone is one its package runs, by the one
        rule (code_volume.runs): the install gives it RUN_MODE, and fingerprints, which spells the rule out, finds the
        package's digest the digest of the release it makes."""
        modes = {"owner": 0o744, "group": 0o654, "others": 0o645, "none": 0o644}
        shutil.rmtree(self.src, ignore_errors=True)
        code_tests.package(self.src, "2026.11.19")
        for name, mode in modes.items():
            code_tests.write(self.src, f"tools/{name}.sh", "true\n", mode)
        self.volume.install(code_tests.archive(self.src))
        release = self.current()
        self.assertEqual(
            {
                name: (release / "tools" / f"{name}.sh").stat().st_mode & 0o7777
                for name in modes
            },
            {
                name: (
                    code_volume.RUN_MODE
                    if code_volume.runs(mode)
                    else code_volume.FILE_MODE
                )
                for name, mode in modes.items()
            },
        )
        self.assertEqual(
            [name for name, mode in modes.items() if code_volume.runs(mode)],
            ["owner", "group", "others"],
        )
        self.assertEqual(
            code_volume.digest(self.src), code_volume.release_info(release)["digest"]
        )


if __name__ == "__main__":
    unittest.main()
