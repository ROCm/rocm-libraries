# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 5's rk commands: `rk logins` (every login and secret something may need, in the secrets registry's order, with
the features that need it, its state and the command that sets it, from made-up probe results), `rk secret-set` (every
secret the registry lists and the claude login's key, from stdin into its file, mode 600, never printed nor its length;
empty or wrong-kind input refused; a login kept in a folder, an older name or a name the registry does not list a usage
error; --remove) and `rk read-login` (a GitHub token as the read login, probed in a folder of its own first: one that
can write is refused, as is every token while repository.name names no public repository, and nothing changes; one that
reads only takes the read login's place whole; never printed); and the atlassian site check that names a site the login
does not reach.
"""

import io
import json
import os
import pathlib
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    atlassian,
    bindings,
    config,
    container_layout,
    doctor,
    gh,
    ghlogins,
    logins,
    reach,
    writer,
)
from rkmon_lib import secrets_registry as sr  # noqa: E402
from rendered import one_line  # noqa: E402
from test_secrets_registry import a_value  # noqa: E402  (an invented value of a shape)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.cfg = config.normalize(
            {
                "repo": "acme/widgets",
                "operator": {"github_login": "op-login"},
                "paths": {
                    "working_dir": str(self.d / "work"),
                    "secrets_dir": str(self.d / "secrets"),
                },
                "features": {},
            }
        )  # with the places the load derives


class TestLogins(Base):
    def setUp(self):
        super().setUp()
        self.write_secrets = self.d / "secrets-write"  # the writer's volume
        env = dict(
            os.environ, RKMON_CONTAINER="1"
        )  # the Claude session is the container's
        env.pop(
            writer.BROKER_VAR, None
        )  # this process is the writer: it reads its own volume
        for p in (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                container_layout, "WRITE_SECRETS_DIR", str(self.write_secrets)
            ),
        ):
            p.start()
            self.addCleanup(p.stop)
        writer.forget()
        self.addCleanup(writer.forget)

    def rows(self, sides, features=None):
        cfg = dict(self.cfg, features=features or {})
        with mock.patch.object(
            atlassian, "status", return_value={"found": False, "logged_in": False}
        ):
            return {
                r["name"]: r
                for r in logins.rows(cfg, probe=lambda c: ({"sides": sides}, True))
            }

    def test_the_order_and_what_is_needed(self):
        """Each login and secret something may need, in the secrets registry's order, by its registry name, with the
        command that sets it."""
        r = self.rows({"read": {"login": "reader", "can_write": False}})
        self.assertEqual(list(r), [e.name for e in sr.walk_entries(self.cfg)])
        self.assertEqual(
            (r["github_read"]["state"], r["github_read"]["needed_by"]),
            ("SET", [sr.ALWAYS_WORDS]),
        )
        self.assertEqual(
            (r["claude"]["state"], r["claude"]["needed_by"]),
            ("MISSING", ["Claude session"]),
        )  # review automation is enabled by default: its Claude session runs
        self.assertEqual(
            (r["admin_access_token"]["state"], r["admin_access_token"]["needed_by"]),
            ("MISSING", ["admin dashboard"]),
        )  # the admin dashboard is on by default, and every request to it carries the token
        for name in (
            "github_write",
            "atlassian",
            "teams_webhook",
            "web_host_ssh_key",
            "gpu_ssh_key",
        ):
            self.assertEqual(
                r[name]["state"], "NOT NEEDED", name
            )  # nothing goes out in prototype
        for name, row in r.items():
            self.assertEqual(
                row["command"], " ".join(sr.set_command(sr.entry(self.cfg, name)))
            )
        r = self.rows({}, {"review_automation": {"enabled": False}})
        self.assertEqual(r["claude"]["state"], "NOT NEEDED")

    def test_missing_and_wrong(self):
        r = self.rows(
            {
                "read": {
                    "login": "op-login",
                    "can_write": True,
                    "why": "a comment and a push: HTTP 422 (Validation Failed): allowed",
                },
                "write": {"login": None, "can_write": None},
            },
            {
                "review_automation": {"mode": "live"},
                "alerts": {
                    "enabled": True,
                    "mode": "live",
                    "develop_breaks": True,
                    "needs_you": True,
                },
            },
        )
        self.assertEqual(r["github_read"]["state"], "WRONG")
        self.assertEqual(
            r["github_read"]["found"],
            "op-login can write (a comment and a push: HTTP 422 (Validation Failed): allowed): log a read-only "
            "token in",
        )
        self.assertEqual(
            (r["github_write"]["state"], r["github_write"]["needed_by"]),
            ("MISSING", ["review posting"]),
        )
        self.assertEqual(r["claude"]["state"], "MISSING")
        self.assertEqual(
            (r["teams_webhook"]["state"], r["teams_webhook"]["needed_by"]),
            ("MISSING", ["develop break alerts", "needs-you alerts"]),
        )
        text = logins.text(list(r.values()), "./rkmon")
        for fix in (
            "login github_read",
            "login github_write",
            "login claude",
            "secret admin_access_token --generate",
            "secret teams_webhook",
        ):
            self.assertIn(f"   fix: ./rkmon {fix}\n", text)
        self.assertTrue(
            text.endswith(
                "\n5 to do: github_read, github_write, claude, admin_access_token, teams_webhook"
            ),
            text,
        )

    def test_the_write_login_is_judged_by_what_is_on(self):
        """The write login must comment while something that is on writes to GitHub, and push as well while something
        that is on turns auto-merge on or off: one that may comment but not push is SET while reviews post alone, and
        says what that leaves; WRONG once the merge queue would turn auto-merge on, and while the auto-merge guard
        acts, which turns it off, with auto-merge "disabled" too. One that may push but not comment is WRONG while
        anything writes."""
        why = "a comment: HTTP 422 (Validation Failed): allowed; a push: HTTP 403: refused"
        sides = {
            "read": {"login": "reader", "can_write": False},
            "write": {
                "login": "writer",
                "can_comment": True,
                "can_push": False,
                "can_write": True,
                "why": why,
            },
        }
        posting = {"review_automation": {"mode": "live"}}
        merging = dict(
            posting,
            merge_queue={"enabled": True, "mode": "live", "auto_merge": "enabled"},
        )
        said = f"writer can comment but not push: {ghlogins.NO_PUSH}"
        r = self.rows(sides, posting)["github_write"]
        self.assertEqual((r["state"], r["found"]), ("SET", said))
        r = self.rows(sides, merging)["github_write"]
        self.assertEqual((r["state"], r["found"]), ("WRONG", f"{said} ({why})"))
        guarding = dict(
            posting,
            merge_queue={"enabled": True, "mode": "live", "auto_merge": "disabled"},
        )
        r = self.rows(sides, guarding)["github_write"]
        self.assertEqual((r["state"], r["found"]), ("WRONG", f"{said} ({why})"))
        sides["write"].update(can_comment=False, can_push=True)
        r = self.rows(sides, posting)["github_write"]
        self.assertEqual(r["state"], "WRONG")
        self.assertTrue(
            r["found"].startswith(
                "writer can push but not comment: rk post and every other write fail"
            ),
            r["found"],
        )

    def test_an_ssh_key_is_wrong_while_its_host_s_own_key_is_not_recorded(self):
        """The ssh key of a host ssh reaches is WRONG while the state records none of the host's own keys, the only
        keys ssh trusts it by (rkmon_lib/host_keys.py): its row's command is `login`, deploy step 5, which shows that
        key and records it once you accept it (docker/rkmon logins_walk); so too under its older name, which the walk
        counts as set until the one-time move copies it. SET, saying so, once one is recorded. While
        nothing on needs it, it is a secret like any other."""
        (self.d / "secrets").mkdir()
        (self.d / "secrets" / "gpu_ssh_key").write_text(a_value(sr.SSH_PRIVATE_KEY, 1))
        self.write_secrets.mkdir()
        (self.write_secrets / "web_host_ssh_key").write_text(
            a_value(sr.SSH_PRIVATE_KEY, 2)
        )
        cfg = config.normalize(
            dict(
                self.cfg,
                features={"dashboards": {"team": {"enabled": True, "mode": "live"}}},
                gpu={"enabled": True, "login_host": "login.example.com"},
                publish={"html": {"host": "web.example.com"}},
            )
        )
        hosts = (
            ("web_host_ssh_key", "the web host"),
            ("gpu_ssh_key", "the GPU login host"),
        )
        for name, words in hosts:
            self.assertEqual(
                logins.state(cfg, sr.entry(cfg, name)),
                ("WRONG", f"the key is set; {words}'s own key is not recorded"),
                name,
            )
        with mock.patch.object(
            atlassian, "status", return_value={"found": False, "logged_in": False}
        ):
            rs = logins.rows(cfg, probe=lambda c: ({"sides": {}}, True))
        text = logins.text(rs, "./rkmon")
        for r in rs:
            self.assertEqual(r["host_key"], r["name"] in dict(hosts), r["name"])
        for name, words in hosts:
            (r,) = [x for x in rs if x["name"] == name]
            self.assertEqual((r["state"], r["command"]), ("WRONG", "login"))
            self.assertIn(
                f"   the key is set; {words}'s own key is not recorded\n"
                "   fix: ./rkmon login shows it, and records it once you accept it\n",
                text,
            )
        # Under its older name, until the one-time move copies it to its place: the same.
        (self.write_secrets / "web_host_ssh_key").unlink()
        older = self.d / "secrets" / "webhost_ed25519"
        older.write_text(a_value(sr.SSH_PRIVATE_KEY, 2))
        self.assertEqual(
            logins.state(cfg, sr.entry(cfg, "web_host_ssh_key")),
            ("WRONG", "the key is set; the web host's own key is not recorded"),
        )
        cfg["publish"]["html"]["host_keys"] = [
            "web.example.com ssh-ed25519 AAAAExample"
        ]
        cfg["remote_gpu_tests"] = {
            "login_host_keys": ["login.example.com ssh-ed25519 AAAAExample"]
        }
        for name, words in hosts:
            st, found_ = logins.state(cfg, sr.entry(cfg, name))
            self.assertEqual(st, "SET", name)
            self.assertTrue(found_.endswith(f"; {words}'s own key is recorded"), found_)
        self.assertTrue(
            logins.state(cfg, sr.entry(cfg, "web_host_ssh_key"))[1].startswith(
                f"{older}, under its older name webhost_ed25519: "
            )
        )
        # A key recorded for the host or port the settings named before is none of the host's (host_keys.recorded).
        cfg["publish"]["html"]["host"] = "web2.example.com"
        cfg["gpu"]["ssh_port"] = 2222
        for name, words in hosts:
            self.assertEqual(logins.state(cfg, sr.entry(cfg, name))[0], "WRONG", name)
        cfg = dict(self.cfg, features={})
        for name, words in hosts:
            st, found_ = logins.state(cfg, sr.entry(cfg, name))
            self.assertEqual(st, "SET", name)
            self.assertNotIn("own key", found_)

    def test_a_secret_set_is_seen(self):
        """The Teams webhook is the writer's: in the container it is kept in the writer's volume. A copy
        in the monitor's secrets folder is its older place, set until the one-time move copies it (state 7).
        """
        older = self.d / "secrets" / "teams_webhook"
        older.parent.mkdir()
        older.write_text("https://hooks.example.invalid/x\n")
        alerts = {
            "alerts": {
                "enabled": True,
                "mode": "live",
                "develop_breaks": True,
                "needs_you": True,
            }
        }
        r = self.rows({}, alerts)
        self.assertEqual(
            (r["teams_webhook"]["state"], r["teams_webhook"]["found"]),
            (
                "SET",
                f"{older}, its older place: the one-time move copies it to its place",
            ),
        )
        older.unlink()
        r = self.rows({}, alerts)
        self.assertEqual(
            (r["teams_webhook"]["state"], r["teams_webhook"]["found"]),
            ("MISSING", str(self.write_secrets / "teams_webhook")),
        )
        self.write_secrets.mkdir()
        (self.write_secrets / "teams_webhook").write_text(
            "https://hooks.example.invalid/x\n"
        )
        r = self.rows({}, alerts)
        self.assertEqual(r["teams_webhook"]["state"], "SET")
        self.assertNotIn("hooks.example.invalid", json.dumps(r))

    def test_the_first_review_pass_s_key_under_its_older_name(self):
        """The key the local worker's binding read is the first review pass's now: while a binding names that key, a
        copy under the older name counts as set until the one-time move copies it (state 7). With no
        binding that names it, no row has it."""
        older = self.d / "secrets" / "worker_api_key"
        older.parent.mkdir()
        older.write_text("invented-key\n")
        drafts = {"review_automation": {"enabled": True}}
        self.cfg["worker"] = {"api_key_secret": sr.FIRST_PASS_KEY}
        self.cfg["reviewer"] = (
            bindings.WORKER
        )  # the local worker runs the first review pass
        row = self.rows({}, drafts)[sr.FIRST_PASS_KEY]
        self.assertEqual(
            (row["state"], row["found"]),
            (
                "SET",
                f"{older}, under its older name worker_api_key: the one-time move copies it to its place",
            ),
        )
        older.unlink()
        self.assertEqual(self.rows({}, drafts)[sr.FIRST_PASS_KEY]["state"], "MISSING")
        del self.cfg["worker"]
        older.write_text("invented-key\n")
        self.assertNotIn(sr.FIRST_PASS_KEY, self.rows({}, drafts))

    def test_the_claude_login_under_its_older_name(self):
        """An older package kept the Claude session's key in the secret claude_env, beside the session's other
        settings: a copy there counts as set until the one-time move copies the key to the claude login's file
        (state 7). Never a value: only where it is."""
        secrets = self.d / "secrets"
        older = secrets / "claude_env"
        secrets.mkdir()
        older.write_text('{"ANTHROPIC_API_KEY": "Invented-Key-Q"}\n')
        row = self.rows({})["claude"]
        self.assertEqual(
            (row["state"], row["found"]),
            (
                "SET",
                f"{older}, under its older name claude_env: the one-time move copies it to its place",
            ),
        )
        older.unlink()
        row = self.rows({})["claude"]
        self.assertEqual(
            (row["state"], row["found"]), ("MISSING", str(secrets / "claude_key"))
        )
        (secrets / "claude_key").write_text("Invented-Key-Q\n")
        row = self.rows({})["claude"]
        self.assertEqual(
            (row["state"], row["found"]), ("SET", str(secrets / "claude_key"))
        )
        self.assertNotIn("Invented-Key-Q", json.dumps(self.rows({})))

    def test_the_monitor_asks_the_writer(self):
        """The monitor cannot read the writer's volume: whether each of the writer's secrets is set is the writer's
        answer (`rk writer-secrets --json`, through sudo), never a value; no answer is UNKNOWN, not MISSING.
        """
        self.write_secrets.mkdir()
        (self.write_secrets / "teams_webhook").write_text(
            "https://hooks.example.invalid/x\n"
        )  # the monitor never looks at it
        asked = []

        def through(argv, input=None, timeout_s=None):
            asked.append(argv)
            return subprocess.CompletedProcess(argv, 0, json.dumps(answer), "")

        answer = {
            "teams_webhook": {"set": False, "where": "/w/teams_webhook", "mark": ""}
        }
        with mock.patch.dict(
            os.environ, {writer.BROKER_VAR: "/w/rk_write.py"}
        ), mock.patch.object(writer, "through", through):
            r = self.rows(
                {},
                {
                    "alerts": {
                        "enabled": True,
                        "mode": "live",
                        "develop_breaks": True,
                        "needs_you": True,
                    }
                },
            )
            self.assertEqual(
                (r["teams_webhook"]["state"], r["teams_webhook"]["found"]),
                ("MISSING", "/w/teams_webhook"),
            )
            self.assertEqual(asked, [writer.SECRETS_PROBE])  # asked once, then kept
            writer.forget()
            answer = {}
            r = self.rows(
                {},
                {
                    "alerts": {
                        "enabled": True,
                        "mode": "live",
                        "develop_breaks": True,
                        "needs_you": True,
                    }
                },
            )
            self.assertEqual(
                (r["teams_webhook"]["state"], r["teams_webhook"]["found"]),
                ("UNKNOWN", logins.NOT_SAID),
            )
            self.assertEqual(
                r["claude"]["found"], str(self.d / "secrets" / "claude_key")
            )  # the monitor's own: looked at here

    def test_a_model_binding_s_key_comes_last(self):
        """A secret a model binding names is walked after the fixed entries, needed while its reviews run on the
        binding."""
        self.cfg["model_bindings"] = {"b": {"api_key_secret": "b_api_key"}}
        self.cfg["reviewer"] = "b"
        rows = list(self.rows({}).values())
        self.assertEqual(
            {k: rows[-1][k] for k in ("name", "state", "needed_by", "command")},
            {
                "name": "b_api_key",
                "state": "MISSING",
                "needed_by": ["reviews"],
                "command": "secret b_api_key",
            },
        )
        rows = list(self.rows({}, {"review_automation": {"enabled": False}}).values())
        self.assertEqual(
            (rows[-1]["name"], rows[-1]["state"]), ("b_api_key", "NOT NEEDED")
        )

    def test_an_older_copy_counts_as_set_until_the_move_ends(self):
        """In state 7, an entry missing from its own place but found under an older name that the one-time
        move copies to it is SET, and says where; one the move only removes (not `copied`) is no copy of it.
        """
        entry = sr.Secret(
            "new_secret",
            "a secret",
            sr.ONE_LINE,
            "you",
            sr.MONITOR,
            (sr.ALWAYS,),
            older=(sr.Older("old_secret"), sr.Older("left_secret", copied=False)),
        )
        secrets = self.d / "secrets"
        secrets.mkdir()

        def row():
            with mock.patch.object(sr, "_fixed", return_value=(entry,)):
                r = self.rows({})["new_secret"]
            return r["state"], r["found"]

        self.assertEqual(row(), ("MISSING", str(secrets / "new_secret")))
        (secrets / "left_secret").write_text("x\n")
        self.assertEqual(row(), ("MISSING", str(secrets / "new_secret")))
        (secrets / "old_secret").write_text("x\n")
        self.assertEqual(
            row(),
            (
                "SET",
                f"{secrets / 'old_secret'}, under its older name old_secret: the one-time move copies it to its "
                "place",
            ),
        )
        (secrets / "new_secret").write_text("x\n")
        self.assertEqual(row(), ("SET", str(secrets / "new_secret")))
        # The admin dashboard's older token is made anew, never copied: it is no copy of the access token.
        (secrets / "admin_token").write_text("x\n")
        self.assertEqual(self.rows({})["admin_access_token"]["state"], "MISSING")


class TestSecretSet(Base):
    def run_(self, name, value):
        out = io.StringIO()
        rc = rkmon.cmd_secret_set(name, self.cfg, stdin=io.StringIO(value), out=out)
        return rc, out.getvalue()

    def test_it_writes_mode_600_and_never_prints(self):
        # The secret's path part has capitals and a hyphen, which no temporary folder's name holds.
        rc, out = self.run_(
            "teams_webhook", "https://hooks.example.invalid/Hook-Path-Q\n"
        )
        self.assertEqual(rc, 0, out)
        f = self.d / "secrets" / "teams_webhook"
        self.assertEqual(f.read_text(), "https://hooks.example.invalid/Hook-Path-Q\n")
        self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE((self.d / "secrets").stat().st_mode) & 0o077, 0)
        self.assertNotIn("Hook-Path-Q", out)
        self.assertNotIn("hooks.example.invalid", out)
        self.assertIn("not shown", out)

    def test_empty_or_the_wrong_kind_is_refused(self):
        self.assertEqual(self.run_("teams_webhook", "\n")[0], 4)
        rc, out = self.run_("web_host_ssh_key", "not a key")
        self.assertEqual(rc, 4)
        self.assertIn("nothing written", out)
        self.assertFalse((self.d / "secrets" / "web_host_ssh_key").exists())
        rc, out = self.run_("claude", '{"ANTHROPIC_API_KEY": "Invented-Key-Q"}')
        self.assertEqual(rc, 4)  # the claude login is the key alone
        self.assertNotIn("Invented-Key-Q", out)
        self.assertFalse((self.d / "secrets" / "claude_key").exists())

    def test_a_link_in_the_secret_s_place_is_refused(self):
        """The secret is written as a private file: a link in its place is refused, and the fix removes the link
        alone, then runs the command again: neither another file in the secret's own key (the load derives that place,
        and keeps none config.json gives) nor another secrets folder, which every other secret would miss. The file
        it points to stays as it was, and the link stays.
        """
        d = self.d / "secrets"
        d.mkdir(mode=0o700)
        elsewhere = self.d / "elsewhere"
        elsewhere.write_text("left alone\n")
        (d / "teams_webhook").symlink_to(elsewhere)
        home = self.d / "home"
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}):
            rc, out = self.run_(
                "teams_webhook", "https://hooks.example.invalid/Hook-Path-Q\n"
            )
        self.assertEqual(rc, 4, out)
        said = one_line(out)
        self.assertIn("REFUSED: teams_webhook: nothing written:", said)
        link = d / "teams_webhook"
        self.assertIn(f"{link} is a link, not a file", said)
        self.assertIn(
            f"fix: rm {shlex.quote(str(link))} (the link alone), then run the command again",
            said,
        )
        self.assertNotIn("teams.webhook_file", said)
        self.assertNotIn("paths.secrets_dir", said)
        self.assertTrue((d / "teams_webhook").is_symlink())
        self.assertEqual(elsewhere.read_text(), "left alone\n")
        self.assertNotIn("Hook-Path-Q", out)

    def test_every_secret_the_registry_lists_and_never_its_value_nor_its_length(self):
        """rk secret-set takes each secret the registry lists for the configuration, a model binding's key among them,
        and each login kept in a file, the claude login's key (`rkmon login claude` sets it through here); its one
        outcome line names the file, never the value nor its length."""
        self.cfg["model_bindings"] = {"b": {"api_key_secret": "b_api_key"}}
        secrets = [e for e in sr.entries(self.cfg) if sr.held_files(e) is None]
        self.assertIn("b_api_key", [e.name for e in secrets])
        self.assertIn("claude", [e.name for e in secrets])
        for i, e in enumerate(secrets):
            value = a_value(e.shape, i)
            with self.subTest(name=e.name):
                rc, out = self.run_(e.name, value)
                f = pathlib.Path(sr.path(self.cfg, e))
                self.assertEqual(
                    (rc, out), (0, f"{e.name}: set ({f}, mode 600; not shown)\n")
                )
                self.assertEqual(f.read_text(), value.strip() + "\n")
                self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)

    def test_a_login_or_a_name_the_registry_does_not_list_is_a_usage_error(self):
        for name, why in (
            (
                "github_write",
                "github_write is a login: rkmon login github_write sets it",
            ),
            ("github_read", "github_read is a login: rkmon login github_read sets it"),
            (
                "gh-read",
                "gh-read is the older name of the login github_read: rkmon login github_read sets it",
            ),
            ("teams-webhook", "no secret is named teams-webhook"),
            (
                "webhost_ed25519",
                "webhost_ed25519 is the older name of the secret web_host_ssh_key: rkmon secret "
                "web_host_ssh_key sets it",
            ),
            ("b_api_key", "no secret is named b_api_key"),  # no binding names it
            (
                "admin_token",
                "admin_token is the older name of the secret admin_access_token: rkmon secret "
                "admin_access_token --generate sets it",
            ),
            (
                "claude_env",
                "claude_env is the older name of the login claude: rkmon login claude sets it",
            ),
        ):
            with self.subTest(name=name), mock.patch(
                "sys.stderr", new_callable=io.StringIO
            ) as err:
                self.assertEqual(self.run_(name, a_value(sr.ONE_LINE, 3)), (2, ""))
                self.assertIn(
                    f"rk secret-set: {why} (the names it takes: claude, admin_access_token, ",
                    err.getvalue(),
                )
        self.assertFalse((self.d / "secrets").exists())  # nothing written

    def test_remove_takes_only_the_file_at_its_own_place(self):
        """--remove removes the secret's file at its own place in the secrets folder; a file a configuration names in
        the secret's config key is left as it is, as the load keeps no place a configuration gives (derived.py).
        """

        def remove(cfg):
            out = io.StringIO()
            return rkmon.cmd_secret_set("teams_webhook", cfg, out=out, remove=True), (
                out.getvalue()
            )

        self.assertEqual(self.run_("teams_webhook", a_value(sr.HTTPS_URL, 1))[0], 0)
        own = self.d / "secrets" / "teams_webhook"
        self.assertEqual(remove(self.cfg), (0, f"teams_webhook: removed ({own})\n"))
        self.assertFalse(own.exists())
        self.assertEqual(
            remove(self.cfg), (0, f"teams_webhook: not set ({own}); nothing removed\n")
        )
        elsewhere = self.d / "elsewhere"
        elsewhere.write_text("left alone\n")
        self.assertEqual(self.run_("teams_webhook", a_value(sr.HTTPS_URL, 2))[0], 0)
        given = config.normalize(dict(self.cfg, teams={"webhook_file": str(elsewhere)}))
        self.assertEqual(remove(given), (0, f"teams_webhook: removed ({own})\n"))
        self.assertFalse(own.exists())
        self.assertEqual(elsewhere.read_text(), "left alone\n")


class TestTheDoctorAgrees(Base):
    """The doctor says what the logins step says: no write login is no failure while nothing writes, and no webhook is
    no warning while nothing posts to the channel."""

    def test_no_write_login_while_nothing_writes(self):
        sides = {
            "read": {"dir": "/secrets/gh-read", "login": "reader", "can_write": False},
            "write": {
                "dir": "/secrets-write/gh",
                "login": None,
                "can_write": False,
                "why": "no gh login",
            },
        }
        with mock.patch(
            "rkmon_lib.ghlogins.folders", return_value={"read": "r", "write": "w"}
        ):
            checks = doctor.gh_split_checks(
                self.cfg, check=lambda c, s, f: ({"sides": sides}, True)
            )
            self.assertEqual(
                {c.name: c.status for c in checks},
                {"GitHub read login": "PASS", "GitHub write login": "OFF"},
            )
            posting = dict(self.cfg, features={"review_automation": {"mode": "live"}})
            checks = doctor.gh_split_checks(
                posting, check=lambda c, s, f: ({"sides": sides}, True)
            )
            self.assertEqual(
                [c.status for c in checks if c.name == "GitHub write login"], ["FAIL"]
            )

    def test_no_webhook_while_nothing_posts_to_the_channel(self):
        """The row's commands are the wrapper's, as the operator types them (RKMON_WRAPPER)."""
        cfg = dict(self.cfg, teams={"channel": "ops"})
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            self.assertEqual(reach.chat(cfg)[0].status, "OFF")
            cfg["features"] = {
                "alerts": {"enabled": True, "develop_breaks": True, "needs_you": True}
            }
            (row,) = reach.chat(cfg)
            self.assertEqual(
                (row.status, row.found),
                (
                    "OFF",
                    "features.alerts.mode is prototype: nothing goes out until it is live, so no webhook is needed "
                    "yet (./rkmon secret teams_webhook sets it)",
                ),
            )  # in prototype the alerts send nothing
            cfg["features"]["alerts"]["mode"] = "live"
            (row,) = reach.chat(cfg)
            self.assertEqual(
                (row.status, str(row.fix)),
                (
                    "WARN",
                    "./rkmon secret teams_webhook, and give an https:// address on one line",
                ),
            )  # the secret's own fix (secrets_registry.set_fix)


class TestReadLogin(Base):
    """`rk read-login` (ghlogins.set_read), with gh's login and probe made up: the token is logged in to a folder of its
    own first, where the probe asks whether it can write; only one that reads only goes on to the read login's own
    folder, whole, in that folder's place. The token is never printed."""

    TOKEN = "invented-read-" + "token-Q7"
    READS_ONLY = {
        "login": "reader",
        "can_write": False,
        "why": "HTTP 403: writes are refused",
    }

    def setUp(self):
        super().setUp()
        self.read = pathlib.Path(sr.path(self.cfg, sr.entry(self.cfg, "github_read")))

    @staticmethod
    def hosts(*accounts) -> str:
        """gh's hosts.yml with an entry for each of ACCOUNTS, the last one the active one (made up: no token in it)."""
        users = "".join(f"        {a}: {{}}\n" for a in accounts)
        return f"github.com:\n    users:\n{users}    user: {accounts[-1]}\n"

    @staticmethod
    def accounts(hosts: str) -> list:
        """The accounts the hosts.yml text HOSTS holds an entry for."""
        return [
            ln.strip()[: -len(": {}")]
            for ln in hosts.splitlines()
            if ln.startswith(" " * 8) and ln.endswith(": {}")
        ]

    def run_(self, probed, took=("",)):
        """set_read with PROBED as the probe's answer and TOOK as gh's words at each login ("": it took the token), in
        the container, as `rkmon login github_read` runs it: (exit code, its output, the folders it logged the token
        in to). The login is gh's: an entry for the token's account (reader) in the folder's hosts.yml, beside the
        entries of the accounts logged in there before."""
        logged = []

        def login(folder, token):
            self.assertEqual(token, self.TOKEN)
            logged.append(folder)
            why = took[len(logged) - 1]
            if not why:
                f = pathlib.Path(folder)
                f.mkdir(mode=0o700, parents=True, exist_ok=True)
                hosts = f / gh.HOSTS_FILE
                had = self.accounts(hosts.read_text()) if hosts.is_file() else []
                hosts.write_text(
                    self.hosts(*[a for a in had if a != "reader"], "reader")
                )
                (f / "config.yml").write_text("version: 1\n")
            return why

        def probe(folder, repo, issue):
            self.assertEqual((folder, repo), (logged[-1], "acme/widgets"))
            self.assertTrue(pathlib.Path(folder, gh.HOSTS_FILE).is_file())
            return probed

        out = io.StringIO()
        with mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        ):
            rc = ghlogins.set_read(
                self.cfg, self.TOKEN + "\n", out=out, login=login, probe=probe
            )
        self.assertNotIn(self.TOKEN, out.getvalue())
        return rc, out.getvalue(), logged

    def test_a_token_that_reads_only_goes_in_to_the_read_login_s_own_folder(self):
        rc, out, logged = self.run_(self.READS_ONLY)
        self.assertEqual(
            (rc, out), (0, f"github_read: set ({self.read}; reader reads only)\n")
        )
        self.assertEqual(len(logged), 1)  # once: no second login
        self.assertNotEqual(logged[0], str(self.read))  # probed in a folder of its own
        self.assertFalse(os.path.exists(logged[0]))  # which is gone
        # its files make the read login's folder, 700, its files 600, and nothing is left beside it
        self.assertEqual(
            {p.name: p.read_text() for p in self.read.iterdir()},
            {gh.HOSTS_FILE: self.hosts("reader"), "config.yml": "version: 1\n"},
        )
        self.assertEqual(stat.S_IMODE(self.read.stat().st_mode), 0o700)
        self.assertEqual(
            {stat.S_IMODE(p.stat().st_mode) for p in self.read.iterdir()}, {0o600}
        )
        self.assertEqual(os.listdir(self.read.parent), [self.read.name])

    def test_a_new_login_takes_the_earlier_one_s_place_whole(self):
        """gh keeps an entry for each account in its folder, so a token logged in over an earlier login would leave
        that account's token there, inactive. The probed login's files take the folder's place instead: hosts.yml
        holds its one account, and nothing of the earlier login stays."""
        self.read.mkdir(parents=True, mode=0o700)
        (self.read / gh.HOSTS_FILE).write_text(self.hosts("earlier"))
        (self.read / "state.yml").write_text("checked: never\n")
        rc, out, _logged = self.run_(self.READS_ONLY)
        self.assertEqual(rc, 0, out)
        self.assertEqual(
            self.accounts((self.read / gh.HOSTS_FILE).read_text()), ["reader"]
        )
        self.assertEqual(sorted(os.listdir(self.read)), ["config.yml", gh.HOSTS_FILE])
        self.assertEqual(os.listdir(self.read.parent), [self.read.name])

    def test_a_failed_rename_leaves_the_earlier_login_as_it_was(self):
        """The new login's folder is renamed into the read login's place once the earlier one is set aside: when that
        rename fails, the earlier login is put back as it was, and the new folder goes.
        """
        self.read.mkdir(parents=True, mode=0o700)
        (self.read / gh.HOSTS_FILE).write_text(self.hosts("earlier"))
        rename = os.rename

        def failing(src, dst):
            if os.fspath(src).endswith(".new"):
                raise OSError(5, "Input/output error")
            rename(src, dst)

        with mock.patch.object(ghlogins.os, "rename", failing):
            rc, out, _logged = self.run_(self.READS_ONLY)
        self.assertEqual(
            (rc, out),
            (
                1,
                f"ERROR: github_read: reader reads only, but its login could not be put in {self.read} ([Errno 5] "
                "Input/output error); nothing changed\n",
            ),
        )
        self.assertEqual(os.listdir(self.read), [gh.HOSTS_FILE])
        self.assertEqual((self.read / gh.HOSTS_FILE).read_text(), self.hosts("earlier"))
        self.assertEqual(os.listdir(self.read.parent), [self.read.name])

    def test_what_an_earlier_read_login_left_is_removed(self):
        """A kill between those steps leaves a new folder or a set-aside one beside the read login's: the next read
        login removes them first. One it cannot remove is named after the set line, and the next one removes it.
        """
        self.read.mkdir(parents=True, mode=0o700)
        (self.read / gh.HOSTS_FILE).write_text(self.hosts("earlier"))
        for end in ("k1ll3d00.new", "k1ll3d01.old"):
            left = self.read.parent / f".{self.read.name}.{end}"
            left.mkdir(mode=0o700)
            (left / gh.HOSTS_FILE).write_text(self.hosts("left"))
        set_line = f"github_read: set ({self.read}; reader reads only)\n"
        self.assertEqual(self.run_(self.READS_ONLY)[:2], (0, set_line))
        self.assertEqual(os.listdir(self.read.parent), [self.read.name])
        rmtree = ghlogins.shutil.rmtree

        def refused(path, *args, **kwargs):
            if os.fspath(path).endswith(".old"):
                raise PermissionError(13, "Permission denied")
            rmtree(path, *args, **kwargs)

        with mock.patch.object(ghlogins.shutil, "rmtree", refused):
            rc, out, _logged = self.run_(self.READS_ONLY)
        (aside,) = [n for n in os.listdir(self.read.parent) if n != self.read.name]
        aside = pathlib.Path(os.path.realpath(self.read.parent)) / aside
        self.assertEqual(
            (rc, out),
            (
                0,
                f"{set_line}github_read: {aside}, left by an earlier login, could not be removed (Permission "
                "denied)\n",
            ),
        )
        self.assertEqual(self.run_(self.READS_ONLY)[:2], (0, set_line))
        self.assertEqual(os.listdir(self.read.parent), [self.read.name])

    def test_with_no_read_login_yet_the_older_login_is_left_as_it_is(self):
        """While there is no read login, gh reads with the login an older package kept in the monitor's secrets folder
        (gh.login_dirs "unsplit"): the token goes in to the read login's own folder, never that one.
        """
        older = self.d / "secrets" / "gh"
        older.mkdir(parents=True)
        (older / gh.HOSTS_FILE).write_text("github.com: {}\n")
        self.assertEqual(gh.login_dirs(self.cfg).get("unsplit"), str(older))
        rc, out, logged = self.run_(self.READS_ONLY)
        self.assertEqual(rc, 0, out)
        self.assertEqual(
            self.accounts((self.read / gh.HOSTS_FILE).read_text()), ["reader"]
        )
        self.assertNotIn(str(older), logged)
        self.assertEqual((older / gh.HOSTS_FILE).read_text(), "github.com: {}\n")

    def test_a_token_that_can_write_is_refused_and_nothing_changes(self):
        rc, out, logged = self.run_(
            {
                "login": "op-login",
                "can_write": True,
                "why": "HTTP 422 (validation): writes are allowed",
            }
        )
        self.assertEqual(rc, 4, out)
        said = one_line(out)
        self.assertTrue(
            said.startswith("REFUSED: github_read: nothing changed: FAIL"), said
        )
        self.assertIn(
            "op-login can write to acme/widgets (HTTP 422 (validation): writes are allowed); the read login must "
            "not fix: on GitHub, make a fine-grained token with acme as its resource owner that only reads "
            "acme/widgets and acme's team members, then ./rkmon login github_read",
            said,
        )
        self.assertEqual(len(logged), 1)
        self.assertFalse(os.path.exists(logged[0]))
        self.assertFalse(self.read.exists())

    def test_github_s_404_is_refused_with_the_repository_s_name_or_the_probe_s_issue_as_its_fix(
        self,
    ):
        """A 404 to the probe: repository.name names no public repository (the monitor watches public ones only, so a
        private one is a wrong name, not a wrong token), or the probe's issue is gone. Running the command again gets
        the same answer, and so does another token, so the row names both causes, and its fix is each one's cure:
        repository.name set to a public repository, then confirm and apply, or another probe issue, which the package's
        maintainers name in the repository's defaults; nothing changes. A 5xx, no answer in time or no network stay not
        known, where running it again is the fix (test_what_is_refused_or_not_known_changes_nothing).
        """
        rc, out, logged = self.run_(
            {
                "login": "reader",
                "http": 404,
                "can_write": None,
                "why": "gh: Not Found (HTTP 404)",
            }
        )
        self.assertEqual(rc, 4, out)
        self.assertEqual(
            one_line(out),
            "REFUSED: github_read: nothing changed: FAIL github_read: repository.name, acme/widgets, names no public "
            "repository, or GitHub finds no issue 1 in it, which the probe comments on (gh: Not Found (HTTP 404)) "
            "fix: set repository.name in your deploy settings file to the public repository as GitHub names it, "
            "<org>/<repo>, then ./rkmon confirm and ./rkmon apply; when repository.name is right, the probe's issue "
            "is gone: the package's maintainers name another in the repository's defaults",
        )
        self.assertEqual(len(logged), 1)
        self.assertFalse(os.path.exists(logged[0]))
        self.assertFalse(self.read.exists())

    def test_no_temporary_folder_to_try_the_token_in_is_said_in_its_own_words(self):
        """The token is tried in a temporary folder of its own first: when this user may make none there (a container
        whose /tmp and /var/tmp only root may write), the outcome says so, not that gh did not run; gh is not asked,
        and nothing changes."""

        def unasked(*args):
            self.fail(f"gh asked: {args}")

        none = FileNotFoundError(
            2, "No usable temporary directory found in ['/tmp', '/var/tmp']"
        )
        out = io.StringIO()
        with mock.patch.object(ghlogins.tempfile, "mkdtemp", side_effect=none):
            rc = ghlogins.set_read(
                self.cfg, self.TOKEN, out=out, login=unasked, probe=unasked
            )
        self.assertEqual(
            (rc, out.getvalue()),
            (
                1,
                "ERROR: github_read: no temporary folder to try the token in ([Errno 2] No usable temporary directory "
                "found in ['/tmp', '/var/tmp']); nothing changed\n",
            ),
        )
        self.assertFalse(self.read.exists())

    def test_what_is_refused_or_not_known_changes_nothing(self):
        for probed, took, code, said in (
            (
                {},
                ("error validating token: HTTP 401: Bad credentials",),
                4,
                "REFUSED: github_read: GitHub did not take the token (error validating token: HTTP 401: Bad "
                "credentials); nothing changed\n",
            ),
            (
                {},
                (f"bad token {self.TOKEN}",),  # gh's words never carry the token out
                4,
                "REFUSED: github_read: GitHub did not take the token (bad token (the token)); nothing changed\n",
            ),
            (
                {
                    "login": None,
                    "can_write": False,
                    "why": "not logged in, or the token is expired or revoked",
                },
                ("",),
                4,
                "REFUSED: github_read: GitHub does not know the token (not logged in, or the token is expired or "
                "revoked); nothing changed\n",
            ),
            (
                {"login": "reader", "http": 502, "can_write": None, "why": "HTTP 502"},
                ("",),
                1,
                "ERROR: github_read: whether reader can write to acme/widgets is not known (HTTP 502); nothing "
                "changed\n",
            ),
            (
                {
                    "login": None,
                    "http": None,
                    "can_write": None,
                    "why": "gh did not run: Command 'gh' timed out after 60 seconds",
                },
                ("",),
                1,
                "ERROR: github_read: whether the token can write to acme/widgets is not known (gh did not run: "
                "Command 'gh' timed out after 60 seconds); nothing changed\n",
            ),
        ):
            with self.subTest(said=said):
                rc, out, logged = self.run_(probed, took)
                self.assertEqual((rc, out), (code, said))
                self.assertEqual(len(logged), 1)
                self.assertFalse(self.read.exists())

        def unasked(*args):
            self.fail(f"gh asked: {args}")

        for cfg, token, said in (
            (self.cfg, "\n", "no token on stdin; nothing changed"),
            (
                dict(self.cfg, repo=""),
                self.TOKEN,
                "the configuration names no repository, where the probe asks whether the token can write; "
                "nothing changed",
            ),
        ):
            with self.subTest(said=said):
                out = io.StringIO()
                rc = ghlogins.set_read(
                    cfg, token, out=out, login=unasked, probe=unasked
                )
                self.assertEqual(
                    (rc, out.getvalue()), (4, f"REFUSED: github_read: {said}\n")
                )


class TestSites(Base):
    def test_a_site_the_login_does_not_reach(self):
        listing = json.dumps(
            [{"name": "site-a", "url": "https://site-a.example.invalid"}]
        )
        cfg = dict(self.cfg, atlassian={"site": "site-b"})
        with mock.patch.object(
            atlassian.shutil, "which", return_value="/bin/atlassian"
        ):
            wrong = atlassian.wrong_sites(cfg, run=lambda cmd: (0, listing))
        self.assertEqual(
            wrong,
            [
                f"{atlassian.site_key()} site-b is not a site your login reaches (site-a): fix it in your deploy "
                "settings file, or log in to that site"
            ],
        )
        cfg["atlassian"]["site"] = "site-a"
        with mock.patch.object(
            atlassian.shutil, "which", return_value="/bin/atlassian"
        ):
            self.assertEqual(
                atlassian.wrong_sites(cfg, run=lambda cmd: (0, listing)), []
            )

    def test_jira_s_address_names_its_site_as_the_cli_does(self):
        """The deploy settings file gives JIRA's site by its address (jira_url), and the CLI names a site by its name,
        the address's first label (atlassian.site_name): the status asks that site whether it answers, and the
        login's sites are matched by it."""
        site = "site-a"
        listing = json.dumps([{"name": site, "url": f"https://{site}.example.invalid"}])

        def run(cmd):
            if cmd[1:3] == ["site", "list"]:
                return 0, listing
            if (
                cmd[1] == "--site"
            ):  # `site active`: the CLI answers for a site by its name alone
                return (0, "ok") if cmd[2] == site else (1, "no such site")
            return 0, "Logged in as op"

        for value in (f"https://{site}.atlassian.net/", f"{site}.atlassian.net", site):
            with self.subTest(value=value):
                cfg = dict(self.cfg, atlassian={"site": value})
                self.assertEqual(atlassian.jira_site(cfg), site)
                with mock.patch.object(
                    atlassian.shutil, "which", return_value="/bin/atlassian"
                ):
                    self.assertTrue(atlassian.status(cfg, run=run)["site_ok"])
                    self.assertEqual(atlassian.wrong_sites(cfg, run=run), [])


if __name__ == "__main__":
    unittest.main()
