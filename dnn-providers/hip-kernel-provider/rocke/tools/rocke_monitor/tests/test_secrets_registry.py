# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The secrets registry (rkmon_lib/secrets_registry.py) against its contract: the secrets and logins it lists, the
binding keys a configuration adds, the GitHub logins' permissions, the command that sets each one, the check of a value
that never shows any part of it, a value made here, where each one is kept, what the features that are on need and in
which words, the login walk's entries, the file names the ignore files take and that may hold a secret, and the modules
an import of it loads. And what reads it: every name the docs and the templates give `rkmon secret` and `rkmon login`,
and the deploy schema's rule for a secret's name.
"""

import fnmatch
import io
import json
import os
import pathlib
import random
import re
import string
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
from rkmon_lib import (
    bindings,
    checks,
    config,
    features,
    file_types,
    gh,
    logins,
    paths,
    private_files,
    program,
    state_store,
)  # noqa: E402
from rkmon_lib import secrets_registry as sr  # noqa: E402

FIXED = [
    "github_read",
    "github_write",
    "atlassian",
    "claude",
    "admin_access_token",
    "teams_webhook",
    "web_host_ssh_key",
    "gpu_ssh_key",
]
OFF = {
    "review_automation": {"enabled": False},
    "merge_queue": {"enabled": False},
    "alerts": {"enabled": False},
    "dashboards": {"admin": {"enabled": False}, "team": {"enabled": False}},
    "ticket_creation": {"enabled": False},
}
# Ticket creation with each type of ticket on (each one is off until it is set).
TICKETS = {
    "enabled": True,
    "triggers": {"review_follow_up": True, "out_of_scope_issue": True},
}
BEGIN, END = (
    "-----BEGIN" + " OPENSSH PRIVATE KEY-----",
    "-----END" + " OPENSSH PRIVATE KEY-----",
)
# (features turned on, the rest of the configuration): one of each that changes what is needed
CONFIGURATIONS = (
    ({}, {}),
    ({"review_automation": {"enabled": True}}, {}),
    ({"review_automation": {"enabled": True, "mode": "live"}}, {}),
    ({"merge_queue": {"enabled": True}}, {}),
    ({"merge_queue": {"enabled": True, "mode": "live"}}, {}),
    ({"alerts": {"enabled": True, "develop_breaks": True, "needs_you": True}}, {}),
    (
        {
            "alerts": {
                "enabled": True,
                "mode": "live",
                "develop_breaks": True,
                "needs_you": True,
            }
        },
        {},
    ),
    (
        {"dashboards": {"team": {"enabled": True}}},
        {"team_dashboard": {"mode": "publish"}},
    ),
    (
        {"dashboards": {"team": {"enabled": True, "mode": "live"}}},
        {"team_dashboard": {"mode": "serve"}},
    ),
    (
        {"dashboards": {"team": {"enabled": True, "mode": "live"}}},
        {"team_dashboard": {"mode": "publish"}},
    ),
    ({"ticket_creation": TICKETS}, {}),
    ({"ticket_creation": dict(TICKETS, mode="live")}, {}),
)


def made_up(seed: int, n: int = 40) -> str:
    """Invented secret material: letters and digits no message holds."""
    return "".join(
        random.Random(seed).choices(string.ascii_letters + string.digits, k=n)
    )


def by_name(cfg: dict | None = None) -> dict:
    return {e.name: e for e in sr.entries(cfg or {})}


def a_value(shape: str, seed: int) -> str:
    """An invented value of SHAPE, one its check passes (sr.check_value)."""
    t = made_up(seed)
    return {
        sr.LOGIN: t,
        sr.HTTPS_URL: f"https://hooks.example.invalid/{t}",
        sr.SSH_PRIVATE_KEY: f"{BEGIN}\n{t}\n{END}\n",
        sr.ONE_LINE: t,
    }[shape]


class SecretTest(unittest.TestCase):
    def entry(self, **kw):
        args = dict(
            name="a_secret",
            what="a secret",
            shape=sr.ONE_LINE,
            made_by="you",
            keeper=sr.MONITOR,
        )
        return sr.Secret(**{**args, **kw})

    def test_a_bad_entry_is_refused(self):
        for kw in (
            {"shape": "a word"},
            {"keeper": "nobody"},
            {"name": "A-Secret"},
            {"name": "a-secret"},  # a secret's name has no -
            {"name": "-login", "shape": sr.LOGIN},
            {"what": " "},
            {"made_by": ""},
            {"needed_when": ("Review posting",)},
            {"needed_when": ("auto-merge-guard=",)},
            {"permissions": (("Issues", "read"),)},  # only a login has permissions
            {"shape": sr.LOGIN, "name": "gh", "permissions": (("Issues", "admin"),)},
            {"shape": sr.LOGIN, "name": "gh", "permissions": (("Issues",),)},
            {"key": "Not a key"},
            {"key": "a..b"},
            {"variable": "lower_case"},
            {"place": "a/b"},
            {"place": ".."},
            {"files": ("hosts.yml",)},  # only a login has files
            {"shape": sr.LOGIN, "name": "gh", "files": ("gh/hosts.yml",)},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.entry(**kw)

    def test_a_login_name_may_hold_a_hyphen_and_lists_become_tuples(self):
        e = self.entry(
            name="gh-read",
            shape=sr.LOGIN,
            needed_when=["always", "auto-merge-guard=act"],
            permissions=[["Issues", "read"]],
            files=["hosts.yml"],
        )
        self.assertEqual(e.needed_when, ("always", "auto-merge-guard=act"))
        self.assertEqual(e.permissions, (("Issues", "read"),))
        self.assertEqual(e.files, ("hosts.yml",))
        hash(e)


class EntriesTest(unittest.TestCase):
    def test_the_fixed_entries_are_today_s_names(self):
        e = by_name()
        self.assertEqual(list(e), FIXED)
        self.assertEqual(
            {n for n, x in e.items() if x.shape == sr.LOGIN},
            {"github_write", "github_read", "atlassian", "claude"},
        )  # what `rkmon login` takes
        self.assertEqual(
            {n for n, x in e.items() if x.shape != sr.LOGIN},
            {
                "admin_access_token",
                "teams_webhook",
                "web_host_ssh_key",
                "gpu_ssh_key",
            },
        )  # what `rkmon secret` takes, with the key a model binding names
        self.assertEqual(
            [n for n, x in e.items() if sr.held_files(x) is None],
            [
                "claude",
                "admin_access_token",
                "teams_webhook",
                "web_host_ssh_key",
                "gpu_ssh_key",
            ],
        )  # what `rk secret-set` takes: each one kept in a file, the claude login's key among them

    def test_each_one_s_key_place_and_files(self):
        e = by_name()
        keys = {n: x.key for n, x in e.items() if x.key}
        self.assertEqual(
            keys,
            {
                "github_read": "gh.read_config_dir",
                "github_write": "gh.write_config_dir",
                "admin_access_token": "admin_dashboard.access_token_file",
                "teams_webhook": "teams.webhook_file",
                "web_host_ssh_key": "publish.html.key",
                "gpu_ssh_key": "gpu.ssh_key",
            },
        )
        defaults = config.load_defaults()
        for name, key in keys.items():
            with self.subTest(key=key):
                self.assertIsNotNone(
                    config.lookup(defaults, key), name
                )  # a key the package has
        self.assertEqual(
            {n: x.place for n, x in e.items() if x.place},
            {"github_read": "gh-read", "github_write": "gh", "claude": "claude_key"},
        )  # the GitHub logins' folders keep their older names, the claude login's file has a name no file
        # of the package has; every other is named after it
        self.assertEqual(
            {n: x.files for n, x in e.items() if x.files},
            {
                "github_read": ("hosts.yml",),
                "github_write": ("hosts.yml",),
                "atlassian": ("credentials",),
            },
        )

    def test_no_secret_holds_headers_nothing_sends(self):
        """The local worker's extra headers and the JIRA MCP server's are no secrets: no binding sends headers of its
        own, and the Claude session has no JIRA MCP server, an unused feature. Their files are
        an older package's, which the ignore lists keep out of git and the image; no shape is left for headers. The
        Claude session's login is its key alone: where the key goes and how it is sent are integrations.claude_code's
        settings.
        """
        for name in ("worker_headers", "jira_mcp_headers"):
            with self.subTest(name=name):
                self.assertIsNone(sr.entry({}, name))
                self.assertIn(name, sr.OLDER_SECRET_FILES)
        self.assertEqual(
            sr.SHAPES,
            (sr.LOGIN, sr.HTTPS_URL, sr.SSH_PRIVATE_KEY, sr.ONE_LINE),
        )

    def test_shapes_keepers_and_who_makes_each(self):
        e = by_name()
        shapes = {n: x.shape for n, x in e.items()}
        self.assertEqual(
            shapes,
            {
                "github_read": sr.LOGIN,
                "github_write": sr.LOGIN,
                "atlassian": sr.LOGIN,
                "claude": sr.LOGIN,
                "admin_access_token": sr.ONE_LINE,
                "teams_webhook": sr.HTTPS_URL,
                "web_host_ssh_key": sr.SSH_PRIVATE_KEY,
                "gpu_ssh_key": sr.SSH_PRIVATE_KEY,
            },
        )
        self.assertEqual(
            [n for n, x in e.items() if x.keeper == sr.WRITER],
            ["github_write", "atlassian", "teams_webhook", "web_host_ssh_key"],
        )  # the outward writes' secrets are the writer's; everything else the monitor's
        self.assertEqual(
            [n for n, x in e.items() if x.made_by == sr.GENERATED],
            ["admin_access_token"],
        )

    def test_the_github_logins_permissions(self):
        """What each login's calls need (gh.CALLS), in name order: the write login reads only what the functions that
        write read (PRs and the develop issues), so it needs neither Actions nor Commit statuses nor Members.
        """
        e = by_name()
        self.assertEqual(
            e["github_read"].permissions,
            (
                ("Actions", "read"),
                ("Commit statuses", "read"),
                ("Contents", "read"),
                ("Issues", "read"),
                ("Members", "read"),
                ("Metadata", "read"),
                ("Pull requests", "read"),
            ),
        )
        self.assertEqual(
            e["github_write"].permissions,
            (
                ("Contents", "write"),
                ("Issues", "write"),
                ("Metadata", "read"),
                ("Pull requests", "write"),
            ),
        )
        self.assertEqual(
            [n for n, x in e.items() if x.permissions], ["github_read", "github_write"]
        )

    def test_one_entry_per_secret_a_binding_names(self):
        cfg = {
            "worker": {"api_key_secret": sr.FIRST_PASS_KEY},
            "model_bindings": {
                "b2": {"api_key_secret": "shared_api_key"},
                "b1": {"api_key_secret": " shared_api_key "},
                "b3": {"api_key_secret": "Not-A-Name"},
                "b4": {"api_key_secret": ""},
                "b5": {},
                "b6": "not a mapping",
                "b7": {"api_key_secret": "b7_api_key"},
                "b8": {"api_key_secret": "gpu_ssh_key"},
            },
            # a pass on b2 and on b8 (the first review pass is the local worker's: no reviewer is named)
            "models": {"ci_triage": "b2", "quality_draft": "b8"},
        }
        e = by_name(cfg)
        self.assertEqual(
            list(e),
            FIXED + ["b7_api_key", "first_review_pass_api_key", "shared_api_key"],
        )
        self.assertEqual(
            (e[sr.FIRST_PASS_KEY].what, e[sr.FIRST_PASS_KEY].needed_when),
            (
                "the API key model_bindings.worker.api_key_secret_name names",
                ("review-drafts",),
            ),
        )  # the first review pass's key, which setup names
        shared = e["shared_api_key"]
        self.assertEqual(
            shared.what,
            "the API key model_bindings.b1.api_key_secret_name and model_bindings.b2.api_key_secret_name name",
        )
        self.assertEqual(
            (shared.shape, shared.keeper, shared.needed_when),
            (sr.ONE_LINE, sr.MONITOR, ("review-drafts",)),
        )
        self.assertEqual(
            (e["b7_api_key"].what, e["b7_api_key"].needed_when),
            ("the API key model_bindings.b7.api_key_secret_name names", ()),
        )  # no pass runs on b7: nothing needs its key
        self.assertEqual(
            e["gpu_ssh_key"].needed_when, ("remote-gpu-tests", "review-drafts")
        )  # the fixed entry, with the binding's need
        self.assertEqual(list(by_name({"model_bindings": ["a list"]})), FIXED)

    def test_an_entry_by_its_name(self):
        cfg = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}}
        self.assertEqual(sr.entry(cfg, "github_read"), by_name()["github_read"])
        self.assertEqual(sr.entry(cfg, "b_api_key"), by_name(cfg)["b_api_key"])
        self.assertIsNone(sr.entry({}, "b_api_key"))
        self.assertIsNone(sr.entry(cfg, "no_such_secret"))

    def test_each_condition_names_a_capability_and_a_value_it_can_have(self):
        cfg = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}}
        for e in sr.entries(cfg):
            for c in e.needed_when:
                if c == sr.ALWAYS:
                    continue
                cap, _, value = c.partition("=")
                with self.subTest(entry=e.name, condition=c):
                    self.assertIn(cap, features.CAPABILITIES)
                    if value:
                        self.assertIn(value, features.CAPABILITIES[cap].values)

    def test_a_binding_s_key_must_be_a_secret_s_name(self):
        bad = {
            "model_bindings": {
                "b": {
                    "provider": "openai-compatible",
                    "url": "https://models.example.invalid",
                    "model": "m",
                    "api_key_secret": "Not-A-Name",
                }
            }
        }
        self.assertIn(
            "model_bindings.b.api_key_secret must be a secret's name (lowercase letters, digits and _)",
            bindings.problems(bad),
        )
        bad["model_bindings"]["b"]["api_key_secret"] = "b_api_key"
        self.assertFalse([p for p in bindings.problems(bad) if "api_key_secret" in p])
        # The rule is the registry's own (secrets_registry.NAME), not a copy: another rule there moves the check.
        with mock.patch.object(sr, "NAME", re.compile(r"[A-Za-z-]+")):
            self.assertTrue(
                [p for p in bindings.problems(bad) if "api_key_secret" in p]
            )
            bad["model_bindings"]["b"]["api_key_secret"] = "Not-A-Name"
            self.assertFalse(
                [p for p in bindings.problems(bad) if "api_key_secret" in p]
            )


class CheckValueTest(unittest.TestCase):
    TOK, TOK2 = made_up(7), made_up(11)

    def entry(self, shape: str) -> sr.Secret:
        return sr.Secret(
            "gh" if shape == sr.LOGIN else "a_secret",
            "a secret",
            shape,
            "you",
            sr.MONITOR,
        )

    def good(self) -> dict:
        t, t2 = self.TOK, self.TOK2
        return {
            sr.LOGIN: [t],
            sr.HTTPS_URL: [
                f"https://hooks.example.invalid/{t}",
                f"  https://hooks.example.invalid/{t}\n",
            ],
            sr.SSH_PRIVATE_KEY: [f"{BEGIN}\n{t}\n{t2}\n{END}\n"],
            sr.ONE_LINE: [t, f"{t} {t2}"],
        }

    def bad(self) -> dict:
        t, t2 = self.TOK, self.TOK2
        return {
            sr.LOGIN: [f"{t} {t2}", f"{t}\n{t2}"],
            sr.HTTPS_URL: [
                f"http://hooks.example.invalid/{t}",
                f"https://hooks.example.invalid/{t}\n{t2}",
                f"https:///{t}",
                f"https://hooks.example.invalid/{t} {t2}",
                f"https://[{t}/x",
                t,
            ],
            sr.SSH_PRIVATE_KEY: [
                t,
                f"{BEGIN}\n{t}\n",
                f"{BEGIN}\n{END}",
                f"ssh-ed25519 {t} {t2}",
            ],
            sr.ONE_LINE: [f"{t}\n{t2}"],
        }

    def test_a_value_of_the_shape_passes(self):
        for shape, values in self.good().items():
            for v in values:
                with self.subTest(shape=shape, value=v[:12]):
                    row = sr.check_value(self.entry(shape), v)
                    self.assertEqual(row.status, checks.Status.PASS)
                    self.assertIn(sr.SHAPE_WORDS[shape], row.found)

    def test_a_value_of_another_shape_fails_with_the_command_that_sets_it(self):
        for shape, values in self.bad().items():
            command = "login" if shape == sr.LOGIN else "secret"
            for v in values + [" \n "]:
                with self.subTest(shape=shape, value=v[:12]):
                    e = self.entry(shape)
                    row = sr.check_value(e, v)
                    self.assertEqual(row.status, checks.Status.FAIL)
                    self.assertEqual(
                        row.fix,
                        checks.Steps(
                            (
                                checks.Command(("rkmon", command, e.name)),
                                f", and give {sr.SHAPE_WORDS[shape]}",
                            )
                        ),
                    )
        self.assertEqual(
            str(sr.check_value(self.entry(sr.HTTPS_URL), "x").fix),
            "rkmon secret a_secret, and give an https:// address on one line",
        )
        made_here = by_name()["admin_access_token"]
        self.assertEqual(
            sr.check_value(made_here, "a\nb").fix,
            checks.Command(("rkmon", "secret", "admin_access_token", "--generate")),
        )
        self.assertEqual(
            sr.check_value(self.entry(sr.ONE_LINE), "\t\n").found, "the value is empty"
        )

    def test_no_part_of_the_value_is_ever_shown(self):
        windows = {
            s[i : i + 6] for s in (self.TOK, self.TOK2) for i in range(len(s) - 5)
        }
        for table in (self.good(), self.bad()):
            for shape, values in table.items():
                for v in values:
                    row = sr.check_value(self.entry(shape), v)
                    shown = (
                        checks.render([row]) + f"\n{row.name}\n{row.found}\n{row.fix}"
                    )
                    with self.subTest(shape=shape, value=v[:12]):
                        self.assertEqual([w for w in windows if w in shown], [])
                        self.assertNotIn(str(len(v.strip())), shown)

    def test_a_value_that_is_not_text_is_refused(self):
        with self.assertRaises(TypeError):
            sr.check_value(self.entry(sr.ONE_LINE), self.TOK.encode())

    def test_the_claude_login_is_its_key_alone(self):
        """The Claude session's login is its key alone, a token on one line: an older package's secret,
        the session's settings with the key among them, is refused, and the fix is the command that sets the key.
        """
        e = by_name()["claude"]
        self.assertEqual(sr.check_value(e, f" {self.TOK}\n").status, checks.Status.PASS)
        for v in (
            json.dumps({"ANTHROPIC_API_KEY": self.TOK}),
            f"{self.TOK} {self.TOK2}",
            f"{self.TOK}\n{self.TOK2}",
        ):
            with self.subTest(value=v[:12]):
                row = sr.check_value(e, v)
                self.assertEqual(row.status, checks.Status.FAIL)
                self.assertEqual(
                    row.fix,
                    checks.Steps(
                        (
                            checks.Command(("rkmon", "login", "claude")),
                            f", and give {sr.SHAPE_WORDS[sr.LOGIN]}",
                        )
                    ),
                )


class SetCommandTest(unittest.TestCase):
    def test_the_command_that_sets_each_one(self):
        cfg = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}}
        self.assertEqual(
            {e.name: " ".join(sr.set_command(e)) for e in sr.entries(cfg)},
            {
                "github_read": "login github_read",
                "github_write": "login github_write",
                "atlassian": "login atlassian",
                "claude": "login claude",
                "admin_access_token": "secret admin_access_token --generate",
                "teams_webhook": "secret teams_webhook",
                "web_host_ssh_key": "secret web_host_ssh_key",
                "gpu_ssh_key": "secret gpu_ssh_key",
                "b_api_key": "secret b_api_key",
            },
        )

    def test_the_entry_and_its_command_in_words(self):
        e = by_name()
        for name, words in (
            ("teams_webhook", "the secret teams_webhook: rkmon secret teams_webhook"),
            ("github_write", "the login github_write: rkmon login github_write"),
            ("claude", "the login claude: rkmon login claude"),
            (
                "admin_access_token",
                "the secret admin_access_token: rkmon secret admin_access_token --generate",
            ),
        ):
            with self.subTest(name=name):
                self.assertEqual(sr.how_to_set(e[name]), words)

    def test_the_fix_that_sets_each_one(self):
        """set_fix: the one fix that sets an entry, the one its failed check gives, and the admin dashboard's
        refusal."""
        e = by_name()
        for name, words in (
            (
                "teams_webhook",
                f"rkmon secret teams_webhook, and give {sr.SHAPE_WORDS[e['teams_webhook'].shape]}",
            ),
            (
                "github_write",
                f"rkmon login github_write, and give {sr.SHAPE_WORDS[sr.LOGIN]}",
            ),
            ("claude", f"rkmon login claude, and give {sr.SHAPE_WORDS[sr.LOGIN]}"),
            ("admin_access_token", "rkmon secret admin_access_token --generate"),
        ):
            with self.subTest(name=name):
                self.assertEqual(str(sr.set_fix(e[name])), words)
        for entry in e.values():
            with self.subTest(entry=entry.name):
                self.assertEqual(sr.check_value(entry, "").fix, sr.set_fix(entry))
        from rkmon_lib import webserve

        with tempfile.TemporaryDirectory() as d:
            none = {"admin_dashboard": {"access_token_file": f"{d}/admin_access_token"}}
            self.assertEqual(
                str(webserve.access_refusal(none).fix),
                str(sr.set_fix(e[webserve.ACCESS_SECRET])),
            )

    def test_the_command_is_named_as_the_operator_types_it(self):
        """In the container, the name of the wrapper that runs the command there (doctor.wrapper_name); on the
        host, the name rkmon was run by."""
        e = by_name()
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            self.assertEqual(
                str(sr.set_fix(e["claude"])),
                f"./rkmon login claude, and give {sr.SHAPE_WORDS[sr.LOGIN]}",
            )
            self.assertEqual(
                sr.how_to_set(e["github_write"]),
                "the login github_write: ./rkmon login github_write",
            )
        with mock.patch.dict(os.environ), mock.patch.object(
            program, "name", return_value="../docker/rkmon"
        ):
            os.environ.pop("RKMON_WRAPPER", None)
            self.assertEqual(
                str(sr.set_fix(e["admin_access_token"])),
                "../docker/rkmon secret admin_access_token --generate",
            )

    def test_claude_env_s_fix_and_check_16_s_both_ways(self):
        """claude_env_fix: the file's edit alone, as the translation's stops show it. claude_env_or_login_fix: that
        edit before the first start, or the claude login's command, as check 16's rows show it, which cannot tell
        whether root's one-time move has dealt with the file."""
        from rkmon_lib import claude_env_shape as ces

        either = ", before the first start; or set the key with rkmon login claude"
        for parts, file in (
            (
                (ces.Edit("ANTHROPIC_BASE_URL", "the gateway's address"),),
                "set ANTHROPIC_BASE_URL in claude_env to the gateway's address",
            ),
            (
                (
                    ces.Edit("ANTHROPIC_API_KEY"),
                    ", or",
                    ces.Edit("ANTHROPIC_AUTH_TOKEN"),
                    ", whichever the gateway does not take",
                ),
                "delete ANTHROPIC_API_KEY from claude_env, or delete ANTHROPIC_AUTH_TOKEN from claude_env, whichever "
                "the gateway does not take",
            ),
        ):
            with self.subTest(file=file):
                self.assertEqual(str(sr.claude_env_fix(parts)), file)
                self.assertEqual(sr.claude_env_fix(parts).commands, ())
                both = sr.claude_env_or_login_fix(parts)
                self.assertEqual(str(both), file + either)
                self.assertEqual(both.commands, ("rkmon login claude",))

    def test_a_value_made_here(self):
        made_here = by_name()["admin_access_token"]
        first, second = sr.generate(made_here), sr.generate(made_here)
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"\A[A-Za-z0-9_-]{32}\Z")  # 24 random bytes, URL-safe
        self.assertEqual(sr.check_value(made_here, first).status, checks.Status.PASS)
        for e in sr.entries({}):
            if e.made_by != sr.GENERATED:
                with self.subTest(entry=e.name), self.assertRaises(ValueError):
                    sr.generate(e)


class PathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.secrets = self.d / "secrets"
        self.raw = {
            "paths": {
                "working_dir": str(self.d / "work"),
                "secrets_dir": str(self.secrets),
            },
            "model_bindings": {"b": {"api_key_secret": "b_api_key"}},
        }
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "RKMON_GH_READ_DIR",
                gh.BROKERED,
                "RKMON_CONTAINER",
            )
        }  # a host's: the container's own case sets its variables
        p = mock.patch.dict(os.environ, env, clear=True)
        p.start()
        self.addCleanup(p.stop)
        self.cfg = config.normalize(self.raw)  # the places the load derives

    def where(self, cfg: dict) -> dict:
        return {e.name: sr.path(cfg, e) for e in sr.entries(cfg)}

    def test_a_key_s_place_is_the_one_the_load_derives_and_no_other(self):
        """The place of an entry with a config key is the one the key holds, which the load derives
        (rkmon_lib/derived.py): a blank key is no place, whatever secrets folder is known, for path() keeps no rule of
        its own. An entry with no key is in its home."""
        keyed = [e for e in sr.entries(self.raw) if e.key]
        self.assertTrue(keyed)
        for e in keyed:
            with self.subTest(entry=e.name):
                with self.assertRaises(paths.PathsError) as c:
                    sr.path(self.raw, e)
                self.assertIn(e.name, str(c.exception))
                # the place normalize gives it, as the load does
                self.assertEqual(sr.path(self.cfg, e), config.lookup(self.cfg, e.key))
        for e in sr.entries(self.raw):
            if not e.key:
                with self.subTest(entry=e.name):
                    self.assertEqual(sr.path(self.raw, e), sr.home(self.raw, e))

    def test_each_one_s_place(self):
        w = self.where(self.cfg)
        s = str(self.secrets)
        self.assertEqual(
            w,
            {
                "github_read": f"{s}/gh-read",
                "github_write": f"{s}/gh",
                "atlassian": f"{s}/atlassian",
                **{n: f"{s}/{n}" for n in FIXED[3:]},
                "claude": f"{s}/claude_key",
                "b_api_key": f"{s}/b_api_key",
            },
        )  # on a host the one user keeps the writer's secrets too, in <SECRETS_DIR>
        self.assertEqual(
            {e.name: sr.home(self.cfg, e) for e in sr.entries(self.cfg)}, w
        )  # nothing names another place: each one's home

    def test_a_place_of_its_own(self):
        e = sr.Secret(
            name="a_secret",
            what="a secret",
            shape=sr.ONE_LINE,
            made_by="you",
            keeper=sr.MONITOR,
            place="a-place",
        )
        self.assertEqual(sr.home(self.cfg, e), str(self.secrets / "a-place"))
        self.assertEqual(sr.path(self.cfg, e), str(self.secrets / "a-place"))

    def test_the_writer_s_secrets_are_in_its_own_volume_in_the_container(self):
        """In the container a secret the writer keeps is in the writer's volume, which no other user reads, and a
        config key that names another file for it is not read (the monitor's processes can edit the configuration);
        the monitor's secrets stay in <SECRETS_DIR>."""
        from rkmon_lib import container_layout

        volume = self.d / "secrets-write"
        cfg = dict(
            self.cfg,
            teams={"webhook_file": str(self.d / "elsewhere" / "hook")},
            publish={"html": {"key": str(self.d / "elsewhere" / "key")}},
            admin_dashboard={"access_token_file": str(self.d / "elsewhere" / "token")},
        )
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"}), mock.patch.object(
            container_layout, "WRITE_SECRETS_DIR", str(volume)
        ):
            w = self.where(cfg)
            homes = {e.name: sr.home(cfg, e) for e in sr.entries(cfg)}
        for name, place in (
            ("github_write", "gh"),
            ("atlassian", "atlassian"),
            ("teams_webhook", "teams_webhook"),
            ("web_host_ssh_key", "web_host_ssh_key"),
        ):
            with self.subTest(name=name):
                self.assertEqual(w[name], str(volume / place))
                self.assertEqual(homes[name], w[name])
        self.assertEqual(
            (w["github_read"], w["admin_access_token"], w["claude"]),
            (
                str(self.secrets / "gh-read"),
                str(self.d / "elsewhere" / "token"),
                str(self.secrets / "claude_key"),
            ),
        )
        hook = by_name()["teams_webhook"]
        with mock.patch.object(container_layout, "WRITE_SECRETS_DIR", str(volume)):
            for environ, keeper, home in (
                ({"RKMON_CONTAINER": "1"}, "", volume / "teams_webhook"),
                ({"RKMON_CONTAINER": "1"}, sr.MONITOR, self.secrets / "teams_webhook"),
                ({}, "", self.secrets / "teams_webhook"),
            ):
                with self.subTest(environ=environ, keeper=keeper):
                    self.assertEqual(sr.home(cfg, hook, environ, keeper), str(home))

    def test_a_file_a_config_key_names(self):
        hook = self.d / "elsewhere" / "hook"
        cfg = dict(self.cfg, teams={"webhook_file": str(hook)})
        self.assertEqual(self.where(cfg)["teams_webhook"], str(hook))

    def test_a_binding_s_key_is_where_the_binding_reads_it(self):
        cfg = config.normalize(
            dict(
                self.raw,
                admin_dashboard={"access_token_file": str(self.d / "token")},
                model_bindings={
                    "b": {"api_key_secret": "b_api_key"},
                    "t": {"api_key_secret": "admin_access_token"},
                    "r": {"api_key_secret": "gpu_ssh_key"},
                    "f": {
                        "api_key_secret": "b_api_key",
                        "api_key_file": str(self.d / "elsewhere"),
                    },
                },
                worker={"api_key_secret": "gpu_ssh_key"},
            )
        )  # fixed secrets, each at the place the load derives: a file its config key names is not kept
        # f names a file as well as a secret: it reads the secret where the registry keeps it
        where = self.where(cfg)
        self.assertEqual(
            (
                where["b_api_key"],
                where["admin_access_token"],
                where["gpu_ssh_key"],
            ),
            (
                str(self.secrets / "b_api_key"),
                str(self.secrets / "admin_access_token"),
                str(self.secrets / "gpu_ssh_key"),
            ),
        )
        for b in ("b", "t", "r", "f"):
            with self.subTest(binding=b):
                name = cfg["model_bindings"][b]["api_key_secret"]
                self.assertEqual(
                    bindings.binding_by_name(cfg, b)["api_key_file"], where[name]
                )
                self.assertEqual(sr.binding_key_file(cfg, name), where[name])
        self.assertEqual(config.worker(cfg)["api_key_file"], where["gpu_ssh_key"])
        self.assertEqual(sr.binding_key_file(cfg, "no_such_secret"), "")
        self.assertEqual(
            sr.binding_key_file({"worker": {"api_key_secret": "w_key"}}, "w_key"), ""
        )  # no secrets folder known

    def test_the_worker_s_settings_read_no_file(self):
        """config.worker: the local worker's settings as the configuration holds them, with no file read. Its key file
        is the one the load derives for the secret its binding names (worker.api_key_file): the reader derives none
        itself. The secrets folder's files of an older package (the local worker's key and headers) are never read by
        their names.
        """
        self.secrets.mkdir()
        for name in sr.OLDER_SECRET_FILES:
            (self.secrets / name).write_text("invented\n", encoding="utf-8")
        raw = dict(self.raw, worker={"url": "https://gw.example.com/v1/", "model": "m"})
        w = config.worker(config.normalize(raw))
        self.assertEqual(
            [w[k] for k in ("url", "model", "api_key_file")],
            ["https://gw.example.com/v1", "m", ""],
        )
        raw["worker"] = dict(raw["worker"], api_key_secret=sr.FIRST_PASS_KEY)
        self.assertEqual(config.worker(raw)["api_key_file"], "")
        self.assertEqual(
            config.worker(config.normalize(raw))["api_key_file"],
            str(self.secrets / "first_review_pass_api_key"),
        )

    def test_the_read_login_alone_has_a_variable(self):
        """The suite's variable for the read login's folder is the only one: the write login's folder is its key's, and
        in the container its home (container_layout.WRITE_LOGIN_DIR)."""
        self.assertEqual(
            {e.name: e.variable for e in sr.entries(self.cfg) if e.variable},
            {"github_read": "RKMON_GH_READ_DIR"},
        )

    def test_a_login_s_folder_its_key_or_its_variable_names(self):
        cfg = dict(self.cfg, gh=dict(self.cfg["gh"], read_config_dir="~/reads"))
        self.assertEqual(
            self.where(cfg)["github_read"],
            os.path.join(os.path.expanduser("~"), "reads"),
        )
        reads = str(self.d / "env-reads")
        with mock.patch.dict(os.environ, {"RKMON_GH_READ_DIR": reads}):
            self.assertEqual(self.where(cfg)["github_read"], reads)  # before the key
        self.assertEqual(
            sr.path(cfg, by_name()["github_read"], {"RKMON_GH_READ_DIR": reads}), reads
        )  # the environment given

    def test_the_read_login_s_place_is_its_own_whichever_folder_gh_reads_with(self):
        """A write login made before the split and no read login: gh reads with the write login meanwhile, and in the
        broker with the write folder; the read login's place stays its own folder."""
        old = self.secrets / "gh"
        old.mkdir(parents=True)
        (old / "hosts.yml").write_text("github.com: {}\n", encoding="utf-8")
        own = str(self.secrets / "gh-read")
        self.assertEqual(gh.login_dirs(self.cfg)["read"], str(old))
        self.assertEqual(self.where(self.cfg)["github_read"], own)
        with mock.patch.dict(os.environ, {gh.BROKERED: "1"}):
            self.assertEqual(gh.login_dirs(self.cfg)["read"], str(old))
            self.assertEqual(self.where(self.cfg)["github_read"], own)

    def test_no_secrets_folder(self):
        cfg = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}}
        for e in sr.entries(cfg):
            with self.subTest(entry=e.name), self.assertRaises(paths.PathsError) as c:
                sr.path(cfg, e)
            self.assertIn("paths.secrets_dir", str(c.exception))

    def test_the_edit_a_refused_write_shows(self):
        """A refused write of a secret's file names another secrets folder in the config.json that places it, the one
        place of the secrets config.json gives; by default the monitor's own config.json, found only when a refusal
        names it."""
        from rkmon_lib import private_files

        file = str(self.d / "config.json")
        another = "another folder of yours, outside the package folder, git and cloud-synced folders"
        self.assertEqual(
            sr.retry(file)("/x/hook"),
            checks.Edit("paths.secrets_dir", file, checks.Described(another)),
        )
        self.assertEqual(
            str(sr.retry(file)(private_files.FILE)),
            f"set paths.secrets_dir in {file} to the folder of FILE",
        )
        with mock.patch.object(config, "config_path", side_effect=AssertionError):
            later = sr.retry()  # no config.json is looked for yet
        with mock.patch.object(config, "config_path", return_value=pathlib.Path(file)):
            self.assertEqual(
                str(later("/x/hook")), f"set paths.secrets_dir in {file} to {another}"
            )


class OlderNamesTest(unittest.TestCase):
    """Each entry's older names and places, read here by everything else: the names `rkmon secret`,
    `rkmon login` and `rk secret-set` refuse, naming the entry; the places the one-time move copies from, never the
    entry's own; and the files and folders the ignore lists take."""

    BASE = dict(
        name="a_secret",
        what="a secret",
        shape=sr.ONE_LINE,
        made_by="you",
        keeper=sr.MONITOR,
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.secrets = self.d / "secrets"
        p = mock.patch.dict(os.environ)
        p.start()
        self.addCleanup(p.stop)
        for name in ("RKMON_CONTAINER", "RKMON_GH_READ_DIR"):
            os.environ.pop(name, None)
        # with the places the load derives
        self.cfg = config.normalize({"paths": {"secrets_dir": str(self.secrets)}})

    def test_an_older_name_is_checked_as_it_is_made(self):
        for kw in ({"name": "a/b"}, {"name": ""}, {"name": "x", "keeper": "nobody"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                sr.Older(**kw)
        for kw in (
            {"older": ("an_older_name",)},  # an Older, not a name
            {"older": (sr.Older("an_older_token", file=True),)},  # a secret is a file
            {"shape": sr.LOGIN, "older": (sr.Older("a_token", file=True),)},
        ):  # a login of a key alone is a file
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                sr.Secret(**{**self.BASE, **kw})
        e = sr.Secret(**self.BASE, older=[sr.Older("an_older_name")])
        self.assertEqual(e.older, (sr.Older("an_older_name", sr.MONITOR, True, False),))
        hash(e)

    def test_each_entry_s_older_names(self):
        """(name, keeper, copied, kept in a file): an older copy the one-time move copies, or one it only removes as it
        ends."""
        self.assertEqual(
            {
                e.name: [(o.name, o.keeper, o.copied, o.file) for o in e.older]
                for e in sr.entries({})
                if e.older
            },
            {
                "github_read": [
                    ("gh-read", sr.MONITOR, True, False),
                    ("gh_read_token", sr.MONITOR, False, True),
                    ("gh-read-token", sr.MONITOR, False, True),
                ],
                "github_write": [("gh", sr.MONITOR, True, False)],
                "atlassian": [("atlassian", sr.MONITOR, False, False)],
                "claude": [("claude_env", sr.MONITOR, True, False)],
                "admin_access_token": [("admin_token", sr.MONITOR, False, False)],
                "teams_webhook": [("teams_webhook", sr.MONITOR, True, False)],
                "web_host_ssh_key": [("webhost_ed25519", sr.MONITOR, True, False)],
            },
        )
        self.assertEqual(
            sr.older_names(by_name()["admin_access_token"]), ["admin_token"]
        )

    def test_the_writer_s_older_places(self):
        """A container from before the writer kept the writer's secrets and the atlassian login in the monitor's
        secrets folder, the web host's key under its algorithm's name. On a host the one user keeps
        both keepers' secrets in <SECRETS_DIR>, so there only the older name is another place.
        """
        from rkmon_lib import container_layout

        e, s = by_name(), self.secrets

        def places(environ=None):
            return {
                n: [
                    (o.name, o.copied, p)
                    for o, p in sr.older_places(self.cfg, e[n], environ)
                ]
                for n in ("atlassian", "teams_webhook", "web_host_ssh_key")
            }

        key = [("webhost_ed25519", True, str(s / "webhost_ed25519"))]
        self.assertEqual(
            places(),
            {"atlassian": [], "teams_webhook": [], "web_host_ssh_key": key},
        )
        volume = self.d / "secrets-write"
        with mock.patch.object(container_layout, "WRITE_SECRETS_DIR", str(volume)):
            self.assertEqual(
                places({"RKMON_CONTAINER": "1"}),
                {
                    "atlassian": [("atlassian", False, str(s / "atlassian"))],
                    "teams_webhook": [
                        ("teams_webhook", True, str(s / "teams_webhook"))
                    ],
                    "web_host_ssh_key": key,
                },
            )

    def test_the_claude_login_s_older_copy_is_the_older_secret_s_file(self):
        """An older package kept the Claude session's key in the secret claude_env, beside the session's other
        settings: a file of the monitor's secrets folder, which the one-time move copies the key alone from;
        until then a copy there counts as set (state 7: logins.py's rows).
        """
        e = by_name()["claude"]
        self.assertIsNone(sr.held_files(e))  # a login of a key alone: a file
        self.assertEqual(
            [
                (o.name, o.copied, sr.held_files(e, o), p)
                for o, p in sr.older_places(self.cfg, e)
            ],
            [("claude_env", True, None, str(self.secrets / "claude_env"))],
        )
        self.assertEqual(sr.renamed({}, "claude_env"), e)
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            self.assertEqual(
                sr.older_name_refusal("claude_env", e),
                "claude_env is the older name of the login claude: ./rkmon login claude sets it",
            )

    def test_where_each_older_name_keeps_it_but_its_own_place(self):
        from rkmon_lib import container_layout

        e = sr.Secret(
            **dict(self.BASE, keeper=sr.WRITER, place="its_file"),
            older=(
                sr.Older("its_file", sr.WRITER),  # its own place: none
                sr.Older("its_file"),  # the monitor's: another place in the container
                sr.Older("old_file", copied=False),
                sr.Older("old_file"),  # that place again: once
            ),
        )
        s = self.secrets
        self.assertEqual(
            sr.older_places(self.cfg, e),
            [(sr.Older("old_file", copied=False), str(s / "old_file"))],
        )  # on a host the one user keeps both keepers' secrets in <SECRETS_DIR>
        self.assertEqual(sr.older_names(e), ["its_file", "old_file"])
        volume = self.d / "secrets-write"
        with mock.patch.object(container_layout, "WRITE_SECRETS_DIR", str(volume)):
            self.assertEqual(
                sr.older_places(self.cfg, e, {"RKMON_CONTAINER": "1"}),
                [
                    (sr.Older("its_file"), str(s / "its_file")),
                    (sr.Older("old_file", copied=False), str(s / "old_file")),
                ],
            )
        with self.assertRaises(paths.PathsError):
            sr.older_places({}, e)

    def test_the_files_held_takes(self):
        login = sr.Secret(
            **dict(self.BASE, name="a-login", shape=sr.LOGIN, files=("hosts.yml",)),
            older=(sr.Older("a-login-token", file=True), sr.Older("an-older-login")),
        )
        self.assertEqual(sr.held_files(login), ["hosts.yml"])
        self.assertEqual(sr.held_files(login, login.older[0]), None)
        self.assertEqual(sr.held_files(login, login.older[1]), ["hosts.yml"])
        alone = sr.Secret(**dict(self.BASE, name="a-key", shape=sr.LOGIN))
        self.assertIsNone(sr.held_files(alone))  # a login of a key alone: a file
        self.assertIsNone(sr.held_files(sr.Secret(**self.BASE)))
        self.secrets.mkdir()
        (self.secrets / "a-key").write_text("k\n", encoding="utf-8")
        self.assertTrue(sr.held(str(self.secrets / "a-key"), []))
        self.assertFalse(sr.held(str(self.secrets), []))  # a folder is no file

    def test_an_older_name_is_refused_naming_the_entry(self):
        self.assertEqual(sr.renamed({}, "admin_token"), by_name()["admin_access_token"])
        self.assertEqual(sr.renamed({}, "gh-read-token"), by_name()["github_read"])
        for name in ("admin_access_token", "no_such_secret", ""):
            with self.subTest(name=name):
                self.assertIsNone(sr.renamed({}, name))
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            self.assertEqual(
                sr.older_name_refusal("admin_token", by_name()["admin_access_token"]),
                "admin_token is the older name of the secret admin_access_token: ./rkmon secret "
                "admin_access_token --generate sets it",
            )
            self.assertEqual(
                sr.older_name_refusal("gh_read_token", by_name()["github_read"]),
                "gh_read_token is the older name of the login github_read: ./rkmon login github_read sets it",
            )
        e = sr.Secret(**self.BASE, older=(sr.Older("a_secret"),))
        with mock.patch.object(sr, "_fixed", return_value=(e,)):
            self.assertIsNone(
                sr.renamed({}, "a_secret")
            )  # its own name is no older name
            self.assertEqual(sr.older_names(e), [])

    def test_why_a_name_is_refused_where_a_secret_s_name_is_asked_for(self):
        """name_refusal: nothing for a secret's name, a model binding's key among them as the configuration names it;
        a login's name, with the command that sets it; an older name, with the new one; any other name, shown only
        when it reads as a name."""
        cfg = {"model_bindings": {"g": {"api_key_secret": "g_api_key"}}}
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            for name, why in (
                ("teams_webhook", ""),
                ("g_api_key", ""),
                ("claude", "claude is a login: ./rkmon login claude sets it"),
                (
                    "g_key",
                    "g_key is the older name of the secret g_api_key: ./rkmon secret g_api_key sets it",
                ),
                ("b_api_key", "no secret is named b_api_key"),  # no binding names it
                ("teams-hook", "no secret is named teams-hook"),
                (
                    "Bearer " + "made-up-Zq7",
                    "a name given is no secret's, and is not shown: it does not read as a name, and may be a value",
                ),
            ):
                with self.subTest(name=name):
                    self.assertEqual(sr.name_refusal(cfg, name), why)

    def test_with_no_configuration_known_a_name_a_binding_may_give_is_not_refused(self):
        """name_refusal with no configuration known (None: rkmon secret before the first confirm): the fixed entries
        alone; a name that reads as a secret's which they do not refuse is not refused, as a model binding may name it,
        while a login's name, an older name of a fixed entry and a name that does not read as a secret's are.
        """
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            for name, why in (
                ("teams_webhook", ""),
                ("g_api_key", ""),
                ("g_key", ""),
                ("claude", "claude is a login: ./rkmon login claude sets it"),
                (
                    "admin_token",
                    "admin_token is the older name of the secret admin_access_token: ./rkmon secret "
                    "admin_access_token --generate sets it",
                ),
                ("teams-hook", "no secret is named teams-hook"),
                (
                    "Bearer " + "made-up-Zq7",
                    "a name given is no secret's, and is not shown: it does not read as a name, and may be a value",
                ),
            ):
                with self.subTest(name=name):
                    self.assertEqual(sr.name_refusal(None, name), why)


class BindingKeyNameTest(unittest.TestCase):
    def test_the_name_the_move_copies_a_binding_s_key_to(self):
        """Every binding's key's name ends in _api_key: the local worker's key is the first review
        pass's; a name without that end gets it, in the place of a last _key; any other name stays.
        """
        for given, name in (
            ("worker_api_key", "first_review_pass_api_key"),
            ("other_key", "other_api_key"),
            ("glm", "glm_api_key"),
            ("gw_token", "gw_token_api_key"),
            ("other_api_key", "other_api_key"),
            ("Not-A-Name", "Not-A-Name"),
            ("", ""),
        ):
            with self.subTest(given=given):
                self.assertEqual(sr.binding_key_name(given), name)

    def test_a_binding_s_key_s_older_names(self):
        """The names binding_key_name gives its name from, each copied by the one-time move while a binding names the
        key; never another entry's name or place, nor a fixed entry's older name."""
        cfg = {
            "worker": {"api_key_secret": sr.FIRST_PASS_KEY},
            "model_bindings": {
                # claude_key is the claude login's place, and claude its name: no older name of this key
                "c": {"api_key_secret": "claude_api_key"},
                "g": {"api_key_secret": "glm_api_key"},
                # glm_key is an entry of its own here: never another entry's older name
                "k": {"api_key_secret": "glm_key"},
                "s": {"api_key_secret": "gpu_ssh_api_key"},
                "w": {"api_key_secret": "webhost_ed25519_api_key"},
                "x": {"api_key_secret": "x_api_api_key"},
            },
        }
        e = by_name(cfg)
        self.assertEqual(
            {
                n: [(o.name, o.keeper, o.copied, o.file) for o in e[n].older]
                for n in e
                if n not in FIXED
            },
            {
                "claude_api_key": [],
                "first_review_pass_api_key": [
                    ("worker_api_key", sr.MONITOR, True, False),
                    ("first_review_pass_key", sr.MONITOR, True, False),
                    ("first_review_pass", sr.MONITOR, True, False),
                ],
                "glm_api_key": [("glm", sr.MONITOR, True, False)],
                "glm_key": [],
                "gpu_ssh_api_key": [("gpu_ssh", sr.MONITOR, True, False)],
                "webhost_ed25519_api_key": [
                    ("webhost_ed25519_key", sr.MONITOR, True, False)
                ],
                "x_api_api_key": [("x_api", sr.MONITOR, True, False)],
            },
        )
        self.assertEqual(sr.renamed(cfg, "worker_api_key"), e[sr.FIRST_PASS_KEY])
        self.assertIsNone(sr.renamed(cfg, "glm_key"))  # an entry's own name
        # No binding names the first review pass's key: the older name is an older package's file (OLDER_SECRET_FILES).
        self.assertIsNone(sr.renamed({}, "worker_api_key"))
        self.assertIn("worker_api_key", sr.OLDER_SECRET_FILES)

    def test_a_binding_s_key_under_its_older_name_is_set_until_the_move(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        secrets = pathlib.Path(tmp.name) / "secrets"
        cfg = {
            "paths": {"secrets_dir": str(secrets)},
            "worker": {"api_key_secret": sr.FIRST_PASS_KEY},
        }
        e = sr.entry(cfg, sr.FIRST_PASS_KEY)
        self.assertEqual(
            [(o.name, p) for o, p in sr.older_places(cfg, e)],
            [
                ("worker_api_key", str(secrets / "worker_api_key")),
                ("first_review_pass_key", str(secrets / "first_review_pass_key")),
                ("first_review_pass", str(secrets / "first_review_pass")),
            ],
        )


class NeededTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": "1"}
        )  # the Claude session is the container's
        p.start()
        self.addCleanup(p.stop)

    def config(self, feats: dict, **cfg) -> dict:
        merged = json.loads(json.dumps(OFF))
        for k, v in feats.items():
            merged[k] = {**merged.get(k, {}), **v}
        return {"features": merged, **cfg}

    def needed(self, feats: dict, **cfg) -> list:
        return [e.name for e in sr.needed(self.config(feats, **cfg))]

    def test_what_needs_each_one_in_words(self):
        def title(capability: str) -> str:
            return features.lowered(features.CAPABILITIES[capability].title)

        e = by_name()
        off = self.config({})
        self.assertEqual(sr.needed_by(off, e["github_read"]), [sr.ALWAYS_WORDS])
        self.assertEqual(
            [n for n, x in e.items() if sr.needed_by(off, x)], ["github_read"]
        )
        live = self.config({"review_automation": {"enabled": True, "mode": "live"}})
        self.assertIn(title("review-posting"), sr.needed_by(live, e["github_write"]))
        guard = self.config({"merge_queue": {"enabled": True, "mode": "live"}})
        self.assertIn(
            title("auto-merge-guard") + " (turning auto-merge off)",
            sr.needed_by(guard, e["github_write"]),
        )
        team = self.config(
            {"dashboards": {"team": {"enabled": True, "mode": "live"}}},
            team_dashboard={"mode": "publish"},
        )
        self.assertEqual(
            sr.needed_by(team, e["web_host_ssh_key"]),
            [title("team-dashboard") + " (publish)"],
        )
        alerts = self.config(
            {
                "alerts": {
                    "enabled": True,
                    "mode": "live",
                    "develop_breaks": True,
                    "needs_you": True,
                }
            }
        )
        self.assertEqual(
            sr.needed_by(alerts, e["teams_webhook"]),
            [
                title(c)
                for c in e["teams_webhook"].needed_when
                if features.on(alerts, c)
            ],
        )
        self.assertTrue(sr.needed_by(alerts, e["teams_webhook"]))

    def test_each_condition_in_words(self):
        # What the README's logins table says of each entry, whatever is on.
        for condition, words in (
            (sr.ALWAYS, "the monitor, always"),
            ("review-posting", "review posting"),
            ("team-dashboard=publish", "team dashboard (publish)"),
            ("auto-merge-guard=act", "auto-merge guard (turning auto-merge off)"),
            ("headless-claude", "Claude session"),
        ):
            with self.subTest(condition=condition):
                self.assertEqual(sr.condition_words(condition), words)
        for e in sr.entries({}):
            for c in e.needed_when:
                with self.subTest(entry=e.name, condition=c):
                    self.assertTrue(sr.condition_words(c))
        with self.assertRaises(KeyError):
            sr.condition_words("no-such-capability")

    def test_a_configuration_features_cannot_read_has_nothing_on(self):
        cfg = self.config({"review_automation": {"enabled": True, "mode": "live"}})
        with mock.patch.object(
            features, "configured", side_effect=ValueError("unread")
        ):
            self.assertEqual([e.name for e in sr.needed(cfg)], ["github_read"])
            self.assertEqual(sr.needed_by(cfg, by_name()["github_write"]), [])

    def test_the_words_agree_with_what_is_needed(self):
        mb = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}, "reviewer": "b"}
        for feats, cfg in CONFIGURATIONS:
            full = self.config(feats, **cfg, **mb)
            needed = [e.name for e in sr.needed(full)]
            for e in sr.entries(full):
                with self.subTest(features=feats, cfg=cfg, entry=e.name):
                    self.assertEqual(bool(sr.needed_by(full, e)), e.name in needed)

    def test_the_login_walk_s_entries(self):
        mb = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}}
        self.assertEqual(
            [e.name for e in sr.walk_entries(dict(mb, reviewer="b"))],
            [
                "github_read",
                "github_write",
                "atlassian",
                "claude",
                "admin_access_token",
                "teams_webhook",
                "web_host_ssh_key",
                "gpu_ssh_key",
                "b_api_key",
            ],
        )  # each one something may need: never one that is only used when it is set
        self.assertNotIn(
            "b_api_key", [e.name for e in sr.walk_entries(mb)]
        )  # no pass runs on b: nothing may need its key
        self.assertIn(
            sr.FIRST_PASS_KEY,
            [
                e.name
                for e in sr.walk_entries(
                    {"worker": {"api_key_secret": sr.FIRST_PASS_KEY}}
                )
            ],
        )  # a binding that names it adds its need

    def test_standby_changes_no_need(self):
        """What a live feature needs stays needed while the monitor is in standby: standby passes, and
        its logins are needed again the moment it ends."""
        cfg = self.config(
            {
                "review_automation": {"enabled": True, "mode": "live"},
                "merge_queue": {"enabled": True, "mode": "live"},
            }
        )
        guard = by_name()["github_write"]
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(
            os.environ, {"RKMON_HOME": home}
        ):
            before = (self.needed(cfg["features"]), sr.needed_by(cfg, guard))
            state_store.set(home, "standby.on", True)
            self.assertTrue(features.standby())
            self.assertEqual(features.value(cfg, "auto-merge-guard"), "alert")
            self.assertEqual(
                (self.needed(cfg["features"]), sr.needed_by(cfg, guard)), before
            )
        self.assertIn("github_write", before[0])
        self.assertIn(
            features.lowered(features.CAPABILITIES["auto-merge-guard"].title)
            + " (turning auto-merge off)",
            before[1],
        )

    def test_with_everything_off_only_the_read_login(self):
        self.assertEqual(self.needed({}), ["github_read"])

    def test_the_write_login_waits_for_something_that_writes(self):
        self.assertNotIn(
            "github_write",
            self.needed({"merge_queue": {"enabled": True, "mode": "prototype"}}),
        )  # the guard alerts: nothing goes out
        self.assertIn(
            "github_write",
            self.needed({"merge_queue": {"enabled": True, "mode": "live"}}),
        )
        self.assertIn(
            "github_write",
            self.needed({"review_automation": {"enabled": True, "mode": "live"}}),
        )

    def test_the_web_host_key_only_when_the_team_dashboard_is_published(self):
        on = {"dashboards": {"team": {"enabled": True, "mode": "live"}}}
        self.assertIn(
            "web_host_ssh_key", self.needed(on, team_dashboard={"mode": "publish"})
        )
        self.assertNotIn(
            "web_host_ssh_key", self.needed(on, team_dashboard={"mode": "serve"})
        )
        # Where it goes is the capability's value by the settings (features.configured), which the registry reads and
        # nothing else.
        says = {"team-dashboard": "publish"}
        with mock.patch.object(
            features, "configured", side_effect=lambda cfg, name: says.get(name, False)
        ):
            self.assertEqual(
                self.needed(on, team_dashboard={"mode": "serve"}),
                ["github_read", "web_host_ssh_key"],
            )

    def test_each_feature_s_own(self):
        for feats, cfg, name in (
            (
                {
                    "alerts": {
                        "enabled": True,
                        "mode": "live",
                        "develop_breaks": True,
                        "needs_you": True,
                    }
                },
                {},
                "teams_webhook",
            ),
            ({"dashboards": {"admin": {"enabled": True}}}, {}, "admin_access_token"),
            ({"ticket_creation": dict(TICKETS, mode="live")}, {}, "atlassian"),
            ({"review_automation": {"enabled": True}}, {}, "claude"),
            (
                {"review_automation": {"enabled": True}},
                {"gpu": {"enabled": True}},
                "gpu_ssh_key",
            ),
        ):
            with self.subTest(name=name):
                self.assertIn(name, self.needed(feats, **cfg))
                self.assertNotIn(name, self.needed({}, **cfg))

    def test_in_prototype_no_login_that_sends(self):
        """A feature in prototype sends nothing, so it needs none of the logins and secrets its sends go out with:
        the alerts' webhook, the web host's key and the atlassian login wait for live, as the write
        login does."""
        for feats, cfg, name in (
            (
                {
                    "alerts": {
                        "enabled": True,
                        "develop_breaks": True,
                        "needs_you": True,
                    }
                },
                {},
                "teams_webhook",
            ),
            (
                {"dashboards": {"team": {"enabled": True}}},
                {"team_dashboard": {"mode": "publish"}},
                "web_host_ssh_key",
            ),
            ({"ticket_creation": TICKETS}, {}, "atlassian"),
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, self.needed(feats, **cfg))
                self.assertEqual(
                    sr.needed_by(self.config(feats, **cfg), by_name()[name]), []
                )

    def test_what_each_one_waits_for_names_the_mode(self):
        """While nothing that is on needs it, an entry that a feature in prototype would need once live waits for
        that feature (waits_for), and its words name the mode (waiting_words). Nothing waits while it is
        needed, nor for a feature whose going live would not need it. Standby changes nothing, and the configuration
        given is not changed."""
        alerts = {
            "alerts": {"enabled": True, "develop_breaks": True, "needs_you": True}
        }
        team = {"dashboards": {"team": {"enabled": True}}}
        both = {
            "review_automation": {"enabled": True},
            "merge_queue": {"enabled": True},
        }
        for feats, cfg, name, waits in (
            (alerts, {}, "teams_webhook", ["alerts"]),
            (
                team,
                {"team_dashboard": {"mode": "publish"}},
                "web_host_ssh_key",
                ["dashboards.team"],
            ),
            ({"ticket_creation": TICKETS}, {}, "atlassian", ["ticket_creation"]),
            (both, {}, "github_write", ["review_automation", "merge_queue"]),
            ({"merge_queue": {"enabled": True}}, {}, "github_write", ["merge_queue"]),
            # needed now: it waits for nothing
            (
                dict(both, review_automation={"enabled": True, "mode": "live"}),
                {},
                "github_write",
                [],
            ),
            ({"alerts": dict(alerts["alerts"], mode="live")}, {}, "teams_webhook", []),
            (
                {"review_automation": {"enabled": True}},
                {"gpu": {"enabled": True}},
                "gpu_ssh_key",
                [],
            ),  # remote GPU tests run in prototype too
            # going live would not need it either
            (
                {
                    "alerts": {
                        "enabled": True,
                        "develop_breaks": False,
                        "needs_you": False,
                    }
                },
                {},
                "teams_webhook",
                [],
            ),
            (team, {"team_dashboard": {"mode": "serve"}}, "web_host_ssh_key", []),
            ({}, {}, "github_write", []),
        ):
            full = self.config(feats, **cfg)
            entry = by_name(full)[name]
            with self.subTest(name=name, features=feats, cfg=cfg):
                self.assertEqual(sr.waits_for(full, entry), waits)
                self.assertEqual(
                    sr.waiting_words(full, entry),
                    "; ".join(features.prototype_words(f) for f in waits),
                )
        full = self.config(alerts)
        given = json.loads(json.dumps(full))
        entry = by_name(full)["teams_webhook"]
        self.assertEqual(
            sr.waiting_words(full, entry),
            "features.alerts.mode is prototype: nothing goes out until it is live",
        )
        self.assertEqual(full, given)
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(
            os.environ, {"RKMON_HOME": home}
        ):
            state_store.set(home, "standby.on", True)
            self.assertTrue(features.standby())
            self.assertEqual(sr.waits_for(full, entry), ["alerts"])

    def test_going_live_needs_what_waits_for_it(self):
        """An entry that waits for features is needed once any one of them goes live (features.as_live), and going
        live with any other feature does not need it: the words agree with what is needed.
        """
        mb = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}, "reviewer": "b"}
        for feats, cfg in CONFIGURATIONS:
            full = self.config(feats, **cfg, **mb)
            for e in sr.entries(full):
                waits = sr.waits_for(full, e)
                with self.subTest(features=feats, cfg=cfg, entry=e.name):
                    if waits:
                        self.assertEqual(sr.needed_by(full, e), [])
                    for f in features.SENDS:
                        if sr.needed_by(full, e) or features.live(full, f):
                            continue
                        self.assertEqual(
                            bool(sr.needed_by(features.as_live(full, f), e)),
                            f in waits,
                            f,
                        )

    def test_a_binding_s_key_while_review_automation_is_on(self):
        """A binding's key, while review automation is on and a pass runs on its binding."""
        mb = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}}
        on = {"review_automation": {"enabled": True}}
        for runs in ({"reviewer": "b"}, {"models": {"notice_draft": "b"}}):
            with self.subTest(runs=runs):
                self.assertIn("b_api_key", self.needed(on, **mb, **runs))
                self.assertNotIn("b_api_key", self.needed({}, **mb, **runs))
        self.assertNotIn(
            "b_api_key", self.needed(on, **mb)
        )  # no pass runs on b: nothing needs its key
        self.assertNotIn(
            sr.FIRST_PASS_KEY, self.needed({"review_automation": {"enabled": True}})
        )  # no binding names it: a model that takes no key has none

    def test_it_agrees_with_the_login_walk(self):
        """`rk logins` walks the registry's entries by their names, each one with what needs it exactly while the
        registry says it is needed, and NOT NEEDED otherwise."""
        mb = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}, "reviewer": "b"}
        for feats, cfg in CONFIGURATIONS:
            full = self.config(feats, **cfg, **mb)
            got = [e.name for e in sr.needed(full)]
            with mock.patch.object(logins, "state", return_value=("MISSING", "")):
                rows = logins.rows(full, probe=lambda c: ({}, True))
            self.assertEqual(
                [r["name"] for r in rows], [e.name for e in sr.walk_entries(full)]
            )
            for r in rows:
                with self.subTest(features=feats, cfg=cfg, row=r["name"]):
                    self.assertEqual(bool(r["needed_by"]), r["name"] in got)
                    self.assertEqual(r["state"] == "NOT NEEDED", r["name"] not in got)


class DocumentedNamesTest(unittest.TestCase):
    """Every name the docs, the commands' help, the schemas the package's files are read with (file_types.SCHEMAS)
    and every settings template give `rkmon secret` or `rkmon login` is the registry's: a login's for `rkmon login`; for
    `rkmon secret`, a secret it lists, or a model binding's key by its usual name (*_api_key). `rk secret-set` takes
    each such secret, and each login kept in a file (the claude login's key), with a value of its shape, and refuses
    each login kept in a folder.
    """

    COMMAND = re.compile(r"rkmon (secret|login) ((?:\\?\||[A-Za-z0-9_.<>-])+)")
    # Prose after a command's name: "rkmon login and rkmon secret set them", "rkmon login sets them", "rkmon login
    # installs those", "rkmon login lists the logins they need", "docker/rkmon login walks them".
    PROSE = {"and", "set", "sets", "installs", "lists", "walks"}

    def texts(self) -> dict:
        from rkmon_lib import settings_templates

        files = [
            *ROOT.glob("*.md"),
            *ROOT.glob("docs/**/*.md"),
            *file_types.SCHEMAS.glob("*.json"),
            ROOT / "docker" / "rkmon",
            ROOT / "rkmon.py",
        ]
        out = {
            str(p.relative_to(ROOT)): p.read_text(encoding="utf-8")
            for p in sorted(files)
        }
        for t in settings_templates.templates():
            out[f"the {t.label} template"] = t.text()
        return out

    def documented(self) -> dict:
        """{(command, name): [where]}: each name given `rkmon secret` or `rkmon login`, placeholders aside."""
        found: dict = {}
        for where, text in self.texts().items():
            for m in self.COMMAND.finditer(text):
                for name in re.split(r"\\?\|", m.group(2)):
                    name = name.rstrip(".,;:")
                    if (
                        not name
                        or name.startswith(("<", "-"))
                        or name.isupper()
                        or name in self.PROSE
                    ):
                        continue  # a placeholder (NAME, <name>), an option (--no-input) or prose
                    found.setdefault((m.group(1), name), []).append(where)
        return found

    def test_each_name_is_the_registry_s(self):
        found = self.documented()
        for named in (
            ("login", "github_write"),
            ("login", "github_read"),
            ("login", "atlassian"),
            ("login", "claude"),
            ("secret", "teams_webhook"),
            ("secret", "web_host_ssh_key"),
            ("secret", "admin_access_token"),
        ):
            self.assertIn(named, found)  # the scan reads the docs
        self.assertTrue(
            [w for w in found[("login", "claude")] if w.endswith(" template")]
        )  # and the templates
        logins_ = {e.name for e in sr.entries({}) if e.shape == sr.LOGIN}
        secrets = {e.name for e in sr.entries({}) if e.shape != sr.LOGIN}
        for (command, name), where in sorted(found.items()):
            with self.subTest(command=command, name=name, where=where[0]):
                if command == "login":
                    self.assertIn(name, logins_)
                else:
                    self.assertTrue(
                        name in secrets
                        or (sr.NAME.fullmatch(name) and name.endswith("_api_key")),
                        name,
                    )

    def test_rk_secret_set_takes_each_documented_secret_and_refuses_each_login(self):
        found = set(self.documented())
        names = sorted({n for c, n in found if c == "secret"})
        # a model binding names each key the registry does not list itself
        named = {n: {"api_key_secret": n} for n in names if sr.entry({}, n) is None}
        with tempfile.TemporaryDirectory() as d:
            cfg = config.normalize(
                {
                    "paths": {"working_dir": d, "secrets_dir": f"{d}/secrets"},
                    "model_bindings": named,
                }
            )
            for i, name in enumerate(names):
                with self.subTest(secret=name):
                    out = io.StringIO()
                    value = a_value(sr.entry(cfg, name).shape, i)
                    rc = rkmon.cmd_secret_set(
                        name, cfg, stdin=io.StringIO(value), out=out
                    )
                    self.assertEqual(rc, 0, out.getvalue())
            taken = []
            for name in sorted({n for c, n in found if c == "login"}):
                with self.subTest(login=name), mock.patch(
                    "sys.stderr", new_callable=io.StringIO
                ):
                    out = io.StringIO()
                    rc = rkmon.cmd_secret_set(
                        name, cfg, stdin=io.StringIO(made_up(1)), out=out
                    )
                    if sr.held_files(sr.entry(cfg, name)) is None:
                        self.assertEqual(rc, 0, out.getvalue())
                        taken.append(name)
                    else:
                        self.assertEqual((rc, out.getvalue()), (2, ""))
            self.assertEqual(taken, ["claude"])  # the one login kept in a file


class NoOlderNameTest(unittest.TestCase):
    """No message, doc or skill of the package names a login or a secret by an older name, nor the
    Claude section of setup, which `rkmon login claude` replaced: only the files that read an older package's files by
    those names do (READERS), and the tests."""

    # The files that read an older package's files by their older names, and the names each one reads (None: every
    # one): the registry's attribute; the ignore lists, made from it; the one-time move's copies (docker/rk_root.py);
    # the translation of a production deployment (tools/translate_production.py); layout 1's schemas; the reader of the
    # older Claude secret's shape (rkmon_lib/claude_env_shape.py); and check 16 of rkmon confirm, which compares that
    # shape with the settings and names the file in its rows (rkmon_lib/confirm.py).
    READERS = {
        "rkmon_lib/secrets_registry.py": None,
        ".gitignore": None,
        ".dockerignore": None,
        "docker/rk_root.py": None,
        "tools/translate_production.py": None,
        "schema/layout1/": None,
        "rkmon_lib/claude_env_shape.py": {"claude_env"},
        "rkmon_lib/confirm.py": {"claude_env"},
    }
    SECTION = re.compile(r"--section[ =]claude\b|section\(\s*[\"']claude[\"']\s*\)")

    def names(self) -> set:
        """Each fixed entry's older names and the older package's secret files, but a name an entry has now: its
        own, its place (a GitHub login's folder keeps its older name), and the host's gh split's folder.
        """
        fixed = sr.entries({})
        now = {e.name for e in fixed} | {e.place for e in fixed if e.place}
        older = {o.name for e in fixed for o in e.older} | set(sr.OLDER_SECRET_FILES)
        return older - now - set(sr.OTHER_LOGIN_FOLDERS)

    @staticmethod
    def word(names) -> re.Pattern:
        """Any of NAMES as a word: never a part of a longer name."""
        alternatives = "|".join(map(re.escape, sorted(names)))
        return re.compile(r"(?<![\w-])(" + alternatives + r")(?![\w-])")

    def test_the_names_it_looks_for(self):
        self.assertEqual(
            sorted(self.names()),
            [
                "admin_token",
                "claude_env",
                "gh-read-token",
                "gh_read_token",
                "jira_mcp_headers",
                "webhost_ed25519",
                "worker_api_key",
                "worker_headers",
            ],
        )

    def test_no_older_name_but_where_it_is_read(self):
        from rkmon_lib import package_files

        word = self.word(self.names())
        files = package_files.tracked(ROOT)
        self.assertIn(ROOT / "README.md", files)  # the scan reads the package
        found = []
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            if rel.startswith("tests/"):
                continue
            allowed = next(
                (
                    names
                    for k, names in self.READERS.items()
                    if rel == k or (k.endswith("/") and rel.startswith(k))
                ),
                set(),
            )
            if allowed is None:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # not text
            for i, line in enumerate(text.splitlines(), 1):
                found += [
                    f"{rel}:{i}: {m.group(1)}"
                    for m in word.finditer(line)
                    if m.group(1) not in allowed
                ]
                found += [
                    f"{rel}:{i}: {m.group(0)}" for m in self.SECTION.finditer(line)
                ]
        self.assertEqual(found, [])

    def test_it_finds_one(self):
        """The scan's rule finds an older name in prose and in code, never inside a longer name."""
        word = self.word(["claude_env", "gh-read-token"])
        for line, hit in (
            ("the secret `claude_env` keeps it", "claude_env"),
            ('open("gh-read-token")', "gh-read-token"),
            ("claude_env_shape reads it", None),
            ("my-gh-read-token-2", None),
        ):
            with self.subTest(line=line):
                m = word.search(line)
                self.assertEqual(m and m.group(1), hit)
        self.assertTrue(self.SECTION.search("./rkmon setup --section claude"))
        self.assertTrue(self.SECTION.search('doctor.section("claude")'))
        self.assertFalse(self.SECTION.search("./rkmon setup --section claude_code"))


class SchemaPatternTest(unittest.TestCase):
    def test_the_deploy_schema_names_a_secret_by_the_registry_s_rule(self):
        """The secret a model binding's key is in, in the deploy settings file (file_types.DEPLOY's
        api_key_secret_name), is named by the registry's rule for a binding's key (secrets_registry.BINDING_KEY_NAME:
        a secret's name that ends in _api_key, which the ignore lists match), or blank.
        """
        patterns = []

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "api_key_secret_name" and "pattern" in (
                        v if isinstance(v, dict) else {}
                    ):
                        patterns.append(v["pattern"])
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(file_types.DEPLOY.load())
        self.assertTrue(patterns)
        for p in patterns:
            with self.subTest(pattern=p):
                rule = p[len("^$|") :] if p.startswith("^$|") else p
                self.assertEqual(rule, f"^{sr.BINDING_KEY_NAME.pattern}$")
        self.assertTrue(sr.BINDING_KEY_NAME.pattern.startswith(sr.NAME.pattern))
        for name in ("glm_api_key", "first_review_pass_api_key", "worker_api_key"):
            with self.subTest(name=name):
                self.assertTrue(sr.BINDING_KEY_NAME.fullmatch(name))
                self.assertTrue(sr.holds_secret(name))
        for name in (
            "glm_key",
            "api_key",
            "_api_key",
            "Big_api_key",
            "glm_api_key.txt",
        ):
            with self.subTest(name=name):
                self.assertFalse(sr.BINDING_KEY_NAME.fullmatch(name))


class IgnorePatternsTest(unittest.TestCase):
    def matches(self, name: str) -> list:
        return [p for p in sr.ignore_patterns() if fnmatch.fnmatchcase(name, p)]

    def test_every_file_that_may_hold_a_secret(self):
        pats = sr.ignore_patterns()
        self.assertEqual(len(pats), len(set(pats)))
        # No folder in a pattern: a folder's own pattern ends in /, and has no other.
        self.assertEqual([p for p in pats if "/" in p.removesuffix("/")], [])
        for name in [
            *(e.name for e in sr.entries({}) if e.shape != sr.LOGIN),
            *sr.OLDER_SECRET_FILES,
            "worker_headers.prev",
            "admin_token",  # an older name (admin_access_token's)
            "gh-read-token",
            "gh_read_token.prev",  # an older package's copy of the read login
            "hosts.yml",
            "credentials",
            "claude_key",  # the Claude session's key, a login kept in a file
            "claude_env",  # its older name
            "glm_api_key",
            "gateway_api_key.new",
            "webhost_ed25519.pub",
            "gpu_ssh_key.private-copy",
            "teams_webhook.new",
            "id_ed25519",
            "id_rsa.pub",
            "server.key",
            "client.pem",
        ]:
            with self.subTest(name=name):
                self.assertTrue(self.matches(name))

    def test_the_package_s_own_private_files_and_their_copies(self):
        for name in [
            ".env",
            "compose.override.yaml",
            ".env.prev",
            "compose.override.yaml.prev",
            "records.json",
            "records.json.new",
        ]:
            with self.subTest(name=name):
                self.assertTrue(self.matches(name))
        # The package's own folder of them, whatever its files are named.
        self.assertIn(
            pathlib.PurePosixPath(private_files.RECORDS_FOLDER).name + "/",
            sr.ignore_patterns(),
        )

    def test_your_own_private_files_by_their_usual_names(self):
        """The settings files (the guides' rkmon-deploy.yaml and rkmon-build.yaml, layout 1's rkmon-settings*, any
        *settings*.yaml) and their copies, the deny list, and your network's root certificates.
        """
        for name in [
            "rkmon-deploy.yaml",
            "rkmon-build.yaml",
            "rkmon-deploy-2.yaml",
            "rkmon-deploy.yaml.bak",
            "rkmon-deploy.yaml.v1",
            "rkmon-build.yml",
            "rkmon-settings.yaml",
            "rkmon-settings-20261111.json",
            "my-settings.yaml",
            "build_settings.yml.old",
            "privacy_deny.txt",
            "privacy_deny.txt.old",
            "corp-root.crt",
            "corp-root.pem",
        ]:
            with self.subTest(name=name):
                self.assertTrue(self.matches(name))
        for name in ["build_settings.schema.json", "compose.yaml", "answers.yaml"]:
            with self.subTest(name=name):
                self.assertFalse(self.matches(name))

    def test_every_folder_that_may_hold_a_secret(self):
        """A secret's folders, wherever they are, as NAME/: a login's (its place), the one an older package's gh
        split kept the write login in, and the secrets folders; then the package's own folder of private files.
        """
        folders = [p[:-1] for p in sr.ignore_patterns() if p.endswith("/")]
        for name in (
            "gh",
            "gh-read",
            "gh-write",
            "atlassian",
            ".config",
            "secrets",
            "secrets-write",
            ".rkmon",
        ):
            with self.subTest(name=name):
                self.assertIn(name, folders)
        self.assertEqual(
            folders,
            sr.secret_folders()
            + [pathlib.PurePosixPath(private_files.RECORDS_FOLDER).name],
        )

    def test_git_leaves_out_the_package_s_own_private_files_where_they_are(self):
        """docker/.env, the override, their .prev copies, and every file of the host's records folder, its lock files
        among them, as git reads .gitignore (in a repository of its own, so a copy of the package that is no checkout
        is tested too)."""
        records = private_files.RECORDS_FOLDER
        paths = [
            *private_files.PACKAGE_FILES,
            f"{records}/records.lock",
            f"{records}/under_way.lock",
        ]
        with tempfile.TemporaryDirectory() as d:
            try:
                subprocess.run(["git", "init", "-q", d], check=True)
            except (OSError, subprocess.CalledProcessError):
                self.skipTest("no git")
            (pathlib.Path(d) / ".gitignore").write_text(
                (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
            )
            p = subprocess.run(
                ["git", "-C", d, "check-ignore", "--no-index", *paths],
                capture_output=True,
                text=True,
            )
        self.assertEqual(p.stdout.split(), paths)

    def test_no_file_of_the_package_matches(self):
        try:
            files = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        hits = [f for f in files for part in f.split("/") if self.matches(part)]
        self.assertEqual(hits, [])
        folders = [p[:-1] for p in sr.ignore_patterns() if p.endswith("/")]
        hits = [f for f in files for part in f.split("/")[:-1] if part in folders]
        self.assertEqual(hits, [])

    def test_no_file_of_the_package_is_left_out_where_case_is_ignored(self):
        """As git reads .gitignore where case is ignored (core.ignorecase=true: a clone on Windows or macOS), no file
        of the package is left out, nor one a package may hold anywhere: Claude Code's CLAUDE.md, a folder named
        claude."""
        try:
            files = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        files += ["CLAUDE.md", "docs/Claude.md", "claude/notes.md", "docs/Claude/x.md"]
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True)
            (pathlib.Path(d) / ".gitignore").write_text(
                (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
            )
            p = subprocess.run(
                ["git", "-C", d, "-c", "core.ignorecase=true"]
                + ["check-ignore", "--no-index", "--stdin"],
                input="\n".join(files),
                capture_output=True,
                text=True,
            )
        self.assertEqual((p.stdout.split(), p.stderr), ([], ""))

    def test_the_ignore_files_hold_these_patterns(self):
        """.gitignore holds each pattern as it is, and .dockerignore each after **/ (a folder's trailing / dropped:
        docker reads a pattern as a file's or a folder's), in the registry's order and last; the stray-file scan takes
        them from the registry itself. tools/generated.py writes both files (tests/test_generated.py keeps them in
        step with their sources)."""
        pats = sr.ignore_patterns()

        def lines(name: str) -> list:
            text = (ROOT / name).read_text(encoding="utf-8")
            return [x for x in text.splitlines() if x and not x.startswith("#")]

        self.assertEqual(lines(".gitignore")[-len(pats) :], pats)
        self.assertEqual(
            lines(".dockerignore")[-len(pats) :],
            [f"**/{p.removesuffix('/')}" for p in pats],
        )

    def test_the_secret_patterns_leave_out_the_package_s_own_files(self):
        """The secrets' patterns are among the ignore lists'; the ignore lists add the private files that hold no
        secret (the package's own with their kin, yours by their usual names, root certificates) and the folders.
        """
        secret, ignore = sr.secret_patterns(), sr.ignore_patterns()
        self.assertEqual(len(secret), len(set(secret)))
        self.assertEqual([p for p in secret if p not in ignore], [])
        own = [p for p in ignore if p not in secret]
        package = [
            x
            for n in private_files.PACKAGE_FILES
            for x in private_files.own_names(pathlib.PurePosixPath(n).name)
        ]
        yours = [
            x
            for n in sr.YOUR_FILES
            for x in ((n,) if n.endswith("*") else (n, n + ".*"))
        ]
        folders = [p for p in ignore if p.endswith("/")]
        self.assertEqual(own, package + yours + folders + list(sr.CERTIFICATE_FILES))
        for name in (".env", "compose.override.yaml.prev", "records.json"):
            with self.subTest(name=name):
                self.assertFalse([p for p in secret if fnmatch.fnmatchcase(name, p)])
        # What a write of one of them left, when it stopped part-way (private_files.NEW_FILE).
        for name in ("..env.4242.0a1b2c3d.new", ".records.json.4242.0a1b2c3d.new"):
            with self.subTest(name=name):
                self.assertTrue([p for p in ignore if fnmatch.fnmatchcase(name, p)])

    def test_what_may_hold_a_secret_wherever_it_is(self):
        for name in (
            "gh",
            "gh-read",
            "gh-write",  # an older package's gh split kept the write login there
            "atlassian",
            ".config",
            "secrets",
            "secrets-write",
            "hosts.yml",
            "credentials",
            "claude_key",  # the Claude session's key
            "claude_key.prev",
            "claude_env",  # its older name
            "teams_webhook.new",
            "gh-read-token",
            "gpu_ssh_key.private-copy",
            "webhost_ed25519.pub",
            "glm_api_key",
            "id_ed25519",
            "server.pem",
            "admin_token",  # an older name (admin_access_token's), and its copy
            "admin_token.bak",
        ):
            with self.subTest(name=name):
                self.assertTrue(sr.holds_secret(name))
        for name in (
            "state.json",
            "config.json",
            "outputs",
            "gh.log",
            ".env",
            "records.json",
            "README.md",
            "claude",  # Claude Code's own folder
            "CLAUDE.md",  # Claude Code's own file
        ):
            with self.subTest(name=name):
                self.assertFalse(sr.holds_secret(name))


class LoadsTest(unittest.TestCase):
    def test_an_import_loads_the_standard_library_and_checks_alone(self):
        """An import of the registry, in a Python of its own, loads the standard library, and of the package
        rkmon_lib/checks.py and what checks.py loads alone, as the registry's docstring says.
        """
        probe = (
            "import json, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "before = set(sys.modules)\n"
            "import rkmon_lib.secrets_registry\n"
            "print(json.dumps(sorted(set(sys.modules) - before)))\n"
        )
        r = subprocess.run(
            [sys.executable, "-I", "-B", "-c", probe, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(
            [
                m
                for m in json.loads(r.stdout)
                if m.split(".")[0] not in sys.stdlib_module_names
            ],
            [
                "rkmon_lib",
                "rkmon_lib.checks",
                "rkmon_lib.program",
                "rkmon_lib.secrets_registry",
                "rkmon_lib.wrapping",
            ],
        )


if __name__ == "__main__":
    unittest.main()
