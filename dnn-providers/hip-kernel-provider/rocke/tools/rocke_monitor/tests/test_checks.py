# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The check rows (rkmon_lib/checks.py): a status is its word; a fix is a command, an edit (that sets a key or deletes
it) or steps of them, never words alone, shown as the terminal shows them; a row is checked as it is made (a WARN and
a FAIL carry their fix); render prints this module's sample check outputs as they are written, wraps at WIDTH without
breaking a word, keeps the caller's own line breaks and indents, and shows only the problems when asked.
"""

import ast
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

from rkmon_lib import checks, repository_defaults  # noqa: E402
from rkmon_lib.checks import (  # noqa: E402
    DELETE,
    Check,
    Command,
    Described,
    Edit,
    Status,
    Steps,
)

BUILD = "~/rkmon-build.yaml"
DEPLOY = "~/rkmon-deploy.yaml"


def build_check_rows():
    """The rows of `rkmon check` on a sample build settings file, which render as BUILD_CHECK."""
    return [
        Check("docker", "PASS", "server <version>, buildx <version>"),
        Check(
            "settings file",
            Status.PASS,
            f"{BUILD} is yours only, outside the package, git and cloud sync",
        ),
        Check(
            "atlassian_cli.package_url",
            Status.FAIL,
            "still the template's <URL>",
            Edit(
                "atlassian_cli.package_url",
                BUILD,
                Described(
                    "the address of the atlassian tool's Linux binary for x86-64, from wherever your organization "
                    "keeps it"
                ),
            ),
        ),
        Check(
            "atlassian_cli.package_sha256",
            Status.FAIL,
            "still the template's <sha256>",
            Steps(
                [
                    "download that binary once, run",
                    Command(["sha256sum", "FILE"]),
                    "on it, and",
                    Edit(
                        "atlassian_cli.package_sha256",
                        BUILD,
                        Described("the 64 hex digits, in double quotes"),
                    ),
                ]
            ),
        ),
        Check(
            "TLS",
            Status.FAIL,
            "this network inspects TLS, and no root certificate is named",
            Edit(
                "operating_system.ca_certificates",
                BUILD,
                Described("your network's root certificate files (PEM)"),
            ),
        ),
        Check(
            "github_cli.version",
            Status.WARN,
            "not pinned; this build would take the newest release, <version>",
            Steps(
                [
                    Edit("github_cli.version", BUILD, "<version>"),
                    ", so a later build makes the same image",
                ]
            ),
        ),
        Check("disk", Status.PASS, "541 GiB free at /var/lib/docker"),
    ]


BUILD_CHECK = """\
  PASS  docker: server <version>, buildx <version>
  PASS  settings file: ~/rkmon-build.yaml is yours only, outside the package, git and cloud sync
  FAIL  atlassian_cli.package_url: still the template's <URL>
        fix: set atlassian_cli.package_url in ~/rkmon-build.yaml to the address of the atlassian tool's Linux binary
             for x86-64, from wherever your organization keeps it
  FAIL  atlassian_cli.package_sha256: still the template's <sha256>
        fix: download that binary once, run sha256sum FILE on it, and set atlassian_cli.package_sha256 in
             ~/rkmon-build.yaml to the 64 hex digits, in double quotes
  FAIL  TLS: this network inspects TLS, and no root certificate is named
        fix: set operating_system.ca_certificates in ~/rkmon-build.yaml to your network's root certificate files
             (PEM)
  WARN  github_cli.version: not pinned; this build would take the newest release, <version>
        fix: set github_cli.version in ~/rkmon-build.yaml to "<version>", so a later build makes the same image
  PASS  disk: 541 GiB free at /var/lib/docker"""


class StatusTest(unittest.TestCase):
    def test_a_status_is_its_word(self):
        for word in ("PASS", "WARN", "FAIL", "OFF", "SKIP"):
            s = Status(word)
            self.assertEqual(str(s), word)
            self.assertEqual(f"{s}", word)
            self.assertEqual(f"{s:<6}|", word.ljust(6) + "|")
            self.assertEqual(s, word)
        self.assertEqual(
            [str(s) for s in Status], ["PASS", "WARN", "FAIL", "OFF", "SKIP"]
        )

    def test_any_other_word_is_no_status(self):
        for word in ("OK", "pass", "", "ERROR"):
            with self.assertRaises(ValueError):
                Status(word)

    def test_the_words_have_one_home(self):
        """Status is the home of the five words, and PROBLEMS of the two whose rows carry a fix: the toolkit and the
        container tests read them there, and spell out neither a pattern of the words ("(PASS|WARN|...)") nor the
        problem statuses as a collection of their own."""
        alt = "|".join(str(s) for s in Status)
        pattern = re.compile(rf"\((?:{alt})(?:\|(?:{alt}))+\)")
        problems = {str(s) for s in checks.PROBLEMS}
        spelled = []
        for f in [
            ROOT / "rkmon.py",
            *sorted((ROOT / "rkmon_lib").glob("*.py")),
            *sorted((ROOT / "tests" / "container").glob("*.py")),
        ]:
            if f == ROOT / "rkmon_lib" / "checks.py":
                continue
            for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
                    said = {e.value for e in n.elts if isinstance(e, ast.Constant)}
                    if said == problems and len(n.elts) == len(problems):
                        spelled.append(f"{f.relative_to(ROOT)}:{n.lineno}")
                elif (
                    isinstance(n, ast.Constant)
                    and isinstance(n.value, str)
                    and pattern.search(n.value)
                ):
                    spelled.append(f"{f.relative_to(ROOT)}:{n.lineno}")
        self.assertEqual(spelled, [], "read checks.Status and checks.PROBLEMS")


class FixTest(unittest.TestCase):
    def test_a_command_shows_as_a_shell_line(self):
        self.assertEqual(
            str(Command(["rkmon", "login", "claude"])), "rkmon login claude"
        )
        self.assertEqual(
            str(Command(("rkmon", "secret", "admin_access_token", "--generate"))),
            "rkmon secret admin_access_token --generate",
        )
        self.assertEqual(
            str(Command(["rkmon", "check", "--settings", "my settings.yaml"])),
            "rkmon check --settings 'my settings.yaml'",
        )
        self.assertEqual(Command(["rkmon", 2]).argv, ("rkmon", "2"))

    def test_a_word_that_starts_with_the_home_folder_keeps_it(self):
        self.assertEqual(
            str(Command(["mv", "/a/b c.yaml", "~/b c.yaml"])),
            "mv '/a/b c.yaml' ~/'b c.yaml'",
        )
        self.assertEqual(
            str(Command(["rkmon", "check", "--settings", "~/rkmon-build.yaml"])),
            "rkmon check --settings ~/rkmon-build.yaml",
        )
        self.assertEqual(str(Command(["ls", "~/"])), "ls ~/")
        self.assertEqual(str(Command(["echo", "a~/b", "~"])), "echo 'a~/b' '~'")

    def test_a_command_names_its_program(self):
        for argv in ([], [""], ["  "]):
            with self.assertRaises(ValueError):
                Command(argv)

    def test_an_edit_names_key_file_and_the_value_as_json(self):
        self.assertEqual(
            str(Edit("integrations.claude_code.api_key_sent_as", DEPLOY, "bearer")),
            'set integrations.claude_code.api_key_sent_as in ~/rkmon-deploy.yaml to "bearer"',
        )
        self.assertEqual(
            str(Edit("host.resources.cpus", DEPLOY, 4)),
            "set host.resources.cpus in ~/rkmon-deploy.yaml to 4",
        )
        self.assertEqual(
            str(Edit("features.alerts.develop_breaks", DEPLOY, False)),
            "set features.alerts.develop_breaks in ~/rkmon-deploy.yaml to false",
        )
        self.assertEqual(
            str(
                Edit("serve.address", pathlib.PurePosixPath("/tmp/d.yaml"), "127.0.0.1")
            ),
            'set serve.address in /tmp/d.yaml to "127.0.0.1"',
        )

    def test_an_edit_names_its_key_and_its_file(self):
        for key, file in (("", DEPLOY), ("a.b", ""), (" ", DEPLOY)):
            for value in (1, DELETE):
                with self.assertRaises(ValueError):
                    Edit(key, file, value)

    def test_an_edit_may_delete_its_key(self):
        bind = Edit("features.dashboards.admin.bind", DEPLOY, DELETE)
        self.assertEqual(
            str(bind), "delete features.dashboards.admin.bind from ~/rkmon-deploy.yaml"
        )
        self.assertEqual(bind, Edit("features.dashboards.admin.bind", DEPLOY, DELETE))
        self.assertIs(Check("admin dashboard", "FAIL", "set", bind).fix, bind)
        self.assertEqual(repr(DELETE), "DELETE")
        self.assertEqual(  # the word is a value like any other: only DELETE deletes
            str(Edit("a.b", DEPLOY, "DELETE")),
            'set a.b in ~/rkmon-deploy.yaml to "DELETE"',
        )

    def test_a_value_only_you_know_says_where_it_comes_from(self):
        self.assertEqual(
            str(
                Edit(
                    "atlassian_cli.package_url",
                    BUILD,
                    Described(" the address of the atlassian tool's Linux binary "),
                )
            ),
            "set atlassian_cli.package_url in ~/rkmon-build.yaml to the address of the atlassian tool's Linux binary",
        )
        self.assertEqual(  # words in a plain value are a string, in double quotes
            str(Edit("a.b", DEPLOY, "the address")),
            'set a.b in ~/rkmon-deploy.yaml to "the address"',
        )
        for words in ("", "  "):
            with self.assertRaises(ValueError):
                Described(words)

    def test_steps_are_commands_and_edits_in_order_with_the_words_that_join_them(
        self,
    ):
        mv = Command(["mv", "/p/x.yaml", "~/x.yaml"])
        again = Command(["rkmon", "check", "--settings", "~/x.yaml"])
        self.assertEqual(
            str(Steps([mv, ", then", again])),
            "mv /p/x.yaml ~/x.yaml, then rkmon check --settings ~/x.yaml",
        )
        self.assertEqual(
            str(Steps(["run", again, "; QUICK_BUILD_GUIDE.md says more"])),
            "run rkmon check --settings ~/x.yaml; QUICK_BUILD_GUIDE.md says more",
        )
        self.assertEqual(  # a line break or a bracket joins without a space; blank words are left out
            str(Steps([Edit("a.b", DEPLOY, 2), "\n(", again, " lists them)", "  "])),
            "set a.b in ~/rkmon-deploy.yaml to 2\n(rkmon check --settings ~/x.yaml lists them)",
        )
        self.assertEqual(  # the join rules are the words': a command that starts with ./ keeps its space
            str(
                Steps(
                    [
                        mv,
                        ", then",
                        Command(["./rkmon", "check", "--settings", "~/x.yaml"]),
                    ]
                )
            ),
            "mv /p/x.yaml ~/x.yaml, then ./rkmon check --settings ~/x.yaml",
        )
        self.assertEqual(
            str(Steps([Command(["echo", "("]), Command(["./rkmon", "start"])])),
            "echo '(' ./rkmon start",
        )
        self.assertEqual(Steps((mv,)).parts, (mv,))

    def test_words_after_the_last_step_name_the_doc_section_that_says_more(self):
        readme = '; README.md, "The admin dashboard", says how to reach it from another machine'
        fix = Steps([Edit("features.dashboards.admin.bind", DEPLOY, DELETE), readme])
        self.assertEqual(
            str(fix),
            'delete features.dashboards.admin.bind from ~/rkmon-deploy.yaml; README.md, "The admin dashboard", says '
            "how to reach it from another machine",
        )
        mv = Command(["mv", "/p/x.yaml", "~/x.yaml"])
        again = Command(["rkmon", "check", "--settings", "~/x.yaml"])
        self.assertEqual(  # words between the steps, and after the last
            str(Steps([mv, ", then", again, "; QUICK_BUILD_GUIDE.md says more"])),
            "mv /p/x.yaml ~/x.yaml, then rkmon check --settings ~/x.yaml; QUICK_BUILD_GUIDE.md says more",
        )
        row = Check("admin dashboard", "FAIL", "bind is set", fix)
        found, shown = checks.render([row]).split("\n        fix: ")
        self.assertEqual(found, "  FAIL  admin dashboard: bind is set")
        self.assertEqual(" ".join(shown.split()), str(fix))

    def test_steps_are_never_words_alone(self):
        for parts in ([], ["read the guide"], ["a", "b"]):
            with self.assertRaisesRegex(ValueError, "never words alone"):
                Steps(parts)
        for part in (4, None, ["mv", "a", "b"], Steps([Command(["ls"])])):
            with self.assertRaises(ValueError):
                Steps([Command(["ls"]), part])

    def test_a_fix_names_the_commands_it_gives_as_they_show(self):
        chmod = Command(["chmod", "600", BUILD])
        named = Command(["echo", "a b"])
        self.assertEqual(chmod.commands, ("chmod 600 ~/rkmon-build.yaml",))
        self.assertEqual(named.commands, ("echo 'a b'",))
        self.assertEqual(Edit("a.b", DEPLOY, 2).commands, ())
        self.assertEqual(
            Steps(
                ["make it yours:", chmod, ", then", Edit("a.b", DEPLOY, 2), named]
            ).commands,
            ("chmod 600 ~/rkmon-build.yaml", "echo 'a b'"),
        )


class FixJsonTest(unittest.TestCase):
    def test_a_fix_is_kept_as_json_by_its_kind_and_read_back_the_same(self):
        """fix_to_json and fix_from_json, the JSON form of a fix that a file keeping rows uses (rk doctor's cache):
        each kind by its own key, an edit's value by what it is, steps' words as text; read back, the same fix.
        """
        mount = Edit("paths.x", DEPLOY, "/srv/x")
        cases = [
            (None, None),
            (Command(["rk", "login", "gh"]), {"command": ["rk", "login", "gh"]}),
            (Edit("a.b", DEPLOY, 7), {"edit": ["a.b", DEPLOY], "value": 7}),
            (
                Edit("a.b", DEPLOY, [1, "x"]),
                {"edit": ["a.b", DEPLOY], "value": [1, "x"]},
            ),
            (
                Edit("models.planner", DEPLOY, Described("a model it lists")),
                {"edit": ["models.planner", DEPLOY], "described": "a model it lists"},
            ),
            (Edit("a.c", DEPLOY, DELETE), {"edit": ["a.c", DEPLOY], "delete": True}),
            (
                Steps(
                    (
                        "mount it:",
                        mount,
                        ", then",
                        Command(["rkmon", "doctor"]),
                        "again",
                    )
                ),
                {
                    "steps": [
                        "mount it:",
                        {"edit": ["paths.x", DEPLOY], "value": "/srv/x"},
                        ", then",
                        {"command": ["rkmon", "doctor"]},
                        "again",
                    ]
                },
            ),
        ]
        for fix, kept in cases:
            with self.subTest(fix=str(fix)):
                self.assertEqual(checks.fix_to_json(fix), kept)
                self.assertEqual(
                    checks.fix_from_json(json.loads(json.dumps(kept))), fix
                )

    def test_a_row_is_kept_as_json_and_read_back_the_same_checked_again(self):
        """check_to_json and check_from_json (rk doctor --json, and its cache): the row's name, status, found text
        and fix in fix_to_json's form; read back, the same row, checked as Check checks it; a form that is no row's is
        a KeyError or a TypeError."""
        fix = Steps(("set it, then", Command(["rkmon", "confirm"])))
        row = Check("model fast", "FAIL", "no answer", fix)
        kept = {
            "name": "model fast",
            "status": "FAIL",
            "found": "no answer",
            "fix": checks.fix_to_json(fix),
        }
        self.assertEqual(checks.check_to_json(row), kept)
        self.assertEqual(checks.check_from_json(json.loads(json.dumps(kept))), row)
        passing = {**kept, "status": "PASS", "fix": None}
        self.assertEqual(
            checks.check_from_json(passing), Check("model fast", "PASS", "no answer")
        )
        for bad, error in (
            ({**kept, "fix": None}, ValueError),
            ({**kept, "status": "OK"}, ValueError),
            ({"name": "x"}, KeyError),
            ([], TypeError),
        ):
            with self.subTest(bad=bad), self.assertRaises(error):
                checks.check_from_json(bad)


class CheckTest(unittest.TestCase):
    def test_the_status_word_is_taken(self):
        c = Check("disk", "PASS", "541 GiB free")
        self.assertIs(c.status, Status.PASS)
        self.assertIsNone(c.fix)

    def test_a_status_that_is_none_of_the_five_is_refused(self):
        with self.assertRaises(ValueError):
            Check("disk", "OK", "541 GiB free")

    def test_a_row_names_what_it_checked_and_what_it_found(self):
        for name, found in (("", "x"), ("x", ""), (" ", "x"), ("x", "  ")):
            with self.assertRaises(ValueError):
                Check(name, "PASS", found)

    def test_a_warn_and_a_fail_carry_their_fix(self):
        prune = Command(["docker", "image", "ls"])
        for status in ("WARN", "FAIL"):
            with self.assertRaises(ValueError):
                Check("disk", status, "full")
            self.assertIs(Check("disk", status, "full", prune).fix, prune)

    def test_a_fix_is_a_command_an_edit_or_steps_never_words_alone(self):
        for fix in (
            Command(["rkmon", "doctor"]),
            Edit("a.b", DEPLOY, 1),
            Steps(["run", Command(["rkmon", "doctor"])]),
        ):
            self.assertIs(Check("disk", "FAIL", "full", fix).fix, fix)
        for fix in ("free some space", "", "  ", ["rkmon", "doctor"], 4, {"key": "a"}):
            for status in ("FAIL", "PASS"):
                with self.subTest(fix=fix, status=status):
                    with self.assertRaises(ValueError):
                        Check("disk", status, "full", fix)

    def test_a_row_with_nothing_to_fix_may_carry_a_fix_it_never_shows(self):
        for status in ("PASS", "OFF", "SKIP"):
            fix = Command(["docker", "image", "ls"])
            c = Check("disk", status, "541 GiB free", fix)
            self.assertIs(c.fix, fix)
            self.assertNotIn("fix", checks.render([c]))

    def test_a_row_cannot_be_changed(self):
        c = Check("disk", "PASS", "541 GiB free")
        with self.assertRaises(AttributeError):
            c.status = Status.FAIL

    def test_failed_is_whether_a_row_is_a_fail(self):
        prune = Command(["docker", "image", "ls"])
        rows = [
            Check("disk", "PASS", "541 GiB free"),
            Check("memory", "WARN", "tight", prune),
            Check("alerts", "OFF", "off"),
            Check("image", "SKIP", "checked at step 4"),
        ]
        self.assertFalse(checks.failed([]))
        self.assertFalse(checks.failed(rows))
        self.assertTrue(checks.failed(rows + [Check("ports", "FAIL", "busy", prune)]))


class RenderTest(unittest.TestCase):
    def test_the_build_check_word_for_word(self):
        self.assertEqual(checks.render(build_check_rows()), BUILD_CHECK)

    def test_problems_shows_the_warn_and_fail_rows_only(self):
        want = "\n".join(
            line for line in BUILD_CHECK.split("\n") if not line.startswith("  PASS")
        )
        self.assertEqual(checks.render(build_check_rows(), rows="problems"), want)
        passing = [r for r in build_check_rows() if r.status == "PASS"] + [
            Check("alerts", "OFF", "off")
        ]
        self.assertEqual(checks.render(passing, rows="problems"), "")
        self.assertEqual(checks.render([]), "")

    def test_rows_is_all_or_problems(self):
        with self.assertRaises(ValueError):
            checks.render(build_check_rows(), rows="fails")

    def test_a_row_without_its_fix_when_the_caller_shows_it_elsewhere(self):
        row = Check(
            "doctor after", "FAIL", "model fast: no answer", Command(["rkmon", "logs"])
        )
        self.assertEqual(
            checks.render([row]),
            "  FAIL  doctor after: model fast: no answer\n        fix: rkmon logs",
        )
        self.assertEqual(
            checks.render([row], fixes=False),
            "  FAIL  doctor after: model fast: no answer",
        )

    def test_the_deploy_check_word_for_word(self):
        rows = [
            Check("docker", "PASS", "server <version>, compose <version>"),
            Check(
                "settings file",
                "PASS",
                f"{DEPLOY} is yours only, outside the package, git and cloud sync",
            ),
            Check(
                "settings",
                "PASS",
                "every key fits the deploy schema, and no <...> is left",
            ),
            Check(
                "features",
                "PASS",
                "review automation prototype · merge queue off · admin dashboard on · team dashboard off\n"
                "· alerts off · ticket creation off",
            ),
            Check("disk", "PASS", "541 GiB free at /var/lib/docker"),
            Check("memory", "PASS", "the container may use 8 of this host's 76 GiB"),
            Check(
                "CPUs",
                "WARN",
                "host.resources.cpus 2 is fewer than the reviews and builds that may run at once (4)",
                Edit("host.resources.cpus", DEPLOY, 4),
            ),
            Check("ports", "PASS", "127.0.0.1:18614 is free"),
            Check(
                "container name",
                "FAIL",
                "rocke-monitor is used by another deployment on this host",
                Steps(
                    [
                        Edit(
                            "container.name",
                            DEPLOY,
                            Described("a name of its own, such as rocke-monitor-2"),
                        ),
                        "\n(",
                        Command(["docker", "compose", "ls", "--all"]),
                        "lists the names in use)",
                    ]
                ),
            ),
        ]
        self.assertEqual(
            checks.render(rows),
            """\
  PASS  docker: server <version>, compose <version>
  PASS  settings file: ~/rkmon-deploy.yaml is yours only, outside the package, git and cloud sync
  PASS  settings: every key fits the deploy schema, and no <...> is left
  PASS  features: review automation prototype · merge queue off · admin dashboard on · team dashboard off
        · alerts off · ticket creation off
  PASS  disk: 541 GiB free at /var/lib/docker
  PASS  memory: the container may use 8 of this host's 76 GiB
  WARN  CPUs: host.resources.cpus 2 is fewer than the reviews and builds that may run at once (4)
        fix: set host.resources.cpus in ~/rkmon-deploy.yaml to 4
  PASS  ports: 127.0.0.1:18614 is free
  FAIL  container name: rocke-monitor is used by another deployment on this host
        fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as rocke-monitor-2
             (docker compose ls --all lists the names in use)""",
        )

    def test_the_login_walk_word_for_word(self):
        rows = [
            Check("github_read", "PASS", "a read-only token for <org>/<repo>"),
            Check(
                "claude", "FAIL", "no Claude key", Command(["rkmon", "login", "claude"])
            ),
            Check(
                "admin_access_token",
                "FAIL",
                "not set",
                Command(["rkmon", "secret", "admin_access_token", "--generate"]),
            ),
            Check(
                "github_write",
                "OFF",
                "nothing that is on writes to GitHub (review automation is in prototype)",
            ),
            Check("teams_webhook", "OFF", "alerts are off"),
            Check("web_host_ssh_key", "OFF", "the team dashboard is off"),
            Check("atlassian", "OFF", "ticket creation is off"),
            Check("gpu_ssh_key", "OFF", "remote GPU tests are off"),
            Check("image", "SKIP", "checked at step 4, once it is on this host"),
        ]
        self.assertEqual(
            checks.render(rows),
            """\
  PASS  github_read: a read-only token for <org>/<repo>
  FAIL  claude: no Claude key
        fix: rkmon login claude
  FAIL  admin_access_token: not set
        fix: rkmon secret admin_access_token --generate
  OFF   github_write: nothing that is on writes to GitHub (review automation is in prototype)
  OFF   teams_webhook: alerts are off
  OFF   web_host_ssh_key: the team dashboard is off
  OFF   atlassian: ticket creation is off
  OFF   gpu_ssh_key: remote GPU tests are off
  SKIP  image: checked at step 4, once it is on this host""",
        )

    def test_the_settings_checks_word_for_word(self):
        rows = [
            Check(
                "RAM disk",
                "WARN",
                "host.resources.ramdisk_gib 4 is less than the starting reviews and builds need (about 7 GiB)",
                Edit(
                    "host.resources.ramdisk_gib",
                    DEPLOY,
                    Described("7 or more, or to 0 for no RAM disk"),
                ),
            ),
            Check(
                "alerts",
                "FAIL",
                "features.alerts.develop_breaks is on, and features.alerts.mode is missing",
                Steps(
                    [
                        "run",
                        Command(["rkmon", "template", "section", "features.alerts"]),
                        f", paste its text over features.alerts in {DEPLOY}, then set develop_breaks.enabled back to"
                        ' true and fill in each key whose comment says "Required when"',
                    ]
                ),
            ),
            Check(
                "Claude Code",
                "FAIL",
                "claude_env sends the key as a bearer token, and integrations.claude_code.api_key_sent_as\n"
                'says "header"',
                Edit("integrations.claude_code.api_key_sent_as", DEPLOY, "bearer"),
            ),
            Check(
                "Claude Code",
                "PASS",
                "claude_env sets ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_OPUS_MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL and"
                " ANTHROPIC_DEFAULT_HAIKU_MODEL; the move leaves them behind, since each pass asks for its model"
                " (features.review_automation) by its built-in id",
            ),
            Check(
                "ticket creation",
                "FAIL",
                'the project requires <field id> ("<field name>") on a new "Task", and\n'
                "required_fields does not set it",
                Edit(
                    "features.ticket_creation.required_fields.<field id>",
                    DEPLOY,
                    "<one of: <value>, <value>>",
                ),
            ),
        ]
        self.assertEqual(
            checks.render(rows),
            """\
  WARN  RAM disk: host.resources.ramdisk_gib 4 is less than the starting reviews and builds need (about 7 GiB)
        fix: set host.resources.ramdisk_gib in ~/rkmon-deploy.yaml to 7 or more, or to 0 for no RAM disk
  FAIL  alerts: features.alerts.develop_breaks is on, and features.alerts.mode is missing
        fix: run rkmon template section features.alerts, paste its text over features.alerts in ~/rkmon-deploy.yaml,
             then set develop_breaks.enabled back to true and fill in each key whose comment says "Required when"
  FAIL  Claude Code: claude_env sends the key as a bearer token, and integrations.claude_code.api_key_sent_as
        says "header"
        fix: set integrations.claude_code.api_key_sent_as in ~/rkmon-deploy.yaml to "bearer"
  PASS  Claude Code: claude_env sets ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_OPUS_MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL
        and ANTHROPIC_DEFAULT_HAIKU_MODEL; the move leaves them behind, since each pass asks for its model
        (features.review_automation) by its built-in id
  FAIL  ticket creation: the project requires <field id> ("<field name>") on a new "Task", and
        required_fields does not set it
        fix: set features.ticket_creation.required_fields.<field id> in ~/rkmon-deploy.yaml to
             "<one of: <value>, <value>>\"""",
        )

    def test_no_line_is_longer_than_the_width_and_no_word_is_broken(self):
        words = [f"word{i}" * (1 + i % 5) for i in range(200)]
        text = " ".join(words)
        out = checks.render([Check("long", "FAIL", text, Command(["echo", *words]))])
        for line in out.split("\n"):
            self.assertLessEqual(len(line), checks.WIDTH)
        found, fix = out.split("\n        fix: ")
        self.assertEqual(" ".join(found.split()[2:]), text)
        self.assertEqual(" ".join(fix.split()), "echo " + text)
        for line in found.split("\n")[1:]:
            self.assertRegex(line, r"^ {8}\S")
        for line in fix.split("\n")[1:]:
            self.assertRegex(line, r"^ {13}\S")

    def test_a_word_longer_than_a_line_goes_on_a_line_of_its_own(self):
        url = "https://example.com/" + "a" * 150
        out = checks.render(
            [
                Check(
                    "source", "FAIL", f"cannot fetch {url} here", Command(["open", url])
                )
            ]
        ).split("\n")
        self.assertEqual(out[0], "  FAIL  source: cannot fetch")
        self.assertEqual(out[1], " " * 8 + url)
        self.assertEqual(out[2], " " * 8 + "here")
        self.assertEqual(out[3], " " * 8 + "fix: open")
        self.assertEqual(out[4], " " * 13 + url)

    def test_a_quoted_phrase_that_fits_on_a_line_is_never_split(self):
        phrase = '"' + " ".join(["quoted"] * 12) + '"'
        out = checks.render(
            [
                Check(
                    "name",
                    "FAIL",
                    "x" * 60 + f" says {phrase}",
                    Edit("key", "x" * 60, " ".join(["quoted"] * 12)),
                )
            ]
        ).split("\n")
        self.assertEqual(out[1], " " * 8 + phrase)
        self.assertEqual(out[3], " " * 13 + phrase)

    def test_a_command_of_a_fix_that_fits_on_a_line_is_never_split(self):
        # Where the line would pass the width, the command moves whole to the fix's next line: a line copied runs it.
        again = Command(
            ["~/the-package-folder/docker/rkmon", "check", "--settings", BUILD]
        )
        fix = Steps([f"correct {BUILD} as the line above says, then run", again])
        out = checks.render([Check("settings file", "FAIL", "it cannot be read", fix)])
        self.assertEqual(
            out.split("\n")[1:],
            [
                f"        fix: correct {BUILD} as the line above says, then run",
                " " * 13 + str(again),
            ],
        )


class TestMessagesNameTheContainersCommands(unittest.TestCase):
    """The toolkit's messages, fixes, docstrings and comments name what replaces the interview: an edit
    of the deploy settings file, then confirm and apply; a login or a secret; the start; the web host's install; or a
    container command. None names ./configure, setup or a section of the interview, but the lines LEFT names.
    """

    SETUP = re.compile(
        r"\./configure|\bconfigure --section|\brk setup|rkmon setup|setup --settings|--section "
    )
    # Lines that keep such words: rk overlaps' own --section, and the blank config.json the package ships
    # (package_files.BLANK_CONFIG).
    LEFT = (
        "rk overlaps $N --section",
        "--section needs a PR number",
        "Blank on purpose. ./configure writes",
    )

    FOLDERS = (
        "rkmon_lib",
        "worker",
        "tools",
        "docker",
        "requirements",
        "agentic",
        "schema",
        "repository",
    )

    def swept(self):
        """The package's code and tables: rk, rkmon.py and FOLDERS, but the guides (*.md)."""
        root = pathlib.Path(__file__).resolve().parents[1]
        files = [root / "rk", root / "rkmon.py"] + [
            f
            for d in self.FOLDERS
            for f in sorted((root / d).rglob("*"))
            if f.is_file()
            and f.suffix in (".py", ".txt", ".sh", ".yaml", ".json", "")
            and "__pycache__" not in f.parts
        ]
        return root, files

    def test_no_swept_file_names_the_interview(self):
        root, files = self.swept()
        for f in (
            "rkmon.py",
            "rkmon_lib/doctor.py",
            "docker/rkmon",
            "docker/compose.yaml",
            "agentic/skills/rocke-monitor/lessons_for.py",
            "schema/deploy_settings.schema.json",
            repository_defaults.home().path.relative_to(root).as_posix(),
        ):
            self.assertIn(root / f, files)
        found = [
            f"{f.relative_to(root)}:{n}: {line.strip()}"
            for f in files
            for n, line in enumerate(
                f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            )
            if self.SETUP.search(line) and not any(x in line for x in self.LEFT)
        ]
        self.assertEqual(found, [])

    def test_the_commands_the_messages_name_are_the_wrapper_s(self):
        from rkmon_lib import commands, doctor

        env = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "rkmon"}
        said = doctor.apply_words(env)
        self.assertEqual(
            said,
            "edit your deploy settings file, then rkmon confirm and rkmon apply",
        )
        for words in (
            ["confirm"],
            ["apply"],
            ["doctor"],
            ["login", "atlassian"],
            ["login", "github_read"],
            ["secret", "teams_webhook"],
            ["web-host", "install"],
        ):
            with self.subTest(words=words):
                commands.table_parser().parse_args(words)

    def test_every_settings_fix_ends_with_confirm_and_apply(self):
        """Each settings fix a row prints ends with the one home of how an edit of the deploy settings file comes to
        run (doctor.apply_commands: `<wrapper> confirm` and `<wrapper> apply`), as the operator types it on the host,
        and apply_words says the same: settings_parts and settings_fix, every settings section, settings_file and
        cannot_run_fix."""
        from rkmon_lib import doctor
        from rkmon_lib.checks import Described, Steps

        env = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        value = Described("a project")
        fixes = {
            "settings_parts": Steps(doctor.settings_parts("jira.project", value, env)),
            "settings_fix": doctor.settings_fix("jira.project", value, env),
            "settings_file": Steps(("fix it in", *doctor.settings_file(env))),
            "cannot_run_fix": doctor.cannot_run_fix(env),
            "apply_words": doctor.apply_words(env),
            **{f"section {n}": doctor.section(n, env) for n in doctor.SECTION_SETTINGS},
        }
        for name, fix in fixes.items():
            with self.subTest(name):
                self.assertTrue(
                    str(fix).endswith(", then ./rkmon confirm and ./rkmon apply"), fix
                )

    @mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"})
    def test_each_row_shown_before_the_first_start_ends_its_fix_with_apply(self):
        """Each row whose settings fix may show before the first start shows after it too, where a later start never
        applies, so its fix ends with confirm and apply, as every settings fix does (doctor.apply_commands). Each fix
        comes from the row's own code: the claude login's test call with no answer, and refused (`rkmon login claude`
        makes it whenever a login is redone); config.load with no configuration, a config.json that is none, and a
        configuration that cannot run; the applied record, at every start; a step of the start; the request service's
        install with settings it cannot use."""
        from rkmon_lib import (
            config,
            orchestrator,
            provision,
            reach,
            runtime_config,
            web_host_install,
        )

        fixes = {}
        cfg = config.load_defaults()
        cfg = dict(
            cfg, claude=dict(cfg.get("claude") or {}, gateway="https://gw.example.org")
        )
        for name, answer in (
            ("no answer", (0, "timed out")),
            ("refused", (404, "not_found_error")),
        ):
            with mock.patch.object(
                orchestrator, "read_key", return_value="sk-test-123"
            ):
                row = reach.claude_test(
                    cfg, reach.again(os.environ), post=lambda *a, said=answer: said
                )
            fixes[f"the claude login's test call, {name}"] = row.fix
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"RKMON_HOME": d}
        ):
            home = pathlib.Path(d)
            on_a_host = {"RKMON_CONTAINER": ""}  # where config.load reads config.json
            for name, text in (
                ("no configuration", None),
                ("a config.json that is none", "[]"),
                ("a configuration that cannot run", '{"repo": "acme/widgets"}'),
            ):
                if text is not None:
                    (home / "config.json").write_text(text, encoding="utf-8")
                with mock.patch.dict(os.environ, on_a_host):
                    with self.assertRaises(config.ConfigError) as e:
                        config.load()
                fixes[f"config.load, {name}"] = e.exception.fix("./rkmon")
            with self.assertRaises(config.ConfigError) as e:
                runtime_config.record_applied(
                    home, runtime_config.Resolved({}, {}, ["a problem"])
                )
            fixes["the applied record"] = e.exception.fix("./rkmon")
            fixes["a step of the start"] = provision._settings_fix("repo")
            out = io.StringIO()
            with mock.patch.object(
                web_host_install.publish, "key_here", return_value=True
            ):
                with mock.patch.object(
                    web_host_install, "nothing_to_install", return_value=""
                ):
                    self.assertEqual(web_host_install.install({}, run=None, out=out), 4)
            fixes["the request service's install"] = out.getvalue().split(
                checks.FIX_LABEL, 1
            )[1]
        for name, fix in fixes.items():
            with self.subTest(name):
                said = " ".join(str(fix).split())
                confirms = said.count("./rkmon confirm")
                self.assertTrue(confirms, said)
                self.assertEqual(
                    said.count("./rkmon confirm and ./rkmon apply"), confirms, said
                )

    def test_each_command_the_fix_helpers_name_is_the_wrapper_s(self):
        """Each command the fix helpers name, whatever this process's place, starts with the wrapper's name and
        parses with the container's command table: the fix of each part of the deployment, the checks again,
        the removal by hand then the stop and the start, the confirm and the apply, an rk command, the GitHub logins,
        the image's fix, the request service's install.
        """
        from rkmon_lib import commands, doctor, gh, reach, web_host_install
        from rkmon_lib.checks import Command, Steps

        def named(fix) -> list:
            return [
                p
                for p in (fix.parts if isinstance(fix, Steps) else (fix,))
                if isinstance(p, Command)
            ]

        for env in (
            {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "rkmon"},
            {"RKMON_WRAPPER": "rkmon"},
        ):
            sections = [*doctor.SECTION_STARTS, *doctor.SECTION_SETTINGS]
            fixes = [doctor.section(name, env) for name in sections] + [
                doctor.check_again(env),
                doctor.remove_then_start("a piece", env),
                reach.again(env),
                Steps(doctor.apply_commands(env)),
                doctor.rk_command("features", environ=env),
                doctor.gh_login_fix(gh.READ, env),
                doctor.gh_login_fix(gh.WRITE, env),
                doctor.cannot_run_fix(env),
                doctor.image_fix(env),
                web_host_install.install_command(env),
            ]
            for fix in fixes:
                for c in named(fix):
                    with self.subTest(env=env, command=str(c)):
                        self.assertEqual(c.argv[0], "rkmon")
                        commands.table_parser().parse_args(list(c.argv[1:]))

    def test_the_fix_of_each_part_of_the_deployment(self):
        """Each part's fix, as the operator types it on the host (doctor.section): the stop and the start; or the
        edit of the sections of the deploy settings file that hold its settings, then confirm and apply.
        """
        from rkmon_lib import doctor

        env = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        file = "your deploy settings file, then ./rkmon confirm and ./rkmon apply"
        want = {
            "deploy": "./rkmon stop --drain, then ./rkmon start",
            "agentic": "./rkmon stop --drain, then ./rkmon start",
            "reviewer": "edit features.review_automation.passes.first_review_pass and model_bindings in "
            + file,
            "model": f"edit model_bindings in {file}",
            "bindings": f"edit model_bindings in {file}",
            "jira": f"edit features.ticket_creation and integrations.atlassian_cli in {file}",
            "gpu": f"edit features.review_automation.remote_gpu_tests in {file}",
            "publish": f"edit features.dashboards.team in {file}",
        }
        sections = [*doctor.SECTION_STARTS, *doctor.SECTION_SETTINGS]
        self.assertEqual({n: str(doctor.section(n, env)) for n in sections}, want)

    def test_each_section_a_settings_fix_names_is_a_key_of_the_deploy_settings_file(
        self,
    ):
        """Each section of the deploy settings file a section's fix names (doctor.SECTION_SETTINGS) is a key of the
        file's schema (deploy_tree.node)."""
        from rkmon_lib import deploy_tree, doctor

        for name, keys in doctor.SECTION_SETTINGS.items():
            for key in keys:
                with self.subTest(section=name, key=key):
                    self.assertIsNotNone(deploy_tree.node(key)[1], key)

    @staticmethod
    def names_the_interview(node) -> bool:
        """Whether the argv NODE (a tuple or list) names the interview: setup as a program's first word, --section
        but rk overlaps' own, or ./configure.
        """
        words = [e.value if isinstance(e, ast.Constant) else None for e in node.elts]
        if "--section" in words and "overlaps" not in words:
            return True
        if "./configure" in words:
            return True
        return words[1:2] == ["setup"] and words[0] in (None, "rk", "rkmon", "./rkmon")

    def test_no_argv_names_the_interview(self):
        """No argv of the swept code (a tuple or list, as Command takes it) names the interview: the sweep above reads
        the text, this the code."""
        root, files = self.swept()
        found = [
            f"{f.relative_to(root)}:{node.lineno}: {ast.unparse(node)}"
            for f in files
            if f.suffix == ".py"
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8")))
            if isinstance(node, (ast.Tuple, ast.List))
            and self.names_the_interview(node)
        ]
        self.assertEqual(found, [])
        for argv, named in (
            ('(w, "setup", "--section", name)', True),
            ('("rkmon", "setup")', True),
            ('("./configure", "--section", "gpu")', True),
            ('("./configure", "--check")', True),
            ('(rk, "overlaps", n, "--section")', False),
            ('("secrets", "setup", "exported_at")', False),
        ):
            with self.subTest(argv=argv):
                node = ast.parse(argv, mode="eval").body
                self.assertEqual(self.names_the_interview(node), named)

    @staticmethod
    def named_value(root, f, node):
        """The value NODE gives in the swept file F of the package at ROOT: a constant's, or that of a name of F's
        module (NAME) or of a module of F's package (MODULE.NAME); None for anything else.
        """
        import importlib

        if isinstance(node, ast.Constant):
            return node.value
        module = ".".join(f.relative_to(root).with_suffix("").parts)
        if isinstance(node, ast.Name):
            name, attr = module, node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            package = module.rpartition(".")[0]
            name = f"{package}.{node.value.id}" if package else node.value.id
            attr = node.attr
        else:
            return None
        try:
            return getattr(importlib.import_module(name), attr, None)
        except ImportError:
            return None

    def test_each_rk_command_the_code_names_is_one_of_rk_s(self):
        """Each rk command the swept code names (doctor.rk_command's word, or an argv's word after "rk") is one that
        rkmon.py's own parser takes: `rk WORD --help` shows its help and exits 0. rk_command's word is a string, a
        module's name for one (`named_value`), or, for rk_command(*args), the first word of each command of the table
        the call passes (`tables`); the test fails on any other, which it cannot read.
        """
        import contextlib
        import io
        from unittest import mock

        import rkmon

        # The calls that pass a table's commands (rk_command(*args)), by their file and function: the table's.
        tables = {
            ("rkmon.py", "_run_instead"): [a for a, _ in rkmon.INSTEAD.values()],
        }
        root, files = self.swept()
        words, unread = {}, []
        for f in files:
            if f.suffix != ".py":
                continue
            where = str(f.relative_to(root))
            tree = ast.parse(f.read_text(encoding="utf-8"))
            within = {}  # a node's id -> the name of the function it is in
            for fn in ast.walk(tree):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    within.update((id(n), fn.name) for n in ast.walk(fn))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    called = (
                        fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                    )
                    first = node.args[0] if node.args else None
                    if called != "rk_command" or first is None:
                        continue
                    at = f"{where}:{node.lineno}"
                    if isinstance(first, ast.Starred):
                        table = tables.get((where, within.get(id(node))), [])
                        named = [argv[0] for argv in table]
                    else:
                        named = [self.named_value(root, f, first)]
                    if not named or not all(isinstance(w, str) for w in named):
                        unread.append(at)
                    for word in named:
                        if isinstance(word, str):
                            words.setdefault(word, at)
                elif isinstance(node, (ast.Tuple, ast.List)):
                    argv = [
                        e.value if isinstance(e, ast.Constant) else None
                        for e in node.elts
                    ]
                    for a, b in zip(argv, argv[1:]):
                        if (
                            a == "rk"
                            and isinstance(b, str)
                            and re.fullmatch(r"[a-z][a-z0-9-]*", b)
                        ):
                            words.setdefault(b, f"{where}:{node.lineno}")
        self.assertEqual(unread, [])
        # A string, a module's name (atlassian.py's writer.ATLASSIAN) and a table's commands (INSTEAD's) are read.
        for word in ("gpu-check", "atlassian", "pause", "stop", "config"):
            self.assertIn(word, words)
        for word, where in sorted(words.items()):
            with self.subTest(word=word, where=where):
                with mock.patch(
                    "os.execv",
                    side_effect=AssertionError(
                        f"rk {word} --help runs another program"
                    ),
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    with self.assertRaises(SystemExit) as shown:
                        rkmon.main([word, "--help"])
                self.assertEqual(shown.exception.code, 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as shown:
            rkmon.main(["gpu-chek", "--help"])
        self.assertNotEqual(shown.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
