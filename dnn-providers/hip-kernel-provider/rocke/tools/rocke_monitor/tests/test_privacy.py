# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The privacy check (rkmon_lib/privacy.py), through its one way in, `run` (or its command line): a fake of each kind,
planted in a checked file, fails it; a clean file passes; the report gives labels and counts, never a value. Every fake
is built at run time, so this file itself passes the check."""

import ast
import contextlib
import copy
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import sandbox  # noqa: E402
from rkmon_lib import (  # noqa: E402
    checks,
    file_types,
    paths,
    privacy,
    registry,
    settings_file,
    settings_templates,
)

AT, DOT, SLASH = "@", ".", "/"
CORP = "acme" + "-corp" + DOT + "com"  # a made-up company domain
# An address that names a site of its own beside its placeholder (a made-up site), bare and in a URL.
SITE_HOST = "<site>" + DOT + "corp-example" + DOT + "net"
SITE_URL = "https" + "://" + SITE_HOST + "/v1"
# An internal product's domain, whose site label a placeholder may be.
ATL = "atlassian" + DOT + "net"
PRN = "143" + "21"  # a PR number the fake live state knows
# A fake machine's environment: its git identity, user and host names, and WSL distribution.
FAKE_ENV = {
    "GITNAME": "Opal " + "Operator",
    "GITEMAIL": "opal" + AT + CORP,
    "WINUSER": "winuser" + "-x",
    "WINHOST": "HOST" + "-X99",
    "WSL_DISTRO_NAME": "Distro" + "-X",
}


def fake_live(d: pathlib.Path) -> tuple:
    """Live sources with fake values, written into `d`: a config, a state, a Claude workspace and a deny list
    (deny.txt). Returns the three live sources, (config, state, workspace), in the order `privacy.run` and
    `privacy.sources` take them."""
    cfg = {
        "operator": {"github_login": "op" + "-login-x"},
        "paths": {
            "root": SLASH + "srv" + SLASH + "mon-x",
            "dashboard": "/srv/out-x/q.md",
        },
        "teams": {
            "channel": "Widget " + "Team Room",
            "webhook_file": "~/.config/rkmon/hook-x",
        },
        "publish": {
            "html": {
                "host": "web7" + DOT + CORP,
                "user": "pub" + "user-x",
                "url": "http" + "://web7." + CORP + ":8080/x",
            }
        },
        "github_jira_users": {"dev" + "-mapped-x": "mapped" + AT + CORP},
        "incident_issue": {
            "note": "Jane " + "Roe" + " via the user, 2026-10-29: file it"
        },
    }
    st = {
        "prs": {
            "1": {
                "author": "author" + "-x1",
                "reviews": [{"login": "reviewer" + "-x2", "state": "APPROVED"}],
                "review_requests": ["team-reviewers", "requested" + "-x3"],
            },
            PRN: {"n": int(PRN)},
        },
        "conflicts": {
            "186" + "13": {}
        },  # a number that is a setting (the dashboard's port), whatever the state holds
        "codeowners": {"teams": {"team-p": ["owner" + "-x4"]}},
    }
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "state.json").write_text(json.dumps(st), encoding="utf-8")
    ws = d / "ws"
    (ws / ".claude" / "skills" / "rocke-monitor").mkdir(parents=True)
    (ws / ".claude" / "project_config.md").write_text(
        "## JIRA\n| `project_key` | `ZQK"
        + "X` |\n| `default_epic` | `ZQKX-"
        + "77` |\n"
        "| `default_component` | `Widget "
        + "Kernels` |\n| `default_labels` | `zq"
        + "-label, ab` |\n"
        "| `base_url` | `https://wikihost-x.invalid/wiki` |\n## Confluence\n"
        "| `site` | `docs` |\n"  # a bare site name that is a word of a public site: never a value to look for
        "| `space_key` | `WQS" + "P` |\n| `space_id` | `99887" + "7665` |\n",
        encoding="utf-8",
    )
    (ws / ".claude" / "skills" / "rocke-monitor" / "rules.md").write_text(
        "(Rich " + "Moe's request, relayed by the operator)\n", encoding="utf-8"
    )
    (d / "deny.txt").write_text(
        "# local values\ndenied" + "-value-x\n", encoding="utf-8"
    )
    return d / "config.json", d / "state.json", ws


# label -> a line that must fail with it
PLANTS = {
    privacy.L_ME: "posted as op" + "-login-x today",
    privacy.L_GIT: "signed off by Opal " + "Operator",
    privacy.L_LOGINS: "a fix from reviewer" + "-x2",
    privacy.L_CFG: "alert the Widget " + "Team Room",
    privacy.L_PEOPLE: "as Rich " + "Moe asked",
    privacy.L_MACHINE: "built on HOST" + "-X99",
    privacy.L_PATHS: "see /srv/" + "mon-x/workers",
    privacy.L_WORKSPACE: "file it in WQS" + "P",
    privacy.L_DENY: "mentions denied" + "-value-x here",
    privacy.P_EMAIL: "write to someone" + AT + "elsewhere" + DOT + "org",
    privacy.P_HOME: "open " + SLASH + "home" + SLASH + "alice" + SLASH + "x",
    privacy.P_SYNC: "C:" + "\\Users\\u\\One" + "Drive - Widget\\x",
    privacy.P_TOKEN: "token " + "gh" + "p_" + "A1b2C3d4" * 5,
    privacy.P_TICKET: "fixes " + "ABC" + "-1234",
    privacy.P_URL: "see https" + "://intranet" + DOT + "widgets" + DOT + "io/x",
    privacy.P_HOST: "ssh to build3" + DOT + CORP,
    privacy.P_WIKI: "the guide (page" + "Id=12" + "3456)",
    privacy.P_PR: "since #" + "14" + "059 merged",
    privacy.P_GPU: "measured on an MI" + "300X",
    privacy.P_DOC: "see ~/notes/" + "plan.md",
    privacy.P_HANDLE: "thanks " + AT + "jdoe7 for the fix",
    privacy.P_PERF: "it reaches 512 " + "TFLOP/s now",
    privacy.P_SETTINGS: "rkmon" + "_settings: 1\nconfig:\n  repo: acme/widgets",
    privacy.P_ARTIFACTS: "ATLASSIAN_CLI_PACKAGE_URL=https"
    + "://bins"
    + DOT
    + "widgets"
    + DOT
    + "io:8443/"
    + "art"
    + "ifactory/"
    "TOOLS-LOCAL/cli/linux-amd64/cli",
    privacy.P_DATED: "# 10" + "-31: the tick printed it twice",
    privacy.P_ATTRIB: "the merge plan (the user" + "'s request)",
    privacy.P_INCIDENT: "it reopened INC" + "-2026-10-31-7",
}
# more forms of the same kinds
MORE = {
    privacy.P_PR: [
        "PR " + "13" + "975 fixed it",
        "github.com/ROCm/example/pull/" + "14" + "059",
    ],
    privacy.P_GPU: ["the Rad" + "eon card", "a St" + "rix laptop"],
    privacy.P_DOC: [
        "C:" + "\\Users\\x\\Down" + "loads\\a.txt",
        "/mnt/c/x/" + "spec.pdf",
    ],
    privacy.P_HANDLE: ["cc " + AT + "jrandom", "(" + AT + "some-org/real-team)"],
    privacy.P_PERF: [
        "1.6 " + "TB/s of bandwidth",
        "an " + "M" + "FU of 45%",
        "the kernel runs in 3.2 " + "ms",
        "900 " + "GB/s",
    ],
    # a settings file of either kind, deploy or build, whatever its name or form: comments first, or the header
    # anywhere at the top level, in JSON or YAML
    privacy.P_SETTINGS: [
        "# exported by rk\n\n  rkmon" + "_settings: 1",
        '{"rkmon' + '_settings": 1, "config": {}}',
        '{"config": {}, "rkmon' + '_settings": 1}',
        "# a deploy settings file\nkind: deploy" + "_settings\nschema_version: 1",
        "kind: build" + "_settings\nschema_version: 1\nrocm:\n  version: 7.2",
        "# the older layout\n\nrkmon_build" + "_settings: 1\nrocm: {}",
        '{"schema_version": 1, "kind": "build' + '_settings"}',
        '{"rocm": {}, "rkmon_build' + '_settings": 1}',
        "schema_version: 1\nkind: deploy"
        + "_settings\nrepository:\n  name: acme/widgets",
        "# made by hand\nrocm:\n  version: 7.2\nkind: build" + "_settings",
        "config:\n  repo: acme/widgets\nrkmon" + "_settings: 1",
    ],
    # an artifact store under an example host, or only its name in a host or a path, fails all the same
    privacy.P_ARTIFACTS: [
        "https" + "://store.example.com/" + "art" + "ifactory/REPO-LOCAL/tool",
        "fetched from the build" + "art" + "ifactory host",
        "an " + "Art" + "ifactory repository",
    ],
    # a dated narrative in a docstring, a parenthesis, a bullet, a table cell, with a year or a comma
    privacy.P_DATED: [
        '"""11' + '-04, the fix landed at 03:50Z."""',
        "a log (10" + "-30: the gate printed it)",
        "- 2026-11" + "-04: a run on Docker Desktop",
        "| 10" + "-31: x |",
        "10" + "-29:",
        '"""11' + '-02 #2222: dev-y approved twice"""',
        '"""10' + "-31 02:2xZ: GitHub answered 502",
        "# 11" + "-06 #NNNN fu4: the brief missed it",
    ],
    # a person's answer as the reason, or a decision with its date, the user's or the operator's
    privacy.P_ATTRIB: [
        "the server (the user" + "'s answer: other teams may ask)",
        "never a node (the operator" + "'s decision, 11-05)",
        "Stopping (the operator" + "'s request, 2026-11-05): pause",
        "a card (user, 2026-11" + "-03)",
    ],
    privacy.P_INCIDENT: [
        "closed INC" + "-2026-11-04-9 at 05:40Z",
        '"id": "INC' + '-2026-10-29-5"',
    ],
}
# Lines like the narratives above that are facts, data or ranges: none fails.
CLEAN_DATED = (
    "at 10:31Z",
    "since 2026-10-31T12:00:00Z",
    "a range 10-31",
    "lines 10-20: the header",
    "rows = {'2026-11-04': 3}",
    "only on the operator's instruction",
    "it is the user's call",
    "on the operator's request",
    "INC-1 opened",
    "a fixture INC-2026-01-01-1",
    "INC-<date>-<n>",
    "1.10-12: no date",
    "every 10-12 minutes, 3-5: a ratio",
    # a review table's row keys, the post format's own: a reviewer and the date of their item
    "| Our 11-02 N3: what | OPEN |",
    "| Ours (11-01) N1: `answer` returns 41 |",
    "dave (11-02): keep z.py",
)
CLEAN = (
    "Invented PR #1111 and #12345 link https://github.com/ROCm/example/pull/1234 and https://rocm.docs.amd.com/x, "
    "the admin dashboard http://127.0.0.1:18613/, a lesson L-526, finding N3, SHA-256 and UTF-8, "
    "the incident INC-2026-01-01-7, a <JIRA_PROJECT>-NNNN tag, /home/<user>/x and the team rocke-platform, "
    "gfx942 and gfx950, a CSS colour --muted:#898781, the release github.com/cli/cli/releases, "
    "a @dataclass, @pytest.mark.slow, jq's @tsv, @dev-a and @acme/team-p, rkmon@localhost, 1.3x faster, a 30 s "
    "timeout, the model GLM-5.3-Flash, | https://cli.github.com | a table cell |."
)


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.d.name)
        self.live = fake_live(self.root)
        self.deny = self.root / "deny.txt"
        self.values = privacy.sources(*self.live, environ=FAKE_ENV, deny=self.deny)[0]

    def tearDown(self):
        self.d.cleanup()

    def scanned(self, name: str, text: str):
        """The check, with the fake live sources, over a folder that holds one file, `name`, with `text`: (its report,
        whether it passed, the (line, label) of each match)."""
        folder = self.root / "files"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        (folder / name).write_text(text, encoding="utf-8")
        report, ok, found = privacy.run(
            [folder], self.deny, *self.live, environ=FAKE_ENV
        )
        self.assertIn(row("PASS", privacy.S_FILES, "1 file(s) read"), report)
        return report, ok, [(i, lab) for _f, i, lab in found]

    def check_file(self, name: str, text: str):
        """`scanned`: (its report, whether it passed, the labels it found)."""
        report, ok, found = self.scanned(name, text)
        return report, ok, {lab for _i, lab in found}

    def check_line(self, line: str):
        return self.check_file("file.md", line + "\n")

    def test_every_planted_kind_fails_with_its_label(self):
        for label, line in PLANTS.items():
            with self.subTest(label=label):
                text, ok, found = self.check_line(line)
                self.assertIn(label, found, line)
                self.assertFalse(ok)
                self.assertIn(f"  FAIL  {label}:", text)

    def test_more_forms_fail(self):
        for label, lines in MORE.items():
            for line in lines:
                with self.subTest(line=line):
                    self.assertIn(label, self.check_line(line)[2], line)

    def test_a_clean_file_passes(self):
        text, ok, found = self.check_line(CLEAN)
        self.assertEqual(found, set())
        self.assertTrue(ok, text)
        self.assertIn("Result: PASS (0 match(es) in 0 file(s) and 0 template(s))", text)

    def test_a_host_with_a_placeholder_is_the_host_its_fixed_part_names(self):
        """A URL whose host holds a <placeholder> names nothing when, its placeholders and its port taken out, nothing
        or a top-level domain alone is left, or when the placeholder is the site's label in an internal product's
        domain; any other is the host its fixed part names, and is checked as that host is.
        """
        for url in (
            "https" + "://<site>" + DOT + ATL,
            "https" + "://<site>" + DOT + ATL + "/wiki/x",
            "https" + "://<host>",
            "http" + "://<host>:<port>/x",
            "http" + "://<host address>:11434",
            "https" + "://<company>" + DOT + "com",
            "https" + "://<user>" + AT + "<host>/x",
            "https" + "://<site>" + DOT + "example" + DOT + "com/x",
            "http" + "://127.0.0.1:<port>/",
            "https" + "://<x>.rocm.docs.amd.com/x",
            # a JSON text's escaped quotes around it
            '\\"https' + "://<site>" + DOT + ATL + '\\"',
        ):
            with self.subTest(url=url):
                self.assertEqual(self.check_line(f"see {url} here")[2], set(), url)
        for url in (
            SITE_URL,
            "https" + "://<a>" + DOT + "<b>" + DOT + "corp-example" + DOT + "net",
            "https" + "://<site>-x" + DOT + ATL,
            "https" + "://docs" + DOT + "<x>" + DOT + "com",
            "https" + "://<x>" + DOT + "github" + DOT + "com/acme/widgets",
            "http" + "://<host>" + DOT + "corp-example" + DOT + "net:<port>/x",
        ):
            with self.subTest(url=url):
                self.assertIn(privacy.P_URL, self.check_line(f"see {url} here")[2], url)

    def test_a_json_file_s_strings_are_read_decoded_too(self):
        """A .json file is read with its strings decoded too: a URL with its slashes escaped
        (`https:\\/\\/<host>\\/page`) is found as the plain one is, on its line, a string over lines included; a line
        that shows a match in both readings counts it as often as the one that shows it more.
        """
        plain = "https" + "://wiki" + DOT + "corp-example" + DOT + "net/page"
        escaped = plain.replace(SLASH, "\\" + SLASH)
        for url in (plain, escaped):
            with self.subTest(url=url):
                report, ok, found = self.scanned(
                    "data.json", '{\n  "page": "' + url + '"\n}\n'
                )
                self.assertEqual(found, [(2, privacy.P_URL)])
                self.assertFalse(ok)
        both = '{"a": "' + plain + '", "b": "' + escaped + '"}\n'
        self.assertEqual(self.scanned("data.json", both)[2], [(1, privacy.P_URL)] * 2)
        over_lines = '{\n  "a": "x\\nsee ' + escaped + '"\n}\n'
        self.assertEqual(self.scanned("data.json", over_lines)[2], [(2, privacy.P_URL)])

    def test_facts_data_and_ranges_are_no_dated_narrative_attribution_or_incident(self):
        for line in CLEAN_DATED:
            with self.subTest(line=line):
                self.assertEqual(self.check_line(line)[2], set(), line)

    def scan_file(self, name: str, text: str):
        return self.check_file(name, text)[2]

    def test_the_live_states_pr_numbers_fail_as_bare_numbers_in_code_and_tests(self):
        self.assertIn(PRN, self.values[privacy.L_PRNUMS])
        self.assertNotIn(
            "186" + "13", self.values[privacy.L_PRNUMS]
        )  # a port: allowed whatever the state holds
        for text in (
            f"refs('cat /w/tmp/v{PRN}fu4/notes.md')\n",
            f"N = {PRN}\n",
            f"wt = 'wt{PRN}m'\n",
            f"def test_{PRN}_stays():\n",
        ):
            with self.subTest(text=text):
                self.assertIn(privacy.L_PRNUMS, self.scan_file("t.py", text))
        # an extensionless Python entry point is code too
        self.assertIn(
            privacy.L_PRNUMS,
            self.scan_file("tool", f"#!/usr/bin/env python3\nN = {PRN}\n"),
        )
        # not code: a document, data, a shell script; nor a number the state does not know, a longer number, a hex
        # digest, a port, or #N (the PR pattern's, counted once there)
        self.assertNotIn(privacy.L_PRNUMS, self.scan_file("notes.md", f"N = {PRN}\n"))
        self.assertNotIn(
            privacy.L_PRNUMS, self.scan_file("tool", f"#!/bin/sh\nN={PRN}\n")
        )
        for text in (
            "N = 143" + "22\n",
            f"N = 9{PRN}\n",
            f"sha = 'ab{PRN}cd'\n",
            "PORT = 18613\n",
            f"x = '#{PRN}'\n",
        ):
            with self.subTest(text=text):
                self.assertNotIn(privacy.L_PRNUMS, self.scan_file("t.py", text))

    def test_the_admin_dashboards_public_port_is_no_pr_number(self):
        """The deploy schema's default for the admin dashboard's public port is a port, never a PR number, whatever
        the state has."""
        port = str(registry.default("admin_dashboard.public_port"))
        self.assertIn(port, privacy.safe_numbers())
        st = {"prs": {port: {"n": int(port)}, PRN: {"n": int(PRN)}}}
        self.assertEqual(privacy.state_pr_numbers(st), {PRN})

    def test_the_pr_numbers_seeded_in_the_deny_list_keep_their_scope(self):
        deny = self.root / "seeded.txt"
        privacy.seed_deny(deny, self.values)
        self.assertIn("pr:" + PRN, privacy.deny_values(deny))
        # No live source: the deny list alone.
        alone = privacy.sources(deny=deny, environ={})[0]
        self.assertIn(PRN, alone[privacy.L_PRNUMS])
        self.assertNotIn("pr:" + PRN, alone.get(privacy.L_DENY, set()))
        # The check with that deny list alone finds the number in code; in a document it is not looked for.
        folder = self.root / "seeded"
        folder.mkdir()
        for name in ("t.py", "notes.md"):
            (folder / name).write_text(f"N = {PRN}\n", encoding="utf-8")
        _text, ok, found = privacy.run([folder], deny, environ={})
        self.assertFalse(ok)
        self.assertEqual(
            {(pathlib.Path(f).name, lab) for f, _i, lab in found},
            {("t.py", privacy.L_PRNUMS)},
        )

    def test_a_file_that_only_mentions_a_settings_header_passes(self):
        # The README shows the header in an example, and the code names it: neither is a settings file.
        found = self.check_line(
            "A settings file starts with this line:\n\n    rkmon" + "_settings: 1\n"
        )[2]
        self.assertNotIn(privacy.P_SETTINGS, found)
        for text in (
            'HEADER = "rkmon' + '_settings"',
            'KINDS = ("deploy' + '_settings", "build' + '_settings")\nkind = KINDS[0]',
            "# Settings\n\nAn example:\n\n```yaml\nschema_version: 1\nkind: deploy"
            + "_settings\n```",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.check_line(text)[2], set())

    def test_the_report_never_holds_a_value(self):
        text, ok, _found = self.check_line("\n".join(PLANTS.values()))
        self.assertFalse(ok)
        for vals in self.values.values():
            for v in vals:
                self.assertNotIn(v, text)
        for line in PLANTS.values():
            self.assertNotIn(line, text)
        # a match's fix runs the check again with --where, which names each match's place, never its value
        self.assertIn(str(privacy.WHERE_FIX), fixes(text))

    def test_seeding_the_deny_list_writes_no_value_anywhere_else(self):
        deny = self.root / "secrets" / "privacy_deny.txt"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = privacy.main(
                [
                    "--config",
                    str(self.root / "config.json"),
                    "--state",
                    str(self.root / "state.json"),
                    "--seed-deny",
                    str(deny),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(deny.stat().st_mode & 0o777, 0o600)
        listed = privacy.deny_values(deny)
        self.assertIn("op" + "-login-x", listed)
        self.assertIn("reviewer" + "-x2", listed)
        # One value per line: no `VALUE => TEXT` or `VALUE -> TEXT` line.
        lines = [
            x
            for x in deny.read_text(encoding="utf-8").splitlines()
            if not x.startswith("#")
        ]
        self.assertEqual(sorted(lines), sorted(listed))
        for line in lines:
            self.assertNotIn(privacy.WITH_TEXT, line)
            self.assertNotIn(privacy.NOT_LOOKED_FOR, line)
        # The site values too, the secrets' files with the home folder spelt out; no path written from the home folder.
        cfg = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.assertLessEqual(privacy.site_values(cfg), listed)
        self.assertIn(os.path.expanduser("~/.config/rkmon/hook-x"), listed)
        self.assertEqual([v for v in listed if v.startswith("~")], [])
        # The deny list's own path is printed; it may hold this machine's user name (a TMPDIR under the home folder).
        printed = out.getvalue().replace(str(deny), "<the deny list>")
        for v in listed:
            self.assertNotIn(v, printed)
        again = privacy.seed_deny(
            deny, privacy.sources(self.root / "config.json", environ={})[0]
        )
        self.assertEqual(again, 0)  # nothing new: each value once

    def test_a_workspace_s_values_seeded_in_the_deny_list_are_found_with_it_alone(self):
        """The package build checks with the deny list alone: seeding from a Claude workspace writes its JIRA and
        wiki values and the people its skill names into the list, so that check finds them.
        """
        deny = self.root / "seeded.txt"
        with contextlib.redirect_stdout(io.StringIO()):
            code = privacy.main(
                ["--workspace", str(self.live[2]), "--seed-deny", str(deny)]
            )
        self.assertEqual(code, 0)
        for label in (privacy.L_WORKSPACE, privacy.L_PEOPLE):
            with self.subTest(label=label):
                folder = self.root / "files"
                shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir()
                (folder / "notes.md").write_text(PLANTS[label] + "\n", encoding="utf-8")
                _text, ok, found = privacy.run([folder], deny, environ={})
                self.assertFalse(ok)
                self.assertEqual({lab for _f, _i, lab in found}, {privacy.L_DENY})

    def test_the_checks_command_line_fails_on_a_finding(self):
        """`python3 -m rkmon_lib.privacy` exits 1 when it finds something and 0 on a clean tree: the package build and
        the sync stop on its exit code."""
        f = self.root / "pkg" / "file.md"
        f.parent.mkdir(exist_ok=True)
        for text, code in ((PLANTS[privacy.L_ME], 1), (CLEAN, 0)):
            with self.subTest(code=code):
                f.write_text(text + "\n", encoding="utf-8")
                p = subprocess.run(
                    privacy.command(
                        "--config",
                        self.root / "config.json",
                        "--state",
                        self.root / "state.json",
                        "--deny",
                        self.root / "deny.txt",
                        f.parent,
                    ),
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                )
                self.assertEqual(p.returncode, code, p.stdout + p.stderr)

    def test_a_configuration_with_the_older_me_key_is_looked_for(self):
        """A config.json that still says `me` (from before the rename) keeps its login among the values the check
        looks for."""
        login = "old" + "-login-x"
        old = self.root / "old.json"
        old.write_text(json.dumps({"me": login}), encoding="utf-8")
        v = privacy.sources(old, environ={})[0]
        self.assertIn(login, v[privacy.L_ME])
        f = self.root / "files" / "file.md"
        f.parent.mkdir()
        f.write_text(f"posted as {login} today\n", encoding="utf-8")
        _text, _ok, found = privacy.run([f.parent], self.deny, old, environ={})
        self.assertIn(privacy.L_ME, {lab for _f, _i, lab in found})

    def test_the_live_values_are_read_from_each_source(self):
        v = self.values
        self.assertEqual(
            len(v[privacy.L_LOGINS]), 4
        )  # author, reviewer, requested reviewer, code owner; no team
        self.assertIn("Jane Roe", v[privacy.L_CFG])  # a person named in a config note
        # project, epic, the site's host, space key and id; not the component and labels (descriptive words), nor a
        # bare site name such as "docs"
        self.assertEqual(len(v[privacy.L_WORKSPACE]), 5)
        self.assertEqual(privacy.company_domains(v), {CORP})

    def test_a_cloud_sites_name_alone_is_looked_for(self):
        """A JIRA or wiki address on a cloud site (`https://<site>.atlassian.net`) gives the check the site's host and
        its name alone, as the atlassian CLI names the site (`--site NAME`): a line with either fails.
        """
        site = "zq" + "cloud-x"
        (self.live[2] / ".claude" / "project_config.md").write_text(
            f"## JIRA\n| `base_url` | `https://{site}.atlassian.net` |\n",
            encoding="utf-8",
        )
        for line in (f"see {site}.atlassian.net", f"atlassian --site {site}"):
            with self.subTest(line=line):
                text, ok, found = self.check_line(line)
                self.assertIn(privacy.L_WORKSPACE, found, text)
                self.assertFalse(ok)


class TestBots(unittest.TestCase):
    def test_githubs_own_bots_are_no_logins(self):
        # The follow-up scan names Copilot's reviewer to count its threads: a bot, not a person.
        st = {
            "prs": {
                "1": {
                    "author": "dev-a",
                    "reviews": [
                        {"login": "copilot-pull-request-reviewer"},
                        {"login": "github-actions"},
                    ],
                }
            }
        }
        self.assertEqual(privacy.state_logins(st), {"dev-a"})


class TestDenyList(unittest.TestCase):
    """The deny list: values, PR numbers and company domains, a line that starts with `#` a comment; the two other
    kinds of line a list may hold; and its seeding."""

    def test_its_forms(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "deny.txt"
            login = "op" + "-login-q"
            f.write_text(
                "# a comment\nplain-x\n"
                + login
                + "\npr:"
                + PRN
                + "\ndomain:"
                + CORP
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                privacy.deny_values(f),
                {"plain-x", login, "pr:" + PRN, "domain:" + CORP},
            )
            v = privacy.sources(deny=f, environ={})[0]
            self.assertEqual(v[privacy.L_DENY], {"plain-x", login})
            self.assertIn(PRN, v[privacy.L_PRNUMS])
            self.assertEqual(privacy.company_domains(v), {CORP})
            g = pathlib.Path(d) / "files" / "notes.md"
            g.parent.mkdir()
            g.write_text(
                "host api7." + CORP + " and docs here; by " + login + "\n",
                encoding="utf-8",
            )
            _text, _ok, found = privacy.run([g.parent], f, environ={})
            self.assertEqual(
                sorted({lab for _f, _i, lab in found}),
                sorted({privacy.P_HOST, privacy.L_DENY}),
            )

    def test_the_two_other_kinds_of_line_a_list_may_hold(self):
        """`VALUE => TEXT` is read as VALUE, which the check looks for and the seeding does not add again; `VALUE ->
        TEXT` gives nothing to look for."""
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "deny.txt"
            login = "op" + "-login-q"
            f.write_text(
                login + " => <user_login>\nmodel-x:1 -> <WORKER_MODEL>\n",
                encoding="utf-8",
            )
            self.assertEqual(privacy.deny_values(f), {login})
            files = pathlib.Path(d) / "files"
            files.mkdir()
            (files / "a.md").write_text("by " + login + "\n", encoding="utf-8")
            (files / "b.md").write_text("with model-x:1\n", encoding="utf-8")
            _text, _ok, found = privacy.run([files], f, environ={})
            self.assertEqual(
                {(pathlib.Path(x).name, lab) for x, _i, lab in found},
                {("a.md", privacy.L_DENY)},
            )
            self.assertEqual(privacy.seed_deny(f, {privacy.L_ME: {login}}), 0)

    def test_the_seeding_adds_no_path_written_from_the_home_folder(self):
        """A path written from the home folder (`~/...`) leads to nobody: the seeding adds the other values, not it."""
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "deny.txt"
            n = privacy.seed_deny(
                f,
                {privacy.L_PATHS: {"~/mon-q/state", "/srv/mon-q"}},
                site={"~/mon-q", "/srv/out-q"},
            )
            self.assertEqual(n, 2)
            self.assertEqual(privacy.deny_values(f), {"/srv/mon-q", "/srv/out-q"})

    def test_the_seeding_ends_the_list_s_last_line_before_it_adds(self):
        """Each value added is a line of its own, after a last line written without a line end."""
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "deny.txt"
            f.write_text("plain-x", encoding="utf-8")
            self.assertEqual(privacy.seed_deny(f, {privacy.L_CFG: {"value-y"}}), 1)
            self.assertEqual(f.read_text(encoding="utf-8"), "plain-x\nvalue-y\n")

    def test_the_check_requires_it(self):
        """`--deny FILE` is the one way to name the deny list: a check run without it is a usage error (exit 2)."""
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "file.md"
            f.write_text("a clean line\n", encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as e:
                privacy.main([str(f)])
            self.assertEqual(e.exception.code, 2)
            self.assertIn("--deny FILE is required", err.getvalue())

    def test_the_command_line_names_it_in_one_form(self):
        """`--deny FILE` or `--deny=FILE`: no option is abbreviated (privacy.parser, which the sync reads the check's
        arguments with too), so `--den FILE` is a usage error."""
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "files" / "file.md"
            f.parent.mkdir()
            f.write_text("a clean line\n", encoding="utf-8")
            deny = pathlib.Path(d) / "deny.txt"
            deny.write_text("denied" + "-value-x\n", encoding="utf-8")
            for argv in (["--deny", str(deny)], ["--deny=" + str(deny)]):
                with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(privacy.main([*argv, str(f.parent)]), 0)
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as e:
                privacy.main(["--den", str(deny), str(f.parent)])
            self.assertEqual(e.exception.code, 2)
            self.assertIn("unrecognized arguments: --den", err.getvalue())


def row(status: str, name: str, found: str = "") -> str:
    """The start of a report's row (checks.render): its status, its name and what was found."""
    return f"  {status:<6}{name}: {found}"


def fixes(text: str) -> list:
    """Each fix a report shows (checks.render: `fix: ` at column 8, its further lines at column 13), on one line."""
    return [
        re.sub(r"\n {13}", " ", m.group(1))
        for m in re.finditer(r"(?m)^ {8}fix: (.*(?:\n {13}\S.*)*)", text)
    ]


class TestSiteValues(unittest.TestCase):
    """The values the seeding adds to a live configuration's: the other forms of its folders and files, and the team
    dashboard's address."""

    def test_the_forms_a_configuration_s_values_lack(self):
        home = SLASH + "home" + SLASH + "user-q"
        users = "Users"
        win = "/mnt/c/" + users + SLASH + "win-q"
        url = "https" + "://web7." + CORP + "/q/"
        cfg = {
            "paths": {
                "root": home + "/mon-q",
                "repo": home + "/src-q",
                "dashboard": win + "/out-q/dash.html",
                "secrets_dir": "~/.config/sec-q",
            },
            "teams": {"webhook_file": "~/.config/sec-q/hook"},
            "admin_dashboard": {"access_token_file": "~/.config/sec-q/access-q"},
            "worker": {"model": "model-q:1"},
            "publish": {"html": {"url": url}},
        }
        with mock.patch.dict(os.environ, {"HOME": home}):
            got = privacy.site_values(cfg, pathlib.Path(win + "/ws-q"), home=home)
        for v in (
            home + "/mon-q/monitor/rk",
            home + "/mon-q/monitor",
            home + "/src-q",
            home + "/mon-q",
            win + "/out-q",
            "C:\\" + users + "\\win-q\\out-q",
            "C:/" + users + "/win-q/out-q",
            "/c/" + users + "/win-q/out-q",
            win + "/ws-q",
            "C:\\" + users + "\\win-q\\ws-q",
            win,
            "C:\\" + users + "\\win-q",
            home + "/.config/sec-q/hook",
            home + "/.config/sec-q/access-q",
            home + "/.config/sec-q",
            home,
            url,
            url.rstrip("/"),
        ):
            with self.subTest(v=v):
                self.assertIn(v, got)
        # A path written from the home folder and the reviewer model's name lead to nobody.
        self.assertEqual([v for v in got if v.startswith("~")], [])
        self.assertNotIn("model-q:1", got)

    def test_the_check_and_the_agentic_files_import_neither_the_other(self):
        """The seeding is the privacy check's own: privacy.py imports nothing of agentic.py, nor agentic.py of
        privacy.py."""
        for name, other in (("privacy", "agentic"), ("agentic", "privacy")):
            tree = ast.parse(
                (ROOT / "rkmon_lib" / f"{name}.py").read_text(encoding="utf-8")
            )
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported |= {a.name for a in node.names}
                    imported.add((node.module or "").rsplit(".", 1)[-1])
                elif isinstance(node, ast.Import):
                    imported |= {a.name.rsplit(".", 1)[-1] for a in node.names}
            with self.subTest(name=name):
                self.assertNotIn(other, imported)


class TestNotChecked(unittest.TestCase):
    """The check cannot pass while checking nothing: a source named that cannot be read or gives no value, and a path
    named that does not exist or holds no file, are NOT CHECKED, a FAIL row each; a kind of identifying value with no
    values is not reported as checked. The rows name no path and no value."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.d.name)
        fake_live(self.root)
        self.deny = self.root / "deny.txt"
        self.pkg = self.root / "pkg"
        self.pkg.mkdir()
        (self.pkg / "file.md").write_text(CLEAN + "\n", encoding="utf-8")

    def tearDown(self):
        self.d.cleanup()

    def git_folder(self, name: str, email: str) -> pathlib.Path:
        """A git folder of its own whose identity is `name` and `email`."""
        g = pathlib.Path(tempfile.mkdtemp(dir=self.root))
        subprocess.run(["git", "init", "-q", str(g)], check=True, capture_output=True)
        for key, value in (("user.name", name), ("user.email", email)):
            subprocess.run(
                ["git", "-C", str(g), "config", key, value],
                check=True,
                capture_output=True,
            )
        return g

    def test_every_source_read_and_a_clean_file_pass(self):
        git = self.git_folder("Gale " + "Gitter", "gale" + AT + CORP)
        named = (
            self.root / "config.json",
            self.root / "state.json",
            self.root / "ws",
            [git],
        )
        text, ok, found = privacy.run([self.pkg], self.deny, *named, environ={})
        self.assertTrue(ok, text)
        self.assertEqual(found, [])
        for name in (
            privacy.S_CONFIG,
            privacy.S_STATE,
            privacy.S_WORKSPACE,
            privacy.S_GIT,
            privacy.S_DENY,
        ):
            self.assertRegex(text, re.escape(row("PASS", name)) + r"\d+ value\(s\)")
        self.assertIn(row("PASS", privacy.S_FILES, "1 file(s) read"), text)
        self.assertIn("Result: PASS (0 match(es) in 0 file(s) and 0 template(s))", text)
        self.assertNotIn(str(self.root), text)
        machine = privacy.sources(environ={})[0]
        values = privacy.sources(*named, environ={}, deny=self.deny)[0]
        for label, vals in values.items():
            for v in vals - machine.get(label, set()):
                self.assertNotIn(v, text)

    def test_a_deny_list_that_cannot_be_read_or_gives_no_value_fails(self):
        comments = self.root / "comments.txt"
        comments.write_text("# a comment, and no value\n\n", encoding="utf-8")
        latin = self.root / "latin.txt"
        latin.write_bytes(b"caf\xe9-x\n")  # not UTF-8
        for deny, why in (
            (self.root / "missing.txt", privacy.NOT_READ),
            (self.root, privacy.NOT_READ),  # a folder
            (latin, privacy.NOT_READ),
            ("", privacy.NOT_READ),
            (None, privacy.NOT_READ),
            (comments, privacy.NO_VALUE),
        ):
            with self.subTest(deny=str(deny)):
                text, ok, found = privacy.run([self.pkg], deny, environ={})
                self.assertFalse(ok, text)
                self.assertEqual(found, [])
                self.assertIn(row("FAIL", privacy.S_DENY, why), text)
                self.assertIn(
                    "Result: FAIL (0 match(es) in 0 file(s) and 0 template(s); 1 NOT CHECKED)",
                    text,
                )
        text, ok, _found = privacy.run([self.pkg], self.deny, environ={})
        self.assertTrue(ok, text)

    def test_each_named_source_that_cannot_be_read_or_gives_no_value_fails(self):
        missing = self.root / "missing"
        not_json = self.root / "not.json"
        not_json.write_text("{not json", encoding="utf-8")
        a_list = self.root / "list.json"
        a_list.write_text("[1, 2]", encoding="utf-8")
        nothing = self.root / "nothing.json"
        nothing.write_text("{}", encoding="utf-8")
        empty = self.root / "empty"
        empty.mkdir()
        own = self.git_folder(
            "rkmon", "rkmon" + AT + "localhost"
        )  # the toolkit's own identity: no value
        for option, value, name, why in (
            ("config", missing, privacy.S_CONFIG, privacy.NOT_READ),
            ("config", not_json, privacy.S_CONFIG, privacy.NOT_READ),
            ("config", a_list, privacy.S_CONFIG, privacy.NOT_READ),
            ("config", nothing, privacy.S_CONFIG, privacy.NO_VALUE),
            ("state", missing, privacy.S_STATE, privacy.NOT_READ),
            ("state", not_json, privacy.S_STATE, privacy.NOT_READ),
            ("state", nothing, privacy.S_STATE, privacy.NO_VALUE),
            ("workspace", missing, privacy.S_WORKSPACE, privacy.NOT_READ),
            ("workspace", empty, privacy.S_WORKSPACE, privacy.NO_VALUE),
            ("git_dirs", [missing], privacy.S_GIT, privacy.NOT_READ),
            ("git_dirs", [own, missing], privacy.S_GIT, privacy.NOT_READ),
            ("git_dirs", [own], privacy.S_GIT, privacy.NO_VALUE),
        ):
            with self.subTest(option=option, value=str(value)):
                text, ok, _found = privacy.run(
                    [self.pkg], self.deny, environ={}, **{option: value}
                )
                self.assertFalse(ok, text)
                self.assertIn(row("FAIL", name, why), text)
                self.assertIn("; 1 NOT CHECKED)", text)
                self.assertNotIn(str(self.root), text)

    def test_each_fix_runs_the_check_again_with_a_stand_in_for_each_path(self):
        """In the report, each fix for a source, the files, the package folder or a match runs the check again: its own
        command line in the package folder (privacy.AGAIN), with a stand-in for each path. Each parses as the check's
        arguments, each option once, the stand-in in the option of the source it names, and shows no path. A deny list
        that gives no value is seeded from a monitor's files (privacy.SEED_FIX)."""

        def arguments(fix):
            """The arguments of the fix's one command, which is the check's own command line."""
            self.assertIsInstance(fix, checks.Steps)
            commands = [p for p in fix.parts if isinstance(p, checks.Command)]
            self.assertEqual(len(commands), 1, fix)
            argv = commands[0].argv
            n = len(privacy.COMMAND_LINE)
            self.assertEqual(argv[:n], privacy.COMMAND_LINE)
            options = [w for w in argv if w.startswith("--")]
            self.assertEqual(len(options), len(set(options)), fix)
            self.assertNotIn(str(self.root), str(fix))
            return privacy.parser().parse_args(argv[n:])

        missing = self.root / "missing"
        nothing = self.root / "nothing.json"
        nothing.write_text("{}", encoding="utf-8")
        empty = self.root / "empty"
        empty.mkdir()
        comments = self.root / "comments.txt"
        comments.write_text("# a comment, and no value\n", encoding="utf-8")
        own = self.git_folder("rkmon", "rkmon" + AT + "localhost")  # no value
        not_read = privacy.sources(
            missing, missing, missing, [self.pkg, missing], environ={}, deny=missing
        )[1]
        no_value = privacy.sources(
            nothing, nothing, empty, [own], environ={}, deny=comments
        )[1]
        named = (
            privacy.S_CONFIG,
            privacy.S_STATE,
            privacy.S_WORKSPACE,
            privacy.S_GIT,
            privacy.S_DENY,
        )
        self.assertEqual(
            [(r.name, r.found) for r in not_read + no_value],
            [(name, privacy.NOT_READ) for name in named]
            + [(name, privacy.NO_VALUE) for name in named],
        )
        option = {
            privacy.S_CONFIG: ("config", privacy.FILE),
            privacy.S_STATE: ("state", privacy.FILE),
            privacy.S_WORKSPACE: ("workspace", privacy.FOLDER),
            privacy.S_GIT: ("git", privacy.FOLDER),
            privacy.S_DENY: ("deny", privacy.FILE),
        }
        for r in not_read + no_value[:-1]:
            with self.subTest(row=r.name, found=r.found):
                a = arguments(r.fix)
                dest, stand_in = option[r.name]
                self.assertEqual(a.paths, ["."])
                self.assertEqual(
                    getattr(a, dest), [stand_in] if dest == "git" else stand_in
                )
                if dest != "deny":
                    self.assertEqual(a.deny, privacy.DENY_LIST)
                self.assertIn(f", {stand_in} being ", str(r.fix))
        self.assertIs(no_value[-1].fix, privacy.SEED_FIX)
        a = arguments(privacy.SEED_FIX)
        self.assertEqual(
            (a.paths, a.seed_deny, a.config, a.state, a.deny),
            ([], privacy.FILE, "CONFIG", "STATE", None),
        )
        # the files, the parts of a package folder and its templates (here another folder's code), and the matches
        report = [
            privacy._files_row([(missing, [])]),
            *privacy._part_rows(self.pkg, []),
            privacy._scan_templates(self.pkg, {})[1],
        ]
        self.assertEqual({r.status for r in report}, {checks.Status.FAIL})
        for fix, where in [(r.fix, False) for r in report] + [
            (privacy.WHERE_FIX, True),
            (privacy.SETTINGS_FIX, True),
            (privacy._template_fix(), True),
        ]:
            with self.subTest(fix=str(fix)):
                a = arguments(fix)
                self.assertEqual(
                    (a.paths, a.deny, a.where), (["."], privacy.DENY_LIST, where)
                )

    def test_a_sources_fix_names_another_in_the_command_you_ran(self):
        """With the command you ran (`typed`, its program first), a source's fix is that command with the stand-in in
        the place of the path you named for that source (private_files.again), for the git folders the first that is
        no folder; when the git folders give no value, `--git FOLDER` follows the command.
        """
        names = {
            key: str(self.root / ("no-" + key))
            for key in ("config", "state", "workspace", "git", "deny")
        }
        own = str(self.git_folder("rkmon", "rkmon" + AT + "localhost"))  # no value
        typed = (
            *privacy.COMMAND_LINE,
            "--config",
            names["config"],
            "--state",
            names["state"],
            "--workspace",
            names["workspace"],
            "--git",
            own,
            "--git",
            names["git"],
            "--deny=" + names["deny"],
            ".",
        )
        rows = privacy.sources(
            names["config"],
            names["state"],
            names["workspace"],
            [own, names["git"]],
            environ={},
            deny=names["deny"],
            typed=typed,
        )[1]
        self.assertEqual([r.found for r in rows], [privacy.NOT_READ] * 5)
        for r, key, stand_in in zip(
            rows,
            ("config", "state", "workspace", "git", "deny"),
            (
                privacy.FILE,
                privacy.FILE,
                privacy.FOLDER,
                privacy.FOLDER,
                privacy.FILE,
            ),
        ):
            with self.subTest(row=r.name):
                self.assertEqual(
                    r.fix.parts[0].argv,
                    tuple(w.replace(names[key], stand_in) for w in typed),
                )
                self.assertIn(f", {stand_in} being ", str(r.fix))
        typed = (*privacy.COMMAND_LINE, "--git", own, "--seed-deny", "seeded.txt")
        r = privacy.sources(git_dirs=[own], environ={}, typed=typed)[1][0]
        self.assertEqual(r.found, privacy.NO_VALUE)
        self.assertEqual(r.fix.parts[0].argv, (*typed, "--git", privacy.FOLDER))

    def test_a_path_that_does_not_exist_or_holds_no_file_fails(self):
        """Each path named is counted on its own: one that holds no file fails the check beside one that holds files."""
        empty = self.root / "empty"
        empty.mkdir()
        (empty / "__pycache__").mkdir()  # a cache is no file to check
        (empty / "__pycache__" / "x.pyc").write_bytes(b"\0")
        missing = self.root / "missing"
        for paths, why in (
            ([missing], "1 of the 1 path(s) named do not exist"),
            ([self.pkg, missing], "1 of the 2 path(s) named do not exist"),
            ([empty], "1 of the 1 path(s) named hold no file"),
            ([self.pkg, empty], "1 of the 2 path(s) named hold no file"),
            (
                [missing, self.pkg, empty],
                "of the 3 path(s) named, 1 do not exist and 1 hold no file",
            ),
            ([], "no file or folder is named"),
        ):
            with self.subTest(paths=[p.name for p in paths]):
                text, ok, _found = privacy.run(paths, self.deny, environ={})
                self.assertFalse(ok, text)
                self.assertIn(row("FAIL", privacy.S_FILES, "NOT CHECKED: " + why), text)
                self.assertIn("; 1 NOT CHECKED)", text)

    def test_a_kind_with_no_values_is_not_reported_as_checked(self):
        text, ok, _found = privacy.run([self.pkg], self.deny, environ={})
        self.assertTrue(ok, text)
        self.assertIn(
            row("PASS", privacy.L_DENY, "1 value(s) checked, 0 match(es)"), text
        )
        for label in (
            privacy.L_ME,
            privacy.L_LOGINS,
            privacy.L_CFG,
            privacy.L_PEOPLE,
            privacy.L_PATHS,
            privacy.L_WORKSPACE,
            privacy.L_PRNUMS,
        ):
            self.assertNotIn(label, text)

    def test_the_command_line_fails_on_a_deny_list_it_cannot_read(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = privacy.main(
                ["--deny", str(self.root / "missing.txt"), str(self.pkg)]
            )
        self.assertEqual(code, 1)
        self.assertIn(row("FAIL", privacy.S_DENY, privacy.NOT_READ), out.getvalue())

    def test_seeding_stops_on_a_source_it_cannot_read(self):
        """The seeding stops on a source it cannot read, and the row's fix is the command you ran, FILE in the place
        of that source."""
        seeded = self.root / "seeded.txt"
        state = self.root / "state.json"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = privacy.main(
                [
                    "--config",
                    str(self.root / "missing.json"),
                    "--state",
                    str(state),
                    "--seed-deny",
                    str(seeded),
                ]
            )
        self.assertEqual(code, 1)
        self.assertFalse(seeded.exists())
        self.assertIn(row("FAIL", privacy.S_CONFIG, privacy.NOT_READ), out.getvalue())
        self.assertIn("nothing was added to the deny list", out.getvalue())
        again = checks.Command(
            (
                *privacy.COMMAND_LINE,
                "--config",
                privacy.FILE,
                "--state",
                state,
                "--seed-deny",
                seeded,
            )
        )
        self.assertEqual(
            fixes(out.getvalue()),
            [
                f"{again}, {privacy.FILE} being the monitor's config.json (a JSON object), or leave --config out"
            ],
        )

    def test_seeding_names_no_file_to_check(self):
        seeded = self.root / "seeded.txt"
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as e:
            privacy.main(["--seed-deny", str(seeded), str(self.pkg)])
        self.assertEqual(e.exception.code, 2)
        self.assertIn("checks nothing", err.getvalue())
        self.assertFalse(seeded.exists())


T_BUILD = settings_templates.Template(file_types.BUILD)
T_FULL = settings_templates.Template(file_types.DEPLOY, "full")
T_EVERY = settings_templates.Template(file_types.DEPLOY, every_key=True)
# A build settings template, each private key's value to fill in: rocm.package_url (line 6) and
# atlassian_cli.package_url (line 8), both download locations, private in the build schema.
BUILD_TEMPLATE = (
    "{first}\n"
    "kind: build_settings\n"
    "schema_version: 2\n"
    "rocm:\n"
    "  version: 10.1.0\n"
    "  package_url: {rocm}\n"
    "atlassian_cli:\n"
    "  package_url: {cli}  # the CLI's download URL\n"
)
# A deploy settings template with one model binding, an Ollama model: its model_id (line 7) is private in each of the
# three forms an entry takes.
BINDING_TEMPLATE = (
    "# A deploy settings file\n"
    "kind: deploy_settings\n"
    "schema_version: 2\n"
    "model_bindings:\n"
    "  local:\n"
    '    provider: "ollama"\n'
    "    model_id: {model}\n"
)


class Fixed:
    """A template with a fixed text, in the place of the package's own: its label, file type and command are
    TEMPLATE's. An exception as the text is raised when the text is asked for: a template that cannot be rendered.
    """

    def __init__(self, template, text):
        self.label, self.file_type, self.body = template.label, template.file_type, text
        self.command = template.command

    def text(self) -> str:
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class TestSettingsTemplates(unittest.TestCase):
    """A package folder's settings templates are rendered and checked as files are, every line of them, and for
    absolute paths and private values (a private key's value is blank, <...> placeholders only or its default); a
    package folder's parts are each read. A part with no file, templates the check does not run as the folder's own
    code, a template that cannot be rendered and one in which no key is private are NOT CHECKED. The check runs as the
    package folder's own code here: paths.PACKAGE is the folder.
    """

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.d.name)
        self.pkg = self.root / "pkg"
        for rel, text in (
            ("rkmon.py", ""),
            ("rkmon_lib/__init__.py", ""),
            ("README.md", "The package's guide.\n"),
            ("schema/x.schema.json", "{}\n"),
            ("tests/test_x.py", "X = 1\n"),
            ("agentic/skills/x/SKILL.md", "A skill.\n"),
        ):
            (self.pkg / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.pkg / rel).write_text(text, encoding="utf-8")
        own = mock.patch.object(paths, "PACKAGE", self.pkg.resolve())
        own.start()
        self.addCleanup(own.stop)
        self.deny = self.root / "deny.txt"
        self.deny.write_text("denied" + "-value-x\n", encoding="utf-8")

    def tearDown(self):
        self.d.cleanup()

    def build(self, rocm='""', cli="<URL>", first="# A build settings file") -> dict:
        """The templates: a build template whose private keys hold `rocm` and `cli`, its first line `first`."""
        return {T_BUILD: BUILD_TEMPLATE.format(first=first, rocm=rocm, cli=cli)}

    def check(self, texts=None):
        """The check over the package folder, with its own templates or `texts` (settings_templates.Template ->
        text) in their place: (its report, whether it passed, the findings in its templates).
        """
        with contextlib.ExitStack() as stack:
            if texts is not None:
                fixed = tuple(Fixed(t, text) for t, text in texts.items())
                stack.enter_context(
                    mock.patch.object(settings_templates, "templates", lambda: fixed)
                )
            text, ok, found = privacy.run([self.pkg], self.deny, environ={})
        return text, ok, [x for x in found if x[0].startswith("template ")]

    def test_every_template_is_rendered_and_the_packages_own_pass(self):
        self.assertEqual(
            [t.label for t in settings_templates.templates()],
            [
                "build",
                "deploy full",
                "deploy review-only",
                "deploy merge-queue",
                "deploy every key",
            ],
        )
        text, ok, found = self.check()
        self.assertEqual(found, [])
        self.assertTrue(ok, text)
        # The row counts the private keys checked in each template (checks.render wraps it: one line here).
        flat = re.sub(r"\n {8}", " ", text)
        start = row(
            "PASS", privacy.S_TEMPLATES, "5 rendered (private keys checked: build "
        )
        self.assertRegex(
            flat,
            re.escape(start)
            + r"[1-9]\d*, deploy full [1-9]\d*, deploy review-only [1-9]\d*, deploy merge-queue [1-9]\d*, "
            + r"deploy every key [1-9]\d*\)",
        )

    def test_the_check_reads_each_template_whole_as_the_package_writes_it(self):
        """What the check reads of each template is its whole text, line 1 too, as settings_file writes it: the text
        docker/rkmon prints (tests/test_settings_templates.py)."""
        read = []
        real = privacy._scan_text

        def reading(text, *args, **kw):
            read.append(text)
            return real(text, *args, **kw)

        with mock.patch.object(privacy, "_scan_text", reading):
            text, ok, _found = privacy.run([self.pkg], self.deny, environ={})
        self.assertTrue(ok, text)
        for t in settings_templates.templates():
            with self.subTest(template=t.label):
                written = settings_file.template(t.file_type, t.profile, t.every_key)
                self.assertIn(written, read)

    def test_a_blank_placeholders_only_or_the_default_passes(self):
        """A private key's template value passes when it is blank, <...> placeholders only (no letter or digit beside
        them), or the key's own default (blank, for both keys of the build schema here); any other value fails.
        """
        for value in (
            '""',
            "null",
            "[]",
            "{}",
            "<URL>",
            "<the CLI URL>",
            '"<org>/<repo>"',
            '"<model>:<tag>"',
            '["<label>", "<another>"]',
        ):
            with self.subTest(value=value):
                text, ok, found = self.check(self.build(rocm=value, cli=value))
                self.assertEqual(found, [])
                self.assertTrue(ok, text)
        for value in (
            "x",
            '"<URL> x"',
            '"<org>/repo"',
            '"<a>1<b>"',
            '"/"',
            '["<label>", "real"]',
            "0",
        ):
            with self.subTest(value=value):
                text, ok, found = self.check(self.build(rocm=value))
                self.assertEqual(found, [("template build", 6, privacy.P_PRIVATE)])
                self.assertFalse(ok, text)

    def test_a_private_key_of_a_map_s_entries_is_checked_whatever_the_entry_s_name(
        self,
    ):
        """A private key of a map's entries (model_bindings.NAME.model_id, in each of the forms an entry takes) is
        checked whatever the entry's name: the forms' placeholders pass; a model's own name fails.
        """
        for model in ('"<model>:<tag>"', '"<model>"'):
            with self.subTest(model=model):
                text, ok, found = self.check(
                    {T_FULL: BINDING_TEMPLATE.format(model=model)}
                )
                self.assertEqual(found, [])
                self.assertTrue(ok, text)
                self.assertIn("(private keys checked: deploy full 1)", text)
        text, ok, found = self.check(
            {T_FULL: BINDING_TEMPLATE.format(model='"llama3.1:8b"')}
        )
        self.assertEqual(found, [("template deploy full", 7, privacy.P_PRIVATE)])
        self.assertFalse(ok, text)

    def test_a_private_key_a_template_shows_commented_out_is_checked_as_the_others(
        self,
    ):
        """The every-key text shows blocks commented out: a section its chooser does not choose (the team
        dashboard's serve) and a map's example entries (model_bindings'). The check reads their keys under the keys
        above each block, so that it checks every private key of the schema there, and a private value fails in a key
        commented out as in one that is not."""
        text = T_EVERY.text()
        lines = text.splitlines()
        read = {k for _i, k, _value in privacy._key_lines(text)}
        for pattern, _nodes in privacy._private_keys(file_types.DEPLOY):
            with self.subTest(key=pattern.pattern):
                self.assertTrue(any(pattern.fullmatch(k) for k in read))
        # A URL there fails as a URL in a file does, too.
        both, alone = (privacy.P_URL, privacy.P_PRIVATE), (privacy.P_PRIVATE,)
        for old, value, labels in (
            ('    base_url: ""', SITE_URL, both),  # integrations.claude_code's
            ('#     base_url: "<url>"', SITE_URL, both),  # an example binding's
            ('      #   address: "127.0.0.1"', SITE_HOST, alone),  # serve's
        ):
            with self.subTest(line=old):
                self.assertEqual(lines.count(old), 1)
                at = lines.index(old)
                planted = list(lines)
                planted[at] = old.split(": ")[0] + f': "{value}"'
                report, ok, found = self.check({T_EVERY: "\n".join(planted) + "\n"})
                self.assertFalse(ok, report)
                self.assertEqual(
                    sorted(found),
                    sorted(("template deploy every key", at + 1, x) for x in labels),
                )

    def test_a_template_in_which_no_key_is_private_is_not_checked(self):
        """A template with no private key gives the private values nothing to check: NOT CHECKED, a FAIL, naming the
        template; its fix classes the keys in the schemas' folder."""
        text, ok, found = self.check(
            {
                T_BUILD: "# A build settings file\nkind: build_settings\nschema_version: 2\n"
            }
        )
        self.assertEqual(found, [])
        self.assertFalse(ok, text)
        self.assertIn(
            row(
                "FAIL",
                privacy.S_TEMPLATES,
                "NOT CHECKED: no private key in template(s) build (private keys checked: build 0)",
            ),
            text,
        )
        folder = file_types.SCHEMAS.relative_to(file_types.PACKAGE).as_posix()
        self.assertIn(
            f'in its schema ({folder}/), give each key whose value can name a person, a site or a machine "x-class": '
            '"private", then python3 -B -m rkmon_lib.privacy . --deny DENY_LIST',
            fixes(text),
        )

    def test_the_package_s_own_build_template_fails_on_a_private_value(self):
        """The package's own build template, with its schema changed in memory: a private key whose template value
        (x-template) is no placeholder fails; the key's own default, a generic value, shown in its place, passes.
        """
        url = "https" + "://downloads" + DOT + "widgets" + DOT + "example/rocm.run"
        build = copy.deepcopy(file_types.BUILD.load())
        rocm = build["properties"]["rocm"]["properties"]["package_url"]
        self.assertEqual(rocm.get("x-class"), "private")
        rocm["x-template"] = url

        def built_from(schema):
            """The build file type, its schema SCHEMA."""
            b = file_types.BUILD
            return mock.patch.object(
                file_types,
                "BUILD",
                settings_file.FileType(
                    b.kind, b.version, schema, template_command=b.template_command
                ),
            )

        with built_from(build):
            text, ok, found = self.check()
        self.assertFalse(ok, text)
        self.assertEqual(
            [(t, label) for t, _i, label in found],
            [("template build", privacy.P_PRIVATE)],
        )
        del rocm["x-template"]
        rocm["default"] = "rkmon" + "-downloads"
        with built_from(build):
            text, ok, found = self.check()
        self.assertEqual(found, [])
        self.assertTrue(ok, text)

    def test_a_templates_fix_names_the_folder_of_the_schemas_it_is_made_from(self):
        """A private value in a template is fixed in its schema, in the folder the templates are made from
        (file_types.SCHEMAS), where the profiles' values are too (x-profiles)."""
        folder = file_types.SCHEMAS.relative_to(file_types.PACKAGE).as_posix()
        fix = str(privacy._template_fix())
        self.assertIn(
            f"in its schema ({folder}/), or in its profile there (x-profiles)", fix
        )

    def test_a_rendered_template_with_a_private_value_fails(self):
        url = "https" + "://downloads" + DOT + "widgets" + DOT + "example/rocm.run"
        text, ok, found = self.check(self.build(rocm=url))
        self.assertEqual(found, [("template build", 6, privacy.P_PRIVATE)])
        # The templates were rendered (their row passes, with the private keys checked): the finding is what fails
        # the check.
        self.assertIn(
            row(
                "PASS",
                privacy.S_TEMPLATES,
                "1 rendered (private keys checked: build 2)",
            ),
            text,
        )
        self.assertFalse(ok)
        self.assertIn(row("FAIL", privacy.P_PRIVATE, "1 match(es)"), text)
        found = self.check(self.build(cli="cli" + "-build-7"))[2]
        self.assertEqual(found, [("template build", 8, privacy.P_PRIVATE)])

    def test_a_url_in_a_template_takes_the_rule_a_url_in_a_file_does(self):
        """A template's lines take the URL rule a file's lines do (`_public_url`): a host whose placeholder is the
        site's label in an internal product's domain passes; one whose fixed part names a site of its own fails.
        """
        texts = self.build()
        texts[T_BUILD] += "# https" + "://<site>" + DOT + ATL + "/wiki\n"
        self.assertEqual(self.check(texts)[2], [])
        texts[T_BUILD] += "# " + SITE_URL + "\n"
        self.assertEqual(self.check(texts)[2], [("template build", 10, privacy.P_URL)])

    def test_an_absolute_path_fails_on_any_line(self):
        for line in (
            "# kept in " + SLASH + "srv" + SLASH + "builds",
            "# or C:" + "\\builds\\x",
            "# or c:" + SLASH + "builds",
            "# or \\\\filer" + "\\share\\x",
        ):
            with self.subTest(line=line):
                texts = self.build()
                texts[T_BUILD] += line + "\n"
                found = self.check(texts)[2]
                self.assertEqual(found, [("template build", 9, privacy.P_ABSOLUTE)])
        # line 1 is read as any other: a schema line that names a path fails (a public address, or no line)
        found = self.check(
            self.build(first="# $schema=" + SLASH + "opt" + SLASH + "s.json")
        )[2]
        self.assertEqual(found, [("template build", 1, privacy.P_ABSOLUTE)])
        # a URL, a relative path and a folder under the home folder are no absolute path
        texts = self.build()
        texts[T_BUILD] += "# https://example.com/a/b, owner/name, a/b and ~/x\n"
        self.assertEqual(self.check(texts)[2], [])

    def test_the_templates_are_checked_only_as_the_package_folder_s_own_code(self):
        """The code that runs the check renders the templates: over a package folder whose code is another, they are
        NOT CHECKED, and the check's command line, run in the folder, renders them as the folder's own code. Here the
        folder's own build schema puts an absolute path in its template: neither way passes.
        """
        other = self.root / "other"
        shutil.copytree(self.pkg, other)
        shutil.rmtree(other / "rkmon_lib")
        shutil.copytree(
            ROOT / "rkmon_lib",
            other / "rkmon_lib",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copytree(ROOT / "schema", other / "schema", dirs_exist_ok=True)
        # The package's defaults, which its code reads (the deploy templates' comments name some).
        shutil.copy(ROOT / "config.defaults.json", other / "config.defaults.json")
        # The copies keep the modes of a read-only package folder (the image's): they are the test's to change.
        sandbox.writable(other)
        build = other / "schema" / "build_settings.schema.json"
        text = build.read_text(encoding="utf-8")
        first = '"Extra root certificates ('
        self.assertEqual(text.count(first), 1)
        kept = SLASH + "srv" + SLASH + "builds"
        build.write_text(text.replace(first, first[:-1] + f"kept in {kept} ("), "utf-8")
        text, ok, found = privacy.run([other], self.deny, environ={})
        self.assertFalse(ok, text)
        self.assertEqual([x for x in found if x[0].startswith("template ")], [])
        self.assertIn(
            row(
                "FAIL",
                privacy.S_TEMPLATES,
                "NOT CHECKED: the check runs as another folder's code",
            ),
            text,
        )
        p = subprocess.run(
            privacy.command("--deny", self.deny, "--where", "."),
            cwd=other,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            timeout=300,
        )
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn(row("PASS", privacy.S_TEMPLATES, "5 rendered"), p.stdout)
        self.assertIn(row("FAIL", privacy.P_ABSOLUTE, "1 match(es)"), p.stdout)
        self.assertRegex(
            p.stdout, r"(?m)^template build:\d+: " + re.escape(privacy.P_ABSOLUTE) + "$"
        )

    def test_a_template_that_cannot_be_rendered_or_read_back_is_not_checked(self):
        """Each template is rendered and read back on its own: one that cannot be is NOT CHECKED, and the row names it
        alone, with the command that prints it, which shows the error; the others are checked, and counted.
        """
        real = settings_file.template

        def broken(file_type, profile=None, every_key=False):
            if profile == "full" or every_key:
                raise RuntimeError("no template")
            return real(file_type, profile, every_key)

        with mock.patch.object(settings_file, "template", broken):
            text, ok, found = privacy.run([self.pkg], self.deny, environ={})
        self.assertEqual((ok, found), (False, []))
        self.assertIn(
            row(
                "FAIL",
                privacy.S_TEMPLATES,
                "NOT CHECKED: template(s) deploy full, deploy every key cannot be rendered and read back "
                "(RuntimeError); 3 rendered (private keys checked: build ",
            ),
            re.sub(r"\n {8}", " ", text),
        )
        self.assertIn(
            "rkmon template deploy --profile full - prints deploy full, rkmon template section prints deploy every "
            "key: fix the error each shows",
            fixes(text),
        )
        # The others are checked: the build template's private value fails.
        url = "https" + "://downloads" + DOT + "widgets" + DOT + "example/rocm.run"
        texts = self.build(rocm=url)
        texts[T_FULL] = RuntimeError("no template")
        text, ok, found = self.check(texts)
        self.assertEqual(found, [("template build", 6, privacy.P_PRIVATE)])
        self.assertIn(
            row(
                "FAIL",
                privacy.S_TEMPLATES,
                "NOT CHECKED: template(s) deploy full cannot be rendered and read back (RuntimeError); 1 rendered "
                "(private keys checked: build 2)",
            ),
            re.sub(r"\n {8}", " ", text),
        )
        self.assertIn(
            "rkmon template deploy --profile full - prints it: fix the error it shows",
            fixes(text),
        )
        # A settings file takes no YAML anchor: this one cannot be read back.
        text, ok, found = self.check({T_BUILD: "kind: &a build_settings\n"})
        self.assertEqual((ok, found), (False, []))
        self.assertIn(
            row(
                "FAIL",
                privacy.S_TEMPLATES,
                "NOT CHECKED: template(s) build cannot be rendered and read back (",
            ),
            text,
        )
        self.assertIn(
            "rkmon template build - prints it: fix the error it shows", fixes(text)
        )

    def test_templates_that_cannot_be_listed_are_none_of_them_checked(self):
        """When the templates cannot be listed (the schemas name the profiles), none is checked: the fix names the
        template commands, which show the error."""

        def unlisted():
            raise ValueError("no schema")

        with mock.patch.object(settings_templates, "templates", unlisted):
            text, ok, found = privacy.run([self.pkg], self.deny, environ={})
        self.assertEqual((ok, found), (False, []))
        self.assertIn(
            row(
                "FAIL",
                privacy.S_TEMPLATES,
                "NOT CHECKED: they cannot be listed (ValueError)",
            ),
            text,
        )
        self.assertIn(
            "rkmon template build - prints build, rkmon template deploy - prints deploy: fix the error each shows",
            fixes(text),
        )

    def test_the_check_over_a_package_folder_reads_its_parts_and_templates(self):
        text, ok, _found = privacy.run([self.pkg], self.deny, environ={})
        self.assertTrue(ok, text)
        for name in (
            privacy.S_DOCS,
            privacy.S_SCHEMAS,
            privacy.S_SKILLS,
            privacy.S_TESTS,
        ):
            self.assertRegex(text, re.escape(row("PASS", name)) + r"\d+ file\(s\) read")
        self.assertIn(row("PASS", privacy.S_TEMPLATES, "5 rendered"), text)
        for label in privacy.TEMPLATE_LABELS:
            self.assertIn(row("PASS", label, "0 match(es)"), text)
        # The result counts the matches, and the files and the templates they are in, apart.
        (self.pkg / "notes.md").write_text(
            "denied" + "-value-x\n" + "denied" + "-value-x\n", encoding="utf-8"
        )
        text, ok, found = self.check(self.build(rocm="x"))
        self.assertFalse(ok)
        self.assertIn(row("FAIL", privacy.P_PRIVATE, "1 match(es)"), text)
        self.assertEqual(found, [("template build", 6, privacy.P_PRIVATE)])
        self.assertIn("Result: FAIL (3 match(es) in 1 file(s) and 1 template(s))", text)
        # a folder that is no package folder: no part, no template
        text, ok, _found = privacy.run([self.pkg / "tests"], self.deny, environ={})
        self.assertTrue(ok, text)
        self.assertNotIn(privacy.S_TEMPLATES, text)
        self.assertNotIn(privacy.P_PRIVATE, text)

    def test_a_skill_under_claude_skills_is_one_of_the_skills(self):
        """A skill in the package folder's .claude/skills/ (one Claude Code reads there) counts in the skills part, not
        in the docs: a package folder whose only skill is there has its skills read."""
        shutil.rmtree(self.pkg / "agentic")
        skill = self.pkg / ".claude" / "skills" / "setup" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("A skill.\n", encoding="utf-8")
        text, ok, _found = privacy.run([self.pkg], self.deny, environ={})
        self.assertTrue(ok, text)
        self.assertIn(row("PASS", privacy.S_SKILLS, "1 file(s) read"), text)
        self.assertIn(row("PASS", privacy.S_DOCS, "1 file(s) read"), text)

    def test_a_part_with_no_file_is_not_checked(self):
        for name, part in (
            (privacy.S_DOCS, "README.md"),
            (privacy.S_SCHEMAS, "schema"),
            (privacy.S_SKILLS, "agentic"),
            (privacy.S_TESTS, "tests"),
        ):
            with self.subTest(part=part):
                pkg = self.root / ("without-" + part)
                shutil.copytree(self.pkg, pkg)
                p = pkg / part
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                with mock.patch.object(paths, "PACKAGE", pkg.resolve()):
                    text, ok, _found = privacy.run([pkg], self.deny, environ={})
                self.assertFalse(ok, text)
                self.assertIn(row("FAIL", name, "NOT CHECKED"), text)
                self.assertIn(row("PASS", privacy.S_TEMPLATES, "5 rendered"), text)


class TestThePackageItself(unittest.TestCase):
    """The hard check: the package's own files, as published, hold nothing that leads to a person, a machine or a
    site: no login, name or email, no home folder with a user name, no internal host, link or webhook, no ticket key, no
    token. The check runs over a copy of them as the copy's own code (its command line, in its folder), with an
    invented deny list: it looks for this machine's names and the generic patterns, so this file names no value. The
    deny list adds the concrete values in the check that the package build (tools/make_package.py) and the sync
    (tools/sync_from_live.py, which tools/publish_package.py runs) run before their tests.
    """

    @classmethod
    def files(cls) -> list:
        """The package's files, as published: no git data, caches, backups or binaries."""
        skip = {".git", "__pycache__", ".pytest_cache", "state", "dist"}
        return [
            f
            for f in sorted(ROOT.rglob("*"))
            if f.is_file()
            and not (set(f.relative_to(ROOT).parts) & skip)
            and ".bak-" not in f.name
            and f.suffix not in (".pyc", ".gz", ".png")
        ]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        copy = pathlib.Path(cls.tmp.name) / "pkg"
        cls.n = 0
        for f in cls.files():
            dest = copy / f.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            cls.n += 1
        deny = pathlib.Path(cls.tmp.name) / "deny.txt"
        deny.write_text("invented" + "-deny-value-q\n", encoding="utf-8")
        cls.check = subprocess.run(
            privacy.command("--deny", deny, "--where", "."),
            cwd=copy,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            timeout=600,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_file_of_the_package_leads_to_a_person_or_a_site(self):
        out = self.check.stdout
        self.assertGreater(self.n, 100)
        self.assertIn(row("PASS", privacy.S_FILES, f"{self.n} file(s) read"), out)
        where = [x for x in out.splitlines() if re.match(r"\S[^:]*:\d+: ", x)]
        self.assertEqual(
            where, [], "\n".join(where[:40])
        )  # labels only, never the value
        self.assertEqual(self.check.returncode, 0, out + self.check.stderr)
        self.assertIn("Result: PASS", out)

    def test_the_check_and_the_deny_list_have_one_name_each(self):
        """ "The privacy check" and "the deny list", in code, messages, docs and tests: "gate" is the merge queue's
        word. The pattern is split, so this file does not match it."""
        other = re.compile(
            r"privacy[\s_-]*"
            + r"gate|(?:private|local|maintainer's)[\s_-]+"
            + r"(?:deny[\s_-]+)?list",
            re.IGNORECASE,
        )
        where = []
        for f in self.files():
            text = f.read_text(encoding="utf-8", errors="replace")
            where += [
                f"{f.relative_to(ROOT)}:{text.count(chr(10), 0, m.start()) + 1}"
                for m in other.finditer(text)
            ]
        self.assertEqual(
            where, [], "another name for the privacy check or the deny list"
        )

    def test_its_parts_are_read_and_its_settings_templates_pass(self):
        out = self.check.stdout
        for name, _inside in privacy.PARTS:
            self.assertRegex(out, re.escape(row("PASS", name)) + r"\d+ file\(s\) read")
        self.assertIn(row("PASS", privacy.S_TEMPLATES, "5 rendered"), out)
        for label in privacy.TEMPLATE_LABELS:
            self.assertIn(row("PASS", label, "0 match(es)"), out)


if __name__ == "__main__":
    unittest.main()
