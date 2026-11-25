# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tickets waiting for you (the dashboard card): a proposed ticket for a problem
found along the way waits on the admin dashboard's card; Approve files it through the atlassian CLI, records the key in
the ledger and shows it in the row; Decline drops it and journals it; one action per proposed ticket; nothing is
filed while ticket creation (or its out_of_scope_issue trigger) is off or the monitor is stopped, nor a proposed ticket
rk ticket-draft's check refuses, which one of its title replaces; in prototype and in standby the card's Approve keeps
the ticket as an unsent item, and `rk ticket-file` typed is refused; the session proposes and never decides (its guard
refuses rk ticket-file); the card is the admin dashboard's only; what a person reads names a proposed ticket, never a
draft."""

import base64
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import (
    features,
    orchestrator,
    report,
    state_store,
    ticket_fields,
    tickets,
    webdash,
    webserve,
)  # noqa: E402
from rkmon_lib.store import Store, boot_id, empty_state  # noqa: E402

FAKES = ROOT / "tests" / "fakes"
PROJ = (
    "TK" + "T"
)  # an invented project key, built at run time (the privacy check looks for <KEY>-<n>)
ACCOUNT = "0123456789abcdef01234567"

DRAFT = """---
title: The nightly lane skips the fp8 cases on every runner
type: Bug
priority: P2
assignee: dev-a
summary: tests/fp8 skip on every runner since the image update; nothing runs them
labels: ci, nightly
pr: 77
source: https://ci.example.com/runs/1
---
What happens: every fp8 case reports SKIPPED on develop.

Why it matters: a regression in them would land unseen.
"""


def make_cfg(tmp: str, **kw) -> dict:
    c = {
        "repo": "ROCm/example",
        "operator": {"github_login": "op"},
        "features": {
            "ticket_creation": {
                "enabled": True,
                "mode": "live",
                "triggers": {"review_follow_up": True, "out_of_scope_issue": True},
            }
        },
        "jira": {
            "project": PROJ,
            "url": "https://jira.example.com",
            "issue_type": "",
            "labels": ["monitor"],
            "component": "Platform",
            "priority_map": {"P2": "Major"},
            "extra_fields": {
                "customfield_1": {"value": "Moderate"},
                "customfield_2": "plain",
            },
            "accounts": {"dev-a": "dev-a@example.com"},
        },
        "atlassian": {"site": "example-site"},
        "paths": {"working_dir": tmp},
        "admin_dashboard": {
            "port": 18613,
            "access_token_file": f"{tmp}/admin_access_token",
        },
    }
    c.update(kw)
    return c


def cli_ready(test: unittest.TestCase, tmp: str) -> None:
    """The atlassian CLI here and logged in, as the ticket features need it (atlassian.missing): a stand-in on PATH and
    a credentials file in <SECRETS_DIR>/atlassian (<tmp>/.config, the working folder's default).
    """
    b = pathlib.Path(tmp) / "atlassian-bin"
    b.mkdir(exist_ok=True)
    (b / "atlassian").write_text("#!/bin/sh\nexit 0\n")
    (b / "atlassian").chmod(0o755)
    creds = pathlib.Path(tmp) / ".config" / "atlassian" / "credentials"
    creds.parent.mkdir(parents=True, exist_ok=True)
    creds.write_text("{}")
    p = mock.patch.dict(os.environ, {"PATH": f"{b}:{os.environ['PATH']}"})
    p.start()
    test.addCleanup(p.stop)


class FakeCli:
    """The atlassian CLI as tickets calls it: (exit code, output) per command, recorded."""

    def __init__(self, fail: str = ""):
        self.calls, self.n, self.fail, self.made = [], 0, fail, []

    def __call__(self, cmd):
        self.calls.append(cmd)
        if (
            "search" in cmd
        ):  # the issues made, newest first; "nosearch": the search fails
            if "nosearch" in self.fail:
                return 1, "Error: Atlassian API error: HTTP 503"
            return 0, json.dumps(
                {
                    "issues": [
                        {"key": k, "fields": {"summary": s}}
                        for k, s in reversed(self.made)
                    ]
                }
            )
        if "create" in cmd:
            if self.fail == "create":
                return 1, "Error: 400 Bad Request: Field 'priority' is not valid"
            self.n += 1
            self.made.append((f"{PROJ}-{self.n}", cmd[cmd.index("--summary") + 1]))
            if self.fail.startswith("lost"):  # JIRA took it; the answer was lost
                return 126, "Command timed out after 90 seconds"
            return 0, "A new version is available.\n" + json.dumps(
                {"key": f"{PROJ}-{self.n}", "id": "1"}
            )
        if cmd[-3:-1] == ["user", "get"]:
            return 0, json.dumps({"accountId": ACCOUNT})
        if "update" in cmd:
            return (1, "Error: 403") if self.fail == "assign" else (0, "{}")
        return 2, "unknown"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(pathlib.Path(self.tmp.name) / "state")
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        self.store.save(st)
        self.cfg = make_cfg(self.tmp.name)
        cli_ready(self, self.tmp.name)
        self.draft = tickets.parse(DRAFT)

    def add(self, draft=None, cfg=None) -> tuple:
        with self.store.locked(wait=True):
            st = self.store.load()
            out = tickets.add(cfg or self.cfg, st, self.store, draft or self.draft)
            self.store.save(st)
        return out

    def item(self, tid: str) -> dict:
        return tickets.find(self.store.load(), tid)

    def without_a_priority(self, **more) -> str:
        """DRAFT, waiting as another deployment recorded it without a priority (adopt brings its state over), with
        MORE: its id."""
        _ok, _msg, tid = self.add()
        st = self.store.load()
        tickets.find(st, tid).update(priority="", **more)
        self.store.save(st)
        return tid

    def journal(self) -> str:
        return (
            self.store.journal_path.read_text()
            if self.store.journal_path.exists()
            else ""
        )

    def run_rk(
        self, *argv, cfg=None, stdin: str = "", env: dict | None = None
    ) -> tuple[int, str]:
        import rkmon

        out = []
        with mock.patch.object(
            rkmon, "load_config", lambda: dict(cfg or self.cfg)
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch.object(
            sys, "stdin", io.StringIO(stdin)
        ), mock.patch.dict(
            os.environ, env or {}
        ), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            rc = rkmon.main(list(argv))
        return rc, "\n".join(out)


class TestDrafts(Base):
    def test_a_draft_file_and_its_checks(self):
        d = self.draft
        self.assertEqual(
            (d["title"], d["type"], d["priority"], d["assignee"], d["labels"], d["pr"]),
            (
                "The nightly lane skips the fp8 cases on every runner",
                "Bug",
                "P2",
                "dev-a",
                ["ci", "nightly"],
                77,
            ),
        )
        self.assertTrue(d["description"].startswith("What happens: every fp8 case"))
        self.assertIn("\n\nWhy it matters", d["description"])  # its lines kept
        as_json = tickets.parse(
            json.dumps(
                {
                    "title": "x",
                    "priority": "P3",
                    "summary": "y",
                    "description": "z",
                    "assignee": "none",
                }
            )
        )
        self.assertEqual(
            (
                as_json["priority"],
                as_json["assignee"],
                as_json["labels"],
                as_json["pr"],
            ),
            ("P3", "", [], None),
        )
        for text, why in (
            ("title: x\n", "starts with a --- block"),
            (
                "---\ntitle: x\n---\n",
                "priority: required .*; summary: required .*; description: required",
            ),
            ("---\ntitle: x\nsummary: y\nowner: z\n---\nbody", "unknown keys owner"),
            (
                "---\ntitle: " + "t" * 201 + "\nsummary: y\n---\nbody",
                "title: at most 200 characters",
            ),
            ("---\ntitle: x\nsummary: y\nlabels: a b\n---\nbody", "labels: at most 10"),
            ("---\ntitle: x\nsummary: y\npr: soon\n---\nbody", "pr: a PR number"),
            ('{"title": 1', "not valid JSON"),
        ):
            with self.assertRaisesRegex(tickets.DraftError, why):
                tickets.parse(text)

    def test_a_draft_without_a_priority_is_refused(self):
        """Check 6 counts the priority among the fields every new ticket gets (ticket_fields.FILLED), so the filer
        sends one with each ticket: a draft without one, or with a blank one, is refused, in either form.
        """
        self.assertIn("priority", ticket_fields.FILLED)
        given = {"title": "x", "summary": "y", "description": "z"}
        for n, text in enumerate(
            (
                DRAFT.replace("priority: P2\n", ""),
                DRAFT.replace("priority: P2\n", "priority:   \n"),
                json.dumps(given),
                json.dumps(dict(given, priority=" ")),
            )
        ):
            with self.subTest(n=n), self.assertRaisesRegex(
                tickets.DraftError, r"^priority: required \(one line\)$"
            ):
                tickets.parse(text)

    def test_added_once_with_the_projects_defaults(self):
        ok, msg, tid = self.add()
        self.assertTrue(ok, msg)
        self.assertRegex(tid, r"^[0-9a-f]{8}$")
        self.assertTrue(msg.startswith(f"TICKET_DRAFTED {tid}: The nightly lane"))
        self.assertIn(
            "waits for op's OK on the admin dashboard's \"Tickets waiting for you\" card",
            msg,
        )
        it = self.item(tid)
        self.assertEqual(
            (it["status"], it["type"], it["labels"], it["component"]),
            ("waiting", "Bug", ["ci", "nightly"], "Platform"),
        )
        bare = tickets.parse(
            "---\ntitle: Another one\npriority: P3\nsummary: s\n---\nbody"
        )
        ok, _msg, tid2 = self.add(bare)
        self.assertEqual(
            (self.item(tid2)["type"], self.item(tid2)["labels"]), ("Task", ["monitor"])
        )  # the defaults
        # The fields filled from the configuration are those tickets.unfit leaves out.
        self.assertEqual(
            {k for k in tickets.FIELDS if not bare[k] and self.item(tid2)[k]},
            set(tickets.FROM_SETTINGS),
        )
        ok, msg, same = self.add(
            tickets.parse(DRAFT.replace("The nightly lane", "the  NIGHTLY lane"))
        )
        self.assertFalse(ok)
        self.assertEqual(
            (same, msg),
            (
                tid,
                f"proposed ticket {tid} has this title and waits already: nothing added",
            ),
        )
        self.assertRegex(
            self.journal(),
            rf"ticket +proposed {tid}: The nightly lane .* \(Bug, P2; by the monitor's",
        )

    def test_a_draft_of_its_title_takes_the_place_of_a_waiting_draft_that_cannot_be_filed(
        self,
    ):
        """A waiting proposed ticket that check refuses cannot be filed: one of its title takes its place, under its
        id, which is how such a proposed ticket is edited. A waiting one that can be filed keeps its title.
        """
        tid = self.without_a_priority()
        ok, msg, same = self.add()
        self.assertEqual((ok, same), (True, tid), msg)
        self.assertTrue(
            msg.startswith(
                f"TICKET_DRAFTED {tid}: The nightly lane skips the fp8 cases on every runner. It takes the place of "
                "the waiting proposed ticket of this title, which could not be filed: priority: required (one "
                "line). It "
                "waits for op's OK"
            ),
            msg,
        )
        self.assertEqual(
            [
                (it["id"], it["priority"], it["status"])
                for it in tickets.waiting(self.store.load())
            ],
            [(tid, "P2", "waiting")],
        )
        self.assertRegex(
            self.journal(),
            rf"ticket +proposed again {tid}: The nightly lane .* \(Bug, P2; by the monitor's",
        )
        self.assertEqual(
            self.add()[:2],
            (
                False,
                f"proposed ticket {tid} has this title and waits already: nothing added",
            ),
        )


class TestApprove(Base):
    def test_approve_files_it_records_the_key_and_assigns_it(self):
        _ok, _msg, tid = self.add()
        cli = FakeCli()
        ok, msg, extra = tickets.approve(self.cfg, self.store, tid, by="op", run=cli)
        key = f"{PROJ}-1"
        self.assertTrue(ok, msg)
        self.assertEqual(
            msg,
            f"Filed proposed ticket {tid} as {key}: The nightly lane skips the fp8 cases on every runner "
            f"(https://jira.example.com/browse/{key}).",
        )
        self.assertEqual(
            extra,
            {
                "key": key,
                "url": f"https://jira.example.com/browse/{key}",
                "row": f"filed as {key}",
            },
        )
        create = cli.calls[0]
        self.assertEqual(create[1:4], ["--site", "example-site", "--json"])
        self.assertEqual(create[4:9], ["jira", "issue", "create", "--project", PROJ])
        args = dict(zip(create[9::2], create[10::2]))
        self.assertEqual(
            (args["--summary"], args["--type"], args["--priority"], args["--labels"]),
            (
                "The nightly lane skips the fp8 cases on every runner",
                "Bug",
                "Major",
                "ci,nightly",
            ),
        )
        desc = json.loads(
            next(x for x in create if x.startswith("description="))[
                len("description=") :
            ]
        )
        self.assertEqual(
            desc["type"], "doc"
        )  # ADF: the CLI's --description would show Markdown marks as typed
        words = json.dumps(desc)
        self.assertIn("Found on PR #77: ", words)
        self.assertIn('"href": "https://github.com/ROCm/example/pull/77"', words)
        self.assertIn("approved by op on", words)
        self.assertIn('components=[{"name": "Platform"}]', create)
        cli2 = FakeCli()  # several components, comma-separated: each its own
        tickets.file_ticket(
            self.cfg,
            dict(self.item(tid), component="Platform, Platform CI"),
            "op",
            run=cli2,
        )
        self.assertIn(
            'components=[{"name": "Platform"}, {"name": "Platform CI"}]', cli2.calls[0]
        )
        self.assertIn('customfield_1={"value": "Moderate"}', create)
        self.assertIn("customfield_2=plain", create)
        self.assertEqual(cli.calls[1][4:], ["jira", "user", "get", "dev-a@example.com"])
        self.assertEqual(
            cli.calls[2][4:], ["jira", "issue", "update", key, "--assignee", ACCOUNT]
        )
        it = self.item(tid)
        self.assertEqual(
            (it["status"], it["key"], it["decided_by"]), ("filed", key, "op")
        )
        ledger = [r for r in self.store.ledger() if r["kind"] == "ticket"]
        self.assertEqual(
            [(r["pr"], r["head"], r["ref"]) for r in ledger], [(77, tid, key)]
        )
        self.assertRegex(
            self.journal(), rf"ticket +filed {tid} as {key}: The nightly lane"
        )
        # One action per proposed ticket.
        for again in (
            lambda: tickets.approve(self.cfg, self.store, tid, by="op", run=cli),
            lambda: (
                lambda st: tickets.decline(self.cfg, st, self.store, tid, by="op")
            )(self.store.load()),
        ):
            got = again()
            self.assertFalse(got[0])
            self.assertIn(
                f"proposed ticket {tid} is filed already as {key} (approved by op",
                got[1],
            )
            self.assertIn("one action per proposed ticket, nothing changed", got[1])
        self.assertEqual(len([c for c in cli.calls if "create" in c]), 1)

    def test_a_blank_required_field_is_left_out(self):
        """A required_fields value that is blank (null, "", [] or {}) sets nothing, as check 6 reads it: the filer
        sends no field for it, never the text "None"."""
        _ok, _msg, tid = self.add()
        extra = {
            "customfield_1": None,
            "customfield_2": "",
            "customfield_3": [],
            "customfield_4": {},
            "customfield_5": "plain",
        }
        cfg = dict(self.cfg, jira=dict(self.cfg["jira"], extra_fields=extra))
        cli = FakeCli()
        tickets.file_ticket(cfg, self.item(tid), "op", run=cli)
        self.assertEqual(
            [x for x in cli.calls[0] if x.startswith("customfield_")],
            ["customfield_5=plain"],
        )

    def test_nothing_is_filed_while_stopped_or_with_the_feature_off(self):
        _ok, _msg, tid = self.add()
        cli = FakeCli()
        for off in (
            dict(self.cfg, features={"ticket_creation": {"enabled": False}}),
            dict(
                self.cfg,
                features={
                    "ticket_creation": {
                        "enabled": True,
                        "triggers": {"out_of_scope_issue": False},
                    }
                },
            ),
        ):
            ok, msg, _ = tickets.approve(off, self.store, tid, by="op", run=cli)
            self.assertFalse(ok)
            self.assertIn("OFF: out-of-scope issue tickets is off", msg)
        self.assertEqual(tickets.status_text(self.cfg, self.item(tid)), "waiting")
        no_cli = dict(self.cfg, atlassian={"site": ""})
        self.assertIn(
            "NOT SET UP: out-of-scope issue tickets is on but needs",
            tickets.approve(no_cli, self.store, tid, by="op", run=cli)[1],
        )
        st = self.store.load()
        st["stopped"] = {"at": "2026-11-05T00:00:00Z", "how": "stop", "boot": boot_id()}
        self.store.save(st)
        self.assertIn(
            "nothing is filed until rk start; the proposed ticket waits",
            tickets.approve(self.cfg, self.store, tid, by="op", run=cli)[1],
        )
        self.assertEqual((cli.calls, self.item(tid)["status"]), ([], "waiting"))

    def test_a_failed_filing_leaves_it_waiting_with_the_reason(self):
        _ok, _msg, tid = self.add()
        ok, msg, extra = tickets.approve(
            self.cfg, self.store, tid, by="op", run=FakeCli(fail="create")
        )
        self.assertFalse(ok)
        self.assertIn(
            "was not filed: atlassian jira issue create failed (exit 1): Error: 400 Bad Request",
            msg,
        )
        self.assertEqual(self.item(tid)["status"], "waiting")
        self.assertIn("the last filing failed", extra["row"])
        self.assertEqual([r for r in self.store.ledger() if r["kind"] == "ticket"], [])
        self.assertRegex(
            self.journal(),
            rf"ticket +filing {tid} failed: atlassian jira issue create failed",
        )
        ok, msg, _ = tickets.approve(
            self.cfg, self.store, tid, by="op", run=FakeCli(fail="assign")
        )
        self.assertTrue(ok, msg)  # filed, but nobody assigned: the row says so
        self.assertIn("Not assigned to dev-a: Error: 403.", msg)

    def test_a_filing_cut_off_is_never_filed_twice(self):
        _ok, _msg, tid = self.add()
        st = self.store.load()
        tickets.find(st, tid).update(
            status="filing", filing_at="2026-11-05T00:00:00Z", decided_by="op"
        )
        self.store.save(st)
        self.store.ledger_add(
            "ticket", 77, tid, f"{PROJ}-9", note="x"
        )  # the write went through before the cut
        cli = FakeCli()
        self.assertIn(
            "is being filed since",
            tickets.approve(self.cfg, self.store, tid, by="op", run=cli)[1],
        )
        ok, msg, _ = tickets.approve(
            self.cfg, self.store, tid, by="op", again=True, run=cli
        )
        self.assertTrue(ok, msg)
        self.assertEqual((cli.calls, self.item(tid)["key"]), ([], f"{PROJ}-9"))
        self.assertIn("It was filed before (the ledger has it).", msg)

    def test_a_filing_whose_answer_was_lost_is_found_in_jira_not_filed_again(self):
        """A create that timed out after JIRA took it: JIRA is searched for the summary before it is called failed."""
        _ok, _msg, tid = self.add()
        cli = FakeCli(fail="lost")
        ok, msg, _ = tickets.approve(self.cfg, self.store, tid, by="op", run=cli)
        self.assertTrue(ok, msg)
        self.assertEqual(sum("create" in c for c in cli.calls), 1)
        self.assertEqual(self.item(tid)["key"], f"{PROJ}-1")
        self.assertEqual(
            [r["ref"] for r in self.store.ledger() if r["kind"] == "ticket"],
            [f"{PROJ}-1"],
        )

    def test_a_filing_jira_cannot_settle_stays_in_doubt_and_the_retry_searches_first(
        self,
    ):
        _ok, _msg, tid = self.add()
        cli = FakeCli(fail="lost-nosearch")
        ok, msg, _ = tickets.approve(self.cfg, self.store, tid, by="op", run=cli)
        self.assertFalse(ok)
        self.assertTrue(
            msg.startswith(f"proposed ticket {tid} may have been filed: "), msg
        )
        self.assertIn(f"rk ticket-file {tid} --again", msg)
        item = self.item(tid)
        self.assertEqual(
            item["status"], "filing"
        )  # never back to waiting: a second Approve could file it twice
        self.assertTrue(item.get("doubt_since"))
        self.assertIn(
            "is being filed since",
            tickets.approve(self.cfg, self.store, tid, by="op", run=cli)[1],
        )
        cli.fail = (
            ""  # JIRA answers again: the retry finds the issue the first try made
        )
        ok, msg, _ = tickets.approve(
            self.cfg, self.store, tid, by="op", again=True, run=cli
        )
        self.assertTrue(ok, msg)
        self.assertEqual(sum("create" in c for c in cli.calls), 1)
        self.assertEqual(self.item(tid)["key"], f"{PROJ}-1")
        self.assertIn("an earlier filing that was in doubt did go in", self.journal())

    def test_a_filing_in_doubt_that_check_refuses_is_searched_for_never_sent_again(
        self,
    ):
        """Its retry searches JIRA first; JIRA does not have it, and check refuses the proposed ticket: nothing is
        sent, and it waits, with check's words as its note, out of doubt: JIRA's search does not find it.
        """
        at = "2026-11-05T00:00:00Z"
        tid = self.without_a_priority(
            status="filing", filing_at=at, doubt_since=at, decided_by="op"
        )
        cli = FakeCli()
        ok, msg, extra = tickets.approve(
            self.cfg, self.store, tid, by="op", again=True, run=cli
        )
        self.assertFalse(ok)
        self.assertTrue(
            msg.startswith(
                f"proposed ticket {tid} was not filed: priority: required (one line). It waits: propose it "
                "again"
            ),
            msg,
        )
        self.assertEqual([c[4:7] for c in cli.calls], [["jira", "issue", "search"]])
        item = self.item(tid)
        self.assertEqual(
            (item["status"], item["note"], "doubt_since" in item, extra["row"]),
            (
                "waiting",
                "priority: required (one line)",
                False,
                "waiting; priority: required (one line)",
            ),
        )
        self.assertRegex(
            self.journal(), rf"ticket +filing {tid} refused: priority: required"
        )

    def test_decline_drops_it_and_journals_it(self):
        _ok, _msg, tid = self.add()
        st = self.store.load()
        ok, msg = tickets.decline(
            self.cfg, st, self.store, tid, by="op", why="known already"
        )
        self.store.save(st)
        self.assertEqual(
            (ok, msg),
            (
                True,
                f"Declined proposed ticket {tid} (The nightly lane skips the fp8 cases on every "
                "runner): dropped from the card, and journaled.",
            ),
        )
        st = self.store.load()
        self.assertEqual((tickets.shown(st), tickets.waiting(st)), ([], []))
        self.assertRegex(
            self.journal(),
            rf"ticket +declined {tid}: The nightly lane .* \(by op\): known already",
        )
        ok, msg, _ = (
            self.add()
        )  # the same title again, while the decline is recent: refused
        self.assertFalse(ok)
        self.assertIn(f"declined by op on", msg)
        self.assertIn(
            f"was declined by op",
            tickets.approve(self.cfg, self.store, tid, by="op", run=FakeCli())[1],
        )

    def test_what_a_person_reads_names_a_proposed_ticket_never_a_draft(self):
        """A ticket waiting for your approval is a proposed ticket, and "draft" is a pass's output. The
        messages, journal lines, rows, page and description of a proposed ticket's whole cycle say so; the command
        rk ticket-draft, its tag TICKET_DRAFTED and the page's route keep their names.
        """
        said = []
        for bad in ("{", "title: x\n", "---\ntitle x\n---\nbody"):
            with self.assertRaises(tickets.DraftError) as refused:
                tickets.parse(bad)
            said.append(str(refused.exception))
        tid = self.without_a_priority()
        cli = FakeCli()
        said.append(tickets.approve(self.cfg, self.store, tid, by="op", run=cli)[1])
        said.append(self.add()[1])  # it takes the place of the one check refuses
        said.append(self.add()[1])  # it waits already
        said.append(tickets.approve(self.cfg, self.store, tid, by="op", run=cli)[1])
        said.append(tickets.approve(self.cfg, self.store, tid, by="op", run=cli)[1])
        said.append(
            tickets.decline(self.cfg, self.store.load(), self.store, tid, by="op")[1]
        )
        said.append(self.add()[1])  # filed already
        _ok, _msg, other = self.add(
            tickets.parse("---\ntitle: Another\npriority: P3\nsummary: s\n---\nbody")
        )
        st = self.store.load()
        said.append(tickets.decline(self.cfg, st, self.store, other, by="op")[1])
        said.append(tickets.approve_refusal(self.cfg, st, other) or "")
        said.append(tickets.approve_refusal(self.cfg, st, "0000aaaa") or "")
        item = tickets.find(st, tid)
        said += tickets.lines(self.cfg, st)
        said += [
            tickets.page(self.cfg, item),
            tickets.description(self.cfg, item, "op"),
            tickets.unsent_text(self.cfg, item, "op"),
            self.journal(),
        ]
        self.assertTrue(all(said), said)
        text = "\n".join(said)
        for name in ("rk ticket-draft", "TICKET_DRAFTED", "/ticket-draft?id="):
            text = text.replace(name, "")
        self.assertNotIn("draft", text.lower(), text)


class TestCard(Base):
    def render(self, variant, cfg=None):
        cfg = cfg or self.cfg
        st = self.store.load()
        return webdash.render(
            cfg,
            st,
            self.store,
            [],
            [],
            report.board(cfg, st, self.store, [], []),
            variant,
        )

    def test_the_card_is_the_admin_dashboards_only(self):
        _ok, _msg, tid = self.add()
        self.add(
            tickets.parse(
                "---\ntitle: <script>x</script>\npriority: P3\nsummary: s & t\n---\nbody"
            )
        )
        local, published = self.render(webdash.LOCAL), self.render(webdash.PUBLISHED)
        card = local[local.index("<section id='tickets'") :]
        card = card[: card.index("</section>")]
        self.assertIn(
            "Tickets waiting for you <span class='c'>2 waiting · op approves or declines each</span>",
            card,
        )
        self.assertIn(
            "<td>The nightly lane skips the fp8 cases on every runner</td><td class='nw'>Bug</td>"
            "<td class='nw'>P2</td><td class='nw'>dev-a</td>",
            card,
        )
        self.assertIn(
            f"<a href='http://127.0.0.1:18613/ticket-draft?id={tid}' target='_blank' "
            "rel='noopener noreferrer'>full text</a>",
            card,
        )
        self.assertIn(
            f"<button type='button' class='tk-act' data-act='approve' data-id='{tid}'>Approve</button>",
            card,
        )
        self.assertIn(
            "<td class='nw' data-local-only hidden>", card
        )  # shown once the service answers
        self.assertIn("&lt;script&gt;x&lt;/script&gt;", card)
        self.assertNotIn("<script>x", local)
        self.assertIn("<a href='#tickets'>Tickets <span class='c'>2</span></a>", local)
        self.assertIn(
            "post(base+'/ticket-'+b.dataset.act,{id:b.dataset.id})", local
        )  # the browser re-sends the access token itself
        for absent in (
            "Tickets waiting for you",
            "class='tk-act'",
            "ticket-draft",
            "nightly lane",
            "#tickets",
        ):
            self.assertNotIn(absent, published)
        # Both boards say only how many wait (the Needs-you list is on the team dashboard too).
        self.assertIn(
            "2 proposed tickets wait for your OK: approve or decline on the admin dashboard&#x27;s Tickets card",
            published,
        )
        # Filed: the key, linked.
        tickets.approve(self.cfg, self.store, tid, by="op", run=FakeCli())
        local = self.render(webdash.LOCAL)
        self.assertIn(
            f"filed as <a href='https://jira.example.com/browse/{PROJ}-1' target='_blank' "
            f"rel='noopener noreferrer'>{PROJ}-1</a>",
            local,
        )
        self.assertNotIn(
            f"data-act='approve' data-id='{tid}'", local
        )  # one action per proposed ticket

    def test_in_prototype_the_card_says_approve_keeps_the_ticket(self):
        self.add()
        proto = dict(
            self.cfg,
            features={
                "ticket_creation": dict(
                    self.cfg["features"]["ticket_creation"], mode="prototype"
                )
            },
        )
        self.assertIn(
            f"Approve files nothing now, because {tickets.held_words(proto)}. It keeps the proposed ticket as an "
            "unsent item, listed under <a href='#not-sent'>Not sent</a>.",
            self.render(webdash.LOCAL, proto),
        )
        self.assertNotIn("Approve files nothing now", self.render(webdash.LOCAL))

    def test_the_card_names_proposed_tickets_in_live_prototype_and_standby(self):
        """A ticket waiting for your Approve is a proposed ticket, and "draft" is not its name. While ticket
        creation is not live (prototype, or the monitor in standby), the card says that Approve keeps the proposed
        ticket as an unsent item, listed under "Not sent", whose card is on the page."""
        self.add()
        proto = dict(
            self.cfg,
            features={
                "ticket_creation": dict(
                    self.cfg["features"]["ticket_creation"], mode="prototype"
                )
            },
        )
        home = pathlib.Path(self.tmp.name) / "home"
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}):
            pages = {
                "live": (self.render(webdash.LOCAL), ""),
                "prototype": (
                    self.render(webdash.LOCAL, proto),
                    tickets.held_words(proto),
                ),
            }
            state_store.set(home, "standby.on", True)
            pages["standby"] = (
                self.render(webdash.LOCAL),
                tickets.held_words(self.cfg),
            )
        for mode, (page, held) in pages.items():
            with self.subTest(mode=mode):
                card = page[page.index("<section id='tickets'") :]
                card = card[: card.index("</section>")]
                self.assertIn(
                    "Proposed tickets, for problems the monitor found along the way (not breakages, not review "
                    "findings): Approve files the ticket in JIRA through the atlassian CLI, and its key shows here; "
                    "Decline drops it. One action per proposed ticket.",
                    card,
                )
                self.assertIn("<th>Summary</th><th>Text</th><th>Status</th>", card)
                # The full text's address keeps the route's name.
                self.assertNotIn("draft", card.lower().replace("/ticket-draft?id=", ""))
                if mode == "live":
                    self.assertNotIn("unsent item", card)
                    continue
                self.assertIn(mode, held)
                self.assertIn(
                    f"Approve files nothing now, because {held}. It keeps the proposed ticket as an unsent item, "
                    "listed under <a href='#not-sent'>Not sent</a>.",
                    card,
                )
                self.assertIn("<section id='not-sent'", page)

    def test_no_card_while_the_feature_is_off_and_nothing_waits(self):
        off = dict(self.cfg, features={"ticket_creation": {"enabled": False}})
        self.assertNotIn("id='tickets'", self.render(webdash.LOCAL, off))


OWN = "http://127.0.0.1:18613"  # the served page's origin


class TestRoutes(Base):
    """The card's requests: the priority box's checks (the served page's origin, the X-Rkmon header, the access
    token on every request)."""

    def setUp(self):
        super().setUp()
        self.token = "invented-" + "access-token-for-the-card"  # built at run time
        path = pathlib.Path(self.cfg["admin_dashboard"]["access_token_file"])
        path.write_text(self.token + "\n", encoding="utf-8")
        path.chmod(0o600)
        self.access = webserve.access_info(self.cfg, 18613, {})

    def basic(self, token: str) -> str:
        return "Basic " + base64.b64encode(f"anyone:{token}".encode()).decode()

    def hdrs(self, **change):
        h = {
            "Origin": OWN,
            "Host": "127.0.0.1:18613",
            "X-Rkmon": "1",
            "Authorization": self.basic(self.token),
            "Content-Type": "application/json",
        }
        h.update(change)
        return h

    def ask(self, method, path, headers, body=b""):
        return webserve.answer(
            self.cfg, self.store, method, path, headers, body, 18613, self.access
        )

    def test_approve_and_decline_from_the_card(self):
        _ok, _msg, tid = self.add()
        _ok, _msg, other = self.add(
            tickets.parse("---\ntitle: Another\npriority: P3\nsummary: s\n---\nbody")
        )
        self.assertEqual(
            self.ask(
                "POST",
                "/ticket-approve",
                self.hdrs(Authorization=self.basic("x")),
                json.dumps({"id": tid}).encode(),
            )[0],
            401,
        )
        self.assertEqual(
            self.ask(
                "POST",
                "/ticket-approve",
                self.hdrs(Origin="http://evil.example"),
                json.dumps({"id": tid}).encode(),
            )[0],
            403,
        )
        self.assertEqual(
            self.ask("POST", "/ticket-approve", self.hdrs(), b'{"id": "nope"}')[0:3:2],
            (
                400,
                {"ok": False, "message": 'send a proposed ticket\'s id: {"id": "..."}'},
            ),
        )
        with mock.patch.object(
            tickets,
            "_cli",
            lambda cfg, run: (["atlassian", "--site", "s", "--json"], FakeCli()),
        ):
            status, h, out = self.ask(
                "POST", "/ticket-approve", self.hdrs(), json.dumps({"id": tid}).encode()
            )
            self.assertEqual(
                (status, out["ok"], out["key"], h["Access-Control-Allow-Origin"]),
                (200, True, f"{PROJ}-1", OWN),
            )
            status, _h, out = self.ask(
                "POST", "/ticket-approve", self.hdrs(), json.dumps({"id": tid}).encode()
            )
            self.assertEqual((status, out["ok"]), (409, False))
            self.assertIn("one action per proposed ticket", out["message"])
        status, _h, out = self.ask(
            "POST", "/ticket-decline", self.hdrs(), json.dumps({"id": other}).encode()
        )
        self.assertEqual((status, out["ok"]), (200, True))
        self.assertEqual(self.item(other)["status"], "declined")
        j = self.journal()
        self.assertRegex(
            j,
            rf"web +POST /ticket-approve {tid} from {OWN}: done: Filed proposed ticket {tid} as ",
        )
        self.assertRegex(
            j,
            rf"web +POST /ticket-decline {other} from {OWN}: done: Declined proposed ticket {other}",
        )
        self.assertRegex(
            j,
            rf"web +refused POST /ticket-approve {tid} from http://evil.example: origin not allowed",
        )

    def test_in_prototype_the_cards_approve_keeps_the_ticket_as_an_unsent_item(self):
        self.cfg["features"]["ticket_creation"]["mode"] = "prototype"
        _ok, _msg, tid = self.add()
        status, _h, out = self.ask(
            "POST", "/ticket-approve", self.hdrs(), json.dumps({"id": tid}).encode()
        )
        self.assertEqual((status, out["ok"]), (200, True), out)
        held = tickets.held_words(self.cfg)
        self.assertEqual(
            (out["message"], out["row"]),
            (
                f"UNSENT: proposed ticket {tid} for the JIRA project {PROJ}, kept as an unsent item: {held}",
                f"not filed: kept as an unsent item ({held})",
            ),
        )
        self.assertEqual(self.item(tid)["status"], "unsent")
        self.assertEqual(
            [i["type"] for i in state_store.unsent_items(self.store.dir)],
            [state_store.TICKET],
        )

    def test_the_full_text_link(self):
        _ok, _msg, tid = self.add(
            tickets.parse(
                "---\ntitle: T <b>\npriority: P3\nsummary: s\n---\nline one\n<script>x</script>"
            )
        )
        nav = {
            "Host": "127.0.0.1:18613",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
            "Authorization": self.basic(self.token),
        }  # a navigation from another page: no secret in the draft, so it may, with the access token
        self.assertEqual(
            self.ask("GET", f"/ticket-draft?id={tid}", dict(nav, Authorization=""))[0],
            401,
        )
        status, h, body = self.ask("GET", f"/ticket-draft?id={tid}", nav)
        self.assertEqual((status, h["X-Frame-Options"]), (200, "DENY"))
        page = body.decode("utf-8")
        self.assertIn("<h1>T &lt;b&gt;</h1>", page)
        self.assertIn("<pre>line one\n&lt;script&gt;x&lt;/script&gt;</pre>", page)
        self.assertEqual(
            self.ask(
                "GET",
                f"/ticket-draft?id={tid}",
                dict(nav, **{"Sec-Fetch-Dest": "iframe"}),
            )[0],
            403,
        )
        self.assertEqual(
            self.ask(
                "GET",
                f"/ticket-draft?id={tid}",
                dict(nav, Host="rebound.example:18613"),
            )[0],
            403,
        )
        self.assertEqual(
            self.ask("GET", "/ticket-draft?id=0000aaaa", nav)[0:3:2],
            (404, {"ok": False, "message": "no such proposed ticket"}),
        )


class TestCli(Base):
    def fake_cli(self) -> tuple[dict, pathlib.Path]:
        """The fake atlassian CLI on PATH: the environment rk runs with, and the log of its calls."""
        bindir = pathlib.Path(self.tmp.name) / "bin"
        bindir.mkdir()
        shutil.copy(FAKES / "fake_atlassian", bindir / "atlassian")
        (bindir / "atlassian").chmod(0o755)
        log = pathlib.Path(self.tmp.name) / "atlassian.log"
        env = {
            "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
            "FAKE_ATLASSIAN_LOG": str(log),
        }
        return env, log

    def test_draft_list_file_and_decline(self):
        f = pathlib.Path(self.tmp.name) / "draft.md"
        f.write_text(DRAFT, encoding="utf-8")
        rc, out = self.run_rk("ticket-draft", str(f))
        self.assertEqual(rc, 0, out)
        tid = out.split()[1].rstrip(":")
        rc, out = self.run_rk(
            "ticket-draft",
            "-",
            stdin="---\ntitle: Second\npriority: P3\nsummary: s\n---\nbody",
        )
        self.assertEqual(rc, 0, out)
        second = out.split()[1].rstrip(":")
        rc, out = self.run_rk("ticket-draft", "-", stdin="no front matter")
        self.assertEqual(rc, 4)
        self.assertTrue(
            out.startswith(
                "REFUSED: the proposed ticket: a proposed ticket starts with a --- block"
            ),
            out,
        )
        rc, out = self.run_rk("tickets")
        self.assertIn(f"{tid} waiting · Bug · P2 · dev-a · The nightly lane", out)
        self.assertIn("(2 waiting for your OK", out)
        # rk ticket-file: the user's fallback, through the CLI on PATH (the fake one), under the write lock.
        env, log = self.fake_cli()
        rc, out = self.run_rk("ticket-file", tid, env=env)
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.startswith(f"Filed proposed ticket {tid} as {PROJ}-1"), out)
        calls = [json.loads(x) for x in log.read_text().splitlines()]
        self.assertEqual(
            [c[3:6] for c in calls],
            [
                ["jira", "issue", "create"],
                ["jira", "user", "get"],
                ["jira", "issue", "update"],
            ],
        )
        self.assertEqual(
            self.run_rk("ticket-file", tid, env=env)[0], 4
        )  # one action per proposed ticket
        rc, out = self.run_rk("ticket-decline", second, "a", "duplicate")
        self.assertEqual((rc, self.item(second)["note"]), (0, "a duplicate"))
        rc, out = self.run_rk(
            "ticket-draft",
            str(f),
            cfg=dict(self.cfg, features={"ticket_creation": {"enabled": False}}),
        )
        self.assertTrue(out.startswith("OFF: out-of-scope issue tickets is off"), out)

    def test_a_draft_check_refuses_is_not_filed(self):
        """rk ticket-file runs rk ticket-draft's check first: a waiting proposed ticket it refuses is not sent; it
        waits, with check's words as its note, and rk ticket-file exits non-zero, naming the two fixes.
        """
        tid = self.without_a_priority()
        env, log = self.fake_cli()
        self.assertEqual(
            self.run_rk("ticket-file", tid, env=env),
            (
                4,
                f"REFUSED: proposed ticket {tid} was not filed: priority: required (one line). It waits: propose "
                "it again under its title with what is missing (rk ticket-draft FILE; it takes this one's place), "
                f"or decline it (Decline on the card, or rk ticket-decline {tid}).",
            ),
        )
        self.assertFalse(log.exists())  # nothing reached the CLI
        item = self.item(tid)
        self.assertEqual(
            (item["status"], item["note"], tickets.status_text(self.cfg, item)),
            (
                "waiting",
                "priority: required (one line)",
                "waiting; priority: required (one line)",
            ),
        )
        self.assertRegex(
            self.journal(),
            rf"ticket +filing {tid} refused: priority: required \(one line\)",
        )


class TestPrototypeAndStandby(Base):
    """While ticket creation does not send (prototype, or the monitor in standby), the card's Approve files
    nothing: the ticket is kept as an unsent item (for the JIRA project; its content the fields it would be filed with
    and its description), the proposed ticket is done, and no CLI login is needed (in prototype). `rk ticket-file`
    typed then is refused, naming the mode or standby, and the proposed ticket waits; Decline drops one as in live.
    """

    def setUp(self):
        super().setUp()
        self.cfg["features"]["ticket_creation"]["mode"] = "prototype"
        self.creds = (
            pathlib.Path(self.tmp.name) / ".config" / "atlassian" / "credentials"
        )
        self.creds.unlink()  # no CLI login: none is needed in prototype
        self.home = pathlib.Path(self.tmp.name) / "home"
        env = mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)

    def items(self) -> list[dict]:
        return state_store.unsent_items(self.store.dir)

    def test_the_cards_approve_keeps_the_ticket_as_an_unsent_item(self):
        _ok, _msg, tid = self.add()
        cli = FakeCli()
        held = tickets.held_words(self.cfg)
        self.assertIn("prototype", held)
        self.assertTrue(features.active(self.cfg, tickets.FEATURE))
        self.assertEqual(
            tickets.approve(self.cfg, self.store, tid, by="op", run=cli, own=True),
            (
                True,
                f"UNSENT: proposed ticket {tid} for the JIRA project {PROJ}, kept as an unsent item: {held}",
                {"row": f"not filed: kept as an unsent item ({held})"},
            ),
        )
        it = self.item(tid)
        self.assertEqual((it["status"], it["decided_by"]), ("unsent", "op"))
        (item,) = self.items()
        self.assertEqual(
            (item["type"], item["for"], item["feature"], item["why"]),
            (
                state_store.TICKET,
                f"the JIRA project {PROJ}",
                "ticket_creation",
                state_store.PROTOTYPE,
            ),
        )
        self.assertTrue(
            item["content"].startswith(
                "summary: The nightly lane skips the fp8 cases on every runner\ntype: Bug\npriority: P2\n"
                "assignee: dev-a\nlabels: ci, nightly\ncomponent: Platform\n\n"
                "What happens: every fp8 case reports SKIPPED on develop."
            ),
            item["content"],
        )
        self.assertIn("approved by op on", item["content"])
        journal = self.journal()
        self.assertIn(
            f"ticket for the JIRA project {PROJ}: not sent (ticket creation in prototype)",
            journal,
        )
        # The item's one line names no content: the title is in the proposed ticket's own line alone.
        self.assertEqual(journal.count("The nightly lane"), 1, journal)
        self.assertIn(f"proposed {tid}: The nightly lane", journal)
        self.assertNotIn("SKIPPED", journal)
        self.assertEqual([r for r in self.store.ledger() if r["kind"] == "ticket"], [])
        # One action per proposed ticket, and the same title is not proposed again meanwhile.
        for again in (
            lambda: tickets.approve(
                self.cfg, self.store, tid, by="op", run=cli, own=True
            ),
            lambda: tickets.decline(
                self.cfg, self.store.load(), self.store, tid, by="op"
            ),
        ):
            got = again()
            self.assertFalse(got[0])
            self.assertIn(
                f"proposed ticket {tid} was kept as an unsent item (approved by op",
                got[1],
            )
        ok, msg, _ = self.add()
        self.assertFalse(ok)
        self.assertIn("kept as an unsent item on ", msg)
        self.assertEqual((len(self.items()), cli.calls), (1, []))

    def test_the_cards_approve_keeps_no_unsent_item_of_a_draft_check_refuses(self):
        """As live would not file it, the card's Approve does not keep it as an unsent item: it waits, with check's
        words as its note."""
        tid = self.without_a_priority()
        cli = FakeCli()
        ok, msg, extra = tickets.approve(
            self.cfg, self.store, tid, by="op", run=cli, own=True
        )
        self.assertFalse(ok)
        self.assertTrue(
            msg.startswith(
                f"proposed ticket {tid} was not filed: priority: required (one line). It waits: propose it "
                "again"
            ),
            msg,
        )
        self.assertEqual(
            (self.items(), cli.calls, self.item(tid)["status"], extra["row"]),
            ([], [], "waiting", "waiting; priority: required (one line)"),
        )

    def test_a_ticket_you_file_by_hand_is_refused_naming_the_mode_then_standby(self):
        _ok, _msg, tid = self.add()
        _ok, _msg, other = self.add(
            tickets.parse("---\ntitle: Another\npriority: P3\nsummary: s\n---\nbody")
        )
        for standby in (False, True):
            if standby:
                self.cfg["features"]["ticket_creation"]["mode"] = "live"
                self.creds.write_text("{}")  # live needs the login, standby or not
                state_store.set(self.home, "standby.on", True)
            held = tickets.held_words(self.cfg)
            self.assertIn("standby" if standby else "prototype", held)
            self.assertEqual(
                self.run_rk("ticket-file", tid),
                (
                    4,
                    f"REFUSED: proposed ticket {tid} is not sent: {held}; it waits",
                ),
            )
            self.assertEqual(self.item(tid)["status"], "waiting")
        self.assertEqual(self.run_rk("ticket-decline", other, "not", "needed")[0], 0)
        self.assertEqual(self.item(other)["status"], "declined")
        self.assertEqual(self.items(), [])
        # The card's Approve, the monitor's own, keeps it in standby as in prototype.
        ok, msg, _ = tickets.approve(
            self.cfg, self.store, tid, by="op", run=FakeCli(), own=True
        )
        self.assertTrue(ok, msg)
        self.assertEqual(
            [(i["type"], i["why"]) for i in self.items()],
            [(state_store.TICKET, state_store.STANDBY)],
        )

    def test_where_the_writer_files_tickets_it_is_not_asked_while_they_do_not_go_out(
        self,
    ):
        from rkmon_lib import atlassian, writer

        _ok, _msg, tid = self.add()
        with mock.patch.object(
            atlassian, "login_here", lambda cfg: False
        ), mock.patch.object(
            writer, "through", side_effect=AssertionError("the writer was asked")
        ):
            ok, msg, _ = tickets.approve(self.cfg, self.store, tid, by="op", own=True)
        self.assertTrue(ok, msg)
        self.assertEqual(self.item(tid)["status"], "unsent")


class TestTheEpic(Base):
    """features.ticket_creation.default_epic: the ticket is created without its description, linked to the epic with
    `jira issue link-epic KEY --epic EPIC`, and only then given its description (linking can garble one set before),
    through the atlassian CLI on PATH (the fake one, which records each call)."""

    def file_with(self, cfg: dict, fail: str = "") -> tuple:
        _ok, _msg, tid = self.add()
        bindir = pathlib.Path(self.tmp.name) / f"bin{fail or 'ok'}"
        bindir.mkdir()
        shutil.copy(FAKES / "fake_atlassian", bindir / "atlassian")
        (bindir / "atlassian").chmod(0o755)
        log = pathlib.Path(self.tmp.name) / f"atlassian{fail or 'ok'}.log"
        env = {
            "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
            "FAKE_ATLASSIAN_LOG": str(log),
            "FAKE_ATLASSIAN_STATE": str(
                pathlib.Path(self.tmp.name) / f"n{fail or 'ok'}"
            ),
            **({"FAKE_ATLASSIAN_FAIL": fail} if fail else {}),
        }
        with mock.patch.dict(os.environ, env):
            ok, msg, _ = tickets.approve(cfg, self.store, tid, by="op")
        calls = [json.loads(x) for x in log.read_text().splitlines()]
        return ok, msg, calls

    def test_create_then_link_then_the_description(self):
        epic = PROJ + "-77"
        cfg = dict(self.cfg, jira=dict(self.cfg["jira"], default_epic=epic))
        ok, msg, calls = self.file_with(cfg)
        self.assertTrue(ok, msg)
        steps = [c[3:6] for c in calls]
        self.assertEqual(
            steps[:3],
            [
                ["jira", "issue", "create"],
                ["jira", "issue", "link-epic"],
                ["jira", "issue", "update"],
            ],
        )
        create, link, update = calls[0], calls[1], calls[2]
        self.assertNotIn(
            "description=", " ".join(create)
        )  # no description before the link
        self.assertEqual(link[6:], [f"{PROJ}-1", "--epic", epic])
        self.assertEqual(update[6], f"{PROJ}-1")
        self.assertTrue(any(a.startswith("description=") for a in update))

    def test_without_an_epic_the_description_goes_with_the_create(self):
        ok, msg, calls = self.file_with(self.cfg)
        self.assertTrue(ok, msg)
        self.assertNotIn(["jira", "issue", "link-epic"], [c[3:6] for c in calls])
        self.assertTrue(any(a.startswith("description=") for a in calls[0]))

    def test_a_link_that_fails_leaves_the_ticket_filed_with_its_description_and_says_so(
        self,
    ):
        epic = PROJ + "-77"
        cfg = dict(self.cfg, jira=dict(self.cfg["jira"], default_epic=epic))
        ok, msg, calls = self.file_with(cfg, fail="link-epic")
        self.assertTrue(ok, msg)  # the ticket is filed
        self.assertIn(f"Not linked to the epic {epic}", msg)
        self.assertIn(
            ["jira", "issue", "update"], [c[3:6] for c in calls]
        )  # its description still set


class TestTheSession(unittest.TestCase):
    def test_it_drafts_and_never_decides(self):
        for cmd in (
            "rk ticket-file 0a1b2c3d",
            "/work/monitor/rk ticket-decline 0a1b2c3d dup",
            "sh /opt/rocke_monitor/rk --home /work/monitor ticket-file 0a1b2c3d --again",
            "python3 -B /opt/rocke_monitor/rkmon.py ticket-file 0a1b2c3d",
        ):
            refusal = orchestrator.guard(cmd, tickets=True)
            self.assertIn(
                "is for a person to run, or for rk itself, never for the session",
                refusal,
                cmd,
            )
            self.assertIn(
                "a found issue's ticket is proposed with rk ticket-draft", refusal, cmd
            )
        for cmd in (
            "rk ticket-draft /work/scratch/post/t.md",
            "rk tickets",
            "cat t.md | rk ticket-draft -",
        ):
            self.assertIsNone(orchestrator.guard(cmd), cmd)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cli_ready(self, tmp.name)
        cfg = {
            "features": {
                "ticket_creation": {
                    "enabled": True,
                    "triggers": {"review_follow_up": False, "out_of_scope_issue": True},
                }
            },
            "jira": {"project": PROJ},
            "atlassian": {"site": "s"},
            "paths": {"working_dir": tmp.name},
        }
        self.assertFalse(
            orchestrator.tickets_on(cfg)
        )  # a found issue's ticket opens no JIRA write to the session
        self.assertTrue(features.active(cfg, "found-issue-tickets"))

    def test_the_skill_says_so(self):
        skill = ROOT / "agentic" / "skills" / "rocke-monitor"
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        text = text[text.index("## The package: features and models") :]
        self.assertIn("`<RK> ticket-draft FILE`", text)
        self.assertIn("Never run\n  `rk ticket-file` or `rk ticket-decline`", text)
        rules = (skill / "rules.md").read_text(encoding="utf-8")
        self.assertIn("Draft it with `<RK> ticket-draft FILE`", rules)
        self.assertIn("nor run `rk ticket-file` or `rk ticket-decline`", rules)


if __name__ == "__main__":
    unittest.main()
