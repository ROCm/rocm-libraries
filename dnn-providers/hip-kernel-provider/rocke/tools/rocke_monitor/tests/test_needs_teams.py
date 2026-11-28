# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The Needs-you decisions copied to the Teams channel (needs_teams.py): which items count and their keys, one message
per batch with a cap, once per key, the gap between messages, the feature switch, failures, and no secret anywhere;
in prototype and in standby, the message kept as an unsent item, and one you type refused.
"""

import datetime as dt
import json
import os
import pathlib
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard_labels  # noqa: E402

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    features,
    idle,
    needs_teams,
    outward,
    post,
    report,
    reviewer_name,
    state_store,
    tickfilter,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402

SECRET = "https://hooks.example.invalid/workflows/abc?sig=TOPSECRET123"


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def rec(n, head=None):
    return {
        "n": n,
        "head": head or (str(n) * 40)[:40],
        "status": "open",
        "draft": False,
        "head_seen_at": ago(60),
        "author": f"dev{n}",
        "title": f"t{n}",
    }


def ask(i: int, kind: str = "HOLD_STALE", text: str | None = None) -> dict:
    return report.need_ask(
        kind,
        i,
        f"h{i}",
        text or f"#{i} has been held 30 h without a push: keep its place?",
    )


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.d.name)
        self.store = Store(self.dir / "state")
        (self.dir / "hook").write_text(SECRET + "\n")
        self.cfg = {
            "incident_issue": guard_labels.INCIDENT_ISSUE,
            "revert_offer_after_min": 120,
            "operator": {"github_login": "opsuser"},
            "repo": "ROCm/r",
            "codeowner_teams": {"platform": ["plat"]},
            "max_posts_per_hour": 6,
            "features": {
                "alerts": {
                    "enabled": True,
                    "mode": "live",
                    "develop_breaks": True,
                    "needs_you": True,
                }
            },
            "teams": {"channel": "Chan", "webhook_file": str(self.dir / "hook")},
        }
        self.st = empty_state()
        self.st["enabled"] = True
        self.st["develop"]["sha"] = "d" * 40
        self.posted = []
        self.post = mock.patch.object(
            post, "_teams_post", lambda cfg, payload: self.posted.append(payload) or 202
        )
        self.post.start()

    def tearDown(self):
        mock.patch.stopall()
        self.d.cleanup()

    def card_texts(self, payload: dict) -> list[str]:
        return [b.get("text", "") for b in payload["attachments"][0]["content"]["body"]]


class TestTheItems(Base):
    def test_decisions_carry_stable_keys_and_status_lines_are_not_decisions(self):
        self.st["prs"] = {str(n): rec(n) for n in (2, 3, 4, 5, 7)}
        self.st["prs"]["7"]["size"] = 80000  # an in-depth review to approve
        self.st["batches"] = [
            {"members": [3], "heads": {"3": "3" * 40}, "status": "validated"}
        ]
        self.st["holds"] = {
            "2": {
                "head": "2" * 40,
                "reason": "G5 golden",
                "status": "held",
                "since": ago(1500),
                "stale_told": True,
            }
        }
        self.st["hold_reviews"] = {
            "4": {"head": "4" * 40, "reason": "G6 tests", "dismissed": ago(30)}
        }
        self.st["merged_over"] = {
            "5": {
                "kind": "MERGED_OVER_RED",
                "by": "dev9",
                "head": "5" * 40,
                "at": ago(60),
                "why": "while our 🔴 review at 5555555555 is open",
            },
            "6": {
                "kind": "HELD_MERGED",
                "by": "dev9",
                "head": "6" * 40,
                "at": ago(25 * 60),
                "why": "while held",
            },
        }
        plan = [
            {
                "prio": 0,
                "kind": "ALERT",
                "pr": None,
                "text": "develop red",
                "critical": True,
            },
            {"prio": 3, "kind": "FIRST_ROUND", "pr": 7, "text": "", "critical": False},
        ]
        b = report.board(self.cfg, self.st, self.store, plan, [])
        kinds = {a["kind"]: a for a in b["asks"]}
        self.assertEqual(
            set(kinds),
            {
                "PROPOSE",
                "HOLD_STALE",
                "HOLD_DISMISSED",
                "MERGED_OVER_RED",
                "IN_DEPTH_REVIEW",
            },
            b["asks"],
        )
        self.assertEqual(kinds["IN_DEPTH_REVIEW"]["key"], "IN_DEPTH_REVIEW:7:")
        self.assertTrue(
            any(t.startswith("Critical: develop red") for t in b["needs"])
        )  # on the list, not a decision
        self.assertFalse(
            any("#6 merged" in t for t in b["needs"])
        )  # past MERGED_SHOWN_H
        self.assertEqual(kinds["PROPOSE"]["key"], f"PROPOSE:3:3@{'3' * 12}")
        for a in b["asks"]:
            self.assertIn(
                a["text"], b["needs"]
            )  # each decision is a Needs-you line, word for word
        again = report.board(self.cfg, self.st, self.store, plan, [])
        self.assertEqual(
            [a["key"] for a in again["asks"]], [a["key"] for a in b["asks"]]
        )
        self.st["holds"]["2"]["head"] = "9" * 40  # a new fact: a new key
        self.assertNotEqual(
            {
                a["key"]
                for a in report.board(self.cfg, self.st, self.store, plan, [])["asks"]
            },
            {a["key"] for a in b["asks"]},
        )

    def test_an_in_depth_review_goes_out_in_the_teams_words(self):
        """The Teams copy of an in-depth review's decision is the team's line, whole: the PR, its size, its agents and
        where to decide it, with no cost, which stays on the admin side, and no command.
        """
        from rkmon_lib import split

        self.st["prs"] = {"7": dict(rec(7), size=80000)}
        plan = [
            {"prio": 3, "kind": "FIRST_ROUND", "pr": 7, "text": "", "critical": False}
        ]
        with mock.patch.object(split, "agent_cost", return_value=8.0):
            b = report.board(self.cfg, self.st, self.store, plan, [])
        self.assertTrue(
            any("about 11 agents, about $88" in t for t in b["needs"]), b["needs"]
        )
        needs_teams.send(self.cfg, self.st, self.store, b["asks"])
        self.assertIn(
            "**IN_DEPTH_REVIEW** · [#7](https://github.com/ROCm/r/pull/7) · #7: an in-depth review of 80,000 changed "
            "lines needs your OK: about 11 agents. Approve or decline it on the admin dashboard.",
            self.card_texts(self.posted[0]),
        )

    def test_an_auto_merge_the_guard_turns_off_itself_is_news(self):
        self.st["prs"] = {"7": dict(rec(7), author="dev7")}
        self.st["automerge_risks"] = {
            "7": {
                "head": "7" * 40,
                "by": "dev7",
                "reasons": ["G6 fails"],
                "since": ago(5),
            }
        }
        with mock.patch("rkmon_lib.merge.automerge_guard_acts", lambda cfg, r: True):
            b = report.board(self.cfg, self.st, self.store, [], [])
        self.assertTrue(
            any(t.startswith("Auto-merge is on for #7") for t in b["needs"])
        )
        self.assertEqual(b["asks"], [])
        with mock.patch("rkmon_lib.merge.automerge_guard_acts", lambda cfg, r: False):
            b = report.board(self.cfg, self.st, self.store, [], [])
        self.assertEqual(
            [a["key"] for a in b["asks"]], [f"AUTOMERGE_RISK:7:{'7' * 12}"]
        )

    def test_the_merge_step_records_a_merge_over_our_findings_and_drops_it_later(self):
        from rkmon_lib import merge

        st = {
            "prs": {
                "8": dict(
                    rec(8),
                    status="merged",
                    ours={"verdict": "\U0001f534", "head": "8" * 40},
                )
            }
        }
        lines = merge.merged_over_red(
            self.cfg, st, self.store, [{"kind": "MERGED", "pr": 8, "by": "dev9"}], set()
        )
        self.assertTrue(lines[0].startswith("!! MERGED_OVER_RED #8"))
        self.assertEqual(st["merged_over"]["8"]["kind"], "MERGED_OVER_RED")
        st["merged_over"]["8"]["at"] = ago(report.MERGED_SHOWN_H * 60 + 1)
        merge.prune_merged_over(st)
        self.assertEqual(st["merged_over"], {})


class TestSending(Base):
    def test_once_per_key_and_again_after_it_left_the_list(self):
        asks = [ask(1), ask(2)]
        self.assertEqual(
            needs_teams.send(self.cfg, self.st, self.store, asks),
            ["NEEDS_YOU_TEAMS sent 2 items to the team chat (HTTP 202)"],
        )
        self.assertEqual(len(self.posted), 1)
        post.mark_sent(self.st, post.KIND_NEEDS_YOU, ago(60))  # past the gap
        self.assertEqual(needs_teams.send(self.cfg, self.st, self.store, asks), [])
        self.assertEqual(len(self.posted), 1)
        needs_teams.send(
            self.cfg, self.st, self.store, [ask(1)]
        )  # #2 left the list: its key is forgotten
        self.assertEqual(set(self.st["needs_teams"]["sent"]), {ask(1)["key"]})
        needs_teams.send(
            self.cfg, self.st, self.store, asks
        )  # back again: sent again, alone
        self.assertEqual(len(self.posted), 2)
        self.assertEqual(sum("#2" in t for t in self.card_texts(self.posted[1])), 1)
        self.assertFalse(any("#1 " in t for t in self.card_texts(self.posted[1])))
        rows = [r for r in self.store.ledger() if r["kind"] == "teams"]
        self.assertEqual([r["ref"] for r in rows], ["needs-you", "needs-you"])
        self.assertTrue(
            rows[0]["note"].startswith(
                "HTTP 202; 2 item(s): HOLD_STALE:1:h1 HOLD_STALE:2:h2"
            )
        )

    def test_one_message_per_batch_with_a_cap(self):
        asks = [ask(i) for i in range(1, 21)]
        needs_teams.send(self.cfg, self.st, self.store, asks)
        self.assertEqual(len(self.posted), 1)
        texts = self.card_texts(self.posted[0])
        self.assertEqual(
            texts[0], f"Automated message by {reviewer_name.name(self.cfg)}"
        )
        self.assertEqual(
            texts[1], "opsuser: 20 decisions for you on the Needs-you list"
        )
        lines = [t for t in texts if t.startswith("**HOLD_STALE**")]
        self.assertEqual(len(lines), needs_teams.MAX_LINES)
        self.assertEqual(
            lines[0],
            "**HOLD_STALE** · [#1](https://github.com/ROCm/r/pull/1) · #1 has been held 30 h "
            "without a push: keep its place?",
        )
        self.assertIn("…and 5 more on the dashboard.", texts)
        self.assertEqual(texts[-1], "See the dashboard for the full list.")
        self.assertEqual(
            len(self.st["needs_teams"]["sent"]), 20
        )  # the message counted them all
        self.assertFalse(any("@" in t for t in texts))  # addressed by name, no mention

    def test_the_gap_between_messages_holds_new_items_for_the_next(self):
        needs_teams.send(self.cfg, self.st, self.store, [ask(1)])
        out = needs_teams.send(self.cfg, self.st, self.store, [ask(1), ask(2), ask(3)])
        self.assertTrue(
            out[0].startswith(
                "NEEDS_YOU_TEAMS waiting: 2 new items go in the next message, at "
            ),
            out,
        )
        self.assertEqual(tickfilter.classify(out[0])[0], tickfilter.ROUTINE)
        self.assertEqual(len(self.posted), 1)
        due = dict(needs_teams._due(self.cfg, self.st, self.store))
        self.assertEqual(
            due,
            {
                "needs-you Teams message": idle.plus(
                    post.last_sent(self.st, post.KIND_NEEDS_YOU), 30
                )
            },
        )
        post.mark_sent(self.st, post.KIND_NEEDS_YOU, ago(31))
        self.assertEqual(
            needs_teams.send(self.cfg, self.st, self.store, [ask(1), ask(2), ask(3)]),
            ["NEEDS_YOU_TEAMS sent 2 items to the team chat (HTTP 202)"],
        )
        self.assertEqual(dict(needs_teams._due(self.cfg, self.st, self.store)), {})
        self.cfg["teams"]["message_cooldown_min"] = -1
        with self.assertRaisesRegex(
            ValueError,
            r"^features\.alerts\.message_cooldown_minutes must be a number of minutes",
        ):
            post.message_cooldown(self.cfg)

    def test_each_kind_of_message_keeps_its_own_clock(self):
        """A develop break message just sent holds no Needs-you message back, and a Needs-you message holds no develop
        break message back: each kind waits features.alerts.message_cooldown_minutes after its own last one.
        """
        post.mark_sent(self.st, post.KIND_BREAK, ago(1))
        self.assertEqual(
            needs_teams.send(self.cfg, self.st, self.store, [ask(1)]),
            ["NEEDS_YOU_TEAMS sent 1 item to the team chat (HTTP 202)"],
        )
        self.assertEqual(
            post.cooldown_until(self.cfg, self.st, post.KIND_BREAK)[:13], ago(-29)[:13]
        )
        self.assertTrue(post.cooldown_until(self.cfg, self.st, post.KIND_NEEDS_YOU))
        self.st[post.SENT_KEY].pop(post.KIND_BREAK)
        self.assertEqual(post.cooldown_until(self.cfg, self.st, post.KIND_BREAK), "")
        cfg = dict(self.cfg, teams=dict(self.cfg["teams"], message_cooldown_min=0))
        self.assertEqual(post.cooldown_until(cfg, self.st, post.KIND_NEEDS_YOU), "")

    def test_the_feature_off_sends_nothing(self):
        for features in (
            {},
            {"alerts": {"enabled": False}},
            {"alerts": {"enabled": True, "develop_breaks": True, "needs_you": False}},
            None,
        ):
            cfg = dict(self.cfg, features=features)
            board = {"asks": [ask(1)]}
            self.assertEqual(needs_teams.tick(cfg, self.st, self.store, board), [])
            self.assertNotIn("needs_teams", self.st)
            self.assertEqual(list(needs_teams._due(cfg, self.st, self.store)), [])
            with self.assertRaisesRegex(
                post.PostRefused,
                r"needs-you alerts is off \(it runs while features\.alerts\.needs_you\.enabled is true\)",
            ):
                needs_teams.command(cfg, self.st, self.store, board, dry_run=False)
        self.assertEqual(self.posted, [])
        self.assertFalse(self.store.ledger_path.exists())

    def test_a_configuration_without_features_keeps_it_off(self):
        """A configuration with no `features` section takes the schema's defaults: alerts off."""
        from rkmon_lib import features

        cfg = {k: v for k, v in self.cfg.items() if k != "features"}
        self.assertFalse(features.on(cfg, needs_teams.FEATURE))
        self.assertEqual(
            needs_teams.tick(cfg, self.st, self.store, {"asks": [ask(1)]}), []
        )
        self.assertEqual(self.posted, [])

    def test_a_failed_post_leaves_the_items_unsent_for_the_next_tick(self):
        def boom(cfg, payload):
            raise post.PostRefused(
                "the Teams webhook failed: HTTP Error 502: Bad Gateway"
            )

        with mock.patch.object(post, "_teams_post", boom):
            out = needs_teams.tick(
                self.cfg, self.st, self.store, {"asks": [ask(1), ask(2)]}
            )
        self.assertEqual(
            out,
            [
                "(needs-you Teams message not sent: the Teams webhook failed: HTTP Error 502: Bad "
                "Gateway; its 2 items wait for the next full tick)"
            ],
        )
        self.assertEqual(tickfilter.classify(out[0])[0], tickfilter.ROUTINE)
        self.assertEqual(self.st["needs_teams"]["sent"], {})
        self.assertEqual(
            dict(needs_teams._due(self.cfg, self.st, self.store)),
            {
                "needs-you Teams message": idle.plus(
                    self.st["needs_teams"]["last_try"], 30
                )
            },
        )
        self.assertFalse(self.store.ledger_path.exists())
        self.assertEqual(
            needs_teams.tick(self.cfg, self.st, self.store, {"asks": [ask(1), ask(2)]}),
            ["NEEDS_YOU_TEAMS sent 2 items to the team chat (HTTP 202)"],
        )
        (self.dir / "hook").write_text("")
        cfg = dict(self.cfg, teams=dict(self.cfg["teams"], message_cooldown_min=0))
        # On but not set up: the tick skips it (features.tick_reports names the webhook); a send says why it failed.
        self.assertEqual(
            needs_teams.tick(cfg, self.st, self.store, {"asks": [ask(3)]}), []
        )
        out = needs_teams.send(cfg, self.st, self.store, [ask(3)])
        self.assertTrue(
            out[0].startswith(
                "(needs-you Teams message not sent: no Teams webhook configured in "
            ),
            out,
        )

    def test_public_rules_and_the_dashboard_link(self):
        self.cfg["publish"] = {"html": {"url": "http://dash.example.invalid/monitor/"}}
        asks = [
            ask(1, text="#1 is 3.2" " ms slower: hold it?"),
            report.need_ask("BACKLOG_GROWING", None, "losing", "x"),
        ]
        needs_teams.send(self.cfg, self.st, self.store, asks)
        texts = self.card_texts(self.posted[0])
        self.assertIn(
            "**HOLD_STALE** · [#1](https://github.com/ROCm/r/pull/1) · details on the dashboard",
            texts,
        )
        self.assertIn("**BACKLOG_GROWING** · x", texts)
        self.assertEqual(
            texts[-1],
            "The dashboard has the full list: http://dash.example.invalid/monitor/",
        )
        self.assertEqual(
            self.posted[0]["attachments"][0]["content"]["actions"][0]["url"],
            "http://dash.example.invalid/monitor/",
        )


class TestNoSecret(Base):
    def test_the_webhook_url_is_used_but_never_shown(self):
        self.post.stop()
        seen = []

        class Resp:
            status = 202

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def ok(req, timeout):
            seen.append(req.full_url)
            return Resp()

        def fail(req, timeout):
            raise urllib.error.URLError("timed out")

        out = needs_teams.command(
            self.cfg, self.st, self.store, {"asks": [ask(1)]}, dry_run=True
        )
        with mock.patch("urllib.request.urlopen", fail):
            out += needs_teams.tick(self.cfg, self.st, self.store, {"asks": [ask(1)]})
        with mock.patch("urllib.request.urlopen", ok):
            out += needs_teams.tick(self.cfg, self.st, self.store, {"asks": [ask(1)]})
        self.assertEqual(seen, [SECRET])
        self.assertTrue(
            out[-1].startswith("NEEDS_YOU_TEAMS sent 1 item to the team chat"), out
        )
        written = (
            "\n".join(out)
            + json.dumps(self.st)
            + self.store.ledger_path.read_text()
            + self.store.journal_path.read_text()
        )
        for part in ("TOPSECRET123", "hooks.example.invalid", "sig="):
            self.assertNotIn(part, written)


class TestCommand(Base):
    def test_the_dry_run_prints_the_message_and_writes_nothing(self):
        self.st["prs"] = {"2": rec(2)}
        self.st["holds"] = {
            "2": {
                "head": "2" * 40,
                "reason": "G5 golden",
                "status": "held",
                "since": ago(1500),
                "stale_told": True,
            }
        }
        self.store.save(self.st)
        before = {p.name: p.stat().st_mtime_ns for p in self.store.dir.iterdir()}
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print"
        ) as pr:
            self.assertEqual(rkmon.main(["needs-you-teams", "--dry-run"]), 0)
        out = "\n".join(str(c.args[0]) for c in pr.call_args_list)
        self.assertIn(
            "DRY RUN (nothing sent, nothing written): needs-you alerts are on; message label Chan; webhook set",
            out,
        )
        self.assertIn(
            "- HOLD_STALE #2 (https://github.com/ROCm/r/pull/2) · #2 has been held", out
        )
        self.assertEqual(self.posted, [])
        self.assertEqual(
            {p.name: p.stat().st_mtime_ns for p in self.store.dir.iterdir()}, before
        )
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print"
        ):
            self.assertEqual(rkmon.main(["needs-you-teams"]), 0)
        self.assertEqual(len(self.posted), 1)
        sent = list(self.store.load()["needs_teams"]["sent"])
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].startswith(f"HOLD_STALE:2:{'2' * 12}:"), sent)


class TestOwnSwitch(Base):
    def test_the_dry_run_reads_the_needs_you_alerts_own_switch(self):
        """Each kind of alert has its own switch: with the alerts on and needs_you off, the dry run says the Needs-you
        alerts are off."""
        self.cfg["features"]["alerts"]["needs_you"] = False
        out = needs_teams.command(
            self.cfg, self.st, self.store, {"asks": [ask(1)]}, dry_run=True
        )
        self.assertIn(
            "DRY RUN (nothing sent, nothing written): needs-you alerts are off;", out[0]
        )


class TestPrototypeAndStandby(Base):
    """While alerts do not send (prototype, or the monitor in standby), a message is made as live makes it,
    at the same times, and kept as an unsent item instead: nothing is posted, no webhook is needed, its items count as
    sent and its clock runs, and the journal's one line for it names its type and what it was for, never its content.
    One you type is refused, naming the mode or standby. Once standby ends, only new items go: a kept one is never sent.
    """

    def setUp(self):
        super().setUp()
        self.cfg["features"]["alerts"]["mode"] = "prototype"
        (self.dir / "hook").write_text(
            ""
        )  # no webhook: none is needed while nothing goes out
        self.home = self.dir / "home"
        env = mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)

    def items(self) -> list[dict]:
        return state_store.unsent_items(self.store.dir)

    def kept_line(self, what: str) -> str:
        return (
            f"UNSENT: {what} for the team chat channel (Chan), kept as an unsent item: "
            + features.held_words(self.cfg, "alerts")
        )

    def journal(self) -> list[str]:
        return [
            ln.split(" unsent ", 1)[1].strip()
            for ln in self.store.journal_lines()
            if " unsent " in ln
        ]

    def test_in_prototype_the_message_is_kept_and_nothing_is_posted(self):
        self.assertIn("prototype", features.held_words(self.cfg, "alerts"))
        board = {"asks": [ask(1), ask(2)]}
        self.assertEqual(
            needs_teams.tick(self.cfg, self.st, self.store, board),
            ["NEEDS_YOU_TEAMS " + self.kept_line("the Needs-you message (2 items)")],
        )
        (item,) = self.items()
        self.assertEqual(
            (item["type"], item["for"], item["feature"], item["why"]),
            (
                state_store.ALERT,
                "the team chat channel (Chan)",
                "alerts",
                state_store.PROTOTYPE,
            ),
        )
        self.assertEqual(
            item["content"],
            post.card_text(needs_teams.message(self.cfg, board["asks"])),
        )
        self.assertIn("#2 has been held 30 h", item["content"])
        # Its items count as sent and its clock runs, as live's would: the same list keeps nothing more.
        self.assertEqual(
            set(self.st["needs_teams"]["sent"]), {ask(1)["key"], ask(2)["key"]}
        )
        self.assertTrue(post.cooldown_until(self.cfg, self.st, post.KIND_NEEDS_YOU))
        self.assertEqual(needs_teams.tick(self.cfg, self.st, self.store, board), [])
        post.mark_sent(self.st, post.KIND_NEEDS_YOU, ago(31))
        needs_teams.tick(
            self.cfg, self.st, self.store, {"asks": [ask(1), ask(2), ask(3)]}
        )
        self.assertEqual(
            [i["content"].count("#3 has been held") for i in self.items()], [0, 1]
        )
        self.assertEqual(
            self.journal(),
            ["alert for the team chat channel (Chan): not sent (alerts in prototype)"]
            * 2,
        )
        self.assertNotIn("has been held", "\n".join(self.store.journal_lines()))
        self.assertEqual(self.posted, [])
        self.assertFalse(self.store.ledger_path.exists())

    def test_in_standby_nothing_goes_out_and_once_it_ends_only_new_items_go(self):
        self.cfg["features"]["alerts"]["mode"] = "live"
        (self.dir / "hook").write_text(SECRET + "\n")
        state_store.set(self.home, "standby.on", True)
        self.assertIn("standby", features.held_words(self.cfg, "alerts"))
        self.assertEqual(
            needs_teams.tick(self.cfg, self.st, self.store, {"asks": [ask(1)]}),
            ["NEEDS_YOU_TEAMS " + self.kept_line("the Needs-you message (1 item)")],
        )
        self.assertEqual(
            [(i["type"], i["why"]) for i in self.items()],
            [(state_store.ALERT, state_store.STANDBY)],
        )
        with self.assertRaisesRegex(
            post.PostRefused,
            "^the Needs-you message is not sent: the monitor is in standby: nothing goes out until standby ends",
        ):
            needs_teams.command(
                self.cfg, self.st, self.store, {"asks": [ask(2)]}, dry_run=False
            )
        self.assertEqual(self.posted, [])
        state_store.set(self.home, "standby.on", False)
        post.mark_sent(self.st, post.KIND_NEEDS_YOU, ago(31))
        self.assertEqual(
            needs_teams.tick(self.cfg, self.st, self.store, {"asks": [ask(1), ask(2)]}),
            ["NEEDS_YOU_TEAMS sent 1 item to the team chat (HTTP 202)"],
        )
        (sent,) = self.posted
        self.assertTrue(any("#2 has been held" in t for t in self.card_texts(sent)))
        self.assertFalse(any("#1 has been held" in t for t in self.card_texts(sent)))
        self.assertEqual(len(self.items()), 1)

    def test_a_message_you_type_is_refused_naming_the_mode_and_the_monitors_own_is_kept(
        self,
    ):
        self.st["prs"] = {"2": rec(2)}
        self.st["holds"] = {
            "2": {
                "head": "2" * 40,
                "reason": "G5 golden",
                "status": "held",
                "since": ago(1500),
                "stale_told": True,
            }
        }
        self.store.save(self.st)
        out = []
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["needs-you-teams"]), 4)
            self.assertEqual(
                out,
                [
                    "REFUSED: the Needs-you message is not sent: "
                    + features.held_words(self.cfg, "alerts")
                ],
            )
            self.assertEqual(self.items(), [])
            with mock.patch.dict(os.environ, {outward.OWN_VAR: "1"}):
                self.assertEqual(rkmon.main(["needs-you-teams"]), 0)
        self.assertEqual(
            out[1:],
            ["NEEDS_YOU_TEAMS " + self.kept_line("the Needs-you message (1 item)")],
        )
        self.assertEqual([i["type"] for i in self.items()], [state_store.ALERT])
        self.assertEqual(self.posted, [])

    def test_a_develop_break_alert_is_kept_when_the_monitor_asks_and_refused_when_typed(
        self,
    ):
        """Kept, it counts as sent everywhere: its clock runs, and the ledger records the channel as alerted about the
        issue, so once alerts are live no run sends it."""
        from rkmon_lib import gh

        issue = {
            "number": 7,
            "title": "x fails on develop",
            "state": "open",
            "html_url": "https://github.com/ROCm/r/issues/7",
            "body": "",
            "labels": ["red-develop"],
        }
        with mock.patch.object(
            gh, "rest", lambda path, jq=None: dict(issue)
        ), mock.patch.object(post.daylog, "add") as day:
            with self.assertRaisesRegex(
                post.PostRefused,
                "^a develop break alert is not sent: features.alerts.mode is prototype",
            ):
                post.teams_breakage(self.cfg, self.st, self.store, 7)
            self.assertEqual(post.last_sent(self.st, post.KIND_BREAK), "")
            with mock.patch.dict(os.environ, {outward.OWN_VAR: "1"}):
                self.assertEqual(
                    post.teams_breakage(self.cfg, self.st, self.store, 7),
                    self.kept_line("a develop break alert"),
                )
        (item,) = self.items()
        self.assertEqual(
            (item["type"], item["for"]),
            (state_store.ALERT, "the team chat channel (Chan)"),
        )
        self.assertIn("develop is broken (#7)", item["content"])
        self.assertTrue(
            post.last_sent(self.st, post.KIND_BREAK)
        )  # its clock runs as a sent message's
        self.assertEqual(
            self.journal(),
            ["alert for the team chat channel (Chan): not sent (alerts in prototype)"],
        )
        self.assertEqual(self.posted, [])
        (rec,) = self.store.ledger()
        self.assertEqual(
            (rec["kind"], rec["pr"], rec["head"], rec["note"]),
            ("teams", 7, "issue-7", self.kept_line("a develop break alert")),
        )
        day.assert_not_called()
        self.cfg["features"]["alerts"]["mode"] = "live"
        (self.dir / "hook").write_text(SECRET + "\n")
        with mock.patch.object(
            gh, "rest", lambda path, jq=None: dict(issue)
        ), mock.patch.object(post.daylog, "add"), mock.patch.dict(
            os.environ, {outward.OWN_VAR: "1"}
        ), self.assertRaisesRegex(
            post.PostRefused, "^the channel was already alerted about #7$"
        ):
            post.teams_breakage(self.cfg, self.st, self.store, 7, force=True)
        self.assertEqual(self.posted, [])

    def test_the_auto_merge_guards_alert_follows_the_alerts_mode(self):
        """In prototype the merge queue's guard alerts and never acts; its alert, on the Needs-you list, goes out
        while alerts are live, and is kept while they are in prototype."""
        self.cfg["features"]["merge_queue"] = {"enabled": True, "mode": "prototype"}
        self.assertEqual(features.guard_mode(self.cfg), "alert")
        risk = ask(4, kind="AUTOMERGE_RISK", text="#4 has auto-merge on and fails G1")
        needs_teams.send(self.cfg, self.st, self.store, [risk], own=True)
        self.assertEqual([i["type"] for i in self.items()], [state_store.ALERT])
        self.assertIn("AUTOMERGE_RISK", self.items()[0]["content"])
        self.cfg["features"]["alerts"]["mode"] = "live"
        (self.dir / "hook").write_text(SECRET + "\n")
        post.mark_sent(self.st, post.KIND_NEEDS_YOU, ago(31))
        again = ask(5, kind="AUTOMERGE_RISK", text="#5 has auto-merge on and fails G1")
        self.assertEqual(
            needs_teams.send(self.cfg, self.st, self.store, [risk, again], own=True),
            ["NEEDS_YOU_TEAMS sent 1 item to the team chat (HTTP 202)"],
        )
        self.assertEqual(features.guard_mode(self.cfg), "alert")
        self.assertEqual(len(self.posted), 1)


class TestBlankLabel(Base):
    """A blank message label (the deploy settings' features.alerts.message_label) leaves the messages unlabelled
    and stops none of them: the webhook decides the channel, and no line names the label as the channel.
    """

    def setUp(self):
        super().setUp()
        self.cfg["teams"]["channel"] = ""

    def test_the_needs_you_message_goes_without_a_label(self):
        self.assertEqual(
            needs_teams.send(self.cfg, self.st, self.store, [ask(1)]),
            ["NEEDS_YOU_TEAMS sent 1 item to the team chat (HTTP 202)"],
        )
        self.assertEqual(
            self.posted[0]["summary"], needs_teams.headline(self.cfg, [ask(1)])
        )
        labelled = dict(self.cfg, teams=dict(self.cfg["teams"], channel="Chan"))
        self.assertEqual(
            needs_teams.message(labelled, [ask(1)])["summary"],
            "Chan: " + needs_teams.headline(labelled, [ask(1)]),
        )

    def test_the_dry_run_says_there_is_no_label(self):
        out = needs_teams.command(
            self.cfg, self.st, self.store, {"asks": [ask(1)]}, dry_run=True
        )
        self.assertIn(
            "needs-you alerts are on; message label none; webhook set", out[0]
        )

    def test_the_test_card_and_the_develop_break_alert_go_without_a_label(self):
        from rkmon_lib import gh

        self.assertEqual(
            post.teams_test(self.cfg, self.store),
            "test card sent to the team chat (HTTP 202)",
        )
        self.assertEqual(self.posted[-1]["summary"], "rocKE monitor test")
        issue = {
            "number": 7,
            "title": "x fails on develop",
            "state": "open",
            "html_url": "https://github.com/ROCm/r/issues/7",
            "body": "",
            "labels": ["red-develop"],
        }
        with mock.patch.object(
            gh, "rest", lambda path, jq=None: dict(issue)
        ), mock.patch.object(post.daylog, "add"):
            self.assertEqual(
                post.teams_breakage(self.cfg, self.st, self.store, 7),
                "alerted the team chat about #7 (HTTP 202)",
            )
        self.assertEqual(self.posted[-1]["summary"], "develop is broken (#7)")
        self.assertEqual(
            post.teams_breakage_card(issue, None, "Chan", "rkmon-b")["summary"],
            "Chan: develop is broken (#7)",
        )


class TestTheDeploymentsName(Base):
    """Each team chat message names the deployment that sends it, so that the messages of several deployments in one
    channel say which sent each: its first line gives the automated reviewer name, else the container name
    (reviewer_name.name). The writer's check and the unsent item's text read that line by its form.
    """

    def test_each_message_names_the_deployment(self):
        from rkmon_lib import gh

        issue = {
            "number": 7,
            "title": "x fails on develop",
            "state": "open",
            "html_url": "https://github.com/ROCm/r/issues/7",
            "body": "",
            "labels": ["red-develop"],
        }
        named = {
            **self.cfg["features"],
            "review_automation": {"automated_reviewer_name": "Team A monitor"},
        }
        for i, (extra, name) in enumerate(
            (
                ({"deployment": {"project": "rkmon-b"}}, "rkmon-b"),  # no reviewer name
                (
                    {"deployment": {"project": "rkmon-b"}, "features": named},
                    "Team A monitor",
                ),
            )
        ):
            with self.subTest(name=name):
                cfg = {**self.cfg, **extra}
                store = Store(self.dir / f"state{i}")
                post.teams_test(cfg, store)
                with mock.patch.object(
                    gh, "rest", lambda path, jq=None: dict(issue)
                ), mock.patch.object(post.daylog, "add"):
                    st = json.loads(json.dumps(self.st))
                    post.teams_breakage(cfg, st, store, 7)
                for payload in (
                    needs_teams.message(cfg, [ask(1)]),
                    *self.posted[-2:],  # the test card, the develop break alert
                ):
                    self.assertEqual(
                        self.card_texts(payload)[0], f"Automated message by {name}"
                    )
                    self.assertTrue(post.is_message(payload))
                    self.assertNotIn("Automated message", post.card_text(payload))
        message = needs_teams.message(self.cfg, [ask(1)])
        for line in (
            "Automated message by ",
            "Automated message by  ",
            "Automated review by x",
        ):
            with self.subTest(line=line):
                card = message["attachments"][0]["content"]
                card["body"][0] = dict(card["body"][0], text=line)
                self.assertFalse(post.is_message(message))


class TestTheTestCard(Base):
    """rk teams-test's card says it is a test that needs no action, and names the alerts the channel gets: the
    features of rk teams-test's row in rkmon_lib/outward.py, each by its title and what it does.
    """

    def test_the_card_says_it_is_a_test_and_names_the_alerts(self):
        post.teams_test(self.cfg, self.store)
        alerts = [
            features.CAPABILITIES[f] for f in outward.row(outward.TEAMS_TEST).features
        ]
        self.assertIn(features.CAPABILITIES["chat-alerts"], alerts)
        self.assertEqual(
            post.card_text(self.posted[-1]).splitlines(),
            [
                "Chan: rocKE monitor test",
                "Test: the rocKE monitor can post to this channel",
                "No action needed: this is a test card. The rocKE monitor sends these alerts to this channel, each "
                "while it is on:",
                *(f"{c.title}: {c.what}" for c in alerts),
            ],
        )


if __name__ == "__main__":
    unittest.main()
