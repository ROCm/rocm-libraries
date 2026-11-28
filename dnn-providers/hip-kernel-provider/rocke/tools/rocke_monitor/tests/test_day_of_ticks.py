# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A day of the monitor's ticks with every feature that sends on, against fakes for GitHub, the team chat, the web host
and JIRA: live, in prototype, and in standby.

At each tick the monitor makes its writes as a deployment makes them: the tick its own (the Needs-you message,
rkmon_lib/needs_teams.py; the team dashboard's page, rkmon_lib/publish.py); the Claude session its rk commands, in the
environment it is given (orchestrator.session_env, which marks its writes as the monitor's own), those the tick's lines
ask for (rules_tick.rule) and those of its own work (a review, a hold's change request, a develop issue and a label, a
develop break alert, a follow-up ticket); and the admin dashboard's card the Approve of a found issue's ticket
(tickets.approve, as rkmon_lib/webserve.py calls it).

- Live sends each write.
- In prototype nothing leaves but the protective write (turning off an auto-merge the monitor armed while it was live),
  and each other write that live sent is kept as one unsent item of its type, for what it was sent to; the page is one
  item, replaced at each tick. The journal names each item's type and what it was for, never its content.
- In standby nothing leaves, the protective write included: it waits, and once standby.on is cleared the next tick
  makes it, and no unsent item is sent.
- Turning auto-merge off is the protective write only for the PR the monitor armed: for another PR, or with nothing
  armed, it is an auto-merge write, kept or refused as each other write is. The reason for a turn-off made while live,
  left unposted, is a queue notice, kept or refused too.
- A write you type is refused, naming the mode, then standby, and nothing leaves.
- Every command that sends (each OWN and GATED row of rkmon_lib/outward.py's table), as the monitor's own and typed:
  in prototype and in standby only the armed PR's disarm leaves, in prototype; the monitor's own is kept, its line alone
  saying so (exit 0), and one you type is refused. A kept review's AFTER line runs each as the monitor's own, but those
  the session may not run, which it never runs.

The writer's refusal of a send the gate stops is tests/test_writer.py's (TestTheWriterAsksTheOutwardGate), and the drop
of old unsent items tests/test_state_store.py's (test_retention_keeps_the_items_one_rotation_then_drops_them).
"""

import collections
import contextlib
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
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
import test_mirror  # noqa: E402
from rkmon_lib import (  # noqa: E402
    bindings,
    breakage,
    config,
    features,
    gh,
    merge,
    needs_teams,
    orchestrator,
    outward,
    post,
    premises,
    publish,
    report,
    rules_tick,
    state_store,
    tickets,
    writer,
)
from rkmon_lib import secrets_registry as sr  # noqa: E402
from rkmon_lib import spec as spec_mod  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

FAKES = ROOT / "tests" / "fakes"
REPO = "acme/widgets"
GITHUB = f"https://github.com/{REPO}"
PROJ = (
    "TK" + "T"
)  # an invented JIRA project key, built at run time (the privacy check looks for <KEY>-<n>)
SITE = "example-site"
RED, NEW_LABEL = "red-develop", "flaky-test"
HOOK = "https://hooks.example.invalid/workflows/Day-Hook"  # an invented webhook
DEVELOP = "d" * 40
HEADS = {4: "e" * 40, 5: "5" * 40, 7: "7" * 40, 8: "8" * 40}  # each PR's head now
ARMED_AT = (
    "4" * 40
)  # #4's head when the monitor armed it, the merge queue live: it has moved since
BROKEN = 31  # an open develop issue that makes CI red, filed earlier: the day's develop break alert is about it
BROKEN_ISSUE = {
    "number": BROKEN,
    "title": "[develop] the gemm build fails",
    "html_url": f"{GITHUB}/issues/{BROKEN}",
    "state": "open",
    "body": "The gemm build fails on develop.",
    "labels": [RED],
    "assignees": [],
    "pr": False,  # an issue, not a pull request
}
ASK = report.need_ask("READY_FOR_APPROVAL", 9, "h9", "#9 is ready for your approval")
DRAFT = """---
title: The nightly lane skips the widget cases on every runner
type: Bug
priority: P3
summary: every widget case reports SKIPPED
---
What happens: every widget case reports SKIPPED on develop.
"""
FOLLOW_UP = [
    writer.ATLASSIAN,
    "--site",
    SITE,
    "jira",
    "issue",
    "create",
    "--project",
    PROJ,
    "--type",
    "Task",
    "--summary",
    "Follow up: the widget test is still skipped",
]
JIRA_WRITES = {
    "create",
    "update",
    "link-epic",
    "comment",
    "remote-link",
}  # a JIRA call that writes

# The day's ticks: the lines the merge queue plans (the session acts on each as the rules say, rules_tick.rule; while a
# PR the monitor armed has moved on, the queue asks first that its auto-merge be turned off: Day.lines), then the
# monitor's other writes, by name (Day.write).
DAY = (
    ((), ("review", "needs-you", "page")),
    (
        (
            "QUEUE_NOTICE #5: next (dddddddddd); run `rk queue-notice 5`",
            "ARM_READY #5: batch [5] validated on develop dddddddddd; run `rk arm 5`",
        ),
        ("hold", "issue", "alert", "page"),
    ),
    ((), ("label", "follow-up", "approve", "page")),
)
PROTECTIVE = "disarm 4"
# The feature each write of the day sends for (features.SENDS): its unsent item's, and the one a refusal names.
FEATURE = {
    PROTECTIVE: "merge_queue",
    "queue-notice 5": "merge_queue",
    "arm 5": "merge_queue",
    "review": "review_automation",
    "needs-you": "alerts",
    "page": "dashboards.team",
    "hold": "merge_queue",
    "issue": "merge_queue",
    "alert": "alerts",
    "label": "merge_queue",
    "follow-up": "ticket_creation",
    "approve": "ticket_creation",
}


class Answer:
    """A webhook's answer: urlopen's response, as a context manager."""

    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Day(unittest.TestCase):
    """A deployment on a host, in a home of its own for each day, its edges faked: what leaves (self.sent: where it
    went, and what it was), and what GitHub answers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        bindir = self.root / "bin"
        bindir.mkdir()
        shutil.copy(FAKES / "fake_atlassian", bindir / "atlassian")
        (bindir / "atlassian").chmod(0o755)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in (writer.BROKER_VAR, gh.BROKERED, outward.OWN_VAR)
        }
        env.update(
            RKMON_CONTAINER="",
            PATH=f"{bindir}:{env.get('PATH', '')}",
        )
        self.runs = 0
        for p in (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(post, "_gh_write", self.gh_write),
            mock.patch.object(gh, "rest", self.gh_rest),
            mock.patch.object(gh, "_exec", self.no_gh),
            mock.patch("urllib.request.urlopen", self.urlopen),
            mock.patch.object(publish, "_rsync", self.rsync),
            # Every capability set up as live needs it (what each lacks is tests/test_features.py's).
            mock.patch.object(features, "missing", lambda cfg, name: []),
            # `rk queue-notice` builds the queue; the state's validated batch stands, #5 due a notice.
            mock.patch.object(rkmon.queue_mod, "build_queue", lambda *a: []),
            mock.patch.object(
                merge, "desired_notices", lambda cfg, st, q: {5: ("next", "")}
            ),
            mock.patch.object(
                merge,
                "notice_text",
                lambda cfg, st, n, state, detail: f"{post.notice_line(cfg, 'queue')}\n\n#{n} is next in the queue.\n",
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    # --- the edges ----------------------------------------------------------------------------------------------------

    def gh_write(self, args, **_kw):
        """GitHub takes the write: what it was, and its URL."""
        self.sent.append(
            ("github", " ".join(args[:3] + [a for a in args if a == "--disable-auto"]))
        )
        url = {
            ("pr", "comment"): f"{GITHUB}/pull/{args[2]}#issuecomment-{len(self.sent)}",
            ("issue", "create"): f"{GITHUB}/issues/{BROKEN + 1}",
        }.get(tuple(args[:2]), "")
        return subprocess.CompletedProcess(args, 0, url + "\n", "")

    def gh_rest(self, path, jq=None, paginate=False, **_kw):
        """What GitHub answers the writes' reads: the PRs open at their heads, mergeable, auto-merge on, no review of
        the operator's; the develop issues' label there, the new one not yet; the one open develop issue.
        """
        repo = f"repos/{REPO}/"
        m = re.fullmatch(re.escape(repo) + r"pulls/(\d+)", path)
        if m:
            head = HEADS[int(m.group(1))]
            if jq == ".head.sha":
                return head
            return {
                "state": "open",
                "merged": False,
                "draft": False,
                "head": head,
                "mergeable": True,
                "mergeable_state": "clean",
                "auto": True,
            }
        if path == f"{repo}labels/{NEW_LABEL}":
            raise gh.GhError("gh api: Not Found (HTTP 404)")
        if path.startswith(f"{repo}labels/"):
            return path.rsplit("/", 1)[1]
        # The develop issue: its fields by name, as each read projects them (jq), or its assignees' logins.
        if path == f"{repo}issues/{BROKEN}":
            return (
                BROKEN_ISSUE["assignees"]
                if jq == "[.assignees[].login]"
                else BROKEN_ISSUE
            )
        if path.startswith(
            f"{repo}issues?"
        ):  # the open develop issues (breakage.fetch_open)
            i = BROKEN_ISSUE
            return [
                {
                    "n": i["number"],
                    "title": i["title"],
                    "url": i["html_url"],
                    "author": "dev-c",
                    "labels": i["labels"],
                    "assignees": [],
                    "created": "2026-11-20T08:00:00Z",
                    "body": i["body"],
                }
            ]
        if path.endswith("/reviews?per_page=100"):
            return []
        raise AssertionError(f"a GitHub read the day does not expect: {path}")

    @staticmethod
    def no_gh(*a, **_kw):
        raise AssertionError(f"a gh call outside the fakes: {a[:1]}")

    def urlopen(self, req, timeout=None):
        """The team chat takes the message."""
        self.sent.append(("teams", json.loads(req.data)["summary"]))
        return Answer()

    def rsync(self, cfg, src, dest, run, *opts, missing_ok=False):
        """The web host takes the page."""
        self.sent.append(("web", dest))
        return None

    def jira_calls(self) -> list[list[str]]:
        log = self.home / "atlassian.log"
        return (
            [json.loads(x) for x in log.read_text().splitlines()]
            if log.exists()
            else []
        )

    # --- the deployment -----------------------------------------------------------------------------------------------

    def deploy(self, mode: str, standby: bool = False) -> None:
        """A new deployment with every feature that sends on, each in MODE, and STANDBY: its home, settings, secrets
        and state at the day's start (#4 armed by the monitor while it was live, its head moved since; #5 cleared and
        next, #7 to review, #8 held), and the files the session writes."""
        self.runs += 1
        self.home = self.root / f"{self.runs}-{mode}{'-standby' if standby else ''}"
        self.sent = []
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo=REPO,
            rocke_path="a/b",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
            max_posts_per_hour=50,  # the day's ticks run in seconds here: the hourly limit stays out of the way
            incident_issue=dict(
                c["incident_issue"], labels=[RED], stability_label="silent-break"
            ),
            publish={
                "html": dict(
                    c["publish"]["html"],
                    host="web.example",
                    user="op",
                    dest="site/index.html",
                )
            },
            team_dashboard=dict(c.get("team_dashboard") or {}, mode="publish"),
            atlassian=dict(c.get("atlassian") or {}, site=SITE),
            jira=dict(
                c.get("jira") or {}, project=PROJ, url="https://jira.example.com"
            ),
            features={
                "review_automation": {"enabled": True, "mode": mode},
                "merge_queue": dict(
                    c["features"]["merge_queue"], mode=mode, auto_merge="enabled"
                ),
                "dashboards": {
                    "admin": {"enabled": True},
                    "team": {"enabled": True, "mode": mode},
                },
                "alerts": {
                    "enabled": True,
                    "mode": mode,
                    "develop_breaks": True,
                    "needs_you": True,
                },
                "ticket_creation": {
                    "enabled": True,
                    "mode": mode,
                    "triggers": {"review_follow_up": True, "out_of_scope_issue": True},
                },
            },
        )
        c["paths"] = dict(
            c["paths"],
            working_dir=str(self.home / "work"),
            repo=str(self.root / "none"),
            secrets_dir=str(self.home / "secrets"),
        )
        self.cfg = config.normalize(c)
        self.home.mkdir()
        (self.home / "config.json").write_text(json.dumps(self.cfg), encoding="utf-8")
        hook = pathlib.Path(sr.path(self.cfg, sr.entry(self.cfg, "teams_webhook")))
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text(HOOK + "\n")
        hook.chmod(0o600)
        for k, v in (
            ("RKMON_HOME", str(self.home)),
            ("FAKE_ATLASSIAN_LOG", str(self.home / "atlassian.log")),
            ("FAKE_ATLASSIAN_STATE", str(self.home / "jira")),
        ):
            os.environ[k] = v
        self.store = Store(self.home / "state")
        st = empty_state()
        st["develop"]["sha"] = DEVELOP
        st["prs"] = {
            str(n): {
                "n": n,
                "head": head,
                "status": "open",
                "draft": False,
                "size": 10,
                "author": "dev-a" if n < 7 else "dev-b",
                "ours": None,
            }
            for n, head in HEADS.items()
        }
        st["queue_state"] = {
            "armed": {
                "pr": 4,
                "head": ARMED_AT,
                "develop": DEVELOP,
                "at": "2026-11-20T18:00:00Z",
            }
        }
        st["batches"] = [{"status": "validated", "members": [5]}]
        st["cleared"] = [5]
        st["holds"] = {
            "8": {
                "status": "held",
                "head": HEADS[8],
                "reason": "the widget test fails with it merged",
            }
        }
        self.store.save(st)
        state_store.set(self.home, "standby.on", standby)
        files = self.home / "session"
        files.mkdir()
        self.files = {
            "review": files / "review.md",
            "hold": files / "hold.md",
            "issue": files / "issue.md",
        }
        self.files["review"].write_text(
            f"{bindings.first_line(self.cfg)}\n\n"
            "## PR #7 — Add a widget (dev-b) — 🟢 First-round: the widget review text\n"
        )
        self.files["hold"].write_text(
            f"{post.notice_line(self.cfg, 'hold')}\n\nThe widget hold text: fix it, then re-request a review.\n"
        )
        self.files["issue"].write_text(
            f"---\ntitle: [develop] the widget build fails\nlabels: {RED}\n---\n"
            f"{post.notice_line(self.cfg, 'report')}\n\nThe widget issue text.\n"
        )
        self.own = orchestrator.session_env(self.cfg)[outward.OWN_VAR]

    # --- the day ------------------------------------------------------------------------------------------------------

    def rk(self, argv: list[str], own: bool = True) -> tuple[int, str]:
        """rk ARGV in this process, as the Claude session runs it (OWN: with the mark its environment gives its writes,
        the monitor's own), or as a person types it (no mark): its exit code and what it printed.
        """
        out = io.StringIO()
        mark = {outward.OWN_VAR: self.own} if own else {}
        with mock.patch.dict(os.environ, mark), contextlib.redirect_stdout(
            out
        ), contextlib.redirect_stderr(out):
            try:
                code = rkmon.main(list(argv))
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue().strip()

    def lines(self, planned: tuple) -> list[str]:
        """The merge queue's lines of a tick (merge.advance): while the PR the monitor armed has moved on, first that its
        auto-merge must be turned off, and nothing else armed meanwhile (one armed PR at a time); then the lines it
        PLANNED."""
        armed = (self.store.load().get("queue_state") or {}).get("armed")
        if armed and HEADS[armed["pr"]] != armed["head"]:
            n = armed["pr"]
            return [f"!! DISARM_NEEDED #{n}: its head moved. Run `rk disarm {n}`"] + [
                ln for ln in planned if not ln.startswith("ARM_READY")
            ]
        return list(planned)

    def write(self, name: str, tick: int):
        """The monitor's write NAME at TICK, as the monitor makes it: what it said."""
        if name == "needs-you":  # the full tick's step, under its lock
            with self.store.locked(wait=True):
                st = self.store.load()
                said = needs_teams.tick(self.cfg, st, self.store, {"asks": [ASK]})
                self.store.save(st)
            return said
        if name == "page":  # the page the tick wrote, published
            page = self.home / "published.html"
            page.write_text(f"<p>The widget page at tick {tick}</p>\n")
            return publish.team(self.cfg, self.store.load(), self.store, str(page))
        if (
            name == "approve"
        ):  # the session drafts a found issue's ticket; the operator's card approves it
            with self.store.locked(wait=True):
                st = self.store.load()
                ok, said, tid = tickets.add(
                    self.cfg, st, self.store, tickets.parse(DRAFT)
                )
                self.store.save(st)
            self.assertTrue(ok, said)
            return tickets.approve(self.cfg, self.store, tid, by="op-login", own=True)[
                :2
            ]
        return self.rk(self.commands()[name])

    def commands(self) -> dict:
        """The session's own rk commands of the day, by name."""
        return {
            "review": ["post", "7", str(self.files["review"])],
            "hold": ["request-changes", "8", str(self.files["hold"])],
            "issue": ["issue-file", str(self.files["issue"])],
            "alert": ["teams-breakage", str(BROKEN)],
            "label": [
                "label-create",
                NEW_LABEL,
                "--description",
                "a test that fails now and then",
                "--color",
                "fbca04",
            ],
            "follow-up": FOLLOW_UP,
        }

    def step(self, name: str, run) -> tuple:
        """Run one write: (its name, what left while it ran, the unsent items kept while it ran, what it said)."""
        sent, kept, jira = len(self.sent), len(self.items()), len(self.jira_calls())
        said = run()
        self.sent += [
            ("jira", " ".join(c))
            for c in self.jira_calls()[jira:]
            if JIRA_WRITES & set(c)
        ]
        return name, self.sent[sent:], self.items()[kept:], said

    def tick(self, n: int, planned: tuple = (), writes: tuple = ()) -> list[tuple]:
        """Tick N: the session acts on the queue's lines, then the day's other writes. Each write's step."""
        out = []
        for line in self.lines(planned):
            cmd = rules_tick.rule(line)
            out.append(self.step(" ".join(cmd), lambda cmd=cmd: self.rk(cmd)))
        for name in writes:
            out.append(self.step(name, lambda name=name: self.write(name, n)))
        return out

    def day(self, mode: str, standby: bool = False) -> list[tuple]:
        """The day, in a new deployment in MODE (and STANDBY): each write's step, in order."""
        self.deploy(mode, standby)
        out = []
        for n, (planned, writes) in enumerate(DAY, 1):
            out += self.tick(n, planned, writes)
        return out

    def items(self) -> list[dict]:
        return state_store.unsent_items(self.store.dir)

    def kept_for(self) -> dict:
        """The unsent item prototype keeps for each write of the day: its type, and what it is for; None for the
        protective write, which goes out."""
        chat, jira = post.teams_channel(self.cfg), tickets.unsent_for(self.cfg)
        return {
            PROTECTIVE: None,
            "queue-notice 5": (state_store.COMMENT, "pull request 5"),
            "arm 5": (state_store.AUTO_MERGE, "pull request 5"),
            "review": (state_store.REVIEW, "pull request 7"),
            "needs-you": (state_store.ALERT, chat),
            "page": (state_store.PAGE, publish.PAGE_FOR),
            "hold": (state_store.CHANGE_REQUEST, "pull request 8"),
            "issue": (state_store.ISSUE, f"a new issue in {REPO}"),
            "alert": (state_store.ALERT, chat),
            "label": (state_store.LABEL, f"the repository {REPO}"),
            "follow-up": (state_store.TICKET, jira),
            "approve": (state_store.TICKET, jira),
        }

    def typed(self, held: str) -> None:
        """Each of the session's rk commands of the day, typed by a person while every feature is held for HELD
        ("prototype", "standby"): refused, naming its feature's mode, or standby; nothing leaves, nothing is kept. In
        standby the protective write too (in prototype it goes out, typed or not)."""
        sent, kept = len(self.sent), self.items()
        argvs = {
            "queue-notice 5": ["queue-notice", "5"],
            "arm 5": ["arm", "5"],
            **self.commands(),
        }
        if held == state_store.STANDBY:
            argvs[PROTECTIVE] = PROTECTIVE.split()
        for name, argv in argvs.items():
            with self.subTest(typed=name, held=held):
                code, out = self.rk(argv, own=False)
                words = features.held_words(self.cfg, FEATURE[name])
                self.assertIn(held, words)
                self.assertEqual(code, 4, out)
                self.assertTrue(out.startswith("REFUSED: "), out)
                self.assertIn(f" is not sent: {words}", out)
        self.assertEqual(self.sent[sent:], [])
        self.assertEqual(self.items(), kept)

    def journal(self) -> list[tuple[str, str]]:
        """The journal's lines: (kind, text)."""
        return [tuple(ln.split(None, 2)[1:]) for ln in self.store.journal_lines()]

    def unsent_lines(self) -> list[str]:
        """The journal's lines for the unsent items (post.keep)."""
        return [text for kind, text in self.journal() if kind == "unsent"]


class TestADayOfTicks(Day):
    def test_live_sends_each_write(self):
        steps = self.day("live")
        names = [name for name, _sent, _kept, _said in steps]
        self.assertEqual(sorted(set(names)), sorted(FEATURE))  # each write of the day
        for name, sent, kept, said in steps:
            with self.subTest(write=name):
                self.assertTrue(sent, said)
                self.assertEqual(kept, [], said)
        self.assertEqual(self.items(), [])
        self.assertEqual(self.unsent_lines(), [])

    def test_in_prototype_only_the_protective_write_leaves_and_each_other_is_kept(
        self,
    ):
        live = self.day("live")
        steps = self.day("prototype")
        self.assertEqual(
            [name for name, *_ in steps], [name for name, *_ in live]
        )  # the same day
        want = self.kept_for()
        pages = 0
        for (name, sent_live, _k, _s), (_n, sent, kept, said) in zip(live, steps):
            with self.subTest(write=name):
                if want[name] is None:  # protective: it goes, as in live
                    self.assertEqual((sent, kept), (sent_live, []), said)
                    continue
                self.assertEqual(sent, [], said)
                if name == "page" and pages:  # one page item, replaced at each tick
                    self.assertEqual(kept, [], said)
                    continue
                pages += name == "page"
                self.assertEqual(
                    [(i["type"], i["for"], i["feature"], i["why"]) for i in kept],
                    [(*want[name], FEATURE[name], state_store.PROTOTYPE)],
                    said,
                )
        items = self.items()
        each = collections.Counter(
            want[name] for name, *_ in steps if want[name] and name != "page"
        )
        self.assertEqual(
            collections.Counter((i["type"], i["for"]) for i in items),
            each + collections.Counter([want["page"]]),
        )
        (page,) = [i for i in items if i["type"] == state_store.PAGE]
        self.assertEqual(page["file"], str(self.home / "published.html"))
        # The journal: one line for each item, as it was kept, naming its type and what it was for; no other line
        # speaks of one, and none holds what the item holds.
        self.assertEqual(
            self.unsent_lines(),
            [
                f"{i['type']} for {i['for']}: not sent ({features.SENDS[i['feature']]} in prototype)"
                for _n, _s, kept, _said in steps
                for i in kept
            ],
        )
        self.assertEqual(
            [
                (kind, text)
                for kind, text in self.journal()
                if kind != "unsent" and "unsent" in text.lower()
            ],
            [],
        )
        journal = "\n".join(self.store.journal_lines())
        for words in (
            "the widget review text",
            "The widget hold text",
            "The widget issue text",
            "is next in the queue",
            "is ready for your approval",
            "The widget page",
            "every widget case reports SKIPPED",
            "still skipped",
        ):
            self.assertNotIn(words, journal)
        self.typed(state_store.PROTOTYPE)

    def test_in_standby_nothing_leaves_until_it_ends(self):
        self.day("prototype")
        proto_items = [(i["type"], i["for"]) for i in self.items()]
        steps = self.day("live", standby=True)
        for name, sent, kept, said in steps:
            with self.subTest(write=name):
                self.assertEqual(sent, [], said)
                if name == PROTECTIVE:  # it waits: nothing kept, the PR still armed
                    self.assertEqual(kept, [], said)
                    self.assertTrue(said[1].startswith(f"{post.WAITING}: "), said)
        # The queue arms nothing while the PR it armed waits to be turned off.
        self.assertEqual(
            [name for name, *_ in steps if name.startswith("disarm")],
            [PROTECTIVE] * len(DAY),
        )
        self.assertNotIn("arm 5", [name for name, *_ in steps])
        items = self.items()
        self.assertEqual(
            [(i["type"], i["for"]) for i in items],
            [x for x in proto_items if x[0] != state_store.AUTO_MERGE],
        )
        self.assertEqual({i["why"] for i in items}, {state_store.STANDBY})
        self.typed(state_store.STANDBY)
        # Standby ends: the next tick turns the auto-merge off, still due; the Needs-you message it kept is not sent
        # later (its decisions count as sent: nothing new to send), and the page this tick made goes out.
        state_store.set(self.home, "standby.on", False)
        after = self.tick(len(DAY) + 1, (), ("needs-you", "page"))
        self.assertEqual(
            [
                (name, [where for where, _what in sent], kept)
                for name, sent, kept, _ in after
            ],
            [
                (PROTECTIVE, ["github"], []),
                ("needs-you", [], []),
                ("page", ["web"], []),
            ],
            after,
        )
        self.assertEqual(after[0][1], [("github", "pr merge 4 --disable-auto")])
        self.assertEqual(after[1][3], [])
        self.assertIsNone(self.store.load()["queue_state"]["armed"])
        self.assertEqual(self.items(), items)  # none sent, none gone


class TestTurningAutoMergeOff(Day):
    def test_only_the_pr_the_monitor_armed_is_the_protective_write(self):
        """`rk disarm N` is the protective write only for the PR the monitor armed (queue_state's armed PR). For another
        PR (#5, #4 armed), or with nothing armed (#4), it is an auto-merge write: in prototype and in standby the
        monitor's own is kept as an unsent item, and one you type is refused, naming the mode, then standby. Nothing
        leaves, nothing is ledgered, and the armed record stands."""
        for mode, standby, why in (
            ("prototype", False, state_store.PROTOTYPE),
            ("live", True, state_store.STANDBY),
        ):
            for n, armed in ((5, 4), (4, None)):
                with self.subTest(n=n, armed=armed, held=why):
                    self.deploy(mode, standby)
                    if armed is None:
                        st = self.store.load()
                        st["queue_state"]["armed"] = None
                        self.store.save(st)
                    held = features.held_words(self.cfg, "merge_queue")
                    self.assertIn(why, held)
                    argv = ["disarm", str(n), "its", "head", "moved"]
                    self.assertEqual(
                        [self.rk(argv, own=False), self.rk(argv)],
                        [
                            (4, f"REFUSED: turning auto-merge off is not sent: {held}"),
                            (
                                0,
                                f"UNSENT: turning auto-merge off for pull request {n}, kept as an unsent "
                                f"item: {held}",
                            ),
                        ],
                    )
                    self.assertEqual(self.sent, [])
                    self.assertEqual(
                        [
                            (i["type"], i["for"], i["content"], i["feature"], i["why"])
                            for i in self.items()
                        ],
                        [
                            (
                                state_store.AUTO_MERGE,
                                f"pull request {n}",
                                "auto-merge turned off: its head moved",
                                "merge_queue",
                                why,
                            )
                        ],
                    )
                    self.assertEqual(self.store.ledger(), [])
                    self.assertEqual(
                        (self.store.load()["queue_state"]["armed"] or {}).get("pr"),
                        armed,
                    )


class TestTheReasonForATurnOff(Day):
    def test_the_reason_left_over_from_a_turn_off_is_a_queue_notice(self):
        """The reason for a turn-off made while live, left unposted (the user's, with `rk automerge-off N --user`, and
        the guard's explanation), is a queue notice, no protective write: in prototype and in standby the monitor's own
        is kept as an unsent item and says so in its line alone, and one you type is refused, naming the mode, then
        standby. Nothing leaves."""
        reason = "Other merges go first; auto-merge comes back on after them."
        for mode, standby, why in (
            ("prototype", False, state_store.PROTOTYPE),
            ("live", True, state_store.STANDBY),
        ):
            with self.subTest(held=why):
                self.deploy(mode, standby)
                self.store.ledger_add("automerge-off-user", 5, HEADS[5], reason)
                self.store.ledger_add(
                    "automerge-off", 7, HEADS[7], "the gate failed on develop"
                )
                file = self.home / "session" / "reason.md"
                file.write_text(reason + "\n")
                held = features.held_words(self.cfg, "merge_queue")
                self.assertIn(why, held)
                argv = ["automerge-off", "5", "--user", str(file)]
                self.assertEqual(
                    [self.rk(argv, own=False), self.rk(argv)],
                    [
                        (4, f"REFUSED: turning auto-merge off is not sent: {held}"),
                        (
                            0,
                            "UNSENT: a queue notice for pull request 5, kept as an unsent item: "
                            + held,
                        ),
                    ],
                )
                # The guard's own explanation, as its command asks for it (while the merge queue sends nothing, the
                # command only alerts: test_write_gates.py's test_the_guard_s_off_line_names_the_mode_then_standby).
                st = self.store.load()
                with self.assertRaises(post.PostRefused) as typed:
                    post.automerge_off(self.cfg, st, self.store, 7)
                with mock.patch.dict(os.environ, {outward.OWN_VAR: self.own}):
                    own = post.automerge_off(self.cfg, st, self.store, 7)
                self.assertEqual(
                    (str(typed.exception), own),
                    (
                        f"a queue notice is not sent: {held}",
                        "UNSENT: a queue notice for pull request 7, kept as an unsent item: "
                        + held,
                    ),
                )
                self.assertEqual(self.sent, [])
                items = self.items()
                self.assertEqual(
                    [(i["type"], i["for"], i["feature"], i["why"]) for i in items],
                    [
                        (state_store.COMMENT, f"pull request {n}", "merge_queue", why)
                        for n in (5, 7)
                    ],
                )
                self.assertEqual(
                    items[0]["content"],
                    f"{post.notice_line(self.cfg, 'queue')}\n\n{reason}\n",
                )

    def test_a_reason_held_since_its_turn_off_says_so_on_a_line_of_its_own(self):
        """Live, standby begins right after the turn-off went out: its reason is kept as an unsent item, and says so on
        a line of its own, after the words that say auto-merge was turned off."""
        self.deploy("live")
        file = self.home / "session" / "reason.md"
        file.write_text("Other merges go first.\n")

        def gh_write(args, **kw):
            done = self.gh_write(args, **kw)
            state_store.set(self.home, "standby.on", True)
            return done

        with mock.patch.object(post, "_gh_write", gh_write):
            code, out = self.rk(["automerge-off", "5", "--user", str(file)])
        held = features.held_words(self.cfg, "merge_queue")
        self.assertIn(state_store.STANDBY, held)
        self.assertEqual(
            (code, out.splitlines()),
            (
                0,
                [
                    "auto-merge disabled on the user's instruction",
                    f"UNSENT: a queue notice for pull request 5, kept as an unsent item: {held}",
                ],
            ),
        )
        self.assertEqual(self.sent, [("github", "pr merge 5 --disable-auto")])
        self.assertEqual(
            [(i["type"], i["for"], i["why"]) for i in self.items()],
            [(state_store.COMMENT, "pull request 5", state_store.STANDBY)],
        )


# What a send of the sweep does that keeps no item (TestEveryCommandThatSends.expected).
GOES, OFF, REFUSED = "goes", "off", "refused"


class TestEveryCommandThatSends(Day):
    """Every command that sends (the OWN and GATED rows of outward.TABLE), as the monitor's own and typed, in prototype
    and in standby, each in a new deployment, nothing able to leave but through the day's fakes, which record it
    (test_mirror.nothing_leaves): only the armed PR's disarm leaves, in prototype. The monitor's own keeps one unsent item
    for each write it would make, and its line alone says so (`UNSENT: ...`; a review's `UNPOSTED: ...`), exit 0; one you
    type is refused, naming the mode, then standby, exit 4, and nothing is kept. A kept review's AFTER line
    (premises.finish) runs each as the monitor's own does, but those the session may not run, which it never runs.
    """

    # Each send of the sweep, by name: a row's command, or what it adds to it (arrange).
    NAMES = (
        "post",
        "spec-post",
        "notice",
        "arm",
        "disarm",
        PROTECTIVE,
        "queue-notice",
        "automerge-off",
        "automerge-off --user",
        "automerge-off --user, its reason left",
        "issue-file",
        "issue-comment",
        "issue-relabel",
        "issue-assign",
        "label-create",
        "request-changes",
        "fix-link",
        "teams-breakage",
        "needs-you-teams",
        "ticket-file",
        "teams-send",
        "web-push",
        "atlassian",
    )
    # The OWN rows the sweep leaves to their own tests, and why.
    NOT_SWEPT = {
        outward.TEAMS_TEST: "a check you run yourself, sent whatever the mode: the session never runs it, nor does a "
        "review's AFTER line (test_a_kept_review_s_after_line_runs_each_as_the_monitor_s_own_or_not_at_all)",
        "gpu-row": "the GPU tests' row of a posted review asks review posting alone (rkmon_lib/gpu.py), not the one "
        "outward gate",
    }

    def expected(self, name: str) -> tuple:
        """What the send NAME does while it is held: (the feature it sends for, and what the monitor's own does). That
        is the unsent items it keeps, as (type, what each is for); or GOES, the protective write (it goes out in
        prototype, and waits in standby); OFF, the auto-merge guard only alerts while the merge queue sends nothing;
        REFUSED, the writer refuses it (its sender keeps its own item before it asks the writer: post.writer_item).
        """
        pr, issue = "pull request {}".format, f"issue {BROKEN}"
        chat, jira = post.teams_channel(self.cfg), tickets.unsent_for(self.cfg)
        merge_queue = {
            "notice": [(state_store.COMMENT, pr(5))],
            "arm": [(state_store.AUTO_MERGE, pr(5))],
            "disarm": [(state_store.AUTO_MERGE, pr(5))],
            PROTECTIVE: GOES,
            "queue-notice": [(state_store.COMMENT, pr(5))],
            "automerge-off": OFF,
            "automerge-off --user": [(state_store.AUTO_MERGE, pr(5))],
            "automerge-off --user, its reason left": [(state_store.COMMENT, pr(5))],
            "issue-file": [(state_store.ISSUE, f"a new issue in {REPO}")],
            "issue-comment": [(state_store.COMMENT, issue)],
            "issue-relabel": [(state_store.LABEL, issue)],
            "issue-assign": [(state_store.ISSUE, issue)],
            "label-create": [(state_store.LABEL, f"the repository {REPO}")],
            "request-changes": [(state_store.CHANGE_REQUEST, pr(8))],
            "fix-link": [(state_store.COMMENT, issue), (state_store.COMMENT, pr(5))],
        }
        if name in merge_queue:
            return "merge_queue", merge_queue[name]
        return {
            "post": ("review_automation", [(state_store.REVIEW, pr(7))]),
            "spec-post": ("review_automation", [(state_store.REVIEW, pr(7))]),
            "teams-breakage": ("alerts", [(state_store.ALERT, chat)]),
            "needs-you-teams": ("alerts", [(state_store.ALERT, chat)]),
            "ticket-file": ("ticket_creation", [(state_store.TICKET, jira)]),
            "teams-send": ("alerts", REFUSED),
            "web-push": ("dashboards.team", REFUSED),
            "atlassian": ("ticket_creation", [(state_store.TICKET, jira)]),
        }[name]

    def arrange(self, name: str) -> tuple:
        """The send NAME, arranged in this deployment: its rk command, and the patches it runs under."""
        files = self.home / "session"
        notice, comment, reason, page = (
            files / f for f in ("notice.md", "comment.md", "reason.md", "page.html")
        )
        notice.write_text(
            f"{post.notice_line(self.cfg, 'queue')}\n\n#5 is next in the queue.\n"
        )
        comment.write_text(
            f"{post.notice_line(self.cfg, 'report')}\n\nThe widget build fails from its first commit on.\n"
        )
        reason.write_text(
            "Other merges go first; auto-merge comes back on after them.\n"
        )
        page.write_text("<p>The widget page</p>\n")
        patches, draft = contextlib.ExitStack(), ""
        if name == "arm":
            # One armed PR at a time: the one the monitor armed was turned off.
            st = self.store.load()
            st["queue_state"]["armed"] = None
            self.store.save(st)
        if name == "automerge-off --user, its reason left":  # turned off while live
            self.store.ledger_add(
                "automerge-off-user", 5, HEADS[5], "Other merges go first"
            )
        if name == "ticket-file":  # the session's draft of a found issue's ticket
            with self.store.locked(wait=True):
                st = self.store.load()
                ok, said, draft = tickets.add(
                    self.cfg, st, self.store, tickets.parse(DRAFT)
                )
                self.store.save(st)
            self.assertTrue(ok, said)
        if name == "spec-post":
            # Its spec written, its premises holding: then the post's own steps (premises.post_checked). The checks
            # give the post filled in and its round's copy, which a post or a keep writes (premises._post_checks).
            ready = {
                "text": self.files["review"].read_text(),
                "copy": premises.posts_dir(self.cfg) / "pr7_fr.md",
            }
            for module, attr, fake in (
                (spec_mod, "write", lambda *a: (["wrote the spec"], 0)),
                (premises, "load", lambda *a: {"SRC": str(self.files["review"])}),
                (premises, "check", lambda *a, **k: ([], ready)),
                (premises, "finish", lambda *a, **k: ([], True)),
            ):
                patches.enter_context(mock.patch.object(module, attr, fake))
        if name == "needs-you-teams":  # the board's decision for the user
            patches.enter_context(
                mock.patch.object(report, "board", lambda *a: {"asks": [ASK]})
            )
        day = self.commands()
        argv = {
            "post": day["review"],
            "spec-post": ["spec-post", "7", "--round", "fr"],
            "notice": ["notice", "5", str(notice), "--kind", "queue", "--key", "k"],
            "arm": ["arm", "5"],
            "disarm": ["disarm", "5", "its", "turn", "comes", "later"],
            PROTECTIVE: PROTECTIVE.split(),
            "queue-notice": ["queue-notice", "5"],
            "automerge-off": ["automerge-off", "5"],
            "automerge-off --user": ["automerge-off", "5", "--user", str(reason)],
            "automerge-off --user, its reason left": [
                "automerge-off",
                "5",
                "--user",
                str(reason),
            ],
            "issue-file": day["issue"],
            "issue-comment": ["issue-comment", str(BROKEN), str(comment), "--key", "k"],
            "issue-relabel": ["issue-relabel", str(BROKEN), "--to", breakage.STABILITY],
            "issue-assign": ["issue-assign", str(BROKEN), "dev-a"],
            "label-create": day["label"],
            "request-changes": day["hold"],
            "fix-link": ["fix-link", str(BROKEN), "5"],
            "teams-breakage": day["alert"],
            "needs-you-teams": ["needs-you-teams"],
            "ticket-file": ["ticket-file", draft],
            "teams-send": ["teams-send", "--type", post.KIND_BREAK],
            "web-push": ["web-push", str(page)],
            "atlassian": FOLLOW_UP,
        }[name]
        return argv, patches

    def held_as_it_should_be(self, name: str, own: bool, why: str, run: tuple) -> None:
        """The send NAME, run as the monitor's own or typed (OWN) while held for WHY (prototype, standby), did as the
        class says. RUN: its exit code and its lines, what left, the unsent items kept and the ledger's new lines.
        """
        code, lines, sent, items, ledgered = run
        feature, does = self.expected(name)
        held = features.held_words(self.cfg, feature)
        self.assertIn(why, held)
        if does == GOES and why == state_store.PROTOTYPE:  # it goes out, typed or not
            self.assertEqual(
                (code, sent, items),
                (0, [("github", "pr merge 4 --disable-auto")], []),
                lines,
            )
            self.assertIsNone(self.store.load()["queue_state"]["armed"])
            return
        self.assertEqual(sent, [], lines)
        # A develop break alert kept counts as sent (post.teams_breakage): its ledger line says so.
        alert = []
        if does == OFF:
            self.assertEqual(
                (code, lines, items),
                (
                    0,
                    [
                        f"OFF: {features.guard_alert_words(self.cfg)}: auto-merge was left on."
                    ],
                    [],
                ),
            )
        elif does == REFUSED or not own:
            self.assertEqual((code, items), (4, []), lines)
            self.assertTrue(lines[-1].startswith("REFUSED: "), lines)
            self.assertIn(f" is not sent: {held}", lines[-1])
        elif does == GOES:  # in standby it waits, and nothing is kept
            self.assertEqual((code, len(lines), items), (0, 1, []), lines)
            self.assertTrue(lines[0].startswith(f"{post.WAITING}: "), lines)
        else:
            self.assertEqual(code, 0, lines)
            self.assertEqual(
                [(i["type"], i["for"], i["feature"], i["why"]) for i in items],
                [(*item, feature, why) for item in does],
            )
            # The Needs-you message's line has the tick's tag before it.
            said = [ln.removeprefix("NEEDS_YOU_TEAMS ") for ln in lines]
            kept = [
                ln
                for ln in said
                if ln.startswith((f"{post.UNSENT}: ", f"{post.KEPT_KIND}: "))
            ]
            self.assertEqual(len(kept), len(does), lines)
            # Each held line alone: no success word before it, and no REFUSED anywhere.
            self.assertEqual(
                [ln for ln in said if "kept as an unsent item" in ln], kept
            )
            self.assertEqual([ln for ln in lines if "REFUSED" in ln], [])
            if name == "teams-breakage":
                alert = [("teams", kept[0])]
        self.assertEqual([(r["kind"], r["note"]) for r in ledgered], alert, lines)

    def sweep(self, name: str, mode: str, standby: bool, own: bool) -> tuple:
        """The send NAME in a new deployment in MODE (and STANDBY), as the monitor's own or typed (OWN), nothing able to
        leave but through the fakes: its exit code and its lines, what left, the unsent items kept and the ledger's new
        lines."""
        self.deploy(mode, standby)
        argv, patches = self.arrange(name)
        ledger = self.store.ledger()
        with patches, test_mirror.nothing_leaves():
            code, out = self.rk(argv, own=own)
        return (
            code,
            out.splitlines(),
            self.sent,
            self.items(),
            self.store.ledger()[len(ledger) :],
        )

    def test_the_sweep_is_every_command_that_sends(self):
        self.assertEqual(
            sorted({name.split()[0] for name in self.NAMES} | set(self.NOT_SWEPT)),
            sorted(outward.names(gate=outward.OWN) + outward.names(gate=outward.GATED)),
        )

    def test_only_the_armed_pr_s_disarm_leaves_in_prototype_and_nothing_in_standby(
        self,
    ):
        for mode, standby, why in (
            ("prototype", False, state_store.PROTOTYPE),
            ("live", True, state_store.STANDBY),
        ):
            for name in self.NAMES:
                for own in (False, True):
                    with self.subTest(send=name, held=why, own=own):
                        self.held_as_it_should_be(
                            name, own, why, self.sweep(name, mode, standby, own)
                        )

    def test_a_kept_review_s_after_line_runs_each_as_the_monitor_s_own_or_not_at_all(
        self,
    ):
        """A review kept while review automation sends nothing (premises.finish, kept): its AFTER line `$RK ...` runs
        the send as the session's mark makes it the monitor's own, which keeps what it would send; one of a command the
        session may not run is not run, and its line says so. The kept review stands."""
        finish = premises.finish  # the sweep's spec-post puts a stand-in in its place
        for mode, standby, why in (
            ("prototype", False, state_store.PROTOTYPE),
            ("live", True, state_store.STANDBY),
        ):
            for name in (*self.NAMES, outward.TEAMS_TEST):
                with self.subTest(send=name, held=why):
                    self.deploy(mode, standby)
                    argv, patches = (
                        ([name], contextlib.ExitStack())
                        if name == outward.TEAMS_TEST
                        else self.arrange(name)
                    )
                    notes = self.home / "session" / "notes.md"
                    notes.write_text("The widget review's notes.\n")
                    spec = {
                        "ROUND": "fr",
                        "SAVE": "fr: the widget review kept",
                        "SRC": str(self.files["review"]),
                        "NOTES": str(notes),
                        "STOP": "",
                        "AFTER": [["$RK", *argv]],
                    }
                    ready = {
                        "copy": self.files["review"],
                        "sha": HEADS[7],
                        "dev": DEVELOP,
                    }
                    ledger, ran = self.store.ledger(), []

                    def run(cmd, env=None, **_kw):
                        if not env or cmd[0] != env.get("RK"):
                            raise AssertionError(f"an outward call: {cmd[:2]}")
                        # rk in the environment finish gives it, the session's: its mark is the session's.
                        with mock.patch.dict(os.environ, env, clear=True):
                            code, out = self.rk(cmd[1:], own=False)
                        ran.append((env.get(outward.OWN_VAR), code, out.splitlines()))
                        return subprocess.CompletedProcess(cmd, code, out + "\n", "")

                    with patches, test_mirror.nothing_leaves(), mock.patch.object(
                        premises.subprocess, "run", run
                    ), mock.patch.dict(os.environ, {outward.OWN_VAR: self.own}):
                        lines, ok = finish(
                            self.cfg, self.store, 7, spec, ready, kept=True
                        )
                    after = [ln for ln in lines if ln.startswith(("AFTER", "!! AFTER"))]
                    if not outward.row(argv[0]).session:
                        self.assertEqual(
                            (ran, after, self.sent, self.items(), ok),
                            (
                                [],
                                [
                                    f"AFTER not run: rk {argv[0]} is for op-login to run, or for rk itself, never "
                                    f"for the session: $RK {' '.join(argv)}"
                                ],
                                [],
                                [],
                                True,
                            ),
                            lines,
                        )
                        continue
                    ((mark, code, said),) = ran
                    self.assertEqual((mark, after, ok), (self.own, [], True), lines)
                    self.held_as_it_should_be(
                        name,
                        True,
                        why,
                        (
                            code,
                            said,
                            self.sent,
                            self.items(),
                            self.store.ledger()[len(ledger) :],
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
