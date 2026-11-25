# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tool-loop drafter (worker/drafter.py) on an offline PR (tests/drafter_fixture.py): the scripted model's tool
calls go through the real client, limiter and tools, and the draft, its evidence and its token records come out. Its
phases: the script's settling of the prior items (data, unchanged since our last review, closed, saved probes), one
short task per item left open, and the hunt beside them, each within its own caps; then the evidence check and
prior_settled.md. No socket is opened: the binding points at the closed port 127.0.0.1:9 and the client's `post` is the
fake endpoint."""

import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import drafter  # noqa: E402
import drafter_fixture as fx  # noqa: E402
import llm_client as lc  # noqa: E402
from rkmon_lib import bindings, evidence, prior_settled, registry  # noqa: E402

# The first round: I1 dev-b's approval (data), I2 dev-c's review, I3 Copilot's thread, I4 dev-a's reply (the author's).
PRIOR = {
    "prior:I2": [
        fx.call(
            "settle",
            status="OPEN",
            light="🟡",
            gist="helper has no test",
            point="no test calls helper (read)",
            file="proj/x.py",
            line=5,
            quote="def helper(v):",
        )
    ],
    "prior:I3": [
        fx.call("read_prior", id="I3"),
        fx.call(
            "settle",
            status="FIXED",
            gist="helper adds one by design",
            point="adds one on purpose (read)",
            file="proj/x.py",
            line=6,
            quote="return v + 1",
        ),
    ],
}
HUNT = [
    fx.call("diff_hunk", id="H1.1"),
    fx.call("read_lines", path="x.py", start=1, end=6),
    fx.call("grep_code", pattern="return 41"),
    fx.call(
        "add_finding",
        severity="🟠",
        file="proj/x.py",
        line=2,
        problem="answer returns 41, not 42",
        impact="every caller gets a wrong answer",
        fix="return 42",
        quote="return 41",
    ),
    [
        fx.call("record_evidence", id="N1", claim="grep finds it", run="$LAST_RUN"),
        fx.call(
            "add_finding",
            severity="yellow",
            file="proj/x.py",
            line=5,
            problem="helper has no test",
            impact="a regression goes unseen",
            fix="add a test",
        ),
    ],
    fx.call(
        "record_evidence",
        id="N2",
        claim="wrong line",
        file="proj/x.py",
        line=3,
        quote="def helper(v):",
    ),
    fx.call(
        "finish",
        reason="off by one",
        utility_light="🟡",
        utility="fixes nothing yet: the answer is wrong",
    ),
]
# The follow-up: I1 dev-b's approval, I2-I4 our last review's rows (N1, N2, the closed earlier items), I5 its
# finding N4 (a saved probe), I6 dev-a's reply naming N2 (the author's), I7 dev-d's review after ours.
FOLLOW = {
    "prior:I3": [
        fx.call(
            "settle",
            status="FIXED",
            gist="flags keeps fast",
            point="setdefault keeps it (read)",
            file="proj/y.py",
            line=3,
            quote="out.setdefault('fast', False)",
        )
    ],
    "prior:I7": [
        fx.call(
            "settle",
            status="OPEN",
            light="🟡",
            gist="keep z.py",
            point="z.py unchanged (read)",
            file="proj/z.py",
            line=1,
            quote="NOTES = 'unchanged'",
        )
    ],
}


class TestSettings(unittest.TestCase):
    def test_the_drafter_s_settings_are_the_package_s_entry_by_entry(self):
        """The drafter's settings are config.defaults.json's (config drafter): an entry a configuration gives
        replaces that entry alone, a blank one takes the package's, the limiter's folder is under the working
        folder while it is blank, and a name the package's map does not have is not a setting.
        """
        package = registry.default("drafter")
        self.assertIn("review_wall_min", package)
        cfg = {
            "drafter": {"review_wall_min": 50, "hunt_steps": ""},
            "paths": {"root": "/w"},
        }
        self.assertEqual(
            bindings.drafter_settings(cfg),
            dict(package, review_wall_min=50, limiter_dir="/w/llm_limits"),
        )
        cfg["drafter"] = {"speed": 1}
        self.assertIn(
            "drafter.speed is not a setting; the settings are " + ", ".join(package),
            bindings.problems(cfg),
        )

    def test_the_readers_ask_the_package(self):
        """The drafter's settings and their check read the package's map itself, no copy of their own: a
        package map with another value and another entry gives them."""
        real = registry.default

        def other(name):
            if name != "drafter":
                return real(name)
            return dict(real(name), review_wall_min=7, other_entry=1)

        cfg = {"drafter": {"other_entry": 2}, "paths": {"root": "/w"}}
        with mock.patch.object(registry, "default", other):
            got = bindings.drafter_settings(cfg)
            said = bindings.problems(cfg)
        self.assertEqual((got["review_wall_min"], got["other_entry"]), (7, 2))
        self.assertEqual([s for s in said if s.startswith("drafter.other_entry")], [])


class Draft(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = fx.make(pathlib.Path(self.tmp.name))
        self.cfg = self.fx["cfg"]
        self.ws = self.fx["ws"]

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, endpoint, **drafter_over) -> drafter.Session:
        cfg = dict(self.cfg, drafter=dict(self.cfg["drafter"], **drafter_over))
        s = bindings.client_settings(cfg, "drafter")

        def chat(s_, msgs, tools, effort, limiter=None, max_tokens=None):
            return lc.chat(
                s_,
                msgs,
                tools,
                effort,
                limiter=limiter,
                max_tokens=max_tokens,
                post=endpoint,
                sleep=lambda _s: None,
            )

        return drafter.Session(self.ws, 7, cfg, s, chat=chat)

    def read(self, name):
        return (self.ws / name).read_text(encoding="utf-8")

    def records(self, name):
        return [json.loads(x) for x in self.read(name).splitlines() if x.strip()]

    def follow_up(self):
        self.fx = fx.follow_up(self.fx)


class TestFirstRound(Draft):
    def test_the_draft_its_rows_settlers_and_evidence(self):
        ep = fx.FakeEndpoint(dict(PRIOR, hunt=HUNT))
        self.assertTrue(self.session(ep).run_draft())
        text = self.read("review_agent.md")
        lines = text.splitlines()
        self.assertEqual(lines[0], fx.LINE1)
        self.assertEqual(
            lines[2],
            f"## PR #7 — make the answer right (dev-a, {fx.TAG}-123) — 🟠 First-round: off by one",
        )
        self.assertIn("| Prior item | Status | Settled by | Evidence |", text)
        self.assertIn(
            "| dev-b (11-01) APPROVED at `abcdef12` | 🟢 APPROVED `abcdef12`<br>• older head; still counts on "
            "develop | script: data | P1: reviews.md:4 (staged) |",
            text,
        )
        self.assertIn(
            "| dev-c (11-01): helper has no test | 🟡 OPEN<br>• no test calls helper (read) | cand | "
            "P2: x.py:5 (head) |",
            text,
        )
        self.assertIn(
            "| Copilot (11-01): helper adds one by design | 🟢 FIXED<br>• adds one on purpose (read) | cand | "
            "P3: x.py:6 (head) |",
            text,
        )
        self.assertNotIn(
            "Rebased on develop", text
        )  # the author's reply holds no review point
        self.assertNotIn("therock-pr-bot", text)  # a bot's comment is no prior item
        self.assertIn(
            "| Threads | 🟡 1 open of 1<br>• open: `x.py:5` @Copilot<br>• 0 resolved, 0 outdated (read) | "
            "script: data | C4 |",
            text,
        )
        self.assertIn(
            "| Bots / CI | 🔴 Red<br>• fails: build (gfx942) — cause not checked<br>• 1 ✓ (read)<br>• merge: "
            "mergeable (read) | script: data | C3 |",
            text,
        )
        self.assertIn(
            f"| Our checks | 🟢 Clean<br>• merge vs `{self.fx['base'][:10]}`: clean (ran)<br>• rules 1–3 "
            "clean (ran)<br>• build, tests and goldens: the checks job's | script: data | C1, C2 |",
            text,
        )
        self.assertIn(
            "| N1 | 🟠 | `x.py:2` | **answer returns 41, not 42**<br>• impact: every caller gets a wrong "
            "answer<br>• fix: return 42 |",
            text,
        )
        self.assertIn("| N2 | 🟡 | `x.py:5` | **helper has no test**", text)
        self.assertIn(
            "**Utility:** 🟡 Mixed — fixes nothing yet: the answer is wrong", text
        )
        self.assertIn(
            "_First-round review: first pass by Cand-1, verified by the orchestrator (the Reference Model "
            "1.0).",
            text,
        )
        # The evidence: the script's claims, its data rows, the tasks' quotes, the hunt's quotes and runs, and a "none"
        # for the claim without any.
        ev = {(e["id"], e["kind"]) for e in self.records("evidence.jsonl")}
        self.assertEqual(
            ev,
            {
                ("C1", "ran"),
                ("C2", "ran"),
                ("C3", "read"),
                ("C4", "data"),
                ("P1", "read"),
                ("P2", "read"),
                ("P3", "read"),
                ("N1", "read"),
                ("N1", "ran"),
                ("N2", "none"),
            },
        )
        runs = self.records("runs.jsonl")
        self.assertEqual(
            [(r["run"], r["tool"]) for r in runs],
            [("R1", "merge_conflicts"), ("R2", "rule_scan"), ("R3", "grep_code")],
        )
        log = self.read("agent_log.txt")
        self.assertIn("REJECTED: the quote is not at proj/x.py:3 (head)", log)
        self.assertIn("the quote is at line 5", log)
        found = json.loads(self.read("draft_findings.json"))
        self.assertEqual((found["method"], found["role"]), ("phased", "drafter"))
        self.assertEqual(
            [(f["id"], f["severity"]) for f in found["findings"]],
            [("N1", "🟠"), ("N2", "🟡")],
        )
        self.assertEqual(
            [(p["id"], p["settler"], p["how"], p["flip"]) for p in found["priors"]],
            [
                ("P1", "script", "data", False),
                ("P2", "model", "task", False),
                ("P3", "model", "task", False),
            ],
        )
        self.assertEqual(
            found["checks"],
            {
                "C1": "Our checks",
                "C2": "Our checks",
                "C3": "Bots / CI",
                "C4": "Threads",
            },
        )
        self.assertEqual(
            found["skipped"], {"I4": "the PR author's reply or status note"}
        )
        # Each phase in draft_run.json, with its requests and tokens.
        run = json.loads(self.read("draft_run.json"))
        self.assertEqual((run["method"], run["stopped"]), ("phased", ""))
        self.assertEqual(
            (
                run["passes"]["settle"]["script"],
                run["passes"]["prior"]["tasks"],
                run["passes"]["prior"]["requests"],
                run["passes"]["hunt"]["requests"],
            ),
            (1, 2, 3, 7),
        )
        tot = json.loads(self.read("tokens_total.json"))
        a = tot["attempts"][-1]
        self.assertEqual(
            (a["requests"], a["cached_tokens"], a["reasoning_tokens"]), (10, 1000, 800)
        )
        self.assertIn("prompt_tokens_details.cached_tokens", a["usage_keys"])
        self.assertEqual(set(a["passes"]), {"prior", "hunt"})
        self.assertIn(
            "| P1 | dev-b (11-01) APPROVED at `abcdef12` | 🟢 APPROVED `abcdef12` | data |",
            self.read("prior_leads.md"),
        )
        self.assertIn(
            "No review of ours is staged: our earlier items are not among these rows.",
            self.read("prior_leads.md"),
        )

    def test_lean_requests(self):
        # the hunt asks at the first review pass's effort (bindings.drafter_effort), the prior tasks at their own
        self.cfg = dict(self.cfg, effort={"first_pass": "low"})
        ep = fx.FakeEndpoint(dict(PRIOR, hunt=HUNT))
        self.assertTrue(self.session(ep).run_draft())
        by = {}
        for b in ep.bodies:
            by.setdefault(fx.FakeEndpoint.name_of(b).split(":")[0], []).append(b)
        self.assertEqual({b.get("reasoning_effort") for b in by["prior"]}, {"medium"})
        self.assertEqual({b.get("reasoning_effort") for b in by["hunt"]}, {"low"})
        for b in ep.bodies:
            self.assertLessEqual(
                len(b["messages"][1]["content"]), 3000 * lc.CHARS_PER_TOKEN
            )  # the task card
            self.assertTrue(
                b["stream"] and b["stream_options"] == {"include_usage": True}
            )
            self.assertNotIn(fx.KEY, json.dumps(b))
        # The hunt shows only the last four tool results; older ones are masked behind a note of the call and its round.
        last = by["hunt"][-1]["messages"]
        tools = [m for m in last if m["role"] == "tool"]
        self.assertEqual(
            sum(1 for m in tools if not m["content"].startswith("[hidden: ")), 4
        )
        self.assertTrue(
            tools[0]["content"].startswith("[hidden: diff_hunk(H1.1) from round 0, "),
            tools[0],
        )
        self.assertEqual(
            set(tools[0]), {"role", "tool_call_id", "content"}
        )  # the note's own keys are not sent
        # The hunt's card lists the PR map and the prior items, not the diff; a task's card holds its item and code.
        hunt = by["hunt"][0]["messages"][1]["content"]
        self.assertIn(
            "F1 x.py +2/-2: H1.1 @" + "1+2/-2", hunt
        )  # in two pieces: no @handle in the source
        self.assertIn(
            "Copilot (11-01): helper(v) changed its result; is that intended?", hunt
        )
        self.assertNotIn("diff --git", hunt)
        task = next(b for b in by["prior"] if fx.FakeEndpoint.name_of(b) == "prior:I3")[
            "messages"
        ][1]["content"]
        self.assertIn(
            f"Item I3: an inline thread at `x.py:5` [OPEN] by {fx.AT}copilot-pull-request-reviewer (11-01):",
            task,
        )
        self.assertIn("     6+     return v + 1", task)  # the hunt the thread sits on
        self.assertIn("You have 3 rounds; the last offers only settle.", task)
        self.assertEqual(
            {t["function"]["name"] for t in by["prior"][-1]["tools"]}
            <= set(drafter.Session.MENUS["prior"]),
            True,
        )
        self.assertEqual(
            ep.keys[0], fx.KEY
        )  # the key reaches the request, and only the request

    def test_the_key_is_never_written(self):
        ep = fx.FakeEndpoint(dict(PRIOR, hunt=HUNT))
        self.session(ep).run_draft()
        for p in list(self.ws.rglob("*")) + list(
            (self.fx["root"] / "limits").rglob("*")
        ):
            if p.is_file():
                self.assertNotIn(
                    fx.KEY, p.read_text(encoding="utf-8", errors="replace"), p
                )


class TestFollowUp(Draft):
    def test_the_script_settles_what_needs_no_model(self):
        self.follow_up()
        then, now = self.fx["then"][:10], self.fx["head"][:10]
        ep = fx.FakeEndpoint(dict(FOLLOW, hunt=HUNT[3:4] + HUNT[-1:]))
        self.assertTrue(self.session(ep).run_draft())
        text = self.read("review_agent.md")
        self.assertIn(
            "| dev-b (11-01) APPROVED at `abcdef12` | 🟢 APPROVED `abcdef12`<br>• older head; still counts on "
            "develop | script: data | P1: reviews.md:4 (staged) |",
            text,
        )
        # N1's file is as it was at our last review's head: it carries, with its blobs as evidence.
        self.assertIn(
            f"| Ours (11-01) N1: `answer` returns 41 | 🟠 OPEN<br>• unchanged since `{then}`: x.py (read)"
            f"<br>• still at `x.py:2` (read) | script: unchanged | P2: unchanged since `{then}` |",
            text,
        )
        # N2's file changed (and the author names it since): a task settles it.
        self.assertIn(
            "| Ours (11-01) N2: `flags` drops the fast flag | 🟢 FIXED<br>• setdefault keeps it (read) | cand "
            "| P3: y.py:3 (head) |",
            text,
        )
        # A row our last review closed that cites no file carries closed.
        self.assertIn(
            "| Earlier items: ours N3; dev-c's nits | 🟢 FIXED<br>• closed in our last review; cites no file "
            "(read) | script: closed | P4: ours.md:15 (staged) |",
            text,
        )
        # Our N4 has a saved probe: its verdict settles it.
        self.assertIn(
            "| Ours (11-02) N4: helper adds one | 🟠 OPEN<br>• probe N4_helper: REPRODUCES: helper adds one "
            "(ran) | script: probe | P5: R3 probe |",
            text,
        )
        self.assertIn(
            "| dev-d (11-02): keep z.py | 🟡 OPEN<br>• z.py unchanged (read) | cand | P7: z.py:1 (head) |",
            text,
        )
        self.assertNotIn(
            "| dev-c (11-01)", text
        )  # before our last review: its rows cover it
        self.assertNotIn("N2 is fixed in the last push", text)  # the author's reply
        ev = {
            e["id"]: e
            for e in self.records("evidence.jsonl")
            if e["id"].startswith("P")
        }
        self.assertEqual(ev["P2"]["kind"], "unchanged")
        x = ev["P2"]["files"]["proj/x.py"]
        self.assertEqual(
            (ev["P2"]["since"][:10], ev["P2"]["head"][:10], x[0] == x[1], x[2] == x[3]),
            (then, now, True, True),
        )
        self.assertEqual((ev["P5"]["kind"], ev["P5"]["tool"]), ("ran", "probe"))
        merged = json.loads(self.read("draft_run.json"))["merged"]
        self.assertEqual(
            self.read("probes.txt"),
            f"# probes of pr7 at {merged}: N4_helper\n"
            "N4_helper: REPRODUCES: helper adds one\n",
        )
        # The tasks ran only for the items left open; the hunt looks at the delta first.
        self.assertEqual(
            sorted({fx.FakeEndpoint.name_of(b) for b in ep.bodies}),
            ["hunt", "prior:I3", "prior:I7"],
        )
        hunt = next(b for b in ep.bodies if fx.FakeEndpoint.name_of(b) == "hunt")[
            "messages"
        ][1]["content"]
        self.assertIn("Changed since our last review (Δ): F2", hunt)
        self.assertIn("F2 Δ y.py +1/-1: H2.1", hunt)
        self.assertLess(hunt.index("F2 Δ"), hunt.index("F1 x.py"))
        task = next(b for b in ep.bodies if fx.FakeEndpoint.name_of(b) == "prior:I3")[
            "messages"
        ][1]["content"]
        self.assertIn(
            f"Our last review (at `{then}`) said: 🟠 OPEN; `y.py:3` pops it (read)",
            task,
        )
        self.assertIn(
            f"Since our last review, {fx.AT}dev-a (11-02): N2 is fixed in the last push.",
            task,
        )
        self.assertIn("The change to y.py since our last review:", task)
        self.assertIn("+    out.setdefault('fast', False)", task)
        leads = self.read("prior_leads.md")
        for row in ("| P1 |", "| P2 |", "| P4 |", "| P5 |"):
            self.assertIn(row, leads)
        self.assertNotIn("| P3 |", leads)  # a task's, not the script's
        self.assertIn(
            f"Our last review: at `{then}`, posted 2026-11-02T09:00:00Z; develop then `",
            leads,
        )
        # The evidence check passes, and prior_settled.md asks for a look at the 🟠 that closed.
        lines, code = evidence.cmd(self.cfg, 7)
        self.assertEqual(code, 0, lines)
        ps = prior_settled.load(self.ws)
        self.assertEqual(
            {r["id"]: r["group"] for r in ps["rows"]},
            {
                "P1": "settled",
                "P2": "settled",
                "P3": "look",
                "P4": "settled",
                "P5": "settled",
                "P7": "settled",
            },
        )
        self.assertEqual(
            ps["counts"],
            {
                "rows": 6,
                "script": 4,
                "script_pass": 4,
                "model": 2,
                "model_pass": 2,
                "look": 1,
                "unsettled": 0,
            },
        )
        self.assertIn(
            ps["spot"], {"P2", "P4", "P5", "P7"}
        )  # an approval only when nothing else settled
        md = self.read("prior_settled.md")
        self.assertIn(
            "| P3 | Ours (11-01) N2: `flags` drops the fast flag | 🟢 FIXED | 🟠 OPEN | flips a 🔴/🟠: 🟠 OPEN "
            "→ 🟢 FIXED |",
            md,
        )
        self.assertIn(f"Spot-check one: **{ps['spot']}**", md)
        self.assertIn(
            f"Our last review: at `{then}` (posted 2026-11-02T09:00:00Z).", md
        )

    def test_a_review_kept_unposted_is_our_last_review(self):
        """While review posting is off our last review can be one kept unposted (`rk stage-ours` stages it so): its rows
        are prior items as a post's are, and prior_leads.md and prior_settled.md name it kept, at the time it was kept.
        """
        self.follow_up()
        then = self.fx["then"][:10]
        ours = self.ws / "pr" / "ours.md"
        ours.write_text(
            ours.read_text(encoding="utf-8")
            .replace("# Our last review post on", "# Our last review on")
            .replace("\n- Posted: ", "\n- Kept unposted: "),
            encoding="utf-8",
        )
        ep = fx.FakeEndpoint(dict(FOLLOW, hunt=HUNT[3:4] + HUNT[-1:]))
        self.assertTrue(self.session(ep).run_draft())
        self.assertIn(
            f"| Ours (11-01) N1: `answer` returns 41 | 🟠 OPEN<br>• unchanged since `{then}`: x.py (read)",
            self.read("review_agent.md"),
        )
        self.assertIn(
            f"Our last review: at `{then}`, kept unposted 2026-11-02T09:00:00Z; develop then `",
            self.read("prior_leads.md"),
        )
        lines, code = evidence.cmd(self.cfg, 7)
        self.assertEqual(code, 0, lines)
        self.assertIn(
            f"Our last review: at `{then}` (kept unposted 2026-11-02T09:00:00Z).",
            self.read("prior_settled.md"),
        )

    def test_a_probe_with_no_verdict_leaves_its_item_to_the_next_rule(self):
        # A probe that cannot show its finding on this tree (exit 2) settles nothing: N4's file is unchanged since our
        # last review, so it carries; once it cites a changed file, a task settles it.
        self.follow_up()
        (self.fx["root"] / "probes" / "pr7" / "N4_helper.sh").write_text(
            "echo 'cannot show it here'\nexit 2\n"
        )
        settle = {
            "prior:I5": [
                fx.call(
                    "settle",
                    status="OPEN",
                    light="🟠",
                    point="still adds one (read)",
                    file="proj/x.py",
                    line=6,
                    quote="return v + 1",
                )
            ]
        }
        self.assertTrue(
            self.session(
                fx.FakeEndpoint(dict(FOLLOW, hunt=HUNT[-1:], **settle))
            ).run_draft()
        )
        self.assertIn(
            "| Ours (11-02) N4: helper adds one | 🟠 OPEN<br>• unchanged since",
            self.read("review_agent.md"),
        )
        ours = self.ws / "pr" / "ours.md"
        ours.write_text(
            ours.read_text(encoding="utf-8").replace(
                "| N4 | 🟠 | `x.py:6` |", "| N4 | 🟠 | `y.py:3` |"
            ),
            encoding="utf-8",
        )
        ep = fx.FakeEndpoint(dict(FOLLOW, hunt=HUNT[-1:], **settle))
        self.assertTrue(self.session(ep).run_draft())
        self.assertIn(
            "| Ours (11-02) N4: helper adds one | 🟠 OPEN<br>• still adds one (read) | cand | P5: x.py:6 "
            "(head) |",
            self.read("review_agent.md"),
        )
        task = next(b for b in ep.bodies if fx.FakeEndpoint.name_of(b) == "prior:I5")[
            "messages"
        ][1]["content"]
        self.assertIn("Its saved probes settled nothing: N4_helper: ERROR rc=2", task)


class TestPhases(Draft):
    def test_the_prior_tasks_and_the_hunt_run_at_once(self):
        started, seen = threading.Event(), []

        def prior():
            seen.append(
                started.wait(10)
            )  # a task's request waits for the hunt's first one, which runs beside it

        ep = fx.FakeEndpoint(
            dict(PRIOR, hunt=HUNT), before={"prior": prior, "hunt": started.set}
        )
        t0 = time.monotonic()
        self.assertTrue(self.session(ep).run_draft())
        self.assertTrue(seen and all(seen), seen)
        self.assertLess(time.monotonic() - t0, 10)

    def test_each_phase_keeps_its_own_wall_time(self):
        # The tasks' first requests take longer than the prior phase's whole time: their second ones are refused, and
        # their items are not settled; the hunt, with its own time, finishes.
        slow = {
            "prior:I2": [fx.call("read_prior", id="I2")],
            "prior:I3": [fx.call("read_prior", id="I3")],
        }
        ep = fx.FakeEndpoint(
            dict(slow, hunt=HUNT), before={"prior": lambda: time.sleep(0.5)}
        )
        sess = self.session(ep, prior_wall_min=0.005)
        self.assertTrue(sess.run_draft())
        text = self.read("review_agent.md")
        self.assertIn(
            "| dev-c (11-01): The helper should get a test. | ⚪ NOT SETTLED<br>• stopped: the prior phase's "
            "wall time (0 s) is spent | — | P2: none |",
            text,
        )
        self.assertIn("| N1 | 🟠 | `x.py:2` |", text)
        run = json.loads(self.read("draft_run.json"))
        self.assertEqual(
            (
                run["passes"]["prior"]["status"],
                run["passes"]["hunt"]["status"],
                run["stopped"],
            ),
            ("2 item(s) not settled", "ok", ""),
        )

    def test_each_phase_keeps_its_own_token_budget(self):
        slow = {
            "prior:I2": [fx.call("read_prior", id="I2")],
            "prior:I3": [fx.call("read_prior", id="I3")],
        }
        ep = fx.FakeEndpoint(dict(slow, hunt=HUNT))
        self.assertTrue(self.session(ep, prior_token_budget=1500).run_draft())
        text = self.read("review_agent.md")
        self.assertIn("the prior phase's token budget (1500) would be passed", text)
        self.assertIn("| N1 | 🟠 | `x.py:2` |", text)  # the hunt keeps its own
        self.assertEqual(
            json.loads(self.read("draft_run.json"))["passes"]["hunt"]["status"], "ok"
        )

    def test_a_task_has_at_most_its_steps_and_its_last_offers_only_settle(self):
        idle = {"prior:I2": [fx.call("read_prior", id="I2")] * 5}
        ep = fx.FakeEndpoint(dict(idle, hunt=HUNT[-1:]))
        self.assertTrue(self.session(ep).run_draft())
        mine = [b for b in ep.bodies if fx.FakeEndpoint.name_of(b) == "prior:I2"]
        self.assertEqual(len(mine), 3)
        self.assertEqual([t["function"]["name"] for t in mine[-1]["tools"]], ["settle"])
        self.assertIn(
            "| dev-c (11-01): The helper should get a test. | ⚪ NOT SETTLED<br>• no verdict in its 3 "
            "rounds | — | P2: none |",
            self.read("review_agent.md"),
        )
        self.assertIn(
            "refused: read_prior is not available now (available: settle)",
            self.read("agent_log.txt"),
        )

    def test_the_hunt_finishes_only_with_evidence_for_every_serious_finding(self):
        hunt = [
            fx.call(
                "add_finding",
                severity="🟠",
                file="proj/x.py",
                line=2,
                problem="answer returns 41",
                impact="wrong answer",
                fix="return 42",
            ),
            fx.call(
                "finish",
                reason="off by one",
                utility_light="🟡",
                utility="wrong answer",
            ),
            fx.call(
                "record_evidence",
                id="N1",
                claim="returns 41",
                file="proj/x.py",
                line=2,
                quote="return 41",
            ),
            fx.call(
                "finish",
                reason="off by one",
                utility_light="🟡",
                utility="wrong answer",
            ),
        ]
        ep = fx.FakeEndpoint(dict(PRIOR, hunt=hunt))
        self.assertTrue(self.session(ep).run_draft())
        log = self.read("agent_log.txt")
        self.assertIn("REJECTED: N1 has no evidence yet", log)
        self.assertIn("-> finished", log)
        self.assertIn(
            "**Utility:** 🟡 Mixed — wrong answer", self.read("review_agent.md")
        )

    def test_every_yellow_blocker_is_kept_no_cap(self):
        """Scope first: every blocker needed to merge is listed, each lit, a second 🟡 too."""
        hunt = [
            fx.call(
                "add_finding",
                severity="🟡",
                file="proj/x.py",
                line=2,
                problem=f"minor {i}",
                impact="docs drift",
                fix="fix it",
                quote="return 41",
            )
            for i in (1, 2)
        ] + [fx.call("finish", reason="two minors", utility_light="🟡", utility="minor")]
        ep = fx.FakeEndpoint(dict(PRIOR, hunt=hunt))
        self.assertTrue(self.session(ep).run_draft())
        self.assertNotIn("at most one", self.read("agent_log.txt"))
        draft = self.read("review_agent.md")
        self.assertIn("| N1 | 🟡 |", draft)
        self.assertIn("| N2 | 🟡 |", draft)

    def test_rounds_spent_end_in_a_low_effort_wrap_up(self):
        hunt = [fx.call("grep_code", pattern="answer")] * 3 + [
            fx.call(
                "add_finding",
                severity="🟠",
                file="proj/x.py",
                line=2,
                problem="answer returns 41",
                impact="wrong answer",
                fix="return 42",
                quote="return 41",
            ),
            fx.call(
                "finish",
                reason="off by one",
                utility_light="🟡",
                utility="wrong answer",
            ),
        ]
        ep = fx.FakeEndpoint(dict(PRIOR, hunt=hunt))
        self.assertTrue(self.session(ep, hunt_steps=3).run_draft())
        mine = [b for b in ep.bodies if fx.FakeEndpoint.name_of(b) == "hunt"]
        self.assertEqual([b.get("reasoning_effort") for b in mine][-2:], ["low", "low"])
        self.assertEqual(
            {t["function"]["name"] for t in mine[-1]["tools"]},
            {"add_finding", "record_evidence", "drop_finding", "finish"},
        )
        self.assertIn("| N1 | 🟠 |", self.read("review_agent.md"))

    def test_the_run_token_budget_stops_the_draft_and_keeps_what_it_recorded(self):
        ep = fx.FakeEndpoint(dict(PRIOR, hunt=HUNT))
        sess = self.session(
            ep, review_token_budget=5000, prior_token_budget=0, hunt_token_budget=0
        )
        self.assertTrue(sess.run_draft())
        head = json.loads(self.read("draft_run.json"))
        self.assertTrue(
            head["stopped"].startswith(
                "stopped: the run's token budget (5000) would be passed"
            ),
            head,
        )
        self.assertTrue(head["draft"])
        self.assertTrue(self.read("review_agent.md").startswith(fx.LINE1))

    def test_no_answer_no_draft(self):
        def refuse(s, body, key):
            raise lc.GatewayError("HTTP 401: " + lc.explain(401), 401)

        sess = self.session(refuse)
        self.assertFalse(sess.run_draft())
        self.assertFalse((self.ws / "review_agent.md").exists())
        head = json.loads(self.read("draft_run.json"))
        self.assertFalse(head["draft"])
        self.assertIn("HTTP 401", head["passes"]["hunt"]["status"])
        self.assertEqual(
            json.loads(self.read("tokens_total.json"))["attempts"][-1]["requests"], 0
        )

    def test_a_second_attempt_starts_clean_and_adds_up_its_tokens(self):
        self.session(fx.FakeEndpoint(dict(PRIOR, hunt=HUNT))).run_draft()
        self.session(fx.FakeEndpoint(dict(PRIOR, hunt=HUNT))).run_draft()
        tot = json.loads(self.read("tokens_total.json"))
        self.assertEqual([a["attempt"] for a in tot["attempts"]], [1, 2])
        self.assertEqual(tot["total"]["requests"], 20)
        self.assertEqual(
            len([e for e in self.records("evidence.jsonl") if e["id"] == "N1"]), 2
        )  # this attempt's


class TestEvidenceCheck(Draft):
    def test_pass_fail_and_the_marks_in_the_draft(self):
        self.session(fx.FakeEndpoint(dict(PRIOR, hunt=HUNT))).run_draft()
        lines, code = evidence.cmd(self.cfg, 7)
        self.assertEqual(code, 0, lines)
        report = json.loads(self.read("evidence_check.json"))
        by = {c["id"]: c for c in report["claims"]}
        self.assertEqual(
            {k: v["result"] for k, v in by.items()},
            {
                "N1": "PASS",
                "P1": "PASS",
                "P2": "PASS",
                "P3": "PASS",
                "C1": "PASS",
                "C2": "PASS",
                "C3": "PASS",
                "C4": "PASS",
                "N2": "MISSING",
            },
        )
        text = self.read("review_agent.md")
        self.assertIn(
            "**helper has no test**<br>• impact: a regression goes unseen<br>• fix: add a test<br>• "
            "**evidence check MISSING**: no evidence recorded |",
            text,
        )
        kept = pathlib.Path(lines[-1].split("kept in ", 1)[1])
        for name in (
            "evidence_check.json",
            "review_agent.md",
            "prior_settled.md",
            "prior_leads.md",
        ):
            self.assertTrue((kept / name).is_file(), name)
        # A quote that moved, and a recorded command that is not the tool's: FAIL, and marked once per check.
        recs = self.records("evidence.jsonl")
        for e in recs:
            if e["id"] == "N1" and e["kind"] == "read":
                e["line"] = 1
        (self.ws / "evidence.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in recs), encoding="utf-8"
        )
        runs = self.records("runs.jsonl")
        runs[0]["argv"] = runs[0]["argv"] + ["--extra"]
        (self.ws / "runs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8"
        )
        for _ in range(2):
            lines, code = evidence.cmd(self.cfg, 7)
        self.assertEqual(code, 1)
        by = {
            c["id"]: c for c in json.loads(self.read("evidence_check.json"))["claims"]
        }
        self.assertEqual(by["N1"]["result"], "FAIL")
        self.assertIn(
            "the quote is not at proj/x.py:1 (it is at line 2)", by["N1"]["why"]
        )
        self.assertEqual(
            (by["C1"]["result"], by["C1"]["why"]),
            ("FAIL", "the recorded command is not the one the " "tool builds"),
        )
        text = self.read("review_agent.md")
        self.assertEqual(
            text.count("**evidence check FAIL**"), 2
        )  # N1 and the Our checks row (C1, C2)
        self.assertEqual(text.count("**evidence check MISSING**"), 1)

    def test_the_scripts_evidence_is_read_again(self):
        self.follow_up()
        self.session(fx.FakeEndpoint(dict(FOLLOW, hunt=HUNT[-1:]))).run_draft()
        self.assertEqual(evidence.cmd(self.cfg, 7)[1], 0)
        by = {
            c["id"]: c for c in json.loads(self.read("evidence_check.json"))["claims"]
        }
        self.assertEqual(
            (by["P2"]["result"], by["P5"]["result"], by["C4"]["result"]),
            ("PASS", "PASS", "PASS"),
        )
        self.assertIn("the probe and its tree are the ones that ran", by["P5"]["why"])
        # A recorded blob that is not the clone's, a staged fact that reads otherwise, a probe edited since it ran.
        recs = self.records("evidence.jsonl")
        for e in recs:
            if e["kind"] == "unchanged":
                e["files"]["proj/x.py"][1] = "0" * 40
        (self.ws / "evidence.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in recs), encoding="utf-8"
        )
        with open(self.ws / "pr" / "threads.md", "a", encoding="utf-8") as f:
            f.write(
                f"\n## proj/y.py:2 [RESOLVED]\n### {fx.AT}dev-e (2026-11-02T13:00:00Z)\nok\n"
            )
        with open(
            self.fx["root"] / "probes" / "pr7" / "N4_helper.sh", "a", encoding="utf-8"
        ) as f:
            f.write("# edited\n")
        lines, code = evidence.cmd(self.cfg, 7)
        self.assertEqual(code, 1)
        by = {
            c["id"]: c for c in json.loads(self.read("evidence_check.json"))["claims"]
        }
        self.assertEqual(
            by["P2"]["why"], "the blobs of proj/x.py are not the recorded ones"
        )
        self.assertIn("threads.md reads", by["C4"]["why"])
        self.assertEqual(by["P5"]["why"], "the saved probe changed since it ran")
        self.assertEqual(
            {
                r["id"]
                for r in prior_settled.load(self.ws)["rows"]
                if r["group"] == "look"
            },
            {"P2", "P3", "P5"},
        )

    def test_nothing_to_check(self):
        self.assertEqual(evidence.cmd(self.cfg, 8)[1], 2)
        self.assertEqual(
            evidence.cmd(self.cfg, 7, str(self.fx["root"] / "nowhere"))[1], 2
        )


class TestEntryPoint(Draft):
    def test_main_drafts_with_the_role_binding_and_the_configured_verifier(self):
        """`drafter.py <folder> <N>`: the configuration's binding for the role, line 1 for it, and the verifier the
        footer names (premises.verifier_name: the audit model in words)."""
        from unittest import mock
        from rkmon_lib import premises

        seen = {}

        class Fake:
            def __init__(self, ws, n, cfg, s, **kw):
                seen.update(cfg=cfg, s=s, kw=kw)
                self.stopped = ""

            def run_draft(self):
                return True

        with mock.patch(
            "rkmon_lib.store.load_config", lambda: dict(self.cfg)
        ), mock.patch.object(drafter, "Session", Fake), mock.patch("builtins.print"):
            self.assertEqual(drafter.main([str(self.ws), "7"]), 0)
        self.assertEqual(
            seen["cfg"]["verifier_name"], "the Reference Model 1.0"
        )  # the fixture names it
        cfg = {k: v for k, v in self.cfg.items() if k != "verifier_name"}
        with mock.patch(
            "rkmon_lib.store.load_config", lambda: dict(cfg)
        ), mock.patch.object(drafter, "Session", Fake), mock.patch("builtins.print"):
            self.assertEqual(drafter.main([str(self.ws), "7"]), 0)
        self.assertEqual(seen["cfg"]["verifier_name"], premises.verifier_name(cfg))
        with mock.patch(
            "rkmon_lib.store.load_config", lambda: dict(cfg, verifier_name="")
        ), mock.patch.object(drafter, "Session", Fake), mock.patch("builtins.print"):
            self.assertEqual(drafter.main([str(self.ws), "7"]), 0)
        self.assertEqual(
            seen["cfg"]["verifier_name"], premises.verifier_name(cfg)
        )  # a blank one too
        self.assertEqual(
            (seen["s"]["binding"], seen["kw"]["role"]), ("cand", "drafter")
        )

    def test_a_config_file_is_read_whole_or_refused(self):
        """--config: its keys replace the configuration's; a file that cannot be read, is not valid JSON or holds no
        JSON object is refused in plain words, exit 4, and nothing is drafted."""
        import contextlib
        import io
        from unittest import mock

        seen = {}

        class Fake:
            def __init__(self, ws, n, cfg, s, **kw):
                seen.update(cfg=cfg)
                self.stopped = ""

            def run_draft(self):
                return True

        files = {
            "none.json": None,
            "broken.json": "{",
            "list.json": "[]",
            "over.json": json.dumps({"verifier_name": "the Override Model"}),
        }
        for name, text in files.items():
            if text is not None:
                (self.ws / name).write_text(text)
        said = {}
        for name in files:
            seen.clear()
            out = io.StringIO()
            with mock.patch(
                "rkmon_lib.store.load_config", lambda: dict(self.cfg)
            ), mock.patch.object(drafter, "Session", Fake), contextlib.redirect_stdout(
                out
            ):
                code = drafter.main(
                    [str(self.ws), "7", "--config", str(self.ws / name)]
                )
            said[name] = (
                code,
                out.getvalue() if code else seen["cfg"]["verifier_name"],
            )
        f = self.ws
        self.assertEqual(
            said,
            {
                "none.json": (
                    4,
                    f"REFUSED: the --config file {f}/none.json cannot be read: No such file or directory\n",
                ),
                "broken.json": (
                    4,
                    f"REFUSED: the --config file {f}/broken.json is not valid JSON: Expecting property name "
                    "enclosed in double quotes: line 1 column 2 (char 1)\n",
                ),
                "list.json": (
                    4,
                    f"REFUSED: the --config file {f}/list.json is no JSON object\n",
                ),
                "over.json": (0, "the Override Model"),
            },
        )


class TestProfile(unittest.TestCase):
    def test_the_profile_is_the_one_in_profile_dir_and_a_blank_one_is_the_generic_profile(
        self,
    ):
        home = drafter.load_profile({"profile_dir": str(fx.PROFILE)})
        self.assertEqual(
            (home["dir"], home["project"], home["checks_note"]),
            (
                str(fx.PROFILE),
                json.loads((fx.PROFILE / "drafter.json").read_text(encoding="utf-8"))[
                    "project"
                ],
                "build, tests and goldens: the checks job's",
            ),
        )
        self.assertNotEqual(home["rules"], drafter.GENERIC_RULES)
        generic = drafter.load_profile({"profile_dir": ""})
        self.assertEqual(
            (generic["dir"], generic["project"], generic["rules"]),
            ("", drafter.PROFILE_DEFAULTS["project"], drafter.GENERIC_RULES),
        )


if __name__ == "__main__":
    unittest.main()
