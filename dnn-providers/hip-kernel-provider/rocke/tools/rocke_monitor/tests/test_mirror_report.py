# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk mirror-report` (rkmon_lib/mirror_report.py): a live monitor's posts (review_cost.jsonl) paired with its mirror's
reviews (the out folder's cost.json and post.md) by PR and head, the live post the one of the feed line the mirror
took; each post parsed (header light, findings, priors, Code quality, hygiene); the live side's Claude lanes and
session counted from its transcripts as the mirror counts its own, in dollars, with the session shared over the
reviews in its window; the totals per arm with a day at the live rate; the checks; every lost review and why.
"""

import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import rkmon  # noqa: E402
import sandbox  # noqa: E402
from rkmon_lib import mirror_report as mr  # noqa: E402

H, H9, H9B, H10, H11 = "c" * 40, "9" * 40, "b" * 40, "a" * 40, "d" * 40
LIVE = """Automated review: Model - qwen, effort: max

## PR #7 — make the answer right (dev-a) — 🟠 Follow-up: off by one

| Prior item | Status |
|---|---|
| Ours (11-01) N1: answer returns 41 | 🟠 OPEN<br>• still at `x.py:2` (read) |
| dev-b (11-01): helper has no test | 🟢 FIXED `abcdef12`<br>• a test calls it (read) |
| Bots / CI | 🟢 Green<br>• 3 ✓ (read) |
| Our checks | 🟢 Clean<br>• merge clean (ran) |
| Code quality | 🟠 Should-fix<br>• a second home for the helper (read) |

| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🟠 | `proj/x.py:2` | **answer returns 41, not 42**<br>• impact: every caller gets a wrong answer<br>• fix: return 42 |
| N2 | 🟡 | `proj/y.py:9` | **flags drops the fast flag**<br>• fix: keep it |

**Utility:** 🟡 Mixed — fixes nothing yet
"""
MIRROR = """Automated review: Model - Cand-1, effort: high

## PR #7 — make the answer right (dev-a) — 🟡 Follow-up: one nit

| Prior item | Status |
|---|---|
| Ours (11-01) N1: answer returns 41 | 🟠 OPEN<br>• still at `x.py:2` (read) |
| dev-b (11-01): helper has no test | 🟠 OPEN<br>• no test calls it (read) |
| Bots / CI | 🟢 Green<br>• 3 ✓ (read) |
| Our checks | 🟢 Clean<br>• merge clean (ran) |
| Code quality | 🟢 Clean<br>• one home per fact (read) |

| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🟡 | `proj/x.py:3` | **answer returns 41**<br>• fix: return 42, as the user asked |
| N2 | 🟡 | `proj/z.py:1` | **notes left unused**<br>• fix: drop them |

**Utility:** 🟡 Mixed — fixes nothing yet
"""
FEED_AT = "2026-11-07T10:00:00Z"


def jsonl(path: pathlib.Path, recs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def call(mid, t, inp=10, read=0, write=0, w1h=0, out=0, effort="max"):
    u = {
        "input_tokens": inp,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
        "cache_creation": {
            "ephemeral_5m_input_tokens": write - w1h,
            "ephemeral_1h_input_tokens": w1h,
        },
        "output_tokens": out,
    }
    return {
        "type": "assistant",
        "timestamp": t,
        "perTurnEffort": effort,
        "message": {"id": mid, "usage": u},
    }


def run(sub: pathlib.Path, rid: str, kind: str, desc: str, recs) -> None:
    jsonl(sub / f"agent-{rid}.jsonl", recs)
    (sub / f"agent-{rid}.meta.json").write_text(
        json.dumps({"agentType": kind, "description": desc})
    )


class Report(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        root = pathlib.Path(self.d.name)
        self.live, self.out, self.claude = (
            root / "live_state",
            root / "mirror",
            root / "live_claude",
        )
        post = root / "pr7_fu2.md"
        post.write_text(LIVE, encoding="utf-8")
        local = {
            "by": "qwen",
            "method": "local-worker",
            "tokens": None,
            "wall_min": 18.3,
        }
        jsonl(
            self.live / "review_cost.jsonl",
            [
                {
                    "pr": 7,
                    "round": "fu1",
                    "head": H,
                    "post": "",
                    "started": "2026-11-07T08:00:00Z",
                    "at": "2026-11-07T08:50:00Z",
                    "draft": local,
                    "review_min": 50.0,
                },  # an earlier post at this head
                {
                    "pr": 7,
                    "round": "fu2",
                    "head": H,
                    "post": str(post),
                    "started": FEED_AT,
                    "at": "2026-11-07T10:55:00Z",
                    "verifier": 190000,
                    "quality": 230000,
                    "draft": local,
                    "review_min": 55.0,
                },
                {
                    "pr": 8,
                    "round": "fr",
                    "head": "8" * 40,
                    "post": "",
                    "started": "2026-11-07T10:30:00Z",
                    "at": "2026-11-07T11:20:00Z",
                    "draft": {},
                    "review_min": 50.0,
                },  # in #7's window
                {
                    "pr": 10,
                    "round": "fr",
                    "head": H10,
                    "post": "/live/pr10.md",
                    "started": "2026-11-06T09:00:00Z",
                    "at": "2026-11-06T10:00:00Z",
                },
            ],
        )
        with open(self.live / "review_cost.jsonl", "a", encoding="utf-8") as f:
            f.write("not json\n")
        jsonl(
            self.live / "mirror_feed.jsonl",
            [
                {"pr": 9, "head": H9, "round": "fr", "at": "2026-11-07T09:00:00Z"},
                {
                    "pr": 9,
                    "head": H9B,
                    "round": "fr",
                    "at": "2026-11-07T09:30:00Z",
                },  # #9 moved: no live post at H9
                {"pr": 7, "head": H, "round": "fu2", "at": FEED_AT},
            ],
        )
        # The live transcripts: the monitor session, #7's verifier and code quality pass, an earlier round's verifier.
        proj = self.claude / "ws"
        jsonl(
            proj / "s1.jsonl",
            [
                {
                    "type": "user",
                    "timestamp": "2026-11-07T07:00:00Z",
                    "message": {
                        "content": "<command-name>/rocke-monitor</command-name> tick"
                    },
                },
                call(
                    "c1",
                    "2026-11-07T10:10:00Z",
                    read=100000,
                    write=1000,
                    w1h=1000,
                    out=50,
                ),
                call("c2", "2026-11-07T10:40:00Z", read=200000, out=50),
                call("c3", "2026-11-07T12:00:00Z", read=900000),
            ],
        )  # after the post
        jsonl(
            proj / "other.jsonl", [call("z1", "2026-11-07T10:20:00Z", read=999999)]
        )  # not the monitor's session
        sub = proj / "s1" / "subagents"
        run(
            sub,
            "v1",
            "rocke-review-verifier",
            "Verify PR 7 follow-up 2",
            [
                call(
                    "v1a",
                    "2026-11-07T10:20:00Z",
                    inp=5,
                    read=50000,
                    write=2000,
                    out=100,
                    effort="high",
                ),
                call(
                    "v1b",
                    "2026-11-07T10:30:00Z",
                    inp=5,
                    read=80000,
                    out=100,
                    effort="high",
                ),
            ],
        )
        run(
            sub,
            "q1",
            "rocke-code-quality-reviewer",
            "Code quality lane PR 7 fu2",
            [
                call(
                    "q1a",
                    "2026-11-07T10:15:00Z",
                    inp=5,
                    read=70000,
                    write=3000,
                    w1h=500,
                    out=300,
                )
            ],
        )
        run(
            sub,
            "v0",
            "rocke-review-verifier",
            "Verify PR 7 follow-up 1",
            [call("v0a", "2026-11-07T08:10:00Z")],
        )
        # The mirror: #7 (arm B), #9 written with no live post, #11 written then dropped, #10 dropped before.
        d = self.out / f"pr7_fu2_{H[:10]}"
        d.mkdir(parents=True)
        (d / "post.md").write_text(MIRROR, encoding="utf-8")
        vt = {
            "uncached": 10,
            "cache_read": 60000,
            "cache_write_5m": 1000,
            "cache_write_1h": 0,
            "output": 100,
            "calls": 3,
            "processed": 61010,
            "usd": 0.019,
            "ieq": 4760,
            "efforts": {"high": 3},
        }
        (d / "cost.json").write_text(
            json.dumps(
                {
                    "pr": 7,
                    "head": H,
                    "round": "fu2",
                    "arm": "B",
                    "quality_on_binding": True,
                    "feed_at": FEED_AT,
                    "started": "2026-11-07T10:01:00Z",
                    "draft": {
                        "by": "cand",
                        "requests": 30,
                        "prompt_tokens": 180000,
                        "completion_tokens": 30000,
                        "reasoning_tokens": 12000,
                        "tokens": 210000,
                        "wall_min": 25.0,
                    },
                    "quality": {
                        "by": "cand",
                        "requests": 9,
                        "prompt_tokens": 70000,
                        "completion_tokens": 10000,
                        "reasoning_tokens": 6000,
                        "tokens": 80000,
                        "wall_min": 10.0,
                    },
                    "verifier": {"by": "claude", "runs": 1, "usd": 0.019, "tally": vt},
                    "notice": None,
                    "claude": {
                        "lanes": {"verifier": {"runs": 1, "tally": vt}},
                        "session": {
                            "calls": 4,
                            "processed": 200000,
                            "output": 400,
                            "usd": 0.05,
                            "ieq": 12500,
                            "efforts": {"high": 4},
                        },
                        "session_overlap": 0,
                    },
                    "minutes": {"review": 70.0, "checks": 20.0, "verify": 15.0},
                }
            )
        )
        for n, head in ((9, H9), (11, H11)):
            e = self.out / f"pr{n}_fr_{head[:10]}"
            e.mkdir()
            (e / "post.md").write_text("x\n")
            (e / "cost.json").write_text(
                json.dumps(
                    {
                        "pr": n,
                        "head": head,
                        "round": "fr",
                        "arm": "A",
                        "feed_at": "2026-11-07T09:00:00Z",
                    }
                )
            )
        jsonl(
            self.out / "drops.jsonl",
            [
                {
                    "pr": 10,
                    "head": H10,
                    "round": "fr",
                    "arm": "B",
                    "why": "it is closed on GitHub",
                    "dir": None,
                },
                {
                    "pr": 11,
                    "head": H11,
                    "round": "fr",
                    "arm": "A",
                    "dir": str(self.out / f"pr11_fr_{H11[:10]}"),
                    "why": f"the live review of {H11[:10]} was dropped: the head moved",
                },
            ],
        )

    def test_the_pair_its_posts_and_disagreements(self):
        rep = mr.build(self.live, self.out, self.claude)
        self.assertEqual(len(rep["pairs"]), 1)
        p = rep["pairs"][0]
        self.assertEqual(
            (p["pr"], p["round"], p["arm"], p["live"]["cost"]["started"]),
            (7, "fu2", "B", FEED_AT),
        )
        lp, mp = p["live"]["parsed"], p["mirror"]["parsed"]
        self.assertEqual((lp["light"], mp["light"]), ("🟠", "🟡"))
        self.assertEqual(
            [(f["id"], f["light"], f["file"], f["line"]) for f in lp["findings"]],
            [("N1", "🟠", "proj/x.py", 2), ("N2", "🟡", "proj/y.py", 9)],
        )
        self.assertEqual(
            [(x["light"], x["status"]) for x in lp["priors"]],
            [("🟠", "OPEN"), ("🟢", "FIXED")],
        )
        self.assertEqual(
            lp["data"], {"Bots / CI": "🟢", "Our checks": "🟢", "Code quality": "🟠"}
        )
        self.assertEqual(lp["hygiene"], [])
        self.assertEqual(len(mp["hygiene"]), 1)
        self.assertEqual(
            p["disagreements"],
            [
                "header light: live 🟠, mirror 🟡",
                "severity: live N1 🟠, mirror N1 🟡: answer returns 41, not 42",
                "only live: N2 🟡 `proj/y.py:9` flags drops the fast flag",
                "only mirror: N2 🟡 `proj/z.py:1` notes left unused",
                'prior "dev-b (11-01): helper has no test": live 🟢 FIXED, mirror 🟠 OPEN',
                "code quality: live 🟠, mirror 🟢",
                f"hygiene (mirror): {mp['hygiene'][0][:160]}",
            ],
        )
        self.assertTrue(
            mp["hygiene"][0].startswith("hygiene hit: 15:| N1 | 🟡 |"), mp["hygiene"]
        )

    def test_both_sides_counted_one_way_in_dollars(self):
        rep = mr.build(self.live, self.out, self.claude)
        p = rep["pairs"][0]
        lanes = p["live"]["claude"]["lanes"]
        self.assertEqual(
            {k: v["runs"] for k, v in lanes.items()}, {"verifier": 1, "quality": 1}
        )  # not fu1's run
        v, q = lanes["verifier"]["tally"], lanes["quality"]["tally"]
        self.assertEqual(
            (v["processed"], v["output"], v["efforts"]), (132010, 200, {"high": 2})
        )
        self.assertEqual(
            v["usd"], round((10 * 4 + 130000 * 0.2 + 2000 * 5 + 200 * 20) / 1e6, 4)
        )
        self.assertEqual(
            (q["cache_write_5m"], q["cache_write_1h"], q["efforts"]),
            (2500, 500, {"max": 1}),
        )
        s = p["live"]["claude"]["session"]
        self.assertEqual(
            (s["calls"], s["cache_write_1h"]), (2, 1000)
        )  # the window only; the monitor's session only
        self.assertEqual(
            s["usd"], round((20 * 4 + 300000 * 0.2 + 1000 * 8 + 100 * 20) / 1e6, 4)
        )
        # #8's review ran in the window, and #9's at H9B (a feed line with no post yet: an hour from its start)
        self.assertEqual(p["live"]["claude"]["session_overlap"], 2)
        lf, mf = p["live"]["figures"], p["mirror"]["figures"]
        self.assertAlmostEqual(lf["lanes_usd"], 0.04 + 0.0365, places=4)
        self.assertAlmostEqual(lf["share_usd"], s["usd"] / 3, places=3)
        self.assertAlmostEqual(
            lf["review_usd"], lf["lanes_usd"] + lf["share_usd"], places=4
        )
        self.assertEqual(
            (lf["model_tokens"], lf["minutes"]), (None, 55.0)
        )  # the local worker counts no tokens
        self.assertEqual(
            (mf["lanes_usd"], mf["share_usd"], mf["review_usd"], mf["lanes_ieq"]),
            (0.019, 0.05, 0.069, 4760),
        )
        self.assertEqual(
            (mf["model_tokens"], mf["model_requests"], mf["minutes"]),
            (290000, 39, 70.0),
        )
        self.assertEqual(rep["rate"], {"per_day": 3.8, "posts": 4, "days": 1.06})
        self.assertEqual(sorted(rep["totals"]), ["live B", "mirror B"])
        self.assertEqual(
            rep["totals"]["mirror B"]["usd_per_day"], round(0.069 * 3.8, 2)
        )
        self.assertEqual(
            rep["checks"], ["#7 live quality: 500 tokens written to the 1-hour cache"]
        )

    def test_only_a_review_that_got_the_binding_s_pass_owes_its_record(self):
        """cost.json says whether the review got the quality binding's code quality pass (quality_on_binding): one that
        got none by design (a merge-only head, a review too large for the autopilot) is no missing record, and its
        worklist names no quality/ folder; one that got it and left no record is the check's.
        """
        d = self.out / f"pr7_fu2_{H[:10]}"
        rec = json.loads((d / "cost.json").read_text())
        text = "\n".join(mr.render(mr.build(self.live, self.out, self.claude)))
        self.assertIn(
            f"- mirror: {d}/post.md (notes.md, draft/, quality/ beside it)", text
        )
        (d / "cost.json").write_text(
            json.dumps(dict(rec, quality=None, quality_on_binding=False))
        )
        rep = mr.build(self.live, self.out, self.claude)
        self.assertEqual(
            rep["checks"], ["#7 live quality: 500 tokens written to the 1-hour cache"]
        )
        self.assertEqual(rep["pairs"][0]["mirror"]["figures"]["model_tokens"], 210000)
        text = "\n".join(mr.render(rep))
        self.assertIn(f"- mirror: {d}/post.md (notes.md, draft/ beside it)", text)
        (d / "cost.json").write_text(
            json.dumps(dict(rec, quality=None, quality_on_binding=True))
        )
        self.assertIn(
            "#7 mirror arm B: the quality binding's lane left no record",
            mr.build(self.live, self.out, self.claude)["checks"],
        )

    def test_every_lost_review_and_why(self):
        lost = mr.build(self.live, self.out, self.claude)["lost"]
        self.assertEqual(
            [(r["pr"], r.get("side", "mirror")) for r in lost],
            [(10, "mirror"), (11, "mirror"), (9, "live")],
        )
        self.assertEqual(
            lost[0]["live_post"], "/live/pr10.md"
        )  # the mirror lost it; the live monitor posted it
        self.assertIn(
            f"the live monitor started a review of {H9B[:10]} at 2026-11-07T09:30:00Z",
            lost[2]["why"],
        )

    def test_a_review_closed_without_a_post_is_no_live_post(self):
        """The live monitor's line of a review `rk review-close` closed without a post (`closed`) is no post of its
        own: the live side leaves it out, so it pairs with no mirrored review and counts in no rate.
        """
        closed = {
            "pr": 12,
            "round": "fu1",
            "head": "c" * 40,
            "post": "https://example.com/o/r/review-1",
            "closed": True,
            "at": "2026-11-07T11:00:00Z",
        }
        with open(self.live / "review_cost.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(closed) + "\n")
        self.assertEqual(
            sorted(mr.live_side(self.live)), [(7, H), (8, "8" * 40), (10, H10)]
        )
        rep = mr.build(self.live, self.out, self.claude)
        self.assertEqual(rep["rate"], {"per_day": 3.8, "posts": 4, "days": 1.06})

    def test_the_rendered_report_and_the_command(self):
        text = "\n".join(mr.render(mr.build(self.live, self.out, self.claude)))
        self.assertTrue(
            text.startswith("# Mirror report: 1 pair(s), live vs mirror"), text[:80]
        )
        self.assertIn(
            "$4 input, $0.2 cache read, $5 5-minute and $8 1-hour cache write, $20 output",
            text,
        )
        self.assertIn(
            f"| 7 | fu2 | `{H[:10]}` | B | 🟠 / 🟡 | 0·1·1 / 0·0·2 | 2 / 2 | 🟠 / 🟢 | 0 / 1 | $0.10 / $0.07 | "
            "$0.08 / $0.02 | $0.02 / $0.05 | - / 290k | 55.0 / 70.0 |",
            text,
        )
        self.assertIn(
            "| 7 | B | $0.04 · 132k · 200 (1, high) | $0.02 · 61k · 100 (1, high) | "
            "$0.04 · 73k · 300 (1, max) | - | - | - | $0.07 (3) / $0.05 (1) |",
            text,
        )
        self.assertIn(
            "| 7 | B | 180k · 30k · 12k, 30 req, 25.0 min | 70k · 10k · 6k, 9 req, 10.0 min | 20.0 | 15.0 |",
            text,
        )
        self.assertIn(
            "a day at the live monitor's rate: 3.8 reviews a day (4 live posts in 1.06 days)",
            text,
        )
        self.assertIn(
            "| mirror B | $0.07 (1) | $0.02 (1) | $0.05 (1) | $0.05 (1) | 5k (1) | 290k · 39 | 70.0 (1) | "
            "$0.26 |",
            text,
        )
        self.assertIn(f"### #7 fu2 at `{H[:10]}` (arm B)", text)
        self.assertIn("- #7 live quality: 500 tokens written to the 1-hour cache", text)
        self.assertIn(
            f"- #10 fr at `{H10[:10]}`, arm B: the mirror's review: it is closed on GitHub; the live monitor "
            "posted it: /live/pr10.md",
            text,
        )
        self.assertIn(
            f"- #9 fr at `{H9[:10]}`, arm A: the live review: no live post at this head; the live monitor",
            text,
        )
        buf = io.StringIO()
        # The command reads the configuration in the monitor's home: one of the test's own.
        sandbox.own_home(self)
        with redirect_stdout(buf):
            code = rkmon.main(
                [
                    "mirror-report",
                    "--live-state",
                    str(self.live),
                    "--mirror",
                    str(self.out),
                    "--live-claude",
                    str(self.claude),
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        rep = json.loads(buf.getvalue())
        self.assertEqual(
            (rep["pairs"][0]["pr"], rep["live_claude"]), (7, str(self.claude))
        )


if __name__ == "__main__":
    unittest.main()
