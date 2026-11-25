# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Claude's tokens from Claude Code's transcripts (rkmon_lib/transcripts.py), counted one way for both sides of a
mirror test: a call counted once by its message id at its last part; the kinds (uncached, cache reads, 5-minute and
1-hour writes, output), `processed` and the weighted `ieq`; each subagent run's lane and the PR its brief names; the
monitor session's calls per window, in one pass, never a sidechain's or another session's.
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import transcripts as tr  # noqa: E402


def call(mid, t, inp=0, read=0, write=0, w1h=None, out=0, side=False, effort=""):
    u = {
        "input_tokens": inp,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
        "output_tokens": out,
    }
    if w1h is not None:
        u["cache_creation"] = {
            "ephemeral_5m_input_tokens": write - w1h,
            "ephemeral_1h_input_tokens": w1h,
        }
    rec = {
        "type": "assistant",
        "timestamp": t,
        "isSidechain": side,
        "message": {"id": mid, "usage": u},
    }
    return dict(rec, perTurnEffort=effort) if effort else rec


def write(path: pathlib.Path, recs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


class TestTally(unittest.TestCase):
    def test_a_call_counts_once_and_the_kinds_weigh_as_opus(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "s.jsonl"
            write(
                p,
                [
                    call(
                        "m1", "2026-11-07T01:00:00Z", inp=10, read=100, write=40, out=1
                    ),  # a part, then the last
                    call(
                        "m1",
                        "2026-11-07T01:00:01Z",
                        inp=10,
                        read=100,
                        write=40,
                        out=8,
                        effort="max",
                    ),
                    call(
                        "m2",
                        "2026-11-07T01:05:00Z",
                        inp=2,
                        read=1000,
                        write=200,
                        w1h=150,
                        out=20,
                        effort="high",
                    ),
                    {
                        "type": "user",
                        "timestamp": "2026-11-07T01:06:00Z",
                        "message": {"content": "x"},
                    },
                ],
            )
            cs = tr.calls(p)
            t = tr.tally(cs)
        self.assertEqual(len(cs), 2)
        self.assertEqual(
            (
                t["uncached"],
                t["cache_read"],
                t["cache_write_5m"],
                t["cache_write_1h"],
                t["output"],
            ),
            (12, 1100, 90, 150, 28),
        )
        self.assertEqual((t["calls"], t["processed"]), (2, 12 + 1100 + 90 + 150))
        self.assertEqual(
            tr.prices({}), tr.DEFAULT_PRICES
        )  # no config prices: Opus 5.5's
        self.assertEqual(
            t["ieq"], round(12 + 1100 * 0.05 + 90 * 1.25 + 150 * 2 + 28 * 5)
        )
        self.assertEqual(
            t["usd"], round((12 * 4 + 1100 * 0.2 + 90 * 5 + 150 * 8 + 28 * 20) / 1e6, 4)
        )
        self.assertEqual(
            (t["efforts"], tr.effort_of(t)), ({"max": 1, "high": 1}, "high 1 · max 1")
        )
        both = tr.add(t, t)
        self.assertEqual(
            (both["processed"], both["efforts"]),
            (2 * t["processed"], {"max": 2, "high": 2}),
        )
        self.assertAlmostEqual(both["usd"], 2 * t["usd"])
        self.assertEqual(tr.effort_of(tr.tally(cs[1:])), "high")

    def test_the_prices_are_config_prices_as_rk_spend_reads_them(self):
        """The mirror's cost and rk spend agree: both price from config `prices` (its newest Opus entry)."""
        from rkmon_lib import spend

        table = {
            "_source": "test",
            "claude-opus-5-5": {
                "in": 5.0,
                "out": 25.0,
                "cache_read": 0.5,
                "cache_write_5m": 6.25,
                "cache_write_1h": 10.0,
            },
        }
        price = tr.prices({"prices": table})
        self.assertEqual(
            price,
            {
                "uncached": 5.0,
                "cache_read": 0.5,
                "cache_write_5m": 6.25,
                "cache_write_1h": 10.0,
                "output": 25.0,
            },
        )
        cs = [
            (
                "2026-11-07T01:00:00",
                {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 200,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 50,
                        "ephemeral_1h_input_tokens": 150,
                    },
                    "output_tokens": 40,
                },
                "high",
            )
        ]
        t = tr.tally(cs, price)
        use = {
            "opus": [100, 50, 150, 1000, 40, 40, 1]
        }  # spend_scan's row: in, w5m, w1h, read, out, out est., requests
        self.assertAlmostEqual(
            t["usd"], round(spend.usd(spend.price_table({"prices": table}), use), 4)
        )
        self.assertEqual(
            t["ieq"], round(100 + 1000 * 0.1 + 50 * 1.25 + 150 * 2 + 40 * 5)
        )

    def test_times_compare_to_the_second(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "s.jsonl"
            write(
                p,
                [
                    call("m1", "2026-11-07T01:00:00.500Z", inp=1),
                    call("m2", "2026-11-07T01:59:59.900Z", inp=1),
                ],
            )
            got = tr.calls(
                p, since="2026-11-07T01:00:00Z", until="2026-11-07T01:59:59Z"
            )
        self.assertEqual(
            [c[0] for c in got], ["2026-11-07T01:00:00"]
        )  # a call in the start's second is in


class TestRunsAndSession(unittest.TestCase):
    def test_lanes_by_pr_and_the_monitor_session_per_window(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d) / "projects" / "ws"
            main = root / "s1.jsonl"
            write(
                main,
                [
                    {
                        "type": "user",
                        "timestamp": "2026-11-07T00:59:00Z",
                        "message": {
                            "content": "<command-name>/rocke-monitor</command-name> tick"
                        },
                    },
                    call("a", "2026-11-07T01:00:00Z", read=1000),
                    call("b", "2026-11-07T01:30:00Z", read=2000),
                    call(
                        "c", "2026-11-07T01:31:00Z", read=50000, side=True
                    ),  # a sidechain: a subagent's
                    call("d", "2026-11-07T02:30:00Z", read=4000),
                ],
            )
            write(
                root / "other.jsonl", [call("z", "2026-11-07T01:10:00Z", read=9999)]
            )  # not the monitor's
            sub = root / "s1" / "subagents"
            write(
                sub / "agent-v1.jsonl",
                [
                    {
                        "type": "user",
                        "timestamp": "2026-11-07T01:01:00Z",
                        "message": {
                            "content": "Verify PR #7 follow-up 2: the inputs are ..."
                        },
                    },
                    call("v1", "2026-11-07T01:02:00Z", read=300, out=5),
                    call("v2", "2026-11-07T01:20:00Z", read=600, out=5),
                ],
            )
            (sub / "agent-v1.meta.json").write_text(
                json.dumps(
                    {
                        "agentType": "rocke-review-verifier",
                        "description": "Verify #7 fu2",
                    }
                )
            )
            write(
                sub / "agent-q1.jsonl", [call("q1", "2026-11-07T01:03:00Z", read=700)]
            )
            (sub / "agent-q1.meta.json").write_text(
                json.dumps(
                    {
                        "agentType": "rocke-code-quality-reviewer",
                        "description": "Quality lane #7 fu2",
                    }
                )
            )
            write(sub / "agent-n1.jsonl", [call("n1", "2026-11-07T01:04:00Z", read=70)])
            (sub / "agent-n1.meta.json").write_text(
                json.dumps(
                    {
                        "agentType": "rocke-code-quality-reviewer",
                        "description": "Overlap notice draft for #8",
                    }
                )
            )
            runs = tr.runs(pathlib.Path(d) / "projects", since="2026-11-07T00:00:00Z")
            got = {
                r["id"]: (r["lane"], r["pr"], r["round"], r["tally"]["processed"])
                for r in runs
            }
            self.assertEqual(
                got,
                {
                    "v1": ("verifier", 7, "fu2", 900),
                    "q1": ("quality", 7, "fu2", 700),
                    "n1": ("notice", 8, "", 70),
                },
            )
            lanes = tr.lanes_for(runs, 7, "2026-11-07T01:00:00Z")
            self.assertEqual(
                {k: (v["runs"], v["tally"]["processed"]) for k, v in lanes.items()},
                {"verifier": (1, 900), "quality": (1, 700)},
            )
            self.assertEqual(
                tr.lanes_for(runs, 7, "2026-11-07T01:02:30Z"),
                {"quality": lanes["quality"]},
            )
            w = tr.session_windows(
                pathlib.Path(d) / "projects",
                [
                    ("2026-11-07T01:00:00Z", "2026-11-07T02:00:00Z"),
                    ("2026-11-07T02:00:00Z", "2026-11-07T03:00:00Z"),
                    ("2026-11-07T05:00:00Z", "2026-11-07T06:00:00Z"),
                ],
            )
            self.assertEqual([x.get("processed") for x in w], [3000, 4000, None])
            every = tr.session(
                pathlib.Path(d) / "projects",
                "2026-11-07T01:00:00Z",
                "2026-11-07T02:00:00Z",
                marker="",
            )
            self.assertEqual(
                every["processed"], 3000 + 9999
            )  # the container's: every main session is the monitor's


if __name__ == "__main__":
    unittest.main()
