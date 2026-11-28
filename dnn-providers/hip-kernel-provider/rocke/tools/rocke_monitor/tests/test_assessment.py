# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The assessment of the drafters against the verified post (rkmon_lib/assessment.py): the notes' lines per drafter tag,
the records `rk post --premises` writes at post time (one per drafter: the live one and the shadow), the card both
dashboards show (each drafter on its current method, the phases scored apart); and the roles' bindings
(rkmon_lib/bindings.py) it names its candidates by. Model endpoints point at the closed port 127.0.0.1:9.
"""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import (
    assessment,
    bindings,
    config,
    draft_stage,
    premises,
    report,
)  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

BINDING = {
    "provider": "openai",
    "url": "http://127.0.0.1:9/onprem",
    "model": "Cand-1",
    "tpm": 1000,
    "rpm": 10,
}
NOTES = """settled: something
- glm G1 match N1
glm G2 new N3
glm G3 false the quoted line is in a comment
`glm G4 dropped true, but cosmetic`
GLM missed N2 🟠
glm missed N4 yellow
glm prior dev-r (11-04) golden stale right
glm prior Copilot (11-05) ×3: ABI bump wrong
glm G9 maybe something else
another line that is not ours
"""
POST = """Automated review: Model - Cand-1, effort: high

## PR #7 — t (a) — 🔴 Follow-up: wrong answer

| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🔴 | `x.py:2` | **a** |
| N2 | 🟠 | `x.py:5` | **b** |
| N3 | 🟠 | `y.py:9` | **c** |
| N4 | 🟡 | `z.py:1` | **d** |
"""
# The live draft (the local worker's) and the shadow draft's lines for one post.
BOTH = """qwen G1 match N1
qwen missed N2 🟠
qwen prior Ours (11-01) N1: answer right
glm G1 match N2
glm G2 false not a defect
glm prior P2 right
glm prior P3 wrong
glm prior Ours (11-02) N4: helper right
"""
QWEN_DRAFT = """| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🔴 | `x.py:2` | **a** |
"""


def a_cfg(**over) -> dict:
    cfg = {
        "worker_backend": "glm",
        "model_bindings": {"glm": dict(BINDING)},
        "review_first_line": "Automated review by rocke-monitor: Model - Cand-1, effort: high",
    }
    cfg.update(over)
    return cfg


class TestLines(unittest.TestCase):
    def test_parse(self):
        p = assessment.parse(NOTES, "glm")
        self.assertEqual(p["match"], [{"g": 1, "n": 1}])
        self.assertEqual(p["new"], [{"g": 2, "n": 3}])
        self.assertEqual(
            p["false"], [{"g": 3, "why": "the quoted line is in a comment"}]
        )
        self.assertEqual(p["dropped"], [{"g": 4, "why": "true, but cosmetic"}])
        self.assertEqual(
            p["missed"],
            [{"n": 2, "severity": "orange"}, {"n": 4, "severity": "yellow"}],
        )
        self.assertEqual(
            p["prior"],
            [
                {"row": "dev-r (11-04) golden stale", "ok": True},
                {"row": "Copilot (11-05) ×3: ABI bump", "ok": False},
            ],
        )
        self.assertEqual(p["unparsed"], ["glm G9 maybe something else"])
        self.assertEqual(len(p["lines"]), 9)
        self.assertEqual(assessment.parse(NOTES, "other")["lines"], [])
        self.assertEqual(
            len(assessment.parse(BOTH, "qwen")["lines"]), 3
        )  # any drafter's tag

    def test_the_record_counts_by_severity(self):
        draft = {
            "status": "ready",
            "findings": [
                {"id": "N1", "severity": "🔴"},
                {"id": "N2", "severity": "🟠"},
                {"id": "N3", "severity": "🟡"},
                {"id": "N4", "severity": "🟠"},
            ],
            "tokens": 41000,
            "requests": 22,
            "wall_s": 610.0,
            "late": False,
            "model": "Cand-1",
        }
        rec = assessment.build(
            7,
            "fu2",
            "a" * 40,
            NOTES,
            POST,
            draft,
            "glm",
            "Cand-1",
            "Ref Model 1.0",
            "T",
        )
        self.assertEqual(
            (rec["match"], rec["new"], rec["missed"]),
            ({"red": 1}, {"orange": 1}, {"orange": 1, "yellow": 1}),
        )
        self.assertEqual(
            (rec["false"], rec["dropped"]), ({"yellow": 1}, {"orange": 1})
        )  # the draft's own lights
        self.assertEqual(rec["prior"], {"right": 1, "wrong": 1})
        self.assertEqual(
            rec["prior_by"], {"other": {"right": 1, "wrong": 1}}
        )  # no settled rows on record
        self.assertEqual(
            (
                rec["candidate"]["binding"],
                rec["candidate"]["method"],
                rec["reference"]["model"],
                rec["tokens"],
                rec["wall_s"],
            ),
            ("glm", "single-pass", "Ref Model 1.0", 41000, 610.0),
        )
        self.assertEqual((rec["draft_findings"], rec["post_findings"]), (4, 4))

    def test_a_prior_line_names_its_row_by_id_or_first_cell(self):
        rows = [
            {
                "id": "P2",
                "key": "Ours (11-01) N1: `answer` returns 41",
                "settler": "script",
            },
            {"id": "P3", "key": "dave (11-02): keep z.py", "settler": "model"},
        ]
        self.assertEqual(assessment.row_of("p3", rows)["settler"], "model")
        self.assertEqual(assessment.row_of("Ours (11-01) N1: answer", rows)["id"], "P2")
        self.assertIsNone(assessment.row_of("nobody", rows))


class TestRecordAndCard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = pathlib.Path(self.tmp.name)
        self.store = Store(self.work / "state")
        self.st = empty_state()
        self.notes = self.work / "notes.md"
        self.notes.write_text(NOTES, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_post_time_record_and_the_card(self):
        cfg = a_cfg(
            paths={"root": str(self.work), "workers": str(self.work / "workers")}
        )
        self.st["drafts"] = [
            {
                "pr": 7,
                "head": "a" * 40,
                "status": "ready",
                "binding": "glm",
                "model": "Cand-1",
                "findings": [{"id": "N1", "severity": "🔴"}],
                "tokens": 40000,
                "requests": 20,
                "wall_s": 600.0,
                "late": True,
            },
            {
                "pr": 8,
                "head": "b" * 40,
                "status": "missing",
                "binding": "glm",
                "tokens": 9000,
                "requests": 4,
                "wall_s": 300.0,
                "late": False,
            },
        ]
        self.store.save(self.st)
        lines = assessment.record(
            cfg,
            self.store,
            7,
            {"NOTES": str(self.notes), "ROUND": "fu2"},
            {"sha": "a" * 40, "text": POST},
        )
        self.assertEqual(
            len(lines), 1, lines
        )  # the live drafter's (a binding): no shadow
        self.assertTrue(
            lines[0].startswith(
                "assessment recorded (drafter_assessment.jsonl, glm single-pass): 1 match, "
                "1 new, 1 false, 1 dropped, 2 missed; priors 1 right, 1 wrong"
            ),
            lines,
        )
        self.assertIn("1 line(s) not understood", lines[0])
        (rec,) = [
            json.loads(x)
            for x in (self.store.dir / assessment.FILE).read_text().splitlines()
        ]
        self.assertEqual(
            (rec["pr"], rec["round"], rec["draft"], rec["tokens"]),
            (7, "fu2", "ready", 40000),
        )
        self.assertEqual(
            rec["candidate"],
            {
                "role": "drafter",
                "binding": "glm",
                "tag": "glm",
                "model": "Cand-1",
                "method": "single-pass",
            },
        )
        c = assessment.card(cfg, self.store.load(), self.store)
        self.assertEqual(
            c["title"],
            f"GLM drafter vs {bindings.reference_name(premises.verifier_name(cfg))}",
        )
        self.assertEqual(c["columns"], ["glm (single-pass, live)"])
        rows = {name: vals[0] for name, vals in c["rows"]}
        self.assertEqual(
            rows["New: 🔴/🟠 recall"], "67% (2 of 3)"
        )  # N1 (match), N3 (new) of N1, N2, N3
        self.assertEqual(rows["New: 🟡 recall"], "0% (0 of 1)")
        self.assertEqual(rows["New: false claims per draft"], "1.0 (1 in 1 drafts)")
        self.assertEqual(rows["Prior: model statuses right"], "50% (1 of 2)")
        self.assertEqual(rows["Prior: script statuses right"], "—")
        self.assertEqual(rows["Tokens per draft (by phase)"], "24k")
        self.assertEqual(
            rows["Drafts missing or late at post time"], "0 missing, 1 late of 1"
        )  # the review waited
        md = "\n".join(assessment.card_md(cfg, self.store.load(), self.store))
        self.assertIn("## GLM drafter vs", md)
        self.assertIn("| Measure | glm (single-pass, live) |", md)
        self.assertIn("| New: 🔴/🟠 recall | 67% (2 of 3) |", md)
        dash = report.dashboard(
            {
                "features": {"review_automation": {"mode": "live"}},
                "paths": {},
            },
            self.st,
            [],
            [],
            [],
        )
        self.assertNotIn(
            "drafter vs", dash
        )  # `rk drafts` prints the card; no dashboard shows it

    def test_the_live_and_the_shadow_drafter_side_by_side(self):
        # The local worker drafts live; glm drafts in the shadow with the phased method. Its earlier drafts (another
        # method) are left out of its column, and counted.
        cfg = a_cfg(
            worker_backend=None,
            worker_model="qwen3.6:35b",
            roles={"shadow_drafter": "glm"},
            paths={"root": str(self.work), "workers": str(self.work / "workers")},
        )
        live = draft_stage.folder(cfg, 7)
        live.mkdir(parents=True)
        (live / "review_agent.md").write_text(QWEN_DRAFT, encoding="utf-8")
        (live / "stats_agent.json").write_text(json.dumps({"wall_s": 600.0}))
        shadow = draft_stage.shadow_folder(cfg, 7)
        shadow.mkdir(parents=True)
        (shadow / "prior_settled.json").write_text(
            json.dumps(
                {
                    "head": "a" * 40,
                    "method": "phased",
                    "rows": [
                        {
                            "id": "P2",
                            "key": "Ours (11-01) N1: answer",
                            "settler": "script",
                        },
                        {
                            "id": "P3",
                            "key": "dave (11-02): keep z.py",
                            "settler": "model",
                        },
                        {
                            "id": "P5",
                            "key": "Ours (11-02) N4: helper",
                            "settler": "script",
                        },
                    ],
                    "counts": {
                        "rows": 5,
                        "script": 3,
                        "script_pass": 3,
                        "model": 2,
                        "model_pass": 1,
                        "look": 1,
                        "unsettled": 0,
                    },
                }
            )
        )
        self.st["drafts"] = [
            {
                "pr": 5,
                "head": "e" * 40,
                "status": "ready",
                "binding": "glm",
                "tokens": 50000,
            },
            {
                "pr": 6,
                "head": "f" * 40,
                "status": "missing",
                "binding": "glm",
                "tokens": 9000,
            },
            {
                "pr": 7,
                "head": "a" * 40,
                "status": "ready",
                "binding": "glm",
                "role": "shadow_drafter",
                "method": "phased",
                "findings": [
                    {"id": "N1", "severity": "🟠"},
                    {"id": "N2", "severity": "🟡"},
                ],
                "tokens": 190000,
                "requests": 29,
                "wall_s": 1810.0,
                "late": False,
                "phases": {
                    "settle": {"wall_s": 6.0},
                    "prior": {"wall_s": 300.0, "tokens": 40000},
                    "hunt": {"wall_s": 1500.0, "tokens": 150000},
                },
            },
        ]
        self.store.save(self.st)
        self.notes.write_text(BOTH, encoding="utf-8")
        lines = assessment.record(
            cfg,
            self.store,
            7,
            {"NOTES": str(self.notes), "ROUND": "fu3"},
            {"sha": "a" * 40, "text": POST},
        )
        self.assertEqual(len(lines), 2, lines)
        recs = {
            r["candidate"]["tag"]: r
            for r in (
                json.loads(x)
                for x in (self.store.dir / assessment.FILE).read_text().splitlines()
            )
        }
        self.assertEqual(
            recs["qwen"]["candidate"],
            {
                "role": "drafter",
                "binding": "",
                "tag": "qwen",
                "model": "qwen3.6:35b",
                "method": "local-worker",
            },
        )
        self.assertEqual(
            (recs["qwen"]["draft"], recs["qwen"]["wall_s"], recs["qwen"]["match"]),
            ("ready", 600.0, {"red": 1}),
        )
        g = recs["glm"]
        self.assertEqual(
            (g["candidate"]["role"], g["candidate"]["method"], g["draft"]),
            ("shadow_drafter", "phased", "ready"),
        )
        self.assertEqual(
            g["prior_by"],
            {"script": {"right": 2, "wrong": 0}, "model": {"right": 0, "wrong": 1}},
        )
        self.assertEqual(
            (g["judged"], g["settled"]["rows"], g["false"]), (3, 5, {"yellow": 1})
        )
        c = assessment.card(cfg, self.store.load(), self.store)
        self.assertEqual(
            c["title"],
            f"GLM drafter vs {bindings.reference_name(premises.verifier_name(cfg))}",
        )
        self.assertEqual(
            c["columns"], ["glm (phased, shadow)", "qwen (local-worker, live)"]
        )
        self.assertEqual(
            c["left_out"], "2 glm drafts of an earlier method left out (single-pass 2)"
        )
        rows = dict(c["rows"])
        self.assertEqual(rows["Prior: script statuses right"], ["100% (2 of 2)", "—"])
        self.assertEqual(
            rows["Prior: model statuses right"], ["0% (0 of 1)", "100% (1 of 1)"]
        )
        self.assertEqual(
            rows["Prior rows: script / model (evidence pass)"],
            ["3 / 2 (50% (1 of 2))", "—"],
        )
        self.assertEqual(rows["Prior rows the verifier judged"], ["3 of 5", "1"])
        self.assertEqual(rows["New: 🔴/🟠 recall"], ["100% (1 of 1)", "50% (1 of 2)"])
        self.assertEqual(
            rows["Tokens per draft (by phase)"], ["prior 40k, hunt 150k", "—"]
        )
        self.assertEqual(
            rows["Minutes per draft (by phase)"],
            ["settle 0 min, prior 5 min, hunt 25 min", "10 min"],
        )
        md = "\n".join(assessment.card_md(cfg, self.store.load(), self.store))
        self.assertIn(
            "| Measure | glm (phased, shadow) | qwen (local-worker, live) |", md
        )
        self.assertIn("2 glm drafts of an earlier method left out (single-pass 2).", md)

    def test_a_shadow_not_done_at_post_time_is_late(self):
        cfg = a_cfg(
            worker_backend=None,
            roles={"shadow_drafter": "glm"},
            paths={"root": str(self.work), "workers": str(self.work / "workers")},
        )
        self.st["shadow_pool"] = {
            "7": {"pr": 7, "head": "a" * 40, "binding": "glm", "job": {"id": "s7"}}
        }
        self.store.save(self.st)
        self.notes.write_text("glm none\n", encoding="utf-8")
        lines = assessment.record(
            cfg,
            self.store,
            7,
            {"NOTES": str(self.notes)},
            {"sha": "a" * 40, "text": POST},
        )
        self.assertTrue(any("glm" in x and "draft late" in x for x in lines), lines)
        rec = [
            json.loads(x)
            for x in (self.store.dir / assessment.FILE).read_text().splitlines()
        ][-1]
        self.assertEqual((rec["candidate"]["tag"], rec["draft"]), ("glm", "late"))

    def test_the_shadow_scores_its_live_draft_and_records_a_missing_one(self):
        cfg = a_cfg(
            worker_backend=None,
            worker_model="qwen3.6:35b",
            roles={"shadow_drafter": "glm"},
            paths={"root": str(self.work), "workers": str(self.work / "workers")},
        )
        self.st["drafts"] = [
            {
                "pr": 7,
                "head": "a" * 40,
                "status": "ready",
                "binding": "glm",
                "tokens": 50000,
                "findings": [{"id": "N1", "severity": "🔴"}],
            }
        ]  # drafted live, before the shadow role
        self.store.save(self.st)
        self.notes.write_text("glm G1 match N1\n", encoding="utf-8")
        lines = assessment.record(
            cfg,
            self.store,
            7,
            {"NOTES": str(self.notes)},
            {"sha": "a" * 40, "text": POST},
        )
        self.assertEqual(
            len(lines), 1, lines
        )  # the local worker has no line: not scored
        self.notes.write_text("settled: nothing about the drafts\n", encoding="utf-8")
        lines = assessment.record(
            cfg,
            self.store,
            8,
            {"NOTES": str(self.notes)},
            {"sha": "b" * 40, "text": POST},
        )
        recs = [
            json.loads(x)
            for x in (self.store.dir / assessment.FILE).read_text().splitlines()
        ]
        self.assertEqual(
            [
                (r["pr"], r["candidate"]["role"], r["candidate"]["method"], r["draft"])
                for r in recs
            ],
            [
                (7, "shadow_drafter", "single-pass", "ready"),
                (8, "shadow_drafter", "single-pass", "missing"),
            ],
        )

    def test_a_binding_that_drafted_the_head_is_scored_whatever_the_roles_say_now(self):
        cfg = a_cfg(
            worker_backend=None,
            worker_model="qwen3.6:35b",
            paths={"root": str(self.work), "workers": str(self.work / "workers")},
        )
        self.st["drafts"] = [
            {
                "pr": 7,
                "head": "a" * 40,
                "status": "ready",
                "binding": "glm",
                "model": "Cand-1",
                "tokens": 50000,
                "findings": [{"id": "N1", "severity": "🔴"}],
            }
        ]
        self.store.save(self.st)
        self.notes.write_text("glm G1 match N1\n", encoding="utf-8")
        lines = assessment.record(
            cfg,
            self.store,
            7,
            {"NOTES": str(self.notes)},
            {"sha": "a" * 40, "text": POST},
        )
        self.assertEqual(len(lines), 1, lines)
        rec = json.loads((self.store.dir / assessment.FILE).read_text().splitlines()[0])
        self.assertEqual(
            (
                rec["candidate"]["tag"],
                rec["candidate"]["role"],
                rec["candidate"]["method"],
                rec["match"],
            ),
            ("glm", "drafter", "single-pass", {"red": 1}),
        )

    def test_the_single_drafters_records_are_read_as_single_pass(self):
        cfg = a_cfg(paths={"root": str(self.work)})
        old = {
            "at": "T",
            "pr": 3,
            "head": "c" * 40,
            "candidate": {"role": "drafter", "binding": "glm"},
            "draft": "ready",
            "match": {"orange": 1},
            "new": {},
            "false": {},
            "dropped": {},
            "missed": {},
            "prior": {"right": 1, "wrong": 0},
        }
        (self.store.dir / assessment.file_name("glm")).write_text(
            json.dumps(old) + "\n"
        )
        recs = assessment.records(self.store, ["glm"])
        self.assertEqual(
            recs[0]["candidate"],
            {
                "role": "drafter",
                "binding": "glm",
                "tag": "glm",
                "method": "single-pass",
            },
        )
        c = assessment.card(cfg, self.st, self.store)
        self.assertEqual(dict(c["rows"])["New: 🔴/🟠 recall"], ["100% (1 of 1)"])

    def test_nothing_without_a_drafter_or_a_draft(self):
        self.assertEqual(assessment.card({}, self.st, self.store), None)
        self.assertEqual(assessment.card_md({}, self.st, self.store), [])
        self.store.save(self.st)
        cfg = {"paths": {"root": str(self.work), "workers": str(self.work / "workers")}}
        self.assertEqual(
            assessment.record(
                cfg, self.store, 7, {"NOTES": str(self.notes)}, {"sha": "a" * 40}
            ),
            [],
        )
        # A drafter binding but neither a draft for the head nor a line: nothing recorded.
        self.notes.write_text("settled: x\n", encoding="utf-8")
        self.assertEqual(
            assessment.record(
                a_cfg(**cfg),
                self.store,
                7,
                {"NOTES": str(self.notes)},
                {"sha": "a" * 40},
            ),
            [],
        )
        self.assertFalse((self.store.dir / assessment.FILE).exists())


class TestBindings(unittest.TestCase):
    def test_roles_and_bindings(self):
        cfg = a_cfg(
            worker_model="Cand-1",
            model_bindings={"glm": {"url": "http://127.0.0.1:9", "tpm": 5}},
        )
        b = bindings.binding(cfg, "drafter")
        self.assertEqual(
            (b["binding"], b["model"], b["name"], b["where"], b["provider"]),
            ("glm", "Cand-1", "Cand-1", "local", "openai"),
        )
        self.assertEqual(
            bindings.binding(a_cfg(roles={"drafter": "other"}), "drafter"), None
        )
        self.assertEqual(bindings.binding({}, "drafter"), None)
        self.assertEqual(
            bindings.binding({"worker_backend": "qwen"}, "drafter"), None
        )  # no binding of that name
        s = bindings.client_settings(a_cfg(paths={"root": "/r"}), "drafter")
        self.assertEqual(
            (s["review_token_budget"], s["review_wall_min"], s["limiter_dir"]),
            (1_000_000, 45, "/r/llm_limits"),
        )
        self.assertEqual(
            (s["prior_wall_min"], s["hunt_wall_min"], s["prior_task_steps"]),
            (15, 35, 3),
        )
        self.assertEqual(
            bindings.first_line(a_cfg(review_first_line="")),
            "Automated review by rocke-monitor: Model - Cand-1, effort: the model's default",
        )
        self.assertEqual(bindings.tag(a_cfg()), "glm")
        self.assertEqual(bindings.reference_name("Vendor Family 5.5"), "Family")
        self.assertEqual(bindings.problems(a_cfg()), [])
        bad = a_cfg(
            roles={"drafter": "x", "poet": "glm"},
            drafter={"speed": 1},
            review_first_line="Automated review: Model - Other, effort: high",
            model_bindings={"glm": dict(BINDING, colour=1, effort="huge")},
        )
        text = "\n".join(bindings.problems(bad))
        for words in (
            "roles.poet is not a role",
            "roles.drafter names 'x'",
            "model_bindings.glm.colour is not",
            "model_bindings.glm.effort must be",
            "drafter.speed is not a setting",
        ):
            self.assertIn(words, text)
        # A line 1 the configuration gives is not read: line 1 is the drafter's, so no given line can name another
        line1 = a_cfg(review_first_line="Automated review: Model - Other, effort: high")
        self.assertEqual(
            (bindings.first_line(line1), bindings.problems(line1)),
            (
                "Automated review by rocke-monitor: Model - Cand-1, effort: the model's default",
                [],
            ),
        )
        self.assertTrue(
            bindings.describe(a_cfg())[0].startswith(
                "first pass: glm: Cand-1 via openai-compatible at "
            )
        )
        self.assertTrue(
            bindings.describe(a_cfg())[1].startswith(
                "drafter: glm = Cand-1 via openai-compatible at 127.0.0.1 "
                "(local), 1000 tokens and 10 requests a minute"
            )
        )
        self.assertEqual(
            bindings.describe({})[1],
            "drafter: no binding (the local review loop drafts)",
        )

    def test_the_shadow_drafter_and_the_candidates(self):
        cfg = a_cfg(
            worker_backend=None,
            worker_model="qwen3.6:35b",
            roles={"shadow_drafter": "glm"},
            review_first_line="Automated review: Model - qwen3.6:35b, effort: high",
        )
        self.assertEqual(
            bindings.candidates(cfg),
            [
                {
                    "role": "drafter",
                    "tag": "qwen",
                    "binding": "",
                    "model": "qwen3.6:35b",
                },
                {
                    "role": "shadow_drafter",
                    "tag": "glm",
                    "binding": "glm",
                    "model": "Cand-1",
                },
            ],
        )
        self.assertEqual(
            (
                bindings.tag(cfg),
                bindings.tag(cfg, "drafter"),
                bindings.tag(cfg, "shadow_drafter"),
            ),
            ("glm", "qwen", "glm"),
        )
        self.assertEqual(
            bindings.tag(dict(cfg, worker=dict(cfg.get("worker") or {}, tag="local"))),
            "glm",
        )
        self.assertEqual(
            bindings.tag(
                dict(cfg, worker=dict(cfg.get("worker") or {}, tag="local")), "drafter"
            ),
            "local",
        )
        self.assertEqual(
            bindings.first_line(cfg, "shadow_drafter"),
            "Automated review by rocke-monitor: Model - Cand-1, effort: high",
        )
        self.assertEqual(
            bindings.first_line(cfg),
            f"Automated review by rocke-monitor: Model - qwen3.6:35b, effort: {config.MODEL_DEFAULT}",
        )  # the local worker's, at its effort (none: worker.think is unset); the configuration's own line 1 is not read
        self.assertEqual(bindings.problems(cfg), [])
        self.assertIn("shadow_drafter: glm = Cand-1", "\n".join(bindings.describe(cfg)))
        same = a_cfg(roles={"shadow_drafter": "glm"})
        self.assertIn(
            "roles.shadow_drafter names the live drafter's binding 'glm'",
            "\n".join(bindings.problems(same)),
        )

    def test_the_phases_caps_fit_the_runs(self):
        text = "\n".join(
            bindings.problems(
                a_cfg(
                    drafter={
                        "hunt_wall_min": 50,
                        "prior_token_budget": 900_000,
                        "prior_steps": 6,
                    }
                )
            )
        )
        self.assertIn(
            "drafter.hunt_wall_min (50) is over drafter.review_wall_min (45)", text
        )
        self.assertIn(
            "drafter.prior_token_budget + drafter.hunt_token_budget (1400000) is over "
            "drafter.review_token_budget (1000000)",
            text,
        )
        self.assertIn("drafter.prior_steps is not a setting", text)


class TestPriorNote(unittest.TestCase):
    def test_prior_line_with_a_trailing_note_still_parses(self):
        from rkmon_lib import assessment as a

        notes = "glm prior dev-a (11-06) APPROVED right (but omits the older head)\nglm none\n"
        r = a.parse(notes, "glm")
        self.assertEqual(
            [(x["row"], x["ok"]) for x in r["prior"]],
            [("dev-a (11-06) APPROVED", True)],
        )
        self.assertTrue(r["none"] is True and r["unparsed"] == [])


if __name__ == "__main__":
    unittest.main()
