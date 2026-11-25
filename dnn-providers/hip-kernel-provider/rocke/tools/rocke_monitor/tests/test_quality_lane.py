# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The code quality lane on a binding (worker/quality_lane.py) on an offline PR (tests/drafter_fixture.py): the
scripted model's tool calls go through the real client, limiter and tools, and quality.md, its evidence and its token
records come out; a 🔴/🟠 needs evidence beside its own line before finish; the rendered team member is its card, or the
generic card without one. Also tools/quality_run.py's folder and `rk review-prep`'s copy of quality.md. No socket is
opened: the binding points at the closed port 127.0.0.1:9 and the client's `post` is the fake endpoint.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import drafter_fixture as fx  # noqa: E402
import llm_client as lc  # noqa: E402
import quality_lane as ql  # noqa: E402
from rkmon_lib import bindings, prep, review  # noqa: E402

FINISH = fx.call(
    "finish",
    points="duplicates a helper of the module (read); one test covers the change (read)",
    searched="grep for helper and for v + 1 in proj",
    standards="none in this tree",
    overlaps="none listed",
    lead="none",
)
LANE = [
    fx.call("diff_hunk", id="H1.1"),
    fx.call("grep_code", pattern="def helper"),
    fx.call(
        "add_finding",
        severity="🟠",
        file="proj/x.py",
        line=5,
        problem="helper adds a second job",
        evidence="the answer module already owns it",
        fix="keep answer() the only home",
        quote="def helper(v):",
    ),
    FINISH,  # refused: Q1 has no evidence beside its own line
    fx.call(
        "record_evidence",
        id="Q1",
        claim="the existing home",
        file="proj/x.py",
        line=1,
        quote="def answer():",
    ),
    FINISH,
]


class Lane(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = fx.make(pathlib.Path(self.tmp.name))
        self.cfg = dict(self.fx["cfg"], roles={"quality": "cand"})
        self.q = self.fx["ws"] / "quality"
        self.q.mkdir()
        (self.q / "pr").symlink_to("../pr")
        (self.q / "mkwt.log").write_text((self.fx["ws"] / "mkwt.log").read_text())
        (self.q / "overlap_section.md").write_text(
            "## Overlapping work in other open PRs\nnone\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, endpoint, member: str = "", **over) -> ql.QualitySession:
        cfg = dict(self.cfg, drafter=dict(self.cfg["drafter"], **over))
        s = ql.settings(cfg)

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

        return ql.QualitySession(
            self.q, 7, cfg, s, member=member, trees=self.fx["root"], chat=chat
        )

    def read(self, name: str) -> str:
        return (self.q / name).read_text(encoding="utf-8")


class TestLane(Lane):
    def test_quality_md_its_evidence_and_its_records(self):
        ep = fx.FakeEndpoint({"hunt": LANE})
        self.assertTrue(self.session(ep).run_lane())
        lines = self.read("quality.md").splitlines()
        self.assertEqual(
            lines[0],
            "| Code quality | 🟠 Should-fix<br>• duplicates a helper of the module (read)<br>• "
            "one test covers the change (read) |",
        )
        self.assertEqual(
            lines[2:5],
            [
                "| # | Sev | file:line | Finding |",
                "|---|---|---|---|",
                "| Q1 | 🟠 | `x.py:5` | **helper adds a second job**<br>• evidence: the answer "
                "module already owns it (x.py:1 (head))<br>• fix: keep answer() the only home |",
            ],
        )
        self.assertEqual(
            lines[6:12],
            [
                "notes:",
                "- searched: grep for helper and for v + 1 in proj",
                "- standards read: none in this tree",
                "- overlaps: none listed",
                "- lead (pre-existing, not this PR's): none",
                lines[11],
            ],
        )
        self.assertTrue(
            lines[11].startswith(
                "- the pass: the quality binding cand (Cand-1), 6 requests, "
            ),
            lines[11],
        )
        self.assertNotIn("pending", self.read("quality.md"))
        # The refused finish: Q1 had only its own line.
        self.assertIn(
            "REJECTED: Q1 has no evidence beside its own line",
            self.read("agent_log.txt"),
        )
        ev = [json.loads(x) for x in self.read("evidence.jsonl").splitlines()]
        self.assertEqual(
            [(e["id"], e["kind"], e.get("own", False)) for e in ev],
            [("Q1", "read", True), ("Q1", "read", False)],
        )
        run = json.loads(self.read("quality_run.json"))
        self.assertEqual(
            (run["written"], run["light"], run["findings"], run["method"], run["role"]),
            (True, "🟠", 1, "quality-lane", "quality"),
        )
        q = run["passes"]["quality"]
        self.assertEqual((q["status"], q["requests"]), ("ok", 6))
        self.assertGreater(q["tokens"], 0)
        tot = json.loads(self.read("tokens_total.json"))
        self.assertEqual((tot["method"], tot["total"]["requests"]), ("quality-lane", 6))
        # The menu, and the card: the generic rules without a rendered member.
        body = ep.bodies[0]
        self.assertEqual(
            [t["function"]["name"] for t in body["tools"]],
            [
                "diff_hunk",
                "read_lines",
                "grep_code",
                "git_log",
                "add_finding",
                "record_evidence",
                "drop_finding",
                "finish",
            ],
        )
        self.assertIn("THE CODE QUALITY PASS'S RULES", body["messages"][0]["content"])
        self.assertIn("YOUR WORK HERE.", body["messages"][0]["content"])
        user = body["messages"][1]["content"]
        self.assertTrue(user.startswith("TASK: code quality pass. PR #7"), user[:80])
        self.assertIn("## Overlapping work in other open PRs\nnone", user)

    def test_the_rendered_member_is_its_card(self):
        ep = fx.FakeEndpoint({"hunt": [FINISH]})
        self.assertTrue(
            self.session(
                ep, member="## Team Member: the quality lane\nthe operator's rules"
            ).run_lane()
        )
        system = ep.bodies[0]["messages"][0]["content"]
        self.assertIn("the operator's rules", system)
        self.assertNotIn("THE CODE QUALITY PASS'S RULES", system)
        self.assertEqual(
            self.read("quality.md").splitlines()[0],
            "| Code quality | 🟢 Clean<br>• duplicates a helper "
            "of the module (read)<br>• one test covers the "
            "change (read) |",
        )
        self.assertNotIn("| # |", self.read("quality.md"))  # no finding: no table

    def test_a_lane_out_of_rounds_still_writes_a_final_row(self):
        idle = {"hunt": [fx.call("grep_code", pattern="helper")] * 6}
        self.assertTrue(self.session(fx.FakeEndpoint(idle), quality_steps=2).run_lane())
        row = self.read("quality.md").splitlines()[0]
        self.assertTrue(
            row.startswith(
                "| Code quality | 🟢 Clean<br>• the pass ended unfinished ("
            ),
            row,
        )
        self.assertNotIn("pending", row)

    def test_settings_and_the_member_file(self):
        s = ql.settings(self.cfg)
        self.assertEqual(
            (s["binding"], s["review_wall_min"], s["review_token_budget"]),
            ("cand", 20, 400_000),
        )
        self.assertIsNone(ql.settings(dict(self.cfg, roles={})))
        agentic = pathlib.Path(self.tmp.name) / "agentic" / "team_members"
        agentic.mkdir(parents=True)
        (agentic / "rocke_code_quality_reviewer.md").write_text("the rendered lane\n")
        cfg = dict(self.cfg, paths=dict(self.cfg["paths"], agentic=str(agentic.parent)))
        self.assertEqual(ql.member_text(cfg), "the rendered lane\n")
        self.assertEqual(bindings.drafter_settings(cfg)["quality_steps"], 24)


class TestEntryPoint(unittest.TestCase):
    def test_a_config_file_is_read_whole_or_refused(self):
        """--config: its keys replace the configuration's; a file that cannot be read, is not valid JSON or holds no
        JSON object is refused in plain words, exit 4, and the lane does not run."""
        import contextlib
        import io
        from unittest import mock

        seen = []
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d)
            (f / "broken.json").write_text("{")
            (f / "list.json").write_text("[]")
            (f / "over.json").write_text(json.dumps({"reviewer": "the override"}))
            said = {}
            for name in ("none.json", "broken.json", "list.json", "over.json"):
                out = io.StringIO()
                with mock.patch(
                    "rkmon_lib.store.load_config", lambda: {"reviewer": "kept"}
                ), mock.patch.object(
                    ql, "settings", lambda cfg: seen.append(cfg["reviewer"])
                ), mock.patch.object(
                    ql, "QualitySession", None
                ), contextlib.redirect_stdout(
                    out
                ):
                    code = ql.main([d, "7", "--config", str(f / name)])
                said[name] = (code, out.getvalue().split("\n")[0])
        self.assertEqual(
            said,
            {
                "none.json": (
                    4,
                    f"REFUSED: the --config file {f}/none.json cannot be read: No such file or directory",
                ),
                "broken.json": (
                    4,
                    f"REFUSED: the --config file {f}/broken.json is not valid JSON: Expecting property name "
                    "enclosed in double quotes: line 1 column 2 (char 1)",
                ),
                "list.json": (
                    4,
                    f"REFUSED: the --config file {f}/list.json is no JSON object",
                ),
                # no quality binding (settings gives None): it stops there, after the override
                "over.json": (
                    1,
                    "the quality role has no binding (roles.quality naming a model_bindings entry)",
                ),
            },
        )
        self.assertEqual(seen, ["the override"])


class TestJobAndPrep(unittest.TestCase):
    def test_quality_run_prepares_its_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            env = {"PATH": "/usr/bin:/bin", "RKMON_REVIEW_ROOT": str(root)}
            script = [
                sys.executable,
                "-B",
                str(ROOT / "tools" / "quality_run.py"),
                "7",
                "--prepare-only",
            ]
            p = subprocess.run(
                script, capture_output=True, text=True, timeout=60, env=env
            )
            self.assertEqual(p.returncode, 1)
            self.assertEqual(
                p.stdout,
                f"PR 7 is not staged ({root}/workers/pr7/pr/meta.md): no code quality pass\n",
            )
            ws = root / "workers" / "pr7"
            (ws / "pr").mkdir(parents=True)
            (ws / "pr" / "meta.md").write_text("# PR #7: t\n")
            (ws / "mkwt.log").write_text("head=aaaaaaaaaa\n")
            (ws / "quality").mkdir()
            (ws / "quality" / "stale.md").write_text("an earlier lane's\n")
            (ws / "quality" / "stale.md").chmod(0o444)
            p = subprocess.run(
                script, capture_output=True, text=True, timeout=120, env=env
            )
            self.assertEqual((p.returncode, p.stdout), (0, ""), p.stdout + p.stderr)
            q = ws / "quality"
            self.assertFalse((q / "stale.md").exists())  # made afresh
            self.assertEqual(os.readlink(q / "pr"), "../pr")
            self.assertEqual((q / "pr" / "meta.md").read_text(), "# PR #7: t\n")
            self.assertEqual((q / "mkwt.log").read_text(), "head=aaaaaaaaaa\n")
            self.assertTrue((q / "overlap_section.md").is_file())

    def prep_case(self, root: pathlib.Path) -> tuple[dict, pathlib.Path]:
        """A configuration whose quality role has a binding, and `rk review-prep`'s review directory for PR 7."""
        cfg = {
            "paths": {"working_dir": str(root)},
            "roles": {"quality": "cand"},
            "model_bindings": {
                "cand": {"url": "http://127.0.0.1:9", "model": "Cand-1"}
            },
        }
        out = root / "v7fr"
        out.mkdir()
        return cfg, out

    def test_review_prep_copies_the_lane_s_file_for_the_head(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, out = self.prep_case(pathlib.Path(d))
            st = {"prs": {"7": {"head": "a" * 40}}}
            self.assertEqual(prep.quality_copy(cfg, st, 7, out), [])
            q = review.quality_folder(cfg, 7)
            q.mkdir(parents=True)
            (q / "quality.md").write_text("| Code quality | 🟢 Clean<br>• x (read) |\n")
            (q / "quality_run.json").write_text(
                json.dumps({"head": "b" * 40, "binding": "cand"})
            )
            self.assertEqual(
                prep.quality_copy(cfg, st, 7, out),
                [
                    "quality.md: the code quality pass's is for bbbbbbbbbb, not the head aaaaaaaaaa: not "
                    "copied; dispatch the code quality reviewer as usual"
                ],
            )
            self.assertFalse((out / "quality.md").exists())
            (q / "quality_run.json").write_text(
                json.dumps({"head": "a" * 40, "binding": "cand"})
            )
            self.assertEqual(
                prep.quality_copy(cfg, st, 7, out),
                [
                    "quality.md: the code quality pass ran on the quality binding cand at aaaaaaaaaa (copied "
                    "here): the verifier folds it in; don't dispatch the code quality reviewer"
                ],
            )
            self.assertEqual(
                (out / "quality.md").read_text(),
                "| Code quality | 🟢 Clean<br>• x (read) |\n",
            )

    def test_a_review_too_large_for_the_autopilot_gets_no_line_and_no_copy(self):
        """`rk review-prep` says nothing of the lane there, whatever head the lane's file is for: REVIEW_READY and the
        Split review procedure say what takes the pass's place."""
        with tempfile.TemporaryDirectory() as d:
            cfg, out = self.prep_case(pathlib.Path(d))
            q = review.quality_folder(cfg, 7)
            q.mkdir(parents=True)
            (q / "quality.md").write_text("| Code quality | 🟢 Clean<br>• x (read) |\n")
            for head in ("a" * 40, "b" * 40):
                with self.subTest(lane_head=head[:10]):
                    (q / "quality_run.json").write_text(
                        json.dumps({"head": head, "binding": "cand"})
                    )
                    self.assertEqual(
                        prep.quality_copy(
                            cfg,
                            {"prs": {"7": {"head": "a" * 40, "size": 40000}}},
                            7,
                            out,
                        ),
                        [],
                    )
                    self.assertFalse((out / "quality.md").exists())

    def test_a_review_with_no_pass_on_the_binding_by_design_gets_no_line(self):
        """The review's REVIEW_READY kept whether it gets the pass on the binding (review.quality_runs_for): one that
        does not (a merge-only head, a mirror's arm A) is told nothing of an earlier review's file; one that does is
        told to dispatch the code quality reviewer when the file is not for its head."""
        with tempfile.TemporaryDirectory() as d:
            cfg, out = self.prep_case(pathlib.Path(d))
            q = review.quality_folder(cfg, 7)
            q.mkdir(parents=True)
            (q / "quality.md").write_text("| Code quality | 🟢 Clean<br>• x (read) |\n")
            (q / "quality_run.json").write_text(
                json.dumps({"head": "b" * 40, "binding": "cand"})
            )
            for gets, said in (
                (False, []),
                (
                    True,
                    [
                        "quality.md: the code quality pass's is for bbbbbbbbbb, not the head aaaaaaaaaa: not "
                        "copied; dispatch the code quality reviewer as usual"
                    ],
                ),
            ):
                with self.subTest(quality_on_binding=gets):
                    st = {
                        "prs": {"7": {"head": "a" * 40, "size": 10}},
                        "review_ready": [
                            {"pr": 7, "head": "a" * 40, "quality_on_binding": gets}
                        ],
                    }
                    self.assertEqual(prep.quality_copy(cfg, st, 7, out), said)
                    self.assertFalse((out / "quality.md").exists())


if __name__ == "__main__":
    unittest.main()
