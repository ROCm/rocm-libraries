# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/fixline.py: the fix-line premise of `rk post --premises`. A fix that guards a test on a module CI lacks
stops the post; a "CI still runs it" claim is checked with lane-of at the head; other skip guards, tier changes and
test moves need a `fixcheck:` line in the notes. And premises.check runs it on the filled post.
"""

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rkmon_lib import fixline, premises  # noqa: E402

HEAD = "a" * 40
# The modules this test's CI lacks (config ci_absent_modules).
CFG = {
    "ci_absent_modules": ["torch", "matplotlib", "rocke_engine", "rocke_library_engine"]
}


def post(*rows: str) -> str:
    return (
        "Automated review: x\n\n## PR #7 — t — 🔴\n\n**Needed to merge:**\n\n| # | Sev | file:line | Item |\n"
        "|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def row(rid: str, light: str, where: str, fix: str) -> str:
    return f"| {rid} | {light} | {where} | **Title**<br>• impact: x<br>• fix: {fix} |"


class FakeLaneOf:
    def __init__(self, verdict="runs", error=None):
        self.verdict, self.error, self.calls = verdict, error, []

    def __call__(self, cfg, targets, rev=None, collect=False):
        self.calls.append((list(targets), rev, collect))
        if self.error:
            return {"error": self.error, "results": []}
        return {
            "error": None,
            "results": [
                {
                    "target": t,
                    "verdict": self.verdict,
                    "why": 'module-level importorskip("torch"); CI has no torch',
                    "entry": "rocke_pytest_core",
                    "tier": "standard",
                }
                for t in targets
            ],
        }


def run(text, notes="", lane_of=None, cfg=None):
    return fixline.check(
        CFG if cfg is None else cfg, text, notes, HEAD, lane_of=lane_of or FakeLaneOf()
    )


class TestRows(unittest.TestCase):
    def test_red_and_orange_rows_their_fix_and_their_tests(self):
        text = post(
            row(
                "N1",
                "🔴",
                "`platform/tests/core/test_a.py:12`",
                "move the import into `test_b`",
            ),
            row("N2", "🟡", "`x.py:1`", 'importorskip("torch")'),
            "| N3 | 🟠 | `library/tests/test_c.py:4`, `…/test_d.py:9` | Prose finding → add a skipif |",
            row(
                "F1", "🟠", "`a.py:1`", "see `platform/tests/test_e.py::TestX::test_y`"
            ),
        )
        got = {r["id"]: r for r in fixline.rows(text)}
        self.assertEqual(sorted(got), ["F1", "N1", "N3"])  # 🟡 rows are not checked
        self.assertEqual(got["N1"]["fix"], "move the import into `test_b`")
        self.assertEqual(got["N1"]["tests"], ["platform/tests/core/test_a.py"])
        self.assertEqual(got["N3"]["fix"], "add a skipif")
        self.assertEqual(
            got["N3"]["tests"], ["library/tests/test_c.py"]
        )  # a `…/` path is no path
        self.assertEqual(
            got["F1"]["tests"], ["platform/tests/test_e.py::TestX::test_y"]
        )

    def test_fixcheck_lines(self):
        notes = "- fixcheck: N1 ran rk lane-of\nfixcheck: F2 read the tier YAML\nfixcheck: N3 maybe\n"
        self.assertEqual(fixline.fixchecks(notes), {"N1", "F2"})


class TestCheck(unittest.TestCase):
    def test_a_guard_on_a_module_ci_lacks_stops_the_post(self):
        for fix in (
            'per-file `torch = pytest.importorskip("torch")`',
            "`@pytest.mark.skipif(torch is None, reason='no torch')`",
            'importorskip(\\"rocke_engine\\")',
        ):
            with self.subTest(fix=fix), self.assertRaises(fixline.FixStop) as e:
                run(
                    post(row("N1", "🔴", "`t/test_a.py:1`", fix)),
                    notes="fixcheck: N1 ran x",
                )
            self.assertIn("CI would never run it", e.exception.args[0])
        # The configured list decides: mpmath is absent there.
        with self.assertRaises(fixline.FixStop):
            run(
                post(
                    row(
                        "N1",
                        "🔴",
                        "`t/test_a.py:1`",
                        '`importorskip("mpmath")` for trig',
                    )
                ),
                cfg={"ci_absent_modules": ["mpmath"]},
            )

    def test_without_absent_modules_a_guard_needs_a_fixcheck(self):
        # None configured: no module is known absent, so a guard on one is a skip guard the verifier checks.
        text = post(row("N1", "🔴", "`t/test_a.py:1`", 'importorskip("torch")'))
        with self.assertRaises(fixline.FixStop) as e:
            run(text, cfg={})
        self.assertIn("`fixcheck: N1 ran|read HOW`", e.exception.args[0])
        self.assertEqual(
            run(text, cfg={}, notes="fixcheck: N1 ran rk lane-of"),
            ["fix lines: N1 fixcheck in the notes"],
        )

    def test_other_skip_guards_tiers_and_moves_need_a_fixcheck(self):
        for fix in (
            '`importorskip("numpy")` in the helper',
            "`SKIP_RETURN_CODE 5` on the entry",
            "list it in the standard tier beside the others",
            "move the block into `tests/cmake/InstanceTests.cmake`",
            "add it to `ctest_lists.cmake`",
        ):
            text = post(row("N4", "🟠", "`t/test_a.py:1`", fix))
            with self.subTest(fix=fix), self.assertRaises(fixline.FixStop) as e:
                run(text)
            self.assertIn("`fixcheck: N4 ran|read HOW`", e.exception.args[0])
            self.assertEqual(
                run(text, notes="fixcheck: N4 ran rk lane-of"),
                ["fix lines: N4 fixcheck in the notes"],
            )

    def test_a_ci_claim_is_checked_with_lane_of_at_the_head(self):
        text = post(
            row(
                "N2",
                "🔴",
                "`platform/tests/core/test_a.py:3`",
                "move the import into the test; CI still runs it",
            )
        )
        lane = FakeLaneOf("runs")
        self.assertEqual(
            run(text, lane_of=lane),
            [
                "fix lines: N2 claims CI runs it; lane-of: platform/tests/core/test_a.py: runs "
                "(rocke_pytest_core, standard)"
            ],
        )
        self.assertEqual(lane.calls, [(["platform/tests/core/test_a.py"], HEAD, True)])
        with self.assertRaises(fixline.FixStop) as e:
            run(text, lane_of=FakeLaneOf("skips"), notes="fixcheck: N2 ran x")
        self.assertIn(
            'lane-of at the head reads platform/tests/core/test_a.py: CI: skips(module-level importorskip("torch")',
            e.exception.args[0],
        )
        # lane-of could not answer: the verifier's fixcheck settles it, else the post stops.
        with self.assertRaises(fixline.FixStop) as e:
            run(text, lane_of=FakeLaneOf(error="no cmake"))
        self.assertIn("lane-of could not answer (no cmake)", e.exception.args[0])
        self.assertEqual(
            run(
                text,
                lane_of=FakeLaneOf(error="no cmake"),
                notes="fixcheck: N2 ran the installed run",
            ),
            ["fix lines: N2 claims CI runs it; lane-of NOT RUN, fixcheck in the notes"],
        )

    def test_a_ci_claim_naming_no_test_needs_a_fixcheck(self):
        text = post(
            row(
                "N2",
                "🟠",
                "`platform/CMakeLists.txt:9`",
                "keep the entry; CI still runs it",
            )
        )
        with self.assertRaises(fixline.FixStop):
            run(text)
        self.assertEqual(
            run(text, notes="fixcheck: N2 read the shard log"),
            ["fix lines: N2 fixcheck in the notes"],
        )

    def test_a_claim_on_a_test_the_head_lacks_needs_a_fixcheck(self):
        text = post(
            row(
                "N2",
                "🟠",
                "`t/test_new.py:1`",
                "add the case there, so CI still runs it",
            )
        )
        lane = FakeLaneOf("not run")
        with self.assertRaises(fixline.FixStop) as e:
            run(text, lane_of=lane)
        self.assertIn("lane-of could not answer", e.exception.args[0])
        self.assertEqual(
            run(text, lane_of=lane, notes="fixcheck: N2 ran the installed run"),
            ["fix lines: N2 claims CI runs it; lane-of NOT RUN, fixcheck in the notes"],
        )

    def test_a_pytestmark_that_skips_nothing_is_no_guard(self):
        fix = 'module-level `pytestmark = pytest.mark.filterwarnings("error::X")`'
        self.assertEqual(run(post(row("N1", "🔴", "`t/test_a.py:1`", fix))), [])

    def test_a_plain_fix_passes_silently(self):
        lane = FakeLaneOf()
        self.assertEqual(
            run(post(row("N1", "🔴", "`a.py:1`", "rename `x` to `y`")), lane_of=lane),
            [],
        )
        self.assertEqual(lane.calls, [])


class TestPremises(unittest.TestCase):
    def test_check_runs_the_fix_line_premise_on_the_filled_post(self):
        with tempfile.TemporaryDirectory() as d:
            notes = pathlib.Path(d) / "notes.md"
            notes.write_text("nothing\n")
            spec = {"NOTES": str(notes)}
            ready = {
                "text": post(
                    row("N1", "🔴", "`t/test_a.py:1`", 'importorskip("torch")')
                ),
                "sha": HEAD,
            }
            with mock.patch.object(
                premises, "_premises", return_value={}
            ), mock.patch.object(premises, "_post_checks", return_value=ready):
                lines, got = premises.check(CFG, None, 7, spec)
                self.assertIsNone(got)
                self.assertIn("CI would never run it", lines[-1])
                ready["text"] = post(row("N1", "🔴", "`a.py:1`", "rename it"))
                lines, got = premises.check(CFG, None, 7, spec)
                self.assertIs(got, ready)
                self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
