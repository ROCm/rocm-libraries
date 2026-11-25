# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A merge-only head's `threads` and `title` flags against what our review saw (rkmon_lib/peer_carry.py): the items
the checks write, the baseline at a head our review covers, the carry when every flagged item repeats, and the micro
round (the verifier alone on the new items) when one does not."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import test_followup_gap as fg  # noqa: E402  (its merge-only harness; its own tests run in their module)
from rkmon_lib import peer_carry, review  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
T1 = {
    "reviewer": "yr",
    "where": "rk/a.py:3",
    "url": "https://github.com/ROCm/r/pull/7#discussion_r1",
    "at": "2026-11-09T10:00",
    "replies": 0,
}
T2 = {
    "reviewer": "copilot-pull-request-reviewer",
    "where": "rk/b.py:9",
    "url": "https://github.com/ROCm/r/pull/7#discussion_r2",
    "at": "2026-11-10T10:00",
    "replies": 1,
}


NO_ITEMS = {"tests": None, "installed": None, "prnumbers": None}
# A failing test and a pull-request reference, as the checks name them, and our last review's open rows on them
# (the reference on another line of the same file: the review cites the line it was at then).
TEST = "lib/tests/test_pack.py::test_every_kind_packs"
TEST_NEW = "lib/tests/test_pack.py::test_one_kind_packs"
REF_FILE = "tools/gen/tests/test_rank.py"
REF = {
    "at": f"{REF_FILE}:12",
    "what": "`PR 4321`: (PR 4321 review): nothing exercised the ranker",
}
REF_NEW = {"at": "tools/gen/tests/test_other.py:3", "what": "`PR 4321`: see PR 4321"}
OURS_BODY = (
    "| Our N2 (11-01): packaging test fails | 🔴 OPEN: N2 |\n"
    "| N2 | 🔴 | `repo/sub/lib/tests/test_pack.py:40` | **The packaging test fails**<br>• impact: "
    "`test_every_kind_packs` fails on the merge and installed (ran) |\n"
    f'| N5 | 🟡 | `{REF_FILE}:9` | **A PR number in a committed docstring**<br>• impact: "(PR 4321 '
    'review)" |\n'
)


def write_tested(d: pathlib.Path, tests=None, installed=None, refs=None):
    """The tests' failing ids, the installed tests' results and the references' item file; None leaves one out."""
    if tests is not None:
        (d / peer_carry.TESTS_FILES[0]).parent.mkdir(exist_ok=True)
        (d / peer_carry.TESTS_FILES[0]).write_text("".join(f"{t}\n" for t in tests))
    if installed is not None:
        (d / "installed").mkdir(exist_ok=True)
        (d / "installed" / "results.json").write_text(
            json.dumps(
                {
                    "labels": {
                        **{
                            f"rk_pack_pytest {t}": ["pr", "a file the PR adds"]
                            for t in installed
                        },
                        "rk_pack_pytest lib/tests/test_old.py::test_dev": [
                            "develop",
                            "fails on develop too",
                        ],
                    }
                }
            )
        )
    if refs is not None:
        (d / peer_carry.ITEMS["prnumbers"]).write_text(
            json.dumps(
                {"prnumbers": {"check": "prnumbers", "items": refs, "error": None}}
            )
        )


def write_items(
    d: pathlib.Path,
    threads=None,
    title=None,
    why="no `type(scope): ` prefix",
    error=None,
):
    """The checks' item files, as tools/peer_checks.py --items-json writes them; None leaves a file out."""
    if threads is not None:
        (d / peer_carry.ITEMS["threads"]).write_text(
            json.dumps(
                {
                    "threads": {
                        "check": "threads",
                        "items": threads,
                        "error": error,
                        "live": len(threads),
                    }
                }
            )
        )
    if title is not None:
        (d / peer_carry.ITEMS["title"]).write_text(
            json.dumps(
                {
                    "title": {
                        "check": "title",
                        "title": title,
                        "items": [{"why": why}] if why else [],
                        "error": error,
                    }
                }
            )
        )


class TestItems(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)

    def test_read_and_flagged(self):
        self.assertEqual(
            peer_carry.read(self.d),
            dict.fromkeys(("threads", "title", "prnumbers", "tests", "installed")),
        )  # no lists: unknown
        write_items(self.d, threads=[T2, T1], title="Add a thing")
        items = peer_carry.read(self.d)
        self.assertEqual([t["url"] for t in items["threads"]], [T2["url"], T1["url"]])
        self.assertEqual(
            items["title"], {"title": "Add a thing", "why": "no `type(scope): ` prefix"}
        )
        self.assertEqual(
            peer_carry.flagged(items),
            {"threads": [T1["url"], T2["url"]], "title": "Add a thing", **NO_ITEMS},
        )
        write_items(self.d, threads=[], title="feat(rk): x", why=None)
        self.assertEqual(
            peer_carry.flagged(peer_carry.read(self.d)),
            {"threads": [], "title": "", **NO_ITEMS},
        )
        write_tested(self.d, tests=[TEST], installed=[TEST], refs=[REF])
        self.assertEqual(
            {
                k: v
                for k, v in peer_carry.flagged(peer_carry.read(self.d)).items()
                if k in NO_ITEMS
            },
            {
                "tests": [TEST],
                "installed": [TEST],
                "prnumbers": [[REF_FILE, REF["what"]]],
            },
        )
        write_items(
            self.d, threads=[T1], error="GitHub not read: timeout"
        )  # a check that did not run is unknown
        self.assertIsNone(peer_carry.read(self.d)["threads"])

    def test_the_baseline_is_a_covered_heads(self):
        rec = {
            "head": "n" * 40,
            "ours": {"head": "o" * 40, "equivalent_heads": ["e" * 40]},
        }
        self.assertIsNone(peer_carry.baseline(rec, self.d))
        write_items(self.d, threads=[T1], title="Add a thing")
        (self.d / "head_sha").write_text(
            "o" * 40 + "\n"
        )  # the checks at our reviewed head, before staging wipes them
        self.assertEqual(
            peer_carry.baseline(rec, self.d),
            {"threads": [T1["url"]], "title": "Add a thing", **NO_ITEMS},
        )
        (self.d / "head_sha").write_text(
            "f" * 40 + "\n"
        )  # checks at a head our review does not cover
        self.assertIsNone(peer_carry.baseline(rec, self.d))
        peer_carry.record(
            rec, self.d, "e" * 40
        )  # the last carried head's, recorded when its checks finished
        self.assertEqual(
            peer_carry.baseline(rec, self.d),
            {"threads": [T1["url"]], "title": "Add a thing", **NO_ITEMS},
        )
        rec["peer_seen"] = {
            "head": "e" * 40,
            "threads": [],
            "title": "",
        }  # a record from before the item lists
        self.assertEqual(
            peer_carry.baseline(rec, self.d), {"threads": [], "title": "", **NO_ITEMS}
        )
        peer_carry.record(rec, self.d, "f" * 40)
        self.assertIsNone(peer_carry.baseline(rec, self.d))

    def test_judge(self):
        status = {
            "threads": ("fail", "2 live threads"),
            "title": ("fail", "not a Conventional Commits title"),
        }
        now = {
            "threads": [
                {"url": T1["url"], "reviewer": "yr", "where": "rk/a.py:3"},
                {"url": T2["url"], "reviewer": "cp", "where": "rk/b.py:9"},
            ],
            "title": {"title": "Add a thing", "why": "no prefix"},
        }
        base = {"threads": [T1["url"], T2["url"]], "title": "Add a thing"}
        self.assertEqual(
            peer_carry.judge(status, now, base), (["threads", "title"], [])
        )
        self.assertEqual(
            peer_carry.judge(
                status, now, {"threads": [T1["url"]], "title": "Add a thing!"}
            ),
            (
                ["threads", "title"],
                [
                    f"thread cp at rk/b.py:9 {T2['url']}",
                    "title 'Add a thing': no prefix",
                ],
            ),
        )
        settled, new = peer_carry.judge(
            status, now, None
        )  # nothing known: every item is new
        self.assertEqual((settled, len(new)), (["threads", "title"], 3))
        # A failing check with no readable items is not accounted for; a passing one is not looked at.
        self.assertEqual(
            peer_carry.judge(status, {"threads": None, "title": {}}, base), ([], [])
        )
        self.assertEqual(
            peer_carry.judge({"threads": ("pass", "")}, now, None), ([], [])
        )

    FAIL3 = {
        "tests": ("fail", "1 failing"),
        "installed": ("fail", "1 failing"),
        "prnumbers": ("fail", "1 reference"),
    }

    def test_our_last_review_s_open_rows_are_its_findings_then_its_open_priors(self):
        """review_rows: each finding row as its where and text, then each prior item its status table still calls
        OPEN."""
        rows = peer_carry.review_rows(OURS_BODY)
        self.assertEqual(len(rows), 3, rows)
        self.assertTrue(rows[0].startswith("`repo/sub/lib/tests/test_pack.py:40` "))
        self.assertTrue(rows[1].startswith(f"`{REF_FILE}:9` "))
        self.assertTrue(rows[2].startswith("Our N2 (11-01): packaging test fails"))

    def test_failing_tests_and_references_our_last_review_names_carry(self):
        now = {"tests": [TEST], "installed": [TEST], "prnumbers": [REF]}
        base = {
            "threads": [],
            "title": "",
            **NO_ITEMS,
        }  # a record from before the item lists
        self.assertEqual(
            peer_carry.judge(self.FAIL3, now, base, lambda: OURS_BODY),
            (["tests", "installed", "prnumbers"], []),
        )
        # Flagged at a head our review covers: no review body needed (and none read).
        base = {
            "tests": [TEST],
            "installed": [TEST],
            "prnumbers": [[REF_FILE, REF["what"]]],
        }
        self.assertEqual(
            peer_carry.judge(self.FAIL3, now, base, lambda: self.fail("review read")),
            (["tests", "installed", "prnumbers"], []),
        )
        self.assertEqual(
            peer_carry.judge(self.FAIL3, now, None, None), ([], [])
        )  # neither known: full flags

    def test_a_new_failing_test_does_not_carry(self):
        now = {"tests": [TEST, TEST_NEW], "installed": [TEST_NEW], "prnumbers": [REF]}
        self.assertEqual(
            peer_carry.judge(self.FAIL3, now, {"tests": [TEST]}, lambda: OURS_BODY),
            (["prnumbers"], []),
        )

    def test_a_new_reference_does_not_carry(self):
        now = {"tests": [TEST], "installed": [TEST], "prnumbers": [REF, REF_NEW]}
        self.assertEqual(
            peer_carry.judge(self.FAIL3, now, None, lambda: OURS_BODY),
            (["tests", "installed"], []),
        )
        # The same reference text in another file is new too.
        moved = dict(REF_NEW, what=REF["what"])
        self.assertEqual(
            peer_carry.judge(
                self.FAIL3, {"prnumbers": [moved]}, None, lambda: OURS_BODY
            ),
            ([], []),
        )

    def test_a_flag_without_its_items_stays_a_full_flag(self):
        now = {"tests": None, "installed": [], "prnumbers": None}
        self.assertEqual(
            peer_carry.judge(self.FAIL3, now, None, lambda: OURS_BODY), ([], [])
        )


class TestTheChecksWriteTheItems(unittest.TestCase):
    def test_the_tool_writes_its_result(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "peer_title.json"
            r = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "peer_checks.py"),
                    "title",
                    "--title",
                    "Add a thing",
                    "--items-json",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(r.stdout.splitlines()[-1].startswith("TITLE fail "))
            res = json.loads(out.read_text())["title"]
            self.assertEqual((res["title"], len(res["items"])), ("Add a thing", 1))
            self.assertEqual(peer_carry.read(d)["title"]["title"], "Add a thing")

    def test_the_review_checks_pass_the_files(self):
        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        script = review.checks_script(
            {
                "rocke_path": "rk",
                "repo": "o/r",
                "paths": {"working_dir": "/w"},
            },
            7,
            "a" * 40,
            "follow-up",
            plan,
            None,
        )
        self.assertIn('--pr $N --items-json "$OUT/peer_threads.json"', script)
        self.assertIn(
            '--meta $WK/pr$N/pr/meta.md --items-json "$OUT/peer_title.json"', script
        )
        self.assertIn("prnumbers --repo", script)
        self.assertIn('--items-json "$OUT/peer_prnumbers.json"', script)


class TestMergeOnlyPeerFlags(unittest.TestCase):
    """The merge-only verdict's threads and title flags (test_followup_gap's harness)."""

    setUp = fg.TestMergeOnlyVerdict.setUp
    run_review = fg.TestMergeOnlyVerdict.run_review
    STATUS, OLD, NEW = (
        fg.TestMergeOnlyVerdict.STATUS,
        fg.TestMergeOnlyVerdict.OLD,
        fg.TestMergeOnlyVerdict.NEW,
    )
    THREADS = "threads fail 2 live threads with no owner reply: copilot-pull-request-reviewer 1, yr 1\n"
    TITLE = "title fail not a Conventional Commits title: no `type(scope): ` prefix\n"

    def test_flags_our_review_saw_carry_the_head(self):
        write_items(self.chk, threads=[T1, T2], title="Add a thing")
        lines = self.run_review(
            status=self.STATUS + self.THREADS + self.TITLE,
            peer={"threads": [T1["url"], T2["url"]], "title": "Add a thing"},
        )
        self.assertEqual(
            lines,
            [
                f"REVIEW_SKIP #7: merge-only since our review (1 merge; its checks are clean; its "
                f"threads and title flags repeat what our review saw, so nothing is posted); recorded "
                f"{self.NEW[:10]} as equivalent"
            ],
        )
        rec = self.st["prs"]["7"]
        self.assertEqual(rec["ours"]["equivalent_heads"], [self.NEW])
        self.assertEqual(
            rec["peer_seen"],
            {
                "head": self.NEW,
                "threads": [T1["url"], T2["url"]],
                "title": "Add a thing",
                **NO_ITEMS,
            },
        )
        self.assertEqual(self.store.ledger(), [])

    def test_a_new_thread_makes_a_micro_round(self):
        write_items(self.chk, threads=[T1, T2])
        lines = self.run_review(
            status=self.STATUS + self.THREADS,
            peer={"threads": [T1["url"]], "title": ""},
        )
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(
            lines[0].startswith(
                f"REVIEW_READY #7 (follow-up): worker done, checks done. Merge-only since our review; micro: thread "
                f"copilot-pull-request-reviewer at rk/b.py:9 {T2['url']}. The verifier alone (no code quality "
                "pass), briefed on these items only; the other rows carry from our last review."
            ),
            lines[0],
        )
        self.assertNotIn(T1["url"], lines[0])
        self.assertEqual(
            self.st["review_ready"][0]["micro"],
            [f"thread copilot-pull-request-reviewer at rk/b.py:9 {T2['url']}"],
        )
        self.assertNotIn("equivalent_heads", self.st["prs"]["7"]["ours"])
        self.assertIn(
            "ready for verification (micro: 1 new item)",
            self.store.journal_path.read_text(),
        )

    def test_a_changed_title_or_no_baseline_is_new(self):
        write_items(self.chk, title="Add another thing")
        lines = self.run_review(
            status=self.STATUS + self.TITLE,
            peer={"threads": [], "title": "Add a thing"},
        )
        self.assertIn(
            "micro: title 'Add another thing': no `type(scope): ` prefix.", lines[0]
        )
        write_items(self.chk, threads=[T1])
        lines = self.run_review(
            status=self.STATUS + self.THREADS
        )  # no baseline known: every item is new
        self.assertIn(f"micro: thread yr at rk/a.py:3 {T1['url']}", lines[0])

    def test_unknown_items_or_another_flag_mean_full_verification(self):
        lines = self.run_review(
            status=self.STATUS + self.THREADS, peer={"threads": [], "title": ""}
        )  # no item files
        self.assertIn("the checks flag: threads fails (2 live threads", lines[0])
        self.assertNotIn("micro", lines[0])
        write_items(self.chk, threads=[T1, T2])
        status = (
            self.STATUS
            + self.THREADS
            + "trackers fail 1 tracker ID in added lines: TKEY"
            "-2967\n"
        )
        lines = self.run_review(
            status=status, peer={"threads": [T1["url"]], "title": ""}
        )
        self.assertIn("the checks flag: ", lines[0])
        self.assertIn("threads fails", lines[0])
        self.assertIn("trackers fails", lines[0])
        self.assertNotIn("micro", lines[0])
        self.assertNotIn("micro", self.st["review_ready"][0])

    def flagged_status(self, test: str) -> str:
        return (
            self.STATUS.replace(
                "tests skip no test to run\n",
                f"tests fail 1 failing (on the merge only): {test}\n",
            ).replace(
                "installed pass 4 run of 4\n",
                f"installed fail 1 run of 1: 1 failing (on the merge only): "
                f"{test} - AssertionError\n",
            )
            + f"prnumbers fail 1 pull-request reference: {REF['at']}\n"
        )

    def stage_ours(self):
        (self.chk.parent / "pr").mkdir(exist_ok=True)
        (self.chk.parent / "pr" / "ours.md").write_text(
            f"# Our last review post on PR #7\n- Head: {self.OLD}\n- Posted: 2026-11-01T10:00:00Z\n\n## Review\n"
            + OURS_BODY
        )

    def test_failing_tests_and_references_our_post_reports_carry_the_head(self):
        """Our post reports the failing test and the reference; the merge flags the same test and the reference on a
        moved line, and every probe reads as before: nothing is posted."""
        self.stage_ours()
        write_tested(self.chk, tests=[TEST], installed=[TEST], refs=[REF])
        lines = self.run_review(
            status=self.flagged_status(TEST), peer={"threads": [], "title": ""}
        )
        self.assertEqual(
            lines,
            [
                f"REVIEW_SKIP #7: merge-only since our review (1 merge; its checks are clean; its "
                f"tests, installed and prnumbers flags repeat what our review saw, so nothing is "
                f"posted); recorded {self.NEW[:10]} as equivalent"
            ],
        )
        rec = self.st["prs"]["7"]
        self.assertEqual(
            {k: rec["peer_seen"][k] for k in NO_ITEMS},
            {
                "tests": [TEST],
                "installed": [TEST],
                "prnumbers": [[REF_FILE, REF["what"]]],
            },
        )
        self.assertEqual(self.store.ledger(), [])

    def test_a_new_failing_test_or_reference_goes_to_verification(self):
        self.stage_ours()
        write_tested(self.chk, tests=[TEST_NEW], installed=[TEST], refs=[REF])
        lines = self.run_review(
            status=self.flagged_status(TEST_NEW), peer={"threads": [], "title": ""}
        )
        self.assertIn(
            f"the checks flag: tests fails (1 failing (on the merge only): {TEST_NEW})",
            lines[0],
        )
        self.assertNotIn("micro", lines[0])
        write_tested(self.chk, tests=[TEST], refs=[REF, REF_NEW])
        lines = self.run_review(
            status=self.flagged_status(TEST), peer={"threads": [], "title": ""}
        )
        self.assertIn("Merge-only since our review, but the checks flag: ", lines[0])
        self.assertIn("prnumbers fails (1 pull-request reference", lines[0])
        self.assertNotIn("micro", lines[0])

    def test_a_flag_without_an_item_record_goes_to_verification(self):
        self.stage_ours()  # the post names them, but the checks wrote no item lists
        lines = self.run_review(
            status=self.flagged_status(TEST), peer={"threads": [], "title": ""}
        )
        for flag in ("tests fails", "installed fails", "prnumbers fails"):
            self.assertIn(flag, lines[0])

    def test_a_micro_round_beside_known_ci_infrastructure(self):
        write_items(self.chk, threads=[T1, T2])
        self.ci = {self.NEW: {"Windows shard 1"}, self.OLD: set()}
        self.infra = {"Windows shard 1": "signature `pwsh: command not found`"}
        lines = self.run_review(
            status=self.STATUS + self.THREADS,
            peer={"threads": [T1["url"]], "title": ""},
        )
        self.assertTrue(lines[0].startswith(f"CI_INFRA #7 at {self.NEW[:10]}: "), lines)
        self.assertIn("micro: thread copilot", lines[1])
        self.assertNotIn("CI fails at the head", lines[1])
        # Every item seen, beside known infrastructure: carried.
        lines = self.run_review(
            status=self.STATUS + self.THREADS, peer={"threads": [T1["url"], T2["url"]]}
        )
        self.assertTrue(
            lines[0].startswith(
                "REVIEW_SKIP #7: merge-only since our review (1 merge; its only flag is "
                "CI failures that are known infrastructure; its threads flag repeats"
            ),
            lines,
        )


if __name__ == "__main__":
    unittest.main()
