# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rk draft-lint (rkmon_lib/draft_lint.py, lessons L-501): the worker's known errors are flagged against the draft's own
inputs, the draft is never edited, and a well-formed draft gets no flags."""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import draft_lint  # noqa: E402

TK = "TK" + "EY"  # a fake ticket key, built at run time
LINE1 = "Automated review by rocke-monitor: Model - qwen, effort: high"
META = f"""# PR #7: feat(rocke): a change
- Author: @dev-a
- State: OPEN; mergeable: MERGEABLE
- Head: users/dev-a/x @ abcdef1234; base: develop @ 1234567890
- Labels: rocke: platform
- Operator: @op-login
- Round: Follow-up
- Line 1: {LINE1}

## Description
ISSUE ID: {TK}-100
"""
COMMITS = """# Commits (oldest first)
- aaaa1111 2026-11-01T10:00:00Z feat: first
- bbbb2222 2026-11-03T10:00:00Z fix: second
"""
THREADS = """# Inline review threads (our own comments omitted)

## rk/a.py:10 [RESOLVED]
### @copilot (2026-11-02T09:00:00Z)
Possible None dereference.
### @dev-a (2026-11-02T12:00:00Z)
Fixed in bbbb2222.

## rk/b.py:20
### @dev-b (2026-11-04T09:00:00Z)
Please add a test for the empty case.
"""
REVIEWS = """# Reviews from OTHER reviewers (oldest first; commit = the head the reviewer saw).

## @dev-b COMMENTED at bbbb2222 (2026-11-04T09:00:01Z)
(no body)
"""
COMMENTS = "# Conversation comments from OTHERS (oldest first). Our own are omitted.\n"
CHECKS = """conclusion\tcheck\turl (cancelled = superseded or stopped, not a failure)
success\tpre-commit\thttps://x/1
success\tComponent CI Summary\thttps://x/2
in_progress\tLinux shard 1/2\thttps://x/3
"""
LOG_GOOD = """
=== step 0 (t+10s) think=500c calls=2 ===
  TOOL rule_scan({})
  -> == tracker: 0
== perf: 0
  TOOL merge_conflicts({})
  -> (no output)

=== step 1 (t+40s) think=300c calls=1 ===
  TOOL run_pytest({"paths": "rk/tests/test_a.py"})
  -> ....s
4 passed, 1 skipped in 2.00s

=== step 2 (t+60s) think=100c calls=0 ===
"""
GOOD = f"""{LINE1}

## PR #7 — a change (dev-a, {TK}-100) — 🟠 Follow-up: test the empty case

| Prior item | Status |
|---|---|
| Copilot (11-02) ×1: None dereference | 🟢 FIXED `bbbb2222`<br>• guard added (read) |
| dev-b (11-04) ×1: empty-case test | 🟠 OPEN<br>• no test yet (read) |
| Bots / CI | 🟡 running<br>• 2 ✓ · 1 running · 0 ✗ |
| Our checks | 🟢 Pass<br>• merge: clean (ran)<br>• own test run: 4 ✓ / 1 skip (ran) |

Net-new: none.

**Utility:** 🟢 Positive: a small fix.

---
**Legend:** 🔴 blocking · 🟠 should-fix · 🟡 minor nit · 🟢 info/good · ⚪ retracted

_Follow-up review: first pass by a local qwen model._
"""
BAD = (
    f"""## PR #7 — a change (feature/x, {TK}-999) — 🟢 ACCEPT: looks good

| Prior item | Status |
|---|---|
| Copilot (11-02) ×3: None dereference | 🟢 FIXED `deadbeef`<br>• guard added (read) |
| dev-b (11-04) ×1: empty-case test | 🟡 DEFERRED<br>• later (read) |
| Ours (11-03) N1: a leak | 🟢 FIXED (read) |
| Approvals | Stale |
| Bots / CI | 🟢 green<br>• 3 ✓; the shard failure is unrelated |
| Our checks | 🟢 clean<br>• merge: clean (ran)<br>• own test run: 12 ✓ (ran)<br>• `rk/c.py` reads fine (ran) |

| # | Sev | file:line | Finding |
|---|---|---|---|
| N1 | 🟡 | `rk/a.py:3` | **`read_text()` without `encoding=`**<br>• fix: add encoding (ran) |
| N2 | 🟡 | `rk/b.py:9` | **perf: the loop slows the throughput**<br>• fix: hoist it |
"""
    + "".join(f"| N9 | 🟡 | `rk/x.py:1` | **same** |\n" for _ in range(40))
    + """
**Utility:** 🟢 CI clean, 3 commits behind develop. Saved to review.md.

_First-round review: first pass._
"""
)
LOG_BAD = (
    "".join(
        f"""
=== step {k} (t+{k * 10}s) think={40000 if k == 3 else 100}c calls=1 ===
  TOOL read_file({{"path": "rk/a.py"}})
  -> x = 1
"""
        for k in range(8)
    )
    + "\n=== finalize (t+900s): forcing review, think off ===\nfinalize produced 900c\n"
)


def make(d: pathlib.Path, draft: str, log: str) -> pathlib.Path:
    (d / "pr").mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("meta.md", META),
        ("commits.md", COMMITS),
        ("threads.md", THREADS),
        ("reviews.md", REVIEWS),
        ("comments.md", COMMENTS),
        ("checks.txt", CHECKS),
    ):
        (d / "pr" / name).write_text(text, encoding="utf-8")
    (d / "review_agent.md").write_text(draft, encoding="utf-8")
    (d / "agent_log.txt").write_text(log, encoding="utf-8")
    (d / "stats_agent.json").write_text('{"steps": 3}', encoding="utf-8")
    return d


class TestDraftLint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_well_formed_draft_gets_no_flags(self):
        ws = make(self.d, GOOD, LOG_GOOD)
        self.assertEqual(draft_lint.lint(ws), [])
        lines, code, k = draft_lint.run(7, ws)
        self.assertEqual((code, k), (0, 0))
        self.assertIn(
            "No flags.", (ws / "checks" / "lint.md").read_text(encoding="utf-8")
        )

    def test_draft_lint_flags_known_worker_errors(self):
        ws = make(self.d, BAD, LOG_BAD)
        flags = draft_lint.lint(ws)
        text = "\n".join(f"{a}: {b}" for a, b in flags)
        expected = [
            ("L-001", "the verdict word ACCEPT"),
            ("L-001", "line 1 is '## PR #7"),
            ("L-001", "the header lacks"),
            ("L-009", "the header's author is feature/x; meta.md's `Author:` is dev-a"),
            ("L-014", f"{TK}-999 is in neither meta.md nor commits.md"),
            ("L-010", '"Stale"'),
            ("L-012", 'a "Saved to" line'),
            ("L-002", "a behind/ahead count"),
            (
                "L-003",
                'own test run: 12 ✓ (ran)": no run_pytest call printed a pytest summary',
            ),
            (
                "L-003",
                '`rk/c.py` reads fine (ran)": no run_pytest, rule_scan or merge_conflicts output backs it',
            ),
            ("L-003", 'merge: clean (ran)": no merge_conflicts call'),
            ("L-004", "Bots / CI does not say running; checks.txt has 1 not completed"),
            ("L-004", '"unrelated": write "cause not checked"'),
            ("L-004", "Bots / CI says 3 ✓; checks.txt has 2"),
            ("L-004", '"CI clean" while checks.txt has 0 failed and 1 not completed'),
            ("L-006", 'prior row "Ours (11-03) N1: a leak": our own items'),
            ("L-010", "FIXED at `deadbeef`, which is not in commits.md"),
            ("L-017", "×3, but threads.md has 1 thread(s) by Copilot"),
            (
                "L-019",
                'prior row "dev-b (11-04) ×1: empty-case test": DEFERRED with no reply and no later commit',
            ),
            ("L-013", "finding N1: a text-I/O hint as a finding"),
            ("L-016", "finding N2: a perf hint with no measured time or throughput"),
            ("L-101", "a table row repeated 40 times"),
            ("L-101", "the worker was forced to finalize"),
            ("L-101", "a step thought 40,000 characters"),
            ("L-101", 'read_file({"path": "rk/a.py"}) in 8 of 8 steps'),
        ]
        for lesson, part in expected:
            self.assertTrue(
                any(a == lesson and part in b for a, b in flags),
                f"{lesson} {part!r} not in:\n{text}",
            )
        # No flag for the clean merge itself: merge_conflicts was never called, which L-003 names once.
        self.assertFalse(any(a == "L-005" for a, _ in flags), text)

    def test_the_draft_is_never_edited_and_the_report_lands_in_checks(self):
        ws = make(self.d, BAD, LOG_BAD)
        before = (ws / "review_agent.md").read_bytes()
        lines, code, k = draft_lint.run(7, ws)
        self.assertEqual((ws / "review_agent.md").read_bytes(), before)
        self.assertEqual(code, 1)
        report = (ws / "checks" / "lint.md").read_text(encoding="utf-8")
        self.assertTrue(report.startswith("# Draft lint for #7"))
        self.assertIn(f"{k} flag(s): ", report)
        self.assertEqual(lines[0], "# Draft lint for #7")

    def test_merge_words_against_merge_conflicts(self):
        log = LOG_GOOD.replace("  -> (no output)", "  -> rk/a.py\nrk/b.py")
        ws = make(self.d, GOOD, log)
        self.assertIn(
            ("L-005", "the draft says merge clean; merge_conflicts listed 2 file(s)"),
            draft_lint.lint(ws),
        )

    def test_test_counts_must_match_a_pytest_summary(self):
        ws = make(self.d, GOOD.replace("4 ✓ / 1 skip", "40 ✓ / 1 skip"), LOG_GOOD)
        flags = draft_lint.lint(ws)
        self.assertEqual([a for a, _ in flags], ["L-003"], flags)
        self.assertIn("no pytest summary printed these counts", flags[0][1])

    def test_round_and_footer_against_meta(self):
        draft = GOOD.replace("🟠 Follow-up: test", "🟠 First-round: test")
        flags = draft_lint.lint(make(self.d, draft, LOG_GOOD))
        self.assertIn(
            ("L-001", "the header says First-round; meta.md's `Round:` is Follow-up"),
            flags,
        )
        self.assertIn(
            ("L-001", "the footer says Follow-up; the header says First-round"), flags
        )

    def test_without_meta_s_line_1_line_1_must_give_an_automated_reviewer_name(self):
        """With no `Line 1:` in meta.md, the draft's line 1 must give a name, as an older monitor's did not."""
        ws = make(self.d, GOOD, LOG_GOOD)
        (ws / "pr" / "meta.md").write_text(
            META.replace(f"- Line 1: {LINE1}\n", ""), encoding="utf-8"
        )
        self.assertEqual(draft_lint.lint(ws), [])
        older = "Automated review: Model - qwen, effort: high"
        (ws / "review_agent.md").write_text(
            GOOD.replace(LINE1, older), encoding="utf-8"
        )
        self.assertEqual(
            draft_lint.lint(ws),
            [
                (
                    "L-001",
                    f"line 1 is {older!r}: no `Automated review by <automated_reviewer_name>:` line",
                )
            ],
        )

    def test_no_draft_is_exit_2_and_ready_note_says_nothing(self):
        (self.d / "pr").mkdir()
        lines, code, k = draft_lint.run(7, self.d)
        self.assertEqual((code, k), (2, 0))
        self.assertEqual(draft_lint.ready_note(7, self.d), "")
        self.assertFalse((self.d / "checks" / "lint.md").exists())

    def test_ready_note_names_the_report_and_never_raises(self):
        ws = make(self.d, BAD, LOG_BAD)
        note = draft_lint.ready_note(7, ws)
        self.assertTrue(note.startswith(f" Draft lint: {ws}/checks/lint.md ("), note)
        self.assertIn("flags; open it after the blind pass)", note)
        (ws / "checks" / "lint.md").unlink()
        (ws / "checks").rmdir()
        (ws / "checks").write_text(
            "not a folder", encoding="utf-8"
        )  # the report cannot be written
        self.assertTrue(draft_lint.ready_note(7, ws).startswith(" Draft lint failed: "))


class TestReviewReady(unittest.TestCase):
    def test_review_ready_runs_the_lint_on_a_local_draft(self):
        """REVIEW_READY of a local worker draft names checks/lint.md, written beside the draft."""
        from unittest import mock
        from rkmon_lib import review
        from rkmon_lib.store import Store, empty_state

        cfg = {
            "review_first_line": LINE1,
            "rocke_path": "rk",
            "repo": "o/r",
            "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
        }
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            ws = make(root / "workers" / "pr1", BAD, LOG_BAD)
            st = empty_state()
            st["prs"] = {
                "1": {
                    "n": 1,
                    "head": "1" * 40,
                    "status": "open",
                    "draft": False,
                    "size": 10,
                }
            }
            st["review_pipes"] = [
                {
                    "pr": 1,
                    "kind": "follow-up",
                    "stage": "worker",
                    "head": "1" * 40,
                    "worker": {"id": "w", "started": "2000-01-01T00:00:00Z"},
                    "checks": {"id": "c", "started": "2000-01-01T00:00:00Z"},
                }
            ]
            cfg["paths"] = {"working_dir": str(root)}
            with (
                mock.patch.object(
                    review.jobs, "status", lambda store, job: "done" if job else "none"
                ),
                mock.patch.object(
                    review, "read_checks_status", lambda path: ({}, None, None)
                ),
            ):
                lines = review.advance(cfg, st, Store(root / "state"), [])
            self.assertTrue(lines[0].startswith("REVIEW_READY #1"), lines)
            self.assertIn(f" Draft lint: {ws}/checks/lint.md (", lines[0])
            self.assertTrue((ws / "checks" / "lint.md").is_file())


if __name__ == "__main__":
    unittest.main()
