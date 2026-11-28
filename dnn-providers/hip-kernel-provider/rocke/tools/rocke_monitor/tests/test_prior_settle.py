# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""worker/prior_settle.py, the rules Phase 0 settles prior items by before any model: our last review's rows and
findings (pr/ours.md), their labels and the files they cite, the saved probes' labels, what holds no review point, the
Threads row's data, carrying an item whose cited files did not change, and which statuses flip a 🔴/🟠.
"""

import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.dont_write_bytecode = True

import prior_settle as ps  # noqa: E402

OURS = """# Our last review post on PR #9
- Head: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
- Posted: 2026-11-06T20:46:23Z

## Review
Automated review: Model - X, effort: high

## PR #9 — t (a) — 🟠 Follow-up: open

| Prior item | Status |
|---|---|
| Ours (11-06) N12: `BINARY_ABI = 10` beside `ROCKE_ABI_VERSION 11` | 🟢 FIXED `eb22e07d43`<br>• loads at v11 (ran) |
| Ours (11-06) N10, Copilot (11-05): ABI bump; its history entry | 🟠 OPEN<br>• `abi.h:102-107` relabels 10 (ran) |
| dev-j (11-04), Copilot (11-05): `register_prefetch` unwired | 🟡 DEFERRED<br>• to 56789 (read) |
| Earlier items: ours N1–N3, N5; dev-k P2 and thread nudges | 🟢 FIXED, still hold<br>• goldens PASS (ran) |
| Approvals: dev-o | 🟢 APPROVED `05a74c739e`<br>• older head |
| Bots / CI | 🟡 Running<br>• 24 ✓ |
| Our checks | 🟠 Pass<br>• merge vs `8deaef86ac`: clean, 14 behind (ran) |
| Code quality | 🟡 Delta holds |

| # | Sev | file:line | Finding |
|---|---|---|---|
| N21 | 🟠 | `src/a/CMakeLists.txt`, `b.hpp` + 5 | **Conflicts with develop**<br>• impact: no merge<br>• fix: merge develop |
| N22 | 🟡 | `x.py:6` | **helper adds one**<br>• fix: return v |
"""


class TestOurLastReview(unittest.TestCase):
    def test_its_rows_findings_head_and_develop(self):
        ours = ps.parse_ours(OURS)
        self.assertEqual(
            (ours["head"], ours["at"], ours["kept"], ours["develop"]),
            ("a" * 40, "2026-11-06T20:46:23Z", False, "8deaef86ac"),
        )
        self.assertEqual(
            [(r["light"], r["status"], r["sha"]) for r in ours["rows"]][:4],
            [
                ("🟢", "FIXED", "eb22e07d43"),
                ("🟠", "OPEN", ""),
                ("🟡", "DEFERRED", ""),
                ("🟢", "FIXED", ""),
            ],
        )
        self.assertEqual(
            ours["rows"][0]["line"], 12
        )  # its line in ours.md, for the quote that is its evidence
        self.assertEqual(
            [(f["id"], f["light"], f["problem"]) for f in ours["findings"]],
            [("N21", "🟠", "Conflicts with develop"), ("N22", "🟡", "helper adds one")],
        )
        self.assertIsNone(ps.parse_ours("no review here"))

    def test_its_items_leave_the_data_rows_out(self):
        items = ps.row_items(ps.parse_ours(OURS))
        self.assertEqual(
            [i["key"][:30] for i in items],
            [
                "Ours (11-06) N12: `BINARY_ABI ",
                "Ours (11-06) N10, Copilot (11-",
                "dev-j (11-04), Copilot (11-05)",
                "Earlier items: ours N1–N3, N5;",
                "Ours (11-06) N21: Conflicts wi",
                "Ours (11-06) N22: helper adds ",
            ],
        )
        rows = {i["key"][:17]: i for i in items}
        self.assertEqual(rows["Ours (11-06) N12:"]["labels"], ["N12"])
        self.assertEqual(
            rows["Earlier items: ou"]["labels"], ["N1", "N2", "N3", "N5", "P2"]
        )
        self.assertEqual(rows["Ours (11-06) N10,"]["cites"], [("abi.h", 102)])
        self.assertEqual(rows["Ours (11-06) N10,"]["prev"]["status"], "OPEN")
        n21 = next(i for i in items if i["labels"] == ["N21"])
        self.assertEqual(
            n21["cites"], [("src/a/CMakeLists.txt", 0), ("b.hpp", 0)]
        )  # a cell may name several files
        n22 = next(i for i in items if i["labels"] == ["N22"])
        self.assertEqual(
            (n22["where"], n22["cites"], n22["prev"]["status"]),
            ("x.py:6", [("x.py", 6)], "OPEN"),
        )

    def test_a_review_kept_unposted_reads_as_a_post_and_as_kept(self):
        """While review posting is off our last review can be one kept unposted: `rk stage-ours` stages it as "Our last
        review", with the time it was kept in place of the post's. Its head, time, rows and findings read as a post's,
        and it reads as kept.
        """
        kept = OURS.replace(
            "# Our last review post on", "# Our last review on"
        ).replace("\n- Posted: ", "\n- Kept unposted: ")
        self.assertNotEqual(kept, OURS)
        ours, posted = ps.parse_ours(kept), ps.parse_ours(OURS)
        self.assertEqual((ours.pop("kept"), posted.pop("kept")), (True, False))
        self.assertEqual(ours, posted)


class TestRules(unittest.TestCase):
    def test_labels_and_probe_labels(self):
        self.assertEqual(
            ps.labels("Our 11-03 – 11-05 N1–N3, N16-N18; C1–C2 `N9 in code`"),
            ["N1", "N2", "N3", "N16", "N17", "N18", "C1", "C2"],
        )
        self.assertEqual(
            [
                ps.probe_label(n)
                for n in (
                    "N12_abi_mirror",
                    "N2b_1102_x",
                    "N2fu4_aot",
                    "KH2_1103_y",
                    "n1_helper",
                )
            ],
            ["N12", "N2b", "N2", "KH2", ""],
        )

    def test_citations(self):
        text = (
            "see `platform/python/rocke/core/lower_llvm.py:3542-3575` and `lower_hip.py:2103`, the intrinsic "
            "`llvm.amdgcn.global.atomic.fadd.v2f16`, `register_prefetch`, platform/tests/golden/all.json, and "
            "https://github.com/ROCm/r/blob/develop/guide/FAQ.md"
        )
        self.assertEqual(
            ps.citations(text),
            [
                ("platform/python/rocke/core/lower_llvm.py", 3542),
                ("lower_hip.py", 2103),
                ("platform/tests/golden/all.json", 0),
            ],
        )

    def test_a_name_resolves_to_one_file_or_a_few(self):
        tree = [
            "k/a/abi.h",
            "k/b/abi.py",
            "k/c/__init__.py",
            "k/d/__init__.py",
            "k/lower_hip.py",
        ]
        r = ps.Resolver(["k/b/abi.py"], lambda: tree, "k")
        self.assertEqual(
            r.resolve("abi.py"), ["k/b/abi.py"]
        )  # the PR's own files first
        self.assertEqual(r.resolve("abi.h"), ["k/a/abi.h"])
        self.assertEqual(r.resolve("a/abi.h"), ["k/a/abi.h"])  # under the project path
        self.assertEqual(r.resolve("./lower_hip.py"), ["k/lower_hip.py"])
        self.assertEqual(
            r.resolve("__init__.py"), ["k/c/__init__.py", "k/d/__init__.py"]
        )
        self.assertEqual(r.resolve("gone.py"), [])
        many = ps.Resolver([], lambda: [f"k/{i}/x.py" for i in range(9)], "k")
        self.assertEqual(
            many.resolve("x.py"), []
        )  # names too many files: cites none of them

    def test_what_flips_a_serious_status(self):
        self.assertTrue(ps.flips("🟠", "OPEN", "🟢", "FIXED"))  # a 🟠 closes
        self.assertTrue(ps.flips("🟢", "FIXED", "🟠", "OPEN"))  # one reopens
        self.assertTrue(
            ps.flips("🔴", "OPEN", "🟠", "OPEN")
        )  # it moves between 🔴 and 🟠
        self.assertTrue(ps.flips("", "", "🔴", "OPEN"))  # a new 🔴
        self.assertFalse(ps.flips("🟠", "OPEN", "🟠", "OPEN"))
        self.assertFalse(ps.flips("🟡", "OPEN", "🟢", "FIXED"))
        self.assertFalse(ps.flips("", "", "🟡", "PARTIAL"))
        self.assertFalse(ps.flips("🟠", "DEFERRED", "🟠", "DEFERRED"))

    def test_what_holds_no_review_point(self):
        rx = {"a bot's overview": "<!-- ccr-overview"}
        self.assertEqual(
            ps.skip_reason(
                {"kind": "comment", "who": "Alice", "text": "Rebased."}, "alice", rx
            ),
            "the PR author's reply or status note",
        )
        self.assertEqual(
            ps.skip_reason(
                {
                    "kind": "review",
                    "who": "copilot",
                    "text": "<!-- ccr-overview-v2 -->",
                },
                "alice",
                rx,
            ),
            "a bot's overview",
        )
        self.assertEqual(
            ps.skip_reason(
                {"kind": "thread", "who": "bob", "text": "a real point"}, "alice", rx
            ),
            "",
        )

    def test_the_threads_rows_data(self):
        text = (
            "# Inline review threads\n\n## a/x.py:5 [OPEN]\n### @dev-b (t)\nfix this\n\n"
            "## a/y.py:9 [RESOLVED, outdated]\n### @dev-c (t)\nok\n"
        )
        st = ps.threads_state(text)
        self.assertEqual(
            ps.threads_value(st), {"total": 2, "open": 1, "resolved": 1, "outdated": 1}
        )
        self.assertEqual(
            [(o["path"], o["line"], o["who"]) for o in st["open_list"]],
            [("a/x.py", "5", "dev-b")],
        )

    def test_unchanged_files_carry_by_their_blobs(self):
        def git(*a):
            return subprocess.run(
                ["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        with tempfile.TemporaryDirectory() as d:
            git("init", "-q")
            pathlib.Path(d, "a.py").write_text("1\n")
            pathlib.Path(d, "b.py").write_text("1\n")
            git("add", "-A")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            pathlib.Path(d, "b.py").write_text("2\n")
            git("commit", "-qam", "then")
            then = git("rev-parse", "HEAD")
            pathlib.Path(d, "b.py").write_text("3\n")
            git("commit", "-qam", "now")
            now = git("rev-parse", "HEAD")
            same, rec = ps.unchanged(d, then, now, base, base, ["a.py"])
            self.assertTrue(same)
            self.assertEqual(
                set(rec), {"since", "head", "develop_then", "develop", "files"}
            )
            same, rec = ps.unchanged(d, then, now, base, base, ["a.py", "b.py"])
            self.assertFalse(same)
            self.assertEqual(ps.changed_paths(rec), ["b.py"])
            self.assertEqual(
                ps.unchanged(d, then, now, base, "f" * 40, ["a.py"])[0], None
            )  # not in the clone


if __name__ == "__main__":
    unittest.main()
