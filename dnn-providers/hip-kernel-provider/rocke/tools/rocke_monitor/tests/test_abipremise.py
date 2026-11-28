# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ABI premise of `rk post --premises` (rkmon_lib/abipremise.py): a post's "no bump" against the checks' ABI
section, and the ABI numbers it names against develop's counters. Temp folders, no git, no network.
"""

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

from rkmon_lib import abipremise, premises  # noqa: E402

DEV = {"binary": 16, "recipe": 4}
SECTION_NOTES = (
    "## ABI (abi.h counters and abi.py mirrors, C struct layouts and exported functions in include/rocke/)\n"
    "ROCKE_ABI_VERSION: develop 16; no bump\n"
    "NOTE (G9 waits for `rk abi-ok N` after review): C struct layouts change without a binary bump: rocke_gemm_spec "
    "(a field added anywhere changes the size: bump ROCKE_ABI_VERSION)\n"
    "ABI fail 1 note: C struct layouts change without a binary bump\n\n## rule_scan\nnothing\n"
)
SECTION_CLEAN = (
    "## ABI (abi.h counters and abi.py mirrors)\nROCKE_ABI_VERSION: develop 16; no bump\n"
    "no struct layout or exported function change in include/rocke/ without a bump\nABI pass no ABI change\n"
)


def cfg_for(d: str) -> dict:
    return {"paths": {"working_dir": d, "workers": f"{d}/workers"}, "rocke_path": "rk"}


def checks(d: str, n: int, text: str) -> None:
    p = pathlib.Path(d, "workers", f"pr{n}", "checks", "checks.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# checks\n" + text, encoding="utf-8")


class TestParts(unittest.TestCase):
    def test_the_no_bump_phrases(self):
        for s in (
            "no bump needed",
            "needs no ABI bump",
            "does not need a bump",
            "doesn't require an ABI bump",
            "no ABI change",
            "ABI-neutral",
        ):
            self.assertTrue(abipremise.NO_BUMP.search(s), s)
        for s in ("bump ROCKE_ABI_VERSION", "the bump to 17", "no bumps of the tier"):
            self.assertFalse(abipremise.NO_BUMP.search(s), s)

    def test_the_numbers_a_post_names_per_counter(self):
        self.assertEqual(
            abipremise.numbers(
                "bump ROCKE_ABI_VERSION to 17; abi.py BINARY_ABI = 17; ABI 16 -> 17; recipe ABI 5; "
                "RECIPE_ABI 5; see abi.h:42 and 3 PRs"
            ),
            {"binary": [17, 17, 16, 17], "recipe": [5, 5]},
        )

    def test_the_section_and_its_notes(self):
        body = abipremise.section("# x\n" + SECTION_NOTES)
        self.assertNotIn("rule_scan", body)
        self.assertEqual(len(abipremise.notes_of(body)), 1)
        self.assertIsNone(abipremise.section("# x\n## other\n"))
        self.assertEqual(
            abipremise.abi_line("- abi: abi.h's rule: an appended function, no bump\n"),
            "abi.h's rule: an appended function, no bump",
        )
        self.assertIsNone(abipremise.abi_line("abi: trust it\n"))  # names no abi.h rule


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.cfg = cfg_for(self.d.name)

    def run_check(self, text, notes="", n=7):
        return abipremise.check(self.cfg, n, text, notes, None, counters=DEV)

    def test_no_bump_over_a_changed_layout_stops_unless_the_notes_name_abi_h(self):
        checks(self.d.name, 7, SECTION_NOTES)
        with self.assertRaises(abipremise.AbiStop) as stop:
            self.run_check("| N2 | 🟢 | ABI | no bump needed: the struct only grows |")
        self.assertIn(
            "lists 1 change without a bump (C struct layouts change",
            str(stop.exception),
        )
        self.assertIn("abi.h's when-to-bump rule decides", str(stop.exception))
        lines = self.run_check(
            "| N2 | 🟢 | ABI | no bump needed |",
            notes="abi: abi.h's rule exempts internal-only structs; rocke_gemm_spec is not exported\n",
        )
        self.assertIn("the notes' abi: line", lines[0])

    def test_no_bump_over_a_clean_section_or_none_passes_and_says_so(self):
        checks(self.d.name, 7, SECTION_CLEAN)
        self.assertIn("lists no change", self.run_check("no ABI change here")[0])
        self.assertIn("no ABI section", self.run_check("no ABI change", n=8)[0])
        self.assertEqual(self.run_check("rename the helper"), [])

    def test_a_number_neither_develops_nor_the_next_stops(self):
        checks(self.d.name, 7, SECTION_CLEAN)
        self.assertIn(
            "ROCKE_ABI_VERSION 16, 17 against develop 16",
            self.run_check(
                "develop is at ABI 16; this takes ROCKE_ABI_VERSION 17 when it lands"
            )[0],
        )
        with self.assertRaises(abipremise.AbiStop) as stop:
            self.run_check("bump ROCKE_ABI_VERSION to 18")
        self.assertIn(
            "ROCKE_ABI_VERSION 18, which is neither develop's 16 nor develop + 1 (17)",
            str(stop.exception),
        )
        with self.assertRaises(abipremise.AbiStop):
            self.run_check("recipe ABI 7")
        self.assertIn(
            "allowed by the notes' abi: line",
            self.run_check(
                "ABI 15 is taken", notes="abi: abi.h history shows 15 used by develop\n"
            )[0],
        )

    def test_develop_unread_checks_no_number(self):
        lines = abipremise.check(self.cfg, 7, "ABI 30", "", None, counters={})
        self.assertIn("not read", lines[0])


class TestPremises(unittest.TestCase):
    def test_check_runs_the_abi_premise_on_the_filled_post(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_for(d)
            checks(d, 7, SECTION_NOTES)
            notes = pathlib.Path(d) / "notes.md"
            notes.write_text("nothing\n")
            ready = {
                "text": "| N1 | 🟠 | abi.h | no bump needed |",
                "sha": "a" * 40,
                "dev": "b" * 40,
            }
            with mock.patch.object(
                premises, "_premises", return_value={}
            ), mock.patch.object(
                premises, "_post_checks", return_value=ready
            ), mock.patch.object(
                abipremise, "develop_counters", return_value=DEV
            ):
                lines, got = premises.check(cfg, None, 7, {"NOTES": str(notes)})
                self.assertIsNone(got)
                self.assertIn("abi premise", lines[-1])
                self.assertTrue(lines[-1].endswith(": stop"))
                ready["text"] = "| N1 | 🟠 | a.py:1 | rename it |"
                lines, got = premises.check(cfg, None, 7, {"NOTES": str(notes)})
                self.assertIs(got, ready)


if __name__ == "__main__":
    unittest.main()
