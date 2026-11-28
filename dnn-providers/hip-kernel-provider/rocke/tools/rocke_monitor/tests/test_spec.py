# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rk spec: reading a verifier's notes.md and matching its FAILOK names to the live failing jobs."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rkmon_lib import spec  # noqa: E402


class TestNamed(unittest.TestCase):
    def test_the_same_name(self):
        self.assertTrue(
            spec._named("HipDNN Windows Superbuild", "HipDNN Windows Superbuild")
        )

    def test_a_cut_name(self):
        self.assertTrue(
            spec._named(
                "Windows::release / Build Multi-Arch...",
                "Windows::release / Build Multi-Arch Stages",
            )
        )

    def test_a_name_followed_by_its_reason(self):
        self.assertTrue(
            spec._named(
                'hipDNN Superbuild Summary (Windows runner lacks pwsh, checks.md "CI at head")',
                "hipDNN Superbuild Summary",
            )
        )

    def test_a_longer_note_does_not_name_a_shorter_job_without_the_reason_parenthesis(
        self,
    ):
        self.assertFalse(spec._named("hipDNN Superbuild Summary", "hipDNN Superbuild"))


class TestNotesFacts(unittest.TestCase):
    def test_failok_names_split_on_semicolons(self):
        f = spec.notes_facts(
            "FAILOK: HipDNN Windows Superbuild; hipDNN Superbuild Summary (runner lacks pwsh)\n"
        )
        self.assertEqual(len(f["failok"]), 2)
        self.assertTrue(spec._named(f["failok"][1], "hipDNN Superbuild Summary"))

    def test_a_failok_line_that_names_no_job(self):
        for value in (
            "none",
            "None",
            "(none)",
            "(None)",
            "NONE.",
            "-",
            "(-)",
            "",
            "none: all green",
            "(none) (rk ci-fails, latest attempts)",
        ):
            self.assertEqual(
                spec.notes_facts(f"FAILOK: {value}\n")["failok"], [], value
            )
        self.assertEqual(
            spec.notes_facts("FAILOK: a job; (none)\n")["failok"], ["a job"]
        )
        self.assertIsNone(
            spec.notes_facts("premises: head abc1234\n")["failok"]
        )  # no line: not the same as none


class TestSpecLists(unittest.TestCase):
    def test_failok_and_runok_entries_that_name_no_job_are_dropped(self):
        from rkmon_lib import premises

        self.assertEqual(
            premises.job_names(
                ["(none)", "none", "-", "", " NONE ", "a job", "none-such job"]
            ),
            ["a job", "none-such job"],
        )


if __name__ == "__main__":
    unittest.main()
