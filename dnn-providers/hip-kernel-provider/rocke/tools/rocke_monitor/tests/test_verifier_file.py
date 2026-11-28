# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The verifier's team member file as a review in parts and check 18 read it (rkmon_lib/verifier_file.py):
the package's file keeps each checklist lead a part quotes, and a lead a file lacks is missing to both, by
the one reading of the leads they share.
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import split, verifier_file  # noqa: E402

PACKAGE_FILE = ROOT / "agentic" / verifier_file.FILE


def package_text() -> str:
    return PACKAGE_FILE.read_text(encoding="utf-8")


def without(text: str, lead: str) -> str:
    """TEXT with LEAD's bullet left in, but its lead no longer in bold: the file lacks the lead."""
    out = text.replace(f"- **{lead}", f"- {lead}")
    if out == text:
        raise AssertionError(f"no bullet starts with **{lead}")
    return out


class TestLeads(unittest.TestCase):
    def test_the_package_s_verifier_keeps_each_lead(self):
        self.assertEqual(verifier_file.missing(package_text()), [])

    def test_each_lead_once_in_the_order_of_each_kind_s(self):
        self.assertEqual(
            verifier_file.leads(["tests", "build"]),
            [
                "New test assertions",
                '"CTest labels and tiers"',
                '"References to moved or deleted files"',
            ],
        )
        every = verifier_file.leads()
        self.assertEqual(len(every), len(set(every)))
        self.assertEqual(
            set(every),
            {lead for leads in verifier_file.CHECKS.values() for lead in leads},
        )
        self.assertEqual(verifier_file.leads(["not a kind"]), [])

    def test_a_bullet_at_any_indent_starts_with_its_lead_in_bold(self):
        lines = [
            "- Launch paths, in plain words",
            "  - **New guards, defaults**, and where they apply",
            "- **Launch paths**",
        ]
        self.assertEqual(verifier_file.bullet(lines, "New guards, defaults"), 1)
        self.assertEqual(verifier_file.bullet(lines, "Launch paths"), 2)
        self.assertEqual(verifier_file.bullet(lines, "New IR op handlers"), -1)

    def test_a_lead_the_file_lacks_is_missing_to_check_18_and_to_a_part(self):
        """What check 18 names as missing is what a part of each kind that quotes the lead says is missing."""
        lead = "New IR op handlers"
        text = without(package_text(), lead)
        self.assertEqual(verifier_file.missing(text), [lead])
        for kind, leads in verifier_file.CHECKS.items():
            with self.subTest(kind=kind):
                said = [
                    line
                    for line in split.checklist(text, [kind])
                    if line.startswith("- !! ")
                ]
                missing = [f"- !! not in the verifier team file: **{lead}**"]
                self.assertEqual(said, missing if lead in leads else [])


if __name__ == "__main__":
    unittest.main()
