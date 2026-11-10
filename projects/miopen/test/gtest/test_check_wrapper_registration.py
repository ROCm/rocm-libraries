#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for check_wrapper_registration.py.

Dropping the per-architecture wrapper_abi_check and forwarding_failure_modes entries
leaves every other test passing while TheRock's GPU jobs stop selecting the checks.
These tests feed the check registrations built the way CMake writes them, so it fails
here if it stops noticing a missing or mislabelled entry.

Written against the standard library's unittest rather than pytest: this runs as a
ctest entry in a wrapper-enabled build, and nothing provisions pytest for a machine
that builds MIOpen.

    python3 -m unittest test_check_wrapper_registration
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import check_wrapper_registration as registration

TIERS = ["quick", "standard", "comprehensive", "full"]
GPU_LABELS = ["ex_gpu_gfx110X", "ex_gpu_gfx950"]
CHECKS = ("wrapper_abi_check", "forwarding_failure_modes")


def registered(drop=(), relabel=None, tiers=TIERS):
    """Return {entry name: labels} as a correct registration, then damaged."""
    entries = {registration.PARITY: {registration.PARITY_ONLY_LABEL, *tiers}}
    for check in CHECKS:
        entries[check] = set(tiers)
        for label in GPU_LABELS:
            entries[f"{check}_{label}"] = {*tiers, label}
    for name in drop:
        del entries[name]
    entries.update(relabel or {})
    return entries


def as_ctest_json(entries):
    return json.dumps(
        {
            "tests": [
                {
                    "name": name,
                    "properties": [{"name": "LABELS", "value": sorted(labels)}],
                }
                for name, labels in entries.items()
            ]
        }
    )


def as_packaged_list(entries, unlabelled=(), not_added=()):
    """The text miopen_append_installed_wrapper_check and the parity entry write."""
    lines = []
    for name, labels in entries.items():
        if name not in not_added:
            lines.append(f"add_test({name} python3 ../miopen_wrapper_tests/x.py)")
        if name not in unlabelled:
            lines.append(
                f'set_tests_properties({name} PROPERTIES TIMEOUT 300 LABELS "{";".join(sorted(labels))}")'
            )
    return "\n".join(lines) + "\n"


class ProblemsTest(unittest.TestCase):
    def problems(self, entries, labels=GPU_LABELS):
        return registration.problems(entries, labels, "build tree")

    def test_a_complete_registration_has_no_problems(self):
        self.assertEqual(self.problems(registered()), [])

    def test_a_missing_per_arch_entry_is_named(self):
        found = self.problems(registered(drop=["wrapper_abi_check_ex_gpu_gfx950"]))
        self.assertEqual(
            found, ["build tree: wrapper_abi_check_ex_gpu_gfx950 is not registered"]
        )

    def test_both_checks_are_required_on_every_architecture(self):
        entries = registered(
            drop=[f"{check}_{label}" for check in CHECKS for label in GPU_LABELS]
        )
        self.assertEqual(len(self.problems(entries)), 4)

    def test_a_missing_bare_entry_is_named(self):
        found = self.problems(registered(drop=["forwarding_failure_modes"]))
        self.assertEqual(
            found, ["build tree: forwarding_failure_modes is not registered"]
        )

    def test_a_per_arch_entry_without_its_arch_label_is_reported(self):
        found = self.problems(
            registered(relabel={"wrapper_abi_check_ex_gpu_gfx110X": set(TIERS)})
        )
        self.assertEqual(len(found), 1)
        self.assertIn("wrapper_abi_check_ex_gpu_gfx110X has labels", found[0])

    def test_a_check_that_lost_a_tier_label_is_reported(self):
        found = self.problems(
            registered(relabel={"wrapper_abi_check": {"quick", "standard", "full"}})
        )
        self.assertEqual(len(found), 1)
        self.assertIn("wrapper_abi_check has labels", found[0])

    def test_tier_labels_come_from_the_parity_entry_not_from_the_checks(self):
        """Both checks losing a tier together must still be noticed."""
        entries = registered()
        for name in entries:
            if name != registration.PARITY:
                entries[name] = entries[name] - {"comprehensive"}
        self.assertTrue(self.problems(entries))

    def test_no_expected_labels_is_a_problem_rather_than_a_pass(self):
        """An emptied label list would otherwise make every registration look right."""
        found = self.problems(registered(drop=["wrapper_abi_check_ex_gpu_gfx950"]), [])
        self.assertEqual(len(found), 1)
        self.assertIn("no ex_gpu_* labels", found[0])

    def test_a_missing_parity_entry_is_a_problem(self):
        found = self.problems(registered(drop=[registration.PARITY]))
        self.assertEqual(len(found), 1)
        self.assertIn(registration.PARITY, found[0])


class LoadersTest(unittest.TestCase):
    def test_ctest_json_labels_are_read_per_entry(self):
        entries = registered()
        self.assertEqual(
            registration.entries_from_ctest_json(as_ctest_json(entries)), entries
        )

    def test_the_packaged_list_is_read_per_entry(self):
        entries = registered()
        self.assertEqual(
            registration.entries_from_packaged(as_packaged_list(entries)), entries
        )

    def test_a_packaged_entry_without_labels_does_not_count(self):
        text = as_packaged_list(
            registered(), unlabelled=["forwarding_failure_modes_ex_gpu_gfx950"]
        )
        found = registration.problems(
            registration.entries_from_packaged(text), GPU_LABELS, "packaged list"
        )
        self.assertEqual(
            found,
            ["packaged list: forwarding_failure_modes_ex_gpu_gfx950 is not registered"],
        )

    def test_a_packaged_entry_with_labels_but_no_add_test_does_not_count(self):
        text = as_packaged_list(
            registered(), not_added=["wrapper_abi_check_ex_gpu_gfx950"]
        )
        found = registration.problems(
            registration.entries_from_packaged(text), GPU_LABELS, "packaged list"
        )
        self.assertEqual(
            found, ["packaged list: wrapper_abi_check_ex_gpu_gfx950 is not registered"]
        )

    def test_labels_of_a_property_line_with_other_properties_are_found(self):
        """The parity entry puts RUN_SERIAL and ENVIRONMENT around its labels."""
        text = (
            "add_test(miopen_gtest_forwarding_parity python3 x.py)\n"
            "set_tests_properties(miopen_gtest_forwarding_parity PROPERTIES RUN_SERIAL TRUE "
            'TIMEOUT 3600 LABELS "forwarding_parity;quick" ENVIRONMENT "A=B")\n'
        )
        self.assertEqual(
            registration.entries_from_packaged(text),
            {"miopen_gtest_forwarding_parity": {"forwarding_parity", "quick"}},
        )


class MainTest(unittest.TestCase):
    def run_main(self, packaged_text, labels=";".join(GPU_LABELS)):
        with tempfile.TemporaryDirectory() as holder:
            packaged = Path(holder) / "install_CTestTestfile.cmake"
            packaged.write_text(packaged_text)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = registration.main(
                    ["--labels", labels, "--packaged", str(packaged)]
                )
        return code, output.getvalue()

    def test_a_complete_packaged_list_exits_zero(self):
        code, output = self.run_main(as_packaged_list(registered()))
        self.assertEqual(code, 0, output)

    def test_a_damaged_packaged_list_exits_non_zero_and_says_why(self):
        text = as_packaged_list(registered(drop=["wrapper_abi_check_ex_gpu_gfx110X"]))
        code, output = self.run_main(text)
        self.assertEqual(code, 1)
        self.assertIn("wrapper_abi_check_ex_gpu_gfx110X is not registered", output)


if __name__ == "__main__":
    unittest.main()
