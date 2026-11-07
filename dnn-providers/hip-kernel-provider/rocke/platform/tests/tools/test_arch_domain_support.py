# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only support-policy controls, collected in source and installed CI.

The exhaustive compiler sweep remains a separate CTest. These tests exercise
losses, gains and invalid records without depending on a real LLVM installation.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# CMake stages the standalone generator and its private imports beside installed
# tests. Prefer that copy when present; never reach back into a source checkout.
_TESTS = Path(__file__).resolve().parents[1]
_TOOLS = _TESTS / "arch_domain"
if not (_TOOLS / "gen_arch_domain.py").is_file():
    _TOOLS = _TESTS.parent / "tools"
sys.path.insert(0, str(_TOOLS))
import gen_arch_domain as G

_STATUSES = {
    G.STATUS_OK,
    G.STATUS_NAME_ABSENT,
    G.STATUS_ARCH_ABSENT,
    G.STATUS_TARGET_UNSUPPORTED,
    G.STATUS_TOOLCHAIN_CRASH,
    G.STATUS_TOOLCHAIN_TIMEOUT,
    G.STATUS_PROBE_ERROR,
}
_NEEDS_EVIDENCE = _STATUSES - {G.STATUS_OK}


class SupportComparisonTest(unittest.TestCase):
    """Supported capabilities are a floor; observations retain their states."""

    ARCH = "gfx942"

    def _doc(self, status=G.STATUS_OK, **toolchain) -> str:
        tc = {
            "flavor": "llvm22",
            "clang": "AMD clang version 22.0.0git (... roc-7.2.4 26084 abc)",
            "arches": [self.ARCH],
            "generator": G.GENERATOR,
            "probe_cflags": ["-O0", "-nogpulib"],
            "probe_timeout_s": 60,
        }
        tc.update(toolchain)
        return json.dumps(
            {
                "schema": G.SCHEMA,
                "toolchain": tc,
                "keys": {
                    "a.b": {
                        arch: {
                            "status": status,
                            "verified_on": tc["flavor"],
                            "evidence": (
                                "" if status == G.STATUS_OK else "probe diagnostic"
                            ),
                        }
                        for arch in tc["arches"]
                    }
                },
                "canonical": {},
                "processors": {arch: {"accepted": True} for arch in tc["arches"]},
            }
        )

    def _assert_invalid_on_either_or_both_sides(self, valid, invalid, *diagnostics):
        for side, before, after in (
            ("baseline", invalid, valid),
            ("current", valid, invalid),
            ("both", invalid, invalid),
        ):
            with self.subTest(side=side):
                comparison = G._compare_support(json.dumps(before), json.dumps(after))
                self.assertTrue(comparison.failures, "malformed input passed")
                failures = "; ".join(comparison.failures)
                for diagnostic in diagnostics:
                    self.assertIn(diagnostic, failures)
                for label in (("baseline", "current") if side == "both" else (side,)):
                    self.assertIn(label, failures)

    def test_missing_or_invalid_compiler_identity_fails(self):
        valid = json.loads(self._doc())
        for identity in (
            {},
            *({"clang": value} for value in (None, "", " \t", 22, [], {})),
        ):
            with self.subTest(identity=identity):
                invalid = json.loads(self._doc())
                del invalid["toolchain"]["clang"]
                invalid["toolchain"].update(identity)
                self._assert_invalid_on_either_or_both_sides(
                    valid,
                    invalid,
                    "toolchain clang",
                    "nonblank string",
                    f"got {identity.get('clang')!r}",
                )

    def test_non_ok_cells_require_diagnostic_evidence(self):
        for status in sorted(_NEEDS_EVIDENCE - {G.STATUS_PROBE_ERROR}):
            for evidence in (
                {},
                *({"evidence": value} for value in (None, "", " \t", 1, [], {})),
            ):
                with self.subTest(status=status, evidence=evidence):
                    valid, invalid = json.loads(self._doc(status)), json.loads(
                        self._doc(status)
                    )
                    del invalid["keys"]["a.b"][self.ARCH]["evidence"]
                    invalid["keys"]["a.b"][self.ARCH].update(evidence)
                    self._assert_invalid_on_either_or_both_sides(
                        valid,
                        invalid,
                        f"a.b/{self.ARCH}",
                        "evidence",
                        "nonblank string",
                        f"got {evidence.get('evidence')!r}",
                    )

    def test_rejected_processors_require_diagnostic_evidence(self):
        valid = json.loads(self._doc(G.STATUS_TARGET_UNSUPPORTED))
        valid["processors"][self.ARCH] = {
            "accepted": False,
            "evidence": "processor diagnostic",
        }
        for evidence in (
            {},
            *({"evidence": value} for value in (None, "", " \t", 1, [], {})),
        ):
            with self.subTest(evidence=evidence):
                invalid = json.loads(json.dumps(valid))
                del invalid["processors"][self.ARCH]["evidence"]
                invalid["processors"][self.ARCH].update(evidence)
                self._assert_invalid_on_either_or_both_sides(
                    valid,
                    invalid,
                    f"processor {self.ARCH}",
                    "evidence",
                    "nonblank string",
                    f"got {evidence.get('evidence')!r}",
                )

    def test_probe_witness_must_be_an_integer_on_an_ok_cell(self):
        for status in sorted(_STATUSES - {G.STATUS_PROBE_ERROR}):
            values = (None, True, False, "invalid-witness", "1", 1.0, [], {})
            if status != G.STATUS_OK:
                values = (*values, 1)
            for value in values:
                with self.subTest(status=status, witness=value):
                    valid, invalid = json.loads(self._doc(status)), json.loads(
                        self._doc(status)
                    )
                    invalid["keys"]["a.b"][self.ARCH]["probe_imm"] = value
                    self._assert_invalid_on_either_or_both_sides(
                        valid,
                        invalid,
                        f"a.b/{self.ARCH}",
                        "probe_imm",
                        "integer on an ok cell",
                        f"got {value!r}",
                    )

    def test_valid_optional_witnesses_and_gain_witnesses_pass(self):
        for value in (0, 1, -1):
            with self.subTest(witness=value):
                supported = json.loads(self._doc())
                supported["keys"]["a.b"][self.ARCH]["probe_imm"] = value
                text = json.dumps(supported)
                self.assertEqual(G._compare_support(text, text).failures, [])
                gain = G._compare_support(self._doc(G.STATUS_ARCH_ABSENT), text)
                self.assertEqual(gain.failures, [])
                self.assertEqual(len(gain.changes["gain"]), 1)
        self.assertEqual(G._compare_support(self._doc(), self._doc()).failures, [])

    def test_transition_matrix_covers_every_known_pair(self):
        self.assertEqual(set(G._CELL_TRANSITIONS), _STATUSES)
        actions = {
            "unchanged",
            "loss",
            "gain",
            "improved",
            "degraded",
            "invalid",
            "unknown_changed",
            "unavailable_changed",
        }
        for before, row in G._CELL_TRANSITIONS.items():
            with self.subTest(before=before):
                self.assertEqual(set(row), _STATUSES)
                self.assertLessEqual(set(row.values()), actions)
        for before, after in (
            (None, G.STATUS_OK),
            (G.STATUS_OK, {}),
            ("unknown", G.STATUS_OK),
        ):
            with self.subTest(before=before, after=after):
                self.assertEqual(G._cell_transition(before, after), "invalid")

    def test_cell_transition_policy(self):
        for before, after, expected in (
            (G.STATUS_OK, G.STATUS_OK, "unchanged"),
            (G.STATUS_ARCH_ABSENT, G.STATUS_OK, "gain"),
            (G.STATUS_NAME_ABSENT, G.STATUS_OK, "gain"),
            (G.STATUS_TOOLCHAIN_CRASH, G.STATUS_OK, "gain"),
            (G.STATUS_TOOLCHAIN_TIMEOUT, G.STATUS_OK, "gain"),
            (G.STATUS_TARGET_UNSUPPORTED, G.STATUS_OK, "gain"),
            (G.STATUS_TOOLCHAIN_CRASH, G.STATUS_ARCH_ABSENT, "improved"),
            (G.STATUS_TOOLCHAIN_TIMEOUT, G.STATUS_NAME_ABSENT, "improved"),
            (G.STATUS_ARCH_ABSENT, G.STATUS_TOOLCHAIN_CRASH, "degraded"),
            (G.STATUS_NAME_ABSENT, G.STATUS_TARGET_UNSUPPORTED, "degraded"),
            (G.STATUS_TOOLCHAIN_CRASH, G.STATUS_TOOLCHAIN_TIMEOUT, "unknown_changed"),
            (G.STATUS_NAME_ABSENT, G.STATUS_ARCH_ABSENT, "unavailable_changed"),
            (G.STATUS_PROBE_ERROR, G.STATUS_OK, "invalid"),
            (G.STATUS_OK, G.STATUS_PROBE_ERROR, "invalid"),
            (G.STATUS_OK, "unexpected", "invalid"),
        ):
            with self.subTest(before=before, after=after):
                self.assertEqual(G._cell_transition(before, after), expected)

    def test_every_loss_of_working_support_fails_and_names_the_cell(self):
        for status in (
            G.STATUS_ARCH_ABSENT,
            G.STATUS_NAME_ABSENT,
            G.STATUS_TARGET_UNSUPPORTED,
            G.STATUS_TOOLCHAIN_CRASH,
            G.STATUS_TOOLCHAIN_TIMEOUT,
        ):
            with self.subTest(status=status):
                self.assertEqual(G._cell_transition(G.STATUS_OK, status), "loss")
                comparison = G._compare_support(
                    self._doc(), self._doc(status, clang="other build")
                )
                self.assertEqual(len(comparison.failures), 1)
                self.assertIn(
                    f"a.b/{self.ARCH}: ok -> {status}", comparison.failures[0]
                )

    def test_gains_and_measurement_improvements_pass(self):
        for before, after, category in (
            (G.STATUS_ARCH_ABSENT, G.STATUS_OK, "gain"),
            (G.STATUS_NAME_ABSENT, G.STATUS_OK, "gain"),
            (G.STATUS_TOOLCHAIN_CRASH, G.STATUS_OK, "gain"),
            (G.STATUS_TOOLCHAIN_CRASH, G.STATUS_ARCH_ABSENT, "improved"),
        ):
            with self.subTest(before=before, after=after):
                comparison = G._compare_support(self._doc(before), self._doc(after))
                self.assertEqual(comparison.failures, [])
                self.assertEqual(
                    comparison.changes[category],
                    [f"a.b/{self.ARCH}: {before} -> {after}"],
                )

    def test_degraded_measurement_is_reported_not_called_support_loss(self):
        comparison = G._compare_support(
            self._doc(G.STATUS_ARCH_ABSENT), self._doc(G.STATUS_TOOLCHAIN_CRASH)
        )
        self.assertEqual(comparison.failures, [])
        self.assertEqual(len(comparison.changes["degraded"]), 1)
        self.assertEqual(
            comparison.changes["unvalidated"], [f"a.b/{self.ARCH}: toolchain_crash"]
        )

    def test_provenance_and_diagnostic_changes_do_not_fail_support(self):
        for status in (G.STATUS_OK, G.STATUS_ARCH_ABSENT, G.STATUS_TOOLCHAIN_CRASH):
            with self.subTest(status=status):
                fresh = json.loads(self._doc(status, clang="other build"))
                fresh["keys"]["a.b"][self.ARCH]["evidence"] = "different wording"
                fresh["canonical"] = {"a.b": "llvm.new.canonical.name"}
                self.assertEqual(
                    G._compare_support(self._doc(status), json.dumps(fresh)).failures,
                    [],
                )

    def test_probe_method_changes_still_fail(self):
        for field, value in (
            ("probe_cflags", ["-nogpulib"]),
            ("probe_timeout_s", 5),
            ("generator", G.GENERATOR + 1),
            ("flavor", "llvm23"),
            ("arches", ["gfx950" if self.ARCH == "gfx942" else "gfx942"]),
        ):
            with self.subTest(field=field):
                comparison = G._compare_support(
                    self._doc(), self._doc(clang="other build", **{field: value})
                )
                failures = "; ".join(comparison.failures)
                self.assertIn("toolchain scope or probe method changed", failures)
                self.assertIn(
                    f"{field}: baseline={json.loads(self._doc())['toolchain'][field]!r}, current={value!r}",
                    failures,
                )

    def test_metadata_changes_fail_and_name_fields_and_values(self):
        for value in (None, "", {"policy": "candidate"}):
            for operation in ("add", "remove", "change"):
                with self.subTest(value=value, operation=operation):
                    before, after = json.loads(self._doc()), json.loads(self._doc())
                    if operation != "add":
                        before["policy_metadata"] = "recorded"
                    if operation != "remove":
                        after["policy_metadata"] = value
                    comparison = G._compare_support(
                        json.dumps(before), json.dumps(after)
                    )
                    self.assertEqual(len(comparison.failures), 1)
                    old = "<missing>" if operation == "add" else repr("recorded")
                    new = "<missing>" if operation == "remove" else repr(value)
                    self.assertIn("artifact metadata changed", comparison.failures[0])
                    self.assertIn(
                        f"policy_metadata: baseline={old}, current={new}",
                        comparison.failures[0],
                    )

    def test_matching_extra_metadata_and_changed_comments_pass(self):
        before, after = json.loads(self._doc()), json.loads(self._doc())
        for doc in (before, after):
            doc["policy_metadata"] = {"policy": "recorded"}
        before["_comment"] = "recorded commentary"
        after["_comment"] = "current commentary"
        self.assertEqual(
            G._compare_support(json.dumps(before), json.dumps(after)).failures, []
        )

    def test_method_diagnostics_list_each_changed_field(self):
        before, after = json.loads(self._doc()), json.loads(self._doc())
        before["toolchain"]["extra_method"] = None
        after["toolchain"].update(generator=G.GENERATOR + 1, probe_timeout_s=5)
        comparison = G._compare_support(json.dumps(before), json.dumps(after))
        self.assertEqual(len(comparison.failures), 1)
        self.assertIn(
            "extra_method: baseline=None, current=<missing>", comparison.failures[0]
        )
        self.assertIn(
            f"generator: baseline={G.GENERATOR}, current={G.GENERATOR + 1}",
            comparison.failures[0],
        )
        self.assertIn("probe_timeout_s: baseline=60, current=5", comparison.failures[0])

    def test_generator_comes_from_this_test_environment(self):
        self.assertEqual(Path(G.__file__).resolve().parent, _TOOLS.resolve())

    def test_missing_or_invalid_answers_still_fail(self):
        for status in (G.STATUS_OK, G.STATUS_ARCH_ABSENT):
            for damage in ("cell", "row", "status"):
                with self.subTest(status=status, damage=damage):
                    fresh = json.loads(self._doc(status))
                    if damage == "cell":
                        del fresh["keys"]["a.b"][self.ARCH]
                    elif damage == "row":
                        del fresh["keys"]["a.b"]
                    else:
                        fresh["keys"]["a.b"][self.ARCH]["status"] = G.STATUS_PROBE_ERROR
                    comparison = G._compare_support(
                        self._doc(status), json.dumps(fresh)
                    )
                    self.assertTrue(comparison.failures)
                    self.assertIn("a.b", "; ".join(comparison.failures))

    def test_invalid_schema_and_structure_fail_not_crash(self):
        for text in ("{oops", "[]", '{"schema": "wrong"}'):
            with self.subTest(text=text):
                self.assertTrue(G._compare_support(text, self._doc()).failures)
        fresh = json.loads(self._doc())
        fresh["keys"]["a.b"][self.ARCH] = None
        self.assertIn(
            "cell is not a mapping",
            "; ".join(G._compare_support(self._doc(), json.dumps(fresh)).failures),
        )

    def test_unmeasured_architectures_cannot_be_reported_as_gains(self):
        fresh = json.loads(self._doc())
        fresh["processors"] = {"extra-arch": {"accepted": True}}
        self.assertIn(
            "outside probe scope",
            "; ".join(G._compare_support(self._doc(), json.dumps(fresh)).failures),
        )
        both = json.loads(self._doc())
        both["keys"]["a.b"]["extra-arch"] = {"status": G.STATUS_OK}
        self.assertIn(
            "architecture coverage",
            "; ".join(G._compare_support(json.dumps(both), json.dumps(both)).failures),
        )
        both["toolchain"]["arches"] = None
        self.assertIn(
            "arches must be a list",
            "; ".join(G._compare_support(json.dumps(both), json.dumps(both)).failures),
        )

    def test_missing_processor_records_fail_on_either_or_both_sides(self):
        for accepted in (True, False):
            for side in ("baseline", "current", "both"):
                for damage in ("mapping", "entry"):
                    with self.subTest(accepted=accepted, side=side, damage=damage):
                        before, after = json.loads(self._doc()), json.loads(self._doc())
                        for doc in (before, after):
                            doc["processors"][self.ARCH] = {
                                "accepted": accepted,
                                "evidence": "processor diagnostic",
                            }
                        damaged = (
                            (before, after)
                            if side == "both"
                            else ((before,) if side == "baseline" else (after,))
                        )
                        for doc in damaged:
                            if damage == "mapping":
                                del doc["processors"]
                            else:
                                del doc["processors"][self.ARCH]
                        comparison = G._compare_support(
                            json.dumps(before), json.dumps(after)
                        )
                        self.assertTrue(comparison.failures)
                        self.assertIn("processors", "; ".join(comparison.failures))
                        self.assertEqual(
                            comparison.changes["gain"], [], "missing data became a gain"
                        )

    def test_missing_verified_on_fails_on_either_or_both_sides(self):
        for status in (G.STATUS_OK, G.STATUS_ARCH_ABSENT, G.STATUS_TOOLCHAIN_CRASH):
            for side in ("baseline", "current", "both"):
                with self.subTest(status=status, side=side):
                    before, after = json.loads(self._doc(status)), json.loads(
                        self._doc(status)
                    )
                    damaged = (
                        (before, after)
                        if side == "both"
                        else ((before,) if side == "baseline" else (after,))
                    )
                    for doc in damaged:
                        del doc["keys"]["a.b"][self.ARCH]["verified_on"]
                    comparison = G._compare_support(
                        json.dumps(before), json.dumps(after)
                    )
                    self.assertTrue(comparison.failures)
                    self.assertIn(f"a.b/{self.ARCH}", "; ".join(comparison.failures))
                    self.assertIn("verified_on", "; ".join(comparison.failures))

    def test_wrong_or_invalid_verified_on_still_fails(self):
        for value in (None, "llvm24", 22):
            with self.subTest(value=value):
                after = json.loads(self._doc())
                after["keys"]["a.b"][self.ARCH]["verified_on"] = value
                self.assertIn(
                    "verified_on",
                    "; ".join(
                        G._compare_support(self._doc(), json.dumps(after)).failures
                    ),
                )

    def test_probe_witness_changes_still_fail(self):
        before, after = json.loads(self._doc()), json.loads(self._doc())
        before["keys"]["a.b"][self.ARCH]["probe_imm"] = 1
        after["keys"]["a.b"][self.ARCH]["probe_imm"] = 2
        self.assertIn(
            "probe witness changed",
            "; ".join(
                G._compare_support(json.dumps(before), json.dumps(after)).failures
            ),
        )

    def test_processor_acceptance_is_monotonic(self):
        before, after = json.loads(self._doc()), json.loads(self._doc())
        before["processors"] = {
            self.ARCH: {"accepted": False, "evidence": "old wording"}
        }
        after["processors"] = {
            self.ARCH: {"accepted": False, "evidence": "different wording"}
        }
        self.assertEqual(
            G._compare_support(json.dumps(before), json.dumps(after)).failures, []
        )
        after["processors"] = {self.ARCH: {"accepted": True}}
        comparison = G._compare_support(json.dumps(before), json.dumps(after))
        self.assertEqual(comparison.failures, [])
        self.assertEqual(len(comparison.changes["gain"]), 1)
        self.assertIn(
            "accepted -> unsupported",
            "; ".join(
                G._compare_support(json.dumps(after), json.dumps(before)).failures
            ),
        )
        for invalid in (None, {"accepted": "yes"}):
            after["processors"][self.ARCH] = invalid
            self.assertIn(
                "accepted must be boolean",
                "; ".join(
                    G._compare_support(json.dumps(before), json.dumps(after)).failures
                ),
            )


class SupportComparisonGfx950Test(SupportComparisonTest):
    """Apply the same support floor and diagnostic contracts to gfx950."""

    ARCH = "gfx950"
