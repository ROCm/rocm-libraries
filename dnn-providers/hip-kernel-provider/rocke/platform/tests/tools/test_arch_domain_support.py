# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only support-policy controls, collected in source and installed CI.

The exhaustive compiler sweep remains a separate CTest. These tests exercise
losses, gains and invalid records without depending on a real LLVM installation.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
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
                "probe_contracts": (
                    {
                        "a.b": G._probe_contract(
                            G.ProbeSpec("declare i32 @llvm.test(i32)", None, None)
                        )
                    }
                    if status == G.STATUS_OK
                    else {}
                ),
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

    def test_unknown_or_misattributed_compiler_identity_fails(self):
        valid = json.loads(self._doc())
        for identity in (
            "(unknown)",
            " \t(unknown) \n",
            "AMD clang version 20.0.0git (another-build)",
            "AMD LLVM version 23.0.0git",
        ):
            with self.subTest(identity=identity):
                invalid = json.loads(self._doc(clang=identity))
                self._assert_invalid_on_either_or_both_sides(
                    valid, invalid, "toolchain clang", repr(identity)
                )

    def test_same_flavor_builds_and_reworded_compiler_banners_pass(self):
        for identity in (
            "AMD clang version 22.0.1 (another-build)",
            "AMD LLVM version 22.0.0git",
            "a vendor reworded the banner",
        ):
            with self.subTest(identity=identity):
                self.assertEqual(
                    G._compare_support(self._doc(), self._doc(clang=identity)).failures,
                    [],
                )

    def test_measured_arch_cells_require_an_accepted_processor(self):
        for status in (G.STATUS_OK, G.STATUS_ARCH_ABSENT):
            with self.subTest(status=status):
                invalid = json.loads(self._doc(status))
                invalid["processors"][self.ARCH] = {
                    "accepted": False,
                    "evidence": "processor diagnostic",
                }
                for other_status in (G.STATUS_OK, G.STATUS_ARCH_ABSENT):
                    valid = json.loads(self._doc(other_status))
                    self._assert_invalid_on_either_or_both_sides(
                        valid,
                        invalid,
                        f"a.b/{self.ARCH}",
                        f"status {status!r}",
                        "accepted=False",
                    )
                    self.assertEqual(
                        G._compare_support(
                            json.dumps(invalid), json.dumps(valid)
                        ).changes["gain"],
                        [],
                    )

    def test_flavor_axis_and_no_data_can_precede_rejected_processors(self):
        for status in (
            G.STATUS_NAME_ABSENT,
            G.STATUS_TARGET_UNSUPPORTED,
            G.STATUS_TOOLCHAIN_CRASH,
            G.STATUS_TOOLCHAIN_TIMEOUT,
        ):
            with self.subTest(status=status):
                before = json.loads(self._doc(status))
                before["processors"][self.ARCH] = {
                    "accepted": False,
                    "evidence": "processor diagnostic",
                }
                self.assertEqual(
                    G._compare_support(json.dumps(before), self._doc()).failures, []
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

    def test_column_serialization_keeps_contract_entries_on_single_lines(self):
        doc = json.loads(self._doc())
        doc["policy_metadata"] = {
            "probe_contracts": {"nested": {"detail": 'quoted "value"\n\u03bb'}},
            "enabled": True,
        }
        text = G._serialize_column(doc)
        self.assertEqual(json.loads(text), doc)
        self.assertIn(
            '  "probe_contracts": {\n'
            '    "a.b": {"operands": [{"source": "argument", "type": "i32"}], "return_type": "i32", "version": 1}\n'
            "  },\n",
            text,
        )
        # A nested similarly named field must keep ordinary pretty printing.
        ordinary = json.dumps(doc["policy_metadata"], indent=2, sort_keys=True)
        self.assertIn('  "policy_metadata": ' + ordinary.replace("\n", "\n  "), text)
        self.assertTrue(text.endswith("\n"))
        self.assertFalse(text.endswith("\n\n"))
        self.assertEqual(G._serialize_column(json.loads(text)), text)
        self.assertEqual(
            G._compare_support(json.dumps(doc, indent=2), text).failures, []
        )

    def test_column_serialization_is_independent_of_mapping_insertion_order(self):
        def reversed_mappings(value):
            if isinstance(value, dict):
                return {key: reversed_mappings(value[key]) for key in reversed(value)}
            if isinstance(value, list):
                return [reversed_mappings(item) for item in value]
            return value

        doc = json.loads(self._doc())
        doc["keys"]["new.key"] = doc["keys"]["a.b"]
        doc["probe_contracts"]["new.key"] = doc["probe_contracts"]["a.b"]
        self.assertEqual(
            G._serialize_column(doc), G._serialize_column(reversed_mappings(doc))
        )

    def test_empty_and_legacy_contract_sections_do_not_invent_records(self):
        doc = json.loads(self._doc(G.STATUS_NAME_ABSENT))
        text = G._serialize_column(doc)
        self.assertIn('  "probe_contracts": {},\n', text)
        self.assertEqual(json.loads(text), doc)
        del doc["probe_contracts"]
        self.assertEqual(
            G._serialize_column(doc), json.dumps(doc, indent=2, sort_keys=True) + "\n"
        )
        self.assertEqual(G._serialize_column({}), "{}\n")

    def test_newly_added_positive_key_requires_its_own_contract(self):
        invalid = json.loads(self._doc())
        invalid["keys"]["new.key"] = invalid["keys"]["a.b"]
        valid = json.loads(json.dumps(invalid))
        valid["probe_contracts"]["new.key"] = valid["probe_contracts"]["a.b"]
        self._assert_invalid_on_either_or_both_sides(
            valid, invalid, f"new.key/{self.ARCH}", "missing positive probe contract"
        )
        for text in (json.dumps(valid, indent=2), G._serialize_column(valid)):
            self.assertEqual(G._compare_support(text, text).failures, [])

    def test_probe_contracts_cannot_extend_intrinsic_key_coverage(self):
        valid = json.loads(self._doc())
        invalid = json.loads(json.dumps(valid))
        invalid["probe_contracts"]["orphan.key"] = valid["probe_contracts"]["a.b"]
        diagnostic = "probe contract 'orphan.key' is outside intrinsic coverage"
        self.assertEqual(
            G._validate_column(invalid, "baseline"), [f"baseline {diagnostic}"]
        )
        self._assert_invalid_on_either_or_both_sides(valid, invalid, diagnostic)
        valid["keys"]["orphan.key"] = valid["keys"]["a.b"]
        valid["probe_contracts"] = invalid["probe_contracts"]
        self.assertEqual(G._validate_column(valid, "baseline"), [])
        text = json.dumps(valid)
        self.assertEqual(G._compare_support(text, text).failures, [])

    def test_positive_probe_contract_changes_fail(self):
        for old, new in (
            (
                G.ProbeSpec("declare i32 @llvm.test(i32)", None, None),
                G.ProbeSpec("declare i32 @llvm.test(i32 immarg)", None, None),
            ),
            (
                G.ProbeSpec("declare i32 @llvm.test(i32 immarg)", None, None),
                G.ProbeSpec("declare i32 @llvm.test(i32)", None, None),
            ),
            (
                G.ProbeSpec("declare i32 @llvm.test(i32)", 1, None),
                G.ProbeSpec("declare i32 @llvm.test(i32)", 4, None),
            ),
            (
                G.ProbeSpec("declare i32 @llvm.test(i32)", None, None),
                G.ProbeSpec("declare i64 @llvm.test(i64)", None, None),
            ),
        ):
            with self.subTest(old=old, new=new):
                before, after = json.loads(self._doc()), json.loads(self._doc())
                before["probe_contracts"]["a.b"] = G._probe_contract(old)
                after["probe_contracts"]["a.b"] = G._probe_contract(new)
                for a, b in ((before, after), (after, before)):
                    comparison = G._compare_support(json.dumps(a), json.dumps(b))
                    self.assertIn(
                        f"a.b/{self.ARCH}: positive probe contract changed",
                        "; ".join(comparison.failures),
                    )
                    self.assertIn("baseline=", "; ".join(comparison.failures))
                    self.assertIn("current=", "; ".join(comparison.failures))

    def test_contract_describes_the_operands_actually_rendered(self):
        spec = G.ProbeSpec(
            "declare void @llvm.test(i32, i1 immarg, i32 immarg, ptr, ptr addrspace(3), ptr addrspace(8), metadata)",
            4,
            1,
        )
        contract = G._probe_contract(spec)
        self.assertEqual(
            contract,
            {
                "version": G.PROBE_CONTRACT_VERSION,
                "return_type": "void",
                "operands": [
                    {"type": "i32", "source": "literal", "value": "4"},
                    {"type": "i1", "source": "immediate", "value": "true"},
                    {"type": "i32", "source": "immediate", "value": "1"},
                    {"type": "ptr", "source": "base_pointer"},
                    {"type": "ptr addrspace(3)", "source": "addrspacecast"},
                    {"type": "ptr addrspace(8)", "source": "poison"},
                    {"type": "metadata", "source": "agent_metadata"},
                ],
            },
        )
        self.assertEqual(G._validate_probe_contract(contract, "current"), [])
        text, why = G._probe_module(spec.decl, "", spec.literal_ints, spec.imm_int)
        self.assertEqual(why, "")
        self.assertIn(
            "@llvm.test(i32 4, i1 true, i32 1, ptr %base, ptr addrspace(3) %p4, ptr addrspace(8) poison, metadata !0)",
            text,
        )
        batch, why = G._batch_probe_module([spec, spec], "")
        self.assertEqual(why, "")
        self.assertEqual(batch.count("i32 4, i1 true, i32 1"), 2)

    def test_explicit_witness_agrees_with_the_effective_contract(self):
        valid = json.loads(self._doc())
        valid["probe_contracts"]["a.b"] = G._probe_contract(
            G.ProbeSpec("declare i32 @llvm.test(i32 immarg)", None, 1)
        )
        valid["keys"]["a.b"][self.ARCH]["probe_imm"] = 1
        invalid = json.loads(json.dumps(valid))
        invalid["keys"]["a.b"][self.ARCH]["probe_imm"] = 2
        self._assert_invalid_on_either_or_both_sides(
            valid, invalid, "probe_imm disagrees"
        )

    def test_prune_validates_columns_before_using_record_mappings(self):
        valid = json.loads(self._doc())
        cases = ["{oops", "[]", "null"]
        for field in ("keys", "canonical", "toolchain", "probe_contracts"):
            invalid = json.loads(json.dumps(valid))
            invalid[field] = None
            cases.append(json.dumps(invalid))
        for text in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "intrinsic_arch_domain.llvm22.json"
                path.write_text(text)
                with (
                    mock.patch.object(G, "DATA_DIR", Path(directory)),
                    mock.patch.object(G, "_decl_table") as declarations,
                ):
                    self.assertEqual(G._prune(), 1)
                declarations.assert_not_called()
                self.assertEqual(path.read_text(), text)

    def test_prune_refuses_a_post_transform_validation_failure_without_writing(self):
        """Fault injection covers the final write boundary, not a production defect."""
        doc = json.loads(self._doc())
        doc["keys"]["stale.key"] = doc["keys"]["a.b"]
        doc["probe_contracts"]["stale.key"] = doc["probe_contracts"]["a.b"]
        validate = G._validate_column
        labels = []

        def fail_after_pruning(column, label, **kwargs):
            self.assertEqual(validate(column, label, **kwargs), [])
            labels.append(label)
            if label == "pruned":
                self.assertNotIn("stale.key", column["keys"])
                self.assertNotIn("stale.key", column["probe_contracts"])
                return ["pruned injected validation failure"]
            return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intrinsic_arch_domain.llvm22.json"
            text = G._serialize_column(doc)
            path.write_text(text)
            output = io.StringIO()
            with (
                mock.patch.object(G, "DATA_DIR", Path(directory)),
                mock.patch.object(G, "_decl_table", return_value={"a.b": ""}),
                mock.patch.object(
                    G, "_validate_column", side_effect=fail_after_pruning
                ),
                mock.patch.object(Path, "write_text") as write,
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(G._prune(), 1)
            write.assert_not_called()
            self.assertEqual(labels, ["baseline", "pruned"])
            self.assertIn("pruned injected validation failure", output.getvalue())
            self.assertEqual(path.read_text(), text)

    def test_prune_removes_contracts_with_keys_without_inventing_metadata(self):
        for legacy in (False, True):
            with (
                self.subTest(legacy=legacy),
                tempfile.TemporaryDirectory() as directory,
            ):
                doc = json.loads(self._doc())
                doc["keys"]["fabs.f32"] = doc["keys"].pop("a.b")
                doc["keys"]["stale.key"] = doc["keys"]["fabs.f32"]
                doc["probe_contracts"]["fabs.f32"] = doc["probe_contracts"].pop("a.b")
                doc["probe_contracts"]["stale.key"] = doc["probe_contracts"]["fabs.f32"]
                if legacy:
                    del doc["probe_contracts"]
                path = Path(directory) / "intrinsic_arch_domain.llvm22.json"
                path.write_text(json.dumps(doc))
                with mock.patch.object(G, "DATA_DIR", Path(directory)):
                    self.assertEqual(G._prune(), 0)
                expected = json.loads(json.dumps(doc))
                del expected["keys"]["stale.key"]
                if not legacy:
                    del expected["probe_contracts"]["stale.key"]
                text = path.read_text()
                self.assertEqual(json.loads(text), expected)
                self.assertEqual(text, G._serialize_column(expected))
                if not legacy:
                    self.assertEqual(
                        len(
                            [
                                line
                                for line in text.splitlines()
                                if line.startswith('    "fabs.f32": {"operands":')
                            ]
                        ),
                        1,
                    )
                with mock.patch.object(G, "DATA_DIR", Path(directory)):
                    self.assertEqual(G._prune(), 0)
                self.assertEqual(
                    path.read_text(), text, "a no-op prune changed canonical bytes"
                )

    def test_effective_contract_ignores_names_and_unused_attributes(self):
        old = G.ProbeSpec("declare i32 @llvm.old(i32)", None, None)
        new = G.ProbeSpec("declare noundef i32 @llvm.new(i32 noundef)", None, None)
        self.assertEqual(G._probe_contract(old), G._probe_contract(new))
        before, after = json.loads(self._doc()), json.loads(self._doc())
        before["probe_contracts"]["a.b"] = G._probe_contract(old)
        after["probe_contracts"]["a.b"] = G._probe_contract(new)
        self.assertEqual(
            G._compare_support(json.dumps(before), json.dumps(after)).failures, []
        )

    def test_unparseable_declarations_have_named_contract_and_rendering_refusals(self):
        for decl in ("", "not a declaration", "declare i32 @llvm.test("):
            with self.subTest(decl=decl):
                diagnostic = f"cannot parse declare: {decl!r}"
                spec = G.ProbeSpec(decl, None, None)
                with self.assertRaises(ValueError) as error:
                    G._probe_contract(spec)
                self.assertEqual(str(error.exception), diagnostic)
                self.assertEqual(G._probe_plan(decl), (None, diagnostic))
                self.assertEqual(G._probe_module(decl, ""), ("", diagnostic))
        valid = G.ProbeSpec("declare i32 @llvm.test(i32)", None, None)
        self.assertEqual(
            G._probe_contract(valid), json.loads(self._doc())["probe_contracts"]["a.b"]
        )
        text, why = G._probe_module(valid.decl, "")
        self.assertEqual(why, "")
        self.assertIn("call i32 @llvm.test(i32 %a0)", text)

    def test_missing_historical_positive_contract_cannot_pass(self):
        valid = json.loads(self._doc())
        for damage in ("mapping", "entry"):
            with self.subTest(damage=damage):
                invalid = json.loads(self._doc())
                if damage == "mapping":
                    del invalid["probe_contracts"]
                else:
                    invalid["probe_contracts"].clear()
                self._assert_invalid_on_either_or_both_sides(
                    valid, invalid, "probe contract"
                )

    def test_probe_contract_records_are_validated_not_defaulted(self):
        valid = json.loads(self._doc())
        contract = valid["probe_contracts"]["a.b"]
        candidates = (
            None,
            [],
            {},
            {**contract, "version": True},
            {**contract, "version": 2},
            {**contract, "return_type": None},
            {**contract, "operands": None},
            {**contract, "operands": [None]},
            {**contract, "operands": [{"type": "i32", "source": "immediate"}]},
            {
                **contract,
                "operands": [{"type": "i1", "source": "immediate", "value": "1"}],
            },
            {
                **contract,
                "operands": [{"type": "i32", "source": "argument", "value": "0"}],
            },
            {**contract, "operands": [{"type": "i32", "source": []}]},
        )
        for value in candidates:
            with self.subTest(contract=value):
                invalid = json.loads(self._doc())
                invalid["probe_contracts"]["a.b"] = value
                self._assert_invalid_on_either_or_both_sides(valid, invalid, "probe")
        for value in (None, [], "bad"):
            invalid = json.loads(self._doc())
            invalid["probe_contracts"] = value
            self._assert_invalid_on_either_or_both_sides(
                valid, invalid, "probe_contracts"
            )

    def test_malformed_method_values_fail_on_either_side(self):
        valid = json.loads(self._doc())
        for field, value in (
            ("generator", True),
            ("generator", 0),
            ("generator", "7"),
            ("probe_cflags", "-O0"),
            ("probe_cflags", [None]),
            ("probe_timeout_s", True),
            ("probe_timeout_s", 0),
            ("probe_timeout_s", float("inf")),
            ("arches", [self.ARCH, self.ARCH]),
            ("arches", [""]),
        ):
            with self.subTest(field=field, value=value):
                invalid = json.loads(self._doc())
                invalid["toolchain"][field] = value
                self._assert_invalid_on_either_or_both_sides(valid, invalid, field)

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

    def test_schema_is_validated_independently_of_other_record_fields(self):
        valid = json.loads(self._doc())
        diagnostic = f"schema must be {G.SCHEMA!r}"
        for fields in ({}, *({"schema": value} for value in (None, "wrong", 22, []))):
            with self.subTest(schema=fields):
                invalid = json.loads(json.dumps(valid))
                del invalid["schema"]
                invalid.update(fields)
                self.assertEqual(
                    G._validate_column(invalid, "baseline"), [f"baseline {diagnostic}"]
                )
                self._assert_invalid_on_either_or_both_sides(valid, invalid, diagnostic)
        self.assertEqual(G._validate_column(valid, "baseline"), [])
        self.assertEqual(G._compare_support(self._doc(), self._doc()).failures, [])

    def test_architecture_rows_must_be_mappings_before_comparison(self):
        valid = json.loads(self._doc())
        diagnostic = "architecture row is not a mapping"
        for row in (None, [], "invalid row", 22):
            with self.subTest(row=row):
                invalid = json.loads(json.dumps(valid))
                invalid["keys"]["a.b"] = row
                self.assertEqual(
                    G._validate_column(invalid, "baseline"),
                    [f"a.b: baseline {diagnostic}"],
                )
                self._assert_invalid_on_either_or_both_sides(
                    valid, invalid, "a.b", diagnostic
                )
        self.assertEqual(G._validate_column(valid, "baseline"), [])
        self.assertEqual(G._compare_support(self._doc(), self._doc()).failures, [])

    def test_probe_error_is_invalid_before_transition_classification(self):
        valid = json.loads(self._doc(G.STATUS_ARCH_ABSENT))
        invalid = json.loads(self._doc(G.STATUS_PROBE_ERROR))
        diagnostic = f"invalid status {G.STATUS_PROBE_ERROR!r}"
        self.assertEqual(
            G._validate_column(invalid, "baseline"),
            [f"a.b/{self.ARCH}: baseline {diagnostic}"],
        )
        self._assert_invalid_on_either_or_both_sides(valid, invalid, diagnostic)
        self.assertEqual(G._validate_column(valid, "baseline"), [])

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
                        status = (
                            G.STATUS_OK if accepted else G.STATUS_TARGET_UNSUPPORTED
                        )
                        before, after = json.loads(self._doc(status)), json.loads(
                            self._doc(status)
                        )
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
        before, after = json.loads(self._doc(G.STATUS_TARGET_UNSUPPORTED)), json.loads(
            self._doc(G.STATUS_TARGET_UNSUPPORTED)
        )
        before["processors"] = {
            self.ARCH: {"accepted": False, "evidence": "old wording"}
        }
        after["processors"] = {
            self.ARCH: {"accepted": False, "evidence": "different wording"}
        }
        self.assertEqual(
            G._compare_support(json.dumps(before), json.dumps(after)).failures, []
        )
        after = json.loads(self._doc())
        comparison = G._compare_support(json.dumps(before), json.dumps(after))
        self.assertEqual(comparison.failures, [])
        self.assertEqual(
            comparison.changes["gain"],
            [
                f"a.b/{self.ARCH}: target_unsupported -> ok",
                f"processor {self.ARCH}: unsupported -> accepted",
            ],
        )
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
