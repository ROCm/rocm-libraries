# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the byte-identity gate's pass/fail rule (run_diff.py).

The gate prints "RESULT: GREEN - engine builds and .ll emission is
byte-identical to Python" on a zero exit code, so every way the runner can exit
zero is a sentence the gate asserts. These tests pin the rule to the dashboards
it is allowed to say that about.

Lives beside its subject: the differential harness needs a compiler and the
source tree, so CMake excludes this directory from the installed test tree, and
a test that imports ``run_diff`` has to be excluded with it. These tests
therefore run in ``run_all.py``'s pytest pass over the source tree -- the same
lane that runs the gate itself -- and not in the installed ctest lane.

Host-only. ``gate_verdict`` and ``family_status`` are pure over their arguments,
and the tests that drive ``run_family``/``main`` stub out every subprocess.
Nothing here builds an archive, compiles an emitter, or needs a GPU -- though
importing ``run_diff`` does create its /tmp working dir.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

# The differential harness is runner tooling, not a package; import it the way
# its other callers do (see ir_artifact_diff.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_diff  # noqa: E402 -- after sys.path shim
from run_diff import (  # noqa: E402
    DRIFT_VERDICTS,
    GATE_FAIL,
    GATE_PASS,
    family_status,
    gate_verdict,
)


def pre_fix_exit_code(results):
    """run_diff.py's old rule: fail only on DRIFT, pass everything else.

    Kept as the baseline each regression test measures against. A test that only
    asserts the new rule fails cannot tell a real fix from a status name that
    was always failing; asserting this returns 0 on the same dashboard proves
    the case is one the shipped runner used to wave through.

    Scoped to run_diff.py's exit code. check_byte_identity.py separately grepped
    its stdout for COMPILE_FAIL, so that one status was already fatal a layer
    up -- this change moves the rule into the exit code and drops the grep.
    """
    return 1 if [r for r in results if r["status"] == "DRIFT"] else 0


def green(family, n=4):
    return {"family": family, "status": "GREEN", "n": n, "bad": 0, "canon": 0}


class TestGateVerdict(unittest.TestCase):
    """The pass/fail rule over a whole dashboard."""

    def assert_newly_failing(self, results):
        """The dashboard passed the old rule and fails the new one."""
        self.assertEqual(pre_fix_exit_code(results), 0, "not a regression case")
        exit_code, failures = gate_verdict(results)
        self.assertEqual(exit_code, 1)
        return failures

    def test_the_two_partitions_are_disjoint(self):
        # "Named in exactly one of these two maps" is the whole claim, and
        # gate_verdict consults GATE_FAIL first -- so an overlapping key would
        # silently shadow its GATE_PASS entry and read as classified twice.
        self.assertEqual(set(GATE_PASS) & set(GATE_FAIL), set())

    def test_clean_dashboard_passes(self):
        results = [green("gemm_fp16"), green("conv_fwd"), green("attention_tiled")]
        self.assertEqual(gate_verdict(results), (0, []))

    def test_a_run_with_no_families_fails(self):
        # The whole-run form of NO_CONFIGS: --only matched nothing, or the
        # parity dirs are missing. Exiting 0 here claims byte-identity over an
        # empty set.
        exit_code, failures = gate_verdict([])
        self.assertEqual(exit_code, 1)
        self.assertEqual(failures[0].status, "NO_FAMILIES")

    def test_range_drift_fails(self):
        # One engine stopped emitting configs before the other: the two
        # disagree about how many kernels exist.
        results = [
            green("gemm_fp16"),
            {
                "family": "conv_fwd",
                "status": "RANGE_DRIFT",
                "n": 3,
                "bad": 0,
                "canon": 0,
                "range_drift": {"idx": 3, "c_end": True, "p_end": False},
            },
        ]
        failures = self.assert_newly_failing(results)
        self.assertEqual([f.family for f in failures], ["conv_fwd"])
        self.assertIn("C++ ended first", failures[0].reason)

    def test_family_that_compared_nothing_fails(self):
        # Zero configs is a vacuous pass, not a clean one.
        results = [
            green("gemm_fp16"),
            {"family": "conv_fwd", "status": "NO_CONFIGS", "n": 0, "bad": 0},
        ]
        failures = self.assert_newly_failing(results)
        self.assertEqual([f.family for f in failures], ["conv_fwd"])

    def test_family_that_only_rejected_fails(self):
        # Configs were sampled but no bytes were ever compared.
        results = [{"family": "conv_fwd", "status": "ALL_REJECTED", "n": 6, "bad": 0}]
        self.assert_newly_failing(results)

    def test_unrecognized_status_fails(self):
        # The generator of the others: a status nobody classified. This is what
        # a future batching status would arrive as.
        results = [green("gemm_fp16"), {"family": "conv_fwd", "status": "BATCH_SHORT"}]
        failures = self.assert_newly_failing(results)
        self.assertEqual(failures[0][:2], ("conv_fwd", "BATCH_SHORT"))
        self.assertIn("unrecognized", failures[0].reason)

    def test_compile_fail_fails(self):
        # A family that did not build compared nothing, while the gate's GREEN
        # line claims "engine builds".
        results = [{"family": "conv_fwd", "status": "COMPILE_FAIL", "configs": []}]
        self.assert_newly_failing(results)

    def test_drift_still_fails(self):
        results = [
            green("gemm_fp16"),
            {"family": "conv_fwd", "status": "DRIFT", "n": 6, "bad": 2, "canon": 0},
        ]
        exit_code, failures = gate_verdict(results)
        self.assertEqual(exit_code, 1)
        self.assertIn("2 of 6", failures[0].reason)

    def test_canon_only_passes(self):
        results = [{"family": "gemm_fp16", "status": "CANON_ONLY", "n": 4, "bad": 0}]
        self.assertEqual(gate_verdict(results), (0, []))

    def test_mode_unsupported_passes(self):
        # Admissible only because the gating mode cannot produce it -- an
        # emitter predating the mode arg is a coverage gap in the non-gating
        # ir/verify lanes. TestLlModeInvariant pins that guard.
        results = [{"family": "conv_fwd", "status": "MODE_UNSUPPORTED", "mode": "ir"}]
        self.assertEqual(gate_verdict(results), (0, []))

    def test_every_failure_is_reported_not_just_the_first(self):
        results = [
            {
                "family": "a",
                "status": "RANGE_DRIFT",
                "range_drift": {"idx": 1, "c_end": False, "p_end": True},
            },
            green("b"),
            {"family": "c", "status": "NO_CONFIGS", "n": 0},
        ]
        _, failures = gate_verdict(results)
        self.assertEqual([f.family for f in failures], ["a", "c"])

    def test_evidence_is_omitted_rather_than_guessed(self):
        # A record carrying no end-of-range evidence must not name an engine.
        # Guessing which side truncated prints a confident falsehood.
        _, failures = gate_verdict(
            [{"family": "a", "status": "RANGE_DRIFT", "range_drift": {"idx": 1}}]
        )
        self.assertNotIn("ended first", failures[0].reason)


class TestFamilyStatus(unittest.TestCase):
    """The per-family ladder that produces the statuses the gate rules on."""

    @staticmethod
    def configs(*verdicts):
        return [{"idx": i, "verdict": v} for i, v in enumerate(verdicts)]

    def test_all_identical_is_green(self):
        self.assertEqual(
            family_status(self.configs("IDENTICAL", "IDENTICAL"), None),
            ("GREEN", 0, 0),
        )

    def test_no_configs_is_not_green(self):
        self.assertEqual(family_status([], None).status, "NO_CONFIGS")

    def test_a_family_that_only_rejected_is_not_green(self):
        # n > 0, nothing byte-compared: NO_CONFIGS one config-list longer.
        self.assertEqual(
            family_status(self.configs("BOTH_REJECTED", "BOTH_REJECTED"), None).status,
            "ALL_REJECTED",
        )

    def test_canon_equal_alone_is_canon_only(self):
        # Without this the rung is unpinned: delete it and the family falls
        # through to GREEN, silently erasing the SSA-id-numbering signal.
        self.assertEqual(
            family_status(self.configs("CANON_EQUAL"), None), ("CANON_ONLY", 0, 1)
        )

    def test_short_range_is_range_drift(self):
        rd = {"idx": 2, "c_end": True, "p_end": False}
        self.assertEqual(
            family_status(self.configs("IDENTICAL", "IDENTICAL"), rd).status,
            "RANGE_DRIFT",
        )

    def test_short_range_outranks_canon_only(self):
        # A family can differ only in SSA-id numbering *and* be truncated; the
        # truncation is the finding, and CANON_ONLY passes the gate.
        rd = {"idx": 1, "c_end": False, "p_end": True}
        self.assertEqual(
            family_status(self.configs("CANON_EQUAL"), rd).status, "RANGE_DRIFT"
        )

    def test_drift_outranks_short_range(self):
        rd = {"idx": 1, "c_end": True, "p_end": False}
        status, nbad, _ = family_status(self.configs("MISMATCH"), rd)
        self.assertEqual((status, nbad), ("DRIFT", 1))

    def test_every_drift_verdict_counts_as_drift(self):
        # Driven off DRIFT_VERDICTS itself, so a verdict added there is covered
        # here without anyone remembering to update this list.
        self.assertTrue(DRIFT_VERDICTS)
        for verdict in DRIFT_VERDICTS:
            with self.subTest(verdict=verdict):
                self.assertEqual(
                    family_status(self.configs(verdict), None).status, "DRIFT"
                )

    def test_rejects_alongside_a_comparison_are_green(self):
        # Both engines rejecting the same spec is agreement, not drift -- as
        # long as something else in the family was actually compared.
        self.assertEqual(
            family_status(self.configs("BOTH_REJECTED", "IDENTICAL"), None).status,
            "GREEN",
        )


class TestLlModeInvariant(unittest.TestCase):
    """MODE_UNSUPPORTED passes only because ll mode cannot produce it."""

    def test_ll_mode_never_reports_mode_unsupported(self):
        # An emitter that fumbles argv prints a usage line. In ir/verify that is
        # a known coverage gap; in ll -- the gating mode -- it must not be able
        # to masquerade as one.
        usage = (2, b"", "usage: gemm_emit <idx> [mode]")

        def status(mode):
            with mock.patch.multiple(
                run_diff,
                compile_c=mock.DEFAULT,
                run_c=mock.DEFAULT,
                run_py=mock.DEFAULT,
            ) as m:
                m["compile_c"].return_value = (Path("emit_c"), "")
                m["run_c"].return_value = usage
                m["run_py"].return_value = usage
                return run_diff.run_family("gemm", Path("archive.a"), mode)["status"]

        self.assertEqual(status("ir"), "MODE_UNSUPPORTED")
        ll = status("ll")
        self.assertNotEqual(ll, "MODE_UNSUPPORTED")
        # It compared nothing, so it must not pass under another name either.
        self.assertIn(ll, GATE_FAIL)


class TestMainReturnsTheGateVerdict(unittest.TestCase):
    """The pure rule is only a gate if main() actually returns it."""

    def drive_main(self, family_record, tmp):
        archive = tmp / "librocke_core.a"
        archive.touch()
        argv = [
            "run_diff.py",
            "--archive",
            str(archive),
            "--json",
            str(tmp / "dashboard.json"),
        ]
        families = [("fake", tmp)] if family_record else []
        with mock.patch.object(run_diff, "find_families", return_value=families):
            with mock.patch.object(run_diff, "archive_build_id", return_value="stub"):
                with mock.patch.object(
                    run_diff, "run_family", return_value=family_record
                ):
                    with mock.patch.object(sys, "argv", argv):
                        return run_diff.main()

    def test_a_range_drift_family_exits_nonzero(self):
        with TemporaryDirectory() as d:
            self.assertEqual(
                self.drive_main(
                    {
                        "family": "fake",
                        "status": "RANGE_DRIFT",
                        "n": 1,
                        "bad": 0,
                        "canon": 0,
                        "range_drift": {"idx": 1, "c_end": True, "p_end": False},
                        "configs": [],
                    },
                    Path(d),
                ),
                1,
            )

    def test_a_green_family_exits_zero(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            code = self.drive_main(
                {
                    "family": "fake",
                    "status": "GREEN",
                    "n": 1,
                    "bad": 0,
                    "canon": 0,
                    "range_drift": None,
                    "configs": [{"idx": 0, "verdict": "IDENTICAL", "ref_sha": "ab"}],
                },
                tmp,
            )
            self.assertEqual(code, 0)
            # And the dashboard it wrote is the one that verdict came from.
            self.assertEqual(
                gate_verdict(json.loads((tmp / "dashboard.json").read_text())), (0, [])
            )

    def test_discovering_no_families_exits_nonzero(self):
        with TemporaryDirectory() as d:
            self.assertEqual(self.drive_main(None, Path(d)), 1)


if __name__ == "__main__":
    unittest.main()
