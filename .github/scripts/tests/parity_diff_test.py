# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import parity_diff  # noqa: E402


class ParityDiffTest(unittest.TestCase):
    def test_classify_verdicts(self):
        self.assertEqual(parity_diff.classify("passed", "passed"), "MATCH")
        self.assertEqual(parity_diff.classify("failed", "failed"), "MATCH")
        self.assertEqual(
            parity_diff.classify("passed", "failed"), "EMU_FAIL_HW_PASS"
        )
        self.assertEqual(
            parity_diff.classify("failed", "passed"), "EMU_PASS_HW_FAIL"
        )
        self.assertEqual(
            parity_diff.classify("passed", "unsupported"), "UNSUPPORTED"
        )

    def test_not_run_is_outside_parity_percentage(self):
        hw = {
            "run_id": 1,
            "job_id": 2,
            "gtest": [
                {"id": "case/0", "status": "passed", "seconds": 0.4},
                {"id": "case/1", "status": "passed", "seconds": 0.1},
            ],
        }
        emu = {
            "runner_label": "aws-linux-scale-rocm-prod",
            "rocm_version": "10.2.0a20260928",
            "gtest_filter": "case/0",
            "cases": [{"id": "case/0", "status": "passed", "seconds": 14.4}],
        }
        result = parity_diff.diff_results(hw, emu)
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["denominator"], 1)
        self.assertEqual(result["parity_pct"], 100.0)
        self.assertEqual(result["baseline_count"], 2)
        self.assertEqual(result["counts"]["NOT_RUN"], 1)
        self.assertEqual(result["not_run"], ["case/1"])

    def test_checked_in_baseline_xorwow_case(self):
        baseline = (
            Path(__file__).resolve().parents[3]
            / "baseline"
            / "hiprand-gfx942-hw.json"
        )
        hw = json.loads(baseline.read_text(encoding="utf-8"))
        emu = {
            "gtest_filter": "hiprand_32/hiprand_api_32.hiprand_generate_test/0",
            "cases": [
                {
                    "id": "hiprand_32/hiprand_api_32.hiprand_generate_test/0",
                    "status": "passed",
                    "seconds": 14.423,
                }
            ],
        }
        result = parity_diff.diff_results(hw, emu)
        self.assertEqual(len(hw["gtest"]), 364)
        # Twelve ordering cases are recorded twice in the job log, so the
        # join key set is 352 unique ids.
        self.assertEqual(result["baseline_count"], 352)
        self.assertEqual(result["rows"][0]["class"], "MATCH")
        self.assertEqual(result["denominator"], 1)
        self.assertEqual(result["counts"]["NOT_RUN"], 351)

    def test_duplicate_ids_are_counted_once(self):
        hw = {
            "gtest": [
                {"id": "case/0", "status": "passed", "seconds": 0.1},
                {"id": "case/1", "status": "passed", "seconds": 0.04},
            ]
        }
        emu = {
            "cases": [
                {"id": "case/0", "status": "passed", "seconds": 11.0},
                {"id": "case/0", "status": "passed", "seconds": 11.0},
                {"id": "case/1", "status": "failed", "seconds": 1000.0},
                {"id": "case/1", "status": "failed", "seconds": 1000.0},
            ],
            "binaries": [
                {"name": "test_hiprand_api", "status": "timeout", "seconds": 7200.12},
            ],
        }
        result = parity_diff.diff_results(hw, emu)
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["counts"]["EMU_FAIL_HW_PASS"], 1)
        self.assertEqual(result["coverage_pct"], 100.0)
        self.assertEqual(result["timed_out_binaries"], 1)
        self.assertEqual(result["slowdowns"][0]["id"], "case/1")
        self.assertAlmostEqual(result["slowdowns"][0]["slowdown"], 25000.0)
        text = parity_diff.render_report({}, emu, result)
        self.assertIn("test_hiprand_api | timeout", text)
        self.assertIn("Timed-out binaries: 1", text)
        self.assertIn("slowdown", text)

    def test_report_roundtrip(self):
        hw = {"gtest": [{"id": "a", "status": "passed", "seconds": 1}]}
        emu = {"cases": [{"id": "a", "status": "failed", "seconds": 2}]}
        result = parity_diff.diff_results(hw, emu)
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.md"
            summary = Path(tmp) / "summary.json"
            hw_path = Path(tmp) / "hw.json"
            emu_path = Path(tmp) / "emu.json"
            hw_path.write_text(json.dumps(hw), encoding="utf-8")
            emu_path.write_text(json.dumps(emu), encoding="utf-8")
            sys.argv = [
                "parity_diff.py",
                "--hw",
                str(hw_path),
                "--emu",
                str(emu_path),
                "--report",
                str(report),
                "--summary-json",
                str(summary),
            ]
            self.assertEqual(parity_diff.main(), 0)
            text = report.read_text(encoding="utf-8")
            self.assertIn("EMU_FAIL_HW_PASS", text)
            parsed = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(parsed["matches"], 0)
            self.assertEqual(parsed["counts"]["EMU_FAIL_HW_PASS"], 1)


if __name__ == "__main__":
    unittest.main()
