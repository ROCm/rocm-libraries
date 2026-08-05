# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CLI tests: the subprocess contract as the caller experiences it.

The caller is a subprocess orchestrator, so it sees exactly two things -- an
exit code and a result file. The invariant worth defending is that it always
sees both. A rejected request that produced no file would leave the caller
unable to distinguish "rocKE declined this shape" from "rocKE crashed", and
those want very different follow-up.

CPU-only. The measured lanes are skipped here by ``--plan-only`` rather than by
detecting their absence, so these tests assert the same behaviour on a machine
that does have a GPU.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from serve.__main__ import EXIT_DECLINED, EXIT_ERROR, EXIT_OK, main
from serve.protocol import REQUEST_SCHEMA

_ENTRY = {
    "attention_request": {
        "op": "attention",
        "batch": 8,
        "nhead_q": 64,
        "nhead_k": 8,
        "seqlen_q": 1,
        "seqlen_k": 8192,
        "hdim_q": 128,
        "hdim_v": 128,
        "arch": "gfx950",
        "mask_type": 1,
        "kv_block_size": 16,
        "num_sms": 256,
        "dtype": "bf16",
    },
    "problem": {
        "total_q": 8,
        "num_seqs": 8,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_seqlen_q": 1,
        "max_seqlen_k": 8192,
        "dtype": "bf16",
    },
    "call_count": 900,
    "softmax_scale": 0.088,
}


class _Harness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_request(self, **overrides) -> Path:
        payload = {
            "schema": REQUEST_SCHEMA,
            "op": "attention",
            "arch": "gfx950",
            "requests": [copy.deepcopy(_ENTRY)],
            "output_dir": str(self.tmp),
        }
        payload.update(overrides)
        path = self.tmp / "request.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def run_cli(self, *argv) -> tuple[int, dict]:
        out = self.tmp / "result.json"
        code = main([*argv, str(out)] if argv[0] != "probe" else list(argv))
        payload = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        return code, payload


class TestRun(_Harness):
    def test_a_servable_request_plans_every_shape(self):
        request = self.write_request()
        code, result = self.run_cli("run", str(request), "--plan-only")
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(result["status"], "planned")
        self.assertEqual(len(result["plans"]), 1)
        self.assertTrue(result["plans"][0]["servable"])

    def test_planning_alone_claims_neither_speed_nor_correctness(self):
        request = self.write_request()
        _, result = self.run_cli("run", str(request), "--plan-only")
        self.assertIsNone(result["micro_speedup"])
        self.assertIsNone(result["correctness_passed"])
        self.assertNotEqual(result["status"], "ok")

    def test_an_unservable_request_is_declined_with_a_file(self):
        entry = copy.deepcopy(_ENTRY)
        entry["attention_request"].update({"hdim_q": 48, "hdim_v": 48})
        entry["problem"]["head_size"] = 48
        request = self.write_request(requests=[entry])

        code, result = self.run_cli("run", str(request), "--plan-only")
        self.assertEqual(code, EXIT_DECLINED)
        self.assertEqual(result["status"], "declined")
        self.assertFalse(result["plans"][0]["servable"])
        self.assertTrue(result["plans"][0]["reason"])

    def test_a_malformed_request_still_produces_a_result_file(self):
        request = self.write_request(schema="something/else")
        code, result = self.run_cli("run", str(request), "--plan-only")
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["reasons"])

    def test_an_unreadable_request_is_an_error_not_a_traceback(self):
        code, result = self.run_cli("run", str(self.tmp / "absent.json"), "--plan-only")
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(result["status"], "error")

    def test_advisory_requests_are_flagged_in_the_result(self):
        request = self.write_request(advisory=True)
        _, result = self.run_cli("run", str(request), "--plan-only")
        self.assertTrue(any("advisory" in r for r in result["reasons"]))
        self.assertIn("advisory", result["report"])

    def test_the_report_names_the_selected_kernel(self):
        request = self.write_request()
        _, result = self.run_cli("run", str(request), "--plan-only")
        self.assertIn("attention_unified_3d", result["report"])

    def test_mixed_requests_report_each_shape_separately(self):
        bad = copy.deepcopy(_ENTRY)
        bad["attention_request"].update({"hdim_q": 48, "hdim_v": 48})
        bad["problem"]["head_size"] = 48
        request = self.write_request(requests=[copy.deepcopy(_ENTRY), bad])

        code, result = self.run_cli("run", str(request), "--plan-only")
        # One servable shape is enough to proceed; the other is reported.
        self.assertEqual(code, EXIT_OK)
        self.assertEqual([p["servable"] for p in result["plans"]], [True, False])
        self.assertIn("Declined shapes", result["report"])


class TestPlan(_Harness):
    def test_plan_writes_the_same_plans_as_run(self):
        request = self.write_request()
        code, result = self.run_cli("plan", str(request))
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["plans"][0]["candidate"], "attention_unified_3d")


class TestProbe(_Harness):
    def test_probe_reports_coverage_for_an_arch_without_a_request(self):
        out = self.tmp / "probe.json"
        code = main(["probe", "--arch", "gfx950", "--output", str(out)])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("candidates", payload["coverage"])

        # Request-free arch coverage needs declared capabilities. Where the
        # dispatcher has none, the probe must say so rather than answer.
        if payload["candidates_for_arch"] is None:
            self.assertTrue(payload["candidates_for_arch_reason"])
        else:
            self.assertIn("attention_unified_2d", payload["candidates_for_arch"])

    def test_probe_always_names_every_registered_candidate(self):
        out = self.tmp / "probe.json"
        main(["probe", "--output", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        names = {c["name"] for c in payload["coverage"]["candidates"]}
        self.assertIn("attention_unified_2d", names)
        self.assertIn("attention_unified_3d", names)

    def test_probe_states_whether_the_measured_lanes_can_run(self):
        out = self.tmp / "probe.json"
        main(["probe", "--output", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("measured_lanes_available", payload)
        if not payload["measured_lanes_available"]:
            self.assertTrue(payload["measured_lanes_reason"])


if __name__ == "__main__":
    unittest.main()
