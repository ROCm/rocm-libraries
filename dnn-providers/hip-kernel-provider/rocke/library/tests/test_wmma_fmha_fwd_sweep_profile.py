# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import subprocess
import unittest
from unittest import mock

from builders.gfx1151.attention import wmma_fmha_fwd_sweep_profile as subject


def _args(**overrides):
    values = {
        "arch": "gfx1201",
        "seqlen_q": 64,
        "seqlen_k": 128,
        "head_size": 64,
        "heads": 4,
        "kv_heads": 2,
        "batch": 1,
        "causal": False,
        "warmup": 10,
        "iters": 100,
        "profile_repeats": 1,
        "cache": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestWmmaFmhaFwdSweepProfile(unittest.TestCase):
    def test_parse_perfjson_uses_structured_line(self):
        got = subject._parse_perfjson('noise\nPerfJSON: {"ms": 0.25, "tflops": 2.0}\n')
        self.assertEqual(got["ms"], 0.25)

    def test_parse_perfjson_rejects_missing_timing(self):
        with self.assertRaisesRegex(RuntimeError, "valid PerfJSON timing"):
            subject._parse_perfjson('PerfJSON: {"tflops": 2.0}\n')

    def test_parse_perfjson_rejects_malformed_json(self):
        with self.assertRaises(json.JSONDecodeError):
            subject._parse_perfjson("PerfJSON: {not-json}\n")

    def test_variant_args_rejects_unknown_variant(self):
        with self.assertRaisesRegex(ValueError, "unknown variant"):
            subject._variant_args("unknown")

    def test_benchmark_command_selects_vlds(self):
        cmd = subject._benchmark_command(_args(causal=True), "vlds")
        self.assertIn("--v-lds-stage", cmd)
        self.assertIn("--causal", cmd)

    def test_profile_command_preserves_shape_and_warmup(self):
        cmd = subject._profile_command(_args(cache="/tmp/cache"), "vgather")
        warmup = cmd[cmd.index("--warmup") + 1]
        shape = json.loads(cmd[cmd.index("--shape") + 1])
        self.assertEqual(warmup, "11")
        self.assertEqual(shape["variant"], "vgather")
        self.assertEqual(shape["kv_heads"], 2)
        self.assertEqual(cmd[cmd.index("--cache") + 1], "/tmp/cache")

    def test_profile_exit_one_accepted_only_for_regression(self):
        payload = {
            "record": {"wall": {"ms_median": 1.0}},
            "selfcheck": {"verdict": "regressed"},
        }
        proc = subprocess.CompletedProcess(
            args=["profile"], returncode=1, stdout=json.dumps(payload), stderr=""
        )
        with mock.patch.object(subject.subprocess, "run", return_value=proc):
            self.assertEqual(subject._run_profile(["profile"]), payload)

    def test_profile_exit_one_genuine_failure_rejected(self):
        proc = subprocess.CompletedProcess(
            args=["profile"], returncode=1, stdout="", stderr="launch failed"
        )
        with mock.patch.object(subject.subprocess, "run", return_value=proc):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                subject._run_profile(["profile"])

    def test_profile_exit_one_non_regression_json_rejected(self):
        payload = {
            "record": {"wall": {"ms_median": 1.0}},
            "selfcheck": {"verdict": "within_noise"},
        }
        proc = subprocess.CompletedProcess(
            args=["profile"], returncode=1, stdout=json.dumps(payload), stderr=""
        )
        with mock.patch.object(subject.subprocess, "run", return_value=proc):
            with self.assertRaisesRegex(RuntimeError, "profile command failed"):
                subject._run_profile(["profile"])


if __name__ == "__main__":
    unittest.main()
