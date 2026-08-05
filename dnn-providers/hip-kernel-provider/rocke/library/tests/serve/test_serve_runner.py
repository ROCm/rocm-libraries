# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measured-lane tests.

Two halves with different requirements. The availability probe must work with no
torch at all -- it is what the CLI calls to decide whether to try -- so it is
tested unconditionally. Everything else needs torch and is skipped without it.

The tensor-construction tests deliberately stop short of launching. They cover
the part that is easy to get quietly wrong: the paged-cache geometry the wire
format does not carry and this module has to derive. A block table that indexes
past the cache, or a cache too small to hold every sequence, produces a kernel
launch that reads out of bounds rather than an obvious failure.
"""

from __future__ import annotations

import math
import unittest

import pytest

from serve.runner import torch_gpu_available

_PROBLEM = {
    "total_q": 8,
    "num_seqs": 8,
    "num_query_heads": 64,
    "num_kv_heads": 8,
    "head_size": 128,
    "block_size": 16,
    "max_seqlen_q": 1,
    "max_seqlen_k": 8192,
    "dtype": "bf16",
    "sliding_window": 0,
    "softcap": 0.0,
    "use_sinks": False,
    "use_alibi": False,
    "use_fp8": False,
}


class TestAvailabilityProbe(unittest.TestCase):
    """Runs with or without torch; the point is that it never raises."""

    def test_probe_reports_a_reason_whenever_it_says_no(self):
        available, reason = torch_gpu_available()
        self.assertIsInstance(available, bool)
        if not available:
            self.assertTrue(reason, "an unavailable lane must say why")
        else:
            self.assertEqual(reason, "")


class TestShapeDerivation(unittest.TestCase):
    def setUp(self):
        pytest.importorskip("torch")
        from serve.runner import shape_from_problem

        self.shape_from_problem = shape_from_problem

    def test_paged_cache_is_large_enough_for_every_sequence(self):
        shape = self.shape_from_problem(dict(_PROBLEM), softmax_scale=0.088)
        expected_per_seq = math.ceil(_PROBLEM["max_seqlen_k"] / _PROBLEM["block_size"])
        self.assertEqual(shape.max_blocks_per_seq, expected_per_seq)
        # Every sequence's block table is filled with indices < num_blocks, so
        # the cache must hold all of them to keep those indices meaningful.
        self.assertGreaterEqual(
            shape.num_blocks, shape.max_blocks_per_seq * shape.num_seqs
        )

    def test_an_absent_softmax_scale_falls_back_to_the_standard_one(self):
        shape = self.shape_from_problem(dict(_PROBLEM), softmax_scale=0.0)
        self.assertAlmostEqual(shape.softmax_scale, 1.0 / math.sqrt(128))

    def test_sliding_window_round_trips_through_the_harness_convention(self):
        # The harness stores window_size[0] == sliding_window - 1.
        problem = {**_PROBLEM, "sliding_window": 1024}
        shape = self.shape_from_problem(problem, softmax_scale=0.088)
        self.assertEqual(shape.window_size[0] + 1, 1024)

        no_window = self.shape_from_problem(dict(_PROBLEM), softmax_scale=0.088)
        self.assertEqual(no_window.window_size, (-1, -1))

    def test_an_fp8_kv_cache_narrows_only_k_and_v(self):
        shape = self.shape_from_problem(
            {**_PROBLEM, "use_fp8": True}, softmax_scale=0.088
        )
        self.assertEqual(shape.q_dtype, "torch.bfloat16")
        self.assertEqual(shape.out_dtype, "torch.bfloat16")
        self.assertIn("float8", shape.k_dtype)
        self.assertIn("float8", shape.v_dtype)

    def test_an_unsupported_dtype_is_rejected(self):
        with self.assertRaises(ValueError):
            self.shape_from_problem({**_PROBLEM, "dtype": "fp32"}, softmax_scale=0.088)

    def test_the_observed_total_q_is_carried_through(self):
        shape = self.shape_from_problem(
            {**_PROBLEM, "num_seqs": 4, "max_seqlen_q": 2048, "total_q": 7000},
            softmax_scale=0.088,
        )
        self.assertEqual(shape.total_q, 7000)


if __name__ == "__main__":
    unittest.main()
