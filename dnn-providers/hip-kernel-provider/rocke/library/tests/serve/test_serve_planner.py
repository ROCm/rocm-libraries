# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Planning tests: the decision, and the two properties that make it useful.

The planner is thin, so testing that it calls the dispatcher would prove
nothing. What is worth pinning is why the caller sends what it sends:

- Path selection turns on ``num_seqs``. The caller goes to real trouble to
  capture that field in the serving process, because tensor geometry alone does
  not determine it. If the same head geometry and the same query-row count
  routed to one path regardless of how the batch decomposes, that capture work
  would be unnecessary -- so this asserts the sensitivity directly.
- Planning is reproducible for an arch that is not attached, which is what lets
  a caller find out that a shape is unservable without paying for a GPU node.

All of it is CPU-only; no test here needs a device.
"""

from __future__ import annotations

import copy
import unittest

from serve.planner import build_attention_request, build_problem, plan_all, plan_entry
from serve.protocol import ShapeEntry

_BASE = {
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
}


def _entry(**problem_overrides) -> ShapeEntry:
    raw = copy.deepcopy(_BASE)
    raw["problem"].update(problem_overrides)
    if "num_seqs" in problem_overrides:
        raw["attention_request"]["batch"] = problem_overrides["num_seqs"]
    if "max_seqlen_q" in problem_overrides:
        raw["attention_request"]["seqlen_q"] = problem_overrides["max_seqlen_q"]
    if "head_size" in problem_overrides:
        raw["attention_request"]["hdim_q"] = problem_overrides["head_size"]
        raw["attention_request"]["hdim_v"] = problem_overrides["head_size"]
    return ShapeEntry.from_dict(raw, index=0)


class TestPathSelection(unittest.TestCase):
    def test_batch_decomposition_decides_the_path(self):
        """Few sequences take split-KV; many take the 2D tiled path.

        This is the property the whole shape-capture path exists to serve: the
        head geometry is identical across these cases and only ``num_seqs``
        moves, yet the selected kernel changes.
        """
        paths = {}
        for num_seqs in (1, 8, 64, 256, 1087):
            plan = plan_entry(
                _entry(num_seqs=num_seqs, total_q=num_seqs), arch="gfx950"
            )
            self.assertTrue(plan["servable"], plan.get("reason"))
            paths[num_seqs] = plan["path"]

        self.assertEqual(paths[1], "3d")
        self.assertEqual(paths[1087], "2d")
        self.assertGreater(
            len(set(paths.values())),
            1,
            f"num_seqs no longer changes the path: {paths}",
        )

    def test_the_selected_candidate_matches_the_selected_path(self):
        for num_seqs, expected in (
            (1, "attention_unified_3d"),
            (1087, "attention_unified_2d"),
        ):
            with self.subTest(num_seqs=num_seqs):
                plan = plan_entry(
                    _entry(num_seqs=num_seqs, total_q=num_seqs), arch="gfx950"
                )
                self.assertEqual(plan["candidate"], expected)


class TestTheTwoViews(unittest.TestCase):
    def test_dispatch_pads_while_the_runtime_problem_keeps_what_was_observed(self):
        entry = _entry(num_seqs=4, max_seqlen_q=2048, total_q=7000)
        request = build_attention_request(entry, arch="gfx950")
        problem = build_problem(entry, num_sms=256)

        self.assertEqual(request.batch * request.seqlen_q, 8192)
        self.assertEqual(problem.total_q, 7000)

    def test_the_plan_reports_the_runtime_problem(self):
        plan = plan_entry(
            _entry(num_seqs=4, max_seqlen_q=2048, total_q=7000), arch="gfx950"
        )
        self.assertEqual(plan["problem"]["total_q"], 7000)


class TestReproducibilityAndRobustness(unittest.TestCase):
    def test_planning_works_for_an_arch_that_is_not_attached(self):
        # No device is consulted, so every declared target plans from any host.
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                plan = plan_entry(_entry(), arch=arch)
                self.assertTrue(plan["servable"], plan.get("reason"))
                self.assertEqual(plan["arch"], arch)
                self.assertEqual(plan["kernel_id"]["arch"], arch)

    def test_the_envelope_arch_overrides_the_entry(self):
        # The entry says gfx950; the envelope is what the caller resolved.
        plan = plan_entry(_entry(), arch="gfx942")
        self.assertEqual(plan["arch"], "gfx942")

    def test_an_unservable_shape_is_data_not_an_exception(self):
        plan = plan_entry(_entry(head_size=48), arch="gfx950")
        self.assertFalse(plan["servable"])
        self.assertIn("48", plan["reason"])

    def test_unknown_request_fields_are_ignored(self):
        raw = copy.deepcopy(_BASE)
        raw["attention_request"]["some_future_field"] = "x"
        plan = plan_entry(ShapeEntry.from_dict(raw, index=0), arch="gfx950")
        self.assertTrue(plan["servable"], plan.get("reason"))

    def test_plan_all_preserves_order_and_length(self):
        entries = (_entry(num_seqs=1, total_q=1), _entry(head_size=48), _entry())
        plans = plan_all(entries, arch="gfx950")
        self.assertEqual(len(plans), 3)
        self.assertEqual([p["servable"] for p in plans], [True, False, True])


if __name__ == "__main__":
    unittest.main()
