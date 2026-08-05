# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wire-format tests for ``rocke-serve``.

The format is a boundary with a caller we do not build, so these tests are
mostly about what happens to input we did not expect: an older schema, a field
that has since been added upstream, a request whose two views of the same shape
disagree. Each of those has a defined outcome, and none of them may be a
traceback escaping into the caller's log.
"""

from __future__ import annotations

import unittest

from serve.protocol import (
    PROBLEM_FIELDS,
    REQUEST_SCHEMA,
    ProtocolError,
    ServeRequest,
    ShapeEntry,
    make_result,
)


def _entry(**overrides):
    problem = {
        "total_q": 8,
        "num_seqs": 8,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_seqlen_q": 1,
        "max_seqlen_k": 8192,
        "dtype": "bf16",
    }
    problem.update(overrides.pop("problem", {}))
    entry = {
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
            "kv_block_size": 16,
            "dtype": "bf16",
        },
        "problem": problem,
    }
    entry.update(overrides)
    return entry


def _request(**overrides):
    payload = {
        "schema": REQUEST_SCHEMA,
        "op": "attention",
        "arch": "gfx950",
        "requests": [_entry()],
    }
    payload.update(overrides)
    return payload


class TestRequestParsing(unittest.TestCase):
    def test_parses_a_well_formed_request(self):
        req = ServeRequest.from_dict(_request())
        self.assertEqual(req.arch, "gfx950")
        self.assertEqual(len(req.entries), 1)

    def test_rejects_an_unknown_schema(self):
        with self.assertRaises(ProtocolError) as ctx:
            ServeRequest.from_dict(_request(schema="hyperloom.rocke.serve_request/v2"))
        self.assertIn("unsupported schema", str(ctx.exception))

    def test_rejects_a_non_attention_op(self):
        with self.assertRaises(ProtocolError):
            ServeRequest.from_dict(_request(op="gemm"))

    def test_rejects_an_arch_that_is_not_a_gfx_target(self):
        for arch in ("", "MI355X", "cdna"):
            with self.subTest(arch=arch), self.assertRaises(ProtocolError):
                ServeRequest.from_dict(_request(arch=arch))

    def test_rejects_an_empty_request_list(self):
        for requests in ([], None, {}):
            with self.subTest(requests=requests), self.assertRaises(ProtocolError):
                ServeRequest.from_dict(_request(requests=requests))

    def test_arch_is_normalized_to_lower_case(self):
        self.assertEqual(ServeRequest.from_dict(_request(arch="GFX950")).arch, "gfx950")


class TestShapeEntry(unittest.TestCase):
    def test_missing_problem_fields_are_named_in_the_error(self):
        entry = _entry()
        del entry["problem"]["block_size"]
        with self.assertRaises(ProtocolError) as ctx:
            ShapeEntry.from_dict(entry, index=3)
        message = str(ctx.exception)
        self.assertIn("block_size", message)
        self.assertIn("requests[3]", message)

    def test_unknown_problem_fields_are_dropped(self):
        # The caller may run ahead of us. An unrecognized key must not reach
        # UnifiedAttentionProblem(**problem), where it would be a TypeError at
        # the point of use rather than a parse-time decision.
        entry = _entry(problem={"invented_upstream_knob": 7})
        parsed = ShapeEntry.from_dict(entry, index=0)
        self.assertNotIn("invented_upstream_knob", parsed.problem)
        self.assertTrue(set(parsed.problem).issubset(set(PROBLEM_FIELDS)))

    def test_both_views_of_a_ragged_shape_are_preserved(self):
        # The dispatch view is padded (4 x 2048) and the runtime view is what
        # was observed (7000). Collapsing them would either over-report the
        # measured work or under-declare what the kernel must cover.
        entry = _entry(
            problem={"num_seqs": 4, "max_seqlen_q": 2048, "total_q": 7000},
            ragged=True,
            observed_total_q=7000,
            request_total_q=8192,
        )
        entry["attention_request"].update({"batch": 4, "seqlen_q": 2048})
        parsed = ShapeEntry.from_dict(entry, index=0)
        self.assertTrue(parsed.ragged)
        self.assertEqual(parsed.problem["total_q"], 7000)
        self.assertEqual(
            parsed.attention_request["batch"] * parsed.attention_request["seqlen_q"],
            8192,
        )

    def test_entry_must_be_an_object_with_both_views(self):
        for bad in ({}, {"problem": {}}, {"attention_request": {}}, []):
            with self.subTest(bad=bad), self.assertRaises(ProtocolError):
                ShapeEntry.from_dict(bad, index=0)


class TestResult(unittest.TestCase):
    def test_unmeasured_lanes_stay_none(self):
        # The caller promotes a kernel on these two fields. An unmeasured lane
        # must not arrive looking like a measured one that broke even or passed.
        result = make_result(status="planned")
        self.assertIsNone(result["micro_speedup"])
        self.assertIsNone(result["correctness_passed"])

    def test_carries_the_result_schema_and_status(self):
        result = make_result(status="ok", micro_speedup=1.5, correctness_passed=True)
        self.assertEqual(result["schema"], "hyperloom.rocke.serve_result/v1")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["micro_speedup"], 1.5)
        self.assertIs(result["correctness_passed"], True)


if __name__ == "__main__":
    unittest.main()
