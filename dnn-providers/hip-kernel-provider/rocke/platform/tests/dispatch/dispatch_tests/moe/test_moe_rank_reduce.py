# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dispatch tests for local rank-staged MoE reduction epilogues."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.moe_rank_reduce import (
    MoeRankReduceRequest,
    dispatch_moe_rank_reduce,
    moe_rank_reduce_candidates,
)
from rocke.instances.common.moe_rank_reduce import (
    MoeRankReduceRMSNormSpec,
    MoeRankReduceScatterSpec,
)


def _request(operation: str, **overrides) -> MoeRankReduceRequest:
    values = {
        "rows": 8,
        "width": 3584 if operation == "rmsnorm" else 7168,
        "world_size": 8,
        "rank": 3,
        "arch": "gfx950",
        "operation": operation,
        "dtype": "bf16",
    }
    values.update(overrides)
    return MoeRankReduceRequest(**values)


class TestMoeRankReduceDispatch(unittest.TestCase):
    def test_rmsnorm_selects_matching_spec_and_geometry(self) -> None:
        result = dispatch_moe_rank_reduce(_request("rmsnorm"))
        self.assertIsInstance(result.spec, MoeRankReduceRMSNormSpec)
        self.assertEqual((result.spec.block_size, result.spec.vec), (256, 2))
        self.assertEqual(result.grid, (8, 1, 1))
        self.assertEqual(result.block, (256, 1, 1))

    def test_scatter_selects_shard_geometry(self) -> None:
        result = dispatch_moe_rank_reduce(_request("scatter"))
        self.assertIsInstance(result.spec, MoeRankReduceScatterSpec)
        self.assertEqual(result.spec.shard_width, 896)
        self.assertEqual((result.spec.block_size, result.spec.vec), (64, 2))
        self.assertEqual(result.block, (64, 1, 1))

    def test_fp16_alias_is_normalized(self) -> None:
        result = dispatch_moe_rank_reduce(_request("rmsnorm", dtype="fp16"))
        self.assertEqual(result.spec.dtype, "f16")

    def test_precision_mode_changes_identity(self) -> None:
        narrow = dispatch_moe_rank_reduce(_request("rmsnorm"))
        wide = dispatch_moe_rank_reduce(_request("rmsnorm", fp32_internal=True))
        self.assertNotEqual(narrow.kernel_id.spec_hash, wide.kernel_id.spec_hash)

    def test_candidates_are_operation_exclusive(self) -> None:
        request = _request("scatter")
        supported = [
            candidate.spec_id
            for candidate in moe_rank_reduce_candidates()
            if candidate.admits(request)[0]
        ]
        self.assertEqual(supported, ["scatter"])

    def test_rejects_invalid_rank_partition_and_arch(self) -> None:
        with self.assertRaises(ValueError):
            dispatch_moe_rank_reduce(_request("scatter", rank=8))
        with self.assertRaises(ValueError):
            dispatch_moe_rank_reduce(_request("scatter", width=7169))
        with self.assertRaises(ValueError):
            dispatch_moe_rank_reduce(_request("scatter", arch="gfx942"))

    def test_explanation_states_transport_boundary(self) -> None:
        result = dispatch_moe_rank_reduce(_request("rmsnorm"))
        self.assertTrue(
            any("transport is caller-owned" in line for line in result.explanation)
        )


if __name__ == "__main__":
    unittest.main()
