# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dispatch tests for fused top-k active packing."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.moe_routing import (
    MoeRoutingRequest,
    dispatch_moe_routing,
    moe_routing_candidates,
)


def _request(**overrides) -> MoeRoutingRequest:
    values = {
        "tokens": 8,
        "experts": 896,
        "topk": 16,
        "arch": "gfx950",
    }
    values.update(overrides)
    return MoeRoutingRequest(**values)


class TestMoeRoutingDispatch(unittest.TestCase):
    def test_selects_active_pack_target(self) -> None:
        result = dispatch_moe_routing(_request())
        self.assertEqual(result.candidate.spec_id, "topk_active_pack")
        self.assertEqual(result.grid, (1, 1, 1))
        self.assertEqual(result.block, (1024, 1, 1))
        self.assertEqual(result.spec.max_blocks, 128)
        self.assertEqual(len(moe_routing_candidates()), 1)

    def test_identity_tracks_shape_and_normalization(self) -> None:
        base = dispatch_moe_routing(_request())
        different_tokens = dispatch_moe_routing(_request(tokens=1))
        no_norm = dispatch_moe_routing(_request(renormalize=False))
        self.assertNotEqual(
            base.kernel_id.spec_hash, different_tokens.kernel_id.spec_hash
        )
        self.assertNotEqual(base.kernel_id.spec_hash, no_norm.kernel_id.spec_hash)

    def test_rejects_non_f32_multi_group_and_wrong_arch(self) -> None:
        for request in (
            _request(dtype="bf16"),
            _request(num_expert_groups=8, topk_groups=4),
            _request(arch="gfx942"),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                dispatch_moe_routing(request)

    def test_candidate_build_is_reachable(self) -> None:
        result = dispatch_moe_routing(_request(tokens=1))
        kernel = result.candidate.build(result.spec, "gfx950")
        self.assertEqual(kernel.name, result.spec.kernel_name())


if __name__ == "__main__":
    unittest.main()
