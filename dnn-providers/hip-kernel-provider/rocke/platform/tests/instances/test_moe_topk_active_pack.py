# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Structure tests for correction-biased top-k active packing."""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.instances.common.moe_topk_active_pack import (
    MoeTopkActivePackSpec,
    build_moe_topk_active_pack,
    is_valid_spec,
    moe_topk_active_pack_grid,
    moe_topk_active_pack_signature,
)


class TestMoeTopkActivePack(unittest.TestCase):
    def test_target_decode_specs_are_valid(self) -> None:
        for tokens in (1, 8):
            spec = MoeTopkActivePackSpec(tokens=tokens, experts=896, topk=16)
            self.assertEqual(is_valid_spec(spec), (True, "ok"))
            self.assertEqual(spec.total_pairs, tokens * 16)
            self.assertEqual(spec.max_blocks, tokens * 16)
            self.assertEqual(spec.max_padded_pairs, tokens * 16 * 16)

    def test_rejects_unsupported_grouping_and_oversized_batch(self) -> None:
        grouped = MoeTopkActivePackSpec(
            tokens=8,
            experts=896,
            topk=16,
            num_expert_groups=8,
            topk_groups=4,
        )
        ok, why = is_valid_spec(grouped)
        self.assertFalse(ok)
        self.assertIn("one-group", why)

        oversized = MoeTopkActivePackSpec(tokens=65, experts=896, topk=16)
        ok, why = is_valid_spec(oversized)
        self.assertFalse(ok)
        self.assertIn("tokens*topk", why)

    def test_signature_and_static_launch_contract(self) -> None:
        spec = MoeTopkActivePackSpec(tokens=8, experts=896, topk=16)
        self.assertEqual(moe_topk_active_pack_grid(spec), (1, 1, 1))
        self.assertEqual(
            [entry["name"] for entry in moe_topk_active_pack_signature(spec)],
            [
                "Logits",
                "CorrectionBias",
                "SortedTokenIds",
                "SortedTopkIds",
                "SortedWeights",
                "BlockExpertIds",
                "Counts",
                "BlockOffsets",
                "NumBlocks",
                "tokens",
                "experts",
                "routed_scale",
            ],
        )

    def test_build_verifies_and_contains_fused_stages(self) -> None:
        spec = MoeTopkActivePackSpec(tokens=8, experts=896, topk=16)
        kernel = build_moe_topk_active_pack(spec)
        self.assertEqual(verify(kernel), [])
        llvm = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("define amdgpu_kernel", llvm)
        self.assertIn("atomicrmw add ptr addrspace(3)", llvm)
        self.assertNotIn("atomicrmw add ptr addrspace(1)", llvm)
        self.assertNotIn("%token = phi i32", llvm)
        self.assertNotIn("%topk_slot = phi i32", llvm)
        self.assertIn("@llvm.amdgcn.ds.bpermute", llvm)

    def test_precision_and_tie_break_primitives_are_present(self) -> None:
        kernel = build_moe_topk_active_pack(
            MoeTopkActivePackSpec(tokens=1, experts=896, topk=16)
        )
        llvm = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("@llvm.exp2.f32", llvm)
        self.assertIn("fcmp ogt", llvm)
        self.assertIn("fcmp oeq", llvm)
        self.assertIn("icmp slt", llvm)


if __name__ == "__main__":
    unittest.main()
