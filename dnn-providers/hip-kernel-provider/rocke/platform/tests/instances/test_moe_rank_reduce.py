# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Structure and lowering tests for local rank-staged MoE reductions."""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.instances.common.moe_rank_reduce import (
    MoeRankReduceRMSNormSpec,
    MoeRankReduceScatterSpec,
    build_moe_rank_reduce_rmsnorm,
    build_moe_rank_reduce_scatter,
    is_valid_rmsnorm_spec,
    is_valid_scatter_spec,
    moe_rank_reduce_rmsnorm_grid,
    moe_rank_reduce_rmsnorm_signature,
    moe_rank_reduce_scatter_grid,
    moe_rank_reduce_scatter_signature,
)


class TestMoeRankReduce(unittest.TestCase):
    def test_target_shapes_are_valid(self) -> None:
        rms = MoeRankReduceRMSNormSpec(width=3584, world_size=8)
        scatter = MoeRankReduceScatterSpec(width=7168, world_size=8)
        self.assertEqual(is_valid_rmsnorm_spec(rms), (True, "ok"))
        self.assertEqual(is_valid_scatter_spec(scatter), (True, "ok"))
        self.assertEqual(scatter.shard_width, 896)

    def test_validation_rejects_bad_partition_and_geometry(self) -> None:
        bad_partition = MoeRankReduceScatterSpec(width=7169, world_size=8)
        ok, why = is_valid_scatter_spec(bad_partition)
        self.assertFalse(ok)
        self.assertIn("divisible by world_size", why)

        bad_rms = MoeRankReduceRMSNormSpec(
            width=3584, world_size=8, block_size=256, vec=4
        )
        ok, why = is_valid_rmsnorm_spec(bad_rms)
        self.assertFalse(ok)
        self.assertIn("block_size*vec", why)

    def test_names_encode_compile_time_contract(self) -> None:
        narrow = MoeRankReduceRMSNormSpec(width=3584, world_size=8)
        wide = MoeRankReduceRMSNormSpec(width=3584, world_size=8, fp32_internal=True)
        self.assertIn("N3584_R8", narrow.kernel_name())
        self.assertNotEqual(narrow.kernel_name(), wide.kernel_name())
        self.assertTrue(wide.kernel_name().endswith("_f32"))

    def test_grid_and_signatures_match_abis(self) -> None:
        rms = MoeRankReduceRMSNormSpec(width=3584, world_size=8)
        scatter = MoeRankReduceScatterSpec(width=7168, world_size=8)
        self.assertEqual(moe_rank_reduce_rmsnorm_grid(7, rms), (7, 1, 1))
        self.assertEqual(moe_rank_reduce_scatter_grid(7, scatter), (7, 1, 1))
        self.assertEqual(
            [entry["name"] for entry in moe_rank_reduce_rmsnorm_signature(rms)],
            ["Partials", "Gamma", "Y", "rows", "width", "eps"],
        )
        self.assertEqual(
            [entry["name"] for entry in moe_rank_reduce_scatter_signature(scatter)],
            ["Partials", "Y", "rows", "width", "rank"],
        )

    def test_builds_verify_and_lower(self) -> None:
        rms = build_moe_rank_reduce_rmsnorm(
            MoeRankReduceRMSNormSpec(width=3584, world_size=8)
        )
        scatter = build_moe_rank_reduce_scatter(
            MoeRankReduceScatterSpec(width=7168, world_size=8)
        )
        self.assertEqual(verify(rms), [])
        self.assertEqual(verify(scatter), [])

        rms_ll = lower_kernel_to_llvm(rms, arch="gfx950")
        scatter_ll = lower_kernel_to_llvm(scatter, arch="gfx950")
        self.assertIn("define amdgpu_kernel", rms_ll)
        self.assertIn("llvm.amdgcn.rsq.f32", rms_ll)
        self.assertIn("<2 x bfloat>", rms_ll)
        self.assertIn("define amdgpu_kernel", scatter_ll)
        self.assertNotIn("atomicrmw", scatter_ll)


if __name__ == "__main__":
    unittest.main()
