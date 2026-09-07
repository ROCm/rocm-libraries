# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Structure tests for compact routed gather plus FP8 quantization."""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.instances.common.moe_compact_gather_quant import (
    MoeCompactGatherQuantSpec,
    build_moe_compact_gather_quant,
    is_valid_spec,
    moe_compact_gather_quant_grid,
    moe_compact_gather_quant_signature,
)


class TestMoeCompactGatherQuant(unittest.TestCase):
    def test_target_shape_geometry(self) -> None:
        spec = MoeCompactGatherQuantSpec(tokens=8, hidden=3584, max_blocks=128)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        self.assertEqual(spec.hidden_blocks, 28)
        self.assertEqual(spec.output_rows, 2048)
        self.assertEqual(spec.passes_per_thread, 2)
        self.assertEqual(moe_compact_gather_quant_grid(spec), (28, 128, 1))

    def test_signature_matches_pipeline_buffers(self) -> None:
        spec = MoeCompactGatherQuantSpec(tokens=1, hidden=3584, max_blocks=16)
        self.assertEqual(
            [entry["name"] for entry in moe_compact_gather_quant_signature(spec)],
            [
                "X",
                "SortedTokenIds",
                "BlockExpertIds",
                "A",
                "AScale",
                "tokens",
                "hidden",
            ],
        )

    def test_build_verifies_and_lowers_packed_conversion(self) -> None:
        spec = MoeCompactGatherQuantSpec(tokens=8, hidden=3584, max_blocks=128)
        kernel = build_moe_compact_gather_quant(spec)
        self.assertEqual(verify(kernel), [])
        llvm = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("define amdgpu_kernel", llvm)
        self.assertIn("cvt.pk.fp8.f32", llvm)
        self.assertIn("load <4 x bfloat>", llvm)
        self.assertIn("store <4 x i8>", llvm)

    def test_rejects_non_aligned_hidden_and_vec(self) -> None:
        for spec in (
            MoeCompactGatherQuantSpec(tokens=8, hidden=3585, max_blocks=128),
            MoeCompactGatherQuantSpec(tokens=8, hidden=3584, max_blocks=128, vec=2),
        ):
            with self.subTest(spec=spec):
                self.assertFalse(is_valid_spec(spec)[0])


if __name__ == "__main__":
    unittest.main()
