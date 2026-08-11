# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convolution kernel tests, excised from platform/tests/test_rocke.py.

Only the tests whose *subject* is a convolution kernel moved here: the
direct-conv transform-descriptor suite and the three implicit-GEMM /
direct-conv builder smoke tests. Tests that merely use a conv spec as a
vehicle for platform behaviour (CDNA primitive emission, target
intrinsics, pack-args kernarg ABI, CK Tile / HIP lowering coverage) stay
in the platform suite and reach the builders through ``kernels.*`` --
platform *tests* may import the library, only the SDK package may not.
"""

from __future__ import annotations

import unittest

from rocke import lower_kernel_to_llvm

from kernels import (
    ConvProblem,
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    ImplicitGemmConvSpec,
    build_direct_conv_4c,
    build_direct_conv_16c,
    build_implicit_gemm_conv,
)


class TestConvDirectGroupedTransforms(unittest.TestCase):
    """The transform-descriptor migration must keep the kernels building."""

    def test_16c_kernel_lowers_to_llvm(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from kernels import (
            DirectConv16cSpec,
            DirectConvProblem,
            build_direct_conv_16c,
        )

        spec = DirectConv16cSpec(
            problem=DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16)
        )
        kernel = build_direct_conv_16c(spec)
        ll = lower_kernel_to_llvm(kernel)
        # Smoke check: the generated LLVM IR mentions amdgpu and the
        # kernel name (proves the body got emitted).
        self.assertIn("amdgpu", ll)
        self.assertIn(kernel.name, ll)

    def test_4c_kernel_lowers_to_llvm(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from kernels import (
            DirectConv4cSpec,
            DirectConvProblem,
            build_direct_conv_4c,
        )

        spec = DirectConv4cSpec(
            problem=DirectConvProblem(N=1, H=8, W=8, groups=16, cpg=4, kpg=4)
        )
        kernel = build_direct_conv_4c(spec)
        ll = lower_kernel_to_llvm(kernel)
        self.assertIn("amdgpu", ll)
        self.assertIn(kernel.name, ll)


class TestConvInstanceBuilders(unittest.TestCase):
    """Implicit-GEMM and grouped direct-conv builder smoke tests."""

    def test_implicit_gemm_conv_builds(self):
        prob = ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
        )
        kernel = build_implicit_gemm_conv(spec)
        ll = lower_kernel_to_llvm(kernel)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", ll)
        # The buffer rsrc DW3 flag-word must be 0x00027000, not 0 — the
        # critical correctness fix from the bake-off debugging session.
        self.assertIn("159744", ll)  # 0x27000 = 159744

    def test_direct_conv_16c_builds(self):
        prob = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        spec = DirectConv16cSpec(problem=prob, block_groups=4, fold_k32=True)
        kernel = build_direct_conv_16c(spec)
        ll = lower_kernel_to_llvm(kernel)
        # K32-folded hot loop emits ONLY the wide 16x16x32 MFMA: S=0/1 fold
        # into one wide atom and the S=2 residual is promoted to a SECOND
        # wide atom (zero-padded upper K) so both atoms on the accumulator
        # are the same width. Mixing a 16x16x16 residual into the same
        # accumulator triggered a cross-width MFMA accumulator hazard that
        # both comgr and hipcc miscompiled at the H-edges; the all-wide fold
        # is bit-correct.
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll)
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll)
        # The unfolded (gfx942-capable) path still uses only 16x16x16.
        spec_nf = DirectConv16cSpec(problem=prob, block_groups=4, fold_k32=False)
        ll_nf = lower_kernel_to_llvm(build_direct_conv_16c(spec_nf))
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll_nf)
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll_nf)

    def test_direct_conv_4c_builds(self):
        prob = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        spec = DirectConv4cSpec(problem=prob, block_q=8, block_groups=16)
        kernel = build_direct_conv_4c(spec)
        ll = lower_kernel_to_llvm(kernel)
        # 4x4x4 atom emits one MFMA per (r, s) tile (9 per output row).
        self.assertIn("@llvm.amdgcn.mfma.f32.4x4x4f16", ll)


if __name__ == "__main__":
    unittest.main(verbosity=2)
