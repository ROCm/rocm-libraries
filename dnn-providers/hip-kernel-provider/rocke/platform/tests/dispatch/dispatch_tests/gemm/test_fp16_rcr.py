# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the FP16 RCR GEMM dispatcher case."""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.dispatch import (
    GemmRequest,
    dispatch_gemm_fp16,
    gemm_fp16_candidates,
    gemm_fp16_sweep_space,
)
from rocke.dispatch.gemm import build_kernel
from rocke.dispatch.gemm.support import (
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
)
from rocke.instances.common.gemm_universal import is_valid_spec


class TestGemmFp16Dispatch(unittest.TestCase):
    def test_dispatch_gfx950_request(self):
        req = GemmRequest(M=4096, N=4096, K=4096, arch="gfx950")
        result = dispatch_gemm_fp16(req)
        self.assertEqual(result.kernel_id.op, "gemm")
        self.assertEqual(result.kernel_id.family, "gemm_fp16_rcr")
        self.assertEqual(result.kernel_id.arch, "gfx950")
        self.assertEqual(result.kernel_id.algorithm, "universal_gemm")
        self.assertEqual(result.kernel_id.spec_id, "cdna_cshuffle_default")
        self.assertEqual(result.candidate.name, "universal_gemm_fp16_cdna_cshuffle")
        self.assertEqual(result.grid, (32, 32, 1))
        self.assertEqual(result.block, (256, 1, 1))
        self.assertEqual(
            [item["name"] for item in result.signature],
            ["A", "B", "C", "M", "N", "K"],
        )
        ok, why = is_valid_spec(result.spec, arch=req.arch)
        self.assertTrue(ok, why)

    def test_dispatch_wave32_arch(self):
        req = GemmRequest(M=128, N=64, K=64, arch="gfx1151")
        result = dispatch_gemm_fp16(req)
        self.assertEqual(result.kernel_id.spec_id, "rdna_wmma_default")
        self.assertEqual(result.spec.wave_size, 32)
        self.assertEqual(result.spec.trait.pipeline, "mem")
        self.assertEqual(result.spec.trait.epilogue, "default")
        ok, why = is_valid_spec(result.spec, arch=req.arch)
        self.assertTrue(ok, why)

    def test_kernel_id_is_deterministic(self):
        req = GemmRequest(M=1024, N=2048, K=512, arch="gfx950")
        a = dispatch_gemm_fp16(req).kernel_id
        b = dispatch_gemm_fp16(req).kernel_id
        self.assertEqual(a, b)
        self.assertEqual(a.cache_key, b.cache_key)

    def test_rejects_unsupported_dtype_and_layout(self):
        with self.assertRaisesRegex(ValueError, "fp16 only"):
            dispatch_gemm_fp16(GemmRequest(M=1, N=1, K=1, arch="gfx950", dtype="bf16"))
        with self.assertRaisesRegex(ValueError, "RCR only"):
            dispatch_gemm_fp16(GemmRequest(M=1, N=1, K=1, arch="gfx950", layout="RRR"))
        with self.assertRaisesRegex(ValueError, "algorithm"):
            dispatch_gemm_fp16(
                GemmRequest(M=1, N=1, K=1, arch="gfx950", algorithm="direct_gemm")
            )

    def test_rejects_shapes_without_required_tile_granularity(self):
        # Registered phase-1 variants have pad_m/n/k disabled, so the problem
        # shape must be an exact multiple of the selected CTA tile. This mirrors
        # tile_engine's split between config validity and problem-shape support.
        with self.assertRaisesRegex(ValueError, "M=130 is not divisible"):
            dispatch_gemm_fp16(
                GemmRequest(
                    M=130,
                    N=128,
                    K=32,
                    arch="gfx950",
                    spec_id="cdna_cshuffle_default",
                )
            )
        self.assertEqual(
            gemm_fp16_sweep_space(GemmRequest(M=130, N=128, K=32, arch="gfx950")),
            (),
        )

    def test_config_support_uses_arch_mma_and_tile_knobs(self):
        result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch="gfx950"))
        query = support_query_from_universal_spec(result.spec, arch="gfx950")
        ok, why = gemm_config_supported(query)
        self.assertTrue(ok, why)

        # Same CTA shape with an unsupported warp tile must be rejected by the
        # arch-backed MMA catalog, independent of problem shape.
        bad_query = type(query)(
            **{
                **query.__dict__,
                "warp_tile": (64, 64, 16),
            }
        )
        ok, why = gemm_config_supported(bad_query)
        self.assertFalse(ok)
        self.assertIn("unsupported", why)

    def test_request_shape_granularity_is_separate_from_config_support(self):
        result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch="gfx950"))
        ok, why = request_shape_supported(
            GemmRequest(M=130, N=128, K=32, arch="gfx950"), result.spec
        )
        self.assertFalse(ok)
        self.assertIn("M=130", why)

    def test_registered_variants_can_be_selected_by_spec_id(self):
        cdna = dispatch_gemm_fp16(
            GemmRequest(
                M=1024,
                N=2048,
                K=512,
                arch="gfx950",
                algorithm="universal_gemm",
                spec_id="cdna_mem_64x128",
            )
        )
        self.assertEqual(cdna.candidate.name, "universal_gemm_fp16_cdna_mem")
        self.assertEqual(cdna.spec.trait.pipeline, "mem")
        self.assertEqual(cdna.spec.trait.epilogue, "default")
        self.assertEqual(cdna.grid, (16, 16, 1))

        rdna = dispatch_gemm_fp16(
            GemmRequest(
                M=128,
                N=128,
                K=64,
                arch="gfx1151",
                spec_id="rdna_wmma_32x32",
            )
        )
        self.assertEqual(rdna.candidate.name, "universal_gemm_fp16_rdna_wmma_small")
        self.assertEqual(rdna.grid, (4, 4, 1))

    def test_registry_exposes_multiple_variants(self):
        candidates = gemm_fp16_candidates()
        self.assertGreaterEqual(len(candidates), 4)
        self.assertIn("cdna_cshuffle_default", {c.spec_id for c in candidates})
        self.assertIn("rdna_wmma_default", {c.spec_id for c in candidates})

    def test_ranker_can_override_auto_priority(self):
        def prefer_mem(_request, candidates):
            return sorted(candidates, key=lambda c: c.spec_id != "cdna_mem_64x128")

        result = dispatch_gemm_fp16(
            GemmRequest(M=1024, N=2048, K=512, arch="gfx950"),
            ranker=prefer_mem,
        )
        self.assertEqual(result.kernel_id.spec_id, "cdna_mem_64x128")
        self.assertEqual(result.candidate.name, "universal_gemm_fp16_cdna_mem")

    def test_sweep_space_is_bounded_and_valid(self):
        for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
            req = GemmRequest(M=512, N=512, K=512, arch=arch)
            specs = gemm_fp16_sweep_space(req)
            self.assertGreaterEqual(len(specs), 1)
            self.assertLessEqual(len(specs), 4)
            for spec in specs:
                ok, why = is_valid_spec(spec, arch=arch)
                self.assertTrue(ok, f"{arch}: {why}")

    def test_selected_kernel_lowers_to_llvm(self):
        req = GemmRequest(M=128, N=128, K=64, arch="gfx950")
        result = dispatch_gemm_fp16(req)
        kernel = build_kernel(result)
        llvm = lower_kernel_to_llvm(kernel, arch=req.arch)
        self.assertIn("define amdgpu_kernel void", llvm)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", llvm)


if __name__ == "__main__":
    unittest.main()
