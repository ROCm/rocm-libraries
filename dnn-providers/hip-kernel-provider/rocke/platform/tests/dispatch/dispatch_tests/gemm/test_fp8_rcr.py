# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support tests for the FP8 RCR block-scale GEMM dispatcher case.

CPU-only: everything here is selection, spec derivation, and IR construction,
none of which needs a device. The numeric check lives with the runtime tests.
"""

from __future__ import annotations

import unittest

from rocke.dispatch.gemm import gemm_fp8_candidates, gemm_fp8_sweep_space
from rocke.dispatch.gemm.fp8_rcr import (
    GEMM_FP8_REGISTRY,
    Fp8GemmRequest,
    dispatch_gemm_fp8,
)

# The three fp8 GEMMs a Qwen3-30B-A3B decode step issues per layer: the fused
# QKV projection, the attention output projection, and the MoE router gate.
_QKV = (32, 5120, 2048)
_O_PROJ = (32, 2048, 4096)
_ROUTER = (32, 128, 2048)


def _req(M, N, K, arch="gfx950", **kw):
    return Fp8GemmRequest(M=M, N=N, K=K, arch=arch, **kw)


class TestFp8RcrDispatch(unittest.TestCase):
    def test_decode_shapes_all_select_the_per_tensor_candidate(self):
        for M, N, K in (_QKV, _O_PROJ, _ROUTER):
            with self.subTest(M=M, N=N, K=K):
                r = dispatch_gemm_fp8(_req(M, N, K))
                self.assertEqual(r.candidate.spec_id, "cdna_mfma_16x16_per_tensor")
                self.assertEqual(r.spec.layout, "RCR")
                self.assertEqual(r.spec.dtype_c, "bf16")
                self.assertEqual(r.spec.quant_mode, "abquant")

    def test_per_tensor_scaling_is_the_whole_tensor_group(self):
        """One scale per side is the degenerate ``(M, N, K)`` grouping, which is
        what makes the body's block-scale index arithmetic land on element 0."""
        r = dispatch_gemm_fp8(_req(*_ROUTER))
        self.assertEqual(r.spec.group_size_mnk, (32, 128, 2048))

    def test_grid_covers_the_output_in_16x16_tiles(self):
        r = dispatch_gemm_fp8(_req(*_ROUTER))
        self.assertEqual(r.grid, (128 // 16, 32 // 16, 1))
        self.assertEqual(r.block, (64, 1, 1))

    def test_signature_is_the_scaled_mm_calling_convention(self):
        r = dispatch_gemm_fp8(_req(*_ROUTER))
        self.assertEqual(
            [p["name"] for p in r.signature],
            ["A", "B", "AScale", "BScale", "C", "M", "N", "K"],
        )

    def test_torch_dtype_spelling_is_accepted(self):
        r = dispatch_gemm_fp8(_req(*_ROUTER, dtype="float8_e4m3fn"))
        self.assertEqual(r.spec.mantissa_dtype, "fp8e4m3")

    def test_bf8_selects_the_same_candidate_with_its_own_atom(self):
        r = dispatch_gemm_fp8(_req(*_ROUTER, dtype="bf8e5m2"))
        self.assertEqual(r.spec.mantissa_dtype, "bf8e5m2")
        self.assertEqual(r.candidate.spec_id, "cdna_mfma_16x16_per_tensor")

    def test_block_scaled_requests_are_refused_rather_than_reinterpreted(self):
        """The per-tensor kernel reads one scale; serving a block-scaled request
        with it would read element 0 of an array meant to be indexed per K
        block, which is a wrong answer rather than a slow one."""
        with self.assertRaises(ValueError) as cm:
            dispatch_gemm_fp8(_req(*_ROUTER, scale_mode="block"))
        self.assertIn("requires features", str(cm.exception))

    def test_non_fp8_dtype_is_rejected(self):
        for dtype in ("bf16", "fp16"):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    dispatch_gemm_fp8(_req(*_ROUTER, dtype=dtype))

    def test_partial_tiles_are_rejected(self):
        """v1 has no partial-tile path, so a shape that does not fill whole
        16x16 output tiles must not be selected."""
        for M, N, K in ((32, 127, 2048), (31, 128, 2048)):
            with self.subTest(M=M, N=N):
                with self.assertRaises(ValueError):
                    dispatch_gemm_fp8(_req(M, N, K))

    def test_k_must_be_a_whole_number_of_mfma_atoms(self):
        with self.assertRaises(ValueError):
            dispatch_gemm_fp8(_req(32, 128, 2000))

    def test_rdna_arch_is_not_served(self):
        """The body is an MFMA low-bit path; RDNA exposes WMMA instead."""
        with self.assertRaises(ValueError):
            dispatch_gemm_fp8(_req(*_ROUTER, arch="gfx1151"))

    def test_gfx942_and_gfx950_both_select(self):
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                r = dispatch_gemm_fp8(_req(*_ROUTER, arch=arch))
                self.assertEqual(r.kernel_id.arch, arch)

    def test_kernel_id_separates_compile_identity_from_the_problem(self):
        a = dispatch_gemm_fp8(_req(*_ROUTER)).kernel_id
        b = dispatch_gemm_fp8(_req(*_QKV)).kernel_id
        self.assertNotEqual(a.selection_key, b.selection_key)
        # The body specialises on M/N/K, so two shapes are two compiles.
        self.assertNotEqual(a.compile_key, b.compile_key)

    def test_dispatch_is_reproducible(self):
        first = dispatch_gemm_fp8(_req(*_ROUTER)).kernel_id
        second = dispatch_gemm_fp8(_req(*_ROUTER)).kernel_id
        self.assertEqual(first, second)

    def test_selection_builds_ir(self):
        kernel = dispatch_gemm_fp8(_req(*_ROUTER)).build()
        self.assertIn("rcr", kernel.name)
        self.assertIn("bf16", kernel.name)

    def test_sweep_space_is_bounded_and_empty_for_a_bad_request(self):
        self.assertEqual(len(gemm_fp8_sweep_space(_req(*_ROUTER))), 1)
        self.assertEqual(gemm_fp8_sweep_space(_req(*_ROUTER, dtype="bf16")), ())

    def test_registry_requires_buildable_and_bindable_candidates(self):
        self.assertTrue(GEMM_FP8_REGISTRY.require_build)
        self.assertTrue(GEMM_FP8_REGISTRY.require_binding)
        for c in gemm_fp8_candidates():
            with self.subTest(candidate=c.name):
                self.assertIsNotNone(c.build)
                self.assertIsNotNone(c.bind)

    def test_unique_candidate_names(self):
        names = [c.name for c in gemm_fp8_candidates()]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
