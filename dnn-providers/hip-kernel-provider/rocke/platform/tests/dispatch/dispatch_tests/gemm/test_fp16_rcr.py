# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the FP16 RCR GEMM dispatcher case."""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.dispatch import (
    GemmRequest,
    dispatch_gemm_fp16,
    dispatch_gemm_fp16_all,
    gemm_fp16_candidates,
    gemm_fp16_sweep_space,
    registered_gemm_fp16_combos,
)
from rocke.dispatch.gemm import build_kernel
from rocke.dispatch.gemm.support import (
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
)
from rocke.instances.common.gemm_universal import is_valid_spec
from rocke.tests.arch_marks import COVERED_ARCHES

# The 32x32 fp16 MFMA intrinsic that the cdna_cshuffle candidate lowers to.
_MFMA_32X32_FP16 = {
    "gfx942": "@llvm.amdgcn.mfma.f32.32x32x8f16",
    "gfx950": "@llvm.amdgcn.mfma.f32.32x32x16.f16",
}


class TestGemmFp16Dispatch(unittest.TestCase):
    def test_dispatch_cdna_request(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = GemmRequest(M=4096, N=4096, K=4096, arch=arch)
                result = dispatch_gemm_fp16(req)
                self.assertEqual(result.kernel_id.op, "gemm")
                self.assertEqual(result.kernel_id.family, "gemm_fp16_rcr")
                self.assertEqual(result.kernel_id.arch, arch)
                self.assertEqual(result.kernel_id.algorithm, "universal_gemm")
                self.assertEqual(result.kernel_id.spec_id, "cdna_cshuffle_default")
                self.assertEqual(
                    result.candidate.name, "universal_gemm_fp16_cdna_cshuffle"
                )
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
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = GemmRequest(M=1024, N=2048, K=512, arch=arch)
                a = dispatch_gemm_fp16(req).kernel_id
                b = dispatch_gemm_fp16(req).kernel_id
                self.assertEqual(a, b)
                self.assertEqual(a.cache_key, b.cache_key)

    def test_rejects_unsupported_dtype_and_layout(self):
        # dtype and layout are declared coverage, so the capability prefilter
        # rejects them before the residual predicate is consulted.
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(
                    ValueError, r"dtype 'bf16' not in \('fp16',\)"
                ):
                    dispatch_gemm_fp16(
                        GemmRequest(M=1, N=1, K=1, arch=arch, dtype="bf16")
                    )
                with self.assertRaisesRegex(
                    ValueError, r"layout 'RRR' not in \('RCR',\)"
                ):
                    dispatch_gemm_fp16(
                        GemmRequest(M=1, N=1, K=1, arch=arch, layout="RRR")
                    )
                with self.assertRaisesRegex(ValueError, "algorithm"):
                    dispatch_gemm_fp16(
                        GemmRequest(M=1, N=1, K=1, arch=arch, algorithm="direct_gemm")
                    )

    def test_rejects_shapes_without_required_tile_granularity(self):
        # Registered phase-1 variants have pad_m/n/k disabled, so the problem
        # shape must be an exact multiple of the selected CTA tile. This mirrors
        # tile_engine's split between config validity and problem-shape support.
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(ValueError, "M=130 is not divisible"):
                    dispatch_gemm_fp16(
                        GemmRequest(
                            M=130,
                            N=128,
                            K=32,
                            arch=arch,
                            spec_id="cdna_cshuffle_default",
                        )
                    )
                self.assertEqual(
                    gemm_fp16_sweep_space(GemmRequest(M=130, N=128, K=32, arch=arch)),
                    (),
                )

    def test_config_support_uses_arch_mma_and_tile_knobs(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch=arch))
                query = support_query_from_universal_spec(result.spec, arch=arch)
                ok, why = gemm_config_supported(query)
                self.assertTrue(ok, why)

                # Same CTA shape with an unsupported warp tile must be rejected by
                # the arch-backed MMA catalog, independent of problem shape.
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
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch=arch))
                ok, why = request_shape_supported(
                    GemmRequest(M=130, N=128, K=32, arch=arch), result.spec
                )
                self.assertFalse(ok)
                self.assertIn("M=130", why)

    def test_registered_variants_can_be_selected_by_spec_id(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                cdna = dispatch_gemm_fp16(
                    GemmRequest(
                        M=1024,
                        N=2048,
                        K=512,
                        arch=arch,
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

    def test_ranker_can_override_auto_priority(self):
        def prefer_mem(_request, candidates):
            return sorted(candidates, key=lambda c: c.spec_id != "cdna_mem_64x128")

        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                result = dispatch_gemm_fp16(
                    GemmRequest(M=1024, N=2048, K=512, arch=arch),
                    ranker=prefer_mem,
                )
                self.assertEqual(result.kernel_id.spec_id, "cdna_mem_64x128")
                self.assertEqual(result.candidate.name, "universal_gemm_fp16_cdna_mem")

    def test_offered_specs_are_valid_on_the_requested_arch(self):
        for arch in (*COVERED_ARCHES, "gfx1151", "gfx1201"):
            req = GemmRequest(M=512, N=512, K=512, arch=arch)
            specs = gemm_fp16_sweep_space(req)
            for spec in specs:
                ok, why = is_valid_spec(spec, arch=arch)
                self.assertTrue(ok, f"{arch}: {why}")

    def test_dispatch_all_enumerates_every_eligible_variant(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = GemmRequest(M=512, N=512, K=512, arch=arch)
                winner = dispatch_gemm_fp16(req)
                results = dispatch_gemm_fp16_all(req)
                self.assertGreater(len(results), 1)
                self.assertEqual(results[0].candidate.name, winner.candidate.name)
                self.assertEqual(len(results), len(gemm_fp16_sweep_space(req)))
                self.assertEqual(len(results), len(registered_gemm_fp16_combos(req)))

    def test_selected_kernel_lowers_to_llvm(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = GemmRequest(M=128, N=128, K=64, arch=arch)
                result = dispatch_gemm_fp16(req)
                kernel = build_kernel(result)
                llvm = lower_kernel_to_llvm(kernel, arch=req.arch)
                self.assertIn("define amdgpu_kernel void", llvm)
                self.assertIn(_MFMA_32X32_FP16[arch], llvm)


class TestGfx1201Compv4Dispatch(unittest.TestCase):
    def _pick(self, shape, **selectors):
        return dispatch_gemm_fp16(GemmRequest(*shape, arch="gfx1201", **selectors))

    def test_direct_a_classes_choose_their_tiles(self):
        cases = (
            ((512, 512, 512), "gfx1201_direct_a_small", (64, 64, 64), 4),
            ((8192, 128, 2048), "gfx1201_direct_a_tall", (128, 64, 64), 8),
            ((2048, 6144, 2560), "gfx1201_direct_a_wide96", (128, 96, 64), 4),
            ((768, 1536, 2048), "gfx1201_direct_a_wide96", (128, 96, 64), 4),
            ((4096, 4096, 4096), "gfx1201_direct_a_default", (128, 64, 64), 4),
            ((128, 8192, 2048), "gfx1201_direct_a_default", (128, 64, 64), 4),
        )
        for shape, spec_id, tile, group in cases:
            with self.subTest(shape=shape):
                result = self._pick(shape)
                spec = result.spec
                self.assertEqual(result.candidate.spec_id, spec_id)
                self.assertEqual(
                    (spec.tile.tile_m, spec.tile.tile_n, spec.tile.tile_k), tile
                )
                self.assertEqual(spec.trait.group_m, group)
                self.assertEqual(is_valid_spec(spec, "gfx1201"), (True, "ok"))

    def test_direct_a_takes_partial_m_and_n_tiles(self):
        for shape, grid in (
            ((1000, 1000, 1024), (16, 8, 1)),
            ((3000, 1500, 2048), (24, 24, 1)),
        ):
            with self.subTest(shape=shape):
                result = self._pick(shape)
                self.assertEqual(result.spec.trait.pipeline, "direct_a")
                self.assertEqual(tuple(result.grid), grid)

    def test_direct_a_refuses_what_it_measured_slower_or_cannot_address(self):
        # Short K loops, 16 KiB row strides, the smallest outputs, K that is
        # not whole 64-wide tiles, and a C span of 2 GiB or more.
        for shape in (
            (1024, 1024, 128),
            (2048, 2048, 8192),
            (256, 256, 1024),
            (2048, 2048, 1040),
            (46336, 46336, 256),
        ):
            with self.subTest(shape=shape):
                self.assertEqual(self._pick(shape).spec.trait.pipeline, "compv4")

    def test_output_bands_and_reduction_bands_choose_legal_tiles(self):
        cases = (
            ((128, 6144, 128), (64, 64, 64), False, 4, 8),
            ((1024, 1024, 192), (128, 128, 64), True, 4, 8),
            ((1152, 1024, 192), (128, 128, 32), False, 4, 8),
            ((4096, 4096, 8192), (128, 128, 64), True, 8, 0),
        )
        for shape, tile, one_wg, group, stagger in cases:
            with self.subTest(shape=shape):
                spec = self._pick(shape).spec
                self.assertEqual(
                    (spec.tile.tile_m, spec.tile.tile_n, spec.tile.tile_k), tile
                )
                self.assertEqual(spec.trait.lds_db_allow_1wg, one_wg)
                self.assertEqual(
                    (spec.trait.group_m, spec.trait.k_stagger), (group, stagger)
                )
                self.assertEqual(is_valid_spec(spec, "gfx1201"), (True, "ok"))

    def test_alignment_and_short_k_fall_through_without_padding(self):
        cases = (
            ((96, 1024, 1024), (32, 64, 64)),
            ((1024, 96, 1024), (64, 32, 64)),
            ((160, 160, 1024), (32, 32, 64)),
            ((2048, 2048, 1040), (128, 128, 16)),
            ((256, 256, 32), (128, 128, 32)),
            ((128, 128, 16), (64, 32, 16)),
        )
        for shape, tile in cases:
            with self.subTest(shape=shape):
                spec = self._pick(shape).spec
                self.assertEqual(
                    (spec.tile.tile_m, spec.tile.tile_n, spec.tile.tile_k), tile
                )
                self.assertFalse(
                    spec.trait.pad_m or spec.trait.pad_n or spec.trait.pad_k
                )
                self.assertEqual(is_valid_spec(spec, "gfx1201"), (True, "ok"))

    def test_long_row_stride_band_does_not_include_neighboring_k(self):
        # A 16 KiB row stride takes the K-64 LDS tile; its neighbours stay direct.
        for k, pipeline in (
            (8128, "direct_a"),
            (8192, "compv4"),
            (8256, "direct_a"),
            (16384, "compv4"),
        ):
            with self.subTest(k=k):
                spec = self._pick((2048, 2048, k)).spec
                self.assertEqual(spec.trait.pipeline, pipeline)
                self.assertEqual(spec.tile.tile_k, 64)
                self.assertEqual(is_valid_spec(spec, "gfx1201"), (True, "ok"))

    def test_long_row_stride_never_takes_a_k32_tile(self):
        for shape, spec_id in (
            ((4096, 4096, 8192), "gfx1201_compv4_long_stride"),
            ((1024, 1024, 8192), "gfx1201_compv4_long_stride_medium"),
            ((1024, 1088, 16384), "gfx1201_compv4_long_stride_small_grouped"),
            ((576, 576, 8192), "gfx1201_compv4_long_stride_small_grouped"),
            ((512, 512, 8192), "gfx1201_compv4_long_stride_small"),
            ((192, 192, 8192), "gfx1201_compv4_long_stride_small"),
        ):
            with self.subTest(shape=shape):
                result = self._pick(shape)
                self.assertEqual(result.candidate.spec_id, spec_id)
                self.assertEqual(result.spec.trait.pipeline, "compv4")
                self.assertEqual(result.spec.tile.tile_k, 64)

    def test_one_step_reductions_keep_legacy_selection(self):
        for shape, pin in (
            ((1024, 1024, 16), "rdna_wmma_default"),
            ((96, 96, 16), "rdna_wmma_32x32"),
        ):
            with self.subTest(shape=shape):
                result = self._pick(shape)
                self.assertEqual(result.candidate.spec_id, pin)
                self.assertEqual(result.spec.trait.pipeline, "mem")

    def test_explicit_legacy_pins_are_not_retuned(self):
        for pin, tile in (
            ("rdna_wmma_default", (64, 32, 16)),
            ("rdna_wmma_32x32", (32, 32, 16)),
        ):
            with self.subTest(pin=pin):
                spec = self._pick((128, 128, 64), spec_id=pin).spec
                self.assertEqual(
                    (spec.tile.tile_m, spec.tile.tile_n, spec.tile.tile_k), tile
                )
                self.assertEqual(spec.trait.pipeline, "mem")
                self.assertFalse(spec.trait.lds_db_allow_1wg)

    def test_pinning_the_selected_candidate_reproduces_its_kernel(self):
        """A replay or sweep that pins the candidate auto dispatch chose must
        build the same kernel, prepared-read addressing included."""
        for shape in ((512, 512, 512), (1024, 1024, 192), (2048, 6144, 2560)):
            with self.subTest(shape=shape):
                auto = self._pick(shape)
                pinned = self._pick(
                    shape,
                    algorithm=auto.candidate.algorithm,
                    spec_id=auto.candidate.spec_id,
                )
                self.assertEqual(pinned.kernel_id.spec_hash, auto.kernel_id.spec_hash)

    def test_candidates_measured_slower_with_prepared_reads_keep_legacy(self):
        for shape, spec_id in (
            ((2048, 2048, 1040), "gfx1201_compv4_k16"),
            ((1024, 1024, 32), "gfx1201_compv4_short_k32"),
        ):
            with self.subTest(shape=shape):
                picked = self._pick(shape)
                self.assertEqual(picked.candidate.spec_id, spec_id)
                self.assertEqual(picked.spec.trait.global_addressing, "legacy")

    def test_pin_cannot_bypass_shape_or_dtype_constraints(self):
        with self.assertRaises(ValueError):
            self._pick((512, 512, 512), spec_id="gfx1201_compv4_medium")
        with self.assertRaises(ValueError):
            self._pick((1024, 1024, 1024), dtype="bf16")
        with self.assertRaises(ValueError):
            self._pick((130, 128, 24))

    def test_m_and_n_no_tile_divides_still_dispatch(self):
        for shape, spec_id, grid in (
            ((300, 400, 640), "gfx1201_direct_a_any", (7, 5, 1)),
            ((130, 128, 32), "gfx1201_direct_a_any_k16", (4, 3, 1)),
            ((77, 33, 16), "gfx1201_direct_a_any_k16", (2, 2, 1)),
        ):
            with self.subTest(shape=shape):
                result = self._pick(shape)
                self.assertEqual(result.candidate.spec_id, spec_id)
                self.assertEqual(tuple(result.grid), grid)
                self.assertEqual(is_valid_spec(result.spec, "gfx1201"), (True, "ok"))

    def test_last_resort_does_not_displace_a_tile_that_divides(self):
        self.assertEqual(self._pick((96, 96, 16)).candidate.spec_id, "rdna_wmma_32x32")
        self.assertEqual(
            self._pick((256, 256, 1024)).candidate.spec_id, "gfx1201_compv4_small"
        )

    def test_other_targets_and_bf16_keep_their_selections(self):
        from rocke.dispatch import dispatch_gemm_bf16

        for arch, expected in (
            ("gfx1151", "rdna_wmma_default"),
            ("gfx942", "cdna_cshuffle_default"),
            ("gfx950", "cdna_cshuffle_default"),
        ):
            with self.subTest(arch=arch):
                result = dispatch_gemm_fp16(GemmRequest(512, 512, 512, arch=arch))
                self.assertEqual(result.candidate.spec_id, expected)
        # BF16 on gfx1201 has no compv4 palette: direct-A classes, else mem.
        for shape, expected in (
            ((512, 512, 512), "gfx1201_direct_a_small"),
            ((128, 128, 64), "rdna_wmma_default"),
            ((2048, 2048, 8192), "rdna_wmma_default"),
        ):
            with self.subTest(bf16=shape):
                result = dispatch_gemm_bf16(
                    GemmRequest(*shape, arch="gfx1201", dtype="bf16")
                )
                self.assertEqual(result.candidate.spec_id, expected)
                self.assertEqual(result.spec.data.dtype_a, "bf16")
        for candidate in gemm_fp16_candidates():
            if not candidate.spec_id.startswith("gfx1201_compv4"):
                continue
            for arch in ("gfx1200", "gfx1151", "gfx1250", "gfx942", "gfx950"):
                with self.subTest(candidate=candidate.spec_id, arch=arch):
                    self.assertFalse(
                        candidate.admits(GemmRequest(1024, 1024, 1024, arch=arch))[0]
                    )


if __name__ == "__main__":
    unittest.main()
