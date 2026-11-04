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
from rocke.dispatch.core import PinRefused
from rocke.dispatch.gemm import build_kernel
from rocke.dispatch.gemm.support import (
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
)
from rocke.instances.common.gemm_universal import is_valid_spec
from rocke.tests.arch_marks import arch_specific


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
        # dtype and layout are declared coverage, so the capability prefilter
        # rejects them before the residual predicate is consulted.
        with self.assertRaisesRegex(ValueError, r"dtype 'bf16' not in \('fp16',\)"):
            dispatch_gemm_fp16(GemmRequest(M=1, N=1, K=1, arch="gfx950", dtype="bf16"))
        with self.assertRaisesRegex(ValueError, r"layout 'RRR' not in \('RCR',\)"):
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

    def test_dispatch_all_enumerates_every_eligible_variant(self):
        req = GemmRequest(M=512, N=512, K=512, arch="gfx950")
        winner = dispatch_gemm_fp16(req)
        results = dispatch_gemm_fp16_all(req)
        self.assertGreater(len(results), 1)
        self.assertEqual(results[0].candidate.name, winner.candidate.name)
        self.assertEqual(len(results), len(gemm_fp16_sweep_space(req)))
        self.assertEqual(len(results), len(registered_gemm_fp16_combos(req)))

    def test_selected_kernel_lowers_to_llvm(self):
        req = GemmRequest(M=128, N=128, K=64, arch="gfx950")
        result = dispatch_gemm_fp16(req)
        kernel = build_kernel(result)
        llvm = lower_kernel_to_llvm(kernel, arch=req.arch)
        self.assertIn("define amdgpu_kernel void", llvm)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", llvm)


@arch_specific(
    ("gfx1201",),
    reason="the size-gated WMMA candidates are registered for gfx1201 only",
)
class TestGfx1201Fp16Dispatch(unittest.TestCase):
    _MEDIUM = "gfx1201_wmma_128x128x16"
    _TALL = "gfx1201_wmma_128x64x32"
    _WIDE = "gfx1201_wmma_64x128x32"
    _LARGE = "gfx1201_wmma_128x128x16_large"
    _NEW = (_MEDIUM, _TALL, _WIDE, _LARGE)
    _DEFAULT = "rdna_wmma_default"
    _SMALL = "rdna_wmma_32x32"

    def _assert_pin_refused(self, req, reason):
        # A refused pin never falls back: it raises PinRefused naming the
        # pinned candidate and the gate that refused it.
        with self.assertRaisesRegex(PinRefused, reason) as caught:
            dispatch_gemm_fp16(req)
        self.assertEqual(caught.exception.spec_id, req.spec_id)
        self.assertEqual(
            list(caught.exception.refusals),
            [f"universal_gemm_fp16_{req.spec_id}"],
        )

    def _assert_unrouted(self, req):
        with self.assertRaisesRegex(ValueError, "^no candidate supports request") as e:
            dispatch_gemm_fp16(req)
        self.assertNotIsInstance(e.exception, PinRefused)
        self.assertEqual(dispatch_gemm_fp16_all(req), ())

    def test_selected_tiles_and_launch_geometry(self):
        cases = (
            (1024, 1024, self._MEDIUM, (128, 128, 16), (4, 1, 1), (8, 8, 1), 128),
            (2048, 2048, self._TALL, (128, 64, 32), (2, 1, 1), (32, 16, 1), 64),
            (512, 2112, self._TALL, (128, 64, 32), (2, 1, 1), (33, 4, 1), 64),
            (2112, 512, self._WIDE, (64, 128, 32), (1, 2, 1), (4, 33, 1), 64),
        )
        for m, n, spec_id, tile_shape, warps, grid, block_size in cases:
            with self.subTest(M=m, N=n, spec_id=spec_id):
                req = GemmRequest(M=m, N=n, K=256, arch="gfx1201")
                result = dispatch_gemm_fp16(req)
                self.assertEqual(result.kernel_id.spec_id, spec_id)
                tile = result.spec.tile
                self.assertEqual((tile.tile_m, tile.tile_n, tile.tile_k), tile_shape)
                self.assertEqual((tile.warp_m, tile.warp_n, tile.warp_k), warps)
                self.assertEqual(result.grid, grid)
                self.assertEqual(result.block, (block_size, 1, 1))
                self.assertEqual(result.spec.wave_size, 32)
                trait = result.spec.trait
                self.assertEqual(
                    (trait.pipeline, trait.scheduler, trait.epilogue),
                    ("mem", "intrawave", "default"),
                )
                self.assertEqual((trait.pad_m, trait.pad_n, trait.pad_k), (False,) * 3)
                ok, why = is_valid_spec(result.spec, arch=req.arch)
                self.assertTrue(ok, why)

    def test_rectangular_output_area_bounds_are_inclusive(self):
        # Adjacent 128-column tiles straddle each area boundary while keeping
        # every new tile divisible. Neither individual output axis is a gate.
        cases = (
            (128, 8064, self._DEFAULT),
            (128, 8192, self._MEDIUM),
            (8192, 128, self._MEDIUM),
            (128, 16384, self._MEDIUM),
            (16384, 128, self._MEDIUM),
            (128, 16512, self._TALL),
        )
        for m, n, expected in cases:
            with self.subTest(M=m, N=n):
                result = dispatch_gemm_fp16(
                    GemmRequest(M=m, N=n, K=256, arch="gfx1201")
                )
                self.assertEqual(result.kernel_id.spec_id, expected)

    def test_output_area_selection_is_independent_of_reduction_length(self):
        for m, n, expected in (
            (512, 512, self._DEFAULT),
            (512, 2048, self._MEDIUM),
            (512, 8192, self._TALL),
        ):
            for k in (256, 4096):
                with self.subTest(M=m, N=n, K=k):
                    result = dispatch_gemm_fp16(
                        GemmRequest(M=m, N=n, K=k, arch="gfx1201")
                    )
                    self.assertEqual(result.kernel_id.spec_id, expected)

    def test_eligible_priority_order_and_k_granularity(self):
        old = [self._DEFAULT, self._SMALL]
        cases = (
            (512, 512, 256, old),
            (1024, 1024, 224, old),
            (1024, 1024, 256, [self._MEDIUM, self._TALL, self._WIDE, *old]),
            (1024, 1024, 272, [self._MEDIUM, *old]),
            (2048, 2048, 256, [self._TALL, self._WIDE, self._LARGE, *old]),
            (2048, 2048, 272, [self._LARGE, *old]),
        )
        for m, n, k, expected in cases:
            with self.subTest(M=m, N=n, K=k):
                req = GemmRequest(M=m, N=n, K=k, arch="gfx1201")
                results = dispatch_gemm_fp16_all(req)
                self.assertEqual([r.kernel_id.spec_id for r in results], expected)
                self.assertEqual(dispatch_gemm_fp16(req).kernel_id.spec_id, expected[0])

    def test_large_outputs_without_k32_granularity_keep_k16_tile(self):
        for m, n, k, grid in (
            (2048, 2048, 272, (16, 16, 1)),
            (4096, 1024, 1040, (8, 32, 1)),
        ):
            with self.subTest(M=m, N=n, K=k):
                req = GemmRequest(M=m, N=n, K=k, arch="gfx1201")
                result = dispatch_gemm_fp16(req)
                self.assertEqual(result.kernel_id.spec_id, self._LARGE)
                tile = result.spec.tile
                self.assertEqual(
                    (tile.tile_m, tile.tile_n, tile.tile_k), (128, 128, 16)
                )
                self.assertEqual((tile.warp_m, tile.warp_n, tile.warp_k), (4, 1, 1))
                self.assertEqual(result.grid, grid)
                self.assertEqual(result.block, (128, 1, 1))
                ok, why = is_valid_spec(result.spec, arch=req.arch)
                self.assertTrue(ok, why)

    def test_tile_divisibility_falls_back_without_padding(self):
        for m, n, expected, grid, block_size in (
            (1088, 1088, self._DEFAULT, (34, 17, 1), 64),
            (1024, 1056, self._DEFAULT, (33, 16, 1), 64),
            (1056, 1024, self._SMALL, (32, 33, 1), 128),
        ):
            with self.subTest(M=m, N=n):
                result = dispatch_gemm_fp16(
                    GemmRequest(M=m, N=n, K=256, arch="gfx1201")
                )
                self.assertEqual(result.kernel_id.spec_id, expected)
                self.assertEqual(result.grid, grid)
                self.assertEqual(result.block, (block_size, 1, 1))
                ok, why = is_valid_spec(result.spec, arch="gfx1201")
                self.assertTrue(ok, why)

    def test_old_pins_keep_their_tiles_and_compile_identity(self):
        for spec_id, tile_shape, grid, block_size in (
            (self._DEFAULT, (64, 32, 16), (32, 16, 1), 64),
            (self._SMALL, (32, 32, 16), (32, 32, 1), 128),
        ):
            with self.subTest(spec_id=spec_id):
                small = dispatch_gemm_fp16(
                    GemmRequest(M=512, N=512, K=256, arch="gfx1201", spec_id=spec_id)
                )
                medium = dispatch_gemm_fp16(
                    GemmRequest(M=1024, N=1024, K=256, arch="gfx1201", spec_id=spec_id)
                )
                self.assertEqual(medium.kernel_id.spec_id, spec_id)
                tile = medium.spec.tile
                self.assertEqual((tile.tile_m, tile.tile_n, tile.tile_k), tile_shape)
                self.assertEqual(medium.grid, grid)
                self.assertEqual(medium.block, (block_size, 1, 1))
                self.assertEqual(
                    small.kernel_id.compile_key, medium.kernel_id.compile_key
                )

    def test_new_pins_identify_fixed_specs(self):
        compile_keys = set()
        for spec_id in (self._MEDIUM, self._TALL, self._WIDE):
            with self.subTest(spec_id=spec_id):
                square = dispatch_gemm_fp16(
                    GemmRequest(M=1024, N=1024, K=256, arch="gfx1201", spec_id=spec_id)
                )
                rectangle = dispatch_gemm_fp16(
                    GemmRequest(M=512, N=2048, K=512, arch="gfx1201", spec_id=spec_id)
                )
                self.assertEqual(square.kernel_id.spec_id, spec_id)
                self.assertEqual(rectangle.kernel_id.spec_id, spec_id)
                self.assertEqual(
                    square.kernel_id.compile_key, rectangle.kernel_id.compile_key
                )
                compile_keys.add(square.kernel_id.compile_key)
        self.assertEqual(len(compile_keys), 3)

    def test_ranker_overrides_priority_but_not_admissibility(self):
        for preferred in (self._WIDE, self._DEFAULT):
            with self.subTest(preferred=preferred):

                def prefer(_request, candidates):
                    return sorted(candidates, key=lambda c: c.spec_id != preferred)

                result = dispatch_gemm_fp16(
                    GemmRequest(M=1024, N=1024, K=256, arch="gfx1201"), ranker=prefer
                )
                self.assertEqual(result.kernel_id.spec_id, preferred)
                below_gate = dispatch_gemm_fp16(
                    GemmRequest(M=512, N=512, K=256, arch="gfx1201"), ranker=prefer
                )
                self.assertEqual(below_gate.kernel_id.spec_id, self._DEFAULT)

    def test_new_pins_cannot_bypass_shape_gates(self):
        # Every new candidate refuses a small output and, at an output area it
        # otherwise admits, a reduction shorter than 256.
        for spec_id, admitted in (
            (self._MEDIUM, 1024),
            (self._TALL, 1024),
            (self._WIDE, 1024),
            (self._LARGE, 2048),
        ):
            for m, k, reason in (
                (512, 256, "output_elements=262144 < min"),
                (admitted, 224, "K=224 < min 256"),
            ):
                with self.subTest(spec_id=spec_id, M=m, N=m, K=k):
                    self._assert_pin_refused(
                        GemmRequest(M=m, N=m, K=k, arch="gfx1201", spec_id=spec_id),
                        "capability: " + reason,
                    )
        for spec_id, m, n, k, reason in (
            (self._MEDIUM, 2048, 2048, 256, "output_elements=4194304 > max 2097152"),
            (self._MEDIUM, 1088, 1024, 256, "M=1088 is not divisible by tile_m=128"),
            (self._TALL, 1088, 1024, 256, "M=1088 is not divisible by tile_m=128"),
            (self._WIDE, 1024, 1088, 256, "N=1088 is not divisible by tile_n=128"),
            (self._TALL, 1024, 1024, 272, "K=272 is not divisible by tile_k=32"),
            (self._WIDE, 1024, 1024, 272, "K=272 is not divisible by tile_k=32"),
            (self._LARGE, 1024, 1024, 256, "output_elements=1048576 < min 2097153"),
        ):
            with self.subTest(spec_id=spec_id, M=m, N=n, K=k):
                self._assert_pin_refused(
                    GemmRequest(M=m, N=n, K=k, arch="gfx1201", spec_id=spec_id),
                    reason,
                )

    def test_other_architectures_keep_existing_selections(self):
        for arch, expected in (
            ("gfx1151", [self._DEFAULT, self._SMALL]),
            ("gfx11-generic", [self._DEFAULT, self._SMALL]),
            ("gfx942", ["cdna_cshuffle_default", "cdna_mem_64x128"]),
            ("gfx950", ["cdna_cshuffle_default", "cdna_mem_64x128"]),
        ):
            for size in (1024, 2048):
                with self.subTest(arch=arch, size=size):
                    req = GemmRequest(M=size, N=size, K=256, arch=arch)
                    self.assertEqual(
                        [r.kernel_id.spec_id for r in dispatch_gemm_fp16_all(req)],
                        expected,
                    )
                    self.assertEqual(
                        dispatch_gemm_fp16(req).kernel_id.spec_id, expected[0]
                    )
            for spec_id in self._NEW:
                with self.subTest(arch=arch, spec_id=spec_id):
                    self._assert_pin_refused(
                        GemmRequest(M=1024, N=1024, K=256, arch=arch, spec_id=spec_id),
                        f"capability: arch '{arch}' not in \\('gfx1201',\\)",
                    )

    def test_rejects_invalid_shapes_and_architectures(self):
        for m, n, k in (
            (1025, 1024, 256),
            (1024, 1025, 256),
            (1024, 1024, 257),
            (0, 1024, 256),
            (1024, -32, 256),
            (1024, 1024, 0),
        ):
            with self.subTest(M=m, N=n, K=k):
                self._assert_unrouted(GemmRequest(M=m, N=n, K=k, arch="gfx1201"))
        self._assert_unrouted(GemmRequest(M=1024, N=1024, K=256, arch="gfx0000"))
        # The new candidates are gfx1201-only; other gfx12/gfx1250 targets do
        # not inherit them, whatever else those targets may support.
        for arch in ("gfx1200", "gfx12-generic", "gfx1250"):
            for spec_id in self._NEW:
                with self.subTest(arch=arch, spec_id=spec_id):
                    self._assert_pin_refused(
                        GemmRequest(M=2048, N=2048, K=256, arch=arch, spec_id=spec_id),
                        f"capability: arch '{arch}' not in \\('gfx1201',\\)",
                    )


if __name__ == "__main__":
    unittest.main()
