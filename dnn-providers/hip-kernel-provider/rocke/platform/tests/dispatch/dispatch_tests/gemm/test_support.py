# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Support predicate tests for GEMM dispatcher candidates."""

from __future__ import annotations

import unittest
from dataclasses import replace

from rocke.dispatch import GemmRequest, dispatch_gemm_fp16
from rocke.dispatch.gemm.support import (
    GemmSupportQuery,
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
)

_CDNA_ARCHES = ("gfx942", "gfx950")

# A 32x32 fp16 MFMA warp tile that the arch catalog accepts (K differs by arch).
_WARP_TILE_32X32 = {"gfx942": (32, 32, 8), "gfx950": (32, 32, 16)}


class TestGemmSupportPredicates(unittest.TestCase):
    def _cdna_query(self, arch: str) -> GemmSupportQuery:
        result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch=arch))
        return support_query_from_universal_spec(result.spec, arch=arch)

    def test_valid_query_is_supported(self):
        for arch in _CDNA_ARCHES:
            with self.subTest(arch=arch):
                ok, why = gemm_config_supported(self._cdna_query(arch))
                self.assertTrue(ok, why)

    def test_rejects_wave_size_mismatch(self):
        for arch in _CDNA_ARCHES:
            with self.subTest(arch=arch):
                ok, why = gemm_config_supported(
                    replace(self._cdna_query(arch), wave_size=32)
                )
                self.assertFalse(ok)
                self.assertIn("wave_size", why)

    def test_rejects_block_size_mismatch(self):
        for arch in _CDNA_ARCHES:
            with self.subTest(arch=arch):
                ok, why = gemm_config_supported(
                    replace(self._cdna_query(arch), block_size=128)
                )
                self.assertFalse(ok)
                self.assertIn("block_size", why)

    def test_rejects_unsupported_mma_intrinsic_shape(self):
        for arch in _CDNA_ARCHES:
            with self.subTest(arch=arch):
                ok, why = gemm_config_supported(
                    replace(self._cdna_query(arch), warp_tile=(64, 64, 16))
                )
                self.assertFalse(ok)
                self.assertIn("unsupported", why)

    def test_rejects_lds_overflow(self):
        for arch in _CDNA_ARCHES:
            with self.subTest(arch=arch):
                q = replace(
                    self._cdna_query(arch),
                    cta_tile=(512, 512, 256),
                    warp_shape=(4, 4, 1),
                    warp_tile=_WARP_TILE_32X32[arch],
                    block_size=1024,
                )
                ok, why = gemm_config_supported(q)
                self.assertFalse(ok)
                self.assertIn("LDS budget", why)

    def test_rejects_wmma_pipeline_and_epilogue_restrictions(self):
        rdna = dispatch_gemm_fp16(GemmRequest(M=64, N=32, K=16, arch="gfx1151"))
        q = support_query_from_universal_spec(rdna.spec, arch="gfx1151")
        ok, why = gemm_config_supported(replace(q, pipeline="compv4"))
        self.assertFalse(ok)
        self.assertIn("WMMA path supports only the 'mem' pipeline", why)
        ok, why = gemm_config_supported(replace(q, epilogue="cshuffle"))
        self.assertFalse(ok)
        self.assertIn("WMMA path supports only the 'default' epilogue", why)

    def test_request_shape_support_respects_padding_flags(self):
        for arch in _CDNA_ARCHES:
            with self.subTest(arch=arch):
                result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch=arch))
                ok, why = request_shape_supported(
                    GemmRequest(M=130, N=128, K=32, arch=arch), result.spec
                )
                self.assertFalse(ok)
                self.assertIn("M=130", why)

                padded = replace(
                    result.spec, trait=replace(result.spec.trait, pad_m=True)
                )
                ok, why = request_shape_supported(
                    GemmRequest(M=130, N=128, K=32, arch=arch), padded
                )
                self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
