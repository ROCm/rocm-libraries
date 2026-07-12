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


class TestGemmSupportPredicates(unittest.TestCase):
    def _gfx950_query(self) -> GemmSupportQuery:
        result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch="gfx950"))
        return support_query_from_universal_spec(result.spec, arch="gfx950")

    def test_valid_query_is_supported(self):
        ok, why = gemm_config_supported(self._gfx950_query())
        self.assertTrue(ok, why)

    def test_rejects_wave_size_mismatch(self):
        ok, why = gemm_config_supported(replace(self._gfx950_query(), wave_size=32))
        self.assertFalse(ok)
        self.assertIn("wave_size", why)

    def test_rejects_block_size_mismatch(self):
        ok, why = gemm_config_supported(replace(self._gfx950_query(), block_size=128))
        self.assertFalse(ok)
        self.assertIn("block_size", why)

    def test_rejects_unsupported_mma_intrinsic_shape(self):
        ok, why = gemm_config_supported(
            replace(self._gfx950_query(), warp_tile=(64, 64, 16))
        )
        self.assertFalse(ok)
        self.assertIn("unsupported", why)

    def test_rejects_lds_overflow(self):
        q = replace(
            self._gfx950_query(),
            cta_tile=(512, 512, 256),
            warp_shape=(4, 4, 1),
            warp_tile=(32, 32, 16),
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
        result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch="gfx950"))
        ok, why = request_shape_supported(
            GemmRequest(M=130, N=128, K=32, arch="gfx950"), result.spec
        )
        self.assertFalse(ok)
        self.assertIn("M=130", why)

        padded = replace(result.spec, trait=replace(result.spec.trait, pad_m=True))
        ok, why = request_shape_supported(
            GemmRequest(M=130, N=128, K=32, arch="gfx950"), padded
        )
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
