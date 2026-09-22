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
        # Matched as a prefix: the WMMA path also accepts the scheduled
        # 'wmma_v1' pipeline, so the message names both. What this asserts is
        # that a CDNA pipeline like compv4 is refused, not the exact phrasing.
        self.assertIn("WMMA path supports only the 'mem'", why)
        ok, why = gemm_config_supported(replace(q, epilogue="cshuffle"))
        self.assertFalse(ok)
        self.assertIn("WMMA path supports only the 'default' epilogue", why)

    def test_gfx1250_wmma_atom_and_pipeline(self):
        """gfx1250's WMMA atom is the K=32 16x16x32 form, not gfx11's 16x16x16.

        This predicate duplicates the gate in
        ``gemm_universal.is_valid_spec``; when the copies drift, a legal spec
        becomes undispatchable. gfx1250 is also ``family="cdna"`` at wave32, so
        it must not be admitted to the wave64 MFMA rules.
        """
        res = dispatch_gemm_fp16(GemmRequest(M=4096, N=4096, K=4096, arch="gfx1250"))
        q = support_query_from_universal_spec(res.spec, arch="gfx1250")
        self.assertEqual(q.warp_tile, (16, 16, 32))
        ok, why = gemm_config_supported(q)
        self.assertTrue(ok, why)

        # The gfx11-era atom is not legal here. It is refused by the atom
        # catalog before the WMMA gate is even reached, because gfx1250's
        # MmaCatalog carries no 16x16x16 fp16 entry at all.
        ok, why = gemm_config_supported(replace(q, warp_tile=(16, 16, 16)))
        self.assertFalse(ok)
        self.assertIn("(16, 16, 16)", why)
        self.assertIn("gfx1250", why)

        # ... and the scheduled WMMA pipeline is.
        ok, why = gemm_config_supported(replace(q, pipeline="wmma_v1"))
        self.assertTrue(ok, why)

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
