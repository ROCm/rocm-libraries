# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + arch-gating tests for the conv dgrad dispatcher family."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.conv_dgrad import (
    ConvDgradRequest,
    conv_dgrad_candidates,
    dispatch_conv_dgrad,
)


def _dgrad(arch, **kw):
    base = dict(
        N=4,
        C=64,
        K=64,
        Hi=56,
        Wi=56,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        arch=arch,
    )
    base.update(kw)
    return ConvDgradRequest(**base)


class TestConvDgradDispatch(unittest.TestCase):

    # ---- request validation --------------------------------------------------

    def test_rejects_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_conv_dgrad(_dgrad("gfx950", dtype="fp8"))

    def test_rejects_groups(self):
        with self.assertRaises(ValueError):
            dispatch_conv_dgrad(_dgrad("gfx950", G=2))

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_conv_dgrad(_dgrad("gfx000"))

    def test_rejects_degenerate_output(self):
        # 5×5 filter on 3×3 input, no pad → Ho ≤ 0.
        with self.assertRaises(ValueError):
            dispatch_conv_dgrad(
                _dgrad("gfx950", Hi=3, Wi=3, Y=5, X=5, pad_h=0, pad_w=0)
            )

    def test_wrong_op_rejected(self):
        req = ConvDgradRequest(
            N=4, C=64, K=64, Hi=56, Wi=56, Y=3, X=3, arch="gfx950", op="conv"
        )
        with self.assertRaises(ValueError):
            dispatch_conv_dgrad(req)

    # ---- gfx950 selection ----------------------------------------------------

    def test_gfx950_fp16_selects_hiperf_gfx950(self):
        # gfx950 has the 32x32x16 f16 atom → highest-priority candidate.
        r = dispatch_conv_dgrad(_dgrad("gfx950"))
        self.assertEqual(r.candidate.spec_id, "cdna_hiperf_gfx950_64x64")
        self.assertEqual(r.spec.warp_tile_k, 16)

    def test_gfx950_bf16_selects_hiperf_gfx950(self):
        r = dispatch_conv_dgrad(_dgrad("gfx950", dtype="bf16"))
        self.assertEqual(r.candidate.spec_id, "cdna_hiperf_gfx950_64x64")

    # ---- gfx942 selection ----------------------------------------------------

    def test_gfx942_fp16_skips_gfx950_candidate(self):
        # gfx942 lacks mfma32x32x16_f16; must fall through to hiperf (32x32x8).
        r = dispatch_conv_dgrad(_dgrad("gfx942"))
        self.assertNotEqual(r.candidate.spec_id, "cdna_hiperf_gfx950_64x64")
        self.assertIn("cdna", r.candidate.name)

    def test_gfx942_hiperf_gfx950_candidate_unsupported(self):
        for c in conv_dgrad_candidates():
            if c.spec_id == "cdna_hiperf_gfx950_64x64":
                ok, _ = c.admits(_dgrad("gfx942"))
                self.assertFalse(ok)

    # ---- RDNA selection -------------------------------------------------------

    def test_rdna_gfx1151_selects_wmma(self):
        r = dispatch_conv_dgrad(
            ConvDgradRequest(N=2, C=32, K=32, Hi=16, Wi=16, Y=1, X=1, arch="gfx1151")
        )
        self.assertEqual(r.candidate.spec_id, "rdna_wmma_32x32")

    def test_rdna_gfx1201_selects_wmma(self):
        r = dispatch_conv_dgrad(
            ConvDgradRequest(N=2, C=32, K=32, Hi=16, Wi=16, Y=1, X=1, arch="gfx1201")
        )
        self.assertEqual(r.candidate.spec_id, "rdna_wmma_32x32")

    def test_rdna_candidate_unsupported_on_cdna(self):
        req = _dgrad("gfx950")
        for c in conv_dgrad_candidates():
            if "rdna" in c.name:
                ok, _ = c.admits(req)
                self.assertFalse(ok)

    def test_cdna_candidates_unsupported_on_rdna(self):
        req = ConvDgradRequest(N=2, C=32, K=32, Hi=16, Wi=16, Y=1, X=1, arch="gfx1151")
        for c in conv_dgrad_candidates():
            if "cdna" in c.name:
                ok, _ = c.admits(req)
                self.assertFalse(ok)

    # ---- grid and signature --------------------------------------------------

    def test_grid_is_1d_flat_tiles(self):
        # Grid x-dim = total flat tiles over all sub-GEMMs, y=1, z=split_k=1.
        r = dispatch_conv_dgrad(_dgrad("gfx942"))
        self.assertEqual(r.grid[1], 1)  # y = 1
        self.assertEqual(r.grid[2], 1)  # z = split_k = 1
        self.assertGreater(r.grid[0], 0)  # x = flat_tiles > 0

    def test_grid_stride2_is_flat(self):
        # stride=2 → 4 sub-GEMMs; flat_tiles = sum of each sub-GEMM's tile count.
        r = dispatch_conv_dgrad(_dgrad("gfx942", stride_h=2, stride_w=2))
        self.assertEqual(r.grid[1], 1)
        sub_gemms = r.spec.compute_sub_gemms()
        self.assertEqual(r.grid[0], sub_gemms[-1].block_end)

    def test_signature_has_sub_gemm_buf(self):
        r = dispatch_conv_dgrad(_dgrad("gfx950"))
        names = [s["name"] for s in r.signature]
        self.assertIn("sub_gemm_buf", names)
        self.assertIn("num_sub_gemms", names)

    def test_block_is_single_workgroup(self):
        r = dispatch_conv_dgrad(_dgrad("gfx950"))
        self.assertEqual(r.block[1], 1)
        self.assertEqual(r.block[2], 1)
        self.assertGreater(r.block[0], 0)

    # ---- stride / needs_atomic -----------------------------------------------

    def test_stride1_needs_no_atomic(self):
        r = dispatch_conv_dgrad(_dgrad("gfx942"))
        self.assertFalse(r.spec.needs_atomic)

    def test_stride2_no_atomic(self):
        # stride>1 + split_k=1: tilde decomposition guarantees disjoint writes
        # so direct buffer_store is used — no atomic needed.
        r = dispatch_conv_dgrad(_dgrad("gfx942", stride_h=2, stride_w=2))
        self.assertFalse(r.spec.needs_atomic)

    # ---- dispatch result fields ----------------------------------------------

    def test_result_has_kernel_id(self):
        r = dispatch_conv_dgrad(_dgrad("gfx950"))
        self.assertEqual(r.kernel_id.op, "conv_dgrad")
        self.assertIsNotNone(r.kernel_id.request_hash)
        self.assertIsNotNone(r.kernel_id.spec_hash)

    def test_unique_candidate_names(self):
        names = [c.name for c in conv_dgrad_candidates()]
        self.assertEqual(len(names), len(set(names)))

    # ---- spec_id pinning via request -----------------------------------------

    def test_spec_id_pin_selects_mem(self):
        r = dispatch_conv_dgrad(_dgrad("gfx950", spec_id="cdna_mem_64x64"))
        self.assertEqual(r.candidate.spec_id, "cdna_mem_64x64")

    def test_spec_id_pin_unknown_raises(self):
        with self.assertRaises(ValueError):
            dispatch_conv_dgrad(_dgrad("gfx950", spec_id="nonexistent"))


if __name__ == "__main__":
    unittest.main()
