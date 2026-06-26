# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support tests for the norm2d (rmsnorm/layernorm) dispatcher."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.norm import (
    NormRequest,
    dispatch_norm,
    norm_candidates,
)


def _rms(rows, cols, arch, dtype="fp16"):
    return NormRequest(rows=rows, cols=cols, kind="rmsnorm", arch=arch, dtype=dtype)


def _ln(rows, cols, arch, dtype="fp16"):
    return NormRequest(rows=rows, cols=cols, kind="layernorm", arch=arch, dtype=dtype)


class TestNormDispatch(unittest.TestCase):
    def test_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            dispatch_norm(
                NormRequest(rows=4, cols=4096, kind="groupnorm", arch="gfx950")
            )

    def test_rejects_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_norm(
                NormRequest(
                    rows=4, cols=4096, kind="rmsnorm", arch="gfx950", dtype="fp8"
                )
            )

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_norm(_rms(4, 4096, "gfx000"))

    def test_picks_largest_blocksize_then_widest_vec(self):
        # cols=4096: b1024*v4=4096 divides; b1024*v8=8192 does not.
        r = dispatch_norm(_rms(4096, 4096, "gfx950"))
        self.assertEqual(r.spec.block_size, 1024)
        self.assertEqual(r.spec.vec, 4)
        self.assertEqual(r.candidate.algorithm, "rmsnorm")

    def test_widest_vec_when_divisible(self):
        # cols=8192: b1024*v8=8192 divides -> widest vec wins.
        r = dispatch_norm(_rms(4096, 8192, "gfx950"))
        self.assertEqual((r.spec.block_size, r.spec.vec), (1024, 8))

    def test_layernorm_kind_selects_layernorm_candidate(self):
        r = dispatch_norm(_ln(1024, 2048, "gfx950"))
        self.assertEqual(r.candidate.algorithm, "layernorm")
        self.assertIn("layernorm", r.candidate.name)

    def test_grid_is_one_cta_per_row(self):
        r = dispatch_norm(_rms(777, 4096, "gfx950"))
        self.assertEqual(r.grid, (777, 1, 1))
        self.assertEqual(r.block, (r.spec.block_size, 1, 1))

    def test_bf16_supported(self):
        r = dispatch_norm(_rms(512, 2048, "gfx950", dtype="bf16"))
        self.assertEqual(r.spec.dtype, "bf16")

    def test_runs_on_rdna_arch(self):
        # Norm is arch-family agnostic; gfx1151 (RDNA, wave32) must select too.
        r = dispatch_norm(_rms(512, 768, "gfx1151"))
        self.assertTrue(r.candidate.spec_id)
        # wave_size baked from the target arch.
        self.assertEqual(r.spec.wave_size, 32)

    def test_kind_gate_filters_candidates(self):
        req = _rms(4096, 4096, "gfx950")
        for c in norm_candidates():
            ok, _ = c.supports(req)
            if ok:
                self.assertEqual(c.algorithm, "rmsnorm")

    def test_unique_candidate_names(self):
        names = [c.name for c in norm_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_explicit_spec_id_selects_that_candidate(self):
        req = NormRequest(
            rows=4096,
            cols=4096,
            kind="rmsnorm",
            arch="gfx950",
            spec_id="rmsnorm_b256_v4",
        )
        r = dispatch_norm(req)
        self.assertEqual(r.candidate.spec_id, "rmsnorm_b256_v4")


if __name__ == "__main__":
    unittest.main()
