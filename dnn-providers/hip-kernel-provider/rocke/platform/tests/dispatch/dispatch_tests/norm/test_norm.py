# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support tests for the norm2d (rmsnorm/layernorm) dispatcher."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.norm import (
    NormRequest,
    dispatch_norm,
    dispatch_norm_all,
    norm_candidates,
    norm_sweep_space,
)
from rocke.tests.arch_marks import COVERED_ARCHES


def _rms(rows, cols, arch, dtype="fp16"):
    return NormRequest(rows=rows, cols=cols, kind="rmsnorm", arch=arch, dtype=dtype)


def _ln(rows, cols, arch, dtype="fp16"):
    return NormRequest(rows=rows, cols=cols, kind="layernorm", arch=arch, dtype=dtype)


class TestNormDispatch(unittest.TestCase):
    def test_rejects_unknown_kind(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                dispatch_norm(
                    NormRequest(rows=4, cols=4096, kind="groupnorm", arch=arch)
                )

    def test_rejects_unsupported_dtype(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                dispatch_norm(
                    NormRequest(
                        rows=4, cols=4096, kind="rmsnorm", arch=arch, dtype="fp8"
                    )
                )

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_norm(_rms(4, 4096, "gfx000"))

    def test_picks_widest_vec_then_sweet_spot_blocksize(self):
        # cols=4096: block_size=256 at vec=8 is legal (256*8=2048 divides
        # 4096) and is the real measured sweet spot -- see norm.py's module
        # docstring for the cross-shape measurements that replaced the old
        # "largest block_size first" rule (which picked block_size=1024 here
        # and measured slower on every shape checked).
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_norm(_rms(4096, 4096, arch))
                self.assertEqual(r.spec.block_size, 256)
                self.assertEqual(r.spec.vec, 8)
                self.assertEqual(r.candidate.algorithm, "rmsnorm")

    def test_widest_vec_wins_within_sweet_spot_blocksize(self):
        # cols=8192: block_size=256, vec=8 (256*8=2048 divides 8192) remains
        # the pick; vec=8 is preferred within any legal block_size.
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_norm(_rms(4096, 8192, arch))
                self.assertEqual((r.spec.block_size, r.spec.vec), (256, 8))

    def test_layernorm_kind_selects_layernorm_candidate(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_norm(_ln(1024, 2048, arch))
                self.assertEqual(r.candidate.algorithm, "layernorm")
                self.assertIn("layernorm", r.candidate.name)

    def test_grid_is_one_cta_per_row(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_norm(_rms(777, 4096, arch))
                self.assertEqual(r.grid, (777, 1, 1))
                self.assertEqual(r.block, (r.spec.block_size, 1, 1))

    def test_bf16_supported(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_norm(_rms(512, 2048, arch, dtype="bf16"))
                self.assertEqual(r.spec.dtype, "bf16")

    def test_runs_on_rdna_arch(self):
        # Norm is arch-family agnostic; gfx1151 (RDNA, wave32) must select too.
        r = dispatch_norm(_rms(512, 768, "gfx1151"))
        self.assertTrue(r.candidate.spec_id)
        # wave_size baked from the target arch.
        self.assertEqual(r.spec.wave_size, 32)

    def test_kind_gate_filters_candidates(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _rms(4096, 4096, arch)
                for c in norm_candidates():
                    ok, _ = c.admits(req)
                    if ok:
                        self.assertEqual(c.algorithm, "rmsnorm")

    def test_unique_candidate_names(self):
        names = [c.name for c in norm_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_explicit_spec_id_selects_that_candidate(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = NormRequest(
                    rows=4096,
                    cols=4096,
                    kind="rmsnorm",
                    arch=arch,
                    spec_id="rmsnorm_b256_v4",
                )
                r = dispatch_norm(req)
                self.assertEqual(r.candidate.spec_id, "rmsnorm_b256_v4")

    def test_sweep_space_enumerates_valid_block_vec_combos(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _rms(4096, 4096, arch)
                winner = dispatch_norm(req)
                specs = norm_sweep_space(req)
                results = dispatch_norm_all(req)
                self.assertGreater(len(specs), 1)
                self.assertEqual(len(results), len(specs))
                self.assertEqual(results[0].candidate.name, winner.candidate.name)


if __name__ == "__main__":
    unittest.main()
