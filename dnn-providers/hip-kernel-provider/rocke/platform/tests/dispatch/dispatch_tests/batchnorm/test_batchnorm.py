# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support tests for the batchnorm2d dispatcher."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.batchnorm import (
    BatchNormRequest,
    batchnorm_candidates,
    batchnorm_sweep_space,
    dispatch_batchnorm,
    dispatch_batchnorm_all,
)
from rocke.tests.arch_marks import COVERED_ARCHES


def _req(n, c, h, w, arch, training=True, dtype="fp16", **kw):
    return BatchNormRequest(
        n=n, c=c, h=h, w=w, arch=arch, training=training, dtype=dtype, **kw
    )


class TestBatchNormDispatch(unittest.TestCase):
    def test_rejects_unsupported_dtype(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                dispatch_batchnorm(_req(1, 64, 64, 64, arch, dtype="fp8"))

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_batchnorm(_req(1, 64, 64, 64, "gfx000"))

    def test_rejects_nonpositive_dims(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                dispatch_batchnorm(_req(0, 64, 64, 64, arch))

    def test_picks_largest_blocksize_then_widest_vec(self):
        # hw=4096: b1024*v4=4096 divides; b1024*v8=8192 does not.
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_batchnorm(_req(1, 32, 64, 64, arch))
                self.assertEqual(r.spec.block_size, 1024)
                self.assertEqual(r.spec.vec, 4)
                self.assertEqual(r.candidate.algorithm, "batchnorm")

    def test_widest_vec_when_divisible(self):
        # hw=8192: b1024*v8=8192 divides -> widest vec wins.
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_batchnorm(_req(1, 16, 64, 128, arch))
                self.assertEqual((r.spec.block_size, r.spec.vec), (1024, 8))

    def test_grid_is_one_cta_per_channel(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_batchnorm(_req(1, 777, 64, 64, arch))
                self.assertEqual(r.grid, (777, 1, 1))
                self.assertEqual(r.block, (r.spec.block_size, 1, 1))

    def test_bf16_supported(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_batchnorm(_req(1, 32, 32, 64, arch, dtype="bf16"))
                self.assertEqual(r.spec.dtype, "bf16")

    def test_runs_on_rdna_arch(self):
        # BatchNorm is arch-family agnostic; gfx1151 (RDNA, wave32) must select too.
        r = dispatch_batchnorm(_req(1, 32, 24, 32, "gfx1151"))
        self.assertTrue(r.candidate.spec_id)
        self.assertEqual(r.spec.wave_size, 32)

    def test_training_and_inference_both_dispatchable(self):
        for arch in COVERED_ARCHES:
            for training in (True, False):
                with self.subTest(arch=arch, training=training):
                    r = dispatch_batchnorm(
                        _req(1, 32, 64, 64, arch, training=training)
                    )
                    self.assertEqual(r.spec.training, training)

    def test_unique_candidate_names(self):
        names = [c.name for c in batchnorm_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_explicit_spec_id_selects_that_candidate(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(1, 32, 64, 64, arch, spec_id="batchnorm_b256_v4")
                r = dispatch_batchnorm(req)
                self.assertEqual(r.candidate.spec_id, "batchnorm_b256_v4")

    def test_channels_per_block_default_is_one_and_not_auto_selected(self):
        # channels_per_block is a default-off lever: auto dispatch must keep
        # picking the channels_per_block=1 candidate even though cpb>1
        # candidates are registered and legal for this shape (c=32 divides
        # every value in _CHANNELS_PER_BLOCK).
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_batchnorm(_req(1, 32, 64, 64, arch))
                self.assertEqual(r.spec.channels_per_block, 1)

    def test_channels_per_block_reachable_via_spec_id_and_shrinks_grid(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(1, 32, 64, 64, arch, spec_id="batchnorm_b256_v4_cpb4")
                r = dispatch_batchnorm(req)
                self.assertEqual(r.spec.channels_per_block, 4)
                self.assertEqual(r.grid, (32 // 4, 1, 1))

    def test_channels_per_block_appears_in_sweep_space(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(1, 32, 64, 64, arch)
                specs = batchnorm_sweep_space(req)
                self.assertTrue(any(s.channels_per_block > 1 for s in specs))

    def test_sweep_space_enumerates_valid_block_vec_combos(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(1, 32, 64, 64, arch)
                winner = dispatch_batchnorm(req)
                specs = batchnorm_sweep_space(req)
                results = dispatch_batchnorm_all(req)
                self.assertGreater(len(specs), 1)
                self.assertEqual(len(results), len(specs))
                self.assertEqual(results[0].candidate.name, winner.candidate.name)


if __name__ == "__main__":
    unittest.main()
