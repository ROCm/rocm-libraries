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

    def test_n_groups_default_is_one_and_not_auto_selected(self):
        # n_groups is a default-off, inference-only lever: auto dispatch
        # must keep picking the n_groups=1 candidate even though ng>1
        # candidates are registered and legal for this shape (n=4 divides
        # every value in _N_GROUPS, training=False makes them eligible).
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_batchnorm(_req(4, 32, 64, 64, arch, training=False))
                self.assertEqual(r.spec.n_groups, 1)

    def test_n_groups_reachable_via_spec_id(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(
                    4,
                    32,
                    64,
                    64,
                    arch,
                    training=False,
                    spec_id="batchnorm_b256_v4_ng4",
                )
                r = dispatch_batchnorm(req)
                self.assertEqual(r.spec.n_groups, 4)

    def test_n_groups_appears_in_sweep_space(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(4, 32, 64, 64, arch, training=False)
                specs = batchnorm_sweep_space(req)
                self.assertTrue(any(s.n_groups > 1 for s in specs))

    def test_n_groups_supported_when_training(self):
        # n_groups > 1 was inference-only; it now also supports training
        # (see instances/common/batchnorm2d.py's n_groups docstring) --
        # pinning its spec_id for a training request must dispatch
        # successfully, not raise.
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(
                    4,
                    32,
                    64,
                    64,
                    arch,
                    training=True,
                    spec_id="batchnorm_b256_v4_ng4",
                )
                result = dispatch_batchnorm(req)
                self.assertEqual(result.spec.n_groups, 4)
                self.assertTrue(result.spec.training)

    def test_n_groups_still_default_off_under_auto_with_training(self):
        # n_groups remains a default-off, opt-in lever even though it is now
        # legal for training=True requests: unpinned ``auto`` dispatch must
        # keep picking the n_groups=1 candidate for a shape where n_groups>1
        # candidates are also valid.
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(4, 32, 64, 64, arch, training=True)
                result = dispatch_batchnorm(req)
                self.assertEqual(result.spec.n_groups, 1)

    def test_n_groups_supported_on_non_divisible_hw(self):
        # n_groups's hw_threads floor/exact-divisibility requirement was
        # relaxed (masked tail, mirroring NG==1's own tail support) so it
        # now reaches shapes no power-of-two hw_threads can tile exactly
        # (7x7=49, 14x14=196 -- real CNN shapes), not just ones that
        # happen to divide evenly.
        for arch in COVERED_ARCHES:
            for training in (True, False):
                for h, w in ((7, 7), (14, 14)):
                    with self.subTest(arch=arch, training=training, hw=(h, w)):
                        req = _req(
                            4,
                            4,
                            h,
                            w,
                            arch,
                            training=training,
                            spec_id="batchnorm_b256_v2_ng4",
                        )
                        result = dispatch_batchnorm(req)
                        self.assertEqual(result.spec.n_groups, 4)
                        self.assertEqual(result.spec.training, training)

    def test_fast_reduce_default_is_false_and_not_auto_selected(self):
        # fast_reduce is a default-off, training-only lever: auto dispatch
        # must keep picking the fast_reduce=False candidate even though the
        # fast_reduce=True candidates are registered and legal for this
        # shape (training=True makes them eligible).
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_batchnorm(_req(4, 32, 64, 64, arch, training=True))
                self.assertFalse(r.spec.fast_reduce)

    def test_fast_reduce_reachable_via_spec_id(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(
                    4,
                    32,
                    64,
                    64,
                    arch,
                    training=True,
                    spec_id="batchnorm_b256_v4_fr",
                )
                r = dispatch_batchnorm(req)
                self.assertTrue(r.spec.fast_reduce)

    def test_fast_reduce_appears_in_sweep_space(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(4, 32, 64, 64, arch, training=True)
                specs = batchnorm_sweep_space(req)
                self.assertTrue(any(s.fast_reduce for s in specs))

    def test_fast_reduce_rejected_when_inference(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(
                    4,
                    32,
                    64,
                    64,
                    arch,
                    training=False,
                    spec_id="batchnorm_b256_v4_fr",
                )
                with self.assertRaises(ValueError):
                    dispatch_batchnorm(req)

    def test_dispatchable_on_non_power_of_two_hw_real_model_shapes(self):
        # Real CNN feature maps whose H*W has no power-of-two divisor large
        # enough to satisfy the old hard n_per_block % (block_size*vec) == 0
        # requirement: 14x14=196, 24x24=576, 56x56=3136. The batchnorm2d
        # scalar tail (see rocke.instances.common.batchnorm2d) now handles
        # these directly, so dispatch must succeed and yield a non-empty
        # sweep space instead of raising/returning nothing.
        shapes = [(14, 14), (24, 24), (56, 56)]
        for arch in COVERED_ARCHES:
            for training in (True, False):
                for h, w in shapes:
                    with self.subTest(arch=arch, training=training, hw=(h, w)):
                        req = _req(1, 16, h, w, arch, training=training)
                        r = dispatch_batchnorm(req)
                        self.assertEqual(r.spec.h, h)
                        self.assertEqual(r.spec.w, w)
                        specs = batchnorm_sweep_space(req)
                        self.assertGreater(len(specs), 0)

    def test_sweep_space_enumerates_valid_block_vec_combos(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                req = _req(1, 32, 64, 64, arch)
                winner = dispatch_batchnorm(req)
                specs = batchnorm_sweep_space(req)
                results = dispatch_batchnorm_all(req)
                self.assertGreater(len(specs), 1)
                self.assertEqual(len(results), len(specs))
                # dispatch_all enumerates every eligible candidate in
                # registration order and explicitly does not rank (see its
                # docstring); dispatch_batchnorm's winner comes from a
                # tail-aware ranker (see _default_ranker) that can reorder
                # same-tier candidates relative to that raw enumeration, so
                # the only invariant to check here is membership, not
                # positional equality.
                self.assertIn(
                    winner.candidate.name, {r.candidate.name for r in results}
                )


if __name__ == "__main__":
    unittest.main()
