# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN decode dispatch wiring: the ``problem -> spec`` direction.

CPU-only by construction -- it builds specs and reads the registry, never
compiling or launching. That is deliberate: a family whose only tests need a
GPU contributes nothing on a CPU CI machine, so the selection logic is covered
here and the numeric behaviour is covered separately by the on-device test.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict

from dispatch.gdn import (
    GDN_REGISTRY,
    GdnDecodeRequest,
    dispatch_gdn_decode,
    gdn_candidates,
    gdn_sweep_space,
)
from dispatch.gdn.gfx950 import ARCH, TUNED_SPEC_IDS, tile_for_batch
from kernels.gfx950.gdn_decode import (
    gdn_decode_grid,
    gdn_decode_signature,
    is_valid_spec,
)

_TILE = lambda s: (s.num_warps, s.warp_threads_k, s.blocks_per_v_dim)  # noqa: E731


def _req(batch: int, **kw) -> GdnDecodeRequest:
    kw.setdefault("arch", ARCH)
    return GdnDecodeRequest(batch=batch, **kw)


class TestRegistration(unittest.TestCase):
    def test_every_tuned_tile_is_registered(self):
        names = {c.spec_id for c in gdn_candidates()}
        self.assertEqual(names, set(TUNED_SPEC_IDS))

    def test_registry_family_is_consistent(self):
        for cand in gdn_candidates():
            self.assertEqual(cand.family, GDN_REGISTRY.family)


class TestTunedSelection(unittest.TestCase):
    """The measured anchors must select the tile the sweep actually won with."""

    ANCHORS = {1: (4, 16, 8), 16: (2, 8, 2), 64: (1, 8, 1), 256: (8, 16, 1)}

    def test_measured_anchors_select_their_tile(self):
        for batch, tile in self.ANCHORS.items():
            with self.subTest(batch=batch):
                self.assertEqual(_TILE(dispatch_gdn_decode(_req(batch)).spec), tile)

    def test_band_edges_are_where_the_table_says(self):
        # Guards against an off-by-one that would silently mis-tune a whole band.
        for batch, expected in ((4, (4, 16, 8)), (5, (2, 8, 2)),
                                (32, (2, 8, 2)), (33, (1, 8, 1)),
                                (128, (1, 8, 1)), (129, (8, 16, 1))):
            with self.subTest(batch=batch):
                self.assertEqual(_TILE(dispatch_gdn_decode(_req(batch)).spec), expected)

    def test_tile_for_batch_agrees_with_dispatch(self):
        for batch in (1, 2, 5, 17, 63, 100, 200, 4096):
            with self.subTest(batch=batch):
                self.assertEqual(
                    _TILE(dispatch_gdn_decode(_req(batch)).spec), tile_for_batch(batch)
                )

    def test_selected_spec_is_always_buildable(self):
        for batch in (1, 4, 5, 16, 33, 64, 129, 256, 8192):
            with self.subTest(batch=batch):
                ok, why = is_valid_spec(dispatch_gdn_decode(_req(batch)).spec, arch=ARCH)
                self.assertTrue(ok, why)


class TestRequestRejection(unittest.TestCase):
    def test_other_arch_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(8, arch="gfx942"))
        self.assertIn("gfx942", str(ctx.exception))

    def test_head_ratio_must_divide(self):
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(8, num_v_heads=33))
        self.assertIn("multiple", str(ctx.exception))

    def test_non_positive_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(_req(0))

    def test_unsupported_dtype_is_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(_req(8, dtype="fp8"))


class TestSpecIdPin(unittest.TestCase):
    def test_pin_overrides_the_tuning_table(self):
        # A tuner must be able to force a non-default tile, otherwise the tuned
        # table could never be re-measured or challenged.
        result = dispatch_gdn_decode(_req(256, spec_id="b4"))
        self.assertEqual(result.candidate.spec_id, "b4")
        self.assertEqual(_TILE(result.spec), (4, 16, 8))

    def test_every_pin_is_reachable_at_any_batch(self):
        for spec_id in TUNED_SPEC_IDS:
            with self.subTest(spec_id=spec_id):
                got = dispatch_gdn_decode(_req(64, spec_id=spec_id))
                self.assertEqual(got.candidate.spec_id, spec_id)


class TestLaunchGeometry(unittest.TestCase):
    def test_grid_and_block_track_the_selected_spec(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                got = dispatch_gdn_decode(_req(batch))
                self.assertEqual(got.grid, gdn_decode_grid(batch, got.spec))
                self.assertEqual(got.block, (got.spec.block_size, 1, 1))

    def test_dtype_aliases_normalize(self):
        a = dispatch_gdn_decode(_req(16, dtype="bfloat16"))
        b = dispatch_gdn_decode(_req(16, dtype="bf16"))
        self.assertEqual(a.spec.kernel_name(), b.spec.kernel_name())


class TestKernelIdentity(unittest.TestCase):
    def test_same_request_gives_a_stable_cache_key(self):
        a = dispatch_gdn_decode(_req(16)).kernel_id
        b = dispatch_gdn_decode(_req(16)).kernel_id
        self.assertEqual(a.spec_hash, b.spec_hash)
        self.assertEqual(a.compile_key, b.compile_key)

    def test_different_tiles_do_not_share_a_cache_key(self):
        # Two batches in different bands must not collide, or one would run the
        # other's compiled kernel -- the same failure mode the kernel name guards.
        seen = {}
        for batch in (1, 16, 64, 256):
            kid = dispatch_gdn_decode(_req(batch)).kernel_id
            self.assertNotIn(kid.compile_key, seen,
                             f"batch {batch} collides with batch {seen.get(kid.compile_key)}")
            seen[kid.compile_key] = batch

    def test_spec_hash_covers_the_tile(self):
        from rocke.dispatch.core import stable_json_hash

        a = dispatch_gdn_decode(_req(1)).spec
        b = dispatch_gdn_decode(_req(256)).spec
        self.assertNotEqual(stable_json_hash(asdict(a), n=16),
                            stable_json_hash(asdict(b), n=16))


class TestSweepSpace(unittest.TestCase):
    def test_sweep_space_is_non_empty_and_valid(self):
        specs = gdn_sweep_space(_req(16))
        self.assertTrue(specs)
        for spec in specs:
            ok, why = is_valid_spec(spec, arch=ARCH)
            self.assertTrue(ok, why)

    def test_sweep_space_of_a_bad_request_is_empty(self):
        self.assertEqual(gdn_sweep_space(_req(8, num_v_heads=33)), ())


class TestDispatchResultContract(unittest.TestCase):
    """The result must be sufficient to drive a launch on its own.

    A caller should not need to reach back into the kernel module for the
    signature or the grid; if the result disagrees with the spec it carries,
    kernel arguments would be packed against one layout and the kernel compiled
    against another.
    """

    def test_build_returns_the_kernel_the_spec_names(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                result = dispatch_gdn_decode(_req(batch))
                kernel = result.build()
                self.assertEqual(kernel.name, result.spec.kernel_name())

    def test_signature_matches_the_spec(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                result = dispatch_gdn_decode(_req(batch))
                self.assertEqual(
                    tuple(result.signature),
                    tuple(gdn_decode_signature(result.spec)),
                )

    def test_compile_key_names_arch_and_abi(self):
        kid = dispatch_gdn_decode(_req(16)).kernel_id
        self.assertIn(ARCH, kid.compile_key)
        self.assertIn("rocke-gdn-decode", kid.compile_key)


if __name__ == "__main__":
    unittest.main()
