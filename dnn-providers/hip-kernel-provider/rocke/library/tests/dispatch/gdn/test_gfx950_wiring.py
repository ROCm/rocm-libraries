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
    request_errors,
)
from dispatch.gdn.gfx950 import (
    ARCH,
    TUNED_SPEC_IDS,
    tile_for_batch,
    tile_for_work,
)
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
        for batch, expected in (
            (4, (4, 16, 8)),
            (5, (2, 8, 2)),
            (32, (2, 8, 2)),
            (33, (1, 8, 1)),
            (128, (1, 8, 1)),
            (129, (8, 16, 1)),
        ):
            with self.subTest(batch=batch):
                spec = dispatch_gdn_decode(_req(batch)).spec
                self.assertEqual(_TILE(spec), expected)
                self.assertEqual(_TILE(spec), tile_for_batch(batch))

    def test_selected_spec_is_always_buildable(self):
        for batch in (1, 4, 5, 16, 33, 64, 129, 256, 8192):
            with self.subTest(batch=batch):
                ok, why = is_valid_spec(
                    dispatch_gdn_decode(_req(batch)).spec, arch=ARCH
                )
                self.assertTrue(ok, why)

    def test_supported_geometry_falls_back_when_tuned_tile_is_invalid(self):
        # batch 1's tuned tile is b4 (warp_threads_k=16 -> warp_tile_k=128), which
        # is invalid for head_k_dim=64; b32 (warp_tile_k=64) is a valid fallback,
        # so a kernel-supported request must still dispatch, not fail.
        result = dispatch_gdn_decode(_req(1, head_k_dim=64))
        ok, why = is_valid_spec(result.spec, arch=ARCH)
        self.assertTrue(ok, why)
        self.assertEqual(result.spec.head_k_dim, 64)
        self.assertNotEqual(result.candidate.spec_id, "b4")  # fell off the tuned tile


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

    def test_kda_d128_is_admitted(self):
        result = dispatch_gdn_decode(
            _req(
                1,
                gate_kind="kda",
                num_k_heads=32,
                num_v_heads=32,
                head_k_dim=128,
                head_v_dim=128,
            )
        )
        self.assertEqual(result.spec.gate_kind, "kda")
        self.assertEqual((result.spec.head_k_dim, result.spec.head_v_dim), (128, 128))

    def test_kda_non_d128_is_loudly_scoped_out(self):
        for head_k_dim, head_v_dim in ((64, 128), (128, 64)):
            with self.subTest(head_k_dim=head_k_dim, head_v_dim=head_v_dim):
                with self.assertRaises(ValueError) as ctx:
                    dispatch_gdn_decode(
                        _req(
                            1,
                            gate_kind="kda",
                            num_k_heads=32,
                            num_v_heads=32,
                            head_k_dim=head_k_dim,
                            head_v_dim=head_v_dim,
                        )
                    )
                self.assertIn("NOT_YET_IMPLEMENTED", str(ctx.exception))
                self.assertIn("128", str(ctx.exception))

    def test_gdn_d64_remains_supported(self):
        result = dispatch_gdn_decode(_req(1, head_k_dim=64))
        self.assertEqual(result.spec.gate_kind, "gdn")
        self.assertEqual(result.spec.head_k_dim, 64)


class TestSpecIdPin(unittest.TestCase):
    def test_pin_overrides_the_tuning_table(self):
        # A tuner must be able to force a non-default tile, otherwise the tuned
        # table could never be re-measured or challenged.
        result = dispatch_gdn_decode(_req(256, spec_id="b4"))
        self.assertEqual(result.candidate.spec_id, "b4")
        self.assertEqual(_TILE(result.spec), (4, 16, 8))

    def test_every_pin_is_reachable_with_its_own_gate_kind(self):
        # Each tuned tile belongs to exactly one gate kind's table, so a pin is
        # reachable from a request of that kind and only that kind.
        for spec_id in TUNED_SPEC_IDS:
            gate_kind = "kda" if spec_id.startswith("kda_") else "gdn"
            with self.subTest(spec_id=spec_id, gate_kind=gate_kind):
                got = dispatch_gdn_decode(
                    GdnDecodeRequest(
                        batch=64, arch=ARCH, spec_id=spec_id, gate_kind=gate_kind
                    )
                )
                self.assertEqual(got.candidate.spec_id, spec_id)
                self.assertEqual(got.spec.gate_kind, gate_kind)

    def test_a_pin_cannot_cross_gate_kinds(self):
        # Serving a KDA pin to a GDN request would hand it a tile tuned for a
        # different kernel. That must fail loudly, not silently fall back.
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(
                GdnDecodeRequest(
                    batch=64, arch=ARCH, spec_id="kda_w128", gate_kind="gdn"
                )
            )

    def test_algorithm_pin_is_honoured_and_an_unknown_one_is_rejected(self):
        # The `algorithm` pin is a separate selector from `spec_id` above, and
        # until this test nothing exercised it: a tuner re-measuring the table,
        # or anyone bisecting a routing regression, forces the algorithm rather
        # than the tile. The family ships one algorithm today, so the case that
        # would silently rot is the REJECTION -- a pin nobody serves must fail
        # loudly instead of falling through to the tuned default.
        got = dispatch_gdn_decode(_req(64, algorithm="warp_tiled"))
        self.assertEqual(got.candidate.algorithm, "warp_tiled")
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(64, algorithm="no_such_algorithm"))
        self.assertIn("algorithm", str(ctx.exception))


class TestRegistryContract(unittest.TestCase):
    def test_the_family_refuses_an_unbuildable_candidate(self):
        """`require_build=True` is what stops a candidate being selectable but
        not compilable -- it fails at registration instead of at launch. The
        sibling family asserts this (tests/dispatch/kda); without the assertion
        the flag could be dropped and nothing would go red."""
        self.assertTrue(GDN_REGISTRY.require_build)
        for candidate in gdn_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertIsNotNone(candidate.build)


class TestDtypeCoverage(unittest.TestCase):
    def test_kernel_and_capability_name_the_same_dtypes(self):
        """The dtype set is declared once by the kernel and re-exported by
        dispatch. This pins the CONTENT as well as the sharing: narrowing the
        tuple would otherwise silently shrink coverage -- every test still
        passes, there are just fewer of them -- which is the quiet direction
        the re-export was meant to prevent."""
        from kernels.gfx950.gdn_decode import GDN_DTYPES

        self.assertEqual(set(GDN_DTYPES), {"bf16", "f16"})
        for candidate in gdn_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertEqual(set(candidate.capability.dtypes), set(GDN_DTYPES))
        for dtype in GDN_DTYPES:
            with self.subTest(dtype=dtype):
                got = dispatch_gdn_decode(_req(16, dtype=dtype))
                self.assertEqual(got.spec.dtype, dtype)


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
            self.assertNotIn(
                kid.compile_key,
                seen,
                f"batch {batch} collides with batch {seen.get(kid.compile_key)}",
            )
            seen[kid.compile_key] = batch

    def test_spec_hash_covers_the_tile(self):
        from rocke.dispatch.core import stable_json_hash

        a = dispatch_gdn_decode(_req(1)).spec
        b = dispatch_gdn_decode(_req(256)).spec
        self.assertNotEqual(
            stable_json_hash(asdict(a), n=16), stable_json_hash(asdict(b), n=16)
        )


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


class TestGateKindWiring(unittest.TestCase):
    """The request carries gate_kind through to the spec, GDN by default."""

    def test_request_defaults_to_the_gdn_gate(self):
        self.assertEqual(_req(1).gate_kind, "gdn")
        self.assertEqual(dispatch_gdn_decode(_req(1)).spec.gate_kind, "gdn")

    def test_kda_request_selects_a_kda_spec(self):
        req = GdnDecodeRequest(batch=4, arch=ARCH, gate_kind="kda")
        spec = dispatch_gdn_decode(req).spec
        self.assertEqual(spec.gate_kind, "kda")
        self.assertIn("kda", spec.kernel_name())

    def test_gate_kind_reaches_the_compile_key(self):
        # Two requests differing only in gate_kind select different kernels, so
        # they must not collapse onto one compile-cache entry.
        gdn = dispatch_gdn_decode(GdnDecodeRequest(batch=4, arch=ARCH))
        kda = dispatch_gdn_decode(GdnDecodeRequest(batch=4, arch=ARCH, gate_kind="kda"))
        self.assertNotEqual(gdn.kernel_id.compile_key, kda.kernel_id.compile_key)

    def test_unknown_gate_kind_is_rejected(self):
        errors = request_errors(GdnDecodeRequest(batch=1, arch=ARCH, gate_kind="mamba"))
        self.assertTrue(any("gate_kind" in e for e in errors), errors)


class TestWorkKeyedTable(unittest.TestCase):
    """The new KDA table is keyed on work = batch * num_v_heads."""

    def test_equal_work_selects_the_same_kda_tile(self):
        self.assertEqual(tile_for_work(8 * 32, "kda"), tile_for_work(32 * 8, "kda"))
        self.assertEqual(tile_for_work(1 * 32, "kda"), tile_for_work(4 * 8, "kda"))

    def test_kda_dispatch_uses_work_not_batch(self):
        full = dispatch_gdn_decode(
            GdnDecodeRequest(
                batch=4,
                arch=ARCH,
                gate_kind="kda",
                num_k_heads=32,
                num_v_heads=32,
            )
        ).spec
        sharded = dispatch_gdn_decode(
            GdnDecodeRequest(
                batch=4,
                arch=ARCH,
                gate_kind="kda",
                num_k_heads=8,
                num_v_heads=8,
            )
        ).spec
        self.assertEqual(_TILE(full), tile_for_work(4 * 32, "kda"))
        self.assertEqual(_TILE(sharded), tile_for_work(4 * 8, "kda"))

    def test_tile_for_work_agrees_with_kda_dispatch(self):
        for batch in (1, 8, 32, 128):
            with self.subTest(batch=batch):
                spec = dispatch_gdn_decode(
                    GdnDecodeRequest(
                        batch=batch,
                        arch=ARCH,
                        gate_kind="kda",
                        num_k_heads=32,
                        num_v_heads=32,
                    )
                ).spec
                self.assertEqual(
                    _TILE(spec),
                    tile_for_work(batch * spec.num_v_heads, "kda"),
                )

    def test_kda_table_is_total_over_work(self):
        for work in (1, 4, 5, 128, 129, 4096, 4097, 10**6):
            self.assertIsNotNone(tile_for_work(work, "kda"))


class TestGdnSelectionIsFrozen(unittest.TestCase):
    """GDN stays batch-keyed; only the new KDA mode is work-keyed.

    Re-keying GDN on ``batch * num_v_heads`` reroutes every sharded-head
    deployment even though this PR measured only the new KDA gate. Pin the
    original GDN selector across head counts, not only the Hv=32 case where the
    old and new keys happen to be algebraically equivalent.
    """

    # The original shipped GDN table, keyed on batch.
    _ORIGINAL = (
        (4, (4, 16, 8)),
        (32, (2, 8, 2)),
        (128, (1, 8, 1)),
        (None, (8, 16, 1)),
    )

    def _original_tile(self, batch):
        for max_batch, tile in self._ORIGINAL:
            if max_batch is None or batch <= max_batch:
                return tile
        raise AssertionError("unreachable")

    def test_gdn_selection_matches_original_table_across_head_counts(self):
        for num_v_heads in (4, 8, 16, 32, 64):
            for batch in (1, 4, 5, 16, 32, 33, 64, 128, 129, 256):
                with self.subTest(num_v_heads=num_v_heads, batch=batch):
                    result = dispatch_gdn_decode(
                        GdnDecodeRequest(
                            batch=batch,
                            arch=ARCH,
                            num_k_heads=max(1, num_v_heads // 2),
                            num_v_heads=num_v_heads,
                        )
                    )
                    self.assertEqual(_TILE(result.spec), self._original_tile(batch))
                    self.assertEqual(tile_for_batch(batch), self._original_tile(batch))
                    self.assertEqual(result.spec.gate_kind, "gdn")

    def test_kda_tuning_cannot_reach_the_gdn_table(self):
        from dispatch.gdn.gfx950 import _TUNED_TILES_GDN, _TUNED_TILES_KDA

        gdn_tiles = {t for _, t, _ in _TUNED_TILES_GDN}
        gdn_ids = {sid for _, _, sid in _TUNED_TILES_GDN}
        kda_ids = {sid for _, _, sid in _TUNED_TILES_KDA}

        self.assertEqual(gdn_tiles, {(4, 16, 8), (2, 8, 2), (1, 8, 1), (8, 16, 1)})
        self.assertFalse(gdn_ids & kda_ids, "spec ids must not collide")
