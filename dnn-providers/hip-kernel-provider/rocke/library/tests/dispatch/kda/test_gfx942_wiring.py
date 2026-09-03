# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wiring for the gfx942 chunkwise KDA candidates.

The kernels had specs, validators, builders, grid helpers and signatures long
before anything referenced them from dispatch. These tests pin what registering
them did -- and, just as much, what it deliberately did not do: the split path
is reachable only when the caller names it, because no measured fused/split
crossover exists yet.

CPU-only. Numeric coverage lives in ``test_kda_chunkwise_gfx942_numeric.py``.
"""

from __future__ import annotations

import unittest

from dispatch.kda import (
    KDA_PARTITION_HEAD_V,
    KDA_REGISTRY,
    KdaRequest,
    dispatch_kda,
    kda_candidates,
    kda_sweep_space,
)

_FUSED = "kda_gfx942_chunk_fused"
_PREP = "kda_gfx942_chunk_prep"
_SCAN = "kda_gfx942_chunk_scan"
_ALL = (_FUSED, _PREP, _SCAN)


def _req(**kw) -> KdaRequest:
    base = dict(
        batch=2,
        num_heads=8,
        seqlen=1024,
        arch="gfx942",
        head_k=128,
        head_v=128,
        chunk_size=16,
    )
    base.update(kw)
    return KdaRequest(**base)


def _candidate(name: str):
    return KDA_REGISTRY.get(name)


class TestRegistration(unittest.TestCase):
    def test_every_kernel_is_registered(self):
        self.assertEqual({c.name for c in kda_candidates()}, set(_ALL))

    def test_identity(self):
        for name, algorithm, spec_id in (
            (_FUSED, "chunk_fused", "gfx942_chunk_fused"),
            (_PREP, "chunk_prep", "gfx942_chunk_prep"),
            (_SCAN, "chunk_scan", "gfx942_chunk_scan"),
        ):
            with self.subTest(candidate=name):
                c = _candidate(name)
                self.assertEqual(c.algorithm, algorithm)
                self.assertEqual(c.spec_id, spec_id)
                self.assertEqual(c.capability.arches, ("gfx942",))

    def test_every_candidate_declares_a_build(self):
        # The whole point of the registration: the three builders were
        # unreachable from dispatch before these candidates existed.
        for name in _ALL:
            with self.subTest(candidate=name):
                self.assertIsNotNone(_candidate(name).build)

    def test_the_family_refuses_an_unbuildable_candidate(self):
        self.assertTrue(KDA_REGISTRY.require_build)

    def test_coverage_is_queryable_without_a_request(self):
        coverage = KDA_REGISTRY.coverage()
        self.assertEqual(coverage["family"], "kda_chunkwise")
        self.assertTrue(all(c["buildable"] for c in coverage["candidates"]))


class TestRouting(unittest.TestCase):
    def test_default_routing_is_the_fused_kernel(self):
        self.assertEqual(dispatch_kda(_req()).candidate.name, _FUSED)

    def test_named_algorithm_selects_a_split_half(self):
        for name, algorithm in ((_PREP, "chunk_prep"), (_SCAN, "chunk_scan")):
            with self.subTest(candidate=name):
                result = dispatch_kda(_req(algorithm=algorithm))
                self.assertEqual(result.candidate.name, name)

    def test_named_spec_id_selects_a_split_half(self):
        result = dispatch_kda(_req(spec_id="gfx942_chunk_scan"))
        self.assertEqual(result.candidate.name, _SCAN)

    def test_the_refusal_explains_the_opt_in(self):
        for name in (_PREP, _SCAN):
            with self.subTest(candidate=name):
                ok, why = _candidate(name).admits(_req())
                self.assertFalse(ok)
                self.assertIn("opt-in", why)

    def test_sweep_space_offers_only_what_is_reachable(self):
        # auto routes to fused alone, so the sweep space is one spec -- not the
        # three it would be if the opt-in gate were advisory.
        self.assertEqual(len(kda_sweep_space(_req())), 1)


class TestArchGate(unittest.TestCase):
    def test_no_candidate_admits_an_arch_it_did_not_declare(self):
        from rocke.core.arch import known_arches

        for name in _ALL:
            for arch in known_arches():
                if arch == "gfx942":
                    continue
                with self.subTest(candidate=name, arch=arch):
                    req = _req(arch=arch, algorithm=_candidate(name).algorithm)
                    self.assertFalse(_candidate(name).admits(req)[0])

    def test_registry_serves_them_only_to_gfx942(self):
        self.assertEqual({c.name for c in KDA_REGISTRY.for_arch("gfx942")}, set(_ALL))
        self.assertEqual(KDA_REGISTRY.for_arch("gfx950"), ())


class TestCapabilityGates(unittest.TestCase):
    """Data gates, checked before any spec is constructed."""

    def test_fp16_is_rejected(self):
        ok, why = _candidate(_FUSED).admits(_req(dtype="fp16"))
        self.assertFalse(ok)
        self.assertIn("fp16", why)

    def test_a_seqlen_that_does_not_tile_the_chunk_is_rejected(self):
        # No varlen path here, so a ragged length must be padded by the caller
        # rather than silently truncated to whole chunks.
        ok, why = _candidate(_FUSED).admits(_req(seqlen=1000))
        self.assertFalse(ok)
        self.assertIn("num_chunks", why)

    def test_an_unsupported_chunk_length_is_rejected(self):
        self.assertFalse(_candidate(_FUSED).admits(_req(chunk_size=64))[0])

    def test_head_v_must_partition_exactly(self):
        # 96 channels cannot be cut into 64-channel workgroups, and admitting
        # it would move the failure to launch.
        self.assertFalse(_candidate(_FUSED).admits(_req(head_v=96))[0])

    def test_the_split_path_can_serve_a_problem_that_wants_a_final_state(self):
        # Regression: withholding the state features from the tile builder made
        # the whole split path unreachable for the default request, since
        # store_final_state defaults on. The flags describe the problem; both
        # halves take part in serving it.
        for name in (_PREP, _SCAN):
            with self.subTest(candidate=name):
                req = _req(
                    algorithm=_candidate(name).algorithm,
                    has_initial_state=True,
                    store_final_state=True,
                )
                self.assertTrue(_candidate(name).admits(req)[0])

    def test_the_state_flags_do_not_reach_the_tile_builder(self):
        # KdaChunkPrepSpec has no state fields: the tile phase is
        # state-independent and the scan half is what applies them.
        spec = dispatch_kda(_req(algorithm="chunk_prep", has_initial_state=True)).spec
        self.assertFalse(hasattr(spec, "has_initial_state"))
        self.assertTrue(
            dispatch_kda(
                _req(algorithm="chunk_scan", has_initial_state=True)
            ).spec.has_initial_state
        )


class TestResidualGates(unittest.TestCase):
    """Gates the validators compute from the spec, not from the request."""

    def test_c32_is_admitted_by_capability_then_refused_on_lds(self):
        # chunk=32 is a declared chunk length, so the prefilter passes it; the
        # gfx942 LDS budget is what rejects it. Pinning both halves of that
        # split is the point -- a capability that also encoded LDS would have
        # to recompute the spec to do it.
        candidate = _candidate(_FUSED)
        req = _req(chunk_size=32)
        self.assertTrue(candidate.capability.check(req)[0])
        ok, why = candidate.admits(req)
        self.assertFalse(ok)
        self.assertIn("LDS", why)


class TestGeometry(unittest.TestCase):
    def test_fused_grid_is_one_workgroup_per_partitioned_head(self):
        result = dispatch_kda(_req())
        parts = 128 // KDA_PARTITION_HEAD_V
        self.assertEqual(result.grid, (2 * 8 * parts, 1, 1))
        self.assertEqual(result.block, (256, 1, 1))

    def test_prep_grid_is_one_workgroup_per_chunk(self):
        result = dispatch_kda(_req(algorithm="chunk_prep"))
        parts = 128 // KDA_PARTITION_HEAD_V
        self.assertEqual(result.grid, (2 * 8 * parts * (1024 // 16), 1, 1))

    def test_scan_grid_matches_the_fused_grid(self):
        fused = dispatch_kda(_req()).grid
        scan = dispatch_kda(_req(algorithm="chunk_scan")).grid
        self.assertEqual(fused, scan)

    def test_the_spec_carries_the_partition_not_the_logical_width(self):
        # The request describes a DV=128 head; the kernel runs 64 channels.
        self.assertEqual(dispatch_kda(_req()).spec.head_v, KDA_PARTITION_HEAD_V)
        self.assertEqual(dispatch_kda(_req()).spec.head_k, 128)


class TestSignatures(unittest.TestCase):
    def test_each_signature_matches_its_kernel_abi(self):
        from kernels.gfx942.kda_chunkwise import (
            kda_chunk_fused_signature,
            kda_chunk_prep_signature,
            kda_chunk_scan_signature,
        )

        for algorithm, signature_fn in (
            ("chunk_fused", kda_chunk_fused_signature),
            ("chunk_prep", kda_chunk_prep_signature),
            ("chunk_scan", kda_chunk_scan_signature),
        ):
            with self.subTest(algorithm=algorithm):
                result = dispatch_kda(_req(algorithm=algorithm))
                expected = [a["name"] for a in signature_fn(result.spec)]
                self.assertEqual([a["name"] for a in result.signature], expected)


class TestBuild(unittest.TestCase):
    def test_it_builds_what_it_selects(self):
        for algorithm, needle in (
            ("chunk_fused", "fused"),
            ("chunk_prep", "prep"),
            ("chunk_scan", "scan"),
        ):
            with self.subTest(algorithm=algorithm):
                kernel = dispatch_kda(_req(algorithm=algorithm)).build()
                self.assertIn(needle, kernel.name)
                self.assertIn("dk128", kernel.name)
                self.assertIn("dv64", kernel.name)

    def test_the_built_kernel_takes_the_declared_signature(self):
        result = dispatch_kda(_req())
        names = [p.name for p in result.build().params]
        self.assertEqual(names, [a["name"] for a in result.signature])

    def test_initial_state_reaches_the_kernel_name(self):
        kernel = dispatch_kda(_req(has_initial_state=True)).build()
        self.assertIn("h0", kernel.name)


if __name__ == "__main__":
    unittest.main()
