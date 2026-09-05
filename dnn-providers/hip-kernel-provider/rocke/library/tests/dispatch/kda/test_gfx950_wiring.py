# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wiring for the gfx950 chunkwise KDA candidates.

The kernels had specs, validators, builders, grid helpers and signatures long
before anything referenced them from dispatch. These tests pin what registering
them did -- and, just as much, what it deliberately did not do: the split path
is reachable only when the caller names it, because the fused/split crossover
has not been measured at enough shapes to encode a threshold.

CPU-only. Numeric coverage lives in ``test_kda_chunkwise_gfx950_numeric.py``.
"""

from __future__ import annotations

import unittest

from dispatch.kda import (
    KDA_REGISTRY,
    KdaRequest,
    dispatch_kda,
    kda_candidates,
    kda_sweep_space,
)

_FUSED = "kda_gfx950_chunk_fused"
_PREP = "kda_gfx950_chunk_prep"
_SCAN = "kda_gfx950_chunk_scan"
_ALL = (_FUSED, _PREP, _SCAN)

_BATCH, _HEADS, _SEQLEN, _CHUNK = 2, 8, 1024, 32


def _req(**kw) -> KdaRequest:
    base = dict(
        batch=_BATCH,
        num_heads=_HEADS,
        seqlen=_SEQLEN,
        arch="gfx950",
        head_k=128,
        head_v=128,
        chunk_size=_CHUNK,
    )
    base.update(kw)
    return KdaRequest(**base)


def _raw_split_req(algorithm: str, **kw) -> KdaRequest:
    base = {
        "algorithm": algorithm,
        "raw_inputs": True,
        "token_major_io": True,
        "fuse_qk_l2norm": True,
        "fuse_gate": True,
        "fuse_beta_sigmoid": True,
        "has_dt_bias": True,
    }
    base.update(kw)
    return _req(**base)


def _candidate(name: str):
    return KDA_REGISTRY.get(name)


class TestRegistration(unittest.TestCase):
    def test_every_kernel_is_registered(self):
        self.assertLessEqual(set(_ALL), {c.name for c in kda_candidates()})

    def test_identity(self):
        for name, algorithm, spec_id in (
            (_FUSED, "chunk_fused", "gfx950_chunk_fused"),
            (_PREP, "chunk_prep", "gfx950_chunk_prep"),
            (_SCAN, "chunk_scan", "gfx950_chunk_scan"),
        ):
            with self.subTest(candidate=name):
                c = _candidate(name)
                self.assertEqual(c.algorithm, algorithm)
                self.assertEqual(c.spec_id, spec_id)
                self.assertEqual(c.capability.arches, ("gfx950",))

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

    def test_unspecified_chunk_uses_the_gfx950_default(self):
        self.assertEqual(dispatch_kda(_req(chunk_size=None)).spec.tile.chunk, 32)

    def test_named_algorithm_selects_a_split_half(self):
        for name, algorithm in ((_PREP, "chunk_prep"), (_SCAN, "chunk_scan")):
            with self.subTest(candidate=name):
                result = dispatch_kda(_req(algorithm=algorithm))
                self.assertEqual(result.candidate.name, name)

    def test_named_spec_id_selects_a_split_half(self):
        result = dispatch_kda(_req(spec_id="gfx950_chunk_scan"))
        self.assertEqual(result.candidate.name, _SCAN)

    def test_the_refusal_explains_the_opt_in(self):
        for name in (_PREP, _SCAN):
            with self.subTest(candidate=name):
                ok, why = _candidate(name).admits(_req())
                self.assertFalse(ok)
                self.assertIn("opt-in", why)

    def test_raw_token_major_contract_selects_the_split_halves(self):
        prep = dispatch_kda(_raw_split_req("chunk_prep", value_splits=4))
        scan = dispatch_kda(_raw_split_req("chunk_scan", value_splits=4))

        self.assertTrue(prep.spec.raw_inputs)
        self.assertTrue(prep.spec.fuse_qk_l2norm)
        self.assertTrue(prep.spec.fuse_gate)
        self.assertTrue(prep.spec.fuse_beta_sigmoid)
        self.assertTrue(prep.spec.has_dt_bias)
        self.assertEqual(prep.spec.tile.block_size, 256)
        self.assertTrue(scan.spec.token_major_io)
        self.assertEqual(scan.spec.value_splits, 4)
        self.assertEqual(scan.spec.tile.block_size, 128)

    def test_fused_kernel_rejects_the_raw_split_contract(self):
        ok, why = _candidate(_FUSED).admits(_raw_split_req("chunk_fused"))
        self.assertFalse(ok)
        self.assertIn("raw split contract", why)

    def test_sweep_space_offers_only_what_is_reachable(self):
        # auto routes to fused alone, so the sweep space is one spec -- not the
        # three it would be if the opt-in gate were advisory.
        self.assertEqual(len(kda_sweep_space(_req())), 1)


class TestArchGate(unittest.TestCase):
    def test_no_candidate_admits_an_arch_it_did_not_declare(self):
        from rocke.core.arch import known_arches

        for name in _ALL:
            for arch in known_arches():
                if arch == "gfx950":
                    continue
                with self.subTest(candidate=name, arch=arch):
                    req = _req(arch=arch, algorithm=_candidate(name).algorithm)
                    self.assertFalse(_candidate(name).admits(req)[0])

    def test_registry_serves_them_only_to_gfx950(self):
        self.assertEqual({c.name for c in KDA_REGISTRY.for_arch("gfx950")}, set(_ALL))
        self.assertFalse(set(_ALL) & {c.name for c in KDA_REGISTRY.for_arch("gfx942")})


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

    def test_an_unsupported_value_split_is_rejected(self):
        ok, why = _candidate(_SCAN).admits(_raw_split_req("chunk_scan", value_splits=3))
        self.assertFalse(ok)
        self.assertIn("value_splits", why)

    def test_raw_inputs_require_fused_preprocessing(self):
        ok, why = _candidate(_PREP).admits(
            _req(algorithm="chunk_prep", raw_inputs=True)
        )
        self.assertFalse(ok)
        self.assertIn("raw_inputs", why)

    def test_fp32_beta_dtype_requires_raw_inputs(self):
        ok, why = _candidate(_PREP).admits(
            _req(algorithm="chunk_prep", fp32_beta_dtype=True)
        )
        self.assertFalse(ok)
        self.assertIn("fp32_beta_dtype", why)

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

    def test_c16_is_admitted_by_capability_then_split_on_the_schedule(self):
        # chunk=16 is a declared chunk length, so the prefilter passes it for
        # every candidate. What separates them is the tile schedule each one
        # starts from: the fused kernel's narrower 16x16 atom tiles a 16-token
        # chunk, the split path's chunk-wide 32x32 atom does not. Pinning both
        # halves of that split is the point -- a capability that also encoded
        # the atom would have to recompute the spec to do it.
        for name in _ALL:
            with self.subTest(candidate=name):
                req = _req(chunk_size=16, algorithm=_candidate(name).algorithm)
                self.assertTrue(_candidate(name).capability.check(req)[0])

        self.assertTrue(_candidate(_FUSED).admits(_req(chunk_size=16))[0])
        for name in (_PREP, _SCAN):
            with self.subTest(candidate=name):
                req = _req(chunk_size=16, algorithm=_candidate(name).algorithm)
                ok, why = _candidate(name).admits(req)
                self.assertFalse(ok)
                self.assertIn("16", why)

    def test_a_scan_whose_tile_builder_is_unbuildable_is_refused(self):
        # head_k=32 passes every rule the scan itself has, and fails the tile
        # builder's cumsum row-group fold. Admitting the scan alone would leave
        # a kernel nothing can feed.
        req = _req(head_k=32, algorithm="chunk_scan")
        self.assertTrue(_candidate(_SCAN).capability.check(req)[0])
        ok, why = _candidate(_SCAN).admits(req)
        self.assertFalse(ok)
        self.assertIn("tile builder", why)


class TestGeometry(unittest.TestCase):
    def test_fused_grid_is_one_workgroup_per_head(self):
        # gfx950 holds a whole DV=128 state mirror, so a logical head is not
        # partitioned the way it is on gfx942.
        result = dispatch_kda(_req())
        self.assertEqual(result.grid, (_BATCH * _HEADS, 1, 1))
        self.assertEqual(result.block, (512, 1, 1))

    def test_prep_grid_is_one_workgroup_per_chunk(self):
        result = dispatch_kda(_req(algorithm="chunk_prep"))
        self.assertEqual(result.grid, (_BATCH * _HEADS * (_SEQLEN // _CHUNK), 1, 1))
        self.assertEqual(result.block, (256, 1, 1))

    def test_scan_grid_adds_value_bands_when_underfilled(self):
        scan = dispatch_kda(_req(algorithm="chunk_scan"))
        self.assertEqual(scan.spec.value_splits, 4)
        self.assertEqual(scan.grid, (_BATCH * _HEADS * 4, 1, 1))
        self.assertEqual(scan.block, (128, 1, 1))

    def test_scan_grid_stays_unsplit_when_saturated(self):
        scan = dispatch_kda(_req(batch=32, algorithm="chunk_scan"))
        self.assertEqual(scan.spec.value_splits, 1)
        self.assertEqual(scan.grid, (32 * _HEADS, 1, 1))
        self.assertEqual(scan.block, (256, 1, 1))

    def test_scan_grid_uses_two_bands_in_the_midrange(self):
        scan = dispatch_kda(_req(batch=12, num_heads=12, algorithm="chunk_scan"))
        self.assertEqual(scan.spec.value_splits, 2)
        self.assertEqual(scan.grid, (12 * 12 * 2, 1, 1))
        self.assertEqual((scan.block[0], scan.spec.scan_atom.m), (256, 16))

    def test_scan_and_prep_keep_one_materialized_tile_layout(self):
        prep = dispatch_kda(_req(algorithm="chunk_prep"))
        scan = dispatch_kda(_req(algorithm="chunk_scan"))
        self.assertEqual(scan.spec.prep, prep.spec)

    def test_scan_dispatch_reports_the_optimized_schedule(self):
        scan = dispatch_kda(_req(algorithm="chunk_scan"))
        self.assertTrue(scan.spec.prefetch_tiles)
        self.assertIn("value_splits=4", scan.explanation)
        self.assertIn("prefetch_tiles=True", scan.explanation)

    def test_the_spec_carries_the_requested_head_widths(self):
        spec = dispatch_kda(_req()).spec
        self.assertEqual(spec.head_k, 128)
        self.assertEqual(spec.head_v, 128)

    def test_each_path_keeps_its_own_tile_schedule(self):
        # The fused kernel's tuned schedule must not leak into the split scan,
        # whose LDS budget is capped for two workgroups per CU.
        fused = dispatch_kda(_req()).spec.tile
        scan = dispatch_kda(_req(algorithm="chunk_scan")).spec.tile
        self.assertEqual((fused.block_size, fused.scan_atom_m), (512, 16))
        self.assertEqual((scan.block_size, scan.scan_atom_m), (128, 16))
        self.assertEqual((fused.chunk, scan.chunk), (_CHUNK, _CHUNK))


class TestSignatures(unittest.TestCase):
    def test_each_signature_matches_its_kernel_abi(self):
        from kernels.gfx950.kda_chunkwise import (
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

    def test_raw_beta_signatures_are_distinct_and_strided(self):
        bf16 = dispatch_kda(_raw_split_req("chunk_prep"))
        fp32 = dispatch_kda(_raw_split_req("chunk_prep", fp32_beta_dtype=True))
        prep_names = [arg["name"] for arg in bf16.signature]
        bf16_types = {arg["name"]: arg["type"] for arg in bf16.signature}
        fp32_types = {arg["name"]: arg["type"] for arg in fp32.signature}

        self.assertEqual(
            prep_names[-9:],
            [
                "a_log_ptr",
                "dt_bias_ptr",
                "batch",
                "heads",
                "tseq",
                "nc",
                "beta_stride_batch",
                "beta_stride_token",
                "beta_stride_head",
            ],
        )
        self.assertEqual(bf16_types["beta_ptr"], "ptr<bf16, global>")
        self.assertEqual(fp32_types["beta_ptr"], "ptr<f32, global>")
        self.assertNotEqual(bf16.spec.kernel_name(), fp32.spec.kernel_name())
        self.assertNotEqual(bf16.kernel_id.spec_hash, fp32.kernel_id.spec_hash)


class TestBuild(unittest.TestCase):
    def test_it_builds_what_it_selects(self):
        for algorithm, needle in (
            ("chunk_fused", "fused"),
            ("chunk_prep", "prep"),
            ("chunk_scan", "scan"),
        ):
            with self.subTest(algorithm=algorithm):
                result = dispatch_kda(_req(algorithm=algorithm))
                kernel = result.build()
                self.assertIn(needle, kernel.name)
                self.assertIn("dk128", kernel.name)
                self.assertIn("dv128", kernel.name)
                self.assertEqual(kernel.name, result.spec.kernel_name())

    def test_the_built_kernel_takes_the_declared_signature(self):
        result = dispatch_kda(_req())
        names = [p.name for p in result.build().params]
        self.assertEqual(names, [a["name"] for a in result.signature])

    def test_initial_state_reaches_the_kernel_name(self):
        kernel = dispatch_kda(_req(has_initial_state=True)).build()
        self.assertIn("h0", kernel.name)


if __name__ == "__main__":
    unittest.main()
