# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wiring for the gfx1250 WMMA FMHA candidate.

The kernel had a spec, validator, builder, and grid helper long before anything
referenced them. These tests pin what registering it did and, just as much,
what it deliberately did not do: gfx1250 prefill still routes to unified_2d
unless the caller asks for this kernel by name.
"""

from __future__ import annotations

import unittest

from dispatch.attention import (
    ATTENTION_REGISTRY,
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)

_NAME = "attention_gfx1250_wmma"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=2,
        nhead_q=16,
        nhead_k=4,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch="gfx1250",
        dtype="fp16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _opt_in(**kw) -> AttentionRequest:
    return _req(algorithm="wmma_attention_fwd", **kw)


def _candidate():
    return ATTENTION_REGISTRY.get(_NAME)


class TestRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        self.assertIn(_NAME, {c.name for c in attention_candidates()})

    def test_identity(self):
        c = _candidate()
        self.assertEqual(c.algorithm, "wmma_attention_fwd")
        self.assertEqual(c.spec_id, "gfx1250_wmma_fwd")
        self.assertEqual(c.capability.arches, ("gfx1250",))

    def test_it_declares_a_build(self):
        # The whole point of the registration: build_wmma_attention_fwd was
        # unreachable from dispatch before this candidate existed.
        self.assertIsNotNone(_candidate().build)


class TestOptIn(unittest.TestCase):
    def test_default_gfx1250_routing_is_unchanged(self):
        # Registering a kernel must not silently re-route the arch. gfx1250
        # prefill goes to unified_2d, which is the path its benchmark covers.
        self.assertEqual(
            dispatch_attention(_req()).candidate.name, "attention_unified_2d"
        )

    def test_named_algorithm_selects_it(self):
        self.assertEqual(dispatch_attention(_opt_in()).candidate.name, _NAME)

    def test_named_spec_id_selects_it(self):
        req = _req(spec_id="gfx1250_wmma_fwd")
        self.assertEqual(dispatch_attention(req).candidate.name, _NAME)

    def test_the_refusal_explains_the_opt_in(self):
        ok, why = _candidate().admits(_req())
        self.assertFalse(ok)
        self.assertIn("opt-in", why)


class TestArchGate(unittest.TestCase):
    def test_it_rejects_every_other_arch(self):
        from rocke.core.arch import known_arches

        for arch in known_arches():
            if arch == "gfx1250":
                continue
            with self.subTest(arch=arch):
                self.assertFalse(_candidate().admits(_opt_in(arch=arch))[0])

    def test_registry_serves_it_only_to_gfx1250(self):
        served = {c.name for c in ATTENTION_REGISTRY.for_arch("gfx1250")}
        self.assertIn(_NAME, served)
        self.assertNotIn(_NAME, {c.name for c in ATTENTION_REGISTRY.for_arch("gfx950")})


class TestCapabilityGates(unittest.TestCase):
    """The spec raises from __post_init__, so these must be caught earlier."""

    def test_bf16_is_rejected_without_constructing_a_spec(self):
        ok, why = _candidate().admits(_opt_in(dtype="bf16"))
        self.assertFalse(ok)
        self.assertIn("bf16", why)

    def test_head_size_must_be_a_multiple_of_the_wmma_k_tile(self):
        self.assertFalse(_candidate().admits(_opt_in(hdim_q=48, hdim_v=48))[0])

    def test_seqlen_q_must_tile_exactly(self):
        # The grid helper refuses a remainder rather than launching a partial
        # tile, so admitting such a request would move the failure to launch.
        self.assertFalse(_candidate().admits(_opt_in(seqlen_q=1000))[0])

    def test_sinks_are_not_claimed(self):
        self.assertFalse(_candidate().admits(_opt_in(use_sinks=True))[0])

    def test_sliding_window_is_not_claimed(self):
        # The spec carries a sliding_window field, but its mask_mode vocabulary
        # is "none"/"causal" and apply_attention_mask reads the window only
        # under a "sliding_window" mode this spec cannot express. Claiming the
        # feature admits the request and compiles plain causal for it.
        ok, why = _candidate().admits(_opt_in(sliding_window=256))
        self.assertFalse(ok)
        self.assertIn("sliding_window", why)

    def test_the_window_field_is_inert_in_the_kernel(self):
        # Why the gate above must be a rejection: the emitted kernel is
        # identical with and without a window, so an admitted request would get
        # unwindowed numerics and no diagnostic. If this ever stops holding,
        # the kernel grew real support and the capability should widen.
        from kernels.gfx1250.wmma_attention_fwd import (
            WmmaAttentionFwdSpec,
            build_wmma_attention_fwd,
        )
        from rocke.core.ir_serialize import canonical_equal

        def _kernel(window):
            spec = WmmaAttentionFwdSpec(
                head_size=128,
                num_query_heads=16,
                num_kv_heads=4,
                mask_mode="causal",
                sliding_window=window,
            )
            return build_wmma_attention_fwd(spec, arch="gfx1250")

        self.assertTrue(canonical_equal(_kernel(0), _kernel(256)))


class TestGeometryAndBuild(unittest.TestCase):
    def test_grid_and_block_are_real_not_deferred(self):
        # Unlike the unified paths, this kernel knows its own geometry.
        result = dispatch_attention(_opt_in())
        self.assertEqual(result.grid, (1024 // 16, 16, 2))
        self.assertEqual(result.block, (32, 1, 1))  # one wave32 per CTA

    def test_signature_matches_the_kernel_abi(self):
        signature = dispatch_attention(_opt_in()).signature
        names = [a["name"] for a in signature]
        self.assertEqual(names[:4], ["Q", "K", "V", "O"])
        self.assertEqual(names[4], "scale_log2")
        self.assertEqual(len(names), 15)

    def test_it_builds_what_it_selects(self):
        kernel = dispatch_attention(_opt_in()).build()
        self.assertIn("wmma16x16x32", kernel.name)
        self.assertIn("causal", kernel.name)

    def test_mask_mode_follows_the_request(self):
        self.assertIn("none", dispatch_attention(_opt_in(mask_type=0)).build().name)


if __name__ == "__main__":
    unittest.main()
