# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wiring for the gfx1151 transposed-QK WMMA FMHA candidate.

``kernels/gfx1151/wmma_fmha_swapqk.py`` is the production Strix Halo prefill
kernel, but dispatch had no gfx1151 entry at all -- it was reachable only
through the standalone verify scripts. These tests pin what registering it did
and, just as much, what it deliberately did not do: gfx1151 prefill still
routes to unified_2d unless the caller asks for this kernel by name.
"""

from __future__ import annotations

import unittest

from dispatch.attention import (
    ATTENTION_REGISTRY,
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)

_NAME = "attention_gfx1151_swapqk"
_ARCH = "gfx1151"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=2,
        nhead_q=16,
        nhead_k=4,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch=_ARCH,
        dtype="fp16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _opt_in(**kw) -> AttentionRequest:
    return _req(algorithm="wmma_fmha_swapqk", **kw)


def _candidate():
    return ATTENTION_REGISTRY.get(_NAME)


class TestRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        self.assertIn(_NAME, {c.name for c in attention_candidates()})

    def test_identity(self):
        c = _candidate()
        self.assertEqual(c.algorithm, "wmma_fmha_swapqk")
        self.assertEqual(c.spec_id, "gfx1151_swapqk")
        self.assertEqual(c.capability.arches, (_ARCH,))

    def test_it_declares_a_build(self):
        # The whole point of the registration: build_wmma_fmha_swapqk was
        # unreachable from dispatch before this candidate existed.
        self.assertIsNotNone(_candidate().build)


class TestOptIn(unittest.TestCase):
    def test_default_gfx1151_routing_is_unchanged(self):
        # Registering a kernel must not silently re-route the arch. gfx1151
        # prefill goes to unified_2d, which is the path its benchmark covers.
        self.assertEqual(
            dispatch_attention(_req()).candidate.name, "attention_unified_2d"
        )

    def test_named_algorithm_selects_it(self):
        self.assertEqual(dispatch_attention(_opt_in()).candidate.name, _NAME)

    def test_named_spec_id_selects_it(self):
        req = _req(spec_id="gfx1151_swapqk")
        self.assertEqual(dispatch_attention(req).candidate.name, _NAME)

    def test_the_refusal_explains_the_opt_in(self):
        ok, why = _candidate().admits(_req())
        self.assertFalse(ok)
        self.assertIn("opt-in", why)


class TestArchGate(unittest.TestCase):
    def test_it_rejects_every_other_arch(self):
        from rocke.core.arch import known_arches

        for arch in known_arches():
            if arch == _ARCH:
                continue
            with self.subTest(arch=arch):
                self.assertFalse(_candidate().admits(_opt_in(arch=arch))[0])

    def test_registry_serves_it_only_to_gfx1151(self):
        served = {c.name for c in ATTENTION_REGISTRY.for_arch(_ARCH)}
        self.assertIn(_NAME, served)
        self.assertNotIn(
            _NAME, {c.name for c in ATTENTION_REGISTRY.for_arch("gfx1250")}
        )


class TestCapabilityGates(unittest.TestCase):
    """``is_valid_spec`` raises for most of these, so they must be caught here."""

    def test_bf16_is_rejected(self):
        # dtype_ir is hardcoded F16 and SwapQKCfg carries no dtype field, so a
        # bf16 request would compile fp16 and return garbage.
        ok, why = _candidate().admits(_opt_in(dtype="bf16"))
        self.assertFalse(ok)
        self.assertIn("bf16", why)

    def test_head_size_must_pair_the_dual_gather_subtiles(self):
        # dual_gather pairs adjacent d-subtiles, so head_size % 32 != 0.
        self.assertFalse(_candidate().admits(_opt_in(hdim_q=48, hdim_v=48))[0])

    def test_seqlen_q_must_tile_exactly(self):
        # swapqk_grid refuses a remainder rather than launching a partial tile,
        # so admitting such a request would move the failure to launch.
        self.assertFalse(_candidate().admits(_opt_in(seqlen_q=1000))[0])

    def test_seqlen_k_must_tile_exactly(self):
        # The kv loop bound is seqlen_k / block_n by integer division, so a
        # remainder is silently dropped -- wrong numbers, no diagnostic.
        self.assertFalse(_candidate().admits(_opt_in(seqlen_k=1000))[0])

    def test_gqa_grouping_must_divide(self):
        self.assertFalse(_candidate().admits(_opt_in(nhead_q=16, nhead_k=5))[0])

    def test_sinks_are_not_claimed(self):
        self.assertFalse(_candidate().admits(_opt_in(use_sinks=True))[0])

    def test_sliding_window_is_not_claimed(self):
        # SwapQKCfg has no window field at all; claiming the feature would admit
        # the request and compile plain causal for it.
        ok, why = _candidate().admits(_opt_in(sliding_window=256))
        self.assertFalse(ok)
        self.assertIn("sliding_window", why)


class TestGeometryAndBuild(unittest.TestCase):
    def test_grid_and_block_are_real_not_deferred(self):
        # Unlike the unified paths, this kernel knows its own geometry.
        spec = dispatch_attention(_opt_in()).spec
        result = dispatch_attention(_opt_in())
        self.assertEqual(result.grid, (1024 // spec.q_rows_per_cta, 16, 2))
        self.assertEqual(result.block, (spec.block_size, 1, 1))

    def test_signature_matches_the_kernel_abi(self):
        signature = dispatch_attention(_opt_in()).signature
        names = [a["name"] for a in signature]
        self.assertEqual(names[:4], ["Q", "K", "V", "O"])
        self.assertEqual(names[4], "scale_log2")
        self.assertEqual(len(names), 15)

    def test_it_builds_what_it_selects(self):
        kernel = dispatch_attention(_opt_in()).build()
        self.assertIn("wmma_fmha_swapqk", kernel.name)
        self.assertIn("causal", kernel.name)

    def test_mask_mode_follows_the_request(self):
        self.assertIn("none", dispatch_attention(_opt_in(mask_type=0)).build().name)

    def test_the_selected_spec_carries_the_request_shape(self):
        spec = dispatch_attention(_opt_in()).spec
        self.assertEqual(spec.head_size, 128)
        self.assertEqual(spec.num_query_heads, 16)
        self.assertEqual(spec.kv_heads, 4)


class TestVLayoutContract(unittest.TestCase):
    def test_the_selected_spec_asks_for_a_transposed_v(self):
        # A caller contract, not a dispatcher guarantee: select_spec hands back
        # the kernel's own SwapQKCfg, whose v_transposed default makes V a
        # [B, Hk, D, Sk] tensor. bind_swapqk_torch shape-checks it so a
        # row-major V fails loudly instead of returning plausible garbage.
        self.assertTrue(dispatch_attention(_opt_in()).spec.v_transposed)


if __name__ == "__main__":
    unittest.main()
