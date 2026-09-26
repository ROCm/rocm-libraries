# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for gfx950 dense prefill dispatch wiring.

Covers:
  - Sliding-window pass-through from AttentionRequest to AttentionDenseSpec
  - Kernel name includes swa<W> token for sliding window
  - Ragged and sliding_window mutual exclusion constraint
  - Sinks support and capability metadata
  - SWA-sink composition (both features together)
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from dispatch.attention import (
    AttentionMaskType,
    AttentionRequest,
    attention_candidates,
    dense_spec_for_request as routed_dense_spec_for_request,
    registered_attention_combos,
)
from dispatch.attention.gfx950 import dense_spec_for_request
from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
from kernels.gfx950.attention_dense import (
    AttentionDenseSpec,
    GFX950_DENSE_LAYOUTS,
    Gfx950AttentionDenseSpec,
    attention_dense_grid,
    build_attention_dense,
    supports_attention_dense,
)


def _gfx950_dense_req(**kw) -> AttentionRequest:
    """Helper to build a gfx950 dense attention request with common defaults."""
    base = dict(
        batch=2,
        nhead_q=8,
        nhead_k=1,  # GQA-8 (common for dense D64 cohort)
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=64,
        hdim_v=64,
        arch="gfx950",
        dtype="bf16",
        algorithm="attention_dense",  # Opt-in required for dense
        mask_type=1,  # causal
    )
    base.update(kw)
    return AttentionRequest(**base)


class TestDenseSpecArchRouter(unittest.TestCase):
    def test_routes_to_concrete_gfx950_spec(self):
        spec = routed_dense_spec_for_request(_gfx950_dense_req())
        self.assertIsInstance(spec, Gfx950AttentionDenseSpec)

    def test_routes_to_concrete_gfx942_spec(self):
        spec = routed_dense_spec_for_request(_gfx950_dense_req(arch="gfx942"))
        self.assertIsInstance(spec, Gfx942AttentionDenseSpec)

    def test_requires_explicit_arch(self):
        for arch in ("", None):
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(ValueError, "requires an explicit arch"):
                    routed_dense_spec_for_request(_gfx950_dense_req(arch=arch))

    def test_rejects_arch_without_dense_factory(self):
        with self.assertRaisesRegex(ValueError, "no spec factory"):
            routed_dense_spec_for_request(_gfx950_dense_req(arch="gfx1250"))


class TestDenseWavesPerEuWiring(unittest.TestCase):
    def test_default_policy_and_explicit_override(self):
        default = dense_spec_for_request(_gfx950_dense_req())
        overridden = dense_spec_for_request(_gfx950_dense_req(dense_waves_per_eu=4))
        self.assertEqual(default.waves_per_eu, 2)
        self.assertEqual(overridden.waves_per_eu, 4)
        self.assertNotIn("wpe", default.kernel_name())
        self.assertIn("wpe4", overridden.kernel_name())
        self.assertNotEqual(default.kernel_name(), overridden.kernel_name())
        self.assertEqual(
            build_attention_dense(overridden, arch="gfx950").attrs["waves_per_eu"],
            4,
        )

    def test_invalid_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dense_waves_per_eu"):
            dense_spec_for_request(_gfx950_dense_req(dense_waves_per_eu=9))

    def _swept_waves(self, level: str, *, pin: int = 0) -> set[int]:
        req = _gfx950_dense_req(
            hdim_q=128,
            hdim_v=128,
            dense_persistent="off",
            dense_tile="default",
            dense_wide_lds_dma="off",
            dense_waves_per_eu=pin,
        )
        return {
            spec.waves_per_eu
            for _candidate, spec in registered_attention_combos(
                req,
                candidate_prefix="attention_gfx950_dense_grid_default",
                sweep_level=level,
            )
        }

    def test_production_and_full_sweeps_expand_wpe(self):
        self.assertEqual(self._swept_waves("production"), {2, 4})
        self.assertEqual(self._swept_waves("full"), {1, 2, 3, 4})

    def test_explicit_override_pins_sweep(self):
        self.assertEqual(self._swept_waves("production", pin=3), {3})
        self.assertEqual(self._swept_waves("full", pin=3), {3})


class TestDenseGqaPairWiring(unittest.TestCase):
    """Invariant-compatible shapes select a balanced GQA-local decode."""

    def test_exact_llama3_8b_prefill_auto_selects_gqa_pair(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=8192,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="on",
            dense_persist_decode="auto",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")
        self.assertIn("gqapair", spec.kernel_name())
        self.assertTrue(spec.wide_lds_dma)
        self.assertIn("wdma", spec.kernel_name())
        self.assertEqual(spec.num_persistent, 256)

    def test_explicit_gqa_pair_passes_through_dispatcher(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=8192,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="on",
            dense_persist_decode="gqa_pair",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.persist_decode, "gqa_pair")
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")
        self.assertTrue(spec.wide_lds_dma)
        self.assertIn("gqapair", spec.kernel_name())

    def test_invalid_dense_decode_is_rejected_at_dispatch(self):
        req = _gfx950_dense_req(dense_persist_decode="not-a-decode")
        with self.assertRaisesRegex(ValueError, "persist_decode"):
            dense_spec_for_request(req)

    def test_s4096_shape_selects_two_phase_pair_and_wide_dma(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=4096,
            seqlen_k=4096,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="on",
            dense_persist_decode="auto",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair_2phase")
        self.assertTrue(spec.wide_lds_dma)
        self.assertIn("gqapair2", spec.kernel_name())
        self.assertIn("wdma", spec.kernel_name())

    def test_bf16_shape_uses_invariant_compatible_fast_path(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=8192,
            hdim_q=128,
            hdim_v=128,
            dtype="bf16",
            dense_persistent="on",
            dense_persist_decode="auto",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")
        self.assertTrue(spec.wide_lds_dma)

    def test_s2048_h64_selects_two_phase_pair(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=64,
            nhead_k=8,
            seqlen_q=2048,
            seqlen_k=2048,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="on",
            dense_persist_decode="auto",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair_2phase")
        self.assertTrue(spec.wide_lds_dma)

    def test_explicit_two_phase_pair_passes_through_dispatcher(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=4096,
            seqlen_k=4096,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="on",
            dense_persist_decode="gqa_pair_2phase",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.persist_decode, "gqa_pair_2phase")
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair_2phase")
        self.assertTrue(spec.wide_lds_dma)

    def test_exact_shape_with_sinks_keeps_gqa_pair(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=8192,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="on",
            use_sinks=True,
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")
        self.assertTrue(spec.use_sinks)
        self.assertFalse(spec.wide_lds_dma)

    def test_sliding_window_falls_back_to_qb_major(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=8192,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="on",
            sliding_window=512,
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.resolved_persist_decode, "qb_major")
        self.assertFalse(spec.wide_lds_dma)

    def test_mha_falls_back_to_qb_major(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=32,
            seqlen_q=8192,
            seqlen_k=8192,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="on",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.resolved_persist_decode, "qb_major")
        self.assertTrue(spec.wide_lds_dma)
        self.assertNotIn("gqapair", spec.kernel_name())


class TestDenseBottomRightWiring(unittest.TestCase):
    @staticmethod
    def _moving_req(mask_type, **kw):
        base = dict(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=12288,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            mask_type=mask_type,
        )
        base.update(kw)
        return _gfx950_dense_req(**base)

    def test_aligned_and_ragged_moving_masks_force_nonpersistent_narrow_dma(self):
        shapes = (
            ("aligned", 8192, 12288, False),
            ("ragged", 8180, 12270, True),
        )
        for mask_type in (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2):
            for name, sq, sk, ragged in shapes:
                with self.subTest(mask_type=mask_type, shape=name):
                    spec = dense_spec_for_request(
                        self._moving_req(
                            mask_type,
                            seqlen_q=sq,
                            seqlen_k=sk,
                            dense_persistent="auto",
                        )
                    )
                    self.assertTrue(spec.causal)
                    self.assertTrue(spec.causal_bottom_right)
                    self.assertEqual(spec.ragged, ragged)
                    self.assertFalse(spec.persistent)
                    self.assertFalse(spec.wide_lds_dma)
                    self.assertIn("br", spec.kernel_name().split("_"))
                    self.assertNotIn("persist", spec.kernel_name())
                    self.assertNotIn("wdma", spec.kernel_name())

    def test_explicit_persistent_on_rejects_moving_bottom_right(self):
        for mask_type in (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2):
            with self.subTest(mask_type=mask_type):
                with self.assertRaisesRegex(ValueError, "bottom-right"):
                    dense_spec_for_request(
                        self._moving_req(mask_type, dense_persistent="on")
                    )

    def test_explicit_persistent_off_stays_off_for_moving_bottom_right(self):
        for mask_type in (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2):
            with self.subTest(mask_type=mask_type):
                spec = dense_spec_for_request(
                    self._moving_req(mask_type, dense_persistent="off")
                )
                self.assertFalse(spec.persistent)
                self.assertFalse(spec.wide_lds_dma)
                self.assertTrue(spec.causal_bottom_right)

    def test_equal_length_bottom_right_preserves_gqa_pair_and_wide_dma(self):
        common = dict(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=8192,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            dense_persistent="auto",
        )
        mask_pairs = (
            (AttentionMaskType.TOP_LEFT_CAUSAL, 2),
            (1, AttentionMaskType.BOTTOM_RIGHT_CAUSAL),
        )
        for top_left, bottom_right in mask_pairs:
            with self.subTest(top_left=top_left, bottom_right=bottom_right):
                top_left_spec = dense_spec_for_request(
                    _gfx950_dense_req(mask_type=top_left, **common)
                )
                bottom_right_spec = dense_spec_for_request(
                    _gfx950_dense_req(mask_type=bottom_right, **common)
                )
                self.assertEqual(bottom_right_spec, top_left_spec)
                self.assertFalse(bottom_right_spec.causal_bottom_right)
                self.assertTrue(bottom_right_spec.persistent)
                self.assertEqual(bottom_right_spec.resolved_persist_decode, "gqa_pair")
                self.assertTrue(bottom_right_spec.wide_lds_dma)


class TestDenseGeometrySpec(unittest.TestCase):
    """Tile geometry is explicit, validated, and kernel-identity-safe."""

    @staticmethod
    def _spec(**kw) -> AttentionDenseSpec:
        base = dict(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="fp16",
        )
        base.update(kw)
        return AttentionDenseSpec(**base)

    def test_dispatch_captures_named_default_geometry(self):
        spec = dense_spec_for_request(
            _gfx950_dense_req(
                batch=1,
                nhead_q=32,
                nhead_k=8,
                hdim_q=128,
                hdim_v=128,
                dtype="fp16",
            )
        )
        defaults = DENSE_TILE_GEOMETRIES["default"]
        layout = GFX950_DENSE_LAYOUTS["default"]
        self.assertEqual(spec.block_m, defaults["block_m"])
        self.assertEqual(spec.block_n, defaults["block_n"])
        self.assertEqual(spec.lds_v_row_pad, layout["lds_v_row_pad"])

    def test_block_m_controls_grid_and_kernel_identity(self):
        default = self._spec()
        bm128 = replace(default, block_m=128)
        self.assertEqual(attention_dense_grid(default), (2, 32, 1))
        self.assertEqual(attention_dense_grid(bm128), (4, 32, 1))
        self.assertNotEqual(default.kernel_name(), bm128.kernel_name())
        self.assertIn("bm128", bm128.kernel_name())
        ok, why = supports_attention_dense(bm128)
        self.assertTrue(ok, why)

    def test_v_row_pad_controls_kernel_identity(self):
        default = self._spec()
        vpad16 = replace(default, lds_v_row_pad=16)
        self.assertNotEqual(default.kernel_name(), vpad16.kernel_name())
        self.assertIn("vpad16", vpad16.kernel_name())

    def test_unknown_block_m_is_rejected_by_gfx950_support(self):
        spec = self._spec(seqlen_q=384, block_m=192)
        ok, why = supports_attention_dense(spec)
        self.assertFalse(ok)
        self.assertIn("block_m", why)

    def test_wide_dma_requires_default_v_row_pad(self):
        with self.assertRaisesRegex(ValueError, "K/V slab padding"):
            self._spec(
                persistent=True,
                num_persistent=16,
                persist_decode="gqa_pair",
                wide_lds_dma=True,
                lds_v_row_pad=16,
            )


class TestDenseSlidingWindowWiring(unittest.TestCase):
    """Verify sliding_window passes through dispatch to the dense spec."""

    def test_sliding_window_zero_by_default(self):
        """Without sliding_window in request, spec gets 0 (full causal)."""
        req = _gfx950_dense_req()
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.sliding_window, 0)
        # No swa token in kernel name for full causal
        self.assertNotIn("swa", spec.kernel_name())

    def test_sliding_window_passes_through_to_spec(self):
        """Request with sliding_window=128 produces spec with sliding_window=128."""
        req = _gfx950_dense_req(sliding_window=128)
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.sliding_window, 128)

    def test_sliding_window_appears_in_kernel_name(self):
        """Spec with sliding_window > 0 includes 'swa<W>' token in kernel_name."""
        req = _gfx950_dense_req(sliding_window=256)
        spec = dense_spec_for_request(req)
        self.assertIn("swa256", spec.kernel_name())

    def test_different_window_sizes(self):
        """Multiple window sizes each produce correct spec and kernel_name."""
        for window in [64, 128, 256, 512]:
            with self.subTest(window=window):
                req = _gfx950_dense_req(sliding_window=window)
                spec = dense_spec_for_request(req)
                self.assertEqual(spec.sliding_window, window)
                self.assertIn(f"swa{window}", spec.kernel_name())

    def test_sliding_window_with_persistent_mode(self):
        """Sliding window works with persistent mode (both appear in kernel_name)."""
        req = _gfx950_dense_req(
            sliding_window=128,
            seqlen_q=4096,  # Large enough to trigger persistent mode
            dense_persistent="on",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.sliding_window, 128)
        self.assertTrue(spec.persistent)
        kname = spec.kernel_name()
        self.assertIn("swa128", kname)
        self.assertIn("persist", kname)

    def test_sliding_window_with_ragged_shape_rejected_at_dispatch(self):
        """Ragged and sliding_window rejected early in _dense_spec.

        The requirement says 'explicit decision rather than letting the spec
        validator raise at dispatch time.' This test verifies _dense_spec()
        itself catches the constraint and raises a clear error.
        """
        # Create a ragged-shaped request: seqlen_q=seqlen_k, not a multiple
        # of the default block_m=256 / block_n=64 geometry.
        req = _gfx950_dense_req(
            seqlen_q=500,
            seqlen_k=500,
            sliding_window=128,  # Conflict: ragged + window
        )

        # _dense_spec should reject this at dispatch time with a clear error
        with self.assertRaises(ValueError) as cm:
            dense_spec_for_request(req)

        err_msg = str(cm.exception)
        self.assertIn("ragged", err_msg.lower())
        self.assertIn("sliding_window", err_msg.lower())

    def test_sliding_window_without_ragged_accepted(self):
        """Sliding window works fine on non-ragged shapes (block-aligned seqlens)."""
        # Non-ragged shape: seqlen_q is a multiple of default block_m=256.
        req = _gfx950_dense_req(
            seqlen_q=2048,  # 2048 % 256 == 0, no ragged
            seqlen_k=2048,
            sliding_window=256,
        )
        spec = dense_spec_for_request(req)

        # Should NOT trigger ragged mode
        self.assertFalse(spec.ragged)
        self.assertEqual(spec.sliding_window, 256)


class TestDenseCapabilitySlidingWindow(unittest.TestCase):
    """Verify dense candidate capability metadata includes sliding_window."""

    def test_sliding_window_in_supports_features(self):
        """Dense candidate declares 'sliding_window' in supports_features."""
        candidate = next(
            c for c in attention_candidates() if c.name == "attention_gfx950_dense"
        )
        self.assertIn("sliding_window", candidate.capability.supports_features)

    def test_sinks_also_in_supports_features(self):
        """Dense candidate also declares 'sinks.'"""
        candidate = next(
            c for c in attention_candidates() if c.name == "attention_gfx950_dense"
        )
        self.assertIn("sinks", candidate.capability.supports_features)

    def test_causal_in_supports_features(self):
        """Dense candidate declares 'causal' (existing feature)."""
        candidate = next(
            c for c in attention_candidates() if c.name == "attention_gfx950_dense"
        )
        self.assertIn("causal", candidate.capability.supports_features)

    def test_dispatch_selects_dense_for_sinks_request(self):
        """A non-wide-DMA dense variant admits sinks; production widedma does not."""
        req = _gfx950_dense_req(
            use_sinks=True,
            hdim_q=128,
            hdim_v=128,
            nhead_q=32,
            nhead_k=8,
        )
        names = {
            c.name
            for c in attention_candidates()
            if c.algorithm == "attention_dense" and c.admits(req)[0]
        }
        self.assertTrue(names, "A gfx950 dense variant should admit sinks requests")
        self.assertNotIn("attention_gfx950_dense", names)
        self.assertIn("attention_gfx950_dense_persist_default", names)

    def test_dispatch_selects_dense_for_sliding_window_request(self):
        """A non-wide-DMA dense variant admits SWA; production widedma does not."""
        req = _gfx950_dense_req(
            sliding_window=256,
            hdim_q=128,
            hdim_v=128,
            nhead_q=32,
            nhead_k=8,
        )
        names = {
            c.name
            for c in attention_candidates()
            if c.algorithm == "attention_dense" and c.admits(req)[0]
        }
        self.assertTrue(
            names, "A gfx950 dense variant should admit sliding_window requests"
        )
        self.assertNotIn("attention_gfx950_dense", names)
        self.assertIn("attention_gfx950_dense_persist_default", names)


class TestSWASinkComposition(unittest.TestCase):
    """Verify SWA-sink (sliding_window + sinks) composes correctly."""

    def test_swa_sink_both_flags_pass_through(self):
        """Request with both sliding_window and use_sinks produces spec with both."""
        req = _gfx950_dense_req(sliding_window=256, use_sinks=True)
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.sliding_window, 256)
        self.assertTrue(spec.use_sinks)

    def test_swa_sink_kernel_name_has_both_tokens(self):
        """Kernel name includes both 'swa<W>' and 'sinks' tokens."""
        req = _gfx950_dense_req(sliding_window=128, use_sinks=True)
        spec = dense_spec_for_request(req)
        kname = spec.kernel_name()
        self.assertIn("swa128", kname)
        self.assertIn("sinks", kname)


class TestSinksValidation(unittest.TestCase):
    """Verify run_attention_dense_torch validates sinks parameter correctly."""

    def test_sinks_rejected_when_use_sinks_false(self):
        """Providing sinks when spec.use_sinks=False raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=False,  # Sinks disabled
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape)
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Minimal mock for sinks (validation checks if it's not None)
        sinks = [0.0] * spec.num_query_heads

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,  # Should be rejected
            )

        self.assertIn("sinks provided but spec.use_sinks is False", str(cm.exception))

    def test_sinks_required_when_use_sinks_true(self):
        """Not providing sinks when spec.use_sinks=True raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,  # Sinks enabled
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape)
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=None,  # Missing sinks
            )

        self.assertIn("spec.use_sinks=True requires sinks", str(cm.exception))

    def test_sinks_wrong_shape_rejected(self):
        """Sinks with incorrect shape raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Mock sinks with wrong shape (16 instead of spec.num_query_heads=8)
        wrong_size = spec.num_query_heads * 2
        sinks = SimpleNamespace(
            shape=(wrong_size,),
            dtype="bfloat16",
            is_contiguous=lambda: True,
            is_cuda=True,
        )

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,
            )

        err_msg = str(cm.exception)
        self.assertIn("sinks must have shape", err_msg)
        self.assertIn(f"({spec.num_query_heads},)", err_msg)
        self.assertIn(f"({wrong_size},)", err_msg)

    def test_sinks_wrong_dtype_rejected(self):
        """Sinks with dtype mismatch raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Mock sinks with wrong dtype (float16 instead of bfloat16)
        sinks = SimpleNamespace(
            shape=(spec.num_query_heads,),
            dtype="float16",
            is_contiguous=lambda: True,
            is_cuda=True,
        )

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,
            )

        err_msg = str(cm.exception)
        self.assertIn("sinks dtype", err_msg)
        self.assertIn("must match q dtype", err_msg)

    def test_sinks_non_contiguous_rejected(self):
        """Non-contiguous sinks raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Mock non-contiguous sinks
        sinks = SimpleNamespace(
            shape=(spec.num_query_heads,),
            dtype="bfloat16",
            is_contiguous=lambda: False,
            is_cuda=True,
        )

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,
            )

        self.assertIn("sinks must be contiguous", str(cm.exception))

    def test_sinks_cpu_tensor_rejected(self):
        """CPU sinks tensor raises ValueError (would silently produce garbage)."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Mock CPU tensor (is_cuda=False)
        sinks = SimpleNamespace(
            shape=(spec.num_query_heads,),
            dtype="bfloat16",
            is_contiguous=lambda: True,
            is_cuda=False,
        )

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,
            )

        self.assertEqual(str(cm.exception), "sinks must be a CUDA tensor")


class TestGfx950DenseVariants(unittest.TestCase):
    """One registered candidate per frozen (tile x persist x wide-DMA) combo."""

    def test_six_dense_candidates_are_registered(self):
        from dispatch.attention.gfx950 import GFX950_DENSE_VARIANTS

        names = [c.name for c in attention_candidates()]
        expected = [v.candidate_name for v in GFX950_DENSE_VARIANTS]
        self.assertEqual(len(expected), 6)
        for name in expected:
            with self.subTest(name=name):
                self.assertIn(name, names)

    def test_d128_admits_all_six_dense_variants(self):
        from dispatch.attention.gfx950 import GFX950_DENSE_VARIANTS

        req = _gfx950_dense_req(
            hdim_q=128,
            hdim_v=128,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=2048,
            seqlen_k=2048,
        )
        names = {
            c.name
            for c in attention_candidates()
            if c.algorithm == "attention_dense" and c.admits(req)[0]
        }
        self.assertEqual(names, {v.candidate_name for v in GFX950_DENSE_VARIANTS})

    def test_d64_refuses_wide_dma_variants(self):
        req = _gfx950_dense_req()
        names = {
            c.name
            for c in attention_candidates()
            if c.algorithm == "attention_dense" and c.admits(req)[0]
        }
        self.assertIn("attention_gfx950_dense_grid_default", names)
        self.assertNotIn("attention_gfx950_dense", names)
        self.assertNotIn("attention_gfx950_dense_persist_widedma_bm128", names)

    def test_dense_tile_bm128_pin_filters_and_selects_block_m(self):
        req = _gfx950_dense_req(
            hdim_q=128,
            hdim_v=128,
            nhead_q=32,
            nhead_k=8,
            dense_tile="bm128",
        )
        names = {
            c.name
            for c in attention_candidates()
            if c.algorithm == "attention_dense" and c.admits(req)[0]
        }
        self.assertEqual(
            names,
            {
                "attention_gfx950_dense_grid_bm128",
                "attention_gfx950_dense_persist_bm128",
                "attention_gfx950_dense_persist_widedma_bm128",
            },
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.block_m, 128)
        self.assertTrue(spec.persistent)
        self.assertTrue(spec.wide_lds_dma)

    def test_unpinned_llama3_8b_s8192_is_persist_widedma_default(self):
        req = _gfx950_dense_req(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=8192,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
        )
        spec = dense_spec_for_request(req)
        self.assertEqual(spec.block_m, 256)
        self.assertTrue(spec.persistent)
        self.assertTrue(spec.wide_lds_dma)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")
        from dispatch.attention import dispatch_attention

        result = dispatch_attention(req)
        self.assertEqual(result.candidate.name, "attention_gfx950_dense")

    def test_registered_combos_include_dense_not_unified_2d(self):
        from dispatch.attention import registered_attention_combos

        req = _gfx950_dense_req(
            hdim_q=128,
            hdim_v=128,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=2048,
            seqlen_k=2048,
            algorithm="auto",
        )
        names = {
            c.name for c, _spec in registered_attention_combos(req, tuning_sample=2)
        }
        self.assertIn("attention_gfx950_dense", names)
        self.assertIn("attention_gfx950_dense_grid_bm128", names)
        self.assertNotIn("attention_unified_2d", names)

    def test_d256_combos_exclude_dense_and_routing_labels(self):
        from dispatch.attention import registered_attention_combos

        req = _gfx950_dense_req(
            hdim_q=256,
            hdim_v=256,
            nhead_q=16,
            nhead_k=2,
            seqlen_q=2048,
            seqlen_k=2048,
            dtype="bf16",
            algorithm="auto",
        )
        combos = registered_attention_combos(req, tuning_sample=2)
        names = {c.name for c, _spec in combos}
        self.assertNotIn("attention_gfx950_d256", names)
        self.assertFalse(
            any(c.algorithm == "attention_dense" for c, _spec in combos),
            names,
        )


if __name__ == "__main__":
    unittest.main()
