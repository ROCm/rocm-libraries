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

from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
)
from dispatch.attention.gfx950 import dense_spec_for_request


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
        # Create a ragged-shaped request: seqlen_q=seqlen_k, not a multiple of block sizes
        # _BLOCK_M=256, _DENSE_BLOCK_N=64, so 500 triggers ragged
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
        # Non-ragged shape: seqlen_q is a multiple of _BLOCK_M=256
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
        """Dispatch selects dense candidate for sinks requests."""
        req = _gfx950_dense_req(use_sinks=True)

        # Find which candidates admit this request
        candidates = [c for c in attention_candidates() if c.admits(req)[0]]

        # Verify dense is among them
        dense = next(
            (c for c in candidates if c.name == "attention_gfx950_dense"), None
        )
        self.assertIsNotNone(dense, "Dense candidate should admit sinks requests")

    def test_dispatch_selects_dense_for_sliding_window_request(self):
        """Dispatch selects dense candidate for sliding_window requests (AICK-1933)."""
        req = _gfx950_dense_req(sliding_window=256)

        # Find which candidates admit this request
        candidates = [c for c in attention_candidates() if c.admits(req)[0]]

        # Verify dense is among them
        dense = next(
            (c for c in candidates if c.name == "attention_gfx950_dense"), None
        )
        self.assertIsNotNone(
            dense, "Dense candidate should admit sliding_window requests"
        )


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


if __name__ == "__main__":
    unittest.main()
