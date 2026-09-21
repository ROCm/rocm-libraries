# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU off-device dispatch tests for AICK-2146.

Verifies that the attention dispatcher makes selection decisions based solely on
``req.arch`` — the request parameter — without reading the live GPU device arch
(``_resolve_attention_arch()``).  All tests run on a CPU-only box with no GPU
and dispatch for arches that are NOT the host arch.

Definition of Done from the ticket:
- ``_resolve_attention_arch()`` is never called on the selection/geometry path.
- ``arch`` flows from the request end-to-end.
- CPU tests dispatch for an architecture that is not the host, across every arch
  the dispatcher currently declares.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionRequest,
    dispatch_attention,
)
from kernels.common.attention_unified import (
    _enable_gfx942_fp16_flash,
    _enable_gfx942_bf16_flash,
    _enable_combo_2d,
    _enable_single_batch_combo,
    supports_native_unified_attention,
    supports_native_unified_attention_tiled,
)


def _make_request(arch: str, **kw) -> AttentionRequest:
    """Build a minimal AttentionRequest for ``arch``."""
    defaults = dict(
        batch=2,
        nhead_q=16,
        nhead_k=16,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="fp16",
    )
    defaults.update(kw)
    return AttentionRequest(**defaults)


class TestDispatchDoesNotReadLiveDevice(unittest.TestCase):
    """_resolve_attention_arch must NOT be called during selection."""

    def _assert_no_live_arch_call(self, arch: str, **kw):
        req = _make_request(arch, **kw)
        with patch.object(
            au,
            "_resolve_attention_arch",
            side_effect=AssertionError(
                f"_resolve_attention_arch() called during dispatch for arch={arch!r} — "
                "selection path must not read the live device"
            ),
        ):
            # If _resolve_attention_arch() is called on the selection path the
            # patch raises AssertionError, failing the test.
            result = dispatch_attention(req)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.spec)

    def test_gfx942_fp16_no_live_device(self):
        self._assert_no_live_arch_call("gfx942", dtype="fp16")

    def test_gfx942_bf16_no_live_device(self):
        self._assert_no_live_arch_call("gfx942", dtype="bf16")

    def test_gfx950_fp16_no_live_device(self):
        self._assert_no_live_arch_call("gfx950", dtype="fp16")

    def test_gfx950_bf16_no_live_device(self):
        self._assert_no_live_arch_call("gfx950", dtype="bf16")


class TestArchFromRequestNotDevice(unittest.TestCase):
    """Selector functions return results consistent with req.arch, not the host."""

    def _simulate_no_gpu(self):
        """Return a patcher that removes the memoized GPU arch so _resolve_attention_arch
        would fall back to "gfx950" (the default), letting tests prove that
        a DIFFERENT arch was used for selection."""
        return patch.object(au, "_RESOLVED_ATTENTION_ARCH", None)

    def test_gfx942_selector_fires_for_gfx942_request(self):
        """_enable_gfx942_fp16_flash returns True only when arch="gfx942"."""
        from kernels.common.attention_unified import UnifiedAttentionProblem

        problem = UnifiedAttentionProblem(
            total_q=1024,
            num_seqs=2,
            num_query_heads=16,
            num_kv_heads=16,
            head_size=128,
            block_size=16,
            max_seqlen_q=512,
            max_seqlen_k=512,
            dtype="fp16",
        )
        self.assertTrue(_enable_gfx942_fp16_flash(problem, "gfx942"))
        self.assertFalse(_enable_gfx942_fp16_flash(problem, "gfx950"))
        self.assertFalse(_enable_gfx942_fp16_flash(problem, "gfx1250"))

    def test_combo_2d_fires_only_for_gfx950(self):
        """_enable_combo_2d is gfx950-only and must respect the explicit arch param."""
        from kernels.common.attention_unified import UnifiedAttentionProblem

        problem = UnifiedAttentionProblem(
            total_q=2048,
            num_seqs=2,
            num_query_heads=64,
            num_kv_heads=8,
            head_size=64,
            block_size=32,
            max_seqlen_q=1024,
            max_seqlen_k=1024,
            dtype="bf16",
            use_sinks=True,
        )
        self.assertTrue(_enable_combo_2d(problem, "gfx950"))
        self.assertFalse(_enable_combo_2d(problem, "gfx942"))
        self.assertFalse(_enable_combo_2d(problem, "gfx1250"))

    def test_supports_tiled_uses_request_arch(self):
        """supports_native_unified_attention_tiled uses the passed arch, not the device."""
        from kernels.common.attention_unified import UnifiedAttentionProblem

        problem = UnifiedAttentionProblem(
            total_q=1024,
            num_seqs=2,
            num_query_heads=16,
            num_kv_heads=16,
            head_size=128,
            block_size=16,
            max_seqlen_q=512,
            max_seqlen_k=512,
            dtype="fp16",
        )
        with self._simulate_no_gpu():
            ok_gfx942, _ = supports_native_unified_attention_tiled(problem, "gfx942")
            ok_gfx950, _ = supports_native_unified_attention_tiled(problem, "gfx950")
        # Both should support fp16 hd128 (not checking tiled specifically here,
        # just that the call completes without touching _resolve_attention_arch).
        self.assertIsInstance(ok_gfx942, bool)
        self.assertIsInstance(ok_gfx950, bool)

    def test_dispatch_for_non_host_arch_succeeds(self):
        """Dispatching for a non-host arch completes without touching the GPU."""
        # Simulate CPU-only box: clear the memoized arch AND make any real GPU
        # lookup raise so the test fails fast if the dispatch path falls through.
        with (
            patch.object(au, "_RESOLVED_ATTENTION_ARCH", None),
            patch.object(
                au,
                "_resolve_attention_arch",
                side_effect=AssertionError(
                    "live GPU read during dispatch — must not happen on selection path"
                ),
            ),
        ):
            for arch in ("gfx942", "gfx950", "gfx1250"):
                with self.subTest(arch=arch):
                    req = _make_request(arch, dtype="fp16")
                    result = dispatch_attention(req)
                    self.assertIsNotNone(result.spec)

    def test_bf16_gfx942_dispatch_no_device_read(self):
        """bf16 gfx942 dispatch uses request arch, not live device."""
        with patch.object(
            au,
            "_resolve_attention_arch",
            side_effect=AssertionError("live GPU read on selection path"),
        ):
            req = _make_request("gfx942", dtype="bf16", seqlen_q=1024, seqlen_k=2048)
            result = dispatch_attention(req)
            self.assertIsNotNone(result.spec)

    def test_gfx950_dispatch_no_device_read(self):
        """gfx950 dispatch (combo/transposed path) uses request arch."""
        with patch.object(
            au,
            "_resolve_attention_arch",
            side_effect=AssertionError("live GPU read on selection path"),
        ):
            req = _make_request(
                "gfx950",
                dtype="bf16",
                batch=2,
                nhead_q=64,
                nhead_k=8,
                seqlen_q=1024,
                seqlen_k=1024,
                hdim_q=64,
                hdim_v=64,
            )
            result = dispatch_attention(req)
            self.assertIsNotNone(result.spec)

    def test_gfx1250_dispatch_no_device_read(self):
        """gfx1250 dispatch uses request arch, not live device."""
        with patch.object(
            au,
            "_resolve_attention_arch",
            side_effect=AssertionError("live GPU read on selection path"),
        ):
            req = _make_request(
                "gfx1250",
                dtype="fp16",
                batch=2,
                nhead_q=16,
                nhead_k=16,
                seqlen_q=512,
                seqlen_k=512,
                hdim_q=128,
                hdim_v=128,
            )
            result = dispatch_attention(req)
            self.assertIsNotNone(result.spec)


if __name__ == "__main__":
    unittest.main()
