# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Signature-parity guard for the gfx950 ``supports_tiled_2d`` feasibility check.

The shared caller ``supports_native_unified_attention_tiled`` builds one kwarg
list and passes ``use_d256_gfx942_fast=`` to whichever arch's
``supports_tiled_2d`` applies. That flag was added to the gfx942 signature (with
the D256 4-warp fast path) but originally missed on gfx950, so every gfx950
problem routed through the check raised
``TypeError: unexpected keyword argument 'use_d256_gfx942_fast'``. It went
unnoticed because production (``run_unified_attention_torch``) skips this check;
only the bench/parity ``path="auto"`` path hits it.

These tests lock the parity so the mismatch can't regress. Arch is pinned via
``_RESOLVED_ATTENTION_ARCH``, so they run GPU-free on any host.
"""

from __future__ import annotations

import inspect
import unittest

import kernels.common.attention_unified as au
import kernels.gfx942.attention_tiled_2d as t2d_942
import kernels.gfx950.attention_tiled_2d as t2d_950


class _PinArch:
    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self.arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


class TestGfx950SupportsSignatureParity(unittest.TestCase):
    def test_gfx950_signature_has_d256_fast_flag(self):
        # Parity with the gfx942 signature and the shared dispatch caller.
        self.assertIn(
            "use_d256_gfx942_fast",
            inspect.signature(t2d_950.supports_tiled_2d).parameters,
        )
        self.assertIn(
            "use_d256_gfx942_fast",
            inspect.signature(t2d_942.supports_tiled_2d).parameters,
        )

    def test_gfx950_supports_accepts_flag(self):
        # Direct call with the flag must not raise and must return (bool, str).
        ok, why = t2d_950.supports_tiled_2d(
            head_size=128,
            block_size=16,
            dtype="fp16",
            num_queries_per_kv=4,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
            tile_size=64,
            num_warps=2,
            arch="gfx950",
            use_d256_gfx942_fast=False,
        )
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(why, str)

    def test_shared_caller_on_gfx950_does_not_raise(self):
        # Reproduces the original bug: the shared feasibility check passes
        # use_d256_gfx942_fast to gfx950's supports_tiled_2d.
        p = au.UnifiedAttentionProblem(
            total_q=8192,
            num_seqs=1,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            block_size=16,
            max_seqlen_q=8192,
            max_seqlen_k=8192,
            dtype="fp16",
        )
        with _PinArch("gfx950"):
            ok, why = au.supports_native_unified_attention_tiled(p)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(why, str)


if __name__ == "__main__":
    unittest.main()
