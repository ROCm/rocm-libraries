# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Signature-parity guard for ``supports_tiled_2d`` across arches.

The shared feasibility check ``supports_native_unified_attention_tiled`` builds
one kwarg list and passes ``use_d256_fast=`` to whichever arch's
``supports_tiled_2d`` applies. Every arch signature must therefore accept that
flag. It was originally declared only on gfx942 (added with the D256 4-warp fast
path), so gfx950 and gfx1250 raised
``TypeError: unexpected keyword argument`` for any problem routed through the
check. It went unnoticed because production (``run_unified_attention_torch``)
skips this check; only the bench/parity ``path="auto"`` path hits it.

These tests lock the parity across all three arches so the mismatch cannot
regress. Arch is pinned via ``_RESOLVED_ATTENTION_ARCH``, so they run GPU-free
on any host.
"""

from __future__ import annotations

import inspect
import unittest

import kernels.common.attention_unified as au
import kernels.gfx942.attention_tiled_2d as t2d_942
import kernels.gfx950.attention_tiled_2d as t2d_950
import kernels.gfx1250.attention_tiled_2d as t2d_1250

_SUPPORTS = {
    "gfx942": t2d_942.supports_tiled_2d,
    "gfx950": t2d_950.supports_tiled_2d,
    "gfx1250": t2d_1250.supports_tiled_2d,
}


class _PinArch:
    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self.arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _d128_bf16_problem() -> "au.UnifiedAttentionProblem":
    return au.UnifiedAttentionProblem(
        total_q=8192,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=8192,
        max_seqlen_k=8192,
        dtype="bf16",
    )


class TestSupportsSignatureParity(unittest.TestCase):
    def test_every_arch_signature_accepts_use_d256_fast(self):
        for arch, fn in _SUPPORTS.items():
            self.assertIn(
                "use_d256_fast",
                inspect.signature(fn).parameters,
                msg=f"{arch} supports_tiled_2d must accept use_d256_fast",
            )

    def test_no_arch_keeps_the_old_flag_name(self):
        for arch, fn in _SUPPORTS.items():
            self.assertNotIn(
                "use_d256_gfx942_fast",
                inspect.signature(fn).parameters,
                msg=f"{arch} still uses the old kwarg name",
            )

    def test_shared_caller_does_not_raise_per_arch(self):
        # Reproduces the bug: the shared feasibility check passes use_d256_fast to
        # each arch's supports_tiled_2d. Must return (bool, str), never TypeError.
        for arch in _SUPPORTS:
            with _PinArch(arch):
                ok, why = au.supports_native_unified_attention_tiled(
                    _d128_bf16_problem()
                )
            self.assertIsInstance(ok, bool, msg=f"{arch}: {why!r}")
            self.assertIsInstance(why, str)


if __name__ == "__main__":
    unittest.main()
