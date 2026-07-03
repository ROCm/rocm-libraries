# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the 2D-tiled LDS-budget resolver.

Pure codegen (no GPU, no subprocess). The resolver deterministically shrinks an
over-budget register-PV 2D spec until it fits the arch LDS cap, and is a strict
no-op for every already-fitting / non-register-PV / non-gfx950 config.

The load-bearing case: bf16 head_dim=256 long prefill on gfx950 overflows LDS at
the default (K double-buffered) geometry -- 204800 B > the 163840 B cap -- and the
resolver must single-buffer K so it compiles. The pre-fix codegen matrix only
covered D256 in fp16, so this path was previously untested.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import kernels.common.attention_unified as au
from kernels import UnifiedAttentionProblem


def _d256_bf16_long_prefill(block_size: int = 64, max_seqlen: int = 4096):
    # No sinks / sliding-window / softcap / alibi / qq_bias, so use_register_pv
    # is eligible and the spec routes onto the register-PV 2D path.
    return UnifiedAttentionProblem(
        total_q=max_seqlen,
        num_seqs=1,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=256,
        block_size=block_size,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dtype="bf16",
    )


class TestLdsBudgetResolver(unittest.TestCase):
    def test_d256_gfx950_bf16_prefill_shrinks_to_fit(self):
        """D256 bf16 register-PV overflows at the default geometry; the resolver
        must single-buffer K so the resolved spec fits the gfx950 160 KB cap."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            spec = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
            self.assertTrue(spec.use_register_pv)
            # The cheapest lever (single-buffer K) was applied ...
            self.assertTrue(spec.use_k_single_buffer)
            # ... and the resolved geometry fits the arch cap.
            self.assertLessEqual(au._lds_bytes_regpv(spec), au._lds_capacity_bytes())

    def test_pre_resolve_geometry_actually_overflows(self):
        """Proves the resolver was necessary: the K-double-buffered geometry it
        started from does exceed the cap (so the no-op guard didn't fire)."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            spec = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
            k_double = replace(spec, use_k_single_buffer=False)
            self.assertGreater(au._lds_bytes_regpv(k_double), au._lds_capacity_bytes())

    def test_resolver_is_noop_when_already_fitting(self):
        """A spec that already fits is returned byte-identical (same object)."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            fitted = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
            self.assertLessEqual(au._lds_bytes_regpv(fitted), au._lds_capacity_bytes())
            self.assertIs(au._resolve_lds_budget(fitted), fitted)

    def test_resolver_is_noop_off_gfx950(self):
        """On a non-gfx950 arch the resolver never engages -- it returns the spec
        unchanged (the footprint model is only validated for gfx950)."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            spec = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx942"):
            self.assertIs(au._resolve_lds_budget(spec), spec)

    def test_acc_lds_scales_with_geometry(self):
        """The auxiliary term is the epilogue Acc_lds, computed from geometry
        (BLOCK_M x OUT_STRIPE_COLS x 2), not a hard-coded constant."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            spec = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
        block_m = spec.num_warps * spec.block_m_per_warp
        out_stripe = 32 if spec.head_size <= 64 else spec.head_size
        self.assertEqual(au._acc_lds_bytes(spec), block_m * out_stripe * 2)


if __name__ == "__main__":
    unittest.main()
