# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the 2D-tiled LDS budget resolver.

Pure codegen (no GPU, no subprocess). The resolver deterministically shrinks an
over-budget register-PV 2D spec until it fits the arch LDS cap, and is a strict
no-op for every already-fitting or non-register-PV config, on any arch.

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
    def setUp(self):
        # The production D256 gfx950 fast route (``_d256_gfx950_fast``) targets
        # this exact cohort (D256 / gfx950 / bf16 prefill) and overrides
        # ``use_register_pv=False``, which would bypass the resolver under test.
        # Disable it here so these unit tests exercise the register-PV LDS-shrink
        # path directly, decoupled from the production routing decision -- which
        # is covered separately by ``TestD256ProductionRouting`` below.
        _p = mock.patch.object(au, "_d256_gfx950_fast", return_value=False)
        _p.start()
        self.addCleanup(_p.stop)

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

    def test_resolver_arch_agnostic_targets_dynamic_cap(self):
        """Arch-agnostic: the resolver targets whatever LDS cap the arch-target
        API reports -- not a hard-coded gfx950 / 163840 constant. Same D256
        K-double geometry (204800 B): with a larger reported cap it already fits
        and is returned byte-identical; with a tiny cap it engages and raises."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            spec = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
            k_double = replace(spec, use_k_single_buffer=False)  # 204800 B
            with mock.patch.object(au, "_lds_capacity_bytes", return_value=262144):
                self.assertIs(au._resolve_lds_budget(k_double), k_double)
            with mock.patch.object(au, "_lds_capacity_bytes", return_value=65536):
                with self.assertRaises(RuntimeError):
                    au._resolve_lds_budget(replace(spec, use_k_single_buffer=False))

    def test_regpv_footprint_uses_shared_helper(self):
        """W: the register-PV footprint is the shared ``_tiled_2d_lds_bytes`` model
        parameterised for the register-PV layout (Q/P^T dropped) -- a single source
        of truth, so it cannot drift from the gfx942 admission gate."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            spec = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
        block_m = spec.num_warps * spec.block_m_per_warp
        expected = au._tiled_2d_lds_bytes(
            tile_size=spec.tile_size,
            head_size=spec.head_size,
            block_m=block_m,
            kv_elem_bytes=au._kv_lds_elem_bytes(spec),
            k_slots=1 if spec.use_k_single_buffer else 2,
            v_slots=2 if spec.use_v_double_buffer else 1,
        )
        self.assertEqual(au._lds_bytes_regpv(spec), expected)

    def test_shared_helper_models_both_layouts(self):
        """W: one helper serves both consumers -- the conservative gfx942 gate
        layout (Q_lds/P_lds staged) and the exact register-PV layout (dropped)."""
        common = dict(tile_size=128, head_size=256, block_m=16, kv_elem_bytes=2)
        conservative = au._tiled_2d_lds_bytes(
            k_slots=2, v_slots=1, include_q_lds=True, include_p_lds=True, **common
        )
        regpv = au._tiled_2d_lds_bytes(k_slots=2, v_slots=1, **common)
        # Dropping the P_lds term makes the register-PV footprint the smaller of
        # the two; the conservative gate estimate never under-counts it.
        self.assertGreater(conservative, regpv)
        self.assertEqual(regpv, 204800)  # matches the comgr-verified D256 number

    def test_ldsfix_levers_report_reasons(self):
        """S: the shrink levers return ``(None, reason)`` when they cannot apply,
        so the resolver surfaces *why* rather than swallowing it."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            spec = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
        # single-K applies (returns a spec, no reason) ...
        cand, why = au._ldsfix_single_k(replace(spec, use_k_single_buffer=False))
        self.assertIsNotNone(cand)
        self.assertIsNone(why)
        # ... but is a no-op once K is already single-buffered (reason given).
        cand, why = au._ldsfix_single_k(replace(spec, use_k_single_buffer=True))
        self.assertIsNone(cand)
        self.assertIn("single-buffered", why)
        # tile64 is a no-op at T<=64.
        cand, why = au._ldsfix_tile64(replace(spec, tile_size=64))
        self.assertIsNone(cand)
        self.assertIn("64", why)

    def test_infeasible_budget_raises_with_lever_diagnostics(self):
        """S: when no lever fits, the resolver raises with each attempted lever's
        result -- a diagnostic, not a cryptic downstream comgr abort."""
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            spec = au._tiled_spec_from_problem(_d256_bf16_long_prefill())
            # Force an impossibly small cap so even single-K + T=64 overflows.
            with mock.patch.object(au, "_lds_capacity_bytes", return_value=1024):
                with self.assertRaises(RuntimeError) as ctx:
                    au._resolve_lds_budget(replace(spec, use_k_single_buffer=False))
        msg = str(ctx.exception)
        self.assertIn("single-K", msg)
        self.assertIn("T=64", msg)


class TestD256ProductionRouting(unittest.TestCase):
    """Production routing (``_d256_gfx950_fast`` live, not mocked): the
    D256 / gfx950 / bf16 prefill cohort is served by the 32x32 transposed fast
    path, NOT the register-PV LDS-shrink resolver. This is the routing the PR
    introduces; the resolver becomes a no-op for it (register-PV disabled)."""

    def test_d256_gfx950_prefill_routes_to_fast_path(self):
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            problem = _d256_bf16_long_prefill()
            # The cohort matches the fast-route predicate ...
            self.assertTrue(au._d256_gfx950_fast(problem))
            spec = au._tiled_spec_from_problem(problem)
            # ... so the built spec is the 32x32 interleave fast path, not
            # register-PV: register-PV is off and the 32x32 stack is on.
            self.assertFalse(spec.use_register_pv)
            self.assertTrue(spec.use_mfma_32x32)
            self.assertTrue(spec.use_transposed_qk_32x32)
            self.assertTrue(spec.use_softmax_mfma_interleave)
            self.assertEqual(spec.softmax_interleave_mode, 2)
            self.assertEqual(spec.softmax_interleave_groups, 4)
            # The fast route single-buffers K itself, so the spec fits the cap
            # and the resolver is a strict no-op (register-PV off -> same object).
            self.assertTrue(spec.use_k_single_buffer)
            self.assertIs(au._resolve_lds_budget(spec), spec)
            self.assertLessEqual(au._lds_bytes_regpv(spec), au._lds_capacity_bytes())


if __name__ == "__main__":
    unittest.main()
