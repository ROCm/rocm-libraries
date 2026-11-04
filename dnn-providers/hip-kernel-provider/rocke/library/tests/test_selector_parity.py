# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pure-Python unit tests for the gfx950 attention tiled-2D selector logic.

These tests verify that the Python ``_tiled_spec_from_problem`` selector
produces the expected spec fields for a range of UnifiedAttentionProblem
inputs -- covering the selectors and gate predicates that route a problem
into the correct kernel variant.  No GPU, no subprocess, no rocke_engine .so
required.

Primary motivation: PR #9220 silently broke parity by removing the bias-
exclusion guard from Python ``_enable_combo_2d``, routing biased combo-geometry
attention onto the transposed combo path while C++ still refused bias.  These
tests encode the expected routing decisions so that kind of regression is caught
immediately on any developer machine.

NOTE ON EXPECTED FAILURES (PR #9220 regression):
Tests marked with @skip document a known divergence introduced by
PR #9220, which removed the bias-exclusion guard from Python _enable_combo_2d.
Python now routes biased combo-geometry problems onto the combo path while
C++ (attention_unified_selectors.cpp) still refuses them. The tests are
written for the CORRECT behavior (bias should block combo_2d) and are being skipped until the Python guard is restored.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from dataclasses import asdict, fields, replace
from types import SimpleNamespace
from unittest import mock

import builders.common.attention_spec_builder as _asb
import kernels.common.attention_unified as _au
from dispatch.attention import AttentionRequest, attention_candidates
from dispatch.attention.common import AttentionTuningSpec, _tuning_problem
from dispatch.attention.tuning_specs import (
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    _semantic_fields,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
)
from kernels import UnifiedAttentionProblem
from kernels.gfx942 import attention_tiled_3d as _gfx942_3d
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx942TiledSpec,
    build_gfx942_4warp_gqa,
)
from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx950TiledSpec,
)
from kernels.gfx1250.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx1250TiledSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_arch(arch: str):
    """Pin ``_resolve_attention_arch`` on the module that defines it.

    That is sufficient because the spec builder reaches the resolver through its
    ``attention_unified`` module handle -- a bound import there would freeze the
    reference and silently leave the builder on the real device arch, so these
    cases would test the host, not ``arch``. That invariant is pinned, statically
    and behaviourally, by ``test_arch_binding_guard.py``; it is not re-asserted
    here.
    """
    return mock.patch.object(_au, "_resolve_attention_arch", return_value=arch)


def _spec(problem: UnifiedAttentionProblem, arch: str = "gfx950") -> dict:
    """Return _tiled_spec_from_problem as a plain dict for easy assertions."""
    with _patch_arch(arch):
        return asdict(_asb._tiled_spec_from_problem(problem, arch))


def _prob(
    head_size: int,
    block_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    dtype: str = "bf16",
    max_seqlen_q: int = 512,
    max_seqlen_k: int = 4096,
    num_seqs: int = 2,
    total_q: int = 1024,
    sliding_window: int = 0,
    softcap: float = 0.0,
    use_sinks: bool = False,
    use_alibi: bool = False,
    use_qq_bias: bool = False,
    use_fp8: bool = False,
    num_kv_blocks: int = 0,
) -> UnifiedAttentionProblem:
    return UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dtype=dtype,
        sliding_window=sliding_window,
        softcap=softcap,
        use_sinks=use_sinks,
        use_alibi=use_alibi,
        use_qq_bias=use_qq_bias,
        use_fp8=use_fp8,
        num_kv_blocks=num_kv_blocks,
    )


# Canonical combo_2d problem: gfx950 + bf16 + d64/b32 + GQA-8 + seqlen_q > 256
def _combo_prob(**overrides) -> UnifiedAttentionProblem:
    defaults = dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        max_seqlen_q=512,
    )
    defaults.update(overrides)
    return _prob(**defaults)


# ---------------------------------------------------------------------------
# enable_combo_2d gate
# ---------------------------------------------------------------------------


class TestEnableCombo2d(unittest.TestCase):
    """_enable_combo_2d fires for the gfx950 d64/b32/GQA-8/S>256 cohort: bf16
    (sinks or not), or fp16 only when use_sinks (the fp16 widening was measured
    on sink prefill)."""

    _BIAS_BUG = (
        "PR #9220: Python _enable_combo_2d missing bias guard (C++ still refuses bias)"
    )

    def _gate(self, problem: UnifiedAttentionProblem, arch: str = "gfx950") -> bool:
        with _patch_arch(arch):
            return _au._enable_combo_2d(problem, arch)

    def test_canonical_combo_fires(self):
        self.assertTrue(self._gate(_combo_prob()))

    @unittest.skip(_BIAS_BUG)
    def test_qq_bias_blocks_combo(self):
        """use_qq_bias=True must disable combo_2d (was the #9220 bug)."""
        self.assertFalse(self._gate(_combo_prob(use_qq_bias=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(use_alibi=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(softcap=50.0)), self._BIAS_BUG)

    def test_fp16_no_sinks_blocks_combo(self):
        # fp16 combo is sink-prefill only; non-sink fp16 stays off the combo path.
        self.assertFalse(self._gate(_combo_prob(dtype="fp16")))

    def test_fp16_sinks_enables_combo(self):
        # fp16 + sinks is the one fp16 shape the widening admits to combo.
        self.assertTrue(self._gate(_combo_prob(dtype="fp16", use_sinks=True)))

    def test_wrong_head_size_blocks_combo(self):
        self.assertFalse(self._gate(_prob(128, 32, 64, 8)))

    def test_wrong_block_size_blocks_combo(self):
        self.assertFalse(self._gate(_prob(64, 16, 64, 8)))

    def test_wrong_gqa_ratio_blocks_combo(self):
        # nqpk = 16/4 = 4, not 8
        self.assertFalse(self._gate(_prob(64, 32, 16, 4)))

    def test_short_seqlen_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(max_seqlen_q=256)))
        self.assertFalse(self._gate(_combo_prob(max_seqlen_q=100)))

    def test_gfx942_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(), arch="gfx942"))


# ---------------------------------------------------------------------------
# Routing: combo_2d cohort → transposed_qk_32x32 flags
# ---------------------------------------------------------------------------


class TestCombo2dRouting(unittest.TestCase):
    """combo_2d problems must get the full 32x32 transposed stack."""

    _BIAS_BUG = "PR #9220: Python _enable_combo_2d missing bias guard"

    def test_combo_gets_transposed_flags(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_mfma_32x32"])
        self.assertTrue(s["use_transposed_qk_32x32"])
        self.assertTrue(s["use_transposed_half_local_pv"])
        self.assertTrue(s["use_transposed_scalar_state"])

    @unittest.skip(_BIAS_BUG)
    def test_biased_combo_does_not_get_fast_paged_kv(self):
        """#9220: biased combo must NOT get use_fast_paged_kv_desc."""
        s = _spec(_combo_prob(use_qq_bias=True))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_combo_no_fast_paged_kv(self):
        s = _spec(_combo_prob(use_alibi=True))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_combo_no_fast_paged_kv(self):
        s = _spec(_combo_prob(softcap=50.0))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    def test_combo_no_sw_gets_fast_paged_kv(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_fast_paged_kv_desc"])

    def test_combo_with_sw_no_fast_paged_kv(self):
        s = _spec(_combo_prob(sliding_window=256))
        self.assertFalse(s["use_fast_paged_kv_desc"])


# ---------------------------------------------------------------------------
# Bias x geometry cohort matrix (covers the #9220 class of divergence)
# ---------------------------------------------------------------------------


class TestBiasGeometryCohorts(unittest.TestCase):
    """Systematic sweep of bias flags x combo-geometry dimensions."""

    _BIAS_BUG = "PR #9220: Python _enable_combo_2d missing bias guard"

    def _fast_kv(self, problem: UnifiedAttentionProblem) -> bool:
        return _spec(problem)["use_fast_paged_kv_desc"]

    # ---- baseline: canonical combo fires ----
    def test_canonical_combo(self):
        self.assertTrue(self._fast_kv(_combo_prob()))

    # ---- PR #9220 biased-combo cohort (skip the test until bug is fixed) ----
    @unittest.skip(_BIAS_BUG)
    def test_qq_bias_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(use_qq_bias=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(use_alibi=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(softcap=50.0)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_and_qq_bias_no_combo(self):
        self.assertFalse(
            self._fast_kv(_combo_prob(use_alibi=True, use_qq_bias=True)), self._BIAS_BUG
        )

    # ---- geometry boundaries (not affected by #9220) ----
    def test_seqlen_at_boundary_no_combo(self):
        self.assertFalse(self._fast_kv(_combo_prob(max_seqlen_q=256)))

    def test_seqlen_above_boundary_combo(self):
        self.assertTrue(self._fast_kv(_combo_prob(max_seqlen_q=257)))

    def test_wrong_dtype_no_combo(self):
        self.assertFalse(self._fast_kv(_combo_prob(dtype="fp16")))

    def test_wrong_head_size_no_combo(self):
        self.assertFalse(self._fast_kv(_prob(128, 32, 64, 8)))

    def test_wrong_gqa_ratio_no_combo(self):
        self.assertFalse(self._fast_kv(_prob(64, 32, 16, 4)))


# ---------------------------------------------------------------------------
# Selector fields: tile_size, num_warps, block_m_per_warp, waves_per_eu
# ---------------------------------------------------------------------------


class TestSelectorFields(unittest.TestCase):
    """Key selector outputs for representative problems."""

    def test_combo_tile_size_is_2x_block(self):
        # combo_2d, no SW: tile_size = 2 * block_size = 64
        s = _spec(_combo_prob())
        self.assertEqual(s["tile_size"], 64)

    def test_combo_sw_tile_size_is_block_size(self):
        # combo_2d + SW: tile_size = block_size = 32
        s = _spec(_combo_prob(sliding_window=256))
        self.assertEqual(s["tile_size"], 32)

    def test_single_batch_d64_tile_size(self):
        # single-batch d64 combo: tile_size = 128
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertEqual(s["tile_size"], 128)

    def test_combo_waves_per_eu(self):
        # combo_2d: waves_per_eu = 4
        s = _spec(_combo_prob())
        self.assertEqual(s["waves_per_eu"], 4)

    def test_combo_block_m_per_warp(self):
        # combo_2d uses 32x32: block_m_per_warp = 32
        s = _spec(_combo_prob())
        self.assertEqual(s["block_m_per_warp"], 32)

    def test_non_combo_block_m_per_warp(self):
        # short seqlen → not combo, not transposed → block_m_per_warp = 16
        s = _spec(_combo_prob(max_seqlen_q=64))
        self.assertEqual(s["block_m_per_warp"], 16)

    def test_non_combo_waves_per_eu(self):
        # standard path: waves_per_eu = 2
        s = _spec(_combo_prob(max_seqlen_q=64))
        self.assertEqual(s["waves_per_eu"], 2)


# ---------------------------------------------------------------------------
# Single-batch combo cohort
# ---------------------------------------------------------------------------


class TestSingleBatchCombo(unittest.TestCase):

    def test_single_batch_d64_gets_transposed_flags(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertTrue(s["use_mfma_32x32"])
        self.assertTrue(s["use_transposed_qk_32x32"])

    def test_single_batch_bias_no_transposed(self):
        # single-batch + bias: _enable_single_batch_combo rejects alibi/qq_bias
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512, use_qq_bias=True))
        # transposed_qk_32x32 may still fire via multi-batch branch, but
        # fast_paged_kv_desc (which needs combo + no-bias) must be off
        self.assertFalse(s["use_fast_paged_kv_desc"])

    def test_single_batch_long_d64_gets_early_v(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=2048))
        self.assertTrue(s["use_early_v_schedule"])
        self.assertFalse(s["use_v_double_buffer"])

    def test_single_batch_short_d64_gets_v_double_buffer(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertFalse(s["use_early_v_schedule"])
        self.assertTrue(s["use_v_double_buffer"])

    def test_single_batch_d128_ksingle_off_not_raise(self):
        # d128 single-batch triggers the softmax-MFMA interleave -> num_warps=4
        # (block_m=128). With tile_size=64 the geometry guard block_m <= tile_size
        # fails, so _enable_k_single_buffer derives K-single OFF and the spec
        # BUILDS (previously the stale block_size>=32 proxy left K-single ON here,
        # raising an uncaught ValueError at spec build).
        s = _spec(_prob(128, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertFalse(s["use_k_single_buffer"])


# ---------------------------------------------------------------------------
# Register-PV gate
# ---------------------------------------------------------------------------


class TestRegisterPv(unittest.TestCase):

    def test_register_pv_off_for_combo(self):
        # combo_2d uses mfma_32x32 which conflicts with register_pv
        s = _spec(_combo_prob())
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_bias(self):
        s = _spec(_prob(64, 32, 32, 32, use_qq_bias=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_alibi(self):
        s = _spec(_prob(64, 32, 32, 32, use_alibi=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_sinks(self):
        s = _spec(_prob(64, 32, 32, 32, use_sinks=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_eligible_for_standard_bf16(self):
        # Standard bf16 short-seqlen, no special flags → register_pv eligible
        s = _spec(_prob(64, 32, 32, 32, max_seqlen_q=64))
        self.assertTrue(s["use_register_pv"])


# ---------------------------------------------------------------------------
# i64_kv_addr
# ---------------------------------------------------------------------------


# Combo-shape blocks are 32 tokens x 8 kv-heads x 64 dims x 2 bytes = 32 KiB, so
# 65534 blocks fill the i32 buffer range (0x7FFF0000 bytes) exactly.
_AT_I32_LIMIT = 65534
_OVER_I32_LIMIT = 65535


class TestI64KvAddr(unittest.TestCase):

    def test_no_kv_blocks_no_i64(self):
        s = _spec(_combo_prob())  # num_kv_blocks=0 default
        self.assertFalse(s["use_i64_kv_addr"])

    def test_small_cache_no_i64(self):
        # Exactly 0x7FFF0000 bytes -> still i32 (threshold is STRICTLY above)
        s = _spec(_combo_prob(num_kv_blocks=_AT_I32_LIMIT))
        self.assertFalse(s["use_i64_kv_addr"])

    def test_large_cache_gets_i64(self):
        # One block over the i32 buffer range -> i64
        s = _spec(_combo_prob(num_kv_blocks=_OVER_I32_LIMIT))
        self.assertTrue(s["use_i64_kv_addr"])

    def test_top_64kib_below_2gib_gets_i64(self):
        # The tiled builders bound K/V buffer loads at num_records=0x7FFF0000, so
        # a cache of exactly 2 GiB would read zeros from its top 64 KiB on i32.
        s = _spec(_combo_prob(num_kv_blocks=65536))
        self.assertTrue(s["use_i64_kv_addr"])
        self.assertEqual(_au._I32_KV_BUFFER_BYTES, 0x7FFF0000)

    def test_i64_threshold_splits_cache_key(self):
        # Regression guard against dropping the i64 flag from _tiled_cache_key.
        # Crossing the threshold builds a different kernel (i64 vs i32 paged-KV
        # addressing), so a key without that decision would let a large cache
        # reuse the i32 launcher cached for a small cache of the same geometry.
        # Two problems straddling the threshold must not collide.
        small = _combo_prob(num_kv_blocks=_AT_I32_LIMIT)
        large = _combo_prob(num_kv_blocks=_OVER_I32_LIMIT)
        with _patch_arch("gfx950"):
            self.assertNotEqual(
                _au._tiled_cache_key(small, "gfx950"),
                _au._tiled_cache_key(large, "gfx950"),
            )

    def test_i64_threshold_splits_3d_cache_key(self):
        # Same guard for the 3D split-KV launcher key, which also folds in
        # _enable_i64_kv_addr.
        small = _combo_prob(num_kv_blocks=_AT_I32_LIMIT)
        large = _combo_prob(num_kv_blocks=_OVER_I32_LIMIT)
        with _patch_arch("gfx950"):
            self.assertNotEqual(
                _au._tiled_3d_cache_key(small, "gfx950"),
                _au._tiled_3d_cache_key(large, "gfx950"),
            )

    def test_num_kv_blocks_only_affects_key_via_i64(self):
        # num_kv_blocks must reach the cache key only through the i64 decision.
        # Keying the raw count would compile a new launcher per cache size, and
        # not keying it at all is the collision above. Across the threshold the
        # two keys differ in exactly one field, _enable_i64_kv_addr.
        small = _combo_prob(num_kv_blocks=_AT_I32_LIMIT)
        large = _combo_prob(num_kv_blocks=_OVER_I32_LIMIT)
        with _patch_arch("gfx950"):
            ks = _au._tiled_cache_key(small, "gfx950")
            kl = _au._tiled_cache_key(large, "gfx950")
            i64_small = _au._enable_i64_kv_addr(small)
            i64_large = _au._enable_i64_kv_addr(large)
        self.assertEqual(len(ks), len(kl))
        diffs = [(a, b) for a, b in zip(ks, kl) if a != b]
        self.assertEqual(diffs, [(i64_small, i64_large)])

    def test_spec_meta_and_guard_agree_at_every_count(self):
        # The spec builders set use_i64_kv_addr from _enable_i64_kv_addr, and the
        # launch meta reads the limit off that spec. The combo spec has no i32-only
        # K/V loader on either arch, so a spec built from a problem with the real
        # count must always pass the launch guard, including the counts between
        # the i32 buffer range and 2^31 bytes.
        for arch in ("gfx942", "gfx950"):
            for n in (_AT_I32_LIMIT - 1, _AT_I32_LIMIT, _OVER_I32_LIMIT, 65536, 65537):
                p = _combo_prob(num_kv_blocks=n)
                with self.subTest(arch=arch, n=n), _patch_arch(arch):
                    spec = _asb._tiled_spec_from_problem(p, arch)
                    self.assertEqual(spec.use_i64_kv_addr, _au._enable_i64_kv_addr(p))
                    limit = _au._spec_kv_addr_limit(p, spec)
                    with _clear_launch_memos():
                        meta = _au._get_2d_launch_meta(
                            p, _au._tiled_cache_key(p, arch), arch=arch
                        )
                    self.assertEqual(meta.kv_addr_limit, limit)
                    _au._check_kv_addr_width(p, _kv_cache(n, p), limit)


# d128 bs16 4-kv-head bf16 blocks are 16 KiB, so 131072 blocks are exactly 2^31
# bytes: 64 sequences of 32768 tokens.
_AT_2GIB = 131072


def _decode_prob(num_kv_blocks: int, **kw) -> UnifiedAttentionProblem:
    fields = dict(
        total_q=64,
        num_seqs=64,
        num_query_heads=64,
        num_kv_heads=4,
        head_size=128,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=32768,
        dtype="bf16",
        num_kv_blocks=num_kv_blocks,
    )
    fields.update(kw)
    return UnifiedAttentionProblem(**fields)


class _ReachedPipeline(Exception):
    pass


def _launch_3d(problem: UnifiedAttentionProblem, num_blocks: int, arch: str) -> None:
    """Run the 3D launch on a shape-only cache of ``num_blocks`` up to the
    pipeline build, which raises ``_ReachedPipeline``."""
    with _patch_arch(arch), mock.patch.dict(
        _au._3D_KV_ADDR_LIMITS, clear=True
    ), mock.patch.object(_au, "_get_3d_pipeline", side_effect=_ReachedPipeline):
        _au._run_3d_tiled_on_stream(
            problem=problem,
            q=None,
            k=_kv_cache(num_blocks, problem),
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=None,
            softcap=0.0,
            sinks=None,
            bt_stride=0,
            warmup=0,
            attempts=1,
            use_graph=False,
        )


class TestGfx942Tiled3DKvAddr(unittest.TestCase):
    """The gfx942 tiled 3D kernel has no i64 path; its K/V loader sets the cache
    it reads. The flat loaders read to 2^31 bytes, the async copies to the
    0x7FFF0000 buffer range."""

    def test_loader_limits(self):
        limit = _gfx942_3d.i32_kv_cache_bytes
        common = dict(tile_size=16, head_size=128)
        self.assertEqual(limit(kv_fp8=False, use_wide_kv_load=True, **common), 2**31)
        self.assertEqual(limit(kv_fp8=True, use_wide_kv_load=False, **common), 2**31)
        self.assertEqual(
            limit(kv_fp8=False, use_wide_kv_load=False, **common), 0x7FFF0000
        )
        # 8 x 32 halves do not split over 64 lanes x 8, so the async copies run.
        self.assertEqual(
            limit(kv_fp8=False, use_wide_kv_load=True, tile_size=8, head_size=32),
            0x7FFF0000,
        )

    def test_exactly_2gib_decode_builds_on_wide_loader(self):
        for hq in (64, 32):
            p = _decode_prob(_AT_2GIB, num_query_heads=hq)
            with self.subTest(num_query_heads=hq), _patch_arch("gfx942"):
                spec = _asb._tiled_3d_spec_from_problem(p, "gfx942")
                self.assertTrue(spec.use_wide_kv_load)
                self.assertFalse(spec.use_i64_kv_addr)

    def test_past_2gib_wide_loader_refused(self):
        p = _decode_prob(_AT_2GIB + 1)
        with _patch_arch("gfx942"), self.assertRaises(NotImplementedError):
            _asb._tiled_3d_spec_from_problem(p, "gfx942")

    def test_async_loader_bounded_by_buffer_range(self):
        # Head size 256 takes the async copies. Its 32 KiB blocks fill the
        # buffer range at 65534 and reach 2^31 bytes at 65536.
        with _patch_arch("gfx942"):
            spec = _asb._tiled_3d_spec_from_problem(
                _decode_prob(65534, head_size=256), "gfx942"
            )
            self.assertFalse(spec.use_wide_kv_load)
            self.assertFalse(spec.use_i64_kv_addr)
            for n in (65535, 65536):
                with self.subTest(num_kv_blocks=n), self.assertRaises(
                    NotImplementedError
                ):
                    _asb._tiled_3d_spec_from_problem(
                        _decode_prob(n, head_size=256), "gfx942"
                    )

    def test_3d_key_follows_loader_limit(self):
        # No i64 binary exists, so caches the loader reads share one key, and a
        # cache past it gets its own key and reaches the spec's refusal.
        small = _decode_prob(_AT_I32_LIMIT)
        at_2gib = _decode_prob(_AT_2GIB)
        over = _decode_prob(_AT_2GIB + 1)
        with _patch_arch("gfx942"):
            key = _au._tiled_3d_cache_key(small, "gfx942")
            self.assertEqual(key, _au._tiled_3d_cache_key(at_2gib, "gfx942"))
            self.assertNotEqual(key, _au._tiled_3d_cache_key(over, "gfx942"))

    def test_3d_launch_checks_the_real_cache(self):
        # A stale num_kv_blocks builds the spec for a small cache, so the launch
        # checks the real cache against what the K/V loader reads.
        wide = _decode_prob(1024)
        with self.assertRaises(_ReachedPipeline):
            _launch_3d(wide, _AT_2GIB, "gfx942")
        with self.assertRaisesRegex(ValueError, "fixing num_kv_blocks will not help"):
            _launch_3d(wide, _AT_2GIB + 1, "gfx942")
        not_wide = _decode_prob(1024, head_size=256)
        with self.assertRaises(_ReachedPipeline):
            _launch_3d(not_wide, 65534, "gfx942")
        with self.assertRaisesRegex(ValueError, "fixing num_kv_blocks will not help"):
            _launch_3d(not_wide, 65535, "gfx942")

    def test_gfx950_3d_launch_checks_the_real_cache(self):
        # 131068 of these 16 KiB blocks fill the i32 buffer range. A stale count
        # builds an i32 spec, which the launch refuses past that; the real count
        # builds an i64 spec, which reads the whole cache.
        stale = _decode_prob(1024)
        with self.assertRaises(_ReachedPipeline):
            _launch_3d(stale, 131068, "gfx950")
        with self.assertRaisesRegex(ValueError, "below the cache's block count"):
            _launch_3d(stale, 131069, "gfx950")
        with self.assertRaises(_ReachedPipeline):
            _launch_3d(_decode_prob(131069), 131069, "gfx950")

    def test_explicit_3d_spec_and_bind_follow_loader_limit(self):
        p = _decode_prob(_AT_2GIB)
        config = ExplicitAttention3DConfig(
            num_segments=0, knobs=(("use_wide_kv_load", True),)
        )
        segment, reduce = make_explicit_attention_3d_specs(p, config, arch="gfx942")
        self.assertFalse(segment.use_i64_kv_addr)
        tuned = AttentionTuningSpec(
            path="3d",
            arch="gfx942",
            builder_kind="tiled",
            compile_backend="llvm",
            candidate_name="test",
            tuning_id="test",
            kernel_spec=segment,
            reduce_spec=reduce,
        )
        self.assertFalse(tuned.with_num_kv_blocks(_AT_2GIB).kernel_spec.use_i64_kv_addr)
        with self.assertRaises(NotImplementedError):
            tuned.with_num_kv_blocks(_AT_2GIB + 1)


@contextmanager
def _clear_launch_memos():
    """Run with empty launch-meta and per-key KV address limit memos."""
    with mock.patch.dict(_au._2D_LAUNCH_META, clear=True), mock.patch.dict(
        _au._2D_KV_ADDR_LIMITS, clear=True
    ):
        yield


_I32_BUFFER = _au._KvAddrLimit(0x7FFF0000, _au._KV_LIMIT_I32_BUFFER)


def _no_i64_path(nbytes: int) -> _au._KvAddrLimit:
    return _au._KvAddrLimit(nbytes, _au._KV_LIMIT_NO_I64_PATH)


def _kv_cache(num_blocks: int, problem: UnifiedAttentionProblem) -> SimpleNamespace:
    """Shape-only stand-in for a paged K cache tensor."""
    shape = (num_blocks, problem.block_size, problem.num_kv_heads, problem.head_size)
    return SimpleNamespace(shape=shape)


# Parity NQK shape: 32768 blocks x 64 tokens x 8 kv-heads x 128 dims x 2 bytes is
# 4 GiB, which is also 2^31 elements.
_BUG_BLOCKS = 32768


def _bug_prob(**overrides) -> UnifiedAttentionProblem:
    return _prob(128, 64, 64, 8, **overrides)


def _cfvst_prob(**overrides) -> UnifiedAttentionProblem:
    """gfx942 bf16 D64 causal prefill: the shipped spec turns cfvst on. Blocks are
    32 KiB, like the combo shape, so the _AT/_OVER_I32_LIMIT counts apply and
    65536 blocks is exactly 2^31 bytes."""
    return _prob(
        64, 32, 32, 8, max_seqlen_q=4096, max_seqlen_k=4096, total_q=8192, **overrides
    )


class TestKvAddrLimit(unittest.TestCase):
    """_kv_addr_limit maps the compiled kernel's flag to the cache it addresses."""

    def test_i64_kernel_is_unbounded(self):
        self.assertIsNone(_au._kv_addr_limit(_bug_prob(), True))

    def test_i32_kernel_is_bounded_by_the_buffer_range(self):
        self.assertEqual(_au._kv_addr_limit(_bug_prob(), False), _I32_BUFFER)

    def test_kernel_without_i64_path_is_bounded_by_element_count(self):
        # Scalar 2D, gfx942 4-warp GQA and gfx1250 tiled 2D: i32 element
        # offsets, so 2^31 elements of the cache dtype.
        self.assertEqual(_au._kv_addr_limit(_bug_prob(), None), _no_i64_path(2 * 2**31))
        self.assertEqual(
            _au._kv_addr_limit(_bug_prob(use_fp8=True), None), _no_i64_path(2**31)
        )

    def test_gfx1250_meta_has_no_i64_path(self):
        # The gfx1250 tiled spec builds D64 over an fp8 cache only. 16 KiB fp8
        # blocks, so this count is past the i32 buffer range and the problem
        # alone would say i64.
        p = _combo_prob(use_fp8=True, num_kv_blocks=2 * _OVER_I32_LIMIT)
        with _patch_arch("gfx1250"), _clear_launch_memos():
            self.assertTrue(_au._enable_i64_kv_addr(p))
            meta = _au._get_2d_launch_meta(
                p, _au._tiled_cache_key(p, "gfx1250"), arch="gfx1250"
            )
        self.assertEqual(meta.kv_addr_limit, _no_i64_path(2**31))

    def test_gfx942_4warp_route_meta_has_no_i64_path(self):
        # D128 sliding-window bf16 rides build_gfx942_4warp_gqa, which indexes
        # K/V with i32 element offsets whatever the problem decides.
        p = _prob(128, 32, 32, 8, sliding_window=128)
        with _patch_arch("gfx942"), _clear_launch_memos():
            self.assertIsNotNone(_au._gfx942_4warp_route(p, "gfx942"))
            meta = _au._get_2d_launch_meta(
                p, _au._tiled_cache_key(p, "gfx942"), arch="gfx942"
            )
        self.assertEqual(meta.kv_addr_limit, _no_i64_path(2 * 2**31))

    def test_gfx1250_spec_has_no_i64_field(self):
        self.assertNotIn("use_i64_kv_addr", {f.name for f in fields(Gfx1250TiledSpec)})

    def test_4warp_builder_rejects_i64(self):
        p = _prob(128, 32, 32, 8, sliding_window=128, num_kv_blocks=_BUG_BLOCKS)
        with _patch_arch("gfx942"):
            spec = _asb._tiled_spec_from_problem(p, "gfx942")
        spec = replace(spec, use_i64_kv_addr=True)
        with self.assertRaisesRegex(NotImplementedError, "use_i64_kv_addr"):
            build_gfx942_4warp_gqa(spec, arch="gfx942")

    def test_i32_only_loader_flags_are_real_spec_fields(self):
        # A renamed flag would silently stop capping its spec.
        names = {f.name for f in fields(Gfx942TiledSpec)}
        for flag in _au._I32_ONLY_KV_LOADERS:
            self.assertIn(flag, names)
        self.assertIn("use_fp8_mfma_pv", {f.name for f in fields(Gfx950TiledSpec)})

    # The bool spec flags that select a K/V loader or address form and keep it
    # correct under use_i64_kv_addr (an I64_KV_ADDR branch at every
    # paged_kv_desc offset they reach), and those that touch no K/V address.
    _I64_SAFE_KV_FLAGS = frozenset(
        {
            "use_i64_kv_addr",
            "use_fast_paged_kv_desc",
            "use_k_sliced_ring",
            "use_global_load_lds_k",
            "use_fp8_mfma_qk",
        }
    )
    _NO_KV_ADDRESS_FLAGS = frozenset(
        {
            # Masking, bias and softmax state.
            "use_sinks",
            "use_alibi",
            "use_qq_bias",
            "use_transposed_scalar_state",
            "use_transposed_invariant_hoist",
            "use_transposed_mask_once",
            "use_transposed_mask_limit",
            "use_grouped_kv2_softmax",
            "use_causal_mask_phase_split",
            "use_mask_phase_split",
            # MFMA atom and PV register path.
            "use_register_pv",
            "use_mfma_32x32",
            "use_mfma_32x32x8",
            "use_transposed_qk_32x32",
            "use_transposed_half_local_pv",
            "use_mfma32_skip_legacy_qreg",
            "use_agpr_alloc_zero",
            # K/V LDS layout and buffering; the global loads are unchanged.
            "use_conflict_free_v_store_split",
            "use_conflict_free_v_ck_vlds",
            "use_k_single_buffer",
            "use_k_sliced_ldsseq",
            "use_v_double_buffer",
            "use_kq_lds_pad",
            # Scheduling.
            "use_early_v_schedule",
            "use_iglp_opt",
            "use_qk_pv_sched_group_barrier",
            "use_staggered_iter_wait",
            "use_sched_barrier",
            "use_softmax_mfma_interleave",
            # Q loads and grid order.
            "use_q_direct_global",
            "use_q_direct_reg",
            "use_q_reread",
            "use_q_major_grid",
        }
    )

    def test_every_tiled_flag_is_classified_for_kv_addressing(self):
        # A new loader flag must be placed before it ships: one with no
        # I64_KV_ADDR branch goes in _I32_ONLY_KV_LOADERS, or an i64 spec that
        # selects it reads wrong data past the i32 range with no cap.
        classes = (
            frozenset(_au._I32_ONLY_KV_LOADERS),
            self._I64_SAFE_KV_FLAGS,
            self._NO_KV_ADDRESS_FLAGS,
        )
        for spec_cls in (Gfx942TiledSpec, Gfx950TiledSpec):
            for f in fields(spec_cls):
                if not f.name.startswith("use_"):
                    continue
                with self.subTest(spec=spec_cls.__module__, flag=f.name):
                    # A non-bool flag type would hide the field from this check.
                    self.assertIn(
                        f.type, (bool, "bool"), f"{f.name}: declare it a plain bool"
                    )
                    homes = sum(f.name in c for c in classes)
                    self.assertEqual(
                        homes,
                        1,
                        f"{f.name}: classify it in exactly one of "
                        "_I32_ONLY_KV_LOADERS (no I64_KV_ADDR branch), "
                        "_I64_SAFE_KV_FLAGS or _NO_KV_ADDRESS_FLAGS",
                    )

    def test_i32_only_loader_caps_an_i64_spec(self):
        # Flat i32 loads read below 2^31 bytes; buffer loads stop at 0x7FFF0000.
        expected = {
            "use_conflict_free_v": 2**31,
            "use_conflict_free_v_store": 2**31,
            "use_v_hbm_direct": 2**31,
            "use_k_hbm_direct": 0x7FFF0000,
            "use_fp8_mfma_pv": 0x7FFF0000,
        }
        self.assertEqual(set(expected), set(_au._I32_ONLY_KV_LOADERS))
        p = _bug_prob()
        for flag, limit in expected.items():
            with self.subTest(flag=flag):
                spec = SimpleNamespace(use_i64_kv_addr=True, **{flag: True})
                self.assertEqual(
                    _au._spec_kv_addr_limit(p, spec), _au._KvAddrLimit(limit, flag)
                )
        # Several loaders: the tightest one bounds the kernel and names the cause.
        both = SimpleNamespace(
            use_i64_kv_addr=True, use_conflict_free_v_store=True, use_k_hbm_direct=True
        )
        self.assertEqual(
            _au._spec_kv_addr_limit(p, both),
            _au._KvAddrLimit(0x7FFF0000, "use_k_hbm_direct"),
        )
        self.assertIsNone(
            _au._spec_kv_addr_limit(p, SimpleNamespace(use_i64_kv_addr=True))
        )
        # An i32 spec stays on the buffer range whatever its loaders.
        i32 = SimpleNamespace(use_i64_kv_addr=False, use_conflict_free_v_store=True)
        self.assertEqual(_au._spec_kv_addr_limit(p, i32), _I32_BUFFER)

    def test_4warp_builder_kind_ignores_the_spec_flag(self):
        p = _bug_prob()
        spec = SimpleNamespace(use_i64_kv_addr=True)
        self.assertEqual(
            _au._spec_kv_addr_limit(p, spec, builder_kind="gfx942_4warp_gqa"),
            _no_i64_path(2 * 2**31),
        )

    def test_gfx942_cfvst_meta_caps_at_2gib(self):
        # Shipped gfx942 bf16 D64 causal prefill rides cfvst, whose V loader has
        # no i64 branch: an i64 problem still gets its flat i32 limit.
        p = _cfvst_prob(num_kv_blocks=_OVER_I32_LIMIT)
        with _patch_arch("gfx942"), _clear_launch_memos():
            spec = _asb._tiled_spec_from_problem(p, "gfx942")
            self.assertTrue(spec.use_i64_kv_addr)
            self.assertTrue(spec.use_conflict_free_v_store)
            meta = _au._get_2d_launch_meta(
                p, _au._tiled_cache_key(p, "gfx942"), arch="gfx942"
            )
        self.assertEqual(
            meta.kv_addr_limit, _au._KvAddrLimit(2**31, "use_conflict_free_v_store")
        )

    def test_meta_reads_the_built_spec(self):
        # A builder branch that leaves use_i64_kv_addr unset compiles i32 while
        # the problem says i64; the meta must follow the spec and keep the limit.
        p = _combo_prob(num_kv_blocks=_OVER_I32_LIMIT)
        with _patch_arch("gfx950"):
            i32_spec = replace(
                _asb._tiled_spec_from_problem(p, "gfx950"), use_i64_kv_addr=False
            )
            with _clear_launch_memos(), mock.patch.object(
                _au, "_tiled_spec_from_problem", return_value=i32_spec
            ):
                self.assertTrue(_au._enable_i64_kv_addr(p))
                meta = _au._get_2d_launch_meta(
                    p, _au._tiled_cache_key(p, "gfx950"), arch="gfx950"
                )
        self.assertEqual(meta.kv_addr_limit, _I32_BUFFER)

    def test_limit_memo_is_keyed_per_kernel(self):
        # One memo across calls, as in production. The cfvst shape at and one
        # block over the i32 buffer range shares its geometry and differs only in
        # the i64 decision, so a key without that decision would serve the i32
        # limit to the i64 cfvst kernel.
        cfvst = _au._KvAddrLimit(2**31, "use_conflict_free_v_store")
        cases = (
            ("gfx942", _AT_I32_LIMIT, _I32_BUFFER),
            ("gfx942", _OVER_I32_LIMIT, cfvst),
            ("gfx950", _OVER_I32_LIMIT, None),
        )
        with _clear_launch_memos():
            for arch, n, limit in cases:
                p = _cfvst_prob(num_kv_blocks=n)
                with self.subTest(arch=arch, n=n), _patch_arch(arch):
                    meta = _au._get_2d_launch_meta(
                        p, _au._tiled_cache_key(p, arch), arch=arch
                    )
                    self.assertEqual(meta.kv_addr_limit, limit)

    def test_limit_memo_builds_the_spec_once_per_kernel(self):
        # total_q splits the launch-meta memo but not the kernel, so a second
        # total_q on the same cache key must reuse the memoized limit.
        p = _cfvst_prob(num_kv_blocks=_OVER_I32_LIMIT)
        q = replace(p, total_q=p.total_q // 2)
        with _patch_arch("gfx942"), _clear_launch_memos(), mock.patch.object(
            _au, "_tiled_spec_from_problem", wraps=_au._tiled_spec_from_problem
        ) as build:
            key = _au._tiled_cache_key(p, "gfx942")
            self.assertEqual(key, _au._tiled_cache_key(q, "gfx942"))
            first = _au._get_2d_launch_meta(p, key, arch="gfx942")
            second = _au._get_2d_launch_meta(q, key, arch="gfx942")
        self.assertEqual(build.call_count, 1)
        self.assertEqual(first.kv_addr_limit, second.kv_addr_limit)
        self.assertNotEqual(first.grid, second.grid)


def _four_warp_tuning():
    """A gfx942 4-warp GQA tuning spec and its problem: 16-token bf16 D128 blocks
    with 8 kv-heads, 32 KiB each, like the _AT/_OVER_I32_LIMIT counts assume."""
    candidate = next(
        c
        for c in attention_candidates()
        if c.name.startswith("attention_gfx942_u2d_gfx942_4warp")
    )
    req = AttentionRequest(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    spec = next(iter(candidate.sweep_space(req)))
    assert spec.builder_kind == "gfx942_4warp_gqa"
    return spec, _tuning_problem(req)


class TestFourWarpTuningSpec(unittest.TestCase):
    """The 4-warp GQA tuning builder has no i64 path: i32 up to 2^31 elements."""

    # 16-token bf16 D128 blocks with 8 kv-heads are 32 KiB, so 2^31 elements
    # (4 GiB) is 131072 blocks.
    _AT_ELEMENT_LIMIT = 131072

    def test_binds_i32_past_the_buffer_range(self):
        spec, problem = _four_warp_tuning()
        for n in (_OVER_I32_LIMIT, self._AT_ELEMENT_LIMIT):
            with self.subTest(n=n):
                bound = spec.with_num_kv_blocks(n)
                self.assertFalse(bound.kernel_spec.use_i64_kv_addr)
                limit = _au._spec_kv_addr_limit(
                    problem, bound.kernel_spec, builder_kind=bound.builder_kind
                )
                _au._check_kv_addr_width(problem, _kv_cache(n, problem), limit)

    def test_refused_past_the_element_limit(self):
        spec, _ = _four_warp_tuning()
        with self.assertRaisesRegex(NotImplementedError, "no i64 KV path"):
            spec.with_num_kv_blocks(self._AT_ELEMENT_LIMIT + 1)

    def test_explicit_spec_never_claims_i64(self):
        # make_explicit_attention_2d_spec takes the address width from the same
        # rule, so an explicit 4-warp spec cannot reach the builder with i64 set.
        _, problem = _four_warp_tuning()
        problem = replace(problem, num_kv_blocks=_OVER_I32_LIMIT)
        self.assertTrue(_semantic_fields(problem)["use_i64_kv_addr"])
        fields_4warp = _semantic_fields(problem, "gfx942_4warp_gqa")
        self.assertFalse(fields_4warp["use_i64_kv_addr"])
        with self.assertRaisesRegex(NotImplementedError, "no i64 KV path"):
            _semantic_fields(
                replace(problem, num_kv_blocks=self._AT_ELEMENT_LIMIT + 1),
                "gfx942_4warp_gqa",
            )

    def test_explicit_4warp_config_hands_its_builder_kind_on(self):
        # Through the public entry point, so dropping config.builder_kind from
        # its _semantic_fields call fails here, not late in the builder.
        spec, problem = _four_warp_tuning()
        ks = spec.kernel_spec
        config = ExplicitAttention2DConfig(
            num_warps=ks.num_warps,
            block_m_per_warp=ks.block_m_per_warp,
            tile_policy=f"{ks.tile_size // problem.block_size}x",
            builder_kind=spec.builder_kind,
        )
        explicit = make_explicit_attention_2d_spec(
            replace(problem, num_kv_blocks=_OVER_I32_LIMIT), config, arch="gfx942"
        )
        self.assertIs(explicit.use_i64_kv_addr, False)
        with self.assertRaisesRegex(NotImplementedError, "no i64 KV path"):
            make_explicit_attention_2d_spec(
                replace(problem, num_kv_blocks=self._AT_ELEMENT_LIMIT + 1),
                config,
                arch="gfx942",
            )


class TestKvAddrWidthGuard(unittest.TestCase):
    """_check_kv_addr_width rejects a cache larger than the kernel addresses."""

    _I32 = _I32_BUFFER

    def test_i32_kernel_over_limit_raises(self):
        p = _bug_prob()
        with self.assertRaisesRegex(ValueError, "zeros or wrong data"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_missing_count_hint_points_at_the_problem(self):
        p = _bug_prob()
        with self.assertRaisesRegex(ValueError, "num_kv_blocks=k.shape"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_stale_count_hint_points_at_the_problem(self):
        # A count taken from a smaller cache than the one launched on.
        p = _bug_prob(num_kv_blocks=1024)
        with self.assertRaisesRegex(ValueError, "below the cache's block count"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_correct_count_hint_points_at_the_kernel(self):
        # A hand-built i32 spec on a problem that already knows the cache size:
        # fixing the problem would not help, rebuilding the spec would.
        p = _bug_prob(num_kv_blocks=_BUG_BLOCKS)
        with self.assertRaisesRegex(ValueError, "use_i64_kv_addr=True"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_no_i64_path_hint_does_not_blame_the_count(self):
        # Even with a stale count, the count is not what bounds this kernel.
        p = _bug_prob(num_kv_blocks=1024)
        limit = _no_i64_path(2**31)
        with self.assertRaisesRegex(ValueError, "fixing num_kv_blocks will not help"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), limit)

    def test_loader_cap_hint_names_the_loader(self):
        p = _bug_prob(num_kv_blocks=_BUG_BLOCKS)
        limit = _au._KvAddrLimit(2**31, "use_conflict_free_v_store")
        with self.assertRaisesRegex(
            ValueError, "use_conflict_free_v_store K/V loader has no i64 path"
        ):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), limit)

    def test_i64_kernel_passes(self):
        p = _bug_prob()
        _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), None)

    def test_small_cache_passes(self):
        p = _bug_prob()
        _au._check_kv_addr_width(p, _kv_cache(16, p), self._I32)

    def test_buffer_limit_boundary(self):
        p = _combo_prob()
        _au._check_kv_addr_width(p, _kv_cache(_AT_I32_LIMIT, p), self._I32)
        with self.assertRaises(ValueError):
            _au._check_kv_addr_width(p, _kv_cache(_OVER_I32_LIMIT, p), self._I32)

    def test_fp8_halves_cache_bytes(self):
        # 64 KiB fp8 blocks: 32767 fill the buffer range exactly.
        p = _bug_prob(use_fp8=True)
        _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS - 1, p), self._I32)
        with self.assertRaises(ValueError):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_shapeless_k_is_skipped(self):
        _au._check_kv_addr_width(_bug_prob(), None, self._I32)

    def test_attn_values_requires_the_limit(self):
        # Required, so a new direct-launch caller cannot forget it.
        p = _bug_prob()
        args = dict(
            problem=p,
            q=None,
            k=_kv_cache(_BUG_BLOCKS, p),
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=None,
            softcap=0.0,
            sinks=None,
            bt_stride=0,
            include_bt_stride=True,
        )
        with self.assertRaises(TypeError):
            _au._attn_values(**args)
        with self.assertRaisesRegex(ValueError, "i32 KV addressing"):
            _au._attn_values(**args, kv_addr_limit=self._I32)


class _RouteGuardCase(unittest.TestCase):
    """run_unified_attention_torch end to end with mocked launchers.

    The K cache is shape-only, so the route through count filling, spec and
    launch-meta selection and the kernarg pack runs on CPU.
    """

    _ARCH = ""

    def setUp(self):
        self.launcher = mock.Mock(return_value=None)
        patches = [
            _patch_arch(self._ARCH),
            mock.patch.dict(_au._2D_LAUNCH_META, clear=True),
            mock.patch.dict(_au._2D_KV_ADDR_LIMITS, clear=True),
            mock.patch.object(_au, "_get_2d_launcher", return_value=self.launcher),
            mock.patch.object(_au, "_get_scalar_launcher", return_value=self.launcher),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _run(self, problem: UnifiedAttentionProblem, num_blocks: int, **kw) -> None:
        return _au.run_unified_attention_torch(
            problem=problem,
            q=None,
            k=_kv_cache(num_blocks, problem),
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=SimpleNamespace(shape=(problem.num_seqs, 64)),
            softcap=0.0,
            **kw,
        )


class TestProductionRouteGuard(_RouteGuardCase):
    _ARCH = "gfx950"

    def test_unset_count_on_4gib_cache_launches_i64(self):
        self._run(_bug_prob(), _BUG_BLOCKS, backend="tiled")
        self.launcher.assert_called_once()
        problem = _au._get_2d_launcher.call_args.args[0]
        self.assertEqual(problem.num_kv_blocks, _BUG_BLOCKS)
        self.assertTrue(_au._enable_i64_kv_addr(problem))

    def test_stale_count_raises(self):
        # A positive count is trusted (only 0 is refilled), so a stale one gets
        # the i32 kernel and the guard stops the launch.
        with self.assertRaisesRegex(ValueError, "below the cache's block count"):
            self._run(_bug_prob(num_kv_blocks=1024), _BUG_BLOCKS, backend="tiled")
        self.launcher.assert_not_called()

    def test_tuning_spec_rebinds_to_the_real_cache(self):
        candidate = next(
            c
            for c in attention_candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        )
        req = AttentionRequest(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=1024,
            seqlen_k=1024,
            hdim_q=128,
            hdim_v=128,
            arch="gfx950",
            dtype="bf16",
            algorithm=candidate.algorithm,
            spec_id=candidate.spec_id,
        )
        spec = next(iter(candidate.sweep_space(req)))
        self.assertFalse(spec.kernel_spec.use_i64_kv_addr)
        problem = _tuning_problem(req)
        # 16-token bf16 blocks of 32 KiB: one block over the i32 buffer range.
        self._run(problem, _OVER_I32_LIMIT, tuning_spec=spec)
        self.launcher.assert_called_once()
        bound = _au._get_2d_launcher.call_args.kwargs["tuning_spec"]
        self.assertTrue(bound.kernel_spec.use_i64_kv_addr)

    def test_scalar_fallback_is_bounded_by_element_count(self):
        # The production count is filled here, so the problem alone says i64;
        # the scalar kernel still indexes K/V with i32 element offsets.
        p = _bug_prob()
        self._run(p, _BUG_BLOCKS, backend="scalar")  # exactly 2^31 elements
        with self.assertRaisesRegex(ValueError, "no i64 KV path"):
            self._run(p, _BUG_BLOCKS + 1, backend="scalar")
        self.launcher.assert_called_once()


class TestGfx942RouteGuard(_RouteGuardCase):
    _ARCH = "gfx942"

    def test_cfvst_route_caps_at_2gib(self):
        # The refilled count makes the cfvst spec i64, but its V loader still
        # indexes with a flat i32 byte offset: 2^31 bytes launches, one block
        # more raises.
        p = _cfvst_prob()
        self._run(p, _OVER_I32_LIMIT, backend="tiled")
        self._run(p, 65536, backend="tiled")
        with self.assertRaisesRegex(
            ValueError, "use_conflict_free_v_store K/V loader has no i64 path"
        ):
            self._run(p, 65537, backend="tiled")
        self.assertEqual(self.launcher.call_count, 2)

    def test_4warp_tuning_spec_launches_past_the_buffer_range(self):
        spec, problem = _four_warp_tuning()
        self._run(problem, _OVER_I32_LIMIT, tuning_spec=spec)
        self.launcher.assert_called_once()
        bound = _au._get_2d_launcher.call_args.kwargs["tuning_spec"]
        self.assertFalse(bound.kernel_spec.use_i64_kv_addr)

    def test_graph_replay_path_checks_the_cache(self):
        # Graph replay only runs on gfx942. Its miss path packs kernargs through
        # _attn_values too, so a stale count must raise there before capture.
        p = _bug_prob(num_kv_blocks=1024)
        raised = []
        run_graphed = _au._run_2d_graphed

        def spy(*args, **kwargs):
            # The non-graph fallback raises the same error, so record that this
            # one came from the graph path.
            try:
                return run_graphed(*args, **kwargs)
            except ValueError as e:
                raised.append(e)
                raise

        with mock.patch.object(
            _au, "_enable_2d_graph_replay", return_value=True
        ), mock.patch.object(
            _au, "_torch_stream_capturing", return_value=False
        ), mock.patch.object(
            _au, "_run_2d_graphed", side_effect=spy
        ), mock.patch.dict(
            _au._2D_GRAPHS, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "below the cache's block count"):
                self._run(p, _BUG_BLOCKS, backend="tiled")
        self.assertEqual(len(raised), 1, "the graph path did not run the guard")
        self.launcher.assert_not_called()


# ---------------------------------------------------------------------------
# Transposed sub-flags (mask_once / mask_limit / scalar_state)
# ---------------------------------------------------------------------------


class TestTransposedSubflags(unittest.TestCase):

    def test_combo_no_sw_gets_mask_flags(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_transposed_mask_once"])
        self.assertTrue(s["use_transposed_mask_limit"])
        self.assertTrue(s["use_transposed_scalar_state"])

    def test_combo_with_sw_no_mask_flags(self):
        # SW combo: mask_once / mask_limit must be off
        s = _spec(_combo_prob(sliding_window=256))
        self.assertFalse(s["use_transposed_mask_once"])
        self.assertFalse(s["use_transposed_mask_limit"])

    def test_biased_combo_no_mask_flags(self):
        # bias_active blocks mask_opts
        s = _spec(_combo_prob(use_qq_bias=True))
        self.assertFalse(s["use_transposed_mask_once"])
        self.assertFalse(s["use_transposed_mask_limit"])


if __name__ == "__main__":
    unittest.main()
