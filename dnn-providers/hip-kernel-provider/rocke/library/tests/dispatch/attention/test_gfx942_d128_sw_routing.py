# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for gfx942 D128 sliding-window prefill routing.

D128 sliding-window (SWA) prefill on gfx942 is routed to the 4-warp natural-QK
paged kernel (``build_gfx942_4warp_gqa`` -- the same kernel already in develop
for the D256 fast path), NOT the non-ring wide flash. The 4-warp kernel applies
the windowed mask + windowed KV-skip in-kernel and is ~4x faster than the
non-ring wide flash at block_size 16 (narrowing to ~1.2-1.3x at block_size 64;
MI300X, GQA 32/8, window 4096). These tests guard that routing:

  * bf16 + fp16 D128 SW (bs 16/32/64) take the 4-warp cohort
    (``_gfx942_4warp_fast`` / ``_d128_gfx942_swa_fast``),
  * the spec is the 4-warp *discriminator* (num_warps=1, block_m_per_warp=32,
    no mfma_32x32 / transposed_qk / single-buffer, ring off) -- NOT the flash
    spec (the prior PR opened the flash gate for D128 SW, kept only as the
    fallback for SW edge cases the 4-warp excludes),
  * the D128 SW spec emits through the 4-warp builder without a validator error,
  * D128 *causal* (both dtypes) stays on develop's flash/ring path -- NOT the
    4-warp cohort (no regression),
  * D64 SW stays off the D128 4-warp cohort, and
  * D256 bf16 causal still takes the 4-warp cohort (no regression).

Arch is pinned via ``_RESOLVED_ATTENTION_ARCH`` (memoized process-wide and
monkeypatched wholesale), so these run on any host without a gfx942 GPU.
"""

from __future__ import annotations

import unittest

import kernels.common.attention_unified as au
from kernels.common.attention_unified import _tiled_spec_from_problem
from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa


class _PinArch:
    """Context-manager that pins ``_RESOLVED_ATTENTION_ARCH`` to ``arch``."""

    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self.arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _d128_problem(**kw) -> "au.UnifiedAttentionProblem":
    """A Mistral-7B-like D128 GQA sliding-window prefill problem."""
    base = dict(
        total_q=8192,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=8192,
        max_seqlen_k=8192,
        dtype="bf16",
        sliding_window=4096,
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


class TestGfx942D128SwRouting(unittest.TestCase):
    def test_sw_routes_to_4warp(self):
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                p = _d128_problem(dtype=dt)
                self.assertTrue(
                    au._d128_gfx942_swa_fast(p),
                    msg=f"D128 SW {dt} must be the SW cohort",
                )
                self.assertTrue(
                    au._gfx942_4warp_fast(p),
                    msg=f"D128 SW {dt} must take the 4-warp path",
                )

    def test_sw_dispatch_gate_accepts_both_dtypes(self):
        # The production dispatcher (run_unified_attention_torch path="auto")
        # routes to the scalar kernel unless supports_native_unified_attention_tiled
        # returns True. The 4-warp SELECTOR firing (test_sw_routes_to_4warp) is not
        # enough: fp16 rode the flash branch, which passes use_d256_fast=True into
        # supports_tiled_2d, whose d256-only acceptance rejected head_size=128 ->
        # fp16 D128 SW silently fell back to the scalar kernel (~thousands x slower,
        # still correct). Guard the actual gate for BOTH dtypes / every block_size.
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                for bs in (16, 32, 64):
                    p = _d128_problem(dtype=dt, block_size=bs)
                    ok, reason = au.supports_native_unified_attention_tiled(p)
                    self.assertTrue(
                        ok,
                        msg=f"D128 SW {dt} bs{bs} must reach the tiled path: {reason}",
                    )

    def test_sw_ineligible_falls_back_not_4warp_gate(self):
        # Configs the 4-warp cohort excludes must NOT be admitted by the gate as a
        # 4-warp problem (they route to develop's wide-flash/narrow or scalar).
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                p = _d128_problem(dtype=dt, softcap=30.0)
                self.assertFalse(
                    au._gfx942_4warp_fast(p), msg=f"{dt} softcap must not take 4-warp"
                )

    def test_sw_spec_is_4warp_discriminator(self):
        # The 4-warp cohort must get the discriminator spec, NOT the flash spec.
        # (Guards the attention_spec_builder flash-branch exclusion: a flash spec
        # here builds num_warps=2 / single-buffer, which the __post_init__
        # validator rejects for fp16 bs16/32.)
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                for bs, exp_tile in ((16, 32), (32, 32), (64, 64)):
                    p = _d128_problem(dtype=dt, block_size=bs)
                    spec = _tiled_spec_from_problem(p)
                    tag = f"{dt} bs{bs}"
                    self.assertEqual(spec.num_warps, 1, msg=f"{tag} num_warps")
                    self.assertEqual(spec.block_m_per_warp, 32, msg=f"{tag} block_m")
                    self.assertEqual(spec.tile_size, exp_tile, msg=f"{tag} tile_size")
                    self.assertFalse(spec.use_mfma_32x32x8, msg=f"{tag} mfma_32x32")
                    self.assertFalse(
                        spec.use_transposed_qk_32x32, msg=f"{tag} transposed_qk"
                    )
                    self.assertFalse(
                        spec.use_k_single_buffer, msg=f"{tag} single_buffer"
                    )
                    self.assertFalse(spec.use_k_sliced_ring, msg=f"{tag} ring")
                    self.assertEqual(spec.sliding_window, 4096, msg=f"{tag} window")

    def test_sw_spec_emits_through_4warp_builder(self):
        # Building the kernel exercises the spec __post_init__ validators (which
        # raise on an SW-incompatible flag combo) and the 4-warp emitter.
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                for bs in (16, 32, 64):
                    for sq in (8192, 16384):
                        p = _d128_problem(
                            dtype=dt,
                            block_size=bs,
                            max_seqlen_q=sq,
                            max_seqlen_k=sq,
                            total_q=sq,
                        )
                        spec = _tiled_spec_from_problem(p)
                        build_gfx942_4warp_gqa(
                            spec, arch="gfx942"
                        )  # raises on bad combo

    def test_causal_stays_off_4warp(self):
        # D128 causal stays on develop's flash/ring path -- NOT the 4-warp cohort.
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                pc = _d128_problem(dtype=dt, sliding_window=0)
                self.assertFalse(
                    au._d128_gfx942_swa_fast(pc),
                    msg=f"D128 causal {dt} must not be SW cohort",
                )
                self.assertFalse(
                    au._gfx942_4warp_fast(pc),
                    msg=f"D128 causal {dt} must not take 4-warp",
                )
            # develop's causal routing is unchanged: bf16 flash/ring-off, fp16 flash/ring-on.
            pb = _d128_problem(dtype="bf16", sliding_window=0)
            self.assertTrue(au._enable_gfx942_bf16_flash(pb))
            self.assertFalse(au._enable_gfx942_flash_k_sliced_ring(pb))
            pf = _d128_problem(dtype="fp16", sliding_window=0)
            self.assertTrue(au._enable_gfx942_fp16_flash(pf))
            self.assertTrue(au._enable_gfx942_flash_k_sliced_ring(pf))

    def test_d64_sw_not_4warp(self):
        # Only D128 SW takes the 4-warp cohort; D64 SW does not.
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                p = _d128_problem(dtype=dt, head_size=64)
                self.assertFalse(au._d128_gfx942_swa_fast(p))

    def test_d256_causal_still_4warp(self):
        # Regression: the D256 bf16 causal fast path still routes to the 4-warp.
        with _PinArch("gfx942"):
            p = au.UnifiedAttentionProblem(
                total_q=8192,
                num_seqs=1,
                num_query_heads=16,
                num_kv_heads=2,
                head_size=256,
                block_size=16,
                max_seqlen_q=8192,
                max_seqlen_k=8192,
                dtype="bf16",
                sliding_window=0,
            )
            self.assertTrue(au._d256_gfx942_fast(p))
            self.assertTrue(au._gfx942_4warp_fast(p))

    def test_sw_variants_route_4warp(self):
        # The cohort opens D128 SW 4-warp for every block_size / num_seqs / seqlen.
        with _PinArch("gfx942"):
            variants = (
                dict(block_size=64),  # real production paged block size
                dict(num_seqs=2, total_q=16384),  # multi-batch
                dict(
                    max_seqlen_q=512, max_seqlen_k=512, total_q=512
                ),  # short (Sq < window)
            )
            for dt in ("bf16", "fp16"):
                for kw in variants:
                    p = _d128_problem(dtype=dt, **kw)
                    self.assertTrue(
                        au._gfx942_4warp_fast(p), msg=f"{dt} {kw} must take 4-warp"
                    )
                    spec = _tiled_spec_from_problem(p)
                    self.assertEqual(spec.num_warps, 1, msg=f"{dt} {kw} num_warps")
                    self.assertFalse(spec.use_k_sliced_ring, msg=f"{dt} {kw} ring")
                    build_gfx942_4warp_gqa(spec, arch="gfx942")

    def test_sw_ineligible_stays_off_4warp(self):
        # D128 SW configs the 4-warp cohort excludes (softcap != 0; block_size not
        # in {16,32,64}) must NOT take the 4-warp -- they fall back to develop's
        # wide-flash / narrow path. Locks in the scope boundary: a future selector
        # change that let the 4-warp fire for an unsupported config would fail here.
        with _PinArch("gfx942"):
            for kw in (dict(softcap=30.0), dict(block_size=128)):
                for dt in ("bf16", "fp16"):
                    p = _d128_problem(dtype=dt, **kw)
                    self.assertFalse(
                        au._d128_gfx942_swa_fast(p),
                        msg=f"{dt} {kw} must not be SW cohort",
                    )
                    self.assertFalse(
                        au._gfx942_4warp_fast(p), msg=f"{dt} {kw} must not take 4-warp"
                    )
                    # Fallback still produces a buildable spec for configs within the
                    # tiled block_size domain (softcap only excludes the 4-warp cohort,
                    # not the tiled path). block_size=128 is outside {16,32,64}: the
                    # gate rejects it outright and no tiled spec exists, so skip it.
                    if p.block_size in (16, 32, 64):
                        _tiled_spec_from_problem(p)

    def test_route_descriptor_matches_selectors(self):
        # The _TiledRoute descriptor is the single source of truth: its disc
        # knobs + real launch geometry MUST match the concrete values the
        # per-selector functions return for every in-cohort problem, and
        # route-is-not-None MUST track _gfx942_4warp_fast. Guards the #2/#3
        # "cache-key discriminator / launch geometry can never silently
        # disagree" contract: a future edit to a selector constant or to the
        # descriptor that breaks the correspondence fails here.
        with _PinArch("gfx942"):
            incohort = (
                _d128_problem(head_size=256, sliding_window=0, block_size=16),
                _d128_problem(head_size=256, sliding_window=0, block_size=32),
                _d128_problem(dtype="bf16", block_size=16),
                _d128_problem(dtype="fp16", block_size=64),
            )
            for p in incohort:
                route = au._gfx942_4warp_route(p)
                tag = f"d{p.head_size} {p.dtype} bs{p.block_size} sw{p.sliding_window}"
                self.assertIsNotNone(route, msg=tag)
                self.assertTrue(au._gfx942_4warp_fast(p), msg=tag)
                self.assertIs(route.builder, build_gfx942_4warp_gqa, msg=tag)
                # Discriminator knobs + real geometry pinned to concrete values.
                self.assertEqual(route.disc["num_warps"], 1, msg=tag)
                self.assertEqual(
                    route.disc["tile_size"], max(32, p.block_size), msg=tag
                )
                self.assertEqual(route.disc["block_m_per_warp"], 32, msg=tag)
                self.assertEqual(route.block_m, 128, msg=tag)
                self.assertEqual(route.block_dim, (256, 1, 1), msg=tag)
                # Selectors must read the descriptor (no silent divergence).
                self.assertEqual(
                    au._select_2d_num_warps(p), route.disc["num_warps"], msg=tag
                )
                self.assertEqual(
                    au._select_2d_tile_size(p), route.disc["tile_size"], msg=tag
                )
                self.assertEqual(
                    au._select_2d_block_m_per_warp(p),
                    route.disc["block_m_per_warp"],
                    msg=tag,
                )
            # Out-of-cohort (D128 causal): route is None and the boolean agrees.
            off = _d128_problem(dtype="bf16", sliding_window=0)
            self.assertIsNone(au._gfx942_4warp_route(off))
            self.assertFalse(au._gfx942_4warp_fast(off))


if __name__ == "__main__":
    unittest.main()
