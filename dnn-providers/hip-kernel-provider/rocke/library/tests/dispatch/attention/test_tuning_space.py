# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dependency-valid, unique, complete attention tuning space."""

from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace
from itertools import islice

from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    attention_execution_candidates,
    dispatch_attention,
    dispatch_attention_all,
)
from dispatch.attention.tuning_common import (
    _CODEPATH_KNOBS,
    KNOWN_WRONG_KNOBS,
    AttentionGeometryVariant,
    _gfx950_tuning_lds_bytes,
    _supports_tuning_spec,
    tuning_axes,
)
from dispatch.attention.tuning_specs import _SEMANTIC_FIELDS
from kernels.common.attention_unified import _tiled_2d_impl, _tiled_3d_impl
from kernels.gfx942.attention_tiled_2d import UnifiedAttention2DTiledSpec as Gfx942Spec
from kernels.gfx950.attention_tiled_2d import UnifiedAttention2DTiledSpec as Gfx950Spec


def _request(arch="gfx950", **kw):
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="bf16" if arch == "gfx950" else "fp16",
    )
    base.update(kw)
    return AttentionRequest(**base)


def _specs_for(prefix, n=200, **req_kw):
    """The first ``n`` specs of the (possibly million-spec) sweep stream."""
    candidate = next(c for c in attention_candidates() if c.name.startswith(prefix))
    req = replace(
        _request(**req_kw),
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    return candidate, req, tuple(islice(candidate.sweep_space(req), n))


def _sampled(prefix, n, seed=0, **req_kw):
    candidate = next(c for c in attention_candidates() if c.name.startswith(prefix))
    req = replace(
        _request(**req_kw),
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    return tuple(candidate.sample_space(req, n, seed))


_GEOMETRY_FIELDS = frozenset(
    {
        "num_warps",
        "block_m_per_warp",
        "tile_size",
        "waves_per_eu",
        "num_segments",
        "tile_size_override",
    }
)


_GFX950_2D_VARIANT = AttentionGeometryVariant(
    arch="gfx950",
    path="2d",
    codepath="narrow",
    builder_kind="tiled",
    tile_policy="8x",
)


class TestTuningSpace(unittest.TestCase):
    def test_every_kernel_tuning_field_is_swept(self):
        for arch in ("gfx942", "gfx950"):
            for path, spec_type in (
                ("2d", _tiled_2d_impl(arch)[0]),
                ("3d", _tiled_3d_impl(arch)[0]),
            ):
                with self.subTest(arch=arch, path=path):
                    swept = {
                        name
                        for axis in tuning_axes(arch, path)
                        for choice in axis.choices
                        for name, _value in choice
                    }
                    codepath = {
                        name
                        for (owner, _cp), knobs in _CODEPATH_KNOBS.items()
                        if owner == arch
                        for name in knobs
                    }
                    fields = {f.name for f in dataclasses.fields(spec_type)}
                    missing = (
                        fields
                        - _SEMANTIC_FIELDS
                        - _GEOMETRY_FIELDS
                        - swept
                        - codepath
                        - KNOWN_WRONG_KNOBS[arch]
                    )
                    self.assertFalse(missing, sorted(missing))

    def test_sampling_draws_distinct_reproducible_specs(self):
        prefix = "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm"
        first = _sampled(prefix, 32, seed=3, arch="gfx942")
        again = _sampled(prefix, 32, seed=3, arch="gfx942")
        other = _sampled(prefix, 32, seed=4, arch="gfx942")
        ids = [s.tuning_id for s in first]
        self.assertEqual(len(ids), 32)
        self.assertEqual(len(set(ids)), 32)
        self.assertEqual(ids, [s.tuning_id for s in again])
        self.assertNotEqual(ids, [s.tuning_id for s in other])

    def test_sampling_stops_at_a_small_space(self):
        specs = _sampled(
            "attention_gfx950_u3d_splitkv_seg64_t1xb",
            256,
            seqlen_q=1,
            seqlen_k=4096,
        )
        self.assertEqual(len(specs), 5)

    def test_tuning_ids_are_unique_and_hashed(self):
        _candidate, _req, specs = _specs_for(
            "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm"
        )
        ids = [s.tuning_id for s in specs]
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all("@" in tid for tid in ids))

    def test_pinned_tuning_id_roundtrips(self):
        candidate, req, specs = _specs_for(
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm"
        )
        pinned = specs[0].tuning_id
        again = candidate.select_spec(replace(req, attention_tuning_id=pinned))
        self.assertEqual(again.tuning_id, pinned)
        results = dispatch_attention_all(
            replace(req, attention_tuning_id=pinned),
            candidate_prefix=candidate.name,
        )
        matching = [r for r in results if r.spec.tuning_id == pinned]
        self.assertEqual(len(matching), 1)

    def test_runtime_cache_size_refreshes_i64_spec_and_id(self):
        _candidate, _req, specs = _specs_for(
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm"
        )
        base = specs[0]
        at_limit = base.with_num_kv_blocks(65536)
        above_limit = base.with_num_kv_blocks(65537)
        self.assertFalse(at_limit.kernel_spec.use_i64_kv_addr)
        self.assertEqual(at_limit.tuning_id, base.tuning_id)
        self.assertTrue(above_limit.kernel_spec.use_i64_kv_addr)
        self.assertNotEqual(above_limit.tuning_id, base.tuning_id)
        self.assertEqual(above_limit.num_kv_blocks, 65537)

        _candidate, _req, split_specs = _specs_for(
            "attention_gfx950_u3d_splitkv_seg64_t1xb",
            seqlen_q=1,
            seqlen_k=4096,
        )
        split = split_specs[0]
        split_i64 = split.with_num_kv_blocks(65537)
        self.assertTrue(split_i64.kernel_spec.use_i64_kv_addr)
        self.assertEqual(split_i64.reduce_spec, split.reduce_spec)
        self.assertNotEqual(split_i64.tuning_id, split.tuning_id)

        _candidate, _req, gfx942_split_specs = _specs_for(
            "attention_gfx942_u3d_splitkv_seg64_t1xb",
            arch="gfx942",
            seqlen_q=1,
            seqlen_k=4096,
        )
        with self.assertRaisesRegex(NotImplementedError, "does not support"):
            gfx942_split_specs[0].with_num_kv_blocks(65537)

        _candidate, _req, fp8_specs = _specs_for(
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
            use_fp8=True,
            fp8_fnuz=False,
        )
        self.assertFalse(
            fp8_specs[0].with_num_kv_blocks(131072).kernel_spec.use_i64_kv_addr
        )
        self.assertTrue(
            fp8_specs[0].with_num_kv_blocks(131073).kernel_spec.use_i64_kv_addr
        )

    def test_tuning_wrapper_preserves_fp8_encoding(self):
        gfx950_candidate, _req, gfx950_specs = _specs_for(
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
            use_fp8=True,
            fp8_fnuz=False,
        )
        _gfx942_candidate, _req, gfx942_specs = _specs_for(
            "attention_gfx942_u3d_splitkv_seg64_t1xb",
            arch="gfx942",
            seqlen_q=1,
            seqlen_k=4096,
            use_fp8=True,
            fp8_fnuz=True,
        )
        self.assertTrue(gfx950_specs)
        self.assertTrue(gfx942_specs)
        self.assertFalse(gfx950_specs[0].fp8_fnuz)
        self.assertTrue(gfx942_specs[0].fp8_fnuz)
        self.assertNotIn("fnuz", gfx950_specs[0].kernel_name())
        self.assertIn("fnuz", gfx942_specs[0].kernel_name())
        with self.assertRaisesRegex(ValueError, "requires OCP"):
            gfx950_candidate.built(replace(gfx950_specs[0], fp8_fnuz=True), "gfx950")

    def test_narrow_gfx942_never_offers_k_hbm_direct(self):
        _c, _req, specs = _specs_for(
            "attention_gfx942_u2d_narrow_nw2_mw16_t4xb_llvm", arch="gfx942"
        )
        self.assertTrue(specs)
        self.assertFalse(any(s.kernel_spec.use_k_hbm_direct for s in specs))

    def test_k_hbm_direct_is_rejected_on_narrow_gfx942_specs(self):
        with self.assertRaisesRegex(ValueError, "transposed-x8"):
            Gfx942Spec(
                head_size=128,
                block_size=16,
                num_query_heads=32,
                num_kv_heads=8,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
                num_warps=2,
                block_m_per_warp=16,
                tile_size=64,
                use_k_hbm_direct=True,
            )

    def test_no_khbm_plus_sliced_ring(self):
        with self.assertRaisesRegex(ValueError, "k_sliced_ring"):
            Gfx942Spec(
                head_size=128,
                block_size=16,
                num_query_heads=32,
                num_kv_heads=8,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
                num_warps=2,
                block_m_per_warp=32,
                tile_size=64,
                use_mfma_32x32x8=True,
                use_transposed_qk_32x32=True,
                use_conflict_free_v_store=True,
                use_k_sliced_ring=True,
                ring_depth=2,
                k_slice_hd=32,
                use_k_hbm_direct=True,
            )

    def test_transposed_x8_omits_k_hbm_direct(self):
        _candidate, _req, specs = _specs_for(
            "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm",
            arch="gfx942",
        )
        self.assertTrue(specs)
        self.assertFalse(any(s.kernel_spec.use_k_hbm_direct for s in specs))

    def test_sampling_reaches_independent_knobs(self):
        specs = _sampled("attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm", 256)
        flags = {
            "use_q_reread": False,
            "use_q_direct_reg": False,
            "use_v_double_buffer": False,
            "use_k_single_buffer": False,
            "use_grouped_kv2_softmax": False,
        }
        for spec in specs:
            for name in flags:
                if getattr(spec.kernel_spec, name):
                    flags[name] = True
        self.assertTrue(all(flags.values()), flags)

    def test_gfx950_lds_gate_matches_codegen_limit(self):
        spec = Gfx950Spec(
            head_size=128,
            block_size=16,
            num_query_heads=32,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_warps=8,
            block_m_per_warp=16,
            tile_size=128,
        )
        from rocke.core.arch import ArchTarget

        budget = ArchTarget.from_gfx("gfx950").lds_capacity_bytes
        base_bytes = _gfx950_tuning_lds_bytes(spec)
        self.assertGreater(base_bytes, budget)
        ok, why = _supports_tuning_spec(_GFX950_2D_VARIANT, spec)
        self.assertFalse(ok)
        self.assertIn("LDS budget", why)
        self.assertGreater(
            _gfx950_tuning_lds_bytes(replace(spec, use_v_double_buffer=True)),
            base_bytes,
        )

    def test_gfx950_padded_k_rejects_q_alias(self):
        spec = Gfx950Spec(
            head_size=128,
            block_size=16,
            num_query_heads=32,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_warps=2,
            block_m_per_warp=32,
            tile_size=64,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_k_single_buffer=True,
            use_kq_lds_pad=True,
            kq_lds_pad_halves=16,
        )
        ok, why = _supports_tuning_spec(_GFX950_2D_VARIANT, spec)
        self.assertFalse(ok)
        self.assertIn("does not support aliased Q", why)
        ok, why = _supports_tuning_spec(
            _GFX950_2D_VARIANT, replace(spec, use_q_direct_reg=True)
        )
        self.assertTrue(ok, why)
        ok, why = _supports_tuning_spec(
            _GFX950_2D_VARIANT,
            replace(
                spec,
                use_k_single_buffer=False,
                use_q_direct_reg=True,
            ),
        )
        self.assertFalse(ok)
        self.assertIn("single-K", why)

    def test_registered_gfx950_specs_pass_spec_complete_support(self):
        for prefix, expect_specs in (
            ("attention_gfx950_u2d_narrow_nw8_mw16_t8xb_llvm", True),
            ("attention_gfx950_u2d_narrow_nw4_mw16_t8xb_hipcc", True),
            # Curated production stacks have no single-K buffer on wide32, and
            # the baseline exceeds LDS on this 8x tile.
            ("attention_gfx950_u2d_wide32_nw4_mw32_t8xb_llvm", False),
            ("attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm", True),
        ):
            with self.subTest(candidate=prefix):
                _c, _req, specs = _specs_for(prefix)
                self.assertEqual(bool(specs), expect_specs)
                for tuning_spec in specs:
                    ok, why = _supports_tuning_spec(
                        _GFX950_2D_VARIANT, tuning_spec.kernel_spec
                    )
                    self.assertTrue(ok, (tuning_spec.tuning_id, why))
                    ks = tuning_spec.kernel_spec
                    if ks.use_kq_lds_pad:
                        self.assertTrue(ks.use_k_single_buffer)
                        self.assertTrue(ks.use_q_direct_reg or ks.use_q_reread)

    def test_execution_candidates_include_every_tuning_geometry(self):
        route = [c for c in attention_candidates() if c.algorithm == "unified_tuning"]
        execution = [
            c
            for c in attention_execution_candidates()
            if c.algorithm == "unified_tuning"
        ]
        from dispatch.attention.gfx942_tuning import GFX942_TUNING_VARIANTS
        from dispatch.attention.gfx950_tuning import GFX950_TUNING_VARIANTS

        expected = len(GFX942_TUNING_VARIANTS) + len(GFX950_TUNING_VARIANTS)
        self.assertEqual(len(route), expected)
        self.assertEqual(len(execution), expected)
        self.assertTrue(route)
        self.assertTrue(all(c.opt_in for c in route))
        self.assertTrue(all(c.opt_in for c in execution))

    def test_full_sample_includes_dead_end_knobs(self):
        """Dead ends stay out of KNOWN_WRONG_KNOBS and in the sampled sweep."""
        cases = (
            (
                "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm",
                "gfx950",
                "use_q_reread",
            ),
            (
                "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm",
                "gfx942",
                "use_conflict_free_v",
            ),
        )
        for prefix, arch, knob in cases:
            with self.subTest(knob=knob):
                self.assertNotIn(knob, KNOWN_WRONG_KNOBS[arch])
                specs = _sampled(prefix, 256, seed=0, arch=arch)
                self.assertTrue(any(getattr(s.kernel_spec, knob) for s in specs))

    def test_production_sweep_excludes_dead_end_knobs(self):
        from dispatch.attention.tuning_common import DEAD_END_KNOBS

        cases = (
            (
                "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm",
                "gfx950",
                "use_q_reread",
            ),
            (
                "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm",
                "gfx942",
                "use_conflict_free_v",
            ),
        )
        for prefix, arch, knob in cases:
            with self.subTest(candidate=prefix):
                _c, _req, specs = _specs_for(prefix, n=10000, arch=arch)
                self.assertGreater(len(specs), 1)
                self.assertLess(len(specs), 500)
                for spec in specs:
                    for dead in DEAD_END_KNOBS[arch]:
                        self.assertFalse(getattr(spec.kernel_spec, dead, False), knob)

    def test_full_space_wide32_8x_fits_with_single_k_buffer(self):
        """Baseline exceeds LDS; the full space still fits via single-K."""
        from dispatch.attention.tuning_common import configure_sweep

        prefix = "attention_gfx950_u2d_wide32_nw4_mw32_t8xb_llvm"
        configure_sweep("full", 0)
        try:
            _c, _req, specs = _specs_for(prefix, n=80)
        finally:
            configure_sweep("production", 0)
        self.assertTrue(specs)
        self.assertTrue(any(s.kernel_spec.use_k_single_buffer for s in specs))
        for tuning_spec in specs:
            ok, why = _supports_tuning_spec(_GFX950_2D_VARIANT, tuning_spec.kernel_spec)
            self.assertTrue(ok, (tuning_spec.tuning_id, why))


class TestAutoDispatchUnchanged(unittest.TestCase):
    def test_auto_still_selects_unified_3d_for_decode(self):
        import kernels.common.attention_unified as au

        old = au._RESOLVED_ATTENTION_ARCH
        try:
            au._RESOLVED_ATTENTION_ARCH = "gfx950"
            result = dispatch_attention(_request(seqlen_q=1, seqlen_k=4096))
        finally:
            au._RESOLVED_ATTENTION_ARCH = old
        self.assertEqual(result.candidate.name, "attention_unified_3d")


if __name__ == "__main__":
    unittest.main()
