# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dependency-valid, unique, bounded attention tuning space."""

from __future__ import annotations

import unittest
from dataclasses import replace

from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    attention_execution_candidates,
    dispatch_attention,
    dispatch_attention_all,
)
from dispatch.attention.tuning_common import (
    AttentionGeometryVariant,
    _gfx950_tuning_lds_bytes,
    _supports_tuning_spec,
)
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


def _specs_for(prefix, **req_kw):
    candidate = next(c for c in attention_candidates() if c.name.startswith(prefix))
    req = replace(
        _request(**req_kw),
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    return candidate, req, candidate.sweep_space(req)


_GFX950_2D_VARIANT = AttentionGeometryVariant(
    arch="gfx950",
    path="2d",
    codepath="narrow",
    builder_kind="tiled",
    tile_policy="8x",
)


class TestTuningSpace(unittest.TestCase):
    def test_gfx950_cardinality_is_bounded(self):
        req = replace(_request(), algorithm="unified_tuning")
        n = sum(
            1
            for _c, _s in __import__(
                "dispatch.attention", fromlist=["registered_attention_combos"]
            ).registered_attention_combos(req)
            if _c.algorithm == "unified_tuning"
        )
        self.assertGreater(n, 1)
        self.assertLessEqual(n, 5000)

    def test_gfx942_cardinality_is_bounded(self):
        from dispatch.attention import registered_attention_combos

        req = replace(_request("gfx942"), algorithm="unified_tuning")
        n = sum(
            1
            for c, _s in registered_attention_combos(req)
            if c.algorithm == "unified_tuning"
        )
        self.assertGreater(n, 1)
        self.assertLessEqual(n, 1500)

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
        self.assertFalse(gfx950_specs[0].kernel_spec.fp8_fnuz)
        self.assertTrue(gfx942_specs[0].kernel_spec.fp8_fnuz)
        with self.assertRaisesRegex(ValueError, "disagree on fp8_fnuz"):
            gfx950_candidate.built(replace(gfx950_specs[0], fp8_fnuz=True), "gfx950")
        bad_kernel = replace(gfx950_specs[0].kernel_spec, fp8_fnuz=True)
        with self.assertRaisesRegex(ValueError, "requires OCP"):
            gfx950_candidate.built(
                replace(
                    gfx950_specs[0],
                    fp8_fnuz=True,
                    kernel_spec=bad_kernel,
                ),
                "gfx950",
            )

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

    def test_transposed_x8_keeps_k_hbm_direct_and_emits_direct_loads(self):
        from rocke import lower_kernel_to_llvm

        candidate, _req, specs = _specs_for(
            "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm",
            arch="gfx942",
        )
        khbm = [s for s in specs if s.kernel_spec.use_k_hbm_direct]
        self.assertTrue(khbm)
        built = candidate.built(khbm[0], "gfx942")
        kernel = built if not isinstance(built, tuple) else built[0]
        llvm = lower_kernel_to_llvm(kernel, arch="gfx942")
        self.assertIn("buffer", llvm.lower())

    def test_named_stacks_cover_independent_knobs(self):
        _c, _req, specs = _specs_for(
            "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm"
        )
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
        self.assertEqual(_gfx950_tuning_lds_bytes(spec), 165888)
        ok, why = _supports_tuning_spec(_GFX950_2D_VARIANT, spec)
        self.assertFalse(ok)
        self.assertIn("163840 B LDS budget", why)
        self.assertEqual(
            _gfx950_tuning_lds_bytes(replace(spec, use_v_double_buffer=True)),
            198656,
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
        self.assertEqual(len(route), 160)
        self.assertEqual(len(execution), 160)


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
