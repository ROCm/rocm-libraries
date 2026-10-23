# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the RDNA WMMA prefill spec, ABI, grid, LDS budget and IR."""

from __future__ import annotations

import unittest
from dataclasses import replace

from kernels.gfx1151.wmma_fmha_fwd_v2 import (
    WmmaFmhaFwdV2Spec,
    build_wmma_fmha_fwd_v2,
    is_gfx12_family,
    is_valid_spec,
    wmma_fmha_fwd_v2_grid,
    wmma_fmha_fwd_v2_lds_bytes,
    wmma_fmha_fwd_v2_signature,
)

_ARCHS = ("gfx11-generic", "gfx12-generic")
_BUCKETS = ((64, 64), (96, 96), (128, 128), (192, 128), (256, 128), (256, 256))


class TestWmmaFmhaFwdV2Spec(unittest.TestCase):
    def test_post_init_rejects_bad_enumerations(self):
        bad = (
            dict(dtype="fp32"),
            dict(mask="window"),
            dict(bias="f16"),
            dict(cta_order="qhb"),
            dict(scheduler_strategy="fastest"),
            dict(block_m=-16),
            dict(waves_per_eu=-1),
            dict(layout="packed"),
            dict(layout="paged"),
            dict(layout="paged", page_block_size=24),
            dict(layout="ragged", page_block_size=16),
            dict(page_block_size=16),
            dict(kv_dtype="fp8e5m2"),
        )
        for fields in bad:
            with self.subTest(**fields), self.assertRaises(ValueError):
                WmmaFmhaFwdV2Spec(**fields)

    def test_derived_geometry(self):
        spec = WmmaFmhaFwdV2Spec(head_dim=128)
        self.assertEqual(
            (spec.dv, spec.pair, spec.bm, spec.threads), (128, False, 128, 256)
        )
        pair = WmmaFmhaFwdV2Spec(head_dim=192, head_dim_v=128)
        self.assertEqual(
            (pair.dv, pair.pair, pair.bm, pair.threads), (128, True, 64, 256)
        )
        unsplit = replace(pair, wave_pair=False)
        self.assertEqual((unsplit.bm, unsplit.threads), (128, 256))
        self.assertEqual(spec.normalized().head_dim_v, 128)
        self.assertEqual(spec.normalized().block_m, 128)

    def test_kernel_names_distinguish_every_emitted_variant(self):
        base = WmmaFmhaFwdV2Spec(dtype="fp16", head_dim=128, mask="causal")
        self.assertEqual(
            base.kernel_name(), "rocke_wmma_fmha_fwd_v2_f16_d128_causal_m128n32"
        )
        variants = [
            base,
            replace(base, dtype="bf16"),
            replace(base, head_dim_v=64),
            replace(base, mask="none"),
            replace(base, bias="f32"),
            replace(base, bias="q"),
            replace(base, block_m=64),
            replace(base, block_n=16),
            replace(base, waves_per_eu=0),
            replace(base, scheduler_strategy="max-ilp"),
            replace(base, lazy_rescale=False),
            replace(base, k_pad=0),
            replace(base, wave_pair=True),
            replace(base, cta_order="hbq"),
            replace(base, use_softcap=True),
            replace(base, use_sinks=True),
            replace(base, use_alibi=True),
            replace(base, use_qq_bias=True),
            replace(base, layout="ragged"),
            replace(base, layout="paged", page_block_size=16),
            replace(base, layout="paged", page_block_size=32),
            replace(base, kv_dtype="fp8e4m3"),
        ]
        names = [v.kernel_name() for v in variants]
        self.assertEqual(len(set(names)), len(names))
        # The normalized spec builds the same object.
        self.assertEqual(base.normalized().kernel_name(), base.kernel_name())

    def test_buckets_are_valid_on_both_families(self):
        for arch in _ARCHS:
            for hd, hdv in _BUCKETS:
                for dtype in ("fp16", "bf16"):
                    spec = WmmaFmhaFwdV2Spec(dtype=dtype, head_dim=hd, head_dim_v=hdv)
                    with self.subTest(arch=arch, hd=hd, hdv=hdv, dtype=dtype):
                        self.assertEqual(is_valid_spec(spec, arch=arch), (True, "ok"))

    def test_unsplit_wide_q_is_gfx12_only(self):
        spec = WmmaFmhaFwdV2Spec(head_dim=256, head_dim_v=128, wave_pair=False)
        self.assertTrue(is_valid_spec(spec, arch="gfx12-generic")[0])
        ok, why = is_valid_spec(spec, arch="gfx11-generic")
        self.assertFalse(ok)
        self.assertIn("unsplit", why)

    def test_invalid_targets_and_budgets(self):
        spec = WmmaFmhaFwdV2Spec()
        for arch in ("gfx950", "gfx1250", "gfx9999"):
            with self.subTest(arch=arch):
                self.assertFalse(is_valid_spec(spec, arch=arch)[0])
        too_wide = WmmaFmhaFwdV2Spec(head_dim=128, block_n=64)
        self.assertGreater(
            wmma_fmha_fwd_v2_lds_bytes(too_wide, arch="gfx11-generic"), 65536
        )
        self.assertFalse(is_valid_spec(too_wide, arch="gfx11-generic")[0])
        self.assertFalse(
            is_valid_spec(WmmaFmhaFwdV2Spec(block_m=1024), arch="gfx11-generic")[0]
        )
        self.assertFalse(
            is_valid_spec(WmmaFmhaFwdV2Spec(head_dim=72), arch="gfx11-generic")[0]
        )
        self.assertFalse(
            is_valid_spec(WmmaFmhaFwdV2Spec(k_pad=4), arch="gfx11-generic")[0]
        )
        with self.assertRaises(ValueError):
            build_wmma_fmha_fwd_v2(too_wide, arch="gfx11-generic")

    def test_lds_budget(self):
        # d128, block_n 32, two buffers of K (32 x 136) + V^T (128 x 40) halves.
        spec = WmmaFmhaFwdV2Spec(head_dim=128)
        self.assertEqual(wmma_fmha_fwd_v2_lds_bytes(spec, arch="gfx11-generic"), 37888)
        self.assertEqual(wmma_fmha_fwd_v2_lds_bytes(spec, arch="gfx12-generic"), 37888)

    def test_family_detection(self):
        self.assertFalse(is_gfx12_family("gfx11-generic"))
        self.assertFalse(is_gfx12_family("gfx1151"))
        self.assertTrue(is_gfx12_family("gfx12-generic"))
        self.assertTrue(is_gfx12_family("gfx1201"))

    def test_grid(self):
        spec = WmmaFmhaFwdV2Spec(head_dim=128)
        self.assertEqual(
            wmma_fmha_fwd_v2_grid(spec, batch=2, seqlen_q=129, num_query_heads=3),
            ((2 * 2 * 3, 1, 1), (256, 1, 1)),
        )
        for bad in (
            dict(batch=0, seqlen_q=1, num_query_heads=1),
            dict(batch=1 << 20, seqlen_q=1 << 20, num_query_heads=64),
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                wmma_fmha_fwd_v2_grid(spec, **bad)

    def test_signature_order(self):
        names = [p["name"] for p in wmma_fmha_fwd_v2_signature(WmmaFmhaFwdV2Spec())]
        strides = [
            f"stride_{t}_{a}"
            for t in ("q", "k", "v", "o", "lse")
            for a in ("batch", "token", "head")
        ]
        expected = [
            "Q", "K", "V", "O", "LSE", "scale_log2", "batch", "seqlen_q", "seqlen_k",
            "num_query_heads", "num_kv_heads", "num_v_heads", "head_dim_q", "head_dim_v",
            "bottom_right", "window_left", "window_right", "write_lse",
        ] + strides  # fmt: skip
        self.assertEqual(names, expected)
        biased = [
            p["name"] for p in wmma_fmha_fwd_v2_signature(WmmaFmhaFwdV2Spec(bias="q"))
        ]
        self.assertEqual(
            biased,
            expected
            + ["attn_bias_ptr", "bias_stride_b", "bias_stride_h", "bias_stride_q"],
        )

    def test_feature_signature_order(self):
        base = [p["name"] for p in wmma_fmha_fwd_v2_signature(WmmaFmhaFwdV2Spec())]
        every = WmmaFmhaFwdV2Spec(
            bias="f32",
            use_softcap=True,
            use_sinks=True,
            use_alibi=True,
            use_qq_bias=True,
            layout="paged",
            page_block_size=16,
            kv_dtype="fp8e4m3",
        )
        sig = wmma_fmha_fwd_v2_signature(every)
        self.assertEqual(
            [p["name"] for p in sig],
            base
            + [
                "softcap", "sink_ptr", "alibi_slopes_ptr", "qq_bias_ptr",
                "qq_bias_rows", "qq_bias_cols", "qq_bias_stride", "cu_seqlens_q",
                "seqused_k", "block_table", "block_table_stride", "stride_k_block",
                "stride_v_block", "num_blocks", "k_scale", "v_scale",
                "attn_bias_ptr",
                "bias_stride_b", "bias_stride_h", "bias_stride_q",
            ],
        )  # fmt: skip
        types = {p["name"]: p["type"] for p in sig}
        self.assertIn("fp8e4m3", types["K"])
        self.assertIn("fp8e4m3", types["V"])
        self.assertIn("bf16", types["sink_ptr"])
        self.assertEqual(types["softcap"], "f32")
        ragged = WmmaFmhaFwdV2Spec(layout="ragged")
        self.assertEqual(
            [p["name"] for p in wmma_fmha_fwd_v2_signature(ragged)],
            base + ["cu_seqlens_q", "cu_seqlens_k"],
        )

    def test_features_build_on_both_families(self):
        from rocke.core.ir_serialize import serialize

        specs = (
            WmmaFmhaFwdV2Spec(use_softcap=True, mask="causal"),
            WmmaFmhaFwdV2Spec(head_dim=64, use_sinks=True, mask="causal"),
            WmmaFmhaFwdV2Spec(use_alibi=True, use_qq_bias=True),
            WmmaFmhaFwdV2Spec(layout="ragged", kv_dtype="fp8e4m3", mask="causal"),
            WmmaFmhaFwdV2Spec(layout="paged", page_block_size=16, head_dim=256),
            WmmaFmhaFwdV2Spec(layout="paged", page_block_size=64, use_sinks=True),
        )
        for arch in _ARCHS:
            for spec in specs:
                with self.subTest(arch=arch, name=spec.kernel_name()):
                    self.assertEqual(is_valid_spec(spec, arch=arch), (True, "ok"))
                    text = serialize(build_wmma_fmha_fwd_v2(spec, arch=arch))
                    self.assertIn("kt_full", text)

    def test_feature_off_ir_is_unchanged_by_feature_code(self):
        """Feature-off objects emit no feature ops (no tanh, no page lookups,
        no FP8 decode, no early exit)."""
        from rocke.core.ir_serialize import serialize

        text = serialize(
            build_wmma_fmha_fwd_v2(
                WmmaFmhaFwdV2Spec(mask="causal"), arch="gfx12-generic"
            )
        )
        for op in ("math.tanh", "arith.sitofp", "scf.if", "arith.fdiv"):
            self.assertNotIn(op, text)

    def test_build_attrs_and_ir(self):
        from rocke.core.ir_serialize import serialize

        spec = WmmaFmhaFwdV2Spec(dtype="bf16", head_dim=128, mask="causal", bias="f32")
        for arch, atom in (
            ("gfx11-generic", "wmma_f32_16x16x16_bf16"),
            ("gfx12-generic", "wmma_gfx12_f32_16x16x16_bf16"),
        ):
            with self.subTest(arch=arch):
                kernel = build_wmma_fmha_fwd_v2(spec, arch=arch)
                self.assertEqual(kernel.name, spec.kernel_name())
                self.assertEqual(kernel.attrs["max_workgroup_size"], 256)
                self.assertEqual(tuple(kernel.attrs["waves_per_eu"]), (2, 2))
                self.assertNotIn("scheduler_strategy", kernel.attrs)
                text = serialize(kernel)
                self.assertIn(atom, text)
                for name in ("attn_kv", "kt_full", "kt_msk", "rsc_msk"):
                    self.assertIn(name, text)
        tuned = replace(
            spec, scheduler_strategy="max-ilp", waves_per_eu=0, lazy_rescale=False
        )
        kernel = build_wmma_fmha_fwd_v2(tuned, arch="gfx11-generic")
        self.assertEqual(kernel.attrs["scheduler_strategy"], "max-ilp")
        self.assertNotIn("waves_per_eu", kernel.attrs)
        self.assertNotIn("rsc_msk", serialize(kernel))

    def test_wave_pair_uses_score_exchange(self):
        from rocke.core.ir_serialize import serialize

        text = serialize(
            build_wmma_fmha_fwd_v2(
                WmmaFmhaFwdV2Spec(head_dim=256), arch="gfx11-generic"
            )
        )
        self.assertIn("attn_xs", text)
        text = serialize(
            build_wmma_fmha_fwd_v2(
                WmmaFmhaFwdV2Spec(head_dim=128), arch="gfx11-generic"
            )
        )
        self.assertNotIn("attn_xs", text)


if __name__ == "__main__":
    unittest.main()
