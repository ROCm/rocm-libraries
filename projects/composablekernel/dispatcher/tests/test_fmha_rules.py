#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "codegen"))

from fmha.validation import validate_config, load_arch_specs

SPECS = load_arch_specs()


def _base_config(
    family="fwd",
    dtype="fp16",
    arch="gfx950",
    pipeline="qr_async",
    hdim_q=128,
    hdim_v=128,
    **sig_overrides,
):
    sig = {
        "family": family,
        "data_type": dtype,
        "mode": "batch",
        "vlayout": "r",
        "hdim_q": hdim_q,
        "hdim_v": hdim_v,
        "mask": "no",
        "bias": "no",
        "lse": False,
        "dropout": False,
        "qscale": "no",
        "rope": "none",
        "logits": False,
        "paged_kv": False,
        "fp8_static_quant": False,
        "skip_min_seqlen_q": False,
        "sink": False,
        "dbias": False,
        "store_randval": False,
        "deterministic": False,
        "kv_memory_layout": "vectorized",
        "kv_lookup_table": "sglang",
        "page_size": 1,
    }
    sig.update(sig_overrides)
    alg = {
        "pipeline": pipeline,
        "tile": [128, 128, 32, 128, 32, 128],
        "wave": [4, 1, 1, 4, 1, 1, 1, 1, 1],
        "warp": [32, 32, 16, 32, 32, 16, 16, 16, 16],
        "padding": [True, True, True, True],
        "block_per_cu": 1,
        "num_wave_groups": 1,
        "max_splits_log2": 0,
        "max_seq_len_q": 0,
    }
    return {"signature": sig, "algorithm": alg, "arch": arch}


def _gfx11_batch_prefill_config(pipeline="batch_prefill_gfx11", **sig_overrides):
    """Config on the gfx1100 emission envelope; overrides break one clause.

    hdim 128 uses M128/block256; hdim 96 and 64 use M64/block128.
    """
    sig_overrides.setdefault("paged_kv", True)
    sig_overrides.setdefault("page_size", 16)
    sig_overrides.setdefault("kv_memory_layout", "linear")
    cfg = _base_config(
        family="batch_prefill",
        arch="gfx1100",
        pipeline=pipeline,
        **sig_overrides,
    )
    hdim = cfg["signature"]["hdim_q"]
    m0 = 128 if hdim == 128 else 64
    waves = m0 * 2 // 32
    cfg["signature"]["mode"] = "group"
    cfg["algorithm"]["tile"] = [m0, 32, 32, cfg["signature"]["hdim_v"], 32, hdim]
    cfg["algorithm"]["wave"] = [waves, 1, 1, waves, 1, 1, 1, 1, 1]
    cfg["algorithm"]["warp"] = [16, 16, 16, 16, 16, 16, 16, 16, 16]
    return cfg


class TestValidateConfig(unittest.TestCase):
    def test_valid_basic_config(self):
        r = validate_config(_base_config(), SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_unsupported_arch(self):
        r = validate_config(_base_config(arch="gfx000"), SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("architecture" in e for e in r.errors))

    def test_v3_hdim128_valid(self):
        r = validate_config(_base_config(pipeline="v3", hdim_q=128, hdim_v=128), SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_hdim_not_multiple_of_8(self):
        r = validate_config(_base_config(hdim_q=65, hdim_v=128), SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("multiples of 8" in e for e in r.errors))

    def test_bias_plus_logits_soft_cap(self):
        r = validate_config(_base_config(bias="bias", logits=True), SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("logits_soft_cap" in e for e in r.errors))

    def test_hdim_192_128_with_bias(self):
        r = validate_config(_base_config(hdim_q=192, hdim_v=128, bias="bias"), SPECS)
        has_issue = any("(192,128)" in e for e in r.errors) or any(
            "(192,128)" in w for w in r.warnings
        )
        self.assertTrue(has_issue)

    def test_hdim_192_128_with_dropout(self):
        r = validate_config(_base_config(hdim_q=192, hdim_v=128, dropout=True), SPECS)
        has_issue = any("(192,128)" in e for e in r.errors) or any(
            "(192,128)" in w for w in r.warnings
        )
        self.assertTrue(has_issue)

    def test_appendkv_must_use_appendkv_pipeline(self):
        cfg = _base_config(family="fwd_appendkv", pipeline="qr_async")
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("appendkv pipeline" in e for e in r.errors))

    def test_pagedkv_requires_qr_pagedkv_pipeline(self):
        cfg = _base_config(family="fwd_pagedkv", pipeline="qr_async", paged_kv=True)
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("qr_pagedkv" in e for e in r.errors))

    def test_batch_prefill_requires_group_mode(self):
        cfg = _base_config(
            family="batch_prefill",
            pipeline="qr_async",
            mode="batch",
            paged_kv=True,
            page_size=64,
        )
        cfg["signature"]["mode"] = "batch"
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("group mode" in e for e in r.errors))

    def test_batch_prefill_valid_group(self):
        cfg = _base_config(
            family="batch_prefill", pipeline="qr_async", paged_kv=True, page_size=64
        )
        cfg["signature"]["mode"] = "group"
        r = validate_config(cfg, SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_gfx1100_batch_prefill_gfx11_valid(self):
        r = validate_config(_gfx11_batch_prefill_config(), SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_gfx1100_batch_prefill_rejects_qr_async(self):
        cfg = _gfx11_batch_prefill_config(pipeline="qr_async")
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("batch_prefill_gfx11" in e for e in r.errors), r.errors)

    def test_batch_prefill_gfx11_rejected_for_other_families(self):
        """The arch allowlist admits the pipeline; only batch_prefill can use it.

        Every other family reaches a kernel-body lookup that has no entry for
        this pipeline, so letting the config through turns a config error into a
        KeyError in codegen.
        """
        for family in ("fwd", "fwd_splitkv", "fwd_appendkv"):
            cfg = _gfx11_batch_prefill_config()
            cfg["signature"]["family"] = family
            r = validate_config(cfg, SPECS)
            self.assertFalse(r.valid, f"{family}: {r.errors}")
            self.assertTrue(
                any("only valid for family batch_prefill" in e for e in r.errors),
                f"{family}: {r.errors}",
            )

    def test_batch_prefill_gfx11_requires_k0_equal_k1(self):
        """Mirror of the pipeline's static_assert(kK0 == kK1).

        The async copy strides K into LDS by kK1 while gemm0 reads kK0-deep
        chunks from the same buffer, so an unequal pair compiles nowhere. The
        other tile dimensions are already pinned, K0 was not.
        """
        cfg = _gfx11_batch_prefill_config()
        tile = list(cfg["algorithm"]["tile"])
        self.assertEqual(tile[2], tile[4], "baseline config should have K0 == K1")
        tile[2] = tile[4] * 2
        cfg["algorithm"]["tile"] = tile
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("K0 == K1" in e for e in r.errors), r.errors)

    def test_batch_prefill_gfx11_rejected_on_other_arch(self):
        cfg = _gfx11_batch_prefill_config()
        cfg["arch"] = "gfx950"
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("gfx1100" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_vectorized(self):
        cfg = _gfx11_batch_prefill_config(kv_memory_layout="vectorized")
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("linear" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_accepts_native_hdims(self):
        """128 rides Independent-V; 96 and 64 ride the fallback gemm1 on M64."""
        for hdim in (128, 96, 64):
            with self.subTest(hdim=hdim):
                cfg = _gfx11_batch_prefill_config(hdim_q=hdim, hdim_v=hdim)
                r = validate_config(cfg, SPECS)
                self.assertTrue(r.valid, r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_padded_hdim_pair(self):
        """(96, 128) is the padded pair these native instances replace."""
        cfg = _gfx11_batch_prefill_config(hdim_q=96, hdim_v=128)
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("hdim pairs" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_padded_qk_tile(self):
        """hdim 96 must carry QKHeaddim 96, not ceil_to_qualified_tile_length 128."""
        cfg = _gfx11_batch_prefill_config(hdim_q=96, hdim_v=96)
        tile = list(cfg["algorithm"]["tile"])
        tile[5] = 128
        cfg["algorithm"]["tile"] = tile
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("QKHeaddim=96" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_soft_cap(self):
        """gfx1100 builds with FAST_EXP2=0, which the pipeline asserts against."""
        cfg = _gfx11_batch_prefill_config(logits=True)
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("soft cap" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_unbalanced_gemm1_warps(self):
        """Mirror of TileFmhaShape's NumGemm1Warps % NumGemm0Warps == 0."""
        cfg = _gfx11_batch_prefill_config(hdim_q=96, hdim_v=96)
        cfg["algorithm"]["wave"] = [4, 1, 1, 2, 1, 1, 1, 1, 1]
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("gemm1 warp count" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_m128_for_hdim96(self):
        """Each head dim is pinned to the one M0 that has device evidence."""
        cfg = _gfx11_batch_prefill_config(hdim_q=96, hdim_v=96)
        cfg["algorithm"]["tile"] = [128, 32, 32, 96, 32, 96]
        cfg["algorithm"]["wave"] = [8, 1, 1, 8, 1, 1, 1, 1, 1]
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("block size 128" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_dropout(self):
        cfg = _gfx11_batch_prefill_config(dropout=True)
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("dropout" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_block_size_128(self):
        cfg = _gfx11_batch_prefill_config()
        cfg["algorithm"]["wave"] = [4, 1, 1, 4, 1, 1, 1, 1, 1]
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("block size 256" in e for e in r.errors), r.errors)

    def test_gfx1100_batch_prefill_gfx11_rejects_other_tile(self):
        cfg = _gfx11_batch_prefill_config()
        cfg["algorithm"]["tile"] = [128, 128, 32, 128, 32, 128]
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("N0=32" in e for e in r.errors), r.errors)

    def test_splitkv_combine_bn1_must_be_32(self):
        cfg = _base_config(family="fwd_splitkv_combine", pipeline="qr")
        cfg["algorithm"]["tile"][3] = 64
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("bn1" in e for e in r.errors))

    def test_bwd_dot_do_o_bm0_128_accepted(self):
        cfg = _base_config(family="bwd_dot_do_o", pipeline="qr")
        cfg["algorithm"]["tile"][0] = 128
        r = validate_config(cfg, SPECS)
        # bwd_dot_do_o with bm0=128 is now valid (relaxed from strict bm0=64)
        self.assertTrue(r.valid, r.errors)

    def test_mask_types_all_valid(self):
        for mask in ["no", "top_left", "bottom_right", "generic"]:
            r = validate_config(_base_config(mask=mask), SPECS)
            self.assertTrue(r.valid, f"mask={mask}: {r.errors}")


class TestMaskDistinction(unittest.TestCase):
    """Verify that top_left and bottom_right are distinct after fix."""

    def test_mask_canonical_distinguishes(self):
        from fmha.symbol_map import canonical_mask, MASK_TO_INT

        self.assertEqual(canonical_mask("top_left"), "top_left")
        self.assertEqual(canonical_mask("bottom_right"), "bottom_right")
        self.assertNotEqual(MASK_TO_INT["top_left"], MASK_TO_INT["bottom_right"])


if __name__ == "__main__":
    unittest.main()
