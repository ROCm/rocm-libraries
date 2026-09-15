#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "codegen"))

from fmha.validation import validate_config, load_arch_specs

try:
    # instance_gen pulls in dispatcher/python/fmha_utils, which needs numpy
    from fmha.instance_gen import (
        _supported_hdims,
        generate_fwd_tiles,
        get_pipelines_for_config,
    )

    HAVE_INSTANCE_GEN = True
except ImportError:
    HAVE_INSTANCE_GEN = False

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


def _hdim512_config(**overrides):
    """fwd/qr/hdim 512 on gfx942 with the shipped 64-row, 4-warp tile (gemm1 16x16x16)."""
    cfg = _base_config(arch="gfx942", pipeline="qr", hdim_q=512, hdim_v=512)
    cfg["algorithm"]["tile"] = [64, 128, 32, 512, 32, 512]
    cfg["algorithm"]["wave"] = [4, 1, 1, 4, 1, 1, 1, 1, 1]
    cfg["algorithm"]["warp"] = [16, 16, 32, 16, 16, 16, 16, 16, 16]
    for section, values in overrides.items():
        cfg[section].update(values)
    return cfg


class TestWideHdim(unittest.TestCase):
    """hdim 512 is served by the fwd family's qr pipeline only, with <= 16 rows per warp."""

    def test_qr_hdim512_64_row_tile_valid(self):
        r = validate_config(_hdim512_config(), SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_qr_hdim512_128_row_8_warp_tile_valid(self):
        cfg = _hdim512_config(
            algorithm={
                "tile": [128, 128, 32, 512, 32, 512],
                "wave": [8, 1, 1, 8, 1, 1, 1, 1, 1],
            }
        )
        r = validate_config(cfg, SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_qr_hdim512_128_row_4_warp_tile_rejected(self):
        # 32 rows per warp: Q + f32 O accumulator no longer fit the register file
        cfg = _hdim512_config(
            algorithm={
                "tile": [128, 128, 32, 512, 32, 512],
                "warp": [32, 32, 16, 32, 32, 16, 16, 16, 16],
            }
        )
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("num_warps" in e for e in r.errors), r.errors)

    def test_qr_hdim512_gemm1_warp_k32_rejected(self):
        # 16x16x32 as the P*V warp tile drops keys 4..7 of every 8 at kN1=512
        cfg = _hdim512_config(algorithm={"warp": [16, 16, 32, 16, 16, 32, 16, 16, 16]})
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("gemm1 warp" in e for e in r.errors), r.errors)

    def test_qr_async_hdim512_rejected(self):
        r = validate_config(_hdim512_config(algorithm={"pipeline": "qr_async"}), SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("qr pipeline" in e for e in r.errors), r.errors)

    def test_splitkv_hdim512_rejected(self):
        r = validate_config(_hdim512_config(signature={"family": "fwd_splitkv"}), SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("qr pipeline" in e for e in r.errors), r.errors)

    def test_fp32_hdim512_valid(self):
        cfg = _hdim512_config(
            signature={"data_type": "fp32"},
            algorithm={
                "tile": [64, 64, 32, 512, 16, 512],
                "warp": [16, 16, 16, 16, 16, 16, 16, 16, 16],
            },
        )
        r = validate_config(cfg, SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_hdim512_listed_for_fp16_bf16_fp32(self):
        from fmha.validation import SUPPORTED_HDIMS

        for dtype in ("fp16", "bf16", "fp32"):
            self.assertIn((512, 512), SUPPORTED_HDIMS[dtype])

    @unittest.skipUnless(HAVE_INSTANCE_GEN, "fmha.instance_gen needs numpy")
    def test_tile_engine_only_emits_qr_16_row_tiles(self):
        tags = {s.tag for s in get_pipelines_for_config("gfx942", "fp16", 512, 512, 0)}
        self.assertEqual(tags, {"qr"})

        self.assertEqual(
            generate_fwd_tiles("gfx942", "fp16", 512, 512, pipeline="qr_async"), []
        )
        tiles = generate_fwd_tiles("gfx942", "fp16", 512, 512, pipeline="qr")
        self.assertTrue(tiles)
        for t in tiles:
            self.assertLessEqual(t.bm0 // t.rm0, 16, t)
            self.assertIn(t.bm0, (64, 128), t)
            self.assertEqual(t.bn1, 512, t)
            self.assertEqual(t.wk1, 16, t)

    @unittest.skipUnless(HAVE_INSTANCE_GEN, "fmha.instance_gen needs numpy")
    def test_non_fwd_families_skip_hdim512(self):
        self.assertIn((512, 512), _supported_hdims("fp16", family="fwd"))
        for family in ("fwd_splitkv", "fwd_pagedkv", "fwd_appendkv", "batch_prefill"):
            self.assertNotIn((512, 512), _supported_hdims("fp16", family=family))
        self.assertEqual(
            _supported_hdims("fp16", restrict_hdims=[(512, 512)], family="fwd_splitkv"),
            [],
        )


class TestMaskDistinction(unittest.TestCase):
    """Verify that top_left and bottom_right are distinct after fix."""

    def test_mask_canonical_distinguishes(self):
        from fmha.symbol_map import canonical_mask, MASK_TO_INT

        self.assertEqual(canonical_mask("top_left"), "top_left")
        self.assertEqual(canonical_mask("bottom_right"), "bottom_right")
        self.assertNotEqual(MASK_TO_INT["top_left"], MASK_TO_INT["bottom_right"])


if __name__ == "__main__":
    unittest.main()
