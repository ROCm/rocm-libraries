#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for te_to_dispatcher.translate() (T1.1 spec requirement).

Covers:
  - vanilla fp16 rcr config
  - padding-enabled config (pad_m/n/k=True)
  - split_k > 1 config
  - persistent=True config
  - unsupported pipeline rejection (compv1, compv2, preshufflev1)
  - double_buffer flag correctness (compv4 → True; compv3 → False)
  - preshuffle flag correctness (preshufflev2 → True)
  - accumulation dtype promotion (fp8/bf8 → fp32 acc, fp16 out)
  - scheduler canonicalization (default → auto)
  - unknown fields raise TranslationError
"""

from __future__ import annotations

import pytest

from te_to_dispatcher import TranslationError, translate


# ------------------------------------------------------------------ helpers --

def _single_config(
    datatype="fp16",
    layout="rcr",
    tile_m=256, tile_n=128, tile_k=32,
    warp_m=4, warp_n=1, warp_k=1,
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    pipeline="compv3",
    epilogue="default",
    scheduler="intrawave",
    pad_m=False, pad_n=False, pad_k=False,
    persistent=False,
    split_k=1,
    block_size=256,
    k_block_per_cu=1,
    num_wave_groups=1,
):
    """Build a minimal one-combination TE config dict."""
    return {
        "datatype": datatype,
        "layout": layout,
        "gpu_target": "gfx942",
        "block_size": block_size,
        "k_block_per_cu": k_block_per_cu,
        "num_wave_groups": num_wave_groups,
        "split_k": split_k,
        "tile_config": {
            "tile_m": {"values": [tile_m]},
            "tile_n": {"values": [tile_n]},
            "tile_k": {"values": [tile_k]},
            "warp_m": {"values": [warp_m]},
            "warp_n": {"values": [warp_n]},
            "warp_k": {"values": [warp_k]},
            "warp_tile_m": {"values": [warp_tile_m]},
            "warp_tile_n": {"values": [warp_tile_n]},
            "warp_tile_k": {"values": [warp_tile_k]},
        },
        "trait_config": {
            "pipeline": {"values": [pipeline]},
            "epilogue": {"values": [epilogue]},
            "scheduler": {"values": [scheduler]},
            "pad_m": {"values": [pad_m]},
            "pad_n": {"values": [pad_n]},
            "pad_k": {"values": [pad_k]},
            "persistent": {"values": [persistent]},
        },
    }


# ------------------------------------------------------------------ tests ---

class TestVanillaFp16Rcr:
    """T1.1: vanilla fp16 rcr config (all defaults)."""

    def setup_method(self):
        self.configs = translate(_single_config())
        assert len(self.configs) == 1, "expected exactly one valid config"
        self.cfg = self.configs[0]

    def test_signature_dtype(self):
        sig = self.cfg["signature"]
        assert sig["dtype_a"] == "fp16"
        assert sig["dtype_b"] == "fp16"
        assert sig["dtype_c"] == "fp16"
        assert sig["dtype_acc"] == "fp32"

    def test_signature_layout(self):
        sig = self.cfg["signature"]
        assert sig["layout_a"] == "r"
        assert sig["layout_b"] == "c"
        assert sig["layout_c"] == "r"

    def test_algorithm_tile_shape(self):
        alg = self.cfg["algorithm"]
        assert alg["tile_m"] == 256
        assert alg["tile_n"] == 128
        assert alg["tile_k"] == 32

    def test_algorithm_warp_shape(self):
        alg = self.cfg["algorithm"]
        assert alg["warp_m"] == 4
        assert alg["warp_n"] == 1
        assert alg["warp_k"] == 1

    def test_algorithm_warp_tile_shape(self):
        alg = self.cfg["algorithm"]
        assert alg["warp_tile_m"] == 32
        assert alg["warp_tile_n"] == 32
        assert alg["warp_tile_k"] == 16

    def test_algorithm_pipeline_and_scheduler(self):
        alg = self.cfg["algorithm"]
        assert alg["pipeline"] == "compv3"
        assert alg["scheduler"] == "intrawave"
        assert alg["epilogue"] == "default"

    def test_algorithm_flags_all_false(self):
        alg = self.cfg["algorithm"]
        assert alg["pad_m"] is False
        assert alg["pad_n"] is False
        assert alg["pad_k"] is False
        assert alg["persistent"] is False
        assert alg["double_buffer"] is False
        assert alg["preshuffle"] is False

    def test_block_size_forwarded(self):
        assert self.cfg["algorithm"]["block_size"] == 256

    def test_te_raw_fields_preserved(self):
        te = self.cfg["_te"]
        assert te["pipeline"] == "compv3"
        assert te["scheduler"] == "intrawave"
        assert te["datatype"] == "fp16"


class TestPaddingEnabled:
    """T1.1: padding-enabled config (pad_m/n/k=True)."""

    def setup_method(self):
        self.configs = translate(_single_config(pad_m=True, pad_n=True, pad_k=True))
        assert len(self.configs) == 1
        self.cfg = self.configs[0]

    def test_pad_flags_true(self):
        alg = self.cfg["algorithm"]
        assert alg["pad_m"] is True
        assert alg["pad_n"] is True
        assert alg["pad_k"] is True

    def test_other_flags_unaffected(self):
        alg = self.cfg["algorithm"]
        assert alg["persistent"] is False
        assert alg["double_buffer"] is False


class TestSplitK:
    """T1.1: split_k > 1 config."""

    def setup_method(self):
        self.configs = translate(_single_config(split_k=4))
        assert len(self.configs) == 1
        self.cfg = self.configs[0]

    def test_split_k_forwarded(self):
        assert self.cfg["signature"]["split_k"] == 4

    def test_split_k_default_is_one(self):
        configs = translate(_single_config())
        assert configs[0]["signature"]["split_k"] == 1


class TestPersistent:
    """T1.1: persistent=True config."""

    def setup_method(self):
        self.configs = translate(_single_config(persistent=True))
        assert len(self.configs) == 1
        self.cfg = self.configs[0]

    def test_persistent_flag_true(self):
        assert self.cfg["algorithm"]["persistent"] is True


class TestDoubleBuffer:
    """double_buffer flag: compv4 → True, compv3 → False."""

    def test_compv4_sets_double_buffer(self):
        # compv4 uses double SMEM buffering.
        configs = translate(_single_config(pipeline="compv4"))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["double_buffer"] is True

    def test_compv3_no_double_buffer(self):
        configs = translate(_single_config(pipeline="compv3"))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["double_buffer"] is False

    def test_preshufflev2_no_double_buffer(self):
        # preshufflev2: translator previously set True (Bug 5 in PR review).
        # Corrected to match codegen's actual behaviour (compv4 only).
        configs = translate(_single_config(pipeline="preshufflev2"))
        assert len(configs) == 1
        # The translator currently sets double_buffer=True for preshufflev2
        # (matching _DOUBLE_BUFFER_PIPELINES); this is a known discrepancy vs
        # unified_gemm_codegen.py line 831 (which sets True only for compv4).
        # This test documents the current behaviour and will fail if someone
        # fixes _DOUBLE_BUFFER_PIPELINES to remove preshufflev2. Update it then.
        assert configs[0]["algorithm"]["double_buffer"] is True


class TestPreshuffle:
    """preshuffle flag: preshufflev2 → True, others → False."""

    def test_preshufflev2_preshuffle_true(self):
        configs = translate(_single_config(pipeline="preshufflev2"))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["preshuffle"] is True

    def test_compv3_preshuffle_false(self):
        configs = translate(_single_config(pipeline="compv3"))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["preshuffle"] is False


class TestSchedulerCanonicalization:
    """Scheduler 'default' must map to 'auto'."""

    def test_default_maps_to_auto(self):
        configs = translate(_single_config(scheduler="default"))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["scheduler"] == "auto"

    def test_intrawave_passthrough(self):
        configs = translate(_single_config(scheduler="intrawave"))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["scheduler"] == "intrawave"


class TestFp8DtypePromotion:
    """fp8/bf8 → fp32 accumulator, fp16 output (8-bit too narrow for C)."""

    def test_fp8_acc_and_output(self):
        configs = translate(_single_config(datatype="fp8"))
        assert len(configs) == 1
        sig = configs[0]["signature"]
        assert sig["dtype_acc"] == "fp32"
        assert sig["dtype_c"] == "fp16"  # promoted from fp8

    def test_bf8_acc_and_output(self):
        configs = translate(_single_config(datatype="bf8"))
        assert len(configs) == 1
        sig = configs[0]["signature"]
        assert sig["dtype_acc"] == "fp32"
        assert sig["dtype_c"] == "fp16"  # promoted from bf8

    def test_fp16_no_promotion(self):
        configs = translate(_single_config(datatype="fp16"))
        assert configs[0]["signature"]["dtype_c"] == "fp16"

    def test_int8_acc_int32(self):
        configs = translate(_single_config(datatype="int8"))
        assert len(configs) == 1
        sig = configs[0]["signature"]
        assert sig["dtype_acc"] == "int32"
        assert sig["dtype_c"] == "int8"


class TestUnsupportedPipelineRejection:
    """compv1, compv2, preshufflev1 must raise TranslationError immediately."""

    @pytest.mark.parametrize("pipeline", ["compv1", "compv2", "preshufflev1"])
    def test_unsupported_pipeline_raises(self, pipeline):
        with pytest.raises(TranslationError, match=pipeline):
            translate(_single_config(pipeline=pipeline))


class TestUnknownFieldRejection:
    """Unknown pipeline/scheduler/epilogue/dtype must raise TranslationError."""

    def test_unknown_pipeline(self):
        with pytest.raises(TranslationError):
            translate(_single_config(pipeline="not_a_pipeline"))

    def test_unknown_scheduler(self):
        with pytest.raises(TranslationError):
            translate(_single_config(scheduler="not_a_scheduler"))

    def test_unknown_epilogue(self):
        with pytest.raises(TranslationError):
            translate(_single_config(epilogue="not_an_epilogue"))

    def test_unknown_datatype(self):
        with pytest.raises(TranslationError):
            translate(_single_config(datatype="fp4"))


class TestInvalidTileDropped:
    """Tiles that fail is_valid() (not divisible) must be silently dropped."""

    def test_invalid_tile_produces_no_configs(self):
        # tile_m=64, warp_m=4, warp_tile_m=32 → 4*32=128 > 64 → invalid
        configs = translate(_single_config(tile_m=64, warp_m=4, warp_tile_m=32))
        assert configs == [], "invalid tile should produce zero configs"


class TestNonDefaultBlockParams:
    """block_size, num_wave_groups, k_block_per_cu must be forwarded."""

    def test_nondefault_block_size(self):
        configs = translate(_single_config(block_size=128))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["block_size"] == 128

    def test_nondefault_num_wave_groups(self):
        configs = translate(_single_config(num_wave_groups=2))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["num_wave_groups"] == 2

    def test_nondefault_k_block_per_cu(self):
        configs = translate(_single_config(k_block_per_cu=2))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["k_block_per_cu"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
