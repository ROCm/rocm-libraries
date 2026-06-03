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

from te_to_dispatcher import TranslationError, translate, translate_with_rejections


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
    """double_buffer flag: compv4 → True, preshufflev2 → True, compv3 → False."""

    def test_compv4_sets_double_buffer(self):
        # compv4 uses double SMEM buffering.
        configs = translate(_single_config(pipeline="compv4"))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["double_buffer"] is True

    def test_compv3_no_double_buffer(self):
        configs = translate(_single_config(pipeline="compv3"))
        assert len(configs) == 1
        assert configs[0]["algorithm"]["double_buffer"] is False

    def test_preshufflev2_sets_double_buffer(self):
        # preshufflev2 uses double SMEM buffering, matching codegen:
        # unified_gemm_codegen.py sets DoubleSmemBuffer = (pipeline == "compv4"
        # or pipeline == "preshufflev2"). The translator's _DOUBLE_BUFFER_PIPELINES
        # = {"compv4", "preshufflev2"} agrees, so there is no discrepancy.
        configs = translate(_single_config(pipeline="preshufflev2"))
        assert len(configs) == 1
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


class TestSplitKValidation:
    """split_k > 255 must raise TranslationError (uint8_t overflow in oracle)."""

    def test_split_k_256_raises(self):
        # cpp_identifier_oracle.cpp casts split_k to uint8_t; 256 wraps to 0.
        with pytest.raises(TranslationError, match="split_k=256"):
            translate(_single_config(split_k=256))

    def test_split_k_255_accepted(self):
        configs = translate(_single_config(split_k=255))
        assert len(configs) == 1
        assert configs[0]["signature"]["split_k"] == 255

    def test_split_k_0_raises(self):
        with pytest.raises(TranslationError, match="split_k=0"):
            translate(_single_config(split_k=0))


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


class TestRejectionManifest:
    """translate_with_rejections() returns (valid_configs, rejections) with reason strings."""

    def test_all_valid_no_rejections(self):
        valid, rejected = translate_with_rejections(_single_config())
        assert len(valid) == 1
        assert rejected == []

    def test_invalid_tile_appears_in_rejections(self):
        # tile_m=64, warp_m=4, warp_tile_m=32 → 4*32=128 > 64 → invalid
        valid, rejected = translate_with_rejections(
            _single_config(tile_m=64, warp_m=4, warp_tile_m=32)
        )
        assert valid == []
        assert len(rejected) == 1
        assert rejected[0]["reason"] == "invalid_tile_divisibility"

    def test_rejection_has_combo_and_reason_keys(self):
        _, rejected = translate_with_rejections(
            _single_config(tile_m=64, warp_m=4, warp_tile_m=32)
        )
        assert "combo" in rejected[0]
        assert "reason" in rejected[0]

    def test_valid_plus_rejected_equals_total_combinations(self):
        # A 2-combo config: one valid, one invalid tile.
        data = {
            "datatype": "fp16",
            "layout": "rcr",
            "gpu_target": "gfx942",
            "block_size": 256,
            "k_block_per_cu": 1,
            "num_wave_groups": 1,
            "split_k": 1,
            "tile_config": {
                "tile_m": {"values": [256, 64]},   # 64 invalid with warp_tile_m=32, warp_m=4
                "tile_n": {"values": [128]},
                "tile_k": {"values": [32]},
                "warp_m": {"values": [4]},
                "warp_n": {"values": [1]},
                "warp_k": {"values": [1]},
                "warp_tile_m": {"values": [32]},
                "warp_tile_n": {"values": [32]},
                "warp_tile_k": {"values": [16]},
            },
            "trait_config": {
                "pipeline": {"values": ["compv3"]},
                "epilogue": {"values": ["default"]},
                "scheduler": {"values": ["intrawave"]},
                "pad_m": {"values": [False]},
                "pad_n": {"values": [False]},
                "pad_k": {"values": [False]},
                "persistent": {"values": [False]},
            },
        }
        valid, rejected = translate_with_rejections(data)
        assert len(valid) == 1
        assert len(rejected) == 1

    def test_padding_config_no_rejections(self):
        valid, rejected = translate_with_rejections(
            _single_config(pad_m=True, pad_n=True, pad_k=True)
        )
        assert len(valid) == 1
        assert rejected == []


# ── verification model documentation ─────────────────────────────────────────

class TestVerificationModel:
    """Checks that the 'each stack verifies against its own CPU ref' model
    is correctly documented in code and that the harness output format is
    compatible with check_parity.py parsing."""

    def test_check_parity_documents_own_cpu_ref_model(self):
        """run_te_benchmark docstring must explain the own-CPU-ref limitation."""
        import check_parity
        import inspect
        src = inspect.getsource(check_parity.run_te_benchmark)
        assert "own" in src.lower() or "self-consist" in src.lower(), (
            "run_te_benchmark docstring must document that each stack "
            "verifies against its own CPU reference"
        )

    def test_harness_cpp_uses_fixed_bounded_init(self):
        """harness.cpp must use a fixed, bounded init pattern — not random."""
        harness = (
            __import__("pathlib").Path(__file__).resolve().parent / "harness.cpp"
        )
        text = harness.read_text()
        # Fixed modular pattern keeps values in a bounded range
        assert "% 7" in text or "%7" in text, (
            "harness.cpp must use deterministic (i%7) init to avoid fp16 overflow"
        )
        # Must NOT use FillUniformDistribution (random)
        assert "FillUniformDistribution" not in text, (
            "harness.cpp must not use random init — TE benchmark uses random "
            "so using it here would still be different data due to different seeds"
        )

    def test_porting_decisions_documents_verification_model(self):
        """PORTING_DECISIONS.md §2 must mention the different-init decision."""
        pd = (
            __import__("pathlib").Path(__file__).resolve().parent
            / "PORTING_DECISIONS.md"
        )
        text = pd.read_text()
        assert "verification model" in text.lower() or "own cpu" in text.lower() or (
            "self-consist" in text.lower()
        ), "PORTING_DECISIONS.md must document the verification model decision"

    def test_harness_explicit_warmup_set(self):
        """harness.cpp must set cold_niters_ explicitly (documents warmup count)."""
        harness = (
            __import__("pathlib").Path(__file__).resolve().parent / "harness.cpp"
        )
        text = harness.read_text()
        assert "cold_niters_" in text, (
            "harness.cpp must set cold_niters_ explicitly so warmup=3 is "
            "self-documenting alongside nrepeat_=20"
        )


# ── Python identifier oracle unit tests (T1.2) ──────────────────────────────

class TestIdentifierPython:
    """encode_identifier() must produce the same byte-for-byte string as the C++ oracle.

    These tests use known config dicts and the expected identifier string derived
    from the C++ encode_identifier() rule documented in identifier.py's docstring.
    They do NOT require g++ or a GPU — pure Python, always runnable.
    """

    def _make_cfg(
        self,
        dtype="fp16", layout="rcr",
        pipeline="compv3", epilogue="default", scheduler="intrawave",
        pad_m=False, pad_n=False, pad_k=False, persistent=False,
        tile_m=256, tile_n=128, tile_k=32,
        warp_m=4, warp_n=1, warp_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        split_k=1, elementwise_op="", num_d_tensors=0,
        structured_sparsity=False, preshuffle=False,
    ):
        return {
            "signature": {
                "dtype_a": dtype,
                "layout_a": layout[0], "layout_b": layout[1], "layout_c": layout[2],
                "split_k": split_k,
                "elementwise_op": elementwise_op,
                "num_d_tensors": num_d_tensors,
                "structured_sparsity": structured_sparsity,
            },
            "algorithm": {
                "pipeline": pipeline, "epilogue": epilogue, "scheduler": scheduler,
                "pad_m": pad_m, "pad_n": pad_n, "pad_k": pad_k, "persistent": persistent,
                "tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k,
                "warp_m": warp_m, "warp_n": warp_n, "warp_k": warp_k,
                "warp_tile_m": warp_tile_m, "warp_tile_n": warp_tile_n, "warp_tile_k": warp_tile_k,
                "preshuffle": preshuffle,
            },
        }

    def test_vanilla_fp16_rcr(self):
        from identifier import encode_identifier
        cfg = self._make_cfg()
        ident = encode_identifier(cfg)
        expected = (
            "fp16_rcr_compv3_default_intrawave_"
            "False_False_False_False_"
            "256x128x32_4x1x1_32x32x16"
        )
        assert ident == expected, f"Got: {ident!r}"

    def test_padding_enabled_identifier(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(pad_m=True, pad_n=True, pad_k=True)
        ident = encode_identifier(cfg)
        assert "True_True_True_False_" in ident, f"Padding flags wrong: {ident!r}"

    def test_split_k_suffix(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(split_k=4)
        ident = encode_identifier(cfg)
        assert ident.endswith("_splitk4"), f"Expected _splitk4 suffix: {ident!r}"

    def test_split_k_1_no_suffix(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(split_k=1)
        ident = encode_identifier(cfg)
        assert "_splitk" not in ident, f"split_k=1 must not add suffix: {ident!r}"

    def test_preshuffle_suffix(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(preshuffle=True)
        ident = encode_identifier(cfg)
        assert ident.endswith("_preshuffle"), f"Expected _preshuffle suffix: {ident!r}"

    def test_preshuffle_false_no_suffix(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(preshuffle=False)
        ident = encode_identifier(cfg)
        assert "_preshuffle" not in ident, f"preshuffle=False must not add suffix: {ident!r}"

    def test_passthrough_elementwise_op_skipped(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(elementwise_op="PassThrough")
        ident_pt = encode_identifier(cfg)
        cfg2 = self._make_cfg(elementwise_op="")
        ident_empty = encode_identifier(cfg2)
        assert ident_pt == ident_empty, "PassThrough must not add suffix to identifier"

    def test_nonpassthrough_op_appended(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(elementwise_op="Relu")
        ident = encode_identifier(cfg)
        assert ident.endswith("_Relu"), f"Custom op must appear in identifier: {ident!r}"

    def test_scheduler_auto_used_as_is(self):
        """Canonical scheduler 'auto' must appear verbatim (no further mapping)."""
        from identifier import encode_identifier
        cfg = self._make_cfg(scheduler="auto")
        ident = encode_identifier(cfg)
        assert "_auto_" in ident, f"Scheduler 'auto' must appear in identifier: {ident!r}"

    def test_bf16_dtype(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(dtype="bf16")
        ident = encode_identifier(cfg)
        assert ident.startswith("bf16_"), f"bf16 dtype prefix wrong: {ident!r}"

    def test_persistent_flag(self):
        from identifier import encode_identifier
        cfg = self._make_cfg(persistent=True)
        ident = encode_identifier(cfg)
        # persistent is the 4th bool after pad_m/n/k
        assert "_False_False_False_True_" in ident, \
            f"persistent=True must appear as True: {ident!r}"

    def test_identifier_field_order(self):
        """Full field order: dtype_layout_pipeline_epilogue_scheduler_padM_N_K_persist_tile."""
        from identifier import encode_identifier
        cfg = self._make_cfg()
        ident = encode_identifier(cfg)
        parts = ident.split("_")
        # dtype, layout_abc, pipeline, epilogue, scheduler, padM, padN, padK, persist, tile...
        assert parts[0] == "fp16"
        assert parts[1] == "rcr"
        assert parts[2] == "compv3"
        assert parts[3] == "default"
        assert parts[4] == "intrawave"
        assert parts[5] == "False"   # pad_m
        assert parts[6] == "False"   # pad_n
        assert parts[7] == "False"   # pad_k
        assert parts[8] == "False"   # persistent


# ── te_kernel_name() regression tests ────────────────────────────────────────

class TestTeKernelName:
    """Tests for check_parity.te_kernel_name().

    Bug #1 in pr_review_report: te_kernel_name() was missing the _preshuffle
    suffix for preshufflev2 configs, causing Stage 2 to exit with
    "expected generated header not found" for all preshuffle configs.
    This class is the regression guard for that fix.
    """

    def _cfg_from_translate(self, **kwargs):
        """Translate a single-combination TE config and return the first result."""
        return translate(_single_config(**kwargs))[0]

    def test_vanilla_name_no_preshuffle_suffix(self):
        """compv3 config must NOT have _preshuffle suffix in kernel name."""
        import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from check_parity import te_kernel_name
        cfg = self._cfg_from_translate(pipeline="compv3", scheduler="intrawave")
        name = te_kernel_name(cfg)
        assert "_preshuffle" not in name, (
            f"compv3 kernel name must not have _preshuffle suffix: {name!r}"
        )

    def test_preshufflev2_appends_preshuffle_suffix(self):
        """preshufflev2 config MUST have _preshuffle suffix (Bug #1 fix)."""
        import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from check_parity import te_kernel_name
        cfg = self._cfg_from_translate(pipeline="preshufflev2", scheduler="intrawave")
        name = te_kernel_name(cfg)
        assert name.endswith("_preshuffle"), (
            f"preshufflev2 kernel name must end with _preshuffle (Bug #1 fix): {name!r}"
        )

    def test_kernel_name_uses_raw_te_scheduler_not_canonical(self):
        """Kernel name uses raw TE scheduler string, not the canonical form.

        For scheduler 'default': the registry identifier uses 'auto' (canonical),
        but te_kernel_name() must use 'default' (raw TE string) so the generated
        header filename gemm_..._default_... can be found on disk.
        """
        import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from check_parity import te_kernel_name
        cfg = self._cfg_from_translate(pipeline="compv3", scheduler="default")
        name = te_kernel_name(cfg)
        assert "_default_" in name, (
            f"te_kernel_name must use raw 'default' scheduler (not 'auto'): {name!r}"
        )
        assert "_auto_" not in name, (
            f"te_kernel_name must NOT use canonical 'auto' scheduler: {name!r}"
        )

    def test_kernel_name_contains_tile_shape(self):
        """Kernel name encodes tile shape as MxNxK."""
        import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from check_parity import te_kernel_name
        cfg = self._cfg_from_translate(tile_m=256, tile_n=128, tile_k=32)
        name = te_kernel_name(cfg)
        assert "256x128x32" in name, (
            f"te_kernel_name must include tile shape 256x128x32: {name!r}"
        )

    def test_kernel_name_contains_padding_flags(self):
        """Padding-enabled kernel name encodes True for pad fields."""
        import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from check_parity import te_kernel_name
        cfg = self._cfg_from_translate(pad_m=True, pad_n=True, pad_k=True)
        name = te_kernel_name(cfg)
        assert "_True_True_True_" in name, (
            f"pad_m/n/k=True must appear in kernel name: {name!r}"
        )


# ── parse_sizes() tests ───────────────────────────────────────────────────────

class TestParseSizes:
    """Tests for check_parity.parse_sizes() — the --sizes CLI argument parser.

    This function converts '512x512x512,1024x1024x1024' into
    [(512,512,512),(1024,1024,1024)]. Errors here silently skip all sizes.
    """

    def _parse(self, spec: str):
        import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from check_parity import parse_sizes
        return parse_sizes(spec)

    def test_single_size(self):
        assert self._parse("512x512x512") == [(512, 512, 512)]

    def test_multiple_sizes(self):
        result = self._parse("512x512x512,1024x1024x1024")
        assert result == [(512, 512, 512), (1024, 1024, 1024)]

    def test_non_cubic_size(self):
        result = self._parse("257x257x56")
        assert result == [(257, 257, 56)]

    def test_whitespace_tolerance(self):
        result = self._parse("512x512x512, 1024x1024x1024")
        assert result == [(512, 512, 512), (1024, 1024, 1024)]

    def test_bad_format_raises(self):
        import pytest
        with pytest.raises(ValueError, match="bad size"):
            self._parse("512x512")

    def test_empty_string_raises(self):
        import pytest
        with pytest.raises(ValueError):
            self._parse("")

    def test_default_sizes_parse_cleanly(self):
        """The actual default --sizes value from check_parity.py must parse."""
        import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        import re
        src = (__import__("pathlib").Path(__file__).parent / "check_parity.py").read_text()
        m = re.search(r'default="([^"]+)".*sizes', src)
        if m is None:
            m = re.search(r'sizes.*default="([^"]+)"', src)
        # Extract the size string from the argparse default
        size_str_match = re.search(
            r'"(\d+x\d+x\d+(?:,\s*\d+x\d+x\d+)*)"',
            src[src.find("--sizes"):src.find("--sizes") + 300] if "--sizes" in src else ""
        )
        if size_str_match:
            result = self._parse(size_str_match.group(1))
            assert len(result) >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
