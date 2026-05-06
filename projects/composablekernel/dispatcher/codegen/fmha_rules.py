#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA validation rules.

Uses fmha_arch_specs.json for data-driven tile/constraint validation,
mirroring how GEMM uses arch_filter.py + arch_specs.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from fmha_symbol_map import (
    BWD_DTYPE_MAP,
    FWD_DTYPE_MAP,
    canonical_bias,
    canonical_kv_lookup,
    canonical_kv_memory_layout,
    canonical_mask,
    canonical_qscale,
    canonical_rope,
)

_ARCH_SPECS_PATH = Path(__file__).with_name("fmha_arch_specs.json")


def load_arch_specs() -> dict:
    return json.loads(_ARCH_SPECS_PATH.read_text())


@dataclass
class ValidationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def _validate_tile_against_specs(
    tile: list,
    hdim_q: int,
    hdim_v: int,
    dtype: str,
    pipeline: str,
    arch_info: dict,
    result: ValidationResult,
    family: str = "fwd",
) -> None:
    """Validate tile config against hdim_tile_combos and hdim_tile_constraints."""
    hdim_key = f"{hdim_q}_{hdim_v}"

    combos = arch_info.get("hdim_tile_combos", {}).get(dtype, {})
    valid_tiles = combos.get(hdim_key, [])
    if valid_tiles and tile not in [list(t) for t in valid_tiles]:
        result.add_warning(
            f"tile {tile} not in known combos for {hdim_key}/{dtype} "
            f"(known: {len(valid_tiles)} configs)"
        )

    constraints = arch_info.get("hdim_tile_constraints", {}).get(pipeline, {})
    hdim_constraint = constraints.get(hdim_key, constraints.get("_default", {}))

    if "required_bn0" in hdim_constraint and tile[1] != hdim_constraint["required_bn0"]:
        result.add_error(
            f"{pipeline} with hdim ({hdim_q},{hdim_v}) requires bn0={hdim_constraint['required_bn0']}, "
            f"got bn0={tile[1]}"
        )
    # CK supports bm0 < required_bm0 with adapted warp configs (e.g. w16x16x32).
    # Downgrade to warning -- the kernel compiles and runs correctly.
    if (
        "required_bm0" in hdim_constraint
        and tile[0] != hdim_constraint["required_bm0"]
        and family != "batch_prefill"
    ):
        result.add_warning(
            f"{pipeline} with hdim ({hdim_q},{hdim_v}): bm0={tile[0]} differs from recommended "
            f"bm0={hdim_constraint['required_bm0']}"
        )
    if (
        "forbidden_bk0" in hdim_constraint
        and tile[2] in hdim_constraint["forbidden_bk0"]
    ):
        result.add_error(
            f"{pipeline} with hdim ({hdim_q},{hdim_v}) forbids bk0={tile[2]}"
        )
    if (
        "forbidden_bn0" in hdim_constraint
        and tile[1] in hdim_constraint["forbidden_bn0"]
    ):
        result.add_error(
            f"{pipeline} with hdim ({hdim_q},{hdim_v}) forbids bn0={tile[1]}"
        )

    if "allowed_hdim" in constraints and hdim_key not in constraints["allowed_hdim"]:
        result.add_error(
            f"{pipeline} only supports hdim in {constraints['allowed_hdim']}, got {hdim_key}"
        )


def _validate_global_rules(
    sig: dict, bias: str, result: ValidationResult, global_rules: dict
) -> None:
    """Validate against global rules from arch specs."""
    hdim_q = sig["hdim_q"]
    hdim_v = sig["hdim_v"]
    divisor = global_rules.get("hdim_divisible_by", 8)
    if hdim_q % divisor != 0 or hdim_v % divisor != 0:
        result.add_error(f"Head dimensions must be multiples of {divisor}")

    if global_rules.get("hdim_192_128_no_bias_dropout"):
        if (
            hdim_q == 192
            and hdim_v == 128
            and (bias != "no" or sig.get("dropout", False))
        ):
            result.add_warning(
                "hdim (192,128) with bias/dropout has limited tile support"
            )

    if global_rules.get("logits_requires_no_bias"):
        if bias != "no" and sig.get("logits", False):
            result.add_error("logits_soft_cap cannot be combined with bias")

    k0max_map = global_rules.get("k0max_alignment_map", {})
    k0max_key = str(hdim_q)
    if k0max_key in k0max_map:
        expected_alignment = k0max_map[k0max_key]
        result.add_warning(
            f"hdim_q={hdim_q} should use k0max alignment {expected_alignment} "
            f"(K0_MAX_SUBMAX_MAP)"
        )


def validate_config(
    config: dict, arch_specs: Optional[dict] = None
) -> ValidationResult:
    arch_specs = arch_specs or load_arch_specs()
    result = ValidationResult()

    sig = config["signature"]
    alg = config["algorithm"]
    arch = config["arch"]

    if arch not in arch_specs["architectures"]:
        result.add_error(f"Unsupported FMHA target architecture: {arch}")
        return result

    arch_info = arch_specs["architectures"][arch]
    global_rules = arch_specs.get("global_rules", {})
    dtype = sig["data_type"]
    family = sig["family"]
    pipeline = alg["pipeline"]
    canonical_mask(sig["mask"])  # validated by _validate_tile_against_specs
    bias = canonical_bias(sig["bias"])
    qscale = canonical_qscale(sig["qscale"])
    rope = canonical_rope(sig["rope"])
    kv_memory_layout = canonical_kv_memory_layout(sig["kv_memory_layout"])
    kv_lookup_table = canonical_kv_lookup(sig["kv_lookup_table"])

    # --- Family validation ---
    supported_families = {
        "fwd",
        "fwd_pagedkv",
        "fwd_splitkv",
        "fwd_splitkv_combine",
        "fwd_appendkv",
        "batch_prefill",
        "bwd_dot_do_o",
        "bwd_dq_dk_dv",
        "bwd_convert_dq",
    }
    if family not in supported_families:
        result.add_error(f"Unsupported FMHA family: {family}")

    # --- Dtype validation ---
    supported_dtypes = set(arch_info["supported_dtypes"])
    if dtype not in supported_dtypes:
        result.add_error(f"dtype {dtype} is not supported on {arch}")

    if family.startswith("bwd") and dtype not in BWD_DTYPE_MAP:
        result.add_error(
            f"Backward family {family} only supports {sorted(BWD_DTYPE_MAP)}"
        )

    if (
        family.startswith("fwd")
        and not family.startswith("fwd_append")
        and dtype not in FWD_DTYPE_MAP
    ):
        result.add_error(f"Forward family {family} does not recognize dtype {dtype}")

    # --- Pipeline validation ---
    # Combine kernels use a reduction pipeline, not an attention pipeline
    if (
        family != "fwd_splitkv_combine"
        and pipeline not in arch_info["supported_pipelines"]
    ):
        result.add_error(f"pipeline {pipeline} is not supported on {arch}")

    if pipeline in {"v3", "qr_async_trload_v3"}:
        if not arch_info.get("supports_v3", False):
            result.add_warning(
                f"v3 pipeline on {arch} requires supports_v3 in arch specs"
            )

    if pipeline == "qr_async_trload" and not arch_info.get("supports_trload", False):
        result.add_error("qr_async_trload requires a trload-capable architecture")

    if pipeline in {"qr_async_trload", "v3", "qr_async_trload_v3"} and (
        sig["hdim_q"] != sig["hdim_v"] or sig["hdim_q"] not in {64, 128}
    ):
        result.add_error(f"{pipeline} only supports symmetric head dims 64 or 128")

    # --- Global rules (data-driven) ---
    _validate_global_rules(sig, bias, result, global_rules)

    # --- Tile validation (data-driven) ---
    tile = alg["tile"]
    expected_tile_len = 9 if family == "bwd_dq_dk_dv" else 6
    if len(tile) != expected_tile_len or len(alg["wave"]) != 9 or len(alg["warp"]) != 9:
        result.add_error(
            f"tile/wave/warp must have {expected_tile_len}/9/9 elements for {family}"
        )
    elif family in {"fwd", "fwd_pagedkv", "fwd_splitkv", "batch_prefill"}:
        if not alg.get("skip_tile_validation", False):
            _validate_tile_against_specs(
                tile,
                sig["hdim_q"],
                sig["hdim_v"],
                dtype,
                pipeline,
                arch_info,
                result,
                family=family,
            )

    # --- QR pipeline MFMA instruction count validation ---
    # block_fmha_pipeline_qr_ks_vs.hpp:354 requires NumMfmaInsts % 8 == 0
    # when warp_size == 64 (gfx9) and hdim_q == 256.
    # Only applies to fwd/dq_dk_dv pipelines, NOT to dot_do_o/convert_dq (1D kernels).
    # NumMfmaInsts = (tile_m0/warp_m0) * (tile_n0/warp_n0) * (tile_k0/warp_k0) / (wave_m0*wave_n0)
    _1d_families = {"bwd_dot_do_o", "bwd_convert_dq"}
    if (
        pipeline == "qr"
        and sig["hdim_q"] == 256
        and sig.get("family", "") not in _1d_families
        and arch_info.get("family", "").startswith("cdna")
        and len(tile) >= 3
        and len(alg["wave"]) >= 2
        and len(alg["warp"]) >= 3
    ):
        wm, wn, wk = alg["warp"][0], alg["warp"][1], alg["warp"][2]
        gm, gn = alg["wave"][0], alg["wave"][1]
        if wm > 0 and wn > 0 and wk > 0 and gm > 0 and gn > 0:
            num_mfma = (tile[0] // wm) * (tile[1] // wn) * (tile[2] // wk) // (gm * gn)
            if num_mfma % 8 != 0:
                result.add_error(
                    f"qr pipeline h256 on {arch}: NumMfmaInsts={num_mfma} "
                    f"(must be divisible by 8). tile=({tile[0]},{tile[1]},{tile[2]}), "
                    f"warp=({wm},{wn},{wk}), wave=({gm},{gn})"
                )

    if alg["block_per_cu"] <= 0 and alg["block_per_cu"] != -1:
        result.add_error("block_per_cu must be positive or -1 (auto)")
    if alg["num_wave_groups"] <= 0:
        result.add_error("num_wave_groups must be positive")

    # --- Family-specific rules ---
    if family == "batch_prefill":
        if sig["vlayout"] != "r":
            result.add_error("batch_prefill only supports row-major V layout")
        if not sig["paged_kv"]:
            result.add_error("batch_prefill requires paged_kv=true")
        if sig["page_size"] <= 0 or (sig["page_size"] & (sig["page_size"] - 1)) != 0:
            result.add_error("batch_prefill page_size must be a positive power of two")
        if sig["mode"] != "group":
            result.add_error("batch_prefill requires group mode")
        if pipeline != "qr_async":
            result.add_error("batch_prefill currently uses qr_async pipeline")

    if family == "fwd_appendkv":
        if sig["mode"] != "batch":
            result.add_error("fwd_appendkv uses batch-mode public API surface")
        if pipeline != "appendkv":
            result.add_error("fwd_appendkv must use appendkv pipeline")
        if sig["vlayout"] != "r":
            result.add_error("fwd_appendkv currently only supports row-major V")

    if family == "fwd_splitkv_combine":
        if sig["mode"] not in {"batch", "group"}:
            result.add_error("fwd_splitkv_combine requires batch or group mode")
        combine_bn1 = arch_specs.get("splitkv_combine", {}).get("combine_bn1", 32)
        if tile[3] != combine_bn1:
            result.add_error(f"fwd_splitkv_combine requires bn1={combine_bn1}")
        if sig["hdim_v"] < tile[3] or sig["hdim_v"] % tile[3] != 0:
            result.add_error("fwd_splitkv_combine requires hdim_v divisible by bn1")

    if family == "fwd_pagedkv":
        if pipeline != "qr_pagedkv":
            result.add_error("fwd_pagedkv must use qr_pagedkv pipeline")
        if not sig["paged_kv"]:
            result.add_error("fwd_pagedkv requires paged_kv=true")
        if sig["vlayout"] != "r":
            result.add_error("fwd_pagedkv currently only supports row-major V")

    if family == "fwd_splitkv":
        if pipeline not in {"qr", "qr_nwarp_sshuffle"}:
            result.add_error("fwd_splitkv must use qr or qr_nwarp_sshuffle pipeline")
        if sig["vlayout"] != "r":
            result.add_error("fwd_splitkv currently only supports row-major V")

    if family == "fwd" and sig["vlayout"] != "r":
        result.add_warning("dispatcher forward examples currently assume row-major V")

    if rope != "none" and family != "fwd_appendkv":
        result.add_warning("RoPE is only used by append-KV kernels in the current port")

    if qscale == "kv_blockscale" and family not in {"batch_prefill"}:
        result.add_warning("kv_blockscale is primarily exercised by batch_prefill")

    if kv_memory_layout not in {"vectorized", "linear"}:
        result.add_error(f"Unsupported KV memory layout: {kv_memory_layout}")
    if kv_lookup_table not in {"sglang", "vllm"}:
        result.add_error(f"Unsupported KV lookup table: {kv_lookup_table}")

    if family == "bwd_dot_do_o" and tile[0] not in {16, 32, 64, 128, 256}:
        result.add_error(f"bwd_dot_do_o bm0={tile[0]} not a valid block size")
    if family == "bwd_convert_dq" and tile[0] not in {16, 32, 64, 128, 256}:
        result.add_error("bwd_convert_dq currently expects bm0=64")
    if family == "bwd_dq_dk_dv":
        if tile[3] <= 0 or tile[4] <= 0 or tile[5] <= 0:
            result.add_error("bwd_dq_dk_dv requires valid tile fields")
        if alg["max_seq_len_q"] < 0:
            result.add_error("bwd_dq_dk_dv max_seq_len_q must be >= 0")

    return result
