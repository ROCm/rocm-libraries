#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Grouped Convolution Dispatcher Utilities

Validation, auto-correction, and config helpers for grouped convolution kernels.
Uses shared dispatcher_common for validation logic.

Usage:
    from grouped_conv_utils import (
        GroupedConvValidationResult,
        validate_grouped_conv_config,
        auto_correct_grouped_conv_config,
        get_grouped_conv_default_config,
        GroupedConvDataType,
        format_grouped_conv_summary,
    )

    config = get_grouped_conv_default_config(variant="forward")
    result = validate_grouped_conv_config(config)
    if not result.is_valid:
        config, result = auto_correct_grouped_conv_config(config)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from dispatcher_common import (
    ValidationResultBase,
    auto_correct_trait,
    auto_correct_wave,
    get_arch_filter_data,
    validate_trait_combo,
    validate_wave_config,
    validate_warp_tile_config,
)


# =============================================================================
# GroupedConvValidationResult
# =============================================================================


@dataclass
class GroupedConvValidationResult(ValidationResultBase):
    """Result of grouped conv kernel config validation."""

    variant: str = "forward"

    def __init__(
        self,
        is_valid: bool = True,
        errors: List[str] = None,
        warnings: List[str] = None,
        suggested_fixes: Dict[str, Any] = None,
        variant: str = "forward",
    ):
        super().__init__(
            is_valid=is_valid,
            errors=errors or [],
            warnings=warnings or [],
            suggested_fixes=suggested_fixes or {},
        )
        self.variant = variant


# =============================================================================
# GroupedConvDataType
# =============================================================================


class GroupedConvDataType(Enum):
    """Data types for grouped convolution kernels."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP8 = "fp8"
    BF8 = "bf8"
    INT8 = "int8"


# =============================================================================
# Config Extraction Helpers
# =============================================================================

VALID_VARIANTS = ("forward", "bwd_data", "bwd_weight")
VALID_NDIM_SPATIAL = (1, 2, 3)
BACKWARD_VARIANTS = ("bwd_data", "bwd_weight")
BACKWARD_PIPELINES = ("compv3", "mem")


def _get_tile_config(config: dict) -> dict:
    """Extract tile_config, return empty dict if missing."""
    return config.get("tile_config") or {}


def _get_trait_config(config: dict) -> dict:
    """Extract trait_config, return empty dict if missing."""
    return config.get("trait_config") or {}


def _first(val) -> Any:
    """Get first element if list, else return value."""
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    return val


def _extract_wave_config(tile_config: dict) -> List[int]:
    """Extract [wave_m, wave_n, wave_k] from tile_config.

    Supports both formats:
    - wave_m, wave_n, wave_k (test/codegen format)
    - warp_m, warp_n, warp_k (user spec: wave config stored under warp_*)
    """
    # Prefer wave_m, wave_n, wave_k
    wm = tile_config.get("wave_m") or tile_config.get("warp_m")
    wn = tile_config.get("wave_n") or tile_config.get("warp_n")
    wk = tile_config.get("wave_k") or tile_config.get("warp_k")
    if wm is not None and wn is not None and wk is not None:
        return [_first(wm), _first(wn), _first(wk)]
    return [2, 2, 1]


def _extract_warp_tile_config(tile_config: dict) -> List[int]:
    """Extract [warp_tile_m, warp_tile_n, warp_tile_k] from tile_config."""
    wtm = tile_config.get("warp_tile_m") or tile_config.get("warp_m")
    wtn = tile_config.get("warp_tile_n") or tile_config.get("warp_n")
    wtk = tile_config.get("warp_tile_k") or tile_config.get("warp_k")
    if wtm is not None and wtn is not None and wtk is not None:
        return [_first(wtm), _first(wtn), _first(wtk)]
    return [32, 32, 16]


def _extract_trait_values(trait_config: dict) -> Tuple[str, str, str]:
    """Extract (pipeline, epilogue, scheduler) from trait_config."""
    p = _first(trait_config.get("pipeline", "compv4"))
    e = _first(trait_config.get("epilogue", "cshuffle"))
    s = _first(trait_config.get("scheduler", "intrawave"))
    if isinstance(p, list):
        p = p[0] if p else "compv4"
    if isinstance(e, list):
        e = e[0] if e else "cshuffle"
    if isinstance(s, list):
        s = s[0] if s else "intrawave"
    return (str(p), str(e), str(s))


# =============================================================================
# validate_grouped_conv_config
# =============================================================================


def validate_grouped_conv_config(config: dict) -> GroupedConvValidationResult:
    """Validate a grouped conv kernel config dict.

    Checks:
    - All required keys exist (tile_config, trait_config, variant, ndim_spatial, arch, layout)
    - Wave config via validate_wave_config()
    - Trait combo via validate_trait_combo()
    - Variant is one of "forward", "bwd_data", "bwd_weight"
    - ndim_spatial is 1, 2, or 3
    - Backward variants only use compv3/mem pipeline
    - Arch is supported
    - Warp tile config for arch/dtype

    Returns GroupedConvValidationResult with is_valid, errors, suggested_fixes.
    """
    errors: List[str] = []
    warnings: List[str] = []
    suggested_fixes: Dict[str, Any] = {}

    # Required keys
    required = ("tile_config", "trait_config", "variant", "ndim_spatial", "arch", "layout")
    for key in required:
        if key not in config:
            errors.append(f"Missing required key: {key}")
    if errors:
        return GroupedConvValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            suggested_fixes=suggested_fixes,
            variant=config.get("variant", "forward"),
        )

    tile_config = _get_tile_config(config)
    trait_config = _get_trait_config(config)
    variant = _first(config.get("variant", "forward"))
    ndim_spatial = config.get("ndim_spatial")
    arch = config.get("arch", "gfx942")
    layout = config.get("layout", "nhwgc")
    dtype = config.get("dtype", "fp16")

    if isinstance(variant, list):
        variant = variant[0] if variant else "forward"
    variant = str(variant)

    # Support "2d_fwd" style aliases
    variant_aliases = {
        "2d_fwd": "forward",
        "2d_bwdd": "bwd_data",
        "2d_bwdw": "bwd_weight",
    }
    variant = variant_aliases.get(variant, variant)

    if variant not in VALID_VARIANTS:
        errors.append(
            f"Invalid variant: {variant}. Valid: {', '.join(VALID_VARIANTS)}"
        )
        suggested_fixes["variant"] = "forward"

    if ndim_spatial is not None:
        ndim = ndim_spatial
        if isinstance(ndim, list):
            ndim = ndim[0] if ndim else 2
        if ndim not in VALID_NDIM_SPATIAL:
            errors.append(
                f"Invalid ndim_spatial: {ndim}. Valid: {', '.join(map(str, VALID_NDIM_SPATIAL))}"
            )
            suggested_fixes["ndim_spatial"] = 2

    # Backward variants: only compv3/mem pipeline
    pipeline, epilogue, scheduler = _extract_trait_values(trait_config)
    if variant in BACKWARD_VARIANTS and pipeline not in BACKWARD_PIPELINES:
        errors.append(
            f"Backward variant '{variant}' requires pipeline compv3 or mem, got {pipeline}"
        )
        suggested_fixes["pipeline"] = "compv3"

    # Trait combo
    ok, msg = validate_trait_combo(pipeline, epilogue, scheduler)
    if not ok:
        errors.append(msg)
        suggested_fixes["scheduler"] = "intrawave"

    # Wave config
    wave_cfg = _extract_wave_config(tile_config)
    ok, msg = validate_wave_config(wave_cfg, arch)
    if not ok:
        errors.append(msg)
        arch_data = get_arch_filter_data()
        valid_waves = arch_data["warp_combos"].get(arch, [[2, 2, 1]])
        if valid_waves:
            suggested_fixes["wave_m"] = valid_waves[0][0]
            suggested_fixes["wave_n"] = valid_waves[0][1]
            suggested_fixes["wave_k"] = valid_waves[0][2]

    # Warp tile config (use dtype from config or fp16)
    warp_cfg = _extract_warp_tile_config(tile_config)
    ok, msg = validate_warp_tile_config(warp_cfg, arch, dtype)
    if not ok:
        errors.append(msg)
        arch_data = get_arch_filter_data()
        acc = "int32" if dtype == "int8" else "fp32"
        dtype_key = f"{dtype}_{dtype}_{acc}"
        valid_tiles = (
            arch_data["warp_tile_combos"]
            .get(arch, {})
            .get(dtype_key, [[32, 32, 16], [16, 16, 16]])
        )
        if valid_tiles:
            suggested_fixes["warp_tile_m"] = valid_tiles[0][0]
            suggested_fixes["warp_tile_n"] = valid_tiles[0][1]
            suggested_fixes["warp_tile_k"] = valid_tiles[0][2]

    # Arch supported
    arch_data = get_arch_filter_data()
    if arch not in arch_data["supported_archs"]:
        errors.append(
            f"Unsupported architecture: {arch}. "
            f"Supported: {', '.join(arch_data['supported_archs'])}"
        )

    return GroupedConvValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        suggested_fixes=suggested_fixes,
        variant=variant,
    )


# =============================================================================
# auto_correct_grouped_conv_config
# =============================================================================


def auto_correct_grouped_conv_config(
    config: dict,
) -> Tuple[dict, GroupedConvValidationResult]:
    """Auto-correct invalid grouped conv config.

    Uses shared auto_correct_wave() and auto_correct_trait().
    Returns (corrected_config, validation_result).
    """
    import copy

    result = validate_grouped_conv_config(config)
    corrected = copy.deepcopy(config)

    if result.is_valid:
        return corrected, result

    tile_config = corrected.setdefault("tile_config", {})
    trait_config = corrected.setdefault("trait_config", {})

    # Apply wave correction
    wave_cfg = _extract_wave_config(tile_config)
    arch = config.get("arch", "gfx942")
    fixed_wave = auto_correct_wave(wave_cfg, arch)
    tile_config["wave_m"] = fixed_wave[0]
    tile_config["wave_n"] = fixed_wave[1]
    tile_config["wave_k"] = fixed_wave[2]

    # Apply trait correction
    pipeline, epilogue, scheduler = _extract_trait_values(trait_config)
    fixed_pipeline, fixed_scheduler = auto_correct_trait(pipeline, scheduler)
    trait_config["pipeline"] = fixed_pipeline
    trait_config["scheduler"] = fixed_scheduler

    # Apply pipeline fix for backward variants
    variant = _first(config.get("variant", "forward"))
    if isinstance(variant, list):
        variant = variant[0] if variant else "forward"
    variant_aliases = {"2d_fwd": "forward", "2d_bwdd": "bwd_data", "2d_bwdw": "bwd_weight"}
    variant = variant_aliases.get(str(variant), str(variant))
    if variant in BACKWARD_VARIANTS and fixed_pipeline not in BACKWARD_PIPELINES:
        trait_config["pipeline"] = "compv3"

    # Apply suggested fixes for warp tile if present
    if "warp_tile_m" in result.suggested_fixes:
        tile_config["warp_tile_m"] = result.suggested_fixes["warp_tile_m"]
        tile_config["warp_tile_n"] = result.suggested_fixes["warp_tile_n"]
        tile_config["warp_tile_k"] = result.suggested_fixes["warp_tile_k"]

    # Re-validate
    result = validate_grouped_conv_config(corrected)
    return corrected, result


# =============================================================================
# get_grouped_conv_default_config
# =============================================================================


def get_grouped_conv_default_config(
    variant: str = "forward",
    ndim_spatial: int = 2,
    arch: str = "gfx942",
    layout: str = "nhwgc",
    dtype: str = "fp16",
) -> dict:
    """Return a valid default config dict for grouped conv.

    Supports variant aliases: "2d_fwd" -> forward, "2d_bwdd" -> bwd_data, etc.
    """
    variant_aliases = {
        "2d_fwd": "forward",
        "2d_bwdd": "bwd_data",
        "2d_bwdw": "bwd_weight",
    }
    variant = variant_aliases.get(variant, variant)

    # Backward variants use compv3/mem pipeline
    if variant in BACKWARD_VARIANTS:
        pipeline = "compv3"
    else:
        pipeline = "compv4"

    config = {
        "tile_config": {
            "tile_m": [1],
            "tile_n": [128],
            "tile_k": [128],
            "wave_m": [2],
            "wave_n": [2],
            "wave_k": [1],
            "warp_tile_m": [32],
            "warp_tile_n": [32],
            "warp_tile_k": [16],
        },
        "trait_config": {
            "pipeline": [pipeline],
            "epilogue": ["cshuffle"],
            "scheduler": ["intrawave"],
            "pad_m": [True],
            "pad_n": [True],
            "pad_k": [True],
        },
        "variant": variant,
        "ndim_spatial": ndim_spatial,
        "arch": arch,
        "layout": layout,
        "dtype": dtype,
    }

    # For validation we need scalar values in nested dicts when using
    # the extractors; also support list format for codegen.
    # Return format matching user spec (lists for codegen compatibility)
    return config


# =============================================================================
# format_grouped_conv_summary
# =============================================================================


def format_grouped_conv_summary(config: dict) -> str:
    """Format a grouped conv config into a human-readable multi-line string."""
    lines: List[str] = []
    tile_config = _get_tile_config(config)
    trait_config = _get_trait_config(config)

    variant = config.get("variant", "?")
    ndim = config.get("ndim_spatial", "?")
    arch = config.get("arch", "?")
    layout = config.get("layout", "?")
    dtype = config.get("dtype", "fp16")

    lines.append(f"Grouped Conv Config: {variant} {ndim}D")
    lines.append(f"  Arch:    {arch}")
    lines.append(f"  Layout:  {layout}")
    lines.append(f"  Dtype:   {dtype}")

    if tile_config:
        wave = _extract_wave_config(tile_config)
        warp = _extract_warp_tile_config(tile_config)
        tile_m = _first(tile_config.get("tile_m", 1))
        tile_n = _first(tile_config.get("tile_n", 128))
        tile_k = _first(tile_config.get("tile_k", 128))
        lines.append(f"  Tile:    M={tile_m} N={tile_n} K={tile_k}")
        lines.append(f"  Wave:    {wave[0]}x{wave[1]}x{wave[2]}")
        lines.append(f"  Warp:    {warp[0]}x{warp[1]}x{warp[2]}")

    if trait_config:
        pipeline = _first(trait_config.get("pipeline", "?"))
        epilogue = _first(trait_config.get("epilogue", "?"))
        scheduler = _first(trait_config.get("scheduler", "?"))
        lines.append(f"  Traits:  pipeline={pipeline} epilogue={epilogue} scheduler={scheduler}")

    return "\n".join(lines) if lines else "(empty config)"
