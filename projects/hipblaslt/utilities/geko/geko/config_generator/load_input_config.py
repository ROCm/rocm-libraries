# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Load and normalize input YAML configuration.

Defaults, validation, ARCH-derived hardware (``CUs``, ``XCC``, ``DTYPE_MIs``,
``WGMUnit``), and GA / kernel-cap rules live here so every entry point shares
the same behavior.
"""

from __future__ import annotations

__all__ = [
    "load_prepared_config_from_yaml",
    "validate_input_config",
    "apply_input_config_defaults",
    "get_gemm_problem",
    "gemm_configs_from_gemm_log_path",
    "gemm_configs_from_gemm_dataframe",
]

import sys
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
import logging

from geko.bench.log import parse as parse_gemm_log
from geko.config_generator.constants import (
    CONFIG_DEFAULTS_BY_ARCH,
    ENV_UPDATABLE_KEYS,
    HARDWARE_MAP,
    MAX_NUM_KERNELS_PER_CONFIG,
    REQUIRED_CONFIG_FIELDS,
)
from geko.config_generator.sizes import get_sizes
from geko.constants import GEMM_TYPE_FIELDS
from geko.schemas import GemmConfig, GemmType

logger = logging.getLogger("GEKO")


def _coerce_env_value(key: str, raw_value: str, current_value: Any) -> Any:
    """Coerce env override strings to the current config value type when possible."""
    text = raw_value.strip().lower()

    if isinstance(current_value, bool):
        if text in {"1", "true"}:
            return True
        if text in {"0", "false"}:
            return False
        logger.warning(
            "Ignoring env override %s=%r: expected boolean token",
            key,
            raw_value,
        )
        return current_value

    if isinstance(current_value, int):
        try:
            return int(raw_value)
        except ValueError:
            logger.warning(
                "Ignoring env override %s=%r: expected integer",
                key,
                raw_value,
            )
            return current_value

    if isinstance(current_value, float):
        try:
            return float(raw_value)
        except ValueError:
            logger.warning(
                "Ignoring env override %s=%r: expected float",
                key,
                raw_value,
            )
            return current_value

    return raw_value


def _apply_env_config_overrides(config: Dict[str, Any]) -> None:
    """Override config values with same-name environment variables."""
    for key in ENV_UPDATABLE_KEYS:
        env_value = os.getenv(key)
        if env_value is None:
            continue

        if key not in config:
            logger.warning(
                "Env override %s=%r ignored: no such config key",
                key,
                env_value,
            )
            continue
        
        old_value = config[key]
        new_value = _coerce_env_value(key, env_value, old_value)
        if new_value != old_value:
            config[key] = new_value
            logger.warning(
                "Config override from env %s: %r -> %r",
                key,
                old_value,
                new_value,
            )


def gemm_configs_from_gemm_dataframe(df: pd.DataFrame) -> List[GemmConfig]:
    """Group df by GEMM_TYPE_FIELDS; sizes use M,N,K or m,n,k columns per summarize output.

    Args:
        df: Non-empty GEMM table; empty or None yields [].

    Returns:
        One GemmConfig per dtype/transpose group (GemmType.from_hipblaslt).
    """
    if df is None or df.empty:
        return []
    size_cols = (
        ["M", "N", "batch_count", "K"] if "M" in df.columns else ["m", "n", "batch_count", "k"]
    )
    gemm_configs: List[GemmConfig] = []
    for gemm_key, gby in df.groupby(list(GEMM_TYPE_FIELDS), sort=False):
        sizes = gby[size_cols].values.tolist()
        fields = dict(zip(GEMM_TYPE_FIELDS, gemm_key))
        gt = GemmType.from_hipblaslt(
            fields["transA"],
            fields["transB"],
            fields["a_type"],
            fields["b_type"],
            fields["c_type"],
            fields["compute_type"],
        )
        gemm_configs.append(GemmConfig(gt, sizes))
    return gemm_configs


def gemm_configs_from_gemm_log_path(log_file: str | Path) -> List[GemmConfig]:
    """parse_gemm_log(as_df=True) then gemm_configs_from_gemm_dataframe (no bench)."""
    df = parse_gemm_log(log_file, as_df=True)
    return gemm_configs_from_gemm_dataframe(df)


def get_gemm_problem(config: dict) -> None:
    """Set config['GemmProblems'] to a one-element list from YAML GEMM fields.

    Expects validate_input_config and apply_input_config_defaults to have run.
    Resolves sizes via get_sizes (explicit list or grid). Builds GemmType
    with from_tensile from the YAML dtype strings.

    Legacy keys (DataType, Sizes, …) stay on config for defaults; the
    generator consumes GemmProblems (and per-iteration GemmProblem in
    config_generator.run).

    Args:
        config: Mutable config dict; updated in place.
    """
    gemm_type = GemmType.from_tensile(
        config["TRANSA"],
        config["TRANSB"],
        str(config["DataType"]),
        str(config["DestDataType"]),
        str(config["ComputeDataType"]),
    )
    sizes = get_sizes(config)
    config["GemmProblems"] = [GemmConfig(gemm_type, sizes)]


def _load_config_from_yaml(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(config_path) as stream:
        return yaml.full_load(stream)


def _apply_arch_hardware_defaults(config: Dict[str, Any]) -> None:
    """Fill hardware fields from ``HARDWARE_MAP[ARCH]`` when the key is absent.

    User YAML may set CUs, XCC, and/or WGMUnit to override the defaults for
    the chosen ARCH. DTYPE_MIs is intentionally not populated here; the
    allowlist is deferred to MIDesign, which derives it from ARCH together
    with the active GemmType.

    Requires ARCH to be set. Mutates config in place.
    """
    hw = HARDWARE_MAP[config["ARCH"]]
    config.setdefault("CUs", hw["CUs"])
    config.setdefault("XCC", hw["XCC"])
    config.setdefault("WGMUnit", config["XCC"])


def validate_input_config(config: Dict[str, Any]) -> None:
    """Ensure required keys are present and ARCH is supported.

    Call on a dict loaded from YAML before apply_input_config_defaults.
    Optional fields may be missing; defaults are applied later.

    When SIZE_OPTION is 2, GEMM_LOG_PATH is required and Tensile dtype /
    layout keys from REQUIRED_CONFIG_FIELDS are not. Otherwise all
    REQUIRED_CONFIG_FIELDS must be present.

    ARCH must already be set (YAML or arch= to load_prepared_config_from_yaml).

    Raises:
        ValueError: If required fields or ARCH are invalid.
    """
    try:
        so_int = int(config["SIZE_OPTION"]) if config.get("SIZE_OPTION") is not None else None
    except (TypeError, ValueError):
        so_int = None

    if so_int == 2:
        if not config.get("GEMM_LOG_PATH"):
            raise ValueError("SIZE_OPTION 2 requires GEMM_LOG_PATH")
    else:
        missing = [k for k in REQUIRED_CONFIG_FIELDS if k not in config]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")

    if config["ARCH"] not in HARDWARE_MAP:
        raise ValueError(
            f"Unknown ARCH '{config['ARCH']}'. "
            f"Valid options: {list(HARDWARE_MAP.keys())}"
        )


def apply_input_config_defaults(config: Dict[str, Any]) -> None:
    """Apply per-ARCH optional defaults, hardware fields, and kernel-cap rules in place.

    Run validate_input_config first for YAML-loaded dicts. Callers must set
    ARCH and either classic GEMM keys or GEMM_LOG_PATH with SIZE_OPTION 2;
    optional keys are filled from CONFIG_DEFAULTS_BY_ARCH.

    Mutates config.
    """
    for key, default in CONFIG_DEFAULTS_BY_ARCH[config["ARCH"]].items():
        config.setdefault(key, default)

    _apply_arch_hardware_defaults(config)

    if not config["MACROTILE_OPT"]:
        config["MT_DU"] = None

    if config["MACROTILE_OPT"] and not config["GA"]:
        raise NotImplementedError("MACROTILE_OPT only valid with GA.")

    if config["GA"]:
        config["MAX_NUM_KERNELS_PER_CONFIG"] = sys.maxsize
    else:
        config.setdefault("MAX_NUM_KERNELS_PER_CONFIG", MAX_NUM_KERNELS_PER_CONFIG)
    
    _apply_env_config_overrides(config)

def load_prepared_config_from_yaml(
    config_path: str | Path | None = None,
    *,
    arch: str | None = None,
    gemm_log_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Load YAML (optional), validate, apply defaults, and populate GemmProblems.

    If config_path is None, starts from an empty dict; pass arch and
    gemm_log_path for workload-only mode (sets ARCH, GEMM_LOG_PATH,
    SIZE_OPTION 2).

    After validate_input_config and apply_input_config_defaults,
    get_gemm_problem (SIZE_OPTION 0/1) or gemm_configs_from_gemm_log_path
    (SIZE_OPTION 2) fills GemmProblems.

    Returns:
        Config dict ready for config_generator entry points.
    """
    if config_path is not None:
        config = _load_config_from_yaml(config_path)
    else:
        config = {}

    if arch is not None:
        config["ARCH"] = arch
    if gemm_log_path is not None:
        config["GEMM_LOG_PATH"] = str(Path(gemm_log_path))
        config["SIZE_OPTION"] = 2

    if not config.get("ARCH"):
        raise ValueError("No ARCH defined; set ARCH in YAML or pass arch=.")

    validate_input_config(config)
    apply_input_config_defaults(config)

    _so = config.get("SIZE_OPTION", -1)
    if _so == 2:
        gemm_problems = gemm_configs_from_gemm_log_path(config["GEMM_LOG_PATH"])
        if not gemm_problems:
            raise ValueError(
                f"No GEMM entries found in GEMM_LOG_PATH {config['GEMM_LOG_PATH']!r}"
            )
        config["GemmProblems"] = gemm_problems
    else:
        get_gemm_problem(config)

    return config
