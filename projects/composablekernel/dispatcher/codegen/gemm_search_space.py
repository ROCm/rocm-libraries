# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Dispatcher universal GEMM search space: enumerate, sample, and export configs.

Wraps gemm_utils.expand_sweep to produce a stratified, budget-limited set of
GemmKernelConfig objects across all supported (dtype, layout) combinations,
mirroring tile_engine's daily-tier sampling for CI coverage.

Public surface:
    DEFAULT_CI_CONFIG   -- path to tile_engine's default_ci_config.json
    DTYPES              -- supported dtype list (matches tile engine gfx942 set)
    LAYOUTS             -- supported layout list
    enumerate_configs   -- full search space across all dtype/layout combos
    sample_configs      -- budget-limited stratified subset with seed support
    daily_seed          -- date-based rotating seed (mirrors tile engine convention)
"""

from __future__ import annotations

import hashlib
import random
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

# Path to tile engine's CI config -- reused directly so dispatcher and tile
# engine sweep the same tile/trait dimensions without duplication.
DEFAULT_CI_CONFIG = (
    Path(__file__).parent.parent.parent
    / "tile_engine/ops/gemm/configs/default_ci_config.json"
)

# Supported dtypes on gfx942 for universal GEMM (mirrors tile engine CI flags).
DTYPES = ["fp16", "bf16", "fp8", "bf8"]

# Supported layout combos (A-major, B-major, C-major).
# Column-major C is rejected by the universal GEMM codegen, so all four
# A/B combinations keep row-major C -- matching tile engine's layout flags.
LAYOUTS = ["rcr", "rrr", "crr", "ccr"]


def enumerate_configs(
    arch: str,
    dtypes: Optional[List[str]] = None,
    layouts: Optional[List[str]] = None,
    config_path: Optional[str] = None,
) -> Dict[str, list]:
    """Enumerate the full search space, grouped by (dtype, layout) stratum.

    Args:
        arch:        GPU architecture string (e.g. 'gfx942').
        dtypes:      Dtype list to sweep; defaults to DTYPES.
        layouts:     Layout list to sweep; defaults to LAYOUTS.
        config_path: Path to a TE-format config JSON; defaults to DEFAULT_CI_CONFIG.

    Returns:
        Dict mapping (dtype, layout) tuple-keys to lists of GemmKernelConfig.
        Key format: "<dtype>/<layout>" (e.g. "fp16/rcr").
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    from gemm_utils import expand_sweep

    cfg = str(config_path or DEFAULT_CI_CONFIG)
    dtypes = dtypes or DTYPES
    layouts = layouts or LAYOUTS

    strata: Dict[str, list] = {}
    for dtype in dtypes:
        for layout in layouts:
            key = f"{dtype}/{layout}"
            configs = expand_sweep(cfg, arch=arch, dtype=dtype, layout=layout)
            strata[key] = configs

    return strata


def daily_seed() -> int:
    """Return a date-based integer seed that rotates daily.

    Mirrors tile_engine's daily seed rotation so the sampled subset changes
    each day, giving broad coverage over time without a fixed bias.
    """
    today = date.today().isoformat()
    return int(hashlib.md5(today.encode()).hexdigest(), 16) % (2**31)


def sample_configs(
    strata: Dict[str, list],
    budget: int,
    seed: Optional[int] = None,
) -> list:
    """Stratified budget-limited sample from the search space.

    Distributes the budget equally across all non-empty (dtype, layout) strata,
    then samples randomly within each stratum. If a stratum has fewer configs
    than its share, all are included and the remainder is redistributed.

    Args:
        strata:  Output of enumerate_configs -- dict of stratum -> config list.
        budget:  Total number of configs to return (hard cap).
        seed:    RNG seed. Pass None to use daily_seed() for daily rotation.

    Returns:
        Flat list of GemmKernelConfig, length <= budget.
    """
    if seed is None:
        seed = daily_seed()

    rng = random.Random(seed)

    nonempty = {k: v for k, v in strata.items() if v}
    if not nonempty:
        return []

    n_strata = len(nonempty)
    per_stratum = max(1, budget // n_strata)
    remainder = budget - per_stratum * n_strata

    selected = []
    for i, (key, configs) in enumerate(nonempty.items()):
        # Last stratum absorbs the rounding remainder.
        alloc = per_stratum + (remainder if i == n_strata - 1 else 0)
        alloc = min(alloc, len(configs))
        chosen = rng.sample(configs, alloc)
        selected.extend(chosen)

    rng.shuffle(selected)
    return selected[:budget]
