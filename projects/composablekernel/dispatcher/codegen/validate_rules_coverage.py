#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validation script: compare rule-generated configs against JSON profiler configs.

Usage:
    python validate_rules_coverage.py [--config-set {tests,profiler}] [--extract] [--arch ARCH]

Modes:
    Default: Load JSON configs + generate from rules, report coverage gaps.
    --extract: Dump all unique tile/warp/vec 9-tuples from JSON in Python format.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs" / "grouped_conv"

sys.path.insert(0, str(SCRIPT_DIR))

from unified_grouped_conv_codegen import (
    DepthwiseConvKernelConfig,
    GroupedConvKernelConfig,
    GroupedConvVariant,
    StreamKConfig,
    StreamKReductionStrategy,
    get_default_configs,
)

# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

_VARIANT_DIRS = {
    "forward": "forward",
    "bwd_data": "backward_data",
    "bwd_weight": "backward_weight",
}

_DTYPE_MAP = {
    "bf16": "bf16",
    "fp16": "fp16",
    "fp32": "fp32",
}

_LAYOUT_FILES = {
    2: ["nhwgc"],   # 2D
    3: ["ndhwgc"],  # 3D
}


def load_json_instances(config_set: str = "profiler") -> List[Dict[str, Any]]:
    """Load all instances from JSON config files.

    Returns a flat list of dicts, each with keys matching the JSON instance fields
    plus added 'variant', 'datatype', 'layout', 'ndim_spatial'.
    """
    instances = []
    for variant_key, variant_dir in _VARIANT_DIRS.items():
        dir_path = CONFIGS_DIR / variant_dir / config_set
        if not dir_path.is_dir():
            continue
        for json_file in sorted(dir_path.glob("*.json")):
            stem = json_file.stem  # e.g., "nhwgc_bf16"
            try:
                data = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  WARNING: could not load {json_file}: {e}", file=sys.stderr)
                continue

            file_insts = data.get("instances", [])
            ndim = data.get("ndim_spatial", 2)
            layout = data.get("layout", stem.split("_")[0])
            dtype = data.get("datatype", stem.split("_")[-1] if "_" in stem else "bf16")

            for inst in file_insts:
                if not isinstance(inst, dict):
                    continue
                # Skip malformed instances with missing/zero tile dimensions
                if not inst.get("tile_m") or not inst.get("tile_n") or not inst.get("tile_k"):
                    continue
                enriched = dict(inst)
                enriched.setdefault("variant", variant_key)
                enriched.setdefault("datatype", dtype)
                enriched.setdefault("layout", layout)
                enriched.setdefault("ndim_spatial", ndim)
                instances.append(enriched)

    return instances


# ---------------------------------------------------------------------------
# Canonical key generation
# ---------------------------------------------------------------------------

def _inst_to_key(inst: Dict[str, Any]) -> FrozenSet:
    """Convert a JSON instance dict to a frozenset-based canonical key."""
    sk_enabled = inst.get("streamk_enabled", False)
    sk_strategy = inst.get("streamk_reduction_strategy") or ""
    sk_persistent = inst.get("streamk_persistent", False)

    return frozenset({
        ("variant",             inst.get("variant", "")),
        ("ndim_spatial",        int(inst.get("ndim_spatial", 2))),
        ("tile_m",              int(inst.get("tile_m", 0))),
        ("tile_n",              int(inst.get("tile_n", 0))),
        ("tile_k",              int(inst.get("tile_k", 0))),
        ("warp_m",              int(inst.get("warp_m", 0))),
        ("warp_n",              int(inst.get("warp_n", 0))),
        ("warp_k",              int(inst.get("warp_k", 1))),
        ("warp_tile_m",         int(inst.get("warp_tile_m", 0))),
        ("warp_tile_n",         int(inst.get("warp_tile_n", 0))),
        ("warp_tile_k",         int(inst.get("warp_tile_k", 0))),
        ("pipeline",            inst.get("pipeline", "")),
        ("scheduler",           inst.get("scheduler", "")),
        ("vec_a",               int(inst.get("vector_size_a", inst.get("vec_a", 4)))),
        ("vec_b",               int(inst.get("vector_size_b", inst.get("vec_b", 8)))),
        ("vec_c",               int(inst.get("vector_size_c", inst.get("vec_c", 8)))),
        ("double_smem_buffer",  bool(inst.get("double_smem_buffer", False))),
        ("two_stage",           bool(inst.get("two_stage", False))),
        ("explicit_gemm",       bool(inst.get("explicit_gemm", False))),
        ("split_image",         bool(inst.get("split_image", False))),
        ("num_groups_to_merge", int(inst.get("num_groups_to_merge", 1))),
        ("specialization",      inst.get("specialization", "default") or "default"),
        ("streamk_enabled",     bool(sk_enabled)),
        ("streamk_persistent",  bool(sk_persistent) if sk_enabled else False),
    })


_VARIANT_ENUM = {
    "forward":    GroupedConvVariant.FORWARD,
    "bwd_data":   GroupedConvVariant.BACKWARD_DATA,
    "bwd_weight": GroupedConvVariant.BACKWARD_WEIGHT,
}

_VARIANT_STR = {v: k for k, v in _VARIANT_ENUM.items()}


def _config_to_key(cfg: GroupedConvKernelConfig) -> FrozenSet:
    """Convert a GroupedConvKernelConfig to a canonical key."""
    t = cfg.tile
    tr = cfg.trait
    sk = tr.streamk_config

    return frozenset({
        ("variant",             _VARIANT_STR.get(cfg.variant, str(cfg.variant))),
        ("ndim_spatial",        cfg.ndim_spatial),
        ("tile_m",              t.tile_m),
        ("tile_n",              t.tile_n),
        ("tile_k",              t.tile_k),
        ("warp_m",              t.warp_m),
        ("warp_n",              t.warp_n),
        ("warp_k",              t.warp_k),
        ("warp_tile_m",         t.warp_tile_m),
        ("warp_tile_n",         t.warp_tile_n),
        ("warp_tile_k",         t.warp_tile_k),
        ("pipeline",            tr.pipeline),
        ("scheduler",           tr.scheduler),
        ("vec_a",               cfg.vector_size_a),
        ("vec_b",               cfg.vector_size_b),
        ("vec_c",               cfg.vector_size_c),
        ("double_smem_buffer",  tr.double_smem_buffer),
        ("two_stage",           tr.two_stage),
        ("explicit_gemm",       tr.explicit_gemm),
        ("split_image",         tr.split_image),
        ("num_groups_to_merge", tr.num_groups_to_merge),
        ("specialization",      tr.specialization or "default"),
        ("streamk_enabled",     sk.streamk_enabled),
        ("streamk_persistent",  sk.streamk_persistent if sk.streamk_enabled else False),
    })


def _key_to_dict(key: FrozenSet) -> Dict:
    return dict(key)


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def analyze_coverage(
    json_instances: List[Dict],
    generated_configs: List[GroupedConvKernelConfig],
) -> Dict:
    """Compare JSON instances against generated configs.

    Returns dict with:
      - 'json_keys': set of canonical keys from JSON
      - 'gen_keys': set of canonical keys from generated configs
      - 'covered': JSON keys that appear in generated
      - 'missing': JSON keys absent from generated
      - 'extra': generated keys not in any JSON file
    """
    json_keys: Set[FrozenSet] = set()
    for inst in json_instances:
        json_keys.add(_inst_to_key(inst))

    gen_keys: Set[FrozenSet] = set()
    for cfg in generated_configs:
        gen_keys.add(_config_to_key(cfg))

    covered = json_keys & gen_keys
    missing = json_keys - gen_keys
    extra = gen_keys - json_keys

    return {
        "json_keys": json_keys,
        "gen_keys": gen_keys,
        "covered": covered,
        "missing": missing,
        "extra": extra,
    }


# ---------------------------------------------------------------------------
# Extract mode
# ---------------------------------------------------------------------------

def extract_tile_data(json_instances: List[Dict]) -> None:
    """Print unique tile 9-tuples from JSON instances in Python format."""
    by_variant: Dict[str, Set[Tuple]] = defaultdict(set)

    for inst in json_instances:
        variant = inst.get("variant", "unknown")
        try:
            tup = (
                int(inst["tile_m"]),
                int(inst["tile_n"]),
                int(inst["tile_k"]),
                int(inst["warp_m"]),
                int(inst["warp_n"]),
                int(inst.get("warp_k", 1)),
                int(inst["warp_tile_m"]),
                int(inst["warp_tile_n"]),
                int(inst["warp_tile_k"]),
            )
        except (KeyError, ValueError):
            continue
        by_variant[variant].add(tup)

    all_shapes: Dict[str, Set[Tuple[int,int,int]]] = defaultdict(set)
    for variant, tuples in by_variant.items():
        for tup in tuples:
            all_shapes[variant].add(tup[:3])

    for variant in sorted(by_variant):
        print(f"\n# ---- {variant} ----")
        print(f"{variant}_tile_configs_9 = [")
        for tup in sorted(by_variant[variant]):
            print(f"    {tup},")
        print("]")
        print(f"\n{variant}_tile_shapes_3 = [")
        for tup in sorted(all_shapes[variant]):
            print(f"    {tup},")
        print("]")

    # Global union 3-tuples per direction
    fwd = all_shapes.get("forward", set())
    bwd_d = all_shapes.get("bwd_data", set())
    bwd_w = all_shapes.get("bwd_weight", set())
    common = fwd & bwd_d & bwd_w
    print("\n# ---- Computed tile list sets ----")
    print(f"COMMON_TILES (intersection of all 3 directions): {len(common)}")
    print(f"FWD_TILES (forward only): {len(fwd - common)}")
    print(f"BWD_DATA_TILES (bwd_data only): {len(bwd_d - common)}")
    print(f"BWD_WEIGHT_TILES (bwd_weight only): {len(bwd_w - common)}")


# ---------------------------------------------------------------------------
# Pretty-print a missing instance
# ---------------------------------------------------------------------------

def _format_missing(key: FrozenSet) -> str:
    d = _key_to_dict(key)
    tile = f"({d.get('tile_m')},{d.get('tile_n')},{d.get('tile_k')})"
    wave = f"({d.get('warp_m')},{d.get('warp_n')},{d.get('warp_k')})"
    warp = f"({d.get('warp_tile_m')},{d.get('warp_tile_n')},{d.get('warp_tile_k')})"
    pipe = f"{d.get('pipeline')}/{d.get('scheduler')}"
    vec = f"({d.get('vec_a')},{d.get('vec_b')},{d.get('vec_c')})"
    spec = d.get("specialization", "default")
    flags = []
    if d.get("double_smem_buffer"):
        flags.append("dsb")
    if d.get("two_stage"):
        flags.append("2stage")
    if d.get("explicit_gemm"):
        flags.append("explicit_gemm")
    if d.get("split_image"):
        flags.append("split_image")
    if d.get("num_groups_to_merge", 1) > 1:
        flags.append(f"gm{d['num_groups_to_merge']}")
    if d.get("streamk_enabled"):
        flags.append("streamk" + ("_persistent" if d.get("streamk_persistent") else ""))
    flag_str = " " + ",".join(flags) if flags else ""
    variant = d.get("variant", "?")
    ndim = d.get("ndim_spatial", 2)
    return (
        f"  [{variant}/{ndim}d] tile={tile} wave={wave} warp={warp} "
        f"pipe={pipe} vec={vec} spec={spec}{flag_str}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate rule-generated configs against JSON profiler configs."
    )
    parser.add_argument(
        "--rule-set",
        choices=["profiler", "tests", "default", "tiny"],
        default="profiler",
        help="Which rule set to generate configs from (default: profiler).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract and print unique tile 9-tuples from JSON files in Python format.",
    )
    parser.add_argument(
        "--arch",
        default="gfx950",
        help="Target architecture for rule generation (default: gfx950).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["forward", "bwd_data", "bwd_weight"],
        default=["forward", "bwd_data", "bwd_weight"],
        help="Which variants to analyze.",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=20,
        metavar="N",
        help="Max missing instances to print (0 = all, default: 20).",
    )
    args = parser.parse_args()

    print(f"Loading JSON configs from: {CONFIGS_DIR}")
    print(f"Rule set: {args.rule_set})")

    json_config_set = "profiler" if args.rule_set == "default" else args.rule_set
    if json_config_set == "tiny":
        json_config_set = "tests"

    json_instances = load_json_instances(json_config_set)
    print(f"Loaded {len(json_instances)} JSON instances total.")

    if args.extract:
        print("\n=== EXTRACT MODE: Unique tile data from JSON configs ===\n")
        extract_tile_data(json_instances)
        return

    # Filter to requested variants
    selected_variants = [_VARIANT_ENUM[v] for v in args.variants]
    variant_strs = set(args.variants)
    json_instances = [i for i in json_instances if i.get("variant") in variant_strs]
    print(f"Filtered to {len(json_instances)} instances for variants: {args.variants}")

    # Generate from rules (both 2D and 3D to cover all JSON files)
    print(f"\nGenerating configs from rules (arch={args.arch}, datatypes=[fp16, bf16, fp32], ndims=[2,3])...")
    all_generated = get_default_configs(
        arch=args.arch,
        variants=selected_variants,
        ndims=[2, 3],
        datatypes=["fp16", "bf16", "fp32"],
        rule_set=args.rule_set,
    )
    # Filter out depthwise configs (validated separately via test_depthwise_tile_math.py)
    generated = [c for c in all_generated if isinstance(c, GroupedConvKernelConfig)]
    n_dw = len(all_generated) - len(generated)
    print(f"Generated {len(generated)} GEMM configs from rules"
          + (f" (+ {n_dw} depthwise, validated separately)." if n_dw else "."))

    # Analyze coverage
    print("\nAnalyzing coverage...")
    result = analyze_coverage(json_instances, generated)

    n_json = len(result["json_keys"])
    n_covered = len(result["covered"])
    n_missing = len(result["missing"])
    n_extra = len(result["extra"])

    coverage_pct = 100.0 * n_covered / n_json if n_json > 0 else 0.0

    print("\n" + "=" * 70)
    print("COVERAGE REPORT")
    print("=" * 70)
    print(f"JSON instances (unique keys):  {n_json}")
    print(f"Generated configs (unique):    {len(result['gen_keys'])}")
    print(f"Covered by rules:              {n_covered} ({coverage_pct:.1f}%)")
    print(f"Missing from rules:            {n_missing}")
    print(f"Extra in rules (not in JSON):  {n_extra}")

    if result["missing"]:
        limit = args.show_missing if args.show_missing > 0 else len(result["missing"])
        missing_sorted = sorted(result["missing"], key=lambda k: _key_to_dict(k).get("tile_m", 0))
        print(f"\n--- Missing instances (showing {min(limit, n_missing)} of {n_missing}) ---")
        for key in missing_sorted[:limit]:
            print(_format_missing(key))
        if n_missing > limit:
            print(f"  ... and {n_missing - limit} more.")

    # Summary by variant
    print("\n--- Coverage by variant ---")
    for var_str in args.variants:
        j_keys = {k for k in result["json_keys"] if _key_to_dict(k).get("variant") == var_str}
        c_keys = {k for k in result["covered"] if _key_to_dict(k).get("variant") == var_str}
        m_keys = {k for k in result["missing"] if _key_to_dict(k).get("variant") == var_str}
        pct = 100.0 * len(c_keys) / len(j_keys) if j_keys else 0.0
        print(f"  {var_str:15s}: {len(c_keys):4d}/{len(j_keys):4d} covered ({pct:5.1f}%), {len(m_keys):4d} missing")

    print("=" * 70)

    if n_missing == 0:
        print("\n✓ Rules fully cover all JSON instances!")
    else:
        print(f"\n✗ {n_missing} JSON instances are not covered by rules.")
        sys.exit(1)


if __name__ == "__main__":
    main()
