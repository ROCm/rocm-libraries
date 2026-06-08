#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validation script: compare rule-generated configs against the CK Builder
reference instance set (generated in memory from the ``.conf`` configs).

Usage:
    python validate_rules_coverage.py [--rule-set ...] [--extract] [--arch ARCH]

Modes:
    Default: Generate the builder reference set + the chosen rule set, report
             coverage gaps.
    --extract: Dump all unique tile/warp/vec 9-tuples from the reference set in
               Python format.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR))

from unified_grouped_conv_codegen import (
    DepthwiseConvKernelConfig,
    GroupedConvKernelConfig,
    GroupedConvVariant,
    StreamKConfig,
    StreamKReductionStrategy,
    get_default_configs,
)
from grouped_conv.grouped_config_rules_builder import get_configs as get_builder_configs

# ---------------------------------------------------------------------------
# Reference (CK Builder) loading
# ---------------------------------------------------------------------------

_VARIANT_ENUM = {
    "forward":    GroupedConvVariant.FORWARD,
    "bwd_data":   GroupedConvVariant.BACKWARD_DATA,
    "bwd_weight": GroupedConvVariant.BACKWARD_WEIGHT,
}


def load_reference_configs(
    subset: str,
    arch: str,
    variants: List,
) -> List[GroupedConvKernelConfig]:
    """Generate the CK Builder reference GEMM configs in memory.

    ``subset`` is the builder ``.conf`` subset ("profiler" or "tests"). Depthwise
    configs are filtered out (validated separately via test_depthwise_tile_math.py).
    """
    cfgs = get_builder_configs(
        arch=arch,
        variants=variants,
        ndims=[2, 3],
        datatypes=["fp16", "bf16", "fp32"],
        subset=subset,
    )
    return [c for c in cfgs if isinstance(c, GroupedConvKernelConfig)]


# ---------------------------------------------------------------------------
# Canonical key generation
# ---------------------------------------------------------------------------

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
    reference_configs: List[GroupedConvKernelConfig],
    generated_configs: List[GroupedConvKernelConfig],
) -> Dict:
    """Compare the reference set against generated configs.

    Returns dict with:
      - 'json_keys': set of canonical keys from the reference set
      - 'gen_keys': set of canonical keys from generated configs
      - 'covered': reference keys that appear in generated
      - 'missing': reference keys absent from generated
      - 'extra': generated keys not in the reference set
    """
    json_keys: Set[FrozenSet] = set()
    for cfg in reference_configs:
        json_keys.add(_config_to_key(cfg))

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

def extract_tile_data(reference_configs: List[GroupedConvKernelConfig]) -> None:
    """Print unique tile 9-tuples from the reference configs in Python format."""
    by_variant: Dict[str, Set[Tuple]] = defaultdict(set)

    for cfg in reference_configs:
        inst = _key_to_dict(_config_to_key(cfg))
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
        choices=["profiler", "tests", "full", "full-tests", "default", "tiny"],
        default="full",
        help="Which rule set to generate configs from (default: full).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract and print unique tile 9-tuples from the reference set in Python format.",
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

    print(f"Rule set: {args.rule_set}")

    # The reference (ground truth) is the CK Builder instance set, generated in
    # memory. "tests"-class rule sets validate against the builder "tests" subset;
    # everything else validates against the builder "profiler" subset.
    reference_subset = "tests" if args.rule_set in ("tests", "full-tests", "tiny") else "profiler"
    print(f"Reference: CK Builder '{reference_subset}' subset (generated in memory)")

    selected_variants = [_VARIANT_ENUM[v] for v in args.variants]

    reference_configs = load_reference_configs(reference_subset, args.arch, selected_variants)
    print(f"Loaded {len(reference_configs)} reference instances for variants: {args.variants}")

    if args.extract:
        print("\n=== EXTRACT MODE: Unique tile data from reference set ===\n")
        extract_tile_data(reference_configs)
        return

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
    result = analyze_coverage(reference_configs, generated)

    n_json = len(result["json_keys"])
    n_covered = len(result["covered"])
    n_missing = len(result["missing"])
    n_extra = len(result["extra"])

    coverage_pct = 100.0 * n_covered / n_json if n_json > 0 else 0.0

    print("\n" + "=" * 70)
    print("COVERAGE REPORT")
    print("=" * 70)
    print(f"Reference instances (unique):  {n_json}")
    print(f"Generated configs (unique):    {len(result['gen_keys'])}")
    print(f"Covered by rules:              {n_covered} ({coverage_pct:.1f}%)")
    print(f"Missing from rules:            {n_missing}")
    print(f"Extra in rules (not in ref):   {n_extra}")

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
        print("\n✓ Rules fully cover all reference instances!")
    else:
        print(f"\n✗ {n_missing} reference instances are not covered by rules.")
        sys.exit(1)


if __name__ == "__main__":
    main()
