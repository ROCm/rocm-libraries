#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Parse MIOpenDriver configs from ALL_CONFIGS_FULL.txt and convert to
GroupedConv test problems for dispatcher training data collection.

Filters configs based on dispatcher constraints:
- 2D spatial only (ndim_spatial=2)
- NHWGC layout (dispatcher's native layout)
- Reasonable problem sizes for training diversity
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from grouped_conv_utils import GroupedConvProblem  # noqa: E402


# =============================================================================
# MIOpenDriver parser
# =============================================================================


def parse_miopen_line(line: str) -> Dict:
    """Parse a single MIOpenDriver command line into problem dict."""
    # Extract all --key value pairs
    params = {}

    # Parse with regex
    pattern = r'--(\w+)\s+(\S+)'
    for match in re.finditer(pattern, line):
        key, value = match.groups()
        # Try to convert to int, otherwise keep as string
        try:
            params[key] = int(value)
        except ValueError:
            params[key] = value

    return params


def miopen_to_grouped_conv_problem(params: Dict) -> GroupedConvProblem:
    """Convert MIOpenDriver params to GroupedConvProblem.

    Note: MIOpenDriver uses NCHW layout, but dispatcher uses NHWGC.
    The mathematical problem is the same, just memory layout differs.
    Note: dtype is not part of GroupedConvProblem - it's handled at benchmark level.
    """
    return GroupedConvProblem(
        N=params.get('batchsize', 1),
        C=params.get('in_channels', 1),
        K=params.get('out_channels', 1),
        G=params.get('group_count', 1),
        Hi=params.get('in_h', 1),
        Wi=params.get('in_w', 1),
        Y=params.get('fil_h', 1),
        X=params.get('fil_w', 1),
        stride_h=params.get('conv_stride_h', 1),
        stride_w=params.get('conv_stride_w', 1),
        dilation_h=params.get('dilation_h', 1),
        dilation_w=params.get('dilation_w', 1),
        pad_h=params.get('pad_h', 0),
        pad_w=params.get('pad_w', 0),
    )


# =============================================================================
# Filtering logic
# =============================================================================


def is_dispatcher_compatible(params: Dict) -> bool:
    """Check if config is compatible with dispatcher constraints."""
    # Must be 2D spatial
    if params.get('spatial_dim') != 2:
        return False

    # Skip 3D convolutions
    if params.get('in_d', 1) != 1 or params.get('fil_d', 1) != 1:
        return False

    # Must be forward convolution mode
    if params.get('mode') != 'conv':
        return False

    # Skip extremely tiny problems (H,W < 4) - not representative
    if params.get('in_h', 0) < 4 or params.get('in_w', 0) < 4:
        return False

    # Skip extremely large problems (H,W > 512) - too slow for training sweeps
    if params.get('in_h', 0) > 512 or params.get('in_w', 0) > 512:
        return False

    # Skip large batch sizes (>256) - diminishing returns for diversity
    if params.get('batchsize', 0) > 256:
        return False

    # Skip very large channel counts (>2048) - edge cases
    if params.get('in_channels', 0) > 2048 or params.get('out_channels', 0) > 2048:
        return False

    return True


def deduplicate_problems(problems: List[GroupedConvProblem]) -> List[GroupedConvProblem]:
    """Remove duplicate problems (same shape, different dtype)."""
    seen: Set[str] = set()
    unique: List[GroupedConvProblem] = []

    for prob in problems:
        # Create shape signature (ignore dtype for dedup)
        sig = f"{prob.N}_{prob.C}_{prob.K}_{prob.G}_{prob.Hi}_{prob.Wi}_{prob.Y}_{prob.X}"
        sig += f"_{prob.stride_h}_{prob.stride_w}_{prob.dilation_h}_{prob.dilation_w}"
        sig += f"_{prob.pad_h}_{prob.pad_w}"

        if sig not in seen:
            seen.add(sig)
            unique.append(prob)

    return unique


# =============================================================================
# Sampling strategies for diversity
# =============================================================================


def sample_for_diversity(problems: List[GroupedConvProblem], target_count: int = 200) -> List[GroupedConvProblem]:
    """Sample problems for training diversity.

    Strategy:
    - Stratified sampling across problem categories
    - Ensure coverage of degenerate, small, medium, large
    - Prefer diverse (N, C, K, G, H, W, Y, X) combinations
    """
    if len(problems) <= target_count:
        return problems

    # Categorize problems
    categories = {
        'depthwise': [],  # G == C (depthwise separable)
        'pointwise': [],  # Y == X == 1 (1x1 convolutions)
        'small_spatial': [],  # H,W < 32
        'medium_spatial': [],  # 32 <= H,W < 128
        'large_spatial': [],  # H,W >= 128
        'small_channels': [],  # C,K < 128
        'medium_channels': [],  # 128 <= C,K < 512
        'large_channels': [],  # C,K >= 512
        'stride_2': [],  # stride > 1
        'dilated': [],  # dilation > 1
        'large_filter': [],  # Y,X >= 5
    }

    for prob in problems:
        # Multiple categories per problem
        if prob.G == prob.C:
            categories['depthwise'].append(prob)
        if prob.Y == 1 and prob.X == 1:
            categories['pointwise'].append(prob)
        if prob.Hi < 32 and prob.Wi < 32:
            categories['small_spatial'].append(prob)
        elif prob.Hi < 128 and prob.Wi < 128:
            categories['medium_spatial'].append(prob)
        else:
            categories['large_spatial'].append(prob)
        if prob.C < 128 and prob.K < 128:
            categories['small_channels'].append(prob)
        elif prob.C < 512 and prob.K < 512:
            categories['medium_channels'].append(prob)
        else:
            categories['large_channels'].append(prob)
        if prob.stride_h > 1 or prob.stride_w > 1:
            categories['stride_2'].append(prob)
        if prob.dilation_h > 1 or prob.dilation_w > 1:
            categories['dilated'].append(prob)
        if prob.Y >= 5 or prob.X >= 5:
            categories['large_filter'].append(prob)

    # Allocate samples per category (proportional)
    sampled: Set[int] = set()  # Track by id(prob)
    result: List[GroupedConvProblem] = []

    # Priority categories (ensure representation)
    priority = [
        ('depthwise', 20),
        ('pointwise', 20),
        ('stride_2', 15),
        ('dilated', 10),
        ('large_filter', 10),
    ]

    for cat_name, count in priority:
        cat_probs = categories[cat_name]
        # Sample up to count, but skip already sampled
        for prob in cat_probs:
            if id(prob) not in sampled and len(result) < target_count:
                sampled.add(id(prob))
                result.append(prob)
                if len([p for p in result if id(p) in {id(x) for x in cat_probs}]) >= count:
                    break

    # Fill remaining with balanced sampling from spatial/channel categories
    remaining_budget = target_count - len(result)
    spatial_cats = ['small_spatial', 'medium_spatial', 'large_spatial']
    channel_cats = ['small_channels', 'medium_channels', 'large_channels']

    per_spatial = remaining_budget // len(spatial_cats)
    for cat_name in spatial_cats:
        cat_probs = categories[cat_name]
        added = 0
        for prob in cat_probs:
            if id(prob) not in sampled and added < per_spatial:
                sampled.add(id(prob))
                result.append(prob)
                added += 1

    # Fill any remaining slots with uncategorized problems
    for prob in problems:
        if len(result) >= target_count:
            break
        if id(prob) not in sampled:
            sampled.add(id(prob))
            result.append(prob)

    return result[:target_count]


# =============================================================================
# Generate backward problems from forward
# =============================================================================


def derive_bwd_data_problem(fwd: GroupedConvProblem) -> GroupedConvProblem:
    """Derive backward data problem from forward.

    Backward data computes input gradients: dL/dInput = conv(dL/dOutput, Weight).
    Mathematical swap: C <-> K (input/output channels swap roles).
    """
    # Compute output spatial dims from forward
    Ho = (fwd.Hi + 2 * fwd.pad_h - fwd.dilation_h * (fwd.Y - 1) - 1) // fwd.stride_h + 1
    Wo = (fwd.Wi + 2 * fwd.pad_w - fwd.dilation_w * (fwd.X - 1) - 1) // fwd.stride_w + 1

    return GroupedConvProblem(
        N=fwd.N,
        C=fwd.K,  # Swap: output channels become input channels
        K=fwd.C,  # Swap: input channels become output channels
        G=fwd.G,
        Hi=Ho,  # Output spatial dims become input
        Wi=Wo,
        Y=fwd.Y,
        X=fwd.X,
        stride_h=fwd.stride_h,
        stride_w=fwd.stride_w,
        dilation_h=fwd.dilation_h,
        dilation_w=fwd.dilation_w,
        pad_h=fwd.pad_h,
        pad_w=fwd.pad_w,
    )


def derive_bwd_weight_problem(fwd: GroupedConvProblem) -> GroupedConvProblem:
    """Derive backward weight problem from forward.

    Backward weight computes weight gradients: dL/dWeight = conv(Input, dL/dOutput).
    Same dimensions as forward, but different memory access patterns.
    """
    # Weight gradient problem has same shape as forward
    return GroupedConvProblem(
        N=fwd.N,
        C=fwd.C,
        K=fwd.K,
        G=fwd.G,
        Hi=fwd.Hi,
        Wi=fwd.Wi,
        Y=fwd.Y,
        X=fwd.X,
        stride_h=fwd.stride_h,
        stride_w=fwd.stride_w,
        dilation_h=fwd.dilation_h,
        dilation_w=fwd.dilation_w,
        pad_h=fwd.pad_h,
        pad_w=fwd.pad_w,
    )


# =============================================================================
# Main parsing
# =============================================================================


def parse_all_configs(
    config_file: str,
    max_forward: int = 200,
    max_bwd_data: int = 150,
    max_bwd_weight: int = 150,
) -> Tuple[List[GroupedConvProblem], List[GroupedConvProblem], List[GroupedConvProblem]]:
    """Parse ALL_CONFIGS_FULL.txt and generate all 3 variant problem sets.

    Note: dtype is not part of problem spec - benchmarks will expand each
    problem to multiple dtypes (fp16, bf16, fp32).
    """

    print(f"Parsing {config_file}...")

    # Parse all forward problems
    forward_problems: List[GroupedConvProblem] = []

    errors = {}
    with open(config_file) as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"  Processed {i} lines, found {len(forward_problems)} valid problems...")

            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                params = parse_miopen_line(line)
                if is_dispatcher_compatible(params):
                    prob = miopen_to_grouped_conv_problem(params)
                    forward_problems.append(prob)
            except Exception as e:
                # Track errors for debugging
                err_type = type(e).__name__
                errors[err_type] = errors.get(err_type, 0) + 1
                if i < 100:  # Show first few errors
                    print(f"  Error at line {i}: {e}")
                continue

    if errors:
        print(f"  Errors encountered: {errors}")

    print(f"  Total forward problems parsed: {len(forward_problems)}")

    # Deduplicate
    forward_problems = deduplicate_problems(forward_problems)
    print(f"  After deduplication: {len(forward_problems)}")

    # Sample for diversity
    forward_problems = sample_for_diversity(forward_problems, max_forward)
    print(f"  After diversity sampling: {len(forward_problems)} (target: {max_forward})")

    # Derive backward problems
    print("\nDeriving backward problems from forward...")
    bwd_data_problems = [derive_bwd_data_problem(p) for p in forward_problems]
    bwd_weight_problems = [derive_bwd_weight_problem(p) for p in forward_problems]

    # Sample backward problems separately (may want different diversity)
    bwd_data_problems = sample_for_diversity(bwd_data_problems, max_bwd_data)
    bwd_weight_problems = sample_for_diversity(bwd_weight_problems, max_bwd_weight)

    print(f"  Backward data: {len(bwd_data_problems)} (target: {max_bwd_data})")
    print(f"  Backward weight: {len(bwd_weight_problems)} (target: {max_bwd_weight})")

    return forward_problems, bwd_data_problems, bwd_weight_problems


# =============================================================================
# Export to Python problem sets
# =============================================================================


def export_to_python(problems: List[GroupedConvProblem], output_file: str, variant: str):
    """Export problems to a Python file.

    Note: dtype is not included in GroupedConvProblem - it will be handled
    by the benchmark script when expanding problems for multiple dtypes.
    """
    with open(output_file, 'w') as f:
        f.write("#!/usr/bin/env python3\n\n")
        f.write("# Auto-generated from ALL_CONFIGS_FULL.txt\n")
        f.write(f"# Training problem set for {variant}\n")
        f.write("# Note: dtype will be expanded by benchmark (fp16, bf16, fp32)\n\n")
        f.write("from grouped_conv_utils import GroupedConvProblem\n\n")
        f.write(f"TRAINING_PROBLEMS_{variant.upper()} = [\n")

        for i, prob in enumerate(problems):
            f.write(f"    # Problem {i+1}\n")
            f.write(f"    GroupedConvProblem(\n")
            f.write(f"        N={prob.N}, C={prob.C}, K={prob.K}, G={prob.G},\n")
            f.write(f"        Hi={prob.Hi}, Wi={prob.Wi}, Y={prob.Y}, X={prob.X},\n")
            f.write(f"        stride_h={prob.stride_h}, stride_w={prob.stride_w},\n")
            f.write(f"        dilation_h={prob.dilation_h}, dilation_w={prob.dilation_w},\n")
            f.write(f"        pad_h={prob.pad_h}, pad_w={prob.pad_w},\n")
            f.write(f"    ),\n")

        f.write("]\n")

    print(f"Exported {len(problems)} problems to {output_file}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Parse MIOpenDriver configs and generate dispatcher training sets"
    )
    parser.add_argument(
        '--input',
        default='ALL_CONFIGS_FULL.txt',
        help='Input MIOpenDriver config file'
    )
    parser.add_argument(
        '--max-forward',
        type=int,
        default=200,
        help='Target number of forward problems'
    )
    parser.add_argument(
        '--max-bwd-data',
        type=int,
        default=150,
        help='Target number of backward data problems'
    )
    parser.add_argument(
        '--max-bwd-weight',
        type=int,
        default=150,
        help='Target number of backward weight problems'
    )
    parser.add_argument(
        '--output-dir',
        default='problems',
        help='Output directory for problem sets'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List sampled problems instead of exporting'
    )
    args = parser.parse_args()

    # Parse and sample
    fwd, bwd_data, bwd_weight = parse_all_configs(
        args.input,
        args.max_forward,
        args.max_bwd_data,
        args.max_bwd_weight,
    )

    if args.list:
        print("\n=== FORWARD PROBLEMS ===")
        for i, p in enumerate(fwd[:20]):  # Show first 20
            print(f"{i}: N={p.N} C={p.C} K={p.K} G={p.G} H={p.Hi} W={p.Wi} Y={p.Y} X={p.X}")
        if len(fwd) > 20:
            print(f"... and {len(fwd) - 20} more")

        print("\n=== BACKWARD DATA PROBLEMS ===")
        for i, p in enumerate(bwd_data[:20]):
            print(f"{i}: N={p.N} C={p.C} K={p.K} G={p.G} H={p.Hi} W={p.Wi} Y={p.Y} X={p.X}")
        if len(bwd_data) > 20:
            print(f"... and {len(bwd_data) - 20} more")

        print("\n=== BACKWARD WEIGHT PROBLEMS ===")
        for i, p in enumerate(bwd_weight[:20]):
            print(f"{i}: N={p.N} C={p.C} K={p.K} G={p.G} H={p.Hi} W={p.Wi} Y={p.Y} X={p.X}")
        if len(bwd_weight) > 20:
            print(f"... and {len(bwd_weight) - 20} more")
    else:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Export
        export_to_python(fwd, output_dir / "forward_training.py", "forward")
        export_to_python(bwd_data, output_dir / "bwd_data_training.py", "bwd_data")
        export_to_python(bwd_weight, output_dir / "bwd_weight_training.py", "bwd_weight")

        print(f"\nTotal problems generated: {len(fwd) + len(bwd_data) + len(bwd_weight)}")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
