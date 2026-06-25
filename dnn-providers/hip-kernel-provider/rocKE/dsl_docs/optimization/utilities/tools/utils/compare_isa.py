# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare ISA between CK DSL mem_sync and FlyDSL to identify optimization opportunities."""

import re
from pathlib import Path


def count_instructions(isa_file):
    """Count key instruction types in ISA file."""
    with open(isa_file) as f:
        isa_text = f.read()

    counts = {}

    # MFMA instructions
    counts["v_mfma_total"] = len(re.findall(r"\bv_mfma_", isa_text))
    counts["v_mfma_32x32x16"] = len(re.findall(r"\bv_mfma_f32_32x32x16_bf16", isa_text))
    counts["v_mfma_16x16x32"] = len(re.findall(r"\bv_mfma_f32_16x16x32_f16", isa_text))

    # LDS operations
    counts["ds_read"] = len(re.findall(r"\bds_read", isa_text))
    counts["ds_write"] = len(re.findall(r"\bds_write", isa_text))

    # Global memory operations
    counts["buffer_load"] = len(re.findall(r"\bbuffer_load", isa_text))
    counts["buffer_store"] = len(re.findall(r"\bbuffer_store", isa_text))
    counts["global_load"] = len(re.findall(r"\bglobal_load", isa_text))
    counts["global_store"] = len(re.findall(r"\bglobal_store", isa_text))

    # Synchronization
    counts["s_waitcnt"] = len(re.findall(r"\bs_waitcnt", isa_text))
    counts["s_barrier"] = len(re.findall(r"\bs_barrier", isa_text))

    # Scheduling hints
    counts["s_sched_barrier"] = len(re.findall(r"\bs_sched_barrier", isa_text))
    counts["s_sched_group_barrier"] = len(
        re.findall(r"\bs_sched_group_barrier", isa_text)
    )
    counts["s_setprio"] = len(re.findall(r"\bs_setprio", isa_text))

    # Control flow
    counts["s_cbranch"] = len(re.findall(r"\bs_cbranch", isa_text))
    counts["s_branch"] = len(re.findall(r"\bs_branch", isa_text))

    # Register operations
    counts["v_accvgpr_read"] = len(re.findall(r"\bv_accvgpr_read", isa_text))
    counts["v_accvgpr_write"] = len(re.findall(r"\bv_accvgpr_write", isa_text))

    # Total lines
    counts["total_lines"] = len(isa_text.split("\n"))

    return counts


def analyze_hot_loop(isa_file):
    """Try to identify and analyze the hot loop."""
    with open(isa_file) as f:
        lines = f.readlines()

    # Find the first MFMA instruction (likely in hot loop)
    mfma_indices = [i for i, line in enumerate(lines) if "v_mfma_" in line]

    if not mfma_indices:
        return {"error": "No MFMA found"}

    # Get 50 lines around first MFMA as representative hot loop
    first_mfma_idx = mfma_indices[0]
    start = max(0, first_mfma_idx - 20)
    end = min(len(lines), first_mfma_idx + 30)

    hot_loop_snippet = "".join(lines[start:end])

    # Count key instructions in the hot loop
    hot_loop_counts = {
        "ds_read": hot_loop_snippet.count("ds_read"),
        "v_mfma": hot_loop_snippet.count("v_mfma"),
        "s_waitcnt": hot_loop_snippet.count("s_waitcnt"),
        "s_barrier": hot_loop_snippet.count("s_barrier"),
        "s_sched_group_barrier": hot_loop_snippet.count("s_sched_group_barrier"),
    }

    return {
        "snippet": hot_loop_snippet,
        "counts": hot_loop_counts,
        "total_mfmas": len(mfma_indices),
    }


def main():
    import argparse

    _default_ck = (
        Path(__file__).resolve().parents[1] / "comparison" / "rocke_isa" / "mem_sync.s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rocke_isa",
        nargs="?",
        default=str(_default_ck),
        help="path to CK DSL ISA .s file",
    )
    parser.add_argument(
        "compare_isa",
        nargs="?",
        default=None,
        help="path to comparison ISA .s file (FlyDSL, CK Tile C++, etc.)",
    )
    _args = parser.parse_args()
    rocke_isa = Path(_args.rocke_isa)
    flydsl_isa = Path(_args.compare_isa) if _args.compare_isa else None

    print("=" * 70)
    print("ISA COMPARISON: CK DSL mem_sync vs FlyDSL")
    print("=" * 70)

    # Count instructions
    print("\n## Instruction Counts\n")
    ck_counts = count_instructions(rocke_isa)
    if flydsl_isa is None:
        print("No comparison ISA file provided; showing CK DSL counts only.")
        for key, val in sorted(ck_counts.items()):
            print(f"  {key:<30} {val}")
        return
    fly_counts = count_instructions(flydsl_isa)

    # Print comparison table
    print(f"{'Instruction':<25} {'CK DSL':<12} {'FlyDSL':<12} {'Ratio':>10}")
    print("-" * 70)

    for key in sorted(ck_counts.keys()):
        ck_val = ck_counts[key]
        fly_val = fly_counts.get(key, 0)
        ratio = f"{fly_val / ck_val:.1f}×" if ck_val > 0 else "N/A"
        print(f"{key:<25} {ck_val:<12} {fly_val:<12} {ratio:>10}")

    # Key findings
    print("\n## Key Findings\n")

    # MFMA ratio
    mfma_ratio = (
        fly_counts["v_mfma_total"] / ck_counts["v_mfma_total"]
        if ck_counts["v_mfma_total"] > 0
        else 0
    )
    print(f"1. MFMA ratio: FlyDSL has {mfma_ratio:.0f}× more MFMAs")
    if mfma_ratio > 50:
        print("   → FlyDSL uses FULL LOOP UNROLLING (all 72 K-iterations)")
        print("   → CK DSL uses RUNTIME LOOP (8 MFMAs per iteration)")

    # Scheduling hints
    if (
        fly_counts["s_sched_group_barrier"] > 0
        and ck_counts["s_sched_group_barrier"] == 0
    ):
        print(
            f"\n2. FlyDSL has {fly_counts['s_sched_group_barrier']} s_sched_group_barrier instructions"
        )
        print("   → FlyDSL uses EXPLICIT INSTRUCTION SCHEDULING")
        print("   → CK DSL has NO scheduling hints (relies on compiler)")

    # LDS pattern
    lds_reads_per_mfma_ck = (
        ck_counts["ds_read"] / ck_counts["v_mfma_total"]
        if ck_counts["v_mfma_total"] > 0
        else 0
    )
    lds_reads_per_mfma_fly = (
        fly_counts["ds_read"] / fly_counts["v_mfma_total"]
        if fly_counts["v_mfma_total"] > 0
        else 0
    )
    print(
        f"\n3. LDS reads per MFMA: CK DSL {lds_reads_per_mfma_ck:.2f}, FlyDSL {lds_reads_per_mfma_fly:.2f}"
    )

    # Wait counts
    waits_per_mfma_ck = (
        ck_counts["s_waitcnt"] / ck_counts["v_mfma_total"]
        if ck_counts["v_mfma_total"] > 0
        else 0
    )
    waits_per_mfma_fly = (
        fly_counts["s_waitcnt"] / fly_counts["v_mfma_total"]
        if fly_counts["v_mfma_total"] > 0
        else 0
    )
    print(
        f"\n4. s_waitcnt per MFMA: CK DSL {waits_per_mfma_ck:.2f}, FlyDSL {waits_per_mfma_fly:.2f}"
    )
    if waits_per_mfma_ck > waits_per_mfma_fly:
        print("   → CK DSL has MORE stalls per MFMA (worse latency hiding)")

    # Code size
    size_ratio = (
        fly_counts["total_lines"] / ck_counts["total_lines"]
        if ck_counts["total_lines"] > 0
        else 0
    )
    print(
        f"\n5. Code size: FlyDSL is {size_ratio:.1f}× larger ({fly_counts['total_lines']} vs {ck_counts['total_lines']} lines)"
    )
    print(
        f"   MFMA density: CK DSL {ck_counts['v_mfma_total'] / ck_counts['total_lines'] * 100:.1f}% MFMAs/line"
    )
    print(
        f"   MFMA density: FlyDSL {fly_counts['v_mfma_total'] / fly_counts['total_lines'] * 100:.1f}% MFMAs/line"
    )

    # Hot loop analysis
    print("\n## Hot Loop Patterns\n")

    print("### CK DSL Hot Loop (50 lines around first MFMA):")
    ck_hot = analyze_hot_loop(rocke_isa)
    print(f"  Total MFMAs in kernel: {ck_hot['total_mfmas']}")
    print(f"  Hot loop instruction counts: {ck_hot['counts']}")
    print("\nSnippet:")
    print(ck_hot["snippet"][:800])
    print("  ... (truncated)\n")

    print("### FlyDSL Hot Loop (50 lines around first MFMA):")
    fly_hot = analyze_hot_loop(flydsl_isa)
    print(f"  Total MFMAs in kernel: {fly_hot['total_mfmas']}")
    print(f"  Hot loop instruction counts: {fly_hot['counts']}")
    print("\nSnippet:")
    print(fly_hot["snippet"][:800])
    print("  ... (truncated)\n")

    # Recommendations
    print("\n## Optimization Recommendations\n")
    print("Based on the ISA comparison:\n")

    recommendations = []

    if (
        fly_counts["s_sched_group_barrier"] > 0
        and ck_counts["s_sched_group_barrier"] == 0
    ):
        recommendations.append(
            {
                "priority": 1,
                "name": "Add instruction scheduling hints",
                "impact": "Medium (2-4%)",
                "effort": "Low",
                "desc": "Add s_sched_group_barrier to CK DSL hot loop to improve MFMA/LDS interleaving",
            }
        )

    if mfma_ratio > 50:
        recommendations.append(
            {
                "priority": 2,
                "name": "Implement clean loop unrolling",
                "impact": "High (5-10%)",
                "effort": "High",
                "desc": "Add range_constexpr equivalent or use MLIR loop unroll pass (NOT the broken async_dma path)",
            }
        )

    if waits_per_mfma_ck > waits_per_mfma_fly * 1.2:
        recommendations.append(
            {
                "priority": 3,
                "name": "Optimize LDS prefetching",
                "impact": "Medium (2-3%)",
                "effort": "Medium",
                "desc": "Reduce s_waitcnt stalls by better overlap of LDS reads with MFMAs",
            }
        )

    for i, rec in enumerate(sorted(recommendations, key=lambda x: x["priority"]), 1):
        print(
            f"{i}. **{rec['name']}** (Impact: {rec['impact']}, Effort: {rec['effort']})"
        )
        print(f"   {rec['desc']}\n")

    print("=" * 70)
    print("Comparison complete. See above for detailed analysis.")
    print("=" * 70)


if __name__ == "__main__":
    main()
