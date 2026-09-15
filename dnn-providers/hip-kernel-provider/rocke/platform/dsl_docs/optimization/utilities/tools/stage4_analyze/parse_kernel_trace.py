# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Parse ATT kernel traces and extract structured stall breakdown.

Parses rocprof-compute ATT trace output (code.json, snapshots.json) and provides:
- Stall classification (VMEM, LDS, barriers, MFMA, etc.)
- Top-K stall hotspots mapped to source lines
- Actionable recommendations based on stall types

Usage:
    python parse_kernel_trace.py <dispatch_dir> [--output report.json] [--topk 10]
    python parse_kernel_trace.py ui_output_agent_0_dispatch_1234 --topk 15 --detail
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class Instruction:
    """Single instruction with stall metrics."""

    asm: str
    pc_index: int
    source_loc: str
    pc_addr: int
    exec_count: int
    total_cycles: int
    stall_cycles: int
    issue_cycles: int

    @property
    def stall_pct(self) -> float:
        return (
            100.0 * self.stall_cycles / self.total_cycles if self.total_cycles else 0.0
        )

    @property
    def stall_type(self) -> str:
        """Classify instruction by stall type for bottleneck analysis."""
        asm = self.asm.lower()

        # Memory wait instructions (dominant stall source)
        if "s_waitcnt" in asm:
            if "vmcnt" in asm:
                return "vmem_wait"
            if "lgkmcnt" in asm:
                return "lds_wait"
            if "expcnt" in asm:
                return "exp_wait"
            return "waitcnt"

        # Synchronization barriers
        if "s_barrier" in asm or "s_wait_idle" in asm:
            return "barrier_stall"

        # Global memory ops
        if "buffer_load" in asm or "global_load" in asm or "flat_load" in asm:
            return "vmem_load"
        if "buffer_store" in asm or "global_store" in asm:
            return "vmem_store"

        # LDS ops
        if "ds_read" in asm or "ds_write" in asm:
            return "lds_stall"

        # Scalar memory
        if "s_load" in asm or "s_store" in asm:
            return "smem_stall"

        # Compute (MFMA stalls indicate operand readiness issues)
        if "v_mfma" in asm or "v_fma" in asm:
            return "mfma_stall"

        return "other"


@dataclass
class SourceLineHotspot:
    """Aggregated stall hotspot for a source line."""

    source_loc: str
    total_stall_cycles: int = 0
    total_cycles: int = 0
    instruction_count: int = 0
    dominant_type: str = "other"

    @property
    def stall_pct(self) -> float:
        return (
            100.0 * self.total_stall_cycles / self.total_cycles
            if self.total_cycles
            else 0.0
        )


@dataclass
class StallBreakdown:
    """Overall kernel stall breakdown."""

    total_cycles: int
    total_stall_cycles: int
    vmem_wait_pct: float = 0.0
    vmem_load_pct: float = 0.0
    lds_wait_pct: float = 0.0
    lds_stall_pct: float = 0.0
    barrier_stall_pct: float = 0.0
    mfma_stall_pct: float = 0.0
    other_pct: float = 0.0

    @property
    def overall_stall_pct(self) -> float:
        return (
            100.0 * self.total_stall_cycles / self.total_cycles
            if self.total_cycles
            else 0.0
        )


@dataclass
class KernelTraceReport:
    """Complete kernel trace analysis report."""

    dispatch_dir: str
    kernel_name: str
    instruction_count: int
    stall_breakdown: StallBreakdown
    hotspots: List[Dict]
    recommendations: List[str]


def load_instructions(dispatch_dir: Path) -> List[Instruction]:
    """Load instructions from code.json ATT trace output.

    Args:
        dispatch_dir: Path to ATT dispatch output directory

    Returns:
        List of Instruction objects with stall metrics
    """
    code_json = dispatch_dir / "code.json"
    if not code_json.exists():
        raise FileNotFoundError(f"code.json not found in {dispatch_dir}")

    with open(code_json) as f:
        data = json.load(f)

    instructions = []
    for row in data.get("code", []):
        # code.json format: [asm, label, pc_index, source_loc, ..., pc_addr, exec_count, total_cycles, stall_cycles, issue_cycles]
        if not isinstance(row[2], int) or row[2] == 0:
            continue

        instructions.append(
            Instruction(
                asm=row[0],
                pc_index=row[2],
                source_loc=row[3] if row[3] else "<unknown>",
                pc_addr=row[5] if len(row) > 5 else 0,
                exec_count=row[6] if len(row) > 6 and isinstance(row[6], int) else 0,
                total_cycles=row[7] if len(row) > 7 and isinstance(row[7], int) else 0,
                stall_cycles=row[8] if len(row) > 8 and isinstance(row[8], int) else 0,
                issue_cycles=row[9] if len(row) > 9 and isinstance(row[9], int) else 0,
            )
        )

    return instructions


def compute_stall_breakdown(instructions: List[Instruction]) -> StallBreakdown:
    """Compute stall breakdown by category.

    Args:
        instructions: List of parsed instructions

    Returns:
        StallBreakdown with percentage breakdowns
    """
    total_cycles = sum(i.total_cycles for i in instructions)
    total_stall = sum(i.stall_cycles for i in instructions)

    stalls_by_type = defaultdict(int)
    for inst in instructions:
        if inst.stall_cycles > 0:
            stalls_by_type[inst.stall_type] += inst.stall_cycles

    def pct(cycles):
        return 100.0 * cycles / total_stall if total_stall else 0.0

    return StallBreakdown(
        total_cycles=total_cycles,
        total_stall_cycles=total_stall,
        vmem_wait_pct=pct(stalls_by_type.get("vmem_wait", 0)),
        vmem_load_pct=pct(stalls_by_type.get("vmem_load", 0)),
        lds_wait_pct=pct(stalls_by_type.get("lds_wait", 0)),
        lds_stall_pct=pct(stalls_by_type.get("lds_stall", 0)),
        barrier_stall_pct=pct(stalls_by_type.get("barrier_stall", 0)),
        mfma_stall_pct=pct(stalls_by_type.get("mfma_stall", 0)),
        other_pct=pct(
            sum(
                v
                for k, v in stalls_by_type.items()
                if k
                not in {
                    "vmem_wait",
                    "vmem_load",
                    "lds_wait",
                    "lds_stall",
                    "barrier_stall",
                    "mfma_stall",
                }
            )
        ),
    )


def aggregate_by_source(instructions: List[Instruction], topk: int = 15) -> List[Dict]:
    """Aggregate stalls by source line and return top-K hotspots.

    Args:
        instructions: List of parsed instructions
        topk: Number of top hotspots to return

    Returns:
        List of hotspot dictionaries sorted by stall_cycles
    """
    by_source = {}

    for inst in instructions:
        loc = inst.source_loc
        if loc not in by_source:
            by_source[loc] = {
                "source_loc": loc,
                "total_stall_cycles": 0,
                "total_cycles": 0,
                "instruction_count": 0,
                "stall_types": defaultdict(int),
            }

        hs = by_source[loc]
        hs["total_stall_cycles"] += inst.stall_cycles
        hs["total_cycles"] += inst.total_cycles
        hs["instruction_count"] += 1
        if inst.stall_cycles > 0:
            hs["stall_types"][inst.stall_type] += inst.stall_cycles

    # Add dominant stall type and stall_pct
    for hs in by_source.values():
        if hs["stall_types"]:
            hs["dominant_type"] = max(hs["stall_types"], key=hs["stall_types"].get)
        else:
            hs["dominant_type"] = "other"

        hs["stall_pct"] = (
            100.0 * hs["total_stall_cycles"] / hs["total_cycles"]
            if hs["total_cycles"]
            else 0.0
        )
        # Convert defaultdict to regular dict for JSON serialization
        hs["stall_types"] = dict(hs["stall_types"])

    # Sort by total_stall_cycles and return top-K
    hotspots = sorted(
        by_source.values(), key=lambda x: x["total_stall_cycles"], reverse=True
    )
    return hotspots[:topk]


def generate_recommendations(breakdown: StallBreakdown) -> List[str]:
    """Generate actionable recommendations based on stall breakdown.

    Maps stall types to optimization strategies from Runbook Table 3.1a.

    Args:
        breakdown: Stall breakdown with percentages

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # VMEM wait stalls (global memory latency)
    if breakdown.vmem_wait_pct > 20:
        recommendations.append(
            f"⚠️ VMEM-wait stalls are {breakdown.vmem_wait_pct:.1f}% - "
            "Consider: (1) Increase prefetch distance/stages, "
            "(2) Use async_buffer_load on gfx950, "
            "(3) Software pipelining (2-stage or 3-stage)"
        )

    # LDS wait stalls (LDS latency + bank conflicts)
    if breakdown.lds_wait_pct > 15:
        recommendations.append(
            f"⚠️ LDS-wait stalls are {breakdown.lds_wait_pct:.1f}% - "
            "Check: (1) LDS bank conflicts (run analyze_lds_conflicts.py), "
            "(2) Apply XOR swizzle or padding, "
            "(3) Coalesce LDS accesses"
        )

    # Barrier stalls (synchronization overhead)
    if breakdown.barrier_stall_pct > 10:
        recommendations.append(
            f"⚠️ Barrier stalls are {breakdown.barrier_stall_pct:.1f}% - "
            "Review: (1) Are barriers necessary (can you use async copy)?, "
            "(2) Unroll loops to reduce barrier frequency, "
            "(3) Balance work across waves"
        )

    # MFMA stalls (operand readiness)
    if breakdown.mfma_stall_pct > 5:
        recommendations.append(
            f"⚠️ MFMA stalls are {breakdown.mfma_stall_pct:.1f}% - "
            "Issue: MFMA operands not ready in time. "
            "Fix: (1) Prefetch operands earlier, "
            "(2) Improve instruction scheduling, "
            "(3) Reduce VGPR pressure"
        )

    # Overall stall rate
    if breakdown.overall_stall_pct < 30:
        recommendations.append(
            f"✅ Overall stall rate is low ({breakdown.overall_stall_pct:.1f}%) - kernel is compute-bound"
        )
    elif breakdown.overall_stall_pct > 50:
        recommendations.append(
            f"🔴 Overall stall rate is HIGH ({breakdown.overall_stall_pct:.1f}%) - memory/sync bottlenecked"
        )

    if not recommendations:
        recommendations.append(
            "✅ No major bottlenecks detected - stall profile looks healthy"
        )

    return recommendations


def print_report(report: KernelTraceReport, detail: bool = False):
    """Print formatted trace report to console.

    Args:
        report: Parsed kernel trace report
        detail: Whether to show detailed hotspot breakdown
    """
    print(f"\n{'=' * 90}")
    print(f"  Kernel Trace Analysis: {report.kernel_name}")
    print(f"{'=' * 90}")
    print(f"  Source: {report.dispatch_dir}")
    print(f"  Instructions: {report.instruction_count:,}")
    print(f"  Total cycles: {report.stall_breakdown.total_cycles:,}")
    print(
        f"  Stall cycles: {report.stall_breakdown.total_stall_cycles:,} "
        f"({report.stall_breakdown.overall_stall_pct:.1f}%)"
    )
    print()

    # Stall breakdown
    print("Stall Breakdown by Type:")
    print(f"  {'Type':<18} {'Percentage':>10}")
    print(f"  {'-' * 18} {'-' * 10}")
    bd = report.stall_breakdown
    print(f"  {'VMEM-wait':<18} {bd.vmem_wait_pct:>9.1f}%")
    print(f"  {'VMEM-load':<18} {bd.vmem_load_pct:>9.1f}%")
    print(f"  {'LDS-wait':<18} {bd.lds_wait_pct:>9.1f}%")
    print(f"  {'LDS-stall':<18} {bd.lds_stall_pct:>9.1f}%")
    print(f"  {'Barrier':<18} {bd.barrier_stall_pct:>9.1f}%")
    print(f"  {'MFMA':<18} {bd.mfma_stall_pct:>9.1f}%")
    print(f"  {'Other':<18} {bd.other_pct:>9.1f}%")
    print()

    # Hotspots
    print(f"Top-{len(report.hotspots)} Stall Hotspots:")
    print(f"  {'#':>3}  {'Stall Cycles':>12}  {'% of Total':>10}  {'Type':<14}  Source")
    print(f"  {'-' * 3}  {'-' * 12}  {'-' * 10}  {'-' * 14}  {'-' * 50}")

    total_stall = report.stall_breakdown.total_stall_cycles
    for i, hs in enumerate(report.hotspots, 1):
        pct = 100.0 * hs["total_stall_cycles"] / total_stall if total_stall else 0.0
        src_short = (
            hs["source_loc"][-50:] if len(hs["source_loc"]) > 50 else hs["source_loc"]
        )
        print(
            f"  {i:>3}  {hs['total_stall_cycles']:>12,}  {pct:>9.1f}%  {hs['dominant_type']:<14}  {src_short}"
        )

        if detail and i <= 5:
            # Show stall type breakdown for top 5
            print(f"       Stall types: {dict(hs['stall_types'])}")

    print()

    # Recommendations
    print("Recommendations:")
    for rec in report.recommendations:
        print(f"  {rec}")
    print()


def main():
    """Command-line interface for kernel trace parsing."""
    parser = argparse.ArgumentParser(
        description="Parse ATT kernel traces and extract stall breakdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse trace and show summary
  %(prog)s ui_output_agent_0_dispatch_1234

  # Save JSON report
  %(prog)s ui_output_agent_0_dispatch_1234 --output report.json

  # Show detailed hotspot breakdown
  %(prog)s ui_output_agent_0_dispatch_1234 --detail --topk 20
        """,
    )

    parser.add_argument(
        "dispatch_dir",
        type=Path,
        help="Path to ATT dispatch output directory (contains code.json)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file (default: print to console)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=15,
        help="Number of top hotspots to report (default: 15)",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed stall type breakdown for top hotspots",
    )

    args = parser.parse_args()

    if not args.dispatch_dir.is_dir():
        print(f"Error: Directory not found: {args.dispatch_dir}", file=sys.stderr)
        return 1

    try:
        # Load and parse trace
        instructions = load_instructions(args.dispatch_dir)
        if not instructions:
            print(
                f"Warning: No instructions found in {args.dispatch_dir}/code.json",
                file=sys.stderr,
            )
            return 1

        # Compute metrics
        breakdown = compute_stall_breakdown(instructions)
        hotspots = aggregate_by_source(instructions, topk=args.topk)
        recommendations = generate_recommendations(breakdown)

        # Build report
        report = KernelTraceReport(
            dispatch_dir=str(args.dispatch_dir),
            kernel_name=args.dispatch_dir.name,
            instruction_count=len(instructions),
            stall_breakdown=breakdown,
            hotspots=hotspots,
            recommendations=recommendations,
        )

        # Output
        if args.output:
            with open(args.output, "w") as f:
                # Convert dataclass to dict for JSON serialization
                report_dict = asdict(report)
                json.dump(report_dict, f, indent=2)
            print(f"Report saved to {args.output}")
        else:
            print_report(report, detail=args.detail)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
