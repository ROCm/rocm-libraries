# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Analyze global memory prefetch effectiveness.

Evaluates if prefetch strategies (manual loop-carried values in FlyDSL,
pipeline stages in CK DSL) successfully hide memory latency.

Key metrics:
- VMEM load stall cycles vs compute cycles
- Global memory coalescing efficiency
- Buffer load vectorization width
- Async copy usage (gfx950)

References runbook section 6.1

Usage:
    python analyze_prefetch_efficiency.py --rocprof stats.csv --isa kernel.s --arch gfx950
    python analyze_prefetch_efficiency.py --rocprof stats.csv --isa kernel.s --output report.json
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class VMemCounters:
    """VMEM-related hardware counters from rocprof."""

    insts_vmem: int = 0  # Total VMEM instructions
    wait_inst_vmem: int = 0  # Cycles waiting for VMEM
    active_waves: int = 0  # Average active waves
    total_cycles: int = 0  # Total kernel cycles

    @property
    def vmem_stall_pct(self) -> float:
        """Percentage of time stalled on VMEM."""
        return (
            100.0 * self.wait_inst_vmem / self.total_cycles
            if self.total_cycles
            else 0.0
        )

    @property
    def avg_wait_per_vmem(self) -> float:
        """Average wait cycles per VMEM instruction."""
        return self.wait_inst_vmem / self.insts_vmem if self.insts_vmem else 0.0


@dataclass
class BufferLoadStats:
    """Buffer/global load instruction statistics from ISA."""

    buffer_load_dword: int = 0  # 32-bit loads
    buffer_load_dwordx2: int = 0  # 64-bit loads
    buffer_load_dwordx3: int = 0  # 96-bit loads
    buffer_load_dwordx4: int = 0  # 128-bit loads
    global_load_dword: int = 0
    global_load_dwordx2: int = 0
    global_load_dwordx4: int = 0
    async_copy_detected: bool = False
    prefetch_distance: Optional[int] = (
        None  # Detected prefetch distance in loop iterations
    )

    @property
    def total_loads(self) -> int:
        return (
            self.buffer_load_dword
            + self.buffer_load_dwordx2
            + self.buffer_load_dwordx3
            + self.buffer_load_dwordx4
            + self.global_load_dword
            + self.global_load_dwordx2
            + self.global_load_dwordx4
        )

    @property
    def avg_load_width_bytes(self) -> float:
        """Average load width in bytes."""
        if self.total_loads == 0:
            return 0.0

        total_bytes = (
            self.buffer_load_dword * 4
            + self.buffer_load_dwordx2 * 8
            + self.buffer_load_dwordx3 * 12
            + self.buffer_load_dwordx4 * 16
            + self.global_load_dword * 4
            + self.global_load_dwordx2 * 8
            + self.global_load_dwordx4 * 16
        )
        return total_bytes / self.total_loads

    @property
    def wide_load_pct(self) -> float:
        """Percentage of loads that are >= 128-bit (good coalescing)."""
        if self.total_loads == 0:
            return 0.0
        wide_loads = self.buffer_load_dwordx4 + self.global_load_dwordx4
        return 100.0 * wide_loads / self.total_loads


@dataclass
class PrefetchAnalysis:
    """Complete prefetch efficiency analysis report."""

    arch: str
    vmem_counters: VMemCounters
    buffer_stats: BufferLoadStats
    efficiency_score: float  # 0-100, higher is better
    bottleneck_type: str  # "latency-bound", "bandwidth-bound", "compute-bound"
    recommendations: List[str]
    estimated_gain_pct: float


def parse_rocprof_csv(csv_path: Path) -> VMemCounters:
    """Parse rocprof stats CSV and extract VMEM counters.

    Args:
        csv_path: Path to rocprof kernel_stats.csv or domain_stats.csv

    Returns:
        VMemCounters with parsed values
    """
    counters = VMemCounters()

    if not csv_path.exists():
        raise FileNotFoundError(f"rocprof CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_str = str(row).lower()

            # Extract VMEM instruction count
            if "sq_insts_vmem" in row_str or "insts_vmem" in row_str:
                for val in row.values():
                    try:
                        counters.insts_vmem = int(float(val))
                        break
                    except (ValueError, TypeError):
                        continue

            # Extract VMEM wait cycles
            if "sq_wait_inst_vmem" in row_str or "wait_inst_vmem" in row_str:
                for val in row.values():
                    try:
                        counters.wait_inst_vmem = int(float(val))
                        break
                    except (ValueError, TypeError):
                        continue

            # Extract active waves
            if "sq_wave_cycles" in row_str or "wave_cycles" in row_str:
                for val in row.values():
                    try:
                        counters.active_waves = int(float(val))
                        break
                    except (ValueError, TypeError):
                        continue

            # Extract total kernel duration (might be in different formats)
            if "totaldurationns" in row_str or "duration" in row_str:
                for key, val in row.items():
                    if "duration" in key.lower():
                        try:
                            # Convert nanoseconds to cycles (assuming ~1.6 GHz)
                            ns = float(val)
                            counters.total_cycles = int(ns * 1.6)  # Approximation
                            break
                        except (ValueError, TypeError):
                            continue

    return counters


def parse_isa_for_loads(isa_path: Path) -> Tuple[BufferLoadStats, bool]:
    """Parse ISA assembly for buffer/global load patterns.

    Args:
        isa_path: Path to ISA assembly file (.s)

    Returns:
        Tuple of (BufferLoadStats, has_software_pipelining)
    """
    if not isa_path.exists():
        raise FileNotFoundError(f"ISA file not found: {isa_path}")

    stats = BufferLoadStats()
    has_pipelining = False
    loop_blocks = []

    with open(isa_path, "r") as f:
        lines = f.readlines()

    # Count buffer/global loads by width
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Buffer loads
        if "buffer_load_dwordx4" in line_lower:
            stats.buffer_load_dwordx4 += 1
        elif "buffer_load_dwordx3" in line_lower:
            stats.buffer_load_dwordx3 += 1
        elif "buffer_load_dwordx2" in line_lower:
            stats.buffer_load_dwordx2 += 1
        elif "buffer_load_dword" in line_lower:
            stats.buffer_load_dword += 1

        # Global loads
        if "global_load_dwordx4" in line_lower:
            stats.global_load_dwordx4 += 1
        elif "global_load_dwordx2" in line_lower:
            stats.global_load_dwordx2 += 1
        elif "global_load_dword" in line_lower:
            stats.global_load_dword += 1

        # Detect async copy (gfx950 feature)
        if "async" in line_lower and (
            "buffer_load" in line_lower or "global_load" in line_lower
        ):
            stats.async_copy_detected = True

        # Detect loop blocks for pipelining analysis
        if re.match(r"BB\d+_\d+:", line):
            loop_blocks.append(i)

    # Detect software pipelining (heuristic: multiple loop blocks with loads)
    if len(loop_blocks) >= 3:
        # Check if there are loads distributed across different loop blocks
        loads_per_block = [0] * len(loop_blocks)
        for i in range(len(loop_blocks)):
            start = loop_blocks[i]
            end = loop_blocks[i + 1] if i + 1 < len(loop_blocks) else len(lines)
            block_lines = lines[start:end]
            loads_per_block[i] = sum(
                1
                for line in block_lines
                if "buffer_load" in line.lower() or "global_load" in line.lower()
            )

        # If loads are spread across multiple blocks, likely pipelined
        if sum(1 for count in loads_per_block if count > 0) >= 2:
            has_pipelining = True

    # Estimate prefetch distance (heuristic: count instructions between load and use)
    # This is simplified - a full analysis would track register dependencies
    # For now, we'll just note if there are loads early in the loop
    if loop_blocks:
        first_block_start = loop_blocks[0]
        first_block_end = loop_blocks[1] if len(loop_blocks) > 1 else len(lines)
        first_block = lines[first_block_start:first_block_end]

        # Count instructions between first load and first MFMA
        first_load_idx = None
        first_mfma_idx = None
        for i, line in enumerate(first_block):
            if first_load_idx is None and (
                "buffer_load" in line.lower() or "global_load" in line.lower()
            ):
                first_load_idx = i
            if first_mfma_idx is None and "v_mfma" in line.lower():
                first_mfma_idx = i
            if first_load_idx is not None and first_mfma_idx is not None:
                break

        if (
            first_load_idx is not None
            and first_mfma_idx is not None
            and first_mfma_idx > first_load_idx
        ):
            stats.prefetch_distance = first_mfma_idx - first_load_idx

    return stats, has_pipelining


def calculate_efficiency_score(
    vmem_counters: VMemCounters, buffer_stats: BufferLoadStats
) -> float:
    """Calculate prefetch efficiency score (0-100).

    Higher score = better latency hiding.

    Args:
        vmem_counters: VMEM hardware counters
        buffer_stats: Buffer load statistics

    Returns:
        Efficiency score (0-100)
    """
    score = 100.0

    # Penalize high VMEM stall percentage
    if vmem_counters.vmem_stall_pct > 40:
        score -= 40  # Major penalty
    elif vmem_counters.vmem_stall_pct > 25:
        score -= 25
    elif vmem_counters.vmem_stall_pct > 15:
        score -= 15
    else:
        score -= vmem_counters.vmem_stall_pct  # Linear penalty below 15%

    # Bonus for wide loads (good coalescing)
    if buffer_stats.wide_load_pct > 70:
        score += 10
    elif buffer_stats.wide_load_pct > 50:
        score += 5

    # Bonus for async copy
    if buffer_stats.async_copy_detected:
        score += 5

    # Bonus for prefetch distance
    if buffer_stats.prefetch_distance and buffer_stats.prefetch_distance > 20:
        score += 10
    elif buffer_stats.prefetch_distance and buffer_stats.prefetch_distance > 10:
        score += 5

    return max(0.0, min(100.0, score))


def classify_bottleneck(vmem_counters: VMemCounters, efficiency_score: float) -> str:
    """Classify bottleneck type.

    Args:
        vmem_counters: VMEM hardware counters
        efficiency_score: Calculated efficiency score

    Returns:
        "latency-bound", "bandwidth-bound", or "compute-bound"
    """
    if vmem_counters.vmem_stall_pct > 30:
        # High stall percentage suggests latency issues
        if vmem_counters.avg_wait_per_vmem > 100:
            return "latency-bound"
        else:
            return "bandwidth-bound"
    elif vmem_counters.vmem_stall_pct < 15:
        return "compute-bound"
    else:
        return "latency-bound"


def estimate_performance_gain(
    bottleneck: str, current_efficiency: float, arch: str
) -> float:
    """Estimate potential performance gain from prefetch optimization.

    Args:
        bottleneck: Bottleneck type
        current_efficiency: Current efficiency score
        arch: Architecture

    Returns:
        Estimated percentage performance gain
    """
    if bottleneck == "compute-bound":
        return 0.0  # Already optimized

    if bottleneck == "latency-bound":
        # Latency-bound: significant gains possible with better prefetching
        if current_efficiency < 50:
            return 15.0  # Up to 15% gain
        elif current_efficiency < 70:
            return 8.0
        else:
            return 3.0
    else:  # bandwidth-bound
        # Bandwidth-bound: modest gains from better coalescing
        if current_efficiency < 60:
            return 7.0
        else:
            return 2.0


def generate_recommendations(
    vmem_counters: VMemCounters,
    buffer_stats: BufferLoadStats,
    efficiency_score: float,
    bottleneck: str,
    arch: str,
    has_pipelining: bool,
) -> List[str]:
    """Generate actionable recommendations for prefetch optimization.

    Args:
        vmem_counters: VMEM hardware counters
        buffer_stats: Buffer load statistics
        efficiency_score: Calculated efficiency score
        bottleneck: Bottleneck type
        arch: Architecture
        has_pipelining: Whether software pipelining detected

    Returns:
        List of recommendation strings
    """
    recs = []

    # Overall assessment
    if efficiency_score >= 80:
        recs.append(
            f"✅ Excellent prefetch efficiency ({efficiency_score:.1f}/100) - well optimized!"
        )
    elif efficiency_score >= 60:
        recs.append(
            f"✅ Good prefetch efficiency ({efficiency_score:.1f}/100) - minor improvements possible"
        )
    elif efficiency_score >= 40:
        recs.append(
            f"⚠️ Moderate prefetch efficiency ({efficiency_score:.1f}/100) - optimization recommended"
        )
    else:
        recs.append(
            f"🔴 Poor prefetch efficiency ({efficiency_score:.1f}/100) - significant optimization needed!"
        )

    # Bottleneck-specific recommendations
    recs.append(f"ℹ️ Bottleneck type: {bottleneck.upper()}")

    if bottleneck == "latency-bound":
        recs.append(
            f"💡 Memory latency is limiting performance (VMEM stall: {vmem_counters.vmem_stall_pct:.1f}%)"
        )

        if not has_pipelining:
            recs.append(
                "💡 Implement software pipelining: Separate loop into prefetch, compute, and store stages"
            )
        else:
            recs.append(
                "💡 Increase pipeline depth: Move loads earlier in the loop (3-stage vs 2-stage pipelining)"
            )

        if not buffer_stats.prefetch_distance or buffer_stats.prefetch_distance < 15:
            recs.append(
                "💡 Increase prefetch distance: Move buffer_load instructions at least 15-20 instructions before use"
            )

        if "gfx950" in arch.lower() and not buffer_stats.async_copy_detected:
            recs.append(
                "💡 Use async_buffer_load on gfx950: Enables true async memory ops with lower latency"
            )

    elif bottleneck == "bandwidth-bound":
        recs.append(
            "💡 Memory bandwidth is limiting performance - optimize data movement"
        )

        if buffer_stats.wide_load_pct < 50:
            recs.append(
                f"💡 Improve coalescing: {buffer_stats.wide_load_pct:.1f}% of loads are 128-bit. "
                "Increase to >70% by using dwordx4 loads where possible"
            )

        recs.append(
            "💡 Reduce redundant loads: Check if data can be reused from LDS or registers"
        )

    else:  # compute-bound
        recs.append(
            "✅ Compute-bound - memory latency is well hidden. Focus on compute optimizations."
        )

    # Load width analysis
    if buffer_stats.total_loads > 0:
        recs.append(
            f"ℹ️ Average load width: {buffer_stats.avg_load_width_bytes:.1f} bytes "
            f"({buffer_stats.wide_load_pct:.1f}% are 128-bit)"
        )

        if buffer_stats.avg_load_width_bytes < 8:
            recs.append(
                "💡 Loads are narrow (avg < 8 bytes) - vectorize loads to improve bandwidth utilization"
            )

    # Async copy
    if buffer_stats.async_copy_detected:
        recs.append("✅ Async copy detected - good use of gfx950 features!")

    # Estimated gain
    estimated_gain = estimate_performance_gain(bottleneck, efficiency_score, arch)
    if estimated_gain > 0:
        recs.append(
            f"📈 Estimated performance gain: ~{estimated_gain:.1f}% from prefetch optimization"
        )

    return recs


def print_analysis(analysis: PrefetchAnalysis):
    """Print formatted analysis report to console.

    Args:
        analysis: Prefetch efficiency analysis report
    """
    print(f"\n{'=' * 90}")
    print(f"  Prefetch Efficiency Analysis - {analysis.arch}")
    print(f"{'=' * 90}")
    print()

    # VMEM counters
    print("VMEM Hardware Counters (from rocprof):")
    print(f"  VMEM instructions:     {analysis.vmem_counters.insts_vmem:,}")
    print(f"  VMEM wait cycles:      {analysis.vmem_counters.wait_inst_vmem:,}")
    print(f"  Total kernel cycles:   {analysis.vmem_counters.total_cycles:,}")
    print(f"  VMEM stall %:          {analysis.vmem_counters.vmem_stall_pct:.2f}%")
    print(
        f"  Avg wait/VMEM inst:    {analysis.vmem_counters.avg_wait_per_vmem:.1f} cycles"
    )
    print()

    # Buffer load statistics
    print("Buffer/Global Load Statistics (from ISA):")
    stats = analysis.buffer_stats
    print(f"  buffer_load_dword:     {stats.buffer_load_dword}")
    print(f"  buffer_load_dwordx2:   {stats.buffer_load_dwordx2}")
    print(f"  buffer_load_dwordx4:   {stats.buffer_load_dwordx4}")
    print(
        f"  global_load_*:         {stats.global_load_dword + stats.global_load_dwordx2 + stats.global_load_dwordx4}"
    )
    print(f"  Total loads:           {stats.total_loads}")
    print(f"  Avg load width:        {stats.avg_load_width_bytes:.1f} bytes")
    print(f"  Wide load %:           {stats.wide_load_pct:.1f}% (128-bit)")
    print(
        f"  Async copy:            {'✅ Yes' if stats.async_copy_detected else '❌ No'}"
    )
    if stats.prefetch_distance:
        print(f"  Prefetch distance:     ~{stats.prefetch_distance} instructions")
    print()

    # Efficiency score
    print(f"Efficiency Score: {analysis.efficiency_score:.1f}/100")
    print(f"Bottleneck Type:  {analysis.bottleneck_type.upper()}")
    print()

    # Recommendations
    print("Recommendations:")
    for rec in analysis.recommendations:
        print(f"  {rec}")
    print()


def main():
    """Command-line interface for prefetch efficiency analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze global memory prefetch effectiveness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with rocprof stats and ISA
  %(prog)s --rocprof kernel_stats.csv --isa kernel.s --arch gfx950

  # Save JSON report
  %(prog)s --rocprof stats.csv --isa kernel.s --arch gfx942 --output report.json

  # Analyze without ISA (counters only)
  %(prog)s --rocprof kernel_stats.csv --arch gfx950
        """,
    )

    parser.add_argument(
        "--rocprof",
        type=Path,
        required=True,
        help="Path to rocprof kernel_stats.csv or domain_stats.csv",
    )
    parser.add_argument(
        "--isa",
        type=Path,
        help="Path to ISA assembly file (.s) for load pattern analysis",
    )
    parser.add_argument(
        "--arch",
        choices=["gfx942", "gfx950"],
        default="gfx950",
        help="GPU architecture (default: gfx950)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file (default: print to console)"
    )

    args = parser.parse_args()

    try:
        # Parse rocprof counters
        vmem_counters = parse_rocprof_csv(args.rocprof)

        # Parse ISA if provided
        if args.isa:
            buffer_stats, has_pipelining = parse_isa_for_loads(args.isa)
        else:
            buffer_stats = BufferLoadStats()
            has_pipelining = False
            print(
                "Warning: No ISA file provided - load pattern analysis will be limited",
                file=sys.stderr,
            )

        # Calculate metrics
        efficiency_score = calculate_efficiency_score(vmem_counters, buffer_stats)
        bottleneck = classify_bottleneck(vmem_counters, efficiency_score)
        estimated_gain = estimate_performance_gain(
            bottleneck, efficiency_score, args.arch
        )

        # Generate recommendations
        recommendations = generate_recommendations(
            vmem_counters,
            buffer_stats,
            efficiency_score,
            bottleneck,
            args.arch,
            has_pipelining,
        )

        # Build analysis report
        analysis = PrefetchAnalysis(
            arch=args.arch,
            vmem_counters=vmem_counters,
            buffer_stats=buffer_stats,
            efficiency_score=efficiency_score,
            bottleneck_type=bottleneck,
            recommendations=recommendations,
            estimated_gain_pct=estimated_gain,
        )

        # Output
        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(analysis), f, indent=2)
            print(f"Analysis saved to {args.output}")
        else:
            print_analysis(analysis)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
