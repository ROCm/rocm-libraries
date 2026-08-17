# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Analyze LDS bank conflicts and recommend swizzle strategies.

Deep analysis of LDS (Local Data Share) bank conflicts based on:
- rocprof hardware counters (SQ_LDS_BANK_CONFLICT, SQ_LDS_IDX_ACTIVE)
- ISA disassembly (ds_read/ds_write patterns)
- Architecture-specific conflict period rules (gfx942 vs gfx950)

References runbook sections 6.3, 6.4, 6.4a + LDS_a.md, LDS_b.md

Usage:
    python analyze_lds_conflicts.py --rocprof stats.csv --isa kernel.s --arch gfx950
    python analyze_lds_conflicts.py --rocprof stats.csv --isa kernel.s --output report.json
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


@dataclass
class LDSCounters:
    """LDS-related hardware performance counters from rocprof."""

    bank_conflict: int = 0
    idx_active: int = 0
    unaligned_stall: int = 0

    @property
    def conflict_rate(self) -> float:
        """Conflicts per 1000 LDS operations."""
        return 1000.0 * self.bank_conflict / self.idx_active if self.idx_active else 0.0


@dataclass
class LDSInstructionStats:
    """Statistics from ISA disassembly."""

    ds_read_b32: int = 0
    ds_read_b64: int = 0
    ds_read_b96: int = 0
    ds_read_b128: int = 0
    ds_write_b32: int = 0
    ds_write_b64: int = 0
    ds_write_b96: int = 0
    ds_write_b128: int = 0

    @property
    def total_ds_ops(self) -> int:
        return (
            self.ds_read_b32
            + self.ds_read_b64
            + self.ds_read_b96
            + self.ds_read_b128
            + self.ds_write_b32
            + self.ds_write_b64
            + self.ds_write_b96
            + self.ds_write_b128
        )


@dataclass
class LDSConflictAnalysis:
    """Complete LDS conflict analysis report."""

    arch: str
    bank_count: int
    counters: LDSCounters
    isa_stats: LDSInstructionStats
    conflict_severity: str  # "low", "medium", "high"
    has_xor_swizzle: bool
    has_padding: bool
    detected_power_of_2_stride: bool
    recommendations: List[str]
    estimated_gain_pct: float


def parse_rocprof_csv(csv_path: Path) -> LDSCounters:
    """Parse rocprof stats CSV and extract LDS counters.

    Args:
        csv_path: Path to rocprof kernel_stats.csv or domain_stats.csv

    Returns:
        LDSCounters with parsed values
    """
    counters = LDSCounters()

    if not csv_path.exists():
        raise FileNotFoundError(f"rocprof CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        # Try to detect CSV format (kernel_stats vs domain_stats)
        reader = csv.DictReader(f)
        for row in reader:
            # Different formats have different column names
            # kernel_stats: Name, Calls, TotalDurationNs, ...
            # domain_stats: might have Counter_Name, Value, etc.

            # For now, we'll look for specific counter values in any column
            row_str = str(row).lower()

            if "sq_lds_bank_conflict" in row_str or "lds_bank_conflict" in row_str:
                # Try to extract numeric value
                for val in row.values():
                    try:
                        counters.bank_conflict = int(float(val))
                        break
                    except (ValueError, TypeError):
                        continue

            if "sq_lds_idx_active" in row_str or "lds_idx_active" in row_str:
                for val in row.values():
                    try:
                        counters.idx_active = int(float(val))
                        break
                    except (ValueError, TypeError):
                        continue

            if "sq_lds_unaligned_stall" in row_str or "lds_unaligned" in row_str:
                for val in row.values():
                    try:
                        counters.unaligned_stall = int(float(val))
                        break
                    except (ValueError, TypeError):
                        continue

    return counters


def parse_isa_for_lds(isa_path: Path) -> Tuple[LDSInstructionStats, bool, bool, bool]:
    """Parse ISA assembly for LDS instruction patterns.

    Args:
        isa_path: Path to ISA assembly file (.s)

    Returns:
        Tuple of (LDSInstructionStats, has_xor_swizzle, has_padding, has_power_of_2_stride)
    """
    if not isa_path.exists():
        raise FileNotFoundError(f"ISA file not found: {isa_path}")

    stats = LDSInstructionStats()
    has_xor = False
    has_padding = False
    power_of_2_stride = False

    with open(isa_path, "r") as f:
        lines = f.readlines()

    # Count LDS operations by size
    for line in lines:
        line_lower = line.lower().strip()

        # Count ds_read/ds_write by size
        if "ds_read_b128" in line_lower:
            stats.ds_read_b128 += 1
        elif "ds_read_b96" in line_lower:
            stats.ds_read_b96 += 1
        elif "ds_read_b64" in line_lower:
            stats.ds_read_b64 += 1
        elif "ds_read_b32" in line_lower or "ds_read_" in line_lower:
            stats.ds_read_b32 += 1

        if "ds_write_b128" in line_lower:
            stats.ds_write_b128 += 1
        elif "ds_write_b96" in line_lower:
            stats.ds_write_b96 += 1
        elif "ds_write_b64" in line_lower:
            stats.ds_write_b64 += 1
        elif "ds_write_b32" in line_lower or "ds_write_" in line_lower:
            stats.ds_write_b32 += 1

        # Detect XOR swizzle (v_xor_b32 used in LDS address computation)
        if "v_xor_b32" in line_lower and (
            "offset" in line_lower or "addr" in line_lower
        ):
            has_xor = True

        # Detect padding (irregular offset patterns, non-power-of-2 strides)
        # Look for ds_read/write with offsets like offset:288, offset:640 (not 256, 512)
        if (
            "ds_read" in line_lower or "ds_write" in line_lower
        ) and "offset:" in line_lower:
            match = re.search(r"offset:(\d+)", line_lower)
            if match:
                offset = int(match.group(1))
                # Check if offset is NOT a power of 2 (indicates padding)
                if offset > 0 and (offset & (offset - 1)) != 0:
                    has_padding = True

    # Detect power-of-2 stride pattern (high conflict risk)
    # This is a heuristic: if we see many ds_read/write with power-of-2 offsets
    power_of_2_count = 0
    total_offset_count = 0
    for line in lines:
        if ("ds_read" in line or "ds_write" in line) and "offset:" in line:
            match = re.search(r"offset:(\d+)", line)
            if match:
                offset = int(match.group(1))
                total_offset_count += 1
                if offset > 0 and (offset & (offset - 1)) == 0:
                    power_of_2_count += 1

    if total_offset_count > 10 and power_of_2_count / total_offset_count > 0.7:
        power_of_2_stride = True

    return stats, has_xor, has_padding, power_of_2_stride


def calculate_conflict_severity(conflict_rate: float) -> str:
    """Classify conflict severity based on rate.

    Args:
        conflict_rate: Conflicts per 1000 LDS operations

    Returns:
        "low", "medium", or "high"
    """
    if conflict_rate < 10:
        return "low"
    elif conflict_rate < 50:
        return "medium"
    else:
        return "high"


def estimate_performance_gain(severity: str, has_swizzle: bool, arch: str) -> float:
    """Estimate potential performance gain from conflict elimination.

    Args:
        severity: Conflict severity ("low", "medium", "high")
        has_swizzle: Whether XOR swizzle is already applied
        arch: Architecture (gfx942 or gfx950)

    Returns:
        Estimated percentage performance gain
    """
    if has_swizzle:
        # Already optimized
        return 0.0

    # Conservative estimates based on empirical data
    gain_map = {
        "low": 1.0,  # 1% gain
        "medium": 3.0,  # 3% gain
        "high": 8.0,  # 8% gain (can be 5-15% in practice)
    }

    base_gain = gain_map.get(severity, 0.0)

    # gfx950 has 64 banks vs gfx942's 32 banks, so conflicts are more impactful
    if "gfx950" in arch.lower():
        base_gain *= 1.2

    return base_gain


def generate_recommendations(
    counters: LDSCounters,
    isa_stats: LDSInstructionStats,
    severity: str,
    has_xor: bool,
    has_padding: bool,
    power_of_2_stride: bool,
    arch: str,
) -> List[str]:
    """Generate actionable recommendations for LDS conflict optimization.

    Args:
        counters: LDS hardware counters
        isa_stats: ISA instruction statistics
        severity: Conflict severity
        has_xor: Whether XOR swizzle detected
        has_padding: Whether padding detected
        power_of_2_stride: Whether power-of-2 stride pattern detected
        arch: Architecture (gfx942 or gfx950)

    Returns:
        List of recommendation strings
    """
    recs = []

    # Overall severity assessment
    if severity == "low":
        recs.append("✅ LDS bank conflicts are low - no immediate action needed")
    elif severity == "medium":
        recs.append(
            f"⚠️ Moderate LDS bank conflicts ({counters.conflict_rate:.1f} per 1000 ops)"
        )
    else:
        recs.append(
            f"🔴 HIGH LDS bank conflicts ({counters.conflict_rate:.1f} per 1000 ops) - optimization needed!"
        )

    # Architecture-specific guidance
    bank_count = 64 if "gfx950" in arch.lower() else 32
    recs.append(f"ℹ️ Architecture: {arch} ({bank_count} LDS banks)")

    # XOR swizzle recommendation
    if not has_xor and severity in ["medium", "high"]:
        if "gfx950" in arch.lower():
            recs.append(
                "💡 Apply XOR swizzle: addr_swizzled = addr ^ ((addr >> 7) & 0x1F)  [64-bank gfx950]"
            )
        else:
            recs.append(
                "💡 Apply XOR swizzle: addr_swizzled = addr ^ ((addr >> 6) & 0xF)  [32-bank gfx942]"
            )
    elif has_xor:
        recs.append("✅ XOR swizzle already applied - good!")

    # Padding recommendation
    if power_of_2_stride and not has_padding and severity in ["medium", "high"]:
        recs.append(
            "💡 Add padding: Use non-power-of-2 strides (e.g., pad from 256 to 288 bytes) "
            "to avoid systematic bank conflicts"
        )
    elif has_padding:
        recs.append("✅ Padding detected - helps avoid power-of-2 conflicts")

    # Instruction-specific guidance based on LDS_a.md conflict period rules
    if isa_stats.ds_read_b128 > 50 or isa_stats.ds_write_b128 > 50:
        conflict_period = 64 if "gfx950" in arch.lower() else 32
        recs.append(
            f"ℹ️ Many ds_read/write_b128 instructions (128-bit LDS ops) - "
            f"Conflict period: {conflict_period} dwords. "
            f"Ensure stride > {conflict_period} dwords or apply swizzle"
        )

    # Coalescing recommendation
    if isa_stats.total_ds_ops > 100:
        recs.append(
            "💡 Consider coalescing LDS accesses: Combine multiple narrow reads/writes "
            "into wider operations (e.g., b32 → b64 or b128)"
        )

    # Estimated gain
    estimated_gain = estimate_performance_gain(severity, has_xor, arch)
    if estimated_gain > 0:
        recs.append(
            f"📈 Estimated performance gain: ~{estimated_gain:.1f}% from conflict elimination"
        )

    return recs


def print_analysis(analysis: LDSConflictAnalysis):
    """Print formatted analysis report to console.

    Args:
        analysis: LDS conflict analysis report
    """
    print(f"\n{'=' * 90}")
    print(f"  LDS Bank Conflict Analysis - {analysis.arch}")
    print(f"{'=' * 90}")
    print()

    # Hardware counters
    print("Hardware Counters (from rocprof):")
    print(f"  SQ_LDS_BANK_CONFLICT:  {analysis.counters.bank_conflict:,}")
    print(f"  SQ_LDS_IDX_ACTIVE:     {analysis.counters.idx_active:,}")
    if analysis.counters.unaligned_stall > 0:
        print(f"  SQ_LDS_UNALIGNED:      {analysis.counters.unaligned_stall:,}")
    print(
        f"  Conflict rate:         {analysis.counters.conflict_rate:.2f} per 1000 LDS ops"
    )
    print(f"  Severity:              {analysis.conflict_severity.upper()}")
    print()

    # ISA statistics
    print("LDS Instructions (from ISA):")
    stats = analysis.isa_stats
    print(f"  ds_read_b32:   {stats.ds_read_b32}")
    print(f"  ds_read_b64:   {stats.ds_read_b64}")
    print(f"  ds_read_b128:  {stats.ds_read_b128}")
    print(f"  ds_write_b32:  {stats.ds_write_b32}")
    print(f"  ds_write_b64:  {stats.ds_write_b64}")
    print(f"  ds_write_b128: {stats.ds_write_b128}")
    print(f"  Total LDS ops: {stats.total_ds_ops}")
    print()

    # Pattern detection
    print("Pattern Detection:")
    print(
        f"  XOR swizzle detected:      {'✅ Yes' if analysis.has_xor_swizzle else '❌ No'}"
    )
    print(
        f"  Padding detected:          {'✅ Yes' if analysis.has_padding else '❌ No'}"
    )
    print(
        f"  Power-of-2 stride pattern: {'⚠️ Yes (high risk)' if analysis.detected_power_of_2_stride else '✅ No'}"
    )
    print()

    # Recommendations
    print("Recommendations:")
    for rec in analysis.recommendations:
        print(f"  {rec}")
    print()


def main():
    """Command-line interface for LDS conflict analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze LDS bank conflicts and recommend optimizations",
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
        "--isa", type=Path, help="Path to ISA assembly file (.s) for pattern analysis"
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
        counters = parse_rocprof_csv(args.rocprof)

        # Parse ISA if provided
        if args.isa:
            isa_stats, has_xor, has_padding, power_of_2_stride = parse_isa_for_lds(
                args.isa
            )
        else:
            isa_stats = LDSInstructionStats()
            has_xor = False
            has_padding = False
            power_of_2_stride = False
            print(
                "Warning: No ISA file provided - pattern detection will be limited",
                file=sys.stderr,
            )

        # Calculate severity
        severity = calculate_conflict_severity(counters.conflict_rate)

        # Generate recommendations
        recommendations = generate_recommendations(
            counters,
            isa_stats,
            severity,
            has_xor,
            has_padding,
            power_of_2_stride,
            args.arch,
        )

        # Estimate gain
        estimated_gain = estimate_performance_gain(severity, has_xor, args.arch)

        # Build analysis report
        bank_count = 64 if args.arch == "gfx950" else 32
        analysis = LDSConflictAnalysis(
            arch=args.arch,
            bank_count=bank_count,
            counters=counters,
            isa_stats=isa_stats,
            conflict_severity=severity,
            has_xor_swizzle=has_xor,
            has_padding=has_padding,
            detected_power_of_2_stride=power_of_2_stride,
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
