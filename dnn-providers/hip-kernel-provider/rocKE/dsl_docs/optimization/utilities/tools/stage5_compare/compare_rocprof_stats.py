# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare rocprof stats between CK DSL and CK Tile C++"""

import argparse
import csv
import sys


def parse_kernel_trace(filepath):
    """Extract hardware metrics from kernel trace CSV"""
    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = [
            r
            for r in reader
            if r["Kind"] == "KERNEL_DISPATCH"
            and "rocke" in r.get("Kernel_Name", "").lower()
            or "grouped_conv" in r.get("Kernel_Name", "").lower()
        ]

        if not rows:
            print(f"No kernel dispatches found in {filepath}")
            return None

        # Use first kernel dispatch
        row = rows[0]

        return {
            "kernel_name": row["Kernel_Name"],
            "vgpr": int(row["VGPR_Count"]),
            "agpr": int(row["Accum_VGPR_Count"]),
            "sgpr": int(row["SGPR_Count"]),
            "lds_bytes": int(row["LDS_Block_Size"]),
            "scratch": int(row["Scratch_Size"]),
            "workgroup_size": int(row["Workgroup_Size_X"])
            * int(row["Workgroup_Size_Y"])
            * int(row["Workgroup_Size_Z"]),
            "grid_x": int(row["Grid_Size_X"]),
            "grid_y": int(row["Grid_Size_Y"]),
            "grid_z": int(row["Grid_Size_Z"]),
        }


def calculate_occupancy(vgpr, agpr, lds_bytes):
    """Calculate theoretical occupancy for gfx950"""
    # gfx950 limits (MI355X)
    MAX_VGPR_PER_CU = 512  # Total VGPRs per CU
    MAX_WAVES_PER_CU = 16  # Hardware limit
    LDS_PER_CU = 163840  # 160 KB

    # VGPR limit
    vgpr_limit = MAX_VGPR_PER_CU // vgpr if vgpr > 0 else MAX_WAVES_PER_CU

    # AGPR limit (gfx950 has separate AGPR file, doesn't limit VGPR)
    # But each wave needs VGPRs + AGPRs, so total register pressure matters

    # LDS limit (per workgroup, 4 waves per workgroup on CK kernels)
    waves_per_wg = 4  # Typical for 256 threads (64 threads/wave)
    lds_per_wave = lds_bytes // waves_per_wg if waves_per_wg > 0 else lds_bytes
    lds_limit = LDS_PER_CU // lds_per_wave if lds_per_wave > 0 else MAX_WAVES_PER_CU

    # Take minimum
    occupancy = min(vgpr_limit, lds_limit, MAX_WAVES_PER_CU)

    return occupancy, vgpr_limit, lds_limit


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "rocke_csv",
    nargs="?",
    default="experiments/rocke_rocprof_stats_kernel_trace.csv",
    help="rocprof kernel-trace CSV for CK DSL",
)
parser.add_argument(
    "cktile_csv",
    nargs="?",
    default="experiments/cktile_rocprof_stats_kernel_trace.csv",
    help="rocprof kernel-trace CSV for the comparison baseline (CK Tile C++, FlyDSL, etc.)",
)
args = parser.parse_args()

# Parse both files
rocke = parse_kernel_trace(args.rocke_csv)
cktile = parse_kernel_trace(args.cktile_csv)

if not rocke or not cktile:
    sys.exit(1)

print("=" * 80)
print("CK DSL vs CK Tile C++ Hardware Metrics Comparison")
print("=" * 80)
print()

print(f"{'Metric':<20} {'CK DSL':<20} {'CK Tile C++':<20} {'Analysis':<30}")
print("-" * 90)

# VGPRs
print(f"{'VGPRs':<20} {rocke['vgpr']:<20} {cktile['vgpr']:<20} ", end="")
if rocke["vgpr"] < cktile["vgpr"]:
    print("✓ CK DSL uses FEWER")
elif rocke["vgpr"] > cktile["vgpr"]:
    print("⚠ CK DSL uses MORE")
else:
    print("Same")

# AGPRs
print(f"{'AGPRs':<20} {rocke['agpr']:<20} {cktile['agpr']:<20} ", end="")
if rocke["agpr"] == cktile["agpr"]:
    print("Same")
else:
    print("")

# SGPRs
print(f"{'SGPRs':<20} {rocke['sgpr']:<20} {cktile['sgpr']:<20} ", end="")
if rocke["sgpr"] < cktile["sgpr"]:
    print("✓ CK DSL uses fewer")
else:
    print("")

# LDS
print(
    f"{'LDS (bytes)':<20} {rocke['lds_bytes']:<20} {cktile['lds_bytes']:<20} ", end=""
)
lds_diff_pct = (
    (rocke["lds_bytes"] / cktile["lds_bytes"] - 1) * 100
    if cktile["lds_bytes"] > 0
    else 0
)
print(f"({lds_diff_pct:+.1f}%)")

# Workgroup size
print(
    f"{'Workgroup size':<20} {rocke['workgroup_size']:<20} {cktile['workgroup_size']:<20} ",
    end="",
)
if rocke["workgroup_size"] == cktile["workgroup_size"]:
    print("Same (256 threads)")
else:
    print("")

# Grid size
rocke_grid_total = rocke["grid_x"] * rocke["grid_y"] * rocke["grid_z"]
cktile_grid_total = cktile["grid_x"] * cktile["grid_y"] * cktile["grid_z"]
print(f"{'Grid size':<20} {rocke_grid_total:<20} {cktile_grid_total:<20} ", end="")
if rocke_grid_total == cktile_grid_total:
    print("Same workgroups")
else:
    print("")

print()
print("Occupancy Analysis (gfx950 MI355X):")
print("-" * 90)

rocke_occ, rocke_vgpr_lim, rocke_lds_lim = calculate_occupancy(
    rocke["vgpr"], rocke["agpr"], rocke["lds_bytes"]
)
cktile_occ, cktile_vgpr_lim, cktile_lds_lim = calculate_occupancy(
    cktile["vgpr"], cktile["agpr"], cktile["lds_bytes"]
)

print(
    f"{'Framework':<20} {'VGPR Limit':<15} {'LDS Limit':<15} {'Occupancy':<15} {'Percentage':<15}"
)
print("-" * 90)
print(
    f"{'CK DSL':<20} {rocke_vgpr_lim:<15} {rocke_lds_lim:<15} {rocke_occ:<15} {rocke_occ / 16 * 100:.1f}%"
)
print(
    f"{'CK Tile C++':<20} {cktile_vgpr_lim:<15} {cktile_lds_lim:<15} {cktile_occ:<15} {cktile_occ / 16 * 100:.1f}%"
)

print()
print("Key Findings:")
print("-" * 90)

if rocke["vgpr"] < cktile["vgpr"]:
    print(f"✓ CK DSL has BETTER VGPR efficiency ({rocke['vgpr']} vs {cktile['vgpr']})")
    print(
        f"  → CK DSL should have HIGHER occupancy ({rocke_occ} vs {cktile_occ} waves/CU)"
    )
    if rocke_occ > cktile_occ:
        print(
            "  → Yet CK DSL is SLOWER! This confirms: occupancy is NOT the bottleneck"
        )
        print(
            "  → The gap must be from: LDS bank conflicts, memory access patterns, or instruction scheduling"
        )
else:
    print(
        f"⚠ CK DSL has similar or worse VGPR usage ({rocke['vgpr']} vs {cktile['vgpr']})"
    )

if rocke["lds_bytes"] > cktile["lds_bytes"]:
    print(
        f"⚠ CK DSL uses {lds_diff_pct:+.1f}% MORE LDS ({rocke['lds_bytes']} vs {cktile['lds_bytes']} bytes)"
    )
    print("  → Potential LDS pressure or different allocation strategy")
elif rocke["lds_bytes"] < cktile["lds_bytes"]:
    print(
        f"✓ CK DSL uses {abs(lds_diff_pct):.1f}% LESS LDS ({rocke['lds_bytes']} vs {cktile['lds_bytes']} bytes)"
    )

print()
print("Conclusion:")
print("-" * 90)
if rocke_occ >= cktile_occ:
    print("CK DSL has EQUAL or BETTER occupancy than CK Tile C++")
    print("→ Occupancy is NOT the bottleneck")
    print("→ Performance gap likely from:")
    print("  1. LDS bank conflicts (need ATT profiling to confirm)")
    print("  2. Memory access patterns (coalescing, cache locality)")
    print("  3. Instruction scheduling (low unroll factor from ISA analysis)")
else:
    print("CK DSL has LOWER occupancy than CK Tile C++")
    print("→ Occupancy MAY be a contributing factor")
    print("→ But likely not the primary bottleneck (need ATT to confirm)")
