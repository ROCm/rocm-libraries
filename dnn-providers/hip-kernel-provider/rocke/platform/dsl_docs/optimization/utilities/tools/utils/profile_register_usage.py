# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Profile CK DSL kernels to get VGPR/SGPR/AGPR usage and occupancy."""

import sys
import subprocess
import csv
import tempfile
from pathlib import Path

# Add CK DSL to path — resolve relative to this file so no absolute paths needed.
ROCKE_ROOT = Path(__file__).resolve().parents[7]  # .../python
if str(ROCKE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROCKE_ROOT))

from rocke.helpers import (  # noqa: E402
    compile_kernel,
    make_conv_manifest,
    write_artifact,
)
from rocke.instances.common.conv_implicit_gemm import (  # noqa: E402
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)

# Problem: ResNet50 conv3_1
problem = ConvProblem(
    N=16,
    Hi=56,
    Wi=56,
    C=512,
    K=512,
    Y=3,
    X=3,
    sH=1,
    sW=1,
    pH=1,
    pW=1,
    dH=1,
    dW=1,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "comparison" / "rocke_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def profile_config(name, pipeline, async_dma):
    """Build and profile a CK DSL configuration."""
    print(f"\n{'=' * 60}")
    print(f"Profiling: {name}")
    print(f"  Pipeline: {pipeline}, async_dma: {async_dma}")
    print(f"{'=' * 60}")

    spec = ImplicitGemmConvSpec(
        problem=problem,
        name=f"rocke_{name}",
        tile_m=64,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline=pipeline,
        epilogue="cshuffle",
        async_dma=async_dma,
    )

    # Compile
    print("Compiling...")
    kernel = build_implicit_gemm_conv(spec)
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
    print(f"  Kernel: {artifact.kernel_name}")

    # Save HSACO
    hsaco_path = OUT_DIR / f"{name}.hsaco"
    with open(hsaco_path, "wb") as f:
        f.write(artifact.hsaco)

    # Create manifest and run with rocprof
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        p = problem
        manifest = make_conv_manifest(
            artifact=artifact,
            block_m=spec.tile_m,
            block_n=spec.tile_n,
            block_k=spec.tile_k,
            threads_per_block=spec.block_size,
            conv=[
                p.N,
                p.Hi,
                p.Wi,
                p.C,
                p.K,
                p.Y,
                p.X,
                p.sH,
                p.sW,
                p.pH,
                p.pW,
                p.dH,
                p.dW,
            ],
            groups=1,
            cpg=p.C,
            kpg=p.K,
            conv_layout="implicit_gemm",
            grid_order="NM",
            warmup_iters=5,
            timed_iters=10,  # Fewer iterations for profiling
            atoms=[
                f"tile.mfma_f32_{spec.warp_tile_m}x{spec.warp_tile_n}x{spec.warp_tile_k}_f16"
            ],
            notes=f"{name}",
        )

        written_paths = write_artifact(artifact, tmpdir_path, manifest)
        manifest_path = written_paths["manifest"]

        # Run with rocprof to capture stats
        output_dir = OUT_DIR / f"{name}_rocprof"
        output_dir.mkdir(exist_ok=True)

        print("Running rocprof...")
        cmd = [
            "rocprofv3",
            "--stats",
            "--kernel-trace",
            "-f",
            "csv",
            "-o",
            str(output_dir / "run"),
            "--",
            "python3",
            str(ROCKE_ROOT / "rocke" / "run_manifest.py"),
            str(manifest_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print("  ERROR: rocprof failed")
            print(result.stderr)
            return None

        # Parse kernel stats
        stats_csv = output_dir / "run_kernel_stats.csv"
        if not stats_csv.exists():
            print("  ERROR: No stats CSV found")
            return None

        with open(stats_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("  ERROR: Empty stats CSV")
            return None

        # Get the first kernel row (should be our kernel)
        row = rows[0]

        stats = {
            "name": name,
            "kernel_name": row.get("Kernel_Name", ""),
            "vgpr": int(row.get("VGPRs", 0)) if row.get("VGPRs") else None,
            "sgpr": int(row.get("SGPRs", 0)) if row.get("SGPRs") else None,
            "agpr": int(row.get("AccVGPRs", 0)) if row.get("AccVGPRs") else None,
            "lds_bytes": int(row.get("LDS", 0)) if row.get("LDS") else None,
            "occupancy": (
                float(row.get("OccupancyPct", 0)) if row.get("OccupancyPct") else None
            ),
            "avg_time_ns": (
                float(row.get("AverageNs", 0)) if row.get("AverageNs") else None
            ),
        }

        print("\nStats:")
        print(f"  VGPRs: {stats['vgpr']}")
        print(f"  SGPRs: {stats['sgpr']}")
        print(f"  AccVGPRs (AGPRs): {stats['agpr']}")
        print(f"  LDS bytes: {stats['lds_bytes']}")
        print(f"  Occupancy: {stats['occupancy']:.1f}%")
        print(f"  Avg time: {stats['avg_time_ns'] / 1e6:.3f} ms")

        return stats


if __name__ == "__main__":
    results = []

    # Profile mem_sync (current best)
    stats = profile_config("mem_sync", "mem", async_dma=False)
    if stats:
        results.append(stats)

    # Print summary
    print("\n" + "=" * 60)
    print("REGISTER USAGE SUMMARY")
    print("=" * 60)
    print(
        f"{'Config':<15} {'VGPRs':<8} {'SGPRs':<8} {'AGPRs':<8} {'LDS (KB)':<10} {'Occ %':<8}"
    )
    print("-" * 60)

    for s in results:
        lds_kb = s["lds_bytes"] / 1024 if s["lds_bytes"] else 0
        print(
            f"{s['name']:<15} {s['vgpr'] or 'N/A':<8} {s['sgpr'] or 'N/A':<8} "
            f"{s['agpr'] or 'N/A':<8} {lds_kb:<10.1f} {s['occupancy'] or 0:<8.1f}"
        )

    # Calculate occupancy limits
    print("\n" + "=" * 60)
    print("OCCUPANCY ANALYSIS (gfx950 / MI355X)")
    print("=" * 60)
    print("Limits per CU:")
    print("  Max waves: 8")
    print("  Max VGPRs: 512 total → 512/waves VGPRs per wave")
    print("  Max SGPRs: 800 total → 800/waves SGPRs per wave")
    print("  Max AGPRs: 256 total → 256/waves AGPRs per wave")
    print("  Max LDS: 163,840 bytes")

    if results:
        s = results[0]
        print("\nCurrent usage (mem_sync):")
        print(
            f"  VGPRs: {s['vgpr']} → allows up to {512 // (((s['vgpr'] - 1) // 4 + 1) * 4)} waves"
        )
        print(
            f"  SGPRs: {s['sgpr']} → allows up to {800 // (((s['sgpr'] - 1) // 8 + 1) * 8)} waves"
        )
        print(
            f"  AGPRs: {s['agpr']} → allows up to {256 // (((s['agpr'] - 1) // 4 + 1) * 4) if s['agpr'] else 'unlimited'} waves"
        )
        print(
            f"  LDS: {s['lds_bytes']} → allows up to {163840 // s['lds_bytes']} workgroups/CU"
        )

        # Estimate for unrolled version
        print("\nEstimated for full unroll:")
        # Full unroll may increase VGPR by ~20-30% due to more live values
        est_vgpr = int(s["vgpr"] * 1.25)
        est_agpr = s["agpr"]  # AGPRs shouldn't change (same MFMA accumulator count)
        print(
            f"  Estimated VGPRs: {est_vgpr} → allows up to {512 // (((est_vgpr - 1) // 4 + 1) * 4)} waves"
        )
        print(
            f"  AGPRs: {est_agpr} → allows up to {256 // (((est_agpr - 1) // 4 + 1) * 4) if est_agpr else 'unlimited'} waves"
        )

        if 512 // (((est_vgpr - 1) // 4 + 1) * 4) < 6:
            print(
                "  ⚠️  WARNING: Estimated VGPR usage may reduce occupancy below 6 waves/CU"
            )
            print("      Consider partial unrolling instead of full unrolling")

    print("\n" + "=" * 60)
