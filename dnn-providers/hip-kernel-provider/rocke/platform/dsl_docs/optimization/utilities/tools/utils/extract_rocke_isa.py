# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Extract ISA for CK DSL mem_sync vs mem_async to debug the 36% performance loss."""

import sys
import subprocess
from pathlib import Path

# Add CK DSL to path — resolve relative to this file so no absolute paths needed.
ROCKE_ROOT = Path(__file__).resolve().parents[7]  # .../python
if str(ROCKE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROCKE_ROOT))

from rocke.helpers import compile_kernel  # noqa: E402
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

OUT_DIR = Path(__file__).resolve().parents[1] / "comparison" / "rocke_isa"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_isa(name, pipeline, async_dma, unroll_k=False):
    """Build kernel and extract ISA."""
    print(f"\n{'=' * 60}")
    print(f"Extracting: {name}")
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
        unroll_k=unroll_k,
    )

    # Compile
    print("Compiling...")
    kernel = build_implicit_gemm_conv(spec)
    artifact = compile_kernel(
        kernel, isa="amdgcn-amd-amdhsa--gfx950", capture_ir_text=True
    )
    print(f"  Kernel: {artifact.kernel_name}")

    # Save MLIR IR
    ir_path = OUT_DIR / f"{name}.mlir"
    with open(ir_path, "w") as f:
        f.write(artifact.ir_text)
    print(f"  Saved MLIR: {ir_path}")

    # Save LLVM IR
    llvm_path = OUT_DIR / f"{name}.ll"
    with open(llvm_path, "w") as f:
        f.write(artifact.llvm_text)
    print(f"  Saved LLVM: {llvm_path}")

    # Save HSACO
    hsaco_path = OUT_DIR / f"{name}.hsaco"
    with open(hsaco_path, "wb") as f:
        f.write(artifact.hsaco)
    print(f"  Saved HSACO: {hsaco_path}")

    # Extract ISA using llvm-objdump directly on HSACO
    print("  Extracting ISA...")
    isa_path = OUT_DIR / f"{name}.s"
    result = subprocess.run(
        ["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--mcpu=gfx950", str(hsaco_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    with open(isa_path, "w") as f:
        f.write(result.stdout)
    print(f"  Saved ISA: {isa_path}")

    return ir_path, llvm_path, isa_path


if __name__ == "__main__":
    # Extract both modes
    sync_files = extract_isa("mem_sync", "mem", async_dma=False, unroll_k=False)
    unroll_files = extract_isa("mem_unroll", "mem", async_dma=False, unroll_k=True)

    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {OUT_DIR}")
    print("Files:")
    for f in OUT_DIR.glob("*"):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:30s} {size_kb:8.1f} KB")
