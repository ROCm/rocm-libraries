#!/usr/bin/env python3
"""Simplified test for unrolled loop lowering in CK DSL.

Tests Phase 3 implementation without complex IR builder APIs.
"""
import sys
from pathlib import Path

# Add CK DSL to path
CKDSL_ROOT = Path("/workspace/rocm-libraries/projects/composablekernel/python")
if str(CKDSL_ROOT) not in sys.path:
    sys.path.insert(0, str(CKDSL_ROOT))

from ck_dsl.core.ir import IRBuilder, I32, F32
from ck_dsl.helpers import compile_kernel


def test_unrolled_lowering():
    """Test that unroll=True produces straight-line code."""
    print("Test: Unrolled loop lowering")
    print("-" * 50)

    b = IRBuilder("test_unroll")

    # Simple loop: accumulate 0+1+2+3+4
    lower = b.const_i32(0)
    upper = b.const_i32(5)
    step = b.const_i32(1)
    init_sum = b.const_i32(0)

    # Create UNROLLED loop
    for_op = b.scf_for_iter(
        lower, upper, step,
        [("sum", init_sum)],
        iv_name="i",
        unroll=True  # Request unrolling
    )

    with for_op as (i, [sum_var]):
        new_sum = b.add(sum_var, i)
        b.scf_yield(new_sum)

    # Get result
    result = for_op.results[0]

    # Lower to LLVM IR
    kernel = b.kernel
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
    llvm_ir = artifact.llvm_text

    print("LLVM IR generated:")
    print("-" * 50)
    # Show relevant parts
    lines = llvm_ir.split('\n')
    for i, line in enumerate(lines):
        if 'define' in line or 'iv_const' in line or 'add i32' in line or 'for.' in line or 'phi' in line:
            print(f"{i:3}: {line}")

    print("-" * 50)

    # Verify unrolling
    has_loop_header = "for.header" in llvm_ir
    has_loop_body = "for.body" in llvm_ir
    has_loop_latch = "for.latch" in llvm_ir

    # Count adds (should be 5 for 5 iterations)
    add_count = llvm_ir.count("add i32")

    print(f"\nAnalysis:")
    print(f"  Has loop header: {has_loop_header}")
    print(f"  Has loop body:   {has_loop_body}")
    print(f"  Has loop latch:  {has_loop_latch}")
    print(f"  Add instructions: {add_count}")

    assert not has_loop_header, "Should not have loop header (unrolled)"
    assert not has_loop_body, "Should not have loop body label (unrolled)"
    assert not has_loop_latch, "Should not have loop latch (unrolled)"
    assert add_count >= 5, f"Should have at least 5 adds, got {add_count}"

    print(f"\n✓ Loop is fully unrolled!")
    print()


def test_normal_lowering():
    """Test that unroll=False produces loop structure."""
    print("Test: Normal loop lowering")
    print("-" * 50)

    b = IRBuilder("test_normal")

    # Same loop, but NOT unrolled
    lower = b.const_i32(0)
    upper = b.const_i32(5)
    step = b.const_i32(1)
    init_sum = b.const_i32(0)

    for_op = b.scf_for_iter(
        lower, upper, step,
        [("sum", init_sum)],
        iv_name="i",
        unroll=False  # Normal loop
    )

    with for_op as (i, [sum_var]):
        new_sum = b.add(sum_var, i)
        b.scf_yield(new_sum)

    result = for_op.results[0]

    # Lower to LLVM IR
    kernel = b.kernel
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
    llvm_ir = artifact.llvm_text

    # Verify loop structure
    has_loop_header = "for.header" in llvm_ir
    has_phi = "phi i32" in llvm_ir

    print(f"Analysis:")
    print(f"  Has loop header: {has_loop_header}")
    print(f"  Has phi nodes:   {has_phi}")

    assert has_loop_header, "Should have loop header (normal)"
    assert has_phi, "Should have phi nodes (normal)"

    print(f"\n✓ Normal loop has control flow!")
    print()


def test_realistic_conv_loop():
    """Test realistic 72-iteration conv loop."""
    print("Test: Realistic conv K-loop (72 iterations)")
    print("-" * 50)

    b = IRBuilder("conv_k_loop")

    # 72 iterations like ResNet50 conv3_1
    lower = b.const_i32(0)
    upper = b.const_i32(72)
    step = b.const_i32(1)
    init_acc = b.const_f32(0.0)

    for_op = b.scf_for_iter(
        lower, upper, step,
        [("acc", init_acc)],
        iv_name="k",
        unroll=True
    )

    with for_op as (k, [acc]):
        # Simplified accumulation
        k_float = b.sitofp_f32(k)
        new_acc = b.fadd(acc, k_float)
        b.scf_yield(new_acc)

    result = for_op.results[0]

    # Lower to LLVM IR
    kernel = b.kernel
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
    llvm_ir = artifact.llvm_text

    # Verify unrolling
    has_loop = "for.header" in llvm_ir
    fadd_count = llvm_ir.count("fadd ")

    print(f"Analysis:")
    print(f"  Has loop structure: {has_loop}")
    print(f"  fadd instructions: {fadd_count}")
    print(f"  LLVM IR size: {len(llvm_ir)} bytes")

    assert not has_loop, "Should not have loop (unrolled)"
    assert fadd_count >= 70, f"Should have ~72 fadds, got {fadd_count}"

    print(f"\n✓ 72 iterations fully unrolled!")
    print()


def main():
    print("=" * 70)
    print("CK DSL Phase 3: Unrolled Lowering Test (Simplified)")
    print("=" * 70)
    print()

    test_unrolled_lowering()
    test_normal_lowering()
    test_realistic_conv_loop()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print()
    print("Phase 3 implementation verified:")
    print("  - Unrolled loops produce straight-line code (no control flow)")
    print("  - Normal loops preserve loop structure")
    print("  - Realistic 72-iteration conv loop unrolls successfully")


if __name__ == "__main__":
    main()
