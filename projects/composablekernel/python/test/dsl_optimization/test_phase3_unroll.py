#!/usr/bin/env python3
"""Test Phase 3: Unrolled loop lowering in CK DSL.

Validates that _lower_unrolled_for() correctly emits straight-line code.
"""
import sys
from pathlib import Path

# Add CK DSL to path
CKDSL_ROOT = Path("/workspace/rocm-libraries/projects/composablekernel/python")
if str(CKDSL_ROOT) not in sys.path:
    sys.path.insert(0, str(CKDSL_ROOT))

from ck_dsl.core.ir import IRBuilder, I32, F32
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm


def test_unrolled_lowering():
    """Test that unroll=True produces straight-line code."""
    print("Test 1: Unrolled loop lowering (5 iterations)")
    print("-" * 50)

    b = IRBuilder("test_unroll")

    # Simple loop: sum = 0+1+2+3+4
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

    result = for_op.results[0]

    # Lower to LLVM IR
    kernel = b.kernel
    llvm_ir = lower_kernel_to_llvm(kernel)

    print("LLVM IR generated:")
    print("-" * 50)
    # Show relevant parts
    lines = llvm_ir.split('\n')
    for i, line in enumerate(lines):
        if 'define' in line or 'iv_const' in line or 'add' in line or 'for.' in line or 'phi' in line:
            print(f"{i:3}: {line}")

    print("-" * 50)

    # Verify unrolling
    has_loop_header = "for.header" in llvm_ir
    has_loop_body = "for.body" in llvm_ir
    has_loop_latch = "for.latch" in llvm_ir

    # Count adds (should be 5 for 5 iterations)
    add_count = llvm_ir.count("add nsw i32")

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
    print("Test 2: Normal loop lowering")
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
    llvm_ir = lower_kernel_to_llvm(kernel)

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


def test_multiple_iter_args():
    """Test unrolled loop with multiple iteration arguments."""
    print("Test 3: Multiple iteration arguments")
    print("-" * 50)

    b = IRBuilder("multi_iter")

    # Loop with 2 iter args
    lower = b.const_i32(1)
    upper = b.const_i32(5)
    step = b.const_i32(1)
    init_sum = b.const_i32(0)
    init_prod = b.const_i32(1)

    for_op = b.scf_for_iter(
        lower, upper, step,
        [("sum", init_sum), ("prod", init_prod)],
        iv_name="i",
        unroll=True
    )

    with for_op as (i, [sum_var, prod_var]):
        new_sum = b.add(sum_var, i)
        new_prod = b.mul(prod_var, i)
        b.scf_yield(new_sum, new_prod)

    sum_result, prod_result = for_op.results

    # Lower to LLVM IR
    kernel = b.kernel
    llvm_ir = lower_kernel_to_llvm(kernel)

    # Verify unrolling
    has_loop = "for.header" in llvm_ir
    add_count = llvm_ir.count("add nsw i32")
    mul_count = llvm_ir.count("mul nsw i32")

    print(f"Analysis:")
    print(f"  Has loop structure: {has_loop}")
    print(f"  Add instructions: {add_count}")
    print(f"  Mul instructions: {mul_count}")

    assert not has_loop, "Should not have loop (unrolled)"
    assert add_count >= 4, f"Should have at least 4 adds, got {add_count}"
    assert mul_count >= 4, f"Should have at least 4 muls, got {mul_count}"

    print(f"\n✓ Multiple iter args handled correctly!")
    print()


def test_realistic_conv_loop():
    """Test realistic 72-iteration conv loop."""
    print("Test 4: Realistic conv K-loop (72 iterations)")
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
    llvm_ir = lower_kernel_to_llvm(kernel)

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


def test_non_constant_fallback():
    """Test that non-constant bounds fall back to normal loop."""
    print("Test 5: Non-constant bounds fallback")
    print("-" * 50)

    b = IRBuilder("fallback")

    # Non-constant upper bound (parameter)
    param_n = b.param("N", I32)
    lower = b.const_i32(0)
    upper = param_n  # NOT constant
    step = b.const_i32(1)
    init = b.const_i32(0)

    # Request unrolling, but should fall back to normal loop
    for_op = b.scf_for_iter(
        lower, upper, step,
        [("acc", init)],
        unroll=True  # Request unroll, but can't
    )

    with for_op as (i, [acc]):
        new_acc = b.add(acc, i)
        b.scf_yield(new_acc)

    result = for_op.results[0]

    # Lower to LLVM IR
    kernel = b.kernel
    llvm_ir = lower_kernel_to_llvm(kernel)

    # Should fall back to normal loop
    has_loop = "for.header" in llvm_ir
    has_phi = "phi i32" in llvm_ir

    print(f"Analysis:")
    print(f"  Has loop structure: {has_loop}")
    print(f"  Has phi nodes: {has_phi}")

    assert has_loop, "Should have loop (fallback)"
    assert has_phi, "Should have phi nodes (fallback)"

    print(f"\n✓ Correctly fell back to normal loop lowering!")
    print()


def main():
    print("=" * 70)
    print("CK DSL Phase 3: Unrolled Lowering Test")
    print("=" * 70)
    print()

    test_unrolled_lowering()
    test_normal_lowering()
    test_multiple_iter_args()
    test_realistic_conv_loop()
    test_non_constant_fallback()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print()
    print("Phase 3 implementation verified:")
    print("  - Simple loops unroll correctly (no control flow)")
    print("  - Normal loops preserve loop structure")
    print("  - Multiple iteration arguments handled")
    print("  - Realistic 72-iteration conv loop unrolls")
    print("  - Non-constant bounds gracefully fall back to normal loop")
    print()
    print("Ready for integration testing with actual conv kernel!")


if __name__ == "__main__":
    main()
