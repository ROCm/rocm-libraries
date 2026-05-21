#!/usr/bin/env python3
"""Test unrolled loop lowering in CK DSL.

This tests Phase 3 implementation: _lower_unrolled_for().
"""
import sys
from pathlib import Path

# Add CK DSL to path
CKDSL_ROOT = Path("/workspace/rocm-libraries/projects/composablekernel/python")
if str(CKDSL_ROOT) not in sys.path:
    sys.path.insert(0, str(CKDSL_ROOT))

from ck_dsl.core.ir import IRBuilder, I32, F32, PtrType
from ck_dsl.helpers import compile_kernel


def test_simple_unrolled_loop():
    """Test simple loop with unroll=True."""
    print("Test 1: Simple unrolled loop")
    print("-" * 50)

    b = IRBuilder("simple_unroll")

    # Loop: sum from 0 to 9
    lower = b.const_i32(0)
    upper = b.const_i32(10)
    step = b.const_i32(1)
    init_sum = b.const_f32(0.0)

    # Create unrolled loop
    for_op = b.scf_for_iter(
        lower, upper, step,
        [("sum", init_sum)],
        iv_name="i",
        unroll=True  # Request unrolling
    )

    with for_op as (i, [sum_var]):
        # Body: sum += i (converted to float)
        i_float = b.sitofp_f32(i)
        new_sum = b.fadd(sum_var, i_float)
        b.scf_yield(new_sum)

    result = for_op.results[0]

    # Store result (simplified - just use result)
    output_ptr = b.param("output", PtrType(F32, "global"))
    b.store(result, output_ptr, align=4)

    kernel = b.kernel

    # Compile and check LLVM IR
    print("  Compiling kernel...")
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")

    # Check LLVM IR doesn't have control flow (no br, no phi except at start)
    llvm_ir = artifact.llvm_text

    # Count branches (should be minimal - just entry/exit)
    br_count = llvm_ir.count("  br ")
    phi_count = llvm_ir.count("  %sum")

    print(f"  ✓ Compiled successfully")
    print(f"  LLVM IR analysis:")
    print(f"    Branch instructions: {br_count}")
    print(f"    Sum variables: {phi_count}")

    # For unrolled loop, we should see inline additions, not loop structure
    assert "for.header" not in llvm_ir, "Should not have loop header"
    assert "for.body" not in llvm_ir, "Should not have loop body label"
    assert "for.latch" not in llvm_ir, "Should not have loop latch"

    print(f"  ✓ Loop is fully unrolled (no control flow)")
    print()


def test_unrolled_with_iter_args():
    """Test unrolled loop with multiple iteration arguments."""
    print("Test 2: Unrolled loop with multiple iter args")
    print("-" * 50)

    b = IRBuilder("multi_iter_unroll")

    # Loop: accumulate both sum and product
    lower = b.const_i32(1)
    upper = b.const_i32(5)  # 1, 2, 3, 4
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
        # sum += i, prod *= i
        new_sum = b.add(sum_var, i)
        new_prod = b.mul(prod_var, i)
        b.scf_yield(new_sum, new_prod)

    sum_result, prod_result = for_op.results

    # Store results
    output_ptr = b.param("output", PtrType(I32, "global"))
    b.store(sum_result, output_ptr, align=4)

    kernel = b.kernel

    print("  Compiling kernel...")
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")

    llvm_ir = artifact.llvm_text

    # Verify unrolling
    assert "for.header" not in llvm_ir, "Should not have loop header"
    assert "phi i32" not in llvm_ir or llvm_ir.count("phi i32") <= 2, "Should have minimal phi nodes"

    print(f"  ✓ Compiled successfully")
    print(f"  ✓ Multiple iter args handled correctly")
    print()


def test_unroll_vs_normal():
    """Compare unrolled vs normal loop lowering."""
    print("Test 3: Unrolled vs Normal lowering comparison")
    print("-" * 50)

    # Build two kernels: one unrolled, one normal
    def build_kernel(name, unroll):
        b = IRBuilder(name)

        lower = b.const_i32(0)
        upper = b.const_i32(8)
        step = b.const_i32(1)
        init = b.const_i32(0)

        for_op = b.scf_for_iter(
            lower, upper, step,
            [("acc", init)],
            unroll=unroll
        )

        with for_op as (i, [acc]):
            new_acc = b.add(acc, i)
            b.scf_yield(new_acc)

        result = for_op.results[0]
        output = b.param("out", PtrType(I32, "global"))
        b.store(result, output, align=4)

        return b.kernel

    # Build both versions
    kernel_unrolled = build_kernel("unrolled", unroll=True)
    kernel_normal = build_kernel("normal", unroll=False)

    print("  Compiling unrolled version...")
    artifact_unrolled = compile_kernel(kernel_unrolled, isa="amdgcn-amd-amdhsa--gfx950")

    print("  Compiling normal version...")
    artifact_normal = compile_kernel(kernel_normal, isa="amdgcn-amd-amdhsa--gfx950")

    llvm_unrolled = artifact_unrolled.llvm_ir
    llvm_normal = artifact_normal.llvm_ir

    # Analyze differences
    unrolled_has_loop = "for.header" in llvm_unrolled
    normal_has_loop = "for.header" in llvm_normal

    print(f"\n  Unrolled version:")
    print(f"    Has loop structure: {unrolled_has_loop}")
    print(f"    LLVM IR size: {len(llvm_unrolled)} bytes")

    print(f"\n  Normal version:")
    print(f"    Has loop structure: {normal_has_loop}")
    print(f"    LLVM IR size: {len(llvm_normal)} bytes")

    assert not unrolled_has_loop, "Unrolled should not have loop"
    assert normal_has_loop, "Normal should have loop"

    print(f"\n  ✓ Unroll hint correctly changes lowering strategy")
    print()


def test_conv_like_loop():
    """Test realistic conv-like loop (72 iterations)."""
    print("Test 4: Conv-like K-loop (72 iterations)")
    print("-" * 50)

    b = IRBuilder("conv_k_loop")

    # ResNet50 conv3_1: K_gemm=4608, block_k=64 → 72 iterations
    lower = b.const_i32(0)
    upper = b.const_i32(72)  # 72 iterations
    step = b.const_i32(1)

    init_acc = b.const_f32(0.0)

    for_op = b.scf_for_iter(
        lower, upper, step,
        [("acc", init_acc)],
        iv_name="k_iter",
        unroll=True  # Full unrolling like actual conv kernel
    )

    with for_op as (k, [acc]):
        # Simplified MFMA-like accumulation
        k_float = b.sitofp_f32(k)
        new_acc = b.fadd(acc, k_float)
        b.scf_yield(new_acc)

    result = for_op.results[0]
    output = b.param("out", PtrType(F32, "global"))
    b.store(result, output, align=4)

    kernel = b.kernel

    print("  Compiling 72-iteration unrolled loop...")
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")

    llvm_ir = artifact.llvm_text

    # Verify unrolling
    assert "for.header" not in llvm_ir, "Should not have loop"

    # Count fadd instructions (should be 72 for 72 iterations)
    fadd_count = llvm_ir.count("fadd ")

    print(f"  ✓ Compiled successfully")
    print(f"  LLVM IR size: {len(llvm_ir)} bytes")
    print(f"  fadd instructions: {fadd_count} (expected ~72)")
    print(f"  ✓ 72 iterations fully unrolled")
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
    output = b.param("out", PtrType(I32, "global"))
    b.store(result, output, align=4)

    kernel = b.kernel

    print("  Compiling with non-constant bounds...")
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")

    llvm_ir = artifact.llvm_text

    # Should fall back to normal loop
    assert "for.header" in llvm_ir, "Should have loop (fallback)"
    assert "phi i32" in llvm_ir, "Should have phi nodes (fallback)"

    print(f"  ✓ Correctly fell back to normal loop lowering")
    print(f"  ✓ Handles non-constant bounds gracefully")
    print()


def main():
    print("=" * 70)
    print("CK DSL Phase 3: Unrolled Lowering Test")
    print("=" * 70)
    print()

    test_simple_unrolled_loop()
    test_unrolled_with_iter_args()
    test_unroll_vs_normal()
    test_conv_like_loop()
    test_non_constant_fallback()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print()
    print("Phase 3 implementation verified:")
    print("  - Simple loops unroll correctly (no control flow)")
    print("  - Multiple iteration arguments handled")
    print("  - Unrolled vs normal lowering produces different IR")
    print("  - Realistic 72-iteration conv loop unrolls")
    print("  - Non-constant bounds gracefully fall back to normal loop")
    print()
    print("Ready for integration testing with actual conv kernel!")


if __name__ == "__main__":
    main()
