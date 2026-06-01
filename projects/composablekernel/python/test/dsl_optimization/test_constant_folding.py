#!/usr/bin/env python3
"""Test constant folding helpers in CK DSL lowering pass.

This tests Phase 2 implementation: _is_constant() and _eval_constant().
"""
import sys
from pathlib import Path

# Add CK DSL to path
CKDSL_ROOT = Path("/workspace/rocm-libraries/projects/composablekernel/python")
if str(CKDSL_ROOT) not in sys.path:
    sys.path.insert(0, str(CKDSL_ROOT))

from ck_dsl.core.ir import IRBuilder, I32, F32
from ck_dsl.core.lower_llvm import _Lowerer


def test_is_constant():
    """Test _is_constant() helper function."""
    print("Test 1: _is_constant() detection")
    print("-" * 50)

    b = IRBuilder("test_kernel")

    # Create constants
    c0 = b.const_i32(0)
    c10 = b.const_i32(10)
    c_step = b.const_i32(1)
    f_const = b.const_f32(3.14)

    # Create non-constant (parameter)
    param = b.param("x", I32)

    # Create lowerer
    kernel = b.kernel
    lowerer = _Lowerer(kernel)

    # Test constant detection
    assert lowerer._is_constant(c0), "Should detect c0 as constant"
    assert lowerer._is_constant(c10), "Should detect c10 as constant"
    assert lowerer._is_constant(c_step), "Should detect c_step as constant"
    assert lowerer._is_constant(f_const), "Should detect f_const as constant"

    # Test non-constant detection
    assert not lowerer._is_constant(param), "Should detect param as non-constant"

    print("  ✓ Constant detection works correctly")
    print()


def test_eval_constant():
    """Test _eval_constant() helper function."""
    print("Test 2: _eval_constant() evaluation")
    print("-" * 50)

    b = IRBuilder("test_kernel")

    # Create various constants
    c0 = b.const_i32(0)
    c72 = b.const_i32(72)
    c_neg = b.const_i32(-5)

    # Create lowerer
    kernel = b.kernel
    lowerer = _Lowerer(kernel)

    # Test constant evaluation
    assert lowerer._eval_constant(c0) == 0, "Should evaluate c0 to 0"
    assert lowerer._eval_constant(c72) == 72, "Should evaluate c72 to 72"
    assert lowerer._eval_constant(c_neg) == -5, "Should evaluate c_neg to -5"

    print(f"  ✓ Evaluated c0 = {lowerer._eval_constant(c0)}")
    print(f"  ✓ Evaluated c72 = {lowerer._eval_constant(c72)}")
    print(f"  ✓ Evaluated c_neg = {lowerer._eval_constant(c_neg)}")
    print()


def test_eval_constant_error():
    """Test _eval_constant() raises error on non-constant."""
    print("Test 3: _eval_constant() error handling")
    print("-" * 50)

    b = IRBuilder("test_kernel")

    # Create non-constant
    param = b.param("x", I32)

    # Create lowerer
    kernel = b.kernel
    lowerer = _Lowerer(kernel)

    # Test error on non-constant
    try:
        lowerer._eval_constant(param)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    print()


def test_loop_bounds_detection():
    """Test detection of constant loop bounds."""
    print("Test 4: Loop bounds constant detection")
    print("-" * 50)

    b = IRBuilder("test_kernel")

    # Case 1: Constant bounds (unrollable)
    lower_const = b.const_i32(0)
    upper_const = b.const_i32(72)
    step_const = b.const_i32(8)

    # Case 2: Non-constant bounds (not unrollable)
    param_n = b.param("N", I32)
    lower_var = b.const_i32(0)
    upper_var = param_n
    step_var = b.const_i32(1)

    # Create lowerer
    kernel = b.kernel
    lowerer = _Lowerer(kernel)

    # Test Case 1: Constant bounds
    is_unrollable_1 = (
        lowerer._is_constant(lower_const) and
        lowerer._is_constant(upper_const) and
        lowerer._is_constant(step_const)
    )
    assert is_unrollable_1, "Should detect constant bounds as unrollable"

    if is_unrollable_1:
        lower_val = lowerer._eval_constant(lower_const)
        upper_val = lowerer._eval_constant(upper_const)
        step_val = lowerer._eval_constant(step_const)
        trip_count = (upper_val - lower_val) // step_val

        print(f"  ✓ Case 1 (constant bounds): Unrollable")
        print(f"    lower={lower_val}, upper={upper_val}, step={step_val}")
        print(f"    trip_count={trip_count} iterations")

    # Test Case 2: Non-constant bounds
    is_unrollable_2 = (
        lowerer._is_constant(lower_var) and
        lowerer._is_constant(upper_var) and
        lowerer._is_constant(step_var)
    )
    assert not is_unrollable_2, "Should detect non-constant bounds as not unrollable"
    print(f"  ✓ Case 2 (non-constant bounds): Not unrollable")

    print()


def test_conv_k_loop_bounds():
    """Test realistic K-loop bounds from conv kernel."""
    print("Test 5: Conv K-loop bounds (realistic)")
    print("-" * 50)

    b = IRBuilder("conv_kernel")

    # ResNet50 conv3_1: K_gemm = 4608, block_k = 64
    # K-loop: for (k=0; k<4608; k+=64)
    K_gemm = 4608
    block_k = 64

    lower = b.const_i32(0)
    upper = b.const_i32(K_gemm)
    step = b.const_i32(block_k)

    # Create lowerer
    kernel = b.kernel
    lowerer = _Lowerer(kernel)

    # Check if unrollable
    assert lowerer._is_constant(lower), "lower should be constant"
    assert lowerer._is_constant(upper), "upper should be constant"
    assert lowerer._is_constant(step), "step should be constant"

    # Calculate trip count
    lower_val = lowerer._eval_constant(lower)
    upper_val = lowerer._eval_constant(upper)
    step_val = lowerer._eval_constant(step)
    trip_count = (upper_val - lower_val) // step_val

    print(f"  ✓ K-loop is unrollable:")
    print(f"    K_gemm={upper_val}, block_k={step_val}")
    print(f"    Trip count: {trip_count} iterations")
    print(f"    Total MFMAs (8 per iter): {trip_count * 8}")

    assert trip_count == 72, f"Expected 72 iterations, got {trip_count}"

    print()


def main():
    print("=" * 70)
    print("CK DSL Phase 2: Constant Folding Helpers Test")
    print("=" * 70)
    print()

    test_is_constant()
    test_eval_constant()
    test_eval_constant_error()
    test_loop_bounds_detection()
    test_conv_k_loop_bounds()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print()
    print("Phase 2 implementation verified:")
    print("  - _is_constant() correctly detects compile-time constants")
    print("  - _eval_constant() correctly evaluates constant values")
    print("  - Loop bounds detection works for both constant and variable cases")
    print("  - Ready for Phase 3: Unrolled lowering")


if __name__ == "__main__":
    main()
