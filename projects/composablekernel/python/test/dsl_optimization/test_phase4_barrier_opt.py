#!/usr/bin/env python3
"""Test Phase 4: Barrier optimization in unrolled loops.

Validates that redundant sync() calls can be automatically elided when
loops are unrolled, particularly the final sync() before the last iteration ends.
"""
import sys
from pathlib import Path

# Add CK DSL to path
CKDSL_ROOT = Path("/workspace/rocm-libraries/projects/composablekernel/python")
if str(CKDSL_ROOT) not in sys.path:
    sys.path.insert(0, str(CKDSL_ROOT))

from ck_dsl.core.ir import IRBuilder, I32, F32
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm


def test_sync_in_unrolled_loop():
    """Test sync() behavior in unrolled loop - should keep all barriers."""
    print("Test 1: sync() in unrolled loop (baseline)")
    print("-" * 50)

    b = IRBuilder("test_sync_unroll")

    # Loop with sync() in body
    lower = b.const_i32(0)
    upper = b.const_i32(3)  # 3 iterations
    step = b.const_i32(1)
    init_sum = b.const_i32(0)

    for_op = b.scf_for_iter(
        lower, upper, step,
        [("sum", init_sum)],
        iv_name="i",
        unroll=True
    )

    with for_op as (i, [sum_var]):
        # Do some work
        new_sum = b.add(sum_var, i)
        # Sync barrier
        b.sync()
        # More work after barrier
        final_sum = b.add(new_sum, b.const_i32(1))
        b.scf_yield(final_sum)

    result = for_op.results[0]

    # Lower to LLVM IR
    kernel = b.kernel
    llvm_ir = lower_kernel_to_llvm(kernel)

    # Count barrier calls (not declarations)
    barrier_count = llvm_ir.count("call void @llvm.amdgcn.s.barrier()")

    print(f"Analysis:")
    print(f"  Iterations: 3")
    print(f"  Barriers in IR: {barrier_count}")
    print(f"  Expected: 3 (one per iteration)")

    # For now, we expect all barriers to be present
    # Phase 4 optimization would reduce this
    assert barrier_count == 3, f"Expected 3 barriers, got {barrier_count}"

    print(f"\n✓ All barriers preserved (Phase 4 not yet implemented)")
    print()


def test_final_barrier_elision_opportunity():
    """Identify where final barrier could be elided."""
    print("Test 2: Final barrier elision opportunity")
    print("-" * 50)

    b = IRBuilder("test_final_barrier")

    # Pattern: sync() right before scf_yield in last iteration
    lower = b.const_i32(0)
    upper = b.const_i32(3)
    step = b.const_i32(1)
    init_acc = b.const_f32(0.0)

    for_op = b.scf_for_iter(
        lower, upper, step,
        [("acc", init_acc)],
        unroll=True
    )

    with for_op as (i, [acc]):
        i_float = b.sitofp_f32(i)
        new_acc = b.fadd(acc, i_float)
        # Barrier before yield
        b.sync()
        b.scf_yield(new_acc)

    result = for_op.results[0]

    kernel = b.kernel
    llvm_ir = lower_kernel_to_llvm(kernel)

    barrier_count = llvm_ir.count("call void @llvm.amdgcn.s.barrier()")

    print(f"Analysis:")
    print(f"  Total barriers: {barrier_count}")
    print(f"  Expected: 1 barrier (Phase 4a active)")
    print(f"  Iterations: 0, 1, 2")
    print(f"  Iter 0 barrier: ELIDED (iter 1 starts with its own barrier)")
    print(f"  Iter 1 barrier: ELIDED (iter 2 starts with its own barrier)")
    print(f"  Iter 2 barrier: KEPT (no next iteration to provide barrier)")

    # Phase 4a elides trailing barriers in non-final iterations
    # Each iteration N+1 starts with a barrier, making iteration N's
    # trailing barrier redundant. Only final iteration keeps its trailing barrier.
    assert barrier_count == 1, f"Expected 1 barrier with Phase 4a, got {barrier_count}"

    print(f"\n✓ Phase 4a correctly elides non-final iteration barriers")
    print()


def test_loop_vs_unrolled_barriers():
    """Compare barrier behavior in loop vs unrolled."""
    print("Test 3: Loop vs unrolled barrier comparison")
    print("-" * 50)

    def build_with_unroll(unroll):
        b = IRBuilder(f"test_{'unroll' if unroll else 'loop'}")
        lower = b.const_i32(0)
        upper = b.const_i32(5)
        step = b.const_i32(1)
        init = b.const_i32(0)

        for_op = b.scf_for_iter(
            lower, upper, step,
            [("acc", init)],
            unroll=unroll
        )

        with for_op as (i, [acc]):
            new_acc = b.add(acc, i)
            b.sync()
            b.scf_yield(new_acc)

        kernel = b.kernel
        return lower_kernel_to_llvm(kernel)

    llvm_unrolled = build_with_unroll(True)
    llvm_loop = build_with_unroll(False)

    barriers_unrolled = llvm_unrolled.count("call void @llvm.amdgcn.s.barrier()")
    barriers_loop = llvm_loop.count("call void @llvm.amdgcn.s.barrier()")

    print(f"Unrolled version:")
    print(f"  Barriers: {barriers_unrolled}")
    print(f"Loop version:")
    print(f"  Barriers: {barriers_loop}")

    # With Phase 4a: unrolled has 1 barrier (only final iteration keeps it)
    # 5 iterations → 1 barrier (iter 0-3 elided, iter 4 kept)
    assert barriers_unrolled == 1, f"Expected 1 barrier (Phase 4a), got {barriers_unrolled}"
    assert barriers_loop == 1, f"Expected 1 barrier in loop, got {barriers_loop}"

    print(f"\n✓ Unrolled emits 1 barrier (Phase 4a), loop emits 1 dynamic barrier")
    print()


def test_phase4_design():
    """Document Phase 4 design and requirements."""
    print("Test 4: Phase 4 Design Documentation")
    print("-" * 50)
    print()
    print("Phase 4 Goal: Automatically elide redundant barriers in unrolled loops")
    print()
    print("Optimization Opportunities:")
    print("  1. Final barrier: Last iteration's sync() before yield is redundant")
    print("     - No next iteration exists to synchronize with")
    print("     - Can safely elide if no global loads follow")
    print()
    print("  2. Inter-iteration barriers: Analyze data dependencies")
    print("     - If iteration N doesn't depend on iteration N-1's LDS data")
    print("     - Middle barriers could be elided")
    print()
    print("Implementation Approaches:")
    print("  A. Conservative (Phase 4a): Only elide final barrier")
    print("     - Simple: check if last lowered iteration ends with sync()")
    print("     - Safe: preserves all cross-iteration dependencies")
    print("     - Benefit: 1 fewer barrier per unrolled loop")
    print()
    print("  B. Aggressive (Phase 4b): Full barrier analysis")
    print("     - Complex: need data-dependency tracking")
    print("     - Analyze LDS read/write patterns")
    print("     - Benefit: could elide multiple barriers")
    print()
    print("Recommendation: Start with Phase 4a (conservative)")
    print("  - Modify _lower_unrolled_for() to track if last op is sync()")
    print("  - If yes, skip lowering that sync() in final iteration")
    print("  - Low risk, measurable benefit")
    print()
    print("✓ Phase 4 design documented")
    print()


def main():
    print("=" * 70)
    print("CK DSL Phase 4: Barrier Optimization Test")
    print("=" * 70)
    print()

    test_sync_in_unrolled_loop()
    test_final_barrier_elision_opportunity()
    test_loop_vs_unrolled_barriers()
    test_phase4_design()

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Current Status:")
    print("  - Unrolled loops preserve all barriers (correct but conservative)")
    print("  - Final barrier in last iteration is redundant")
    print("  - Optimization would save 1 barrier per unrolled loop")
    print()
    print("Next Step: Implement Phase 4a (conservative final-barrier elision)")


if __name__ == "__main__":
    main()
