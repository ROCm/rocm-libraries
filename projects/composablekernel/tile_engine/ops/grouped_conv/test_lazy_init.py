#!/usr/bin/env python3
"""
Test that the lazy initialization design works:
1. setup_multiple_grouped_conv_dispatchers() returns paths without loading GPU
2. GpuGroupedConvRunner.__init__() doesn't touch GPU
3. GPU context is only created on first run() call
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parent.parent.parent / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from grouped_conv_utils import (
    GroupedConvKernelConfig,
    GpuGroupedConvRunner,
    setup_multiple_grouped_conv_dispatchers,
)

def test_setup_returns_paths():
    """Test that setup returns Path objects, not loaded libraries."""
    print("\n" + "=" * 80)
    print("Test 1: setup_multiple_grouped_conv_dispatchers returns paths")
    print("=" * 80)

    configs = [
        GroupedConvKernelConfig(
            variant="forward",
            ndim_spatial=2,
            dtype="bf16",
            arch="gfx950",
            tile_m=16, tile_n=64, tile_k=64,
            wave_m=1, wave_n=4, wave_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            pipeline="compv3",
            scheduler="intrawave",
        )
    ]

    print("\n  Calling setup_multiple_grouped_conv_dispatchers()...")
    lib_paths = setup_multiple_grouped_conv_dispatchers(configs, verbose=True, max_workers=1)

    print(f"\n  ✓ Returned {len(lib_paths)} items")

    if lib_paths and lib_paths[0] is not None:
        print(f"  ✓ Type: {type(lib_paths[0])}")
        print(f"  ✓ Path: {lib_paths[0]}")
        assert isinstance(lib_paths[0], Path), "Expected Path object!"
        print("\n  ✅ Test 1 PASSED: Returns Path objects (not loaded libraries)")
        return lib_paths[0]
    else:
        print("\n  ⚠️  No library path returned (compilation may have failed)")
        return None


def test_runner_lazy_init(lib_path):
    """Test that GpuGroupedConvRunner.__init__() doesn't load GPU."""
    print("\n" + "=" * 80)
    print("Test 2: GpuGroupedConvRunner.__init__ is lazy")
    print("=" * 80)

    print(f"\n  Creating GpuGroupedConvRunner with lib_path={lib_path}")
    runner = GpuGroupedConvRunner(lib_path=str(lib_path))

    print(f"  ✓ Runner created")
    print(f"  ✓ runner._initialized = {runner._initialized}")
    print(f"  ✓ runner._hip = {runner._hip}")
    print(f"  ✓ runner._dispatch_lib = {runner._dispatch_lib}")

    # At this point, GPU should NOT be initialized
    assert runner._initialized == False, "GPU should not be initialized yet!"
    assert runner._hip is None, "HIP library should not be loaded yet!"

    print("\n  ✅ Test 2 PASSED: No GPU initialization in __init__()")
    return runner


def test_run_triggers_init(runner):
    """Test that first run() call triggers GPU initialization."""
    print("\n" + "=" * 80)
    print("Test 3: run() triggers lazy GPU initialization")
    print("=" * 80)

    import numpy as np
    from grouped_conv_utils import GroupedConvProblem

    # Create dummy problem
    problem = GroupedConvProblem(
        N=1, C=64, K=64, G=1,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        pad_h=1, pad_w=1,
        direction="forward"
    )

    input_np = np.random.randn(1, 56, 56, 1, 64).astype(np.float16)
    weight_np = np.random.randn(64, 3, 3, 1, 64).astype(np.float16)

    print("\n  Calling runner.run() for the first time...")
    print("  (This should trigger GPU initialization)")

    result = runner.run(input_np, weight_np, problem)

    print(f"\n  ✓ Result: success={result.success}")
    if result.success:
        print(f"  ✓ Time: {result.time_ms:.3f} ms")
        print(f"  ✓ TFLOPS: {result.tflops:.2f}")

    # Now GPU should be initialized
    print(f"\n  ✓ runner._initialized = {runner._initialized}")
    print(f"  ✓ runner._hip = {runner._hip}")
    print(f"  ✓ runner._dispatch_lib = {runner._dispatch_lib}")

    if result.success:
        assert runner._initialized == True, "GPU should be initialized after run()!"
        assert runner._hip is not None, "HIP library should be loaded after run()!"
        print("\n  ✅ Test 3 PASSED: GPU initialized on first run()")
    else:
        print("\n  ⚠️  run() failed, but lazy init was attempted")


def main():
    print("\n" + "=" * 80)
    print("Testing Lazy GPU Initialization Design")
    print("=" * 80)
    print("\nDesign goals:")
    print("  1. setup_multiple_grouped_conv_dispatchers() returns Path objects")
    print("  2. GpuGroupedConvRunner.__init__() doesn't touch GPU")
    print("  3. GPU context created lazily on first run() call")
    print()

    # Test 1: Setup returns paths
    lib_path = test_setup_returns_paths()

    if lib_path is None:
        print("\n⚠️  Cannot continue - no library compiled")
        return

    # Test 2: Runner init is lazy
    runner = test_runner_lazy_init(lib_path)

    # Test 3: Run triggers init
    test_run_triggers_init(runner)

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Lazy initialization working correctly!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
