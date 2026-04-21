#!/usr/bin/env python3
"""
Test the runtime intrinsic validation system (Option B2).

Verifies:
  1. Intrinsics loaded automatically from .st.bc at import time
  2. Arguments can be specified in ANY order
  3. Automatic reordering to match signature
  4. Validation of missing/extra arguments
  5. Mixed types (registers + literals)
"""

import os
import sys
import pytest

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), "../../build/lib")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import stinkytofu as st
except ImportError:
    pytest.skip("StinkyTofu Python module not built", allow_module_level=True)


def test_intrinsics_loaded():
    """Test that intrinsics are automatically loaded from .st.bc."""
    intrinsics = st.list_intrinsics()
    assert len(intrinsics) > 0, "No intrinsics loaded"
    assert "ReluF32" in intrinsics
    assert "ClampF32" in intrinsics
    print(f"\n? Loaded {len(intrinsics)} intrinsics: {intrinsics}")


def test_get_signatures():
    """Test getting intrinsic signatures."""
    # ReluF32 has a simple 2-argument signature
    sig = st.get_intrinsic_signature("ReluF32")
    assert sig == ["dest", "src"]

    info = st.get_intrinsic_info("ReluF32")
    assert info is not None
    assert "signature" in info
    assert "comment" in info

    # ExpF32 has a temp register
    exp_sig = st.get_intrinsic_signature("ExpF32")
    assert exp_sig == ["dest", "src", "temp"]

    print("\n? Signatures accessible")


def test_order_independent():
    """Test that argument order doesn't matter."""
    v0, v1, v2 = st.vgpr(0), st.vgpr(1), st.vgpr(2)

    # Use ExpF32 which has dest, src, temp
    # Different orders - all should work
    r1 = st.Intrinsic("ExpF32", dest=v0, src=v1, temp=v2)
    r2 = st.Intrinsic("ExpF32", temp=v2, src=v1, dest=v0)
    r3 = st.Intrinsic("ExpF32", src=v1, temp=v2, dest=v0)

    assert r1.function_name == "ExpF32"
    assert r2.function_name == "ExpF32"
    assert r3.function_name == "ExpF32"

    print("\n? Order-independent arguments work!")


def test_mixed_types():
    """Test intrinsics with mixed argument types (registers + literals)."""
    v0, v1, v2 = st.vgpr(0), st.vgpr(1), st.vgpr(2)

    # ClampF32 with float literals
    clamp = st.Intrinsic("ClampF32", dest=v0, src=v1, min_val=0.0, max_val=1.0, temp=v2)

    assert clamp.function_name == "ClampF32"
    print("\n? Mixed types (registers + float literals) work!")


def test_missing_argument():
    """Test error handling for missing arguments."""
    v0, v1 = st.vgpr(0), st.vgpr(1)

    # ExpF32 requires dest, src, temp - test with missing temp
    with pytest.raises(ValueError) as exc_info:
        st.Intrinsic("ExpF32", dest=v0, src=v1)  # Missing 'temp'

    assert "missing" in str(exc_info.value).lower()
    assert "temp" in str(exc_info.value)
    print("\n? Missing argument validation works!")


def test_extra_argument():
    """Test error handling for extra arguments."""
    v0, v1, v2 = st.vgpr(0), st.vgpr(1), st.vgpr(2)

    # ReluF32 only takes dest and src - test with extra temp
    with pytest.raises(ValueError) as exc_info:
        st.Intrinsic("ReluF32", dest=v0, src=v1, extra=v2)

    assert "unexpected" in str(exc_info.value).lower()
    print("\n? Extra argument validation works!")


def test_unknown_intrinsic():
    """Test error handling for unknown intrinsics."""
    v0 = st.vgpr(0)

    with pytest.raises(ValueError) as exc_info:
        st.Intrinsic("UnknownOp", dest=v0)

    assert "unknown" in str(exc_info.value).lower()
    print("\n? Unknown intrinsic validation works!")


def test_module_integration():
    """Test that intrinsics work in a complete module."""
    module = st.LogicalModule("test_kernel")

    v0, v1, v2 = st.vgpr(0), st.vgpr(1), st.vgpr(2)

    # Add regular instruction
    module.add(st.VAddF32(dest=v0, src0=v1, src1=v2))

    # Add intrinsic with reordered args (using ReluF32 which only needs dest + src)
    module.add(st.Intrinsic("ReluF32", src=v1, dest=v0))

    dump = module.dump()
    assert "VAddF32" in dump
    assert "IntrinsicCall" in dump
    assert "ReluF32" in dump

    print("\n? Module integration works!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
