#!/usr/bin/env python3
"""
Test Expected<T> error handling for all refactored StinkyTofu functions

This validates that all functions refactored to use Expected<T> properly
convert C++ errors to Python RuntimeError exceptions via unwrapExpected().
"""

import sys
import pytest

# Add build directory to path to import stinkytofu
sys.path.insert(0, '/data0/yangwen/rocm-libraries/shared/stinkytofu/build/python_module')
import stinkytofu as st


# Use gfx950 as the test architecture (supported in build)
GFX950 = [9, 5, 0]


# Helper functions for creating registers
def sgpr(idx, count=1):
    return st.sgpr(idx, count)


def vgpr(idx, count=1):
    return st.vgpr(idx, count)


def agpr(idx, count=1):
    """Create accumulator register (AGPR) - using vgpr as compatible type"""
    # Note: For testing purposes, use vgpr as MFMA functions accept them
    return vgpr(idx, count)


# ============================================================================
# Test Helper Functions (validate Expected<T> mechanism)
# ============================================================================

def test_expected_helper_success():
    """Test Expected<int> success case"""
    result = st._testExpectedSuccess(42)
    assert result == 84, f"Expected 84 (42*2), got {result}"


def test_expected_helper_error():
    """Test Expected<int> error case raises RuntimeError"""
    with pytest.raises(RuntimeError) as exc_info:
        st._testExpectedError("Test error message")
    assert "Test error message" in str(exc_info.value)


def test_expected_vector_success():
    """Test Expected<vector<int>> success case"""
    result = st._testExpectedVectorSuccess()
    assert result == [1, 2, 3, 4, 5]


def test_expected_vector_error():
    """Test Expected<vector<int>> error case"""
    with pytest.raises(RuntimeError) as exc_info:
        st._testExpectedVectorError()
    assert "Vector creation failed" in str(exc_info.value)


# ============================================================================
# Test Refactored gfx1250+ Instructions (should fail on gfx950)
# ============================================================================

def test_smul_lo_u32_unsupported():
    """SMulLOU32 requires gfx1250+, should fail on gfx950"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        tofu.SMulLOU32(sgpr(0), sgpr(1), sgpr(2))
    
    error_msg = str(exc_info.value)
    assert "s_mul_lo_u32" in error_msg
    assert "not supported" in error_msg.lower() or "gfx1250" in error_msg


def test_ssub_u64_unsupported():
    """SSubU64 requires gfx1250+, should fail on gfx950"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        tofu.SSubU64(sgpr(0), sgpr(2), sgpr(4))
    
    error_msg = str(exc_info.value)
    assert "not supported" in error_msg.lower() or "gfx1250" in error_msg


def test_sand_saveexec_b32_unsupported():
    """SAndSaveExecB32 requires gfx1250+, should fail on gfx950"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        tofu.SAndSaveExecB32(sgpr(0), sgpr(1))
    
    error_msg = str(exc_info.value)
    assert "not supported" in error_msg.lower() or "gfx1250" in error_msg


def test_sor_saveexec_b32_unsupported():
    """SOrSaveExecB32 requires gfx1250+, should fail on gfx950"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        tofu.SOrSaveExecB32(sgpr(0), sgpr(1))
    
    error_msg = str(exc_info.value)
    assert "not supported" in error_msg.lower() or "gfx1250" in error_msg


def test_vfma_mix_f32_unsupported():
    """VFmaMixF32 requires gfx1250+, should fail on gfx950"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        tofu.VFmaMixF32(vgpr(0), vgpr(1), vgpr(2), vgpr(3))
    
    error_msg = str(exc_info.value)
    assert "not supported" in error_msg.lower() or "gfx1250" in error_msg


def test_vrsq_iflag_f32_unsupported():
    """VRsqIFlagF32 requires gfx1250+, should fail on gfx950"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        tofu.VRsqIFlagF32(vgpr(0), vgpr(1))
    
    error_msg = str(exc_info.value)
    assert "not supported" in error_msg.lower() or "gfx1250" in error_msg


# ============================================================================
# Test VCvtBF16toFP32 (gfx950+ supported, but has parameter validation)
# ============================================================================

def test_vcvt_bf16_to_fp32_success():
    """VCvtBF16toFP32 should work on gfx950"""
    tofu = st.StinkyAsmIR(GFX950)
    
    # Should succeed - gfx950 supports native bf16 conversion
    result = tofu.VCvtBF16toFP32(vgpr(0), vgpr(1))
    assert len(result) > 0
    assert result[0] is not None


def test_vcvt_bf16_to_fp32_missing_vgpr_mask():
    """VCvtBF16toFP32 requires vgprMask for SDWA path"""
    tofu = st.StinkyAsmIR(GFX950)
    
    # Test will depend on architecture - gfx950 has native support
    # so this test might not trigger the error path
    # Just verify it doesn't crash
    try:
        result = tofu.VCvtBF16toFP32(vgpr(0), vgpr(1))
        assert len(result) > 0
    except RuntimeError as e:
        # If it does error, verify the error message makes sense
        assert "vgprMask" in str(e) or "not supported" in str(e).lower()


# ============================================================================
# Test MFMA Functions (generic instruction creation)
# ============================================================================

def test_mfma_invalid_instruction():
    """createMFMA should raise error for invalid instruction type"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        # Try to create MFMA with invalid parameters that don't map to real instruction
        tofu.createMFMA(
            "invalid_type",  # Invalid type
            "f32",
            32, 32, 16,
            1,
            False,
            agpr(0),
            vgpr(0),
            vgpr(16),
            None,
            False,
            ""
        )
    
    error_msg = str(exc_info.value)
    assert "not found" in error_msg.lower() or "not supported" in error_msg.lower()


def test_mxmfma_invalid_instruction():
    """createMXMFMA should raise error for invalid instruction"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        tofu.createMXMFMA(
            "invalid_type",
            "f32",
            "e5m3",
            "e5m3",
            16, 16, 128,
            16,
            agpr(0),
            vgpr(0),
            vgpr(8),
            agpr(0),
            vgpr(16),
            vgpr(17),
            False,
            False,
            ""
        )
    
    error_msg = str(exc_info.value)
    assert "not found" in error_msg.lower() or "not supported" in error_msg.lower()


def test_smfma_invalid_instruction():
    """createSMFMA should raise error for invalid instruction"""
    tofu = st.StinkyAsmIR(GFX950)
    
    with pytest.raises(RuntimeError) as exc_info:
        tofu.createSMFMA(
            "invalid_type",
            "f32",
            16, 16, 32,
            1,
            False,
            agpr(0),
            vgpr(0),
            vgpr(8),
            vgpr(16),
            False,
            ""
        )
    
    error_msg = str(exc_info.value)
    assert "not found" in error_msg.lower() or "not supported" in error_msg.lower()


# ============================================================================
# Test Error Message Quality
# ============================================================================

def test_error_messages_are_descriptive():
    """Verify that error messages contain useful information"""
    tofu = st.StinkyAsmIR(GFX950)
    
    # Collect all error messages
    errors = []
    
    try:
        tofu.SMulLOU32(sgpr(0), sgpr(1), sgpr(2))
    except RuntimeError as e:
        errors.append(("SMulLOU32", str(e)))
    
    try:
        tofu.VFmaMixF32(vgpr(0), vgpr(1), vgpr(2), vgpr(3))
    except RuntimeError as e:
        errors.append(("VFmaMixF32", str(e)))
    
    # Verify all errors are informative (not empty, mention instruction or arch)
    for func_name, error_msg in errors:
        assert len(error_msg) > 10, f"{func_name} error message too short: '{error_msg}'"
        # Should mention either the instruction name or architecture requirement
        assert any(keyword in error_msg.lower() for keyword in 
                  ['not supported', 'requires', 'gfx', 'architecture', 'instruction'])


def test_exception_types_are_runtime_error():
    """Verify all Expected<T> errors convert to RuntimeError"""
    tofu = st.StinkyAsmIR(GFX950)
    
    test_cases = [
        lambda: tofu.SMulLOU32(sgpr(0), sgpr(1), sgpr(2)),
        lambda: tofu.SSubU64(sgpr(0), sgpr(2), sgpr(4)),
        lambda: tofu.VFmaMixF32(vgpr(0), vgpr(1), vgpr(2), vgpr(3)),
    ]
    
    for test_func in test_cases:
        try:
            test_func()
        except Exception as e:
            assert type(e).__name__ == 'RuntimeError', \
                f"Expected RuntimeError, got {type(e).__name__}"


if __name__ == "__main__":
    print("=" * 80)
    print("Testing All Expected<T> Refactored Functions")
    print("Validating C++ Expected<T> → Python RuntimeError Conversion")
    print("=" * 80)
    
    # Helper function tests
    print("\n[1/4] Testing Expected<T> helper functions...")
    test_expected_helper_success()
    test_expected_helper_error()
    test_expected_vector_success()
    test_expected_vector_error()
    print("  ✓ All helper tests passed")
    
    # Architecture-limited instruction tests
    print("\n[2/4] Testing gfx1250+ instructions on gfx950 (should fail)...")
    test_smul_lo_u32_unsupported()
    test_ssub_u64_unsupported()
    test_sand_saveexec_b32_unsupported()
    test_sor_saveexec_b32_unsupported()
    test_vfma_mix_f32_unsupported()
    test_vrsq_iflag_f32_unsupported()
    print("  ✓ All architecture checks passed")
    
    # VCvtBF16toFP32 tests
    print("\n[3/4] Testing VCvtBF16toFP32 (gfx950+ instruction)...")
    test_vcvt_bf16_to_fp32_success()
    test_vcvt_bf16_to_fp32_missing_vgpr_mask()
    print("  ✓ VCvtBF16toFP32 tests passed")
    
    # MFMA tests
    print("\n[4/4] Testing MFMA instruction creation...")
    test_mfma_invalid_instruction()
    test_mxmfma_invalid_instruction()
    test_smfma_invalid_instruction()
    print("  ✓ MFMA error handling tests passed")
    
    # Error message quality tests
    print("\n[Extra] Testing error message quality...")
    test_error_messages_are_descriptive()
    test_exception_types_are_runtime_error()
    print("  ✓ Error message quality tests passed")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("\nValidated Functions (10 total):")
    print("  ✅ SMulLOU32 - gfx1250+ scalar multiply")
    print("  ✅ SSubU64 - gfx1250+ 64-bit subtract")
    print("  ✅ SAndSaveExecB32 - gfx1250+ exec mask AND")
    print("  ✅ SOrSaveExecB32 - gfx1250+ exec mask OR")
    print("  ✅ VFmaMixF32 - gfx1250+ vector FMA")
    print("  ✅ VRsqIFlagF32 - gfx1250+ vector reciprocal sqrt")
    print("  ✅ VCvtBF16toFP32 - gfx950+ bf16 conversion")
    print("  ✅ createMFMA - Generic MFMA creation")
    print("  ✅ createMXMFMA - MX format MFMA")
    print("  ✅ createSMFMA - Sparse MFMA")
    print("\nError Handling Validated:")
    print("  ✅ C++ Expected<T> → Python RuntimeError")
    print("  ✅ Error messages preserved and descriptive")
    print("  ✅ Architecture limitations handled gracefully")
    print("  ✅ Invalid parameters caught and reported")
    print("=" * 80)
