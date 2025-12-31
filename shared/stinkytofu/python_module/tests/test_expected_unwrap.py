#!/usr/bin/env python3
"""
Test unwrapExpected() functionality in Python bindings.

This tests that Expected<T> is correctly converted to Python exceptions
when errors occur, and successfully unwrapped when values are present.
"""

import sys
import pytest

# Add build directory to path to import stinkytofu
sys.path.insert(0, '/data0/yangwen/rocm-libraries/shared/stinkytofu/build/python_module')
import stinkytofu


def test_expected_success_int():
    """Test that successful Expected<int> returns the value"""
    result = stinkytofu._testExpectedSuccess(21)
    assert result == 42, f"Expected 42, got {result}"
    print("✓ test_expected_success_int passed")


def test_expected_error_raises_exception():
    """Test that Expected<T>::Error raises RuntimeError in Python"""
    with pytest.raises(RuntimeError, match="Test error message"):
        stinkytofu._testExpectedError("Test error message")
    print("✓ test_expected_error_raises_exception passed")


def test_expected_error_custom_message():
    """Test that error messages are properly propagated"""
    custom_msg = "Custom error: architecture not supported"
    with pytest.raises(RuntimeError) as exc_info:
        stinkytofu._testExpectedError(custom_msg)
    
    assert custom_msg in str(exc_info.value), \
        f"Expected '{custom_msg}' in exception, got '{exc_info.value}'"
    print("✓ test_expected_error_custom_message passed")


def test_expected_vector_success():
    """Test that successful Expected<vector<int>> returns a Python list"""
    result = stinkytofu._testExpectedVectorSuccess()
    assert result == [1, 2, 3, 4, 5], f"Expected [1,2,3,4,5], got {result}"
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    print("✓ test_expected_vector_success passed")


def test_expected_vector_error():
    """Test that Expected<vector<T>>::Error raises RuntimeError"""
    with pytest.raises(RuntimeError, match="Vector creation failed"):
        stinkytofu._testExpectedVectorError()
    print("✓ test_expected_vector_error passed")


def test_expected_multiple_calls():
    """Test that unwrapExpected works correctly across multiple calls"""
    # Multiple successful calls
    assert stinkytofu._testExpectedSuccess(10) == 20
    assert stinkytofu._testExpectedSuccess(5) == 10
    assert stinkytofu._testExpectedSuccess(0) == 0
    
    # Multiple error calls
    for i in range(3):
        with pytest.raises(RuntimeError):
            stinkytofu._testExpectedError(f"Error {i}")
    
    # Interleaved success and error
    assert stinkytofu._testExpectedSuccess(100) == 200
    with pytest.raises(RuntimeError):
        stinkytofu._testExpectedError("Interleaved error")
    assert stinkytofu._testExpectedSuccess(50) == 100
    
    print("✓ test_expected_multiple_calls passed")


def test_exception_type():
    """Test that the exception type is RuntimeError (not generic Exception)"""
    with pytest.raises(RuntimeError) as exc_info:
        stinkytofu._testExpectedError("Type check")
    
    # Verify it's specifically RuntimeError, not just Exception
    assert type(exc_info.value).__name__ == 'RuntimeError'
    print("✓ test_exception_type passed")


def test_error_message_integrity():
    """Test that error messages with special characters are preserved"""
    special_msg = "Error: gfx900 doesn't support s_mul_lo_u32 (requires gfx1250+)"
    
    with pytest.raises(RuntimeError) as exc_info:
        stinkytofu._testExpectedError(special_msg)
    
    assert special_msg in str(exc_info.value)
    print("✓ test_error_message_integrity passed")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing unwrapExpected() functionality")
    print("=" * 70)
    
    # Run tests manually
    test_expected_success_int()
    test_expected_error_raises_exception()
    test_expected_error_custom_message()
    test_expected_vector_success()
    test_expected_vector_error()
    test_expected_multiple_calls()
    test_exception_type()
    test_error_message_integrity()
    
    print("=" * 70)
    print("All tests passed! ✅")
    print("=" * 70)

