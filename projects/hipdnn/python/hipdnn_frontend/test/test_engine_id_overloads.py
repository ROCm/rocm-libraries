#!/usr/bin/env python3
"""
Test script for set_preferred_engine_id_ext() and get_preferred_engine_id_ext() overloads.

This script verifies that:
1. Setting by int64 works
2. Setting by string works
3. Setting to None clears the preference
4. Setting to empty string clears the preference
5. Getting the preference returns the correct value
6. Method chaining works

USAGE:
    # From the python/hipdnn_frontend/test directory, after building and installing:
    python test_engine_id_overloads.py
"""

import sys
import os

import hipdnn_frontend as fe


def test_set_by_int():
    """Test setting preferred engine ID by integer."""
    print("Test 1: Set by int...")
    graph = fe.Graph()

    # Set by int
    graph.set_preferred_engine_id_ext(12345)

    # Verify
    engine_id = graph.get_preferred_engine_id_ext()
    assert engine_id is not None, "Engine ID should be set"
    assert engine_id == 12345, f"Expected 12345, got {engine_id}"
    print("  OK Set by int works")


def test_set_by_string():
    """Test setting preferred engine ID by string."""
    print("Test 2: Set by string...")
    graph = fe.Graph()

    # Set by string
    test_engine_name = "TEST_ENGINE_NAME"
    graph.set_preferred_engine_id_ext(test_engine_name)

    # Verify it was set (we can't easily verify the exact ID without engineNameToId in Python)
    engine_id = graph.get_preferred_engine_id_ext()
    assert engine_id is not None, "Engine ID should be set"
    assert isinstance(engine_id, int), f"Engine ID should be int, got {type(engine_id)}"
    print(f"  OK Set by string works (ID: {engine_id})")


def test_clear_with_none():
    """Test clearing preference with None."""
    print("Test 3: Clear with None...")
    graph = fe.Graph()

    # Set then clear
    graph.set_preferred_engine_id_ext(12345)
    assert graph.get_preferred_engine_id_ext() is not None, "Should be set"

    graph.set_preferred_engine_id_ext(None)
    assert graph.get_preferred_engine_id_ext() is None, "Should be cleared"
    print("  OK Clear with None works")


def test_clear_with_empty_string():
    """Test clearing preference with empty string."""
    print("Test 4: Clear with empty string...")
    graph = fe.Graph()

    # Set then clear
    graph.set_preferred_engine_id_ext("TEST_ENGINE")
    assert graph.get_preferred_engine_id_ext() is not None, "Should be set"

    graph.set_preferred_engine_id_ext("")
    assert graph.get_preferred_engine_id_ext() is None, "Should be cleared"
    print("  OK Clear with empty string works")


def test_overload_interaction():
    """Test that overloads can override each other."""
    print("Test 5: Overload interaction...")
    graph = fe.Graph()

    # Set by string
    graph.set_preferred_engine_id_ext("ENGINE_A")
    id_from_string = graph.get_preferred_engine_id_ext()

    # Override with int
    graph.set_preferred_engine_id_ext(999)
    id_from_int = graph.get_preferred_engine_id_ext()

    assert id_from_int == 999, f"Expected 999, got {id_from_int}"
    assert id_from_int != id_from_string, "IDs should be different"
    print("  OK Overload interaction works")


def test_method_chaining():
    """Test that set_preferred_engine_id_ext() supports method chaining."""
    print("Test 6: Method chaining...")
    graph = fe.Graph()

    # Chain multiple setters
    result = (
        graph.set_name("test_graph")
        .set_preferred_engine_id_ext(12345)
        .set_compute_data_type(fe.DataType.FLOAT)
    )

    # Verify chaining returns the graph
    assert result is graph, "Chaining should return the same graph object"

    # Verify values were set
    assert (
        graph.get_name() == "test_graph"
    ), f"Expected name 'test_graph', got '{graph.get_name()}'"
    engine_id = graph.get_preferred_engine_id_ext()
    assert engine_id == 12345, f"Expected engine ID 12345, got {engine_id}"
    assert (
        graph.get_compute_data_type() == fe.DataType.FLOAT
    ), f"Expected FLOAT, got {graph.get_compute_data_type()}"
    print("  OK Method chaining works")


def test_chaining_with_string_overload():
    """Test chaining with string overload."""
    print("Test 7: Chaining with string overload...")
    graph = fe.Graph()

    # Chain with string overload
    result = (
        graph.set_name("test_graph")
        .set_preferred_engine_id_ext("MY_ENGINE")
        .set_io_data_type(fe.DataType.HALF)
    )

    assert result is graph, "Chaining should return the same graph object"
    assert graph.get_preferred_engine_id_ext() is not None
    print("  OK Chaining with string overload works")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing set_preferred_engine_id_ext() overloads")
    print("=" * 60)

    try:
        test_set_by_int()
        test_set_by_string()
        test_clear_with_none()
        test_clear_with_empty_string()
        test_overload_interaction()
        test_method_chaining()
        test_chaining_with_string_overload()

        print("=" * 60)
        print("OK All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\nX Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\nX Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
