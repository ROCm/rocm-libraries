#!/usr/bin/env python3
"""
Contract tests for the hard-select / read-back Graph bindings:
create_execution_plan_ext() and get_execution_plan_engine_id().

These verify the nanobind surface without a device:
1. Both methods are bound on Graph.
2. create_execution_plan_ext() returns an Error object (it must not throw and
   must not return None); on a graph with no built operation graph the Error
   reports is_bad().
3. get_execution_plan_engine_id() maps a bad backend result to a Python
   RuntimeError when no execution plan exists.

A full create -> build -> read-back lifecycle against a real backend is covered
by the dnn-benchmarking integration tests (test_execution.py).

USAGE:
    # From the python/hipdnn_frontend/test directory, after building and installing:
    python test_execution_plan_engine_id_bindings.py
"""

import hipdnn_frontend as fe


def _expect_raises(exc_type, fn):
    """Assert that calling fn() raises exc_type (no pytest dependency)."""
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


def test_bindings_exist():
    """Both new methods must be bound on Graph."""
    print("Test 1: bindings exist...")
    graph = fe.Graph()
    assert hasattr(graph, "create_execution_plan_ext"), "binding is missing"
    assert hasattr(graph, "get_execution_plan_engine_id"), "binding is missing"
    print("  OK both bindings are present")


def test_create_execution_plan_ext_returns_error_object():
    """On an unbuilt graph it returns an Error (is_bad) -- not None, never throws."""
    print("Test 2: create_execution_plan_ext returns an Error...")
    graph = fe.Graph()
    result = graph.create_execution_plan_ext(0)
    assert result is not None, "expected an Error object, got None"
    assert hasattr(result, "is_bad") and hasattr(result, "get_message"), (
        "return value is not an Error"
    )
    assert result.is_bad(), "expected is_bad() on a graph with no operation graph"
    print(f"  OK returns Error (msg: {result.get_message()[:50]!r})")


def test_get_execution_plan_engine_id_raises_without_plan():
    """The value getter maps a bad backend result to a Python RuntimeError."""
    print("Test 3: get_execution_plan_engine_id raises without a plan...")
    graph = fe.Graph()
    _expect_raises(RuntimeError, graph.get_execution_plan_engine_id)
    print("  OK raises RuntimeError when no execution plan exists")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing create_execution_plan_ext / get_execution_plan_engine_id bindings")
    print("=" * 60)
    try:
        test_bindings_exist()
        test_create_execution_plan_ext_returns_error_object()
        test_get_execution_plan_engine_id_raises_without_plan()
        print("=" * 60)
        print("OK All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\nX Test failed: {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"\nX Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
