# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for set_preferred_engine_id_ext() and get_preferred_engine_id_ext() overloads.

Verifies that:
1. Setting by int64 works
2. Setting by string works
3. Setting to None clears the preference
4. Setting to empty string clears the preference
5. Getting the preference returns the correct value
6. Method chaining works
"""

import hipdnn_frontend as fe


def test_set_by_int():
    """Test setting preferred engine ID by integer."""
    graph = fe.Graph()

    graph.set_preferred_engine_id_ext(12345)

    engine_id = graph.get_preferred_engine_id_ext()
    assert engine_id is not None, "Engine ID should be set"
    assert engine_id == 12345, f"Expected 12345, got {engine_id}"


def test_set_by_string():
    """Test setting preferred engine ID by string."""
    graph = fe.Graph()

    test_engine_name = "TEST_ENGINE_NAME"
    graph.set_preferred_engine_id_ext(test_engine_name)

    engine_id = graph.get_preferred_engine_id_ext()
    assert engine_id is not None, "Engine ID should be set"
    assert isinstance(engine_id, int), f"Engine ID should be int, got {type(engine_id)}"


def test_clear_with_none():
    """Test clearing preference with None."""
    graph = fe.Graph()

    graph.set_preferred_engine_id_ext(12345)
    assert graph.get_preferred_engine_id_ext() is not None, "Should be set"

    graph.set_preferred_engine_id_ext(None)
    assert graph.get_preferred_engine_id_ext() is None, "Should be cleared"


def test_clear_with_empty_string():
    """Test clearing preference with empty string."""
    graph = fe.Graph()

    graph.set_preferred_engine_id_ext("TEST_ENGINE")
    assert graph.get_preferred_engine_id_ext() is not None, "Should be set"

    graph.set_preferred_engine_id_ext("")
    assert graph.get_preferred_engine_id_ext() is None, "Should be cleared"


def test_overload_interaction():
    """Test that overloads can override each other."""
    graph = fe.Graph()

    graph.set_preferred_engine_id_ext("ENGINE_A")
    id_from_string = graph.get_preferred_engine_id_ext()

    graph.set_preferred_engine_id_ext(999)
    id_from_int = graph.get_preferred_engine_id_ext()

    assert id_from_int == 999, f"Expected 999, got {id_from_int}"
    assert id_from_int != id_from_string, "IDs should be different"


def test_method_chaining():
    """Test that set_preferred_engine_id_ext() supports method chaining."""
    graph = fe.Graph()

    result = (
        graph.set_name("test_graph")
        .set_preferred_engine_id_ext(12345)
        .set_compute_data_type(fe.DataType.FLOAT)
    )

    assert result is graph, "Chaining should return the same graph object"
    assert (
        graph.get_name() == "test_graph"
    ), f"Expected name 'test_graph', got '{graph.get_name()}'"
    engine_id = graph.get_preferred_engine_id_ext()
    assert engine_id == 12345, f"Expected engine ID 12345, got {engine_id}"
    assert (
        graph.get_compute_data_type() == fe.DataType.FLOAT
    ), f"Expected FLOAT, got {graph.get_compute_data_type()}"


def test_chaining_with_string_overload():
    """Test chaining with string overload."""
    graph = fe.Graph()

    result = (
        graph.set_name("test_graph")
        .set_preferred_engine_id_ext("MY_ENGINE")
        .set_io_data_type(fe.DataType.HALF)
    )

    assert result is graph, "Chaining should return the same graph object"
    assert graph.get_preferred_engine_id_ext() is not None
