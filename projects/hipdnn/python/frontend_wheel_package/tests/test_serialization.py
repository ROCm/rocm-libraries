# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for Graph serialization (to_json/from_json, to_binary/from_binary)."""

import pytest

import hipdnn_frontend as hipdnn

from .helpers import create_float_graph


def build_pointwise_add_graph(n=2, c=4, h=8, w=8):
    """Build a single-node pointwise-add graph for serialization round-trips."""
    graph = create_float_graph()
    graph.set_name("serialization_test")

    a = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    a.set_name("a")

    b = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    b.set_name("b")

    attrs = hipdnn.PointwiseAttributes()
    attrs.set_name("add")
    attrs.set_mode(hipdnn.PointwiseMode.ADD)

    out = graph.pointwise(a, b, attrs)
    out.set_name("out")
    out.set_output(True)

    return graph


class TestJsonSerialization:
    """Topology-only JSON round-trips (no GPU required)."""

    def test_to_json_returns_nonempty_string(self):
        """to_json() serializes a validated graph to a non-empty string."""
        graph = build_pointwise_add_graph()
        assert graph.validate().is_good()

        json_str = graph.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_from_json_restores_topology(self):
        """from_json(json) restores graph topology without a handle."""
        graph = build_pointwise_add_graph()
        assert graph.validate().is_good()
        json_str = graph.to_json()

        restored = hipdnn.Graph()
        assert restored.from_json(json_str).is_good()

    def test_json_round_trip_is_stable(self):
        """Re-serializing a restored graph reproduces the original JSON."""
        graph = build_pointwise_add_graph()
        assert graph.validate().is_good()
        json_str = graph.to_json()

        restored = hipdnn.Graph()
        assert restored.from_json(json_str).is_good()

        assert restored.to_json() == json_str


class TestBinarySerialization:
    """Topology-only binary round-trips (no GPU required)."""

    def test_to_binary_returns_nonempty_bytes(self):
        """to_binary() serializes a validated graph to non-empty bytes."""
        graph = build_pointwise_add_graph()
        assert graph.validate().is_good()

        data = graph.to_binary()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_from_binary_restores_topology(self):
        """from_binary(data) restores graph topology without a handle."""
        graph = build_pointwise_add_graph()
        assert graph.validate().is_good()
        data = graph.to_binary()

        restored = hipdnn.Graph()
        assert restored.from_binary(data).is_good()

    def test_binary_round_trip_is_stable(self):
        """Re-serializing a restored graph reproduces the original bytes."""
        graph = build_pointwise_add_graph()
        assert graph.validate().is_good()
        data = graph.to_binary()

        restored = hipdnn.Graph()
        assert restored.from_binary(data).is_good()

        assert restored.to_binary() == data


@pytest.mark.gpu
class TestHandleFinalizedDeserialization:
    """Handle-finalized deserialization overloads (require GPU)."""

    def test_from_json_with_handle_finalizes(self):
        """from_json(handle, json) deserializes and finalizes for execution."""
        graph = build_pointwise_add_graph()
        assert graph.validate().is_good()
        json_str = graph.to_json()

        handle = hipdnn.create_handle()
        restored = hipdnn.Graph()
        assert restored.from_json(handle, json_str).is_good()

    def test_from_binary_with_handle_finalizes(self):
        """from_binary(handle, data) deserializes and finalizes for execution."""
        graph = build_pointwise_add_graph()
        assert graph.validate().is_good()
        data = graph.to_binary()

        handle = hipdnn.create_handle()
        restored = hipdnn.Graph()
        assert restored.from_binary(handle, data).is_good()
