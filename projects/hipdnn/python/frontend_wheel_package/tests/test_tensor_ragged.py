# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for Tensor ragged-offset accessors (no GPU required)."""

import json

import pytest

import hipdnn_frontend as hipdnn

from .helpers import create_float_graph


class TestTensorRaggedOffset:
    """Round-trip tests for set_ragged_offset()/get_ragged_offset()."""

    def test_round_trip(self):
        """set_ragged_offset()/get_ragged_offset() round-trip."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        offset = hipdnn.Tensor.create([4], hipdnn.DataType.INT32)

        assert tensor.get_ragged_offset() is None

        tensor.set_ragged_offset(offset)

        assert tensor.get_ragged_offset() is offset

    def test_none_clears(self):
        """set_ragged_offset(None) clears the ragged offset."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_ragged_offset(hipdnn.Tensor.create([4], hipdnn.DataType.INT32))

        tensor.set_ragged_offset(None)

        assert tensor.get_ragged_offset() is None

    def test_returns_self(self):
        """set_ragged_offset() returns self for chaining."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        offset = hipdnn.Tensor.create([4], hipdnn.DataType.INT32)

        assert tensor.set_ragged_offset(offset) is tensor

    def test_sdpa_graph_json_references_offset_uid(self):
        """A ragged Q serializes the offset tensor's UID in the graph JSON."""
        if not hasattr(hipdnn.Graph, "sdpa"):
            pytest.skip("SDPA disabled")

        graph = create_float_graph()
        q = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
        k = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
        v = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
        offset = hipdnn.Tensor.create([3, 1, 1, 1], hipdnn.DataType.INT32)
        q.set_ragged_offset(offset)

        attrs = hipdnn.SdpaAttributes().set_generate_stats(True)
        o, _stats = graph.sdpa(q, k, v, attrs)
        o.set_output(True)

        data = json.loads(graph.to_json())
        (q_entry,) = [t for t in data["tensors"] if t["uid"] == q.get_uid()]

        assert offset.has_uid()
        assert q_entry["ragged_offset_tensor_uid"] == offset.get_uid()
        assert any(t["uid"] == offset.get_uid() for t in data["tensors"])
