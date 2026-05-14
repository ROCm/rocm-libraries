# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""API tests for Graph configuration (mostly no GPU required)."""

import hipdnn_frontend as hipdnn


class TestGraphConfiguration:
    """Tests for Graph setter and getter methods."""

    def test_graph_set_name(self):
        """set_name() / get_name() roundtrip."""
        g = hipdnn.Graph()
        g.set_name("test_graph")
        assert g.get_name() == "test_graph"

    def test_graph_set_compute_data_type(self):
        """set_compute_data_type() / get_compute_data_type() roundtrip."""
        g = hipdnn.Graph()
        g.set_compute_data_type(hipdnn.DataType.FLOAT)
        assert g.get_compute_data_type() == hipdnn.DataType.FLOAT

    def test_graph_set_io_data_type(self):
        """set_io_data_type() / get_io_data_type() roundtrip."""
        g = hipdnn.Graph()
        g.set_io_data_type(hipdnn.DataType.FLOAT)
        assert g.get_io_data_type() == hipdnn.DataType.FLOAT

    def test_graph_set_intermediate_data_type(self):
        """set_intermediate_data_type() / get_intermediate_data_type() roundtrip."""
        g = hipdnn.Graph()
        g.set_intermediate_data_type(hipdnn.DataType.FLOAT)
        assert g.get_intermediate_data_type() == hipdnn.DataType.FLOAT

    def test_graph_method_chaining(self):
        """Chained setter calls return the same graph object."""
        g = hipdnn.Graph()
        result = (
            g.set_name("chained_graph")
            .set_io_data_type(hipdnn.DataType.FLOAT)
            .set_compute_data_type(hipdnn.DataType.FLOAT)
            .set_intermediate_data_type(hipdnn.DataType.FLOAT)
        )
        assert result is g
        assert g.get_name() == "chained_graph"


class TestGraphTensorCreation:
    """Tests for creating tensors via the Graph API."""

    def test_graph_tensor_creation(self):
        """graph.tensor() creates a shared tensor from attributes."""
        g = hipdnn.Graph()

        attrs = hipdnn.Tensor()
        attrs.set_dim([2, 3, 4])
        attrs.set_data_type(hipdnn.DataType.FLOAT)
        attrs.set_stride([12, 4, 1])

        t = g.tensor(attrs)
        assert t is not None
        assert t.get_dim() == [2, 3, 4]
        assert t.get_data_type() == hipdnn.DataType.FLOAT

    def test_graph_tensor_like(self):
        """Graph.tensor_like() creates a tensor with matching dims but new uid."""
        original = hipdnn.Tensor.create([4, 8, 16], hipdnn.DataType.FLOAT)
        original.set_name("original")

        copy = hipdnn.Graph.tensor_like(original)
        assert copy is not None
        assert copy.get_dim() == original.get_dim()
        assert copy.get_data_type() == original.get_data_type()
        # tensor_like clears the uid, so has_uid should be False
        assert not copy.has_uid()
