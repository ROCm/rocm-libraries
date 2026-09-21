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


def _build_pointwise_add_graph(dim_a, dim_b):
    """Build a single-node pointwise-add graph over two input tensors."""
    graph = hipdnn.Graph()
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)

    a = hipdnn.Tensor.create(dim_a, hipdnn.DataType.FLOAT)
    a.set_name("a")
    b = hipdnn.Tensor.create(dim_b, hipdnn.DataType.FLOAT)
    b.set_name("b")

    attrs = hipdnn.PointwiseAttributes()
    attrs.set_name("add")
    attrs.set_mode(hipdnn.PointwiseMode.ADD)

    out = graph.pointwise(a, b, attrs)
    out.set_name("out")
    out.set_output(True)
    return graph


class TestGraphValidation:
    """Happy- and unhappy-path validation of multi-node graphs (no GPU)."""

    def test_pointwise_graph_validates(self):
        """A well-formed pointwise-add graph passes validation."""
        graph = _build_pointwise_add_graph([8, 16], [8, 16])

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_mismatched_pointwise_shapes_fail_validation(self):
        """Incompatible operand shapes for pointwise add fail validation."""
        graph = _build_pointwise_add_graph([8, 16], [4, 4])

        result = graph.validate()
        assert not result.is_good()

    def test_matmul_inner_dim_mismatch_fails_validation(self):
        """Matmul with non-matching contraction dims fails validation."""
        graph = hipdnn.Graph()
        graph.set_io_data_type(hipdnn.DataType.FLOAT)
        graph.set_compute_data_type(hipdnn.DataType.FLOAT)
        graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)

        a = hipdnn.Tensor.create([4, 3], hipdnn.DataType.FLOAT)
        a.set_name("A")
        b = hipdnn.Tensor.create([5, 6], hipdnn.DataType.FLOAT)
        b.set_name("B")

        attrs = hipdnn.MatmulAttributes()
        attrs.set_name("matmul")

        c = graph.matmul(a, b, attrs)
        c.set_name("C")
        c.set_output(True)

        result = graph.validate()
        assert not result.is_good()
