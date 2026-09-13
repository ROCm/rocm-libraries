# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""API tests for Graph configuration (mostly no GPU required)."""

import math

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .graph_builders import build_pointwise_add_graph
from .helpers import build_all_plans


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


def _device_buffer(data):
    buf = hipdnn.DeviceBuffer(data.nbytes)
    buf.copy_from_host(data.tobytes())
    return buf


def _workspace_ptr(graph):
    size = graph.get_workspace_size()
    if size <= 0:
        return None, 0
    buf = hipdnn.DeviceBuffer(size)
    return buf, buf.ptr()


class _NullHandle:
    """Stand-in for a Handle whose get() yields a null pointer.

    execute_timed_ext() must reject a graph with no compiled plan before it ever
    dereferences the handle, so this path needs no real device or backend.
    """

    def get(self):
        return 0


def test_execute_timed_ext_without_compiled_plan_is_a_bad_error():
    """No compiled plan -> the same failure execute() would report; timing stays empty."""
    graph = hipdnn.Graph()

    err, timing = graph.execute_timed_ext(_NullHandle(), {}, 0)
    assert err.is_bad()
    assert "compiled execution plan" in err.get_message()
    assert timing.elapsed_ms is None
    assert timing.quality == hipdnn.TimingQuality.INVALID


@pytest.mark.gpu
class TestGraphExecuteTimedExt:
    """Tests for Graph.execute_timed_ext(): exactly-once, device-only-timed execution."""

    def test_uid_keyed_preserves_stub_output_and_reports_timing(self):
        """execute_timed_ext() runs the same variant-pack/backendExecute plumbing as
        execute(): a known sentinel written to the output buffer survives untouched
        under the pinned no-op test stub, and the call reports a finite timing.

        The ABSOLUTE-mode test stub (GoodPlugin) never touches device memory --
        conftest.py's _load_test_good_plugin and helpers.execute_zeros both note its
        execute() is a no-op. This proves the timed-execute plumbing runs end to end
        against a real device; it says nothing about the arithmetic of the operation,
        which is the C++/real-provider tests' job.
        """
        graph, a, b, out = build_pointwise_add_graph(n=1, c=1, h=2, w=2)
        handle = build_all_plans(graph)

        a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(a.get_dim())
        b_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32).reshape(
            b.get_dim()
        )
        sentinel = np.full(out.get_dim(), -1.0, dtype=np.float32)

        a_buf, b_buf = _device_buffer(a_data), _device_buffer(b_data)
        out_buf = _device_buffer(sentinel)
        ws_buf, ws_ptr = _workspace_ptr(graph)
        variant_pack = {
            a.get_uid(): a_buf.ptr(),
            b.get_uid(): b_buf.ptr(),
            out.get_uid(): out_buf.ptr(),
        }

        err, timing = graph.execute_timed_ext(handle, variant_pack, ws_ptr)
        assert err.is_good(), err.get_message()
        assert timing.elapsed_ms is not None
        assert math.isfinite(timing.elapsed_ms)
        assert timing.elapsed_ms >= 0.0
        assert timing.quality in (
            hipdnn.TimingQuality.DEVICE_ONLY,
            hipdnn.TimingQuality.HOST_INCLUDED,
        )

        actual = np.frombuffer(out_buf.copy_to_host(), dtype=np.float32)
        np.testing.assert_array_equal(actual, sentinel.reshape(-1))


def test_execution_timing_types_are_bound():
    """TimingQuality/ExecutionTiming expose the documented enum and read-only fields."""
    assert hipdnn.TimingQuality.DEVICE_ONLY.name == "DEVICE_ONLY"
    assert hipdnn.TimingQuality.HOST_INCLUDED.name == "HOST_INCLUDED"
    assert hipdnn.TimingQuality.INVALID.name == "INVALID"

    timing = hipdnn.ExecutionTiming()
    assert timing.elapsed_ms is None
    assert timing.quality == hipdnn.TimingQuality.INVALID

    with pytest.raises(AttributeError):
        timing.elapsed_ms = 1.0
    with pytest.raises(AttributeError):
        timing.quality = hipdnn.TimingQuality.DEVICE_ONLY
