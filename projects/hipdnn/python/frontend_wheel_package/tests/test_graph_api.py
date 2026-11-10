# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""API tests for Graph configuration (mostly no GPU required)."""

import math
import sys

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .graph_builders import build_pointwise_add_graph, build_sdpa_graph
from .helpers import NullHandle, build_all_plans


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


def test_execute_timed_ext_without_compiled_plan_is_a_bad_error():
    """No compiled plan -> the same failure execute() would report; timing stays empty."""
    graph = hipdnn.Graph()

    err, timing = graph.execute_timed_ext(NullHandle(), {}, 0)
    assert err.is_bad()
    assert not timing.timed_out
    assert timing.elapsed_ms is None
    assert timing.quality == hipdnn.TimingQuality.INVALID


def test_numerical_note_enum_exposes_the_ten_notes_without_not_set():
    """NumericalNote mirrors the frontend enum; NOT_SET is not a reportable note."""
    expected = {
        "TENSOR_CORE",
        "DOWN_CONVERT_INPUTS",
        "REDUCED_PRECISION_REDUCTION",
        "FFT",
        "NONDETERMINISTIC",
        "WINOGRAD",
        "WINOGRAD_TILE_4x4",
        "WINOGRAD_TILE_6x6",
        "WINOGRAD_TILE_13x13",
        "STRICT_NAN_PROP",
    }
    assert {member.name for member in hipdnn.NumericalNote} == expected


def test_numerical_note_is_cudnns_spelling_of_the_enum():
    """cuDNN's Python frontend spells the enum numerical_note."""
    assert hipdnn.numerical_note is hipdnn.NumericalNote


def test_numerical_notes_for_engine_on_unbuilt_graph_raises():
    """The graph must be built before engine notes can be queried."""
    graph = hipdnn.Graph()

    with pytest.raises(RuntimeError):
        graph.get_numerical_notes_for_engine(0)


@pytest.mark.gpu
def test_numerical_notes_for_engine_reports_the_engines_notes():
    """The stub engine declares no numerical notes, so the backend reports NONDETERMINISTIC."""
    graph, _, _, _ = build_pointwise_add_graph(n=1, c=1, h=2, w=2)
    handle = hipdnn.create_handle()
    assert graph.validate().is_good()
    assert graph.build_operation_graph(handle).is_good()

    engine_ids = graph.get_ranked_engine_ids()
    assert engine_ids
    for engine_id in engine_ids:
        notes = graph.get_numerical_notes_for_engine(engine_id)
        assert notes == [hipdnn.NumericalNote.NONDETERMINISTIC]


@pytest.mark.gpu
class TestGraphExecuteTimedExt:
    """Tests for Graph.execute_timed_ext(): exactly-once, device-only-timed execution."""

    def test_uid_keyed_reports_valid_timing(self):
        """A real HIP event measurement through the test plugin reports its method."""
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
        assert not timing.timed_out
        if sys.platform == "win32" and timing.quality == hipdnn.TimingQuality.INVALID:
            # On Windows, hipEventElapsedTime can return a negative value for a short
            # span (https://github.com/ROCm/rocm-systems/issues/12925). The binding
            # reports that as INVALID with no elapsed time, which is its documented
            # contract. Remove this when the runtime fix ships.
            assert timing.elapsed_ms is None
            return
        assert timing.elapsed_ms is not None
        assert math.isfinite(timing.elapsed_ms)
        assert timing.elapsed_ms >= 0.0
        assert timing.quality in (
            hipdnn.TimingQuality.DEVICE_ONLY,
            hipdnn.TimingQuality.UNSTALLED,
        )


class TestCheckTensorUidsSet:
    """Tests for Graph.checkTensorUidsSet()."""

    def test_uids_set_after_graph_construction(self):
        """Tensors created via Graph.pointwise() have auto-assigned UIDs."""
        graph = _build_pointwise_add_graph([8, 16], [8, 16])
        result = graph.checkTensorUidsSet()
        assert result.is_good(), f"checkTensorUidsSet failed: {result.get_message()}"


class TestGetTensorsByUidAndName:
    """Tests for Graph.getTensorsByUid() and getTensorsByName()."""

    def test_get_tensors_by_uid_returns_all_tensors(self):
        """getTensorsByUid() returns a dict with an entry per tensor in the graph."""
        graph = _build_pointwise_add_graph([8, 16], [8, 16])
        tensors_by_uid = graph.getTensorsByUid()
        # pointwise add has 2 inputs + 1 output = 3 tensors
        assert isinstance(tensors_by_uid, dict)
        assert len(tensors_by_uid) == 3

    def test_get_tensors_by_name_returns_named_tensors(self):
        """getTensorsByName() returns a dict keyed by tensor name."""
        graph = _build_pointwise_add_graph([8, 16], [8, 16])
        tensors_by_name = graph.getTensorsByName()
        assert isinstance(tensors_by_name, dict)
        assert "a" in tensors_by_name
        assert "b" in tensors_by_name
        assert "out" in tensors_by_name

    def test_uid_and_name_maps_are_consistent(self):
        """Both maps contain the same tensor objects."""
        graph = _build_pointwise_add_graph([4, 8], [4, 8])
        by_uid = graph.getTensorsByUid()
        by_name = graph.getTensorsByName()
        uid_objects = set(id(t) for t in by_uid.values())
        name_objects = set(id(t) for t in by_name.values())
        assert uid_objects == name_objects


@pytest.mark.gpu
class TestGraphBuild:
    """Tests for Graph.build() one-shot lifecycle."""

    def test_build_produces_executable_graph(self):
        """build() validates, builds, creates plans, and finalizes in one call."""
        graph, a, b, out = build_pointwise_add_graph(n=1, c=1, h=2, w=2)
        handle = hipdnn.create_handle()
        result = graph.build(handle)
        assert result.is_good(), f"build() failed: {result.get_message()}"
        # Graph should be ready for execution
        assert graph.get_execution_plan_count() >= 1

    def test_build_with_invalid_graph_fails(self):
        """build() on an empty graph returns a bad error."""
        graph = hipdnn.Graph()
        handle = hipdnn.create_handle()
        result = graph.build(handle)
        assert result.is_bad()


@pytest.mark.gpu
class TestCompiledPlanSerialization:
    """Tests for to_compiled_plan_binary() / from_compiled_plan_binary()."""

    def test_compiled_plan_round_trip(self):
        """Serialize a compiled plan and restore it on a fresh graph."""
        graph, a, b, out = build_pointwise_add_graph(n=1, c=1, h=2, w=2)
        handle = hipdnn.create_handle()
        assert graph.build(handle).is_good()

        plan_bytes = graph.to_compiled_plan_binary()
        assert isinstance(plan_bytes, bytes)
        assert len(plan_bytes) > 0

        # Restore on a fresh graph
        graph2, _, _, _ = build_pointwise_add_graph(n=1, c=1, h=2, w=2)
        result = graph2.from_compiled_plan_binary(handle, plan_bytes)
        assert result.is_good(), f"from_compiled_plan_binary failed: {result.get_message()}"

    def test_to_compiled_plan_binary_without_build_raises(self):
        """Calling to_compiled_plan_binary() before build() raises RuntimeError."""
        graph = _build_pointwise_add_graph([8, 16], [8, 16])
        with pytest.raises(RuntimeError):
            graph.to_compiled_plan_binary()


_sdpa_available = hasattr(hipdnn.Graph(), "set_override_shape_enabled")


@pytest.mark.skipif(not _sdpa_available, reason="HIPDNN_ENABLE_SDPA not compiled in")
class TestOverrideShapeEnabled:
    """Tests for set_override_shape_enabled() / is_override_shape_enabled()."""

    def test_default_is_disabled(self):
        """Override shape mode is disabled by default."""
        g = hipdnn.Graph()
        assert not g.is_override_shape_enabled()

    def test_enable_and_query(self):
        """set_override_shape_enabled(True) enables the mode."""
        g = hipdnn.Graph()
        result = g.set_override_shape_enabled(True)
        assert result is g  # fluent API returns self
        assert g.is_override_shape_enabled()

    def test_disable_after_enable(self):
        """set_override_shape_enabled(False) disables the mode."""
        g = hipdnn.Graph()
        g.set_override_shape_enabled(True)
        g.set_override_shape_enabled(False)
        assert not g.is_override_shape_enabled()
