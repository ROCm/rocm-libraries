# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for hipdnn.pygraph, the Graph subclass with cuDNN pygraph semantics."""

import json

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

_FLOAT = hipdnn.DataType.FLOAT
_SDPA_ENABLED = hasattr(hipdnn.Graph, "sdpa")

_VALUE_ERROR_CODES = {
    hipdnn.ErrorCode.ATTRIBUTE_NOT_SET,
    hipdnn.ErrorCode.SHAPE_DEDUCTION_FAILED,
    hipdnn.ErrorCode.INVALID_TENSOR_NAME,
    hipdnn.ErrorCode.INVALID_VARIANT_PACK,
}
_NOT_SUPPORTED_CODES = {
    hipdnn.ErrorCode.GRAPH_NOT_SUPPORTED,
    hipdnn.ErrorCode.GRAPH_EXECUTION_PLAN_CREATION_FAILED,
    hipdnn.ErrorCode.HEURISTIC_QUERY_FAILED,
    hipdnn.ErrorCode.UNSUPPORTED_GRAPH_FORMAT,
}


class _NullHandle:
    """Stand-in for a Handle whose get() yields a null pointer.

    Passing it explicitly keeps pygraph from creating its own handle, so the
    failure paths below need no device.
    """

    def get(self):
        return 0


def _expected_exception(error):
    """The exception class pygraph raises for a bad Error, as cuDNN maps it."""
    code = error.get_code()
    if code in _VALUE_ERROR_CODES:
        return ValueError
    if code in _NOT_SUPPORTED_CODES:
        return hipdnn.GraphNotSupportedError
    return RuntimeError


def _add_float_pointwise(graph, dim_a, dim_b):
    """Add a FLOAT pointwise-add node to graph; returns (a, b, out)."""
    a = hipdnn.Tensor.create(dim_a, _FLOAT)
    a.set_name("a")
    b = hipdnn.Tensor.create(dim_b, _FLOAT)
    b.set_name("b")

    attrs = hipdnn.PointwiseAttributes()
    attrs.set_name("add")
    attrs.set_mode(hipdnn.PointwiseMode.ADD)

    out = graph.pointwise(a, b, attrs)
    out.set_name("out")
    out.set_output(True)
    return a, b, out


def _float_pygraph(**kwargs):
    return hipdnn.pygraph(
        io_data_type=_FLOAT,
        intermediate_data_type=_FLOAT,
        compute_data_type=_FLOAT,
        **kwargs,
    )


def _float_graph():
    graph = hipdnn.Graph()
    graph.set_io_data_type(_FLOAT)
    graph.set_intermediate_data_type(_FLOAT)
    graph.set_compute_data_type(_FLOAT)
    return graph


class TestClass:
    """pygraph is a distinct subclass; Graph is unchanged."""

    def test_pygraph_is_a_graph(self):
        assert isinstance(hipdnn.pygraph(), hipdnn.Graph)
        assert issubclass(hipdnn.pygraph, hipdnn.Graph)

    def test_graph_type_is_unchanged(self):
        assert type(hipdnn.Graph()) is hipdnn.Graph
        assert not isinstance(hipdnn.Graph(), hipdnn.pygraph)

    def test_inherited_setters_return_the_same_object(self):
        p = hipdnn.pygraph()
        assert p.set_name("x") is p
        assert p.set_io_data_type(_FLOAT) is p
        assert p.deselect_workspace_greater_than(1 << 20) is p
        assert p.deselect_engines(["no_such_engine"]) is p
        assert p.deselect_engines([12345]) is p
        assert p.get_name() == "x"

    def test_inherited_attribute_op_works(self):
        p = _float_pygraph()
        _, _, out = _add_float_pointwise(p, [2, 3], [2, 3])
        assert isinstance(out, hipdnn.Tensor)
        assert p.validate() is None

    @pytest.mark.skipif(
        not hasattr(hipdnn.Graph, "add"), reason="named pointwise methods absent"
    )
    def test_inherited_keyword_op_works(self):
        p = _float_pygraph()
        a = hipdnn.Tensor.create([2, 3], _FLOAT)
        b = hipdnn.Tensor.create([2, 3], _FLOAT)
        out = p.add(a=a, b=b)
        assert isinstance(out, hipdnn.Tensor)

    def test_graph_not_supported_error_is_a_runtime_error(self):
        assert issubclass(hipdnn.GraphNotSupportedError, RuntimeError)


class TestConstruction:
    """Constructor keywords mirror cudnn.pygraph."""

    def test_defaults_match_graph(self):
        p = hipdnn.pygraph()
        g = hipdnn.Graph()
        assert p.get_name() == g.get_name()
        assert p.get_io_data_type() == hipdnn.DataType.NOT_SET
        assert p.get_intermediate_data_type() == hipdnn.DataType.NOT_SET
        assert p.get_compute_data_type() == hipdnn.DataType.NOT_SET

    def test_keywords_set_name_and_data_types(self):
        p = hipdnn.pygraph(
            name="g",
            io_data_type=hipdnn.DataType.HALF,
            intermediate_data_type=_FLOAT,
            compute_data_type=hipdnn.DataType.BFLOAT16,
        )
        assert p.get_name() == "g"
        assert p.get_io_data_type() == hipdnn.DataType.HALF
        assert p.get_intermediate_data_type() == _FLOAT
        assert p.get_compute_data_type() == hipdnn.DataType.BFLOAT16

    def test_positional_arguments_follow_cudnn_order(self):
        p = hipdnn.pygraph("g", hipdnn.DataType.HALF, _FLOAT, hipdnn.DataType.BFLOAT16)
        assert p.get_name() == "g"
        assert p.get_io_data_type() == hipdnn.DataType.HALF
        assert p.get_intermediate_data_type() == _FLOAT
        assert p.get_compute_data_type() == hipdnn.DataType.BFLOAT16

    def test_default_values_of_unsupported_parameters_are_accepted(self):
        hipdnn.pygraph(
            handle=None,
            sm_count=None,
            sm_version=None,
            kernel_cache=None,
            device_property=None,
            is_dynamic_shape_enabled=False,
            is_override_shape_enabled=False,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sm_count": 80},
            {"sm_version": 90},
            {"kernel_cache": object()},
            {"device_property": object()},
            {"is_dynamic_shape_enabled": True},
        ],
        ids=lambda kwargs: next(iter(kwargs)),
    )
    def test_unsupported_parameters_raise(self, kwargs):
        name = next(iter(kwargs))
        with pytest.raises(NotImplementedError, match=name):
            hipdnn.pygraph(**kwargs)

    def test_override_shape_enabled_follows_sdpa_support(self):
        if _SDPA_ENABLED:
            hipdnn.pygraph(is_override_shape_enabled=True)
        else:
            with pytest.raises(NotImplementedError, match="is_override_shape_enabled"):
                hipdnn.pygraph(is_override_shape_enabled=True)

    def test_handle_must_be_a_handle(self):
        with pytest.raises(TypeError):
            hipdnn.pygraph(handle=object())


def _mismatched_validate(graph):
    _add_float_pointwise(graph, [8, 16], [4, 4])
    return graph.validate()


def _bad_ranking_metric(graph):
    return graph.set_ranking_metric("not_a_metric")


def _plan_index_out_of_range(graph):
    return graph.build_plan_at_index(0)


def _index_spelling_out_of_range(graph):
    return graph.build_plan_at_index(index=0)


def _engine_before_build(graph):
    return graph.add_engine(1)


def _plan_ext_before_build(graph):
    return graph.create_execution_plan_ext(0)


_FAILURE_CASES = [
    _mismatched_validate,
    _bad_ranking_metric,
    _plan_index_out_of_range,
    _index_spelling_out_of_range,
    _engine_before_build,
    _plan_ext_before_build,
]


class TestRaisingLifecycle:
    """pygraph raises where Graph returns a bad Error."""

    @pytest.mark.parametrize("call", _FAILURE_CASES, ids=lambda f: f.__name__)
    def test_failure_raises_on_pygraph_and_returns_error_on_graph(self, call):
        if call is _index_spelling_out_of_range and not _graph_has_index_spelling():
            pytest.skip("Graph.build_plan_at_index(index=) spelling absent")
        error = call(_float_graph())
        assert isinstance(error, hipdnn.Error)
        assert error.is_bad()

        expected = _expected_exception(error)
        with pytest.raises(expected) as excinfo:
            call(_float_pygraph())
        assert type(excinfo.value) is expected
        assert str(excinfo.value) == error.get_message()

    def test_successful_calls_return_none(self):
        p = _float_pygraph()
        _add_float_pointwise(p, [2, 3], [2, 3])
        assert p.validate() is None
        assert p.checkNoDuplicateTensorIds() is None
        assert p.topologicallySortGraph() is None
        assert p.set_ranking_metric("time") is None

    def test_get_workspace_size_raises_without_a_plan(self):
        p = hipdnn.pygraph()
        with pytest.raises(RuntimeError):
            p.get_workspace_size()
        with pytest.raises(RuntimeError):
            p.get_workspace_size(handle=None)
        with pytest.raises(RuntimeError, match="out of bounds"):
            p.get_workspace_size_plan_at_index(0)
        with pytest.raises(RuntimeError, match="out of bounds"):
            p.get_workspace_size_plan_at_index(plan_index=0)
        with pytest.raises(RuntimeError, match="out of bounds"):
            p.get_workspace_size_plan_at_index(index=0)
        with pytest.raises(RuntimeError, match="out of bounds"):
            p.get_workspace_size_plan_at_index(0, None)

    @pytest.mark.parametrize(
        "override", ["override_uids", "override_shapes", "override_strides"]
    )
    def test_workspace_size_overrides_are_unsupported(self, override):
        p = hipdnn.pygraph()
        value = [1] if override == "override_uids" else [[1]]
        with pytest.raises(NotImplementedError, match=override):
            p.get_workspace_size(**{override: value})
        with pytest.raises(NotImplementedError, match=override):
            p.get_workspace_size_plan_at_index(0, **{override: value})


def _graph_has_index_spelling():
    try:
        hipdnn.Graph().build_plan_at_index(index=0)
    except TypeError:
        return False
    return True


_NO_PLAN = "no compiled execution plan"
_OUT_OF_BOUNDS = "out of bounds"


class TestExecuteDispatch:
    """Both execute forms resolve, positionally and by keyword, and raise on failure."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: p.execute(_NullHandle(), {}),
            lambda p: p.execute(_NullHandle(), {}, 0),
            lambda p: p.execute(handle=_NullHandle(), variant_pack={}, workspace=0),
        ],
        ids=["positional", "positional_workspace", "keywords"],
    )
    def test_hipdnn_form(self, call):
        with pytest.raises(RuntimeError, match=_NO_PLAN):
            call(hipdnn.pygraph())

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: p.execute({}, 0, _NullHandle()),
            lambda p: p.execute({1: 0, None: 0}, 0, handle=_NullHandle()),
            lambda p: p.execute(
                tensor_to_device_buffer={}, workspace=0, handle=_NullHandle()
            ),
            lambda p: p.execute({}, 0, _NullHandle(), None, None, None),
        ],
        ids=["positional", "none_key_skipped", "keywords", "explicit_defaults"],
    )
    def test_cudnn_form(self, call):
        with pytest.raises(RuntimeError, match=_NO_PLAN):
            call(hipdnn.pygraph())

    def test_cudnn_form_tensor_keys(self):
        p = hipdnn.pygraph()
        t = hipdnn.Tensor.create([1], _FLOAT)
        t.set_uid(7)
        with pytest.raises(RuntimeError, match=_NO_PLAN):
            p.execute({t: 0}, 0, _NullHandle())

    def test_cudnn_form_overrides(self):
        p = hipdnn.pygraph()
        if _SDPA_ENABLED:
            with pytest.raises(RuntimeError, match=_NO_PLAN):
                p.execute({}, 0, _NullHandle(), override_uids=[1])
        else:
            with pytest.raises(NotImplementedError, match="override_uids"):
                p.execute({}, 0, _NullHandle(), override_uids=[1])

    def test_graph_execute_still_returns_error(self):
        error = hipdnn.Graph().execute(_NullHandle(), {}, 0)
        assert isinstance(error, hipdnn.Error)
        assert error.is_bad()

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: p.execute_plan_at_index(_NullHandle(), {}, 0, 0),
            lambda p: p.execute_plan_at_index(_NullHandle(), {}, 0, plan_index=0),
            lambda p: p.execute_plan_at_index(_NullHandle(), {}, 0, index=0),
            lambda p: p.execute_plan_at_index({}, 0, 0, _NullHandle()),
            lambda p: p.execute_plan_at_index(
                tensor_to_device_buffer={}, workspace=0, index=0, handle=_NullHandle()
            ),
        ],
        ids=[
            "hipdnn_positional",
            "hipdnn_plan_index",
            "hipdnn_index",
            "cudnn_positional",
            "cudnn_keywords",
        ],
    )
    def test_execute_plan_at_index_forms(self, call):
        with pytest.raises(RuntimeError, match=_OUT_OF_BOUNDS):
            call(hipdnn.pygraph())

    def test_execute_plan_at_index_overrides_are_unsupported(self):
        with pytest.raises(NotImplementedError, match="override_shapes"):
            hipdnn.pygraph().execute_plan_at_index(
                {}, 0, 0, _NullHandle(), override_shapes=[[1]]
            )


class TestSerialization:
    """serialize() / deserialize() round-trips (structure only; no device needed)."""

    @staticmethod
    def _validated():
        p = _float_pygraph(name="serialized")
        _add_float_pointwise(p, [2, 3], [2, 3])
        p.validate()
        return p

    def test_serialize_returns_list_of_ints_matching_to_binary(self):
        p = self._validated()
        blob = p.serialize()
        assert isinstance(blob, list)
        assert blob and all(isinstance(byte, int) for byte in blob)
        assert bytes(blob) == p.to_binary()

    @pytest.mark.parametrize("as_bytes", [False, True], ids=["list", "bytes"])
    def test_binary_round_trip(self, as_bytes):
        p = self._validated()
        blob = p.serialize()
        restored = hipdnn.pygraph()
        # An explicit null handle restores the structure without creating a handle.
        assert (
            restored.deserialize(_NullHandle(), bytes(blob) if as_bytes else blob)
            is None
        )
        assert restored.to_json() == p.to_json()

    def test_json_round_trip_needs_no_handle(self):
        p = self._validated()
        restored = hipdnn.pygraph()
        assert restored.deserialize(p.to_json()) is None
        assert restored.to_json() == p.to_json()
        restored = hipdnn.pygraph()
        assert restored.deserialize(None, p.to_json()) is None
        assert restored.to_json() == p.to_json()

    def test_deserialize_rejects_other_types(self):
        with pytest.raises(TypeError):
            hipdnn.pygraph().deserialize(_NullHandle(), 3.5)

    def test_from_json_and_from_binary_return_none(self):
        p = self._validated()
        restored = hipdnn.pygraph()
        assert restored.from_json(p.to_json()) is None
        assert restored.to_json() == p.to_json()
        restored = hipdnn.pygraph()
        assert restored.from_binary(p.to_binary()) is None
        assert restored.to_json() == p.to_json()


def _device_buffers(tensors):
    return {
        tensor: hipdnn.DeviceBuffer(
            int(np.prod(tensor.get_dim())) * np.dtype(np.float32).itemsize
        )
        for tensor in tensors
    }


def _workspace(size):
    return hipdnn.DeviceBuffer(size) if size > 0 else 0


@pytest.mark.gpu
class TestCudnnStyleExecution:
    """cuDNN-style end-to-end flows against the stub engine, with no caller handle."""

    def test_build_and_execute_without_a_handle(self):
        g = _float_pygraph()
        a, b, out = _add_float_pointwise(g, [1, 1, 2, 2], [1, 1, 2, 2])
        assert g.build([hipdnn.HeuristicMode.FALLBACK]) is None

        buffers = _device_buffers([a, b, out])
        workspace = _workspace(g.get_workspace_size())
        assert g.execute(buffers, workspace) is None
        assert (
            g.execute({t.get_uid(): buf for t, buf in buffers.items()}, workspace)
            is None
        )

    def test_supplied_handle_is_used(self):
        handle = hipdnn.create_handle()
        g = _float_pygraph(handle=handle)
        a, b, out = _add_float_pointwise(g, [1, 1, 2, 2], [1, 1, 2, 2])
        g.build()
        g.create_execution_plans([hipdnn.HeuristicMode.FALLBACK])
        g.check_support()
        g.build_plans()

        buffers = _device_buffers([a, b, out])
        workspace = _workspace(g.get_workspace_size())
        assert g.execute(buffers, workspace, handle) is None
        assert g.execute(handle, buffers, workspace) is None
        assert g.execute(None, buffers, workspace) is None

        # Handle-less calls go through the supplied handle, not a fresh one.
        hipdnn.destroy_handle(handle)
        with pytest.raises(RuntimeError, match="Handle has been destroyed"):
            g.execute(buffers, workspace)

    def test_execute_plan_at_index_without_a_handle(self):
        g = _float_pygraph()
        a, b, out = _add_float_pointwise(g, [1, 1, 2, 2], [1, 1, 2, 2])
        g.build()
        g.create_execution_plans([hipdnn.HeuristicMode.FALLBACK])
        g.check_support()
        g.build_plans(hipdnn.BuildPlanPolicy.ALL)
        assert g.get_execution_plan_count() >= 1

        buffers = _device_buffers([a, b, out])
        workspace = _workspace(g.get_workspace_size_plan_at_index(index=0))
        assert g.execute_plan_at_index(buffers, workspace, 0) is None

    def test_deserialize_binary_uses_graph_handle(self):
        g = _float_pygraph()
        _add_float_pointwise(g, [1, 1, 2, 2], [1, 1, 2, 2])
        g.validate()
        blob = g.serialize()

        restored = hipdnn.pygraph()
        assert restored.deserialize(blob) is None
        # Finalizing with a handle assigns the graph an id; the topology is unchanged.
        restored_json = json.loads(restored.to_json())
        restored_json.pop("id")
        assert restored_json == json.loads(g.to_json())
        # Only a handle-finalized graph accepts plan creation directly.
        restored.create_execution_plans([hipdnn.HeuristicMode.FALLBACK])
