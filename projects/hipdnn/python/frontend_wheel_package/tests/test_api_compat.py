# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pins the call shapes existing hipdnn.Graph users rely on.

The cuDNN-style overloads are added next to the existing ones. These tests fail
if any existing call shape starts resolving to a new overload: a different return
type, raising instead of returning an Error, or a rejected keyword.
"""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .graph_builders import build_conv_fprop_graph, build_layernorm_graph


class _NullHandle:
    """Stand-in for a Handle whose get() yields a null pointer."""

    def get(self):
        return 0


def _assert_bad_error(result):
    assert isinstance(result, hipdnn.Error), f"expected an Error, got {result!r}"
    assert result.is_bad()


class TestGraphKeepsReturningErrors:
    """hipdnn.Graph reports failures as Error objects, never by raising."""

    def test_validate_returns_error(self):
        graph = hipdnn.Graph()
        graph.pointwise(
            hipdnn.Tensor.create([2, 3], hipdnn.DataType.FLOAT),
            hipdnn.Tensor.create([4, 5], hipdnn.DataType.FLOAT),
            hipdnn.PointwiseAttributes().set_mode(hipdnn.PointwiseMode.ADD),
        )
        _assert_bad_error(graph.validate())

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.execute(_NullHandle(), {}),
            lambda g: g.execute(_NullHandle(), {}, 0),
            lambda g: g.execute(_NullHandle(), {}, workspace=0),
            lambda g: g.execute(handle=_NullHandle(), variant_pack={}, workspace=0),
            lambda g: g.execute(_NullHandle(), variant_pack={}),
        ],
        ids=["positional", "positional-workspace", "workspace-kw", "all-kw", "pack-kw"],
    )
    def test_execute_call_shapes_return_error(self, call):
        _assert_bad_error(call(hipdnn.Graph()))

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.execute_plan_at_index(_NullHandle(), {1: 1}, 0, 0),
            lambda g: g.execute_plan_at_index(_NullHandle(), {1: 1}, 0, np.int64(0)),
            lambda g: g.execute_plan_at_index(
                handle=_NullHandle(), variant_pack={1: 1}, workspace=0, plan_index=0
            ),
            lambda g: g.execute_plan_at_index(_NullHandle(), {1: 1}, 0, index=0),
        ],
        ids=["positional", "numpy-index", "all-kw", "cudnn-index-kw"],
    )
    def test_execute_plan_at_index_call_shapes_return_error(self, call):
        result = call(hipdnn.Graph())
        _assert_bad_error(result)
        assert "out of bounds" in result.get_message()

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.build_plan_at_index(0),
            lambda g: g.build_plan_at_index(np.int64(0)),
            lambda g: g.build_plan_at_index(plan_index=0),
            lambda g: g.build_plan_at_index(index=0),
        ],
        ids=["positional", "numpy-index", "plan_index-kw", "index-kw"],
    )
    def test_build_plan_at_index_call_shapes_return_error(self, call):
        _assert_bad_error(call(hipdnn.Graph()))

    @pytest.mark.parametrize("keyword", ["plan_index", "index"])
    def test_plan_index_accessors_keep_raising(self, keyword):
        graph = hipdnn.Graph()
        for accessor in (
            graph.get_workspace_size_plan_at_index,
            graph.get_plan_name_at_index,
        ):
            with pytest.raises(RuntimeError, match="out of bounds"):
                accessor(**{keyword: 0})


class TestTensorCallShapes:
    """Existing Tensor/Graph.tensor/tensor_like calls behave as before."""

    def test_graph_tensor_from_tensor(self):
        source = hipdnn.Tensor.create([2, 3], hipdnn.DataType.HALF)
        for created in (
            hipdnn.Graph.tensor(source),
            hipdnn.Graph.tensor(tensor=source),
        ):
            assert created.get_dim() == [2, 3]
            assert created.get_data_type() == hipdnn.DataType.HALF

    def test_tensor_like_keyword_and_errors(self):
        source = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        copy = hipdnn.Graph.tensor_like(tensor=source, name="copy")
        assert copy.get_name() == "copy"
        with pytest.raises(TypeError, match="__dlpack__"):
            hipdnn.Graph.tensor_like(object())

    def test_set_data_type_with_data_type_chains(self):
        tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        assert tensor.set_data_type(hipdnn.DataType.HALF) is tensor
        assert tensor.get_data_type() == hipdnn.DataType.HALF

    @pytest.mark.parametrize("convert", [int, np.int64], ids=["int", "numpy-int"])
    def test_set_data_type_with_integer_value(self, convert):
        tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        assert tensor.set_data_type(convert(hipdnn.DataType.HALF.value)) is tensor
        assert tensor.get_data_type() == hipdnn.DataType.HALF

    def test_set_data_type_rejects_non_dtypes_as_before(self):
        with pytest.raises(TypeError):
            hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT).set_data_type("half")


class TestGraphTypeAndOps:
    """The Graph class and its attribute-object op calls are unchanged."""

    def test_graph_type_is_unchanged(self):
        graph = hipdnn.Graph()
        assert type(graph) is hipdnn.Graph
        assert graph.set_name("g") is graph

    def test_attribute_op_calls(self):
        graph, _x, _weight, y = build_conv_fprop_graph()
        assert isinstance(y, hipdnn.Tensor)
        assert graph.validate().is_good()

        _graph, _x, _scale, _bias, out, mean, inv_variance = build_layernorm_graph()
        assert isinstance(out, hipdnn.Tensor)
        assert (mean, inv_variance) == (None, None)

    def test_attribute_op_keyword_attributes(self):
        graph = hipdnn.Graph()
        x = hipdnn.Tensor.create([1, 4, 8, 8], hipdnn.DataType.FLOAT)
        w = hipdnn.Tensor.create([4, 4, 3, 3], hipdnn.DataType.FLOAT)
        attributes = hipdnn.ConvFpropAttributes().set_padding([1, 1])
        y = graph.conv_fprop(x, w, attributes=attributes)
        assert isinstance(y, hipdnn.Tensor)

    def test_mixing_attributes_and_cudnn_keywords_is_rejected(self):
        graph = hipdnn.Graph()
        x = hipdnn.Tensor.create([1, 4, 8, 8], hipdnn.DataType.FLOAT)
        w = hipdnn.Tensor.create([4, 4, 3, 3], hipdnn.DataType.FLOAT)
        with pytest.raises(TypeError):
            graph.conv_fprop(x, w, hipdnn.ConvFpropAttributes(), name="conv")
