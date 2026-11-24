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
from .helpers import NullHandle


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
            lambda g: g.execute(NullHandle(), {}),
            lambda g: g.execute(NullHandle(), {}, 0),
            lambda g: g.execute(NullHandle(), {}, workspace=0),
            lambda g: g.execute(handle=NullHandle(), variant_pack={}, workspace=0),
            lambda g: g.execute(NullHandle(), variant_pack={}),
        ],
        ids=["positional", "positional-workspace", "workspace-kw", "all-kw", "pack-kw"],
    )
    def test_execute_call_shapes_return_error(self, call):
        _assert_bad_error(call(hipdnn.Graph()))

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.execute_plan_at_index(NullHandle(), {1: 1}, 0, 0),
            lambda g: g.execute_plan_at_index(NullHandle(), {1: 1}, 0, np.int64(0)),
            lambda g: g.execute_plan_at_index(
                handle=NullHandle(), variant_pack={1: 1}, workspace=0, plan_index=0
            ),
            lambda g: g.execute_plan_at_index(NullHandle(), {1: 1}, 0, index=0),
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


def _tensor(dim=(2, 4, 8, 8), data_type=hipdnn.DataType.FLOAT):
    return hipdnn.Tensor.create(list(dim), data_type)


def _pointwise(mode):
    return hipdnn.PointwiseAttributes().set_mode(mode)


_requires_sdpa = pytest.mark.skipif(
    not hasattr(hipdnn.Graph, "sdpa"), reason="SDPA is compiled out of the bindings"
)

# Each entry builds the full argument set of an attributes-object overload, in
# parameter order and keyed by its bound name, so the same arguments can be
# passed positionally and by keyword.
_ATTRIBUTE_OVERLOADS = [
    pytest.param(
        "batchnorm",
        lambda: dict(
            x=_tensor(),
            scale=_tensor(),
            bias=_tensor(),
            attributes=hipdnn.BatchnormAttributes(),
        ),
        id="batchnorm",
    ),
    pytest.param(
        "batchnorm_backward",
        lambda: dict(
            dy=_tensor(),
            x=_tensor(),
            scale=_tensor(),
            attributes=hipdnn.BatchnormBackwardAttributes(),
        ),
        id="batchnorm_backward",
    ),
    pytest.param(
        "pointwise",
        lambda: dict(
            in0=_tensor(), attributes=_pointwise(hipdnn.PointwiseMode.RELU_FWD)
        ),
        id="pointwise-unary",
    ),
    pytest.param(
        "pointwise",
        lambda: dict(
            in0=_tensor(),
            in1=_tensor(),
            attributes=_pointwise(hipdnn.PointwiseMode.ADD),
        ),
        id="pointwise-binary",
    ),
    pytest.param(
        "pointwise",
        lambda: dict(
            in0=_tensor(),
            in1=_tensor(),
            in2=_tensor(),
            attributes=_pointwise(hipdnn.PointwiseMode.BINARY_SELECT),
        ),
        id="pointwise-ternary",
    ),
    pytest.param(
        "conv_fprop",
        lambda: dict(x=_tensor(), w=_tensor(), attributes=hipdnn.ConvFpropAttributes()),
        id="conv_fprop",
    ),
    pytest.param(
        "matmul",
        lambda: dict(a=_tensor(), b=_tensor(), attributes=hipdnn.MatmulAttributes()),
        id="matmul",
    ),
    pytest.param(
        "conv_dgrad",
        lambda: dict(
            dy=_tensor(), w=_tensor(), attributes=hipdnn.ConvDgradAttributes()
        ),
        id="conv_dgrad",
    ),
    pytest.param(
        "conv_wgrad",
        lambda: dict(
            dy=_tensor(), x=_tensor(), attributes=hipdnn.ConvWgradAttributes()
        ),
        id="conv_wgrad",
    ),
    pytest.param(
        "batchnorm_inference_variance_ext",
        lambda: dict(
            x=_tensor(),
            mean=_tensor(),
            variance=_tensor(),
            scale=_tensor(),
            bias=_tensor(),
            epsilon=_tensor([1]),
            attributes=hipdnn.BatchnormInferenceAttributesVarianceExt(),
        ),
        id="batchnorm_inference_variance_ext",
    ),
    pytest.param(
        "layernorm",
        lambda: dict(
            x=_tensor(),
            scale=_tensor(),
            bias=_tensor(),
            attributes=hipdnn.LayernormAttributes(),
        ),
        id="layernorm",
    ),
    pytest.param(
        "layernorm_backward",
        lambda: dict(
            dy=_tensor(),
            x=_tensor(),
            scale=_tensor(),
            attributes=hipdnn.LayernormBackwardAttributes(),
        ),
        id="layernorm_backward",
    ),
    pytest.param(
        "rmsnorm",
        lambda: dict(
            x=_tensor(), scale=_tensor(), attributes=hipdnn.RMSNormAttributes()
        ),
        id="rmsnorm",
    ),
    pytest.param(
        "rmsnorm_backward",
        lambda: dict(
            dy=_tensor(),
            x=_tensor(),
            scale=_tensor(),
            inv_rms=_tensor(),
            attributes=hipdnn.RMSNormBackwardAttributes(),
        ),
        id="rmsnorm_backward",
    ),
    pytest.param(
        "block_scale_dequantize",
        lambda: dict(
            x=_tensor(),
            scale=_tensor(),
            attributes=hipdnn.BlockScaleDequantizeAttributes(),
        ),
        id="block_scale_dequantize",
    ),
    pytest.param(
        "block_scale_quantize",
        lambda: dict(x=_tensor(), attributes=hipdnn.BlockScaleQuantizeAttributes()),
        id="block_scale_quantize",
    ),
    pytest.param(
        "reduction",
        lambda: dict(x=_tensor(), attributes=hipdnn.ReductionAttributes()),
        id="reduction",
    ),
    pytest.param(
        "reduction",
        lambda: dict(x=_tensor(), y=_tensor(), attributes=hipdnn.ReductionAttributes()),
        id="reduction-with_output",
    ),
    pytest.param(
        "moe_grouped_matmul",
        lambda: dict(
            token=_tensor(),
            weight=_tensor(),
            first_token_offset=_tensor(),
            token_index=_tensor(),
            token_ks=_tensor(),
            attributes=hipdnn.MoeGroupedMatmulAttributes(),
        ),
        id="moe_grouped_matmul",
    ),
    pytest.param(
        "custom_op",
        lambda: dict(
            inputs=[_tensor(), _tensor()],
            num_outputs=2,
            attributes=hipdnn.CustomOpAttributes().set_custom_op_id("example.identity"),
        ),
        id="custom_op",
    ),
    pytest.param(
        "sdpa",
        lambda: dict(
            q=_tensor(),
            k=_tensor(),
            v=_tensor(),
            attributes=hipdnn.SdpaAttributes().set_generate_stats(True),
        ),
        id="sdpa",
        marks=_requires_sdpa,
    ),
    pytest.param(
        "sdpa_backward",
        lambda: dict(
            q=_tensor(),
            k=_tensor(),
            v=_tensor(),
            o=_tensor(),
            dO=_tensor(),
            stats=_tensor(),
            attributes=hipdnn.SdpaBackwardAttributes(),
        ),
        id="sdpa_backward",
        marks=_requires_sdpa,
    ),
    pytest.param(
        "resample",
        lambda: dict(x=_tensor(), attributes=hipdnn.ResampleFwdAttributes()),
        id="resample",
    ),
    pytest.param(
        "resample_fwd",
        lambda: dict(x=_tensor(), attributes=hipdnn.ResampleFwdAttributes()),
        id="resample_fwd",
    ),
]


def _output_shape(outputs):
    if isinstance(outputs, (tuple, list)):
        return type(outputs), [type(output) for output in outputs]
    return type(outputs)


@pytest.mark.parametrize("method, arguments", _ATTRIBUTE_OVERLOADS)
def test_attribute_overload_parameter_names(method, arguments):
    """The parameter names of the attributes-object overloads are public: an
    all-keyword call reaches the same overload as the positional call."""
    positional = getattr(hipdnn.Graph(), method)(*arguments().values())
    by_keyword = getattr(hipdnn.Graph(), method)(**arguments())
    assert _output_shape(by_keyword) == _output_shape(positional)
