# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for Graph serialization (to_json/from_json, to_binary/from_binary).

Every graph operation gets one topology-only JSON case and one topology-only
binary case in `_OPERATION_CASES` below, asserting its serialized "type" tag
and that a restored graph re-serializes identically. This is the single
source of per-operation serialization coverage for the whole binding surface
-- new operations are added here, not in a separate class.
"""

import json

import pytest

import hipdnn_frontend as hipdnn

from .helpers import create_float_graph


def _epsilon_tensor():
    tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
    tensor.set_value(1e-5)
    return tensor


def _pointwise_graph():
    graph = create_float_graph()
    a = hipdnn.Tensor.create([2, 4, 8, 8], hipdnn.DataType.FLOAT)
    b = hipdnn.Tensor.create([2, 4, 8, 8], hipdnn.DataType.FLOAT)
    out = graph.pointwise(
        a, b, hipdnn.PointwiseAttributes().set_mode(hipdnn.PointwiseMode.ADD)
    )
    out.set_output(True)
    return graph


def _matmul_graph():
    graph = create_float_graph()
    a = hipdnn.Tensor.create([4, 3], hipdnn.DataType.FLOAT)
    b = hipdnn.Tensor.create([3, 5], hipdnn.DataType.FLOAT)
    c = graph.matmul(a, b, hipdnn.MatmulAttributes())
    c.set_output(True)
    return graph


def _conv_fprop_graph():
    graph = create_float_graph()
    x = hipdnn.Tensor.create([1, 2, 4, 4], hipdnn.DataType.FLOAT)
    weight = hipdnn.Tensor.create([2, 2, 3, 3], hipdnn.DataType.FLOAT)
    attrs = (
        hipdnn.ConvFpropAttributes()
        .set_padding([1, 1])
        .set_stride([1, 1])
        .set_dilation([1, 1])
    )
    y = graph.conv_fprop(x, weight, attrs)
    y.set_output(True)
    return graph


def _conv_dgrad_graph():
    graph = create_float_graph()
    dy = hipdnn.Tensor.create([1, 2, 4, 4], hipdnn.DataType.FLOAT)
    weight = hipdnn.Tensor.create([2, 2, 3, 3], hipdnn.DataType.FLOAT)
    attrs = (
        hipdnn.ConvDgradAttributes()
        .set_pre_padding([1, 1])
        .set_post_padding([1, 1])
        .set_stride([1, 1])
        .set_dilation([1, 1])
    )
    dx = graph.conv_dgrad(dy, weight, attrs)
    dx.set_dim([1, 2, 4, 4])
    dx.set_output(True)
    return graph


def _conv_wgrad_graph():
    graph = create_float_graph()
    dy = hipdnn.Tensor.create([1, 2, 4, 4], hipdnn.DataType.FLOAT)
    x = hipdnn.Tensor.create([1, 2, 4, 4], hipdnn.DataType.FLOAT)
    attrs = (
        hipdnn.ConvWgradAttributes()
        .set_pre_padding([1, 1])
        .set_post_padding([1, 1])
        .set_stride([1, 1])
        .set_dilation([1, 1])
    )
    dw = graph.conv_wgrad(dy, x, attrs)
    dw.set_dim([2, 2, 3, 3])
    dw.set_output(True)
    return graph


def _batchnorm_graph():
    graph = create_float_graph()
    x = hipdnn.Tensor.create([2, 4, 4, 4], hipdnn.DataType.FLOAT)
    scale = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    bias = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    attrs = hipdnn.BatchnormAttributes().set_epsilon(_epsilon_tensor())
    y, mean, inv_variance, _next_mean, _next_var = graph.batchnorm(
        x, scale, bias, attrs
    )
    y.set_output(True)
    mean.set_output(True)
    inv_variance.set_output(True)
    return graph


def _batchnorm_backward_graph():
    graph = create_float_graph()
    dy = hipdnn.Tensor.create([2, 4, 4, 4], hipdnn.DataType.FLOAT)
    x = hipdnn.Tensor.create([2, 4, 4, 4], hipdnn.DataType.FLOAT)
    scale = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    dx, dscale, dbias = graph.batchnorm_backward(
        dy, x, scale, hipdnn.BatchnormBackwardAttributes()
    )
    dx.set_output(True)
    dscale.set_output(True)
    dbias.set_output(True)
    return graph


def _batchnorm_inference_graph():
    graph = create_float_graph()
    x = hipdnn.Tensor.create([2, 4, 4, 4], hipdnn.DataType.FLOAT)
    mean = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    inv_variance = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    scale = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    bias = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    y = graph.batchnorm_inference(
        x, mean, inv_variance, scale, bias, hipdnn.BatchnormInferenceAttributes()
    )
    y.set_output(True)
    return graph


def _batchnorm_inference_variance_graph():
    graph = create_float_graph()
    x = hipdnn.Tensor.create([2, 4, 4, 4], hipdnn.DataType.FLOAT)
    mean = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    variance = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    scale = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    bias = hipdnn.Tensor.create([1, 4, 1, 1], hipdnn.DataType.FLOAT)
    epsilon = hipdnn.Tensor()
    epsilon.set_value(1e-5)
    y = graph.batchnorm_inference_variance_ext(
        x,
        mean,
        variance,
        scale,
        bias,
        epsilon,
        hipdnn.BatchnormInferenceAttributesVarianceExt(),
    )
    y.set_output(True)
    return graph


def _layernorm_graph():
    graph = create_float_graph()
    graph.layernorm(
        hipdnn.Tensor.create([2, 6, 4], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([6, 4], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([6, 4], hipdnn.DataType.FLOAT),
        hipdnn.LayernormAttributes()
        .set_epsilon(_epsilon_tensor())
        .set_forward_phase(hipdnn.NormFwdPhase.INFERENCE),
    )
    return graph


def _layernorm_backward_graph():
    graph = create_float_graph()
    graph.layernorm_backward(
        hipdnn.Tensor.create([16, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([16, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.LayernormBackwardAttributes()
        .set_mean(hipdnn.Tensor.create([16, 1, 1, 1], hipdnn.DataType.FLOAT))
        .set_inv_variance(hipdnn.Tensor.create([16, 1, 1, 1], hipdnn.DataType.FLOAT))
        .set_epsilon(_epsilon_tensor()),
    )
    return graph


def _rmsnorm_graph():
    graph = create_float_graph()
    graph.rmsnorm(
        hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.RMSNormAttributes()
        .set_epsilon(_epsilon_tensor())
        .set_forward_phase(hipdnn.NormFwdPhase.TRAINING),
    )
    return graph


def _rmsnorm_backward_graph():
    graph = create_float_graph()
    graph.rmsnorm_backward(
        hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([1, 1, 1, 1], hipdnn.DataType.FLOAT),
        hipdnn.RMSNormBackwardAttributes(),
    )
    return graph


def _block_scale_dequantize_graph():
    graph = create_float_graph()
    graph.block_scale_dequantize(
        hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 2, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.BlockScaleDequantizeAttributes().set_block_size([32]),
    )
    return graph


def _block_scale_quantize_graph():
    graph = create_float_graph()
    graph.block_scale_quantize(
        hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT),
        hipdnn.BlockScaleQuantizeAttributes().set_block_size(32),
    )
    return graph


def _reduction_graph():
    graph = create_float_graph()
    output = graph.reduction(
        hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT),
        hipdnn.ReductionAttributes().set_mode(hipdnn.ReductionMode.ADD),
    )
    output.set_dim([1, 8]).set_stride([8, 1])
    return graph


def _moe_grouped_matmul_graph():
    graph = create_float_graph()
    graph.moe_grouped_matmul(
        hipdnn.Tensor.create([1, 8, 16], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 16, 32], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 1, 1], hipdnn.DataType.INT32),
        hipdnn.Tensor.create([1, 8, 1], hipdnn.DataType.INT32),
        hipdnn.Tensor.create([1, 8, 1], hipdnn.DataType.INT32),
        hipdnn.MoeGroupedMatmulAttributes()
        .set_mode(hipdnn.MoeGroupedMatmulMode.SCATTER)
        .set_top_k(2),
    )
    return graph


def _custom_op_graph():
    graph = create_float_graph()
    outputs = graph.custom_op(
        [
            hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT),
            hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT),
        ],
        2,
        hipdnn.CustomOpAttributes().set_custom_op_id("example.identity"),
    )
    for output in outputs:
        output.set_dim([4, 8]).set_stride([8, 1])
    return graph


def _resample_fwd_graph():
    graph = create_float_graph()
    graph.resample_fwd(
        hipdnn.Tensor.create([1, 3, 4, 4], hipdnn.DataType.FLOAT),
        hipdnn.ResampleFwdAttributes()
        .set_resample_mode(hipdnn.ResampleMode.MAXPOOL)
        .set_padding_mode(hipdnn.PaddingMode.ZERO_PAD)
        .set_pre_padding([0, 0])
        .set_post_padding([0, 0])
        .set_stride([2, 2])
        .set_window([2, 2]),
    )
    return graph


def _resample_bwd_graph():
    graph = create_float_graph()
    graph.resample_bwd(
        hipdnn.Tensor.create([1, 3, 16, 16], hipdnn.DataType.FLOAT),
        hipdnn.ResampleBwdAttributes()
        .set_resample_mode(hipdnn.ResampleMode.AVGPOOL_EXCLUDE_PADDING)
        .set_padding_mode(hipdnn.PaddingMode.ZERO_PAD)
        .set_pre_padding([1, 1])
        .set_post_padding([1, 1])
        .set_stride([2, 2])
        .set_window([3, 3]),
    )
    return graph


def _sdpa_graph():
    graph = create_float_graph()
    graph.sdpa(
        hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT),
        hipdnn.SdpaAttributes().set_generate_stats(True),
    )
    return graph


def _sdpa_backward_graph():
    graph = create_float_graph()
    graph.sdpa_backward(
        hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT),
        hipdnn.Tensor.create([2, 8, 16, 1], hipdnn.DataType.FLOAT),
        hipdnn.SdpaBackwardAttributes(),
    )
    return graph


# One entry per graph-construction operation exposed by the Python bindings.
# `expected_type` is the flatbuffers union tag written into "type" -- for most
# operations this matches the frontend Attributes class name, but the three
# convolution directions use distinct schema names (Fwd/Bwd/Wrw) unrelated to
# their frontend class names (ConvFprop/ConvDgrad/ConvWgrad); verified
# empirically, do not rename to "look consistent".
_OPERATION_CASES = (
    ("pointwise", _pointwise_graph, "PointwiseAttributes"),
    ("matmul", _matmul_graph, "MatmulAttributes"),
    ("conv_fprop", _conv_fprop_graph, "ConvolutionFwdAttributes"),
    ("conv_dgrad", _conv_dgrad_graph, "ConvolutionBwdAttributes"),
    ("conv_wgrad", _conv_wgrad_graph, "ConvolutionWrwAttributes"),
    ("batchnorm", _batchnorm_graph, "BatchnormAttributes"),
    ("batchnorm_backward", _batchnorm_backward_graph, "BatchnormBackwardAttributes"),
    ("batchnorm_inference", _batchnorm_inference_graph, "BatchnormInferenceAttributes"),
    (
        "batchnorm_inference_variance_ext",
        _batchnorm_inference_variance_graph,
        "BatchnormInferenceAttributesVarianceExt",
    ),
    ("layernorm", _layernorm_graph, "LayernormAttributes"),
    ("layernorm_backward", _layernorm_backward_graph, "LayernormBackwardAttributes"),
    ("rmsnorm", _rmsnorm_graph, "RMSNormAttributes"),
    ("rmsnorm_backward", _rmsnorm_backward_graph, "RMSNormBackwardAttributes"),
    (
        "block_scale_dequantize",
        _block_scale_dequantize_graph,
        "BlockScaleDequantizeAttributes",
    ),
    (
        "block_scale_quantize",
        _block_scale_quantize_graph,
        "BlockScaleQuantizeAttributes",
    ),
    ("reduction", _reduction_graph, "ReductionAttributes"),
    ("moe_grouped_matmul", _moe_grouped_matmul_graph, "MoeGroupedMatmulAttributes"),
    ("custom_op", _custom_op_graph, "CustomOpAttributes"),
    ("resample_fwd", _resample_fwd_graph, "ResampleFwdAttributes"),
    ("resample_bwd", _resample_bwd_graph, "ResampleBwdAttributes"),
    ("sdpa", _sdpa_graph, "SdpaAttributes"),
    ("sdpa_backward", _sdpa_backward_graph, "SdpaBackwardAttributes"),
)
_OPERATION_CASE_PARAMS = [(build, expected) for _, build, expected in _OPERATION_CASES]
_OPERATION_CASE_IDS = [name for name, _, _ in _OPERATION_CASES]


def _skip_if_sdpa_disabled(expected_type):
    if expected_type.startswith("Sdpa") and not hasattr(hipdnn.Graph, "sdpa"):
        pytest.skip("SDPA disabled")


class TestJsonSerialization:
    """Topology-only JSON round-trips for every graph operation (no GPU required)."""

    @pytest.mark.parametrize(
        "build_graph, expected_type", _OPERATION_CASE_PARAMS, ids=_OPERATION_CASE_IDS
    )
    def test_json_round_trip_preserves_operation_type(self, build_graph, expected_type):
        _skip_if_sdpa_disabled(expected_type)

        graph = build_graph()
        assert graph.validate().is_good()
        json_str = graph.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        nodes = json.loads(json_str)["nodes"]
        assert len(nodes) == 1
        assert nodes[0]["type"] == expected_type

        restored = hipdnn.Graph()
        assert restored.from_json(json_str).is_good()
        assert restored.to_json() == json_str


class TestBinarySerialization:
    """Topology-only binary round-trips for every graph operation (no GPU required)."""

    @pytest.mark.parametrize(
        "build_graph, expected_type", _OPERATION_CASE_PARAMS, ids=_OPERATION_CASE_IDS
    )
    def test_binary_round_trip_is_stable(self, build_graph, expected_type):
        _skip_if_sdpa_disabled(expected_type)

        graph = build_graph()
        assert graph.validate().is_good()
        data = graph.to_binary()
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = hipdnn.Graph()
        assert restored.from_binary(data).is_good()
        assert restored.to_binary() == data


@pytest.mark.gpu
class TestHandleFinalizedDeserialization:
    """Handle-finalized deserialization overloads (require GPU).

    Uses one representative operation (pointwise) -- this exercises the
    handle-finalization mechanism itself, not per-operation serialization
    content, which is already covered exhaustively above.
    """

    def test_from_json_with_handle_finalizes(self):
        """from_json(handle, json) deserializes and finalizes for execution."""
        graph = _pointwise_graph()
        assert graph.validate().is_good()
        json_str = graph.to_json()

        handle = hipdnn.create_handle()
        restored = hipdnn.Graph()
        assert restored.from_json(handle, json_str).is_good()

    def test_from_binary_with_handle_finalizes(self):
        """from_binary(handle, data) deserializes and finalizes for execution."""
        graph = _pointwise_graph()
        assert graph.validate().is_good()
        data = graph.to_binary()

        handle = hipdnn.create_handle()
        restored = hipdnn.Graph()
        assert restored.from_binary(handle, data).is_good()
