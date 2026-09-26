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

from .graph_builders import (
    build_batchnorm_backward_graph,
    build_batchnorm_inference_graph,
    build_batchnorm_inference_variance_graph,
    build_batchnorm_training_graph,
    build_block_scale_dequantize_graph,
    build_block_scale_quantize_graph,
    build_conv_dgrad_graph,
    build_conv_fprop_graph,
    build_conv_wgrad_graph,
    build_custom_op_graph,
    build_layernorm_backward_graph,
    build_layernorm_graph,
    build_matmul_graph,
    build_moe_grouped_matmul_graph,
    build_pointwise_add_graph,
    build_reduction_graph,
    build_resample_bwd_graph,
    build_resample_fwd_graph,
    build_rmsnorm_backward_graph,
    build_rmsnorm_graph,
    build_sdpa_backward_graph,
    build_sdpa_graph,
)

# One entry per graph-construction operation exposed by the Python bindings.
# The builders in .graph_builders return (graph, *tensors) for execution
# callers; serialization only needs the topology, hence the `[0]` adapters.
# `expected_type` is the flatbuffers union tag written into "type" -- for most
# operations this matches the frontend Attributes class name, but the three
# convolution directions use distinct schema names (Fwd/Bwd/Wrw) unrelated to
# their frontend class names (ConvFprop/ConvDgrad/ConvWgrad); verified
# empirically, do not rename to "look consistent".
_OPERATION_CASES = (
    ("pointwise", lambda: build_pointwise_add_graph()[0], "PointwiseAttributes"),
    ("matmul", lambda: build_matmul_graph()[0], "MatmulAttributes"),
    ("conv_fprop", lambda: build_conv_fprop_graph()[0], "ConvolutionFwdAttributes"),
    ("conv_dgrad", lambda: build_conv_dgrad_graph()[0], "ConvolutionBwdAttributes"),
    ("conv_wgrad", lambda: build_conv_wgrad_graph()[0], "ConvolutionWrwAttributes"),
    ("batchnorm", lambda: build_batchnorm_training_graph()[0], "BatchnormAttributes"),
    (
        "batchnorm_backward",
        lambda: build_batchnorm_backward_graph()[0],
        "BatchnormBackwardAttributes",
    ),
    (
        "batchnorm_inference",
        lambda: build_batchnorm_inference_graph()[0],
        "BatchnormInferenceAttributes",
    ),
    (
        "batchnorm_inference_variance_ext",
        lambda: build_batchnorm_inference_variance_graph()[0],
        "BatchnormInferenceAttributesVarianceExt",
    ),
    ("layernorm", lambda: build_layernorm_graph()[0], "LayernormAttributes"),
    (
        "layernorm_backward",
        lambda: build_layernorm_backward_graph()[0],
        "LayernormBackwardAttributes",
    ),
    ("rmsnorm", lambda: build_rmsnorm_graph()[0], "RMSNormAttributes"),
    (
        "rmsnorm_backward",
        lambda: build_rmsnorm_backward_graph()[0],
        "RMSNormBackwardAttributes",
    ),
    (
        "block_scale_dequantize",
        lambda: build_block_scale_dequantize_graph()[0],
        "BlockScaleDequantizeAttributes",
    ),
    (
        "block_scale_quantize",
        lambda: build_block_scale_quantize_graph()[0],
        "BlockScaleQuantizeAttributes",
    ),
    ("reduction", lambda: build_reduction_graph()[0], "ReductionAttributes"),
    (
        "moe_grouped_matmul",
        lambda: build_moe_grouped_matmul_graph()[0],
        "MoeGroupedMatmulAttributes",
    ),
    ("custom_op", lambda: build_custom_op_graph()[0], "CustomOpAttributes"),
    ("resample_fwd", lambda: build_resample_fwd_graph()[0], "ResampleFwdAttributes"),
    ("resample_bwd", lambda: build_resample_bwd_graph()[0], "ResampleBwdAttributes"),
    ("sdpa", lambda: build_sdpa_graph()[0], "SdpaAttributes"),
    ("sdpa_backward", lambda: build_sdpa_backward_graph()[0], "SdpaBackwardAttributes"),
)
_OPERATION_CASE_PARAMS = [(build, expected) for _, build, expected in _OPERATION_CASES]
_OPERATION_CASE_IDS = [name for name, _, _ in _OPERATION_CASES]


def _skip_if_sdpa_disabled(expected_type):
    if expected_type.startswith("Sdpa") and not hasattr(hipdnn.Graph, "sdpa"):
        pytest.skip("SDPA disabled")


# CustomOpAttributes serializes flat input_tensor_uids/output_tensor_uids arrays
# rather than per-role keys, so there are no <role>_tensor_uid entries to check.
_OPS_WITHOUT_TENSOR_ROLES = frozenset({"CustomOpAttributes"})


def _node_tensor_roles(node):
    """Every ``<role>_tensor_uid`` entry a serialized node exposes.

    The layout is not uniform. Most operations nest their tensors under
    ``inputs``/``outputs`` maps, while reduction and both resample directions
    write the keys straight onto the node (ReductionAttributes.hpp,
    ResampleFwdAttributes.hpp), so all three levels are scanned. Values are
    usually a scalar uid, but vector-valued roles -- batchnorm's peer_stats --
    carry a list of them.
    """
    roles = {}
    for section in (node, node.get("inputs") or {}, node.get("outputs") or {}):
        roles.update(
            (key, value)
            for key, value in section.items()
            if key.endswith("_tensor_uid")
        )
    return roles


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
        # Graph has no __eq__; re-serializing through both formats is the
        # closest available equality check against the original. to_binary()
        # catches state the JSON writer echoes but from_json() failed to load.
        assert restored.to_json() == json_str
        assert restored.to_binary() == graph.to_binary()

    @pytest.mark.parametrize(
        "build_graph, expected_type", _OPERATION_CASE_PARAMS, ids=_OPERATION_CASE_IDS
    )
    def test_each_tensor_lands_in_its_named_role(self, build_graph, expected_type):
        """Every ``<role>_tensor_uid`` points at the tensor named ``<role>``.

        Catches a binding that forwards its positional arguments into the wrong
        attribute slots -- ``layernorm(x, scale, bias)`` filling bias from
        scale. Sibling tensors usually share a shape, so validation, lowering,
        and stubbed execution all still pass; the serialized role map is the
        only place that wiring is observable from Python.

        Builders name each tensor after the role it belongs in; tensors the
        frontend synthesizes carry no name and are skipped.
        """
        _skip_if_sdpa_disabled(expected_type)

        document = json.loads(build_graph().to_json())
        name_by_uid = {tensor["uid"]: tensor["name"] for tensor in document["tensors"]}
        node = document["nodes"][0]
        roles = _node_tensor_roles(node)

        if expected_type in _OPS_WITHOUT_TENSOR_ROLES:
            assert not roles
            return
        assert roles, f"{expected_type} serialized no tensor roles"

        for key, value in roles.items():
            expected_name = key.removesuffix("_tensor_uid")
            uids = value if isinstance(value, list) else [value]
            for uid in uids:
                actual = name_by_uid.get(uid)
                if not actual:
                    continue
                assert actual == expected_name, (
                    f"{expected_type}.{key} -> uid {uid} is named {actual!r}, "
                    f"expected {expected_name!r}"
                )


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
        assert restored.to_json() == graph.to_json()


@pytest.mark.gpu
class TestHandleFinalizedDeserialization:
    """Handle-finalized deserialization overloads (require GPU).

    Uses one representative operation (pointwise) -- this exercises the
    handle-finalization mechanism itself, not per-operation serialization
    content, which is already covered exhaustively above.
    """

    def test_from_json_with_handle_finalizes(self):
        """from_json(handle, json) deserializes and finalizes for execution."""
        graph = build_pointwise_add_graph()[0]
        assert graph.validate().is_good()
        json_str = graph.to_json()

        handle = hipdnn.create_handle()
        restored = hipdnn.Graph()
        assert restored.from_json(handle, json_str).is_good()

    def test_from_binary_with_handle_finalizes(self):
        """from_binary(handle, data) deserializes and finalizes for execution."""
        graph = build_pointwise_add_graph()[0]
        assert graph.validate().is_good()
        data = graph.to_binary()

        handle = hipdnn.create_handle()
        restored = hipdnn.Graph()
        assert restored.from_binary(handle, data).is_good()
