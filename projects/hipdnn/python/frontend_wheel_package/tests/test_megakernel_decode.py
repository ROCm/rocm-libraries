# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for typed megakernel decode attributes and graph construction."""

import json

import pytest

import hipdnn_frontend as hipdnn

from .helpers import call_attribute_methods, create_float_graph


_PAYLOAD = {
    "model": "megakernel-standin-gfx942",
    "src": "0123456789abcdef",
    "arch": "gfx942",
    "cus": 304,
    "kv": "ROCM_ATTN",
    "block": 784,
    "M": 1,
    "tp": 1,
    "rt": 0,
    "lm_head": False,
    "nlayers": 64,
}


def _attributes():
    attributes = hipdnn.MegakernelDecodeAttributes()
    call_attribute_methods(
        attributes,
        tuple(
            (f"set_{key.lower()}", (value,), f"get_{key.lower()}", value)
            for key, value in _PAYLOAD.items()
        ),
    )
    return attributes


def test_megakernel_decode_serializes_custom_op():
    graph = create_float_graph()
    tab = hipdnn.Tensor.create([16], hipdnn.DataType.INT64).set_uid(1)
    glob = hipdnn.Tensor.create([8], hipdnn.DataType.INT64).set_uid(2)

    assert graph.megakernel_decode(tab=tab, glob=glob, attributes=_attributes()) is None
    assert graph.validate().is_good()
    (node,) = json.loads(graph.to_json())["nodes"]
    assert node["type"] == "CustomOpAttributes"
    assert node["custom_op_id"] == "megakernel.megakernel-standin-gfx942"
    assert node["input_tensor_uids"] == [1, 2]
    assert node["output_tensor_uids"] == []
    assert json.loads(bytes(node["data"])) == {"schema": 1, **_PAYLOAD}


def test_megakernel_decode_rejects_invalid_arguments():
    graph = create_float_graph()
    tab = hipdnn.Tensor.create([16], hipdnn.DataType.INT64)
    glob = hipdnn.Tensor.create([8], hipdnn.DataType.INT64)
    not_int64 = hipdnn.Tensor.create([8], hipdnn.DataType.FLOAT)
    attributes = _attributes()

    with pytest.raises(ValueError, match="INT64"):
        graph.megakernel_decode(tab, not_int64, attributes)
    with pytest.raises(ValueError, match="non-null"):
        graph.megakernel_decode(None, glob, attributes)
    with pytest.raises(ValueError, match="nlayers"):
        graph.megakernel_decode(tab, glob, attributes.set_nlayers(2**31))
    with pytest.raises(ValueError, match="requires model"):
        graph.megakernel_decode(tab, glob, hipdnn.MegakernelDecodeAttributes())
