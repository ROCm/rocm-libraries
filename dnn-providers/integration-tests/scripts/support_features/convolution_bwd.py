# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``ConvolutionBwdAttributes`` (convolution, data grad).

The features come from ``convolution_bwd_attributes.fbs`` and are read as
``convolution`` describes. ``dtype`` is the dtype of the input ``dy``.
"""

from __future__ import annotations

from support_features import convolution

NODE_TYPE = "ConvolutionBwdAttributes"
SCHEMA_FILE = "convolution_bwd_attributes.fbs"
SCHEMA_TABLE = "ConvolutionBwdAttributes"

PORTS = convolution.Ports(
    activation="dx_tensor_uid",
    filter="w_tensor_uid",
    response="dy_tensor_uid",
    data="dy_tensor_uid",
)

FEATURES = convolution.catalog(PORTS)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    return convolution.features(node, tensors_by_uid, PORTS)
