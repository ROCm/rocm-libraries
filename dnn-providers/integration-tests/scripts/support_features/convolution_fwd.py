# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``ConvolutionFwdAttributes`` (convolution, forward).

The features come from ``convolution_fwd_attributes.fbs`` and are read as
``convolution`` describes. ``dtype`` is the dtype of the input ``x``.
"""

from __future__ import annotations

from support_features import convolution

NODE_TYPE = "ConvolutionFwdAttributes"
SCHEMA_FILE = "convolution_fwd_attributes.fbs"
SCHEMA_TABLE = "ConvolutionFwdAttributes"

PORTS = convolution.Ports(
    activation="x_tensor_uid",
    filter="w_tensor_uid",
    response="y_tensor_uid",
    data="x_tensor_uid",
)

FEATURES = convolution.catalog(PORTS)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    return convolution.features(node, tensors_by_uid, PORTS)
