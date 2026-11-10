# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``ConvolutionWrwAttributes`` (convolution, weight grad).

The features come from ``convolution_wrw_attributes.fbs`` and are read as
``convolution`` describes. ``dtype`` is the dtype of the input ``x``, the
forward data input, as for ``ConvolutionFwdAttributes``.
"""

from __future__ import annotations

from support_features import convolution

NODE_TYPE = "ConvolutionWrwAttributes"
SCHEMA_FILE = "convolution_wrw_attributes.fbs"
SCHEMA_TABLE = "ConvolutionWrwAttributes"

PORTS = convolution.Ports(
    activation="x_tensor_uid",
    filter="dw_tensor_uid",
    response="dy_tensor_uid",
    data="x_tensor_uid",
)

FEATURES = convolution.catalog(PORTS)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    return convolution.features(node, tensors_by_uid, PORTS)
