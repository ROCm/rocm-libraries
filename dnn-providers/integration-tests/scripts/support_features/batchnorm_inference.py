# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``BatchnormInferenceAttributes`` (batch norm, inference).

The features come from ``batchnorm_inference_attributes.fbs``: ``dtype``,
``precision``, ``spatial`` and ``layout``, read from ``x`` and ``y`` as
``batchnorm`` describes. ``y`` is often the virtual fp32 intermediate into a
fused pointwise.

``scale``, ``bias``, ``mean`` and ``inv_variance`` are required and, as the
API requires, ``[1, C, 1, ...]``; there is nothing to choose, so they drive no
feature. Unlike training, the API puts no floor on the values per channel:
the statistics are given, not computed from the batch.
"""

from __future__ import annotations

from support_features import FeatureError, batchnorm

NODE_TYPE = "BatchnormInferenceAttributes"
SCHEMA_FILE = "batchnorm_inference_attributes.fbs"
SCHEMA_TABLE = "BatchnormInferenceAttributes"

X, Y = "x_tensor_uid", "y_tensor_uid"
PER_CHANNEL = (
    "scale_tensor_uid",
    "bias_tensor_uid",
    "mean_tensor_uid",
    "inv_variance_tensor_uid",
)

FEATURES = batchnorm.catalog(X, (X, Y))

NOT_FEATURES = {port: batchnorm.REQUIRED_PER_CHANNEL for port in PER_CHANNEL}


def normalize(
    node: dict, tensors_by_uid: dict, per_channel: tuple[str, ...]
) -> set[str]:
    """The features of an inference node whose per-channel ports are ``per_channel``.

    Shared with ``BatchnormInferenceAttributesVarianceExt``, which takes the
    variance in place of its inverse.
    """
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (X, Y, *per_channel):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")

    features, dims = batchnorm.io_features(node, tensors_by_uid, X, (X, Y))
    for port in per_channel:
        batchnorm.check_channel_only(tensors_by_uid[by_port[port]], dims[1], port)
    return features


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    return normalize(node, tensors_by_uid, PER_CHANNEL)
