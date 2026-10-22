# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``BatchnormInferenceAttributesVarianceExt``.

Batch norm inference that takes the variance, not its inverse, and so needs
an epsilon. The features come from
``batchnorm_inference_attributes_variance_ext.fbs``: those of
``batchnorm_inference``, read the same way, and

- ``epsilon`` (exclusive): ``fixed`` (a constant baked into the graph) or
  ``runtime`` (passed with each execution), as ``scalar_source`` reads it.

``scale``, ``bias``, ``mean`` and ``variance`` are required ``[1, C, 1, ...]``
and drive no feature.
"""

from __future__ import annotations

from support_features import (
    Feature,
    FeatureError,
    batchnorm,
    batchnorm_inference,
    scalar_source,
)

NODE_TYPE = "BatchnormInferenceAttributesVarianceExt"
SCHEMA_FILE = "batchnorm_inference_attributes_variance_ext.fbs"
SCHEMA_TABLE = "BatchnormInferenceAttributesVarianceExt"

PER_CHANNEL = (
    "scale_tensor_uid",
    "bias_tensor_uid",
    "mean_tensor_uid",
    "variance_tensor_uid",
)
EPSILON = "epsilon_tensor_uid"

FEATURES = (
    *batchnorm_inference.FEATURES,
    Feature("epsilon:fixed", (EPSILON,), "a constant baked into the graph"),
    Feature("epsilon:runtime", (EPSILON,), "passed with each execution"),
)

NOT_FEATURES = {port: batchnorm.REQUIRED_PER_CHANNEL for port in PER_CHANNEL}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    if not batchnorm.binds(node)(EPSILON):
        raise FeatureError(f"required port {EPSILON} is not bound")
    features = batchnorm_inference.normalize(node, tensors_by_uid, PER_CHANNEL)
    epsilon = tensors_by_uid[batchnorm.bound(node)[EPSILON]]
    features.add(f"epsilon:{scalar_source(epsilon, EPSILON)}")
    return features
