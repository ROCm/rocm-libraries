# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``BatchnormBackwardAttributes`` (batch norm, backward).

The features come from ``batchnorm_backward_attributes.fbs``. ``dtype``,
``precision``, ``spatial`` and ``layout`` are read from ``dy``, ``x`` and ``dx``
as ``batchnorm`` describes, the dtype from ``x``: ``dy`` is often the virtual
fp32 gradient out of a fused pointwise backward, which the caller never sees.
The rest:

- ``saved_stats:on``: the mean and inverse variance the forward pass saved.
  The API takes both or neither; without them the provider recomputes them.
- ``peer_stats:on``: statistics shared across devices (multi-GPU batch norm).

``scale``, ``dscale`` and ``dbias`` are required and, as the API requires,
``[1, C, 1, ...]``; there is nothing to choose, so they drive no feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm

NODE_TYPE = "BatchnormBackwardAttributes"
SCHEMA_FILE = "batchnorm_backward_attributes.fbs"
SCHEMA_TABLE = "BatchnormBackwardAttributes"

DY, X, DX = "dy_tensor_uid", "x_tensor_uid", "dx_tensor_uid"
SCALE = "scale_tensor_uid"
GRADIENTS = ("dscale_tensor_uid", "dbias_tensor_uid")
SAVED_STATS = ("mean_tensor_uid", "inv_variance_tensor_uid")
PEER_STATS = "peer_stats_tensor_uid"

FEATURES = (
    *batchnorm.catalog(X, (DY, X, DX)),
    Feature(
        "saved_stats:on",
        SAVED_STATS,
        "the forward pass's mean and inverse variance, not recomputed",
    ),
    Feature("peer_stats:on", (PEER_STATS,), "statistics shared across devices"),
)

NOT_FEATURES = {port: batchnorm.REQUIRED_PER_CHANNEL for port in (SCALE, *GRADIENTS)}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (DY, X, DX, SCALE, *GRADIENTS):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")

    features, dims = batchnorm.io_features(node, tensors_by_uid, X, (DY, X, DX))
    batchnorm.check_batch_statistics(dims)
    channels = dims[1]

    for port in (SCALE, *GRADIENTS):
        batchnorm.check_channel_only(tensors_by_uid[by_port[port]], channels, port)

    saved = [port for port in SAVED_STATS if has(port)]
    if saved and len(saved) != len(SAVED_STATS):
        raise FeatureError(f"saved statistics {saved} without the rest")
    if saved:
        for port in SAVED_STATS:
            batchnorm.check_channel_only(tensors_by_uid[by_port[port]], channels, port)
        features.add("saved_stats:on")

    if by_port.get(PEER_STATS):
        features.add("peer_stats:on")
    return features
