# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``BatchnormAttributes`` (batch norm, training forward).

The features come from ``batchnorm_attributes.fbs``. ``dtype``,
``precision``, ``spatial`` and ``layout`` are read from ``x`` and ``y`` as
``batchnorm`` describes; the rest, by group:

- ``affine`` (exclusive): the shape ``scale`` and ``bias`` share.
  ``full_rank`` is ``[1, C, 1, ...]`` at the rank of ``x``; ``reduced_rank``
  is ``[C, 1, ...]`` at a lower rank, which the hip-kernel provider reads with
  the channel first. The API leaves the shape to the providers ("any
  broadcastable shape") but rejects per-activation mode (a spatial dim above
  1); a shape neither provider names raises, as does a non-fp32 affine.
- ``epsilon`` (exclusive): ``fixed`` (a constant baked into the graph) or
  ``runtime`` (passed with each execution), as ``scalar_source`` reads it.
- ``running_stats:on``: the four running statistics, which the API takes all
  or none of.
- ``momentum`` (exclusive): how the running-statistics momentum gets its
  value, as ``epsilon``. The API does not tie momentum to the running
  statistics, but neither means anything without the other, so a graph with
  one and not the other raises.
- ``out:mean``, ``out:inv_variance``: the batch statistics saved for the
  backward pass. The API takes each without the other.
- ``peer_stats:on``: statistics shared across devices (multi-GPU batch norm).

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, scalar_source

NODE_TYPE = "BatchnormAttributes"
SCHEMA_FILE = "batchnorm_attributes.fbs"
SCHEMA_TABLE = "BatchnormAttributes"

X, Y = "x_tensor_uid", "y_tensor_uid"
AFFINE = ("scale_tensor_uid", "bias_tensor_uid")
EPSILON = "epsilon_tensor_uid"
MOMENTUM = "momentum_tensor_uid"
RUNNING_STATS = (
    "prev_running_mean_tensor_uid",
    "prev_running_variance_tensor_uid",
    "next_running_mean_tensor_uid",
    "next_running_variance_tensor_uid",
)
SAVED_STATS = {
    "mean_tensor_uid": "out:mean",
    "inv_variance_tensor_uid": "out:inv_variance",
}
PEER_STATS = "peer_stats_tensor_uid"

FEATURES = (
    *batchnorm.catalog(X, (X, Y)),
    Feature("affine:full_rank", AFFINE, "scale and bias are [1, C, 1, ...]"),
    Feature(
        "affine:reduced_rank",
        AFFINE,
        "scale and bias are [C, 1, ...] below the rank of x",
    ),
    Feature("epsilon:fixed", (EPSILON,), "a constant baked into the graph"),
    Feature("epsilon:runtime", (EPSILON,), "passed with each execution"),
    Feature("running_stats:on", RUNNING_STATS, "running mean and variance updated"),
    Feature("momentum:fixed", (MOMENTUM,), "a constant baked into the graph"),
    Feature("momentum:runtime", (MOMENTUM,), "passed with each execution"),
    *(
        Feature(feature, (port,), "saved for the backward pass")
        for port, feature in SAVED_STATS.items()
    ),
    Feature("peer_stats:on", (PEER_STATS,), "statistics shared across devices"),
)

NOT_FEATURES: dict[str, str] = {}


def affine_feature(tensor: dict, x_dims: list[int], port: str) -> str:
    dims, rank, channels = tensor["dims"], len(x_dims), x_dims[1]
    if dims == [1, channels] + [1] * (rank - 2):
        feature = "affine:full_rank"
    elif 1 <= len(dims) < rank and dims == [channels] + [1] * (len(dims) - 1):
        feature = "affine:reduced_rank"
    else:
        raise FeatureError(
            f"{port} dims {dims} are neither [1, {channels}, 1, ...] at rank "
            f"{rank} nor [{channels}, 1, ...] below it"
        )
    batchnorm.check_channel_dtype(tensor, port)
    return feature


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (X, Y, *AFFINE, EPSILON):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")

    features, dims = batchnorm.io_features(node, tensors_by_uid, X, (X, Y))
    batchnorm.check_batch_statistics(dims)
    channels = dims[1]

    scale, bias = (tensors_by_uid[by_port[port]] for port in AFFINE)
    if scale["dims"] != bias["dims"]:
        raise FeatureError(f"scale dims {scale['dims']} != bias dims {bias['dims']}")
    features.update(
        affine_feature(tensors_by_uid[by_port[port]], dims, port) for port in AFFINE
    )

    features.add(f"epsilon:{scalar_source(tensors_by_uid[by_port[EPSILON]], EPSILON)}")

    running = [port for port in RUNNING_STATS if has(port)]
    if running and len(running) != len(RUNNING_STATS):
        raise FeatureError(f"running statistics {running} without the rest")
    if bool(running) != has(MOMENTUM):
        raise FeatureError(
            "momentum without running statistics"
            if has(MOMENTUM)
            else "running statistics without momentum"
        )
    if running:
        for port in RUNNING_STATS:
            batchnorm.check_channel_only(tensors_by_uid[by_port[port]], channels, port)
        features.add("running_stats:on")
        momentum = tensors_by_uid[by_port[MOMENTUM]]
        features.add(f"momentum:{scalar_source(momentum, MOMENTUM)}")

    for port, feature in SAVED_STATS.items():
        if has(port):
            batchnorm.check_channel_only(tensors_by_uid[by_port[port]], channels, port)
            features.add(feature)

    if by_port.get(PEER_STATS):
        features.add("peer_stats:on")
    return features
