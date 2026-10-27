# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Features shared by the batch-normalization ops.

``BatchnormAttributes`` (training, forward), ``BatchnormBackwardAttributes``,
``BatchnormInferenceAttributes`` and ``BatchnormInferenceAttributesVarianceExt``
each normalize an activation per channel. Their serializers
(``utilities/json/Batchnorm*Attributes.hpp``) write every port under
``inputs`` / ``outputs``, an unset optional port as null. Each op names its
activation-shaped ports -- ``x`` and ``y``, or ``dy``, ``x`` and ``dx`` -- and
these groups are read from them:

- ``dtype`` (exclusive): the dtype of the op's data input.
- ``precision:mixed``: the non-virtual activation tensors do not all share one
  dtype. A virtual ``y`` -- the fp32 intermediate into a fused pointwise -- is
  not counted: the caller never sees it.
- ``spatial`` (exclusive): 0 to 3 spatial dims after ``[N, C]``, 0 as
  ``spatial:none``. The API's floor is ``[N, C]``; MIOpen starts at one
  spatial dim, but the API is the denominator.
- ``layout`` (exclusive): channels first or channels last, read from the
  strides as ``convolution.tensor_layouts`` reads them. If no activation
  tensor tells, the graph adds no layout feature.

The per-channel tensors (saved and running statistics, and the scale and bias
wherever the API requires a channel-only shape) are ``[1, C, 1, ...]``, as
``validateChannelOnlyTensorShape`` requires, and fp32. Both providers read
them as fp32 only, so another dtype is not catalogued and raises.
"""

from __future__ import annotations

from typing import Callable

from support_features import Feature, FeatureError, convolution
from support_matrix_units import dtype_label

DTYPES = convolution.DTYPES
SPATIAL = {0: "spatial:none", **convolution.SPATIAL}
CHANNEL_DTYPE = "fp32"

# NOT_FEATURES reason for a per-channel port the API requires.
REQUIRED_PER_CHANNEL = "required and always [1, C, 1, ...] fp32: nothing to choose"


def catalog(data: str, io: tuple[str, ...]) -> tuple[Feature, ...]:
    return (
        *(Feature(f"dtype:{dtype}", (data,)) for dtype in DTYPES),
        Feature(
            "precision:mixed",
            io,
            "the non-virtual activation tensors do not all share one dtype",
        ),
        *(Feature(spatial, io) for spatial in SPATIAL.values()),
        Feature("layout:nchw", io, "channels first"),
        Feature("layout:nhwc", io, "channels last"),
    )


def bound(node: dict) -> dict:
    """Port -> uid (or null, or a list of uids) for every port the node writes."""
    return {**(node.get("inputs") or {}), **(node.get("outputs") or {})}


def binds(node: dict) -> Callable[[str], bool]:
    """``has(port)``: whether the node binds a tensor to that port."""
    by_port = bound(node)
    return lambda port: by_port.get(port) is not None


def io_features(
    node: dict, tensors_by_uid: dict, data: str, io: tuple[str, ...]
) -> tuple[set[str], list[int]]:
    """The shared groups, and the dims of ``data``.

    Raises where the API would reject the node (activation tensors of
    different shapes, fewer than two dims) or where the catalog names no value.
    """
    by_port = bound(node)
    tensors = [tensors_by_uid[by_port[port]] for port in io]
    dims = tensors_by_uid[by_port[data]]["dims"]
    if any(tensor["dims"] != dims for tensor in tensors):
        shapes = {port: t["dims"] for port, t in zip(io, tensors)}
        raise FeatureError(f"activation shapes differ: {shapes}")
    spatial = len(dims) - 2
    if spatial not in SPATIAL:
        raise FeatureError(f"{data} dims {dims}: {spatial} spatial dims")

    dtype = dtype_label(tensors_by_uid[by_port[data]].get("data_type"))
    if dtype not in DTYPES:
        raise FeatureError(f"{data} dtype {dtype!r} is not catalogued")
    labels = {dtype_label(t.get("data_type")) for t in tensors}
    stray = labels - set(DTYPES)
    if stray:
        raise FeatureError(
            f"tensor dtype(s) {sorted(map(str, stray))} are not catalogued"
        )
    seen = {dtype_label(t.get("data_type")) for t in tensors if not t.get("virtual")}

    features = {f"dtype:{dtype}", SPATIAL[spatial]}
    if len(seen) > 1:
        features.add("precision:mixed")
    telling = [
        fits for fits in map(convolution.tensor_layouts, tensors) if len(fits) == 1
    ]
    agreed = set().union(*telling)
    if len(agreed) > 1:
        raise FeatureError("the activation tensors' layouts disagree")
    features.update(f"layout:{name}" for name in agreed)
    return features, dims


def check_channel_dtype(tensor: dict, port: str) -> None:
    if dtype_label(tensor.get("data_type")) != CHANNEL_DTYPE:
        raise FeatureError(
            f"{port} dtype {tensor.get('data_type')!r} is not catalogued "
            f"(per-channel tensors are {CHANNEL_DTYPE})"
        )


def check_channel_only(tensor: dict, channels: int, port: str) -> None:
    """``tensor`` is ``[1, C, 1, ...]`` and fp32, or this raises."""
    dims = tensor["dims"]
    if len(dims) < 2 or dims[:2] != [1, channels] or any(d != 1 for d in dims[2:]):
        raise FeatureError(f"{port} dims {dims} are not [1, {channels}, 1, ...]")
    check_channel_dtype(tensor, port)


def check_batch_statistics(dims: list[int]) -> None:
    """Training statistics need more than one value per channel.

    ``validateBatchNormTrainingSpatialDimensions`` rejects N times the spatial
    dims equal to 1: the batch variance would be undefined.
    """
    values = dims[0]
    for dim in dims[2:]:
        values *= dim
    if values <= 1:
        raise FeatureError(f"dims {dims} leave one value per channel")
