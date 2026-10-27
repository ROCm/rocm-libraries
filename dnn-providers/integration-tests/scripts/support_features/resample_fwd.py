# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``ResampleFwdAttributes``.

Pooling: each spatial cell of ``y`` reduces a ``window`` of ``x``, placed
every ``stride`` cells over ``x`` padded by ``pre_padding`` and
``post_padding``; N and C pass through. The features come from
``resample_fwd_attributes.fbs`` and ``resample_common.fbs``. As for
reduction, the serializer (``utilities/json/ResampleFwdAttributes.hpp``)
writes the ports at the top of the node; it leaves an unset
``index_tensor_uid`` and ``generate_index`` out, and the reader takes a
missing ``generate_index`` as false. The hip-kernel provider (the only one
with the op) and the CPU reference agree on what they take, and the
catalog takes the same. By group:

- ``mode`` (exclusive): the pooling, by its serialized name.
- ``padding_mode`` (exclusive, with ``maxpool`` when a window reaches the
  padding): the value of a padded cell, ``neg_inf_pad`` (skipped) or
  ``zero_pad`` (a 0 candidate). Only then is it read: the average modes say
  themselves whether padded cells count, and a window that stays on ``x``
  reads none.
- ``dtype`` (exclusive): the dtype of ``x``, which ``y`` shares.
- ``spatial`` (exclusive): 2 or 3 spatial dims.
- ``layout`` (exclusive, when the strides tell): ``nchw`` or ``nhwc``, read
  from ``x``, ``y`` and the index, as for convolution.
- ``padding:symmetric``, ``padding:asymmetric``: non-zero padding, with
  pre = post in every dim or not, as for convolution.
- ``stride:strided``: a stride above 1 in some dim.
- ``window:overlapping``, ``window:gapped``: a stride below the window in
  some dim (windows share cells), or above it (cells no window reads).
- ``index`` (exclusive, with ``maxpool``): ``generated`` (the index tensor
  bound and ``generate_index`` on: each ``y`` cell's argmax, for the
  backward) or ``none``.

The API requires ``x``, ``y`` and a mode and padding mode other than their
``NOT_SET``; with ``generate_index`` on under ``maxpool``, an index. The
provider and the CPU reference require ``x`` and ``y`` of rank 4 or 5 (the
API allows 3) and of one dtype; a padding, stride and window entry for each
spatial dim, the padding 0 or above and the stride and window above 0; ``y``
of ``x``'s N and C, each spatial dim ``(x + pre + post - window) / stride +
1`` and above 0; and an index only under ``maxpool``, of ``y``'s dims and
int32 (the API allows int8), and only with ``generate_index`` on. A graph
that breaks these raises. So does an index bound with ``generate_index``
off, which the provider leaves unwritten and the CPU reference fills.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, convolution, layernorm

NODE_TYPE = "ResampleFwdAttributes"
SCHEMA_FILE = "resample_fwd_attributes.fbs"
SCHEMA_TABLE = "ResampleFwdAttributes"

X, Y, INDEX = "x_tensor_uid", "y_tensor_uid", "index_tensor_uid"
PRE, POST, STRIDE, WINDOW = "pre_padding", "post_padding", "stride", "window"
MODE, PADDING_MODE, GENERATE = "resample_mode", "padding_mode", "generate_index"
PARAMETERS = (PRE, POST, STRIDE, WINDOW)

# ResampleMode and PaddingMode as utilities/json/Common.hpp writes them, but
# "not_set" and "padding_not_set", which the API rejects.
MODES = ("maxpool", "avgpool_exclude_padding", "avgpool_include_padding")
PADDING_MODES = ("neg_inf_pad", "zero_pad")
# Tensor rank -> spatial feature: 4D and 5D, as the provider and the
# CPU reference take.
SPATIAL = {4: "spatial:2d", 5: "spatial:3d"}
INDEX_DTYPE = "int32"

FEATURES = (
    *(Feature(f"mode:{mode}", (MODE,)) for mode in MODES),
    Feature(
        "padding_mode:neg_inf_pad",
        (PADDING_MODE, MODE, PRE, POST),
        "maxpool skips a padded cell",
    ),
    Feature(
        "padding_mode:zero_pad",
        (PADDING_MODE, MODE, PRE, POST),
        "maxpool takes a padded cell as 0",
    ),
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in layernorm.DTYPES),
    *(Feature(spatial, (X, Y)) for spatial in SPATIAL.values()),
    Feature("layout:nchw", (X, Y, INDEX), "channels first"),
    Feature("layout:nhwc", (X, Y, INDEX), "channels last"),
    Feature("padding:symmetric", (PRE, POST), "non-zero, pre = post"),
    Feature("padding:asymmetric", (PRE, POST), "pre != post in some dim"),
    Feature("stride:strided", (STRIDE,), "a stride above 1"),
    Feature("window:overlapping", (STRIDE, WINDOW), "windows share cells"),
    Feature("window:gapped", (STRIDE, WINDOW), "cells between windows"),
    Feature("index:generated", (MODE, INDEX, GENERATE), "maxpool writes argmax"),
    Feature("index:none", (MODE, INDEX, GENERATE), "maxpool without an index"),
)

NOT_FEATURES: dict[str, str] = {}


def catalogued_mode(node: dict) -> str:
    mode = node.get(MODE)
    if mode not in MODES:
        raise FeatureError(f"{MODE} {mode!r} is not catalogued")
    padding_mode = node.get(PADDING_MODE)
    if padding_mode not in PADDING_MODES:
        raise FeatureError(f"{PADDING_MODE} {padding_mode!r} is not catalogued")
    return mode


def checked_shapes(
    node: dict, big: dict, big_port: str, small: dict, small_port: str
) -> int:
    """The spatial dim count, or raise where the provider rejects the shapes:
    ``big`` is the pooled tensor (fwd ``x``, bwd ``dx``), ``small`` the
    pooled-to one (fwd ``y``, bwd ``dy``)."""
    big_dims, small_dims = big["dims"], small["dims"]
    if len(big_dims) != len(small_dims):
        raise FeatureError(
            f"{small_port} dims {small_dims} are not the rank of {big_port} dims "
            f"{big_dims}"
        )
    if len(big_dims) not in SPATIAL:
        raise FeatureError(
            f"{big_port} dims {big_dims}: rank {len(big_dims)} is not catalogued"
        )
    spatial = len(big_dims) - 2
    for name in PARAMETERS:
        values = node.get(name)
        if (
            type(values) is not list
            or len(values) != spatial
            or any(type(v) is not int for v in values)
        ):
            raise FeatureError(f"{name} {values!r} does not have {spatial} int entries")
    if any(p < 0 for name in (PRE, POST) for p in node[name]):
        raise FeatureError(f"negative padding {node[PRE]} / {node[POST]}")
    if any(s < 1 for name in (STRIDE, WINDOW) for s in node[name]):
        raise FeatureError(f"{STRIDE} {node[STRIDE]} / {WINDOW} {node[WINDOW]} below 1")
    if big_dims[:2] != small_dims[:2]:
        raise FeatureError(
            f"{small_port} dims {small_dims} do not keep N and C of {big_port} dims "
            f"{big_dims}"
        )
    expected = [
        (d + pre + post - window) // stride + 1
        for d, pre, post, stride, window in zip(
            big_dims[2:], *(node[name] for name in PARAMETERS)
        )
    ]
    if any(e < 1 for e in expected) or small_dims[2:] != expected:
        raise FeatureError(
            f"{small_port} spatial dims {small_dims[2:]} are not {expected}, or one "
            "is below 1"
        )
    return spatial


def reaches_padding(node: dict, big_dims: list[int], small_dims: list[int]) -> bool:
    """Whether some window covers a padded cell: the first window, given
    pre-padding, or the last, reaching past the end of ``big``."""
    for d, out, pre, stride, window in zip(
        big_dims[2:], small_dims[2:], node[PRE], node[STRIDE], node[WINDOW]
    ):
        if pre > 0 or (out - 1) * stride + window - 1 - pre >= d:
            return True
    return False


def geometry_features(node: dict) -> set[str]:
    features = set()
    if node[PRE] != node[POST]:
        features.add("padding:asymmetric")
    elif any(node[PRE]):
        features.add("padding:symmetric")
    if any(s > 1 for s in node[STRIDE]):
        features.add("stride:strided")
    if any(s < w for s, w in zip(node[STRIDE], node[WINDOW])):
        features.add("window:overlapping")
    if any(s > w for s, w in zip(node[STRIDE], node[WINDOW])):
        features.add("window:gapped")
    return features


def layout_features(tensors: dict[str, dict]) -> set[str]:
    """The layout the telling tensors (port -> tensor) agree on, or raise."""
    telling = [
        fits
        for fits in map(convolution.tensor_layouts, tensors.values())
        if len(fits) == 1
    ]
    agreed = set().union(*telling)
    if len(agreed) > 1:
        raise FeatureError(f"the layouts of {', '.join(tensors)} disagree")
    return {f"layout:{name}" for name in agreed}


def checked_index(index: dict, small: dict, small_port: str) -> None:
    if index.get("data_type") != INDEX_DTYPE:
        raise FeatureError(
            f"{INDEX} dtype {index.get('data_type')!r} is not catalogued"
        )
    if index["dims"] != small["dims"]:
        raise FeatureError(
            f"{INDEX} dims {index['dims']} are not {small_port} dims {small['dims']}"
        )


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    for port in (X, Y):
        if node.get(port) is None:
            raise FeatureError(f"required port {port} is not bound")
    x, y = tensors_by_uid[node[X]], tensors_by_uid[node[Y]]

    mode = catalogued_mode(node)
    spatial = checked_shapes(node, x, X, y, Y)

    dtype = layernorm.catalogued_dtype(x, X)
    if layernorm.catalogued_dtype(y, Y) != dtype:
        raise FeatureError(f"{Y} dtype {y.get('data_type')!r} is not that of {X}")

    generate = node.get(GENERATE, False)
    if type(generate) is not bool:
        raise FeatureError(f"{GENERATE} {generate!r} is not a bool")
    has_index = node.get(INDEX) is not None
    if generate and not has_index:
        raise FeatureError(f"{GENERATE} is on but {INDEX} is not bound")
    layout_tensors = {X: x, Y: y}
    if has_index:
        if mode != "maxpool":
            raise FeatureError(f"{INDEX} is bound under {MODE} {mode}")
        if not generate:
            raise FeatureError(f"{INDEX} is bound but {GENERATE} is off")
        layout_tensors[INDEX] = index = tensors_by_uid[node[INDEX]]
        checked_index(index, y, Y)

    features = {
        f"mode:{mode}",
        f"dtype:{dtype}",
        SPATIAL[spatial + 2],
        *layout_features(layout_tensors),
        *geometry_features(node),
    }
    if mode == "maxpool":
        features.add("index:generated" if has_index else "index:none")
        if reaches_padding(node, x["dims"], y["dims"]):
            features.add(f"padding_mode:{node[PADDING_MODE]}")
    return features
