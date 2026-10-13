# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Features shared by the three convolution ops.

``ConvolutionFwdAttributes``, ``ConvolutionBwdAttributes`` and
``ConvolutionWrwAttributes`` carry the same five parameters
(``convolution_*_attributes.fbs``, ``convolution_common.fbs``) and differ only
in which three tensors they bind. Each op module names its ports in a
``Ports``; ``catalog`` and ``features`` do the rest. The tensors play three
shape roles: the activation (fwd ``x``, bwd ``dx``, wrw ``x``), the filter
(fwd ``w``, bwd ``w``, wrw ``dw``) and the response (fwd ``y``, bwd and wrw
``dy``). Groups:

- ``dtype`` (exclusive): the dtype of the op's data input.
- ``precision:mixed``: the op's non-virtual tensors do not all share one
  dtype. A virtual tensor -- the fp32 intermediate between a conv and its
  fused pointwise -- is not counted: the caller never sees it.
- ``rank`` (exclusive): 1, 2 or 3 spatial dims.
- ``layout`` (exclusive): channels first (NCW / NCHW / NCDHW) or channels last
  (NWC / NHWC / NDHWC), read from the strides. A dim of size 1 has no place in
  the order, so it is skipped; a tensor whose remaining order fits both
  layouts (e.g. C = 1) says nothing. The node's layout is the one its
  telling tensors agree on. If none tells, the graph adds no layout feature.
- ``groups:grouped``, ``groups:depthwise``: more than one group (activation
  channels / filter channels); depthwise when there are as many groups as
  activation channels.
- ``stride:strided``, ``dilation:dilated``: a stride / dilation above 1 in any
  spatial dim.
- ``padding:symmetric``, ``padding:asymmetric``: non-zero padding, with
  pre = post in every dim or not.
- ``mode`` (exclusive): ``CROSS_CORRELATION`` or ``CONVOLUTION``.

Filter size (1x1 or not) is not a feature: it is a provider specialization,
not something the API distinguishes.

A graph the API would reject -- parameter lists of the wrong length, channels
that do not divide into groups, ``conv_mode`` ``UNSET`` -- raises rather than
counting.
"""

from __future__ import annotations

from typing import NamedTuple

from support_features import Feature, FeatureError
from support_matrix_units import dtype_label

DTYPES = ("fp32", "fp16", "bf16")
RANKS = {1: "rank:1d", 2: "rank:2d", 3: "rank:3d"}
LAYOUTS = ("nchw", "nhwc")
# The serializer's spelling (Graph.hpp) -> feature id.
MODES = {
    "CROSS_CORRELATION": "mode:cross_correlation",
    "CONVOLUTION": "mode:convolution",
}
PARAMETERS = ("pre_padding", "post_padding", "stride", "dilation")


class Ports(NamedTuple):
    """An op's tensor ports, by shape role."""

    activation: str
    filter: str
    response: str
    # The port whose dtype is the op's ``dtype``.
    data: str


def catalog(ports: Ports) -> tuple[Feature, ...]:
    tensors = (ports.activation, ports.filter, ports.response)
    return (
        *(Feature(f"dtype:{dtype}", (ports.data,)) for dtype in DTYPES),
        Feature(
            "precision:mixed",
            tensors,
            "the non-virtual tensors do not all share one dtype",
        ),
        *(Feature(rank, tensors) for rank in RANKS.values()),
        Feature("layout:nchw", tensors, "channels first"),
        Feature("layout:nhwc", tensors, "channels last"),
        Feature("groups:grouped", (ports.activation, ports.filter), "groups > 1"),
        Feature(
            "groups:depthwise",
            (ports.activation, ports.filter),
            "one group per activation channel",
        ),
        Feature("stride:strided", ("stride",), "a stride above 1"),
        Feature("dilation:dilated", ("dilation",), "a dilation above 1"),
        Feature(
            "padding:symmetric",
            ("pre_padding", "post_padding"),
            "non-zero, pre = post",
        ),
        Feature(
            "padding:asymmetric",
            ("pre_padding", "post_padding"),
            "pre != post in some dim",
        ),
        Feature("mode:cross_correlation", ("conv_mode",)),
        Feature("mode:convolution", ("conv_mode",)),
    )


def _orders(rank: int) -> dict[str, tuple[int, ...]]:
    """Layout -> its dims from outermost to innermost."""
    return {"nchw": tuple(range(rank)), "nhwc": (0, *range(2, rank), 1)}


def tensor_layouts(tensor: dict) -> set[str]:
    """The layouts ``tensor``'s strides fit: one, or both if it cannot tell."""
    dims, strides = tensor["dims"], tensor.get("strides")
    if not strides or len(strides) != len(dims):
        raise FeatureError(f"strides {strides} do not match dims {dims}")
    live = [dim for dim in range(len(dims)) if dims[dim] != 1]
    if len({strides[dim] for dim in live}) != len(live):
        raise FeatureError(f"strides {strides} overlap for dims {dims}")
    order = tuple(sorted(live, key=lambda dim: -strides[dim]))
    fits = {
        name
        for name, full in _orders(len(dims)).items()
        if tuple(dim for dim in full if dim in live) == order
    }
    if not fits:
        raise FeatureError(
            f"strides {strides} for dims {dims} are not a catalogued layout"
        )
    return fits


def features(node: dict, tensors_by_uid: dict, ports: Ports) -> set[str]:
    # The serializers write the ports under "inputs" / "outputs" and the rest
    # under "parameters" (utilities/json/Convolution*Attributes.hpp).
    bound = {**(node.get("inputs") or {}), **(node.get("outputs") or {})}
    params = node.get("parameters") or {}
    x, w, y = (
        tensors_by_uid[bound[port]]
        for port in (ports.activation, ports.filter, ports.response)
    )

    ranks = {len(t["dims"]) for t in (x, w, y)}
    if len(ranks) != 1:
        raise FeatureError(f"tensor ranks differ: {sorted(ranks)}")
    spatial = ranks.pop() - 2
    if spatial not in RANKS:
        raise FeatureError(f"{spatial} spatial dims are not catalogued")
    for name in PARAMETERS:
        if len(params.get(name) or ()) != spatial:
            raise FeatureError(
                f"{name} {params.get(name)} does not have {spatial} entries"
            )
    if any(p < 0 for name in ("pre_padding", "post_padding") for p in params[name]):
        raise FeatureError(
            f"negative padding {params['pre_padding']} / {params['post_padding']}"
        )
    if any(s < 1 for name in ("stride", "dilation") for s in params[name]):
        raise FeatureError(
            f"stride {params['stride']} / dilation {params['dilation']} below 1"
        )

    data = tensors_by_uid[bound[ports.data]]
    dtype = dtype_label(data.get("data_type"))
    if dtype not in DTYPES:
        raise FeatureError(
            f"{ports.data} dtype {data.get('data_type')!r} is not catalogued"
        )
    stray = {dtype_label(t.get("data_type")) for t in (x, w, y)} - set(DTYPES)
    if stray:
        raise FeatureError(f"tensor dtype(s) {sorted(stray)} are not catalogued")
    seen = {dtype_label(t.get("data_type")) for t in (x, w, y) if not t.get("virtual")}

    mode = MODES.get(params.get("conv_mode"))
    if mode is None:
        raise FeatureError(f"conv_mode {params.get('conv_mode')!r} is not catalogued")

    channels, per_group = x["dims"][1], w["dims"][1]
    if per_group < 1 or channels % per_group:
        raise FeatureError(
            f"{channels} activation channels do not split into groups of {per_group}"
        )
    groups = channels // per_group
    if w["dims"][0] % groups:
        raise FeatureError(f"{w['dims'][0]} filters do not split into {groups} groups")

    features = {f"dtype:{dtype}", RANKS[spatial], mode}
    if len(seen) > 1:
        features.add("precision:mixed")

    telling = [fits for fits in map(tensor_layouts, (x, w, y)) if len(fits) == 1]
    agreed = set().union(*telling)
    if len(agreed) > 1:
        raise FeatureError("the tensors' layouts disagree")
    features.update(f"layout:{name}" for name in agreed)

    if groups > 1:
        features.add("groups:depthwise" if groups == channels else "groups:grouped")
    if any(s > 1 for s in params["stride"]):
        features.add("stride:strided")
    if any(d > 1 for d in params["dilation"]):
        features.add("dilation:dilated")
    if params["pre_padding"] != params["post_padding"]:
        features.add("padding:asymmetric")
    elif any(params["pre_padding"]):
        features.add("padding:symmetric")
    return features
