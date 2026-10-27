# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``ResampleBwdAttributes``.

The gradient of pooling: each cell of ``dy`` (the forward ``y``) spreads
back over its ``window`` of ``dx`` (the forward ``x``), evenly for the
average modes, to the argmax the index names for ``maxpool``. The features
come from ``resample_bwd_attributes.fbs`` and ``resample_common.fbs``. As
for the forward, the serializer
(``utilities/json/ResampleBwdAttributes.hpp``) writes the ports at the top
of the node and leaves an unset ``index_tensor_uid`` out. The hip-kernel
provider and the CPU reference agree on what they take, and the catalog
takes the same. By group, as for ``resample_fwd``:

- ``mode`` (exclusive): the pooling, by its serialized name.
- ``dtype`` (exclusive): the dtype of ``dy``, which ``dx`` shares.
- ``spatial`` (exclusive): 2 or 3 spatial dims.
- ``layout`` (exclusive, when the strides tell): ``nchw`` or ``nhwc``, read
  from ``dy``, ``dx`` and the index.
- ``padding:symmetric``, ``padding:asymmetric``: non-zero padding, with
  pre = post in every dim or not.
- ``stride:strided``: a stride above 1 in some dim.
- ``window:overlapping``, ``window:gapped``: a stride below the window in
  some dim (windows share ``dx`` cells, whose gradients sum), or above it
  (``dx`` cells no window reads, whose gradient is 0).

No ``index`` group: ``maxpool`` requires the index and the average modes
reject one, so ``mode`` says it.

The API requires ``dy``, ``dx`` and, under ``maxpool``, an index. The
provider and the CPU reference also require a mode and padding mode other
than their ``NOT_SET``; ``dy`` and ``dx`` of rank 4 or 5 (the API allows 3)
and of one dtype; the padding, stride and window ``resample_fwd`` requires,
with ``dy`` the shape the forward gives ``dx``; and an index only under
``maxpool``, of ``dy``'s dims and int32 (the API allows int8). A graph that
breaks these raises.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, layernorm, resample_fwd
from support_features.resample_fwd import (
    INDEX,
    MODE,
    PADDING_MODE,
    POST,
    PRE,
    SPATIAL,
    STRIDE,
    WINDOW,
)

NODE_TYPE = "ResampleBwdAttributes"
SCHEMA_FILE = "resample_bwd_attributes.fbs"
SCHEMA_TABLE = "ResampleBwdAttributes"

DY, DX = "dy_tensor_uid", "dx_tensor_uid"

FEATURES = (
    *(Feature(f"mode:{mode}", (MODE,)) for mode in resample_fwd.MODES),
    *(Feature(f"dtype:{dtype}", (DY,)) for dtype in layernorm.DTYPES),
    *(Feature(spatial, (DY, DX)) for spatial in SPATIAL.values()),
    Feature("layout:nchw", (DY, DX, INDEX), "channels first"),
    Feature("layout:nhwc", (DY, DX, INDEX), "channels last"),
    Feature("padding:symmetric", (PRE, POST), "non-zero, pre = post"),
    Feature("padding:asymmetric", (PRE, POST), "pre != post in some dim"),
    Feature("stride:strided", (STRIDE,), "a stride above 1"),
    Feature("window:overlapping", (STRIDE, WINDOW), "dx gradients sum"),
    Feature("window:gapped", (STRIDE, WINDOW), "dx cells no window reads"),
)

NOT_FEATURES: dict[str, str] = {
    PADDING_MODE: "read by neither the provider (its plan does not pass it to "
    "the kernel) nor the CPU reference: maxpool's index names the argmax, "
    "and the average modes say whether padded cells count",
}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    for port in (DY, DX):
        if node.get(port) is None:
            raise FeatureError(f"required port {port} is not bound")
    dy, dx = tensors_by_uid[node[DY]], tensors_by_uid[node[DX]]

    # Validates padding_mode too: both reject padding_not_set.
    mode = resample_fwd.catalogued_mode(node)
    spatial = resample_fwd.checked_shapes(node, dx, DX, dy, DY)

    dtype = layernorm.catalogued_dtype(dy, DY)
    if layernorm.catalogued_dtype(dx, DX) != dtype:
        raise FeatureError(f"{DX} dtype {dx.get('data_type')!r} is not that of {DY}")

    has_index = node.get(INDEX) is not None
    layout_tensors = {DY: dy, DX: dx}
    if mode == "maxpool":
        if not has_index:
            raise FeatureError(f"{MODE} maxpool requires {INDEX}")
        layout_tensors[INDEX] = index = tensors_by_uid[node[INDEX]]
        resample_fwd.checked_index(index, dy, DY)
    elif has_index:
        raise FeatureError(f"{INDEX} is bound under {MODE} {mode}")

    return {
        f"mode:{mode}",
        f"dtype:{dtype}",
        SPATIAL[spatial + 2],
        *resample_fwd.layout_features(layout_tensors),
        *resample_fwd.geometry_features(node),
    }
