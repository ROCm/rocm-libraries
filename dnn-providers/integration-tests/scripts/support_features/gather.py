# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``GatherAttributes``.

Gather (index_select / take) along one axis: with ``indices`` of rank m,
``y[a.., i_0..i_{m-1}, b..] = x[a.., indices[i_0..i_{m-1}], b..]``, so
``y.dims = x.dims[:axis] ++ indices.dims ++ x.dims[axis + 1:]``. The features
come from ``gather_attributes.fbs``. The serializer
(``utilities/json/GatherAttributes.hpp``) writes the ports under ``inputs``
and ``outputs`` and ``axis`` at the top level. No provider implements the op,
so the dtypes are those the CPU reference executor registers
(``GatherSignatureKey``): ``x`` and ``y`` of one float dtype, ``indices``
int32 or int64. By group:

- ``dtype`` (exclusive): the dtype of ``x``, which ``y`` shares.
- ``index_dtype`` (exclusive): ``int32`` or ``int64``.
- ``axis`` (exclusive): ``leading`` (axis 0: embedding lookup) or ``inner``
  (any later axis).
- ``indices_rank`` (exclusive): ``single`` (rank 1: a list of ids) or
  ``multi`` (rank >= 2: e.g. ``[batch, seq]`` token ids).

The API and the CPU reference require ``0 <= axis < rank(x)``, ``y`` of the
dims above and the dtype of ``x``, and no ragged tensor. A graph that breaks
these raises. The index values (which rows are read) are data the graph does
not carry.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm

NODE_TYPE = "GatherAttributes"
SCHEMA_FILE = "gather_attributes.fbs"
SCHEMA_TABLE = "GatherAttributes"

X, INDICES, Y = "x_tensor_uid", "indices_tensor_uid", "y_tensor_uid"
AXIS = "axis"
REQUIRED = (X, INDICES, Y)
RAGGED = "ragged_offset_tensor_uid"
INDEX_DTYPES = ("int32", "int64")

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X, Y)) for dtype in layernorm.DTYPES),
    *(Feature(f"index_dtype:{dtype}", (INDICES,)) for dtype in INDEX_DTYPES),
    Feature("axis:leading", (AXIS,), "axis 0 (embedding lookup)"),
    Feature("axis:inner", (AXIS,), "an axis after the first"),
    Feature("indices_rank:single", (INDICES,), "rank-1 indices"),
    Feature("indices_rank:multi", (INDICES,), "indices of rank >= 2"),
)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    x, indices, y = (tensors_by_uid[by_port[p]] for p in REQUIRED)

    for port, tensor in zip(REQUIRED, (x, indices, y)):
        dims = tensor["dims"]
        if not dims or any(d < 0 for d in dims):
            raise FeatureError(f"{port} dims {dims}: not rank >= 1 with dims >= 0")
        # Written only when set (CHECK_NO_RAGGED_TENSORS); uid 0 is one too.
        if tensor.get(RAGGED) is not None:
            raise FeatureError(f"{port} is ragged")

    axis = node.get(AXIS, 0)
    x_dims = x["dims"]
    if not isinstance(axis, int) or not 0 <= axis < len(x_dims):
        raise FeatureError(f"{AXIS} {axis!r} is not in [0, {len(x_dims)})")
    expected = x_dims[:axis] + indices["dims"] + x_dims[axis + 1 :]
    if y["dims"] != expected:
        raise FeatureError(f"{Y} dims {y['dims']} are not {expected}")

    index_dtype = indices.get("data_type")
    if index_dtype not in INDEX_DTYPES:
        raise FeatureError(f"{INDICES} dtype {index_dtype!r} is not int32 or int64")
    dtype = layernorm.catalogued_dtype(x, X)
    if layernorm.catalogued_dtype(y, Y) != dtype:
        raise FeatureError(f"{Y} dtype {y.get('data_type')!r} is not that of {X}")

    return {
        f"dtype:{dtype}",
        f"index_dtype:{index_dtype}",
        "axis:leading" if axis == 0 else "axis:inner",
        "indices_rank:single" if len(indices["dims"]) == 1 else "indices_rank:multi",
    }
