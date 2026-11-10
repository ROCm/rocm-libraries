# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``TopkAttributes``.

Top-k selection along the last dim: for every row of ``x``, the positions
(``indices``) and, when bound, the values (``values``) of the ``k`` largest
(``largest`` true) or smallest entries, sorted best-first; ties keep the lower
position first and NaN compares greater than every number. The features come
from ``topk_attributes.fbs``. The serializer
(``utilities/json/TopkAttributes.hpp``) writes the ports under ``inputs`` and
``outputs`` and ``k`` and ``largest`` at the top level; ``values_tensor_uid``
only when bound. No provider implements the op, so the dtypes are those the
CPU reference executor registers (``TopkSignatureKey``): ``x`` and ``values``
of one float dtype, ``indices`` int32 or int64. By group:

- ``dtype`` (exclusive): the dtype of ``x``, which ``values`` shares.
- ``index_dtype`` (exclusive): ``int32`` or ``int64``.
- ``order`` (exclusive): ``largest`` or ``smallest``.
- ``values`` (exclusive): ``present`` or ``absent`` (indices only).
- ``k`` (exclusive): ``argmax`` (k = 1), ``partial`` (1 < k < last dim) or
  ``full`` (k = last dim, a full sort; k = 1 = last dim counts as ``argmax``).

The API and the CPU reference require ``1 <= k <= x.dims[-1]``, ``indices``
and ``values`` of the dims of ``x`` with the last dim ``k``, ``values`` of the
dtype of ``x``, and no ragged tensor. A graph that breaks these raises.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm

NODE_TYPE = "TopkAttributes"
SCHEMA_FILE = "topk_attributes.fbs"
SCHEMA_TABLE = "TopkAttributes"

X, INDICES, VALUES = "x_tensor_uid", "indices_tensor_uid", "values_tensor_uid"
K, LARGEST = "k", "largest"
REQUIRED = (X, INDICES)
RAGGED = "ragged_offset_tensor_uid"
INDEX_DTYPES = ("int32", "int64")

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X, VALUES)) for dtype in layernorm.DTYPES),
    *(Feature(f"index_dtype:{dtype}", (INDICES,)) for dtype in INDEX_DTYPES),
    Feature("order:largest", (LARGEST,), "the k largest entries"),
    Feature("order:smallest", (LARGEST,), "the k smallest entries"),
    Feature("values:present", (VALUES,), "the selected values are written"),
    Feature("values:absent", (VALUES,), "indices only"),
    Feature("k:argmax", (K,), "k = 1"),
    Feature("k:partial", (K,), "1 < k < last dim"),
    Feature("k:full", (K,), "k = last dim: a full sort"),
)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    ports = [*REQUIRED, *((VALUES,) if has(VALUES) else ())]
    tensors = {port: tensors_by_uid[by_port[port]] for port in ports}

    for port, tensor in tensors.items():
        dims = tensor["dims"]
        if not dims or any(d < 0 for d in dims):
            raise FeatureError(f"{port} dims {dims}: not rank >= 1 with dims >= 0")
        # Written only when set (CHECK_NO_RAGGED_TENSORS); uid 0 is one too.
        if tensor.get(RAGGED) is not None:
            raise FeatureError(f"{port} is ragged")

    x_dims = tensors[X]["dims"]
    k = node.get(K, 0)
    if not isinstance(k, int) or not 1 <= k <= x_dims[-1]:
        raise FeatureError(f"{K} {k!r} is not in [1, {x_dims[-1]}]")
    expected = x_dims[:-1] + [k]
    for port in ports[1:]:
        if tensors[port]["dims"] != expected:
            raise FeatureError(
                f"{port} dims {tensors[port]['dims']} are not {expected}"
            )

    index_dtype = tensors[INDICES].get("data_type")
    if index_dtype not in INDEX_DTYPES:
        raise FeatureError(f"{INDICES} dtype {index_dtype!r} is not int32 or int64")
    dtype = layernorm.catalogued_dtype(tensors[X], X)
    if (
        VALUES in tensors
        and layernorm.catalogued_dtype(tensors[VALUES], VALUES) != dtype
    ):
        values_dtype = tensors[VALUES].get("data_type")
        raise FeatureError(f"{VALUES} dtype {values_dtype!r} is not that of {X}")

    if k == 1:
        k_feature = "k:argmax"
    elif k == x_dims[-1]:
        k_feature = "k:full"
    else:
        k_feature = "k:partial"

    return {
        f"dtype:{dtype}",
        f"index_dtype:{index_dtype}",
        "order:largest" if node.get(LARGEST, True) else "order:smallest",
        "values:present" if VALUES in tensors else "values:absent",
        k_feature,
    }
