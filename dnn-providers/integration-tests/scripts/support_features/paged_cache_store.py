# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``PagedCacheStoreAttributes``.

Paged KV-cache write into the SDPA paged container ``[num_blocks, H,
block_size, D]``: each token with ``slot = slot_mapping[token] >= 0`` writes
``container[slot / block_size, h, slot % block_size, :] = cast(x[token, h, :]
* scale)``; other container elements keep their contents. The features come
from ``paged_cache_store_attributes.fbs``. The serializer
(``utilities/json/PagedCacheStoreAttributes.hpp``) writes the ports under
``inputs`` and ``outputs``. No provider implements the op, so the dtypes are
those the CPU reference executor registers (``PagedCacheStoreSignatureKey``):
``x`` of one float dtype, the container of that dtype or an OCP fp8 dtype,
``slot_mapping`` int32 or int64, ``scale`` fp32. By group:

- ``dtype`` (exclusive): the dtype of ``x``.
- ``container`` (exclusive): ``same`` (the dtype of ``x``), ``fp8e4m3`` or
  ``fp8e5m2`` (a quantizing store).
- ``slot_dtype`` (exclusive): ``int32`` or ``int64``.
- ``layout`` (exclusive): ``token_packed`` (``x [T, H, D]``, slots ``[T]``) or
  ``sdpa_native`` (``x [B, H, S, D]``, slots ``[B, S]``).
- ``scale`` (exclusive): ``absent`` or ``present``.

The API and the CPU reference require a container of rank 4 whose dims 1 and
3 are the head count and head size of ``x``, a one-element fp32 scale, and no
ragged tensor. A graph that breaks these raises. The slot values are data the
graph does not carry.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm
from support_matrix_units import dtype_label

NODE_TYPE = "PagedCacheStoreAttributes"
SCHEMA_FILE = "paged_cache_store_attributes.fbs"
SCHEMA_TABLE = "PagedCacheStoreAttributes"

X, SLOT, CONTAINER, SCALE = (
    "x_tensor_uid",
    "slot_mapping_tensor_uid",
    "container_tensor_uid",
    "scale_tensor_uid",
)
REQUIRED = (X, SLOT, CONTAINER)
RAGGED = "ragged_offset_tensor_uid"
SLOT_DTYPES = ("int32", "int64")
FP8_CONTAINERS = ("fp8e4m3", "fp8e5m2")

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in layernorm.DTYPES),
    Feature("container:same", (CONTAINER,), "the container has the dtype of x"),
    *(
        Feature(f"container:{dtype}", (CONTAINER,), f"a {dtype} container")
        for dtype in FP8_CONTAINERS
    ),
    *(Feature(f"slot_dtype:{dtype}", (SLOT,)) for dtype in SLOT_DTYPES),
    Feature("layout:token_packed", (X, SLOT), "x [T, H, D], slots [T]"),
    Feature("layout:sdpa_native", (X, SLOT), "x [B, H, S, D], slots [B, S]"),
    Feature("scale:absent", (SCALE,)),
    Feature("scale:present", (SCALE,)),
)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    ports = REQUIRED + ((SCALE,) if has(SCALE) else ())
    x, slots, container = (tensors_by_uid[by_port[p]] for p in REQUIRED)

    for port in ports:
        tensor = tensors_by_uid[by_port[port]]
        dims = tensor["dims"]
        if not dims or any(d < 0 for d in dims):
            raise FeatureError(f"{port} dims {dims}: not rank >= 1 with dims >= 0")
        # Written only when set (CHECK_NO_RAGGED_TENSORS); uid 0 is one too.
        if tensor.get(RAGGED) is not None:
            raise FeatureError(f"{port} is ragged")

    x_dims, slot_dims, c_dims = x["dims"], slots["dims"], container["dims"]
    if len(x_dims) == 3 and slot_dims == x_dims[:1]:
        layout = "layout:token_packed"
    elif len(x_dims) == 4 and slot_dims == [x_dims[0], x_dims[2]]:
        layout = "layout:sdpa_native"
    else:
        raise FeatureError(
            f"{X} dims {x_dims} with {SLOT} dims {slot_dims} are neither [T, H, D] + [T] "
            "nor [B, H, S, D] + [B, S]"
        )
    if len(c_dims) != 4 or c_dims[1] != x_dims[1] or c_dims[3] != x_dims[-1]:
        raise FeatureError(
            f"{CONTAINER} dims {c_dims} are not [num_blocks, {x_dims[1]}, block_size, "
            f"{x_dims[-1]}]"
        )

    slot_dtype = slots.get("data_type")
    if slot_dtype not in SLOT_DTYPES:
        raise FeatureError(f"{SLOT} dtype {slot_dtype!r} is not int32 or int64")
    dtype = layernorm.catalogued_dtype(x, X)
    container_label = dtype_label(container.get("data_type"))
    if container_label == dtype:
        container_feature = "container:same"
    elif container_label in FP8_CONTAINERS:
        container_feature = f"container:{container_label}"
    else:
        raise FeatureError(
            f"{CONTAINER} dtype {container.get('data_type')!r} is neither that of {X} nor fp8"
        )

    if has(SCALE):
        scale = tensors_by_uid[by_port[SCALE]]
        if scale.get("data_type") != "float":
            raise FeatureError(f"{SCALE} dtype {scale.get('data_type')!r} is not float")
        count = 1
        for d in scale["dims"]:
            count *= d
        if count != 1:
            raise FeatureError(f"{SCALE} dims {scale['dims']} are not one element")

    return {
        f"dtype:{dtype}",
        container_feature,
        f"slot_dtype:{slot_dtype}",
        layout,
        "scale:present" if has(SCALE) else "scale:absent",
    }
