# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``BlockScaleQuantizeAttributes``.

``y = x / scale`` with one ``scale`` per block of ``block_size`` elements
along one axis of ``x``, the scale chosen from the block's largest
magnitude: the quantize that produces an MX operand. The features come from
``block_scale_quantize_attributes.fbs``. The serializer
(``utilities/json/BlockScaleQuantizeAttributes.hpp``) writes the ports under
``inputs`` (``x``) and ``outputs`` (``y``, ``scale``); it leaves a
``block_size`` of 0 and an unset ``axis`` out, and its reader takes a
missing ``transpose`` as false. No provider implements the op, so the dtypes
are those the CPU reference executor registers
(``BlockScaleQuantizeSignatureKey``). By group:

- ``dtype`` (exclusive): the dtype of ``x``, a float one.
- ``out_dtype`` (exclusive): the dtype of ``y``: an OCP MX dtype (fp8, fp6
  or fp4) or a float one, read as ``block_scale_dequantize`` reads its
  ``x``.
- ``scale_dtype`` (exclusive): ``fp8e8m0`` (a power of two, the MX scale)
  or ``fp32``.
- ``axis`` (exclusive): the blocked dim: ``last_dim`` (``axis`` unset, the
  default, or naming the last) or ``other_dim``.
- ``block_length`` (exclusive): ``32`` (the MX block) or ``other``.
- ``transpose`` (exclusive): ``off`` or ``on``. The backend descriptor
  carries it; the API reads it to infer ``y`` and ``scale`` strides with the
  blocked dim packed.

The API requires ``x``, ``y`` and ``scale``, ``x`` of rank 1 or more, ``y``
of ``x``'s dims, a ``block_size`` above 0 that divides the blocked dim, and
an ``axis`` within the rank of ``x``; it infers ``scale`` as ``x`` with the
blocked dim divided by ``block_size``, the shape the CPU reference requires.
A graph that breaks these raises. As the API rejects a blocked dim that
``block_size`` does not divide, there is no partial block and no ``fit``
group.

No rank or layout feature: outside its blocked dim the op is elementwise,
and ``transpose`` names the one layout choice it makes.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, block_scale_dequantize

NODE_TYPE = "BlockScaleQuantizeAttributes"
SCHEMA_FILE = "block_scale_quantize_attributes.fbs"
SCHEMA_TABLE = "BlockScaleQuantizeAttributes"

X, Y, SCALE = "x_tensor_uid", "y_tensor_uid", "scale_tensor_uid"
BLOCK_SIZE, AXIS, TRANSPOSE = "block_size", "axis", "transpose"

# data_type as utilities/json/Common.hpp writes it -> label.
DTYPES = {"float": "fp32", "half": "fp16", "bfloat16": "bf16"}
OUT_DTYPES = block_scale_dequantize.DTYPES
SCALE_DTYPES = block_scale_dequantize.SCALE_DTYPES
MX_BLOCK = block_scale_dequantize.MX_BLOCK

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in DTYPES.values()),
    *(Feature(f"out_dtype:{dtype}", (Y,)) for dtype in OUT_DTYPES.values()),
    Feature("scale_dtype:fp8e8m0", (SCALE,), "a power of two: the MX scale"),
    Feature("scale_dtype:fp32", (SCALE,), "any fp32 value"),
    Feature("axis:last_dim", (X, AXIS), "the last dim, by default or by index"),
    Feature("axis:other_dim", (X, AXIS), "a dim before the last"),
    Feature(f"block_length:{MX_BLOCK}", (BLOCK_SIZE,), "the MX block"),
    Feature("block_length:other", (BLOCK_SIZE,), "a block that is not 32"),
    Feature("transpose:off", (TRANSPOSE,), "y and scale take x's layout"),
    Feature("transpose:on", (TRANSPOSE,), "y and scale pack the blocked dim"),
)

NOT_FEATURES: dict[str, str] = {}


def blocked_axis(node: dict, rank: int) -> int:
    """The blocked dim of ``x``: ``axis``, or the last when it is unset; or
    raise where the API rejects it."""
    if AXIS not in node:
        return rank - 1
    axis = node[AXIS]
    # type() rather than isinstance: a bool is an int.
    if type(axis) is not int or not 0 <= axis < rank:
        raise FeatureError(f"{AXIS} {axis!r} is not a dim of rank {rank}")
    return axis


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (X, Y, SCALE):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    x, y, scale = (tensors_by_uid[by_port[port]] for port in (X, Y, SCALE))

    dims = x["dims"]
    if not dims or any(d < 1 for d in dims):
        raise FeatureError(f"{X} dims {dims}: not rank >= 1 with dims >= 1")
    if y["dims"] != dims:
        raise FeatureError(f"{Y} dims {y['dims']} are not {X} dims {dims}")
    block_size = node.get(BLOCK_SIZE)
    if type(block_size) is not int or block_size < 1:
        raise FeatureError(f"{BLOCK_SIZE} {block_size!r} is not an int above 0")
    axis = blocked_axis(node, len(dims))
    if dims[axis] % block_size:
        raise FeatureError(
            f"{BLOCK_SIZE} {block_size} does not divide {X} dim {axis} ({dims[axis]})"
        )
    expected = list(dims)
    expected[axis] //= block_size
    if scale["dims"] != expected:
        raise FeatureError(f"{SCALE} dims {scale['dims']} are not {expected}")

    features = {
        f"dtype:{block_scale_dequantize.catalogued(x, X, DTYPES)}",
        f"out_dtype:{block_scale_dequantize.catalogued(y, Y, OUT_DTYPES)}",
        f"scale_dtype:{block_scale_dequantize.catalogued(scale, SCALE, SCALE_DTYPES)}",
        "axis:last_dim" if axis == len(dims) - 1 else "axis:other_dim",
        f"block_length:{MX_BLOCK}" if block_size == MX_BLOCK else "block_length:other",
    }

    transpose = node.get(TRANSPOSE, False)
    if type(transpose) is not bool:
        raise FeatureError(f"{TRANSPOSE} {transpose!r} is not a bool")
    features.add("transpose:on" if transpose else "transpose:off")
    return features
