# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``RopeAttributes``.

Rotary position embedding (RoPE) forward, rotate-half (non-interleaved): the
segment ``x[..., o : o + r]`` of the last dim ``D`` is rotated by the raw FP32
angles in ``freqs`` and every element is scaled by ``output_scale``. ``r`` is
``rope_dim`` (0 means ``D``); ``o`` is ``rope_offset_ext`` when set and
``D - r`` (cuDNN, trailing segment) when absent. The features come from
``rope_attributes.fbs``. The serializer
(``utilities/json/RopeAttributes.hpp``) writes the ports under ``inputs`` and
``outputs`` and ``output_scale``, ``rope_dim`` and ``rope_offset_ext`` at the
top level. The dtypes are those the CPU reference executor registers
(``RopeSignatureKey``), which ``hipkernel:flydsl_rope`` also builds: ``x`` and
``y`` of one float dtype, ``freqs`` fp32. By group:

- ``dtype`` (exclusive): the dtype of ``x``, which ``y`` shares.
- ``rotation`` (exclusive): ``full`` (``r = D``), ``partial_leading``
  (``o = 0``: Llama/Qwen partial rotary), ``partial_trailing`` (``o = D - r``:
  cuDNN) or ``partial_inner`` (any other ``o``).
- ``freqs`` (exclusive): ``broadcast`` (some non-last freqs dim is 1 where
  x's is not, e.g. ``[1, 1, S, r]``) or ``full`` (every non-last dim is x's).
- ``scale`` (exclusive): ``unit`` (``output_scale`` 1) or ``scaled``.

The API and the CPU reference require rank >= 2, ``freqs`` of x's rank with
each non-last dim x's or 1 and last dim ``r``, ``r`` even in ``(0, D]``,
``0 <= o`` and ``o + r <= D``, fp32 ``freqs``, ``y`` of x's dims and dtype, and
no ragged tensor. A graph that breaks these raises. The angle values are data
the graph does not carry.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm

NODE_TYPE = "RopeAttributes"
SCHEMA_FILE = "rope_attributes.fbs"
SCHEMA_TABLE = "RopeAttributes"

X, FREQS, Y = "x_tensor_uid", "freqs_tensor_uid", "y_tensor_uid"
OUTPUT_SCALE, ROPE_DIM, ROPE_OFFSET = "output_scale", "rope_dim", "rope_offset_ext"
REQUIRED = (X, FREQS, Y)
RAGGED = "ragged_offset_tensor_uid"

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X, Y)) for dtype in layernorm.DTYPES),
    Feature("rotation:full", (ROPE_DIM,), "the whole last dim is rotated"),
    Feature(
        "rotation:partial_leading",
        (ROPE_DIM, ROPE_OFFSET),
        "the first rope_dim dims are rotated (Llama/Qwen partial rotary)",
    ),
    Feature(
        "rotation:partial_trailing",
        (ROPE_DIM, ROPE_OFFSET),
        "the last rope_dim dims are rotated (cuDNN)",
    ),
    Feature(
        "rotation:partial_inner",
        (ROPE_DIM, ROPE_OFFSET),
        "a segment touching neither end is rotated",
    ),
    Feature("freqs:broadcast", (FREQS,), "freqs broadcast over some non-last dim"),
    Feature("freqs:full", (FREQS,), "freqs has every non-last dim of x"),
    Feature("scale:unit", (OUTPUT_SCALE,), "output_scale 1"),
    Feature("scale:scaled", (OUTPUT_SCALE,), "output_scale other than 1"),
)

NOT_FEATURES: dict[str, str] = {}


def _rotation(head_dim: int, rope_dim: int, offset: int) -> str:
    if rope_dim == head_dim:
        return "rotation:full"
    if offset == 0:
        return "rotation:partial_leading"
    if offset == head_dim - rope_dim:
        return "rotation:partial_trailing"
    return "rotation:partial_inner"


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    x, freqs, y = (tensors_by_uid[by_port[p]] for p in REQUIRED)

    for port, tensor in zip(REQUIRED, (x, freqs, y)):
        dims = tensor["dims"]
        if len(dims) < 2 or any(d < 0 for d in dims):
            raise FeatureError(f"{port} dims {dims}: not rank >= 2 with dims >= 0")
        # Written only when set (CHECK_NO_RAGGED_TENSORS); uid 0 is one too.
        if tensor.get(RAGGED) is not None:
            raise FeatureError(f"{port} is ragged")

    x_dims, freqs_dims = x["dims"], freqs["dims"]
    if y["dims"] != x_dims:
        raise FeatureError(f"{Y} dims {y['dims']} are not {x_dims}")
    if len(freqs_dims) != len(x_dims):
        raise FeatureError(f"{FREQS} rank {len(freqs_dims)} is not that of {X}")

    head_dim = x_dims[-1]
    rope_dim = node.get(ROPE_DIM, 0)
    if not isinstance(rope_dim, int) or rope_dim < 0:
        raise FeatureError(f"{ROPE_DIM} {rope_dim!r} is negative")
    rope_dim = rope_dim or head_dim
    if rope_dim > head_dim or rope_dim % 2 != 0:
        raise FeatureError(f"{ROPE_DIM} {rope_dim} is not even in (0, {head_dim}]")
    offset = node.get(ROPE_OFFSET)
    if offset is None:
        offset = head_dim - rope_dim
    if not isinstance(offset, int) or offset < 0 or offset + rope_dim > head_dim:
        raise FeatureError(f"{ROPE_OFFSET} {offset!r} + {rope_dim} exceeds {head_dim}")

    if freqs_dims[-1] != rope_dim:
        raise FeatureError(f"{FREQS} last dim {freqs_dims[-1]} is not {rope_dim}")
    pairs = list(zip(freqs_dims[:-1], x_dims[:-1]))
    if any(f not in (x_dim, 1) for f, x_dim in pairs):
        raise FeatureError(f"{FREQS} dims {freqs_dims} do not broadcast to {x_dims}")

    if layernorm.catalogued_dtype(freqs, FREQS) != "fp32":
        raise FeatureError(f"{FREQS} dtype {freqs.get('data_type')!r} is not fp32")
    dtype = layernorm.catalogued_dtype(x, X)
    if layernorm.catalogued_dtype(y, Y) != dtype:
        raise FeatureError(f"{Y} dtype {y.get('data_type')!r} is not that of {X}")

    broadcast = any(f != x_dim for f, x_dim in pairs)
    return {
        f"dtype:{dtype}",
        _rotation(head_dim, rope_dim, offset),
        "freqs:broadcast" if broadcast else "freqs:full",
        "scale:unit" if node.get(OUTPUT_SCALE, 1.0) == 1.0 else "scale:scaled",
    }
