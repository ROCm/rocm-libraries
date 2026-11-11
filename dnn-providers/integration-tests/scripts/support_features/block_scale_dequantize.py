# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``BlockScaleDequantizeAttributes``.

``y = x * scale`` with one ``scale`` per block of ``x``: the dequantize in
front of a block-scaled (MX) matmul. The features come from
``block_scale_dequantize_attributes.fbs``. The serializer
(``utilities/json/BlockScaleDequantizeAttributes.hpp``) writes the ports
under ``inputs`` (``x``, ``scale``) and ``outputs`` (``y``); its reader
takes a missing ``block_size`` as empty and a missing ``is_negative_scale``
as false. The dtypes are those the CPU reference executor registers
(``BlockScaleDequantizeSignatureKey``): an MX ``x`` with an fp8e8m0
``scale``, a float ``x`` with an fp32 one. hipBLASLt reads only the MX
pairs, and only fused into its matmul. By group:

- ``dtype`` (exclusive): the dtype of ``x``: an OCP MX dtype (fp8, fp6 or
  fp4) or a float one. MX is OCP only, so a FNUZ ``x`` raises.
- ``scale_dtype`` (exclusive): ``fp8e8m0`` (a power of two, the MX scale)
  or ``fp32``.
- ``block`` (exclusive): which dims of ``x`` are blocked. ``block_size``
  applies to the trailing dims, right-aligned, and a dim it does not reach
  has a block of 1; a dim is blocked when its block is above 1. ``element``
  (none: a scale per element), ``last_dim`` (the last only), ``other_dim``
  (one other) or ``tile`` (more than one).
- ``block_length`` (exclusive, when a dim is blocked): ``32`` (every
  blocked dim, the MX block) or ``other``.
- ``fit`` (exclusive, when a dim is blocked): ``exact`` (every blocked dim
  is a multiple of its block) or ``ragged`` (one is not, so its last block
  is partial).
- ``scale`` (exclusive): ``multiply`` (``y = x * scale``) or
  ``negative_exponent`` (``is_negative_scale``: ``y = x * 2^-scale``).

The API requires ``x``, ``scale`` and a virtual ``y`` of ``x``'s dims, and
a ``block_size`` of 1 to rank-of-``x`` entries, each above 0. The CPU
reference (``CpuFpReferenceBlockScaleDequantize``) requires ``scale`` of
``x``'s rank with ``ceil(x / block)`` on each dim; no provider takes another
shape. A graph that breaks these raises.

No layout or rank feature: ``y`` takes ``x``'s dims and strides, and the
matmul that consumes it counts them (``a_layout``, ``b_layout``, ``batch``).
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm

NODE_TYPE = "BlockScaleDequantizeAttributes"
SCHEMA_FILE = "block_scale_dequantize_attributes.fbs"
SCHEMA_TABLE = "BlockScaleDequantizeAttributes"

X, SCALE, Y = "x_tensor_uid", "scale_tensor_uid", "y_tensor_uid"
BLOCK_SIZE, NEGATIVE = "block_size", "is_negative_scale"

# data_type as utilities/json/Common.hpp writes it -> label.
DTYPES = {
    "fp8_e4m3": "fp8e4m3",
    "fp8_e5m2": "fp8e5m2",
    "fp6_e2m3": "fp6e2m3",
    "fp6_e3m2": "fp6e3m2",
    "fp4_e2m1": "fp4e2m1",
    "float": "fp32",
    "half": "fp16",
    "bfloat16": "bf16",
}
SCALE_DTYPES = {"fp8_e8m0": "fp8e8m0", "float": "fp32"}
MX_BLOCK = 32

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in DTYPES.values()),
    Feature("scale_dtype:fp8e8m0", (SCALE,), "a power of two: the MX scale"),
    Feature("scale_dtype:fp32", (SCALE,), "any fp32 value"),
    Feature("block:element", (X, BLOCK_SIZE), "no dim blocked: a scale each"),
    Feature("block:last_dim", (X, BLOCK_SIZE), "the last dim only"),
    Feature("block:other_dim", (X, BLOCK_SIZE), "one dim, not the last"),
    Feature("block:tile", (X, BLOCK_SIZE), "more than one dim"),
    Feature(f"block_length:{MX_BLOCK}", (BLOCK_SIZE,), "the MX block"),
    Feature("block_length:other", (BLOCK_SIZE,), "a blocked dim is not 32"),
    Feature("fit:exact", (X, BLOCK_SIZE), "every block is whole"),
    Feature("fit:ragged", (X, BLOCK_SIZE), "a dim ends in a partial block"),
    Feature("scale:multiply", (NEGATIVE,), "y = x * scale"),
    Feature("scale:negative_exponent", (NEGATIVE,), "y = x * 2^-scale"),
)

NOT_FEATURES: dict[str, str] = {
    Y: "virtual (the API requires it) with x's dims and strides; its dtype "
    "is the consuming op's input dtype, which that op's catalog counts",
}


def catalogued(tensor: dict, port: str, labels: dict[str, str]) -> str:
    # type() rather than in: an unhashable value would raise TypeError.
    dtype = tensor.get("data_type")
    if type(dtype) is not str or dtype not in labels:
        raise FeatureError(f"{port} dtype {dtype!r} is not catalogued")
    return labels[dtype]


def effective_block(node: dict, dims: list[int]) -> list[int]:
    """The block on each dim of ``x``: ``block_size`` right-aligned to the
    trailing dims, 1 before them; or raise where the API rejects it."""
    block_size = node.get(BLOCK_SIZE)
    if type(block_size) is not list or not 1 <= len(block_size) <= len(dims):
        raise FeatureError(
            f"{BLOCK_SIZE} {block_size}: not 1 to {len(dims)} entries for {X} dims"
        )
    if any(type(b) is not int or b < 1 for b in block_size):
        raise FeatureError(f"{BLOCK_SIZE} {block_size}: an entry is not an int above 0")
    return [1] * (len(dims) - len(block_size)) + list(block_size)


def block_feature(blocked: list[int], rank: int) -> str:
    if not blocked:
        return "block:element"
    if len(blocked) > 1:
        return "block:tile"
    return "block:last_dim" if blocked[0] == rank - 1 else "block:other_dim"


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (X, SCALE, Y):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    x, scale, y = (tensors_by_uid[by_port[port]] for port in (X, SCALE, Y))

    if not y.get("virtual"):
        raise FeatureError(f"{Y} is not virtual")
    dims = x["dims"]
    if not dims or any(d < 1 for d in dims):
        raise FeatureError(f"{X} dims {dims}: not rank >= 1 with dims >= 1")
    if y["dims"] != dims:
        raise FeatureError(f"{Y} dims {y['dims']} are not {X} dims {dims}")
    block = effective_block(node, dims)
    expected = [-(-d // b) for d, b in zip(dims, block)]
    if scale["dims"] != expected:
        raise FeatureError(f"{SCALE} dims {scale['dims']} are not {expected}")

    features = {
        f"dtype:{catalogued(x, X, DTYPES)}",
        f"scale_dtype:{catalogued(scale, SCALE, SCALE_DTYPES)}",
    }
    blocked = [d for d, b in enumerate(block) if b > 1]
    features.add(block_feature(blocked, len(dims)))
    if blocked:
        lengths = {block[d] for d in blocked}
        features.add(
            f"block_length:{MX_BLOCK}"
            if lengths == {MX_BLOCK}
            else "block_length:other"
        )
        ragged = any(dims[d] % block[d] for d in blocked)
        features.add("fit:ragged" if ragged else "fit:exact")

    negative = node.get(NEGATIVE, False)
    if type(negative) is not bool:
        raise FeatureError(f"{NEGATIVE} {negative!r} is not a bool")
    features.add("scale:negative_exponent" if negative else "scale:multiply")
    return features
