# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``MatmulAttributes``.

``c = a @ b`` over the last two dims, ``a`` ``[..., M, K]`` and ``b``
``[..., K, N]``, batched over the dims before them. The features come from
``matmul_attributes.fbs``: three tensor ports, which the serializer
(``utilities/json/MatmulAttributes.hpp``) writes under ``inputs`` (``a``,
``b``) and ``outputs`` (``c``). The hipBLASLt provider (the only one with
matmul) takes the dtype of each independently. By group:

- ``dtype`` (exclusive): the dtype of ``a``.
- ``precision:mixed_inputs``, ``precision:mixed_output``: ``b``, or ``c``,
  is not the dtype of ``a``. A virtual tensor -- the fp32 intermediate into
  a fused bias or activation, or out of a block-scale dequantize -- is not
  counted, as for convolution: the caller never sees it.
- ``batch`` (exclusive): ``none`` (rank 2), ``single`` (one batch dim) or
  ``multi`` (more, which the provider flattens into one batch count).
- ``broadcast:a``, ``broadcast:b``: ``a`` (or ``b``) is smaller than ``c``
  on some batch dim, so one matrix serves several of ``c``'s.
- ``a_layout``, ``b_layout``, ``c_layout`` (each exclusive): ``row_major``
  (unit stride on the last dim) or ``col_major`` (on the one before, a
  transposed operand), read in that order as the provider's ``getTrans``
  reads them, so a matrix with a dim of 1 is row major. A virtual ``c`` adds
  no layout: the fused op's output is the matrix the provider writes.

The API requires ``a`` and ``b`` of one rank of at least 2, with the same
``K`` and batch dims of which one divides the other on each dim, and infers
``c`` as the larger of each batch dim, then ``M`` and ``N``. A graph that
breaks these raises, as does a matrix with no unit stride on either of its
last two dims, which no layout describes.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm

NODE_TYPE = "MatmulAttributes"
SCHEMA_FILE = "matmul_attributes.fbs"
SCHEMA_TABLE = "MatmulAttributes"

A, B, C = "a_tensor_uid", "b_tensor_uid", "c_tensor_uid"
PORTS = (A, B, C)
LAYOUTS = ("row_major", "col_major")

FEATURES = (
    *(Feature(f"dtype:{dtype}", (A,)) for dtype in layernorm.DTYPES),
    Feature("precision:mixed_inputs", (A, B), "b is not the dtype of a"),
    Feature("precision:mixed_output", (A, C), "c is not the dtype of a"),
    Feature("batch:none", PORTS, "rank 2: one matrix product"),
    Feature("batch:single", PORTS, "one batch dim"),
    Feature("batch:multi", PORTS, "more than one batch dim"),
    Feature("broadcast:a", (A, C), "a is reused across c's batch"),
    Feature("broadcast:b", (B, C), "b is reused across c's batch"),
    *(
        Feature(f"{name}_layout:{layout}", (port,), description)
        for name, port in (("a", A), ("b", B), ("c", C))
        for layout, description in zip(LAYOUTS, ("not transposed", "transposed"))
    ),
)

NOT_FEATURES: dict[str, str] = {}


def matrix_layout(tensor: dict, port: str) -> str:
    """``row_major`` or ``col_major``, as the provider's ``getTrans`` reads
    the last two strides, or raise."""
    strides = tensor["strides"]
    if strides[-1] == 1:
        return "row_major"
    if strides[-2] == 1:
        return "col_major"
    raise FeatureError(f"{port} strides {strides} have no unit stride on a matrix dim")


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in PORTS:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    a, b, c = (tensors_by_uid[by_port[port]] for port in PORTS)

    a_dims, b_dims = a["dims"], b["dims"]
    if any(d < 1 for d in a_dims + b_dims):
        raise FeatureError(f"{A} dims {a_dims} or {B} dims {b_dims} has a dim below 1")
    if len(a_dims) < 2 or len(a_dims) != len(b_dims):
        raise FeatureError(
            f"{A} dims {a_dims} and {B} dims {b_dims}: not one rank >= 2"
        )
    if a_dims[-1] != b_dims[-2]:
        raise FeatureError(f"K differs: {A} dims {a_dims}, {B} dims {b_dims}")
    batch = list(zip(a_dims[:-2], b_dims[:-2]))
    if any(x % y and y % x for x, y in batch):
        raise FeatureError(f"batch dims of {a_dims} and {b_dims} do not broadcast")
    expected = [max(x, y) for x, y in batch] + [a_dims[-2], b_dims[-1]]
    if c["dims"] != expected:
        raise FeatureError(f"{C} dims {c['dims']} are not {expected}")

    dtype = {
        port: layernorm.catalogued_dtype(t, port) for port, t in zip(PORTS, (a, b, c))
    }
    features = {f"dtype:{dtype[A]}"}
    if not a.get("virtual") and not b.get("virtual") and dtype[B] != dtype[A]:
        features.add("precision:mixed_inputs")
    if not a.get("virtual") and not c.get("virtual") and dtype[C] != dtype[A]:
        features.add("precision:mixed_output")

    features.add(
        "batch:none"
        if not batch
        else "batch:single" if len(batch) == 1 else "batch:multi"
    )
    if any(x < y for x, y in batch):
        features.add("broadcast:a")
    if any(y < x for x, y in batch):
        features.add("broadcast:b")

    features.add(f"a_layout:{matrix_layout(a, A)}")
    features.add(f"b_layout:{matrix_layout(b, B)}")
    if not c.get("virtual"):
        features.add(f"c_layout:{matrix_layout(c, C)}")
    return features
