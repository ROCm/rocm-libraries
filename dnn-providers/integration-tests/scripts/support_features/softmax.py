# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``SoftmaxAttributes``.

Softmax along the last dim, mirroring cuDNN's unified softmax:
``s = exp(p - max(p)) / sum(exp(p - max(p)))`` per row, and, when ``stats`` is
bound, the log-sum-exp ``stats = max(p) + ln(sum(exp(p - max(p))))``, in log2
units with ``stats_use_log2``. The features come from
``softmax_attributes.fbs``. The serializer
(``utilities/json/SoftmaxAttributes.hpp``) writes the ports under ``inputs``
and ``outputs`` (``stats_tensor_uid`` null when unbound) and
``stats_use_log2`` at the top level. No provider implements the op, so the
dtypes are those the CPU reference executor registers
(``SoftmaxSignatureKey``): ``s`` of the dtype of ``p`` or fp32, ``stats``
fp32. By group:

- ``dtype`` (exclusive): the dtype of ``p``.
- ``output`` (exclusive): ``same`` (``s`` has the dtype of ``p``) or ``fp32``
  (``s`` is fp32 for an fp16/bf16 ``p``).
- ``stats`` (exclusive): ``absent``, ``natural`` (natural-log units) or
  ``log2``.

The API and the CPU reference require rank >= 1, ``s`` of the dims of ``p``,
``stats`` of the dims of ``p`` with the last dim 1, and no ragged tensor. A
graph that breaks these raises.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm

NODE_TYPE = "SoftmaxAttributes"
SCHEMA_FILE = "softmax_attributes.fbs"
SCHEMA_TABLE = "SoftmaxAttributes"

P, S, STATS = "p_tensor_uid", "s_tensor_uid", "stats_tensor_uid"
STATS_USE_LOG2 = "stats_use_log2"
REQUIRED = (P, S)
RAGGED = "ragged_offset_tensor_uid"

FEATURES = (
    *(Feature(f"dtype:{dtype}", (P,)) for dtype in layernorm.DTYPES),
    Feature("output:same", (S,), "s has the dtype of p"),
    Feature("output:fp32", (S,), "s is fp32 for a narrower p"),
    Feature("stats:absent", (STATS,), "no log-sum-exp output"),
    Feature(
        "stats:natural", (STATS, STATS_USE_LOG2), "log-sum-exp in natural-log units"
    ),
    Feature("stats:log2", (STATS, STATS_USE_LOG2), "log-sum-exp in log2 units"),
)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    ports = [*REQUIRED, *((STATS,) if has(STATS) else ())]
    tensors = {port: tensors_by_uid[by_port[port]] for port in ports}

    for port, tensor in tensors.items():
        dims = tensor["dims"]
        if not dims or any(d < 0 for d in dims):
            raise FeatureError(f"{port} dims {dims}: not rank >= 1 with dims >= 0")
        # Written only when set (CHECK_NO_RAGGED_TENSORS); uid 0 is one too.
        if tensor.get(RAGGED) is not None:
            raise FeatureError(f"{port} is ragged")

    p_dims = tensors[P]["dims"]
    if tensors[S]["dims"] != p_dims:
        raise FeatureError(f"{S} dims {tensors[S]['dims']} are not {p_dims}")
    if STATS in tensors:
        expected = p_dims[:-1] + [1]
        if tensors[STATS]["dims"] != expected:
            raise FeatureError(
                f"{STATS} dims {tensors[STATS]['dims']} are not {expected}"
            )
        if layernorm.catalogued_dtype(tensors[STATS], STATS) != "fp32":
            stats_dtype = tensors[STATS].get("data_type")
            raise FeatureError(f"{STATS} dtype {stats_dtype!r} is not fp32")

    dtype = layernorm.catalogued_dtype(tensors[P], P)
    s_dtype = layernorm.catalogued_dtype(tensors[S], S)
    if s_dtype == dtype:
        output = "output:same"
    elif s_dtype == "fp32":
        output = "output:fp32"
    else:
        s_raw = tensors[S].get("data_type")
        raise FeatureError(f"{S} dtype {s_raw!r} is neither that of {P} nor fp32")

    if STATS not in tensors:
        stats = "stats:absent"
    elif node.get(STATS_USE_LOG2, False):
        stats = "stats:log2"
    else:
        stats = "stats:natural"

    return {f"dtype:{dtype}", output, stats}
