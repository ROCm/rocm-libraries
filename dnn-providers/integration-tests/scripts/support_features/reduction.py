# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``ReductionAttributes``.

``y = reduce(x)`` over the dims ``y`` keeps at 1. The features come from
``reduction_attributes.fbs``. Unlike the other ops, the serializer
(``utilities/json/ReductionAttributes.hpp``) writes the ports at the top of
the node, beside ``mode``, not under ``inputs`` and ``outputs``; its reader
takes a missing ``is_deterministic`` as false. No provider implements the
op, so the dtypes are those the CPU reference executor registers
(``ReductionSignatureKey``): ``x`` in each float dtype, ``y`` in its dtype
or fp32. The integration tests' GPU reference takes more pairs, but it
checks a result rather than computing one for a caller, so those raise. By
group:

- ``mode`` (exclusive): the reduction, by its serialized name.
- ``dtype`` (exclusive): the dtype of ``x``.
- ``precision:mixed_output``: ``y`` is not the dtype of ``x``. A virtual
  tensor is not counted, as for matmul: the caller never sees it.
- ``dims`` (exclusive): the rank of ``x`` (and ``y``), 1 to 5.
- ``reduce`` (exclusive): which dims are reduced (``y`` 1, ``x`` above 1),
  against those kept (``y`` and ``x`` the same, above 1); a dim of 1 in both
  is neither. ``all`` (none kept: one value), ``trailing`` (every reduced dim
  after every kept one, as a per-channel sum over H and W), ``leading``
  (every reduced dim before every kept one, as a sum over the batch) or
  ``interleaved``. Read on the logical dims, layout aside.
- ``layout`` (exclusive, when the strides tell): ``nchw`` or ``nhwc``, read
  from ``x`` and ``y``, as for layernorm.
- ``deterministic`` (exclusive): ``off`` or ``on``.

The API requires ``x``, ``y`` and a ``mode`` other than ``NOT_SET``; and
``x`` and ``y`` of one rank, each dim of ``y`` that of ``x`` or 1, with at
least one dim reduced. The CPU reference requires the same, and ``y`` in
``x``'s dtype or fp32. A graph that breaks these raises.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, convolution, layernorm

NODE_TYPE = "ReductionAttributes"
SCHEMA_FILE = "reduction_attributes.fbs"
SCHEMA_TABLE = "ReductionAttributes"

X, Y = "in_tensor_uid", "out_tensor_uid"
MODE, DETERMINISTIC = "mode", "is_deterministic"

# ReductionMode as utilities/json/ReductionAttributes.hpp writes it, but
# "not_set", which the API rejects.
MODES = ("add", "mul", "min", "max", "amax", "avg", "norm1", "norm2", "mul_no_zeros")
DIMS = layernorm.DIMS

FEATURES = (
    *(Feature(f"mode:{mode}", (MODE,)) for mode in MODES),
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in layernorm.DTYPES),
    Feature("precision:mixed_output", (X, Y), "y is not the dtype of x"),
    *(Feature(feature, (X,)) for feature in DIMS.values()),
    Feature("reduce:all", (X, Y), "every dim: one value"),
    Feature("reduce:trailing", (X, Y), "the dims after those kept"),
    Feature("reduce:leading", (X, Y), "the dims before those kept"),
    Feature("reduce:interleaved", (X, Y), "reduced dims between kept ones"),
    Feature("layout:nchw", (X, Y), "channels first"),
    Feature("layout:nhwc", (X, Y), "channels last"),
    Feature("deterministic:off", (DETERMINISTIC,), "any order of accumulation"),
    Feature("deterministic:on", (DETERMINISTIC,), "a reproducible result"),
)

NOT_FEATURES: dict[str, str] = {}


def reduce_feature(x_dims: list[int], y_dims: list[int]) -> str:
    """The ``reduce`` feature, or raise where the API rejects the shapes."""
    if not x_dims or any(d < 1 for d in x_dims + y_dims):
        raise FeatureError(f"{X} dims {x_dims}: not rank >= 1 with dims >= 1")
    if len(y_dims) != len(x_dims):
        raise FeatureError(f"{Y} dims {y_dims} are not the rank of {X} dims {x_dims}")
    if any(y not in (x, 1) for x, y in zip(x_dims, y_dims)):
        raise FeatureError(f"{Y} dims {y_dims} are not {X} dims {x_dims} or 1")
    reduced = [i for i, (x, y) in enumerate(zip(x_dims, y_dims)) if y < x]
    if not reduced:
        raise FeatureError(f"{Y} dims {y_dims} reduce no dim of {X} dims {x_dims}")
    kept = [i for i, (x, y) in enumerate(zip(x_dims, y_dims)) if y == x > 1]
    if not kept:
        return "reduce:all"
    if max(kept) < min(reduced):
        return "reduce:trailing"
    if max(reduced) < min(kept):
        return "reduce:leading"
    return "reduce:interleaved"


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    for port in (X, Y):
        if node.get(port) is None:
            raise FeatureError(f"required port {port} is not bound")
    x, y = tensors_by_uid[node[X]], tensors_by_uid[node[Y]]

    mode = node.get(MODE)
    if mode not in MODES:
        raise FeatureError(f"{MODE} {mode!r} is not catalogued")

    x_dims = x["dims"]
    features = {f"mode:{mode}", reduce_feature(x_dims, y["dims"])}
    if len(x_dims) not in DIMS:
        raise FeatureError(f"{X} dims {x_dims}: rank {len(x_dims)} is not catalogued")
    features.add(DIMS[len(x_dims)])

    dtype = layernorm.catalogued_dtype(x, X)
    features.add(f"dtype:{dtype}")
    y_dtype = layernorm.catalogued_dtype(y, Y)
    if y_dtype not in (dtype, "fp32"):
        raise FeatureError(f"{Y} dtype {y_dtype} with {X} {dtype} has no plan")
    if not x.get("virtual") and not y.get("virtual") and y_dtype != dtype:
        features.add("precision:mixed_output")

    telling = [
        fits for fits in map(convolution.tensor_layouts, (x, y)) if len(fits) == 1
    ]
    agreed = set().union(*telling)
    if len(agreed) > 1:
        raise FeatureError(f"the layouts of {X} and {Y} disagree")
    features.update(f"layout:{name}" for name in agreed)

    deterministic = node.get(DETERMINISTIC, False)
    if type(deterministic) is not bool:
        raise FeatureError(f"{DETERMINISTIC} {deterministic!r} is not a bool")
    features.add("deterministic:on" if deterministic else "deterministic:off")
    return features
