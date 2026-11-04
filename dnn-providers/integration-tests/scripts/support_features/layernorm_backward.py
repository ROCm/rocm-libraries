# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``LayernormBackwardAttributes`` (layer norm, backward).

The gradients of layer norm: ``dx`` from ``dy``, ``x`` and ``scale``, and
``dscale`` and ``dbias``. The features come from
``layernorm_backward_attributes.fbs`` and are read as ``layernorm`` reads the
forward's. By group:

- ``dtype`` (exclusive): the dtype of ``x``.
- ``precision:mixed_io``, ``precision:mixed_affine``,
  ``precision:mixed_stats``: ``dy`` or ``dx``; ``scale``, ``dscale`` and
  ``dbias``; or the saved ``mean`` and ``inv_variance`` are not the dtype of
  ``x``. As in the forward, a split within the affine or the statistics
  tensors raises.
- ``dims`` (exclusive), ``layout`` (exclusive): as the forward, read from
  ``x``, ``dy`` and ``dx``.
- ``norm`` (exclusive), ``affine`` (exclusive): as the forward, from the
  shape ``scale``, ``dscale`` and ``dbias`` share.
- ``stats`` (exclusive): ``saved`` (the forward's mean and inverse variance
  are passed in) or ``recomputed`` (neither is, and the provider computes
  them from ``x``). The API requires both or neither. Saved statistics are
  ``x`` with 1 on the normalized dims when the affine is ``full_rank`` and
  the dims before them only when it is ``reduced_rank``: the API requires
  both one-padded or neither.
- ``epsilon`` (exclusive): ``default`` (unset: the provider's 1e-5),
  ``fixed`` or ``runtime``, as ``scalar_source`` reads it. Epsilon enters
  only the recomputed statistics, so a graph with saved statistics adds no
  epsilon feature; a bound epsilon is still read, and one the API rejects
  raises.

``normalized_dim_count`` is read with the shapes, as in the forward.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm
from support_features import scalar_source

NODE_TYPE = "LayernormBackwardAttributes"
SCHEMA_FILE = "layernorm_backward_attributes.fbs"
SCHEMA_TABLE = "LayernormBackwardAttributes"

X, DY, DX = "x_tensor_uid", "dy_tensor_uid", "dx_tensor_uid"
AFFINE = ("scale_tensor_uid", "dscale_tensor_uid", "dbias_tensor_uid")
EPSILON = "epsilon_tensor_uid"
STATS = layernorm.STATS
COUNT = layernorm.COUNT

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in layernorm.DTYPES),
    Feature("precision:mixed_io", (X, DY, DX), "dy or dx is not the dtype of x"),
    Feature(
        "precision:mixed_affine",
        (X, *AFFINE),
        "scale and its gradients are not the dtype of x",
    ),
    Feature(
        "precision:mixed_stats",
        (X, *STATS),
        "the saved statistics are not the dtype of x",
    ),
    *(Feature(feature, (X,)) for feature in layernorm.DIMS.values()),
    Feature("layout:nchw", (X, DY, DX), "channels first"),
    Feature("layout:nhwc", (X, DY, DX), "channels last"),
    Feature("norm:last_dim", (COUNT, *AFFINE), "the last dim only"),
    Feature("norm:multi_dim", (COUNT, *AFFINE), "the last k dims, 1 < k < rank"),
    Feature("norm:all_dims", (COUNT, *AFFINE), "every dim: one mean in all"),
    Feature(
        "affine:full_rank",
        AFFINE,
        "the rank of x, 1 on the dims not normalized",
    ),
    Feature("affine:reduced_rank", AFFINE, "the normalized dims only"),
    Feature("stats:saved", STATS, "the forward's mean and inverse variance"),
    Feature("stats:recomputed", STATS, "computed again from x"),
    Feature("epsilon:default", (EPSILON,), "unset: the provider's 1e-5"),
    Feature("epsilon:fixed", (EPSILON,), "a constant baked into the graph"),
    Feature("epsilon:runtime", (EPSILON,), "passed with each execution"),
)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (X, DY, DX, *AFFINE):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    tensor = {port: tensors_by_uid[by_port[port]] for port in by_port if has(port)}

    io = {port: tensor[port] for port in (X, DY, DX)}
    features, dtype, dims = layernorm.io_features(io, X)
    count = layernorm.normalized_dims(node.get(COUNT), dims)
    affine = {port: tensor[port] for port in AFFINE}
    features |= layernorm.norm_features(affine, dims, count)
    if layernorm.role_dtype(affine, "affine") != dtype:
        features.add("precision:mixed_affine")

    bound_stats = [port for port in STATS if has(port)]
    if bound_stats and len(bound_stats) != len(STATS):
        raise FeatureError(f"binds the statistics {bound_stats}, not both")
    epsilon = scalar_source(tensor[EPSILON], EPSILON) if has(EPSILON) else "default"
    if bound_stats:
        features.add("stats:saved")
        stats = {port: tensor[port] for port in STATS}
        padded = "affine:full_rank" in features
        layernorm.check_stats(stats, dims, count, padded)
        if layernorm.role_dtype(stats, "statistics") != dtype:
            features.add("precision:mixed_stats")
    else:
        features |= {"stats:recomputed", f"epsilon:{epsilon}"}
    return features
