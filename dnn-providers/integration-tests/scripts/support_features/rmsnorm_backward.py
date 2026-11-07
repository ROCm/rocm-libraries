# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``RMSNormBackwardAttributes`` (RMS norm, backward).

The gradients of RMS norm: ``dx`` from ``dy``, ``x``, ``scale`` and the
forward's saved ``inv_rms``, and ``dscale`` and, optionally, ``dbias``. The
features come from ``rmsnorm_backward_attributes.fbs`` and are read as
``rmsnorm`` reads the forward's. The serializer
(``utilities/json/RMSNormBackwardAttributes.hpp``) leaves an unset ``dbias``
out. By group:

- ``dtype`` (exclusive): the dtype of ``x``.
- ``precision:mixed_io``, ``precision:mixed_affine``,
  ``precision:mixed_stats``: ``dy`` or ``dx``; ``scale``, ``dscale`` and
  ``dbias``; or ``inv_rms`` are not the dtype of ``x``. ``dy`` is often the
  virtual fp32 gradient out of a fused pointwise backward, which is not
  counted. As in the forward, a split within the affine tensors raises.
- ``dims`` (exclusive), ``layout`` (exclusive): as the forward, read from
  ``x``, ``dy`` and ``dx``.
- ``norm`` (exclusive): as the forward, from the shape ``scale``,
  ``dscale`` and ``dbias`` share.
- ``dbias`` (exclusive): ``none`` or ``bound`` (the forward had a bias, and
  its gradient is computed too).

``inv_rms`` is required (there is no recompute path, so no ``epsilon``) and
is ``x`` with 1 on the normalized dims, as in the forward.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm, rmsnorm

NODE_TYPE = "RMSNormBackwardAttributes"
SCHEMA_FILE = "rmsnorm_backward_attributes.fbs"
SCHEMA_TABLE = "RMSNormBackwardAttributes"

X, DY, DX = "x_tensor_uid", "dy_tensor_uid", "dx_tensor_uid"
SCALE, DSCALE, DBIAS = "scale_tensor_uid", "dscale_tensor_uid", "dbias_tensor_uid"
AFFINE = (SCALE, DSCALE, DBIAS)
INV_RMS = rmsnorm.INV_RMS

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in layernorm.DTYPES),
    Feature("precision:mixed_io", (X, DY, DX), "dy or dx is not the dtype of x"),
    Feature(
        "precision:mixed_affine",
        (X, *AFFINE),
        "scale and its gradients are not the dtype of x",
    ),
    Feature("precision:mixed_stats", (X, INV_RMS), "inv_rms is not the dtype of x"),
    *(Feature(feature, (X,)) for feature in rmsnorm.DIMS.values()),
    Feature("layout:nchw", (X, DY, DX), "channels first"),
    Feature("layout:nhwc", (X, DY, DX), "channels last"),
    Feature("norm:last_dim", AFFINE, "the last dim only"),
    Feature("norm:multi_dim", AFFINE, "the last k dims, 1 < k < rank - 1"),
    Feature("norm:all_but_batch", AFFINE, "every dim but the batch: one RMS each"),
    Feature("dbias:none", (DBIAS,), "dx and dscale only"),
    Feature("dbias:bound", (DBIAS,), "the bias gradient too"),
)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (X, DY, DX, SCALE, DSCALE, INV_RMS):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    tensor = {port: tensors_by_uid[by_port[port]] for port in by_port if has(port)}

    io = {port: tensor[port] for port in (X, DY, DX)}
    features, dtype, dims = layernorm.io_features(io, X)
    if len(dims) not in rmsnorm.DIMS:
        raise FeatureError(f"{X} dims {dims}: rank {len(dims)} is not catalogued")
    affine = {port: tensor[port] for port in AFFINE if has(port)}
    affine_set, count = rmsnorm.affine_features(affine, dims, dtype)
    features |= affine_set
    features.add("dbias:bound" if has(DBIAS) else "dbias:none")
    features |= rmsnorm.stats_features(tensor[INV_RMS], dims, count, dtype)
    return features
