# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``RMSNormAttributes`` (RMS norm, forward).

RMS norm scales ``x`` by the inverse root mean square of its normalized dims,
without subtracting a mean. The features come from ``rmsnorm_attributes.fbs``.
The serializer (``utilities/json/RMSNormAttributes.hpp``) writes the ports
under ``inputs`` / ``outputs`` and leaves an unset optional port out, and
writes ``forward_phase`` by name, leaving it out when it is unset. By group:

- ``dtype`` (exclusive), ``precision:mixed_io``, ``dims`` (exclusive),
  ``layout`` (exclusive): as ``layernorm`` reads them, from ``x`` and ``y``.
  The API's rank floor is 2, not 1.
- ``precision:mixed_affine``, ``precision:mixed_stats``: ``scale`` and
  ``bias``, or the saved ``inv_rms``, are not the dtype of ``x``. The
  hip-kernel provider (the only one with RMS norm) requires the affine
  tensors to agree among themselves, so a split raises.
- ``norm`` (exclusive): the normalized dims. There is no dim count: the API
  reads them from ``scale``, which is the rank of ``x``, 1 on the batch dim
  and the dims before the normalized ones, and ``x``'s on the normalized
  ones (the longest trailing run where the two agree). The batch dim is
  never normalized, so ``all_but_batch`` (one RMS per sample) takes the place
  of layer norm's ``all_dims``; ``last_dim`` is the last only, ``multi_dim``
  more than one but not all of them.
- ``bias`` (exclusive): ``none`` or ``bound``; a bias is ``scale``'s shape.
- ``phase`` (exclusive): ``inference`` or ``training``. Both compute the
  RMS from ``x``; training also writes ``inv_rms`` out, which
  ``Graph::rmsnorm`` binds for training and never for inference. A graph
  that binds it otherwise raises, as does an unset phase, which the API
  rejects. ``inv_rms`` is ``x`` with 1 on the normalized dims.
- ``epsilon`` (exclusive): ``fixed`` or ``runtime``, as ``scalar_source``
  reads it.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm
from support_features import scalar_source

NODE_TYPE = "RMSNormAttributes"
SCHEMA_FILE = "rmsnorm_attributes.fbs"
SCHEMA_TABLE = "RMSNormAttributes"

X, Y = "x_tensor_uid", "y_tensor_uid"
SCALE, BIAS = "scale_tensor_uid", "bias_tensor_uid"
EPSILON = "epsilon_tensor_uid"
INV_RMS = "inv_rms_tensor_uid"
PHASE = "forward_phase"

DIMS = {rank: f"dims:{rank}" for rank in range(2, 6)}
# NormFwdPhase in norm_common.fbs, by the name the serializer writes.
PHASES = {"INFERENCE": "inference", "TRAINING": "training"}

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in layernorm.DTYPES),
    Feature("precision:mixed_io", (X, Y), "y is not the dtype of x"),
    Feature(
        "precision:mixed_affine",
        (X, SCALE, BIAS),
        "scale and bias are not the dtype of x",
    ),
    Feature("precision:mixed_stats", (X, INV_RMS), "inv_rms is not the dtype of x"),
    *(Feature(feature, (X,)) for feature in DIMS.values()),
    Feature("layout:nchw", (X, Y), "channels first"),
    Feature("layout:nhwc", (X, Y), "channels last"),
    Feature("norm:last_dim", (SCALE,), "the last dim only"),
    Feature("norm:multi_dim", (SCALE,), "the last k dims, 1 < k < rank - 1"),
    Feature("norm:all_but_batch", (SCALE,), "every dim but the batch: one RMS each"),
    Feature("bias:none", (BIAS,), "y = scale * x / rms"),
    Feature("bias:bound", (BIAS,), "y = scale * x / rms + bias"),
    Feature("phase:inference", (PHASE,), "y only"),
    Feature("phase:training", (PHASE, INV_RMS), "y and the inverse RMS"),
    Feature("epsilon:fixed", (EPSILON,), "a constant baked into the graph"),
    Feature("epsilon:runtime", (EPSILON,), "passed with each execution"),
)

NOT_FEATURES: dict[str, str] = {}


def normalized_dims(scale: list[int], dims: list[int]) -> int:
    """How many trailing dims of ``dims`` the ``scale`` shape normalizes, as
    ``validateScaleNormalizedShape`` (frontend ``detail/Utilities.hpp``)
    reads it, or raise where it rejects the shape."""
    if len(scale) != len(dims):
        raise FeatureError(f"scale dims {scale} are not the rank of x {dims}")
    matched = 0
    while matched < len(dims) and scale[-1 - matched] == dims[-1 - matched]:
        matched += 1
    # The batch dim is never normalized, even when x's is 1 too; so the
    # leading 1s scale[:-count] always include it.
    count = min(matched, len(dims) - 1)
    if count == 0 or any(s != 1 for s in scale[:-count]):
        raise FeatureError(
            f"scale dims {scale} are not 1 then a trailing run of x {dims}"
        )
    return count


def norm_feature(count: int, rank: int) -> str:
    if count == rank - 1:
        return "norm:all_but_batch"
    return "norm:last_dim" if count == 1 else "norm:multi_dim"


def affine_features(
    affine: dict[str, dict], dims: list[int], dtype: str
) -> tuple[set[str], int]:
    """``norm`` and ``precision:mixed_affine`` from the affine tensors (port ->
    tensor), which share one shape; and the normalized dim count."""
    shapes = {port: t["dims"] for port, t in affine.items()}
    shape = next(iter(shapes.values()))
    if any(s != shape for s in shapes.values()):
        raise FeatureError(f"affine shapes differ: {shapes}")
    count = normalized_dims(shape, dims)
    features = {norm_feature(count, len(dims))}
    if layernorm.role_dtype(affine, "affine") != dtype:
        features.add("precision:mixed_affine")
    return features, count


def stats_features(inv_rms: dict, dims: list[int], count: int, dtype: str) -> set[str]:
    """``precision:mixed_stats``, once ``inv_rms`` is checked against ``x``."""
    layernorm.check_stats({INV_RMS: inv_rms}, dims, count)
    if layernorm.catalogued_dtype(inv_rms, INV_RMS) != dtype:
        return {"precision:mixed_stats"}
    return set()


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (X, Y, SCALE, EPSILON):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    tensor = {port: tensors_by_uid[by_port[port]] for port in by_port if has(port)}

    features, dtype, dims = layernorm.io_features({X: tensor[X], Y: tensor[Y]}, X)
    if len(dims) not in DIMS:
        raise FeatureError(f"{X} dims {dims}: rank {len(dims)} is not catalogued")
    affine = {port: tensor[port] for port in (SCALE, BIAS) if has(port)}
    affine_set, count = affine_features(affine, dims, dtype)
    features |= affine_set
    features.add("bias:bound" if has(BIAS) else "bias:none")

    # type() rather than in: an unhashable value would raise TypeError.
    phase = PHASES.get(node.get(PHASE)) if type(node.get(PHASE)) is str else None
    if phase is None:
        raise FeatureError(f"{PHASE} {node.get(PHASE)!r} is not catalogued")
    features.add(f"phase:{phase}")
    if phase == "inference" and has(INV_RMS):
        raise FeatureError(f"inference binds {INV_RMS}")
    if phase == "training":
        if not has(INV_RMS):
            raise FeatureError(f"training does not bind {INV_RMS}")
        features |= stats_features(tensor[INV_RMS], dims, count, dtype)

    features.add(f"epsilon:{scalar_source(tensor[EPSILON], EPSILON)}")
    return features
