# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``LayernormAttributes`` (layer norm, forward).

Layer norm normalizes ``x`` over its last ``k`` dims: one mean and variance
for each index into the dims before them. The features come from
``layernorm_attributes.fbs``. The serializer
(``utilities/json/LayernormAttributes.hpp``) writes the ports under
``inputs`` / ``outputs``, an unset optional port as null, and
``normalized_dim_count`` and ``forward_phase`` beside them. By group:

- ``dtype`` (exclusive): the dtype of ``x``.
- ``precision:mixed_io``, ``precision:mixed_affine``,
  ``precision:mixed_stats``: ``y``; ``scale`` and ``bias``; or the saved
  ``mean`` and ``inv_variance`` are not the dtype of ``x``. The hip-kernel
  provider (the only one with layer norm) takes the dtype of each of these
  roles independently but requires the affine and the statistics tensors to
  agree among themselves, so a split within a role raises. A virtual ``y``,
  the intermediate into a fused op, is not counted, as for batch norm.
- ``dims`` (exclusive): the rank of ``x``, 1 (the API's floor) to 5.
- ``layout`` (exclusive): channels first or channels last, read from ``x``
  and ``y`` as ``convolution.tensor_layouts`` reads them. Below rank 3, or
  when the strides do not tell, the graph adds no layout feature.
- ``norm`` (exclusive): the normalized dims. ``last_dim`` is the last only
  (a transformer's hidden dim), ``multi_dim`` more than one but not all,
  ``all_dims`` every dim (one mean for the whole tensor).
- ``affine`` (exclusive): the shape ``scale`` and ``bias`` share.
  ``full_rank`` is the rank of ``x``, 1 on the dims not normalized (the
  frontend's default); ``reduced_rank`` is the normalized dims only. When
  every dim is normalized the two are one shape, counted as ``full_rank``.
- ``phase`` (exclusive): ``inference`` or ``training``. Unlike batch norm,
  both compute the statistics from ``x``; training also writes the mean and
  inverse variance out, which ``Graph::layernorm`` binds for training and
  never for inference. A graph that binds them otherwise raises, as does an
  unset phase, which the API rejects.
- ``epsilon`` (exclusive): ``fixed`` (a constant baked into the graph) or
  ``runtime`` (passed with each execution), as ``scalar_source`` reads it.

``normalized_dim_count`` is internal: the frontend infers it from the scale
shape, and the provider ignores it and reads the normalized dims from the
scale and statistics shapes. So it is read together with them, and a count
the shapes do not agree with raises: scale and bias equal ``x`` on the
normalized dims, the statistics are ``x`` with 1 on them.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, convolution
from support_features import scalar_source
from support_matrix_units import dtype_label

NODE_TYPE = "LayernormAttributes"
SCHEMA_FILE = "layernorm_attributes.fbs"
SCHEMA_TABLE = "LayernormAttributes"

X, Y = "x_tensor_uid", "y_tensor_uid"
AFFINE = ("scale_tensor_uid", "bias_tensor_uid")
EPSILON = "epsilon_tensor_uid"
STATS = ("mean_tensor_uid", "inv_variance_tensor_uid")
COUNT = "normalized_dim_count"
PHASE = "forward_phase"

DTYPES = convolution.DTYPES
DIMS = {rank: f"dims:{rank}" for rank in range(1, 6)}
# NormFwdPhase in norm_common.fbs, as the serializer writes it. NOT_SET (0)
# the API rejects.
PHASES = {1: "inference", 2: "training"}

FEATURES = (
    *(Feature(f"dtype:{dtype}", (X,)) for dtype in DTYPES),
    Feature("precision:mixed_io", (X, Y), "y is not the dtype of x"),
    Feature(
        "precision:mixed_affine", (X, *AFFINE), "scale and bias are not the dtype of x"
    ),
    Feature(
        "precision:mixed_stats",
        (X, *STATS),
        "the saved statistics are not the dtype of x",
    ),
    *(Feature(feature, (X,)) for feature in DIMS.values()),
    Feature("layout:nchw", (X, Y), "channels first"),
    Feature("layout:nhwc", (X, Y), "channels last"),
    Feature("norm:last_dim", (COUNT, *AFFINE), "the last dim only"),
    Feature("norm:multi_dim", (COUNT, *AFFINE), "the last k dims, 1 < k < rank"),
    Feature("norm:all_dims", (COUNT, *AFFINE), "every dim: one mean in all"),
    Feature(
        "affine:full_rank",
        AFFINE,
        "the rank of x, 1 on the dims not normalized",
    ),
    Feature("affine:reduced_rank", AFFINE, "the normalized dims only"),
    Feature("phase:inference", (PHASE,), "y only"),
    Feature("phase:training", (PHASE, *STATS), "y, the mean and inverse variance"),
    Feature("epsilon:fixed", (EPSILON,), "a constant baked into the graph"),
    Feature("epsilon:runtime", (EPSILON,), "passed with each execution"),
)

NOT_FEATURES: dict[str, str] = {}


def catalogued_dtype(tensor: dict, port: str) -> str:
    dtype = dtype_label(tensor.get("data_type"))
    if dtype not in DTYPES:
        raise FeatureError(f"{port} dtype {dtype!r} is not catalogued")
    return dtype


def role_dtype(tensors: dict[str, dict], role: str) -> str:
    """The one dtype the ``role``'s tensors (port -> tensor) share, or raise."""
    labels = {port: catalogued_dtype(t, port) for port, t in tensors.items()}
    if len(set(labels.values())) > 1:
        raise FeatureError(f"{role} dtypes differ: {labels}")
    return next(iter(labels.values()))


def io_features(tensors: dict[str, dict], data: str) -> tuple[set[str], str, list[int]]:
    """``dtype``, ``precision:mixed_io``, ``dims`` and ``layout``; the dtype
    and dims of ``data``.

    ``tensors`` maps each activation-shaped port to its tensor; they must
    share one shape, as the API requires.
    """
    dims = tensors[data]["dims"]
    if any(t["dims"] != dims for t in tensors.values()):
        shapes = {port: t["dims"] for port, t in tensors.items()}
        raise FeatureError(f"activation shapes differ: {shapes}")
    if len(dims) not in DIMS:
        raise FeatureError(f"{data} dims {dims}: rank {len(dims)} is not catalogued")

    dtype = catalogued_dtype(tensors[data], data)
    features = {f"dtype:{dtype}", DIMS[len(dims)]}
    for port, tensor in tensors.items():
        if catalogued_dtype(tensor, port) != dtype and not tensor.get("virtual"):
            features.add("precision:mixed_io")

    telling = [
        fits
        for fits in map(convolution.tensor_layouts, tensors.values())
        if len(fits) == 1
    ]
    agreed = set().union(*telling)
    if len(agreed) > 1:
        raise FeatureError("the activation tensors' layouts disagree")
    features.update(f"layout:{name}" for name in agreed)
    return features, dtype, dims


def normalized_dims(count: object, dims: list[int]) -> int:
    """``count``, if it is a number of trailing dims of ``dims``, or raise."""
    if type(count) is not int or not 1 <= count <= len(dims):
        raise FeatureError(f"{COUNT} {count!r} is not 1 to {len(dims)} for {dims}")
    return count


def norm_features(affine: dict[str, dict], dims: list[int], count: int) -> set[str]:
    """``norm`` and ``affine``, from the affine tensors (port -> tensor)."""
    rank = len(dims)
    reduced = dims[rank - count :]
    full = [1] * (rank - count) + reduced
    shapes = {port: t["dims"] for port, t in affine.items()}
    shape = next(iter(shapes.values()))
    if any(s != shape for s in shapes.values()):
        raise FeatureError(f"affine shapes differ: {shapes}")
    if shape == full:
        features = {"affine:full_rank"}
    elif shape == reduced:
        features = {"affine:reduced_rank"}
    else:
        raise FeatureError(
            f"affine dims {shape} are neither {full} nor {reduced}: x {dims} "
            f"normalized over its last {count}"
        )
    if count == rank:
        features.add("norm:all_dims")
    else:
        features.add("norm:last_dim" if count == 1 else "norm:multi_dim")
    return features


def check_stats(
    stats: dict[str, dict], dims: list[int], count: int, padded: bool = True
) -> None:
    """The statistics (port -> tensor) are ``x`` with 1 on the normalized dims
    or, unless ``padded``, the dims before them only."""
    expected = dims[: len(dims) - count] + ([1] * count if padded else [])
    for port, tensor in stats.items():
        if tensor["dims"] != expected:
            raise FeatureError(f"{port} dims {tensor['dims']} are not {expected}")


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in (X, Y, *AFFINE, EPSILON):
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    tensor = {port: tensors_by_uid[by_port[port]] for port in by_port if has(port)}

    features, dtype, dims = io_features({X: tensor[X], Y: tensor[Y]}, X)
    count = normalized_dims(node.get(COUNT), dims)
    affine = {port: tensor[port] for port in AFFINE}
    features |= norm_features(affine, dims, count)
    if role_dtype(affine, "affine") != dtype:
        features.add("precision:mixed_affine")

    # type() rather than ==: True and 2.0 hash as 1 and 2.
    phase = PHASES.get(node.get(PHASE)) if type(node.get(PHASE)) is int else None
    if phase is None:
        raise FeatureError(f"{PHASE} {node.get(PHASE)!r} is not catalogued")
    features.add(f"phase:{phase}")
    bound_stats = [port for port in STATS if has(port)]
    if phase == "inference" and bound_stats:
        raise FeatureError(f"inference binds the statistics {bound_stats}")
    if phase == "training":
        if len(bound_stats) != len(STATS):
            raise FeatureError(f"training binds the statistics {bound_stats}, not all")
        stats = {port: tensor[port] for port in STATS}
        check_stats(stats, dims, count)
        if role_dtype(stats, "statistics") != dtype:
            features.add("precision:mixed_stats")

    features.add(f"epsilon:{scalar_source(tensor[EPSILON], EPSILON)}")
    return features
