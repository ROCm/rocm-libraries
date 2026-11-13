# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``PointwiseAttributes`` (element-wise ops).

The features come from ``pointwise_attributes.fbs``. Groups:

- ``mode`` (exclusive): one feature per ``PointwiseMode`` but ``UNSET``.
  Forward and backward modes count separately, as they are separate kernels.
  The arity of each mode -- which of ``in_1`` / ``in_2`` it takes -- follows
  ``PointwiseValidation.hpp``; a graph that binds the wrong inputs raises.
- ``dtype`` (exclusive): the dtype of ``in_0``, the data input.
- ``precision:mixed``: the float ports do not all share one dtype.
- ``broadcast:in``: an input's dims differ from the output's.
- The mode parameters (relu clips and slope, swish / elu / softplus scalars).
  Each counts only at a value other than its default: a parameter left at
  its default exercises nothing a graph without it does not, so a kernel
  that ignored it would still pass. The defaults are those of the CPU
  reference (``UnaryOperationFunctors.hpp``, ``ReferencePointwiseBase.hpp``)
  and of ``hip-kernel-provider/src/core/Utils.cpp``.

A parameter set on a mode that does not read it raises: the graph is not
testing what it appears to.

The ``dtype`` value set is not in the schema, which types it only as a
tensor. It lists the float dtypes; integer and bool data inputs (to the
logical and comparison modes) raise until they are catalogued.
"""

from __future__ import annotations

from support_features import Feature, FeatureError
from support_matrix_units import dtype_label

NODE_TYPE = "PointwiseAttributes"
SCHEMA_FILE = "pointwise_attributes.fbs"
SCHEMA_TABLE = "PointwiseAttributes"

# PointwiseValidation.hpp, in the serializer's lowercase spelling.
UNARY = (
    "abs",
    "ceil",
    "elu_fwd",
    "erf",
    "exp",
    "floor",
    "gelu_approx_tanh_fwd",
    "gelu_fwd",
    "gen_index",
    "identity",
    "log",
    "logical_not",
    "neg",
    "reciprocal",
    "relu_fwd",
    "rsqrt",
    "sigmoid_fwd",
    "sin",
    "softplus_fwd",
    "sqrt",
    "swish_fwd",
    "tan",
    "tanh_fwd",
)
BINARY = (
    "add",
    "add_square",
    "cmp_eq",
    "cmp_ge",
    "cmp_gt",
    "cmp_le",
    "cmp_lt",
    "cmp_neq",
    "div",
    "elu_bwd",
    "gelu_approx_tanh_bwd",
    "gelu_bwd",
    "logical_and",
    "logical_or",
    "max_op",
    "min_op",
    "mul",
    "relu_bwd",
    "sigmoid_bwd",
    "softplus_bwd",
    "sub",
    "swish_bwd",
    "tanh_bwd",
)
TERNARY = ("binary_select",)
ARITY = {
    **{mode: 1 for mode in UNARY},
    **{mode: 2 for mode in BINARY},
    **{mode: 3 for mode in TERNARY},
}

IN_PORTS = ("in_0_tensor_uid", "in_1_tensor_uid", "in_2_tensor_uid")
OUT_PORT = "out_0_tensor_uid"

DTYPES = ("fp32", "fp16", "bf16")
# A non-data port may also be bool: a comparison's output, a select's condition.
_PORT_DTYPES = DTYPES + ("bool",)

RELU_MODES = ("relu_fwd", "relu_bwd")
# Parameter -> (the modes that read it, its default). A default of None means
# any set value is non-default.
PARAMETERS = {
    "relu_lower_clip": (RELU_MODES, 0.0),
    "relu_upper_clip": (RELU_MODES, None),
    "relu_lower_clip_slope": (RELU_MODES, 0.0),
    "swish_beta": (("swish_fwd", "swish_bwd"), 1.0),
    "elu_alpha": (("elu_fwd", "elu_bwd"), 1.0),
    "softplus_beta": (("softplus_fwd", "softplus_bwd"), 1.0),
    "axis_tensor_uid": (("gen_index",), None),
}
# Parameter -> the feature it drives when non-default. axis has none of its
# own: gen_index needs it, so it is part of mode:gen_index.
PARAMETER_FEATURES = {
    "relu_lower_clip": "relu:lower_clip",
    "relu_upper_clip": "relu:upper_clip",
    "relu_lower_clip_slope": "relu:leaky",
    "swish_beta": "swish:beta",
    "elu_alpha": "elu:alpha",
    "softplus_beta": "softplus:beta",
}


def _mode_fields(mode: str) -> tuple[str, ...]:
    fields = ("operation",) + IN_PORTS[: ARITY[mode]]
    if mode == "gen_index":
        fields += ("axis_tensor_uid",)
    return fields


FEATURES = (
    *(Feature(f"mode:{mode}", _mode_fields(mode)) for mode in ARITY),
    *(Feature(f"dtype:{dtype}", ("in_0_tensor_uid",)) for dtype in DTYPES),
    Feature(
        "precision:mixed",
        IN_PORTS + (OUT_PORT,),
        "the float ports do not all share one dtype",
    ),
    Feature(
        "broadcast:in",
        IN_PORTS + (OUT_PORT,),
        "an input's dims differ from the output's",
    ),
    Feature("relu:lower_clip", ("relu_lower_clip",), "a lower clip other than 0"),
    Feature("relu:upper_clip", ("relu_upper_clip",), "any upper clip"),
    Feature(
        "relu:leaky",
        ("relu_lower_clip_slope",),
        "a slope below the lower clip other than 0",
    ),
    Feature("swish:beta", ("swish_beta",), "a beta other than 1"),
    Feature("elu:alpha", ("elu_alpha",), "an alpha other than 1"),
    Feature("softplus:beta", ("softplus_beta",), "a beta other than 1"),
)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    # The serializer writes every scalar and input port under "inputs" and
    # only out_0 under "outputs" (utilities/json/PointwiseAttributes.hpp).
    inputs = node.get("inputs") or {}
    outputs = node.get("outputs") or {}

    mode = inputs.get("operation")
    if mode not in ARITY:
        raise FeatureError(f"pointwise mode {mode!r} is not catalogued")
    bound = tuple(inputs.get(port) is not None for port in IN_PORTS)
    expected = tuple(i < ARITY[mode] for i in range(len(IN_PORTS)))
    if bound != expected:
        raise FeatureError(
            f"{mode} takes {ARITY[mode]} input(s); bound in_0..in_2: {bound}"
        )

    ins = [tensors_by_uid[inputs[port]] for port in IN_PORTS[: ARITY[mode]]]
    out = tensors_by_uid[outputs[OUT_PORT]]

    dtype = dtype_label(ins[0].get("data_type"))
    if dtype not in DTYPES:
        raise FeatureError(f"in_0 dtype {ins[0].get('data_type')!r} is not catalogued")
    port_dtypes = {dtype_label(t.get("data_type")) for t in ins[1:] + [out]}
    stray = port_dtypes - set(_PORT_DTYPES)
    if stray:
        raise FeatureError(f"port dtype(s) {sorted(stray)} are not catalogued")

    features = {f"mode:{mode}", f"dtype:{dtype}"}
    if len(({dtype} | port_dtypes) & set(DTYPES)) > 1:
        features.add("precision:mixed")
    if any(t["dims"] != out["dims"] for t in ins):
        features.add("broadcast:in")

    for name, (modes, default) in PARAMETERS.items():
        value = inputs.get(name)
        if value is None:
            continue
        if mode not in modes:
            raise FeatureError(f"{name} is set, but {mode} does not read it")
        feature = PARAMETER_FEATURES.get(name)
        if feature is not None and value != default:
            features.add(feature)
    return features
