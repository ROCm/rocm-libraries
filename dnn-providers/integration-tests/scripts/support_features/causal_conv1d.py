# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``CausalConv1dAttributes``.

Depthwise causal 1-D convolution with a carried state: ``x [T, C]`` holds N
token-packed sequences with offsets ``cu_seqlens [N + 1]``; per sequence and
channel, ``y = act(bias + sum_j weight[c, j] * xs[t + j])`` over the W - 1
state columns followed by the tokens, and the final state is the last W - 1
entries. The features come from ``causal_conv1d_attributes.fbs``. The
serializer (``utilities/json/CausalConv1dAttributes.hpp``) writes the ports
under ``inputs`` and ``outputs`` and ``activation`` under ``attributes``. No
provider implements the op, so the dtypes are those the CPU reference executor
registers (``CausalConv1dSignatureKey``): one float dtype for x, weight, y,
bias and the states; int32 or int64 offsets and state indices. By group:

- ``dtype`` (exclusive): the dtype of ``x`` (and of weight, y, bias, states).
- ``cu_seqlens_dtype`` (exclusive): ``int32`` or ``int64``.
- ``bias`` (exclusive): ``absent`` or ``present``.
- ``state`` (exclusive): ``zero`` (no initial state), ``initial`` (one state
  per sequence) or ``pool`` (``state_indices`` address a ``[P, C, W - 1]``
  pool).
- ``final_state`` (exclusive): ``absent`` or ``present``.
- ``activation`` (exclusive): ``identity`` or ``silu``.

The API and the CPU reference require ``x``/``y`` ``[T, C]``, ``weight``
``[C, W]`` with W >= 2, ``cu_seqlens [N + 1]`` with N >= 1, ``bias [C]``,
states ``[N or P, C, W - 1]``, ``state_indices [N]`` only with an initial
state, and no ragged tensor. A graph that breaks these raises. The offsets and
state indices are data the graph does not carry.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm

NODE_TYPE = "CausalConv1dAttributes"
SCHEMA_FILE = "causal_conv1d_attributes.fbs"
SCHEMA_TABLE = "CausalConv1dAttributes"

X, WEIGHT, CU_SEQLENS, Y = (
    "x_tensor_uid",
    "weight_tensor_uid",
    "cu_seqlens_tensor_uid",
    "y_tensor_uid",
)
BIAS, INITIAL_STATE, STATE_INDICES, FINAL_STATE = (
    "bias_tensor_uid",
    "initial_state_tensor_uid",
    "state_indices_tensor_uid",
    "final_state_tensor_uid",
)
ACTIVATION = "activation"
REQUIRED = (X, WEIGHT, CU_SEQLENS, Y)
OPTIONAL = (BIAS, INITIAL_STATE, STATE_INDICES, FINAL_STATE)
RAGGED = "ragged_offset_tensor_uid"
INDEX_DTYPES = ("int32", "int64")
ACTIVATIONS = ("identity", "silu")

FEATURES = (
    *(
        Feature(f"dtype:{dtype}", (X, WEIGHT, Y), "the dtype of every data tensor")
        for dtype in layernorm.DTYPES
    ),
    *(Feature(f"cu_seqlens_dtype:{dtype}", (CU_SEQLENS,)) for dtype in INDEX_DTYPES),
    Feature("bias:absent", (BIAS,)),
    Feature("bias:present", (BIAS,)),
    Feature("state:zero", (INITIAL_STATE,), "no initial state: zeros"),
    Feature("state:initial", (INITIAL_STATE,), "one state per sequence"),
    Feature(
        "state:pool", (INITIAL_STATE, STATE_INDICES), "state_indices address a pool"
    ),
    Feature("final_state:absent", (FINAL_STATE,)),
    Feature("final_state:present", (FINAL_STATE,)),
    *(Feature(f"activation:{name}", (ACTIVATION,)) for name in ACTIVATIONS),
)

NOT_FEATURES: dict[str, str] = {}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    ports = REQUIRED + tuple(p for p in OPTIONAL if has(p))
    tensors = {port: tensors_by_uid[by_port[port]] for port in ports}
    for port, tensor in tensors.items():
        dims = tensor["dims"]
        if not dims or any(d < 0 for d in dims):
            raise FeatureError(f"{port} dims {dims}: not rank >= 1 with dims >= 0")
        # Written only when set (CHECK_NO_RAGGED_TENSORS); uid 0 is one too.
        if tensor.get(RAGGED) is not None:
            raise FeatureError(f"{port} is ragged")

    x_dims, w_dims = tensors[X]["dims"], tensors[WEIGHT]["dims"]
    cu_dims = tensors[CU_SEQLENS]["dims"]
    if len(x_dims) != 2:
        raise FeatureError(f"{X} dims {x_dims} are not [T, C]")
    if tensors[Y]["dims"] != x_dims:
        raise FeatureError(f"{Y} dims {tensors[Y]['dims']} are not those of {X}")
    if len(w_dims) != 2 or w_dims[0] != x_dims[1] or w_dims[1] < 2:
        raise FeatureError(f"{WEIGHT} dims {w_dims} are not [{x_dims[1]}, W >= 2]")
    if len(cu_dims) != 1 or cu_dims[0] < 2:
        raise FeatureError(f"{CU_SEQLENS} dims {cu_dims} are not [N + 1] with N >= 1")
    sequences = cu_dims[0] - 1
    if has(BIAS) and tensors[BIAS]["dims"] != [x_dims[1]]:
        raise FeatureError(f"{BIAS} dims {tensors[BIAS]['dims']} are not [{x_dims[1]}]")

    pooled = has(STATE_INDICES)
    if pooled:
        if not has(INITIAL_STATE):
            raise FeatureError(f"{STATE_INDICES} requires {INITIAL_STATE}")
        if tensors[STATE_INDICES]["dims"] != [sequences]:
            raise FeatureError(
                f"{STATE_INDICES} dims {tensors[STATE_INDICES]['dims']} are not [{sequences}]"
            )
    for port in (INITIAL_STATE, FINAL_STATE):
        if not has(port):
            continue
        dims = tensors[port]["dims"]
        rows_ok = len(dims) == 3 and (dims[0] >= 1 if pooled else dims[0] == sequences)
        if not rows_ok or dims[1] != x_dims[1] or dims[2] != w_dims[1] - 1:
            raise FeatureError(
                f"{port} dims {dims} are not [N or P, {x_dims[1]}, W - 1]"
            )
    if pooled and has(FINAL_STATE):
        if tensors[FINAL_STATE]["dims"][0] != tensors[INITIAL_STATE]["dims"][0]:
            raise FeatureError(f"{FINAL_STATE} pool rows differ from {INITIAL_STATE}'s")

    dtype = layernorm.catalogued_dtype(tensors[X], X)
    for port in (WEIGHT, Y, BIAS, INITIAL_STATE, FINAL_STATE):
        if port in tensors and tensors[port].get("data_type") != tensors[X].get(
            "data_type"
        ):
            raise FeatureError(f"{port} dtype is not that of {X}")
    for port in (CU_SEQLENS, STATE_INDICES):
        if port in tensors and tensors[port].get("data_type") not in INDEX_DTYPES:
            raise FeatureError(
                f"{port} dtype {tensors[port].get('data_type')!r} is not int32 or int64"
            )

    activation = (node.get("attributes") or {}).get(ACTIVATION, "identity")
    if activation not in ACTIVATIONS:
        raise FeatureError(f"{ACTIVATION} {activation!r} is not identity or silu")

    if pooled:
        state = "state:pool"
    elif has(INITIAL_STATE):
        state = "state:initial"
    else:
        state = "state:zero"
    return {
        f"dtype:{dtype}",
        f"cu_seqlens_dtype:{tensors[CU_SEQLENS].get('data_type')}",
        "bias:present" if has(BIAS) else "bias:absent",
        state,
        "final_state:present" if has(FINAL_STATE) else "final_state:absent",
        f"activation:{activation}",
    }
