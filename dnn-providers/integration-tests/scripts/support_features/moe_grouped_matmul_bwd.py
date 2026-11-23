# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``MoeGroupedMatmulBwdAttributes``.

The weight gradient of the mixture-of-experts grouped matmul:
``first_token_offset`` cuts the rows of ``token`` ``[1, rows, K]`` and
``doutput`` ``[1, rows, N]`` into one group per expert, and expert ``e`` of
``dweight`` ``[E, K, N]`` is its group of ``token`` transposed times its
group of ``doutput``. The features come from
``moe_grouped_matmul_bwd_attributes.fbs``. The serializer
(``utilities/json/MoeGroupedMatmulBwdAttributes.hpp``) writes the ports
under ``inputs`` and ``outputs``; its reader requires all four. No provider
implements the op, so the dtypes are those the CPU reference executor
registers (``MoeGroupedMatmulBwdSignatureKey``): ``doutput`` and ``token``
of one float dtype, ``dweight`` in that dtype or, with the other fp32, in
another. By group:

- ``dtype`` (exclusive): the dtype of ``token``, which ``doutput`` shares.
- ``precision:mixed_output``: ``dweight`` is not the dtype of ``token``. A
  virtual tensor is not counted, as for matmul: the caller never sees it.
- ``dweight_layout`` (exclusive, when the strides tell): ``col_major``
  (unit stride on ``K``: the strides the frontend infers, cuDNN's
  documented MoE layout) or ``row_major`` (on ``N``: a caller's), read as
  matmul reads them. The CPU reference reads any strides, so a ``dweight``
  with neither is no layout, not an error.

The API and the CPU reference require all four tensors of rank 3, none
ragged; ``doutput`` and ``token`` ``[1, rows, ...]`` of the same rows; the
offset ``[E, 1, 1]`` and int32; and ``dweight`` ``[E, K, N]``, with ``K``
that of ``token`` and ``N`` that of ``doutput``. A graph that breaks these
raises. The offsets' values (where the groups start) are data the graph
does not carry.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm
from support_features.moe_grouped_matmul import ROUTING_DTYPE

NODE_TYPE = "MoeGroupedMatmulBwdAttributes"
SCHEMA_FILE = "moe_grouped_matmul_bwd_attributes.fbs"
SCHEMA_TABLE = "MoeGroupedMatmulBwdAttributes"

DOUTPUT, TOKEN = "doutput_tensor_uid", "token_tensor_uid"
OFFSET = "first_token_offset_tensor_uid"
DWEIGHT = "dweight_tensor_uid"
REQUIRED = (DOUTPUT, TOKEN, OFFSET, DWEIGHT)
RAGGED = "ragged_offset_tensor_uid"

FEATURES = (
    *(Feature(f"dtype:{dtype}", (DOUTPUT, TOKEN)) for dtype in layernorm.DTYPES),
    Feature("precision:mixed_output", (TOKEN, DWEIGHT), "dweight is not token's dtype"),
    Feature("dweight_layout:col_major", (DWEIGHT,), "unit stride on K"),
    Feature("dweight_layout:row_major", (DWEIGHT,), "unit stride on N"),
)

NOT_FEATURES: dict[str, str] = {
    OFFSET: "[E, 1, 1] int32, its E that of dweight: the shape is fixed, and "
    "where each expert's rows start is data the graph does not carry",
}


def dweight_layout(strides: list[int]) -> set[str]:
    """The ``dweight_layout`` feature, as matmul.matrix_layout reads the
    last two strides; none when neither is 1."""
    if strides[-1] == 1:
        return {"dweight_layout:row_major"}
    if strides[-2] == 1:
        return {"dweight_layout:col_major"}
    return set()


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    doutput, token, offset, dweight = (tensors_by_uid[by_port[p]] for p in REQUIRED)

    for port, tensor in zip(REQUIRED, (doutput, token, offset, dweight)):
        dims = tensor["dims"]
        if len(dims) != 3 or any(d < 1 for d in dims):
            raise FeatureError(f"{port} dims {dims}: not rank 3 with dims >= 1")
        # Written only when set (CHECK_NO_RAGGED_TENSORS); uid 0 is one too.
        if tensor.get(RAGGED) is not None:
            raise FeatureError(f"{port} is ragged")
    _, rows, k = token["dims"]
    n = doutput["dims"][2]
    experts = offset["dims"][0]
    if token["dims"][0] != 1:
        raise FeatureError(f"{TOKEN} dims {token['dims']} are not [1, rows, K]")
    if doutput["dims"][:2] != [1, rows]:
        raise FeatureError(
            f"{DOUTPUT} dims {doutput['dims']} are not [1, {rows}, N]: {TOKEN}'s rows"
        )
    if offset["dims"][1:] != [1, 1]:
        raise FeatureError(f"{OFFSET} dims {offset['dims']} are not [E, 1, 1]")
    if offset.get("data_type") != ROUTING_DTYPE:
        raise FeatureError(f"{OFFSET} dtype {offset.get('data_type')!r} is not int32")
    if dweight["dims"] != [experts, k, n]:
        raise FeatureError(
            f"{DWEIGHT} dims {dweight['dims']} are not {[experts, k, n]}"
        )

    dtype = layernorm.catalogued_dtype(token, TOKEN)
    if layernorm.catalogued_dtype(doutput, DOUTPUT) != dtype:
        raise FeatureError(
            f"{DOUTPUT} dtype {doutput.get('data_type')!r} is not that of {TOKEN}"
        )
    out_dtype = layernorm.catalogued_dtype(dweight, DWEIGHT)
    if out_dtype != dtype and "fp32" not in (dtype, out_dtype):
        raise FeatureError(
            f"{DWEIGHT} dtype {out_dtype} with {TOKEN} {dtype} has no plan"
        )

    features = {f"dtype:{dtype}", *dweight_layout(dweight["strides"])}
    if not token.get("virtual") and not dweight.get("virtual") and out_dtype != dtype:
        features.add("precision:mixed_output")
    return features
