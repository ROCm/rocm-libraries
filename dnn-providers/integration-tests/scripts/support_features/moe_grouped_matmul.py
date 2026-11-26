# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``MoeGroupedMatmulAttributes``.

The mixture-of-experts grouped matmul: ``first_token_offset`` cuts the rows
of ``token`` ``[1, tokens, K]`` into ``G`` groups, and group ``g`` is
multiplied by expert ``g % E`` of ``weight`` ``[E, K, N]`` into ``output``
``[1, rows, N]``; the offset table is ``[batch][expert]``, so ``G`` is the
batch times ``E``. The routing mode places the rows: ``none`` in order,
``gather`` reading the token ``token_index`` names, ``scatter`` writing to
row ``token_index * top_k + token_ks``. The features come from
``moe_grouped_matmul_attributes.fbs``. The serializer
(``utilities/json/MoeGroupedMatmulAttributes.hpp``) writes the ports under
``inputs`` and ``outputs``, leaving an unset index or ks out, and ``mode``
and ``top_k`` at the top of the node, both always: its reader requires
them. No provider implements the op, so the dtypes are those the CPU
reference executor registers (``MoeGroupedMatmulSignatureKey``): ``token``
and ``weight`` of one float dtype, ``output`` in that dtype or, with the
other fp32, in another. By group:

- ``mode`` (exclusive): the routing, by its serialized name.
- ``dtype`` (exclusive): the dtype of ``token``, which ``weight`` shares.
- ``precision:mixed_output``: ``output`` is not the dtype of ``token``. A
  virtual tensor is not counted, as for matmul: the caller never sees it.
- ``batch`` (exclusive): ``single`` (one offset per expert) or ``multi``
  (more: ``G`` a multiple of ``E`` above 1).
- ``top_k`` (exclusive, with ``scatter``): ``one`` (each token to one
  expert) or ``multi``.
- ``weight_layout`` (exclusive): ``row_major`` (unit stride on ``N``) or
  ``col_major`` (on ``K``, cuDNN's documented MoE layout), the two paths of
  the CPU reference, read as matmul reads them.

The API, the backend's routing contract (``MoeGroupedMatmulValidation``)
and the CPU reference require ``token``, ``weight``, ``first_token_offset``
and ``output`` of rank 3; ``token`` and ``output`` ``[1, rows, ...]``, the
offset ``[G, 1, 1]`` and int32, with ``G`` a multiple of ``E``; ``token``'s
``K`` that of ``weight``; under ``none`` no index, no ks and ``top_k`` 0;
under ``gather`` an index, no ks and ``top_k`` 0; under ``scatter`` both and
``top_k`` from 1 to ``E``; an index and a ks int32 and ``[1, rows, 1]``,
under ``scatter`` of ``token``'s rows (the API checks only that they
agree); and ``output`` ``[1, rows, N]``, its rows the index's under
``gather`` and ``token``'s otherwise. A graph that breaks these raises, as
does a ``weight`` with no unit stride on ``K`` or ``N``, which no layout
describes.

Every schema field drives a feature.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, layernorm, matmul

NODE_TYPE = "MoeGroupedMatmulAttributes"
SCHEMA_FILE = "moe_grouped_matmul_attributes.fbs"
SCHEMA_TABLE = "MoeGroupedMatmulAttributes"

TOKEN, WEIGHT = "token_tensor_uid", "weight_tensor_uid"
OFFSET = "first_token_offset_tensor_uid"
INDEX, KS = "token_index_tensor_uid", "token_ks_tensor_uid"
OUTPUT = "output_tensor_uid"
MODE, TOP_K = "mode", "top_k"
REQUIRED = (TOKEN, WEIGHT, OFFSET, OUTPUT)

# MoeGroupedMatmulMode as utilities/json/MoeGroupedMatmulAttributes.hpp
# writes it.
MODES = ("none", "gather", "scatter")
ROUTING_DTYPE = "int32"

FEATURES = (
    Feature("mode:none", (MODE, INDEX, KS), "rows in order"),
    Feature("mode:gather", (MODE, INDEX, KS), "rows read through the index"),
    Feature("mode:scatter", (MODE, INDEX, KS), "rows written through index, ks"),
    *(Feature(f"dtype:{dtype}", (TOKEN, WEIGHT)) for dtype in layernorm.DTYPES),
    Feature("precision:mixed_output", (TOKEN, OUTPUT), "output is not token's dtype"),
    Feature("batch:single", (OFFSET, WEIGHT), "one offset per expert"),
    Feature("batch:multi", (OFFSET, WEIGHT), "offsets for several batches"),
    Feature("top_k:one", (MODE, TOP_K), "scatter: a token to one expert"),
    Feature("top_k:multi", (MODE, TOP_K), "scatter: a token to several"),
    Feature("weight_layout:row_major", (WEIGHT,), "unit stride on N"),
    Feature("weight_layout:col_major", (WEIGHT,), "unit stride on K"),
)

NOT_FEATURES: dict[str, str] = {}


def checked_routing(
    mode: str, top_k: int, experts: int, has_index: bool, has_ks: bool
) -> None:
    """Raise where the backend's routing contract rejects the binding."""
    if mode == "none" and (has_index or has_ks):
        raise FeatureError(f"{MODE} none binds {INDEX} or {KS}")
    if mode == "gather" and (not has_index or has_ks):
        raise FeatureError(f"{MODE} gather does not bind {INDEX} alone")
    if mode == "scatter" and not (has_index and has_ks):
        raise FeatureError(f"{MODE} scatter does not bind both {INDEX} and {KS}")
    if mode == "scatter":
        if not 1 <= top_k <= experts:
            raise FeatureError(f"{TOP_K} {top_k} is not from 1 to {experts} experts")
    elif top_k != 0:
        raise FeatureError(f"{TOP_K} {top_k} is not 0 under {MODE} {mode}")


def routed_rows(tensor: dict, port: str) -> int:
    """The rows of an index or ks ``[1, rows, 1]`` int32, or raise."""
    if tensor.get("data_type") != ROUTING_DTYPE:
        raise FeatureError(f"{port} dtype {tensor.get('data_type')!r} is not int32")
    dims = tensor["dims"]
    if len(dims) != 3 or dims[0] != 1 or dims[2] != 1 or dims[1] < 1:
        raise FeatureError(f"{port} dims {dims} are not [1, rows, 1]")
    return dims[1]


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    token, weight, offset, output = (tensors_by_uid[by_port[p]] for p in REQUIRED)

    mode = node.get(MODE)
    if mode not in MODES:
        raise FeatureError(f"{MODE} {mode!r} is not catalogued")
    top_k = node.get(TOP_K)
    # type() rather than isinstance: a bool is an int.
    if type(top_k) is not int:
        raise FeatureError(f"{TOP_K} {top_k!r} is not an int")

    for port, tensor in zip(REQUIRED, (token, weight, offset, output)):
        dims = tensor["dims"]
        if len(dims) != 3 or any(d < 1 for d in dims):
            raise FeatureError(f"{port} dims {dims}: not rank 3 with dims >= 1")
    experts, k, n = weight["dims"]
    groups = offset["dims"][0]
    if token["dims"][0] != 1 or token["dims"][2] != k:
        raise FeatureError(f"{TOKEN} dims {token['dims']} are not [1, tokens, {k}]")
    if offset["dims"][1:] != [1, 1] or groups % experts:
        raise FeatureError(
            f"{OFFSET} dims {offset['dims']} are not [G, 1, 1] with G a multiple of "
            f"{experts} experts"
        )
    if offset.get("data_type") != ROUTING_DTYPE:
        raise FeatureError(f"{OFFSET} dtype {offset.get('data_type')!r} is not int32")

    has_index, has_ks = has(INDEX), has(KS)
    checked_routing(mode, top_k, experts, has_index, has_ks)
    rows = token["dims"][1]
    if has_index:
        index_rows = routed_rows(tensors_by_uid[by_port[INDEX]], INDEX)
        if mode == "gather":
            rows = index_rows
    if has_ks:
        # checked_routing: a ks is bound only under scatter, with an index.
        ks_rows = routed_rows(tensors_by_uid[by_port[KS]], KS)
        if index_rows != ks_rows or index_rows != rows:
            raise FeatureError(
                f"{INDEX} rows {index_rows} and {KS} rows {ks_rows} are not {TOKEN} "
                f"rows {rows}"
            )
    if output["dims"] != [1, rows, n]:
        raise FeatureError(f"{OUTPUT} dims {output['dims']} are not {[1, rows, n]}")

    dtype = layernorm.catalogued_dtype(token, TOKEN)
    if layernorm.catalogued_dtype(weight, WEIGHT) != dtype:
        raise FeatureError(
            f"{WEIGHT} dtype {weight.get('data_type')!r} is not that of {TOKEN}"
        )
    out_dtype = layernorm.catalogued_dtype(output, OUTPUT)
    if out_dtype != dtype and "fp32" not in (dtype, out_dtype):
        raise FeatureError(
            f"{OUTPUT} dtype {out_dtype} with {TOKEN} {dtype} has no plan"
        )

    features = {
        f"mode:{mode}",
        f"dtype:{dtype}",
        "batch:single" if groups == experts else "batch:multi",
        f"weight_layout:{matmul.matrix_layout(weight, WEIGHT)}",
    }
    if not token.get("virtual") and not output.get("virtual") and out_dtype != dtype:
        features.add("precision:mixed_output")
    if mode == "scatter":
        features.add("top_k:one" if top_k == 1 else "top_k:multi")
    return features
