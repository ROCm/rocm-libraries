# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``MoeRoutingAttributes``.

Mixture-of-experts routing: ``topk_ids`` ``[S, top_k]`` (each token's chosen
experts) becomes ``first_token_offset`` ``[E + 1, 1, 1]`` (the explicit group
boundaries) and ``token_index`` / ``token_ks`` ``[1, S * top_k, 1]`` (each
routed row's source token and slot), ordered by expert, then token, then
slot; ids outside ``[0, E)`` are not routed. The outputs feed MoE grouped
matmul ``gather`` and ``scatter`` as they are. The features come from
``moe_routing_attributes.fbs``. The serializer
(``utilities/json/MoeRoutingAttributes.hpp``) writes ``topk_ids`` under
``inputs``, the three routing tensors under ``outputs`` and ``num_experts``
at the top of the node. No provider implements the op, so the dtypes are
those the CPU reference executor registers (``MoeRoutingSignatureKey``):
``first_token_offset`` int32 or int64, everything else int32. By group:

- ``offset_dtype`` (exclusive): the dtype of ``first_token_offset``.
- ``top_k`` (exclusive): ``one`` (each token to one expert) or ``multi``.

The API and the CPU reference require ``topk_ids`` of rank 2 and int32,
``num_experts`` of at least 2 (the API; one expert needs no routing),
``first_token_offset`` ``[num_experts + 1, 1, 1]``, ``token_index`` and
``token_ks`` ``[1, S * top_k, 1]`` and int32, and no ragged tensor. A graph
that breaks these raises. The ids (which experts are local) are data the
graph does not carry.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm

NODE_TYPE = "MoeRoutingAttributes"
SCHEMA_FILE = "moe_routing_attributes.fbs"
SCHEMA_TABLE = "MoeRoutingAttributes"

TOPK_IDS = "topk_ids_tensor_uid"
OFFSET = "first_token_offset_tensor_uid"
INDEX, KS = "token_index_tensor_uid", "token_ks_tensor_uid"
NUM_EXPERTS = "num_experts"
REQUIRED = (TOPK_IDS, OFFSET, INDEX, KS)
RAGGED = "ragged_offset_tensor_uid"
OFFSET_DTYPES = ("int32", "int64")
ROUTING_DTYPE = "int32"

FEATURES = (
    *(Feature(f"offset_dtype:{dtype}", (OFFSET,)) for dtype in OFFSET_DTYPES),
    Feature("top_k:one", (TOPK_IDS, INDEX, KS), "each token to one expert"),
    Feature("top_k:multi", (TOPK_IDS, INDEX, KS), "each token to several experts"),
)

NOT_FEATURES: dict[str, str] = {
    NUM_EXPERTS: "fixes only the first_token_offset length, already in its dims",
}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")
    topk_ids, offset, index, ks = (tensors_by_uid[by_port[p]] for p in REQUIRED)

    for port, tensor in zip(REQUIRED, (topk_ids, offset, index, ks)):
        # Written only when set (CHECK_NO_RAGGED_TENSORS); uid 0 is one too.
        if tensor.get(RAGGED) is not None:
            raise FeatureError(f"{port} is ragged")

    experts = node.get(NUM_EXPERTS)
    # type() rather than isinstance: a bool is an int.
    if type(experts) is not int or experts < 2:
        raise FeatureError(f"{NUM_EXPERTS} {experts!r} is not an int of at least 2")

    ids_dims = topk_ids["dims"]
    if len(ids_dims) != 2 or any(d < 1 for d in ids_dims):
        raise FeatureError(f"{TOPK_IDS} dims {ids_dims} are not [S, top_k]")
    if topk_ids.get("data_type") != ROUTING_DTYPE:
        raise FeatureError(
            f"{TOPK_IDS} dtype {topk_ids.get('data_type')!r} is not int32"
        )
    tokens, top_k = ids_dims

    if offset["dims"] != [experts + 1, 1, 1]:
        raise FeatureError(
            f"{OFFSET} dims {offset['dims']} are not {[experts + 1, 1, 1]}"
        )
    offset_dtype = offset.get("data_type")
    if offset_dtype not in OFFSET_DTYPES:
        raise FeatureError(f"{OFFSET} dtype {offset_dtype!r} is not int32 or int64")

    rows = [1, tokens * top_k, 1]
    for port, tensor in ((INDEX, index), (KS, ks)):
        if tensor["dims"] != rows:
            raise FeatureError(f"{port} dims {tensor['dims']} are not {rows}")
        if tensor.get("data_type") != ROUTING_DTYPE:
            raise FeatureError(f"{port} dtype {tensor.get('data_type')!r} is not int32")

    return {
        f"offset_dtype:{offset_dtype}",
        "top_k:one" if top_k == 1 else "top_k:multi",
    }
