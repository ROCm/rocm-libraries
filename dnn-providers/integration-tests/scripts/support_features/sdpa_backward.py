# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``SdpaBackwardAttributes`` (scaled dot-product attention,
backward).

The features come from ``sdpa_backward_attributes.fbs``. The exclusive groups
-- ``dtype``, ``head_dim``, ``heads``, ``layout``, ``mask``, ``seq``, ``scale``
-- are read as the forward catalog reads them (``sdpa.core_features``), with
two differences:

- The schema has no fp8 scale tensors, so ``dtype`` is bf16 and fp16 only.
- There is no ``seq:decode``: a backward pass trains, and a single query
  position is an ordinary ``q_eq_kv`` / ``q_ne_kv`` shape.
- There is no ``layout:ragged``: no backward engine reads ragged offsets, so a
  ragged q raises ``FeatureError`` rather than counting as its stride order.
"""

from __future__ import annotations

from support_features import Feature, sdpa

NODE_TYPE = "SdpaBackwardAttributes"
SCHEMA_FILE = "sdpa_backward_attributes.fbs"
SCHEMA_TABLE = "SdpaBackwardAttributes"

DTYPES = {"bfloat16": "bf16", "half": "fp16"}

FEATURES = (
    *(Feature(f"dtype:{dtype}", ("q_tensor_uid",)) for dtype in DTYPES.values()),
    *(Feature(f"head_dim:{d}", sdpa.QV_FIELDS) for d in sdpa.HEAD_DIMS.values()),
    Feature("heads:mha", sdpa.QK_FIELDS, "as many k/v heads as q heads"),
    Feature("heads:gqa", sdpa.QK_FIELDS, "fewer k/v heads than q heads, more than one"),
    Feature("heads:mqa", sdpa.QK_FIELDS, "one k/v head"),
    *(Feature(f"layout:{name}", ("q_tensor_uid",)) for name in sdpa.LAYOUTS.values()),
    Feature("mask:none", sdpa.MASK_FIELDS),
    Feature("mask:causal_top_left", sdpa.MASK_FIELDS),
    Feature("mask:causal_bottom_right", sdpa.MASK_FIELDS),
    Feature("mask:sliding_window", sdpa.MASK_FIELDS),
    Feature("seq:q_eq_kv", sdpa.QK_FIELDS),
    Feature("seq:q_ne_kv", sdpa.QK_FIELDS),
    Feature(
        "varlen:seq_lens",
        ("seq_len_q_tensor_uid", "seq_len_kv_tensor_uid", "padding_mask"),
        "per-batch sequence lengths, or padding",
    ),
    Feature("scale:fixed", ("attn_scale_value",)),
    Feature("scale:runtime", ("scale_tensor_uid",)),
    Feature("scale:default", ("attn_scale_value", "scale_tensor_uid")),
    Feature("bias:additive", ("attn_mask_tensor_uid",)),
    Feature("bias:alibi", ("alibi_mask",)),
    Feature(
        "dropout:on",
        (
            "dropout_probability",
            "seed_tensor_uid",
            "offset_tensor_uid",
            "dropout_mask_tensor_uid",
            "dropout_scale_tensor_uid",
            "dropout_scale_inv_tensor_uid",
        ),
    ),
    Feature("out:dbias", ("dbias_tensor_uid",), "gradient of the additive bias"),
)

NOT_FEATURES = {
    "o_tensor_uid": "required; its shape follows q's, so it varies nothing q does not",
    "do_tensor_uid": "required; its shape follows o's, so it varies nothing q does not",
    "stats_tensor_uid": "required; the forward's softmax stats, shaped by q and k",
    "dq_tensor_uid": "required; its shape follows q's, so it varies nothing q does not",
    "dk_tensor_uid": "required; its shape follows k's, so it varies nothing k does not",
    "dv_tensor_uid": "required; its shape follows v's, so it varies nothing v does not",
}


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    # The backward serializer writes scalars under "parameters" (forward writes
    # "attributes"; see utilities/json/Sdpa*Attributes.hpp).
    attrs = node.get("parameters") or {}
    has = sdpa.ports(node)
    features = sdpa.core_features(
        node, attrs, tensors_by_uid, dtypes=DTYPES, decode=False, ragged=False
    )

    if (
        has("seq_len_q_tensor_uid")
        or has("seq_len_kv_tensor_uid")
        or attrs.get("padding_mask")
    ):
        features.add("varlen:seq_lens")
    if has("attn_mask_tensor_uid"):
        features.add("bias:additive")
    if attrs.get("alibi_mask"):
        features.add("bias:alibi")
    if (
        (attrs.get("dropout_probability") or 0) > 0
        or has("seed_tensor_uid")
        or has("dropout_mask_tensor_uid")
    ):
        features.add("dropout:on")
    if has("dbias_tensor_uid"):
        features.add("out:dbias")
    return features
