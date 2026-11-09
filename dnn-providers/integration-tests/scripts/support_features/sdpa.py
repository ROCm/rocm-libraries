# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``SdpaAttributes`` (scaled dot-product attention, forward).

The features come from ``sdpa_attributes.fbs``. A few are read from the q / k
/ v tensors rather than from attributes -- dtype, head dim, head grouping,
layout and sequence shape -- since the schema carries those only as tensor
uids. ``head_dim`` reads q and v together: ``192x128`` is a q / k head dim of
192 with a v head dim of 128, its own kernel rather than a 192. ``layout`` is
q's stride order, named by its dims from the largest stride to the smallest
(``bshd``: b, then s, then h, then d), or ``ragged`` when q is packed by
per-batch offsets (the tensor's ``ragged_offset_tensor_uid``) whatever its
strides.

Exclusive groups (exactly one feature per graph): ``dtype``, ``head_dim``,
``heads``, ``layout``, ``mask``, ``seq``, ``scale``. ``dtype`` is q's. The rest
are optional inputs and outputs, each present or not, and two precision
features: ``precision:mixed_output``, o's dtype is not q's, as fp8 q / k / v
with a bf16 o; and ``precision:mixed_input``, k's or v's dtype is not q's.

The value sets of ``dtype``, ``head_dim`` and ``layout`` are not in the schema,
which types them only as tensors. Each value is listed; a graph outside them
raises ``FeatureError`` rather than widening the denominator.
FNUZ fp8 is not folded into OCP fp8: gfx942 runs ``fp8e4m3_fnuz``, gfx950
``fp8e4m3``, and the two are not bit-compatible. The same holds for e5m2.
"""

from __future__ import annotations

from support_features import Feature, FeatureError

NODE_TYPE = "SdpaAttributes"
SCHEMA_FILE = "sdpa_attributes.fbs"
SCHEMA_TABLE = "SdpaAttributes"

_FP8_SCALES = (
    "descale_q_tensor_uid",
    "descale_k_tensor_uid",
    "descale_v_tensor_uid",
    "descale_s_tensor_uid",
    "scale_s_tensor_uid",
    "scale_o_tensor_uid",
)
MASK_FIELDS = (
    "causal_mask",
    "causal_mask_bottom_right",
    "left_bound",
    "right_bound",
    "diagonal_alignment",
)
QK_FIELDS = ("q_tensor_uid", "k_tensor_uid")
QV_FIELDS = ("q_tensor_uid", "v_tensor_uid")

# q's data_type as serialized -> dtype value.
DTYPES = {
    "bfloat16": "bf16",
    "half": "fp16",
    "float": "fp32",
    "fp8_e4m3": "fp8e4m3",
    "fp8_e4m3_fnuz": "fp8e4m3_fnuz",
    "fp8_e5m2": "fp8e5m2",
    "fp8_e5m2_fnuz": "fp8e5m2_fnuz",
}
# q / k head dim -> the v head dims catalogued with it. v may differ from q / k:
# the ASM kernels run 192 x 128 (fmha_v3_*/*.csv). The odd sizes sit one either
# side of 64, 96 and 128.
_V_HEAD_DIMS = {
    1: (1, 64, 65, 96, 128, 192, 256, 512, 576),
    63: (96, 128),
    64: (1, 64, 95, 96, 128, 192, 256, 512, 576),
    65: (128, 256),
    95: (128,),
    96: (1, 64, 96, 128, 192, 256, 512, 576),
    97: (128,),
    127: (128,),
    128: (1, 63, 64, 65, 95, 96, 97, 127, 128, 129, 192, 256, 512, 576),
    129: (1, 64, 128),
    192: (1, 64, 96, 97, 128, 192, 256, 512, 576),
    256: (1, 64, 96, 127, 128, 192, 256, 512, 576),
    512: (1, 64, 96, 128, 192, 256, 512, 576),
    576: (1, 64, 96, 128, 192, 256, 512, 576),
}
# (q / k head dim, v head dim) -> head_dim value: ``128`` when they are equal,
# ``192x128`` when they are not.
HEAD_DIMS = {
    (qk, v): str(qk) if qk == v else f"{qk}x{v}"
    for qk, vs in _V_HEAD_DIMS.items()
    for v in vs
}
# Order of q's dims by decreasing stride -> layout name. Dims are (b, h, s, d).
LAYOUTS = {
    (0, 1, 2, 3): "bhsd",
    (0, 1, 3, 2): "bhds",
    (0, 2, 1, 3): "bshd",
    (0, 2, 3, 1): "bsdh",
    (0, 3, 1, 2): "bdhs",
    (2, 0, 1, 3): "sbhd",
}

FEATURES = (
    Feature("dtype:bf16", ("q_tensor_uid",)),
    Feature("dtype:fp16", ("q_tensor_uid",)),
    Feature("dtype:fp32", ("q_tensor_uid",)),
    Feature("dtype:fp8e4m3", ("q_tensor_uid",) + _FP8_SCALES),
    Feature("dtype:fp8e4m3_fnuz", ("q_tensor_uid",) + _FP8_SCALES),
    Feature("dtype:fp8e5m2", ("q_tensor_uid",) + _FP8_SCALES),
    Feature("dtype:fp8e5m2_fnuz", ("q_tensor_uid",) + _FP8_SCALES),
    *(Feature(f"head_dim:{d}", QV_FIELDS) for d in HEAD_DIMS.values()),
    Feature("heads:mha", QK_FIELDS, "as many k/v heads as q heads"),
    Feature("heads:gqa", QK_FIELDS, "fewer k/v heads than q heads, more than one"),
    Feature("heads:mqa", QK_FIELDS, "one k/v head"),
    *(Feature(f"layout:{name}", ("q_tensor_uid",)) for name in LAYOUTS.values()),
    Feature("layout:ragged", ("q_tensor_uid",), "q packed by per-batch offsets"),
    Feature("mask:none", MASK_FIELDS),
    Feature("mask:causal_top_left", MASK_FIELDS),
    Feature("mask:causal_bottom_right", MASK_FIELDS),
    Feature("mask:sliding_window", MASK_FIELDS),
    Feature("seq:q_eq_kv", QK_FIELDS),
    Feature("seq:q_ne_kv", QK_FIELDS),
    Feature("seq:decode", ("q_tensor_uid",), "one query position"),
    Feature(
        "varlen:seq_lens",
        ("seq_len_q_tensor_uid", "seq_len_kv_tensor_uid", "padding_mask"),
        "per-batch sequence lengths, or padding",
    ),
    Feature("scale:fixed", ("attn_scale_value",)),
    Feature("scale:runtime", ("scale_tensor_uid",)),
    Feature("scale:default", ("attn_scale_value", "scale_tensor_uid")),
    Feature("out:stats", ("generate_stats", "stats_tensor_uid")),
    Feature("out:max_sum_exp", ("max_tensor_uid", "sum_exp_tensor_uid")),
    Feature("out:rng_dump", ("rng_dump_tensor_uid",)),
    Feature("fp8:amax", ("amax_s_tensor_uid", "amax_o_tensor_uid")),
    Feature(
        "precision:mixed_output", ("q_tensor_uid", "o_tensor_uid"), "o is not q's dtype"
    ),
    Feature(
        "precision:mixed_input",
        ("q_tensor_uid", "k_tensor_uid", "v_tensor_uid"),
        "k or v is not q's dtype",
    ),
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
        ),
    ),
    Feature(
        "kv:paged",
        ("page_table_k_tensor_uid", "page_table_v_tensor_uid", "max_seq_len_kv"),
    ),
    Feature("input:block_mask", ("block_mask_tensor_uid",)),
    Feature("input:sink_token", ("sink_token_tensor_uid",)),
)

NOT_FEATURES = {
    "mma_core_mode": "a numerics hint to the engine, not a variant of the op",
    "implementation": "an engine-selection hint, not a variant of the op",
}


def mask_kind(attrs: dict) -> str:
    """The mask kind, resolved as ``getMaskType`` (SdpaPlanUtils.hpp) does.

    Forward and backward engines both resolve it this way.
    """
    causal = attrs.get("causal_mask")
    bottom_right = attrs.get("causal_mask_bottom_right")
    if causal and bottom_right:
        raise FeatureError("causal_mask and causal_mask_bottom_right are both set")
    # The deprecated flags win over the bounds.
    if causal:
        return "mask:causal_top_left"
    if bottom_right:
        return "mask:causal_bottom_right"
    # An unset bound is unbounded.
    left = attrs.get("left_bound")
    right = attrs.get("right_bound")
    left = -1 if left is None else left
    right = -1 if right is None else right
    if left == -1 and right == -1:
        return "mask:none"
    if left == -1 and right == 0:
        if attrs.get("diagonal_alignment") == "BOTTOM_RIGHT":
            return "mask:causal_bottom_right"
        return "mask:causal_top_left"
    return "mask:sliding_window"


def ports(node: dict):
    """``has(port)``: whether ``node`` binds ``port`` as an input or output."""
    inputs = node.get("inputs") or {}
    outputs = node.get("outputs") or {}

    def has(port: str) -> bool:
        return inputs.get(port) is not None or outputs.get(port) is not None

    return has


def core_features(
    node: dict,
    attrs: dict,
    tensors_by_uid: dict,
    *,
    dtypes: dict[str, str],
    head_dims: dict[tuple[int, int], str],
    layouts: dict[tuple[int, ...], str],
    decode: bool,
    ragged: bool,
    mixed_input: bool,
) -> set[str]:
    """The exclusive groups shared by forward and backward.

    These are ``dtype``, ``head_dim``, ``heads``, ``layout``, ``mask``, ``seq``
    and ``scale``. ``decode`` says whether a single query position counts as
    ``seq:decode`` or as an ordinary ``q_eq_kv`` / ``q_ne_kv`` shape.
    ``ragged`` says whether a ragged q is ``layout:ragged`` or uncatalogued.
    ``mixed_input`` says whether a k or v dtype other than q's adds
    ``precision:mixed_input`` or raises. ``dtypes`` maps a data_type as
    serialized to its dtype value; ``head_dims`` and ``layouts`` are the
    catalogued values of their groups, keyed as ``HEAD_DIMS`` and ``LAYOUTS``.
    """
    inputs = node.get("inputs") or {}
    q, k, v = (tensors_by_uid[inputs[f"{port}_tensor_uid"]] for port in "qkv")
    ranks = [len(t["dims"]) for t in (q, k, v)]
    if ranks != [4, 4, 4]:
        raise FeatureError(f"q / k / v rank {' / '.join(map(str, ranks))}, not 4")
    _, heads_q, seq_q, head_dim_qk = q["dims"]
    _, heads_kv, seq_kv, _ = k["dims"]
    head_dim_v = v["dims"][3]

    # type() rather than in: an unhashable value would raise TypeError.
    dtype = q.get("data_type")
    if type(dtype) is not str or dtype not in dtypes:
        raise FeatureError(f"q dtype {dtype!r} is not catalogued")
    kv_dtypes = (k.get("data_type"), v.get("data_type"))
    is_mixed_input = kv_dtypes != (dtype, dtype)
    if is_mixed_input:
        if not mixed_input:
            raise FeatureError(
                f"k / v dtype {kv_dtypes[0]!r} / {kv_dtypes[1]!r} is not q's"
            )
        for kv_dtype in kv_dtypes:
            if type(kv_dtype) is not str or kv_dtype not in dtypes:
                raise FeatureError(f"k / v dtype {kv_dtype!r} is not catalogued")
    head_dim = head_dims.get((head_dim_qk, head_dim_v))
    if head_dim is None:
        raise FeatureError(f"head dim {head_dim_qk} x {head_dim_v} is not catalogued")
    if q.get("ragged_offset_tensor_uid") is not None:
        if not ragged:
            raise FeatureError("ragged q is not catalogued")
        layout = "ragged"
    else:
        # Stable sort: dims of equal stride (a size-1 dim) keep their bhsd order.
        order = tuple(sorted(range(4), key=lambda dim: -q["strides"][dim]))
        if order not in layouts:
            raise FeatureError(f"q stride order {order} is not a catalogued layout")
        layout = layouts[order]

    features = {
        f"dtype:{dtypes[dtype]}",
        f"head_dim:{head_dim}",
        f"layout:{layout}",
        mask_kind(attrs),
    }
    if is_mixed_input:
        features.add("precision:mixed_input")
    if heads_q == heads_kv:
        features.add("heads:mha")
    elif heads_kv == 1:
        features.add("heads:mqa")
    else:
        features.add("heads:gqa")
    if decode and seq_q == 1:
        features.add("seq:decode")
    elif seq_q == seq_kv:
        features.add("seq:q_eq_kv")
    else:
        features.add("seq:q_ne_kv")

    if ports(node)("scale_tensor_uid"):
        features.add("scale:runtime")
    elif attrs.get("attn_scale_value") is not None:
        features.add("scale:fixed")
    else:
        features.add("scale:default")
    return features


def mixed_output(node: dict, tensors_by_uid: dict) -> bool:
    """Whether o's dtype is not q's. A virtual q or o takes the dtype of the op
    on its other side, which that op's catalog counts."""
    o_uid = (node.get("outputs") or {}).get("o_tensor_uid")
    if o_uid is None:
        raise FeatureError("required port o_tensor_uid is not bound")
    q, o = tensors_by_uid[node["inputs"]["q_tensor_uid"]], tensors_by_uid[o_uid]
    o_dtype = o.get("data_type")
    if type(o_dtype) is not str or o_dtype not in DTYPES:
        raise FeatureError(f"o dtype {o_dtype!r} is not catalogued")
    if q.get("virtual") or o.get("virtual"):
        return False
    return o_dtype != q["data_type"]


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    # The forward serializer writes scalars under "attributes" (backward
    # writes "parameters"; see utilities/json/Sdpa*Attributes.hpp).
    attrs = node.get("attributes") or {}
    has = ports(node)
    features = core_features(
        node,
        attrs,
        tensors_by_uid,
        dtypes=DTYPES,
        head_dims=HEAD_DIMS,
        layouts=LAYOUTS,
        decode=True,
        ragged=True,
        mixed_input=True,
    )

    if (
        has("seq_len_q_tensor_uid")
        or has("seq_len_kv_tensor_uid")
        or attrs.get("padding_mask")
    ):
        features.add("varlen:seq_lens")
    if attrs.get("generate_stats") or has("stats_tensor_uid"):
        features.add("out:stats")
    if has("max_tensor_uid") or has("sum_exp_tensor_uid"):
        features.add("out:max_sum_exp")
    if has("rng_dump_tensor_uid"):
        features.add("out:rng_dump")
    if has("amax_s_tensor_uid") or has("amax_o_tensor_uid"):
        features.add("fp8:amax")
    if mixed_output(node, tensors_by_uid):
        features.add("precision:mixed_output")
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
    if has("page_table_k_tensor_uid") or has("page_table_v_tensor_uid"):
        features.add("kv:paged")
    if has("block_mask_tensor_uid"):
        features.add("input:block_mask")
    if has("sink_token_tensor_uid"):
        features.add("input:sink_token")
    return features
