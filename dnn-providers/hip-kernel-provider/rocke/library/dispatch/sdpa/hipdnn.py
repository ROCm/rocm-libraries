# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Translate a hipDNN SDPA forward node, in its flatbuffer JSON form, into an ``SdpaRequest``.

The input is what the hipDNN flatbuffer SDK's JSON helpers write
(``utilities/json/SdpaAttributes.hpp`` and ``TensorAttributes.hpp``): a node
with ``inputs`` / ``outputs`` / ``attributes`` maps of uids and values, plus
the graph's ``tensors`` list. Golden bundles and the hipDNN engine share that
format, so the case table and the C++ engine read one input.

Mask resolution follows hipDNN's resolvers: both deprecated causal flags set
is an error; a flag sets ``right_bound = 0`` (and bottom-right alignment for
``causal_mask_bottom_right``) and the stored ``left_bound``, ``right_bound``
and ``diagonal_alignment`` are applied on top; an unset bound reads as
unbounded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .common import SdpaBias, SdpaDropout, SdpaFp8, SdpaPaged, SdpaRequest, SdpaTensor

__all__ = ["HIPDNN_DTYPES", "from_hipdnn", "from_hipdnn_graph"]

# hipDNN ``DataType`` JSON names (``utilities/json/Common.hpp``) -> rocKE dtype.
# flatc writes the enum names in upper case; both spellings are accepted.
HIPDNN_DTYPES: dict[str, str | None] = {
    "unset": None,
    "float": "fp32",
    "half": "fp16",
    "bfloat16": "bf16",
    "double": "fp64",
    "uint8": "uint8",
    "int32": "int32",
    "int8": "int8",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
    "fp8_e8m0": "fp8_e8m0",
    "fp4_e2m1": "fp4_e2m1",
    "fp6_e2m3": "fp6_e2m3",
    "fp6_e3m2": "fp6_e3m2",
    "int4": "int4",
    "int64": "int64",
    "boolean": "bool",
    "fp8_e4m3_fnuz": "fp8_e4m3_fnuz",
    "fp8_e5m2_fnuz": "fp8_e5m2_fnuz",
}


# Every key hipDNN's JSON helpers write. A key outside these sets raises rather
# than being dropped, so a new or misspelt hipDNN field cannot leave the
# request unchanged.
_NODE_KEYS = frozenset(
    {"name", "type", "compute_data_type", "inputs", "outputs", "attributes"}
)
_INPUT_KEYS = frozenset(
    {
        "q_tensor_uid",
        "k_tensor_uid",
        "v_tensor_uid",
        "attn_mask_tensor_uid",
        "scale_tensor_uid",
        "seq_len_q_tensor_uid",
        "seq_len_kv_tensor_uid",
        "seed_tensor_uid",
        "offset_tensor_uid",
        "dropout_mask_tensor_uid",
        "dropout_scale_tensor_uid",
        "page_table_k_tensor_uid",
        "page_table_v_tensor_uid",
        "block_mask_tensor_uid",
        "sink_token_tensor_uid",
        "descale_q_tensor_uid",
        "descale_k_tensor_uid",
        "descale_v_tensor_uid",
        "descale_s_tensor_uid",
        "scale_s_tensor_uid",
        "scale_o_tensor_uid",
    }
)
_OUTPUT_KEYS = frozenset(
    {
        "o_tensor_uid",
        "stats_tensor_uid",
        "max_tensor_uid",
        "sum_exp_tensor_uid",
        "rng_dump_tensor_uid",
        "amax_s_tensor_uid",
        "amax_o_tensor_uid",
    }
)
_ATTRIBUTE_KEYS = frozenset(
    {
        "generate_stats",
        "alibi_mask",
        "padding_mask",
        "causal_mask",
        "causal_mask_bottom_right",
        "dropout_probability",
        "attn_scale_value",
        "left_bound",
        "right_bound",
        "max_seq_len_kv",
        "diagonal_alignment",
        "mma_core_mode",
        "implementation",
    }
)
_TENSOR_KEYS = frozenset(
    {
        "uid",
        "name",
        "data_type",
        "dims",
        "strides",
        "virtual",
        "is_runtime_pass_by_value",
        "value",
        "value_type",
        "ragged_offset_tensor_uid",
        "ragged_offset_multiplier",
        "alignment",
    }
)


def _reject_unknown_keys(where: str, group: Mapping, allowed: frozenset[str]) -> None:
    unknown = sorted(set(group) - allowed)
    if unknown:
        raise ValueError(f"{where} has keys hipDNN does not write: {unknown}")


def _dtype(name: object) -> str | None:
    key = str(name).lower()
    if key not in HIPDNN_DTYPES:
        raise ValueError(f"unknown hipDNN data_type {name!r}")
    return HIPDNN_DTYPES[key]


def _tensor_table(tensors_json: Iterable[Mapping] | Mapping) -> dict[int, Mapping]:
    if isinstance(tensors_json, Mapping):
        return {int(uid): entry for uid, entry in tensors_json.items()}
    return {int(entry["uid"]): entry for entry in tensors_json}


def _sdpa_tensor(entry: Mapping) -> SdpaTensor:
    _reject_unknown_keys(f"tensor uid {entry.get('uid')}", entry, _TENSOR_KEYS)
    dtype = _dtype(entry["data_type"])
    if dtype is None:
        raise ValueError(f"tensor uid {entry.get('uid')} has data_type unset")
    return SdpaTensor(
        dtype=dtype,
        dims=tuple(entry["dims"]),
        strides=tuple(entry["strides"]),
        ragged=entry.get("ragged_offset_tensor_uid") is not None,
        ragged_offset_multiplier=entry.get("ragged_offset_multiplier", 1),
        by_value=bool(entry.get("is_runtime_pass_by_value", False)),
        alignment=entry.get("alignment", 16),
    )


def _resolve_band(attrs: Mapping) -> tuple[str, int, int]:
    """``(diagonal, left_bound, right_bound)``, merging the deprecated flags.

    Follows hipDNN's resolvers (``extractDiagonalBandParams`` in the CPU
    reference, ``resolveDiagonalBand`` in the asm SDPA engine): a deprecated
    flag acts as if its setter ran first (right bound 0, plus bottom-right
    alignment for ``causal_mask_bottom_right``), and the stored bounds and
    alignment are applied on top. ``causal_mask`` keeps the stored alignment.
    """
    causal = bool(attrs.get("causal_mask") or False)
    causal_br = bool(attrs.get("causal_mask_bottom_right") or False)
    if causal and causal_br:
        raise ValueError(
            "causal_mask and causal_mask_bottom_right are mutually exclusive "
            "but both are set"
        )
    alignment = str(attrs.get("diagonal_alignment") or "TOP_LEFT").upper()
    if alignment not in ("TOP_LEFT", "BOTTOM_RIGHT"):
        raise ValueError(f"unknown diagonal_alignment {alignment!r}")
    left = attrs.get("left_bound")
    right = attrs.get("right_bound")
    left = -1 if left is None else int(left)
    right = -1 if right is None else int(right)
    if left < -1 or right < -1:
        raise ValueError(
            "left_bound and right_bound must be >= -1 "
            f"(got left_bound={left}, right_bound={right})"
        )
    if causal or causal_br:
        right = right if right >= 0 else 0
        if causal_br:
            alignment = "BOTTOM_RIGHT"
    return alignment.lower(), left, right


def from_hipdnn(
    node_json: Mapping[str, Any],
    tensors_json: Iterable[Mapping] | Mapping,
    arch: str,
) -> SdpaRequest:
    """Build an :class:`SdpaRequest` from one SDPA node and the graph's tensors.

    ``tensors_json`` is the graph's ``tensors`` list, or a mapping from uid to
    tensor entry. Raises ``ValueError`` for anything the hipDNN frontend or
    mask resolver would also reject, and for any key hipDNN does not write.
    An absent key reads as hipDNN's null.
    """
    node_type = node_json.get("type", "SdpaAttributes")
    if node_type != "SdpaAttributes":
        raise ValueError(f"expected an SdpaAttributes node, got {node_type!r}")
    inputs = node_json.get("inputs", {})
    outputs = node_json.get("outputs", {})
    attrs = node_json.get("attributes", {})
    _reject_unknown_keys("SDPA node", node_json, _NODE_KEYS)
    _reject_unknown_keys("SDPA node inputs", inputs, _INPUT_KEYS)
    _reject_unknown_keys("SDPA node outputs", outputs, _OUTPUT_KEYS)
    _reject_unknown_keys("SDPA node attributes", attrs, _ATTRIBUTE_KEYS)
    table = _tensor_table(tensors_json)

    def entry(uid: object) -> Mapping | None:
        if uid is None:
            return None
        if int(uid) not in table:
            raise ValueError(
                f"node references tensor uid {uid}, which is not in the graph"
            )
        return table[int(uid)]

    def tensor(group: Mapping, key: str) -> SdpaTensor | None:
        found = entry(group.get(key))
        return None if found is None else _sdpa_tensor(found)

    def required(group: Mapping, key: str) -> SdpaTensor:
        if group.get(key) is None:
            raise ValueError(f"SDPA node is missing required {key}")
        return tensor(group, key)

    kwargs: dict[str, Any] = {
        "arch": arch,
        "q": required(inputs, "q_tensor_uid"),
        "k": required(inputs, "k_tensor_uid"),
        "v": required(inputs, "v_tensor_uid"),
        "o": required(outputs, "o_tensor_uid"),
        "compute_dtype": _dtype(node_json.get("compute_data_type", "float")) or "fp32",
    }

    scale_value = attrs.get("attn_scale_value")
    scale_entry = entry(inputs.get("scale_tensor_uid"))
    if scale_value is not None and scale_entry is not None:
        raise ValueError("attn_scale tensor and attn_scale_value cannot both be set")
    if scale_entry is not None:
        folded = scale_entry.get("value")
        if scale_entry.get("is_runtime_pass_by_value") and folded is not None:
            kwargs.update(scale_source="value", scale=float(folded))
        else:
            kwargs.update(scale_source="tensor", scale_tensor=_sdpa_tensor(scale_entry))
    elif scale_value is not None:
        kwargs.update(scale_source="value", scale=float(scale_value))

    diagonal, left, right = _resolve_band(attrs)
    kwargs.update(diagonal=diagonal, left_bound=left, right_bound=right)
    kwargs["alibi"] = bool(attrs.get("alibi_mask") or False)
    bias = tensor(inputs, "attn_mask_tensor_uid")
    if bias is not None:
        kwargs["bias"] = SdpaBias(bias)
    kwargs["block_mask"] = tensor(inputs, "block_mask_tensor_uid")
    kwargs["sink_token"] = tensor(inputs, "sink_token_tensor_uid")

    kwargs["padding_mask"] = bool(attrs.get("padding_mask") or False)
    kwargs["seq_len_q"] = tensor(inputs, "seq_len_q_tensor_uid")
    kwargs["seq_len_kv"] = tensor(inputs, "seq_len_kv_tensor_uid")

    dropout = {
        "seed": tensor(inputs, "seed_tensor_uid"),
        "offset": tensor(inputs, "offset_tensor_uid"),
        "mask": tensor(inputs, "dropout_mask_tensor_uid"),
        "scale": tensor(inputs, "dropout_scale_tensor_uid"),
        "rng_dump": tensor(outputs, "rng_dump_tensor_uid"),
    }
    probability = attrs.get("dropout_probability")
    if probability is not None or any(t is not None for t in dropout.values()):
        kwargs["dropout"] = SdpaDropout(
            probability=0.0 if probability is None else probability, **dropout
        )

    paged = SdpaPaged(
        page_table_k=tensor(inputs, "page_table_k_tensor_uid"),
        page_table_v=tensor(inputs, "page_table_v_tensor_uid"),
        max_seq_len_kv=attrs.get("max_seq_len_kv"),
    )
    if paged != SdpaPaged():
        kwargs["paged"] = paged

    fp8 = SdpaFp8(
        descale_q=tensor(inputs, "descale_q_tensor_uid"),
        descale_k=tensor(inputs, "descale_k_tensor_uid"),
        descale_v=tensor(inputs, "descale_v_tensor_uid"),
        descale_s=tensor(inputs, "descale_s_tensor_uid"),
        scale_s=tensor(inputs, "scale_s_tensor_uid"),
        scale_o=tensor(inputs, "scale_o_tensor_uid"),
        amax_s=tensor(outputs, "amax_s_tensor_uid"),
        amax_o=tensor(outputs, "amax_o_tensor_uid"),
    )
    if fp8 != SdpaFp8():
        kwargs["fp8"] = fp8

    # The hipDNN frontend creates the stats output exactly when generate_stats is
    # true, so either one without the other is contradictory input. Dropping it
    # would let a kernel that writes no LSE be admitted for a node that asked for it.
    stats = tensor(outputs, "stats_tensor_uid")
    if attrs.get("generate_stats"):
        if stats is None:
            raise ValueError("generate_stats is true but stats_tensor_uid is not set")
        kwargs["stats"] = stats
    elif stats is not None:
        raise ValueError("stats_tensor_uid is set but generate_stats is not true")
    kwargs["max"] = tensor(outputs, "max_tensor_uid")
    kwargs["sum_exp"] = tensor(outputs, "sum_exp_tensor_uid")

    kwargs["mma_core_mode"] = _dtype(attrs.get("mma_core_mode") or "unset")
    kwargs["implementation"] = str(attrs.get("implementation") or "AUTO").lower()
    return SdpaRequest(**kwargs)


def from_hipdnn_graph(
    graph_json: Mapping[str, Any], arch: str, *, node_index: int = 0
) -> SdpaRequest:
    """:func:`from_hipdnn` for one node of a whole graph (a golden bundle's ``.json``)."""
    return from_hipdnn(graph_json["nodes"][node_index], graph_json["tensors"], arch)
