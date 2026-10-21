# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``from_hipdnn``: hipDNN flatbuffer JSON -> ``SdpaRequest``.

Graphs are built in the shape hipDNN's JSON helpers and golden generator write:
a node with ``inputs`` / ``outputs`` / ``attributes`` and a ``tensors`` list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dispatch.sdpa import SdpaRequest, from_hipdnn, from_hipdnn_graph

B, H, S, D = 2, 4, 8, 16

_INPUT_UIDS = (
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
)
_OUTPUT_UIDS = (
    "stats_tensor_uid",
    "max_tensor_uid",
    "sum_exp_tensor_uid",
    "rng_dump_tensor_uid",
    "amax_s_tensor_uid",
    "amax_o_tensor_uid",
)


def _tensor(uid, dims, data_type="half", **extra):
    strides, acc = [], 1
    for d in reversed(dims):
        strides.append(acc)
        acc *= d
    entry = {
        "uid": uid,
        "name": f"t{uid}",
        "data_type": data_type,
        "dims": list(dims),
        "strides": list(reversed(strides)),
        "virtual": False,
    }
    entry.update(extra)
    return entry


def _graph(attributes=None, inputs=None, outputs=None, tensors=()):
    """A one-node SDPA graph; extra uids reference ``tensors`` (uids >= 10)."""
    node = {
        "type": "SdpaAttributes",
        "name": "",
        "compute_data_type": "float",
        "inputs": {"q_tensor_uid": 0, "k_tensor_uid": 1, "v_tensor_uid": 2},
        "outputs": {"o_tensor_uid": 3},
        "attributes": {
            "generate_stats": None,
            "alibi_mask": False,
            "padding_mask": False,
            "causal_mask": False,
            "causal_mask_bottom_right": False,
            "dropout_probability": None,
            "attn_scale_value": None,
            "left_bound": None,
            "right_bound": None,
            "max_seq_len_kv": None,
            "diagonal_alignment": "TOP_LEFT",
            "mma_core_mode": "unset",
            "implementation": "AUTO",
        },
    }
    node["inputs"].update({key: None for key in _INPUT_UIDS})
    node["outputs"].update({key: None for key in _OUTPUT_UIDS})
    node["inputs"].update(inputs or {})
    node["outputs"].update(outputs or {})
    node["attributes"].update(attributes or {})
    primaries = [_tensor(uid, (B, H, S, D)) for uid in range(4)]
    return {"nodes": [node], "tensors": primaries + list(tensors)}


def _request(**kwargs) -> SdpaRequest:
    return from_hipdnn_graph(_graph(**kwargs), "gfx950")


def test_minimal_node():
    req = _request()
    assert req.q.dims == (B, H, S, D) and req.q.dtype == "fp16"
    assert (req.scale_source, req.scale) == ("default", 1.0)
    assert (req.diagonal, req.left_bound, req.right_bound) == ("top_left", -1, -1)
    assert req.compute_dtype == "fp32"
    assert req.features() == {"layout_bhsd"}


def test_from_hipdnn_accepts_a_uid_mapping():
    graph = _graph()
    tensors = {t["uid"]: t for t in graph["tensors"]}
    assert from_hipdnn(graph["nodes"][0], tensors, "gfx950") == _request()


# --- mask resolution: the deprecated flags merge with the stored bounds ----------
# Expected values are what hipDNN's resolvers return: ``extractDiagonalBandParams``
# (test_sdk PlanUtils.hpp) and ``resolveDiagonalBand`` (asm SDPA engine).


@pytest.mark.parametrize(
    "attributes, band",
    [
        ({}, ("top_left", -1, -1)),
        ({"right_bound": 0}, ("top_left", -1, 0)),
        ({"left_bound": 3}, ("top_left", 3, -1)),
        (
            {"left_bound": 3, "right_bound": 0, "diagonal_alignment": "BOTTOM_RIGHT"},
            ("bottom_right", 3, 0),
        ),
        # A deprecated flag alone is causal.
        ({"causal_mask": True}, ("top_left", -1, 0)),
        ({"causal_mask_bottom_right": True}, ("bottom_right", -1, 0)),
        # causal_mask keeps the stored alignment; the bottom-right flag forces it.
        (
            {"causal_mask": True, "diagonal_alignment": "BOTTOM_RIGHT"},
            ("bottom_right", -1, 0),
        ),
        (
            {"causal_mask_bottom_right": True, "diagonal_alignment": "TOP_LEFT"},
            ("bottom_right", -1, 0),
        ),
        # The stored bounds apply on top of the flag: a sliding window, not causal.
        ({"causal_mask_bottom_right": True, "left_bound": 3}, ("bottom_right", 3, 0)),
        ({"causal_mask": True, "left_bound": 3}, ("top_left", 3, 0)),
        (
            {
                "causal_mask": True,
                "left_bound": 5,
                "right_bound": 2,
                "diagonal_alignment": "BOTTOM_RIGHT",
            },
            ("bottom_right", 5, 2),
        ),
        (
            {"causal_mask_bottom_right": True, "left_bound": 5, "right_bound": 2},
            ("bottom_right", 5, 2),
        ),
        # An unset right bound (-1) does not override the flag's right bound of 0.
        ({"causal_mask": True, "right_bound": -1}, ("top_left", -1, 0)),
    ],
)
def test_mask_merges_deprecated_flags_like_hipdnn(attributes, band):
    req = _request(attributes=attributes)
    assert (req.diagonal, req.left_bound, req.right_bound) == band


@pytest.mark.parametrize(
    "attributes",
    [
        {"left_bound": -4, "right_bound": -9},
        {"left_bound": -2},
        {"right_bound": -2},
        # The flag's right bound of 0 must not mask an invalid stored bound.
        {"causal_mask": True, "right_bound": -5},
        {"causal_mask_bottom_right": True, "left_bound": -3},
    ],
)
def test_bounds_below_minus_one_are_rejected_like_hipdnn(attributes):
    with pytest.raises(ValueError, match=r"must be >= -1"):
        _request(attributes=attributes)


def test_flag_with_a_stored_left_bound_is_a_window_not_causal():
    """The merged band is a sliding window, so a causal-only kernel must decline it."""
    req = _request(attributes={"causal_mask_bottom_right": True, "left_bound": 3})
    assert req.features() >= {"window"} and "causal_tl" not in req.features()


def test_both_deprecated_causal_flags_are_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _request(attributes={"causal_mask": True, "causal_mask_bottom_right": True})


# --- scale: exactly one source ---------------------------------------------------


def test_scale_value():
    req = _request(attributes={"attn_scale_value": 0.125})
    assert (req.scale_source, req.scale) == ("value", 0.125)


def test_scale_value_one_is_the_default_scale():
    """hipDNN reads an unset scale as 1.0, so a baked 1.0 is the same request."""
    baked = _request(attributes={"attn_scale_value": 1.0})
    assert (baked.scale_source, baked.scale) == ("default", 1.0)
    assert baked == _request() and baked.normalized() == _request().normalized()


@pytest.mark.parametrize("probability", [None, 0.0, 0])
def test_zero_dropout_probability_is_no_dropout(probability):
    """The asm SDPA engine gates on "unset or zero", so 0.0 is not a dropout request."""
    req = _request(attributes={"dropout_probability": probability})
    assert req.dropout is None and "dropout" not in req.features()
    assert req == _request()


def test_zero_probability_with_dropout_tensors_is_still_dropout():
    """The tensors configure dropout, so a candidate must still decline it."""
    seed = [_tensor(10, (1,), "int32")]
    req = _request(
        attributes={"dropout_probability": 0.0},
        inputs={"seed_tensor_uid": 10},
        tensors=seed,
    )
    assert req.dropout is not None and "dropout" in req.features()


def test_scale_tensor():
    req = _request(
        inputs={"scale_tensor_uid": 10}, tensors=[_tensor(10, (1,), "float")]
    )
    assert req.scale_source == "tensor" and req.scale_tensor.dims == (1,)
    assert "scale_tensor" in req.features()


def test_by_value_scale_tensor_folds_to_value():
    scale = _tensor(
        10,
        (1,),
        "float",
        is_runtime_pass_by_value=True,
        value_type="Float32Value",
        value=0.5,
    )
    req = _request(inputs={"scale_tensor_uid": 10}, tensors=[scale])
    assert (req.scale_source, req.scale, req.scale_tensor) == ("value", 0.5, None)


def test_scale_tensor_and_value_are_exclusive():
    with pytest.raises(ValueError, match="cannot both be set"):
        _request(
            attributes={"attn_scale_value": 0.5},
            inputs={"scale_tensor_uid": 10},
            tensors=[_tensor(10, (1,), "float")],
        )


# --- every other field lands somewhere ---------------------------------------------


def test_stats_needs_generate_stats_and_a_tensor():
    lse = [_tensor(10, (B, H, S, 1), "float")]
    # The hipDNN frontend never writes a stats uid without generate_stats.
    with pytest.raises(ValueError, match="generate_stats"):
        _request(outputs={"stats_tensor_uid": 10}, tensors=lse)
    req = _request(
        attributes={"generate_stats": True},
        outputs={"stats_tensor_uid": 10},
        tensors=lse,
    )
    assert req.stats.dims == (B, H, S, 1) and "lse" in req.features()
    # And the reverse: asking for stats without a tensor must not drop the LSE
    # request, or a kernel that writes no LSE would be admitted.
    with pytest.raises(ValueError, match="stats_tensor_uid is not set"):
        _request(attributes={"generate_stats": True})


@pytest.mark.parametrize(
    "kwargs, feature",
    [
        ({"attributes": {"alibi_mask": True}}, "alibi"),
        ({"attributes": {"padding_mask": True}}, "padded"),
        ({"inputs": {"seq_len_q_tensor_uid": 10}}, "padded"),
        ({"inputs": {"attn_mask_tensor_uid": 10}}, "bias"),
        ({"inputs": {"block_mask_tensor_uid": 10}}, "block_mask"),
        ({"inputs": {"sink_token_tensor_uid": 10}}, "sinks"),
        ({"attributes": {"dropout_probability": 0.1}}, "dropout"),
        ({"inputs": {"seed_tensor_uid": 10}}, "dropout"),
        ({"outputs": {"rng_dump_tensor_uid": 10}}, "dropout"),
        ({"inputs": {"page_table_k_tensor_uid": 10}}, "paged"),
        ({"attributes": {"max_seq_len_kv": 64}}, "paged"),
        ({"inputs": {"descale_q_tensor_uid": 10}}, "fp8"),
        ({"inputs": {"scale_o_tensor_uid": 10}}, "fp8_s_o"),
        ({"outputs": {"amax_s_tensor_uid": 10}}, "fp8_s_o"),
        ({"outputs": {"max_tensor_uid": 10}}, "max_sum_exp"),
        ({"outputs": {"sum_exp_tensor_uid": 10}}, "max_sum_exp"),
        ({"attributes": {"mma_core_mode": "float"}}, "mma_core_mode"),
        ({"attributes": {"implementation": "UNIFIED"}}, "implementation_hint"),
    ],
)
def test_field_reaches_features(kwargs, feature):
    extra = [_tensor(10, (1, 1, S, S), "float")]
    req = _request(tensors=extra, **kwargs)
    assert feature in req.features()


def test_ragged_and_dtypes_map():
    graph = _graph()
    for t in graph["tensors"]:
        t["data_type"] = "BFLOAT16"  # flatc spells enums in upper case
        t["strides"] = [S * H * D, D, H * D, 1]
        t["ragged_offset_tensor_uid"] = 10
        t["ragged_offset_multiplier"] = H * D
    graph["nodes"][0]["compute_data_type"] = "half"
    req = from_hipdnn_graph(graph, "gfx950")
    assert req.q.dtype == "bf16" and req.compute_dtype == "fp16"
    assert req.q.ragged and req.q.ragged_offset_multiplier == H * D
    assert {"ragged", "layout_bshd", "compute_dtype"} <= req.features()


@pytest.mark.parametrize(
    "graph_edit, match",
    [
        (lambda g: g["nodes"][0].update(type="MatmulAttributes"), "SdpaAttributes"),
        (lambda g: g["nodes"][0]["inputs"].update(q_tensor_uid=None), "missing"),
        (lambda g: g["nodes"][0]["inputs"].update(seed_tensor_uid=99), "uid 99"),
        (lambda g: g["tensors"][0].update(data_type="fp12"), "data_type"),
        (lambda g: g["tensors"][1].update(dims=[B, H, S, 2 * D]), "head dim"),
    ],
)
def test_malformed_graphs_are_rejected(graph_edit, match):
    graph = _graph()
    graph_edit(graph)
    with pytest.raises(ValueError, match=match):
        from_hipdnn_graph(graph, "gfx950")


@pytest.mark.parametrize(
    "graph_edit",
    [
        pytest.param(lambda g: g["nodes"][0].update(sdpa_version=2), id="node"),
        pytest.param(
            lambda g: g["nodes"][0]["inputs"].update(rope_tensor_uid=None), id="input"
        ),
        pytest.param(
            lambda g: g["nodes"][0]["outputs"].update(lse_tensor_uid=None), id="output"
        ),
        pytest.param(
            lambda g: g["nodes"][0]["attributes"].update(softcap=30.0), id="attribute"
        ),
        pytest.param(
            lambda g: g["nodes"][0]["attributes"].update(causal_mask_botom_right=True),
            id="misspelt_attribute",
        ),
        pytest.param(lambda g: g["tensors"][0].update(stride_order=[0]), id="tensor"),
    ],
)
def test_unknown_keys_are_rejected_not_dropped(graph_edit):
    """A key from_hipdnn does not read must fail, not leave the request unchanged."""
    graph = _graph()
    graph_edit(graph)
    with pytest.raises(ValueError, match="hipDNN does not write"):
        from_hipdnn_graph(graph, "gfx950")


_SDPA_GRAPH_TEMPLATE = (
    Path(__file__).resolve().parents[6]
    / "projects/hipdnn/tools/IngestorGenerator/configs/templates"
    / "attention_sdpa_fwd.graph.json"
)


@pytest.mark.skipif(
    not _SDPA_GRAPH_TEMPLATE.is_file(), reason="IngestorGenerator not in this tree"
)
def test_key_sets_match_the_ingestor_sdpa_graph_template():
    """The ingestor's SDPA graph template carries every key hipDNN writes."""
    from dispatch.sdpa import hipdnn

    node = json.loads(_SDPA_GRAPH_TEMPLATE.read_text())["nodes"][0]
    assert set(node) == hipdnn._NODE_KEYS
    assert set(node["inputs"]) == hipdnn._INPUT_KEYS
    assert set(node["outputs"]) == hipdnn._OUTPUT_KEYS
    assert set(node["attributes"]) == hipdnn._ATTRIBUTE_KEYS


# --- the checked-in hipDNN golden bundles ------------------------------------------

_BUNDLES = (
    Path(__file__).resolve().parents[5]
    / "integration-tests"
    / "integration-test-bundles"
)
_GRAPHS = sorted(
    p
    for p in _BUNDLES.glob("*/SdpaFwd*/**/*.json")
    if not p.name.endswith(("meta.json", "support.json", "sweep.json"))
    and "Default" not in p.parts
)


@pytest.mark.skipif(not _GRAPHS, reason="hipDNN SDPA golden bundles not in this tree")
@pytest.mark.parametrize("path", _GRAPHS, ids=lambda p: str(p.relative_to(_BUNDLES)))
def test_golden_bundle_graphs_translate(path):
    graph = json.loads(path.read_text())
    req = from_hipdnn_graph(graph, "gfx950")
    assert SdpaRequest.from_normalized(json.loads(json.dumps(req.normalized()))) == req
    node = graph["nodes"][0]
    assert ("lse" in req.features()) == bool(node["attributes"]["generate_stats"])
    assert ("padded" in req.features()) == (
        node["inputs"]["seq_len_q_tensor_uid"] is not None
    )
