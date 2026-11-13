# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The hipDNN ``SdpaBackward`` lowering of the native backward engine (CPU only).

* a bundle-shaped dense graph and a hipDNN-shaped THD graph (token offsets,
  per-side multipliers, packed LSE) lower to the expected request fields
  and plan;
* band translation (``left_bound = L`` to the window ``L - 1``, causal flags,
  diagonal alignment) and the band errors;
* every LOWERING rule: operands, ranks, dims, offset tables, a ragged tensor
  the lowering does not consume, stats alignment;
* THD forms: partial raggedness, two tables on one side, stats on the K
  table, mixed offset dtypes, dense stats under THD;
* the smallest declared alignment selects the access width;
* feature mappings decline with the predicate's codes;
* the inverse round-trips, and the plan record lists every launch in ABI
  order.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from dispatch.attention_bwd.hipdnn import (
    ENGINE_ARCHS,
    LOWERING,
    MAIN_TOKENS,
    PREP_TOKENS,
    HipdnnLoweringError,
    engine_policy,
    engine_verdict,
    from_hipdnn_graph,
    plan_record,
    tensor_roles,
    to_hipdnn_graph,
    tuned_keys,
    tuning_key_id,
)
from kernels.common.attention_bwd import attn_bwd_params
from kernels.common.attention_bwd_plan import (
    SHIPPED_DECLARED_GROUPS,
    AttnBwdRequest,
    attn_bwd_plan,
)

_ROLES = {"q": 2, "k": 0, "v": 6, "o": 1, "do": 7, "lse": 8, "dq": 4, "dk": 3, "dv": 5}


def _tensor(uid, name, dims, strides, dtype="bfloat16", **extra):
    entry = {
        "uid": uid,
        "name": name,
        "data_type": dtype,
        "dims": list(dims),
        "strides": list(strides),
        "virtual": False,
        "is_runtime_pass_by_value": False,
        "alignment": 16,
        "ragged_offset_multiplier": 1,
    }
    entry.update(extra)
    return entry


def _node(**params):
    base = {
        "alibi_mask": False,
        "padding_mask": False,
        "causal_mask": False,
        "causal_mask_bottom_right": False,
        "dropout_probability": None,
        "attn_scale_value": None,
        "left_bound": None,
        "right_bound": None,
        "diagonal_alignment": "TOP_LEFT",
    }
    base.update(params)
    inputs = {
        k: None
        for k in (
            "scale_tensor_uid",
            "attn_mask_tensor_uid",
            "seq_len_q_tensor_uid",
            "seq_len_kv_tensor_uid",
            "seed_tensor_uid",
            "offset_tensor_uid",
            "dropout_mask_tensor_uid",
            "dropout_scale_tensor_uid",
            "dropout_scale_inv_tensor_uid",
        )
    }
    for role in ("q", "k", "v", "o", "do"):
        inputs[f"{role}_tensor_uid"] = _ROLES[role]
    inputs["stats_tensor_uid"] = _ROLES["lse"]
    return {
        "name": "SdpaBwd_0",
        "type": "SdpaBackwardAttributes",
        "compute_data_type": "float",
        "inputs": inputs,
        "outputs": {
            "dq_tensor_uid": _ROLES["dq"],
            "dk_tensor_uid": _ROLES["dk"],
            "dv_tensor_uid": _ROLES["dv"],
            "dbias_tensor_uid": None,
        },
        "parameters": base,
    }


def _dense_graph(b=2, h_q=4, h_k=4, h_v=4, s_q=72, s_kv=40, d=64, **params):
    """A dense BHSD graph shaped like the SdpaBackward bundles."""

    def bhsd(h, s):
        return (b, h, s, d), (h * s * d, s * d, d, 1)

    q, kv_k, kv_v = bhsd(h_q, s_q), bhsd(h_k, s_kv), bhsd(h_v, s_kv)
    tensors = [
        _tensor(_ROLES["k"], "K", *kv_k),
        _tensor(_ROLES["o"], "O", *q),
        _tensor(_ROLES["q"], "Q", *q),
        _tensor(_ROLES["dk"], "dK", *kv_k),
        _tensor(_ROLES["dq"], "dQ", *q),
        _tensor(_ROLES["dv"], "dV", *kv_v),
        _tensor(_ROLES["v"], "V", *kv_v),
        _tensor(_ROLES["do"], "dO", *q),
        _tensor(
            _ROLES["lse"],
            "stats",
            (b, h_q, s_q, 1),
            (h_q * s_q, s_q, 1, 1),
            "float",
        ),
    ]
    return {"name": "SdpaBackwardTest", "nodes": [_node(**params)], "tensors": tensors}


_OFF_Q, _OFF_KV = 20, 21


def _thd_graph(b=3, h_q=8, h_k=2, h_v=2, s_q=96, s_kv=80, d=128, **params):
    """A packed THD graph as hipDNN writes it: token-major strides, offsets in
    tokens (multiplier H * D per side), the LSE packed [T, H, 1] and ragged on
    Q's table (multiplier H)."""
    g = _dense_graph(b, h_q, h_k, h_v, s_q, s_kv, d, **params)
    heads = {"q": h_q, "o": h_q, "do": h_q, "dq": h_q}
    heads.update(k=h_k, dk=h_k, v=h_v, dv=h_v)
    by_uid = {t["uid"]: t for t in g["tensors"]}
    for role, h in heads.items():
        t = by_uid[_ROLES[role]]
        s = t["dims"][2]
        t["strides"] = [s * h * d, d, h * d, 1]
        t["ragged_offset_tensor_uid"] = (
            _OFF_Q if role in ("q", "o", "do", "dq") else _OFF_KV
        )
        t["ragged_offset_multiplier"] = h * d
    lse = by_uid[_ROLES["lse"]]
    lse["strides"] = [s_q * h_q, 1, h_q, 1]
    lse["ragged_offset_tensor_uid"] = _OFF_Q
    lse["ragged_offset_multiplier"] = h_q
    for uid, name in ((_OFF_Q, "offsets_q"), (_OFF_KV, "offsets_kv")):
        g["tensors"].append(_tensor(uid, name, (b + 1, 1, 1, 1), (1, 1, 1, 1), "int32"))
    return g


def _entry(g, role):
    uid = _ROLES[role] if isinstance(role, str) else role
    return next(t for t in g["tensors"] if t["uid"] == uid)


def _lower(g, arch="gfx942", num_cus=304):
    return from_hipdnn_graph(g, arch, num_cus=num_cus)


def _verdict(g, arch="gfx942", num_cus=304):
    try:
        req = _lower(g, arch, num_cus)
    except HipdnnLoweringError as exc:
        return LOWERING, str(exc)
    code, detail, _plan = engine_verdict(req, arch)
    return code, detail


# ---------------------------------------------------------------------------
# dense and THD mapping
# ---------------------------------------------------------------------------


def test_dense_bundle_graph_lowers_field_by_field():
    g = _dense_graph(attn_scale_value=0.125)
    req = _lower(g)
    assert (req.b, req.h_q, req.h_k, req.h_v) == (2, 4, 4, 4)
    assert (req.s_q, req.s_kv, req.d_qk, req.d_v) == (72, 40, 64, 64)
    assert req.dtype == "bf16" and req.dq_dtype == "bf16" and req.lse_dtype == "fp32"
    assert req.layout == "dense" and req.lse_ragged is False
    assert req.strides["q"] == (4 * 72 * 64, 72 * 64, 64, 1)
    assert req.lse_dims == (2, 4, 72, 1) and req.lse_strides == (288, 72, 1, 1)
    assert req.scale == 0.125 and req.tensor_alignment == 16 and req.num_cus == 304
    assert (req.left_bound, req.right_bound, req.bottom_right) == (-1, -1, False)
    for arch in ENGINE_ARCHS:
        code, _, plan = engine_verdict(req, arch)
        assert code == "OK" and plan.route == "general"


def test_thd_graph_with_per_side_multipliers():
    g = _thd_graph()
    req = _lower(g)
    assert req.layout == "thd" and req.ragged_same_table
    assert req.ragged_offset_dtype == "int32" and req.ragged_offset_dims == (4,)
    m = dict(req.ragged_offset_multipliers)
    assert m == {
        "q": 1024, "o": 1024, "do": 1024, "dq": 1024,
        "k": 256, "v": 256, "dk": 256, "dv": 256, "lse": 8,
    }  # fmt: skip
    assert req.strides["q"] == (0, 128, 1024, 1) and req.strides["k"] == (
        0,
        128,
        256,
        1,
    )
    assert req.lse_ragged is True
    for arch in ENGINE_ARCHS:
        code, detail, plan = engine_verdict(req, arch)
        assert code == "OK", detail
        main = next(s for s in plan.launches if s.stage == "main").scalars
        assert (main["q_mult"], main["q_div"]) == (1024, 1024)
        assert (main["kv_mult"], main["kv_div"]) == (256, 256)
        assert main["off64"] == 0
        prep = plan.launches[0].scalars
        assert (prep["l_h"], prep["l_t"]) == (1, 8)


def test_thd_int64_and_mqa_and_hk_ne_hv():
    for h_k, h_v in ((1, 1), (4, 2)):
        g = _thd_graph(h_k=h_k, h_v=h_v)
        for name in ("offsets_q", "offsets_kv"):
            uid = _OFF_Q if name == "offsets_q" else _OFF_KV
            _entry(g, uid)["data_type"] = "INT64"
        req = _lower(g)
        assert req.ragged_offset_dtype == "int64"
        code, detail, plan = engine_verdict(req, "gfx950")
        assert code == "OK", detail
        assert plan.launches[1].scalars["off64"] == 1


def test_tensor_roles():
    roles = tensor_roles(_thd_graph())
    assert roles == {**_ROLES, "offsets_q": _OFF_Q, "offsets_kv": _OFF_KV}
    assert tensor_roles(_dense_graph())["offsets_q"] is None


# ---------------------------------------------------------------------------
# band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, band",
    [
        ({}, (-1, -1, False)),
        ({"causal_mask": True}, (-1, 0, False)),
        ({"causal_mask_bottom_right": True}, (-1, 0, True)),
        ({"left_bound": 16, "right_bound": 0}, (15, 0, False)),
        (
            {"left_bound": 1, "right_bound": 3, "diagonal_alignment": "BOTTOM_RIGHT"},
            (0, 3, True),
        ),
        ({"causal_mask": True, "left_bound": 8}, (7, 0, False)),
        ({"left_bound": -1, "right_bound": -1}, (-1, -1, False)),
    ],
)
def test_band_translation(params, band):
    req = _lower(_dense_graph(**params))
    assert (req.left_bound, req.right_bound, req.bottom_right) == band
    assert not req.causal and not req.causal_bottom_right


@pytest.mark.parametrize(
    "params",
    [
        {"causal_mask": True, "causal_mask_bottom_right": True},
        {"left_bound": 0},
        {"right_bound": -2},
        {"diagonal_alignment": "CENTER"},
    ],
)
def test_band_errors_are_lowering_declines(params):
    assert _verdict(_dense_graph(**params))[0] == LOWERING


def test_far_left_bound_is_a_band_decline():
    code, _ = _verdict(_dense_graph(left_bound=2**30 + 1, right_bound=0))
    assert code == "BAND_BOUND"


# ---------------------------------------------------------------------------
# LOWERING rules
# ---------------------------------------------------------------------------


def _drop_input(g, key):
    g["nodes"][0]["inputs"][key] = None


def _set(g, role, **kw):
    _entry(g, role).update(kw)


_LOWERING_MUTATIONS = {
    "missing_q": lambda g: _drop_input(g, "q_tensor_uid"),
    "missing_stats": lambda g: _drop_input(g, "stats_tensor_uid"),
    "missing_dv": lambda g: g["nodes"][0]["outputs"].update(dv_tensor_uid=None),
    "absent_key": lambda g: g["nodes"][0]["inputs"].pop("do_tensor_uid"),
    "unknown_uid": lambda g: g["nodes"][0]["inputs"].update(k_tensor_uid=99),
    "virtual_o": lambda g: _set(g, "o", virtual=True),
    "pass_by_value_dk": lambda g: _set(g, "dk", is_runtime_pass_by_value=True),
    "rank_3_dims": lambda g: _set(g, "q", dims=[2, 4, 72]),
    "rank_3_strides": lambda g: _set(g, "dq", strides=[288, 64, 1]),
    "negative_dim": lambda g: _set(g, "v", dims=[2, 4, -40, 64]),
    "batch_mismatch": lambda g: _set(g, "k", dims=[3, 4, 40, 64]),
    "dq_dims": lambda g: _set(g, "dq", dims=[2, 4, 36, 64]),
    "o_head_dim": lambda g: _set(g, "o", dims=[2, 4, 72, 32]),
    "kv_seq_mismatch": lambda g: _set(g, "v", dims=[2, 4, 41, 64]),
    "k_head_dim": lambda g: _set(g, "k", dims=[2, 4, 40, 32]),
    "stats_dims": lambda g: _set(g, "lse", dims=[2, 4, 71, 1]),
    "stats_alignment": lambda g: _set(g, "lse", alignment=2),
    "missing_scale_tensor": lambda g: g["nodes"][0]["inputs"].update(
        scale_tensor_uid=77
    ),
    "non_bool_flag": lambda g: g["nodes"][0]["parameters"].update(alibi_mask=1),
    "inputs_not_a_map": lambda g: g["nodes"][0].update(inputs=[]),
    "tensors_not_a_list": lambda g: g.update(tensors={}),
    "bool_uid": lambda g: g["tensors"][0].update(uid=True),
}


_LOWERING_MESSAGES = {
    "tensors_not_a_list": "graph tensors is not a list",
    "bool_uid": "has no integer uid",
}


@pytest.mark.parametrize("name", sorted(_LOWERING_MUTATIONS))
def test_lowering_rules(name):
    g = _dense_graph()
    _LOWERING_MUTATIONS[name](g)
    with pytest.raises(HipdnnLoweringError, match=_LOWERING_MESSAGES.get(name)):
        _lower(g)


def _thd_offset_mutation(fn):
    def apply(g):
        fn(g, _entry(g, _OFF_Q))

    return apply


_THD_LOWERING = {
    "offsets_missing": lambda g: _set(g, "q", ragged_offset_tensor_uid=55),
    "offsets_virtual": _thd_offset_mutation(lambda g, t: t.update(virtual=True)),
    "offsets_by_value": _thd_offset_mutation(
        lambda g, t: t.update(is_runtime_pass_by_value=True)
    ),
    "offsets_ragged": _thd_offset_mutation(
        lambda g, t: t.update(ragged_offset_tensor_uid=_OFF_KV)
    ),
    "offsets_rank_1": _thd_offset_mutation(
        lambda g, t: t.update(dims=[4], strides=[1])
    ),
    "offsets_strided": _thd_offset_mutation(
        lambda g, t: t.update(strides=[2, 1, 1, 1])
    ),
    "unconsumed_ragged_tensor": lambda g: g["tensors"].append(
        _tensor(
            40, "extra", (3, 1, 1, 1), (1, 1, 1, 1), ragged_offset_tensor_uid=_OFF_Q
        )
    ),
}


@pytest.mark.parametrize("name", sorted(_THD_LOWERING))
def test_thd_lowering_rules(name):
    g = _thd_graph()
    _THD_LOWERING[name](g)
    assert _verdict(g)[0] == LOWERING


def test_ragged_seq_len_tensor_is_not_consumed():
    g = _dense_graph(padding_mask=True)
    for side, uid in (("q", 30), ("kv", 31)):
        g["nodes"][0]["inputs"][f"seq_len_{side}_tensor_uid"] = uid
        g["tensors"].append(
            _tensor(uid, f"seq_len_{side}", (2, 1, 1, 1), (1, 1, 1, 1), "int32")
        )
    assert _verdict(g)[0] == "NOT_YET_DECLARED"
    g["tensors"].append(_tensor(32, "off", (3, 1, 1, 1), (1, 1, 1, 1), "int32"))
    _set(g, 30, ragged_offset_tensor_uid=32)
    assert _verdict(g)[0] == LOWERING


def test_not_an_sdpa_backward_graph():
    two = _dense_graph()
    two["nodes"].append(copy.deepcopy(two["nodes"][0]))
    fwd = _dense_graph()
    fwd["nodes"][0]["type"] = "SdpaAttributes"
    for g in (two, fwd, {"nodes": [], "tensors": []}):
        with pytest.raises(HipdnnLoweringError, match="not an SdpaBackward graph"):
            _lower(g)
        with pytest.raises(HipdnnLoweringError):
            tensor_roles(g)


# ---------------------------------------------------------------------------
# THD forms the predicate declines
# ---------------------------------------------------------------------------


def _unragged(g, role):
    t = _entry(g, role)
    t.pop("ragged_offset_tensor_uid")
    t["ragged_offset_multiplier"] = 1


def _second_q_table(g):
    g["tensors"].append(_tensor(22, "offsets_q2", (4, 1, 1, 1), (1, 1, 1, 1), "int32"))
    _set(g, "do", ragged_offset_tensor_uid=22)


def _both_tables(g, **kw):
    for uid in (_OFF_Q, _OFF_KV):
        _entry(g, uid).update(kw)


_THD_DECLINES = {
    "partial_ragged_o": (lambda g: _unragged(g, "o"), "RAGGED_FORMAT"),
    "partial_ragged_dv": (lambda g: _unragged(g, "dv"), "RAGGED_FORMAT"),
    "two_q_tables": (_second_q_table, "RAGGED_FORMAT"),
    "stats_on_kv_table": (
        lambda g: _set(g, "lse", ragged_offset_tensor_uid=_OFF_KV),
        "RAGGED_FORMAT",
    ),
    "mixed_offset_dtypes": (
        lambda g: _entry(g, _OFF_KV).update(data_type="int64"),
        "RAGGED_FORMAT",
    ),
    "float_offsets": (
        lambda g: _both_tables(g, data_type="float"),
        "RAGGED_FORMAT",
    ),
    "offset_dims": (
        lambda g: _both_tables(g, dims=[5, 1, 1, 1]),
        "RAGGED_FORMAT",
    ),
    "q_multiplier_not_tokens": (
        lambda g: _set(g, "o", ragged_offset_multiplier=512),
        "RAGGED_FORMAT",
    ),
    "lse_multiplier_not_tokens": (
        lambda g: _set(g, "lse", ragged_offset_multiplier=4),
        "RAGGED_FORMAT",
    ),
    "zero_multiplier": (
        lambda g: _set(g, "k", ragged_offset_multiplier=0),
        "RAGGED_FORMAT",
    ),
    "dense_stats_under_thd": (lambda g: _unragged(g, "lse"), "LSE_FORMAT"),
}


@pytest.mark.parametrize("name", sorted(_THD_DECLINES))
def test_thd_forms_declined_by_the_predicate(name):
    g = _thd_graph()
    fn, code = _THD_DECLINES[name]
    fn(g)
    for arch in ENGINE_ARCHS:
        got, detail = _verdict(g, arch)
        assert got == code, (name, detail)


def test_dense_stats_under_thd_names_the_form():
    g = _thd_graph()
    _unragged(g, "lse")
    _set(g, "lse", strides=[96 * 8, 96, 1, 1])
    code, detail = _verdict(g)
    assert code == "LSE_FORMAT" and "dense per-batch" in detail


def test_ragged_stats_in_a_dense_graph():
    g = _dense_graph()
    g["tensors"].append(
        _tensor(_OFF_Q, "offsets_q", (3, 1, 1, 1), (1, 1, 1, 1), "int32")
    )
    _set(g, "lse", ragged_offset_tensor_uid=_OFF_Q)
    assert _verdict(g)[0] == "LSE_FORMAT"


# ---------------------------------------------------------------------------
# alignment and features
# ---------------------------------------------------------------------------


def test_minimum_declared_alignment():
    g = _dense_graph()
    assert _lower(g).tensor_alignment == 16
    _set(g, "dv", alignment=8)
    req = _lower(g)
    assert req.tensor_alignment == 8
    code, _, plan = engine_verdict(req, "gfx942")
    assert code == "OK" and plan.specs["main"].stage_vec == 1
    # the stats alignment is not a data-tensor alignment
    g2 = _dense_graph()
    _set(g2, "lse", alignment=4)
    assert _lower(g2).tensor_alignment == 16
    _set(g2, "q", alignment=1)
    assert _verdict(g2)[0] == "STRIDE_ALIGN"
    # hipDNN's default alignment when the key is absent
    g3 = _dense_graph()
    for t in g3["tensors"]:
        t.pop("alignment")
    assert _lower(g3).tensor_alignment == 16


@pytest.mark.parametrize(
    "declared, guaranteed", [(24, 8), (48, 16), (6, 2), (3, 1), (12, 4), (32, 32)]
)
def test_alignment_is_the_largest_power_of_two_divisor(declared, guaranteed):
    # A pointer that is a multiple of 24 is only 8-byte aligned.
    g = _dense_graph()
    for role in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        _set(g, role, alignment=declared)
    req = _lower(g)
    assert req.tensor_alignment == guaranteed
    code, _, plan = engine_verdict(req, "gfx942")
    if guaranteed < 2:
        assert code == "STRIDE_ALIGN"
    else:
        assert code == "OK"
        assert plan.specs["main"].stage_vec == (8 if guaranteed >= 16 else 1)


def test_stats_alignment_is_the_largest_power_of_two_divisor():
    g = _dense_graph()
    _set(g, "lse", alignment=6)
    with pytest.raises(HipdnnLoweringError, match="stats alignment below 4 bytes"):
        _lower(g)
    _set(g, "lse", alignment=12)
    assert _lower(g).tensor_alignment == 16


def _with_tensor(g, group, key, entry):
    g["nodes"][0][group][key] = entry["uid"]
    g["tensors"].append(entry)
    return g


_SCALAR = ((1, 1, 1, 1), (1, 1, 1, 1))


@pytest.mark.parametrize(
    "mutate, code",
    [
        (
            lambda g: g["nodes"][0]["parameters"].update(dropout_probability=0.1),
            "DROPOUT",
        ),
        (
            lambda g: _with_tensor(
                g, "inputs", "seed_tensor_uid", _tensor(30, "seed", *_SCALAR, "int64")
            ),
            "DROPOUT",
        ),
        (
            lambda g: _with_tensor(
                g,
                "inputs",
                "attn_mask_tensor_uid",
                _tensor(30, "bias", (2, 4, 72, 40), (11520, 2880, 40, 1)),
            ),
            "BIAS_IN_BACKWARD",
        ),
        (
            lambda g: _with_tensor(
                g,
                "outputs",
                "dbias_tensor_uid",
                _tensor(30, "dbias", (2, 4, 72, 40), (11520, 2880, 40, 1)),
            ),
            "DBIAS",
        ),
        (lambda g: g["nodes"][0]["parameters"].update(alibi_mask=True), "ALIBI"),
        (lambda g: _set(g, "k", data_type="fp8_e4m3"), "FP8"),
        (
            lambda g: g["nodes"][0].update(compute_data_type="half"),
            "UNSUPPORTED_FEATURE",
        ),
        (
            lambda g: _with_tensor(
                g,
                "inputs",
                "scale_tensor_uid",
                _tensor(30, "scale", *_SCALAR, "float", is_runtime_pass_by_value=True),
            ),
            "SCALE_DEVICE_TENSOR",
        ),
        (
            lambda g: _with_tensor(
                g, "inputs", "scale_tensor_uid", _tensor(30, "scale", *_SCALAR, "float")
            ),
            "SCALE_DEVICE_TENSOR",
        ),
        (lambda g: _set(g, "v", data_type="half"), "DTYPE"),
        (lambda g: _set(g, "dq", data_type="float"), "OUT_DTYPE"),
        (lambda g: _set(g, "q", data_type="float"), "DTYPE"),
        (lambda g: _set(g, "lse", data_type="half"), "LSE_FORMAT"),
        (
            lambda g: g["nodes"][0]["parameters"].update(padding_mask=True),
            "SEQ_LEN_FORMAT",
        ),
    ],
)
def test_feature_mappings(mutate, code):
    g = _dense_graph()
    mutate(g)
    assert _verdict(g)[0] == code


def test_scale_tensor_with_a_stored_value_is_a_baked_scale():
    g = _with_tensor(
        _dense_graph(),
        "inputs",
        "scale_tensor_uid",
        _tensor(
            30, "attn_scale", *_SCALAR, "float", value=0.5, value_type="Float32Value"
        ),
    )
    req = _lower(g)
    assert req.scale == 0.5 and not req.scale_is_device_tensor
    assert _verdict(g)[0] == "OK"


@pytest.mark.parametrize("dtype", ["fp8_e4m3", "int8", "uint8"])
def test_scale_constant_of_a_one_byte_dtype_is_a_lowering_error(dtype):
    g = _with_tensor(
        _dense_graph(),
        "inputs",
        "scale_tensor_uid",
        _tensor(
            30, "attn_scale", *_SCALAR, dtype, value=0x38, value_type="Float8Value"
        ),
    )
    assert _verdict(g)[0] == LOWERING


def test_no_scale_is_hipdnn_unit_scale_and_round_trips():
    g = _dense_graph()
    req = _lower(g)
    assert req.scale == 1.0 and not req.scale_is_device_tensor
    assert _verdict(g)[0] == "OK"
    # rocKE's own default (1/sqrt(d)) is written out explicitly
    rocke_default = AttnBwdRequest(
        b=1, h_q=2, h_k=2, h_v=2, s_q=64, s_kv=64, d_qk=64, d_v=64, dtype="bf16"
    )
    back = to_hipdnn_graph(rocke_default)
    assert back["nodes"][0]["parameters"]["attn_scale_value"] == 0.125
    assert from_hipdnn_graph(back, "gfx942", num_cus=304).scale == 0.125


def test_padding_is_not_declared_by_the_engine():
    g = _dense_graph(padding_mask=True)
    for side, uid in (("q", 30), ("kv", 31)):
        g["nodes"][0]["inputs"][f"seq_len_{side}_tensor_uid"] = uid
        g["tensors"].append(
            _tensor(uid, f"seq_len_{side}", (2, 1, 1, 1), (1, 1, 1, 1), "int32")
        )
    req = _lower(g)
    assert req.padding and req.has_seq_len_q and req.has_seq_len_kv
    code, detail = _verdict(g)
    assert code == "NOT_YET_DECLARED" and "padding" in detail
    # mismatched SEQ_LEN tensors are declined before the policy
    _set(g, 31, data_type="int64")
    assert _verdict(g)[0] == "SEQ_LEN_FORMAT"


def test_engine_arch_and_policy():
    req = _lower(_dense_graph())
    code, detail, plan = engine_verdict(req, "gfx1201")
    assert (code, plan) == ("ARCH_UNSUPPORTED", None) and "gfx942" in detail
    pol = engine_policy()
    assert pol.declared_groups == SHIPPED_DECLARED_GROUPS - {"padding"}
    assert not (pol.perf_gate or pol.dense_pack or pol.deterministic)
    assert pol.tuned_table, "the per-arch tuned tables are joined"


def test_unknown_dtype_is_declined_by_name():
    g = _dense_graph()
    _set(g, "v", data_type="weird")
    code, detail = _verdict(g)
    assert code == "DTYPE" and "weird" in detail


# ---------------------------------------------------------------------------
# inverse and plan record
# ---------------------------------------------------------------------------


def _round_trip_requests():
    thd_mults = {n: 1024 for n in ("q", "o", "do", "dq")}
    thd_mults.update({n: 256 for n in ("k", "v", "dk", "dv")})
    base = {
        "b": 2,
        "h_q": 8,
        "h_k": 2,
        "h_v": 2,
        "s_q": 100,
        "s_kv": 77,
        "d_qk": 64,
        "d_v": 64,
    }
    return {
        "dense": AttnBwdRequest(**base),
        "dense_window_br": AttnBwdRequest(
            **base, left_bound=9, right_bound=0, bottom_right=True, scale=0.3
        ),
        "dense_narrow": AttnBwdRequest(**base, tensor_alignment=4, dtype="bf16"),
        "thd_legacy_multiplier": AttnBwdRequest(**base, layout="thd"),
        "thd_tokens": AttnBwdRequest(
            **dict(base, d_qk=128, d_v=128),
            layout="thd",
            ragged_offset_multipliers=thd_mults,
            ragged_offset_dtype="int64",
        ),
        "thd_hk_ne_hv": AttnBwdRequest(**dict(base, h_k=4), layout="thd"),
        "mqa_causal": AttnBwdRequest(**dict(base, h_k=1, h_v=1), right_bound=0),
    }


@pytest.mark.parametrize("name", sorted(_round_trip_requests()))
@pytest.mark.parametrize("arch", ENGINE_ARCHS)
def test_round_trip_keeps_the_plan(name, arch):
    req = _round_trip_requests()[name]
    plan = attn_bwd_plan(req, arch, policy=engine_policy())
    back = _lower(to_hipdnn_graph(req), arch, num_cus=req.num_cus)
    code, detail, plan2 = engine_verdict(back, arch)
    assert code == "OK", detail
    want, got = plan_record(plan, req, arch), plan_record(plan2, back, arch)
    if req.is_thd and req.ragged_offset_multipliers is None:
        # element offsets of Q are not whole LSE (or V) elements: the graph
        # counts its offsets in tokens, which changes only the multipliers
        units = {"q_mult", "kv_mult", "tok_mult"}
        for rec in (want, got):
            for launch in rec["launches"]:
                launch["kernargs"] = [
                    k for k in launch["kernargs"] if k[0] not in units
                ]
        assert dict(back.ragged_offset_multipliers)["q"] == 8 * 64
    assert got == want
    # the lowered request is the canonical form: it lowers to itself
    again = _lower(to_hipdnn_graph(back), arch, num_cus=back.num_cus)
    assert again == back


def test_inverse_refuses_what_hipdnn_cannot_express():
    base = {
        "b": 1,
        "h_q": 2,
        "h_k": 2,
        "h_v": 2,
        "s_q": 8,
        "s_kv": 8,
        "d_qk": 64,
        "d_v": 64,
    }
    for kw in (
        {"paged": True},
        {"deterministic": True},
        {"layout": "thd", "max_total_q": 8, "max_total_kv": 8},
        {"num_xcds": 8},
        {"causal": True, "right_bound": 0},
        {"causal": True, "causal_bottom_right": True},
        {"right_bound": -2},
        {"other_features": ("sink_token",)},
    ):
        with pytest.raises(ValueError):
            to_hipdnn_graph(AttnBwdRequest(**base, **kw))


def test_plan_record_lists_every_launch_in_abi_order():
    req = _lower(_thd_graph(h_k=4, h_v=2))
    _code, _, plan = engine_verdict(req, "gfx942")
    rec = plan_record(plan, req, "gfx942")
    assert [(x["slot"], x["launch"]) for x in rec["launches"]] == [
        ("prep", ""),
        ("main", "main"),
        ("main", "dq"),
        ("main", "convert_dk"),
        ("main", "convert_dv"),
    ]
    for launch, step in zip(rec["launches"], plan.launches):
        assert [k[:2] for k in launch["kernargs"]] == [
            list(p) for p in attn_bwd_params(step.stage)
        ]
        assert launch["kernel"] == step.kernel
    main = {k[0]: k[2] for k in rec["launches"][1]["kernargs"]}
    assert main["Q"] == "q" and main["OFF_KV"] == "offsets_kv"
    assert main["WS_DQ"] == "ws.dq_acc" and main["dK"] is None
    assert main["q_mult"] == 1024 and main["kv_mult"] == 512
    assert main["scale_log2"].startswith("0x") and len(main["scale_log2"]) == 10
    ws = rec["workspace"]
    assert set(ws["regions"]) == {"dq_acc", "lse2", "dsum", "dk_acc", "dv_acc"}
    assert rec["tokens"]["ws.bytes"] == ws["bytes"] == ws["scratch_offset"]
    for name, (off, _n) in ws["regions"].items():
        assert off % 256 == 0 and rec["tokens"][f"ws.{name}.offset"] == off
    t = rec["tokens"]
    want = {"ws.bytes", "ws.scratch_offset"} | {f"ws.{n}.offset" for n in ws["regions"]}
    want |= {f"attention_general_bwd.prep.{f}" for f in PREP_TOKENS}
    want |= {f"attention_general_bwd.main.{f}" for f in MAIN_TOKENS}
    assert set(t) == want
    assert t["attention_general_bwd.main.seq_mode"] == 1
    assert t["attention_general_bwd.main.dkv_mode"] == 1
    assert t["attention_general_bwd.prep.dtype"] == 2


def test_plan_record_direct_dkv_has_no_kv_converts():
    req = _lower(_dense_graph())
    _code, _, plan = engine_verdict(req, "gfx950")
    rec = plan_record(plan, req, "gfx950")
    assert [x["launch"] for x in rec["launches"]] == ["", "main", "dq"]
    assert "ws.dk_acc.offset" not in rec["tokens"]


def test_tuned_keys_are_numbered_in_string_order():
    for arch in ENGINE_ARCHS:
        keys = tuned_keys(arch)
        assert [i for i, _k, _s in keys] == list(range(1, len(keys) + 1))
        assert [s for _i, _k, s in keys] == sorted(s for _i, _k, s in keys)
        for i, k, _s in keys:
            assert tuning_key_id(arch, k) == i
        assert tuning_key_id(arch, None) == 0


def test_tuned_request_records_its_table_key():
    from dispatch.attention_bwd.tuned_tables import general_key_requests

    arch = "gfx950"
    _i, key, s = tuned_keys(arch)[0]
    req = next(r for r in general_key_requests(arch, key) if not r.padding)
    req = dataclasses.replace(req, num_xcds=None)
    code, detail, plan = engine_verdict(req, arch)
    assert code == "OK", detail
    rec = plan_record(plan, req, arch)
    assert rec["tokens"]["attention_general_bwd.main.tuning_key"] == 1
    assert "|" in s
