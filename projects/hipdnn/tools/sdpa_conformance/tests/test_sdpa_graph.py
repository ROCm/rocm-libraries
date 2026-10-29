# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The hipDNN graph hipdnn_sdpa builds, asserted through ``Graph.to_json()``.

No handle and no device: tensors are CPU torch tensors, and serialization lowers
the graph without a backend handle (``graph_bindings.cpp:785``). Needs the built
``hipdnn_frontend`` bindings (SDPA enabled, ``Tensor.set_ragged_offset``), and
skips cleanly where they are absent.
"""

import json

import pytest
import torch
from sdpa_cases import KINDS, make_packed

hipdnn = pytest.importorskip("hipdnn_frontend")
if not hasattr(hipdnn.Graph, "sdpa"):
    pytest.skip("hipdnn_frontend built without SDPA", allow_module_level=True)

from hipdnn_sdpa import Dropout, Mask, Varlen  # noqa: E402
from hipdnn_sdpa import _graph  # noqa: E402
from hipdnn_sdpa._problem import backward_problem, forward_problem  # noqa: E402

JSON_DTYPE = {
    torch.float16: "half",
    torch.bfloat16: "bfloat16",
    torch.float32: "float",
    torch.int32: "int32",
    torch.int64: "int64",
}


class Serialized:
    def __init__(self, prob):
        self.prob = prob
        graph, self.uids = _graph.build(prob)
        doc = json.loads(graph.to_json())
        assert len(doc["nodes"]) == 1
        self.doc = doc
        self.node = doc["nodes"][0]
        self.tensors = {t["uid"]: t for t in doc["tensors"]}
        self.inputs = self.node["inputs"]
        self.outputs = self.node["outputs"]
        self.attrs = self.node["attributes"]

    def tensor(self, role):
        return self.tensors[self.uids[role]]

    def uid_of(self, section, key):
        return (self.inputs if section == "in" else self.outputs).get(key)

    def role_tensor(self, section, key):
        return self.tensors[self.uid_of(section, key)]


def dense_inputs(hq=4, hk=2, sq=16, sk=24, d=32, dtype=torch.float16):
    """BSHD-permuted views (non-contiguous, as a packed QKV buffer would give)."""
    b = 2

    def bshd(h, s):
        return torch.zeros(b, s, h, d, dtype=dtype).permute(0, 2, 1, 3)

    return dict(
        q=bshd(hq, sq),
        k=bshd(hk, sk),
        v=bshd(hk, sk),
        o=bshd(hq, sq),
        do=bshd(hq, sq),
        dq=bshd(hq, sq),
        dk=bshd(hk, sk),
        dv=bshd(hk, sk),
        stats=torch.zeros(b, hq, sq, 1, dtype=torch.float32),
    )


def fwd(t, scale=0.125, bias=None, mask=None, dropout=None, varlen=None):
    return Serialized(
        forward_problem(
            t["q"],
            t["k"],
            t["v"],
            t["o"],
            t["stats"],
            scale,
            bias,
            mask,
            dropout,
            varlen,
        )
    )


def bwd(t, scale=0.125, bias=None, dbias=None, mask=None, dropout=None, varlen=None):
    return Serialized(
        backward_problem(
            t["q"],
            t["k"],
            t["v"],
            t["o"],
            t["do"],
            t["stats"],
            t["dq"],
            t["dk"],
            t["dv"],
            scale,
            dbias,
            bias,
            mask,
            dropout,
            varlen,
        )
    )


BUILDERS = [
    pytest.param(fwd, "SdpaAttributes", id="fwd"),
    pytest.param(bwd, "SdpaBackwardAttributes", id="bwd"),
]


def check_tensor(s, role):
    """The serialized tensor of ``role`` carries the slot's dims, strides and dtype."""
    slot = s.prob.slots[role]
    t = s.tensor(role)
    assert t["name"] == role
    assert t["dims"] == list(slot.dims) and t["strides"] == list(slot.strides), role
    assert t["data_type"] == JSON_DTYPE[slot.dtype], role
    assert t["is_runtime_pass_by_value"] is False, role
    if slot.ragged is None:
        assert "ragged_offset_tensor_uid" not in t, role
    else:
        assert t["ragged_offset_tensor_uid"] == s.uids[slot.ragged], role


@pytest.mark.parametrize("build,node_type", BUILDERS)
def test_dense_tensors_keep_their_real_strides(build, node_type):
    s = build(dense_inputs())
    assert s.node["type"] == node_type
    roles = (
        ("q", "k", "v", "o", "stats")
        if build is fwd
        else ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")
    )
    for role in roles:
        check_tensor(s, role)
    # GQA: K carries its own (smaller) head count
    assert s.tensor("q")["dims"] == [2, 4, 16, 32] and s.tensor("k")["dims"] == [
        2,
        2,
        24,
        32,
    ]
    assert s.tensor("q")["strides"] == [16 * 4 * 32, 32, 4 * 32, 1]
    assert s.tensor("stats")["data_type"] == "float"


def test_forward_wires_roles_and_generates_stats():
    s = fwd(dense_inputs())
    for key, role in (
        ("q_tensor_uid", "q"),
        ("k_tensor_uid", "k"),
        ("v_tensor_uid", "v"),
    ):
        assert s.role_tensor("in", key)["name"] == role
    assert s.role_tensor("out", "o_tensor_uid")["name"] == "o"
    assert s.role_tensor("out", "stats_tensor_uid")["name"] == "stats"
    assert s.attrs["generate_stats"] is True
    assert s.attrs["attn_scale_value"] == pytest.approx(0.125)
    assert s.doc["io_data_type"] == "half"
    assert s.doc["compute_data_type"] == "float"
    assert s.tensor("o")["virtual"] is False and s.tensor("stats")["virtual"] is False


def test_backward_wires_roles():
    s = bwd(dense_inputs())
    names = {"q": "q", "k": "k", "v": "v", "o": "o", "do": "do", "stats": "stats"}
    for role, name in names.items():
        assert s.role_tensor("in", f"{role}_tensor_uid")["name"] == name
    for role in ("dq", "dk", "dv"):
        assert s.role_tensor("out", f"{role}_tensor_uid")["name"] == role
        assert s.role_tensor("out", f"{role}_tensor_uid")["virtual"] is False
    assert s.attrs["attn_scale_value"] == pytest.approx(0.125)


def test_no_optional_features_by_default():
    for s in (fwd(dense_inputs()), bwd(dense_inputs())):
        assert s.attrs["padding_mask"] is False
        assert s.attrs.get("dropout_probability") is None
        assert s.attrs.get("left_bound") is None and s.attrs.get("right_bound") is None
        for key in ("attn_mask_tensor_uid", "seq_len_q_tensor_uid", "seed_tensor_uid"):
            assert s.uid_of("in", key) is None


@pytest.mark.parametrize("build,node_type", BUILDERS)
@pytest.mark.parametrize(
    "mask,alignment,left,right",
    [
        (Mask("tl", None, 0), "TOP_LEFT", None, 0),  # causal, top-left
        (Mask("br", None, 0), "BOTTOM_RIGHT", None, 0),  # causal, bottom-right
        (
            Mask("tl", 5, 0),
            "TOP_LEFT",
            5,
            0,
        ),  # sliding window, (l, r) = (4, 0) -> left 5
        (Mask("tl", 3, 2), "TOP_LEFT", 3, 2),  # two-sided window
        (Mask("br", 6, 1), "BOTTOM_RIGHT", 6, 1),
    ],
)
def test_mask_alignment_and_band_bounds(build, node_type, mask, alignment, left, right):
    s = build(dense_inputs(), mask=mask)
    assert s.attrs["diagonal_alignment"] == alignment
    assert s.attrs.get("left_bound") == left
    assert s.attrs.get("right_bound") == right
    # the deprecated causal flags stay unused
    assert (
        s.attrs["causal_mask"] is False and s.attrs["causal_mask_bottom_right"] is False
    )


@pytest.mark.parametrize("build,node_type", BUILDERS)
def test_scale_value(build, node_type):
    assert build(dense_inputs(), scale=0.0625).attrs[
        "attn_scale_value"
    ] == pytest.approx(0.0625)


@pytest.mark.parametrize("build,node_type", BUILDERS)
def test_seeded_dropout_on_both_directions(build, node_type):
    seed = torch.tensor([1234], dtype=torch.int64)
    offset = torch.tensor([99], dtype=torch.int64)
    s = build(dense_inputs(), dropout=Dropout(0.25, seed, offset))
    assert s.attrs["dropout_probability"] == pytest.approx(0.25)
    for key, role in (("seed_tensor_uid", "seed"), ("offset_tensor_uid", "offset")):
        t = s.role_tensor("in", key)
        assert t["name"] == role and t["data_type"] == "int64" and t["dims"] == [1]
        assert t["is_runtime_pass_by_value"] is False
    assert s.uid_of("in", "dropout_mask_tensor_uid") is None


@pytest.mark.parametrize("build,node_type", BUILDERS)
def test_bias_is_a_real_strided_input(build, node_type):
    bias = torch.zeros(2, 1, 16, 24, dtype=torch.float16).expand(2, 4, 16, 24)
    s = build(dense_inputs(), bias=bias)
    t = s.role_tensor("in", "attn_mask_tensor_uid")
    assert t["name"] == "bias" and t["dims"] == [2, 4, 16, 24]
    assert t["strides"] == list(bias.stride()) and t["data_type"] == "half"


def test_dbias_is_a_backward_output():
    bias = torch.zeros(2, 4, 16, 24, dtype=torch.float16)
    dbias = torch.zeros(2, 4, 16, 24, dtype=torch.float16)
    s = bwd(dense_inputs(), bias=bias, dbias=dbias)
    t = s.role_tensor("out", "dbias_tensor_uid")
    assert (
        t["name"] == "dbias" and t["dims"] == [2, 4, 16, 24] and t["virtual"] is False
    )
    assert s.role_tensor("in", "attn_mask_tensor_uid")["name"] == "bias"
    assert bwd(dense_inputs(), bias=bias).uid_of("out", "dbias_tensor_uid") is None


@pytest.mark.parametrize("build,node_type", BUILDERS)
def test_padded_varlen_sets_padding_mask_and_seq_len(build, node_type):
    t = dense_inputs()
    seq_q = torch.tensor([16, 9], dtype=torch.int32)
    seq_kv = torch.tensor([24, 11], dtype=torch.int32)
    s = build(t, varlen=Varlen(seq_q, seq_kv, None, None, 16, 24))
    assert s.attrs["padding_mask"] is True
    for key, role in (
        ("seq_len_q_tensor_uid", "seq_len_q"),
        ("seq_len_kv_tensor_uid", "seq_len_kv"),
    ):
        st = s.role_tensor("in", key)
        assert st["name"] == role and st["data_type"] == "int32"
        assert st["dims"] == [2, 1, 1, 1] and st["strides"] == [1, 1, 1, 1]
        assert "ragged_offset_tensor_uid" not in st
    # padded BHSD stays BHSD: no ragged offsets anywhere
    assert not any("ragged_offset_tensor_uid" in t for t in s.doc["tensors"])
    assert s.tensor("q")["dims"] == [2, 4, 16, 32]


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("build,node_type", BUILDERS)
def test_token_major_packed_varlen_serializes_as_the_contract_says(
    kind, build, node_type
):
    case = make_packed(kind)
    v = case.views
    if build is fwd:
        prob = forward_problem(
            v.q, v.k, v.v, v.o, v.stats, case.scale, None, None, None, case.varlen
        )
        roles = ("q", "k", "v", "o", "stats")
    else:
        prob = backward_problem(
            v.q,
            v.k,
            v.v,
            v.o,
            v.do,
            v.stats,
            v.dq,
            v.dk,
            v.dv,
            case.scale,
            None,
            None,
            None,
            None,
            case.varlen,
        )
        roles = ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")
    s = Serialized(prob)
    assert s.node["type"] == node_type
    B, hq, hk, d = 3, case.hq, case.hk, case.d
    sq, sk = case.varlen.max_sq, case.varlen.max_sk

    for role in roles:
        check_tensor(s, role)
        assert "ragged_offset_tensor_uid" in s.tensor(role), role
    # [B, H, max_S, D] with the token-major head / token / feature strides
    assert s.tensor("q")["dims"] == [B, hq, sq, d]
    assert s.tensor("q")["strides"][1:] == [d, hq * d, 1]
    assert s.tensor("k")["dims"] == [B, hk, sk, d]
    assert s.tensor("k")["strides"][1:] == [d, hk * d, 1]
    assert (
        s.tensor("stats")["dims"] == [B, hq, sq, 1]
        and s.tensor("stats")["data_type"] == "float"
    )
    assert s.tensor("stats")["strides"][1:3] == [1, hq]

    # ragged offsets: [B+1,1,1,1] INT32, shared per (side, token stride)
    for role in roles:
        off = s.tensors[s.tensor(role)["ragged_offset_tensor_uid"]]
        assert off["dims"] == [B + 1, 1, 1, 1] and off["data_type"] == "int32", role
    q_off = s.tensors[s.tensor("q")["ragged_offset_tensor_uid"]]
    assert (
        s.tensor("o")["ragged_offset_tensor_uid"]
        == s.tensor("q")["ragged_offset_tensor_uid"]
    )
    assert q_off["name"].startswith("ragged_offset_q_")

    # seq_len tensors: padding_mask, [B,1,1,1] INT32, no ragged offset
    assert s.attrs["padding_mask"] is True
    for key in ("seq_len_q_tensor_uid", "seq_len_kv_tensor_uid"):
        st = s.role_tensor("in", key)
        assert st["dims"] == [B, 1, 1, 1] and st["data_type"] == "int32"
        assert "ragged_offset_tensor_uid" not in st
    if build is fwd:
        assert s.attrs["generate_stats"] is True


def test_packed_graph_with_mask_and_dropout():
    case = make_packed("strided")
    v = case.views
    seed = torch.tensor([5], dtype=torch.int64)
    prob = forward_problem(
        v.q,
        v.k,
        v.v,
        v.o,
        v.stats,
        case.scale,
        None,
        Mask("br", None, 0),
        Dropout(0.1, seed, seed),
        case.varlen,
    )
    s = Serialized(prob)
    assert (
        s.attrs["diagonal_alignment"] == "BOTTOM_RIGHT"
        and s.attrs["padding_mask"] is True
    )
    assert s.attrs["dropout_probability"] == pytest.approx(0.1)
    assert "ragged_offset_tensor_uid" not in s.role_tensor("in", "seed_tensor_uid")
