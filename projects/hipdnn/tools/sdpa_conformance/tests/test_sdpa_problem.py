# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""How hipdnn_sdpa declares tensors to hipDNN (CPU only, no bindings needed).

Covers the RFC-0014 span-rule pre-check, the declared ``[B,H,max_S,D]`` shape and
strides of packed tensors, the ragged-offset values, and the plan-cache key.
"""

import pytest
import torch
from sdpa_cases import KINDS, make_packed

from hipdnn_sdpa import Dropout, EngineDeclined, Mask, Varlen
from hipdnn_sdpa._problem import backward_problem, forward_problem


def fwd(case, views=None, **kw):
    v = views or case.views
    return forward_problem(
        v.q,
        v.k,
        v.v,
        v.o,
        v.stats,
        case.scale,
        kw.get("bias"),
        kw.get("mask"),
        kw.get("dropout"),
        case.varlen,
    )


def bwd(case, views=None, **kw):
    v = views or case.views
    return backward_problem(
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
        kw.get("mask"),
        kw.get("dropout"),
        case.varlen,
    )


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("build", [fwd, bwd])
def test_token_major_passes_span_rule_and_is_declared_padded(kind, build):
    case = make_packed(kind)
    prob = build(case)
    B, hq, hk, d = 3, case.hq, case.hk, case.d
    max_sq, max_sk = case.varlen.max_sq, case.varlen.max_sk

    q = prob.slots["q"]
    assert q.dims == (B, hq, max_sq, d)
    assert q.strides == (max_sq * hq * d, d, hq * d, 1)  # batch, head, token, feature
    k = prob.slots["k"]
    assert k.dims == (B, hk, max_sk, d) and k.strides[2] == hk * d
    stats = prob.slots["stats"]
    assert stats.dims == (B, hq, max_sq, 1)
    assert stats.strides[1:3] == (1, hq)
    assert stats.dtype == torch.float32

    ragged = prob.tensors[q.ragged]
    assert ragged.dtype == torch.int32 and tuple(ragged.shape) == (B + 1, 1, 1, 1)
    assert ragged.flatten().tolist() == [p * hq * d for p in case.pos_q]
    kv_ragged = prob.tensors[k.ragged]
    assert kv_ragged.flatten().tolist() == [p * hk * d for p in case.pos_kv]
    assert prob.tensors[stats.ragged].flatten().tolist() == [p * hq for p in case.pos_q]

    # every Q-side tensor of one token stride shares one offset tensor
    assert prob.slots["o"].ragged == q.ragged
    for role in ("seq_len_q", "seq_len_kv"):
        s = prob.slots[role]
        assert s.dims == (B, 1, 1, 1) and s.dtype == torch.int32 and s.ragged is None
    assert prob.padding and prob.packed_q and prob.packed_kv


def test_seq_len_follows_the_unpadded_lengths():
    case = make_packed("strided")
    prob = fwd(case)
    assert prob.tensors["seq_len_q"].flatten().tolist() == case.seq_q
    assert prob.tensors["seq_len_kv"].flatten().tolist() == case.seq_kv
    # the declared sequence extent is the largest slot (length + padding), not the longest length
    assert prob.slots["q"].dims[2] == max(
        s + p for s, p in zip(KINDS["strided"]["seq_q"], KINDS["strided"]["pad_q"])
    )


@pytest.mark.parametrize("build,direction", [(fwd, "fwd"), (bwd, "bwd")])
@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
def test_head_major_tensor_is_declined_before_any_graph(build, direction, role):
    case = make_packed("compact")
    views = case.views
    side_len = case.tok.q.shape[0] if role in ("q", "o") else case.tok.k.shape[0]
    heads = case.hq if role in ("q", "o") else case.hk
    head_major = torch.zeros(
        1, heads, side_len, case.d, dtype=torch.float16
    )  # [1,H,T,D] contiguous
    views = type(views)(**{**vars(views), role: head_major})
    with pytest.raises(EngineDeclined) as exc:
        build(case, views)
    assert exc.value.direction == direction
    assert (
        exc.value.reason
        == f"adapter: packed layout not sequence-outermost (RFC-0014): {role}"
    )


def test_head_major_do_and_heads_major_stats_are_declined():
    case = make_packed("compact")
    views = case.views
    do = torch.zeros(1, case.hq, case.tok.q.shape[0], case.d, dtype=torch.float16)
    with pytest.raises(EngineDeclined) as exc:
        bwd(case, type(views)(**{**vars(views), "do": do}))
    assert exc.value.reason.endswith(": do")

    ht_stats = (
        torch.zeros(case.hq, case.tok.q.shape[0]).unsqueeze(0).unsqueeze(-1)
    )  # (H, T)
    for build in (fwd, bwd):
        with pytest.raises(EngineDeclined) as exc:
            build(case, type(views)(**{**vars(views), "stats": ht_stats}))
        assert exc.value.reason.endswith(": stats")


def test_single_head_head_major_passes_the_span_rule():
    # With H = 1 a token row is D contiguous elements: head-major and token-major coincide.
    case = make_packed("compact", hq=1, hk=1)
    head_major = torch.zeros(1, 1, case.tok.q.shape[0], case.d, dtype=torch.float16)
    views = type(case.views)(**{**vars(case.views), "q": head_major})
    assert fwd(case, views).slots["q"].strides[2] == case.d


def test_padded_varlen_and_dense_are_declared_as_given():
    B, H, S, D = 2, 4, 16, 32
    q = torch.zeros(B, H, S, D, dtype=torch.float16)
    k = torch.zeros(B, 2, S, D, dtype=torch.float16)
    o = torch.zeros_like(q)
    stats = torch.zeros(B, H, S, 1)
    seq = torch.tensor([16, 9], dtype=torch.int32)
    varlen = Varlen(seq, seq, None, None, S, S)
    prob = forward_problem(q, k, k, o, stats, 0.25, None, None, None, varlen)
    assert (
        prob.slots["q"].dims == (B, H, S, D) and prob.slots["q"].strides == q.stride()
    )
    assert not prob.packed_q and prob.padding
    assert all(s.ragged is None for s in prob.slots.values())
    assert prob.slots["seq_len_q"].dims == (B, 1, 1, 1)

    dense = forward_problem(q, k, k, o, stats, 0.25, None, None, None, None)
    assert not dense.padding and "seq_len_q" not in dense.slots


def test_outputs_and_roles():
    case = make_packed("compact")
    assert fwd(case).outputs == ("o", "stats")
    assert bwd(case).outputs == ("dq", "dk", "dv")
    q = torch.zeros(1, 2, 4, 8)
    dbias = torch.zeros(1, 2, 4, 4)
    prob = backward_problem(
        q,
        q,
        q,
        q,
        q,
        torch.zeros(1, 2, 4, 1),
        q,
        q,
        q,
        1.0,
        dbias,
        torch.zeros(1, 2, 4, 4),
        None,
        None,
        None,
    )
    assert prob.outputs == ("dq", "dk", "dv", "dbias")
    with pytest.raises(ValueError):
        backward_problem(
            q,
            q,
            q,
            q,
            q,
            torch.zeros(1, 2, 4, 1),
            q,
            q,
            q,
            1.0,
            dbias,
            None,
            None,
            None,
            None,
        )


def test_bias_with_packed_varlen_is_a_contract_violation():
    case = make_packed("compact")
    with pytest.raises(ValueError):
        fwd(case, bias=torch.zeros(1, 1, 4, 4))


def _dense(k=None, scale=0.25, bias=None, mask=None, dropout=None):
    q = torch.zeros(2, 4, 16, 32, dtype=torch.float16)
    if k is None:
        k = torch.zeros(2, 2, 16, 32, dtype=torch.float16)
    return forward_problem(
        q,
        k,
        k,
        torch.zeros_like(q),
        torch.zeros(2, 4, 16, 1),
        scale,
        bias,
        mask,
        dropout,
        None,
    )


def test_plan_cache_key_covers_every_graph_input():
    base = _dense().key
    seed = torch.zeros(1, dtype=torch.int64)
    variants = {
        "scale": _dense(scale=0.5),
        "mask tl": _dense(mask=Mask("tl", None, 0)),
        "mask br": _dense(mask=Mask("br", None, 0)),
        "window": _dense(mask=Mask("tl", 3, 0)),
        "bias": _dense(bias=torch.zeros(2, 4, 16, 16)),
        "dropout": _dense(dropout=Dropout(0.2, seed, seed)),
        "dtype": _dense(k=torch.zeros(2, 2, 16, 32, dtype=torch.bfloat16)),
        "stride": _dense(
            k=torch.zeros(2, 16, 2, 32, dtype=torch.float16).transpose(1, 2)
        ),
        "shape": _dense(k=torch.zeros(2, 2, 32, 32, dtype=torch.float16)),
    }
    keys = {name: p.key for name, p in variants.items()}
    for name, key in keys.items():
        assert key != base, name
    assert len(set(keys.values())) == len(keys)
    assert hash(base) == hash(_dense().key) and base == _dense().key


def test_plan_cache_key_ignores_data_and_dropout_seed_values():
    seed = torch.zeros(1, dtype=torch.int64)
    other = torch.full((1,), 7, dtype=torch.int64)
    a = _dense(dropout=Dropout(0.2, seed, seed)).key
    b = _dense(dropout=Dropout(0.2, other, other)).key
    assert a == b

    c1, c2 = make_packed("strided", seed=1), make_packed("strided", seed=2)
    assert fwd(c1).key == fwd(c2).key
    assert fwd(make_packed("compact")).key != fwd(make_packed("strided")).key


def test_zero_dropout_is_no_dropout():
    seed = torch.zeros(1, dtype=torch.int64)
    prob = _dense(dropout=Dropout(0.0, seed, seed))
    assert prob.dropout_p is None and "seed" not in prob.slots


@pytest.mark.parametrize("entry", ["fwd", "bwd"])
def test_decline_happens_before_hipdnn_is_touched(monkeypatch, entry):
    import hipdnn_sdpa
    from hipdnn_sdpa import _graph

    def boom():
        raise AssertionError("hipdnn_frontend touched before the span-rule decline")

    monkeypatch.setattr(_graph, "frontend", boom)
    case = make_packed("compact")
    v = case.views
    head_major = torch.zeros(
        1, case.hq, case.tok.q.shape[0], case.d, dtype=torch.float16
    )
    with pytest.raises(EngineDeclined) as exc:
        if entry == "fwd":
            hipdnn_sdpa.sdpa_fwd(
                head_major,
                v.k,
                v.v,
                o=v.o,
                stats=v.stats,
                scale=1.0,
                varlen=case.varlen,
            )
        else:
            hipdnn_sdpa.sdpa_bwd(
                head_major,
                v.k,
                v.v,
                v.o,
                v.do,
                v.stats,
                dq=v.dq,
                dk=v.dk,
                dv=v.dv,
                scale=1.0,
                varlen=case.varlen,
            )
    assert exc.value.direction == entry
