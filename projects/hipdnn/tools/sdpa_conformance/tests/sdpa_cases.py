# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Token-major packed varlen cases (D10) and their per-sequence reference.

The layout is PyTorch ``varlen_attn`` style: Q/K/V/O/dO are ``[T, H, D]`` token
buffers and stats are ``(T, H)``, so one token row of every tensor is contiguous
and RFC-0014's sequence-outermost span rule holds. The adapter receives them as
``[1, H, T, D]`` / ``[1, H, T, 1]`` views.

Kinds, by padded slots per sequence (``slot = seqlen + pad``):

* ``compact``: no padding; ``pos = cumsum(seqlen)``.
* ``strided``: both sides padded; ``seq_len < slot`` and ``pos = cumsum(slot)``.
* ``seqused``: Q compact, K padded (``seq_len_kv`` used lengths inside larger slots).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from hipdnn_sdpa import Varlen

KINDS = {
    "compact": dict(
        seq_q=[3, 5, 2], seq_kv=[4, 6, 3], pad_q=[0, 0, 0], pad_kv=[0, 0, 0]
    ),
    "strided": dict(
        seq_q=[3, 5, 2], seq_kv=[4, 6, 3], pad_q=[2, 1, 3], pad_kv=[1, 2, 0]
    ),
    "seqused": dict(
        seq_q=[3, 5, 2], seq_kv=[4, 6, 3], pad_q=[0, 0, 0], pad_kv=[2, 0, 3]
    ),
}


def _cumsum(values, device):
    out = [0]
    for v in values:
        out.append(out[-1] + v)
    return torch.tensor(out, dtype=torch.int32, device=device)


def make_packed(kind, device="cpu", dtype=torch.float16, hq=4, hk=2, d=64, seed=0):
    """Token-major tensors, their ``[1, H, T, D]`` views and the matching ``Varlen``."""
    spec = KINDS[kind]
    slot_q = [s + p for s, p in zip(spec["seq_q"], spec["pad_q"])]
    slot_kv = [s + p for s, p in zip(spec["seq_kv"], spec["pad_kv"])]
    pos_q, pos_kv = _cumsum(slot_q, device), _cumsum(slot_kv, device)
    tq, tkv = int(pos_q[-1]), int(pos_kv[-1])
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen).to(dtype).to(device)

    def zeros(*shape, dt=dtype):
        return torch.zeros(*shape, dtype=dt, device=device)

    tok = SimpleNamespace(
        q=rnd(tq, hq, d),
        k=rnd(tkv, hk, d),
        v=rnd(tkv, hk, d),
        do=rnd(tq, hq, d),
        o=zeros(tq, hq, d),
        dq=zeros(tq, hq, d),
        dk=zeros(tkv, hk, d),
        dv=zeros(tkv, hk, d),
        stats=zeros(tq, hq, dt=torch.float32),
    )

    def head_view(t):  # [T, H, D] -> [1, H, T, D]
        return t.permute(1, 0, 2).unsqueeze(0)

    views = SimpleNamespace(
        **{
            n: head_view(getattr(tok, n))
            for n in ("q", "k", "v", "do", "o", "dq", "dk", "dv")
        },
        stats=tok.stats.t().unsqueeze(0).unsqueeze(-1),  # (T, H) -> [1, H, T, 1]
    )
    varlen = Varlen(
        seq_len_q=torch.tensor(spec["seq_q"], dtype=torch.int32, device=device),
        seq_len_kv=torch.tensor(spec["seq_kv"], dtype=torch.int32, device=device),
        pos_q=pos_q,
        pos_kv=pos_kv,
        max_sq=max(slot_q),
        max_sk=max(slot_kv),
    )
    return SimpleNamespace(
        kind=kind,
        hq=hq,
        hk=hk,
        d=d,
        tok=tok,
        views=views,
        varlen=varlen,
        seq_q=spec["seq_q"],
        seq_kv=spec["seq_kv"],
        pos_q=pos_q.tolist(),
        pos_kv=pos_kv.tolist(),
        scale=d**-0.5,
    )


def valid_tokens(pos, seqlens, total):
    """Boolean ``[total]`` mask of tokens inside a sequence (not in slot padding)."""
    mask = torch.zeros(total, dtype=torch.bool)
    for start, n in zip(pos[:-1], seqlens):
        mask[start : start + n] = True
    return mask


def reference(case):
    """Per-sequence fp32 attention: ``o [Tq,Hq,D]``, ``lse [Tq,Hq]`` and the dq/dk/dv of ``sum(do * o)``.

    Positions in slot padding are zero. Key/value head ``h // (Hq / Hk)`` serves query head ``h``.
    """
    t = case.tok
    q, k, v, do = (x.detach().float().cpu() for x in (t.q, t.k, t.v, t.do))
    q.requires_grad_(), k.requires_grad_(), v.requires_grad_()
    group = case.hq // case.hk
    o = torch.zeros_like(q)
    lse = torch.zeros(q.shape[0], case.hq)
    loss = 0.0
    parts = []
    for b, (nq, nk) in enumerate(zip(case.seq_q, case.seq_kv)):
        qs, ks = case.pos_q[b], case.pos_kv[b]
        qb = q[qs : qs + nq].transpose(0, 1)  # [Hq, nq, D]
        kb = k[ks : ks + nk].transpose(0, 1).repeat_interleave(group, dim=0)
        vb = v[ks : ks + nk].transpose(0, 1).repeat_interleave(group, dim=0)
        s = case.scale * qb @ kb.transpose(1, 2)
        ob = torch.softmax(s, dim=-1) @ vb  # [Hq, nq, D]
        parts.append(
            (qs, nq, ob.transpose(0, 1), torch.logsumexp(s, dim=-1).transpose(0, 1))
        )
        loss = loss + (ob.transpose(0, 1) * do[qs : qs + nq]).sum()
    dq, dk, dv = torch.autograd.grad(loss, (q, k, v))
    with torch.no_grad():
        for qs, nq, ob, lb in parts:
            o[qs : qs + nq] = ob
            lse[qs : qs + nq] = lb
    return SimpleNamespace(o=o, lse=lse, dq=dq, dk=dk, dv=dv)
