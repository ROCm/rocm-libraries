# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The per-sequence reference the packed GPU cases compare against (CPU only)."""

import pytest
import torch
import torch.nn.functional as F
from sdpa_cases import KINDS, make_packed, reference, valid_tokens


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_reference_matches_torch_sdpa_per_sequence(kind):
    case = make_packed(kind)
    ref = reference(case)
    t = case.tok
    group = case.hq // case.hk
    q, k, v, do = (x.float().requires_grad_() for x in (t.q, t.k, t.v, t.do))
    for b, (nq, nk) in enumerate(zip(case.seq_q, case.seq_kv)):
        qs, ks = case.pos_q[b], case.pos_kv[b]
        qb = q[qs : qs + nq].transpose(0, 1)
        kb = k[ks : ks + nk].transpose(0, 1).repeat_interleave(group, dim=0)
        vb = v[ks : ks + nk].transpose(0, 1).repeat_interleave(group, dim=0)
        ob = F.scaled_dot_product_attention(qb, kb, vb, scale=case.scale).transpose(
            0, 1
        )
        torch.testing.assert_close(ob, ref.o[qs : qs + nq], atol=1e-5, rtol=1e-5)

    vq = valid_tokens(case.pos_q, case.seq_q, t.q.shape[0])
    assert not ref.o[~vq].any() and not ref.lse[~vq].any()
    vk = valid_tokens(case.pos_kv, case.seq_kv, t.k.shape[0])
    assert not ref.dk[~vk].any() and not ref.dv[~vk].any()
    assert ref.dq[vq].abs().sum() > 0 and ref.dk[vk].abs().sum() > 0
