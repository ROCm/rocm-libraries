# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Token-major packed varlen cases (D10) executed on hipDNN (``gpu`` marker).

``[T, H, D]`` Q/K/V/O/dO and ``(T, H)`` stats satisfy RFC-0014's span rule, so the
adapter builds a packed graph. A case passes only when hipDNN computed the
direction and the result matches the per-sequence reference; a decline is recorded
as a ``declined:`` skip with hipDNN's own message. Today no engine executes ragged
offsets, so a decline is the expected outcome.
"""

import pytest
import torch
from sdpa_cases import KINDS, make_packed, reference, valid_tokens

import hipdnn_sdpa
from hipdnn_sdpa import EngineDeclined

pytestmark = pytest.mark.gpu

# fp16 inputs against an fp32 reference
TOL = dict(atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_token_major_forward(kind, cuda_device):
    case = make_packed(kind, device=cuda_device)
    v = case.views
    try:
        hipdnn_sdpa.sdpa_fwd(
            v.q, v.k, v.v, o=v.o, stats=v.stats, scale=case.scale, varlen=case.varlen
        )
    except EngineDeclined as declined:
        pytest.skip(f"declined: fwd {declined.reason}")
    torch.cuda.synchronize()

    ref = reference(case)
    valid = valid_tokens(case.pos_q, case.seq_q, case.tok.q.shape[0])
    torch.testing.assert_close(case.tok.o.float().cpu()[valid], ref.o[valid], **TOL)
    torch.testing.assert_close(
        case.tok.stats.cpu()[valid], ref.lse[valid], atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_token_major_backward(kind, cuda_device):
    case = make_packed(kind, device=cuda_device)
    ref = reference(case)
    case.tok.o.copy_(ref.o)
    case.tok.stats.copy_(ref.lse)
    v = case.views
    try:
        hipdnn_sdpa.sdpa_bwd(
            v.q,
            v.k,
            v.v,
            v.o,
            v.do,
            v.stats,
            dq=v.dq,
            dk=v.dk,
            dv=v.dv,
            scale=case.scale,
            varlen=case.varlen,
        )
    except EngineDeclined as declined:
        pytest.skip(f"declined: bwd {declined.reason}")
    torch.cuda.synchronize()

    valid_q = valid_tokens(case.pos_q, case.seq_q, case.tok.q.shape[0])
    valid_kv = valid_tokens(case.pos_kv, case.seq_kv, case.tok.k.shape[0])
    torch.testing.assert_close(
        case.tok.dq.float().cpu()[valid_q], ref.dq[valid_q], **TOL
    )
    torch.testing.assert_close(
        case.tok.dk.float().cpu()[valid_kv], ref.dk[valid_kv], **TOL
    )
    torch.testing.assert_close(
        case.tok.dv.float().cpu()[valid_kv], ref.dv[valid_kv], **TOL
    )
