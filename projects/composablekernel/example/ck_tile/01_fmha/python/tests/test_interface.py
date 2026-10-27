# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch

from ck_tile_fmha import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from reference import apply_rotary_emb


@pytest.mark.parametrize("api", ["fixed", "varlen", "kvcache"])
def test_softcap_is_rejected(api):
    q = torch.randn(1, 4, 2, 32, dtype=torch.float16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    with pytest.raises(RuntimeError, match="do not support softcap"):
        if api == "fixed":
            flash_attn_func(q, k, v, softcap=1.0)
        elif api == "varlen":
            cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
            flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens, cu_seqlens, 4, 4, softcap=1.0)
        else:
            flash_attn_with_kvcache(q, k, v, softcap=1.0)


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_registered_operator_fake_metadata(varlen, backward):
    # Compare fake metadata with actual CK outputs, including the uint8 dropout
    # buffer and CK-specific variable-length layouts used by torch.compile.
    q = torch.randn(2, 97, 4, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(2, 113, 2, 64, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    sequence_args = ()
    if varlen:
        q, k, v = (x.flatten(0, 1) for x in (q, k, v))
        cu_q = torch.tensor([0, 97, 194], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, 113, 226], device="cuda", dtype=torch.int32)
        sequence_args = (cu_q, cu_k, 97, 113)
    name = "_flash_attn_varlen_" if varlen else "_flash_attn_"
    op = getattr(torch.ops.ck_tile_fmha, name + "forward")
    args = (q, k, v, *sequence_args, 0.17, 0.125, False, -1, -1, 0.0, None, True)
    if backward:
        out, lse, _, rng = op(*args)
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        op = getattr(torch.ops.ck_tile_fmha, name + "backward")
        args = (
            torch.randn_like(out),
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            *sequence_args,
            0.17,
            0.125,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
            rng,
        )
    torch.library.opcheck(op, args, test_utils=("test_faketensor",))


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_reference_batch_offsets(interleaved):
    # Batch zero rotates by pi/2; batch one uses the identity. The last two
    # features are outside rotary_dim and must survive unchanged.
    x = torch.tensor([1, 2, 3, 4, 5, 6], device="cuda", dtype=torch.float16)
    x = x.expand(2, 1, 1, 6).contiguous()
    cos = torch.tensor([[0, 0], [1, 1]], device="cuda", dtype=x.dtype)
    sin = torch.tensor([[1, 1], [0, 0]], device="cuda", dtype=x.dtype)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    actual = apply_rotary_emb(x, cos, sin, interleaved=interleaved, seqlen_offsets=offsets)
    first = [-2, 1, -4, 3, 5, 6] if interleaved else [-3, -4, 1, 2, 5, 6]
    expected = torch.tensor([first, [1, 2, 3, 4, 5, 6]], device="cuda", dtype=x.dtype)
    assert torch.equal(actual[:, 0, 0], expected)
