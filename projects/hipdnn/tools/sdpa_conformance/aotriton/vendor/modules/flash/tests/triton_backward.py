#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import math
import pytest
import torch

from triton_attn_torch_function import attention

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    """
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    """
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    SPARSE_HEAD_SINCE = 5
    SPARSE_SEQ_SINCE = 5
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        if dropout_mask is not None:
            attn_weight.masked_fill_(dropout_mask.logical_not(), float("0.0"))
            value = value / (1 - dropout_p)
        else:
            # assert False, "TESTING dropout_mask code path"
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    else:
        # assert False, "TESTING dropout_mask code path"
        pass
    av = attn_weight @ value
    return av, attn_weight

def _make_block_eyes(q, base=1.0, inc=0.0):
    dhead = q.shape[-1]
    seqlen = q.shape[2]
    assert seqlen % dhead == 0
    scale = base
    for i in range(0, seqlen, dhead):
        q[:, :, i:i+dhead, :] = torch.eye(dhead, device=q.device, dtype=q.dtype) * scale
        scale += inc

'''
Flash Attention is batch operator that evaluates sm(QK')V
Q = batch_size x ... x seqlen_q x head_size
K = batch_size x ... x seqlen_k x head_size
    => K' = batch_size x ... x head_size x seqlen_k
V = batch_size x ... x seqlen_k x head_size
sm(.) = softmax(.)
The output size is
batch_size x ... x seqlen_q x head_size

Note: In Flash V2 API the ... is denoted as "num_heads", serving as uniformly sized sequences
but in PyTorch API it does not present at all
'''

# @pytest.mark.parametrize('BATCH', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('BATCH', [1, 2, 4])
@pytest.mark.parametrize('N_HEADS', [1, 2, 4])
@pytest.mark.parametrize('D_HEAD', [16,32,64,128])
# @pytest.mark.parametrize('D_HEAD', [128])
# @pytest.mark.parametrize('seqlen_q', [16,32,64,128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [16,32,64,128,256,512,1024])
@pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
@pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_q', [32, 128])
# @pytest.mark.parametrize('seqlen_k', [32, 128])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
@pytest.mark.parametrize('storage_flip', [True, False])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
# FIXME: GPU Segfault on 0.0-dtype0-0.0-True-128-256-16-4-1
#        Also all causal=False UTs passed
def test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    SKIP_DQ = False
    torch.manual_seed(20)
    SPARSE_HEAD_SINCE = 1
    SPARSE_SEQ_SINCE = 1
    qdims = (BATCH, N_HEADS, seqlen_q, D_HEAD)
    kdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
    vdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
    if storage_flip:
        qdims = (qdims[0], qdims[2], qdims[1], qdims[3])
        kdims = (kdims[0], kdims[2], kdims[1], kdims[3])
        vdims = (vdims[0], vdims[2], vdims[1], vdims[3])
    q = torch.empty(qdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    k = torch.empty(kdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    v = torch.empty(vdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    if storage_flip:
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
    if not SKIP_DQ:
        q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    autotune = True
    # # triton implementation
    tri_out, encoded_softmax, _ = attention(q, k, v, causal, sm_scale, dropout_p, True, autotune)
    dropout_mask = encoded_softmax >= 0
    ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v,
                                                                dropout_p=dropout_p,
                                                                is_causal=causal,
                                                                scale=sm_scale,
                                                                dropout_mask=dropout_mask)
    dout = torch.randn_like(q)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = None if SKIP_DQ else q.grad.clone(), None
    ref_out.backward(dout, None)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = None if SKIP_DQ else q.grad.clone(), None
    # compare
    if dtype==torch.bfloat16:
        ATOL = 1e-1 * (seqlen_q / 64.0) if seqlen_q >= 16 else 1e-1
    else:
        ATOL = 1e-2 * (seqlen_q / 64.0) if seqlen_q >= 16 else 1e-2
    # RTOL=1e-2 if dtype==torch.float16 else 5e-2
    RTOL=0.0
    print(f'Forward Using ATOL={ATOL} RTOL={RTOL}')
    # FIXME: Need to raise tolerance
    is_allclose = torch.allclose(ref_out, tri_out, atol=ATOL, rtol=RTOL)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
    assert is_allclose
    if dtype == torch.bfloat16:
        ATOL = 1e-1 * ((seqlen_q + D_HEAD) / 32.0)
    if dtype == torch.float32:
        ATOL = 1e-3 * ((seqlen_q + D_HEAD) / 32.0)
    else:
        ATOL = 1e-1 * ((seqlen_q + D_HEAD) / 32.0)
    print(f"Backward Using {ATOL=} {RTOL=}")

    dv_allclose = torch.allclose(ref_dv, tri_dv, atol=ATOL, rtol=RTOL)
    if not dv_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_dv - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=} {q.dtype=}')
        print(f'{k.shape=} {k.stride()=} {k.dtype=}')
        print(f'{v.shape=} {v.stride()=} {v.dtype=}')
        print(f'{q[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{k[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{v[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        # print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{dropout_mask.shape=}')
        print(f'{err_idx=}')
        print(f'{tri_dv[err_idx]=}')
        print(f'{ref_dv[err_idx]=}')
        print(f'{tri_dv[err_idx[:2]]=}')
        print(f'{ref_dv[err_idx[:2]]=}')
        if seqlen_q < 16:
            print(f'{tri_dk[0,0]=}')
            print(f'{ref_dk[0,0]=}')
            print(f'{tri_dv[0,0]=}')
            print(f'{ref_dv[0,0]=}')
            # print(f'{tri_dq[0,0]=}')
            # print(f'{ref_dq[0,0]=}')

    dk_allclose = torch.allclose(ref_dk, tri_dk, atol=ATOL, rtol=RTOL)
    if dv_allclose and not dk_allclose:
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_dk - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')

    dq_allclose = SKIP_DQ or torch.allclose(ref_dq, tri_dq, atol=ATOL, rtol=RTOL)
    if dk_allclose and dv_allclose and not dq_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_dq - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')
    assert dk_allclose and dv_allclose and dq_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=}'
