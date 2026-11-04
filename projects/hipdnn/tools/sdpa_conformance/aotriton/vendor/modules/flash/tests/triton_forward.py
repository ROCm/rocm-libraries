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
    print(f'{query=}')
    print(f'{key=}')
    print(f'BEFORE softmax {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    print(f'BEFORE DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    if dropout_p > 0.0:
        if dropout_mask is not None:
            print(f'BEFORE DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            attn_weight.masked_fill_(dropout_mask.logical_not(), float("0.0"))
            print(f'AFTER DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            value = value / (1 - dropout_p)
        else:
            # assert False, "TESTING dropout_mask code path"
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    else:
        # assert False, "TESTING dropout_mask code path"
        pass
    print(f'{value[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    av = attn_weight @ value
    print(f'{av[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    return av, attn_weight

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
# @pytest.mark.parametrize('seqlen_q', [16, 128])
# @pytest.mark.parametrize('seqlen_k', [16, 128])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
@pytest.mark.parametrize('storage_flip', [True, False])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    torch.manual_seed(20)
    print(f"test_op_fwd {BATCH=}, {N_HEADS=}, {seqlen_q=}, {seqlen_k=}, {D_HEAD=}, {causal=}")
    SPARSE_HEAD_SINCE = 3
    SPARSE_SEQ_SINCE = 3
    Z = BATCH
    H = N_HEADS
    if True: # Real UT
        qdims = (BATCH, N_HEADS, seqlen_q, D_HEAD)
        kdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
        vdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
        if storage_flip:
            qdims = (qdims[0], qdims[2], qdims[1], qdims[3])
            kdims = (kdims[0], kdims[2], kdims[1], kdims[3])
            vdims = (vdims[0], vdims[2], vdims[1], vdims[3])
        q = (
            torch.empty(qdims, dtype=dtype, device="cuda")
            .normal_(mean=0., std=0.5)
            .requires_grad_()
        )
        k = (
            torch.empty(kdims, dtype=dtype, device="cuda")
            .normal_(mean=0., std=0.5)
            .requires_grad_()
        )
        v = (
            torch.empty(vdims, dtype=dtype, device="cuda")
            .normal_(mean=0., std=0.5)
            .requires_grad_()
        )
        if storage_flip:
            q = torch.transpose(q, 1, 2)
            k = torch.transpose(k, 1, 2)
            v = torch.transpose(v, 1, 2)
            assert q.shape == (BATCH, N_HEADS, seqlen_q, D_HEAD)
            assert k.shape == (BATCH, N_HEADS, seqlen_k, D_HEAD)
            assert v.shape == (BATCH, N_HEADS, seqlen_k, D_HEAD)
            assert q.stride() == (N_HEADS * seqlen_q * D_HEAD, D_HEAD, D_HEAD * N_HEADS, 1)
    if False: # Debugging
        q = (
            torch.empty((Z, H, seqlen_q, D_HEAD), dtype=dtype, device="cuda")
            .normal_(mean=0., std=0.5)
            .requires_grad_()
        )
        k = torch.ones((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda") * 1.0
        v = torch.ones((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda") * 1.0
    if False:
        q = torch.ones((Z, H, seqlen_q, D_HEAD), dtype=dtype, device="cuda") * 1.0
        k = torch.ones((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda") * 2.0
        v = torch.ones((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda") * 3.0
    if False:
        import numpy as np
        q = torch.arange(np.prod([Z, H, seqlen_q, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, seqlen_q, D_HEAD))
        k = torch.arange(np.prod([Z, H, seqlen_k, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, seqlen_q, D_HEAD))
        v = torch.arange(np.prod([Z, H, seqlen_k, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, seqlen_q, D_HEAD))
        q = (q - 128.0) * 0.01
        k = (k - 128.0) * 0.01
        v = (v - 128.0) * 0.01
        q[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        k[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        v[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        q[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
        k[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
        v[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0

    '''
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    '''
    return_encoded_softmax = dropout_p > 0.0
    tri_out, encoded_softmax, _ = attention(q, k, v, causal, sm_scale, dropout_p, return_encoded_softmax)

    dropout_mask = encoded_softmax > 0 if encoded_softmax is not None else None
    # assert torch.allclose(dropout_mask, dropout_mask_naive)
    ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v,
                                                                dropout_p=dropout_p,
                                                                is_causal=causal,
                                                                scale=sm_scale,
                                                                dropout_mask=dropout_mask)
    if True:
        mref_out, mref_softmax = scaled_dot_product_attention(q, k, v,
                                     dropout_p=dropout_p,
                                     is_causal=causal,
                                     scale=sm_scale,
                                     dropout_mask=dropout_mask)
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{mref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{q.shape=} {q.stride()=}')
        print(f'{k.shape=} {k.stride()=}')
        print(f'{v.shape=} {v.stride()=}')
        print(f'{tri_out.shape=} {tri_out.stride()=}')
        print(f'{encoded_softmax=}')
        if encoded_softmax is not None:
            print(f'{encoded_softmax.shape=} {encoded_softmax.stride()=}')
            print(f'{encoded_softmax[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_SEQ_SINCE]=}')
            print(f'{dropout_mask.shape=} {dropout_mask.stride()=}')
            print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    if dtype==torch.bfloat16:
        ATOL = 1e-1 * (seqlen_q / 64.0) if seqlen_q >= 16 else 1e-1
    else:
        ATOL = 1e-2 * (seqlen_q / 64.0) if seqlen_q >= 16 else 1e-2
    print(f'Using ATOL={ATOL}')
    is_allclose = torch.allclose(ref_out, tri_out, atol=ATOL, rtol=0)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=} {ref_out[err_idx]=} error: {tri_out[err_idx] - ref_out[err_idx]}')
    # if not is_allclose:
    if True:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{tri_out[0][0][0][:]=}')
        print(f'{ref_out[0][0][0][:]=}')
        print(f'{mref_out[0][0][0][:]=}')
        if encoded_softmax is not None:
            print(f'{encoded_softmax[0][0][0][:]=}')
        print(f'{ref_softmax[0][0][0][:]=}')
        print(f'{tri_out[-1][0][0][:]=}')
        print(f'{ref_out[-1][0][0][:]=}')
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
        if dropout_p > 0:
            # print(f'{unmasked_ref_out[0][0][0][:]=}')
            print(f'{dropout_mask[0][0][0][:]=}')
            print(f'{dropout_mask[err_idx]=}')
        # tri_cpu = tri_out[0, 0].cpu().detach().numpy()
        # print(f'{tri_cpu.shape=}')
    # compare
    assert is_allclose
