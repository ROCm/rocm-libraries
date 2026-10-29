#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os
import math

from attn_torch_function import (
    DEFAULT_PHILOX_SEED,
    DEFAULT_PHILOX_OFFSET,
    attention,
    AttentionExtraArgs
)
from _common_test import SdpaContext, SdpaParams

FOR_RELEASE = bool(int(os.getenv('FOR_RELEASE', default='0')))

POT_HEADDIMS = [16, 32, 64, 128, 256]
NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
PRIME_HEADDIMS = [7, 23, 37, 53, 67, 73, 89, 113, 149, 179, 211, 241]
PRIME_SEQLEN_Q = [11, 17, 37, 67, 157, 257, 523, 1033, 2063, 4919, 10601]
PRIME_SEQLEN_K = [13, 31, 41, 71, 211, 337, 571, 1063, 2081, 5237, 11369]

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

def _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    if causal and bias_type is not None:
        pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
    if sm_scale == 'l1':
        sm_scale = 1.0 / D_HEAD
    elif sm_scale == 'l2':
        sm_scale = 1.0 / math.sqrt(D_HEAD)
    # if BATCH > 1 and seqlen_q >= 1024 and seqlen_k >= 1024:
    #     torch.cuda.empty_cache()
    SKIP_DK_DV = True
    SKIP_DQ = True
    SKIP_DB = True if bias_type is None else False
    USE_AUTOTUNE = False
    torch.manual_seed(20)
    SPARSE_HEAD_SINCE = 1
    SPARSE_SEQ_SINCE = 1
    transpose = (1, 2) if storage_flip else None
    ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                      bias_type=bias_type, storage_flip=transpose, device='cuda')
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=SKIP_DB)
    q, k, v, b = ctx.dev_tensors
    # autotune = True
    # # triton implementation
    ext = AttentionExtraArgs(return_encoded_softmax=dropout_p > 0.0,
                             autotune=False,
                             return_autotune=False)
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    dropout_mask = encoded_softmax >= 0 if dropout_p > 0.0 else None
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)

    is_allclose, adiff, _, _, tfts = ctx.validate_with_reference(tri_out, None, no_backward=True, return_target_fudge_factors=True)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out.to(device=tri_out.device) - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
    assert is_allclose, f'Forward pass {is_allclose=} {tfts=}'
    print(f'{adiff=}')

# @pytest.mark.parametrize('BATCH', [1])
# @pytest.mark.parametrize('N_HEADS', [1])
@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [3])
@pytest.mark.parametrize('N_HEADS', [1, 4] if not FOR_RELEASE else [8])
# @pytest.mark.parametrize('D_HEAD', [16, 32, 64, 128, 256])
# Irregular-only PyTorch set
# @pytest.mark.parametrize('D_HEAD', [8, 21, 72, 96, 160, 192, 203])
# @pytest.mark.parametrize('seqlen_q', [1, 4, 32, 128, 256, 512, 1024, 7, 394, 250, 399, 511, 1019])
# @pytest.mark.parametrize('seqlen_k', [1, 4, 32, 128, 256, 512, 1024, 3, 217, 339, 313, 491, 988])
# PyTorch set
@pytest.mark.parametrize('D_HEAD', POT_HEADDIMS + NPOT_HEADDIMS + PRIME_HEADDIMS)
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# Minimal set
# @pytest.mark.parametrize('seqlen_q', [32, 128])
# @pytest.mark.parametrize('seqlen_k', [32, 128])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 0.125] if not FOR_RELEASE else ['l1', 'l2'])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

# @pytest.mark.parametrize('BATCH', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [3])
@pytest.mark.parametrize('N_HEADS', [1, 4] if not FOR_RELEASE else [8])
@pytest.mark.parametrize('D_HEAD', POT_HEADDIMS + NPOT_HEADDIMS + PRIME_HEADDIMS)
# @pytest.mark.parametrize('D_HEAD', [128])
# Complete set
# @pytest.mark.parametrize('seqlen_q', [4,8,16,17,32,64,128,143,256,512,1024,2048])
# @pytest.mark.parametrize('seqlen_k', [4,8,16,23,32,64,128,256,512,587,1024,2048])
# PyTorch set
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# @pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_q', [128, 113])
# @pytest.mark.parametrize('seqlen_k', [128, 79])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 0.125] if not FOR_RELEASE else ['l1', 'l2'])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_fwd_with_matrix_bias(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
    causal = False
    bias_type = 'matrix'
    '''
    _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
    '''
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [4])
@pytest.mark.parametrize('N_HEADS', [(16, 8), (10, 2)])
@pytest.mark.parametrize('D_HEAD', [8, 203, 256])
@pytest.mark.parametrize('seqlen_q', [4, 143, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 127, 579, 2048])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [0.0, 0.125] if not FOR_RELEASE else ['l1', 'l2'])
@pytest.mark.parametrize('storage_flip', [False])
def test_gqa(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

@pytest.mark.parametrize('BATCH', [3])
@pytest.mark.parametrize('N_HEADS', [7])
@pytest.mark.parametrize('D_HEAD', PRIME_HEADDIMS)
@pytest.mark.parametrize('seqlen_q', PRIME_SEQLEN_Q)
@pytest.mark.parametrize('seqlen_k', PRIME_SEQLEN_K)
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [1.2])
@pytest.mark.parametrize('storage_flip', [False, True])
@pytest.mark.parametrize('bias_type', [None, 'matrix'], ids=['BiasOff', 'BiasOn'])
def test_irregulars(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)


dtype0 = torch.float16
dtype1 = torch.bfloat16
dtype2 = torch.float32

# Testing test_op_fwd_with_matrix_bias from string
def main4():
    # utshort = 'False-1.2-dtype0-0.0-587-2048-32-1-1'
    utshort = 'False-1.2-dtype0-0.0-4-2048-32-1-1'
    # utshort = 'False-1.2-dtype0-0.0-4-1024-32-1-1'
    utlist_str = list(reversed(utshort.split('-')))
    BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dropout_p, dtype, sm_scale, storage_flip = [eval(e) for e in utlist_str]
    causal = False
    bias_type = 'matrix'
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main3():
    # utshort = 'False-1.2-dtype0-0.0-4-2048-32-1-1'
    # utshort = 'False-1.2-dtype0-0.0-4-1024-32-1-1'
    utlist_str = list(reversed(utshort.split('-')))
    BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, dropout_p, dtype, sm_scale, storage_flip = [eval(e) for e in utlist_str]
    bias_type = None
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main2():
    # False-1.2-dtype0-0.0-587-2048-32-1-1
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-4-1
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-1-4
    # False-1.2-dtype0-0.0-False-4-4-72-1-4
    # BATCH = 1
    # D_HEAD = 32
    # N_HEADS = 4
    # seqlen_q = 16
    # seqlen_k = 16
    # causal = False

    BATCH = 2
    D_HEAD = 4
    N_HEADS = 1
    seqlen_q = 8
    seqlen_k = 8
    causal = False

    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    bias_type = None
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main():
    BATCH = 1
    D_HEAD = 80
    N_HEADS = 2
    seqlen_q = 6432
    seqlen_k = 6432
    '''
    N_HEADS = 6432
    seqlen_q = 2
    seqlen_k = 2
    '''
    causal = False
    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.bfloat16
    storage_flip = False
    bias_type = None
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

if __name__ == '__main__':
    main4()
