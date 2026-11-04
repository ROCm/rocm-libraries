#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# FIXME: MUST Import torch before pyaotriton now. pyaotriton.so may resolve to
# system RCCL which is incompatible with torch's RCCL
import torch
from pyaotriton import get_name_suffix
assert get_name_suffix() != "tRiToN_tEsTeR", ("AOTriton is compiled with suffix 'tRiToN_tEsTeR'. "
                                              "This is a signature for AOTriton built to test Triton compiler, "
                                              "which has fewer functionals selected "
                                              "and should not be used for general tests.")
import pytest
from _core_test_backward import (
    ALL_INT_HEADDIMS,
    ALL_TUP_HEADDIMS,
    REGULAR_SEQLEN,
    REGULAR_SEQLEN_2K,
    PRIME_SEQLEN_Q,
    PRIME_SEQLEN_K,
    PRIME_SEQLEN_Q_1K,
    PRIME_SEQLEN_K_1K,
    FOR_RELEASE,
    BWD_IMPL,
    DTYPES,
    BWDOP_ids,
    fmt_nheads,
    fmt_hdim,
    PRIME_HEADDIMS,
    core_test_logsumexp_scaling,
    core_test_matrix_bias_fwd_bwd_symmetry,
    core_test_nonpositive_scale_symmetry,
    NONPOS_SCALES,
    core_test_op_bwd,
    core_test_large_bf16_nan_values,
    core_test_bottom_right_fully_masked_rows,
    core_test_sm_scale_magnitude,
)
from _core_test_common_mistakes import (
    core_test_common_mistakes,
    COMMON_MISTAKES_CASES,
    COMMON_MISTAKES_DTYPES,
)
from _common_test import ALL_LAYOUTS, StorageLayout

if FOR_RELEASE >= 0:
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5, (10, 2)] if BWD_IMPL != 'aiter' else [5], ids=fmt_nheads)
    @pytest.mark.parametrize('D_HEAD', [8, 64, 184, (24, 152), (120, 8), (64, 32)], ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', [11, 523, 2048])
    @pytest.mark.parametrize('seqlen_k', [31, 337, 1063])
    # pairing causal and bias_type to eliminate programmatic skips.
    # Gated on BWD_IMPL != 'aiter' for the same reason line 71 below is: AITER ASM
    # does not support bias, so the matrix-bias pair has to come out of the
    # list rather than be skipped inside the test -- the point of pairing is
    # that there is no programmatic skip left here.
    #
    # 'matrix,nograd' is a bias whose gradient the caller does not want: dB
    # arrives as a null pointer over an all-zero stride triple, which is what
    # PyTorch passes for every bool-masked SDPA and what no other case reaches.
    # See the bias_type parser in _do_test_op_bwd.
    @pytest.mark.parametrize('causal,bias_type',
                             [(False, None), (False, 'matrix'), (False, 'matrix,nograd'), (True, None)]
                             if BWD_IMPL != 'aiter' else [(False, None), (True, None)],
                             ids=['CausalOff-BiasOff', 'CausalOff-BiasOn', 'CausalOff-BiasOnNoGrad', 'CausalOn-BiasOff']
                             if BWD_IMPL != 'aiter' else ['CausalOff-BiasOff', 'CausalOn-BiasOff'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 'aiter' else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_fast(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE > 0:
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', REGULAR_SEQLEN)
    @pytest.mark.parametrize('seqlen_k', REGULAR_SEQLEN)
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 'aiter' else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1', 'l2'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_regular_bwd(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
        bias_type = None
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE > 0 and BWD_IMPL != 'aiter':  # AITER ASM does not support bias ATM
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', REGULAR_SEQLEN_2K)
    @pytest.mark.parametrize('seqlen_k', REGULAR_SEQLEN_2K)
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 'aiter' else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_op_bwd_with_matrix_bias(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
        causal = False
        bias_type = 'matrix'
        '''
        _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
        '''
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE > 0 and BWD_IMPL != 'aiter':  # AITER ASM does not expose GQA
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [(16, 8), (10, 2)])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', [4, 143, 2048])
    @pytest.mark.parametrize('seqlen_k', [4, 127, 579, 2048])
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1', 'l2'])
    @pytest.mark.parametrize('storage_flip', [False])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_gqa(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
        bias_type = None
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE >= 0:
    # The 8xD input contract, exercised. See PRIME_HEADDIMS in
    # _core_test_backward.py for why these were disabled and what changed.
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', PRIME_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', [257])
    @pytest.mark.parametrize('seqlen_k', [571])
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('bias_type', [None], ids=['BiasOff'])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_prime_hdim(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE >= 0:
    # **Every tensor in a different memory layout, with the head dim off the
    # 8-multiple grid.** test_prime_hdim above proves the 8xD contract holds;
    # this proves the descriptor that implements it is derived from the right
    # tensor's stride, which is a question only a permuted layout can ask.
    #
    # `_slab_span_elems` spans a `(batch, head)` slab as
    # `(rows-1)*stride_seq + ceil8(hdim)`, built per tensor from THAT tensor's
    # own stride. Under BHSD every tensor agrees, so reading the wrong one is
    # invisible -- and BHSD plus the single transposition `storage_flip` reaches
    # were the only layouts anything exercised.
    #
    # Sensitivity: end a slab at `hdim` instead and the hardware drops the dword
    # holding columns `hdim-1` and `hdim`, taking the real column with it on the
    # last row of every slab. Measured at these shapes: `O[.., seqlen_q-1,
    # hdim-1]` never stored, `dQ[.., seqlen_q-1, hdim-1]` and `dK/dV[..,
    # seqlen_kv-1, hdim-1]` garbage, and the NaN left in O reaches every element
    # of dK through delta.
    LAYOUT_CASES = [StorageLayout.round_robin(case) for case in range(len(ALL_LAYOUTS))]

    # BATCH and N_HEADS must BOTH be > 1. At 1 the corresponding stride is
    # arbitrary and unconstrained, so the six permutations collapse into fewer
    # than six distinct stride patterns and the test stops asking its question.
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5], ids=fmt_nheads)
    # One prime on each side of the point where the D axis stops being a single
    # tile: 53 rides one block, 179 is loaded as a composed 128+64 pair, so both
    # the simple and the composed store path meet a row whose last dword straddles
    # the slab bound. Both are odd, which is what puts the last real column in the
    # same dword as the first pad column.
    @pytest.mark.parametrize('D_HEAD', [53, 179], ids=fmt_hdim)
    # Off the block grid in both directions, as test_prime_hdim uses them: the
    # defect is on the LAST row of a slab, so a seqlen that divides the block size
    # evenly would never produce a ragged one.
    @pytest.mark.parametrize('seqlen_q', [257])
    @pytest.mark.parametrize('seqlen_k', [571])
    # Paired rather than crossed, the way test_fast pairs them, so there is no
    # programmatic skip: `causal and bias_type is not None` is rejected by
    # _scaled_dot_product_attention, and AITER ASM has no bias at all. Bias is in
    # here because it has a descriptor of its own (`_bias_slab_num_records_bytes`)
    # and an innermost axis that is the KV sequence rather than a head dim.
    @pytest.mark.parametrize('causal,bias_type',
                             [(False, None), (False, 'matrix'), (True, None)]
                             if BWD_IMPL != 'aiter' else [(False, None), (True, None)],
                             ids=['CausalOff-BiasOff', 'CausalOff-BiasOn', 'CausalOn-BiasOff']
                             if BWD_IMPL != 'aiter' else ['CausalOff-BiasOff', 'CausalOn-BiasOff'])
    @pytest.mark.parametrize('dropout_p', [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    # The `storage_flip` slot takes either spelling; _do_test_op_bwd dispatches on
    # the type. See _common_test.StorageLayout for the round robin these are.
    @pytest.mark.parametrize('storage_flip', LAYOUT_CASES,
                             ids=[f'Layouts{case}' for case in range(len(ALL_LAYOUTS))])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_memory_layouts(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE > 1:  # Make the loading faster
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', PRIME_SEQLEN_Q)
    @pytest.mark.parametrize('seqlen_k', PRIME_SEQLEN_K)
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1', 'l2'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('bias_type', [None, 'matrix'], ids=['BiasOff', 'BiasOn'])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_irregulars(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        if bias_type is not None and BWD_IMPL == 'aiter':
            pytest.skip("Bias is not supported in AITER ASM backend")
        if bias_type is not None and (seqlen_q > 2048 or seqlen_k > 2048):
            pytest.skip("Skip large UT with bias to avoid OOM")
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE > 2:  # Testing hdim_qk != hdim_vo
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_TUP_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', PRIME_SEQLEN_Q_1K)
    @pytest.mark.parametrize('seqlen_k', PRIME_SEQLEN_K_1K)
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [False])
    @pytest.mark.parametrize('bias_type', [None, 'matrix'], ids=['BiasOff', 'BiasOn'])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_hdim_qk_ne_vo(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        if bias_type is not None and (seqlen_q > 2048 or seqlen_k > 2048):
            pytest.skip("Skip large UT with bias to avoid OOM")
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

@pytest.mark.parametrize('D_HEAD', [16])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_large_bf16_nan_values(BWDOP, D_HEAD):
    core_test_large_bf16_nan_values(D_HEAD)

# Defined here rather than imported from _core_test_backward, so that conftest.py's
# _FILE_ORDER sees this file. The gpu_id is the second half of the fix:
# core_test_logsumexp_scaling hardcodes device='cuda', so without the context manager
# it runs on the default device regardless of which GPU this xdist worker leased.
@pytest.mark.parametrize('dtype', DTYPES)
def test_logsumexp_scaling(gpu_id, dtype):
    with torch.cuda.device(gpu_id):
        core_test_logsumexp_scaling(dtype)

# Defined here for the same two reasons as test_logsumexp_scaling above.
@pytest.mark.parametrize('bias_val', [0.0, 4.0, 16.0, -16.0, 64.0])
@pytest.mark.parametrize('dtype', DTYPES)
def test_matrix_bias_fwd_bwd_symmetry(gpu_id, dtype, bias_val):
    with torch.cuda.device(gpu_id):
        core_test_matrix_bias_fwd_bwd_symmetry(dtype, bias_val)

# Analytic check: a single-key shape (dQ = dK = 0 checked directly) and a prime one.
NONPOS_SYM_SEQLENS = [(17, 1), (257, 571)]
# Reference sweep: no single-key shape. With one key the exact dQ and dK are 0,
# so there is no reference error to scale a tolerance from, and the kernel's fp32
# rounding noise fails it for either sign of sm_scale.
NONPOS_REF_SEQLENS = [(1, 17), (17, 2), (128, 128), (257, 571)]
# (window_left, window_right), top-left aligned; both edges active. Every query
# row keeps at least one key at the NONPOS_REF_SEQLENS shapes.
NONPOS_WINDOW = (16, 3)

# Ungated (level 0): sm_scale <= 0 must still give finite, correct gradients,
# with a bias and with masked tail tiles. Q = K = 0 makes the answer exact, so
# this checks Out, LSE, dQ, dK, dV and dB directly without a reference.
@pytest.mark.parametrize('seqlen_q,seqlen_k', NONPOS_SYM_SEQLENS)
@pytest.mark.parametrize('sm_scale', NONPOS_SCALES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_nonpositive_scale_symmetry(gpu_id, dtype, sm_scale, seqlen_q, seqlen_k):
    with torch.cuda.device(gpu_id):
        core_test_nonpositive_scale_symmetry(dtype, sm_scale, seqlen_q, seqlen_k)

# sm_scale <= 0 on random inputs against the reference, with bias, causal,
# sliding window, and dropout.
@pytest.mark.parametrize('BATCH', [3])
@pytest.mark.parametrize('N_HEADS', [5], ids=fmt_nheads)
@pytest.mark.parametrize('D_HEAD', [53], ids=fmt_hdim)
@pytest.mark.parametrize('seqlen_q,seqlen_k', NONPOS_REF_SEQLENS)
@pytest.mark.parametrize('causal,bias_type,dropout_p',
                         [(False, None, 0.0), (False, 'matrix', 0.0), (True, None, 0.0),
                          (NONPOS_WINDOW, None, 0.0), (False, None, 0.5)]
                         if BWD_IMPL != 'aiter' else
                         [(False, None, 0.0), (True, None, 0.0), (NONPOS_WINDOW, None, 0.0)],
                         ids=['CausalOff-BiasOff', 'CausalOff-BiasOn', 'CausalOn-BiasOff',
                              'Window16-3-BiasOff', 'CausalOff-BiasOff-Dropout']
                         if BWD_IMPL != 'aiter' else
                         ['CausalOff-BiasOff', 'CausalOn-BiasOff', 'Window16-3-BiasOff'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', NONPOS_SCALES)
@pytest.mark.parametrize('storage_flip', [False])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_nonpositive_sm_scale(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    core_test_op_bwd(request, args, device=gpu_id)

# ROCm/aotriton#235. Ungated (level 0) on purpose: it guards a silent
# wrong-answer path -- bottom-right causal with seqlen_q > seqlen_k returned
# unwritten Out/LSE rows once the tile count exceeded the workgroup count --
# and it costs one small forward per repeat. This is the shape PyTorch's
# memory-efficient attention takes for causal_lower_right, where the
# corresponding upstream test is currently skipped on ROCm.
#
# Not parametrized over BWDOP: the defect is in attn_fwd, and the backward
# backend is irrelevant to it.
def test_bottom_right_fully_masked_rows(gpu_id):
    with torch.cuda.device(gpu_id):
        core_test_bottom_right_fully_masked_rows(f'cuda:{gpu_id}')

if FOR_RELEASE >= 0:
    # The forward's precision as a function of `sm_scale`. See
    # core_test_sm_scale_magnitude for what the defect is, what the measured
    # numbers are, and why both the normal inputs and the absolute bound are
    # required for this to catch anything.
    #
    # Ungated (level 0) on purpose, like test_bottom_right_fully_masked_rows
    # above: it guards a silent wrong-answer path that the fudge factor by
    # construction cannot see, and 24 small forwards cost ~10s.
    #
    # Not parametrized over BWDOP: the defect is in the forward, and the
    # backward backend is irrelevant to it. Defined here rather than imported
    # so conftest.py's _FILE_ORDER sees this file, and wrapped in
    # torch.cuda.device(gpu_id) for the reason given at test_logsumexp_scaling.
    #
    # Axes, and why each is here:
    #   - sm_scale: 'l2' (= 1/sqrt(hdim)) is the control that must keep
    #     passing; 1.2 is the load-bearing value (and the one
    #     test_forward.py::test_irregulars already uses), larger than the torch
    #     UT's 1.0 and therefore strictly harder.
    #   - D_HEAD: sqrt(hdim) is the other half of the product. 512 is the torch
    #     UT's and the worst case; 64 is the mildest hdim that still shows it;
    #     256 sits between and is a separate flyc tile ladder entry.
    #   - dtype: f16 and bf16 round the folded scale differently (bf16 has the
    #     coarser mantissa, f16 the narrower exponent) and are separate kernel
    #     functionals. A literal list rather than DTYPES, and a strict subset of
    #     it under every backend: f32 would need a third MAX_ADIFF entry to
    #     measure a fold that costs it 2**-24, which is nothing.
    #   - (seqlen_q, seqlen_k, causal): the two pairs the torch UT parametrises.
    #
    # 2 x 3 x 2 x 2 = 24 cases, 16 of them on gfx11, which has no kernel above
    # hdim 256 and skips that rung in core_test_sm_scale_magnitude. BATCH and
    # N_HEADS are fixed small -- neither is on the error's axis, and hdim 512
    # is the expensive part of the footprint.
    #
    # The bounds are per dtype and NOT per `sm_scale`, which is the property
    # under test: a kernel that keeps `qk_scale` on the f32 accumulator has an
    # output error set by the input dtype and the shape, not by the scale.
    # Against the table in core_test_sm_scale_magnitude each bound sits 1.9x
    # (bf16) / 7x (f16) above the worst correct value and 2.2x (bf16) / 2.6x
    # (f16) below the smallest incorrect one, so it is not a threshold anybody
    # should be tuning -- if it needs moving, the property stopped holding.
    SM_SCALE_MAX_ADIFF = {torch.float16: 0.01, torch.bfloat16: 0.03}

    @pytest.mark.parametrize('BATCH', [1])
    @pytest.mark.parametrize('N_HEADS', [4])
    @pytest.mark.parametrize('D_HEAD', [64, 256, 512], ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q,seqlen_k,causal',
                             [(289, 289, False), (289, 400, True)],
                             ids=['289x289-CausalOff', '289x400-CausalOn'])
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize('sm_scale', ['l2', 1.2], ids=['ScaleL2', 'Scale1.2'])
    # The torch UT's query is `randn(1, q_len, num_heads, head_dim).transpose(1, 2)`,
    # i.e. BSHD storage read as BHSD -- storage_flip=True is that layout.
    @pytest.mark.parametrize('storage_flip', [True])
    def test_sm_scale_magnitude(gpu_id, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal,
                                sm_scale, dtype, storage_flip):
        with torch.cuda.device(gpu_id):
            core_test_sm_scale_magnitude(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal,
                                         sm_scale, dtype, storage_flip,
                                         max_adiff=SM_SCALE_MAX_ADIFF[dtype],
                                         device_str=f'cuda:{gpu_id}')

# docs/attention-kernel-numerical-error-lessons/: mistakes a new attention
# kernel is likely to repeat, each of which core_test_op_bwd's fudge factors
# pass. Complements test_sm_scale_magnitude above, which bounds the forward's
# output only; this also checks the LSE, every gradient, and the -inf paths.
# Ungated (level 0) like the guards above it: eight small fwd+bwd calls, and
# every case guards a silent wrong answer rather than a crash.
#
# Not parametrized over BWDOP, for the same reason test_bottom_right_fully_masked_rows
# is not: BWD_IMPL pins the backend for the whole process.
@pytest.mark.parametrize('dtype', COMMON_MISTAKES_DTYPES)
@pytest.mark.parametrize('case', COMMON_MISTAKES_CASES)
def test_common_mistakes(gpu_id, case, dtype):
    with torch.cuda.device(gpu_id):
        core_test_common_mistakes(case, dtype, f'cuda:{gpu_id}')

def main2():
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-4-1
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-1-4
    # False-1.2-dtype0-0.0-False-4-4-72-1-4
    BATCH = 8
    D_HEAD = 64
    N_HEADS = 8
    seqlen_q = 256
    seqlen_k = 256
    causal = False

    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    bias_type = None
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    core_test_op_bwd(args)

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
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    core_test_op_bwd(args)

if __name__ == '__main__':
    main2()
    # main_npz()
