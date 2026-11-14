#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import numpy as np
import math
import os
import json

from _core_test_backward import adiffs, RECORD_ADIFFS_TO
from varlen_attn_torch_function import (
    varlen_attention,
    AttentionExtraArgs,
    PROBE_UNSUPPORTED,
    BWD_IMPL,
)
from _common_test import (
    VarlenSdpaContext,
    PaddedVarlenSdpaContext,
    StridedVarlenSdpaContext,
    SdpaParams,
    fmt_hdim,
    FA_WINDOWS,
    FA_WINDOW_IDS,
    PD_CHOICES,
    PD_CHOICE_IDS,
    PD_VARLEN_SEQLENS_K,
    PD_VARLEN_SEQLENS_Q,
    PD_WINDOW_IDS,
    PD_WINDOW_VALUES,
    fa_keep,
    fa_window_to_aotriton,
    pd_causal,
    windowed_keep_mask,
)
# The dtype set, from the one place that decides it. This file used to spell the
# list out and gate it on `BWD_IMPL == 'aiter'` alone, which is half the rule: fp32
# is
# also out when the FORWARD is pinned to flyc, and that half was added to
# _core_test_backward.py only -- so every fp32 varlen case asked for a kernel
# that was never built. Importing the name is what stops the two drifting again.
from _core_test_backward import DTYPES, SKIP_BWD, NONPOS_SCALES, AOTRITON_UT_ARCH, remove_outside

FOR_RELEASE = int(os.getenv('FOR_RELEASE', default='0'))

POT_HEADDIMS = [16, 32, 64, 128, 256]
NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
PRIME_HEADDIMS = [7, 23, 37, 53, 67, 73, 89, 113, 149, 179, 211, 241]

# Same narrowing as _core_test_backward.py: gfx1100's 0.15b retune covers head
# dims 64, 80 and 128 only.
if AOTRITON_UT_ARCH in ('gfx1100@0.15b',):
    POT_HEADDIMS = remove_outside(POT_HEADDIMS, low=48, high=128)
    NPOT_HEADDIMS = remove_outside(NPOT_HEADDIMS, low=48, high=128)
    PRIME_HEADDIMS = remove_outside(PRIME_HEADDIMS, low=48, high=128)

# SEQLEN_Q = [4, 8, 64, 143, 256, 512, 1024, 2048]
# SEQLEN_K = [4, 8, 64, 128, 256, 587, 1024, 2048]

SEQLEN_Q = [4, 8, 64, 143]
SEQLEN_K = [4, 8, 63, 128]

# SEQLEN_Q = [4]
# SEQLEN_K = [4]

POSSIBLE_SEQLEN = sorted(set(SEQLEN_Q + SEQLEN_K))
POSSIBLE_PADLEN = [0, 4, 7]

def rng_seqlens(n_seqlen):
    return np.random.choice(POSSIBLE_SEQLEN, n_seqlen)

def rng_padlens(n_seqlen):
    return np.random.choice(POSSIBLE_PADLEN, n_seqlen)

VARLEN_FACTORY = {
    "compact": VarlenSdpaContext,
    "padded": PaddedVarlenSdpaContext,
    "strided": StridedVarlenSdpaContext,
    # Same tensor layout as strided -- slot z holds seqlens[z] + padlens[z] rows
    # and only the first seqlens[z] participate -- so the same context builds it
    # and the same reference excludes the tail. What differs is purely how the
    # kernel is told: strided reads K's length cumulatively out of one array,
    # seqused reads it individually out of another. A bits difference, not a
    # layout one.
    "seqused": StridedVarlenSdpaContext,
}

# The `causal` axis of the varlen matrix. Besides off and top-left causal it
# carries a numeric sliding window measured from each sequence's bottom-right
# diagonal (VarlenBits::principal_diagonal_choice = BOTTOM_RIGHT). That is a
# property of the kernel's mask code, not of a feature combination, so it rides
# the level-0 matrix: every head dim, dropout, dtype and layout, with
# seqlen_q and seqlen_k drawn independently so d = seqlen_k - seqlen_q differs
# per sequence and takes both signs. aiter accepts only sentinel windows.
VARLEN_WINDOW_BR = pd_causal((16, 4), 1)
VARLEN_CAUSAL_VALUES = [False, True] + ([] if BWD_IMPL == 'aiter' else [VARLEN_WINDOW_BR])
VARLEN_CAUSAL_IDS = ['CausalOff', 'CausalOn'] + ([] if BWD_IMPL == 'aiter' else ['WindowBR'])

# torch.nn.attention.varlen.varlen_attn(..., seqused_k=...): "Number of valid KV
# tokens per batch element; shape (N,). When set, only the first seqused_k[i]
# tokens in the key/value sequence for batch element i participate in attention."
#
# That pairs an INDIVIDUAL (N,) length array on K with a CUMULATIVE (N+1,)
# position array -- two facts from two different tensors -- against a plain
# compact Q. It is the one shipped configuration the retired VarlenType enum
# could not express at all, and the only asymmetric Q/K pairing driven through
# this API.
#
# torch documents seqused_k as inference-only. This tests the BACKWARD too, on
# purpose: nothing in the decoder makes it forward-only -- the backward kernels
# read the same six scalars -- so the restriction is torch's, not ours, and a
# limitation the code does not actually have is one that rots into a surprise.
@pytest.mark.parametrize('n_seqlen', [3, 7])
@pytest.mark.parametrize('causal', VARLEN_CAUSAL_VALUES, ids=VARLEN_CAUSAL_IDS)
def test_seqused_k(request, gpu_id, n_seqlen, causal):
    np.random.seed(4177)
    N_HEADS = 3
    D_HEAD = 64
    seqlens_q = rng_seqlens(n_seqlen)
    if causal is True:
        # Deliberately SHORTER than Q, not equal to it. Under top-left causal a
        # key past index seqlen_q is masked no matter what, so slack appended to
        # a full-length K is invisible and this half of the test would pass with
        # seqused_k ignored -- verified by sabotage, it did. Truncating below the
        # query length puts the used bound back inside the causal window.
        seqlens_k = np.maximum(1, seqlens_q // 2)
    else:
        seqlens_k = rng_seqlens(n_seqlen)
    # The slack is the point: every padlen must be > 0, or the used length
    # coincides with the slot length and an implementation that ignored
    # seqused_k entirely would pass. Q carries none -- it is packed compact.
    padlens_k = 1 + rng_padlens(n_seqlen)
    padlens_q = np.zeros(n_seqlen, dtype=padlens_k.dtype)
    with torch.cuda.device(gpu_id):
        _do_test_varlen(N_HEADS, D_HEAD,
                        np.array([seqlens_q, padlens_q]),
                        np.array([seqlens_k, padlens_k]),
                        causal, 'l1', 0.0, torch.float16, 'seqused')


# Modes whose seqlens arrive as (lens, padlens) pairs and whose tensors carry
# slack between sequences. Named once: the branches below all mean this and not
# "is strided", and the two diverge as soon as a third slotted mode appears.
SLOTTED_TYPES = ('strided', 'seqused')


def _lse_valid_token_ranges(seqlens_q, varlen_type):
    """(start, stop) along the packed token axis for each sequence's own rows.

    Under a slotted layout the gaps between sequences are never written, so a
    whole-buffer comparison would be comparing uninitialized memory.
    """
    if varlen_type in SLOTTED_TYPES:
        lens, pads = seqlens_q
    else:
        lens, pads = seqlens_q, np.zeros_like(seqlens_q)
    start = 0
    for seqlen, padlen in zip(lens, pads):
        yield start, start + int(seqlen)
        start += int(seqlen) + int(padlen)


def _assert_lse_th_is_ht_transposed(lse_th, lse_ht, seqlens_q, num_heads, varlen_type):
    """The two layouts of the same run must be transposes of each other.

    Round-tripping the gradients through test_op_bwd is necessary and NOT
    sufficient -- if all four kernels (attn_fwd writes L, bwd_preprocess writes
    Delta, dk_dv and dq read both) are CONSISTENTLY wrong about where a row
    lives, the gradients still match the reference and the test passes. This is
    the assertion that fails in that case.

    It is still only a RELATIVE check: it pins the two runs to each other, not
    either to the truth. test_logsumexp_layout below closes that with a
    closed-form expected value.
    """
    if varlen_type == 'padded':
        batch = len(seqlens_q)
        max_seqlen_q = int(np.max(seqlens_q))
        assert lse_ht.shape == (batch * num_heads, max_seqlen_q), f'{lse_ht.shape=}'
        assert lse_th.shape == (batch * max_seqlen_q, num_heads), f'{lse_th.shape=}'
        # HT is (B, H, S) flattened over (B, H); TH is (B, S, H) flattened over
        # (B, S). Not a plain 2-D transpose -- the batch axis stays outermost in
        # both, which is exactly the confusion this check exists to catch.
        ht = lse_ht.view(batch, num_heads, max_seqlen_q)
        th = lse_th.view(batch, max_seqlen_q, num_heads)
        for b, seqlen in enumerate(seqlens_q):
            seqlen = int(seqlen)
            assert torch.equal(th[b, :seqlen, :], ht[b, :, :seqlen].t()), \
                    f'LSE layout mismatch on sequence {b}'
        return
    total_seqlen_q = lse_ht.shape[1]
    assert lse_ht.shape == (num_heads, total_seqlen_q), f'{lse_ht.shape=}'
    assert lse_th.shape == (total_seqlen_q, num_heads), f'{lse_th.shape=}'
    for z, (start, stop) in enumerate(_lse_valid_token_ranges(seqlens_q, varlen_type)):
        assert torch.equal(lse_th[start:stop, :], lse_ht[:, start:stop].t()), \
                f'LSE layout mismatch on sequence {z} (tokens {start}:{stop})'




def _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype,
                    varlen_type, lse_layout='HT'):
    assert varlen_type in VARLEN_FACTORY.keys(), f"_do_test_varlen: unknown varlen_type {varlen_type}"
    if isinstance(D_HEAD, int):
        HDIM_QK = HDIM_VO = D_HEAD
    else:
        HDIM_QK, HDIM_VO = D_HEAD
    HDIM_MAX = max(HDIM_QK, HDIM_VO)
    if sm_scale == 'l1':
        sm_scale = 1.0 / HDIM_QK
    elif sm_scale == 'l2':
        sm_scale = 1.0 / math.sqrt(HDIM_QK)
    # Data creation
    SKIP_DK_DV = False
    SKIP_DQ = False
    USE_AUTOTUNE = False
    torch.manual_seed(20)
    factory = VARLEN_FACTORY[varlen_type]
    # Same adiffs protocol as core_test_op_bwd (OOM/NAN/CPUREF/recorded entry).
    utname = os.environ.get('PYTEST_CURRENT_TEST')
    use_adiff_entry = adiffs.get(utname, None)
    if use_adiff_entry == "OOM":
        pytest.skip("[Adiffs] Skip due to known OOM.")
    if use_adiff_entry == "NAN":
        pytest.xfail("[Adiffs] known NAN.")
    adiff_ref_device_policy = None
    if use_adiff_entry == "CPUREF":
        adiff_ref_device_policy = 'cpu'
        use_adiff_entry = None
    ctx = factory(N_HEADS, D_HEAD, seqlens_q, seqlens_k, dtype, device='cuda')
    ctx.create_ref_inputs(target_device_policy=adiff_ref_device_policy)
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=True)
    q, k, v, b = ctx.dev_tensors
    # Forward
    ext = AttentionExtraArgs(return_encoded_softmax=dropout_p > 0.0,
                             autotune=USE_AUTOTUNE,
                             return_autotune=False,
                             fillnan=True)
    # What lse_layout gates HERE is the BACKWARD: bwd_preprocess writes Delta and
    # both key kernels read L and Delta through the same addressing, so a layout
    # the backward disagrees with shows up in dq/dk/dv below. The LSE buffer's
    # own contents are gated by test_logsumexp_layout, which pins them per
    # position against a closed form -- an absolute check, so re-running the
    # forward here under 'HT' and comparing the transpose would only re-prove a
    # weaker version of it at the cost of a second launch on every TH case.
    tri_out, encoded_softmax, _ = varlen_attention(q, k, v, seqlens_q, seqlens_k, causal,
                                                   sm_scale, dropout_p, varlen_type,
                                                   lse_layout, ext)
    dropout_mask = encoded_softmax >= 0 if dropout_p > 0.0 else None
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)

    if SKIP_BWD:
        # Forward-only, the same contract `_core_test_backward` honours: a
        # backward-less backend must not be judged on a backward this harness
        # launched for it. `no_backward=True` keeps the forward assertion below
        # exactly the one the full path makes.
        is_allclose, adiff, _ga, _gd, tfts = ctx.validate_with_reference(
            tri_out, [], no_backward=True, return_target_fudge_factors=True,
            use_adiff_entry=use_adiff_entry)
        if RECORD_ADIFFS_TO is not None and not is_allclose:
            with open(RECORD_ADIFFS_TO, 'a') as f:
                dj = { "adiff" : adiff, "grads_adiff" : [] }
                print(utname, "\t", json.dumps(dj), file=f, flush=True, sep='')
            pytest.xfail(f"RECORD ADIFFS {adiff=} (SKIP_BWD=1)")
        assert is_allclose, f'Forward pass {is_allclose=} {tfts=}'
        print(f'{adiff=} (SKIP_BWD=1, backward not run)')
        return

    # # Backward
    dout = torch.rand_like(tri_out)
    if PROBE_UNSUPPORTED:
        try:
            ctx.compute_backward(tri_out, dout)
        except NotImplementedError as e:
            pytest.xfail("Unsupported Config in AITER")
    else:
        ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff, tfts = ctx.validate_with_reference(tri_out, ctx.dout_tensors, return_target_fudge_factors=True, use_adiff_entry=use_adiff_entry)
    if RECORD_ADIFFS_TO is not None and (not is_allclose or not all(grads_allclose)):
        with open(RECORD_ADIFFS_TO, 'a') as f:
            dj = { "adiff" : adiff, "grads_adiff" : grads_adiff }
            print(utname, "\t", json.dumps(dj), file=f, flush=True, sep='')
        pytest.xfail(f"RECORD ADIFFS {adiff=} {grads_adiff=}")
    torch.set_printoptions(threshold=114514, linewidth=200)

    # Test Forward
    if not is_allclose:
        import numpy as np
        print(f'{ref_out.shape=}')
        print(f'{tri_out.shape=}')
        print(f'{seqlens_q=}')
        print(f'{seqlens_k=}')
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out.cpu() - tri_out.cpu())).numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
        # print(f'{tri_out=}')
        # print(f'{ref_out=}')
    assert is_allclose, f'Forward pass {is_allclose=} {tfts=}'

    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    def TO(ref_tensor):
        return ref_tensor.to(device=q.device, dtype=dtype)
    if not dv_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=} {q.dtype=}')
        print(f'{k.shape=} {k.stride()=} {k.dtype=}')
        print(f'{v.shape=} {v.stride()=} {v.dtype=}')
        print(f'{err_idx=}')
        print(f'{tri_dv[err_idx]=}')
        print(f'{ref_dv[err_idx]=}')
        print(f'{torch.isnan(ref_dv).any()=}')

    if dv_allclose and not dk_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')

    if dk_allclose and dv_allclose and not dq_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')

    if dk_allclose and dv_allclose and dq_allclose and not db_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_db) - tri_db)).cpu().numpy(), ref_db.shape)
        print(f'{err_idx=}')
        print(f'{tri_db[err_idx]=} {ref_db[err_idx]=} error = {torch.abs(tri_db[err_idx] - ref_db[err_idx])}')

    assert dk_allclose and dv_allclose and dq_allclose and db_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=}'
    print(f'{adiff=} {grads_adiff=}')

@pytest.mark.parametrize('N_HEADS', [3])
@pytest.mark.parametrize('D_HEAD', [64, 128, 192] if BWD_IMPL == 'aiter' else [8, 64, 184, (24, 152), (120, 8)], ids=fmt_hdim)
@pytest.mark.parametrize('n_seqlen', range(2, 24, 5))
@pytest.mark.parametrize('causal', VARLEN_CAUSAL_VALUES, ids=VARLEN_CAUSAL_IDS)
@pytest.mark.parametrize('dropout_p', [0.0] if BWD_IMPL == 'aiter' else [0.0, 0.5])
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('sm_scale', ['l1', 'l2'] if FOR_RELEASE > 0 else ['l1'])
@pytest.mark.parametrize('varlen_type', ['compact', 'padded', 'strided'])
@pytest.mark.parametrize('lse_layout', ['HT', 'TH'] if FOR_RELEASE > 0 else ['HT'])
def test_op_bwd(gpu_id, N_HEADS, D_HEAD, n_seqlen, causal, sm_scale, dropout_p, dtype,
                varlen_type, lse_layout):
    np.random.seed(8139)
    seqlens_q = rng_seqlens(n_seqlen)
    # `causal is True`, not truthiness: a window tuple is truthy, and the
    # windowed variant needs independent lengths to have a per-sequence d.
    seqlens_k = seqlens_q if causal is True else rng_seqlens(n_seqlen)
    if varlen_type in SLOTTED_TYPES:
        padlens_q = rng_padlens(n_seqlen)
        padlens_k = padlens_q if causal is True else rng_padlens(n_seqlen)
        seqlens_q = np.array([seqlens_q, padlens_q])
        seqlens_k = np.array([seqlens_k, padlens_k])
    # Sets the current CUDA device for this worker, which is what the bare 'cuda'
    # inside _do_test_varlen resolves to. No device string needs threading through.
    with torch.cuda.device(gpu_id):
        _do_test_varlen(N_HEADS, D_HEAD,
                        seqlens_q, seqlens_k,
                        causal, sm_scale, dropout_p, dtype, varlen_type, lse_layout)

# VarlenBits::principal_diagonal_choice (p) under varlen. d = seqlen_k -
# seqlen_q is PER SEQUENCE, so one batch can hold a q < k, a q == k, a q > k, a
# decode (seqlen_q = 1) and a chunked prefill, each measured from its own
# bottom-right diagonal when p = 1. That is the case no single host-side window
# can express (pytorch/pytorch#201930). The window and p travel in the `causal`
# slot as (left, right, p).
#
# The reference builds each sequence's mask from that sequence's own lengths
# (_common_test.windowed_sdpa_math); a row that keeps no key must give O = 0.
# Its LSE is a placeholder and is not compared.
#
# Select with `-k principal_diagonal`; `-k PD1` for p = 1, `-k FA_` for the
# flash-attention windows, `-k chunked_prefill` for the #201930 scenario.
PD_VARLEN_TYPES = ['compact', 'padded', 'strided']
PD_VARLEN_PADLENS = np.array([0, 4, 7, 0, 4, 7])

def _pd_varlen_lengths(varlen_type, seqlens_q, seqlens_k):
    """seqlens as `_do_test_varlen` wants them for `varlen_type`."""
    if varlen_type in SLOTTED_TYPES:
        padlens = PD_VARLEN_PADLENS[:len(seqlens_q)]
        return np.array([seqlens_q, padlens]), np.array([seqlens_k, padlens])
    return seqlens_q, seqlens_k

if BWD_IMPL != 'aiter':
    @pytest.mark.parametrize('N_HEADS', [3])
    @pytest.mark.parametrize('D_HEAD', [64], ids=fmt_hdim)
    @pytest.mark.parametrize('window', PD_WINDOW_VALUES, ids=PD_WINDOW_IDS)
    @pytest.mark.parametrize('principal_diagonal', PD_CHOICES, ids=PD_CHOICE_IDS)
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize('varlen_type', PD_VARLEN_TYPES)
    def test_varlen_principal_diagonal(gpu_id, N_HEADS, D_HEAD, window, principal_diagonal, dtype, varlen_type):
        seqlens_q, seqlens_k = _pd_varlen_lengths(varlen_type, PD_VARLEN_SEQLENS_Q, PD_VARLEN_SEQLENS_K)
        with torch.cuda.device(gpu_id):
            _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k,
                            pd_causal(window, principal_diagonal),
                            'l1', 0.0, dtype, varlen_type)

    # PyTorch's test_varlen_attention.py windows, translated per the "Matching
    # flash-attention" table (p = 1), against a batch with seqlen_q != seqlen_k
    # in every sequence but one.
    @pytest.mark.parametrize('N_HEADS', [3])
    @pytest.mark.parametrize('D_HEAD', [64], ids=fmt_hdim)
    @pytest.mark.parametrize('fa_window', FA_WINDOWS, ids=FA_WINDOW_IDS)
    @pytest.mark.parametrize('varlen_type', PD_VARLEN_TYPES)
    def test_varlen_principal_diagonal_fa_windows(gpu_id, N_HEADS, D_HEAD, fa_window, varlen_type):
        max_seqlen_q = int(np.max(PD_VARLEN_SEQLENS_Q))
        window = fa_window_to_aotriton(fa_window[0], fa_window[1], max_seqlen_q)
        assert window is not None
        seqlens_q, seqlens_k = _pd_varlen_lengths(varlen_type, PD_VARLEN_SEQLENS_Q, PD_VARLEN_SEQLENS_K)
        with torch.cuda.device(gpu_id):
            _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k,
                            pd_causal(window, 1),
                            'l1', 0.0, torch.float16, varlen_type)

    # The #201930 scenario: a serving batch of chunked prefills (new queries
    # over a long cached context) and decodes, with a numeric sliding window.
    # Every sequence has its own d, from 0 to 199, and p = 1 must anchor each
    # window at its own diagonal.
    @pytest.mark.parametrize('N_HEADS', [3])
    @pytest.mark.parametrize('D_HEAD', [64], ids=fmt_hdim)
    @pytest.mark.parametrize('window', [(8, 0), (4, 4), (0, 0)], ids=['8_0', '4_4', '0_0'])
    @pytest.mark.parametrize('varlen_type', PD_VARLEN_TYPES)
    def test_varlen_principal_diagonal_chunked_prefill(gpu_id, N_HEADS, D_HEAD, window, varlen_type):
        seqlens_q = np.array([16, 16, 1, 1, 32, 5])
        seqlens_k = np.array([100, 48, 57, 200, 32, 5])
        seqlens_q, seqlens_k = _pd_varlen_lengths(varlen_type, seqlens_q, seqlens_k)
        with torch.cuda.device(gpu_id):
            _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k,
                            pd_causal(window, 1),
                            'l1', 0.0, torch.float16, varlen_type)

# Pure Python, no GPU: the translation table against flash-attention's own
# predicate, per sequence, for every sequence of the mixed batch.
@pytest.mark.parametrize('fa_window', FA_WINDOWS + [(-1, 0), (3, -1)],
                         ids=FA_WINDOW_IDS + ['FA_m1_0', 'FA_3_m1'])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
def test_principal_diagonal_fa_translation(fa_window, causal):
    max_seqlen_q = int(np.max(PD_VARLEN_SEQLENS_Q))
    window = fa_window_to_aotriton(fa_window[0], fa_window[1], max_seqlen_q, causal)
    for seqlen_q, seqlen_k in zip(PD_VARLEN_SEQLENS_Q, PD_VARLEN_SEQLENS_K):
        seqlen_q, seqlen_k = int(seqlen_q), int(seqlen_k)
        expected = fa_keep(seqlen_q, seqlen_k, fa_window[0], fa_window[1], causal)
        got = windowed_keep_mask(seqlen_q, seqlen_k, window[0], window[1], 1, device='cpu').numpy()
        assert (got == expected).all(), f'{seqlen_q=} {seqlen_k=} {window=}'

# sm_scale <= 0 with varlen: each sequence's ragged tail (primes, 1, one past a
# block edge) must stay masked. padded pads every sequence to the longest, so
# each one ends in masked slack. No sequence has a single key: see
# NONPOS_REF_SEQLENS in test_backward.py.
@pytest.mark.parametrize('sm_scale', NONPOS_SCALES)
def test_nonpositive_sm_scale_varlen(gpu_id, sm_scale):
    seqlens_q = np.array([17, 1, 67, 128, 31])
    seqlens_k = np.array([31, 17, 64, 129, 2])
    with torch.cuda.device(gpu_id):
        _do_test_varlen(3, 64, seqlens_q, seqlens_k,
                        False, sm_scale, 0.0, torch.float16, 'padded', 'HT')

# Distinct per head, so the logsumexp varies along H. Also the number of heads.
HEAD_SCALES = [1.0, 2.0, 3.0]
# Distinct per sequence, so it varies along T. 63 is deliberately not a multiple
# of any BLOCK_M.
LAYOUT_SEQLENS = [4, 8, 63]
LAYOUT_PADLENS = [0, 7, 4]


def _build_analytic_lse_inputs(varlen_type, num_heads, hdim, device, dtype):
    """Q/K/V for which the logsumexp has a closed form, plus that closed form.

    The trap this construction exists to avoid: test_logsumexp_scaling uses
    identity Q/K/V, which makes EVERY element of L the same number. That is fine
    for a scaling test and useless for a layout test -- if every element is
    equal, a transposed buffer compares equal to a correct one.

    So make L vary along BOTH axes and stay analytic. Every K row is e_0; every Q
    row of head h is HEAD_SCALES[h] * e_0. Then every score in a sequence is
    sm_scale * HEAD_SCALES[h], and

        lse[h, t in sequence z] = sm_scale * HEAD_SCALES[h] + log(seqlens_k[z])

    exactly -- one term per axis, so a swapped index is wrong in a way allclose
    sees.
    """
    seqlens = np.array(LAYOUT_SEQLENS)
    padlens = np.array(LAYOUT_PADLENS)
    scales = torch.tensor(HEAD_SCALES, device=device, dtype=torch.float32)

    def token_axis_len():
        if varlen_type == 'padded':
            return int(np.max(seqlens))
        if varlen_type in SLOTTED_TYPES:
            return int(np.sum(seqlens + padlens))
        return int(np.sum(seqlens))

    def sequence_slices():
        # (batch index, token slice) of each sequence's own rows.
        if varlen_type == 'padded':
            for z, seqlen in enumerate(seqlens):
                yield z, slice(0, int(seqlen))
            return
        start = 0
        for z, seqlen in enumerate(seqlens):
            yield 0, slice(start, start + int(seqlen))
            start += int(seqlen) + (int(padlens[z]) if varlen_type in SLOTTED_TYPES else 0)

    batch = len(seqlens) if varlen_type == 'padded' else 1
    tokens = token_axis_len()
    q = torch.zeros((batch, num_heads, tokens, hdim), device=device, dtype=dtype)
    k = torch.zeros((batch, num_heads, tokens, hdim), device=device, dtype=dtype)
    # V never enters the logsumexp, so anything finite will do -- but make it
    # non-zero so a bug that reads V into L would be visible rather than lucky.
    v = torch.full((batch, num_heads, tokens, hdim), 0.5, device=device, dtype=dtype)
    for b, tok in sequence_slices():
        k[b, :, tok, 0] = 1.0
        for h in range(num_heads):
            q[b, h, tok, 0] = HEAD_SCALES[h]
    return q, k, v, seqlens, padlens, scales, sequence_slices


# A handful of launches, not a second launch on every TH case in test_op_bwd.
# The property is about the LSE buffer alone, so it needs one small case per
# layout -- running it inside the backward matrix bought nothing and doubled the
# forward cost of that matrix's whole TH half.
#
# What this adds over test_logsumexp_layout, which is otherwise strictly
# stronger: RANDOM inputs. That test's closed form needs a contrived
# construction (K rows e_0, Q rows c_h * e_0), so it pins values only for that
# one input shape. This one says the layouts stay transposes for arbitrary data.
@pytest.mark.parametrize('varlen_type', ['compact', 'padded', 'strided'])
def test_lse_layout_th_is_ht_transposed(gpu_id, varlen_type):
    np.random.seed(9137)
    N_HEADS, D_HEAD, n_seqlen = 3, 64, 4
    seqlens_q = rng_seqlens(n_seqlen)
    seqlens_k = rng_seqlens(n_seqlen)
    if varlen_type in SLOTTED_TYPES:
        seqlens_q = np.array([seqlens_q, rng_padlens(n_seqlen)])
        seqlens_k = np.array([seqlens_k, rng_padlens(n_seqlen)])
    with torch.cuda.device(gpu_id):
        ctx = VARLEN_FACTORY[varlen_type](N_HEADS, D_HEAD, seqlens_q, seqlens_k,
                                          torch.float16, device='cuda')
        ctx.create_ref_inputs()
        q, k, v, _ = ctx.dev_tensors
        out = {}
        for layout in ('HT', 'TH'):
            ext = AttentionExtraArgs(return_encoded_softmax=False, autotune=False,
                                     return_autotune=False, fillnan=True,
                                     return_logsumexp=True)
            # Same inputs, same fixed philox constants, only the layout differs.
            _, _, out[layout] = varlen_attention(q, k, v, seqlens_q, seqlens_k,
                                                 False, 1.0 / D_HEAD, 0.0,
                                                 varlen_type, layout, ext)
        _assert_lse_th_is_ht_transposed(out['TH'], out['HT'], seqlens_q,
                                        N_HEADS, varlen_type)


@pytest.mark.parametrize('varlen_type', ['compact', 'padded', 'strided'])
@pytest.mark.parametrize('lse_layout', ['HT', 'TH'])
def test_logsumexp_layout(gpu_id, varlen_type, lse_layout):
    """The logsumexp lands at the index its declared layout says it does.

    Modelled on test_logsumexp_scaling rather than on test_op_bwd's
    parameterization, and for one reason: a CLOSED-FORM expected value is what a
    self-referential `TH == HT.T` comparison lacks. Assertion 2 below is absolute
    and per-position, so it subsumes that pairwise check -- a pair of runs that
    are consistently wrong fails here where they would pass there.

    No dtype axis, unlike test_logsumexp_scaling which sweeps all three because
    *scaling* is where a dtype's exponent range matters. This is about
    *addressing*: L is fp32 whatever the inputs are, and the index arithmetic
    never sees the element type.
    """
    dtype = torch.float16
    num_heads = len(HEAD_SCALES)
    # 64 rather than 16: head dim 16 is excluded from the forward functional
    # inventory on gfx950 (see _attn_fwd_disabled), and nothing here depends on
    # the head dim -- the analytic value does not mention it.
    hdim = 64
    sm_scale = 1.0 / math.sqrt(hdim)

    with torch.cuda.device(gpu_id):
        device = 'cuda'
        q, k, v, seqlens, padlens, scales, sequence_slices = \
                _build_analytic_lse_inputs(varlen_type, num_heads, hdim, device, dtype)
        if varlen_type in SLOTTED_TYPES:
            seqlens_arg = np.array([seqlens, padlens])
        else:
            seqlens_arg = seqlens
        ext = AttentionExtraArgs(return_encoded_softmax=False,
                                 autotune=False,
                                 return_autotune=False,
                                 return_logsumexp=True)
        _, _, lse = varlen_attention(q, k, v, seqlens_arg, seqlens_arg,
                                     False, sm_scale, 0.0, varlen_type, lse_layout, ext)

        # Assertion 1: the SHAPE is what the layout declares. Cheap, and it is
        # what catches an allocation that never got the memo.
        if varlen_type == 'padded':
            tokens = int(np.max(seqlens))
            rows = len(seqlens) * tokens
            want = (rows, num_heads) if lse_layout == 'TH' else (len(seqlens) * num_heads, tokens)
        else:
            tokens = q.shape[2]
            want = (tokens, num_heads) if lse_layout == 'TH' else (num_heads, tokens)
        assert tuple(lse.shape) == want, f'{lse_layout} {varlen_type}: {lse.shape=} {want=}'

        # Reshape both layouts to a common (batch, head, token) view so the
        # per-position assertion is written once. Doing it this way and not by
        # transposing one into the other is the point: each layout is indexed by
        # its OWN formula, so the two cannot agree by construction.
        if varlen_type == 'padded':
            b_axis, t_axis = len(seqlens), int(np.max(seqlens))
            if lse_layout == 'TH':
                view = lse.view(b_axis, t_axis, num_heads).permute(0, 2, 1)
            else:
                view = lse.view(b_axis, num_heads, t_axis)
        else:
            if lse_layout == 'TH':
                view = lse.t().unsqueeze(0)
            else:
                view = lse.unsqueeze(0)

        # Assertion 2: every valid element equals its analytic value, at its
        # layout-specific index. Rows past a sequence's own length under 'padded'
        # (and the inter-sequence gaps under 'strided') are never written, so they
        # are excluded rather than asserted on.
        for z, (b, tok) in enumerate(sequence_slices()):
            expected_per_head = sm_scale * scales + math.log(int(seqlens[z]))
            got = view[b, :, tok]                    # (num_heads, seqlen_q)
            want_full = expected_per_head.unsqueeze(1).expand_as(got)
            assert torch.allclose(got, want_full, atol=1e-3, rtol=1e-3), (
                    f'{lse_layout} {varlen_type} sequence {z}: got {got} want {want_full}')


def main1():
    N_HEADS = 3
    D_HEAD = 8
    seqlens_q = np.array([ 4, 143, 128, 143, 143,])
    seqlens_k = np.array([ 8,  63,   8,  63,  63,])
    # seqlens_q = np.array([4, 8])
    # seqlens_k = seqlens_q
    causal = False
    sm_scale = 1.0 / 8.0
    # dropout_p = 0.5
    dropout_p = 0.0
    dtype = torch.float16
    # varlen_type = 'compact'
    varlen_type = 'padded'
    _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype, varlen_type)

def main2():
    N_HEADS = 3
    D_HEAD = 8
    # seqlens_q = np.array([ 4,  31, 8])
    # seqlens_k = np.array([ 8,  63, 8])
    # padlens_q = np.array([ 2,   3, 0])
    # padlens_k = np.array([ 5,   7, 0])
    seqlens_q = np.array([ 8, 8, 8])
    seqlens_k = np.array([ 8, 8, 8])
    padlens_q = np.array([ 0, 32, 0])
    padlens_k = np.array([ 0, 32, 0])
    causal = False
    seqlens_q = np.array([seqlens_q, padlens_q])
    seqlens_k = np.array([seqlens_k, padlens_k])
    sm_scale = 1.0 / 8.0
    dropout_p = 0.0
    dtype = torch.float16
    varlen_type = 'strided'
    _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype, varlen_type)


if __name__ == '__main__':
    main2()
