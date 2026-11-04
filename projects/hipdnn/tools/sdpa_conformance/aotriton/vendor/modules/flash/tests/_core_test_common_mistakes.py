#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch

from attn_torch_function import attention, AttentionExtraArgs
# Bottom-right causal cannot be spelled with a bool: translate_causal() maps
# True onto TOP_LEFT_ALIGNED.
from aotriton_flash import CausalType
# The backend's dtype set, which already drops fp32 where no kernel exists.
from _core_test_backward import DTYPES

# ---------------------------------------------------------------------------
# Common mistakes. See docs/attention-kernel-numerical-error-lessons/ for the
# history behind each case. modules/flash/kernel/test_backward.py carries the
# JIT twin used to show each case catches the mistake it names; keep the two in
# step.
#
# core_test_op_bwd (_core_test_backward.py) cannot see a precision mistake, for
# two reasons. It measures against PyTorch's own low-precision SDPA with fudge
# factors of 3x-14x on O and 36x-768x on the gradients. And at the sm_scale it
# uses, 1/D or 1/sqrt(D) over N(0, 1) inputs, the logits are too small for an
# error that grows with |S| to show at all. Baking sm_scale*log2(e) into Q
# costs 3x-15x of the floor below once the logits are large. These cases
# measure against that floor instead -- see _common_mistakes_reference.
# ---------------------------------------------------------------------------

COMMON_MISTAKES_CASES = ['sharp_softmax', 'zero_sm_scale', 'negative_sm_scale',
                         'bottom_right_masked_rows']

# fp32 is left out on purpose: every mistake here is a rounding to the input
# dtype, which fp32 inputs make harmless, and the floor model does not capture
# fp32 accumulation error.
COMMON_MISTAKES_DTYPES = [dtype for dtype in DTYPES if dtype != torch.float32]

# A correct kernel lands at or below 1.0x the floor (the Triton kernels on
# gfx90a: 0.7x-1.02x for O, dQ, dK and dV over six seeds). The mistakes
# measured 2.6x-17x on the same inputs, so 2x is both honest and wide.
COMMON_MISTAKES_FLOOR_MULT = 2.0

def _common_mistakes_reference(q, k, v, dout, sm_scale, mask, round_to=None, flush_subnormals=False):
    '''fp64 attention forward and backward from the same low-precision inputs.

    With round_to=None this is the exact answer. With round_to=dtype it rounds
    to dtype exactly where a correct flash kernel has to -- P before P@V and
    P^T@dO, dS before dS@K and dS^T@Q, and every output on store -- and nowhere
    else. Its distance from the exact answer is therefore the error floor: what
    a kernel that keeps everything else in fp32 still gets.

    flush_subnormals also zeroes whatever rounds below the dtype's smallest
    normal. gfx90a's fp16 MFMA flushes subnormal inputs, which is where most of
    P lands once seqlen_k is in the hundreds; without it fp16 dV measures
    2.5x a floor that no hardware-conforming kernel can reach.

    A fully masked row has no softmax. Its contract is P = 0 there, so O = 0 and
    dQ = 0, and the LSE is -inf here (the kernel may store +inf).
    '''
    q, k, v, dout = (t.detach().to(torch.float64) for t in (q, k, v, dout))
    def R(x):
        if round_to is None:
            return x
        x = x.to(round_to).to(torch.float64)
        if flush_subnormals:
            x = torch.where(x.abs() < torch.finfo(round_to).tiny, 0.0, x)
        return x
    s = (q @ k.transpose(-1, -2)) * sm_scale
    s = s.masked_fill(~mask, float('-inf'))
    lse = torch.logsumexp(s, dim=-1, keepdim=True)
    p = torch.where(mask, torch.exp(s - lse), 0.0)
    o = R(R(p) @ v)
    dp = dout @ v.transpose(-1, -2)
    # From the O the kernel stored, which is what its backward reads.
    delta = (dout * o).sum(dim=-1, keepdim=True)
    ds = p * (dp - delta)
    dq = R((R(ds) @ k) * sm_scale)
    dk = R((R(ds).transpose(-1, -2) @ q) * sm_scale)
    dv = R(R(p).transpose(-1, -2) @ dout)
    return o, lse.squeeze(-1), dq, dk, dv

def _common_mistakes_case(case):
    '''(seqlen_q, seqlen_k, input scale, causal, sm_scale) for one case.'''
    D_HEAD = 128
    if case == 'sharp_softmax':
        # Logits with a standard deviation of 16: a trained model's attention
        # is this sharp, and the error of rounding a scaled Q or S to the input
        # dtype grows with |S| while the floor does not.
        return 257, 519, 4.0, False, D_HEAD ** -0.5
    if case == 'zero_sm_scale':
        # Uniform attention over a causal prefix. Any kernel that multiplies a
        # masked -inf by the scale computes -inf * 0 = NaN.
        return 257, 519, 1.0, True, 0.0
    if case == 'negative_sm_scale':
        # Nothing requires sm_scale > 0. A kernel that multiplies a masked
        # -inf by a negative scale computes +inf, which exp2 turns into inf.
        return 257, 519, 1.0, True, -D_HEAD ** -0.5
    if case == 'bottom_right_masked_rows':
        # The first 111 query rows attend to nothing, and 111 is off the grid
        # of every BLOCK_M, so fully masked rows share a tile with live ones
        # and cannot take the whole-tile early exit that
        # core_test_bottom_right_fully_masked_rows exercises.
        return 301, 190, 1.0, CausalType.BOTTOM_RIGHT, D_HEAD ** -0.5
    assert False, f'Unknown case {case}'

def _common_mistakes_mask(causal, seqlen_q, seqlen_k, device):
    i = torch.arange(seqlen_q, device=device)[:, None]
    j = torch.arange(seqlen_k, device=device)[None, :]
    if causal is False:
        return torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=device)
    if causal is True:
        return j <= i
    assert causal == CausalType.BOTTOM_RIGHT
    return j <= i + (seqlen_k - seqlen_q)

def core_test_common_mistakes(case, dtype, device_str='cuda'):
    BATCH, N_HEADS, D_HEAD = 2, 3, 128
    seqlen_q, seqlen_k, input_scale, causal, sm_scale = _common_mistakes_case(case)
    torch.manual_seed(0)
    def randn(seqlen, scale):
        return (torch.randn(BATCH, N_HEADS, seqlen, D_HEAD, device=device_str) * scale).to(dtype)
    q = randn(seqlen_q, input_scale).requires_grad_()
    k = randn(seqlen_k, input_scale).requires_grad_()
    v = randn(seqlen_k, 1.0).requires_grad_()
    dout = randn(seqlen_q, 1.0)

    # is_testing=False so a NaN LSE reaches the diagnostics below instead of
    # the wrapper's own assert. fillnan turns an unwritten element into a NaN
    # rather than whatever the allocator left.
    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False,
                             is_testing=False,
                             fillnan=True,
                             return_logsumexp=True)
    tri_out, _, L = attention(q, k, v, None, causal, sm_scale, 0.0, ext)
    tri_dq, tri_dk, tri_dv = torch.autograd.grad(tri_out, (q, k, v), dout)
    tri_lse = L.view(BATCH, N_HEADS, seqlen_q)

    mask = _common_mistakes_mask(causal, seqlen_q, seqlen_k, device_str)
    live_rows = mask.any(dim=-1)
    exact = _common_mistakes_reference(q, k, v, dout, sm_scale, mask)
    floors = [_common_mistakes_reference(q, k, v, dout, sm_scale, mask, round_to=dtype, flush_subnormals=ftz)
              for ftz in (False, True)]

    ctx = f'{case=} {dtype=} {seqlen_q=} {seqlen_k=} {sm_scale=}'
    names = ['out', 'lse', 'dq', 'dk', 'dv']
    tri = [tri_out, tri_lse, tri_dq, tri_dk, tri_dv]
    for name, t in zip(names, tri):
        assert not torch.isnan(t).any(), f'{ctx}: {name} has NaN'

    # **LSE, to fp32 accuracy.** It is a row reduction of the scores and
    # nothing in it is rounded to the input dtype, so a correct kernel is off
    # by ~1e-6 relative. Rounding a scaled Q (or S itself) to fp16 costs 2**-11
    # of the largest score, bf16 2**-8; a base-2 LSE is off by a factor of
    # 1.44. The bound sits between: 2**-16 of the largest score.
    ref_lse = exact[1]
    s_max = max(1.0, ref_lse[..., live_rows].abs().max().item())
    lse_err = (tri_lse[..., live_rows].double() - ref_lse[..., live_rows]).abs().max().item()
    assert lse_err <= 2.0 ** -16 * s_max, \
        f'{ctx}: LSE off by {lse_err:.3e}, bound {2.0 ** -16 * s_max:.3e} (max |S| {s_max:.1f})'

    # **Fully masked rows are exactly zero.** Rows with no key get P = 0, so O
    # and dQ vanish -- not NaN, not the inits leaking out.
    if not live_rows.all():
        dead = ~live_rows
        for name, t in (('out', tri_out), ('dq', tri_dq)):
            n_nonzero = int((t[:, :, dead] != 0).sum())
            assert n_nonzero == 0, f'{ctx}: {n_nonzero} nonzero elements in fully masked {name} rows'

    # **O and every gradient, against the floor.**
    def relrms(x, ref):
        return ((x.double() - ref).norm() / ref.norm()).item()
    for i in (0, 2, 3, 4):
        name, t, ref = names[i], tri[i], exact[i]
        if ref.norm() == 0:
            # sm_scale == 0 has an exactly zero dQ and dK.
            n_nonzero = int((t != 0).sum())
            assert n_nonzero == 0, f'{ctx}: {name} should be exactly zero, {n_nonzero} elements are not'
            continue
        err = relrms(t, ref)
        floor = max(relrms(f[i], ref) for f in floors)
        assert err <= COMMON_MISTAKES_FLOOR_MULT * floor, \
            f'{ctx}: {name} error {err:.3e} is {err / floor:.1f}x the floor {floor:.3e} (limit {COMMON_MISTAKES_FLOOR_MULT}x)'
