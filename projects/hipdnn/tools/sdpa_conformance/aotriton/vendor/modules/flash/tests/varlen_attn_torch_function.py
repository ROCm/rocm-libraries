#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import numpy as np
from aotriton_flash import (
    attn_fwd_varlen,
    attn_bwd_varlen,
    attn_options,
    hipError_t,
)
from attn_torch_function import (
    AttentionExtraArgs,
    FWD_IMPL_IDX,
    BWD_IMPL_IDX,
    PROBE_UNSUPPORTED,
    FWD_IMPL,
    BWD_IMPL,
)

VERBOSE=False
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET_1 = 0x1D4000
DEFAULT_PHILOX_OFFSET_2 = 0x000B42
DEFAULT_PHILOX_OFFSET = DEFAULT_PHILOX_OFFSET_1 + DEFAULT_PHILOX_OFFSET_2

from aotriton_flash import lazy_dq_acc, lazy_delta

def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0

def is_supported_by_tl_dot(n: int) -> bool:
    return is_power_of_two(n) and n >= 16

class _attention_varlen(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    # DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, seqlens_q, seqlens_k, causal, sm_scale, dropout_p,
                varlen_type, lse_layout='HT',
                attn_extra_args=AttentionExtraArgs()):
        assert lse_layout in ('HT', 'TH'), f'unknown lse_layout {lse_layout}'
        return_encoded_softmax = attn_extra_args.return_encoded_softmax
        autotune = attn_extra_args.autotune
        return_autotune = attn_extra_args.return_autotune
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk
        # assert Lk in {16, 32, 64, 128}
        if varlen_type in ('strided', 'seqused'):
            # Both take (seqlens, padlens) pairs and lay the tensors out the
            # same way: slot z occupies seqlens[z] + padlens[z] rows, of which
            # only the first seqlens[z] participate. What differs is how the
            # kernel is TOLD -- strided reads lengths cumulatively, seqused
            # reads them individually -- which is a bits difference, not a
            # layout one, so one context class serves both.
            seqlens_q, padlens_q = seqlens_q
            seqlens_k, padlens_k = seqlens_k
            total_seqlen_q = int(np.sum(seqlens_q + padlens_q))  # Be explicit
        else:
            total_seqlen_q = int(np.sum(seqlens_q))
        batch = len(seqlens_q)
        num_heads = q.shape[1]
        max_seqlen_q = int(np.max(seqlens_q))
        max_seqlen_k = int(np.max(seqlens_k))
        seqinfo_q0 = np.cumsum(seqlens_q)
        seqinfo_k0 = np.cumsum(seqlens_k)
        seqinfo_q0 = torch.tensor([0] + seqinfo_q0.tolist(), dtype=torch.int32, device=q.device)
        seqinfo_k0 = torch.tensor([0] + seqinfo_k0.tolist(), dtype=torch.int32, device=q.device)
        if varlen_type in ['compact', 'padded']:
            seqinfo_q1 = None
            seqinfo_k1 = None
        elif varlen_type == 'strided':
            seqinfo_q1 = np.cumsum(seqlens_q + padlens_q)
            seqinfo_k1 = np.cumsum(seqlens_k + padlens_k)
            seqinfo_q1 = torch.tensor([0] + seqinfo_q1.tolist(), dtype=torch.int32, device=q.device)
            seqinfo_k1 = torch.tensor([0] + seqinfo_k1.tolist(), dtype=torch.int32, device=q.device)
        elif varlen_type == 'seqused':
            # torch.nn.attention.varlen.varlen_attn(..., seqused_k=...):
            #   cu_seq_q   (N+1,)  -> seqinfo_q0, Q length AND position (REUSE)
            #   seqused_k  (N,)    -> seqinfo_k0, K length only (INDIVIDUAL)
            #   cu_seq_k   (N+1,)  -> seqinfo_k1, K position only  (ARRAY)
            # The K side therefore reads its two facts from two DIFFERENT
            # tensors, which is why seqinfo_?0/?1 are named by role rather than
            # by mode. seqinfo_k0 is rebuilt below as the individual lengths;
            # here we only supply the position array.
            assert np.all(padlens_q == 0), \
                'seqused models a packed Q against a slotted KV cache'
            seqinfo_q1 = None
            seqinfo_k1 = np.cumsum(seqlens_k + padlens_k)
            seqinfo_k1 = torch.tensor([0] + seqinfo_k1.tolist(), dtype=torch.int32, device=q.device)
            # (N,), not (N+1,): INDIVIDUAL means seqinfo_k0[z] IS the length.
            seqinfo_k0 = torch.tensor(np.asarray(seqlens_k).tolist(),
                                      dtype=torch.int32, device=q.device)
        else:
            assert False
        o = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), device=q.device, dtype=q.dtype)
        b = torch.empty((0,0,0,0), device=q.device, dtype=q.dtype)

        # The logsumexp tensor carries NO strides to the kernel: it is always
        # compact, so its layout is fully determined by lse_layout plus the head
        # count and the token pitch, and passing strides alongside would be two
        # sources of truth for one fact. The host's job is therefore to allocate
        # exactly the shape the bits declare -- checking, not inferring.
        if varlen_type == 'padded':
            lse_shape = ((batch * max_seqlen_q, num_heads) if lse_layout == 'TH'
                         else (batch * num_heads, max_seqlen_q))
        else:
            lse_shape = ((total_seqlen_q, num_heads) if lse_layout == 'TH'
                         else (num_heads, total_seqlen_q))
        # zeros for padded (preserved from before this change): rows past a
        # sequence's own length are never written. fillnan below still overrides
        # it when asked, so any comparison on this buffer must mask to the rows
        # each sequence actually owns rather than trusting the whole thing.
        M = (torch.zeros if varlen_type == 'padded' else torch.empty)(
                lse_shape, device=q.device, dtype=torch.float32)
        if attn_extra_args.fillnan:
            for t in (o, M):
                t.fill_(float('nan'))
        if return_encoded_softmax:
            encoded_softmax = torch.zeros((batch, num_heads, max_seqlen_q, max_seqlen_k), device=q.device, dtype=q.dtype)
        else:
            encoded_softmax = None
        if False or VERBOSE:
            print(f'{q.shape=}')
            print(f'{k.shape=}')
            print(f'{v.shape=}')
            print(f'{o.shape=}')
            print(f'{q.data_ptr()=:x}')
            print(f'{k.data_ptr()=:x}')
            print(f'{v.data_ptr()=:x}')
            print(f'{M.data_ptr()=:x}')
            print(f'{o.data_ptr()=:x}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.dtype=}')

        philox_null = torch.empty([0], device=q.device, dtype=torch.uint64)
        if dropout_p > 0.0:
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=q.device, dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=q.device, dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            philox_seed_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            philox_offset_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
        else:
            philox_seed = philox_null
            philox_offset1 = philox_null
            philox_offset2 = 0
            philox_seed_output = philox_null
            philox_offset_output = philox_null

        if causal:
            atomic = torch.zeros([1], device=q.device, dtype=torch.int32)
        else:
            atomic = torch.empty([0], device=q.device, dtype=torch.int32)

        if FWD_IMPL is not None:
            extargs = attn_options()
            extargs.force_backend_index = FWD_IMPL_IDX
        else:
            extargs = None

        attn_fwd_varlen(q, k, v,
                        seqinfo_q0, seqinfo_k0, max_seqlen_q, max_seqlen_k,
                        seqinfo_q1, seqinfo_k1,
                        b, sm_scale, M, o,
                        dropout_p, philox_seed, philox_offset1, philox_offset2,
                        philox_seed_output, philox_offset_output,
                        encoded_softmax, causal, atomic, varlen_type, lse_layout, extargs)

        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.seqlens_q = seqlens_q
        ctx.seqlens_k = seqlens_k
        ctx.seqinfo_q0 = seqinfo_q0
        ctx.seqinfo_k0 = seqinfo_k0
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.seqinfo_q1 = seqinfo_q1
        ctx.seqinfo_k1 = seqinfo_k1
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed_output
        ctx.philox_offset = philox_offset_output
        ctx.philox_offset1 = philox_offset1
        ctx.philox_offset2 = philox_offset2
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.varlen_type = varlen_type
        ctx.lse_layout = lse_layout
        ctx.attn_extra_args = attn_extra_args
        # Honour return_logsumexp, as the dense counterpart already does
        # (attn_torch_function.py). Without it a test cannot see L at all, and L
        # is the only place the lse_layout is directly observable.
        ret3 = M if attn_extra_args.return_logsumexp else None
        return o, encoded_softmax, ret3

    @staticmethod
    def backward(ctx, do, _, __):
        q, k, v, b, o, L = ctx.saved_tensors
        print(f'{b=}')
        seqlens_q = ctx.seqlens_q
        seqlens_k = ctx.seqlens_k
        seqinfo_q0 = ctx.seqinfo_q0
        seqinfo_k0 = ctx.seqinfo_k0
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        batch = len(seqlens_q)
        sm_scale = ctx.sm_scale
        dropout_p = ctx.dropout_p
        philox_seed = ctx.philox_seed
        philox_offset = ctx.philox_offset
        causal = ctx.causal
        # if q.shape[-1] <= 32:
        # do = do.contiguous()
        dq = torch.empty_like(q)
        dq_acc = lazy_dq_acc(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b) if b is not None else None
        if ctx.attn_extra_args.fillnan:
            for t in (dq, dk, dv, db):
                if t is not None:
                    t.fill_(float('nan'))
        delta = lazy_delta(L)
        if BWD_IMPL is not None:
            extargs = attn_options()
            extargs.force_backend_index = BWD_IMPL_IDX
        else:
            extargs = None
        ret = attn_bwd_varlen(q, k, v,
                              seqinfo_q0, seqinfo_k0, max_seqlen_q, max_seqlen_k,
                              ctx.seqinfo_q1, ctx.seqinfo_k1,
                              b, sm_scale, o, do, dq, dk, dv, db, dq_acc, L, delta,
                              dropout_p, philox_seed, philox_offset, 0, causal, ctx.varlen_type,
                              ctx.lse_layout, extargs);
        if PROBE_UNSUPPORTED and ret == hipError_t.hipErrorPeerAccessUnsupported:
            raise NotImplementedError()
        assert ret == hipError_t.hipSuccess, ret
        # One entry per forward input after ctx: q, k, v, seqlens_q, seqlens_k,
        # causal, sm_scale, dropout_p, varlen_type, lse_layout, attn_extra_args.
        return dq, dk, dv, None, None, None, None, None, None, None, None

varlen_attention = _attention_varlen.apply
