#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
from fwd_kernel import attn_fwd as bare_attn_fwd
from bwd_preprocess import bwd_preprocess
from bwd_split_kernel import bwd_kernel_dk_dv, bwd_kernel_dq

VERBOSE=False

def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0

def is_supported_by_tl_dot(n: int) -> bool:
    return is_power_of_two(n) and n >= 16

@triton.autotune(
   configs=[
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 4, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 4, 'pre_load_v': False}, num_stages=1, num_warps=4),
   ],
   key=['seqlen_q', 'seqlen_k', 'STAGE'],
)
@triton.jit
def tuned_attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    seqlen_q,
    seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
):
    bare_attn_fwd(
            Q, K, V, sm_scale, M, Out,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            stride_oz, stride_oh, stride_om, stride_on,
            seqlen_q,
            seqlen_k,
            dropout_p,
            philox_seed,
            philox_offset_base,
            encoded_softmax,
            STAGE,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            pre_load_v,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            )

class _attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dropout_p, return_encoded_softmax,
                autotune=False, return_autotune=False):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        o = torch.empty_like(q)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8

        stage = 3 if causal else 1
        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            q.shape[1],
            q.shape[0],
        )
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if return_encoded_softmax:
            encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=_attention.DEBUG_MASK_DTYPE)
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
            print(f'{stage=}')
            print(f'seqlen_q={q.shape[2]}')
            print(f'seqlen_k={k.shape[2]}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.dtype=}')

        philox_seed = 114514
        philox_offset = 1919810
        MAX_BLOCK_M = 128 if dropout_p == 0 else 64
        MAX_BLOCK_N = 32 if dropout_p == 0 else 32
        MAX_BLOCK_M = MAX_BLOCK_M if is_supported_by_tl_dot(seqlen_q) else 1
        MAX_BLOCK_N = MAX_BLOCK_N if is_supported_by_tl_dot(seqlen_k) else 1
        BLOCK_M=min(MAX_BLOCK_M, q.shape[2], k.shape[2])
        BLOCK_N=min(MAX_BLOCK_N, q.shape[2], k.shape[2])

        bare_attn_fwd[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            seqlen_q=q.shape[2],
            seqlen_k=k.shape[2],
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset_base=philox_offset,
            encoded_softmax=encoded_softmax,
            STAGE=stage,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK_N,
            pre_load_v=False,
            num_stages=1,
            num_warps=4,
            ENABLE_DROPOUT=dropout_p > 0.0,
            RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
        )
        print(f'{BLOCK_M=} {BLOCK_N=}')

        tuning_result = None
        block_m = min(128, q.shape[2], k.shape[2])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[1], q.shape[0])
        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        return o, encoded_softmax, tuning_result

    @staticmethod
    def backward(ctx, do, _, __):
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        # if q.shape[-1] <= 32:
        # do = do.contiguous()  # Not needed for newer bwd_preprocess
        # assert o.stride() == do.stride()
        assert q.stride() == do.stride() # Hard requirement, hard-coded in the triton kernel
        print(f'{o.stride()=}')
        print(f'{do.stride()=}')
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        MAX_BLOCK = 64 if ctx.dropout_p == 0 else 16
        # BLOCK = min(seqlen_q, seqlen_k, q.shape[-1], MAX_BLOCK)
        BLOCK = 16 # FIXME: Variable block size
        BLOCK = BLOCK if is_supported_by_tl_dot(seqlen_q) and is_supported_by_tl_dot(seqlen_k) else 1

        # grid configuration is (batch, head, seqlen_block)
        # block size is (BLOCK_M, D_HEAD)
        # bwd_preprocess[(do.shape[0] * do.shape[1] * triton.cdiv(do.shape[2], BLOCK), )](
        grid_prep = (triton.cdiv(do.shape[2], BLOCK), do.shape[1], do.shape[0])
        bwd_preprocess[grid_prep](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            seqlen_q,
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
            num_stages=1,
            num_warps=4,
        )
        if False or VERBOSE:
            print(f'{q.shape=} {q.stride()=}')
            print(f'{k.shape=} {k.stride()=}')
            print(f'{v.shape=} {v.stride()=}')
            print(f'{o.shape=} {o.stride()=}')
            print(f'{dq.shape=} {dq.stride()=}')
            print(f'{dk.shape=} {dk.stride()=}')
            print(f'{dv.shape=} {dv.stride()=}')
            print(f'{do.shape=} {do.stride()=}')
            print(f'{L=} {L.shape=}')
            print(f'{delta=}')
            print(f'{BLOCK=}')
        # debug_mask = torch.zeros((q.shape[0], q.shape[1], seqlen_q, seqlen_k), device=q.device, dtype=ctx.encoded_softmax.dtype)
        bwd_kernel_dk_dv[(triton.cdiv(seqlen_k, BLOCK), q.shape[1], q.shape[0])](
            q, k, v, ctx.sm_scale,
            o, do,
            dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset_base=ctx.philox_offset,
            # debug_mask=debug_mask,
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            CAUSAL=ctx.causal,
            num_warps=4,
            num_stages=1,
            ENABLE_DROPOUT=ctx.dropout_p > 0.0,
        )
        # mask_allclose = torch.allclose(debug_mask < 0, ctx.encoded_softmax < 0)
        if False:
            mask_allclose = torch.allclose(torch.abs(debug_mask), torch.abs(ctx.encoded_softmax)) # Stores QK
            if not mask_allclose:
                torch.set_printoptions(linewidth=200, threshold=2000)
                import sys
                print(f'bwd mask: {torch.abs(debug_mask[:,:,:2,16:])}')
                print(f'fwd mask: {torch.abs(ctx.encoded_softmax[:,:,:2,16:])}')
                print(f'Full bwd mask: {debug_mask[0,0]}')
                print(f'Full fwd mask: {ctx.encoded_softmax[0,0]}')
                print(f'Full mask div: {debug_mask[0,0] / ctx.encoded_softmax[0,0]}')
                print(f'Full dv: {dv}')
                if seqlen_q == 32:
                    print(f'2nd block bwd mask: {debug_mask[0,0, 16:]}')
                    print(f'2nd block fwd mask: {ctx.encoded_softmax[0,0, 16:]}')
            # print(f'Full q: {q}', file=sys.stderr)
            # assert mask_allclose
        if q.requires_grad:
            dq = torch.zeros_like(q)
            DQ_BLOCK_M = min(seqlen_q, BLOCK)
            bwd_kernel_dq[(triton.cdiv(q.shape[2], DQ_BLOCK_M), q.shape[1], q.shape[0])](
                q, k, v, ctx.sm_scale,
                o, do,
                dq,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset_base=ctx.philox_offset,
                BLOCK_M=DQ_BLOCK_M, BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                CAUSAL=ctx.causal,
                num_warps=4, waves_per_eu=1,
                num_stages=1,
                ENABLE_DROPOUT=ctx.dropout_p > 0.0,
            )
        else:
            dq = None
        # print(h.asm["ttgir"])
        return dq, dk, dv, None, None, None, None, None, None

attention = _attention.apply
