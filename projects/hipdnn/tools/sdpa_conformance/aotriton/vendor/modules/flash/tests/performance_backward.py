#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import pytest
import torch

import triton
from _perf_report import run_report
from collections import defaultdict
from attn_torch_function import attention, AttentionExtraArgs

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None
USE_TFLOPS = bool(int(os.getenv('USE_TFLOPS', default='1')))
print(f'{USE_TFLOPS=}')

d_heads = os.getenv('D_HEADS', default='64,128')
d_heads = list(map(lambda x: int(x), d_heads.split(',')))

n_ctx = os.getenv('N_CTX', default=list(range(10, 14)))
if isinstance(n_ctx, str):
    n_ctx = map(lambda x: int(x), n_ctx.split(','))
X_VALS = list(map(lambda x: 2 ** x, n_ctx))
x_vals = os.getenv('X_VALS', default=None)
if x_vals is not None:
    X_VALS = [int(e) for e in x_vals.split(',')]
print(f'{X_VALS=}')

def _get_modename():
    # Always 'V3', and has been since the V2 path went away. The per-backend
    # label belongs on the LINE (see BACKEND_INDEX below), not on the run;
    # perfmon supersedes this file rather than fixing it.
    return 'V3'

# One line per AOTriton backward BACKEND. See performance_forward.py for the
# rationale; the only difference here is that there are three or four of them
# (triton_split, triton_fuse, aiter, flyc) rather than two, and that BWD_IMPL's
# process-wide pin is what this replaces.
#
# Only the BACKWARD backend varies. The forward runs once, outside do_bench's
# timed region, on whichever backend the operator selects -- held constant
# across lines on purpose, so a difference here is the backward's.
from pyaotriton.v3.flash import OpAttnBwdBackend
BACKEND_INDEX = {name: i for i, name in OpAttnBwdBackend.by_index.items()}

def _aotriton_backends():
    published = list(BACKEND_INDEX)
    want = os.getenv('BACKENDS', default=None)
    if want is None:
        return published
    want = want.split(',')
    missing = [w for w in want if w not in published]
    assert not missing, f'BACKENDS={want} names {missing}, but this build publishes {published}'
    return want

AOTRITON_BACKENDS = _aotriton_backends()
print(f'{AOTRITON_BACKENDS=}')

_UNIT = 'TFLOPS' if USE_TFLOPS else 'ms'
_LINE_COLORS = ['red', 'blue', 'green', 'purple', 'orange']
_LINE_DASHES = ['-', '--', ':']


def _line_styles(n):
    """`n` distinct (color, dash) pairs, cycling instead of truncating.

    The old `_LINE_STYLES[:n]` silently returned FEWER than `n` entries once the
    backend list outgrew it, and triton.testing zips styles against line_vals
    positionally -- so the shortfall is an IndexError at plot time, after the
    whole sweep has run. Backward is already at the five-entry limit (four
    AOTriton backends plus flash), so one more backend was enough.
    """
    return [(_LINE_COLORS[i % len(_LINE_COLORS)],
             _LINE_DASHES[(i // len(_LINE_COLORS)) % len(_LINE_DASHES)])
            for i in range(n)]

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# BATCH, N_HEADS, N_CTX, D_HEAD = 512, 32, 512, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ['bwd']:
    modename = _get_modename()
    # for causal in [False, True]:
    for causal in [False]:
        for D_HEAD in d_heads:
            configs.append(triton.testing.Benchmark(
                x_names=['N_CTX'],
                x_vals=list(X_VALS),
                # x_vals=[2**i for i in range(10, 15)],
                # x_vals=[2**13],
                line_arg='provider',
                line_vals=list(AOTRITON_BACKENDS) + (['flash'] if HAS_FLASH else []),
                line_names=[f'{b}({_UNIT})' for b in AOTRITON_BACKENDS]
                           + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
                styles=_line_styles(len(AOTRITON_BACKENDS) + (1 if HAS_FLASH else 0)),
                ylabel='TFLOPS' if USE_TFLOPS else 'ms',
                plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{modename}-causal={causal}',
                args={
                    'H': N_HEADS,
                    'BATCH': BATCH,
                    'D_HEAD': D_HEAD,
                    'dtype': torch.float16,
                    'mode': mode,
                    'causal': causal,
                    })
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    # print(f"{N_CTX=}")
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 200
    split_kernel = False
    # Bwd pass only supports causal=True right now
    if mode == 'bwd':
        split_kernel = True if causal else split_kernel
    if provider in AOTRITON_BACKENDS:
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        dropout_p = 0.0
        ext = AttentionExtraArgs(return_encoded_softmax=causal,
                autotune=False,
                return_autotune=False,
                is_testing=False,
                force_bwd_backend_index=BACKEND_INDEX.get(provider))
        fn = lambda: attention(q, k, v, None, causal, sm_scale, dropout_p, ext)
        if mode == 'bwd':
            o, _, _ = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        if FLASH_VER == 1:
            lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
            fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
        elif FLASH_VER == 2:
            fn = lambda: flash_attn_func(qkv, causal=causal)
        else:
            raise ValueError(f'unknown {FLASH_VER = }')
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    if USE_TFLOPS:
        return total_flops / ms * 1e-9
    else:
        return ms


# only works on post-Ampere GPUs right now
# save_path=None skips the plot, which needs matplotlib. Set SAVE_PLOT=. (or any
# directory) to draw one; the numbers are printed either way.
run_report(bench_flash_attention, save_path=os.getenv("SAVE_PLOT", default=None))
