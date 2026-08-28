#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side builder for the gfx942 dense flash-attention prefill kernel
(``kernels/gfx942/attention_dense.py``).

Provides the kernel compilation and launch path.  CLI/harness helpers
(resolve_dense_spec, dense_request, etc.) live in
``dispatch.attention.gfx942`` because they depend on dispatch-layer symbols;
import them from there when you need them.

Usage (as a standalone harness, run from the benchmark):
    python benchmarks/gfx942/attention/prefill/benchmark_dense_prefill_live.py
"""
import math
import os
import sys

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from kernels.gfx942.attention_dense import (  # noqa: E402
    AttentionDenseSpec,
    attention_dense_block,
    attention_dense_grid,
    attention_dense_signature,
    build_attention_dense,
    gfx942_kernel_name,
    supports_attention_dense,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig  # noqa: E402

_ARCH = "gfx942"
_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------- #
# compile + launch
# --------------------------------------------------------------------------- #
def _make_launcher(spec: AttentionDenseSpec):
    """kernel-spec generation + compilation + ABI signature -> cached launcher."""
    ok, why = supports_attention_dense(spec, arch=_ARCH)
    if not ok:
        raise ValueError(f"unsupported spec: {why}")
    art = compile_kernel(
        build_attention_dense(spec, arch=_ARCH),
        arch=_ARCH,
        backend="python",
        capture_ir_text=False,
    )
    return KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=attention_dense_signature(spec),
    )


def _launch_config(spec: AttentionDenseSpec, stream) -> LaunchConfig:
    # Geometry is owned by the kernel module (it also handles the persistent grid),
    # never re-derived here -- a third copy of BLOCK_M would defeat the import-time
    # binding in attention_dense.py.
    return LaunchConfig(
        grid=attention_dense_grid(spec),
        block=attention_dense_block(spec),
        stream=stream,
    )


def run(
    spec: AttentionDenseSpec,
    *,
    warmup: int = 15,
    iters: int = 50,
    check: bool = True,
    label: "str | None" = None,
):
    """Compile, launch, parity-check, and benchmark ``spec`` on the current device.

    ``label`` is printed in the result line; defaults to ``gfx942_kernel_name(spec)``.
    """
    dev = "cuda"
    dt = _TORCH_DT[spec.dtype]
    B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    Hq, Hkv, D = spec.num_query_heads, spec.num_kv_heads, spec.head_size
    torch.manual_seed(0)
    q = (torch.randn(B, Sq, Hq, D, dtype=dt, device=dev) * 0.2).contiguous()
    k = (torch.randn(B, Skv, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
    v = (torch.randn(B, Skv, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
    out = torch.zeros(B, Sq, Hq, D, dtype=dt, device=dev)
    scale = 1.0 / math.sqrt(D)

    launcher = _make_launcher(spec)
    stream = torch.cuda.current_stream().cuda_stream
    cfg = _launch_config(spec, stream)
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": scale}

    def call():
        launcher(vals, config=cfg)

    call()
    torch.cuda.synchronize()

    err = float("nan")
    if check:
        qh = q.transpose(1, 2).float()
        rep = Hq // Hkv
        kh = k.transpose(1, 2).repeat_interleave(rep, 1).float()
        vh = v.transpose(1, 2).repeat_interleave(rep, 1).float()
        W = spec.sliding_window
        if spec.causal and W > 0:
            qi = torch.arange(Sq, device=dev).view(-1, 1)
            ki = torch.arange(Skv, device=dev).view(1, -1)
            allowed = (ki <= qi) & (ki > qi - W)
            ref = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=allowed
            ).transpose(1, 2)
        else:
            ref = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=spec.causal
            ).transpose(1, 2)
        err = (out.float() - ref).abs().max().item()

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call()
    e.record()
    e.synchronize()
    ms = s.elapsed_time(e) / iters
    W = spec.sliding_window
    if spec.causal and W > 0:
        pairs = W * Sq - W * (W - 1) // 2 if Sq >= W else Sq * (Sq + 1) // 2
        flops = 4 * B * Hq * D * pairs
    elif spec.causal:
        flops = 4 * B * Hq * D * (Sq * (Sq + 1) // 2)
    else:
        flops = 2 * 2 * B * Hq * D * Sq * Skv
    tf = flops / (ms * 1e-3) / 1e12
    status = "PASS" if (not check or err < 2e-2) else "FAIL"
    tag = label if label is not None else gfx942_kernel_name(spec)
    print(f"[{tag}] {ms:.4f} ms  {tf:.1f} TFLOPS  max_abs={err:.2e}  {status}")
    return ms, tf, err
