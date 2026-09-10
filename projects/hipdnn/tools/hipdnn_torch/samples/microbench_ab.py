#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
microbench_ab.py -- per-op A/B of the hipDNN path vs stock PyTorch, one shape at a
time.

For each shape it routes the functional through hipDNN (``install``), checks numeric
parity against native, times both, and reports whether the call actually landed on
the engine or fell back (and why). It is the honest per-kernel picture -- the shipped
kernels are correctness-first, so do NOT expect a blanket speedup; the value is
parity + the fallback reasons.

Select ops with ``--ops`` and override the shape sweeps with the ``MB_*`` env vars
(see ``--help``). Defaults are LTX-Video-ish. Run::

    HIPDNN_TORCH_PROVIDER_SO=<...>/libhip_kernel_provider.so \
        python samples/microbench_ab.py --ops linear,rmsnorm,sdpa
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402

# LTX-ish defaults; every dim is a multiple of 16 (the reference-kernel constraint).
_LINEAR_SHAPES = [
    (64, 64, 64),  # M, N, K -- matches the substrate parity band
    (256, 256, 256),
    (4096, 2048, 2048),  # attn projection: tokens x hidden x hidden
    (4096, 8192, 2048),  # FFN up
    (4096, 2048, 8192),  # FFN down
]
_RMSNORM_SHAPES = [
    (64, 64),  # M (tokens), N (features)
    (4096, 2048),
]
_SDPA_SHAPES = [
    (1, 32, 32, 32, 64),  # B, H, Sq, Skv, D (small square)
    (1, 32, 64, 48, 64),  # asymmetric (cross-attn in miniature)
    (1, 32, 4096, 4096, 64),  # self-attention
    (1, 32, 4096, 128, 64),  # cross-attention
]
_LAYERNORM_SHAPES = [
    (64, 64),  # M (tokens), N (features)
    (4096, 2048),
]
_ACT_SHAPES = [
    (64, 64),  # M, N -- flattened to numel for the pointwise kernel
    (4096, 8192),  # FFN activation width
]
_CONV_SHAPES = [
    # N, C, H, W, K, R, S -- LTX-Video VAE-ish channels-last convs.
    (1, 32, 128, 128, 32, 3, 3),
    (1, 128, 64, 64, 128, 3, 3),
]


def _time_ms(torch, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def _ab(torch, op, call, ref_native, tag, warmup, iters, dtype):
    """Run one shape: parity (hipDNN vs native) + timing + routed/fell-back."""
    rel = 5e-2 if dtype == torch.bfloat16 else 2e-2
    ov = hipdnn_torch.overrides()[op]

    want = ref_native().float()

    hipdnn_torch.install([op])
    ov.reset()
    got = call().float()
    aot, native = ov.totals()
    hipdnn_torch.uninstall([op])

    if aot == 0:  # never reached the engine
        reason = next(iter(ov.fallback_reasons()), "not applicable")
        print(f"{tag}  fell back to native ({reason})")
        return

    max_err = float((got - want).abs().max().item())
    close = max_err <= max(rel, rel * float(want.abs().max().item()))

    hipdnn_torch.install([op])
    aot_ms = _time_ms(torch, call, warmup, iters)
    hipdnn_torch.uninstall([op])
    pt_ms = _time_ms(torch, ref_native, warmup, iters)
    speedup = pt_ms / aot_ms if aot_ms > 0 else float("nan")

    print(
        f"{tag}  {'OK ' if close else 'BAD'} maxerr={max_err:8.5f}  "
        f"hipDNN={aot_ms*1e3:8.2f}us  PyTorch={pt_ms*1e3:8.2f}us  "
        f"speedup={speedup:5.2f}x"
    )


def _run_linear(torch, shapes, warmup, iters, dtype):
    import torch.nn.functional as F

    dev = torch.device("cuda")
    tok = {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
    print(f"=== linear (RCR / nn.Linear), {tok} ===")
    for m, n, k in shapes:
        torch.manual_seed(0)
        x = torch.randn(m, k, dtype=dtype, device=dev) * 0.1
        w = torch.randn(n, k, dtype=dtype, device=dev) * 0.1  # nn.Linear [N,K]
        tag = f"{tok:4s} M={m:<6} N={n:<6} K={k:<5}"
        _ab(
            torch,
            "linear",
            lambda: F.linear(x, w),
            lambda: F.linear(x, w),
            tag,
            warmup,
            iters,
            dtype,
        )
    print()


def _run_rmsnorm(torch, shapes, warmup, iters, dtype):
    import torch.nn.functional as F

    dev = torch.device("cuda")
    tok = {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
    print(f"=== rms_norm, {tok} ===")
    for m, n in shapes:
        torch.manual_seed(0)
        x = torch.randn(m, n, dtype=dtype, device=dev) * 0.1
        w = torch.randn(n, dtype=dtype, device=dev) * 0.1
        eps = 1e-6
        tag = f"{tok:4s} M={m:<6} N={n:<6}"
        _ab(
            torch,
            "rmsnorm",
            lambda: F.rms_norm(x, (n,), w, eps),
            lambda: F.rms_norm(x, (n,), w, eps),
            tag,
            warmup,
            iters,
            dtype,
        )
    print()


def _run_sdpa(torch, shapes, warmup, iters, dtype):
    import torch.nn.functional as F

    dev = torch.device("cuda")
    tok = {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
    print(f"=== scaled_dot_product_attention, {tok} ===")
    for b, h, sq, skv, d in shapes:
        torch.manual_seed(0)
        q = torch.randn(b, h, sq, d, dtype=dtype, device=dev) * 0.1
        k = torch.randn(b, h, skv, d, dtype=dtype, device=dev) * 0.1
        v = torch.randn(b, h, skv, d, dtype=dtype, device=dev) * 0.1
        scale = 1.0 / math.sqrt(d)
        tag = f"{tok:4s} B={b} H={h:<3} Sq={sq:<5} Skv={skv:<5} D={d:<3}"
        _ab(
            torch,
            "sdpa",
            lambda: F.scaled_dot_product_attention(q, k, v, scale=scale),
            lambda: F.scaled_dot_product_attention(q, k, v, scale=scale),
            tag,
            warmup,
            iters,
            dtype,
        )
    print()


def _run_layernorm(torch, shapes, warmup, iters, dtype):
    import torch.nn.functional as F

    dev = torch.device("cuda")
    tok = {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
    print(f"=== layer_norm, {tok} ===")
    for m, n in shapes:
        torch.manual_seed(0)
        x = torch.randn(m, n, dtype=dtype, device=dev) * 0.1
        w = torch.randn(n, dtype=dtype, device=dev) * 0.1
        b = torch.randn(n, dtype=dtype, device=dev) * 0.1
        eps = 1e-5
        tag = f"{tok:4s} M={m:<6} N={n:<6}"
        _ab(
            torch,
            "layernorm",
            lambda: F.layer_norm(x, (n,), w, b, eps),
            lambda: F.layer_norm(x, (n,), w, b, eps),
            tag,
            warmup,
            iters,
            dtype,
        )
    print()


def _run_activation(torch, op, fn, shapes, warmup, iters, dtype):
    import torch.nn.functional as F  # noqa: F401 -- fn closes over F

    dev = torch.device("cuda")
    tok = {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
    print(f"=== {op}, {tok} ===")
    for m, n in shapes:
        torch.manual_seed(0)
        x = torch.randn(m, n, dtype=dtype, device=dev) * 0.1
        tag = f"{tok:4s} M={m:<6} N={n:<6}"
        _ab(torch, op, lambda: fn(x), lambda: fn(x), tag, warmup, iters, dtype)
    print()


def _run_conv2d(torch, shapes, warmup, iters, dtype):
    import torch.nn.functional as F

    dev = torch.device("cuda")
    tok = {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
    print(f"=== conv2d (fprop, groups=1), {tok} ===")
    for n, c, h, w, k, r, s in shapes:
        torch.manual_seed(0)
        x = torch.randn(n, c, h, w, dtype=dtype, device=dev) * 0.1
        wt = torch.randn(k, c, r, s, dtype=dtype, device=dev) * 0.1
        b = torch.randn(k, dtype=dtype, device=dev) * 0.1
        pad = r // 2
        tag = f"{tok:4s} N={n} C={c:<4} {h}x{w} K={k:<4} {r}x{s}"
        _ab(
            torch,
            "conv2d",
            lambda: F.conv2d(x, wt, b, stride=1, padding=pad),
            lambda: F.conv2d(x, wt, b, stride=1, padding=pad),
            tag,
            warmup,
            iters,
            dtype,
        )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops",
        default="linear,rmsnorm,sdpa,layernorm,silu,gelu,conv2d",
        help="comma-separated subset of "
        "linear,rmsnorm,sdpa,layernorm,silu,gelu,conv2d",
    )
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f16"])
    ap.add_argument(
        "--warmup", type=int, default=int(os.environ.get("MB_WARMUP", "10"))
    )
    ap.add_argument("--iters", type=int, default=int(os.environ.get("MB_ITERS", "50")))
    args = ap.parse_args()

    if not hipdnn_torch.provider_ready():
        print(
            "provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO (see "
            "../README.md).",
            file=sys.stderr,
        )
        return 1

    import torch

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(f"dtype   = {dtype}  warmup={args.warmup} iters={args.iters}")
    print()

    if "linear" in ops:
        _run_linear(torch, _LINEAR_SHAPES, args.warmup, args.iters, dtype)
    if "rmsnorm" in ops:
        _run_rmsnorm(torch, _RMSNORM_SHAPES, args.warmup, args.iters, dtype)
    if "sdpa" in ops:
        _run_sdpa(torch, _SDPA_SHAPES, args.warmup, args.iters, dtype)
    if "layernorm" in ops:
        _run_layernorm(torch, _LAYERNORM_SHAPES, args.warmup, args.iters, dtype)
    if "silu" in ops:
        import torch.nn.functional as F

        _run_activation(
            torch,
            "silu",
            lambda t: F.silu(t),
            _ACT_SHAPES,
            args.warmup,
            args.iters,
            dtype,
        )
    if "gelu" in ops:
        import torch.nn.functional as F

        _run_activation(
            torch,
            "gelu",
            lambda t: F.gelu(t, approximate="tanh"),
            _ACT_SHAPES,
            args.warmup,
            args.iters,
            dtype,
        )
    if "conv2d" in ops:
        _run_conv2d(torch, _CONV_SHAPES, args.warmup, args.iters, dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
