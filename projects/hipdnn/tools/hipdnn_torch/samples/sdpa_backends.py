#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
sdpa_backends.py -- compare the hipDNN SDPA kernel against the REAL fused PyTorch
attention backends on RDNA (gfx1151), not just the math fallback.

Why this matters: on gfx1151 ``F.scaled_dot_product_attention`` silently drops to
the UNFUSED O(S^2) math path unless ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1``
is set BEFORE torch is imported (this script sets it for you). Benchmarking against
that math strawman makes any kernel look ~9-12x better than it is. Here each backend
is timed explicitly against the same shape:

  * math                -- unfused reference (the misleading baseline)
  * aotriton-flash      -- AOTriton flash kernel (the real fused baseline)
  * aotriton-eff        -- AOTriton memory-efficient kernel
  * fa-triton           -- ROCm flash-attention Triton backend, if the ``flash_attn``
                           package is source-built (skipped as N/A otherwise)
  * hipdnn              -- the hipDNN engine kernel, via hipdnn_torch

It first runs a one-shape PROBE (which backends are usable + OK/FAIL), then the full
per-shape/dtype table. See ../README.md "Other PyTorch attention/op backends" for
how to enable each backend and the RDNA-vs-CDNA differences. Run::

    HIPDNN_TORCH_PROVIDER_SO=<...>/libhip_kernel_provider.so \
        python samples/sdpa_backends.py
"""

import os

# MUST precede torch CUDA init (torch is imported lazily inside hipdnn_torch's
# bootstrap, below). Without it, gfx1151 FLASH/EFFICIENT report "still
# experimental" and fall to math.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
# Selects the Triton (AMD) backend inside the flash_attn package, if installed.
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

import math  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402

_SHAPES = [
    (1, 32, 32, 32, 64),  # small square
    (1, 32, 64, 48, 64),  # asymmetric (cross-attn in miniature)
    (1, 32, 4096, 4096, 64),  # LTX self-attention (the dominant cost)
    (1, 32, 4096, 128, 64),  # LTX cross-attention
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


def _hipdnn_sdpa(torch, native_sdpa, q, k, v, scale):
    """Time/eval the hipDNN path by installing the sdpa override around the call."""
    hipdnn_torch.install(["sdpa"])
    try:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    finally:
        hipdnn_torch.uninstall(["sdpa"])


def _make_fa_triton(flash_attn_func):
    def _fa(q, k, v, scale):
        # ROCm Triton FA takes [B,S,H,D]; permute from our [B,H,S,D].
        qt = q.transpose(1, 2).contiguous()
        kt = k.transpose(1, 2).contiguous()
        vt = v.transpose(1, 2).contiguous()
        o = flash_attn_func(
            qt, kt, vt, dropout_p=0.0, softmax_scale=scale, causal=False
        )
        return o.transpose(1, 2)

    return _fa


def _bench_shape(
    torch,
    F,
    sdpa_kernel,
    SDPBackend,
    native_sdpa,
    fa_triton,
    b,
    h,
    sq,
    skv,
    d,
    dtype,
    warmup,
    iters,
):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(b, h, sq, d, dtype=dtype, device=dev) * 0.1
    k = torch.randn(b, h, skv, d, dtype=dtype, device=dev) * 0.1
    v = torch.randn(b, h, skv, d, dtype=dtype, device=dev) * 0.1
    scale = 1.0 / math.sqrt(d)
    tok = {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
    print(f"{tok:4s} B={b} H={h:<3} Sq={sq:<5} Skv={skv:<5} D={d:<3}")

    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        ref = native_sdpa(q, k, v, scale=scale).float()

    results = {}  # label -> (ms, maxerr, status)

    # hipDNN (via the injection path)
    try:
        got = _hipdnn_sdpa(torch, native_sdpa, q, k, v, scale).float()
        err = float((got - ref).abs().max().item())
        ms = _time_ms(
            torch,
            lambda: _hipdnn_sdpa(torch, native_sdpa, q, k, v, scale),
            warmup,
            iters,
        )
        results["hipdnn"] = (ms, err, "OK")
    except Exception as e:  # noqa: BLE001
        results["hipdnn"] = (None, None, f"ERR ({type(e).__name__})")

    # ROCm flash-attention Triton backend
    if fa_triton is None:
        results["fa-triton"] = (None, None, "N/A (flash_attn not installed)")
    else:
        try:
            with torch.no_grad():
                out = fa_triton(q, k, v, scale).float()
                err = float((out - ref).abs().max().item())
                ms = _time_ms(torch, lambda: fa_triton(q, k, v, scale), warmup, iters)
            results["fa-triton"] = (ms, err, "OK")
        except Exception as e:  # noqa: BLE001
            msg = (str(e).strip().splitlines() or [type(e).__name__])[0][:40]
            results["fa-triton"] = (None, None, f"FAIL ({msg})")

    # torch's own SDPA backends
    for label, be in (
        ("math", SDPBackend.MATH),
        ("aotriton-flash", SDPBackend.FLASH_ATTENTION),
        ("aotriton-eff", SDPBackend.EFFICIENT_ATTENTION),
    ):
        try:
            with torch.no_grad(), sdpa_kernel([be]):
                out = native_sdpa(q, k, v, scale=scale).float()
                err = float((out - ref).abs().max().item())
                ms = _time_ms(
                    torch,
                    lambda be=be: native_sdpa(q, k, v, scale=scale),
                    warmup,
                    iters,
                )
            results[label] = (ms, err, "OK")
        except Exception as e:  # noqa: BLE001
            msg = (str(e).strip().splitlines() or [type(e).__name__])[0][:40]
            results[label] = (None, None, f"FAIL ({msg})")

    order = ["math", "aotriton-flash", "aotriton-eff", "fa-triton", "hipdnn"]
    fastest = min((r[0] for r in results.values() if r[0] is not None), default=None)
    for label in order:
        ms, err, status = results[label]
        if ms is None:
            print(f"    {label:16s} : {status}")
            continue
        errs = f"maxerr={err:7.4f}" if err is not None else ""
        rel = f"  ({fastest/ms:4.2f}x of fastest)" if fastest else ""
        star = " *" if ms == fastest else "  "
        print(f"    {label:16s} :{star}{ms*1e3:9.1f} us  {errs}{rel}")
    print()


def _probe(torch, F, sdpa_kernel, SDPBackend, native_sdpa, fa_triton):
    """Which backends can serve one representative shape (self-attn tile)?"""
    dev = torch.device("cuda")
    b, h, s, d = 1, 32, 256, 64
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=dev) * 0.1
    scale = 1.0 / math.sqrt(d)
    print("=== backend probe (bf16 B=1 H=32 S=256 D=64) ===")
    checks = [
        ("math", lambda: _run_be(native_sdpa, sdpa_kernel, SDPBackend.MATH, q, scale)),
        (
            "aotriton-flash",
            lambda: _run_be(
                native_sdpa, sdpa_kernel, SDPBackend.FLASH_ATTENTION, q, scale
            ),
        ),
        (
            "aotriton-eff",
            lambda: _run_be(
                native_sdpa, sdpa_kernel, SDPBackend.EFFICIENT_ATTENTION, q, scale
            ),
        ),
        ("fa-triton", (lambda: fa_triton(q, q, q, scale)) if fa_triton else None),
        ("hipdnn", lambda: _hipdnn_sdpa(torch, native_sdpa, q, q, q, scale)),
    ]
    for label, fn in checks:
        if fn is None:
            print(f"    {label:16s} : N/A (not installed)")
            continue
        try:
            with torch.no_grad():
                fn()
            print(f"    {label:16s} : OK")
        except Exception as e:  # noqa: BLE001
            msg = (str(e).strip().splitlines() or [type(e).__name__])[0][:50]
            print(f"    {label:16s} : FAIL ({msg})")
    print()


def _run_be(native_sdpa, sdpa_kernel, be, q, scale):
    with sdpa_kernel([be]):
        return native_sdpa(q, q, q, scale=scale)


def main() -> int:
    if not hipdnn_torch.provider_ready():
        print(
            "provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO (see "
            "../README.md).",
            file=sys.stderr,
        )
        return 1

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    native_sdpa = F.scaled_dot_product_attention  # original, before any install

    try:
        from flash_attn import flash_attn_func

        fa_triton = _make_fa_triton(flash_attn_func)
    except Exception:  # noqa: BLE001
        fa_triton = None

    props = torch.cuda.get_device_properties(0)
    print(
        f"device     = {torch.cuda.get_device_name(0)} "
        f"({getattr(props, 'gcnArchName', '?')})"
    )
    print(
        f"torch      = {torch.__version__}  hip={getattr(torch.version, 'hip', None)}"
    )
    print(
        f"AOTRITON_EXPERIMENTAL = "
        f"{os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL')}"
    )
    print("  (* = fastest of the row; 'x of fastest' compares each to that)")
    print()

    _probe(torch, F, sdpa_kernel, SDPBackend, native_sdpa, fa_triton)

    warmup, iters = 10, 50
    for dtype in (torch.float16, torch.bfloat16):
        print(
            f"=== {'f16' if dtype == torch.float16 else 'bf16 (LTX native dtype)'} ==="
        )
        for b, h, sq, skv, d in _SHAPES:
            _bench_shape(
                torch,
                F,
                sdpa_kernel,
                SDPBackend,
                native_sdpa,
                fa_triton,
                b,
                h,
                sq,
                skv,
                d,
                dtype,
                warmup,
                iters,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
