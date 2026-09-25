#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
ltx_video_ab.py -- ADVANCED sample: run the real LTX-Video diffusion transformer
(ComfyUI's ``comfy.ldm.lightricks.model.LTXVModel``) with random weights and A/B
its full forward pass native vs the hipdnn_torch injection layer.

This is the real-model end-to-end proof: it instantiates the actual DiT, installs a
selectable subset of overrides (``--ops linear,rmsnorm,sdpa``), verifies the output
is not corrupted, and dumps both the hipdnn_torch intercept report AND a per-op
device-time census (via CUDA events, since kineto GPU capture is dead on this ROCm build).

Random weights are used on purpose: the LTX checkpoint changes pixel values, not the
call sites / shapes / op mix, so it doesn't affect what routes to hipDNN. Because the
shipped gfx1151 kernels are correctness-first references, expect OUTPUT PARITY to
hold and the intended ops to route -- but a blanket full-forward speedup is NOT the
point (uplift is the data-only kernel-tuning follow-on).

Requires an external ComfyUI checkout (for ``comfy.*``) and a provider-compatible
torch. Point ``COMFYUI_PATH`` at your ComfyUI clone::

    COMFYUI_PATH=~/ComfyUI \
        HIPDNN_TORCH_PROVIDER_SO=<...>/libhip_kernel_provider.so \
        TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        python samples/ltx_video_ab.py --ops linear,rmsnorm,sdpa

Set ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` so the *native* SDPA baseline is a
real fused kernel and not the math strawman (see ../README.md).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402


def _add_comfyui_to_path():
    comfy = os.environ.get("COMFYUI_PATH")
    if not comfy:
        print(
            "COMFYUI_PATH is not set -- point it at a ComfyUI checkout "
            "(the source of comfy.*).",
            file=sys.stderr,
        )
        return False
    comfy = os.path.expanduser(comfy)
    if not os.path.isdir(comfy):
        print(f"COMFYUI_PATH does not exist: {comfy}", file=sys.stderr)
        return False
    if comfy not in sys.path:
        sys.path.insert(0, comfy)
    return True


def build_model(comfy_ops, LTXVModel, torch, num_layers, dtype, device):
    model = LTXVModel(
        in_channels=128,
        cross_attention_dim=2048,
        attention_head_dim=64,
        num_attention_heads=32,  # inner_dim = 32 * 64 = 2048
        caption_channels=4096,
        num_layers=num_layers,
        dtype=dtype,
        device=device,
        operations=comfy_ops.disable_weight_init,
    )
    # disable_weight_init skips reset_parameters, so fill every float tensor with
    # small noise (same fill for both A/B runs -> identical weights).
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(1234)
        for p in model.parameters():
            if p.is_floating_point():
                p.copy_(
                    torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.dtype)
                    * 0.02
                )
        for b in model.buffers():
            if b.is_floating_point():
                b.zero_()
    return model.eval()


def build_inputs(torch, frames, hh, ww, ctx_len, dtype, device):
    gen = torch.Generator(device="cpu").manual_seed(7)

    def rnd(*shape):
        return (
            torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype).to(device)
        )

    x = rnd(1, 128, frames, hh, ww)  # latent video [B,C,F,H,W]
    context = rnd(1, ctx_len, 4096)  # text embeddings
    timestep = torch.full((1,), 0.5, dtype=torch.float32, device=device)
    return x, timestep, context


def run_forward(model, x, timestep, context):
    return model(x, timestep, context, attention_mask=None, frame_rate=25)


def time_forward(torch, model, x, timestep, context, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            run_forward(model, x, timestep, context)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            run_forward(model, x, timestep, context)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters


def device_time_census(torch, model, x, timestep, context) -> str:
    """Per-op device time for one forward, via CUDA-event pairs around each hot
    functional (kineto GPU capture is dead on this ROCm build)."""
    F = torch.nn.functional
    reals = {
        "linear (GEMM)": ("linear", F.linear),
        "attention (SDPA)": (
            "scaled_dot_product_attention",
            F.scaled_dot_product_attention,
        ),
        "rms_norm": ("rms_norm", F.rms_norm),
        "layer_norm": ("layer_norm", F.layer_norm),
        "gelu": ("gelu", F.gelu),
        "silu": ("silu", F.silu),
    }
    events = []  # (category, start_event, end_event)

    def make(cat, real):
        def wrapper(*a, **k):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = real(*a, **k)
            e.record()
            events.append((cat, s, e))
            return out

        return wrapper

    with torch.no_grad():
        run_forward(model, x, timestep, context)  # warm
        torch.cuda.synchronize()
        for cat, (attr, real) in reals.items():
            setattr(F, attr, make(cat, real))
        try:
            run_forward(model, x, timestep, context)
            torch.cuda.synchronize()
        finally:
            for cat, (attr, real) in reals.items():
                setattr(F, attr, real)

    totals, calls = {}, {}
    for cat, s, e in events:
        totals[cat] = totals.get(cat, 0.0) + s.elapsed_time(e)
        calls[cat] = calls.get(cat, 0) + 1
    grand = sum(totals.values()) or 1.0
    lines = ["  category            calls   dev_ms   share"]
    for cat in sorted(totals, key=lambda k: -totals[k]):
        lines.append(
            f"  {cat:18s}  {calls[cat]:5d}  {totals[cat]:7.3f}  "
            f"{100 * totals[cat] / grand:5.1f}%"
        )
    lines.append(
        f"  {'(sum of wrapped)':18s}  {sum(calls.values()):5d}  {grand:7.3f}  100.0%"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops",
        default="linear,rmsnorm,sdpa,layernorm,silu,gelu,conv2d",
        help="comma-separated subset of "
        "linear,rmsnorm,sdpa,layernorm,silu,gelu,conv2d to route",
    )
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print(
            "provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO (see "
            "../README.md).",
            file=sys.stderr,
        )
        return 1
    if not _add_comfyui_to_path():
        return 1

    import torch
    import comfy.ops as comfy_ops
    from comfy.ldm.lightricks.model import LTXVModel

    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_layers = int(os.environ.get("LTX_LAYERS", "6"))
    frames = int(os.environ.get("LTX_FRAMES", "4"))
    hh = int(os.environ.get("LTX_H", "32"))
    ww = int(os.environ.get("LTX_W", "32"))
    ctx_len = int(os.environ.get("LTX_CTX", "128"))  # mult of 16 -> cross-attn routes
    warmup = int(os.environ.get("LTX_WARMUP", "3"))
    iters = int(os.environ.get("LTX_ITERS", "10"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"config  = layers={num_layers} tokens={frames*hh*ww} (F{frames}xH{hh}xW{ww}) "
        f"heads=32 head_dim=64 ctx_len={ctx_len} dtype={dtype}"
    )
    print(f"routing = {ops}")
    print()

    model = build_model(comfy_ops, LTXVModel, torch, num_layers, dtype, device)
    x, timestep, context = build_inputs(torch, frames, hh, ww, ctx_len, dtype, device)

    # ---- correctness: native vs injected on identical inputs/weights ----------
    with torch.no_grad():
        hipdnn_torch.uninstall()
        y_native = run_forward(model, x, timestep, context).float()

        hipdnn_torch.install(ops)
        hipdnn_torch.reset()
        y_over = run_forward(model, x, timestep, context).float()
        hipdnn_torch.uninstall()

    max_err = float((y_native - y_over).abs().max().item())
    denom = float(y_native.abs().max().item()) or 1.0
    rel = max_err / denom
    print(
        f"correctness native-vs-injected: {'OK ' if rel < 8e-2 else 'BAD'} "
        f"max_abs_err={max_err:.5f} rel={rel:.4f} (out|max|={denom:.4f})"
    )
    print()
    print(hipdnn_torch.report(ops))
    print()

    # ---- A/B full-forward wall-clock ------------------------------------------
    hipdnn_torch.uninstall()
    native_ms = time_forward(torch, model, x, timestep, context, warmup, iters) * 1e3
    hipdnn_torch.install(ops)
    over_ms = time_forward(torch, model, x, timestep, context, warmup, iters) * 1e3
    hipdnn_torch.uninstall()
    speedup = native_ms / over_ms if over_ms > 0 else float("nan")
    print(f"A/B full-forward wall-clock (warmup={warmup} iters={iters}):")
    print(f"  native   = {native_ms:8.3f} ms/forward")
    print(f"  injected = {over_ms:8.3f} ms/forward   speedup={speedup:5.3f}x")
    print(
        "  (reference kernels are correctness-first; a full-forward speedup is "
        "NOT expected yet)"
    )
    print()

    # ---- device-time census ---------------------------------------------------
    print("=== device-time census: one forward, by op category (cuda events) ===")
    print(device_time_census(torch, model, x, timestep, context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
