#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
sdxl_unet_ab.py -- run one SDXL UNet denoise step and A/B the forward native vs the
hipdnn_torch injection layer. This is the **diffusion UNet** sample: the canonical
conv+attention net (ResNet conv blocks + SiLU + GroupNorm interleaved with cross/self
Transformer2D blocks doing gemm + non-causal SDPA), so it exercises every routed family
at once and is the strongest mixed-op ``default``-routing showcase in the catalog.

Model is ``diffusers.UNet2DConditionModel`` built ``from_config`` with the real
**SDXL base 1.0 UNet** config (embedded below -- no checkpoint download; random weights
are fine for routing/coverage, which depends on call sites / shapes / op mix, not values).
SDXL's ``addition_embed_type="text_time"`` means the forward needs ``added_cond_kwargs``
(``text_embeds`` [B,1280] + ``time_ids`` [B,6]) alongside ``encoder_hidden_states``
[B,77,2048]; all synthesized here.

Always run under ``HIPDNN_TORCH_SELECT=default`` with all providers co-loaded (AOT/
hipblaslt/MIOpen) -- the portable "did hipDNN route this op at all" question. Expect
conv2d -> MIOpen (AOT conv is rank-4 NHWC-groups==1 only; SDXL convs are NCHW/biased),
gemm/sdpa -> AOT or hipblaslt where the shape fits, SiLU -> AOT, and **GroupNorm to
decline everywhere (no gfx1151 engine) = the clearest backlog signal**.

    COMFYUI_PATH unused. HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all .dll> \
        python samples/sdxl_unet_ab.py --ops linear,conv2d,sdpa,silu

Knobs: SDXL_H SDXL_W (latent spatial, default 64 = 512px-equiv; native SDXL is 128=1024px)
SDXL_B SDXL_WARMUP SDXL_ITERS. Set
``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`` -- SDXL uses SDPA, and on this gfx1151 ROCm build
the experimental aotriton fused SDPA crashes the HIP context (see ``llama_block_ab.py``); the
math backend is the correct native baseline.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402

# Real SDXL base 1.0 UNet config (stabilityai/stable-diffusion-xl-base-1.0, unet/config.json).
SDXL_UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "attention_head_dim": [5, 10, 20],
    "block_out_channels": [320, 640, 1280],
    "center_input_sample": False,
    "cross_attention_dim": 2048,
    "down_block_types": [
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "resnet_time_scale_shift": "default",
    "sample_size": 128,
    "transformer_layers_per_block": [1, 2, 10],
    "up_block_types": [
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    ],
    "upcast_attention": False,
    "use_linear_projection": True,
}


def build_unet(torch, device, dtype):
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_config(SDXL_UNET_CONFIG)
    return unet.to(device=device, dtype=dtype).eval()


def make_inputs(torch, batch, lat_h, lat_w, device, dtype):
    gen = torch.Generator(device="cpu").manual_seed(2024)

    def r(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype).to(device)

    sample = r(batch, 4, lat_h, lat_w)
    ehs = r(batch, 77, 2048)  # cross-attn context (CLIP-L 768 + CLIP-G 1280 = 2048)
    added = {"text_embeds": r(batch, 1280), "time_ids": r(batch, 6)}
    timestep = torch.tensor(981, device=device)
    return sample, timestep, ehs, added


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ops", default="linear,conv2d,sdpa,silu", help="overrides to route")
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print("provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr)
        return 1

    global torch
    import torch

    dtype = torch.bfloat16
    device = torch.device("cuda")
    lat_h = int(os.environ.get("SDXL_H", "64"))
    lat_w = int(os.environ.get("SDXL_W", "64"))
    batch = int(os.environ.get("SDXL_B", "1"))
    warmup = int(os.environ.get("SDXL_WARMUP", "1"))
    iters = int(os.environ.get("SDXL_ITERS", "3"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    unet = build_unet(torch, device, dtype)
    sample, timestep, ehs, added = make_inputs(torch, batch, lat_h, lat_w, device, dtype)
    print(
        f"config  = SDXL-base UNet  latent[{batch},4,{lat_h},{lat_w}] "
        f"(~{lat_h*8}x{lat_w*8}px)  ctx[{batch},77,2048]  dtype={dtype}"
    )
    print(f"routing = {ops}")
    print()

    def fwd():
        with torch.no_grad():
            return unet(
                sample, timestep, encoder_hidden_states=ehs, added_cond_kwargs=added
            ).sample

    # ---- correctness: native vs injected on identical input/weights ----------
    with torch.no_grad():
        hipdnn_torch.uninstall()
        y_native = fwd().float()
        hipdnn_torch.install(ops)
        hipdnn_torch.reset()
        y_over = fwd().float()
        hipdnn_torch.uninstall()

    print(f"output  = {tuple(y_native.shape)}")
    fin_n = bool(torch.isfinite(y_native).all())
    fin_o = bool(torch.isfinite(y_over).all())
    if fin_n and fin_o:
        max_err = float((y_native - y_over).abs().max().item())
        denom = float(y_native.abs().max().item()) or 1.0
        rel = max_err / denom
        print(
            f"correctness native-vs-injected: {'OK ' if rel < 8e-2 else 'BAD'} "
            f"max_abs_err={max_err:.5f} rel={rel:.4f}"
        )
    else:
        print(
            f"correctness: parity not measurable "
            f"(native finite={fin_n}, injected finite={fin_o})"
        )
    print()
    print(hipdnn_torch.report(ops))
    print()

    # ---- A/B forward wall-clock -----------------------------------------------
    def timed():
        with torch.no_grad():
            for _ in range(warmup):
                fwd()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fwd()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3

    hipdnn_torch.uninstall()
    native_ms = timed()
    hipdnn_torch.install(ops)
    over_ms = timed()
    hipdnn_torch.uninstall()
    print(f"A/B forward wall-clock (warmup={warmup} iters={iters}):")
    print(f"  native   = {native_ms:8.3f} ms/step")
    print(
        f"  injected = {over_ms:8.3f} ms/step   "
        f"speedup={native_ms/over_ms if over_ms else float('nan'):5.3f}x"
    )
    print()

    print("=== device-time census: one denoise step, by op category (cuda events) ===")
    print(device_time_census(torch, fwd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
