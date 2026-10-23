#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
sd15_unet_ab.py -- run one Stable Diffusion 1.5 UNet denoise step and A/B the forward native
vs the hipdnn_torch injection layer. This is the **classic diffusion UNet** sample and the
smaller sibling of ``sdxl_unet_ab.py``: same ResNet-conv + SiLU + GroupNorm + Transformer2D
(gemm + non-causal SDPA) op mix, but at SD1.5 geometry (cross_attention_dim=768, no
``text_time`` addition embed) and -- crucially -- **``use_linear_projection`` defaults to
False**, so the Transformer2D in/out projections are 1x1 **convs** rather than the SDXL
``Linear``s. That flips a chunk of the projection work from the gemm family to the conv2d
family versus SDXL, making the two UNet entries a controlled conv-vs-gemm contrast.

Model is ``diffusers.UNet2DConditionModel`` built ``from_config`` with the real
**SD 1.5 UNet** config (embedded below -- no checkpoint download; random weights are fine for
routing/coverage). Forward needs only ``encoder_hidden_states`` [B,77,768].

Always run under ``HIPDNN_TORCH_SELECT=default`` with all providers co-loaded. Expect conv2d
-> MIOpen, SiLU -> AOT, and gemm/sdpa/GroupNorm to decline (backlog), same as SDXL.

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all .dll> \
        python samples/sd15_unet_ab.py --ops linear,conv2d,sdpa,silu

Knobs: SD15_H SD15_W (latent spatial, default 64 = 512px) SD15_B SD15_WARMUP SD15_ITERS.
Set ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`` (SDPA + gfx1151
aotriton crash, see ``llama_block_ab.py``).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402

# Real SD 1.5 UNet config (runwayml/stable-diffusion-v1-5, unet/config.json).
SD15_UNET_CONFIG = {
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [320, 640, 1280, 1280],
    "center_input_sample": False,
    "cross_attention_dim": 768,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ],
}


def build_unet(torch, device, dtype):
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_config(SD15_UNET_CONFIG)
    return unet.to(device=device, dtype=dtype).eval()


def make_inputs(torch, batch, lat_h, lat_w, device, dtype):
    gen = torch.Generator(device="cpu").manual_seed(15)

    def r(*shape):
        return (
            torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype).to(device)
        )

    sample = r(batch, 4, lat_h, lat_w)
    ehs = r(batch, 77, 768)  # CLIP-L context
    timestep = torch.tensor(981, device=device)
    return sample, timestep, ehs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops", default="linear,conv2d,sdpa,silu", help="overrides to route"
    )
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print(
            "provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr
        )
        return 1

    global torch
    import torch

    dtype = torch.bfloat16
    device = torch.device("cuda")
    lat_h = int(os.environ.get("SD15_H", "64"))
    lat_w = int(os.environ.get("SD15_W", "64"))
    batch = int(os.environ.get("SD15_B", "1"))
    warmup = int(os.environ.get("SD15_WARMUP", "1"))
    iters = int(os.environ.get("SD15_ITERS", "3"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    unet = build_unet(torch, device, dtype)
    sample, timestep, ehs = make_inputs(torch, batch, lat_h, lat_w, device, dtype)
    print(
        f"config  = SD1.5 UNet  latent[{batch},4,{lat_h},{lat_w}] "
        f"(~{lat_h*8}x{lat_w*8}px)  ctx[{batch},77,768]  dtype={dtype} "
        f"(conv proj, not linear)"
    )
    print(f"routing = {ops}")
    print()

    def fwd():
        with torch.no_grad():
            return unet(sample, timestep, encoder_hidden_states=ehs).sample

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
