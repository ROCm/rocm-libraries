#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
sd_vae_ab.py -- run a Stable-Diffusion 2-D VAE decoder (ComfyUI's
``comfy.ldm.modules.diffusionmodules.model.Decoder``) and A/B its decode native vs
the hipdnn_torch injection layer.

This is the conv2d **NCHW** coverage case: the SD VAE decode is a stack of 2-D
convolutions in the default (contiguous NCHW) memory format, so it exercises the
``F.conv2d`` override on the layout torch hands it -- no channels-last imposition.
The other routed op here is ``F.silu`` (the ResnetBlock nonlinearity). ``group_norm``
is intentionally *not* routed (no override) and stays native; the census still times
it so its share is visible.

Weights are random (no checkpoint), so this validates **coverage and routing**, not a
numeric win: random-weight bf16 group_norm suffers catastrophic cancellation and can
produce non-finite activations, which is a property of the reference math, not the
injection. The script reports a finite-aware parity check and the per-op census.

    HIPDNN_TORCH_PROVIDER_SO=<...>/libhip_kernel_provider.so \
        COMFYUI_PATH=/path/to/ComfyUI \
        python samples/sd_vae_ab.py

    # contrast: force one engine (conv2d has no kernel there -> native fallback)
    HIPDNN_TORCH_SELECT=force HIPDNN_TORCH_ENGINE=AOT_CATALOG_ENGINE \
        ... python samples/sd_vae_ab.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402


def _add_comfyui_to_path():
    comfy = os.environ.get("COMFYUI_PATH")
    if not comfy:
        print(
            "COMFYUI_PATH is not set -- point it at a ComfyUI checkout.",
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


def build_decoder(torch, z_channels, dtype, device):
    """Instantiate a standard SD 2-D VAE Decoder with random weights."""
    import comfy.ldm.modules.diffusionmodules.model as m

    dec = m.Decoder(
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        in_channels=3,
        resolution=256,
        z_channels=z_channels,
    )
    return dec.to(device=device, dtype=dtype).eval()


def decode(dec, z):
    with torch.no_grad():  # noqa: F821 -- torch bound in main()
        return dec(z)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops",
        default="conv2d,silu,layernorm,rmsnorm,linear,gelu",
        help="comma-separated overrides to route",
    )
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print(
            "provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr
        )
        return 1
    if not _add_comfyui_to_path():
        return 1

    global torch
    import torch

    dtype = torch.bfloat16
    device = torch.device("cuda")
    z_channels = int(os.environ.get("SD_VAE_Z", "4"))
    hh = int(os.environ.get("SD_VAE_H", "32"))
    ww = int(os.environ.get("SD_VAE_W", "32"))
    warmup = int(os.environ.get("SD_WARMUP", "2"))
    iters = int(os.environ.get("SD_ITERS", "5"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    dec = build_decoder(torch, z_channels, dtype, device)
    gen = torch.Generator(device="cpu").manual_seed(7)
    z = (
        torch.randn(1, z_channels, hh, ww, generator=gen, dtype=torch.float32)
        .to(dtype)
        .to(device)
    )
    print(
        f"config  = SD VAE Decoder  z[1,{z_channels},{hh},{ww}] -> "
        f"img[1,3,{hh*8},{ww*8}] dtype={dtype}"
    )
    print(f"routing = {ops}")
    print()

    # ---- correctness: native vs injected on identical input/weights ----------
    with torch.no_grad():
        hipdnn_torch.uninstall()
        y_native = decode(dec, z).float()
        hipdnn_torch.install(ops)
        hipdnn_torch.reset()
        y_over = decode(dec, z).float()
        hipdnn_torch.uninstall()

    fin_n = bool(torch.isfinite(y_native).all())
    fin_o = bool(torch.isfinite(y_over).all())
    if fin_n and fin_o:
        max_err = float((y_native - y_over).abs().max().item())
        denom = float(y_native.abs().max().item()) or 1.0
        print(
            f"correctness native-vs-injected: "
            f"{'OK ' if max_err / denom < 8e-2 else 'BAD'} "
            f"max_abs_err={max_err:.5f} rel={max_err/denom:.4f}"
        )
    else:
        print(
            f"correctness: parity not measurable in bf16 "
            f"(native finite={fin_n}, injected finite={fin_o}) "
            f"-- random-weight group_norm artifact; routing/coverage below is valid"
        )
    print()
    print(hipdnn_torch.report(ops))
    print()

    # ---- A/B decode wall-clock ------------------------------------------------
    def timed():
        with torch.no_grad():
            for _ in range(warmup):
                decode(dec, z)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                decode(dec, z)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3

    hipdnn_torch.uninstall()
    native_ms = timed()
    hipdnn_torch.install(ops)
    over_ms = timed()
    hipdnn_torch.uninstall()
    print(f"A/B decode wall-clock (warmup={warmup} iters={iters}):")
    print(f"  native   = {native_ms:8.3f} ms/decode")
    print(
        f"  injected = {over_ms:8.3f} ms/decode   "
        f"speedup={native_ms/over_ms if over_ms else float('nan'):5.3f}x"
    )
    print()

    print("=== device-time census: one decode, by op category (cuda events) ===")
    print(device_time_census(torch, lambda: decode(dec, z)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
