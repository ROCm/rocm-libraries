#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
ltx_vae_ab.py -- run the REAL LTX-Video VAE decoder (ComfyUI's
``comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE``) with REAL weights
and A/B its decode native vs the hipdnn_torch injection layer.

This is the conv3d headline: unlike the transformer samples, the VAE decode is a
stack of causal 3-D convolutions, so it exercises the ``F.conv3d`` override end to
end. Because it loads a real checkpoint (not random weights), the decode is
numerically stable in bf16, so BOTH parity and a real speedup number are meaningful.

It also demonstrates why the ``default`` selection policy matters: under ``default``
hipDNN routes conv3d to whichever loaded engine serves it (MIOpen on this stack); if
you instead ``force`` an engine that has no conv3d kernel, every conv3d drops to
native and the speedup evaporates. Run it both ways and compare the census.

    COMFYUI_PATH=/path/to/ComfyUI \
        HIPDNN_TORCH_PROVIDER_SO=<...>/libhip_kernel_provider.so \
        LTX_VAE=/path/to/LTX-Video-VAE-BF16.safetensors \
        python samples/ltx_vae_ab.py

    # contrast: force one engine (conv3d has no kernel there -> native fallback)
    HIPDNN_TORCH_SELECT=force HIPDNN_TORCH_ENGINE=AOT_CATALOG_ENGINE \
        ... python samples/ltx_vae_ab.py
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


def load_vae(torch, path, dtype, device):
    """Instantiate the LTX-V VideoVAE and load real city96-format weights."""
    from safetensors.torch import load_file
    import comfy.ldm.lightricks.vae.causal_video_autoencoder as cva

    sd = load_file(os.path.expanduser(path))
    prefix = "first_stage_model."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    c1 = sd.get("decoder.up_blocks.0.res_blocks.0.conv1.conv.weight")
    version = 1 if (c1 is not None and c1.shape[0] == 1024) else 0
    model = cva.VideoVAE(version=version, config=None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    return model.to(device=device, dtype=dtype).eval(), version


def decode(vae, latent):
    with torch.no_grad():  # noqa: F821 -- torch bound in main()
        return vae.decode(latent)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops",
        default="conv3d,conv2d,silu,layernorm,rmsnorm,linear,gelu",
        help="comma-separated overrides to route",
    )
    ap.add_argument(
        "--vae",
        default=os.environ.get("LTX_VAE"),
        help="path to LTX-Video VAE safetensors (or set LTX_VAE)",
    )
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print(
            "provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr
        )
        return 1
    if not args.vae:
        print("no VAE checkpoint -- pass --vae or set LTX_VAE.", file=sys.stderr)
        return 1
    if not _add_comfyui_to_path():
        return 1

    global torch
    import torch

    dtype = torch.bfloat16
    device = torch.device("cuda")
    fr = int(os.environ.get("LTX_VAE_F", "2"))
    hh = int(os.environ.get("LTX_VAE_H", "16"))
    ww = int(os.environ.get("LTX_VAE_W", "16"))
    warmup = int(os.environ.get("LTX_WARMUP", "2"))
    iters = int(os.environ.get("LTX_ITERS", "5"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    vae, version = load_vae(torch, args.vae, dtype, device)
    gen = torch.Generator(device="cpu").manual_seed(11)
    latent = (
        torch.randn(1, 128, fr, hh, ww, generator=gen, dtype=torch.float32)
        .to(dtype)
        .to(device)
    )
    print(f"config  = LTX-V VAE v{version}  latent[1,128,{fr},{hh},{ww}] dtype={dtype}")
    print(f"routing = {ops}")
    print()

    # ---- correctness: native vs injected on identical input/weights ----------
    with torch.no_grad():
        hipdnn_torch.uninstall()
        y_native = decode(vae, latent).float()
        hipdnn_torch.install(ops)
        hipdnn_torch.reset()
        y_over = decode(vae, latent).float()
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
            f"correctness: parity not measurable "
            f"(native finite={fin_n}, injected finite={fin_o})"
        )
    print()
    print(hipdnn_torch.report(ops))
    print()

    # ---- A/B decode wall-clock ------------------------------------------------
    def timed():
        with torch.no_grad():
            for _ in range(warmup):
                decode(vae, latent)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                decode(vae, latent)
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
    print(device_time_census(torch, lambda: decode(vae, latent)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
