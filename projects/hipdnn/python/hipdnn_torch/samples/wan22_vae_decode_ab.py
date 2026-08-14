#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
wan22_vae_decode_ab.py -- isolated Wan 2.2 (14B T2V) VAE **decode** at 704x1280, A/B'd
native vs the hipdnn_torch injection layer. This is the ROCM-27995 defect locus, carved
out so it runs in seconds instead of the 11-14 h full 14B text-to-video job.

Wan 2.2 14B T2V reuses the **Wan 2.1 VAE** (ComfyUI ``comfy.ldm.wan.vae.WanVAE``, z_dim=4,
8x spatial / 4x temporal), NOT the newer TI2V-5B ``vae2_2`` VAE. Its decoder is a stack of
``CausalConv3d`` (3x3, temporal-cached) plus ``ops.Conv2d`` upsample (Resample) blocks that
run at increasing resolution, ending in the full-res convs ROCM-27995 fingerprinted:
``{1,256/128,704,1280} -> {1,3,704,1280}`` (MIOpen solvers GemmFwdRest / GemmFwd1x1, the
im2col-workspace path where the silent 0.209-vs-3e-5 corruption appeared).

The A/B here is itself the defect probe: the **native** side runs PyTorch's conv (MIOpen's
chosen solver -- the suspect GemmFwdRest); the **injected** side routes conv2d/conv3d through
hipDNN (AOT/rocKE CK or a different MIOpen solver). A solver-level numerical divergence shows
up in the native-vs-injected diff even with random weights -- so this catches the corruption
locus at model scale without waiting on a full generate. Run under ``default`` so hipDNN picks
across all loaded engines (the portable coverage question). No SDPA in this decode, so the
gfx1151 aotriton crash (see ``llama_block_ab.py``) does not apply.

    COMFYUI_PATH=/path/to/ComfyUI \
        HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all provider .so/.dll> \
        python samples/wan22_vae_decode_ab.py --ops conv2d,conv3d,silu

Random weights bring the conv path + routing + finite/parity check up (Option 1: synthetic
704x1280 latent). To reproduce the *actual* 0.209 corruption values, pass real Wan 2.1 VAE
weights via --vae / WAN_VAE and a real pre-VAE latent via WAN_LATENT (Option 3).

Knobs: WAN_H(704) WAN_W(1280) WAN_F(1 latent frame) WAN_WARMUP WAN_ITERS ; --vae / WAN_VAE
(optional real weights) ; WAN_LATENT (optional real saved latent .pt/.safetensors).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402

_SPATIAL = 8  # Wan 2.1 VAE spatial compression (pixels per latent cell)


def _stub_comfy_aimdo():
    """Register empty stubs for ``comfy_aimdo.*`` so ComfyUI imports don't die.

    AMD's AIMDO ComfyUI build has ``comfy.model_management`` /
    ``comfy.ops`` / ``comfy.model_patcher`` ``import comfy_aimdo.{host_buffer,vram_buffer,
    model_vbar,torch}`` at module load, but that package is an in-tree AMD component that
    isn't checked in or on PyPI. Every *runtime* use of it is gated behind ``hasattr(s,
    "_v")`` -- an attribute set only by ComfyUI's own state-dict loader. We instantiate
    ``WanVAE()`` directly (random or plain ``load_state_dict`` weights), so ``_v`` is never
    set and the AIMDO paths are dead code; the modules only need to *import*. Empty stubs in
    ``sys.modules`` satisfy that without touching the clone or site-packages.
    """
    import types

    if "comfy_aimdo" in sys.modules:
        return
    pkg = types.ModuleType("comfy_aimdo")
    pkg.__path__ = []  # mark as a package so ``import comfy_aimdo.x`` resolves
    sys.modules["comfy_aimdo"] = pkg
    for sub in ("host_buffer", "vram_buffer", "model_vbar", "torch"):
        m = types.ModuleType(f"comfy_aimdo.{sub}")
        sys.modules[f"comfy_aimdo.{sub}"] = m
        setattr(pkg, sub, m)


def _add_comfyui_to_path():
    comfy = os.environ.get("COMFYUI_PATH")
    if not comfy:
        print("COMFYUI_PATH is not set -- point it at a ComfyUI checkout.", file=sys.stderr)
        return False
    comfy = os.path.expanduser(comfy)
    if not os.path.isdir(comfy):
        print(f"COMFYUI_PATH does not exist: {comfy}", file=sys.stderr)
        return False
    if comfy not in sys.path:
        sys.path.insert(0, comfy)
    _stub_comfy_aimdo()
    return True


def build_vae(torch, vae_path, dtype, device):
    """Instantiate Wan 2.1 WanVAE. Load real weights if given, else random (Option 1)."""
    import comfy.ldm.wan.vae as wanvae

    model = wanvae.WanVAE()  # z_dim=4, dim=128, dim_mult=[1,2,4,4] -- the 14B-T2V VAE
    loaded = "random"
    if vae_path:
        from safetensors.torch import load_file

        p = os.path.expanduser(vae_path)
        sd = load_file(p) if p.endswith(".safetensors") else torch.load(p, map_location="cpu")
        for prefix in ("first_stage_model.", "vae."):
            if any(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = f"real ({len(sd)} tensors; {len(missing)} missing, {len(unexpected)} unexpected)"
    return model.to(device=device, dtype=dtype).eval(), loaded


def decode(vae, latent):
    with torch.no_grad():  # noqa: F821 -- torch bound in main()
        return vae.decode(latent)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ops", default="conv2d,conv3d,silu", help="overrides to route")
    ap.add_argument("--vae", default=os.environ.get("WAN_VAE"), help="real Wan2.1 VAE weights")
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print("provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr)
        return 1
    if not _add_comfyui_to_path():
        return 1

    global torch
    import torch

    dtype = torch.bfloat16
    device = torch.device("cuda")
    out_h = int(os.environ.get("WAN_H", "704"))
    out_w = int(os.environ.get("WAN_W", "1280"))
    fr = int(os.environ.get("WAN_F", "1"))
    warmup = int(os.environ.get("WAN_WARMUP", "1"))
    iters = int(os.environ.get("WAN_ITERS", "3"))
    lh, lw = out_h // _SPATIAL, out_w // _SPATIAL

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    vae, loaded = build_vae(torch, args.vae, dtype, device)

    latent_path = os.environ.get("WAN_LATENT")
    if latent_path:
        from safetensors.torch import load_file

        lp = os.path.expanduser(latent_path)
        obj = load_file(lp) if lp.endswith(".safetensors") else torch.load(lp, map_location="cpu")
        latent = (obj if torch.is_tensor(obj) else next(iter(obj.values()))).to(dtype).to(device)
        print(f"latent  = real {tuple(latent.shape)} from {os.path.basename(lp)}")
    else:
        gen = torch.Generator(device="cpu").manual_seed(27995)
        latent = (
            torch.randn(1, 4, fr, lh, lw, generator=gen, dtype=torch.float32).to(dtype).to(device)
        )
        print(f"latent  = synthetic [1,4,{fr},{lh},{lw}] (Option 1)")
    print(f"config  = Wan2.1 VAE (14B-T2V) weights={loaded}  target out {out_h}x{out_w}")
    print(f"routing = {ops}")
    print()

    # ---- correctness: native (MIOpen solver) vs injected (hipDNN) on same input ----
    with torch.no_grad():
        hipdnn_torch.uninstall()
        y_native = decode(vae, latent).float()
        hipdnn_torch.install(ops)
        hipdnn_torch.reset()
        y_over = decode(vae, latent).float()
        hipdnn_torch.uninstall()

    print(f"output  = {tuple(y_native.shape)}")
    fin_n = bool(torch.isfinite(y_native).all())
    fin_o = bool(torch.isfinite(y_over).all())
    if fin_n and fin_o:
        max_err = float((y_native - y_over).abs().max().item())
        denom = float(y_native.abs().max().item()) or 1.0
        rel = max_err / denom
        # ROCM-27995 tell: the corruption showed as rel ~0.209 vs a healthy ~3e-5.
        verdict = "OK " if rel < 8e-2 else "DIVERGENCE (possible ROCM-27995 solver defect)"
        print(
            f"native-vs-injected: {verdict} max_abs_err={max_err:.5f} rel={rel:.5f}"
        )
    else:
        print(
            f"correctness: NOT FINITE (native finite={fin_n}, injected finite={fin_o}) "
            f"-- a non-finite decode is itself the corruption signal"
        )
    print()
    print(hipdnn_torch.report(ops))
    print()

    # ---- A/B decode wall-clock -------------------------------------------------
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
