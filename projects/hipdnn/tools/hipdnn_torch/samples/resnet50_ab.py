#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
resnet50_ab.py -- run a ResNet-50 forward and A/B it native vs the hipdnn_torch injection
layer. This is the **classic CNN** sample: a pure **NCHW conv2d** stack (bottleneck blocks:
1x1 -> 3x3 -> 1x1 with a projection shortcut) + BatchNorm + ReLU, ending in global avgpool
and a single 2-D ``F.linear`` classifier.

Two things make it a useful catalog entry distinct from the diffusion UNets:
  * **NCHW convs.** The AOT conv adapter is rank-4 **NHWC-packed, groups==1** only, so every
    ResNet conv declines AOT and **MIOpen serves under ``default``** -- a clean read on the
    portable conv path when AOT can't take the layout.
  * **A genuinely 2-D GEMM.** The final classifier is ``[B,2048] -> [B,1000]`` at true rank 2,
    unlike the 3-D ``[B,T,C]`` linears in transformers -- so it is the one place a real model
    hits the AOT 2-D ``GemmAdapter [M,N]`` head-on (watch whether it routes to AOT/hipblaslt).
  * **BatchNorm** has no gfx1151 engine -> declines everywhere = a backlog signal (the CNN
    analogue of the UNet GroupNorm gap).

Self-contained (no torchvision) so it can't drag a mismatched CUDA/CPU torch into the ROCm
nightly venv -- the bottleneck architecture is standard and reproduced here with plain
``torch.nn``. Random weights are fine for routing/coverage (BatchNorm running_mean=0/var=1 in
eval keeps activations finite). Always run under ``HIPDNN_TORCH_SELECT=default`` with all
providers co-loaded.

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all .dll> \
        python samples/resnet50_ab.py --ops conv2d,linear

Knobs: RESNET_IMG (default 224) RESNET_B (default 8) RESNET_WARMUP RESNET_ITERS. No SDPA, so
the gfx1151 aotriton crash does not apply.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402


def build_resnet50(torch, device, dtype):
    """A standard ResNet-50 (bottleneck [3,4,6,3]) in plain torch.nn -- no torchvision."""
    nn = torch.nn

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inp, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inp, planes, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, 3, stride=stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            idt = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                idt = self.downsample(x)
            return self.relu(out + idt)

    class ResNet(nn.Module):
        def __init__(self, layers=(3, 4, 6, 3), num_classes=1000):
            super().__init__()
            self.inp = 64
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make(64, layers[0])
            self.layer2 = self._make(128, layers[1], stride=2)
            self.layer3 = self._make(256, layers[2], stride=2)
            self.layer4 = self._make(512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        def _make(self, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inp != planes * Bottleneck.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inp,
                        planes * Bottleneck.expansion,
                        1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * Bottleneck.expansion),
                )
            out = [Bottleneck(self.inp, planes, stride, downsample)]
            self.inp = planes * Bottleneck.expansion
            for _ in range(1, blocks):
                out.append(Bottleneck(self.inp, planes))
            return nn.Sequential(*out)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    return ResNet().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ops", default="conv2d,linear", help="overrides to route")
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
    img = int(os.environ.get("RESNET_IMG", "224"))
    batch = int(os.environ.get("RESNET_B", "8"))
    warmup = int(os.environ.get("RESNET_WARMUP", "3"))
    iters = int(os.environ.get("RESNET_ITERS", "10"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    model = build_resnet50(torch, device, dtype)
    gen = torch.Generator(device="cpu").manual_seed(50)
    x = (
        torch.randn(batch, 3, img, img, generator=gen, dtype=torch.float32)
        .to(dtype)
        .to(device)
    )
    print(
        f"config  = ResNet-50 (bottleneck 3/4/6/3)  input[{batch},3,{img},{img}] dtype={dtype}"
    )
    print(f"routing = {ops}")
    print()

    def fwd():
        with torch.no_grad():
            return model(x)

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
                model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3

    hipdnn_torch.uninstall()
    native_ms = timed()
    hipdnn_torch.install(ops)
    over_ms = timed()
    hipdnn_torch.uninstall()
    print(f"A/B forward wall-clock (warmup={warmup} iters={iters}):")
    print(f"  native   = {native_ms:8.3f} ms/fwd")
    print(
        f"  injected = {over_ms:8.3f} ms/fwd   "
        f"speedup={native_ms/over_ms if over_ms else float('nan'):5.3f}x"
    )
    print()

    print("=== device-time census: one forward, by op category (cuda events) ===")
    print(device_time_census(torch, fwd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
