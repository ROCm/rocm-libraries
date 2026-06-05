# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Interleaved same-process A/B bench for the gfx1151 deep-fusion kernel.

This box auto-clocks +-25-30%, so only same-session interleaved ratios are
valid (see optimization_runbook.md S8.6). This harness builds every named
config once, verifies each against the integer-exact numpy reference, then
benches them round-robin for several rounds and reports the per-config median
plus the ratio to the first (baseline) config.

Usage:
    python -m ck_dsl.examples.gfx1151.deep_conv_fusion.compare_configs \
        [--h 2160 --w 3840] [--rounds 5] [--iters 100]
"""

from __future__ import annotations

import argparse
import statistics
import sys

import numpy as np

from ck_dsl.helpers import compile_kernel
from ck_dsl.instances.gfx1151.deep_fused_conv_pool import (
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    is_valid_spec,
    make_deep_fused_conv_pool_spec,
)
from ck_dsl.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify import (
    _as_u8_buffer,
    _make_inputs,
    _pack_args,
    _reference,
    _unpack_y,
    _useful_flops,
)
from ck_dsl.runtime.hip_module import Runtime

ARCH = "gfx1151"


class Cfg:
    def __init__(self, name, spec):
        ok, why = is_valid_spec(spec, arch=ARCH)
        if not ok:
            raise ValueError(f"invalid spec {name!r}: {why}")
        self.name = name
        self.spec = spec
        self.kernel = build_deep_fused_conv_pool(spec, arch=ARCH)
        self.artifact = compile_kernel(self.kernel, arch=ARCH)
        self.grid = deep_fused_conv_pool_grid(spec)
        self.block = (spec.block_size, 1, 1)
        self.flops = _useful_flops(spec)
        self.samples = []


def _prep(rt, cfg, seed):
    K1 = cfg.spec.problem.conv1_channels
    X, W0, W1, W1_codes, Y = _make_inputs(cfg.spec, seed=seed)
    mod = rt.load_module(cfg.artifact.hsaco)
    fn = mod.get_function(cfg.artifact.kernel_name)
    X_dev = rt.alloc(X.nbytes)
    W0_dev = rt.alloc(W0.nbytes)
    Y_dev = rt.alloc(Y.nbytes)
    W1_dev = rt.alloc(W1.nbytes)
    rt.memcpy_h2d(X_dev, _as_u8_buffer(X), X.nbytes)
    rt.memcpy_h2d(W0_dev, _as_u8_buffer(W0), W0.nbytes)
    rt.memcpy_h2d(W1_dev, _as_u8_buffer(W1), W1.nbytes)
    rt.memset(Y_dev, 0, Y.nbytes)
    args = _pack_args(X_dev, W0_dev, Y_dev, W1_dev)
    # verify
    rt.launch_blocking(fn, cfg.grid, cfg.block, args)
    rt.memcpy_d2h(_as_u8_buffer(Y), Y_dev, Y.nbytes)
    got = _unpack_y(Y, K1)
    ref = _reference(X, W0, W1_codes, cfg.spec)
    bad = int(np.count_nonzero(np.abs(got - ref) > 0))
    return fn, args, (X_dev, W0_dev, Y_dev, W1_dev), bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=2160)
    ap.add_argument("--w", type=int, default=3840)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--l2only",
        action="store_true",
        help="focused base vs compv3 vs compv4 confirmation set",
    )
    args = ap.parse_args()

    def spec(**kw):
        return make_deep_fused_conv_pool_spec(
            h=args.h, w=args.w, c=8, k0=32, k1=24, **kw
        )

    # Shared optimized base (direct-conv0 default, pt2x16, bs256). Every config
    # below is this base plus one or more of the multi-lever toggles, so the
    # interleaved A/B isolates the lever(s) on top of the current best kernel.
    base_kw = dict(
        pool_tile_h=2,
        pool_tile_w=16,
        warp_m=4,
        warp_n=2,
        vectorize_conv0_a=True,
        vectorize_maxpool=True,
        early_w1=True,
        direct_conv0=True,
    )

    def bspec(**kw):
        merged = dict(base_kw)
        merged.update(kw)
        return spec(**merged)

    # First entry is the baseline; ratios are reported against it.
    # Per-lever isolation then the composition stacks (L1=waves_per_eu,
    # L2=sched_policy, L3=mask_maxpool; L4 butterfly is a rejected non-lever).
    full = [
        Cfg("base direct-conv0", bspec()),
        # --- per lever ---
        Cfg("L1 wpe2", bspec(waves_per_eu=2)),
        Cfg("L2 sch compv3", bspec(sched_policy="compv3")),
        Cfg("L2 sch compv4", bspec(sched_policy="compv4")),
        Cfg("L3 maskpool", bspec(mask_maxpool=True)),
        # --- compositions ---
        Cfg("L1+L2", bspec(waves_per_eu=2, sched_policy="compv3")),
        Cfg("L1+L3", bspec(waves_per_eu=2, mask_maxpool=True)),
        Cfg("L2+L3", bspec(sched_policy="compv3", mask_maxpool=True)),
        Cfg(
            "L1+L2+L3", bspec(waves_per_eu=2, sched_policy="compv3", mask_maxpool=True)
        ),
    ]
    # Focused L2 confirmation set (delta was inside the spread in the full run):
    l2only = [
        Cfg("base direct-conv0", bspec()),
        Cfg("L2 sch compv3", bspec(sched_policy="compv3")),
        Cfg("L2 sch compv4", bspec(sched_policy="compv4")),
    ]
    configs = l2only if args.l2only else full

    print(f"shape H={args.h} W={args.w} C=8 K0=32 K1=24")
    rt = Runtime()
    live = []
    for cfg in configs:
        fn, cfgargs, devs, bad = _prep(rt, cfg, args.seed)
        status = "OK" if bad == 0 else f"BAD={bad}"
        print(f"built {cfg.name:22s} grid={cfg.grid} block={cfg.block} verify={status}")
        if bad:
            print(f"  !! {cfg.name} FAILED verification", file=sys.stderr)
        live.append((cfg, fn, cfgargs, devs))

    # warmup all
    for cfg, fn, cfgargs, _ in live:
        for _ in range(args.warmup):
            rt.launch(fn, cfg.grid, cfg.block, cfgargs)
    rt.sync()

    for rnd in range(args.rounds):
        for cfg, fn, cfgargs, _ in live:
            start = rt.event()
            end = rt.event()
            start.record()
            for _ in range(args.iters):
                rt.launch(fn, cfg.grid, cfg.block, cfgargs)
            end.record()
            end.synchronize()
            ms = start.elapsed_to(end) / args.iters
            start.destroy()
            end.destroy()
            cfg.samples.append(ms)
        rt.sync()

    for cfg, _, _, devs in live:
        for d in devs:
            rt.free(d)

    print("\n=== Summary (median of rounds) ===")
    base = statistics.median(configs[0].samples)
    for cfg in configs:
        med = statistics.median(cfg.samples)
        spread = (max(cfg.samples) - min(cfg.samples)) / med * 100
        tflops = cfg.flops / 1e9 / med
        delta = (base / med - 1.0) * 100
        print(
            f"{cfg.name:22s} med={med:.5g} ms  spread={spread:4.1f}%  "
            f"{tflops:6.2f} TFLOP/s  ({delta:+.1f}% vs base)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
