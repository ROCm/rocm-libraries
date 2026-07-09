# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Interleaved same-process A/B bench for the gfx1151 deep-fusion kernel.

This box auto-clocks +-25-30%, so only same-session interleaved ratios are
valid (see optimization_runbook.md S8.6). This harness builds every named
config once, verifies each against the integer-exact numpy reference, then
benches them round-robin for several rounds and reports the per-config median
plus the ratio to the first (baseline) config.

Usage:
    python -m rocke.examples.gfx1151.deep_conv_fusion.compare_configs \
        [--h 2160 --w 3840] [--rounds 5] [--iters 100]
"""

from __future__ import annotations

import argparse
import statistics
import sys

import numpy as np

from rocke.helpers import compile_kernel
from rocke.instances.gfx1151.deep_fused_conv_pool import (
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    is_valid_spec,
    make_deep_fused_conv_pool_spec,
)
from rocke.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify import (
    _as_u8_buffer,
    _make_inputs,
    _pack_args,
    _reference,
    _unpack_y,
    _useful_flops,
)
from rocke.runtime.hip_module import Runtime

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
        "--suite",
        choices=["next", "combo", "warp", "persistent", "sched"],
        default="next",
        help="candidate set to compare against the current winner baseline",
    )
    args = ap.parse_args()

    def spec(**kw):
        return make_deep_fused_conv_pool_spec(
            h=args.h, w=args.w, c=8, k0=32, k1=24, **kw
        )

    # Shared optimized base: current winning full-shape stack. Every config below
    # is this base plus one candidate toggle, so the interleaved A/B isolates the
    # next-winner hypotheses on top of the current best kernel.
    base_kw = dict(
        pool_tile_h=2,
        pool_tile_w=64,
        warp_m=16,
        warp_n=1,
        vectorize_conv0_a=True,
        vectorize_maxpool=True,
        early_w1=True,
        direct_conv0=True,
        native_int=True,
        fused_c0a1=True,
        batch_loads=True,
        sched_policy="compv4",
        conv1_prefetch_k=True,
        conv1_sched_fuse=True,
    )

    def bspec(**kw):
        merged = dict(base_kw)
        merged.update(kw)
        return spec(**merged)

    # First entry is the baseline; ratios are reported against it.
    suites = {
        "next": [
            Cfg("winner", bspec()),
            Cfg("pk-maxpool", bspec(pk_maxpool=True)),
            Cfg("conv1-int8", bspec(conv1_int8=True)),
            Cfg("i8+pk", bspec(conv1_int8=True, pk_maxpool=True)),
            Cfg("static-kmap", bspec(static_direct_kmap=True)),
            Cfg("persist8", bspec(persistent=True, persistent_ctas=8)),
            Cfg("persist16", bspec(persistent=True, persistent_ctas=16)),
            Cfg("persist32", bspec(persistent=True, persistent_ctas=32)),
        ],
        "combo": [
            Cfg("winner", bspec()),
            Cfg("conv1-int8", bspec(conv1_int8=True)),
            Cfg("i8+pk", bspec(conv1_int8=True, pk_maxpool=True)),
            Cfg("i8+static", bspec(conv1_int8=True, static_direct_kmap=True)),
            Cfg(
                "i8+static+pk",
                bspec(conv1_int8=True, static_direct_kmap=True, pk_maxpool=True),
            ),
            Cfg(
                "persist8+i8",
                bspec(persistent=True, persistent_ctas=8, conv1_int8=True),
            ),
            Cfg(
                "persist8+all",
                bspec(
                    persistent=True,
                    persistent_ctas=8,
                    conv1_int8=True,
                    static_direct_kmap=True,
                    pk_maxpool=True,
                ),
            ),
        ],
        "warp": [
            Cfg("winner", bspec()),
            Cfg("warp8x1", bspec(warp_m=8, warp_n=1)),
            Cfg("warp16x1", bspec(warp_m=16, warp_n=1)),
            Cfg("warp8x2", bspec(warp_m=8, warp_n=2)),
        ],
        "persistent": [
            Cfg("winner", bspec()),
            Cfg("persist4", bspec(persistent=True, persistent_ctas=4)),
            Cfg("persist8", bspec(persistent=True, persistent_ctas=8)),
            Cfg("persist16", bspec(persistent=True, persistent_ctas=16)),
            Cfg("persist32", bspec(persistent=True, persistent_ctas=32)),
        ],
        "sched": [
            Cfg("winner", bspec()),
            Cfg("compv3", bspec(sched_policy="compv3")),
            Cfg("compv4", bspec(sched_policy="compv4")),
            Cfg("intrawave", bspec(sched_policy="intrawave")),
            Cfg("no-pfk", bspec(conv1_prefetch_k=False)),
            Cfg("no-schfuse", bspec(conv1_sched_fuse=False)),
        ],
    }
    configs = suites[args.suite]

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
