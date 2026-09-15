# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Interleaved same-process A/B bench for PREBUILT gfx1151 deep-fusion hsacos.

Board analog of ``compare_configs.py``: that one calls ``compile_kernel`` (needs
the gfx950 toolchain), this one loads prebuilt ``gfx11-generic`` hsaco+manifest
pairs so it runs on the toolchain-less Windows Strix Halo board. Every other step
-- buffer setup, bit-exact verify against the integer-exact numpy reference, and
the round-robin timing -- is reused verbatim from ``run_manifest`` so the numbers
are directly comparable.

The board auto-clocks +-25-30%, so only same-session interleaved ratios are
valid: all configs are loaded once, verified, then benched round-robin for
several rounds; we report each config's median and its ratio to the first
(baseline) config.

Usage (on the board, in C:\\Users\\Administrator\\ck_test):
    build_venv\\Scripts\\python.exe \\
      rocke\\examples\\gfx1151\\deep_conv_fusion\\compare_prebuilt.py \\
        --rounds 3 --iters 100 --warmup 200 \\
        base=base.hsaco:manifest_base.json \\
        fuse=fuse.hsaco:manifest_fuse.json
Each positional is ``name=hsaco[:manifest]``; manifest defaults to the hsaco's
sibling ``manifest.json`` when omitted.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from rocke.run_manifest import (
    _deep_fused_conv_pool_i8i4_problem,
    _load,
)
from rocke.runtime.hip_module import Runtime


class Cfg:
    def __init__(self, name: str, hsaco: Path, manifest: Path, verify: bool):
        self.name = name
        man, blob, _ = _load(manifest, hsaco)
        kind = str(man["kind"])
        if kind != "deep_fused_conv_pool_i8i4":
            raise ValueError(
                f"{name!r}: only deep_fused_conv_pool_i8i4 supported, got {kind!r}"
            )
        self.manifest = man
        self.blob = blob
        (
            self._make_args,
            self.grid,
            self.block,
            self.flop,
            self._bytes,
            self._check,
        ) = _deep_fused_conv_pool_i8i4_problem(man, None, verify)
        self.samples: list[float] = []


def _parse_entry(s: str) -> tuple[str, Path, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected name=hsaco[:manifest], got {s!r}")
    name, rest = s.split("=", 1)
    if ":" in rest:
        hs, man = rest.split(":", 1)
        return name, Path(hs), Path(man)
    hs = Path(rest)
    return name, hs, hs.with_name("manifest.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", type=_parse_entry)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument(
        "--rotate",
        action="store_true",
        help=(
            "rotate the config order each round so every config spends equal "
            "time in each load position (defeats the first-position clock-ramp "
            "bias on auto-clocking boards)"
        ),
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the bit-exact check (timing only)",
    )
    args = ap.parse_args()
    verify = not args.no_verify

    rt = Runtime()
    live = []
    for name, hs, man in args.configs:
        cfg = Cfg(name, hs, man, verify)
        module = rt.load_module(cfg.blob)
        fn = module.get_function(str(cfg.manifest["kernel_name"]))
        cargs, ptrs = cfg._make_args(rt)
        # warm one launch then verify
        rt.launch_blocking(fn, cfg.grid, cfg.block, cargs)
        max_abs, bad, total = cfg._check(rt, ptrs)
        status = "OK" if (not verify or bad == 0) else f"BAD={bad}/{total}"
        print(
            f"loaded {cfg.name:18s} grid={cfg.grid} block={cfg.block} "
            f"verify={status} (max_abs_diff={max_abs:g})"
        )
        if verify and bad:
            print(f"  !! {cfg.name} FAILED verification", file=sys.stderr)
        live.append((cfg, fn, cargs, ptrs))

    for cfg, fn, cargs, _ in live:
        for _ in range(args.warmup):
            rt.launch(fn, cfg.grid, cfg.block, cargs)
    rt.sync()

    n = len(live)
    for r in range(args.rounds):
        order = live[r % n :] + live[: r % n] if args.rotate else live
        for cfg, fn, cargs, _ in order:
            e0 = rt.event()
            e1 = rt.event()
            e0.record()
            for _ in range(args.iters):
                rt.launch(fn, cfg.grid, cfg.block, cargs)
            e1.record()
            e1.synchronize()
            cfg.samples.append(e0.elapsed_to(e1) / args.iters)
            e0.destroy()
            e1.destroy()
        rt.sync()

    for cfg, _, _, ptrs in live:
        for p in ptrs:
            rt.free(p)

    print("\n=== Summary (median of rounds) ===")
    base = statistics.median(live[0][0].samples)
    for cfg, _, _, _ in live:
        med = statistics.median(cfg.samples)
        spread = (max(cfg.samples) - min(cfg.samples)) / med * 100
        tflops = cfg.flop / 1e9 / med
        delta = (base / med - 1.0) * 100
        print(
            f"{cfg.name:18s} med={med:.5g} ms  spread={spread:4.1f}%  "
            f"{tflops:6.2f} TFLOP/s  ({delta:+.1f}% vs base)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
