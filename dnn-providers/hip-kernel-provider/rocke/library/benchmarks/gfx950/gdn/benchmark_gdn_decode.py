#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark scenario for the GDN single-token decode kernel.

Sweeps the decode batch range and reports per-launch time for the default
spec. Every scenario is correctness-gated first: a timing number from a
numerically wrong kernel is worse than no number at all, so a shape that fails
the gate is reported as such and excluded rather than timed.

Two timings are reported per shape because they answer different questions:

``eager``   one launch per synchronisation, i.e. what a decode loop that
            dispatches from Python actually pays. Includes host launch cost.
``device``  a captured HIP graph replayed, dividing out the per-launch host
            cost, i.e. what the GPU itself spends.

Decode at small batch is launch-bound, so ``eager`` and ``device`` can differ
by several times; reporting only one of them mis-states where the time goes.

Run::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python \\
        python3 benchmark_gdn_decode.py --batches 1,16,64,256
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

from builders.gfx950.gdn.gdn_decode import (
    TOL,
    check,
    launch,
    launcher_for,
    make_inputs,
    prepare,
)
from kernels.gfx950.gdn_decode import GdnDecodeSpec, gdn_decode_grid

DEFAULT_BATCHES = (1, 16, 64, 256)


def eager_us(spec: GdnDecodeSpec, batch: int, reps: int = 200) -> float:
    """Per-launch time with a synchronisation per iteration.

    Inputs and the launch config are prepared once, outside the timed region:
    re-allocating per iteration would time the caching allocator, which on a
    launch-bound decode kernel is the same order as the launch itself.
    """
    launcher = launcher_for(spec)
    values, cfg = prepare(spec, make_inputs(spec, batch), batch)
    for _ in range(50):
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch(launcher, values, cfg)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def device_us(spec: GdnDecodeSpec, batch: int, reps: int = 64):
    """Per-launch device time from a replayed HIP graph, or None if unavailable.

    Only launches are captured; the buffers are allocated beforehand because
    allocation during capture is illegal. A failed capture leaves the stream in
    an invalidated state, so the failure path resynchronises before returning
    rather than letting the next caller inherit a poisoned stream.
    """
    launcher = launcher_for(spec)
    values, cfg = prepare(spec, make_inputs(spec, batch), batch)
    for _ in range(10):
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                launch(launcher, values, cfg)
    except Exception as exc:  # capture is environment-sensitive; report, don't crash
        print(f"    graph capture unavailable: {type(exc).__name__}", file=sys.stderr)
        torch.cuda.synchronize()
        return None
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(40):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / reps)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches",
        default=",".join(str(b) for b in DEFAULT_BATCHES),
        help="comma-separated decode batch sizes",
    )
    ap.add_argument("--no-device", action="store_true", help="skip HIP-graph timing")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2

    spec = GdnDecodeSpec()
    print(f"kernel: {spec.kernel_name()}  block={spec.block_size}")
    print(f"{'batch':>6} {'grid':>8} {'eager_us':>10} {'device_us':>10}  correctness")

    failures = 0
    for batch in (int(x) for x in args.batches.split(",")):
        out_err, state_err = check(spec, batch)
        err = max(out_err, state_err)
        if err > TOL:
            failures += 1
            print(f"{batch:>6} {gdn_decode_grid(batch, spec)[0]:>8} "
                  f"{'-':>10} {'-':>10}  FAIL max_err={err:.3e}")
            continue
        eager = eager_us(spec, batch)
        device = None if args.no_device else device_us(spec, batch)
        dev_s = f"{device:10.2f}" if device is not None else f"{'n/a':>10}"
        print(f"{batch:>6} {gdn_decode_grid(batch, spec)[0]:>8} "
              f"{eager:10.2f} {dev_s}  max_err={err:.3e}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
