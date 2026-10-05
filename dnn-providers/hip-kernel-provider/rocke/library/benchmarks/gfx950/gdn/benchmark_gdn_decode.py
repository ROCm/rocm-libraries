#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark registered GDN single-token decode candidates.

Candidate enumeration comes only from the dispatch registry. Production
``auto`` is reported separately and remains a static policy.
"""

from __future__ import annotations

import argparse
import statistics
import time
import sys

from dispatch.gdn import (
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
)
from kernels.gfx950.gdn_decode import GdnDecodeSpec

ARCH = "gfx950"

DEFAULT_BATCHES = (1, 16, 64, 256)


def registered_results(req: GdnDecodeRequest):
    """Return the exact registry results consumed by candidate benchmarking."""
    return dispatch_gdn_decode_all(req)


def eager_us(spec: GdnDecodeSpec, batch: int, reps: int = 200) -> float:
    """Median host-observed launch latency in microseconds.

    Inputs and the launch config are prepared once, outside the timed region.
    Each sample measures the CPU call plus the wait for that launch to finish,
    which is the latency a synchronous Python decode loop observes.
    """
    import torch
    from builders.gfx950.gdn.gdn_decode import (
        drain,
        launch,
        launcher_for,
        make_inputs,
        prepare,
    )

    launcher = launcher_for(spec)
    values, cfg = prepare(spec, make_inputs(spec, batch), batch)
    for _ in range(50):
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        launch(launcher, values, cfg)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    drain()
    return statistics.median(samples)


def device_us(spec: GdnDecodeSpec, batch: int, reps: int = 64):
    """Per-launch device time from a replayed HIP graph, or None if unavailable.

    Only launches are captured; the buffers are allocated beforehand because
    allocation during capture is illegal. A failed capture leaves the stream in
    an invalidated state, so the failure path resynchronises before returning
    rather than letting the next caller inherit a poisoned stream.
    """
    import torch
    from builders.gfx950.gdn.gdn_decode import (
        drain,
        launch,
        launcher_for,
        make_inputs,
        prepare,
    )

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
        drain()
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
    drain()
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

    print(
        f"{'batch':>6} {'arm':>5} {'tile':>10} {'spec_id':>22} {'grid':>8} "
        f"{'eager_us':>10} {'device_us':>10}  correctness"
    )

    failures = 0
    for batch in (int(x) for x in args.batches.split(",")):
        request = GdnDecodeRequest(batch=batch, arch=ARCH)
        results = registered_results(request)
        if len(results) != 54:
            print(
                f"batch {batch}: expected 54 legal registry candidates, got {len(results)}",
                file=sys.stderr,
            )
            failures += 1
            continue

        import torch

        if not torch.cuda.is_available():
            print("no HIP device visible", file=sys.stderr)
            return 2
        from builders.gfx950.gdn.gdn_decode import TOL, check

        auto = dispatch_gdn_decode(request)
        for result in (auto, *results):
            spec: GdnDecodeSpec = result.spec
            arm = "auto" if result is auto else "cand"
            tile = f"{spec.num_warps},{spec.warp_threads_k},{spec.blocks_per_v_dim}"
            grid = result.grid[0]
            out_err, state_err = check(spec, batch)
            err = max(out_err, state_err)
            if err > TOL:
                failures += 1
                print(
                    f"{batch:>6} {arm:>5} {tile:>10} "
                    f"{result.candidate.spec_id:>22} {grid:>8} "
                    f"{'-':>10} {'-':>10}  FAIL max_err={err:.3e}"
                )
                continue
            eager = eager_us(spec, batch)
            device = None if args.no_device else device_us(spec, batch)
            dev_s = f"{device:10.2f}" if device is not None else f"{'n/a':>10}"
            print(
                f"{batch:>6} {arm:>5} {tile:>10} "
                f"{result.candidate.spec_id:>22} {grid:>8} "
                f"{eager:10.2f} {dev_s}  max_err={err:.3e}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
