#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark registered GDN single-token decode tiles.

Candidate enumeration comes only from the dispatch registry: each registered
tile's base configuration (every knob at its default). Production ``auto`` is
reported separately and remains a static policy. Timing is cold memory: the
driver's shared helpers (``gdn_decode.device_us`` / ``eager_us``) cycle through
independent copies of the inputs and state (``--rotate-mb``; ``0`` = warm,
debugging only, and the output says so), and report the median.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from dispatch.gdn import (
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_bases,
)
from dispatch.gdn.cli_args import check_rotate_mb, parse_batches
from kernels.gfx950.gdn_decode import GdnDecodeSpec

ARCH = "gfx950"

DEFAULT_BATCHES = (1, 16, 64, 256)


def registered_results(req: GdnDecodeRequest):
    """Return the exact registry results consumed by candidate benchmarking:
    each registered tile's base configuration, in registry order."""
    return dispatch_gdn_decode_bases(req)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches",
        type=parse_batches,
        default=",".join(str(b) for b in DEFAULT_BATCHES),
        help="comma-separated positive decode batch sizes",
    )
    ap.add_argument("--no-device", action="store_true", help="skip HIP-graph timing")
    ap.add_argument(
        "--rotate-mb",
        type=int,
        default=None,
        help="MiB one cold-memory rotation cycle touches at least (default: the "
        "driver's ROTATE_MB; see gdn_decode.cold_copies); 0 times warm, for "
        "debugging only",
    )
    args = ap.parse_args(argv)
    check_rotate_mb(args.rotate_mb, ap)

    failures = 0
    rotate_mb = None
    for batch in args.batches:
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
        from builders.gfx950.gdn.gdn_decode import (
            ROTATE_MB,
            check,
            device_us,
            eager_us,
            make_inputs,
            timing_label,
        )

        if rotate_mb is None:
            rotate_mb = ROTATE_MB if args.rotate_mb is None else args.rotate_mb
            print(timing_label(rotate_mb))
            print(
                f"{'batch':>6} {'arm':>5} {'tile':>10} {'spec_id':>22} {'grid':>8} "
                f"{'eager_us':>10} {'device_us':>10}  correctness"
            )

        auto = dispatch_gdn_decode(request)
        for result in (auto, *results):
            spec: GdnDecodeSpec = result.spec.kernel_spec
            arm = "auto" if result is auto else "cand"
            tile = f"{spec.num_warps},{spec.warp_threads_k},{spec.blocks_per_v_dim}"
            grid = result.grid[0]
            verdict = check(spec, batch)
            err = verdict.worst
            if not verdict.passed:
                failures += 1
                print(
                    f"{batch:>6} {arm:>5} {tile:>10} "
                    f"{result.candidate.spec_id:>22} {grid:>8} "
                    f"{'-':>10} {'-':>10}  FAIL max_err={err:.3e} "
                    f"spill={verdict.spill}"
                )
                continue
            inp = make_inputs(spec, batch)
            eager = eager_us(spec, inp, batch, rotate_mb=rotate_mb)
            device = None
            if not args.no_device:
                timed = device_us(spec, inp, batch, rotate_mb=rotate_mb)
                if timed is not None:
                    device = timed[0]
            dev_s = f"{device:10.2f}" if device is not None else f"{'n/a':>10}"
            print(
                f"{batch:>6} {arm:>5} {tile:>10} "
                f"{result.candidate.spec_id:>22} {grid:>8} "
                f"{eager:10.2f} {dev_s}  max_err={err:.3e}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
