# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Replay fused/unfused peer GEMM correctness; optional measurements stay session-only.

Run from the platform root with the authoring package on PYTHONPATH. GPU resources
must be explicitly allocated by the caller; this entry point never submits jobs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ....benchmark.gemm.peer_runner import PeerGemmRunner
from ....instances.common.peer_gemm import (
    PeerGemmScheduleSpec,
    PeerGemmSpec,
    is_valid_schedule_spec,
    is_valid_spec,
)


def main(argv: list[str] | None = None) -> int:
    """Write correctness metadata, not achieved performance, to an explicit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices", required=True, help="comma-separated allocated HIP ordinals"
    )
    parser.add_argument("--arch", required=True, choices=["gfx942", "gfx950"])
    parser.add_argument("--dtype", default="bf16", choices=["f16", "bf16"])
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--inputs", default="gather", choices=["gather", "replicated"])
    parser.add_argument("--outputs", default="scatter", choices=["local", "scatter"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--correctness-output", type=Path)
    parser.add_argument(
        "--schedule",
        default="legacy",
        choices=["legacy", "source_grouped", "tile_interleaved"],
    )
    parser.add_argument(
        "--max-ctas",
        type=int,
        default=0,
        help="cap submitted CTAs; zero means one CTA per output tile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="untimed iterations per path before opt-in measurement",
    )
    parser.add_argument(
        "--measure-samples",
        type=int,
        default=0,
        help="opt-in session output only; retain numbers only in protected AMD Confluence",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1 or min(args.warmup, args.measure_samples) < 0:
        parser.error(
            "repeats must be positive; warmup and measure-samples must be non-negative"
        )
    try:
        devices = tuple(int(token) for token in args.devices.split(","))
    except ValueError:
        parser.error("devices must be comma-separated integer HIP ordinals")
    if len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        parser.error("devices must be unique non-negative HIP ordinals")
    spec = PeerGemmSpec(
        args.rows, args.cols, args.k, sources=len(devices), dtype=args.dtype
    )
    valid, why = is_valid_spec(spec, args.arch)
    if not valid:
        parser.error(why)
    if args.schedule == "legacy" and args.max_ctas != 0:
        parser.error("max-ctas requires an opt-in non-legacy schedule")
    schedule = (
        None
        if args.schedule == "legacy"
        else PeerGemmScheduleSpec(spec, args.schedule, args.max_ctas)
    )
    if schedule is not None:
        valid, why = is_valid_schedule_spec(schedule, args.arch)
        if not valid:
            parser.error(why)
    report = {
        "spec": asdict(spec),
        "devices": devices,
        "arch": args.arch,
        "schedule": None if schedule is None else asdict(schedule),
        "checks": [],
    }
    with PeerGemmRunner(
        spec,
        devices=devices,
        arch=args.arch,
        input_placement=args.inputs,
        output_placement=args.outputs,
        schedule=schedule,
    ) as runner:
        report["checks"].append(runner.validate(fused=True, repeats=args.repeats))
        report["checks"].append(runner.validate(fused=False, repeats=args.repeats))
        if args.measure_samples:
            for fused in (True, False):
                for _ in range(args.warmup):
                    runner.enqueue(fused=fused)
                runner.drain()
            # Separate session output: correctness JSON never contains measured values.
            print(
                "MEASUREMENTS: session/protected AMD Confluence only; do not redirect into repository artifacts"
            )
            for sample in range(args.measure_samples):
                for fused in ((True, False) if sample % 2 == 0 else (False, True)):
                    print(
                        json.dumps(
                            {
                                "sample": sample,
                                "schedule": report["schedule"],
                                **asdict(runner.measure(fused=fused)),
                            }
                        )
                    )
    # Publish a correctness report only after successful all-rank teardown.
    if args.correctness_output is not None:
        args.correctness_output.parent.mkdir(parents=True, exist_ok=True)
        args.correctness_output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "ranks": len(devices),
                "paths": ["fused", "unfused"],
                "dtype": args.dtype,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
