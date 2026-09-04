# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Live gfx942 comparison of AITER FlyDSL and rocKE FP8 MQA logits.

Both paths consume the same tensors, write the same dense FP32 output, use the
same stream, and are timed by the same HIP event timer. The rocKE build and
launch path comes from the packaged example, so documentation and benchmarking
exercise the same instance builder.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys

from rocke.examples.gfx942.fp8_mqa_logits.fp8_mqa_logits_verify import (
    ARCH,
    build_runner,
    compare_outputs,
    gfx_name,
    make_inputs,
    parse_shape,
    select_spec,
    variant_name,
)
from rocke.runtime import synchronize_and_release, time_launches


DEFAULT_SHAPES = (
    (4096, 4096),
    (8192, 8192),
    (128, 32768),
    (671, 131072),
)


def _time_pair(
    aiter_call,
    rocke_call,
    *,
    stream: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[float, float]:
    """Alternate timing order and return median AITER/rocKE latencies."""

    aiter_samples = []
    rocke_samples = []
    for repeat in range(repeats):
        ordered = (
            (("aiter", aiter_call), ("rocke", rocke_call))
            if repeat % 2 == 0
            else (("rocke", rocke_call), ("aiter", aiter_call))
        )
        for name, call in ordered:
            elapsed = time_launches(
                call,
                warmup=warmup,
                iters=iters,
                stream=stream,
            )
            synchronize_and_release(stream)
            if name == "aiter":
                aiter_samples.append(elapsed)
            else:
                rocke_samples.append(elapsed)
    return statistics.median(aiter_samples), statistics.median(rocke_samples)


def _write_csv(rows: list[dict], output: Path | None) -> None:
    """Write the result table to stdout and, when requested, a CSV file."""

    fields = list(rows[0])
    stdout_writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    stdout_writer.writeheader()
    stdout_writer.writerows(rows)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=DEFAULT_SHAPES,
    )
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-kv", type=int)
    parser.add_argument("--rows-per-block", type=int)
    parser.add_argument("--waves-per-block", type=int)
    parser.add_argument("--waves-per-eu", type=int, default=2)
    parser.add_argument("--target-blocks-per-cu", type=int, default=4)
    parser.add_argument("--num-splits", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="also persist the emitted result table",
    )
    args = parser.parse_args()

    if gfx_name() != ARCH:
        raise RuntimeError(
            f"this comparison requires {ARCH}; current device is {gfx_name()}"
        )
    try:
        from aiter.ops.flydsl import flydsl_fp8_mqa_logits
    except ImportError as exc:
        raise RuntimeError(
            "AITER with the FlyDSL fp8_mqa_logits op must be on PYTHONPATH"
        ) from exc

    rows = []
    for seq_q, seq_kv in args.shapes:
        inputs = make_inputs(
            seq_q,
            seq_kv,
            args.num_heads,
            args.head_dim,
        )
        spec = select_spec(
            seq_q,
            seq_kv,
            args.num_heads,
            args.head_dim,
            block_kv=args.block_kv,
            rows_per_block=args.rows_per_block,
            waves_per_block=args.waves_per_block,
            waves_per_eu=None if args.waves_per_eu == 0 else args.waves_per_eu,
        )
        rocke_call, rocke_output, stream, num_splits, _kernel_name = build_runner(
            inputs,
            seq_q,
            seq_kv,
            spec,
            target_blocks_per_cu=args.target_blocks_per_cu,
            num_splits_override=args.num_splits,
        )

        def aiter_call():
            return flydsl_fp8_mqa_logits(
                inputs["q"],
                inputs["kv"],
                inputs["kv_scales"],
                inputs["weights"],
                inputs["starts"],
                inputs["ends"],
                True,
            )

        aiter_output = aiter_call()
        rocke_call()
        synchronize_and_release(stream)
        diff, _max_abs = compare_outputs(aiter_output, rocke_output, seq_q)
        aiter_ms, rocke_ms = _time_pair(
            aiter_call,
            rocke_call,
            stream=stream,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        rows.append(
            {
                "seq_q": seq_q,
                "seq_kv": seq_kv,
                "aiter_ms": aiter_ms,
                "rocke_ms": rocke_ms,
                "rocke_vs_aiter": aiter_ms / rocke_ms,
                "calc_diff": diff,
                "rocke_variant": variant_name(spec, num_splits),
            }
        )

    _write_csv(rows, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
