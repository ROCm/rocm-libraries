# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark CK DSL `unified_attention` tiled-2D kernel on traced shapes.

Reads aiter trace-shape ndjson files (`tests/aiter_ua_shapes.json`,
`tests/aiter_ua_2_shapes.json`), filters to the prefill-2D subset, and
times the CK DSL tiled-2D kernel (`backend="tiled"`) so the comparison
against AITER's Triton 2D kernel is apples-to-apples.

Environment Variables:
    ROCKE_ROOT:  optional explicit path to composablekernel python root
                 (defaults to the composablekernel python root inferred
                 relative to this file).

Usage:
    source setup_env.sh
    source .venv/bin/activate

    # All BF16 prefill-2D shapes from both trace files (deduped):
    python src/stage1_benchmark/benchmark_rocke_unified_attention.py \\
        --shapes tests/aiter_ua_shapes.json tests/aiter_ua_2_shapes.json \\
        --output results/rocke_ua_prefill2d.csv

    # Single shape by call_idx
    python src/stage1_benchmark/benchmark_rocke_unified_attention.py \\
        --shapes tests/aiter_ua_2_shapes.json --call-idx 0 --limit 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "stage1_benchmark"))

# Make `rocke` importable.
_ROCKE_ROOT = os.environ.get("ROCKE_ROOT")
if _ROCKE_ROOT:
    sys.path.insert(0, _ROCKE_ROOT)
else:
    # This file lives inside Python/rocke/...; REPO_ROOT is the python root.
    if REPO_ROOT.exists():
        sys.path.insert(0, str(REPO_ROOT))

from _ua_shape_utils import (  # noqa: E402
    UAShape,
    attention_flops,
    dedupe_shapes,
    filter_prefill_2d,
    load_shapes,
    make_inputs,
)


_DTYPE_TO_ROCKE = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
}


def _bench_stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


def _time_call(call_once, *, warmup: int, iters: int, stream: int) -> float:
    """Time a callable on ``stream`` using rocke HIP-event timer.

    Uses ``rocke.runtime.time_launches`` on the same stream the CK DSL
    kernel launches into (torch's current stream). Also drains the CK DSL
    retained-args bucket via ``synchronize_and_release`` so the next
    measurement cannot see leftover kernarg buffers.
    """
    from rocke.runtime import synchronize_and_release, time_launches

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    return ms


def benchmark_one(
    shape: UAShape,
    *,
    iterations: int = 20,
    warmup: int = 5,
    seed: int = 0,
    cap_blocks: int | None = 65536,
    num_sms: int = 120,
) -> dict[str, Any]:
    """Time one CK DSL `unified_attention` tiled-2D launch on this shape."""
    from rocke.instances import (
        UnifiedAttentionProblem,
        run_unified_attention_torch,
    )

    dtype_str = _DTYPE_TO_ROCKE.get(shape.q_dtype)
    if dtype_str is None:
        raise ValueError(f"unsupported q_dtype {shape.q_dtype!r} for CK DSL tiled-2D")

    data = make_inputs(shape, seed=seed, cap_blocks=cap_blocks)
    sliding_window = shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0

    problem = UnifiedAttentionProblem(
        total_q=shape.total_q,
        num_seqs=shape.num_seqs,
        num_query_heads=shape.num_query_heads,
        num_kv_heads=shape.num_kv_heads,
        head_size=shape.head_size,
        block_size=shape.block_size,
        max_seqlen_q=shape.max_seqlen_q,
        max_seqlen_k=shape.max_seqlen_k,
        dtype=dtype_str,
        sliding_window=sliding_window,
        softcap=float(shape.softcap),
        use_sinks=shape.has_sinks,
        use_alibi=shape.has_alibi,
        use_qq_bias=False,
        use_fp8=False,
        num_sms=num_sms,
    )

    hip_stream = _bench_stream_handle()

    def call_once():
        run_unified_attention_torch(
            problem=problem,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=data["output"],
            cu_seqlens_q=data["cu_seqlens_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=float(shape.softcap),
            sinks=data["sinks"],
            alibi_slopes=data["alibi_slopes"],
            qq_bias=None,
            qq_bias_stride_0=0,
            backend="tiled",
            stream=hip_stream,
        )

    latency_ms = _time_call(
        call_once, warmup=warmup, iters=iterations, stream=hip_stream
    )
    flops = attention_flops(shape, data["query_lens"], data["kv_lens_list"])
    tflops = (flops / 1e12) / (latency_ms / 1e3) if latency_ms > 0 else 0.0

    return {
        "framework": "rocke_ua_tiled_2d",
        "kernel_name": "unified_attention_tiled_2d",
        "shape_signature": shape.signature,
        "source_file": shape.source_file,
        "call_idx": shape.call_idx,
        "num_seqs": shape.num_seqs,
        "total_q": shape.total_q,
        "num_query_heads": shape.num_query_heads,
        "num_kv_heads": shape.num_kv_heads,
        "head_size": shape.head_size,
        "block_size": shape.block_size,
        "max_seqlen_q": shape.max_seqlen_q,
        "max_seqlen_k": shape.max_seqlen_k,
        "sliding_window": sliding_window,
        "softcap": shape.softcap,
        "has_sinks": shape.has_sinks,
        "has_alibi": shape.has_alibi,
        "dtype": shape.q_dtype,
        "latency_ms": latency_ms,
        "tflops": tflops,
        "iterations": iterations,
        "warmup": warmup,
        "success": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=Path,
        default=[
            REPO_ROOT / "tests" / "aiter_ua_shapes.json",
            REPO_ROOT / "tests" / "aiter_ua_2_shapes.json",
        ],
        help="ndjson shape file(s) to load",
    )
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "all"))
    parser.add_argument(
        "--sliding-window",
        choices=("any", "only", "none"),
        default="any",
        help="filter sliding-window shapes",
    )
    parser.add_argument(
        "--call-idx", type=int, default=None, help="select a single call_idx"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap number of shapes")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cap-blocks",
        type=int,
        default=65536,
        help="cap paged KV-cache num_blocks to bound HBM use (default 65536)",
    )
    parser.add_argument(
        "--num-sms",
        type=int,
        default=120,
        help="num_sms hint passed to UnifiedAttentionProblem (default 120 for MI300/MI350)",
    )
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="optional JSON dump of all results",
    )
    args = parser.parse_args()

    shapes = load_shapes(args.shapes)
    dtype_filter = None if args.dtype == "all" else args.dtype
    sw_filter = {"any": None, "only": True, "none": False}[args.sliding_window]
    shapes = filter_prefill_2d(
        shapes, dtype=dtype_filter, require_sliding_window=sw_filter
    )
    shapes = dedupe_shapes(shapes)

    if args.call_idx is not None:
        shapes = [s for s in shapes if s.call_idx == args.call_idx]
    if args.limit is not None:
        shapes = shapes[: args.limit]

    if not shapes:
        print("No shapes matched the filters.", file=sys.stderr)
        return 1

    print(f"Selected {len(shapes)} prefill-2D shape(s).")
    results: list[dict[str, Any]] = []
    for idx, shape in enumerate(shapes):
        print(f"[{idx + 1}/{len(shapes)}] {shape.signature}")
        try:
            res = benchmark_one(
                shape,
                iterations=args.iterations,
                warmup=args.warmup,
                seed=args.seed,
                cap_blocks=args.cap_blocks,
                num_sms=args.num_sms,
            )
        except (
            Exception
        ) as e:  # noqa: BLE001 - per-shape failures should not abort sweep
            import traceback

            traceback.print_exc()
            res = {
                "framework": "rocke_ua_tiled_2d",
                "shape_signature": shape.signature,
                "source_file": shape.source_file,
                "call_idx": shape.call_idx,
                "success": False,
                "error": repr(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        results.append(res)
        if res.get("success"):
            print(f"  latency={res['latency_ms']:.3f} ms  tflops={res['tflops']:.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cols = sorted({k for r in results for k in r.keys()})
        with args.output.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"Wrote {args.output}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"Wrote {args.json_output}")

    n_ok = sum(1 for r in results if r.get("success"))
    return 0 if n_ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
