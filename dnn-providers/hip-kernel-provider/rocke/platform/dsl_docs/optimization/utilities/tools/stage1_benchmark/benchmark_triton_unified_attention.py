# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark AITER's Triton `unified_attention` 2D kernel on traced shapes.

Reads aiter trace-shape ndjson files (`tests/aiter_ua_shapes.json`,
`tests/aiter_ua_2_shapes.json`), filters to the prefill-2D subset, and
times the AITER Triton kernel forced to its 2D path so the comparison
against the CK DSL tiled-2D kernel is apples-to-apples.

Environment Variables:
    AITER_PATH:  optional explicit path to aiter checkout (defaults to
                 an ``aiter`` checkout sibling to this repo)

Usage:
    source setup_env.sh
    source .venv/bin/activate
    # Activate FlyDSL/triton-friendly env, or any env with aiter installed.

    # All BF16 prefill-2D shapes from both trace files (deduped):
    python src/stage1_benchmark/benchmark_triton_unified_attention.py \\
        --shapes tests/aiter_ua_shapes.json tests/aiter_ua_2_shapes.json \\
        --output results/triton_ua_prefill2d.csv

    # Single shape by call_idx
    python src/stage1_benchmark/benchmark_triton_unified_attention.py \\
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

# Make `aiter` importable.
_AITER_PATH = os.environ.get("AITER_PATH")
if _AITER_PATH:
    sys.path.insert(0, _AITER_PATH)
else:
    for candidate in (REPO_ROOT.parent / "3pint" / "aiter", REPO_ROOT.parent / "aiter"):
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break

from _ua_shape_utils import (  # noqa: E402
    UAShape,
    attention_flops,
    dedupe_shapes,
    filter_prefill_2d,
    load_shapes,
    make_inputs,
)


def _force_2d_path():
    """Patch aiter.unified_attention to always pick the 2D kernel."""
    import aiter.ops.triton.attention.unified_attention as _uam

    _uam.use_2d_kernel = lambda *a, **kw: True
    return _uam


def _bench_stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


def _time_call(call_once, *, warmup: int, iters: int, stream: int) -> float:
    """Time a torch/triton callable on ``stream`` using HIP events.

    Prefer rocke.runtime.time_launches when available (records events
    on the same stream the CK DSL benchmark uses, so the two scripts
    report numbers from an identical clock). Fall back to torch events.
    """
    try:
        from rocke.runtime import time_launches  # type: ignore
    except Exception:
        import torch

        for _ in range(warmup):
            call_once()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            call_once()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) / iters

    return time_launches(call_once, warmup=warmup, iters=iters, stream=stream)


def benchmark_one(
    shape: UAShape,
    *,
    iterations: int = 20,
    warmup: int = 5,
    seed: int = 0,
    cap_blocks: int | None = 65536,
) -> dict[str, Any]:
    """Time one Triton `unified_attention` launch (2D path) on this shape."""
    from aiter.ops.triton.attention.unified_attention import unified_attention

    _force_2d_path()
    data = make_inputs(shape, seed=seed, cap_blocks=cap_blocks)
    window_size = tuple(shape.window_size)  # (size-1, 0) or (-1, -1) per trace

    def call_once():
        unified_attention(
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=data["output"],
            cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=data["max_query_len"],
            seqused_k=data["kv_lens"],
            max_seqlen_k=data["max_kv_len"],
            softmax_scale=data["scale"],
            causal=True,
            window_size=window_size,
            block_table=data["block_tables"],
            softcap=shape.softcap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=data["alibi_slopes"],
            sinks=data["sinks"],
        )

    stream = _bench_stream_handle()
    latency_ms = _time_call(call_once, warmup=warmup, iters=iterations, stream=stream)
    flops = attention_flops(shape, data["query_lens"], data["kv_lens_list"])
    tflops = (flops / 1e12) / (latency_ms / 1e3) if latency_ms > 0 else 0.0

    return {
        "framework": "triton_aiter_ua_2d",
        "kernel_name": "unified_attention_2d",
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
        "sliding_window": (
            (shape.window_size[0] + 1) if shape.window_size[0] >= 0 else 0
        ),
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
            )
        except (
            Exception
        ) as e:  # noqa: BLE001 - per-shape failures should not abort sweep
            import traceback

            traceback.print_exc()
            res = {
                "framework": "triton_aiter_ua_2d",
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
