# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark CK DSL unified-attention 2D on traced AITER prefill shapes.

This reproduces the 142-shape bf16 prefill-2D experiment used by the CK DSL
attention notes. It loads AITER trace records, runs the CK DSL 2D combo policy
with direct LLVM compilation, and optionally joins against a pre-profiled
Triton CSV by ``shape_signature``.

Example:

    PYTHONPATH=Python python \\
      Python/rocke/examples/attention/benchmark_prefill2d_traces.py

The default paths point at the local MLSE benchmark checkout used to collect
the traces. Override ``--shapes`` / ``--triton-csv`` for a different machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]  # rocKE root
sys.path.insert(0, str(ROOT / "Python"))

# Root of the in-tree optimization utilities (replaces the old external MLSE checkout).
_DSL_DOCS = ROOT / "dsl_docs" / "optimization" / "utilities"
DEFAULT_SHAPE_UTILS = _DSL_DOCS / "tools" / "stage1_benchmark"
DEFAULT_SHAPES = DEFAULT_SHAPE_UTILS / "tests" / "aiter_ua_prefill2d_allbf16.json"
DEFAULT_TRITON_CSV = DEFAULT_SHAPE_UTILS / "results" / "triton_ua_prefill2d_bf16.csv"


def _add_shape_utils_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_shape_utils(path: Path):
    _add_shape_utils_path(path)
    from _ua_shape_utils import (  # type: ignore
        attention_flops,
        dedupe_shapes,
        filter_prefill_2d,
        load_shapes,
        make_inputs,
    )

    return attention_flops, dedupe_shapes, filter_prefill_2d, load_shapes, make_inputs


def _bench_stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


def _gm(vals: list[float]) -> float:
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


def _use_combo_policy(shape, sliding_window: int) -> bool:
    """Current experimental 2D policy from parity_unified_attention.py."""

    if shape.q_dtype != "torch.bfloat16":
        return False
    if shape.head_size != 64 or shape.block_size != 32:
        return False
    if shape.num_query_heads != 64 or shape.num_kv_heads != 8:
        return False
    if shape.softcap > 0 or shape.has_alibi:
        return False
    if shape.max_seqlen_q <= 256:
        return False
    # Sliding window is now served by the combo too (with mask_once /
    # mask_limit disabled, which the spec requires for SW): with
    # waves_per_eu=3 the combo wins across the whole SW range. This mirrors
    # production ``_enable_combo_2d`` in attention_unified.py. NOTE: the
    # canonical workbench is now ``benchmark_prefill2d_live.py`` (live
    # Triton + correctness); this script is kept for the CSV-join workflow.
    return True


class RockeComboBench:
    def __init__(self, *, compile_backend: str = "llvm", num_sms: int = 120) -> None:
        self.compile_backend = compile_backend
        self.num_sms = num_sms
        self._launchers: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def _problem(self, shape, sliding_window: int):
        from rocke.instances import UnifiedAttentionProblem

        return UnifiedAttentionProblem(
            total_q=shape.total_q,
            num_seqs=shape.num_seqs,
            num_query_heads=shape.num_query_heads,
            num_kv_heads=shape.num_kv_heads,
            head_size=shape.head_size,
            block_size=shape.block_size,
            max_seqlen_q=shape.max_seqlen_q,
            max_seqlen_k=shape.max_seqlen_k,
            dtype="bf16" if shape.q_dtype == "torch.bfloat16" else "fp16",
            sliding_window=sliding_window,
            softcap=float(shape.softcap),
            use_sinks=shape.has_sinks,
            use_alibi=shape.has_alibi,
            use_qq_bias=False,
            use_fp8=False,
            num_sms=self.num_sms,
            compile_backend=self.compile_backend,
        )

    def _launcher(self, shape, problem, sliding_window: int):
        from rocke import compile_kernel
        from rocke.instances import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
            supports_tiled_2d,
        )
        from rocke.instances.common.attention_unified import _attn_signature
        from rocke.runtime import KernelLauncher

        dtype = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
        ok, reason = supports_tiled_2d(
            head_size=shape.head_size,
            block_size=shape.block_size,
            dtype=dtype,
            num_queries_per_kv=problem.num_queries_per_kv,
            use_alibi=problem.use_alibi,
            use_qq_bias=problem.use_qq_bias,
            use_fp8=problem.use_fp8,
            q_dtype=problem.q_dtype,
            num_warps=4,
            kv_storage_dtype=None,
            tile_size=2 * shape.block_size,
        )
        if not ok:
            raise NotImplementedError(reason)

        use_combo = _use_combo_policy(shape, sliding_window)
        spec = UnifiedAttention2DTiledSpec(
            head_size=shape.head_size,
            block_size=shape.block_size,
            num_query_heads=shape.num_query_heads,
            num_kv_heads=shape.num_kv_heads,
            dtype=dtype,
            use_sinks=shape.has_sinks,
            sliding_window=sliding_window,
            has_softcap=shape.softcap > 0,
            use_alibi=shape.has_alibi,
            use_qq_bias=False,
            num_seqs=shape.num_seqs,
            num_warps=4,
            # waves_per_eu=3 unlocks the 4th WG/CU for the VGPR-limited combo
            # (+15% on the cohort); matches production _select_2d_waves_per_eu.
            waves_per_eu=3 if use_combo else 2,
            tile_size=2 * shape.block_size,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=use_combo,
            # mask_once / mask_limit are only valid for full attention; the
            # spec validator rejects them under sliding window.
            use_transposed_mask_once=use_combo and sliding_window == 0,
            use_transposed_half_local_pv=use_combo,
            use_mfma32_skip_legacy_qreg=use_combo,
            use_transposed_mask_limit=use_combo and sliding_window == 0,
            use_fast_paged_kv_desc=use_combo,
        )
        # The display name intentionally omits several compile-time constants
        # (notably num_seqs/binary_search_iters). Cache on the trace signature
        # as well so shapes with the same kernel_name never reuse an HSACO that
        # was specialized for a different batch geometry.
        key = (shape.signature, spec.kernel_name(), self.compile_backend)
        if key not in self._launchers:
            kernel = build_unified_attention_2d_tiled(spec)
            artifact = compile_kernel(kernel, capture_ir_text=False)
            self._launchers[key] = (
                KernelLauncher(
                    hsaco=artifact.hsaco,
                    kernel_name=artifact.kernel_name,
                    signature=_attn_signature(
                        dtype, include_bt_stride=True, include_qq_bias_stride=True
                    ),
                    cache_key=("prefill2d_trace_combo", key),
                ),
                spec,
            )
        return self._launchers[key], use_combo

    def benchmark(self, shape, data, *, warmup: int, iterations: int, attention_flops):
        from rocke.instances.common.attention_unified import _attn_values
        from rocke.runtime import LaunchConfig, synchronize_and_release, time_launches

        sliding_window = shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0
        problem = self._problem(shape, sliding_window)
        (launcher, spec), use_combo = self._launcher(shape, problem, sliding_window)
        hip_stream = _bench_stream_handle()
        vals = _attn_values(
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
            bt_stride=int(data["block_tables"].stride(0)),
            include_bt_stride=True,
            alibi_slopes=data["alibi_slopes"],
            qq_bias=None,
            qq_bias_stride_0=0,
            include_qq_bias_stride=True,
        )
        cfg = LaunchConfig(
            grid=(
                int(shape.num_kv_heads),
                int(shape.total_q // spec.block_q + shape.num_seqs),
                1,
            ),
            block=(64 * spec.num_warps, 1, 1),
            stream=hip_stream,
        )

        def call_once():
            launcher(vals, config=cfg)

        latency_ms = time_launches(
            call_once, warmup=warmup, iters=iterations, stream=hip_stream
        )
        synchronize_and_release(hip_stream)
        flops = attention_flops(shape, data["query_lens"], data["kv_lens_list"])
        tflops = (flops / 1e12) / (latency_ms / 1e3) if latency_ms > 0 else 0.0
        return {
            "framework": "rocke_ua_tiled_2d_combo",
            "kernel_name": spec.kernel_name(),
            "compile_backend": self.compile_backend,
            "policy": "R4_s1mask_hlpv_combo" if use_combo else "R4_fallback",
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_joined_csv(
    path: Path, ck_rows: list[dict[str, Any]], triton_csv: Path
) -> None:
    if not triton_csv.exists():
        return
    ck_by_sig = {row["shape_signature"]: row for row in ck_rows if row.get("success")}
    joined = []
    with triton_csv.open(newline="") as f:
        for tr in csv.DictReader(f):
            ck = ck_by_sig.get(tr["shape_signature"])
            if ck is None or tr.get("success") != "True":
                continue
            tri_ms = float(tr["latency_ms"])
            ck_ms = float(ck["latency_ms"])
            joined.append(
                {
                    "shape_signature": tr["shape_signature"],
                    "call_idx": tr.get("call_idx", ck.get("call_idx", "")),
                    "dtype": tr.get("dtype", ck.get("dtype", "")),
                    "num_seqs": tr.get("num_seqs", ck.get("num_seqs", "")),
                    "total_q": tr.get("total_q", ck.get("total_q", "")),
                    "head_size": tr.get("head_size", ck.get("head_size", "")),
                    "block_size": tr.get("block_size", ck.get("block_size", "")),
                    "max_seqlen_q": tr.get("max_seqlen_q", ck.get("max_seqlen_q", "")),
                    "max_seqlen_k": tr.get("max_seqlen_k", ck.get("max_seqlen_k", "")),
                    "sliding_window": tr.get(
                        "sliding_window", ck.get("sliding_window", "")
                    ),
                    "triton_latency_ms": tri_ms,
                    "triton_tflops": tr.get("tflops", ""),
                    "rocke_latency_ms": ck_ms,
                    "rocke_tflops": ck.get("tflops", ""),
                    "speedup_triton_over_ckdsl": tri_ms / ck_ms,
                    "rocke_policy": ck.get("policy", ""),
                    "rocke_kernel_name": ck.get("kernel_name", ""),
                    "rocke_compile_backend": ck.get("compile_backend", ""),
                    "triton_kernel_name": tr.get("kernel_name", ""),
                    "iterations": ck.get("iterations", tr.get("iterations", "")),
                    "warmup": ck.get("warmup", tr.get("warmup", "")),
                }
            )
    _write_csv(path, joined)

    speedups = [float(row["speedup_triton_over_ckdsl"]) for row in joined]
    print(f"joined rows: {len(joined)}")
    print(f"overall geomean speedup: {_gm(speedups):.4f}x")
    for topn in (20, 30, 50):
        subset = speedups[:topn]
        if not subset:
            continue
        wins = sum(1 for value in subset if value > 1.0)
        denom = len(subset)
        print(
            f"top {topn}: wins={wins}/{denom} "
            f"win_pct={wins / denom * 100:.1f}% gm={_gm(subset):.2f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shapes", nargs="+", type=Path, default=[DEFAULT_SHAPES])
    parser.add_argument("--dtype", choices=("bf16", "fp16", "all"), default="bf16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cap-blocks", type=int, default=65536)
    parser.add_argument("--num-sms", type=int, default=120)
    parser.add_argument("--compile-backend", choices=("llvm",), default="llvm")
    parser.add_argument(
        "--shape-utils-path",
        type=Path,
        default=DEFAULT_SHAPE_UTILS,
        help="directory containing _ua_shape_utils.py",
    )
    parser.add_argument("--triton-csv", type=Path, default=DEFAULT_TRITON_CSV)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/tmp/rocke_prefill2d_trace_combo.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/tmp/rocke_prefill2d_trace_combo.csv"),
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=Path(
            "Python/rocke/examples/attention/prefill2d_bf16_triton_ckdsl_perf.csv"
        ),
        help="optional Triton+CK joined CSV; relative paths are rooted at composablekernel/",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA/HIP device unavailable", file=sys.stderr)
        return 1

    attention_flops, dedupe_shapes, filter_prefill_2d, load_shapes, make_inputs = (
        _load_shape_utils(args.shape_utils_path)
    )

    dtype_filter = None if args.dtype == "all" else args.dtype
    shapes = filter_prefill_2d(load_shapes(args.shapes), dtype=dtype_filter)
    shapes = dedupe_shapes(shapes)
    if args.limit is not None:
        shapes = shapes[: args.limit]
    if not shapes:
        print("No prefill-2D shapes matched.", file=sys.stderr)
        return 1

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"selected shapes: {len(shapes)}")
    print(f"compile backend: {args.compile_backend}")

    bench = RockeComboBench(
        compile_backend=args.compile_backend,
        num_sms=args.num_sms,
    )
    results: list[dict[str, Any]] = []
    for index, shape in enumerate(shapes, 1):
        print(f"[{index}/{len(shapes)}] {shape.signature}", flush=True)
        try:
            data = make_inputs(shape, seed=args.seed, cap_blocks=args.cap_blocks)
            row = bench.benchmark(
                shape,
                data,
                warmup=args.warmup,
                iterations=args.iterations,
                attention_flops=attention_flops,
            )
            print(
                f"  latency={row['latency_ms']:.6f} ms "
                f"tflops={row['tflops']:.2f} policy={row['policy']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            row = {
                "framework": "rocke_ua_tiled_2d_combo",
                "shape_signature": shape.signature,
                "source_file": shape.source_file,
                "call_idx": shape.call_idx,
                "success": False,
                "error": repr(exc),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        results.append(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, default=str))
    _write_csv(args.output_csv, results)
    print(f"wrote JSON: {args.output_json}")
    print(f"wrote CSV : {args.output_csv}")

    combined = args.combined_csv
    if not combined.is_absolute():
        combined = ROOT / combined
    _write_joined_csv(combined, results, args.triton_csv)
    if args.triton_csv.exists():
        print(f"wrote joined CSV: {combined}")
    else:
        print(f"skipped joined CSV; Triton CSV not found: {args.triton_csv}")

    ok = sum(1 for row in results if row.get("success"))
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
