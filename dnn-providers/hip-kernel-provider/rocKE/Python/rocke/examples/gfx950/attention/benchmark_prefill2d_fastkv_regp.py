# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark experimental CK DSL prefill-2D fastKV + register-P kernels.

This is intentionally separate from ``benchmark_prefill2d_traces.py``.  It
compares the R4 tiled-2D baseline with the experimental kernel from
``attention_tiled_2d_fastkv_regp.py`` and optionally joins against a Triton CSV
by ``shape_signature``.

The emitted 2D kernel name omits specialization constants such as ``num_seqs``
and the derived binary-search trip count.  This harness appends a short shape
hash to each built ``KernelDef.name`` before compiling, so the ROCm module/HSACO
cache never sees two different shape-specialized code objects with the same
kernel symbol.

Example:

    PYTHONPATH=Python Python/rocke/.venv/bin/python \\
      Python/rocke/examples/attention/benchmark_prefill2d_fastkv_regp.py \\
      --limit 10 --iterations 100 --warmup 10

To reproduce the current best measured host-dispatch policy:

    PYTHONPATH=Python Python/rocke/.venv/bin/python \\
      Python/rocke/examples/attention/benchmark_prefill2d_fastkv_regp.py \\
      --smart-dispatch-policy latest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def _sliding_window(shape) -> int:
    return shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0


def _smart_dispatch_variant(policy: str, sliding_window: int) -> str:
    """Return the measured-best CK DSL variant for one trace shape."""

    if policy == "latest":
        return "combo_early_v" if sliding_window == 0 else "combo_t32"
    raise ValueError(f"unknown smart dispatch policy: {policy}")


def _shape_kernel_name(base_name: str, variant: str, shape_signature: str) -> str:
    digest = hashlib.sha1(f"{variant}:{shape_signature}".encode("utf-8")).hexdigest()
    return f"{base_name}_sh{digest[:12]}"


class RockeFastKvRegPBench:
    def __init__(self, *, compile_backend: str = "llvm", num_sms: int = 120) -> None:
        self.compile_backend = compile_backend
        self.num_sms = num_sms
        self._launchers: dict[tuple[Any, ...], tuple[Any, Any, str]] = {}

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

    def _base_r4_spec(self, shape, sliding_window: int, *, tile_mult: int = 2):
        from rocke.instances import UnifiedAttention2DTiledSpec

        dtype = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
        return UnifiedAttention2DTiledSpec(
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
            waves_per_eu=2,
            tile_size=tile_mult * shape.block_size,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
        )

    def _combo_mask_limit(self, shape, sliding_window: int) -> bool:
        return sliding_window == 0 and shape.softcap <= 0 and not shape.has_alibi

    def _combo_spec(self, base, shape, sliding_window: int, *, early_v: bool = False):
        from dataclasses import replace

        return replace(
            base,
            use_transposed_scalar_state=True,
            use_transposed_mask_once=True,
            use_transposed_half_local_pv=True,
            use_mfma32_skip_legacy_qreg=True,
            use_transposed_mask_limit=self._combo_mask_limit(shape, sliding_window),
            use_fast_paged_kv_desc=base.tile_size == 2 * base.block_size,
            use_early_v_schedule=early_v,
        )

    def _variant_spec_and_builder(
        self, shape, problem, variant: str, sliding_window: int
    ):
        from rocke.instances import build_unified_attention_2d_tiled, supports_tiled_2d
        from rocke.instances.gfx950.attention_tiled_2d_fastkv_regp import (
            build_unified_attention_2d_fastkv_register_p,
            make_fastkv_register_p_spec,
            supports_fastkv_register_p_2d,
        )

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

        base = self._base_r4_spec(shape, sliding_window)
        if variant == "r4":
            return base, build_unified_attention_2d_tiled, "R4"
        if variant == "r4_t32":
            return (
                self._base_r4_spec(shape, sliding_window, tile_mult=1),
                (build_unified_attention_2d_tiled),
                "R4_t32",
            )
        if variant == "combo":
            return (
                self._combo_spec(base, shape, sliding_window),
                (build_unified_attention_2d_tiled),
                "R4_s1mask_hlpv_combo",
            )
        if variant == "combo_t32":
            return (
                self._combo_spec(
                    self._base_r4_spec(shape, sliding_window, tile_mult=1),
                    shape,
                    sliding_window,
                ),
                (build_unified_attention_2d_tiled),
                "R4_s1mask_hlpv_combo_t32",
            )
        if variant == "combo_early_v":
            return (
                self._combo_spec(base, shape, sliding_window, early_v=True),
                (build_unified_attention_2d_tiled),
                "R4_s1mask_hlpv_combo_early_v",
            )
        if variant == "fastkv_regp":
            ok, reason = supports_fastkv_register_p_2d(
                head_size=shape.head_size,
                block_size=shape.block_size,
                dtype=dtype,
                num_queries_per_kv=problem.num_queries_per_kv,
                use_alibi=problem.use_alibi,
                use_qq_bias=problem.use_qq_bias,
                use_fp8=problem.use_fp8,
                q_dtype=problem.q_dtype,
                num_query_heads=shape.num_query_heads,
                num_kv_heads=shape.num_kv_heads,
                tile_size=2 * shape.block_size,
            )
            if not ok:
                raise NotImplementedError(reason)
            return (
                make_fastkv_register_p_spec(base),
                build_unified_attention_2d_fastkv_register_p,
                "R4_fastkv_regp",
            )
        if variant == "fastkv_regp_s1mask":
            ok, reason = supports_fastkv_register_p_2d(
                head_size=shape.head_size,
                block_size=shape.block_size,
                dtype=dtype,
                num_queries_per_kv=problem.num_queries_per_kv,
                use_alibi=problem.use_alibi,
                use_qq_bias=problem.use_qq_bias,
                use_fp8=problem.use_fp8,
                q_dtype=problem.q_dtype,
                num_query_heads=shape.num_query_heads,
                num_kv_heads=shape.num_kv_heads,
                tile_size=2 * shape.block_size,
            )
            if not ok:
                raise NotImplementedError(reason)
            use_mask_limit = (
                sliding_window == 0 and shape.softcap <= 0 and not shape.has_alibi
            )
            return (
                make_fastkv_register_p_spec(
                    base,
                    scalar_state=True,
                    mask_once=True,
                    mask_limit=use_mask_limit,
                    skip_legacy_qreg=True,
                ),
                build_unified_attention_2d_fastkv_register_p,
                "R4_s1mask_fastkv_regp",
            )
        if variant == "combo_regp":
            ok, reason = supports_fastkv_register_p_2d(
                head_size=shape.head_size,
                block_size=shape.block_size,
                dtype=dtype,
                num_queries_per_kv=problem.num_queries_per_kv,
                use_alibi=problem.use_alibi,
                use_qq_bias=problem.use_qq_bias,
                use_fp8=problem.use_fp8,
                q_dtype=problem.q_dtype,
                num_query_heads=shape.num_query_heads,
                num_kv_heads=shape.num_kv_heads,
                tile_size=2 * shape.block_size,
            )
            if not ok:
                raise NotImplementedError(reason)
            return (
                make_fastkv_register_p_spec(
                    base,
                    scalar_state=True,
                    mask_once=True,
                    half_local_pv=True,
                    mask_limit=self._combo_mask_limit(shape, sliding_window),
                    skip_legacy_qreg=True,
                ),
                build_unified_attention_2d_fastkv_register_p,
                "R4_s1mask_hlpv_combo_regp",
            )
        raise ValueError(f"unknown variant: {variant}")

    def _launcher(self, shape, problem, variant: str, sliding_window: int):
        from rocke import compile_kernel
        from rocke.instances.common.attention_unified import _attn_signature
        from rocke.runtime import KernelLauncher

        spec, builder, policy = self._variant_spec_and_builder(
            shape, problem, variant, sliding_window
        )
        key = (variant, shape.signature, spec.kernel_name(), self.compile_backend)
        if key not in self._launchers:
            kernel = builder(spec)
            # The display name omits compile-time shape constants.  Keep the
            # human-readable spec name in CSV, but compile a shape-unique symbol
            # to avoid ROCm HSACO/module-cache aliasing across trace records.
            kernel.name = _shape_kernel_name(kernel.name, variant, shape.signature)
            artifact = compile_kernel(kernel, capture_ir_text=False)
            self._launchers[key] = (
                KernelLauncher(
                    hsaco=artifact.hsaco,
                    kernel_name=artifact.kernel_name,
                    signature=_attn_signature(
                        spec.dtype, include_bt_stride=True, include_qq_bias_stride=True
                    ),
                    cache_key=("prefill2d_fastkv_regp", key),
                ),
                spec,
                policy,
            )
        return self._launchers[key]

    def benchmark(
        self,
        shape,
        data,
        *,
        variant: str,
        warmup: int,
        iterations: int,
        attention_flops,
    ):
        from rocke.instances.common.attention_unified import _attn_values
        from rocke.runtime import LaunchConfig, synchronize_and_release, time_launches

        sliding_window = _sliding_window(shape)
        problem = self._problem(shape, sliding_window)
        launcher, spec, policy = self._launcher(shape, problem, variant, sliding_window)
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
            "framework": "rocke_ua_tiled_2d_fastkv_regp",
            "variant": variant,
            "policy": policy,
            "kernel_name": launcher.kernel_name,
            "spec_kernel_name": spec.kernel_name(),
            "compile_backend": self.compile_backend,
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


def _load_triton_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["shape_signature"]: row for row in csv.DictReader(f)}


def _join_triton(
    rows: list[dict[str, Any]],
    triton_csv: Path,
) -> list[dict[str, Any]]:
    triton_by_sig = _load_triton_rows(triton_csv)
    joined: list[dict[str, Any]] = []
    for row in rows:
        tr = triton_by_sig.get(row.get("shape_signature", ""))
        if not row.get("success") or tr is None or tr.get("success") != "True":
            continue
        tri_ms = float(tr["latency_ms"])
        ck_ms = float(row["latency_ms"])
        out = dict(row)
        out.update(
            {
                "triton_latency_ms": tri_ms,
                "triton_tflops": tr.get("tflops", ""),
                "triton_kernel_name": tr.get("kernel_name", ""),
                "rocke_over_triton_latency": ck_ms / tri_ms if tri_ms > 0 else "",
                "triton_over_ckdsl_latency": tri_ms / ck_ms if ck_ms > 0 else "",
            }
        )
        joined.append(out)
    return joined


def _print_summary(rows: list[dict[str, Any]], joined: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("success")]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in ok_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    print("summary:")
    for variant, vals in by_variant.items():
        latencies = [float(row["latency_ms"]) for row in vals]
        print(f"  {variant}: rows={len(vals)} latency_gm={_gm(latencies):.6f} ms")

    by_shape_variant = {
        (row["shape_signature"], row["variant"]): float(row["latency_ms"])
        for row in ok_rows
    }
    r4_shapes = {row["shape_signature"] for row in by_variant.get("r4", [])}
    for variant in sorted(v for v in by_variant if v != "r4"):
        ratios = []
        for sig in r4_shapes:
            base = by_shape_variant.get((sig, "r4"))
            cur = by_shape_variant.get((sig, variant))
            if base and cur:
                ratios.append(cur / base)
        if ratios:
            print(
                f"  {variant}/r4 latency gm={_gm(ratios):.4f}x "
                f"(values < 1.0 are faster than R4)"
            )

    joined_by_variant: dict[str, list[float]] = {}
    for row in joined:
        ratio = row.get("rocke_over_triton_latency")
        if ratio != "":
            joined_by_variant.setdefault(str(row["variant"]), []).append(float(ratio))
    for variant, ratios in joined_by_variant.items():
        print(
            f"  {variant}/triton latency gm={_gm(ratios):.4f}x "
            f"(values < 1.0 are faster than Triton)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shapes", nargs="+", type=Path, default=[DEFAULT_SHAPES])
    parser.add_argument(
        "--shape-signature",
        nargs="+",
        default=None,
        help="optional exact shape_signature filter for targeted experiments",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "all"), default="bf16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cap-blocks", type=int, default=65536)
    parser.add_argument("--num-sms", type=int, default=120)
    parser.add_argument("--compile-backend", choices=("llvm",), default="llvm")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "r4",
            "r4_t32",
            "combo",
            "combo_t32",
            "combo_early_v",
            "fastkv_regp",
            "fastkv_regp_s1mask",
            "combo_regp",
        ),
        default=["r4", "fastkv_regp"],
    )
    parser.add_argument(
        "--smart-dispatch-policy",
        choices=("none", "latest"),
        default="none",
        help=(
            "benchmark one host-selected variant per shape; 'latest' uses "
            "combo_early_v for no-SW and combo_t32 for SW"
        ),
    )
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
        default=Path("/tmp/rocke_prefill2d_fastkv_regp.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/tmp/rocke_prefill2d_fastkv_regp.csv"),
    )
    parser.add_argument(
        "--joined-csv",
        type=Path,
        default=Path("/tmp/rocke_prefill2d_fastkv_regp_triton_joined.csv"),
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
    if args.shape_signature:
        wanted = set(args.shape_signature)
        shapes = [shape for shape in shapes if shape.signature in wanted]
    if args.offset:
        shapes = shapes[args.offset :]
    if args.limit is not None:
        shapes = shapes[: args.limit]
    if not shapes:
        print("No prefill-2D shapes matched.", file=sys.stderr)
        return 1

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"selected shapes: {len(shapes)}")
    if args.smart_dispatch_policy == "none":
        print(f"variants: {', '.join(args.variants)}")
    else:
        print(f"smart dispatch policy: {args.smart_dispatch_policy}")
    print(f"compile backend: {args.compile_backend}")

    bench = RockeFastKvRegPBench(
        compile_backend=args.compile_backend,
        num_sms=args.num_sms,
    )
    results: list[dict[str, Any]] = []
    for index, shape in enumerate(shapes, 1):
        print(f"[{index}/{len(shapes)}] {shape.signature}", flush=True)
        data = make_inputs(shape, seed=args.seed, cap_blocks=args.cap_blocks)
        sliding_window = _sliding_window(shape)
        if args.smart_dispatch_policy == "none":
            variant_requests = [(variant, variant) for variant in args.variants]
        else:
            selected_variant = _smart_dispatch_variant(
                args.smart_dispatch_policy, sliding_window
            )
            variant_requests = [
                (f"smart_{args.smart_dispatch_policy}", selected_variant)
            ]
        for display_variant, variant in variant_requests:
            print(f"  variant={display_variant}", flush=True)
            try:
                row = bench.benchmark(
                    shape,
                    data,
                    variant=variant,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    attention_flops=attention_flops,
                )
                if display_variant != variant:
                    row["selected_variant"] = variant
                    row["variant"] = display_variant
                    row["policy"] = f"{display_variant}:{row['policy']}"
                print(
                    f"    latency={row['latency_ms']:.6f} ms "
                    f"tflops={row['tflops']:.2f} policy={row['policy']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                row = {
                    "framework": "rocke_ua_tiled_2d_fastkv_regp",
                    "variant": display_variant,
                    "selected_variant": variant,
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

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, default=str))
    _write_csv(args.output_csv, results)
    print(f"wrote JSON: {args.output_json}")
    print(f"wrote CSV : {args.output_csv}")

    joined = _join_triton(results, args.triton_csv)
    if joined:
        _write_csv(args.joined_csv, joined)
        print(f"wrote joined CSV: {args.joined_csv}")
    else:
        print(
            f"skipped joined CSV; Triton CSV not found or no matching rows: {args.triton_csv}"
        )

    _print_summary(results, joined)
    ok = sum(1 for row in results if row.get("success"))
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
