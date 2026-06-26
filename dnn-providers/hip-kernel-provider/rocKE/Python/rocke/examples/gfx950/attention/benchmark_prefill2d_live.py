# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Authoritative prefill-2D benchmark: LIVE Triton vs CK DSL variant sweep.

Unlike ``benchmark_prefill2d_traces.py`` (which only times CK DSL and joins a
pre-profiled Triton CSV), this harness:

  * runs AITER's Triton ``unified_attention`` LIVE, forced to the 2D kernel,
    on the same stream + timer as CK DSL (apples-to-apples);
  * sweeps a set of CK DSL 2D kernel variants per shape;
  * checks correctness of every CK DSL variant against the Triton output;
  * reports, per shape and per bucket (sw / no-sw, bf16 / fp8), the best
    correct CK DSL variant and its speedup over Triton.

It is the canonical workbench for closing the prefill-2D gap.

Run:

    export AITER_PATH=<path/to/aiter>
    PYTHONPATH="Python:${AITER_PATH}" \
      python Python/rocke/examples/attention/benchmark_prefill2d_live.py \
        --shapes <path/to/unified_attention_shapes.jsonl> \
        --variants prod combo fallback \
        --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]  # rocKE root
sys.path.insert(0, str(ROOT / "Python"))

DEFAULT_SHAPE_UTILS = ROOT / "dsl_docs/optimization/utilities/tools/stage1_benchmark"


# --------------------------------------------------------------------------
# shape utils + triton
# --------------------------------------------------------------------------
def _load_shape_utils(path: Path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from _ua_shape_utils import (  # type: ignore
        attention_flops,
        dedupe_shapes,
        filter_prefill_2d,
        load_shapes,
        make_inputs,
    )

    return attention_flops, dedupe_shapes, filter_prefill_2d, load_shapes, make_inputs


_UAM = None
_UNIFIED_ATTENTION = None
_ORIG_USE_2D = None


def _import_triton():
    global _UAM, _UNIFIED_ATTENTION, _ORIG_USE_2D
    if _UNIFIED_ATTENTION is not None:
        return
    import aiter.ops.triton.unified_attention as uam  # type: ignore
    from aiter.ops.triton.unified_attention import unified_attention  # type: ignore

    _UAM = uam
    _UNIFIED_ATTENTION = unified_attention
    _ORIG_USE_2D = uam.use_2d_kernel


def _force_triton_2d():
    _import_triton()
    _UAM.use_2d_kernel = lambda *a, **kw: True


def _restore_triton():
    if _UAM is not None and _ORIG_USE_2D is not None:
        _UAM.use_2d_kernel = _ORIG_USE_2D


def _bench_stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


def _gm(vals: list[float]) -> float:
    vals = [v for v in vals if v > 0]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


# --------------------------------------------------------------------------
# CK DSL variant specs
# --------------------------------------------------------------------------
def _variant_flags(name: str, *, sliding_window: int, dtype: str, is_fp8: bool) -> dict:
    """Return the UnifiedAttention2DTiledSpec transposed/opt flags for a variant.

    Variant grammar (``_`` separated tokens layered on a base):
      fallback        : plain R4 (16x16x32, no transpose opts), nw4 mw16
      fallback_nw2    : fallback with num_warps=2
      combo           : full s1mask + half-local-pv + mask-limit + fast-kv-desc
      combo_nw2       : combo with num_warps=2 (BLOCK_M=64)
      combo_t1        : combo with tile_size = 1*block_size (T=32)
      combo_t4        : combo with tile_size = 4*block_size (T=128)
      combo_earlyv    : combo + use_early_v_schedule
      r4_t32          : mfma_32x32 + transposed_qk only
    """
    base = dict(
        num_warps=4,
        block_m_per_warp=32,
        tile_mult=2,  # tile_size = tile_mult * block_size
        use_mfma_32x32=False,
        use_transposed_qk_32x32=False,
        use_transposed_scalar_state=False,
        use_transposed_mask_once=False,
        use_transposed_half_local_pv=False,
        use_mfma32_skip_legacy_qreg=False,
        use_transposed_mask_limit=False,
        use_fast_paged_kv_desc=False,
        use_early_v_schedule=False,
        waves_per_eu=2,
        use_i64_kv_addr=False,
    )
    toks = name.split("_")
    head = toks[0]
    if head == "fallback":
        base.update(num_warps=4, block_m_per_warp=16)
    elif head == "r4" and len(toks) > 1 and toks[1] == "t32":
        base.update(use_mfma_32x32=True, use_transposed_qk_32x32=True)
        toks = [head] + toks[2:]
    elif head == "combo":
        base.update(
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_mask_once=(sliding_window == 0),
            use_transposed_half_local_pv=True,
            use_mfma32_skip_legacy_qreg=True,
            use_transposed_mask_limit=(sliding_window == 0),
            use_fast_paged_kv_desc=True,
            # Match the production dispatcher: ``_select_2d_waves_per_eu``
            # returns 4 for the combo family. The harness used to default the
            # combo to ``waves_per_eu=2`` (the base value below), which builds
            # an occupancy-starved kernel that does NOT match what production
            # ships -- it under-reported the combo by ~25% (0.85x vs the
            # ~1.07x the wpe=4 kernel the dispatcher actually builds gets) on
            # the long-context multi-seq cohort. The ``_we2`` / ``_we3`` /
            # ``_wenone`` modifier tokens still override for sweeps.
            waves_per_eu=4,
        )
    else:
        raise ValueError(f"unknown variant head {head!r}")
    # modifier tokens
    for t in toks[1:]:
        if t == "nw2":
            base["num_warps"] = 2
        elif t == "nw1":
            base["num_warps"] = 1
        elif t == "t1":
            base["tile_mult"] = 1
        elif t == "t4":
            base["tile_mult"] = 4
        elif t == "mw16":
            base["block_m_per_warp"] = 16
        elif t == "nomlim":
            base["use_transposed_mask_limit"] = False
        elif t == "earlyv":
            base["use_early_v_schedule"] = True
        elif t == "i64":
            base["use_i64_kv_addr"] = True
        elif t == "we3":
            base["waves_per_eu"] = 3
        elif t == "we4":
            base["waves_per_eu"] = 4
        elif t == "wenone":
            base["waves_per_eu"] = None
        elif t in ("t32",):
            pass
        else:
            raise ValueError(f"unknown variant modifier {t!r} in {name!r}")
    # mw16 cannot use mfma_32x32 transpose path
    if base["block_m_per_warp"] == 16:
        base.update(
            use_mfma_32x32=False,
            use_transposed_qk_32x32=False,
            use_transposed_scalar_state=False,
            use_transposed_mask_once=False,
            use_transposed_half_local_pv=False,
            use_mfma32_skip_legacy_qreg=False,
            use_transposed_mask_limit=False,
        )
    return base


class CkVariantBench:
    def __init__(self, *, compile_backend: str = "llvm", num_sms: int = 256):
        self.compile_backend = compile_backend
        self.num_sms = num_sms
        self._launchers: dict[tuple, Any] = {}

    def _problem(self, shape, sliding_window: int, is_fp8: bool):
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
            use_fp8=is_fp8,
            num_sms=self.num_sms,
            compile_backend=self.compile_backend,
        )

    def build(self, shape, variant: str, sliding_window: int, is_fp8: bool):
        from rocke import compile_kernel
        from rocke.instances import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
            supports_tiled_2d,
        )
        from rocke.instances.common.attention_unified import _attn_signature
        from rocke.runtime import KernelLauncher

        dtype = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
        problem = self._problem(shape, sliding_window, is_fp8)
        flags = _variant_flags(
            variant, sliding_window=sliding_window, dtype=dtype, is_fp8=is_fp8
        )
        tile_size = flags["tile_mult"] * shape.block_size
        kv_storage_dtype = "fp8e4m3" if is_fp8 else None
        ok, reason = supports_tiled_2d(
            head_size=shape.head_size,
            block_size=shape.block_size,
            dtype=dtype,
            num_queries_per_kv=problem.num_queries_per_kv,
            use_alibi=problem.use_alibi,
            use_qq_bias=False,
            use_fp8=is_fp8,
            q_dtype=problem.q_dtype,
            num_warps=flags["num_warps"],
            kv_storage_dtype=kv_storage_dtype,
            tile_size=tile_size,
        )
        if not ok:
            raise NotImplementedError(f"supports_tiled_2d: {reason}")

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
            num_warps=flags["num_warps"],
            waves_per_eu=flags["waves_per_eu"],
            tile_size=tile_size,
            block_m_per_warp=flags["block_m_per_warp"],
            kv_storage_dtype=kv_storage_dtype,
            use_fp8_mfma_qk=is_fp8,
            use_mfma_32x32=flags["use_mfma_32x32"],
            use_transposed_qk_32x32=flags["use_transposed_qk_32x32"],
            use_transposed_scalar_state=flags["use_transposed_scalar_state"],
            use_transposed_mask_once=flags["use_transposed_mask_once"],
            use_transposed_half_local_pv=flags["use_transposed_half_local_pv"],
            use_mfma32_skip_legacy_qreg=flags["use_mfma32_skip_legacy_qreg"],
            use_transposed_mask_limit=flags["use_transposed_mask_limit"],
            use_fast_paged_kv_desc=flags["use_fast_paged_kv_desc"],
            use_early_v_schedule=flags["use_early_v_schedule"],
            use_i64_kv_addr=flags["use_i64_kv_addr"],
        )
        key = (shape.signature, variant, spec.kernel_name(), self.compile_backend)
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
                    cache_key=("prefill2d_live", key),
                ),
                spec,
                problem,
            )
        return self._launchers[key]

    def run(self, shape, data, variant, sliding_window, is_fp8, *, warmup, iters):
        import torch
        from rocke.instances.common.attention_unified import _attn_values
        from rocke.runtime import LaunchConfig, synchronize_and_release, time_launches

        launcher, spec, problem = self.build(shape, variant, sliding_window, is_fp8)
        hip_stream = _bench_stream_handle()
        out = torch.empty_like(data["query"])
        vals = _attn_values(
            problem=problem,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
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
            k_scale=1.0,
            v_scale=1.0,
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

        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
        return out, ms, spec.kernel_name()


def _run_triton_live(shape, data, sliding_window, is_fp8, *, warmup, iters):
    import torch
    from rocke.runtime import synchronize_and_release, time_launches

    _import_triton()
    out = torch.empty_like(data["query"])
    window_size = (sliding_window - 1, 0) if sliding_window else (-1, -1)
    descale = None
    if is_fp8:
        descale = torch.ones(1, dtype=torch.float32, device=data["query"].device)
    hip_stream = _bench_stream_handle()
    _force_triton_2d()
    try:

        def call_once():
            _UNIFIED_ATTENTION(
                q=data["query"],
                k=data["key_cache"],
                v=data["value_cache"],
                out=out,
                cu_seqlens_q=data["cu_seqlens_q"],
                seqused_k=data["kv_lens"],
                max_seqlen_q=data["max_query_len"],
                max_seqlen_k=data["max_kv_len"],
                softmax_scale=data["scale"],
                causal=True,
                window_size=window_size,
                block_table=data["block_tables"],
                softcap=float(shape.softcap),
                q_descale=None,  # Q is bf16 in these traces; only K/V are fp8
                k_descale=descale,
                v_descale=descale,
                alibi_slopes=data["alibi_slopes"],
                qq_bias=None,
                sinks=data["sinks"],
            )

        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
    finally:
        _restore_triton()
    return out, ms


def _compare(a, b) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).abs().max().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", type=Path, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "all"), default="bf16")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["prod", "combo", "fallback"],
        help="CK DSL variants to sweep: prod combo fallback r4_t32 combo_sw",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="subsample every Nth shape")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    # Production paged-KV caches have hundreds of thousands of blocks, so the
    # KV working set vastly exceeds L2 and attention is HBM-bandwidth-bound.
    # A small cap makes the cache artificially L2-resident, which understates
    # CK DSL (its async-DMA KV loads are more bandwidth-efficient than
    # Triton's, the advantage that only shows once you are HBM-bound). Default
    # to a production-representative cap. Measured: bf16 cohort 0.90x at
    # cap=8192 vs 1.11x at cap=65536.
    ap.add_argument("--cap-blocks", type=int, default=65536)
    ap.add_argument("--num-sms", type=int, default=256)
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--shape-utils-path", type=Path, default=DEFAULT_SHAPE_UTILS)
    ap.add_argument(
        "--output-json", type=Path, default=Path("/tmp/prefill2d_live.json")
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    (
        attention_flops,
        dedupe_shapes,
        filter_prefill_2d,
        load_shapes,
        make_inputs,
    ) = _load_shape_utils(args.shape_utils_path)

    dtype_filter = None if args.dtype == "all" else args.dtype
    shapes = filter_prefill_2d(load_shapes(args.shapes), dtype=dtype_filter)
    shapes = dedupe_shapes(shapes)
    shapes = shapes[:: args.stride]
    if args.limit is not None:
        shapes = shapes[: args.limit]
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"shapes: {len(shapes)}  variants: {args.variants}")

    bench = CkVariantBench(num_sms=args.num_sms)
    results = []
    for i, shape in enumerate(shapes, 1):
        sw = shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0
        is_fp8 = "float8" in shape.k_dtype
        tag = f"[{i}/{len(shapes)}] {shape.signature}"
        try:
            data = make_inputs(shape, seed=args.seed, cap_blocks=args.cap_blocks)
            tri_out, tri_ms = _run_triton_live(
                shape, data, sw, is_fp8, warmup=args.warmup, iters=args.iterations
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}  TRITON FAIL: {exc!r}")
            traceback.print_exc()
            continue

        rec = {
            "signature": shape.signature,
            "sliding_window": sw,
            "is_fp8": is_fp8,
            "num_seqs": shape.num_seqs,
            "total_q": shape.total_q,
            "max_seqlen_k": shape.max_seqlen_k,
            "triton_ms": tri_ms,
            "variants": {},
        }
        best = None
        for v in args.variants:
            try:
                if v in ("prod", "ck3d"):
                    # production dispatch via run_unified_attention_torch
                    ck_out, ck_ms = _run_prod(
                        shape,
                        data,
                        sw,
                        is_fp8,
                        bench,
                        warmup=args.warmup,
                        iters=args.iterations,
                        backend=("3d" if v == "ck3d" else "auto"),
                    )
                    kname = v
                else:
                    ck_out, ck_ms, kname = bench.run(
                        shape,
                        data,
                        v,
                        sw,
                        is_fp8,
                        warmup=args.warmup,
                        iters=args.iterations,
                    )
                err = _compare(ck_out, tri_out)
                ok = err <= args.tol
                spd = tri_ms / ck_ms if ck_ms > 0 else 0.0
                rec["variants"][v] = {
                    "ms": ck_ms,
                    "speedup": spd,
                    "max_abs": err,
                    "ok": ok,
                    "kernel": kname,
                }
                if ok and (best is None or spd > best[1]):
                    best = (v, spd)
            except Exception as exc:  # noqa: BLE001
                rec["variants"][v] = {"error": repr(exc)}
        rec["best_variant"] = best[0] if best else None
        rec["best_speedup"] = best[1] if best else 0.0
        results.append(rec)
        vs = "  ".join(
            f"{v}={rec['variants'][v].get('speedup', 0):.2f}x"
            f"{'' if rec['variants'][v].get('ok') else '!'}"
            for v in args.variants
            if "speedup" in rec["variants"][v]
        )
        print(
            f"{tag} sw={sw} fp8={int(is_fp8)} tri={tri_ms * 1000:.1f}us | {vs} | best={rec['best_variant']}={rec['best_speedup']:.2f}x"
        )

    args.output_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.output_json}  ({len(results)} shapes)")

    # summary
    def bucket(r):
        return (
            "fp8" if r["is_fp8"] else "bf16",
            "sw" if r["sliding_window"] else "nosw",
        )

    buckets: dict[tuple, list] = {}
    for r in results:
        buckets.setdefault(bucket(r), []).append(r)
    print("\n=== geomean best CK DSL speedup over Triton (2d-forced) ===")
    for b in sorted(buckets):
        rs = buckets[b]
        best = [r["best_speedup"] for r in rs if r["best_speedup"] > 0]
        print(
            f"  {b[0]:4s}/{b[1]:4s}  n={len(rs):3d}  best-variant geomean={_gm(best):.3f}x  wins={sum(1 for x in best if x > 1)}/{len(best)}"
        )
    print("\n=== per-variant geomean (correct shapes only) ===")
    for v in args.variants:
        sp = [
            r["variants"][v]["speedup"]
            for r in results
            if v in r["variants"] and r["variants"][v].get("ok")
        ]
        ncorrect = sum(
            1 for r in results if v in r["variants"] and r["variants"][v].get("ok")
        )
        nfail = sum(
            1
            for r in results
            if v in r["variants"] and r["variants"][v].get("ok") is False
        )
        print(
            f"  {v:10s}  geomean={_gm(sp):.3f}x  correct={ncorrect} incorrect={nfail}"
        )
    return 0


def _run_prod(shape, data, sw, is_fp8, bench, *, warmup, iters, backend="auto"):
    """Time the production dispatcher run_unified_attention_torch."""
    import torch
    from rocke.instances import run_unified_attention_torch
    from rocke.runtime import synchronize_and_release, time_launches

    problem = bench._problem(shape, sw, is_fp8)
    out = torch.empty_like(data["query"])
    hip_stream = _bench_stream_handle()

    def call_once():
        run_unified_attention_torch(
            problem=problem,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_seqlens_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=float(shape.softcap),
            sinks=data["sinks"],
            alibi_slopes=data["alibi_slopes"],
            backend=backend,
            stream=hip_stream,
        )

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)
    return out, ms


if __name__ == "__main__":
    sys.exit(main())
