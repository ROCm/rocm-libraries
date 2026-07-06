# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Authoritative prefill-2D benchmark: LIVE Triton vs CK DSL variant sweep (gfx942).

gfx942 (CDNA3 / MI300X) sibling of
``examples/gfx950/attention/benchmark_prefill2d_live.py``.

Unlike ``benchmark_prefill2d_traces.py`` (which only times CK DSL and joins a
pre-profiled Triton CSV), this harness:

  * runs AITER's Triton ``unified_attention`` LIVE, forced to the 2D kernel,
    on the same stream + timer as CK DSL (apples-to-apples);
  * sweeps a set of CK DSL 2D kernel variants per shape;
  * checks correctness of every CK DSL variant against the Triton output;
  * reports, per shape and per bucket (sw / no-sw, bf16 / fp16), the best
    correct CK DSL variant and its speedup over Triton.

gfx942 differences vs gfx950:
  * ``arch="gfx942"`` passed to compile / build / supports helpers.
  * ``num_sms`` defaults to 120 (MI300X production dispatch value).
  * The flash-regime combo variant uses ``use_mfma_32x32x8=True`` (the
    ``mfma_f32_32x32x8_f16`` atom available on gfx942; fp16-only) rather than
    the gfx950 ``use_mfma_32x32=True`` (``mfma_f32_32x32x16_{f16,bf16}``).
  * No FP8 KV-cache path (not supported on gfx942).

Run:

    export AITER_PATH=<path/to/aiter>
    PYTHONPATH="Python:${AITER_PATH}" \
      python rocke/library/builders/gfx942/attention/benchmark_prefill2d_live.py \
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

from rocke.assets import shape_utils_dir

DEFAULT_SHAPE_UTILS = shape_utils_dir()

ARCH = "gfx942"


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
# CK DSL variant specs  (gfx942)
# --------------------------------------------------------------------------
def _variant_flags(name: str, *, sliding_window: int, dtype: str, is_fp8: bool) -> dict:
    """Return the UnifiedAttention2DTiledSpec flags for a gfx942 variant.

    Variant grammar (``_`` separated tokens layered on a base):
      fallback        : narrow 16x16x16 path, nw2 mw16 (matches gfx942 narrow)
      fallback_nw4    : narrow path with num_warps=4
      combo           : gfx942 flash-regime (32x32x8 transposed) wide4 path
      combo_nw1       : flash-regime L4 (WG=64, K single-buffer)
      combo_nw2       : flash-regime wide2 (BLOCK_M=64)
      combo_earlyv    : combo + use_early_v_schedule
      combo_t1        : combo with tile_mult=1 (T=block_size)
      combo_t4        : combo with tile_mult=4 (T=4*block_size)

    Notes:
      * ``use_mfma_32x32x8`` is the gfx942 fp16-only 32x32x8 MFMA atom.
        The gfx950 script uses ``use_mfma_32x32`` (the wider 32x32x16 atom).
      * FP8 is not supported on gfx942; ``is_fp8`` is accepted for API
        compatibility but must be False.
    """
    if is_fp8:
        raise ValueError("FP8 is not supported on gfx942")

    base = dict(
        num_warps=4,
        block_m_per_warp=32,
        tile_mult=2,  # tile_size = tile_mult * block_size
        use_mfma_32x32x8=False,
        use_transposed_qk_32x32=False,
        use_transposed_scalar_state=False,
        use_transposed_mask_once=False,
        use_transposed_half_local_pv=False,
        use_mfma32_skip_legacy_qreg=False,
        use_transposed_mask_limit=False,
        use_fast_paged_kv_desc=False,
        use_early_v_schedule=False,
        use_k_single_buffer=False,
        waves_per_eu=2,
        use_i64_kv_addr=False,
    )
    toks = name.split("_")
    head = toks[0]
    if head == "fallback":
        base.update(num_warps=2, block_m_per_warp=16)
    elif head == "combo":
        # gfx942 flash-regime: 32x32x8 transposed-QK (fp16 only).
        if dtype == "bf16":
            raise NotImplementedError(
                "combo variant requires fp16 on gfx942 (use_mfma_32x32x8 is fp16-only)"
            )
        # wide4 by default (num_warps=4, BLOCK_M=128 > T=64 => K double-buffered).
        base.update(
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_mask_once=(sliding_window == 0),
            use_transposed_half_local_pv=True,
            use_mfma32_skip_legacy_qreg=True,
            use_transposed_mask_limit=(sliding_window == 0),
            use_fast_paged_kv_desc=True,
            use_k_single_buffer=False,  # double-buffered for wide4
            waves_per_eu=4,
        )
    else:
        raise ValueError(f"unknown variant head {head!r}")
    # modifier tokens
    for t in toks[1:]:
        if t == "nw1":
            base["num_warps"] = 1
            # L4 path: BLOCK_M = 32 <= T=64 => K single-buffer
            base["use_k_single_buffer"] = True
        elif t == "nw2":
            base["num_warps"] = 2
            # wide2: BLOCK_M=64 <= T=64 => K single-buffer
            base["use_k_single_buffer"] = True
        elif t == "nw4":
            base["num_warps"] = 4
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
        elif t == "ksb":
            base["use_k_single_buffer"] = True
        else:
            raise ValueError(f"unknown variant modifier {t!r} in {name!r}")
    # mw16 cannot use the 32x32 transpose path
    if base["block_m_per_warp"] == 16:
        base.update(
            use_mfma_32x32x8=False,
            use_transposed_qk_32x32=False,
            use_transposed_scalar_state=False,
            use_transposed_mask_once=False,
            use_transposed_half_local_pv=False,
            use_mfma32_skip_legacy_qreg=False,
            use_transposed_mask_limit=False,
        )
    return base


class CkVariantBench:
    def __init__(self, *, compile_backend: str = "llvm", num_sms: int = 120):
        self.compile_backend = compile_backend
        self.num_sms = num_sms
        self._launchers: dict[tuple, Any] = {}

    def _problem(self, shape, sliding_window: int, is_fp8: bool):
        from kernels import UnifiedAttentionProblem

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
        from kernels import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
            supports_tiled_2d,
        )
        from kernels.common.attention_unified import (
            _attn_signature,
            _tiled_2d_impl,
        )
        from rocke.runtime import KernelLauncher

        # Use the gfx942 spec class (has use_mfma_32x32x8, use_k_single_buffer, etc.)
        spec_cls, _, _ = _tiled_2d_impl(ARCH)

        dtype = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
        problem = self._problem(shape, sliding_window, is_fp8)
        flags = _variant_flags(
            variant, sliding_window=sliding_window, dtype=dtype, is_fp8=is_fp8
        )
        tile_size = flags["tile_mult"] * shape.block_size
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
            tile_size=tile_size,
            arch=ARCH,
        )
        if not ok:
            raise NotImplementedError(f"supports_tiled_2d: {reason}")

        spec = spec_cls(
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
            use_mfma_32x32x8=flags["use_mfma_32x32x8"],
            use_transposed_qk_32x32=flags["use_transposed_qk_32x32"],
            use_transposed_scalar_state=flags["use_transposed_scalar_state"],
            use_transposed_mask_once=flags["use_transposed_mask_once"],
            use_transposed_half_local_pv=flags["use_transposed_half_local_pv"],
            use_mfma32_skip_legacy_qreg=flags["use_mfma32_skip_legacy_qreg"],
            use_transposed_mask_limit=flags["use_transposed_mask_limit"],
            use_fast_paged_kv_desc=flags["use_fast_paged_kv_desc"],
            use_early_v_schedule=flags["use_early_v_schedule"],
            use_k_single_buffer=flags["use_k_single_buffer"],
            use_i64_kv_addr=flags["use_i64_kv_addr"],
        )
        key = (shape.signature, variant, spec.kernel_name(), self.compile_backend)
        if key not in self._launchers:
            kernel = build_unified_attention_2d_tiled(spec, arch=ARCH)
            artifact = compile_kernel(kernel, arch=ARCH, capture_ir_text=False)
            self._launchers[key] = (
                KernelLauncher(
                    hsaco=artifact.hsaco,
                    kernel_name=artifact.kernel_name,
                    signature=_attn_signature(
                        dtype, include_bt_stride=True, include_qq_bias_stride=True
                    ),
                    cache_key=("prefill2d_live_gfx942", key),
                ),
                spec,
                problem,
            )
        return self._launchers[key]

    def run(self, shape, data, variant, sliding_window, is_fp8, *, warmup, iters):
        import torch
        from kernels.common.attention_unified import _attn_values
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
                q_descale=None,
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
    ap.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "all"),
        default="bf16",
        help="gfx942 flash-regime combo is fp16-only; bf16 shapes use the narrow fallback",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="CK DSL variants to sweep: prod combo combo_nw1 combo_nw2 fallback. "
        "Defaults to [prod, combo, fallback] for fp16/all, [prod, fallback] for bf16 "
        "(combo is fp16-only on gfx942).",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="subsample every Nth shape")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    # MI300X has 228 CUs; 120 is the production dispatch value used by the
    # gfx942 attention provider (matches parity_unified_attention.py).
    ap.add_argument("--cap-blocks", type=int, default=65536)
    ap.add_argument("--num-sms", type=int, default=120)
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--shape-utils-path", type=Path, default=DEFAULT_SHAPE_UTILS)
    ap.add_argument(
        "--output-json", type=Path, default=Path("/tmp/prefill2d_live_gfx942.json")
    )
    args = ap.parse_args()

    # Resolve default variant list: combo is fp16-only, so omit it for bf16 runs.
    if args.variants is None:
        if args.dtype == "bf16":
            args.variants = ["prod", "fallback"]
        else:
            args.variants = ["prod", "combo", "fallback"]

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
    print(f"arch:   {ARCH}")
    print(f"shapes: {len(shapes)}  variants: {args.variants}")

    bench = CkVariantBench(num_sms=args.num_sms)
    results = []
    for i, shape in enumerate(shapes, 1):
        sw = shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0
        is_fp8 = "float8" in shape.k_dtype
        if is_fp8:
            print(
                f"[{i}/{len(shapes)}] {shape.signature}  SKIP (fp8 not supported on gfx942)"
            )
            continue
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

        dtype_str = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
        rec = {
            "signature": shape.signature,
            "sliding_window": sw,
            "dtype": dtype_str,
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
                    ck_out, ck_ms, kname = _run_prod(
                        shape,
                        data,
                        sw,
                        is_fp8,
                        bench,
                        warmup=args.warmup,
                        iters=args.iterations,
                        backend=("3d" if v == "ck3d" else "auto"),
                    )
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

        def _fmt_variant(v):
            info = rec["variants"].get(v, {})
            if "speedup" in info:
                ok_mark = "" if info.get("ok") else "!"
                return f"{v}={info['speedup']:.2f}x{ok_mark}"
            return f"{v}=ERR"

        vs = "  ".join(_fmt_variant(v) for v in args.variants)
        print(
            f"{tag} sw={sw} tri={tri_ms * 1000:.1f}us | {vs} | best={rec['best_variant']}={rec['best_speedup']:.2f}x"
        )

    args.output_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.output_json}  ({len(results)} shapes)")

    # summary
    def bucket(r):
        sw_str = "sw" if r["sliding_window"] else "nosw"
        return (r["dtype"], sw_str)

    buckets: dict[tuple, list] = {}
    for r in results:
        buckets.setdefault(bucket(r), []).append(r)
    print("\n=== geomean best CK DSL speedup over Triton (2d-forced) ===")
    for b in sorted(buckets):
        rs = buckets[b]
        best = [r["best_speedup"] for r in rs if r["best_speedup"] > 0]
        dtype_label, sw_label = b
        print(
            f"  {dtype_label:4s}  {sw_label:4s}  n={len(rs):3d}  best-variant geomean={_gm(best):.3f}x  wins={sum(1 for x in best if x > 1)}/{len(best)}"
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
        nerr = sum(1 for r in results if "error" in r["variants"].get(v, {}))
        if nerr == len(results) and results:
            # Every shape errored — surface the first error so it can't look like a
            # real sweep.
            first_err = next(
                r["variants"][v]["error"]
                for r in results
                if "error" in r["variants"].get(v, {})
            )
            print(f"  {v:10s}  SKIPPED on all {len(results)} shapes — {first_err}")
        else:
            print(
                f"  {v:10s}  geomean={_gm(sp):.3f}x  correct={ncorrect} incorrect={nfail}"
                + (f"  errored={nerr}" if nerr else "")
            )
    return 0


def _run_prod(shape, data, sw, is_fp8, bench, *, warmup, iters, backend="auto"):
    """Time the production dispatcher run_unified_attention_torch."""
    import torch
    from kernels import (
        run_unified_attention_torch,
        supports_native_unified_attention_tiled,
        supports_native_unified_attention_3d_tiled,
    )
    from kernels.common.attention_unified import _tiled_spec_from_problem
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

    prefer_2d = backend == "auto" and problem.select_path() == "2d"
    if backend == "3d" or (backend == "auto" and not prefer_2d):
        ok_3d, _ = supports_native_unified_attention_3d_tiled(problem)
        if ok_3d:
            instance_name = "3d"
        else:
            instance_name = "scalar"
    elif backend in ("tiled", "auto"):
        ok_t, _ = supports_native_unified_attention_tiled(problem)
        instance_name = (
            _tiled_spec_from_problem(problem).kernel_name() if ok_t else "scalar"
        )
    else:
        instance_name = "scalar"

    return out, ms, instance_name


if __name__ == "__main__":
    sys.exit(main())
