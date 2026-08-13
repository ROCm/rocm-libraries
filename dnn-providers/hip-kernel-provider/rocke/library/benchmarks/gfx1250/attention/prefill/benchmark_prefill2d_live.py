# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prefill benchmark: LIVE Triton vs DSL dispatcher vs FlyDSL (gfx1250).

For each shape the benchmark runs up to three paths:

* **Triton** — AITER ``unified_attention`` forced to the 2D kernel (baseline).
* **DSL** — the platform dispatcher (:func:`~dispatch.attention.dispatch_attention`)
  selects the registered kernel candidate for gfx1250, then exercises it via
  ``run_unified_attention_torch``.  This is the production provider path.
* **FlyDSL** — the fixed ``flash_attn_varlen_d192_gfx1250`` kernel from aiter
  (dense KV, head_dim_qk=192/head_dim_v=128 only).

FlyDSL constraints on gfx1250:
  * bf16 only (Q/K/V/O)
  * head_dim_qk = 192, head_dim_v = 128  (fixed in the kernel)
  * dense (non-paged) KV: cu_seqlens_k derived from kv_lens
  * no sliding window, no alibi, no softcap, no sinks, no FP8

Shapes that don't satisfy these constraints are skipped for the FlyDSL
measurement and marked ``fly=SKIP`` in the output.  All shapes still run
Triton and DSL so you get full coverage across the suite.

Run:

    export AITER_PATH=<path/to/aiter>
    PYTHONPATH="python:${AITER_PATH}" \
      python rocke/library/benchmarks/gfx1250/attention/prefill/benchmark_prefill2d_live.py \
        --shapes <path/to/unified_attention_shapes.jsonl> \
        --limit 20

Pass ``--skip-triton`` to measure DSL/FlyDSL latency only (no Triton reference,
no correctness check).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

from rocke.assets import shape_utils_dir

DEFAULT_SHAPE_UTILS = shape_utils_dir()

ARCH = "gfx1250"
HEAD_DIM_QK = 192
HEAD_DIM_V = 128


# --------------------------------------------------------------------------
# shape utils + triton
# --------------------------------------------------------------------------
def _load_shape_utils(path: Path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from _ua_shape_utils import (  # type: ignore
        dedupe_shapes,
        filter_prefill_2d,
        load_shapes,
        make_inputs,
    )

    return dedupe_shapes, filter_prefill_2d, load_shapes, make_inputs


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
# FlyDSL support check
# --------------------------------------------------------------------------
def _flydsl_supported(shape) -> tuple[bool, str]:
    """Return (supported, reason) for whether FlyDSL can run this shape."""
    if shape.q_dtype != "torch.bfloat16":
        return False, f"dtype={shape.q_dtype} (need bf16)"
    if shape.head_size != HEAD_DIM_QK:
        return False, f"head_size={shape.head_size} (need {HEAD_DIM_QK})"
    if "float8" in shape.k_dtype:
        return False, "fp8 KV not supported"
    if shape.softcap > 0:
        return False, f"softcap={shape.softcap} not supported"
    if shape.has_alibi:
        return False, "alibi not supported"
    if shape.has_sinks:
        return False, "sinks not supported"
    if shape.window_size[0] >= 0:
        return False, f"sliding_window not supported"
    return True, ""


# --------------------------------------------------------------------------
# FlyDSL runner
# --------------------------------------------------------------------------
_FLYDSL_FN = None


def _import_flydsl():
    global _FLYDSL_FN
    if _FLYDSL_FN is not None:
        return
    from aiter.ops.flydsl.kernels.fmha_gfx1250.fmha_kernel import (  # type: ignore
        flash_attn_varlen_d192_gfx1250,
    )

    _FLYDSL_FN = flash_attn_varlen_d192_gfx1250


def _run_flydsl_live(shape, data, *, warmup, iters):
    """Time FlyDSL flash_attn_varlen_d192_gfx1250 (dense KV, THD layout)."""
    import torch
    from rocke.runtime import synchronize_and_release, time_launches

    _import_flydsl()
    hip_stream = _bench_stream_handle()

    # FlyDSL takes dense (non-paged) KV in THD layout.
    # Reconstruct from the paged cache using kv_lens — this is a one-time
    # setup cost outside the timed loop, acceptable for benchmarking.
    q = data["query"]  # [total_q, nheads_q, head_dim]
    k_cache = data["key_cache"]  # [num_blocks, block_size, nheads_k, head_dim]
    v_cache = data["value_cache"]
    block_tables = data["block_tables"]  # [num_seqs, max_blocks_per_seq]
    kv_lens = data["kv_lens"]  # [num_seqs]  int32
    cu_seqlens_q = data["cu_seqlens_q"]  # [num_seqs+1] int32

    num_seqs = shape.num_seqs
    block_size = shape.block_size
    nheads_k = shape.num_kv_heads
    head_dim = shape.head_size

    # Build dense K/V tensors [total_k_tokens, nheads_k, head_dim]
    kv_lens_cpu = kv_lens.cpu().tolist()
    total_k = sum(kv_lens_cpu)
    k_dense = torch.empty(total_k, nheads_k, head_dim, dtype=q.dtype, device=q.device)
    v_dense = torch.empty(total_k, nheads_k, head_dim, dtype=q.dtype, device=q.device)

    offset = 0
    for seq_idx in range(num_seqs):
        seq_kv_len = kv_lens_cpu[seq_idx]
        blocks_needed = (seq_kv_len + block_size - 1) // block_size
        tokens_copied = 0
        for blk in range(blocks_needed):
            phys = int(block_tables[seq_idx, blk].item())
            n = min(block_size, seq_kv_len - tokens_copied)
            k_dense[offset : offset + n] = k_cache[phys, :n]
            v_dense[offset : offset + n] = v_cache[phys, :n]
            offset += n
            tokens_copied += n

    cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_k[1:] = torch.tensor(
        kv_lens_cpu, dtype=torch.int32, device=q.device
    ).cumsum(0)

    out = torch.empty(
        q.shape[0], q.shape[1], HEAD_DIM_V, dtype=q.dtype, device=q.device
    )

    def call_once():
        _FLYDSL_FN(
            q,
            k_dense,
            v_dense,
            cu_seqlens_q,
            cu_seqlens_k,
            int(data["max_query_len"]),
            int(data["max_kv_len"]),
            softmax_scale=float(data["scale"]),
            causal=True,
            out=out,
            return_lse=False,
        )

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release()
    return out, ms


# --------------------------------------------------------------------------
# Triton runner
# --------------------------------------------------------------------------
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
        synchronize_and_release()
    finally:
        _restore_triton()
    return out, ms


def _compare(a, b) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).abs().max().item())


# --------------------------------------------------------------------------
# Platform dispatcher runner
# --------------------------------------------------------------------------
def _run_dsl_live(shape, data, num_cus: int, *, warmup: int, iters: int):
    """Time the platform dispatcher path via dispatch_attention (gfx1250).

    Builds an :class:`~dispatch.attention.AttentionRequest` and calls
    :func:`~dispatch.attention.dispatch_attention` to select the registered
    kernel candidate for this shape, then runs via ``run_unified_attention_torch``
    — the same code path as the production provider.

    Returns (out, ms, kernel_name) or raises on failure.
    """
    import torch
    from dispatch.attention import AttentionRequest, dispatch_attention
    from kernels import UnifiedAttentionProblem, run_unified_attention_torch
    from rocke.runtime import synchronize_and_release, time_launches

    dtype_str = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
    req = AttentionRequest(
        batch=shape.num_seqs,
        nhead_q=shape.num_query_heads,
        nhead_k=shape.num_kv_heads,
        seqlen_q=shape.max_seqlen_q,
        seqlen_k=shape.max_seqlen_k,
        hdim_q=shape.head_size,
        hdim_v=shape.head_size,
        arch=ARCH,
        dtype=dtype_str,
        sliding_window=shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0,
        kv_block_size=shape.block_size,
        num_cus=num_cus,
    )

    result = dispatch_attention(req)
    dispatched_path = result.spec.path  # "2d" or "3d"
    kernel_name = result.spec.kernel_name()
    run_backend = "tiled" if dispatched_path == "2d" else dispatched_path

    sw = shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0
    is_fp8 = "float8" in shape.k_dtype
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
        sliding_window=sw,
        softcap=float(shape.softcap),
        use_sinks=shape.has_sinks,
        use_alibi=shape.has_alibi,
        use_qq_bias=False,
        use_fp8=is_fp8,
        num_cus=num_cus,
    )

    hip_stream = _bench_stream_handle()
    out = torch.empty_like(data["query"])

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
            backend=run_backend,
            stream=hip_stream,
        )

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)
    return out, ms, kernel_name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", type=Path, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "all"), default="bf16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-blocks", type=int, default=65536)
    ap.add_argument("--num-cus", type=int, default=256)
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument(
        "--skip-triton",
        action="store_true",
        help="Skip Triton reference; measure DSL/FlyDSL latency only (no correctness check)",
    )
    ap.add_argument("--shape-utils-path", type=Path, default=DEFAULT_SHAPE_UTILS)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("/tmp/prefill2d_live_gfx1250_flydsl.json"),
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    dedupe_shapes, filter_prefill_2d, load_shapes, make_inputs = _load_shape_utils(
        args.shape_utils_path
    )

    dtype_filter = None if args.dtype == "all" else args.dtype
    shapes = filter_prefill_2d(load_shapes(args.shapes), dtype=dtype_filter)
    shapes = dedupe_shapes(shapes)
    shapes = shapes[:: args.stride]
    if args.limit is not None:
        shapes = shapes[: args.limit]

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"arch:   {ARCH}")
    print(f"shapes: {len(shapes)}")

    results = []
    n_fly_supported = 0
    n_fly_correct = 0

    for i, shape in enumerate(shapes, 1):
        sw = shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0
        is_fp8 = "float8" in shape.k_dtype
        tag = f"[{i}/{len(shapes)}] {shape.signature}"

        fly_ok_for_shape, fly_skip_reason = _flydsl_supported(shape)

        # ------------------------------------------------------------------
        # Triton reference
        # ------------------------------------------------------------------
        tri_out = None
        tri_ms = None
        if not args.skip_triton:
            try:
                data = make_inputs(shape, seed=args.seed, cap_blocks=args.cap_blocks)
                tri_out, tri_ms = _run_triton_live(
                    shape, data, sw, is_fp8, warmup=args.warmup, iters=args.iterations
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{tag}  TRITON FAIL: {exc!r}")
                traceback.print_exc()
                continue
        else:
            try:
                data = make_inputs(shape, seed=args.seed, cap_blocks=args.cap_blocks)
            except Exception as exc:  # noqa: BLE001
                print(f"{tag}  INPUTS FAIL: {exc!r}")
                continue

        # ------------------------------------------------------------------
        # Platform dispatcher (DSL) measurement
        # ------------------------------------------------------------------
        dsl_ms = None
        dsl_err = None
        dsl_ok = None
        dsl_kernel = None
        dsl_status = "n/a"

        try:
            dsl_out, dsl_ms, dsl_kernel = _run_dsl_live(
                shape, data, args.num_cus, warmup=args.warmup, iters=args.iterations
            )
            if tri_out is not None:
                dsl_err = _compare(dsl_out, tri_out)
                dsl_ok = dsl_err <= args.tol
            dsl_status = f"{dsl_ms * 1000:.1f}us"
            if dsl_ok is not None:
                dsl_status += f"({'ok' if dsl_ok else 'WRONG'})"
        except Exception as exc:  # noqa: BLE001
            dsl_status = f"ERR({exc!r})"

        # ------------------------------------------------------------------
        # FlyDSL measurement
        # ------------------------------------------------------------------
        fly_ms = None
        fly_err = None
        fly_ok = None

        if not fly_ok_for_shape:
            fly_status = f"SKIP({fly_skip_reason})"
        else:
            n_fly_supported += 1
            try:
                fly_out, fly_ms = _run_flydsl_live(
                    shape, data, warmup=args.warmup, iters=args.iterations
                )
                if tri_out is not None:
                    # FlyDSL output has head_dim_v=128; compare the overlap only.
                    fly_err = _compare(fly_out, tri_out[..., :HEAD_DIM_V])
                    fly_ok = fly_err <= args.tol
                    if fly_ok:
                        n_fly_correct += 1
                    fly_status = f"{fly_ms * 1000:.1f}us({'ok' if fly_ok else 'WRONG'})"
                else:
                    fly_status = f"{fly_ms * 1000:.1f}us"
            except Exception as exc:  # noqa: BLE001
                fly_status = f"ERR({exc!r})"

        rec = {
            "signature": shape.signature,
            "sliding_window": sw,
            "is_fp8": is_fp8,
            "num_seqs": shape.num_seqs,
            "total_q": shape.total_q,
            "max_seqlen_k": shape.max_seqlen_k,
            "triton_ms": tri_ms,
            "dsl_ms": dsl_ms,
            "dsl_ok": dsl_ok,
            "dsl_max_abs": dsl_err,
            "dsl_kernel": dsl_kernel,
            "flydsl_ms": fly_ms,
            "flydsl_ok": fly_ok,
            "flydsl_max_abs": fly_err,
            "flydsl_skip_reason": fly_skip_reason if not fly_ok_for_shape else None,
        }
        if tri_ms is not None and dsl_ms is not None:
            rec["speedup_dsl_vs_triton"] = tri_ms / dsl_ms if dsl_ms > 0 else 0.0
        if tri_ms is not None and fly_ms is not None:
            rec["speedup_flydsl_vs_triton"] = tri_ms / fly_ms if fly_ms > 0 else 0.0
        results.append(rec)

        tri_str = f"tri={tri_ms * 1000:.1f}us" if tri_ms is not None else "tri=SKIP"
        dsl_spd_str = ""
        if tri_ms is not None and dsl_ms is not None and dsl_ms > 0:
            dsl_spd_str = f"({tri_ms / dsl_ms:.2f}x)"
        fly_spd_str = ""
        if tri_ms is not None and fly_ms is not None and fly_ms > 0:
            fly_spd_str = f"({tri_ms / fly_ms:.2f}x)"
        print(
            f"{tag} sw={sw} fp8={int(is_fp8)} {tri_str}"
            f" | dsl={dsl_status}{dsl_spd_str}"
            f" | fly={fly_status}{fly_spd_str}"
        )

    args.output_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.output_json}  ({len(results)} shapes)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n=== DSL dispatcher summary ===")
    dsl_results = [r for r in results if r["dsl_ms"] is not None]
    dsl_spds = [
        r["speedup_dsl_vs_triton"] for r in dsl_results if "speedup_dsl_vs_triton" in r
    ]
    print(f"  shapes ran:         {len(dsl_results)}/{len(results)}")
    if dsl_spds:
        wins = sum(1 for x in dsl_spds if x > 1)
        print(
            f"  speedup vs Triton:  geomean={_gm(dsl_spds):.3f}x  wins={wins}/{len(dsl_spds)}"
        )
    if not args.skip_triton and dsl_results:
        n_dsl_correct = sum(1 for r in dsl_results if r.get("dsl_ok"))
        print(
            f"  correctness:        {n_dsl_correct}/{len(dsl_results)} correct (tol={args.tol})"
        )

    print(f"\n=== FlyDSL summary ===")
    fly_results = [r for r in results if r["flydsl_ms"] is not None]
    if fly_results:
        spds = [
            r["speedup_flydsl_vs_triton"]
            for r in fly_results
            if "speedup_flydsl_vs_triton" in r
        ]
        print(f"  shapes total:       {len(results)}")
        print(f"  FlyDSL supported:   {n_fly_supported}")
        print(f"  FlyDSL ran:         {len(fly_results)}")
        if spds:
            wins = sum(1 for x in spds if x > 1)
            print(
                f"  speedup vs Triton:  geomean={_gm(spds):.3f}x  wins={wins}/{len(spds)}"
            )
        if not args.skip_triton and n_fly_supported:
            print(
                f"  correctness:        {n_fly_correct}/{n_fly_supported} correct (tol={args.tol})"
            )
    else:
        print("\nNo FlyDSL results (all shapes skipped or errored).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
