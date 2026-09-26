#!/usr/bin/env python3
"""Sweep every gfx950 decode kernel the dispatcher registry offers.

Per published (model, kv_len) decode shape (seqlen_q=1) this walks
``iter_dispatch_attention_all`` and launches each admitted candidate through
``DispatchResult.bind_torch``. 3D candidates also sweep ``num_cus`` the same
way ``benchmark_decode_live`` does, because that is the decode split-KV axis.

    python -m benchmarks.gfx950.attention.decode.decode_table_sweep --list-only
    rocke-decode-table-sweep --dtype bf16 --output-json results.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from types import SimpleNamespace

from kernels.common.attention_unified import UNIFIED_DTYPES
from dispatch.attention import (
    AttentionRequest,
    attention_dispatch_result,
    iter_dispatch_attention_all,
    iter_registered_attention_combos,
)
from kernels.common.attention_dense_spec import AttentionDenseSpec
from benchmarks.common.attention_flops import attention_flops
from benchmarks.common.attention_combo_sweep import (
    _kernel_name,
    _run_result,
    _unified_tensors,
    init_torch_first,
)

SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_NUM_CUS = [30, 60, 80, 120, 152, 304]

# (label, num_query_heads, num_kv_heads, head_size, kv_lens)
MODELS = [
    ("Llama-3-8B", 32, 8, 128, SEQLENS),
    ("Llama-3-70B", 64, 8, 128, [4096, 8192, 16384]),
    ("Llama-3.1-405B", 128, 8, 128, [4096, 8192]),
    ("Qwen3-235B-A22B", 64, 4, 128, [4096]),
    ("Qwen3-30B-A3B", 32, 4, 128, [4096]),
]

_TOL = 2e-2


def _shape_request(
    *,
    hq: int,
    hkv: int,
    d: int,
    kv_len: int,
    dtype: str,
    algorithm: str,
    kv_block_size: int,
    num_cus: int = 0,
):
    return AttentionRequest(
        batch=1,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=1,
        seqlen_k=kv_len,
        hdim_q=d,
        hdim_v=d,
        arch="gfx950",
        dtype=dtype,
        mask_type=1,
        kv_block_size=kv_block_size,
        algorithm=algorithm,
        num_cus=num_cus,
    )


def _spec_path(spec) -> str:
    return str(getattr(spec, "path", "dense"))


def _iter_shapes(args):
    for label, hq, hkv, d, seqlens in MODELS:
        if args.only_model and args.only_model not in label:
            continue
        for s in seqlens:
            yield label, hq, hkv, d, s


def _run_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
        no_check=args.no_check,
        tolerance=_TOL,
        verbose_errors=True,
    )


def list_combos(args) -> int:
    print(
        f"dtype={args.dtype} algorithm={args.algorithm} "
        f"block={args.kv_block_size} (CPU list-only)"
    )
    for label, hq, hkv, d, s in _iter_shapes(args):
        req = _shape_request(
            hq=hq,
            hkv=hkv,
            d=d,
            kv_len=s,
            dtype=args.dtype,
            algorithm=args.algorithm,
            kv_block_size=args.kv_block_size,
        )
        combos = tuple(
            iter_registered_attention_combos(
                req,
                candidate_prefix=args.candidate_prefix,
                tuning_id_prefix=args.tuning_id_prefix,
                tuning_sample=args.tuning_sample,
                seed=args.seed,
                sweep_level=args.sweep_level,
            )
        )
        print(f"\n{label} Sq=1 Sk={s} Hq={hq} Hkv={hkv} D={d}  n={len(combos)}")
        for candidate, spec in combos:
            extra = ""
            if isinstance(spec, AttentionDenseSpec):
                extra = (
                    f"  bm={getattr(spec, 'block_m', None)} "
                    f"persist={getattr(spec, 'persistent', None)} "
                    f"wdma={getattr(spec, 'wide_lds_dma', None)}"
                )
            print(
                f"  {candidate.name:<48} {candidate.algorithm:<18} "
                f"path={_spec_path(spec):<4} {_kernel_name(spec)}{extra}"
            )
    return 0


def _run_unified_graph(req, result, args) -> dict:
    import torch
    from rocke.runtime import synchronize_and_release, time_launches

    tensors = _unified_tensors(req, args.seed)
    if hasattr(result.spec, "with_num_kv_blocks"):
        runtime_spec = result.spec.with_num_kv_blocks(int(tensors["k"].shape[0]))
        result = attention_dispatch_result(req, result.candidate, runtime_spec)
    stream = torch.cuda.current_stream().cuda_stream
    binding = result.bind_torch(tensors, stream=stream)

    def call():
        binding.launch(stream=stream)

    call()
    torch.cuda.synchronize()
    max_abs = float("nan")
    if not args.no_check:
        from benchmarks.common.attention_combo_sweep import _reference

        ref = _reference(
            tensors["_dense_q"],
            tensors["_dense_k"],
            tensors["_dense_v"],
            causal=bool(req.mask_type),
            sliding_window=int(req.sliding_window),
        )
        out = tensors["out"]
        max_abs = float((out.reshape_as(ref).float() - ref).abs().max().item())

    graph_mode = "internal"
    timed = call
    captured = None
    try:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            call()
        torch.cuda.synchronize()
        graph_mode = "outer"
        timed = captured.replay
    except Exception as exc:  # noqa: BLE001
        print(
            f"  outer CUDAGraph failed ({type(exc).__name__}: {exc}); "
            "timing internal/eager path",
            flush=True,
        )
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        captured = None
        timed = call

    ms = time_launches(timed, warmup=args.warmup, iters=args.iters, stream=stream)
    synchronize_and_release(stream)
    flops = attention_flops(
        req.batch,
        req.nhead_q,
        req.hdim_q,
        req.seqlen_q,
        req.seqlen_k,
        causal=bool(req.mask_type),
        sliding_window=int(req.sliding_window),
    )
    _ = captured
    ok = args.no_check or (max_abs == max_abs and max_abs <= args.tolerance)
    return {
        "status": "ok" if ok else "mismatch",
        "kernel_name": _kernel_name(result.spec),
        "ms": ms,
        "us": ms * 1000.0,
        "tflops": flops / (ms * 1e-3) / 1e12,
        "max_abs": max_abs,
        "path": str(getattr(result.spec, "path", "unified")),
        "num_cus": int(req.num_cus),
        "cuda_graph": graph_mode,
    }


def _store_row(rows: list[dict], rec: dict, args) -> None:
    """Append one row and rewrite the JSON so a crash keeps earlier rows."""
    rows.append(rec)
    path = getattr(args, "output_json", "") or ""
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
        fh.flush()


def sweep(args) -> list[dict]:
    import torch

    rows: list[dict] = []
    run_args = _run_args(args)
    for label, hq, hkv, d, s in _iter_shapes(args):
        base = _shape_request(
            hq=hq,
            hkv=hkv,
            d=d,
            kv_len=s,
            dtype=args.dtype,
            algorithm=args.algorithm,
            kv_block_size=args.kv_block_size,
        )
        try:
            results = tuple(
                iter_dispatch_attention_all(
                    base,
                    candidate_prefix=args.candidate_prefix,
                    tuning_id_prefix=args.tuning_id_prefix,
                    tuning_sample=args.tuning_sample,
                    seed=args.seed,
                    sweep_level=args.sweep_level,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rec = {
                "model": label,
                "seqlen_q": 1,
                "seqlen_k": s,
                "num_query_heads": hq,
                "num_kv_heads": hkv,
                "head_size": d,
                "dtype": args.dtype,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            _store_row(rows, rec, args)
            print(f"ERROR {label} Sk={s} registry: {exc}", flush=True)
            traceback.print_exc()
            continue
        if not results:
            rec = {
                "model": label,
                "seqlen_q": 1,
                "seqlen_k": s,
                "num_query_heads": hq,
                "num_kv_heads": hkv,
                "head_size": d,
                "dtype": args.dtype,
                "status": "unsupported",
                "reason": "no registered attention combo admits this decode shape",
            }
            _store_row(rows, rec, args)
            print(f"SKIP {label} Sk={s}: no registered combo", flush=True)
            continue
        for result in results:
            is_dense = isinstance(result.spec, AttentionDenseSpec)
            cus_list = [0] if is_dense else list(args.num_cus)
            for cus in cus_list:
                req = replace(base, num_cus=int(cus))
                rec = {
                    "model": label,
                    "seqlen_q": 1,
                    "seqlen_k": s,
                    "num_query_heads": hq,
                    "num_kv_heads": hkv,
                    "head_size": d,
                    "dtype": args.dtype,
                    "kv_block_size": args.kv_block_size,
                    "config": result.candidate.name,
                    "candidate": result.candidate.name,
                    "algorithm": result.candidate.algorithm,
                    "spec_id": result.candidate.spec_id,
                    "num_cus": int(cus),
                }
                try:
                    if is_dense or cus == 0:
                        launch = result
                    else:
                        spec = result.candidate.select_spec(
                            replace(
                                req,
                                algorithm=result.candidate.algorithm,
                                spec_id=result.candidate.spec_id,
                                attention_tuning_id=getattr(
                                    result.spec, "tuning_id", "auto"
                                ),
                            )
                        )
                        launch = attention_dispatch_result(req, result.candidate, spec)
                    if is_dense:
                        res = _run_result(req, launch, run_args, 0)
                    else:
                        res = _run_unified_graph(req, launch, run_args)
                    rec.update(**res)
                    print(
                        f"{rec['status'].upper():4} {label} Sk={s} "
                        f"{result.candidate.name} cus={cus}: "
                        f"{rec.get('tflops', float('nan')):.1f} TF  "
                        f"{rec.get('ms', float('nan')):.4f} ms  "
                        f"graph={rec.get('cuda_graph', '-')}  "
                        f"max_abs={rec.get('max_abs', float('nan')):.2e}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    reason = f"{type(exc).__name__}: {exc}"
                    rec.update(status="error", reason=reason)
                    print(
                        f"ERROR {label} Sk={s} {result.candidate.name} cus={cus}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                _store_row(rows, rec, args)
                torch.cuda.empty_cache()
    return rows


def best_table(rows: list[dict]) -> None:
    best: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["model"], r["seqlen_k"])
        if key not in best or r["tflops"] > best[key]["tflops"]:
            best[key] = r

    unsupported = {
        (r["model"], r["seqlen_k"])
        for r in rows
        if r.get("status") == "unsupported" and (r["model"], r["seqlen_k"]) not in best
    }

    models = [m for m in MODELS if any(r.get("model") == m[0] for r in rows)]
    header = f"{'model':<20}" + "".join(f"{s:>10}" for s in SEQLENS)
    print("\n=== best-of-registry decode TFLOP/s (batch 1, Sq=1) ===")
    print(header)
    print("-" * len(header))
    for label, _, _, _, _ in models:
        line = f"{label:<20}"
        for s in SEQLENS:
            r = best.get((label, s))
            if r:
                line += f"{r['tflops']:>10.1f}"
            elif (label, s) in unsupported:
                line += f"{'unsup':>10}"
            else:
                line += f"{'-':>10}"
        print(line)

    print("\n=== winning candidate per shape ===")
    for label, _, _, _, kv_lens in models:
        for s in kv_lens:
            r = best.get((label, s))
            if r:
                print(
                    f"{label:<20} Sk={s:<6} {r.get('candidate', ''):<40} "
                    f"cus={r.get('num_cus', 0):<4} "
                    f"{r['tflops']:>7.1f} TF  {r['ms']:.4f} ms  "
                    f"max_abs={r['max_abs']:.2e}"
                )

    bad = [r for r in rows if r.get("status") == "mismatch"]
    if bad:
        print("\n=== parity mismatches ===")
        for r in bad:
            print(
                f"{r['model']:<20} Sk={r['seqlen_k']:<6} {r.get('candidate', ''):<40} "
                f"max_abs={r['max_abs']:.2e}"
            )


def _rows_exit_code(rows: list[dict]) -> int:
    return 1 if any(r.get("status") not in ("ok", "unsupported") for r in rows) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=UNIFIED_DTYPES)
    ap.add_argument(
        "--algorithm",
        default="auto",
        help="AttentionRequest.algorithm filter; 'auto' enumerates every "
        "executable registry family that admits the decode shape.",
    )
    ap.add_argument("--kv-block-size", type=int, default=16)
    ap.add_argument(
        "--num-cus",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_CUS,
        help="num_cus values for 3D/2D decode (ignored for dense). "
        "0 means auto-resolve to the device CU count.",
    )
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--only-model", default="")
    ap.add_argument("--candidate-prefix", default="")
    ap.add_argument("--tuning-id-prefix", default="")
    ap.add_argument(
        "--sweep-level",
        choices=("production", "full"),
        default="production",
        help="production walks the curated stacks. full samples every kernel knob",
    )
    ap.add_argument(
        "--tuning-sample",
        type=int,
        default=256,
        help="with --sweep-level full: random legal specs per tuning candidate, "
        "seeded by --seed (0 = the full stream). Ignored for production",
    )
    ap.add_argument("--output-json", default="")
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="CPU dump of registered combos per shape; no GPU launch",
    )
    args = ap.parse_args()

    if args.list_only:
        return list_combos(args)

    import torch

    init_torch_first()
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    print(
        f"dtype={args.dtype} algorithm={args.algorithm} "
        f"block={args.kv_block_size} num_cus={args.num_cus} "
        f"check={not args.no_check}"
    )

    rows = sweep(args)
    best_table(rows)

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.output_json}")
    return _rows_exit_code(rows)


if __name__ == "__main__":
    raise SystemExit(main())
