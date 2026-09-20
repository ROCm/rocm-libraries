#!/usr/bin/env python3
"""Sweep every gfx950 attention kernel the dispatcher registry offers.

Per published (model, seqlen) shape this walks ``iter_dispatch_attention_all``
and launches each admitted candidate through ``DispatchResult.bind_torch``.

    python -m benchmarks.gfx950.attention.prefill.dense_prefill_table_sweep --list-only
    rocke-dense-prefill-table-sweep --dtype bf16 --output-json results.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from types import SimpleNamespace

from dispatch.attention import (
    AttentionRequest,
    iter_dispatch_attention_all,
    iter_registered_attention_combos,
)
from kernels.common.attention_dense_spec import AttentionDenseSpec
from benchmarks.common.attention_combo_sweep import (
    _kernel_name,
    _run_result,
    init_torch_first,
)

SEQLENS = [512, 1024, 2048, 4096, 8192]
LONG_ONLY = [2048, 4096, 8192]

# (label, num_query_heads, num_kv_heads, head_size, seqlens)
MODELS = [
    ("Llama-3-8B", 32, 8, 128, SEQLENS),
    ("Llama-3-70B", 64, 8, 128, SEQLENS),
    ("Llama-3.1-405B", 128, 8, 128, SEQLENS),
    ("Qwen2.5-7B", 28, 4, 128, SEQLENS),
    ("Qwen3-14B", 40, 8, 128, SEQLENS),
    ("Qwen3-235B-A22B", 64, 4, 128, SEQLENS),
    ("Llama-2-7B", 32, 32, 128, SEQLENS),
    ("GPT-3-13B", 40, 40, 128, SEQLENS),
    ("d64-serving-GQA8", 64, 8, 64, LONG_ONLY),
    ("Qwen3-Next", 16, 2, 256, LONG_ONLY),
]

_TOL = 2e-2


def _shape_request(
    *, hq: int, hkv: int, d: int, seqlen: int, dtype: str, algorithm: str
):
    return AttentionRequest(
        batch=1,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=seqlen,
        seqlen_k=seqlen,
        hdim_q=d,
        hdim_v=d,
        arch="gfx950",
        dtype=dtype,
        mask_type=1,
        kv_block_size=64,
        algorithm=algorithm,
    )


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
    print(f"dtype={args.dtype} algorithm={args.algorithm} (CPU list-only)")
    for label, hq, hkv, d, s in _iter_shapes(args):
        req = _shape_request(
            hq=hq, hkv=hkv, d=d, seqlen=s, dtype=args.dtype, algorithm=args.algorithm
        )
        combos = tuple(
            iter_registered_attention_combos(
                req,
                candidate_prefix=args.candidate_prefix,
                tuning_id_prefix=args.tuning_id_prefix,
            )
        )
        print(f"\n{label} S={s} Hq={hq} Hkv={hkv} D={d}  n={len(combos)}")
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
                f"{_kernel_name(spec)}{extra}"
            )
    return 0


def sweep(args) -> list[dict]:
    import torch

    rows: list[dict] = []
    run_args = _run_args(args)
    for label, hq, hkv, d, s in _iter_shapes(args):
        req = _shape_request(
            hq=hq, hkv=hkv, d=d, seqlen=s, dtype=args.dtype, algorithm=args.algorithm
        )
        try:
            results = tuple(
                iter_dispatch_attention_all(
                    req,
                    candidate_prefix=args.candidate_prefix,
                    tuning_id_prefix=args.tuning_id_prefix,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rec = {
                "model": label,
                "seqlen": s,
                "num_query_heads": hq,
                "num_kv_heads": hkv,
                "head_size": d,
                "dtype": args.dtype,
                "status": "unsupported",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            rows.append(rec)
            print(f"SKIP {label} S={s} registry: {exc}", flush=True)
            traceback.print_exc()
            continue
        if not results:
            rec = {
                "model": label,
                "seqlen": s,
                "num_query_heads": hq,
                "num_kv_heads": hkv,
                "head_size": d,
                "dtype": args.dtype,
                "status": "unsupported",
                "reason": "no registered attention combo admits this shape",
            }
            rows.append(rec)
            print(f"SKIP {label} S={s}: no registered combo", flush=True)
            continue
        for index, result in enumerate(results):
            rec = {
                "model": label,
                "seqlen": s,
                "num_query_heads": hq,
                "num_kv_heads": hkv,
                "head_size": d,
                "dtype": args.dtype,
                "config": result.candidate.name,
                "candidate": result.candidate.name,
                "algorithm": result.candidate.algorithm,
                "spec_id": result.candidate.spec_id,
            }
            try:
                res = _run_result(req, result, run_args, index)
                ok = res.get("status") == "ok"
                rec.update(status="ok" if ok else res.get("status", "error"), **res)
                print(
                    f"{rec['status'].upper():4} {label} S={s} {result.candidate.name}: "
                    f"{rec.get('tflops', float('nan')):.1f} TF  "
                    f"max_abs={rec.get('max_abs', float('nan')):.2e}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"
                rec.update(status="unsupported", reason=reason)
                print(f"SKIP {label} S={s} {result.candidate.name}: {exc}", flush=True)
                traceback.print_exc()
            rows.append(rec)
            torch.cuda.empty_cache()
    return rows


def best_table(rows: list[dict]) -> None:
    best: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["model"], r["seqlen"])
        if key not in best or r["tflops"] > best[key]["tflops"]:
            best[key] = r

    unsupported = {
        (r["model"], r["seqlen"])
        for r in rows
        if r.get("status") == "unsupported" and (r["model"], r["seqlen"]) not in best
    }

    models = [m for m in MODELS if any(r.get("model") == m[0] for r in rows)]
    header = f"{'model':<20}" + "".join(f"{s:>10}" for s in SEQLENS)
    print("\n=== best-of-registry TFLOP/s (batch 1, causal) ===")
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
    for label, _, _, _, _ in models:
        for s in SEQLENS:
            r = best.get((label, s))
            if r:
                print(
                    f"{label:<20} S={s:<6} {r.get('candidate', r.get('config', '')):<48} "
                    f"{r['tflops']:>7.1f} TF  {r['ms']:.4f} ms  "
                    f"max_abs={r['max_abs']:.2e}"
                )

    bad = [r for r in rows if r.get("status") == "mismatch"]
    if bad:
        print("\n=== parity mismatches ===")
        for r in bad:
            print(
                f"{r['model']:<20} S={r['seqlen']:<6} {r.get('candidate', ''):<48} "
                f"max_abs={r['max_abs']:.2e}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument(
        "--algorithm",
        default="auto",
        help="AttentionRequest.algorithm filter; 'auto' enumerates every "
        "executable registry family that admits the shape.",
    )
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--only-model", default="")
    ap.add_argument("--candidate-prefix", default="")
    ap.add_argument("--tuning-id-prefix", default="")
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
    print(f"dtype={args.dtype} algorithm={args.algorithm} check={not args.no_check}")

    rows = sweep(args)
    best_table(rows)

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
