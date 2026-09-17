#!/usr/bin/env python3
"""Sweep every gfx950 attention kernel the dispatcher registry offers.

Per published (model, seqlen) shape this walks ``registered_attention_combos``
and launches each admitted candidate through the kernel entry
(``run_attention_dense_torch`` / ``run_unified_attention_torch``). It does not
call the dense prefill builder.

    python dense_prefill_table_sweep.py --list-only
    python dense_prefill_table_sweep.py --dtype bf16 --output-json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ROCKE_ROOT") or os.path.abspath(
    os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir)
)
for _sub in ("platform/python", "library"):
    _p = os.path.join(ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dispatch.attention import (  # noqa: E402
    AttentionRequest,
    registered_attention_combos,
)
from dispatch.attention.common import _problem  # noqa: E402
from kernels.common.attention_dense_spec import AttentionDenseSpec  # noqa: E402

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


def _causal_flops(
    *, batch: int, hq: int, d: int, sq: int, sk: int, causal: bool
) -> int:
    if causal:
        return 4 * batch * hq * d * (sq * (sq + 1) // 2)
    return 2 * 2 * batch * hq * d * sq * sk


def _kernel_name(spec) -> str:
    name = getattr(spec, "kernel_name", None)
    if callable(name):
        return str(name())
    return getattr(spec, "name", type(spec).__name__)


def _iter_shapes(args):
    for label, hq, hkv, d, seqlens in MODELS:
        if args.only_model and args.only_model not in label:
            continue
        for s in seqlens:
            yield label, hq, hkv, d, s


def list_combos(args) -> int:
    print(f"rocke_root={ROOT}")
    print(f"dtype={args.dtype} algorithm={args.algorithm} (CPU list-only)")
    for label, hq, hkv, d, s in _iter_shapes(args):
        req = _shape_request(
            hq=hq, hkv=hkv, d=d, seqlen=s, dtype=args.dtype, algorithm=args.algorithm
        )
        combos = registered_attention_combos(
            req,
            candidate_prefix=args.candidate_prefix,
            tuning_id_prefix=args.tuning_id_prefix,
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


def _sdpa_err(q, k, v, out, *, causal: bool):
    qh = q.transpose(1, 2).float()
    hq, hkv = q.shape[2], k.shape[2]
    kh = k.transpose(1, 2).repeat_interleave(hq // hkv, 1).float()
    vh = v.transpose(1, 2).repeat_interleave(hq // hkv, 1).float()
    import torch

    ref = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, is_causal=causal
    ).transpose(1, 2)
    return (out.float() - ref).abs().max().item()


def _run_dense(spec, *, warmup: int, iters: int, seed: int, check: bool) -> dict:
    import torch
    from kernels.gfx950.attention_dense import run_attention_dense_torch
    from rocke.runtime import synchronize_and_release, time_launches

    dt = torch.bfloat16 if spec.dtype == "bf16" else torch.float16
    B, Sq, Hq, D = spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size
    Skv, Hkv = spec.seqlen_kv, spec.num_kv_heads
    torch.manual_seed(seed)
    q = torch.randn(B, Sq, Hq, D, dtype=dt, device="cuda")
    k = (torch.randn(B, Skv, Hkv, D, dtype=dt, device="cuda") * 0.2).contiguous()
    v = (torch.randn(B, Skv, Hkv, D, dtype=dt, device="cuda") * 0.2).contiguous()
    out = torch.zeros(B, Sq, Hq, D, dtype=dt, device="cuda")
    scale = 1.0 / math.sqrt(D)
    stream = torch.cuda.current_stream().cuda_stream

    def call():
        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, scale=scale, stream=stream
        )

    call()
    torch.cuda.synchronize()
    err = float("nan")
    if check:
        err = _sdpa_err(q, k, v, out, causal=spec.causal)
    ms = time_launches(call, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    flops = _causal_flops(batch=B, hq=Hq, d=D, sq=Sq, sk=Skv, causal=spec.causal)
    return {
        "kernel_name": spec.kernel_name(),
        "ms": ms,
        "tflops": flops / (ms * 1e-3) / 1e12,
        "max_abs": err,
        "ok": (not check) or (err < _TOL),
        "block_m": spec.block_m,
        "block_n": spec.block_n,
        "persistent": spec.persistent,
        "wide_lds_dma": spec.wide_lds_dma,
        "persist_decode": spec.resolved_persist_decode,
    }


def _pack_paged(k_dense, page: int):
    import torch

    B, S, Hkv, D = k_dense.shape
    n_pages = (S + page - 1) // page
    pad_s = n_pages * page
    padded = torch.zeros(B, pad_s, Hkv, D, dtype=k_dense.dtype, device=k_dense.device)
    padded[:, :S] = k_dense
    cache = padded.reshape(B * n_pages, page, Hkv, D).contiguous()
    table = (
        torch.arange(B * n_pages, device=k_dense.device, dtype=torch.int32)
        .view(B, n_pages)
        .contiguous()
    )
    return cache, table


def _run_unified(req, spec, *, warmup: int, iters: int, seed: int, check: bool) -> dict:
    import torch
    from kernels import run_unified_attention_torch
    from rocke.runtime import synchronize_and_release, time_launches

    problem = _problem(req)
    dt = torch.bfloat16 if req.dtype.lower() == "bf16" else torch.float16
    B, S, Hq, Hkv, D = (
        int(req.batch),
        int(req.seqlen_q),
        int(req.nhead_q),
        int(req.nhead_k),
        int(req.hdim_q),
    )
    page = int(req.kv_block_size)
    torch.manual_seed(seed)
    q_dense = torch.randn(B, S, Hq, D, dtype=dt, device="cuda")
    k_dense = (torch.randn(B, S, Hkv, D, dtype=dt, device="cuda") * 0.2).contiguous()
    v_dense = (torch.randn(B, S, Hkv, D, dtype=dt, device="cuda") * 0.2).contiguous()
    q = q_dense.reshape(B * S, Hq, D).contiguous()
    out = torch.zeros_like(q)
    k_cache, block_table = _pack_paged(k_dense, page)
    v_cache, _ = _pack_paged(v_dense, page)
    cu_seqlens_q = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device="cuda")
    seqused_k = torch.full((B,), S, dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(D)
    stream = torch.cuda.current_stream().cuda_stream
    path = getattr(spec, "path", "2d")
    backend = "tiled" if path == "2d" else path

    def call():
        run_unified_attention_torch(
            problem=problem,
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            softmax_scale=scale,
            block_table=block_table,
            softcap=0.0,
            backend=backend,
            stream=stream,
            tuning_spec=spec if hasattr(spec, "kernel_spec") else None,
        )

    call()
    torch.cuda.synchronize()
    err = float("nan")
    if check:
        out_dense = out.reshape(B, S, Hq, D)
        err = _sdpa_err(q_dense, k_dense, v_dense, out_dense, causal=True)
    ms = time_launches(call, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    flops = _causal_flops(batch=B, hq=Hq, d=D, sq=S, sk=S, causal=True)
    return {
        "kernel_name": _kernel_name(spec),
        "ms": ms,
        "tflops": flops / (ms * 1e-3) / 1e12,
        "max_abs": err,
        "ok": (not check) or (err < _TOL),
        "path": path,
        "backend": backend,
    }


def sweep(args) -> list[dict]:
    import torch

    rows: list[dict] = []
    for label, hq, hkv, d, s in _iter_shapes(args):
        req = _shape_request(
            hq=hq, hkv=hkv, d=d, seqlen=s, dtype=args.dtype, algorithm=args.algorithm
        )
        try:
            combos = registered_attention_combos(
                req,
                candidate_prefix=args.candidate_prefix,
                tuning_id_prefix=args.tuning_id_prefix,
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
        if not combos:
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
        for candidate, spec in combos:
            rec = {
                "model": label,
                "seqlen": s,
                "num_query_heads": hq,
                "num_kv_heads": hkv,
                "head_size": d,
                "dtype": args.dtype,
                "config": candidate.name,
                "candidate": candidate.name,
                "algorithm": candidate.algorithm,
                "spec_id": candidate.spec_id,
            }
            try:
                if isinstance(spec, AttentionDenseSpec):
                    res = _run_dense(
                        spec,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed,
                        check=not args.no_check,
                    )
                else:
                    res = _run_unified(
                        req,
                        spec,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed,
                        check=not args.no_check,
                    )
                rec.update(status="ok" if res["ok"] else "mismatch", **res)
                print(
                    f"{rec['status'].upper():4} {label} S={s} {candidate.name}: "
                    f"{rec.get('tflops', float('nan')):.1f} TF  "
                    f"max_abs={rec.get('max_abs', float('nan')):.2e}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"
                rec.update(status="unsupported", reason=reason)
                print(f"SKIP {label} S={s} {candidate.name}: {exc}", flush=True)
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
        "registry family that admits the shape (dense + unified + d256).",
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

    print(f"rocke_root={ROOT}")
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
