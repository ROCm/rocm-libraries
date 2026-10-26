#!/usr/bin/env python3
"""Sweep every gfx950 decode kernel the dispatcher registry offers.

Per published (model, kv_len) decode shape (seqlen_q=1) this walks
``iter_dispatch_attention_all`` and launches each admitted candidate through
``DispatchResult.bind_torch``. Candidates whose spec reads ``num_cus`` (the
heuristic split-KV paths) also sweep it the same way ``benchmark_decode_live``
does; tuning specs fix their segment count, so they run once.

    python -m benchmarks.gfx950.attention.decode.decode_table_sweep --list-only
    rocke-decode-table-sweep --dtype bf16 --output-json results.json
    rocke-decode-table-sweep --candidate-prefix attention_gfx950_u \\
        --shard 0/8 --output-json shard0.json        # one GPU's share
    rocke-decode-table-sweep --summarize shard*.json  # best per shape
"""

from __future__ import annotations

import argparse
import glob
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

# (label, num_query_heads, num_kv_heads, head_size, kv_lens, batch)
# batch is the native decode batch each model is published with.
MODELS = [
    ("Llama-3-8B", 32, 8, 128, SEQLENS, 16),
    ("Llama-3-70B", 64, 8, 128, [4096, 8192, 16384], 16),
    ("Llama-3.1-405B", 128, 8, 128, [4096, 8192], 16),
    ("Qwen3-235B-A22B", 64, 4, 128, [4096], 64),
    ("Qwen3-30B-A3B", 32, 4, 128, [4096], 64),
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
    batch: int = 1,
    num_cus: int = 0,
):
    return AttentionRequest(
        batch=batch,
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
    """``(label, hq, hkv, d, kv_len, batch)`` per published shape;
    ``--batch`` overrides the model's native batch when set."""
    for label, hq, hkv, d, seqlens, batch in MODELS:
        if args.only_model and args.only_model not in label:
            continue
        for s in seqlens:
            yield label, hq, hkv, d, s, int(args.batch or batch)


def _parse_shard(text: str):
    index, count = (int(x) for x in str(text).split("/"))
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError(
            f"--shard wants I/N with 0 <= I < N, got {text!r}"
        )
    return index, count


def _reads_num_cus(spec) -> bool:
    """Tuning specs fix their segment count; only the heuristic paths read
    ``num_cus``."""
    return getattr(spec, "path", "") in ("2d", "3d") and not getattr(
        spec, "tuning_id", ""
    )


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
        f"block={args.kv_block_size} batch={args.batch or 'per model'} "
        "(CPU list-only)"
    )
    for label, hq, hkv, d, s, batch in _iter_shapes(args):
        req = _shape_request(
            hq=hq,
            hkv=hkv,
            d=d,
            kv_len=s,
            dtype=args.dtype,
            algorithm=args.algorithm,
            kv_block_size=args.kv_block_size,
            batch=batch,
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
        print(
            f"\n{label} B={batch} Sq=1 Sk={s} Hq={hq} Hkv={hkv} D={d}  "
            f"n={len(combos)}"
        )
        for candidate, spec in combos:
            extra = ""
            kernel_spec = getattr(spec, "kernel_spec", spec)
            if isinstance(kernel_spec, AttentionDenseSpec):
                extra = (
                    f"  bm={kernel_spec.block_m} bn={kernel_spec.block_n} "
                    f"persist={kernel_spec.persistent} "
                    f"wdma={getattr(kernel_spec, 'wide_lds_dma', None)}"
                )
            print(
                f"  {candidate.name:<48} {candidate.algorithm:<18} "
                f"path={_spec_path(spec):<4} {_kernel_name(spec)}{extra}"
            )
    return 0


def _run_unified_graph(req, result, args) -> dict:
    import torch
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.launcher import no_fence

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

    # A launcher call outside no_fence() event-synchronizes, which invalidates
    # a capture; and the launch has to go to the capture stream, not the one
    # the binding was made on.
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured):
            with no_fence():
                binding.launch(stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
    except Exception as exc:
        # An invalidated capture poisons the HIP context for every later
        # launch in this process, so stop rather than record a run of bogus
        # errors.
        raise SystemExit(
            f"outer CUDAGraph capture failed ({type(exc).__name__}: {exc}); "
            "the process cannot launch again"
        ) from exc
    graph_mode = "outer"
    timed = captured.replay

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
    shard_index, shard_count = args.shard
    for label, hq, hkv, d, s, batch in _iter_shapes(args):
        base = _shape_request(
            hq=hq,
            hkv=hkv,
            d=d,
            kv_len=s,
            dtype=args.dtype,
            algorithm=args.algorithm,
            kv_block_size=args.kv_block_size,
            batch=batch,
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
        for index, result in enumerate(results):
            # The stream order does not depend on kv_len, so a shard keeps the
            # same specs on every length of a model and reuses their compiles.
            if index % shard_count != shard_index:
                continue
            is_dense = getattr(result.spec, "path", "") == "dense"
            cus_list = list(args.num_cus) if _reads_num_cus(result.spec) else [0]
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
                    "batch": batch,
                    "kv_block_size": args.kv_block_size,
                    "config": result.candidate.name,
                    "candidate": result.candidate.name,
                    "algorithm": result.candidate.algorithm,
                    "spec_id": result.candidate.spec_id,
                    "path": _spec_path(result.spec),
                    "tuning_id": getattr(result.spec, "tuning_id", ""),
                    "knobs": dict(getattr(result.spec, "knobs", ()) or ()),
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
                                tuning_id=getattr(result.spec, "tuning_id", "")
                                or "auto",
                                tuning_knobs=getattr(result.spec, "knobs", ()),
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
                        f"{rec['tuning_id'] or result.candidate.name} cus={cus}: "
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


def _best_by(rows: list[dict], key) -> dict:
    best: dict = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        k = key(r)
        if k not in best or r["tflops"] > best[k]["tflops"]:
            best[k] = r
    return best


def best_table(rows: list[dict]) -> None:
    best = _best_by(rows, lambda r: (r["model"], r["seqlen_k"]))
    best_path = _best_by(rows, lambda r: (r["model"], r["seqlen_k"], r.get("path", "")))

    unsupported = {
        (r["model"], r["seqlen_k"])
        for r in rows
        if r.get("status") == "unsupported" and (r["model"], r["seqlen_k"]) not in best
    }

    models = [m for m in MODELS if any(r.get("model") == m[0] for r in rows)]
    header = f"{'model':<20}" + "".join(f"{s:>10}" for s in SEQLENS)
    print("\n=== best-of-registry decode TFLOP/s (native batch per model, Sq=1) ===")
    print(header)
    print("-" * len(header))
    for label, *_ in models:
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

    print("\n=== winning config per shape (overall, then best 2D and best 3D) ===")
    for label, _, _, _, kv_lens, _ in models:
        for s in kv_lens:
            r = best.get((label, s))
            if not r:
                continue
            print(
                f"{label:<20} Sk={s:<6} {r.get('path', ''):<3} "
                f"{r.get('tuning_id') or r.get('candidate', ''):<56} "
                f"cus={r.get('num_cus', 0):<4} "
                f"{r['tflops']:>7.1f} TF  {r['us']:>8.2f} us  "
                f"graph={r.get('cuda_graph', '-')}  max_abs={r['max_abs']:.2e}"
            )
            for path in ("2d", "3d"):
                p = best_path.get((label, s, path))
                if p is None:
                    print(f"{'':<27} best {path}: none ran")
                    continue
                print(
                    f"{'':<27} best {path}: "
                    f"{p.get('tuning_id') or p.get('candidate', ''):<56} "
                    f"{p['tflops']:>7.1f} TF  {p['us']:>8.2f} us  "
                    f"knobs={json.dumps(p.get('knobs', {}), sort_keys=True)}"
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
    ap.add_argument(
        "--kv-block-size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=0,
        help="decode batch; 0 uses each model's native batch",
    )
    ap.add_argument(
        "--num-cus",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_CUS,
        help="num_cus values for the heuristic 2D/3D decode paths (tuning and "
        "dense specs do not read it). 0 means auto-resolve to the device CU count.",
    )
    ap.add_argument(
        "--shard",
        type=_parse_shard,
        default=(0, 1),
        metavar="I/N",
        help="run only every N-th spec of each shape's stream, starting at I",
    )
    ap.add_argument(
        "--summarize",
        nargs="+",
        default=None,
        metavar="JSON",
        help="print the best-config tables for saved --output-json files "
        "(globs allowed) and exit; no GPU needed",
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
        help="production walks the curated unified-tuning stacks and sets each "
        "dense knob to every legal value one at a time from the shipped spec. "
        "full samples every knob combination",
    )
    ap.add_argument(
        "--tuning-sample",
        type=int,
        default=256,
        help="with --sweep-level full: random legal specs per tuning or dense "
        "candidate, seeded by --seed (0 = the full stream). Ignored for production",
    )
    ap.add_argument("--output-json", default="")
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="CPU dump of registered combos per shape; no GPU launch",
    )
    args = ap.parse_args()

    if args.summarize:
        rows = []
        for pattern in args.summarize:
            for path in sorted(glob.glob(pattern)) or [pattern]:
                with open(path, encoding="utf-8") as fh:
                    rows.extend(json.load(fh))
        best_table(rows)
        return _rows_exit_code(rows)
    if args.list_only:
        return list_combos(args)

    import torch

    init_torch_first()
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    print(
        f"dtype={args.dtype} algorithm={args.algorithm} "
        f"block={args.kv_block_size} batch={args.batch or 'per model'} "
        f"num_cus={args.num_cus} "
        f"shard={args.shard[0]}/{args.shard[1]} check={not args.no_check}"
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
