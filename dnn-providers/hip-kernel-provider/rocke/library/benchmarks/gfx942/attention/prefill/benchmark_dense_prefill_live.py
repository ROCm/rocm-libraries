# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LIVE benchmark for the dense flash-attention prefill kernel on gfx942 (CDNA3).

Mirrors the gfx950 sibling
(``benchmarks/gfx950/attention/prefill/benchmark_dense_prefill_live.py``): times
the gfx942 dense prefill kernel (``kernels/gfx942/attention_dense.py``:
:class:`AttentionDenseSpec` / :func:`build_attention_dense`) against a torch SDPA
reference on the *same* HIP stream (``rocke.runtime.time_launches`` — HIP events,
not torch events), and reports per shape: windowed-causal TFLOPS, max-abs error vs
SDPA, a PASS/FAIL flag (< 2e-2), and the built kernel name. Emits a JSON (and
optional CSV) report plus a per-mode geomean.

This is the gfx942 NUMERIC + PERF gate (same role the bench plays on gfx950: numeric
correctness lives here, not in a CI pytest — see the gfx950 precedent). It doubles as
the perf harness for the P1-P4 optimization phases.

**P0 scope** (AICK-1664): default grid, dense self-attention (uniform batch via the
``[B, S, H, d]`` grid), causal + full, bf16/fp16, D64/D128, MHA + GQA (incl.
non-pow-2). Sliding-window / varlen / persistent are follow-ups (P1/P4) — their modes
are accepted but skipped with a note until the kernel supports them.

Run as a library module::

    PYTHONPATH=rocke/library python3 -m \\
        benchmarks.gfx942.attention.prefill.benchmark_dense_prefill_live \\
        --mode all --dtype fp16 --output-json /tmp/dense_prefill_live_gfx942.json

or directly with a ROCm-torch venv python::

    ~/.venv/bin/python \\
        rocke/library/benchmarks/gfx942/attention/prefill/benchmark_dense_prefill_live.py \\
        --mode causal --iterations 5 --warmup 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from kernels.gfx942.attention_dense import (  # noqa: E402
    AttentionDenseSpec,
    attention_dense_block,
    attention_dense_grid,
    attention_dense_signature,
    build_attention_dense,
    gfx942_kernel_name,
    supports_attention_dense,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)

_ARCH = "gfx942"
_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}
_TOL = 2e-2
# CLI modes with no P0 spec to construct. This is a UX shortcut only -- the
# authoritative rejection is supports_attention_dense (ValueError from build).
_DEFERRED_MODES = {"swa": "P1+", "varlen": "P1+", "persistent": "P4"}


# --------------------------------------------------------------------------- #
# FLOPs
# --------------------------------------------------------------------------- #
def _pairs(s: int, W: int) -> int:
    if W and W > 0:
        return W * s - W * (W - 1) // 2 if s >= W else s * (s + 1) // 2
    return s * (s + 1) // 2


def _flops(B: int, S: int, causal: bool, W: int, Hq: int, D: int) -> int:
    if causal:
        return B * 4 * Hq * D * _pairs(S, W)
    return B * 2 * 2 * Hq * D * S * S  # full: 2 GEMMs, 2 flop/MAC


def _gm(vals) -> float:
    vals = [v for v in vals if v > 0]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


def _bench_stream_handle() -> int:
    return int(torch.cuda.current_stream().cuda_stream)


# --------------------------------------------------------------------------- #
# launcher (compile + ABI signature)
# --------------------------------------------------------------------------- #
_LAUNCHER_CACHE: dict = {}


def _dense_launcher(spec: AttentionDenseSpec) -> KernelLauncher:
    key = gfx942_kernel_name(
        spec
    )  # batch-unique (kernel bakes batch into buffer extents)
    lch = _LAUNCHER_CACHE.get(key)
    if lch is not None:
        return lch
    ok, why = supports_attention_dense(spec, arch=_ARCH)
    if not ok:
        raise ValueError(f"unsupported spec: {why}")
    art = compile_kernel(
        build_attention_dense(spec, arch=_ARCH),
        arch=_ARCH,
        backend="python",
        capture_ir_text=False,
    )
    lch = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=attention_dense_signature(spec),
    )
    _LAUNCHER_CACHE[key] = lch
    return lch


# --------------------------------------------------------------------------- #
# one benchmark point: build inputs, check parity vs SDPA, time it
# --------------------------------------------------------------------------- #
def bench_dense(
    S: int,
    B: int,
    Hq: int,
    Hkv: int,
    D: int,
    *,
    causal: bool,
    dtype: str,
    block_n: int,
    waves_per_eu: int,
    warmup: int,
    iters: int,
    seed: int,
):
    """Returns (dense_ms, tflops, max_abs, kernel_name). Uniform dense [B,S,H,D]."""
    dev = "cuda"
    dt = _TORCH_DT[dtype]
    scale = 1.0 / math.sqrt(D)
    stream = _bench_stream_handle()
    torch.manual_seed(seed)

    q = (torch.randn(B, S, Hq, D, dtype=dt, device=dev) * 0.2).contiguous()
    k = (torch.randn(B, S, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
    v = (torch.randn(B, S, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
    out = torch.zeros(B, S, Hq, D, dtype=dt, device=dev)

    spec = AttentionDenseSpec(
        batch=B,
        seqlen_q=S,
        seqlen_kv=S,
        num_query_heads=Hq,
        num_kv_heads=Hkv,
        head_size=D,
        causal=causal,
        dtype=dtype,
        block_n=block_n,
        waves_per_eu=waves_per_eu,
    )
    lch = _dense_launcher(spec)
    cfg = LaunchConfig(
        grid=attention_dense_grid(spec),
        block=attention_dense_block(spec),
        stream=stream,
    )
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": scale}

    def call():
        lch(vals, config=cfg)

    call()
    torch.cuda.synchronize()

    # correctness vs SDPA (batched, causal/full, GQA repeat).
    rep = Hq // Hkv
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).repeat_interleave(rep, 1).float()
    vh = v.transpose(1, 2).repeat_interleave(rep, 1).float()
    ref = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, is_causal=causal
    ).transpose(1, 2)
    max_err = (out.float() - ref).abs().max().item()

    ms = time_launches(call, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    tf = _flops(B, S, causal, 0, Hq, D) / (ms * 1e-3) / 1e12
    # gfx942_kernel_name, not spec.kernel_name(): the latter omits batch and
    # waves_per_eu, so a B=4 row would report the B=1 symbol -- the exact confusion
    # behind the cache-collision bug this field exists to make visible.
    return ms, tf, max_err, gfx942_kernel_name(spec)


# --------------------------------------------------------------------------- #
# shape sweeps (P0-supported cohort)
# --------------------------------------------------------------------------- #
def _configs(mode: str, Hq: int, Hkv: int, D: int):
    """Yield (mode, variant, label, S, B, Hq, Hkv, causal) configs."""
    cfgs = []
    if mode in ("causal", "all"):
        for S in (2048, 4096, 8192):
            cfgs.append(("causal", "gqa_causal", f"S={S}", S, 1, Hq, Hkv, True))
        cfgs.append(("causal", "gqa_causal_b4", "S=2048 B=4", 2048, 4, Hq, Hkv, True))
    if mode in ("mha", "all"):
        for H in (16, 32):
            for S in (2048, 4096):
                cfgs.append(("mha", "mha", f"H={H} S={S}", S, 1, H, H, True))
    if mode in ("gqa", "all"):
        # non-power-of-2 GQA groups (common serving shapes).
        for hq, hkv in ((40, 8), (28, 4)):
            cfgs.append(
                ("gqa", "gqa_nonpow2", f"{hq}/{hkv} S=2048", 2048, 1, hq, hkv, True)
            )
    if mode in ("full", "all"):
        for S in (2048, 4096):
            cfgs.append(("full", "non_causal", f"S={S}", S, 1, Hq, Hkv, False))
    return cfgs


def _record(mode, variant, label, S, B, Hq, Hkv, D, causal, res, err_note=None):
    base = {
        "label": label,
        "mode": mode,
        "variant": variant,
        "seqlen": S,
        "batch": B,
        "Hq": Hq,
        "Hkv": Hkv,
        "D": D,
        "causal": causal,
    }
    if res is None:
        return {
            **base,
            "dense_ms": None,
            "tflops": None,
            "max_abs": None,
            "ok": False,
            "kernel_name": None,
            "error": err_note,
        }
    ms, tf, err, kname = res
    return {
        **base,
        "dense_ms": ms,
        "tflops": tf,
        "max_abs": err,
        "ok": bool(err < _TOL),
        "kernel_name": kname,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--mode",
        choices=["causal", "mha", "gqa", "full", "swa", "varlen", "persistent", "all"],
        default="all",
    )
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--hq", type=int, default=128, help="query heads (causal/gqa)")
    ap.add_argument("--hkv", type=int, default=8, help="kv heads (causal/gqa)")
    ap.add_argument("--d", type=int, default=128, help="head size (64 or 128)")
    ap.add_argument("--bn", type=int, default=64, help="block_n (KV tile)")
    ap.add_argument("--waves-per-eu", type=int, default=2, help="occupancy hint")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output-json", type=str, default="/tmp/dense_prefill_live_gfx942.json"
    )
    ap.add_argument("--output-csv", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    if args.mode in _DEFERRED_MODES:
        print(
            f"mode '{args.mode}' is not implemented in the P0 gfx942 dense kernel "
            f"(follow-up: {_DEFERRED_MODES[args.mode]}). Nothing to run.",
            file=sys.stderr,
        )
        return 0

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"mode={args.mode} dtype={args.dtype} Hq={args.hq} Hkv={args.hkv} D={args.d} "
        f"bn={args.bn} wpe={args.waves_per_eu} warmup={args.warmup} iters={args.iterations}"
    )

    cfgs = _configs(args.mode, args.hq, args.hkv, args.d)
    results = []
    for mode, variant, label, S, B, Hq, Hkv, causal in cfgs:
        tag = f"[{mode}/{variant}] {label} Hq={Hq} Hkv={Hkv} D={args.d}"
        try:
            res = bench_dense(
                S,
                B,
                Hq,
                Hkv,
                args.d,
                causal=causal,
                dtype=args.dtype,
                block_n=args.bn,
                waves_per_eu=args.waves_per_eu,
                warmup=args.warmup,
                iters=args.iterations,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 - per-shape failures never abort
            import traceback

            traceback.print_exc()
            results.append(
                _record(
                    mode, variant, label, S, B, Hq, Hkv, args.d, causal, None, repr(exc)
                )
            )
            print(f"{tag}  FAILED ({exc!r})")
            continue

        rec = _record(mode, variant, label, S, B, Hq, Hkv, args.d, causal, res)
        results.append(rec)
        status = "PASS" if rec["ok"] else "FAIL"
        print(
            f"{tag}  {rec['dense_ms']:8.4f} ms  {rec['tflops']:8.1f} TFLOPS  "
            f"max_abs={rec['max_abs']:.2e}  {status}"
        )

    out_json = args.output_json
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {out_json}  ({len(results)} shapes)")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        cols = sorted({k for r in results for k in r.keys()})
        with open(args.output_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"wrote {args.output_csv}")

    print("\n=== per-mode geomean TFLOPS (correct shapes only) ===")
    modes = []
    for r in results:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    for m in modes:
        rs = [r for r in results if r["mode"] == m]
        tfs = [r["tflops"] for r in rs if r["ok"] and r["tflops"]]
        npass = sum(1 for r in rs if r["ok"])
        print(
            f"  {m:12s}  n={len(rs):3d}  geomean={_gm(tfs):8.1f} TFLOPS  pass={npass}/{len(rs)}"
        )
    total_pass = sum(1 for r in results if r["ok"])
    print(f"\nTOTAL PASS {total_pass}/{len(results)}")
    return 0 if total_pass == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
