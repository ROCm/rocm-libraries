# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LIVE benchmark for the dense flash-attention prefill kernel on gfx950.

Times the productized dense prefill kernel
(``kernels/gfx950/attention_dense.py``: :class:`AttentionDenseSpec` /
:func:`build_attention_dense`) across every variant it ships --

  * ``causal``     : headline GQA causal prefill (Hq=128, Hkv=8),
  * ``mha``        : multi-head (Hq==Hkv) causal prefill,
  * ``swa``        : sliding-window (banded causal) prefill,
  * ``varlen``     : packed ragged batch (cu_seqlens) prefill,
  * ``persistent`` : grid-stride persistent variant of the GQA causal shapes,

against a torch SDPA reference on the *same* HIP stream, using the canonical
``rocke.runtime.time_launches`` timer (HIP events, NOT torch events). Every
shape reports honest windowed TFLOPS (banded pair count), the max-abs error vs
SDPA (causal / banded / per-sequence), a PASS/FAIL flag (< 2e-2), and the built
kernel name. Emits a JSON report (and optional CSV) plus a per-mode geomean.

Run as a library module::

    PYTHONPATH=rocke/library python3 -m \\
        benchmarks.gfx950.attention.prefill.benchmark_dense_prefill_live \\
        --mode all --output-json /tmp/dense_prefill_live.json

or directly with the atom-venv python::

    /workspace/atom-venv/bin/python \\
        rocke/library/benchmarks/gfx950/attention/prefill/benchmark_dense_prefill_live.py \\
        --mode causal --iterations 5 --warmup 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

# --- sys.path bootstrap so this runs both as a module and directly (mirrors
#     builders/gfx950/attention/prefill/bench_swa_varlen.py): insert the rocke
#     platform (`platform/python`) and library (`library`) roots. ---
_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from kernels.gfx950.attention_dense import (  # noqa: E402
    AttentionDenseSpec,
    attention_dense_block,
    attention_dense_grid,
    build_attention_dense,
    supports_attention_dense,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.helpers.spec import SignatureBuilder  # noqa: E402
from rocke.runtime import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)

_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}
_TOL = 2e-2


# --------------------------------------------------------------------------- #
# honest FLOPs (banded / windowed causal pair count) -- lifted from
# builders/gfx950/attention/prefill/bench_swa_varlen.py
# --------------------------------------------------------------------------- #
def _pairs(s: int, W: int) -> int:
    """causal (+window) attended (q,k) pairs for one length-s sequence."""
    if W and W > 0:
        return W * s - W * (W - 1) // 2 if s >= W else s * (s + 1) // 2
    return s * (s + 1) // 2


def _flops(seqlens, W: int, Hq: int, D: int) -> int:
    return sum(4 * Hq * D * _pairs(s, W) for s in seqlens)


def _gm(vals) -> float:
    vals = [v for v in vals if v > 0]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


def _bench_stream_handle() -> int:
    return int(torch.cuda.current_stream().cuda_stream)


# --------------------------------------------------------------------------- #
# dense kernel launcher (compile + ABI signature; varlen adds cu_seqlens ptrs)
# --------------------------------------------------------------------------- #
_LAUNCHER_CACHE: dict = {}


def _dense_launcher(spec: AttentionDenseSpec) -> KernelLauncher:
    # `batch` and `waves_per_eu` are baked into the kernel -- batch through the
    # K/V buffer extents, waves_per_eu through the amdgpu-waves-per-eu attribute
    # -- but neither is part of kernel_name(), so both have to be part of the
    # cache key. Keying on the name alone collides here in practice: the varlen
    # sweep runs B=4 and B=6 at the same max_s, which share a name but lower to
    # different IR, so the second silently reuses the first binary.
    key = (spec.kernel_name(), spec.batch, spec.waves_per_eu)
    lch = _LAUNCHER_CACHE.get(key)
    if lch is not None:
        return lch
    ok, why = supports_attention_dense(spec)
    if not ok:
        raise ValueError(f"unsupported spec: {why}")
    art = compile_kernel(
        build_attention_dense(spec),
        arch="gfx950",
        backend="python",
        capture_ir_text=False,
    )
    sb = (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("v_ptr", spec.dtype)
        .ptr("o_ptr", spec.dtype)
        .scalar("scale", "f32")
    )
    if spec.varlen:
        sb = sb.ptr("cu_seqlens_q", "i32").ptr("cu_seqlens_kv", "i32")
    lch = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sb.build()
    )
    _LAUNCHER_CACHE[key] = lch
    return lch


# --------------------------------------------------------------------------- #
# one dense benchmark point: build inputs, check parity vs SDPA, time it
# --------------------------------------------------------------------------- #
def bench_dense(
    seqlens,
    W: int,
    Hq: int,
    Hkv: int,
    D: int,
    *,
    dtype: str,
    block_n: int,
    waves_per_eu: int,
    lds_k_group_pad: int,
    num_persistent: int,
    persistent: bool,
    warmup: int,
    iters: int,
    seed: int,
):
    """Returns (dense_ms, tflops, max_abs, kernel_name)."""
    dev = "cuda"
    dt = _TORCH_DT[dtype]
    B = len(seqlens)
    max_s = max(seqlens)
    total = sum(seqlens)
    scale = 1.0 / math.sqrt(D)
    stream = _bench_stream_handle()
    torch.manual_seed(seed)
    # packed ragged batch when >1 sequence (or a single non-uniform packing).
    varlen = B > 1 or (total != B * max_s)

    if varlen:
        # packed [total_tok, H, D] + int32 cu_seqlens[B+1].
        q = (torch.randn(total, Hq, D, dtype=dt, device=dev) * 0.2).contiguous()
        k = (torch.randn(total, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
        v = (torch.randn(total, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
        out = torch.zeros(total, Hq, D, dtype=dt, device=dev)
        cu = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=dev).cumsum(0)
        spec = AttentionDenseSpec(
            batch=B,
            seqlen_q=max_s,
            seqlen_kv=max_s,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
            head_size=D,
            causal=True,
            dtype=dtype,
            block_n=block_n,
            waves_per_eu=waves_per_eu,
            lds_k_group_pad=lds_k_group_pad,
            sliding_window=W,
            varlen=True,
        )
        lch = _dense_launcher(spec)
        cfg = LaunchConfig(
            grid=attention_dense_grid(spec),
            block=attention_dense_block(spec),
            stream=stream,
        )
        vals = {
            "q_ptr": q,
            "k_ptr": k,
            "v_ptr": v,
            "o_ptr": out,
            "scale": scale,
            "cu_seqlens_q": cu,
            "cu_seqlens_kv": cu,
        }
    else:
        # uniform [B=1, S, H, D].
        S = seqlens[0]
        q = (torch.randn(1, S, Hq, D, dtype=dt, device=dev) * 0.2).contiguous()
        k = (torch.randn(1, S, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
        v = (torch.randn(1, S, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
        out = torch.zeros(1, S, Hq, D, dtype=dt, device=dev)
        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=S,
            seqlen_kv=S,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
            head_size=D,
            causal=True,
            dtype=dtype,
            block_n=block_n,
            waves_per_eu=waves_per_eu,
            lds_k_group_pad=lds_k_group_pad,
            sliding_window=W,
            persistent=persistent,
            num_persistent=num_persistent,
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

    # --- correctness vs per-seq banded / causal SDPA ---
    rep = Hq // Hkv
    max_err = 0.0
    for i, s in enumerate(seqlens):
        if varlen:
            st = int(sum(seqlens[:i]))
            qs, ks, vs_, os_ = (
                q[st : st + s],
                k[st : st + s],
                v[st : st + s],
                out[st : st + s],
            )
            qh = qs.transpose(0, 1).float().unsqueeze(0)
            kh = ks.transpose(0, 1).repeat_interleave(rep, 0).float().unsqueeze(0)
            vh = vs_.transpose(0, 1).repeat_interleave(rep, 0).float().unsqueeze(0)
            ot = os_
        else:
            qh = q[0].transpose(0, 1).float().unsqueeze(0)
            kh = k[0].transpose(0, 1).repeat_interleave(rep, 0).float().unsqueeze(0)
            vh = v[0].transpose(0, 1).repeat_interleave(rep, 0).float().unsqueeze(0)
            ot = out[0]
        if W and W > 0:
            qi = torch.arange(s, device=dev).view(-1, 1)
            ki = torch.arange(s, device=dev).view(1, -1)
            m = (ki <= qi) & (ki > qi - W)
            ref = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=m
            )
        else:
            ref = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=True
            )
        ref = ref.squeeze(0).transpose(0, 1)
        max_err = max(max_err, (ot.float() - ref).abs().max().item())

    ms = time_launches(call, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    tf = _flops(seqlens, W, Hq, D) / (ms * 1e-3) / 1e12
    return ms, tf, max_err, spec.kernel_name()


# --------------------------------------------------------------------------- #
# shape sweeps -- ported from bench_swa_varlen.py main()
# --------------------------------------------------------------------------- #
def _configs(mode: str, Hq: int, Hkv: int, D: int):
    """Yield (mode, variant, label, seqlens, Hq, Hkv, W, persistent) configs."""
    cfgs = []

    if mode in ("causal", "all"):
        for S in (2048, 4096, 8192):
            cfgs.append(("causal", "gqa_causal", f"S={S}", [S], Hq, Hkv, 0, False))

    if mode in ("mha", "all"):
        for H in (16, 32):
            for S in (4096, 8192):
                cfgs.append(("mha", "mha", f"H={H} S={S}", [S], H, H, 0, False))

    if mode in ("swa", "all"):
        for S in (4096, 8192):
            for W in (0, 512, 1024, 2048, 4096):
                if W and W > S:
                    continue
                tag = "full-causal" if W == 0 else f"W={W}"
                cfgs.append(("swa", "swa", f"S={S} {tag}", [S], Hq, Hkv, W, False))

    if mode in ("varlen", "all"):
        batches = [
            [2048, 2048, 2048, 2048],
            [512, 1024, 2048, 4096],
            [256, 512, 768, 1024, 1536, 2048],
            [1024] * 8,
        ]
        for seqlens in batches:
            for W in (0, 1024):
                tag = "full-causal" if W == 0 else f"W={W}"
                label = f"B={len(seqlens)} total={sum(seqlens)} {tag}"
                cfgs.append(
                    ("varlen", "varlen", label, list(seqlens), Hq, Hkv, W, False)
                )

    if mode in ("persistent", "all"):
        for S in (2048, 4096, 8192):
            cfgs.append(("persistent", "persistent", f"S={S}", [S], Hq, Hkv, 0, True))

    return cfgs


def _record(mode, variant, label, seqlens, Hq, Hkv, D, W, knobs, res, err_note=None):
    """One sweep row. ``knobs`` carries the tuning values the row was measured at
    so the report is self-describing: kernel_name() omits some of them, and the
    ones it does carry are positional rather than labelled."""
    rec = {
        "label": label,
        "mode": mode,
        "variant": variant,
        "seqlens": list(seqlens),
        "Hq": Hq,
        "Hkv": Hkv,
        "D": D,
        "sliding_window": W,
        **knobs,
    }
    if res is None:
        rec.update(
            {
                "dense_ms": None,
                "tflops": None,
                "max_abs": None,
                "ok": False,
                "kernel_name": None,
                "error": err_note,
            }
        )
        return rec
    ms, tf, err, kname = res
    rec.update(
        {
            "dense_ms": ms,
            "tflops": tf,
            "max_abs": err,
            "ok": bool(err < _TOL),
            "kernel_name": kname,
        }
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--mode",
        choices=["causal", "mha", "swa", "varlen", "persistent", "all"],
        default="all",
    )
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--hq", type=int, default=128, help="query heads (GQA/SWA/varlen)")
    ap.add_argument("--hkv", type=int, default=8, help="kv heads (GQA/SWA/varlen)")
    ap.add_argument("--d", type=int, default=128, help="head size")
    ap.add_argument("--bn", type=int, default=64, help="block_n (KV tile)")
    ap.add_argument("--waves-per-eu", type=int, default=2, help="occupancy hint")
    ap.add_argument(
        "--lds-k-group-pad",
        type=int,
        default=8,
        help="K LDS pad between DMA row-groups, in elements; live only at D=64 "
        "(at D=128 one DMA fills one row and the per-row pad applies instead). "
        "Must be a multiple of 8; 0 reproduces the unpadded layout",
    )
    ap.add_argument(
        "--num-persistent", type=int, default=256, help="CTAs for persistent variant"
    )
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-json", type=str, default="/tmp/dense_prefill_live.json")
    ap.add_argument("--output-csv", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"mode={args.mode} dtype={args.dtype} Hq={args.hq} Hkv={args.hkv} D={args.d} "
        f"bn={args.bn} wpe={args.waves_per_eu} kpad={args.lds_k_group_pad} "
        f"np={args.num_persistent} warmup={args.warmup} iters={args.iterations}"
    )

    knobs = {
        "dtype": args.dtype,
        "block_n": args.bn,
        "waves_per_eu": args.waves_per_eu,
        "lds_k_group_pad": args.lds_k_group_pad,
    }
    cfgs = _configs(args.mode, args.hq, args.hkv, args.d)
    results = []
    for mode, variant, label, seqlens, Hq, Hkv, W, persistent in cfgs:
        tag = f"[{mode}/{variant}] {label} Hq={Hq} Hkv={Hkv} D={args.d}"
        try:
            res = bench_dense(
                seqlens,
                W,
                Hq,
                Hkv,
                args.d,
                dtype=args.dtype,
                block_n=args.bn,
                waves_per_eu=args.waves_per_eu,
                lds_k_group_pad=args.lds_k_group_pad,
                num_persistent=args.num_persistent,
                persistent=persistent,
                warmup=args.warmup,
                iters=args.iterations,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 - per-shape failures never abort
            import traceback

            traceback.print_exc()
            rec = _record(
                mode,
                variant,
                label,
                seqlens,
                Hq,
                Hkv,
                args.d,
                W,
                knobs,
                None,
                repr(exc),
            )
            results.append(rec)
            print(f"{tag}  FAILED ({exc!r})")
            continue

        rec = _record(mode, variant, label, seqlens, Hq, Hkv, args.d, W, knobs, res)
        results.append(rec)
        status = "PASS" if rec["ok"] else "FAIL"
        print(
            f"{tag}  {rec['dense_ms']:8.4f} ms  {rec['tflops']:8.1f} TFLOPS  "
            f"max_abs={rec['max_abs']:.2e}  {status}"
        )

    # --- JSON report ---
    out_json = args.output_json
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {out_json}  ({len(results)} shapes)")

    # --- optional CSV report ---
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        cols = sorted({k for r in results for k in r.keys()})
        with open(args.output_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in results:
                row = dict(r)
                row["seqlens"] = ",".join(str(x) for x in row["seqlens"])
                w.writerow(row)
        print(f"wrote {args.output_csv}")

    # --- per-mode geomean summary ---
    print("\n=== per-mode geomean TFLOPS (correct shapes only) ===")
    modes = []
    for r in results:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    n_pass = 0
    for m in modes:
        rs = [r for r in results if r["mode"] == m]
        tfs = [r["tflops"] for r in rs if r["ok"] and r["tflops"]]
        npass = sum(1 for r in rs if r["ok"])
        n_pass += npass
        print(
            f"  {m:12s}  n={len(rs):3d}  geomean={_gm(tfs):8.1f} TFLOPS  "
            f"pass={npass}/{len(rs)}"
        )
    total_pass = sum(1 for r in results if r["ok"])
    print(f"\nTOTAL PASS {total_pass}/{len(results)}")
    return 0 if total_pass == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
