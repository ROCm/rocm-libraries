#!/usr/bin/env python3
"""Benchmark dense (rocke) vs flyDSL vs unified_attention across sliding-window
and varlen-batch prefill shapes on gfx950 / MI355X.

Backends
--------
* ``dense``  : the productized rocke dense prefill kernel with the new
               ``sliding_window`` (A) and ``varlen`` (B) features
               (``kernels/gfx950/attention_dense.py``).
* ``flydsl`` : ``flydsl_flash_attn_func`` (DUALWAVE_SWP). Causal dense / packed
               varlen only -- flyDSL prefill has NO sliding-window, so on SWA
               rows it runs FULL causal (a "no-window" reference, marked *).
* ``unified``: rocke ``run_unified_attention_torch`` (paged 2D/3D). Supports
               both sliding-window and varlen (paged).

All three are timed with the same torch CUDA-event timer on the current stream.
Reported TFLOPS uses the honest banded pair count (windowed work); flyDSL's SWA
rows compute full-causal work so its TFLOPS bar is on a *different* workload
(hence the *). Correctness for dense is checked vs a banded-mask SDPA reference.

Usage:
    python bench_swa_varlen.py --mode swa
    python bench_swa_varlen.py --mode varlen
    python bench_swa_varlen.py --mode both --iters 50
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from kernels.gfx950.attention_dense import (  # noqa: E402
    AttentionDenseSpec,
    build_attention_dense,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.helpers.spec import SignatureBuilder  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig  # noqa: E402

_DT = torch.bfloat16


def _pairs(s, W):
    """causal (+window) attended (q,k) pairs for one length-s sequence."""
    if W and W > 0:
        return W * s - W * (W - 1) // 2 if s >= W else s * (s + 1) // 2
    return s * (s + 1) // 2


def _flops(seqlens, W, Hq, D):
    return sum(4 * Hq * D * _pairs(s, W) for s in seqlens)


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


# --------------------------------------------------------------------------- #
# rocke dense kernel
# --------------------------------------------------------------------------- #
_LAUNCHER_CACHE = {}


def _dense_launcher(spec):
    key = spec.kernel_name()
    if key in _LAUNCHER_CACHE:
        return _LAUNCHER_CACHE[key]
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


def bench_dense(seqlens, W, Hq, Hkv, D, warmup, iters, persistent):
    dev = "cuda"
    B = len(seqlens)
    max_s = max(seqlens)
    total = sum(seqlens)
    scale = 1.0 / math.sqrt(D)
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(0)
    varlen = B > 1 or (total != B * max_s)

    if varlen:
        q = (torch.randn(total, Hq, D, dtype=_DT, device=dev) * 0.2).contiguous()
        k = (torch.randn(total, Hkv, D, dtype=_DT, device=dev) * 0.2).contiguous()
        v = (torch.randn(total, Hkv, D, dtype=_DT, device=dev) * 0.2).contiguous()
        out = torch.zeros(total, Hq, D, dtype=_DT, device=dev)
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
            dtype="bf16",
            sliding_window=W,
            varlen=True,
        )
        lch = _dense_launcher(spec)
        cfg = LaunchConfig(
            grid=(max_s // 256, Hq, B), block=(spec.num_waves * 64, 1, 1), stream=stream
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
        S = seqlens[0]
        q = (torch.randn(1, S, Hq, D, dtype=_DT, device=dev) * 0.2).contiguous()
        k = (torch.randn(1, S, Hkv, D, dtype=_DT, device=dev) * 0.2).contiguous()
        v = (torch.randn(1, S, Hkv, D, dtype=_DT, device=dev) * 0.2).contiguous()
        out = torch.zeros(1, S, Hq, D, dtype=_DT, device=dev)
        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=S,
            seqlen_kv=S,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
            head_size=D,
            causal=True,
            dtype="bf16",
            sliding_window=W,
            persistent=persistent,
            num_persistent=256,
        )
        lch = _dense_launcher(spec)
        if persistent:
            grid = (256, 1, 1)
        else:
            grid = (S // 256, Hq, 1)
        cfg = LaunchConfig(grid=grid, block=(spec.num_waves * 64, 1, 1), stream=stream)
        vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": scale}

    def call():
        lch(vals, config=cfg)

    call()
    torch.cuda.synchronize()

    # correctness vs per-seq banded SDPA
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

    ms = _time(call, warmup, iters)
    tf = _flops(seqlens, W, Hq, D) / (ms * 1e-3) / 1e12
    return ms, tf, max_err


# --------------------------------------------------------------------------- #
# flyDSL
# --------------------------------------------------------------------------- #
# flyDSL 0.2.4 (DUALWAVE_SWP: dense/varlen, GQA/MHA, causal; NO sliding window).
# pip-installed into /tmp/flydsl024-venv (torch inherited from atom-venv via .pth);
# kernels come from the flydsl-main HEAD checkout. Runs in a subprocess with its
# own interpreter + PYTHONPATH so its top-level `kernels` package doesn't collide
# with rocke's.
_FLYDSL_PY = "/tmp/flydsl024-venv/bin/python"
_FLYDSL_ENV = {**os.environ, "PYTHONPATH": "/workspace/flydsl-main"}


def _flydsl_run(args):
    import subprocess

    cmd = [_FLYDSL_PY, os.path.join(_HERE, "_flydsl_bench.py"), *args]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=_FLYDSL_ENV
        )
    except Exception as e:  # noqa: BLE001
        return None, None, f"subproc: {e!r}"
    line = ""
    for ln in out.stdout.splitlines():
        if ln.startswith(("OK ", "ERR ")):
            line = ln
    if line.startswith("OK "):
        p = line.split()
        return float(p[1]), float(p[2]), (float(p[3]) if len(p) > 3 else "")
    return (
        None,
        None,
        (
            line[4:]
            if line.startswith("ERR ")
            else (out.stderr.strip().splitlines() or ["no output"])[-1]
        ),
    )


def bench_flydsl_dense(B, S, Hq, Hkv, D, warmup, iters, causal=True):
    return _flydsl_run(
        [
            "dense",
            str(B),
            str(S),
            str(Hq),
            str(Hkv),
            str(D),
            str(int(causal)),
            str(warmup),
            str(iters),
        ]
    )


def bench_flydsl_varlen(seqlens, Hq, Hkv, D, warmup, iters):
    sl = ",".join(str(x) for x in seqlens)
    return _flydsl_run(
        ["varlen", sl, str(Hq), str(Hkv), str(D), str(warmup), str(iters)]
    )


# --------------------------------------------------------------------------- #
# unified_attention (paged)
# --------------------------------------------------------------------------- #
def bench_unified(seqlens, W, Hq, Hkv, D, warmup, iters):
    try:
        from rocke.assets import shape_utils_dir

        sys.path.insert(0, str(shape_utils_dir()))
        import _ua_shape_utils as u
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch
    except Exception as e:  # noqa: BLE001
        return None, None, f"import: {e!r}"
    dev = "cuda"
    B = len(seqlens)
    total = sum(seqlens)
    max_s = max(seqlens)
    block_size = 16
    num_blocks = sum((s + block_size - 1) // block_size for s in seqlens) + B + 16
    max_blocks = (max_s + block_size - 1) // block_size
    win = (W - 1, 0) if (W and W > 0) else (-1, -1)
    try:
        shape = u.UAShape(
            source_file="synthetic",
            line_idx=0,
            call_idx=0,
            kind="prefill",
            all_decode=False,
            num_seqs=B,
            total_q=total,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
            head_size=D,
            block_size=block_size,
            num_blocks=num_blocks,
            max_blocks_per_seq=max_blocks,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            softmax_scale=1.0 / math.sqrt(D),
            softcap=0.0,
            window_size=win,
            has_sinks=False,
            has_alibi=False,
            has_output_scale=False,
            q_dtype="torch.bfloat16",
            k_dtype="torch.bfloat16",
            v_dtype="torch.bfloat16",
            out_dtype="torch.bfloat16",
        )
        data = u.make_inputs(shape, seed=2, cap_blocks=None)
        sw = W if (W and W > 0) else 0
        prob = UnifiedAttentionProblem(
            total_q=shape.total_q,
            num_seqs=shape.num_seqs,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
            head_size=D,
            block_size=block_size,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            dtype="bf16",
            sliding_window=sw,
            softcap=0.0,
            use_sinks=False,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            num_sms=256,
        )
        hip_stream = torch.cuda.current_stream().cuda_stream

        def call():
            run_unified_attention_torch(
                problem=prob,
                q=data["query"],
                k=data["key_cache"],
                v=data["value_cache"],
                out=data["output"],
                cu_seqlens_q=data["cu_seqlens_q"],
                seqused_k=data["kv_lens"],
                softmax_scale=data["scale"],
                block_table=data["block_tables"],
                softcap=0.0,
                sinks=data.get("sinks"),
                alibi_slopes=data.get("alibi_slopes"),
                qq_bias=None,
                qq_bias_stride_0=0,
                backend="auto",
                stream=hip_stream,
            )

        call()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return None, None, f"run: {e!r}"
    ms = _time(call, warmup, iters)
    tf = _flops(seqlens, W, Hq, D) / (ms * 1e-3) / 1e12
    return ms, tf, ""


def _row(label, res):
    ms, tf, note = res
    if ms is None:
        return f"  {label:<10} FAILED ({note})"
    extra = f"  max_abs={note:.2e}" if isinstance(note, float) else ""
    return f"  {label:<10} {ms:8.4f} ms  {tf:8.1f} TFLOPS{extra}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["swa", "varlen", "mha", "both", "all"], default="all"
    )
    ap.add_argument("--hq", type=int, default=128)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--persistent", action="store_true", default=True)
    args = ap.parse_args()
    Hq, Hkv, D = args.hq, args.hkv, args.d

    if args.mode in ("mha", "all"):
        print("\n=== DENSE-CAUSAL GQA (Hq=128, Hkv=8) -- headline prefill config ===")
        for S in (2048, 4096, 8192):
            print(f"\n-- B=1 S={S} Hq={Hq} Hkv={Hkv} D={D} causal --")
            print(
                _row(
                    "dense",
                    bench_dense(
                        [S], 0, Hq, Hkv, D, args.warmup, args.iters, args.persistent
                    ),
                )
            )
            print(
                _row(
                    "flydsl",
                    bench_flydsl_dense(1, S, Hq, Hkv, D, args.warmup, args.iters, True),
                )
            )
            print(
                _row(
                    "unified",
                    bench_unified([S], 0, Hq, Hkv, D, args.warmup, args.iters),
                )
            )

        print("\n=== DENSE-CAUSAL MHA ===")
        for H in (16, 32):
            for S in (4096, 8192):
                print(f"\n-- B=1 S={S} H={H} (MHA) D={D} causal --")
                print(
                    _row(
                        "dense",
                        bench_dense(
                            [S], 0, H, H, D, args.warmup, args.iters, args.persistent
                        ),
                    )
                )
                print(
                    _row(
                        "flydsl",
                        bench_flydsl_dense(
                            1, S, H, H, D, args.warmup, args.iters, True
                        ),
                    )
                )
                print(
                    _row(
                        "unified",
                        bench_unified([S], 0, H, H, D, args.warmup, args.iters),
                    )
                )

    if args.mode in ("swa", "both", "all"):
        print("\n\n=== SLIDING-WINDOW sweep (single-seq prefill, GQA, causal) ===")
        print("    (flyDSL v0.2.0 prefill has NO sliding-window -> not shown)")
        for S in (4096, 8192):
            for W in (0, 512, 1024, 2048, 4096):
                if W and W > S:
                    continue
                tag = "full-causal" if W == 0 else f"W={W}"
                print(f"\n-- S={S}  {tag}  (Hq={Hq} Hkv={Hkv}) --")
                print(
                    _row(
                        "dense",
                        bench_dense(
                            [S], W, Hq, Hkv, D, args.warmup, args.iters, args.persistent
                        ),
                    )
                )
                print(
                    _row(
                        "unified",
                        bench_unified([S], W, Hq, Hkv, D, args.warmup, args.iters),
                    )
                )

    if args.mode in ("varlen", "both", "all"):
        print("\n\n=== VARLEN sweep (packed ragged batch, GQA, causal) ===")
        print("    (flyDSL: W=0 only -- no sliding window in flyDSL prefill)")
        batches = [
            [2048, 2048, 2048, 2048],
            [512, 1024, 2048, 4096],
            [256, 512, 768, 1024, 1536, 2048],
            [1024] * 8,
        ]
        for seqlens in batches:
            for W in (0, 1024):
                tag = "full-causal" if W == 0 else f"W={W}"
                print(
                    f"\n-- B={len(seqlens)} total={sum(seqlens)} "
                    f"seqlens={seqlens} {tag} --"
                )
                print(
                    _row(
                        "dense",
                        bench_dense(
                            seqlens, W, Hq, Hkv, D, args.warmup, args.iters, False
                        ),
                    )
                )
                if W == 0:
                    print(
                        _row(
                            "flydsl",
                            bench_flydsl_varlen(
                                seqlens, Hq, Hkv, D, args.warmup, args.iters
                            ),
                        )
                    )
                print(
                    _row(
                        "unified",
                        bench_unified(seqlens, W, Hq, Hkv, D, args.warmup, args.iters),
                    )
                )


if __name__ == "__main__":
    main()
