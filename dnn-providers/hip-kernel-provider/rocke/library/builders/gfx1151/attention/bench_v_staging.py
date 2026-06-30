# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1151 WMMA FMHA-forward case study: A/B the PV V-operand source.

This is the measurement harness behind ``README.md``. It applies the
optimization runbook's "one lever at a time" discipline to the single WMMA
FMHA-forward kernel, toggling exactly one knob -- ``WmmaFmhaFwdSpec.v_lds_stage``
-- between two variants:

  * ``v0_vgather`` (``v_lds_stage=False``): the correctness-first baseline. The
    PV matmul's B-operand (V in d x k layout) is gathered straight from global
    memory with a per-(d, k) scalar ``global_load`` -- ``n_dk * 16`` scalar
    loads per lane per K-tile.
  * ``v1_vlds`` (``v_lds_stage=True``): each lane loads its own K-row's full
    ``head_size`` slice with 8-wide vector ``global_load``s into a 16 x
    head_size LDS tile once per K-tile, then the PV B-operand is a strided
    *LDS* read. Trades ~3x the global-load traffic for LDS round-trips.

For each variant it (1) builds + compiles the HSACO, (2) gates on a one-shot
numpy correctness check, (3) times the launch with HIP events, and (4)
disassembles the HSACO and counts the memory-instruction mix so the speed
delta can be explained, not just reported. Results are printed as a table and
written to ``v_staging_perf.csv``.

Must run on a gfx1151 device.

    PYTHONPATH=Python python3 -m rocke.examples.gfx1151.attention.bench_v_staging \
        --seqlen-q 512 --seqlen-k 512 --head-size 128 --heads 8 --batch 4
"""

from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from rocke.helpers import compile_kernel
from rocke.instances.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    build_wmma_fmha_fwd,
    wmma_fmha_fwd_grid,
)
from rocke.runtime.hip_module import Runtime
from rocke.runtime.launcher import time_launches

VARIANTS = (
    ("v0_vgather", False),
    ("v1_vlds", True),
)


def _ref_attention(Q, K, V, *, causal: bool):
    """Dense attention reference, Q/K/V shape ``(seqlen, heads, head_size)``."""
    import numpy as np

    d = Q.shape[-1]
    scores = np.einsum("ihd,jhd->ihj", Q.astype(np.float32), K.astype(np.float32))
    scores /= math.sqrt(d)
    if causal:
        q_pos = np.arange(Q.shape[0])[:, None, None]
        k_pos = np.arange(K.shape[0])[None, None, :]
        scores = np.where(k_pos <= q_pos, scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", probs, V.astype(np.float32))
    return out.astype(np.float16)


def _find_objdump() -> str | None:
    """Locate llvm-objdump (PATH, then the rocm-sdk wheel that ships with the
    gfx1151 venv). Returns ``None`` if not found -- ISA counts are then skipped
    rather than failing the benchmark."""
    for cand in ("llvm-objdump", "llvm-objdump.exe"):
        hit = shutil.which(cand)
        if hit:
            return hit
    for base in sys.path:
        p = Path(base)
        # site-packages/_rocm_sdk_core/lib/llvm/bin/llvm-objdump(.exe)
        for sub in (
            "_rocm_sdk_core/lib/llvm/bin",
            "../_rocm_sdk_core/lib/llvm/bin",
        ):
            for name in ("llvm-objdump.exe", "llvm-objdump"):
                c = p / sub / name
                if c.exists():
                    return str(c)
    return None


_MEM_CATEGORIES = (
    ("wmma", lambda m: "wmma" in m),
    ("global_load", lambda m: m.startswith("global_load")),
    ("global_store", lambda m: m.startswith("global_store")),
    ("ds_load", lambda m: m.startswith("ds_load")),
    ("ds_store", lambda m: m.startswith("ds_store")),
)


def _isa_counts(hsaco: bytes, kernel_name: str, objdump: str | None) -> dict:
    """Disassemble and tally the memory/matmul instruction mix. Static counts
    over the (fully unrolled) kernel body -- a proxy for per-K-tile traffic."""
    if objdump is None:
        return {}
    tmp = Path(tempfile.gettempdir()) / (kernel_name + ".hsaco")
    tmp.write_bytes(hsaco)
    try:
        out = subprocess.run(
            [objdump, "-d", str(tmp)], capture_output=True, text=True, check=False
        ).stdout
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    counts = collections.Counter()
    for line in out.splitlines():
        s = line.strip()
        if not s.startswith(("s_", "v_", "ds_", "global_", "buffer_")):
            continue
        mnem = s.split()[0]
        for name, pred in _MEM_CATEGORIES:
            if pred(mnem):
                counts[name] += 1
                break
    return {name: counts.get(name, 0) for name, _ in _MEM_CATEGORIES}


def _make_packed(rt, args, spec):
    import numpy as np

    B, Hq, Hk, D = args.batch, args.heads, spec.kv_heads, args.head_size
    Sq, Sk = args.seqlen_q, args.seqlen_k
    rng = np.random.default_rng(0xA11E)
    Q = (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, Sq, Hq, D), dtype=np.float16)

    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (Q, K, V, Out))
    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(K), K.nbytes)
    rt.memcpy_h2d(vd, u8(V), V.nbytes)
    rt.memset(od, 0, Out.nbytes)

    packed = struct.pack(
        "<QQQQfiiiiiiiiii",
        qd,
        kd,
        vd,
        od,
        scale_log2,
        Sq,
        Sk,
        Hq * D,
        D,  # stride_q_token, stride_q_head
        Hk * D,
        D,  # stride_k_token, stride_k_head
        Hk * D,
        D,  # stride_v_token, stride_v_head
        Hq * D,
        D,  # stride_o_token, stride_o_head
    )
    bufs = {"Q": Q, "K": K, "V": V, "Out": Out, "ptrs": (qd, kd, vd, od), "od": od}
    return packed, bufs


def _verify(args, spec, bufs, rt):
    import numpy as np

    Q, K, V, Out = bufs["Q"], bufs["K"], bufs["V"], bufs["Out"]
    B, Hq, Hk = args.batch, args.heads, spec.kv_heads

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    rt.memcpy_d2h(u8(Out), bufs["od"], Out.nbytes)
    ref = np.empty_like(Out)
    for bi in range(B):
        if Hk != Hq:
            rep = Hq // Hk
            Kb = np.repeat(K[bi], rep, axis=1)
            Vb = np.repeat(V[bi], rep, axis=1)
        else:
            Kb, Vb = K[bi], V[bi]
        ref[bi] = _ref_attention(Q[bi], Kb, Vb, causal=args.causal)
    return float(np.abs(Out.astype(np.float32) - ref.astype(np.float32)).max())


def _run_variant(args, label, v_lds_stage, objdump):
    kvh = args.kv_heads or args.heads
    spec = WmmaFmhaFwdSpec(
        head_size=args.head_size,
        num_query_heads=args.heads,
        num_kv_heads=kvh,
        mask_mode="causal" if args.causal else "none",
        v_lds_stage=v_lds_stage,
        name=f"fmha_case_{label}",
    )
    art = compile_kernel(build_wmma_fmha_fwd(spec, arch=args.arch), arch=args.arch)
    isa = _isa_counts(art.hsaco, art.kernel_name, objdump)

    grid = wmma_fmha_fwd_grid(spec, seqlen_q=args.seqlen_q, batch=args.batch)
    block = (spec.block_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)
    packed, bufs = _make_packed(rt, args, spec)

    # ---- Correctness gate (one launch) before timing ----
    rt.launch(fn, grid, block, packed)
    rt.sync()
    max_abs = _verify(args, spec, bufs, rt)
    ok = max_abs <= args.tol

    ms = time_launches(
        lambda: rt.launch(fn, grid, block, packed),
        warmup=args.warmup,
        iters=args.iters,
    )

    for ptr in bufs["ptrs"]:
        rt.free(ptr)
    module.unload()

    B, Hq, D = args.batch, args.heads, args.head_size
    Sq, Sk = args.seqlen_q, args.seqlen_k
    flops = 4.0 * B * Hq * Sq * Sk * D
    if args.causal:
        flops *= 0.5
    tflops = flops / (ms * 1e-3) / 1e12
    return {
        "variant": label,
        "v_lds_stage": v_lds_stage,
        "hsaco_bytes": art.hsaco_bytes,
        "max_abs": max_abs,
        "ok": ok,
        "us_per_iter": ms * 1e3,
        "tflops": tflops,
        **{f"isa_{k}": v for k, v in isa.items()},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1151")
    p.add_argument("--seqlen-q", type=int, default=512)
    p.add_argument("--seqlen-k", type=int, default=512)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=0, help="0 -> MHA (== heads)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--csv", default=None, help="output CSV path (default beside this file)"
    )
    args = p.parse_args()

    for d, name in (
        (args.seqlen_q, "seqlen_q"),
        (args.seqlen_k, "seqlen_k"),
        (args.head_size, "head_size"),
    ):
        if d % 16:
            raise SystemExit(f"{name}={d} must be a multiple of 16 (WMMA 16x16 tile)")

    objdump = _find_objdump()
    if objdump is None:
        print("[warn] llvm-objdump not found; ISA columns will be blank")

    rows = [_run_variant(args, label, vlds, objdump) for label, vlds in VARIANTS]

    kvh = args.kv_heads or args.heads
    print(
        f"\nWMMA FMHA fwd  arch={args.arch}  B={args.batch} Sq={args.seqlen_q} "
        f"Sk={args.seqlen_k} D={args.head_size} Hq={args.heads} Hk={kvh} "
        f"causal={args.causal}"
    )
    hdr = (
        f"{'variant':<11} {'ok':>3} {'max_abs':>9} {'us/iter':>9} {'TFLOP/s':>8} "
        f"{'gld':>5} {'gst':>5} {'dsld':>5} {'dsst':>5} {'wmma':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    base_us = None
    for r in rows:
        if base_us is None:
            base_us = r["us_per_iter"]
        print(
            f"{r['variant']:<11} {'Y' if r['ok'] else 'N':>3} "
            f"{r['max_abs']:>9.2e} {r['us_per_iter']:>9.1f} {r['tflops']:>8.2f} "
            f"{r.get('isa_global_load', ''):>5} {r.get('isa_global_store', ''):>5} "
            f"{r.get('isa_ds_load', ''):>5} {r.get('isa_ds_store', ''):>5} "
            f"{r.get('isa_wmma', ''):>5}"
        )
    if len(rows) == 2 and rows[0]["us_per_iter"] > 0:
        speedup = rows[0]["us_per_iter"] / rows[1]["us_per_iter"]
        print(f"\nv1_vlds speedup over v0_vgather: {speedup:.2f}x")

    out_csv = args.csv or str(Path(__file__).with_name("v_staging_perf.csv"))
    fields = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
