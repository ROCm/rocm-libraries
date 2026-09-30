# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Ragged MoE grouped GEMM workflow harness for gfx950 (CDNA4) -- test, time, validate.

This harness demonstrates the ragged MoE kernel (fused gather + grouped GEMM + weighted
scatter). The kernel builder lives in rocke.instances.gfx950.ragged_moe; this file is
workflow-only: host prep (_moe_align sort + padding), correctness check, timing loop.

Inputs (moe_align precomputed on host, passed in):
* sorted_token_ids[num_m_tiles*TM] -- expanded token indices grouped by expert, each
  expert's run padded to a multiple of TM (sentinel = num_expanded).
* expert_ids[num_m_tiles] -- expert id per m-tile.
* routing_weights[num_expanded] -- per expanded-row routing weight.
* num_expanded (scalar) -- = num_tokens*top_k; also the padding sentinel and sink-row.

Shape, tile geometry, and every optimization lever are command-line flags (same
convention as ``grouped_gemm_hip.py``); see the ``RaggedMoeSpec`` docstring in
instances/gfx950/ragged_moe.py for what each lever does. Defaults are the
production config; use ``--no-<flag>`` to disable a boolean lever, e.g.
``--no-epifuse``. ``--help`` lists them all.

Run:
    PYTHONPATH=python python3 \
        python/rocke/examples/gfx950/grouped_gemm/ragged_moe_hip.py [--brrr ...]
"""

from __future__ import annotations

import argparse
import math

from rocke.helpers.compile import compile_kernel, compile_kernel_via_hipcc
from rocke.instances.gfx950.ragged_moe import (
    RaggedMoeSpec,
    build_ragged_moe,
    ragged_moe_signature,
)
from rocke.runtime.launcher import KernelLauncher, LaunchConfig


def _moe_align(expert_of, E, TM, num_expanded):
    """Host sort/align: group expanded rows by expert, pad each run to a
    multiple of TM. Returns (sorted_token_ids, expert_ids) as int32 numpy."""
    import numpy as np

    eo = expert_of.astype(np.int64)
    order = np.argsort(eo, kind="stable")  # expanded indices sorted by expert
    eo_sorted = eo[order]
    sorted_ids, expert_ids = [], []
    for e in range(E):
        idxs = order[eo_sorted == e]
        n = len(idxs)
        npad = ((n + TM - 1) // TM) * TM
        padded = np.full(npad, num_expanded, dtype=np.int32)  # sentinel
        padded[:n] = idxs.astype(np.int32)
        sorted_ids.append(padded)
        expert_ids.extend([e] * (npad // TM))
    sorted_ids = (
        np.concatenate(sorted_ids) if sorted_ids else np.zeros(0, dtype=np.int32)
    )
    return sorted_ids.astype(np.int32), np.asarray(expert_ids, dtype=np.int32)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="gfx950 fused ragged MoE bf16 GEMM harness")
    # Problem shape.
    p.add_argument("--num-tokens", type=int, default=524288)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--e", type=int, default=64)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--device", type=int, default=0)
    # Tile / warp geometry + chiplet knobs.
    p.add_argument("--tm", type=int, default=256)
    p.add_argument("--tn", type=int, default=256)
    p.add_argument("--tk", type=int, default=64)
    p.add_argument("--wm", type=int, default=2)
    p.add_argument("--wn", type=int, default=4)
    p.add_argument("--xcds", type=int, default=8)
    p.add_argument("--wgm", type=int, default=8)
    # Production levers (defaults on). Use --no-<flag> to disable.
    B = argparse.BooleanOptionalAction
    p.add_argument("--plog", action=B, default=True, help="overlap prologue B load")
    p.add_argument("--hoist", action=B, default=True, help="hoist gather/load offsets")
    p.add_argument("--rwlds", action=B, default=True, help="stage routing wts in LDS")
    p.add_argument("--epifuse", action=B, default=True, help="fuse scatter into MFMAs")
    p.add_argument("--swz", action=B, default=True, help="st_16x32 LDS XOR swizzle")
    p.add_argument("--asm", action=B, default=True, help="inline-asm ds_read")
    p.add_argument("--deeppipe", action=B, default=True, help="deep operand pipeline")
    p.add_argument("--prio", action=B, default=True, help="s_setprio around MFMAs")
    p.add_argument("--chip", action=B, default=True, help="chiplet super-tile remap")
    p.add_argument("--pin", action=B, default=True, help="pin wave-uniform indices")
    p.add_argument(
        "--burstprio", action=B, default=False, help="per-k-chunk prio toggle"
    )
    p.add_argument(
        "--brrr", action=B, default=False, help="NN weight layout (B=[E,K,N])"
    )
    # Experimental levers: measured NOT wins, kept for exploration (see the
    # RAGGED_CASE_STUDY.md negative results). Default off.
    p.add_argument(
        "--cshuf", action=B, default=False, help="C-shuffle wide-store epilogue"
    )
    p.add_argument(
        "--cksched", action=B, default=False, help="CK-HotLoop sched interleave"
    )
    p.add_argument(
        "--permmasched", action=B, default=False, help="per-MFMA prio + sched_barrier"
    )
    p.add_argument(
        "--tr128", action=B, default=False, help="ds_read_tr16_b128 transpose read"
    )
    p.add_argument("--opsw", action=B, default=False, help="operand-swap packed store")
    p.add_argument(
        "--combine", action=B, default=False, help="fused atomic top-k combine"
    )
    p.add_argument(
        "--storesink", action=B, default=False, help="diagnostic: collapse stores"
    )
    return p.parse_args(argv)


def _main(argv=None) -> int:
    args = _parse_args(argv)

    import numpy as np
    import torch

    N, K, E, TOPK, NTOK = args.n, args.k, args.e, args.topk, args.num_tokens

    spec = RaggedMoeSpec(
        N=N,
        K=K,
        E=E,
        TOPK=TOPK,
        TM=args.tm,
        TN=args.tn,
        TK=args.tk,
        WM=args.wm,
        WN=args.wn,
        plog=args.plog,
        hoist=args.hoist,
        rwlds=args.rwlds,
        epifuse=args.epifuse,
        swz=args.swz,
        asm_reads=args.asm,
        deeppipe=args.deeppipe,
        prio=args.prio,
        burstprio=args.burstprio,
        chiplet=args.chip,
        chiplet_xcds=args.xcds,
        chiplet_wgm=args.wgm,
        pin=args.pin,
        b_rrr=args.brrr,
        cshuf=args.cshuf,
        cksched=args.cksched,
        permmasched=args.permmasched,
        tr128=args.tr128,
        opsw=args.opsw,
        combine=args.combine,
        storesink=args.storesink,
    )

    torch.manual_seed(0)
    dev = f"cuda:{args.device}"
    X = torch.randn(NTOK, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(
        E,
        K if spec.b_rrr else N,
        N if spec.b_rrr else K,
        device=dev,
        dtype=torch.bfloat16,
    )
    rng = np.random.default_rng(1)
    expert_of = rng.choice(E, NTOK * TOPK)
    num_expanded = NTOK * TOPK
    sorted_ids_np, eids_np = _moe_align(expert_of, E, spec.TM, num_expanded)
    sorted_ids = torch.from_numpy(sorted_ids_np).to(dev).to(torch.int32)
    eids = torch.from_numpy(eids_np).to(dev).to(torch.int32)
    num_m_tiles = len(eids_np)
    routing_wt = torch.rand(num_expanded + 1, device=dev, dtype=torch.float32)
    routing_wt[num_expanded] = 0.0

    print(
        f"[rmoe] N={N} K={K} E={E} top_k={TOPK} num_tokens={NTOK} "
        f"num_expanded={num_expanded} num_m_tiles={num_m_tiles}"
    )
    out_rows = NTOK if spec.combine else num_expanded
    C = torch.zeros(out_rows + 1, N, device=dev, dtype=torch.bfloat16)

    kernel, BS, tm, tn = build_ragged_moe(spec)
    combine_needs_llvm = spec.combine and spec.b_rrr
    art = (
        compile_kernel(kernel, arch="gfx950")
        if combine_needs_llvm
        else compile_kernel_via_hipcc(kernel, arch="gfx950")
    )
    sig = ragged_moe_signature()
    L = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=sig,
        cache_key=(kernel.name, N, K, E, TOPK, spec.combine),
    )

    grid = (math.ceil(N / spec.TN), num_m_tiles, 1)
    blk = (BS, 1, 1)
    kw = {
        "A": X.data_ptr(),
        "B": B.data_ptr(),
        "C": C.data_ptr(),
        "sorted_token_ids": sorted_ids.data_ptr(),
        "expert_ids": eids.data_ptr(),
        "routing_weights": routing_wt.data_ptr(),
        "num_expanded": num_expanded,
        "num_m_tiles": num_m_tiles,
    }

    def call():
        L(kw, config=LaunchConfig(stream=0, grid=grid, block=blk))

    # warm
    for _ in range(10):
        call()
    torch.cuda.synchronize()

    # correctness
    Xf, Bf = X.float(), B.float()
    eo_t = torch.from_numpy(expert_of).to(dev)
    ref_exp = torch.zeros(num_expanded, N, device=dev, dtype=torch.float32)
    for e in range(E):
        m = eo_t == e
        if m.any():
            tok = torch.div(torch.nonzero(m).squeeze(-1), TOPK, rounding_mode="floor")
            B_e = Bf[e] if spec.b_rrr else Bf[e].T
            ref_exp[m] = Xf[tok] @ B_e
    ref_exp *= routing_wt[:num_expanded, None]
    if spec.combine:
        ref = torch.zeros(NTOK, N, device=dev, dtype=torch.float32)
        for t in range(NTOK):
            ref[t] = ref_exp[t * TOPK : t * TOPK + TOPK].sum(0)
    else:
        ref = ref_exp

    diff = (C[:out_rows].float() - ref).abs()
    denom = ref.abs() + 1e-3
    max_abs = diff.max().item()
    rel = (diff / denom).max().item()
    status = "PASS" if rel < 0.02 else "FAIL"
    cstr = " COMBINE" if spec.combine else ""
    print(f"[rmoe] grid={grid}{cstr} max_abs={max_abs:.4f} rel={rel:.4f} {status}")
    if status == "FAIL":
        return 1

    # timing
    times = []
    for _ in range(10):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(200):
            call()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 200)
    times.sort()
    med_ms = times[len(times) // 2]
    flops = 2 * num_expanded * N * K
    tflops = flops / (med_ms * 1e-3) / 1e12
    peak_tflops = flops / (times[0] * 1e-3) / 1e12
    print(f"[rmoe] med={tflops:.1f} TF  peak={peak_tflops:.1f} TF")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
