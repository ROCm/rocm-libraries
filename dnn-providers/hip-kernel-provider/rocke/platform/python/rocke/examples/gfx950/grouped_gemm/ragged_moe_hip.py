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

Env levers: PLOG, HOIST, RWLDS, EPIFUSE, CSHUF, BRRR, ASM, SWZ, CHIP, COMBINE
(see instances/gfx950/ragged_moe.py docstring for details).

Run:
    PYTHONPATH=Python python3 Python/rocke/examples/gfx950/grouped_gemm/ragged_moe_hip.py
"""

from __future__ import annotations

import math
import os

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


def _main() -> int:
    import numpy as np
    import torch

    N = int(os.environ.get("N", "1024"))
    K = int(os.environ.get("K", "512"))
    E = int(os.environ.get("E", "64"))
    TOPK = int(os.environ.get("TOPK", "2"))
    NTOK = int(os.environ.get("NTOK", "524288"))

    # Build spec from env (defaults = production opts, env can override)
    spec = RaggedMoeSpec(
        N=N,
        K=K,
        E=E,
        TOPK=TOPK,
        plog=os.environ.get("PLOG", "1") == "1",
        hoist=os.environ.get("HOIST", "1") == "1",
        rwlds=os.environ.get("RWLDS", "1") == "1",
        epifuse=os.environ.get("EPIFUSE", "1") == "1",
        swz=os.environ.get("SWZ", "1") == "1",
        asm_reads=os.environ.get("ASM", "1") == "1",
        chiplet=os.environ.get("CHIP", "1") == "1",
        pin=os.environ.get("PIN", "1") == "1",
        b_rrr=os.environ.get("BRRR", "0") == "1",
        cshuf=os.environ.get("CSHUF", "0") == "1",
        cksched=os.environ.get("CKSCHED", "0") == "1",
        permmasched=os.environ.get("PERMMASCHED", "0") == "1",
        tr128=os.environ.get("TR128", "0") == "1",
        opsw=os.environ.get("OPSW", "0") == "1",
        combine=os.environ.get("COMBINE", "0") == "1",
        storesink=os.environ.get("STORESINK", "0") == "1",
    )

    torch.manual_seed(0)
    dev = f"cuda:{int(os.environ.get('DEVICE', '0'))}"
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
