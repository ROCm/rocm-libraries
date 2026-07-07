# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Ragged grouped GEMM workflow harness for gfx950 (CDNA4) -- test, time, validate.

Pure ragged grouped bf16 GEMM with variable M per expert. No gather/scatter/routing.
Takes m_sizes[E] as input, builds per-tile schedule, compares against torch reference.

Run:
    PYTHONPATH=Python python3 Python/rocke/examples/gfx950/grouped_gemm/ragged_gemm_hip.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.runtime.launcher import KernelLauncher, LaunchConfig


def make_group_sizes(m_total, e, dist, seed=0):
    """Generate per-expert row counts. Returns int64 tensor[E]."""
    if dist == "equal":
        base = m_total // e
        s = [base] * e
        s[-1] += m_total - base * e
        return torch.tensor(s, dtype=torch.int64)
    if dist == "bimodal":
        s = [4096 if i % 2 == 0 else 12288 for i in range(e)]
        rem = m_total - sum(s)
        s[-1] += rem
        return torch.tensor(s, dtype=torch.int64)
    # random ragged
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(e, generator=g) + 0.3
    w = w / w.sum() * m_total
    s = w.long()
    s[-1] += m_total - s.sum().item()
    return s


def build_sched(sizes, block_m, device):
    """Build per-tile schedule from m_sizes.

    Returns (expert_ids, m_offsets, m_valid, num_tiles, m_starts) where, for each
    tile: expert_ids = owning expert, m_offsets = absolute A/C row start,
    m_valid = valid rows in the tile (<= block_m; the tail tile of an expert may
    be partial). m_starts is the per-expert row start (for the reference).
    """
    sizes = sizes.to(device)
    E = sizes.shape[0]
    n_e = (sizes + block_m - 1) // block_m  # tiles per expert
    total_tiles = int(n_e.sum().item())
    tile_start = torch.cumsum(n_e, 0) - n_e
    tile_expert = torch.repeat_interleave(torch.arange(E, device=device), n_e)
    within = torch.arange(total_tiles, device=device) - tile_start[tile_expert]
    m_starts_expanded = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=sizes.dtype), sizes]), 0
    )
    m_starts_per_expert = m_starts_expanded[:-1]
    tile_moff = m_starts_per_expert[tile_expert] + within * block_m
    # valid rows in this tile = min(block_m, expert_size - within*block_m)
    remaining = sizes[tile_expert] - within * block_m
    tile_mvalid = torch.clamp(remaining, max=block_m)
    return (
        tile_expert.to(torch.int32),
        tile_moff.to(torch.int32),
        tile_mvalid.to(torch.int32),
        total_tiles,
        m_starts_per_expert.to(torch.int32),
    )


def _main() -> int:
    from rocke.instances.gfx950.ragged_gemm import (
        RaggedGemmSpec,
        build_ragged_gemm,
        ragged_gemm_signature,
    )

    N = int(os.environ.get("N", "1024"))
    K = int(os.environ.get("K", "512"))
    E = int(os.environ.get("E", "64"))
    M_total = int(os.environ.get("M", "524288"))
    dist = os.environ.get("DIST", "ragged")  # equal, ragged, bimodal

    torch.manual_seed(0)
    dev = f"cuda:{int(os.environ.get('DEVICE', '0'))}"

    # Generate ragged sizes
    m_sizes = make_group_sizes(M_total, E, dist)
    print(f"[rgemm] N={N} K={K} E={E} M_total={M_total} dist={dist}")
    print(
        f"[rgemm] m_sizes: min={m_sizes.min().item()} max={m_sizes.max().item()} mean={m_sizes.float().mean().item():.1f}"
    )

    spec = RaggedGemmSpec(
        N=N,
        K=K,
        E=E,
        hoist=os.environ.get("HOIST", "0") == "1",  # Default OFF: saves VGPR, +4% perf
        deeppipe=os.environ.get("DEEPPIPE", "0")
        == "1",  # Default OFF: saves VGPR, +4% perf
        epifuse=os.environ.get("EPIFUSE", "1") == "1",
        swz=os.environ.get("SWZ", "1") == "1",
        asm_reads=os.environ.get("ASM", "1") == "1",
        chiplet=os.environ.get("CHIP", "1") == "1",
        pin=os.environ.get("PIN", "1") == "1",
        b_rrr=os.environ.get("BRRR", "0") == "1",
        cshuf=os.environ.get("CSHUF", "0") == "1",
    )

    TM = spec.TM
    # A is padded by TM rows so the last tile's over-read stays mapped (see the
    # kernel's dtl_load_a note). The extra rows are never stored.
    X = torch.randn(M_total + TM, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(
        E,
        K if spec.b_rrr else N,
        N if spec.b_rrr else K,
        device=dev,
        dtype=torch.bfloat16,
    )
    # C has one extra sink row (index M_total): padded tile rows write there.
    C = torch.zeros(M_total + 1, N, device=dev, dtype=torch.bfloat16)

    # Build schedule
    expert_ids, m_offsets, m_valid, num_m_tiles, m_starts = build_sched(
        m_sizes, TM, dev
    )

    kernel, BS, tm, tn = build_ragged_gemm(spec)
    art = compile_kernel_via_hipcc(kernel, arch="gfx950")
    sig = ragged_gemm_signature()
    L = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=sig,
        cache_key=(kernel.name, N, K, E, spec.b_rrr),
    )

    grid = (math.ceil(N / spec.TN), num_m_tiles, 1)
    blk = (BS, 1, 1)
    kw = {
        "A": X.data_ptr(),
        "B": B.data_ptr(),
        "C": C.data_ptr(),
        "expert_ids": expert_ids.data_ptr(),
        "m_offsets": m_offsets.data_ptr(),
        "m_valid": m_valid.data_ptr(),
        "M": M_total,
        "num_m_tiles": num_m_tiles,
    }

    def call():
        L(kw, config=LaunchConfig(stream=0, grid=grid, block=blk))

    # warm
    for _ in range(10):
        call()
    torch.cuda.synchronize()

    # correctness (compare valid rows only; row M_total is the padding sink)
    Xf, Bf = X.float(), B.float()
    ref = torch.zeros(M_total, N, device=dev, dtype=torch.float32)
    for e in range(E):
        start = m_starts[e].item()
        sz = m_sizes[e].item()
        if sz > 0:
            B_e = Bf[e] if spec.b_rrr else Bf[e].T
            ref[start : start + sz] = Xf[start : start + sz] @ B_e

    skip_verify = os.environ.get("SKIP_VERIFY", "0") == "1"
    if not skip_verify:
        diff = (C[:M_total].float() - ref).abs()
        denom = ref.abs() + 1e-3
        max_abs = diff.max().item()
        rel = (diff / denom).max().item()
        status = "PASS" if rel < 0.02 else "FAIL"
        print(f"[rgemm] grid={grid} max_abs={max_abs:.4f} rel={rel:.4f} {status}")
        if status == "FAIL":
            return 1
    else:
        print(f"[rgemm] grid={grid} verification SKIPPED")

    # timing
    times = []
    for _ in range(10):
        s, e_ev = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(200):
            call()
        e_ev.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e_ev) / 200)
    times.sort()
    med_ms = times[len(times) // 2]
    flops = 2 * M_total * N * K
    tflops = flops / (med_ms * 1e-3) / 1e12
    peak_tflops = flops / (times[0] * 1e-3) / 1e12
    print(f"[rgemm] med={tflops:.1f} TF  peak={peak_tflops:.1f} TF")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
