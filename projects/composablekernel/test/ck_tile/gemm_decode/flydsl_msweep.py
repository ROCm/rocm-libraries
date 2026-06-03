#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""R1 reference side: best-config M-sweep for FlyDSL's small_m_hgemm kernel.

Mirrors test/ck_tile/gemm_decode/bench_msweep.cpp on the FlyDSL side so the two
CSVs can be joined on (impl, M, N, K). For a fixed (N, K) it sweeps a curated
set of small-M kernel configs (TILE_M=16 MFMA path), compiles each ONCE (the
kernel is compiled for (n, k, config) and takes m as a runtime arg, so a single
compile is reused across the whole M=1..16 sweep), times every (M, config)
cell, and keeps the fastest config per M.

Driven by the venv python with the flydsl wheel layered on via PYTHONPATH, e.g.

  PYTHONPATH=/home/AMD/samremes/dev/.r1_flydsl_pkgs \
    /opt/venv/bin/python3 flydsl_msweep.py \
    --flydsl-repo /home/AMD/samremes/dev/FlyDSL \
    --N 8192 --K 7168 --mmax 16 --warmup 10 --repeat 100 \
    --csv-out /tmp/flydsl_msweep_8192x7168.csv

Emits CSV columns: impl,M,N,K,time_us,tflops,gbytes_s,config
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def _add_paths(flydsl_repo: str) -> None:
    # The flydsl wheel itself is expected on PYTHONPATH already (or installed).
    # The small_m kernel lives in the FlyDSL repo's `kernels` package and uses
    # relative imports, so the repo root must be importable as well.
    if flydsl_repo and flydsl_repo not in sys.path:
        sys.path.insert(0, flydsl_repo)


def _curated_configs(N: int, K: int):
    """A conservative, validated-by-construction config grid.

    The kernel re-validates every config and raises ValueError on anything it
    cannot build, so over-generating here is safe: invalid cells are skipped at
    compile time. We deliberately avoid the exotic wide tile_n values (160/192/
    224) that have been observed to fault on gfx950.
    """
    configs = []

    def add(**cfg):
        cfg.setdefault("N_TILE_REPEAT", 1)
        cfg.setdefault("PERSISTENT_N_TILES", 1)
        cfg.setdefault("WAVES_PER_EU_HINT", 0)
        cfg.setdefault("B_TO_LDS_UNROLL", 0)
        cfg.setdefault("B_TO_LDS", False)
        configs.append(cfg)

    # A focused grid (~56 cells) that still spans every small-M mechanism.
    # Single-threaded MLIR lowering makes each new config a few seconds to
    # build, so we keep the grid deliberately small but representative. split_k
    # matters a lot at tiny M (occupancy), so it is swept widest on the cheap
    # paths and trimmed on the LDS-heavy ones.

    # 1) Plain register-tile path (no B in LDS): many N-warps, one 16-row M tile.
    for tile_n in (64, 128, 256):
        for split_k in (1, 2, 4, 8):
            for bnw in (1, 2):
                add(TILE_N=tile_n, TILE_K=64, SPLIT_K=split_k, BLOCK_N_WARPS=bnw)

    # 2) N_TILE_REPEAT register reuse (FlyDSL analogue of our kNPerWarp):
    #    classic (BNW=1, TILE_N=64).
    for split_k in (1, 2, 4, 8):
        for nr in (2, 4):
            add(TILE_N=64, TILE_K=64, SPLIT_K=split_k, BLOCK_N_WARPS=1,
                N_TILE_REPEAT=nr)

    # 3) B-to-LDS async double-buffered path (the MFMA-heavy "production" path).
    for tile_n in (128, 256):
        for tile_k in (64, 128):
            for split_k in (1, 2):
                for bnw in (2, 4):
                    add(TILE_N=tile_n, TILE_K=tile_k, SPLIT_K=split_k,
                        BLOCK_N_WARPS=bnw, B_TO_LDS=True, B_TO_LDS_UNROLL=16)

    # 4) Persistent-N B-to-LDS path: stay on a small N group longer.
    for tile_n in (128, 256):
        for tile_k in (64, 128):
            for pn in (2, 4):
                add(TILE_N=tile_n, TILE_K=tile_k, SPLIT_K=1, BLOCK_N_WARPS=2,
                    B_TO_LDS=True, PERSISTENT_N_TILES=pn, B_TO_LDS_UNROLL=8)

    return configs


def _config_label(cfg: dict) -> str:
    parts = [f"tn{cfg['TILE_N']}", f"tk{cfg['TILE_K']}", f"spk{cfg['SPLIT_K']}",
             f"bnw{cfg['BLOCK_N_WARPS']}"]
    if cfg.get("N_TILE_REPEAT", 1) > 1:
        parts.append(f"nr{cfg['N_TILE_REPEAT']}")
    if cfg.get("PERSISTENT_N_TILES", 1) > 1:
        parts.append(f"pn{cfg['PERSISTENT_N_TILES']}")
    if cfg.get("B_TO_LDS", False):
        parts.append("bs")
        if cfg.get("B_TO_LDS_UNROLL", 0) > 0:
            parts.append(f"ur{cfg['B_TO_LDS_UNROLL']}")
    return "_".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flydsl-repo", default=os.path.expanduser("~/dev/FlyDSL"))
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--K", type=int, default=7168)
    ap.add_argument("--mmax", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--csv-out", default="/tmp/flydsl_msweep.csv")
    ap.add_argument("--max-configs", type=int, default=0,
                    help="0 = no cap; otherwise keep an evenly-sampled subset.")
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    _add_paths(args.flydsl_repo)

    import torch
    import flydsl.expr as fx
    from flydsl.runtime.device import get_rocm_arch
    from kernels.small_m_hgemm import compile_small_m_hgemm_kernel
    from kernels.tensor_shim import _run_compiled

    arch = get_rocm_arch()
    dev = torch.device("cuda")
    N, K, Mmax = args.N, args.K, args.mmax
    print(f"# FlyDSL small_m M-sweep: arch={arch} N={N} K={K} Mmax={Mmax} "
          f"warmup={args.warmup} repeat={args.repeat}", file=sys.stderr)

    # NOTE: the small_m kernel builds its A/B/C buffer resources via
    # GTensor -> create_buffer_resource -> BufferResourceDescriptor.from_memref,
    # which requires a *memref* operand. So torch tensors are passed directly
    # (memref adaptor); they must NOT be wrapped as bare pointers.

    # Persistent buffers sized for the largest M; smaller M use leading rows.
    A = torch.randn((Mmax, K), dtype=torch.bfloat16, device=dev)
    B = torch.randn((N, K), dtype=torch.bfloat16, device=dev)
    C = torch.zeros((Mmax, N), dtype=torch.bfloat16, device=dev)
    BIAS = torch.zeros((N,), dtype=torch.bfloat16, device=dev)
    # Generous split-K semaphore/signal pools (bm*bn counters; bm=1 here).
    sem = torch.zeros((8192,), dtype=torch.int32, device=dev)
    sig = torch.zeros((8192,), dtype=torch.int32, device=dev)

    configs = _curated_configs(N, K)
    if args.max_configs and len(configs) > args.max_configs:
        stride = len(configs) / args.max_configs
        configs = [configs[int(i * stride)] for i in range(args.max_configs)]
    print(f"# candidate configs: {len(configs)}", file=sys.stderr)

    # Compile each config once (reused across all M). Skip ones the kernel
    # refuses to build for this (N, K).
    compiled = []  # (label, cfg, kernel)
    for cfg in configs:
        label = _config_label(cfg)
        try:
            kernel = compile_small_m_hgemm_kernel("bf16", N, K, **cfg)
        except Exception as e:  # noqa: BLE001 - report and skip invalid cells
            print(f"#   skip {label}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        compiled.append((label, cfg, kernel))
    print(f"# compiled configs: {len(compiled)}", file=sys.stderr)
    if not compiled:
        print("# no FlyDSL configs compiled; aborting", file=sys.stderr)
        return 1

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    def run(kernel, m):
        # The split-K kernel zeroes C itself (zero_c_tile on first arrival) and
        # resets its semaphore/signal at the last departure, so repeated timed
        # launches need no host-side re-init. Tensors are passed directly.
        _run_compiled(kernel, C, A, B, BIAS,
                      int(m), sem, sig, fx.Stream(torch.cuda.current_stream()))

    def time_one(kernel, m):
        for _ in range(args.warmup):
            run(kernel, m)
        torch.cuda.synchronize()
        start_evt.record()
        for _ in range(args.repeat):
            run(kernel, m)
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) * 1000.0 / args.repeat  # us

    # One-time correctness check on the first compiled config at M=Mmax.
    if args.verify:
        label, cfg, kernel = compiled[0]
        run(kernel, Mmax)
        torch.cuda.synchronize()
        ref = (A.float() @ B.float().t())
        got = C.float()
        denom = ref.abs().mean().clamp_min(1e-6)
        rel = (got - ref).abs().mean() / denom
        print(f"# verify config={label} M={Mmax}: mean_rel_err={rel.item():.4e}",
              file=sys.stderr)
        if rel.item() > 5e-2:
            print(f"# WARNING: high rel err {rel.item():.3e}; check harness wiring",
                  file=sys.stderr)

    rows = []
    best_per_m = {}
    for m in range(1, Mmax + 1):
        best = None  # (t_us, label, cfg)
        for label, cfg, kernel in compiled:
            try:
                t_us = time_one(kernel, m)
            except Exception as e:  # noqa: BLE001
                print(f"#   M={m} {label}: run failed {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue
            if best is None or t_us < best[0]:
                best = (t_us, label, cfg)
        if best is None:
            print(f"# M={m}: no working config", file=sys.stderr)
            continue
        t_us, label, cfg = best
        tflops = 2.0 * m * N * K / (t_us * 1e-6) / 1e12
        gbps = (m * K + N * K + m * N) * 2 / (t_us * 1e-6) / 1e9
        best_per_m[m] = best
        rows.append(("flydsl_small_m", m, N, K, t_us, tflops, gbps, label))
        print(f"M={m:2d}  best={t_us:8.2f}us  {tflops:6.2f} TF/s  {gbps:7.1f} GB/s  "
              f"[{label}]", file=sys.stderr)

    with open(args.csv_out, "w") as f:
        f.write("impl,M,N,K,time_us,tflops,gbytes_s,config\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.3f},{r[5]:.3f},"
                    f"{r[6]:.2f},{r[7]}\n")
    print(f"# wrote {len(rows)} rows -> {args.csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
