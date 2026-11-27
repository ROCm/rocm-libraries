# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Ragged grouped GEMM workflow harness for gfx950 (CDNA4) -- test, time, validate.

Pure ragged grouped bf16 GEMM with variable M per expert. No gather/scatter/routing.
Takes m_sizes[E] as input, builds per-tile schedule, compares against a numpy
fp32 reference.

Shape, tile geometry, and every optimization lever are command-line flags (same
convention as ``grouped_gemm_hip.py``); defaults are the measured sweet spot for
M_total=524288, N=1024, K=512, E=64 on gfx950. Use ``--no-<flag>`` to disable a
boolean lever, e.g. ``--no-swz``.

Run (``HIP_VISIBLE_DEVICES`` picks the GPU):
    PYTHONPATH=python python3 \
        python/rocke/examples/gfx950/grouped_gemm/ragged_gemm_hip.py [--brrr ...]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.numeric import storage
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.launcher import (
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)


def make_group_sizes(m_total: int, e: int, dist: str, seed: int = 0) -> np.ndarray:
    """Generate per-expert row counts. Returns an int64 array of length E."""
    if dist == "equal":
        base = m_total // e
        s = [base] * e
        s[-1] += m_total - base * e
        return np.array(s, dtype=np.int64)
    if dist == "bimodal":
        s = [4096 if i % 2 == 0 else 12288 for i in range(e)]
        rem = m_total - sum(s)
        s[-1] += rem
        return np.array(s, dtype=np.int64)
    # random ragged
    w = np.random.default_rng(seed).random(e) + 0.3
    s = (w / w.sum() * m_total).astype(np.int64)
    s[-1] += m_total - s.sum()
    return s


def build_sched(
    sizes: np.ndarray, block_m: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Build per-tile schedule from m_sizes.

    Returns (expert_ids, m_offsets, m_valid, num_tiles, m_starts) where, for each
    tile: expert_ids = owning expert, m_offsets = absolute A/C row start,
    m_valid = valid rows in the tile (<= block_m; the tail tile of an expert may
    be partial). m_starts is the per-expert row start (for the reference). The
    arrays are int32.
    """
    E = sizes.shape[0]
    n_e = (sizes + block_m - 1) // block_m  # tiles per expert
    total_tiles = int(n_e.sum())
    tile_start = np.cumsum(n_e) - n_e
    tile_expert = np.repeat(np.arange(E), n_e)
    within = np.arange(total_tiles) - tile_start[tile_expert]
    m_starts_per_expert = np.cumsum(sizes) - sizes
    tile_moff = m_starts_per_expert[tile_expert] + within * block_m
    # valid rows in this tile = min(block_m, expert_size - within*block_m)
    remaining = sizes[tile_expert] - within * block_m
    tile_mvalid = np.minimum(remaining, block_m)
    return (
        tile_expert.astype(np.int32),
        tile_moff.astype(np.int32),
        tile_mvalid.astype(np.int32),
        total_tiles,
        m_starts_per_expert.astype(np.int32),
    )


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="gfx950 pure ragged grouped bf16 GEMM harness"
    )
    # Problem shape.
    p.add_argument("--m-total", type=int, default=524288)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--e", type=int, default=64)
    p.add_argument(
        "--dist",
        choices=("equal", "ragged", "bimodal"),
        default="ragged",
        help="per-expert row-count distribution",
    )
    # Tile / warp geometry + chiplet knobs.
    p.add_argument("--tm", type=int, default=256)
    p.add_argument("--tn", type=int, default=256)
    p.add_argument("--tk", type=int, default=64)
    p.add_argument("--wm", type=int, default=2)
    p.add_argument("--wn", type=int, default=4)
    p.add_argument("--xcds", type=int, default=8)
    p.add_argument("--wgm", type=int, default=8)
    # Boolean levers (defaults = measured production sweet spot). Use --no-<flag>
    # to disable, e.g. --no-swz.
    B = argparse.BooleanOptionalAction
    p.add_argument(
        "--hoist",
        action=B,
        default=False,
        help="hoist loop-invariant A/B-load offsets (off by default here: the "
        "VGPRs it costs are worth more than the address math it saves)",
    )
    p.add_argument(
        "--deeppipe",
        action=B,
        default=False,
        help="deep operand prefetch pipeline (off by default here, same reason)",
    )
    p.add_argument("--epifuse", action=B, default=True, help="epilogue store fusion")
    p.add_argument("--swz", action=B, default=True, help="st_16x32 LDS XOR swizzle")
    p.add_argument("--asm", action=B, default=True, help="inline-asm ds_read")
    p.add_argument("--chip", action=B, default=True, help="chiplet super-tile remap")
    p.add_argument("--prio", action=B, default=True, help="s_setprio around MFMAs")
    p.add_argument("--pin", action=B, default=True, help="pin wave-uniform indices")
    p.add_argument(
        "--brrr", action=B, default=False, help="NN weight layout (B=[E,K,N])"
    )
    p.add_argument(
        "--cshuf", action=B, default=False, help="C-shuffle wide-store epilogue"
    )
    p.add_argument(
        "--verify", action=B, default=True, help="compare against the fp32 reference"
    )
    return p.parse_args(argv)


def _main(argv=None) -> int:
    from rocke.instances.gfx950.ragged_gemm import (
        RaggedGemmSpec,
        build_ragged_gemm,
        ragged_gemm_signature,
    )

    args = _parse_args(argv)
    N, K, E, M_total, dist = args.n, args.k, args.e, args.m_total, args.dist

    rng = np.random.default_rng(0)

    # Generate ragged sizes
    m_sizes = make_group_sizes(M_total, E, dist)
    print(f"[rgemm] N={N} K={K} E={E} M_total={M_total} dist={dist}")
    print(
        f"[rgemm] m_sizes: min={m_sizes.min()} max={m_sizes.max()} mean={m_sizes.mean():.1f}"
    )

    spec = RaggedGemmSpec(
        N=N,
        K=K,
        E=E,
        TM=args.tm,
        TN=args.tn,
        TK=args.tk,
        WM=args.wm,
        WN=args.wn,
        hoist=args.hoist,
        deeppipe=args.deeppipe,
        epifuse=args.epifuse,
        swz=args.swz,
        asm_reads=args.asm,
        chiplet=args.chip,
        chiplet_xcds=args.xcds,
        chiplet_wgm=args.wgm,
        prio=args.prio,
        pin=args.pin,
        b_rrr=args.brrr,
        cshuf=args.cshuf,
    )

    TM = spec.TM
    # A is padded by TM rows so the last tile's over-read stays mapped (see the
    # kernel's dtl_load_a note). The extra rows are never stored.
    X_host = storage.encode(rng.standard_normal((M_total + TM, K)), "bf16")
    B_host = storage.encode(
        rng.standard_normal((E, K if spec.b_rrr else N, N if spec.b_rrr else K)),
        "bf16",
    )
    X, B = DeviceArray(X_host, "bf16"), DeviceArray(B_host, "bf16")
    # C has one extra sink row (index M_total): padded tile rows write there.
    C = DeviceArray.zeros((M_total + 1, N), "bf16")

    # Build schedule
    expert_ids, m_offsets, m_valid, num_m_tiles, m_starts = build_sched(m_sizes, TM)
    expert_ids, m_offsets, m_valid = (
        DeviceArray(a) for a in (expert_ids, m_offsets, m_valid)
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
    synchronize_and_release()

    if args.verify:
        # correctness (compare valid rows only; row M_total is the padding sink)
        Xf, Bf = storage.decode(X_host, "bf16"), storage.decode(B_host, "bf16")
        ref = np.zeros((M_total, N), dtype=np.float32)
        for e in range(E):
            start, sz = int(m_starts[e]), int(m_sizes[e])
            if sz > 0:
                B_e = Bf[e] if spec.b_rrr else Bf[e].T
                ref[start : start + sz] = Xf[start : start + sz] @ B_e
        out = storage.decode(C.values()[:M_total], "bf16")
        diff = np.abs(out - ref)
        denom = np.abs(ref) + 1e-3
        max_abs = float(diff.max())
        rel = float((diff / denom).max())
        status = "PASS" if rel < 0.02 else "FAIL"
        print(f"[rgemm] grid={grid} max_abs={max_abs:.4f} rel={rel:.4f} {status}")
        if status == "FAIL":
            return 1
    else:
        print(f"[rgemm] grid={grid} verification SKIPPED")

    # timing: ten repeats of 200 launches each (ms per launch)
    times = sorted(time_launches(call, warmup=0, iters=200) for _ in range(10))
    med_ms = times[len(times) // 2]
    flops = 2 * M_total * N * K
    tflops = flops / (med_ms * 1e-3) / 1e12
    peak_tflops = flops / (times[0] * 1e-3) / 1e12
    print(f"[rgemm] med={tflops:.1f} TF  peak={peak_tflops:.1f} TF")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
