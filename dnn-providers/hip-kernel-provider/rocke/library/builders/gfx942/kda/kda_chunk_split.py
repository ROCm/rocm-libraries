#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host builder for the split-path chunkwise KDA forward on gfx942.

Two kernels: :func:`build_kda_chunk_prep` builds the six per-chunk tiles, one
workgroup per chunk and fully parallel over the sequence, then
:func:`build_kda_chunk_scan` walks each (batch, head)'s chunks in order,
staging that chunk's tiles from HBM into LDS and running the state recurrence.

The alternative is the fused kernel in ``kda_chunk_fused.py``, which keeps the
tiles in LDS and never writes them out. This path instead pays the tile round
trip so the two algorithmic phases can be scheduled independently.

Correctness is checked end to end against the same token-serial float64 oracle
the fused builder uses, so the two paths are held to one contract.

Run directly for a parity sweep plus a benchmark::

    python kda_chunk_split.py                 # parity + bench
    python kda_chunk_split.py --no-check      # bench only
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from kernels.gfx942.kda_chunkwise import (  # noqa: E402
    KdaChunkPrepSpec,
    KdaChunkScanSpec,
    KdaTileSpec,
    build_kda_chunk_scan,
    is_valid_scan_spec,
    is_valid_spec,
    kda_chunk_scan_grid,
    kda_chunk_scan_signature,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches  # noqa: E402

import kda_chunk_prep as prep_mod  # noqa: E402
from kda_chunk_fused import (  # noqa: E402
    _pack_initial_state,
    _pack_v_partitions,
    _unpack_outputs,
    make_inputs,
    ref_token_serial,
)

_LAUNCHER_CACHE: dict = {}


def make_launcher(spec: KdaChunkScanSpec) -> KernelLauncher:
    key = spec.kernel_name()
    if key not in _LAUNCHER_CACHE:
        ok, why = is_valid_scan_spec(spec)
        if not ok:
            raise ValueError(f"unsupported spec: {why}")
        art = compile_kernel(
            build_kda_chunk_scan(spec),
            arch="gfx942",
            backend="python",
            capture_ir_text=False,
        )
        _LAUNCHER_CACHE[key] = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=kda_chunk_scan_signature(spec),
        )
    return _LAUNCHER_CACHE[key]


def prep_spec_of(spec: KdaChunkScanSpec) -> KdaChunkPrepSpec:
    """The tile builder that produces this scan's inputs.

    Both halves are derived from one scan spec so the tile layouts cannot drift
    apart: the scan's staging copies assume exactly the flat per-chunk layouts
    the prep kernel's global sink writes.
    """
    return KdaChunkPrepSpec(
        head_k=spec.head_k,
        head_v=spec.head_v,
        dtype=spec.dtype,
        tile=spec.tile,
    )


def run_scan(spec, ws, v, o, ht, bh, nc, h0=None, stream=None):
    """Launch the scan over tiles already materialized by the prep kernel."""
    launcher = make_launcher(spec)
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    cfg = LaunchConfig(
        grid=kda_chunk_scan_grid(spec, bh),
        block=(spec.tile.block_size, 1, 1),
        stream=stream,
    )
    launcher(
        {
            "a_ptr": ws["a"],
            "gk_ptr": ws["gk"],
            "gq_ptr": ws["gq"],
            "aqk_ptr": ws["aqk"],
            "kt_ptr": ws["kt"],
            "dec_ptr": ws["dec"],
            "v_ptr": v,
            "o_ptr": o,
            # unread when the kernel was built with has_initial_state=False
            "h0_ptr": ht if h0 is None else h0,
            "ht_ptr": ht,
            "nc": int(nc),
        },
        config=cfg,
    )


def run_split(spec, q, k, g, beta, v, o, ws, ht, scale, bh, nc, h0=None):
    """Both kernels, back to back on one stream.

    No fence between them: they run in FIFO order on the same stream, so the
    scan already sees the tiles the prep kernel wrote.
    """
    prep_mod.run_prep(prep_spec_of(spec), q, k, g, beta, ws, scale)
    run_scan(spec, ws, v, o, ht, bh, nc, h0=h0)


# ---------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------


def launch_packed(spec, q, k, v, g, beta, h0=None):
    """Run dense ``[B,H,T,D]`` inputs through gfx942 V partitions."""
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    C = spec.tile.chunk
    qf, kf, vf, gf, bf, BH, NC, parts = _pack_v_partitions(spec, q, k, v, g, beta)
    nt = BH * parts * NC
    ws = prep_mod.alloc_tiles(nt, prep_spec_of(spec))
    o = torch.empty(nt, C * spec.head_v, dtype=torch.bfloat16, device=q.device)
    ht = torch.zeros(BH * parts, spec.head_v, DK, dtype=torch.float32, device=q.device)
    h0t = _pack_initial_state(spec, h0, parts)
    run_split(
        spec,
        qf,
        kf,
        gf,
        bf,
        vf,
        o,
        ws,
        ht,
        DK**-0.5,
        BH * parts,
        NC,
        h0=h0t,
    )
    return _unpack_outputs(spec, o, ht, B, H, T, DV, NC, parts)


def check(
    spec,
    B,
    H,
    T,
    gate_low=-0.5,
    tol=3e-2,
    with_h0=False,
    verbose=True,
    logical_head_v=128,
):
    DK, DV = spec.head_k, logical_head_v
    q, k, v, g, beta = make_inputs(B, H, T, DK, DV, gate_low)
    h0 = None
    if with_h0:
        gen = torch.Generator(device="cuda").manual_seed(7)
        h0 = 0.1 * torch.randn(
            B, H, DK, DV, dtype=torch.float32, device="cuda", generator=gen
        )
    o_got, ht_got = launch_packed(spec, q, k, v, g, beta, h0=h0)
    torch.cuda.synchronize()
    o_ref, s_ref = ref_token_serial(q, k, v, g, beta, DK**-0.5, h0=h0)

    worst = 0.0
    for name, got, ref in (("out", o_got, o_ref), ("final_state", ht_got, s_ref)):
        d = (got.double() - ref).abs().max().item()
        den = max(ref.abs().max().item(), 1e-30)
        rel = d / den
        worst = max(worst, rel)
        if verbose:
            print(
                f"  {name:12s} max_abs={d:.3e} rel={rel:.3e} "
                f"ref_absmax={den:.3e} finite={bool(torch.isfinite(got).all())}"
            )
    verdict = "PASS" if worst <= tol else "FAIL"
    if verbose:
        print(f"  -> worst rel {worst:.3e}  tol {tol:.1e}  {verdict}")
    return worst


def bench(
    spec,
    B,
    H,
    T,
    warmup=10,
    iters=30,
    split=True,
    logical_head_v=128,
):
    """Time the pair, and each half alone, on one shape.

    The halves are reported too because they answer different questions: the
    prep time is bandwidth against a known roofline, while the scan time is a
    serial chain whose only lever is occupancy.
    """
    DK, DV = spec.head_k, logical_head_v
    q, k, v, g, beta = make_inputs(B, H, T, DK, DV)
    C = spec.tile.chunk
    qf, kf, vf, gf, bf, BH, NC, parts = _pack_v_partitions(spec, q, k, v, g, beta)
    nt = BH * parts * NC
    ws = prep_mod.alloc_tiles(nt, prep_spec_of(spec))
    o = torch.empty(nt, C * spec.head_v, dtype=torch.bfloat16, device="cuda")
    ht = torch.zeros(BH * parts, spec.head_v, DK, dtype=torch.float32, device="cuda")
    scale = DK**-0.5

    def do_prep():
        prep_mod.run_prep(prep_spec_of(spec), qf, kf, gf, bf, ws, scale)

    def do_scan():
        run_scan(spec, ws, vf, o, ht, BH * parts, NC)

    def do_both():
        do_prep()
        do_scan()

    ms = time_launches(do_both, warmup=warmup, iters=iters)
    # Essential traffic only: q/k/v/g/beta in, o out. The tile round trip this
    # path adds on purpose is excluded, so the number is comparable to the
    # fused kernel's and to FlyDSL's.
    rd = nt * (2 * C * DK * 2 + C * DV * 2 + C * DK * 4 + C * 4)
    wr = nt * C * DV * 2
    gbps = (rd + wr) / (ms * 1e-3) / 2**30
    print(f"  {ms:.4f} ms  {gbps:.1f} GiB/s  (B={B} H={H} T={T}, {nt} chunks)")
    if split:
        p = time_launches(do_prep, warmup=warmup, iters=iters)
        s = time_launches(do_scan, warmup=warmup, iters=iters)
        print(f"    prep {p:.4f} ms   scan {s:.4f} ms   sum {p + s:.4f} ms")
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="8x8x1024,8x16x2048,32x16x2048")
    ap.add_argument("--check-shape", default="2x4x256")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--v-splits", type=int, default=2)
    ap.add_argument("--sb", type=int, default=8)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    spec = KdaChunkScanSpec(
        head_k=args.dk,
        head_v=args.dv // args.v_splits,
        tile=KdaTileSpec(
            chunk=args.chunk,
            block_size=256,
            tile_atom_m=16,
            solve_block=args.sb,
            scan_atom_m=16,
        ),
    )
    ok, why = is_valid_scan_spec(spec)
    print(f"{spec.kernel_name()}  lds={spec.lds_bytes()} B  valid={ok} {why}")
    if not ok:
        return 1
    pspec = prep_spec_of(spec)
    ok, why = is_valid_spec(pspec)
    print(f"{pspec.kernel_name()}  lds={pspec.lds_bytes()} B  valid={ok} {why}")
    if not ok:
        return 1

    worst = 0.0
    if not args.no_check:
        B, H, T = (int(x) for x in args.check_shape.split("x"))
        for gate_low in (-0.1, -0.5, -2.0, -5.0):
            print(f"parity B={B} H={H} T={T} gate in [{gate_low}, 0]:")
            worst = max(
                worst,
                check(
                    spec,
                    B,
                    H,
                    T,
                    gate_low=gate_low,
                    logical_head_v=args.dv,
                ),
            )

    for s in args.shapes.split(","):
        B, H, T = (int(x) for x in s.split("x"))
        bench(spec, B, H, T, iters=args.iters, logical_head_v=args.dv)

    return 0 if worst <= 3e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
