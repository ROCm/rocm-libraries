#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side builder for the chunkwise KDA per-chunk tile kernel
(``kernels/gfx942/kda_chunkwise.py``).

Owns the host path: spec construction, compilation, ABI signature, workspace
allocation and launch -- plus a float64 parity check of all six output tiles
(:func:`rocke.numeric.gated_delta.chunkwise_tiles`) and a throughput report.

The six tiles this kernel emits are exactly the state-independent part of the
chunkwise KDA body, so it can be validated (and benchmarked) on its own, ahead of
the serial state scan that consumes them.

Usage:
    python kda_chunk_prep.py                    # parity + bench, default shapes
    python kda_chunk_prep.py --shapes 8x16x2048
    python kda_chunk_prep.py --chunk 16 --no-check
"""

import argparse
import os
import sys

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches  # noqa: E402
from rocke.runtime.device_array import DeviceArray  # noqa: E402

from builders.common.kda_host import ChunkTiles, check_prep, tile_inputs  # noqa: E402
from kernels.gfx942.kda_chunkwise import (  # noqa: E402
    KdaChunkPrepSpec,
    KdaTileSpec,
    build_kda_chunk_prep,
    is_valid_spec,
    kda_chunk_prep_grid,
    kda_chunk_prep_signature,
)

_LAUNCHER_CACHE = {}


def make_launcher(spec: KdaChunkPrepSpec) -> KernelLauncher:
    key = spec.kernel_name()
    if key not in _LAUNCHER_CACHE:
        ok, why = is_valid_spec(spec)
        if not ok:
            raise ValueError(f"unsupported spec: {why}")
        art = compile_kernel(
            build_kda_chunk_prep(spec),
            arch="gfx942",
            backend="python",
            capture_ir_text=False,
        )
        _LAUNCHER_CACHE[key] = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=kda_chunk_prep_signature(spec),
        )
    return _LAUNCHER_CACHE[key]


def alloc_tiles(num_tiles: int, spec: KdaChunkPrepSpec) -> ChunkTiles:
    """The six per-chunk tiles ``spec``'s prep kernel writes for ``num_tiles``
    chunks (:class:`builders.common.kda_host.ChunkTiles`)."""
    return ChunkTiles(num_tiles, spec.tile.chunk, spec.head_k)


def run_prep(spec, q, k, g, beta, ws, scale, stream=0):
    """Launch the prep kernel over device ``q/k/g/beta`` already packed by
    chunk, on ``stream`` (0: the default stream)."""
    num_tiles = q.shape[0]
    launcher = make_launcher(spec)
    cfg = LaunchConfig(
        grid=kda_chunk_prep_grid(spec, num_tiles),
        block=(spec.tile.block_size, 1, 1),
        stream=stream,
    )
    launcher(
        {
            "q_ptr": q,
            "k_ptr": k,
            "g_ptr": g,
            "beta_ptr": beta,
            "a_ptr": ws["a"],
            "gk_ptr": ws["gk"],
            "gq_ptr": ws["gq"],
            "aqk_ptr": ws["aqk"],
            "kt_ptr": ws["kt"],
            "dec_ptr": ws["dec"],
            "scale": float(scale),
        },
        config=cfg,
    )


# ---------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------


def check(spec, num_tiles, gate_low=-0.5, tol=2e-2, verbose=True):
    """All six tiles of ``num_tiles`` chunks against the float64 reference
    (:func:`builders.common.kda_host.check_prep`)."""
    return check_prep(
        spec, run_prep, num_tiles, gate_low=gate_low, tol=tol, verbose=verbose
    )


def bench(spec, num_tiles, warmup=15, iters=50):
    C, DK = spec.tile.chunk, spec.head_k
    q, k, g, beta = tile_inputs(num_tiles, C, DK)
    ws = alloc_tiles(num_tiles, spec)
    args = (
        spec,
        DeviceArray(q.reshape(num_tiles, -1), "bf16"),
        DeviceArray(k.reshape(num_tiles, -1), "bf16"),
        DeviceArray(g.reshape(num_tiles, -1)),
        DeviceArray(beta),
        ws,
        DK**-0.5,
    )
    # time_launches runs the loop under no_fence and owns the drain. Timing
    # fenced launches instead would charge every iteration a host stream sync
    # (LaunchConfig.fence defaults True), which is not how the kernel runs in a
    # pipeline: the scan consumes the tiles on the same stream, so FIFO order
    # already covers the dependency. At short sequences the sync is a large
    # fraction of the measurement.
    ms = time_launches(lambda: run_prep(*args), warmup=warmup, iters=iters)

    tile_el = 2 * C * C + 2 * C * DK + DK * C
    rd = num_tiles * (2 * C * DK * 2 + C * DK * 4 + C * 4)
    wr = num_tiles * (tile_el * 2 + DK * 4)
    gbps = (rd + wr) / (ms * 1e-3) / 2**30
    print(f"  {ms:.4f} ms  {gbps:.1f} GiB/s  ({num_tiles} chunks)")
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes",
        default="1x8x2048,8x8x1024,8x16x2048",
        help="BxHxT triples; chunk count is B*H*T/chunk",
    )
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--pad-dk", type=int, default=8)
    ap.add_argument("--pad-c", type=int, default=4)
    ap.add_argument("--wpe", type=int, default=0)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    spec = KdaChunkPrepSpec(
        head_k=args.dk,
        head_v=args.dv,
        tile=KdaTileSpec(
            chunk=args.chunk,
            block_size=2 * args.dk,
            pad_dk=args.pad_dk,
            pad_c=args.pad_c,
            waves_per_eu=args.wpe,
        ),
    )
    ok, why = is_valid_spec(spec)
    print(f"{spec.kernel_name()}  lds={spec.lds_bytes()} B  valid={ok} {why}")
    if not ok:
        return 1

    failed = False
    if not args.no_check:
        # Sweep the gate range: near-zero is the typical regime, -5 is the
        # reference gate lower bound and saturates the factored exponents.
        for gl in (-0.1, -0.5, -2.0, -5.0):
            print(f"parity (128 chunks, gate in [{gl}, 0]):")
            failed |= not check(spec, 128, gate_low=gl)

    for s in args.shapes.split(","):
        B, H, T = (int(x) for x in s.split("x"))
        nt = B * H * (T // args.chunk)
        print(f"bench B={B} H={H} T={T}:")
        bench(spec, nt, iters=args.iters)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
