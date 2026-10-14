#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side builder for the chunkwise KDA per-chunk tile kernel
(``kernels/gfx942/kda_chunkwise.py``).

Owns the host path: spec construction, compilation, ABI signature, workspace
allocation and launch -- plus a torch parity check of all six output tiles and a
throughput report.

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

import torch  # noqa: E402

from kernels.gfx942.kda_chunkwise import (  # noqa: E402
    KdaChunkPrepSpec,
    KdaTileSpec,
    build_kda_chunk_prep,
    is_valid_spec,
    kda_chunk_prep_grid,
    kda_chunk_prep_signature,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches  # noqa: E402

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


def alloc_tiles(num_tiles: int, spec: KdaChunkPrepSpec, device="cuda"):
    """The six per-chunk tiles, carved from one pool.

    Six separate allocations show up on the host critical path at short
    sequences, and the tiles are always produced and consumed together.
    """
    C, DK = spec.tile.chunk, spec.head_k
    shapes = [
        ("a", (num_tiles, C, C)),
        ("gk", (num_tiles, C, DK)),
        ("gq", (num_tiles, C, DK)),
        ("aqk", (num_tiles, C, C)),
        ("kt", (num_tiles, DK, C)),
    ]
    counts = [s[0] * s[1] * s[2] for _, s in shapes]
    pool = torch.empty(sum(counts), dtype=torch.bfloat16, device=device)
    ws, off = {}, 0
    for (name, shape), cnt in zip(shapes, counts):
        ws[name] = pool[off : off + cnt].view(*shape)
        off += cnt
    ws["dec"] = torch.empty(num_tiles, DK, dtype=torch.float32, device=device)
    ws["_pool"] = pool
    return ws


def run_prep(spec, q, k, g, beta, ws, scale, stream=None):
    """Launch the prep kernel over ``q/k/g/beta`` already packed by chunk."""
    num_tiles = q.shape[0]
    launcher = make_launcher(spec)
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
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
# reference + parity
# ---------------------------------------------------------------------


def ref_prep_tiles(q, k, g, beta, scale, nt_block=32):
    """float64 torch oracle for the six tiles. q/k/g: [NT, C, DK], beta: [NT, C].

    ``Akk`` and ``Aqk`` are formed from the pairwise exponent *difference*
    ``Gcum_i - Gcum_j`` rather than the ratio ``Gamma_i / Gamma_j``, so the
    oracle never materializes the overflowing intermediate the kernel is
    designed to avoid -- and it does so by a different route (explicit
    ``C x C x DK`` outer product, no midpoint factoring), which is what makes it
    an independent check of the factoring rather than a restatement of it.

    Only the retained triangle is meaningful: for ``i >= j`` the exponent is
    already ``<= 0``, so clamping it there is exact and keeps the discarded
    upper triangle from going to inf. Blocked over NT to bound the
    ``C x C x DK`` temporary.
    """
    NT, C, _ = q.shape
    ar = torch.arange(C, device=q.device)
    lower = (ar[:, None] > ar[None, :]).unsqueeze(0)
    lower_eq = (ar[:, None] >= ar[None, :]).unsqueeze(0)
    zero = torch.zeros((), device=q.device, dtype=torch.float64)

    out = {n: [] for n in ("a", "gk", "gq", "aqk", "kt", "dec")}
    for s in range(0, NT, nt_block):
        e = min(s + nt_block, NT)
        qb, kb = q[s:e].double(), k[s:e].double()
        gcum = g[s:e].double().cumsum(dim=1)
        gl = gcum[:, -1:, :]
        gam = gcum.exp()
        bb = beta[s:e].double()

        # [n, i, j, d] pairwise exponent, clamped where it is discarded anyway.
        dif = (gcum[:, :, None, :] - gcum[:, None, :, :]).clamp(max=0.0).exp()
        akk = (kb[:, :, None, :] * kb[:, None, :, :] * dif).sum(-1)
        # ``dif`` already carries the whole Gamma_i / Gamma_j ratio, so the
        # query side enters as bare q * scale -- folding Gamma_i in via ``gq``
        # here would count the decay twice.
        aqk = (qb[:, :, None, :] * scale * kb[:, None, :, :] * dif).sum(-1)
        gq = qb * gam * scale

        tprime = torch.where(lower, bb[:, :, None] * akk, zero)
        eye = torch.eye(C, device=q.device, dtype=torch.float64).expand_as(tprime)
        out["a"].append(
            torch.linalg.solve_triangular(
                eye + tprime, torch.diag_embed(bb), upper=False
            )
        )
        out["aqk"].append(torch.where(lower_eq, aqk, zero))
        out["gk"].append(kb * gam)
        out["gq"].append(gq)
        out["kt"].append((kb * (gl - gcum).exp()).transpose(1, 2))
        out["dec"].append(gl.squeeze(1).exp())
    return {n: torch.cat(v, 0) for n, v in out.items()}


def make_inputs(num_tiles, C, DK, gate_low=-0.5, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    kw = dict(device=device, generator=gen)
    q = torch.nn.functional.normalize(
        torch.randn(num_tiles, C, DK, dtype=torch.float32, **kw), dim=-1
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(num_tiles, C, DK, dtype=torch.float32, **kw), dim=-1
    ).to(torch.bfloat16)
    # Per-channel log decay in the reference range (gate_lower_bound = -5), which is
    # what stresses the factored C x C construction.
    g = gate_low * torch.rand(num_tiles, C, DK, dtype=torch.float32, **kw)
    beta = torch.rand(num_tiles, C, dtype=torch.float32, **kw)
    return q, k, g, beta


def check(spec, num_tiles, gate_low=-0.5, tol=2e-2, verbose=True):
    C, DK = spec.tile.chunk, spec.head_k
    q, k, g, beta = make_inputs(num_tiles, C, DK, gate_low)
    scale = DK**-0.5
    ws = alloc_tiles(num_tiles, spec)
    run_prep(
        spec,
        q.reshape(num_tiles, -1),
        k.reshape(num_tiles, -1),
        g.reshape(num_tiles, -1),
        beta,
        ws,
        scale,
    )
    torch.cuda.synchronize()

    ref = ref_prep_tiles(q.float(), k.float(), g, beta, scale)
    pairs = [(n, ws[n].float(), ref[n].float()) for n in ref]
    # A's diagonal is beta and dominates its magnitude, so scoring the whole
    # tile against its own absmax would hide the off-diagonal terms -- which are
    # the part that actually came out of the solve. Score them separately.
    pairs.append(("a_low", ws["a"].float().tril(-1), ref["a"].float().tril(-1)))

    worst, worst_name = 0.0, ""
    for name, got, exp in pairs:
        d = (got - exp).abs()
        denom = exp.abs().max().clamp_min(1e-30)
        rel = (d.max() / denom).item()
        if verbose:
            print(
                f"  {name:<5} max_abs={d.max().item():.3e} rel={rel:.3e} "
                f"ref_absmax={exp.abs().max().item():.3e} "
                f"finite={torch.isfinite(got).all().item()}"
            )
        if rel > worst:
            worst, worst_name = rel, name
    ok = worst <= tol
    print(
        f"  -> worst rel {worst:.3e} on {worst_name!r}  tol {tol:.1e}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def bench(spec, num_tiles, warmup=15, iters=50):
    C, DK = spec.tile.chunk, spec.head_k
    q, k, g, beta = make_inputs(num_tiles, C, DK)
    ws = alloc_tiles(num_tiles, spec)
    args = (
        spec,
        q.reshape(num_tiles, -1),
        k.reshape(num_tiles, -1),
        g.reshape(num_tiles, -1),
        beta,
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
