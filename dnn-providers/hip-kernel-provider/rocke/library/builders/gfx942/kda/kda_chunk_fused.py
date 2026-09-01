# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host builder for the fused chunkwise KDA forward on gfx942.

Builds, validates and benchmarks :func:`build_kda_chunk_fused`: one workgroup
per (batch, head) walks that head's chunks in order, so the six per-chunk tiles
are produced and consumed in LDS and the only HBM traffic is q/k/g/beta in and
o (plus the final state) out.

Correctness is checked against a token-serial float64 oracle, which shares no
algebra with the kernel: the kernel evaluates the chunkwise factorization,
while the oracle just walks the gated delta rule one token at a time. Agreement
therefore tests the factorization itself, not only its implementation.

Run directly for a parity sweep plus a benchmark, or emit HSACO +
manifest for ``rocke.run_manifest`` / ``benchmark.remote_test``::

    python kda_chunk_fused.py                 # parity + bench
    python kda_chunk_fused.py --no-check      # bench only
    python -m builders.gfx942.kda.kda_chunk_fused \\
        --no-verify --output-dir /tmp/kda --m 2 --n 4 --k 256
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
    KdaChunkFusedSpec,
    KdaTileSpec,
    build_kda_chunk_fused,
    is_valid_fused_spec,
    kda_chunk_fused_grid,
    kda_chunk_fused_signature,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches  # noqa: E402

_LAUNCHER_CACHE: dict = {}


def make_launcher(spec: KdaChunkFusedSpec) -> KernelLauncher:
    key = spec.kernel_name()
    if key not in _LAUNCHER_CACHE:
        ok, why = is_valid_fused_spec(spec)
        if not ok:
            raise ValueError(f"unsupported spec: {why}")
        art = compile_kernel(
            build_kda_chunk_fused(spec),
            arch="gfx942",
            backend="python",
            capture_ir_text=False,
        )
        _LAUNCHER_CACHE[key] = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=kda_chunk_fused_signature(spec),
        )
    return _LAUNCHER_CACHE[key]


def run_fused(spec, q, k, g, beta, v, o, ht, scale, bh, nc, h0=None, stream=None):
    """Launch the fused kernel over inputs already packed by chunk.

    ``q/k/g`` are ``[BH*NC, C*DK]``, ``beta`` is ``[BH*NC, C]``, ``v``/``o`` are
    ``[BH*NC, C*DV]``, and ``h0``/``ht`` are ``[BH, DV, DK]`` fp32 (the state
    kept transposed, which is the orientation both of its consumers want).
    """
    launcher = make_launcher(spec)
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    cfg = LaunchConfig(
        grid=kda_chunk_fused_grid(spec, bh),
        block=(spec.tile.block_size, 1, 1),
        stream=stream,
    )
    launcher(
        {
            "q_ptr": q,
            "k_ptr": k,
            "g_ptr": g,
            "beta_ptr": beta,
            "v_ptr": v,
            "o_ptr": o,
            # unread when the kernel was built with has_initial_state=False
            "h0_ptr": ht if h0 is None else h0,
            "ht_ptr": ht,
            "scale": float(scale),
            "nc": int(nc),
        },
        config=cfg,
    )


# ---------------------------------------------------------------------
# reference + parity
# ---------------------------------------------------------------------


def ref_token_serial(q, k, v, g, beta, scale, h0=None):
    """Token-serial float64 oracle for the gated delta rule.

    Per token, with ``S`` the ``DK x DV`` state:

    .. code-block:: text

        S <- Diag(exp(g_t)) S
        u <- beta_t (v_t - k_t^T S)
        S <- S + k_t u^T
        o_t = scale * q_t^T S

    Deliberately not chunked: this is the recurrence the chunkwise
    factorization claims to compute, so it is an independent check of that
    claim rather than a re-derivation of it.
    """
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    qd, kd, vd = (x.double() for x in (q, k, v))
    gd, bd = g.double(), beta.double()
    o = torch.zeros(B, H, T, DV, dtype=torch.float64, device=q.device)
    S = (
        torch.zeros(B, H, DK, DV, dtype=torch.float64, device=q.device)
        if h0 is None
        else h0.double().clone()
    )
    for t in range(T):
        S = S * gd[:, :, t, :].exp().unsqueeze(-1)
        kt = kd[:, :, t, :]
        kv = torch.einsum("bhd,bhde->bhe", kt, S)
        u = bd[:, :, t].unsqueeze(-1) * (vd[:, :, t, :] - kv)
        S = S + kt.unsqueeze(-1) * u.unsqueeze(-2)
        o[:, :, t, :] = scale * torch.einsum("bhd,bhde->bhe", qd[:, :, t, :], S)
    return o, S


def make_inputs(B, H, T, DK, DV, gate_low=-0.5, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    kw = dict(device=device, generator=gen)
    q = torch.nn.functional.normalize(
        torch.randn(B, H, T, DK, dtype=torch.float32, **kw), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(
        torch.randn(B, H, T, DK, dtype=torch.float32, **kw), dim=-1
    ).bfloat16()
    v = (torch.randn(B, H, T, DV, dtype=torch.float32, **kw) * 0.2).bfloat16()
    g = gate_low * torch.rand(B, H, T, DK, dtype=torch.float32, **kw)
    beta = torch.rand(B, H, T, dtype=torch.float32, **kw)
    return [x.contiguous() for x in (q, k, v, g, beta)]


def _pack_v_partitions(spec, q, k, v, g, beta):
    """Pack dense inputs into independent gfx942 V partitions.

    gfx942 cannot keep a full DV=128 state mirror plus the C16 tile builder
    under its 64 KiB LDS ceiling. Each workgroup therefore owns
    ``spec.head_v`` channels. Q/K/g/beta are duplicated across partitions;
    V is split, and the device grid gains one independent workgroup per
    partition.
    """
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    C = spec.tile.chunk
    if T % C:
        raise ValueError(f"T ({T}) must be divisible by chunk ({C})")
    if DV % spec.head_v:
        raise ValueError(
            f"logical head_v ({DV}) must be divisible by partition "
            f"head_v ({spec.head_v})"
        )
    BH, NC = B * H, T // C
    parts = DV // spec.head_v

    def duplicate(x, tail):
        return (
            x.view(BH, NC, C, tail)[:, None]
            .expand(BH, parts, NC, C, tail)
            .contiguous()
            .view(BH * parts * NC, C * tail)
        )

    qf = duplicate(q, DK)
    kf = duplicate(k, DK)
    gf = duplicate(g, DK)
    bf = (
        beta.view(BH, NC, C)[:, None]
        .expand(BH, parts, NC, C)
        .contiguous()
        .view(BH * parts * NC, C)
    )
    vf = (
        v.view(BH, NC, C, parts, spec.head_v)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(BH * parts * NC, C * spec.head_v)
    )
    return qf, kf, vf, gf, bf, BH, NC, parts


def _pack_initial_state(spec, h0, parts):
    if h0 is None:
        return None
    B, H, DK, DV = h0.shape
    BH = B * H
    return (
        h0.view(BH, DK, parts, spec.head_v)
        .permute(0, 2, 3, 1)
        .contiguous()
        .view(BH * parts, spec.head_v, DK)
    )


def _unpack_outputs(spec, o, ht, B, H, T, DV, NC, parts):
    C, BH = spec.tile.chunk, B * H
    out = (
        o.view(BH, parts, NC, C, spec.head_v)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(B, H, T, DV)
    )
    final = (
        ht.view(BH, parts, spec.head_v, spec.head_k)
        .permute(0, 3, 1, 2)
        .contiguous()
        .view(B, H, spec.head_k, DV)
    )
    return out, final


def launch_packed(spec, q, k, v, g, beta, h0=None):
    """Run dense ``[B,H,T,D]`` inputs through gfx942 V partitions."""
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    qf, kf, vf, gf, bf, BH, NC, parts = _pack_v_partitions(spec, q, k, v, g, beta)
    nt = BH * parts * NC
    o = torch.empty(
        nt, spec.tile.chunk * spec.head_v, dtype=torch.bfloat16, device=q.device
    )
    ht = torch.zeros(BH * parts, spec.head_v, DK, dtype=torch.float32, device=q.device)
    h0t = _pack_initial_state(spec, h0, parts)
    run_fused(
        spec,
        qf,
        kf,
        gf,
        bf,
        vf,
        o,
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


def bench(spec, B, H, T, warmup=10, iters=30, logical_head_v=128):
    DK, DV = spec.head_k, logical_head_v
    q, k, v, g, beta = make_inputs(B, H, T, DK, DV)
    qf, kf, vf, gf, bf, BH, NC, parts = _pack_v_partitions(spec, q, k, v, g, beta)
    C = spec.tile.chunk
    nt = BH * parts * NC
    o = torch.empty(nt, C * spec.head_v, dtype=torch.bfloat16, device="cuda")
    ht = torch.zeros(BH * parts, spec.head_v, DK, dtype=torch.float32, device="cuda")
    args = (
        spec,
        qf,
        kf,
        gf,
        bf,
        vf,
        o,
        ht,
        DK**-0.5,
        BH * parts,
        NC,
    )
    ms = time_launches(lambda: run_fused(*args), warmup=warmup, iters=iters)
    # Essential traffic only: q/k/v in, g/beta in, o out. The fused kernel adds
    # nothing else, which is the whole point of the design.
    rd = nt * (2 * C * DK * 2 + C * DV * 2 + C * DK * 4 + C * 4)
    wr = nt * C * DV * 2
    gbps = (rd + wr) / (ms * 1e-3) / 2**30
    print(f"  {ms:.4f} ms  {gbps:.1f} GiB/s  (B={B} H={H} T={T}, {nt} chunks)")
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
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="alias of --no-check; remote_test build passes this",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="write HSACO + manifest.json for rocke.run_manifest",
    )
    ap.add_argument("--m", type=int, default=None, help="B (remote-test --shape)")
    ap.add_argument("--n", type=int, default=None, help="H (remote-test --shape)")
    ap.add_argument("--k", type=int, default=None, help="T (remote-test --shape)")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    spec = KdaChunkFusedSpec(
        head_k=args.dk,
        head_v=args.dv // args.v_splits,
        tile=KdaTileSpec(
            chunk=args.chunk,
            block_size=256,
            pad_dk=8,
            pad_cb=8,
            tile_atom_m=16,
            solve_block=args.sb,
            scan_atom_m=16,
        ),
    )
    ok, why = is_valid_fused_spec(spec)
    print(f"{spec.kernel_name()}  lds={spec.lds_bytes()} B  valid={ok} {why}")
    if not ok:
        return 1

    mnk = (args.m, args.n, args.k)
    if any(x is not None for x in mnk) and not all(x is not None for x in mnk):
        print("--m/--n/--k (B/H/T) must be passed together", file=sys.stderr)
        return 2
    if all(x is not None for x in mnk):
        default_shape = (int(args.m), int(args.n), int(args.k))
    else:
        default_shape = tuple(int(x) for x in args.check_shape.split("x"))

    skip_check = args.no_check or args.no_verify

    if args.output_dir:
        from pathlib import Path

        from rocke.helpers.manifest import write_artifact

        try:
            from .manifest import make_kda_fused_manifest
        except ImportError:  # script invocation from this directory
            from manifest import make_kda_fused_manifest

        art = compile_kernel(
            build_kda_chunk_fused(spec),
            arch="gfx942",
            backend="python",
            capture_ir_text=False,
        )
        manifest = make_kda_fused_manifest(
            artifact=art,
            spec=spec,
            args_signature=kda_chunk_fused_signature(spec),
            logical_head_v=args.dv,
            default_shape=default_shape,
        )
        out = Path(args.output_dir)
        write_artifact(art, out, manifest, write_ir_text=False, write_llvm_text=True)
        print(f"wrote artifact to {out}")
        if skip_check:
            return 0

    worst = 0.0
    if not skip_check:
        B, H, T = default_shape
        for gate_low in (-0.1, -0.5, -2.0, -5.0):
            for with_h0 in (False, True):
                if with_h0 and not spec.has_initial_state:
                    continue
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

    if args.output_dir:
        return 0 if worst <= 3e-2 else 1

    for s in args.shapes.split(","):
        B, H, T = (int(x) for x in s.split("x"))
        bench(spec, B, H, T, iters=args.iters, logical_head_v=args.dv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
