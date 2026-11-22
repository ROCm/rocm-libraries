#!/usr/bin/env python3
"""Host builder for the split-path chunkwise KDA forward on gfx950.

Two kernels: :func:`build_kda_chunk_prep` builds the six per-chunk tiles, one
workgroup per chunk and fully parallel over the sequence, then
:func:`build_kda_chunk_scan` walks each (batch, head)'s chunks in order,
staging that chunk's tiles from HBM into LDS and running the state recurrence.

The alternative is the fused kernel in ``kda_chunk_fused.py``, which keeps the
tiles in LDS and never writes them out. That is strictly less traffic and
measurably slower: the combined footprint puts one workgroup over half the LDS
budget, so the scan's latency-bound chain of small matmuls runs with no second
workgroup to cover it. This path pays the tile round trip to keep two resident.

Correctness is checked end to end against the same token-serial float64 oracle
the fused builder uses, so the two paths are held to one contract.

Run directly for a parity sweep plus a benchmark::

    python kda_chunk_split.py                 # parity + bench
    python kda_chunk_split.py --no-check      # bench only
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

try:
    import rocke  # noqa: F401
except ImportError:  # running as a bare script outside the editable install
    _HERE = os.path.dirname(__file__)
    _RK = os.path.abspath(os.path.join(_HERE, "../../../.."))
    sys.path[:0] = [_RK + "/library", _RK + "/platform/python"]

import numpy as np  # noqa: E402
from rocke.core.dtypes import dtype_info  # noqa: E402
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.numeric import storage  # noqa: E402
from rocke.numeric.gated_delta import (  # noqa: E402
    gated_delta_recurrence,
    kda_log_decay,
    l2_normalize,
    sigmoid,
)
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches  # noqa: E402
from rocke.runtime.device_array import DeviceArray  # noqa: E402

from builders.common.kda_host import dense_inputs, initial_state, score  # noqa: E402
from kernels.gfx950.kda_chunkwise import (  # noqa: E402
    KdaChunkPrepSpec,
    KdaChunkScanSpec,
    KdaTileSpec,
    build_kda_chunk_scan,
    g_elem_dtype,
    is_valid_scan_spec,
    is_valid_spec,
    kda_chunk_scan_grid,
    kda_chunk_scan_signature,
)

if __package__:
    from . import kda_chunk_prep as prep_mod  # noqa: E402
    from .kda_chunk_fused import (  # noqa: E402
        device_state,
        read_state,
        reference,
        upload_chunked,
    )
else:
    import kda_chunk_prep as prep_mod  # noqa: E402
    from kda_chunk_fused import (  # noqa: E402
        device_state,
        read_state,
        reference,
        upload_chunked,
    )

_LAUNCHER_CACHE: dict = {}
TOL = 3e-2


def make_launcher(spec: KdaChunkScanSpec) -> KernelLauncher:
    key = spec.kernel_name()
    if key not in _LAUNCHER_CACHE:
        ok, why = is_valid_scan_spec(spec)
        if not ok:
            raise ValueError(f"unsupported spec: {why}")
        art = compile_kernel(
            build_kda_chunk_scan(spec),
            arch="gfx950",
            backend="python",
            capture_ir_text=False,
        )
        _LAUNCHER_CACHE[key] = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=kda_chunk_scan_signature(spec),
        )
    return _LAUNCHER_CACHE[key]


def prep_spec_of(spec: KdaChunkScanSpec, *, raw: bool = False) -> KdaChunkPrepSpec:
    """The tile builder that produces this scan's inputs.

    Both halves are derived from one scan spec so the tile layouts cannot drift
    apart: the scan's staging copies assume exactly the flat per-chunk layouts
    the prep kernel's global sink writes.

    Raw prep keeps the default 256-thread tile builder even when the scan uses a
    narrower block for ``value_splits > 1``.
    """
    raw_kw = {}
    tile = spec.tile
    if raw:
        raw_kw = dict(
            raw_inputs=True,
            fuse_qk_l2norm=True,
            fuse_gate=True,
            fuse_beta_sigmoid=True,
            has_dt_bias=True,
            lower_bound=-5.0,
        )
        tile = dataclasses.replace(spec.tile, block_size=256)
    return KdaChunkPrepSpec(
        head_k=spec.head_k,
        head_v=spec.head_v,
        dtype=spec.dtype,
        tile=tile,
        **raw_kw,
    )


def aligned_split_specs(value_splits: int = 1):
    """Default aligned-C32 split family knobs for the ten benchmark shapes."""
    if value_splits == 8:
        tile = KdaTileSpec(chunk=32, block_size=64, scan_atom_m=16)
    elif value_splits == 4:
        tile = KdaTileSpec(chunk=32, block_size=64)
    elif value_splits == 2:
        tile = KdaTileSpec(chunk=32, block_size=128)
    else:
        tile = KdaTileSpec(chunk=32, block_size=256)
    scan = KdaChunkScanSpec(
        tile=tile,
        value_splits=value_splits,
        token_major_io=True,
    )
    return scan, prep_spec_of(scan, raw=True)


def run_scan(
    spec,
    ws,
    v,
    o,
    ht,
    bh,
    nc,
    h0=None,
    stream=0,
    *,
    batch=None,
    heads=None,
    tseq=None,
):
    """Launch the scan over tiles already materialized by the prep kernel, on
    ``stream`` (0: the default stream)."""
    if h0 is None:
        assert not spec.has_initial_state, "h0=None but the kernel reads h0_ptr"
    launcher = make_launcher(spec)
    cfg = LaunchConfig(
        grid=kda_chunk_scan_grid(spec, bh),
        block=(spec.tile.block_size, 1, 1),
        stream=stream,
    )
    args = {
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
    }
    if spec.token_major_io:
        args.update({"batch": int(batch), "heads": int(heads), "tseq": int(tseq)})
    launcher(args, config=cfg)


def run_split(
    spec,
    q,
    k,
    g,
    beta,
    v,
    o,
    ws,
    ht,
    scale,
    bh,
    nc,
    h0=None,
    *,
    raw=False,
    batch=None,
    heads=None,
    tseq=None,
    a_log=None,
    dt_bias=None,
):
    """Both kernels, back to back on one stream.

    No fence between them: they run in FIFO order on the same stream, so the
    scan already sees the tiles the prep kernel wrote.
    """
    pspec = prep_spec_of(spec, raw=raw)
    prep_mod.run_prep(
        pspec,
        q,
        k,
        g,
        beta,
        ws,
        scale,
        batch=batch,
        heads=heads,
        tseq=tseq,
        nc=nc,
        a_log=a_log,
        dt_bias=dt_bias,
    )
    run_scan(
        spec,
        ws,
        v,
        o,
        ht,
        bh,
        nc,
        h0=h0,
        batch=batch,
        heads=heads,
        tseq=tseq,
    )


def run_raw_split(
    spec,
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    o,
    ht,
    scale,
    batch,
    heads,
    tseq,
    h0=None,
):
    """Aligned raw token-major split path: fused prep + value-split scan."""
    C = spec.tile.chunk
    BH, NC = batch * heads, tseq // C
    nt = BH * NC
    ws = prep_mod.alloc_tiles(nt, prep_spec_of(spec, raw=True))
    run_split(
        spec,
        q,
        k,
        g,
        beta,
        v,
        o,
        ws,
        ht,
        scale,
        BH,
        NC,
        h0=h0,
        raw=True,
        batch=batch,
        heads=heads,
        tseq=tseq,
        a_log=a_log,
        dt_bias=dt_bias,
    )
    return ws


def upload_raw(spec, q, k, v, g, beta, a_log, dt_bias):
    """Host token-major raw inputs as the raw split path's device arrays, each
    in the element type the kernels read: ``q``/``k``/``v`` in ``spec.dtype``,
    ``g`` in the raw prep kernel's gate type, the rest fp32."""
    gate_type = g_elem_dtype(prep_spec_of(spec, raw=True))
    return (
        DeviceArray(q, spec.dtype),
        DeviceArray(k, spec.dtype),
        DeviceArray(v, spec.dtype),
        DeviceArray(g, gate_type),
        DeviceArray(beta, "fp32"),
        DeviceArray(a_log, "fp32"),
        DeviceArray(dt_bias, "fp32"),
    )


def launch_raw(
    spec,
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    h0=None,
):
    """Host token-major ``[B,T,H,D]`` raw inputs in, host ``(o [B,T,H,DV]``
    bf16 bits, V-first final state ``[B,H,DK,DV]`` fp32) out."""
    B, T, H, DK = q.shape
    q_d, k_d, v_d, g_d, beta_d, a_log_d, dt_bias_d = upload_raw(
        spec, q, k, v, g, beta, a_log, dt_bias
    )
    o = DeviceArray.zeros(v.shape, spec.dtype)
    ht = device_state(spec, B, H)
    h0t = None if h0 is None else device_state(spec, B, H, h0)
    run_raw_split(
        spec,
        q_d,
        k_d,
        v_d,
        g_d,
        beta_d,
        a_log_d,
        dt_bias_d,
        o,
        ht,
        DK**-0.5,
        B,
        H,
        T,
        h0=h0t,
    )
    return o.values(), read_state(ht, B, H)


def ref_aligned_raw(
    spec,
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    scale,
    lower_bound=-5.0,
    h0=None,
):
    """Float64 oracle for the aligned raw contract (Aiter preprocessing + KDA)
    on the host inputs :func:`launch_raw` takes, stored as ``spec``'s kernels
    read them: ``(o [B,H,T,DV], final state [B,H,DK,DV])``.

    q/k are L2-normalised and rounded to the I/O type, as the raw prep kernel
    stores them, then the token-serial recurrence runs on the cooked gate and
    beta.
    """
    io = dtype_info(spec.dtype).name
    gate_type = dtype_info(g_elem_dtype(prep_spec_of(spec, raw=True))).name
    B, T, H, DK = q.shape
    q, k, v = (storage.decode(x, io) for x in (q, k, v))
    gate = kda_log_decay(
        storage.decode(g, gate_type),
        dt_bias.reshape(H, DK),
        a_log.reshape(H, 1),
        lower_bound,
    )

    def head_major(x):
        return np.swapaxes(x, 1, 2)

    qn = storage.round_to(l2_normalize(q), io)
    kn = storage.round_to(l2_normalize(k), io)
    return gated_delta_recurrence(
        head_major(qn),
        head_major(kn),
        head_major(v),
        head_major(gate),
        head_major(sigmoid(beta)),
        scale=scale,
        initial_state=h0,
    )


# ---------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------


def launch_packed(spec, q, k, v, g, beta, h0=None):
    """Run dense host ``[B,H,T,D]`` inputs through prep + scan; returns host
    ``(o [B,H,T,DV] bf16 bits, final state [B,H,DK,DV] fp32)``."""
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    C = spec.tile.chunk
    BH, NC = B * H, T // C
    nt = BH * NC
    ws = prep_mod.alloc_tiles(nt, prep_spec_of(spec))
    q_d, k_d, g_d, beta_d, v_d = upload_chunked(spec, q, k, v, g, beta)
    o = DeviceArray.zeros((nt, C * DV), "bf16")
    ht = device_state(spec, B, H)
    h0t = None if h0 is None else device_state(spec, B, H, h0)
    run_split(spec, q_d, k_d, g_d, beta_d, v_d, o, ws, ht, DK**-0.5, BH, NC, h0=h0t)
    return o.values().reshape(B, H, T, DV), read_state(ht, B, H)


def check(spec, B, H, T, gate_low=-0.5, tol=TOL, with_h0=False, verbose=True):
    DK, DV = spec.head_k, spec.head_v
    q, k, v, g, beta = dense_inputs(B, H, T, DK, DV, gate_low=gate_low)
    h0 = initial_state(B, H, DK, DV) if with_h0 else None
    o_got, ht_got = launch_packed(spec, q, k, v, g, beta, h0=h0)
    o_ref, s_ref = reference(q, k, v, g, beta, DK**-0.5, h0=h0)
    pairs = (
        ("out", storage.decode(o_got, "bf16"), o_ref),
        ("final_state", ht_got, s_ref),
    )
    return score(pairs, tol, verbose=verbose)


def bench(spec, B, H, T, warmup=10, iters=30, split=True):
    """Time the pair, and each half alone, on one shape.

    The halves are reported too because they answer different questions: the
    prep time is bandwidth against a known roofline, while the scan time is a
    serial chain whose only lever is occupancy.
    """
    DK, DV = spec.head_k, spec.head_v
    q, k, v, g, beta = dense_inputs(B, H, T, DK, DV)
    C = spec.tile.chunk
    BH, NC = B * H, T // C
    nt = BH * NC
    ws = prep_mod.alloc_tiles(nt, prep_spec_of(spec))
    qf, kf, gf, bf, vf = upload_chunked(spec, q, k, v, g, beta)
    o = DeviceArray.zeros((nt, C * DV), "bf16")
    ht = device_state(spec, B, H)
    scale = DK**-0.5

    def do_prep():
        prep_mod.run_prep(prep_spec_of(spec), qf, kf, gf, bf, ws, scale)

    def do_scan():
        run_scan(spec, ws, vf, o, ht, BH, NC)

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
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--sb", type=int, default=8)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    spec = KdaChunkScanSpec(
        head_k=args.dk,
        head_v=args.dv,
        tile=KdaTileSpec(chunk=args.chunk, solve_block=args.sb),
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
            for with_h0 in (False, True):
                check_spec = (
                    spec
                    if not with_h0
                    else dataclasses.replace(spec, has_initial_state=True)
                )
                state_label = " with h0" if with_h0 else ""
                print(f"parity B={B} H={H} T={T} gate in [{gate_low}, 0]{state_label}:")
                worst = max(
                    worst,
                    check(
                        check_spec,
                        B,
                        H,
                        T,
                        gate_low=gate_low,
                        with_h0=with_h0,
                    ),
                )

    for s in args.shapes.split(","):
        B, H, T = (int(x) for x in s.split("x"))
        bench(spec, B, H, T, iters=args.iters)

    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
