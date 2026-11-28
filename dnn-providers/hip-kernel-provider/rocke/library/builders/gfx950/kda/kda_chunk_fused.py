"""Host builder for the fused chunkwise KDA forward on gfx950.

Builds, validates and benchmarks :func:`build_kda_chunk_fused`: one workgroup
per (batch, head) walks that head's chunks in order, so the six per-chunk tiles
are produced and consumed in LDS and the only HBM traffic is q/k/g/beta in and
o (plus the final state) out.

Correctness is checked against a token-serial float64 oracle
(:func:`rocke.numeric.gated_delta.gated_delta_recurrence`), which shares no
algebra with the kernel: the kernel evaluates the chunkwise factorization,
while the oracle just walks the gated delta rule one token at a time. Agreement
therefore tests the factorization itself, not only its implementation.

Run directly for a parity sweep plus a benchmark::

    python kda_chunk_fused.py                 # parity + bench
    python kda_chunk_fused.py --no-check      # bench only
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
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.numeric import storage  # noqa: E402
from rocke.numeric.gated_delta import gated_delta_recurrence  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches  # noqa: E402
from rocke.runtime.device_array import DeviceArray  # noqa: E402

from builders.common.kda_host import dense_inputs, initial_state, score  # noqa: E402
from kernels.gfx950.kda_chunkwise import (  # noqa: E402
    KdaChunkFusedSpec,
    KdaTileSpec,
    build_kda_chunk_fused,
    is_valid_fused_spec,
    kda_chunk_fused_grid,
    kda_chunk_fused_signature,
)

_LAUNCHER_CACHE: dict = {}
TOL = 3e-2


def make_launcher(spec: KdaChunkFusedSpec) -> KernelLauncher:
    key = spec.kernel_name()
    if key not in _LAUNCHER_CACHE:
        ok, why = is_valid_fused_spec(spec)
        if not ok:
            raise ValueError(f"unsupported spec: {why}")
        art = compile_kernel(
            build_kda_chunk_fused(spec),
            arch="gfx950",
            backend="python",
            capture_ir_text=False,
        )
        _LAUNCHER_CACHE[key] = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=kda_chunk_fused_signature(spec),
        )
    return _LAUNCHER_CACHE[key]


def run_fused(spec, q, k, g, beta, v, o, ht, scale, bh, nc, h0=None, stream=0):
    """Launch the fused kernel over device inputs already packed by chunk, on
    ``stream`` (0: the default stream).

    ``q/k/g`` are ``[BH*NC, C*DK]``, ``beta`` is ``[BH*NC, C]``, ``v``/``o`` are
    ``[BH*NC, C*DV]``, and ``h0``/``ht`` are ``[BH, DV, DK]`` fp32 (the state
    kept transposed, which is the orientation both of its consumers want).
    """
    if h0 is None:
        assert not spec.has_initial_state, "h0=None but the kernel reads h0_ptr"
    launcher = make_launcher(spec)
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
# parity
# ---------------------------------------------------------------------


def upload_chunked(spec, q, k, v, g, beta):
    """Dense host ``[B,H,T,*]`` inputs as the kernels' chunk-major device
    arrays ``(q, k, g, beta, v)``: ``[BH*NC, C*D]`` (``beta``: ``[BH*NC, C]``)."""
    batch, heads, tokens, _ = q.shape
    num_tiles = batch * heads * tokens // spec.tile.chunk

    def chunked(x, dtype):
        return DeviceArray(x.reshape(num_tiles, -1), dtype)

    return (
        chunked(q, "bf16"),
        chunked(k, "bf16"),
        chunked(g, "fp32"),
        chunked(beta, "fp32"),
        chunked(v, "bf16"),
    )


def device_state(spec, batch, heads, h0=None):
    """The kernels' fp32 state buffer ``[BH, DV, DK]``: zero, or ``h0``
    ``[B,H,DK,DV]`` transposed into it."""
    shape = (batch * heads, spec.head_v, spec.head_k)
    if h0 is None:
        return DeviceArray.zeros(shape, "fp32")
    return DeviceArray(np.ascontiguousarray(np.swapaxes(h0, -1, -2)).reshape(shape))


def read_state(state, batch, heads):
    """A device ``[BH, DV, DK]`` state back as host ``[B, H, DK, DV]`` fp32."""
    host = state.values()
    return np.swapaxes(host.reshape(batch, heads, *host.shape[1:]), -1, -2)


def reference(q, k, v, g, beta, scale, h0=None):
    """The token-serial float64 oracle on the stored host inputs:
    ``(o [B,H,T,DV], final state [B,H,DK,DV])``. Deliberately not chunked, so
    agreement tests the chunkwise factorization itself."""
    q, k, v = (storage.decode(x, "bf16") for x in (q, k, v))
    return gated_delta_recurrence(q, k, v, g, beta, scale=scale, initial_state=h0)


def launch_packed(spec, q, k, v, g, beta, h0=None):
    """Run dense host ``[B,H,T,D]`` inputs through the fused kernel; returns
    host ``(o [B,H,T,DV] bf16 bits, final state [B,H,DK,DV] fp32)``."""
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    C = spec.tile.chunk
    BH, NC = B * H, T // C
    q_d, k_d, g_d, beta_d, v_d = upload_chunked(spec, q, k, v, g, beta)
    o = DeviceArray.zeros((BH * NC, C * DV), "bf16")
    ht = device_state(spec, B, H)
    h0t = None if h0 is None else device_state(spec, B, H, h0)
    run_fused(spec, q_d, k_d, g_d, beta_d, v_d, o, ht, DK**-0.5, BH, NC, h0=h0t)
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


def bench(spec, B, H, T, warmup=10, iters=30):
    DK, DV = spec.head_k, spec.head_v
    q, k, v, g, beta = dense_inputs(B, H, T, DK, DV)
    C = spec.tile.chunk
    BH, NC = B * H, T // C
    nt = BH * NC
    q_d, k_d, g_d, beta_d, v_d = upload_chunked(spec, q, k, v, g, beta)
    o = DeviceArray.zeros((nt, C * DV), "bf16")
    ht = device_state(spec, B, H)
    args = (spec, q_d, k_d, g_d, beta_d, v_d, o, ht, DK**-0.5, BH, NC)
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
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--sb", type=int, default=8)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    spec = KdaChunkFusedSpec(
        head_k=args.dk,
        head_v=args.dv,
        tile=KdaTileSpec(
            chunk=args.chunk,
            block_size=512,
            pad_dk=16,
            pad_cb=16,
            tile_atom_m=16,
            solve_block=args.sb,
            scan_atom_m=16,
        ),
    )
    ok, why = is_valid_fused_spec(spec)
    print(f"{spec.kernel_name()}  lds={spec.lds_bytes()} B  valid={ok} {why}")
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

    return 0 if args.no_check or worst <= TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
