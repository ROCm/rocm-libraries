# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host builder for the fused chunkwise KDA forward on gfx942.

Builds, validates and benchmarks :func:`build_kda_chunk_fused`: one workgroup
per (batch, head) walks that head's chunks in order, so the six per-chunk tiles
are produced and consumed in LDS and the only HBM traffic is q/k/g/beta in and
o (plus the final state) out.

Correctness is checked against a token-serial float64 oracle
(:func:`rocke.numeric.gated_delta.gated_delta_recurrence`), which shares no
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


from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.numeric import storage  # noqa: E402
from rocke.numeric.gated_delta import gated_delta_recurrence  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches  # noqa: E402
from rocke.runtime.device_array import DeviceArray  # noqa: E402

from builders.common.kda_host import dense_inputs, initial_state, score  # noqa: E402
from builders.gfx942.kda.hostpack import (  # noqa: E402
    pack_initial_state,
    pack_v_partitions,
    unpack_outputs,
)
from kernels.gfx942.kda_chunkwise import (  # noqa: E402
    KdaChunkFusedSpec,
    KdaTileSpec,
    build_kda_chunk_fused,
    is_valid_fused_spec,
    kda_chunk_fused_grid,
    kda_chunk_fused_signature,
)

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


def run_fused(spec, q, k, g, beta, v, o, ht, scale, bh, nc, h0=None, stream=0):
    """Launch the fused kernel over device inputs already packed by chunk, on
    ``stream`` (0: the default stream).

    ``q/k/g`` are ``[BH*NC, C*DK]``, ``beta`` is ``[BH*NC, C]``, ``v``/``o`` are
    ``[BH*NC, C*DV]``, and ``h0``/``ht`` are ``[BH, DV, DK]`` fp32 (the state
    kept transposed, which is the orientation both of its consumers want).
    """
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


class PartitionedLaunch:
    """Dense host ``[B,H,T,*]`` inputs packed into gfx942 V partitions
    (:func:`builders.gfx942.kda.hostpack.pack_v_partitions`) and uploaded, with
    the zeroed outputs and state the kernels write. :meth:`results` reads them
    back unpacked."""

    def __init__(self, spec, q, k, v, g, beta, h0=None):
        self.batch, self.heads, self.tokens, self.head_k = q.shape
        self.head_v = v.shape[-1]
        packed = pack_v_partitions(spec.tile.chunk, spec.head_v, q, k, v, g, beta)
        self.packed = packed
        self.q = DeviceArray(packed.q, "bf16")
        self.k = DeviceArray(packed.k, "bf16")
        self.v = DeviceArray(packed.v, "bf16")
        self.g = DeviceArray(packed.g)
        self.beta = DeviceArray(packed.beta)
        streams = packed.bh * packed.parts
        self.streams = streams
        self.o = DeviceArray.zeros(packed.v.shape, "bf16")
        self.ht = DeviceArray.zeros((streams, spec.head_v, self.head_k), "fp32")
        self.h0 = (
            None
            if h0 is None
            else DeviceArray(pack_initial_state(spec.head_v, h0, packed.parts))
        )

    def results(self):
        """Host ``(o [B,H,T,DV] bf16 bits, final state [B,H,DK,DV] fp32)``."""
        packed = self.packed
        return unpack_outputs(
            chunk=packed.chunk,
            head_k=packed.head_k,
            head_v=packed.head_v,
            o=self.o.values(),
            ht=self.ht.values(),
            B=self.batch,
            H=self.heads,
            T=self.tokens,
            DV=self.head_v,
            NC=packed.nc,
            parts=packed.parts,
        )


def reference(q, k, v, g, beta, scale, h0=None):
    """The token-serial float64 oracle on the stored dense host inputs:
    ``(o [B,H,T,DV], final state [B,H,DK,DV])``. Deliberately not chunked, so
    agreement tests the chunkwise factorization itself."""
    q, k, v = (storage.decode(x, "bf16") for x in (q, k, v))
    return gated_delta_recurrence(q, k, v, g, beta, scale=scale, initial_state=h0)


def launch_packed(spec, q, k, v, g, beta, h0=None):
    """Run dense host ``[B,H,T,D]`` inputs through gfx942 V partitions; returns
    host ``(o [B,H,T,DV] bf16 bits, final state [B,H,DK,DV] fp32)``."""
    run = PartitionedLaunch(spec, q, k, v, g, beta, h0)
    run_fused(
        spec,
        run.q,
        run.k,
        run.g,
        run.beta,
        run.v,
        run.o,
        run.ht,
        run.head_k**-0.5,
        run.streams,
        run.packed.nc,
        h0=run.h0,
    )
    return run.results()


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
    q, k, v, g, beta = dense_inputs(B, H, T, DK, DV, gate_low=gate_low)
    h0 = initial_state(B, H, DK, DV) if with_h0 else None
    o_got, ht_got = launch_packed(spec, q, k, v, g, beta, h0=h0)
    o_ref, s_ref = reference(q, k, v, g, beta, DK**-0.5, h0=h0)
    pairs = (
        ("out", storage.decode(o_got, "bf16"), o_ref),
        ("final_state", ht_got, s_ref),
    )
    return score(pairs, tol, verbose=verbose)


def bench(spec, B, H, T, warmup=10, iters=30, logical_head_v=128):
    DK, DV = spec.head_k, logical_head_v
    run = PartitionedLaunch(spec, *dense_inputs(B, H, T, DK, DV))
    C = spec.tile.chunk
    nt = run.streams * run.packed.nc
    args = (
        spec,
        run.q,
        run.k,
        run.g,
        run.beta,
        run.v,
        run.o,
        run.ht,
        DK**-0.5,
        run.streams,
        run.packed.nc,
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
