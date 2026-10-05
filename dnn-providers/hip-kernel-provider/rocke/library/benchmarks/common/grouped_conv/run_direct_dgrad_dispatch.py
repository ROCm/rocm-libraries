# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch ONE grouped dgrad request and run exactly what dispatch ships.

Replay driver for
``platform/python/rocke/examples/gfx950/conv_dgrad/grouped_direct_dgrad_dispatch_case_study.md``
(it lives in ``library/`` because it drives the library dispatcher). It asks
``dispatch_conv_grouped`` for the request, prints the selected candidate, its
knobs and launch plan, then (optionally) builds every kernel the plan lists and
verifies / times / loops the whole pipeline. When dispatch picks the igemm
candidate (or the depthwise windowed candidate) it runs that single kernel
instead, so the same command line A/Bs the candidates across checkouts.

* ``--verify``: numpy reference, manifest-runner conv rule
  (``|D - ref| <= 1e-2 + 1e-2*|ref|``), with dX and every workspace pre-filled
  with 0xFF bytes (NaN in fp16 and bf16) so an unwritten element fails.
* ``--time``: median over ``--reps`` of ``time_launches`` means (whole pipeline).
* ``--loop N``: issue N back-to-back pipelines and exit -- run under
  ``rocprofv3 --kernel-trace`` for per-kernel durations. torch is blocked in this
  mode (its HIP runtime registration aborts rocprofv3).

Usage (from ``library/``, ``PYTHONPATH=$(pwd)/../platform/python:$(pwd)``):
    python3 benchmarks/common/grouped_conv/run_direct_dgrad_dispatch.py \\
        --N 128 --C 512 --K 512 --H 14 --W 14 --G 32 --dtype bf16 --verify
    rocprofv3 --kernel-trace --stats -d prof -o run -- python3 \\
        benchmarks/common/grouped_conv/run_direct_dgrad_dispatch.py \\
        --N 128 --C 512 --K 512 --H 14 --W 14 --G 16 --loop 50
"""

import argparse
import ctypes
import statistics
import struct
import sys

_PREPASS_SIG = [
    {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
    {"name": "D", "type": "ptr<f16, global>", "size_bytes": 8},
    {"name": "A_bytes", "type": "i32", "size_bytes": 4},
    {"name": "D_bytes", "type": "i32", "size_bytes": 4},
]


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name, default in (
        ("N", 8),
        ("C", 128),
        ("K", 128),
        ("H", 14),
        ("W", 14),
        ("G", 32),
    ):
        ap.add_argument(f"--{name}", type=int, default=default)
    ap.add_argument("--Y", type=int, default=3, help="square filter size")
    ap.add_argument("--pad", type=int, default=None, help="default (Y-1)/2")
    ap.add_argument("--dtype", default="bf16", choices=("fp16", "bf16"))
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--loop", type=int, default=0)
    return ap.parse_args(argv)


def _rand_bits(np, shape, dtype, seed):
    f = np.random.default_rng(seed).uniform(-1, 1, size=shape).astype(np.float32)
    if dtype == "bf16":
        u = f.view(np.uint32)
        bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        return (bits.astype(np.uint32) << 16).view(np.float32), bits
    bits = f.astype(np.float16).view(np.uint16)
    return bits.view(np.float16).astype(np.float32), bits


def _reference(np, dY, Wt, a, pad):
    cpg, kpg = a.C // a.G, a.K // a.G
    big = a.Y
    dYp = np.pad(dY, ((0, 0), (big, big), (big, big), (0, 0)))
    dX = np.zeros((a.N, a.H, a.W, a.C), dtype=np.float32)
    for r in range(a.Y):
        for s in range(a.Y):
            win = dYp[
                :,
                pad - r + big : pad - r + big + a.H,
                pad - s + big : pad - s + big + a.W,
                :,
            ]
            for g in range(a.G):
                ks = slice(g * kpg, (g + 1) * kpg)
                dX[..., g * cpg : (g + 1) * cpg] += np.einsum(
                    "nhwk,kc->nhwc", win[..., ks], Wt[ks, r, s, :], optimize=True
                )
    return dX


def _bad_count(np, out_bits, ref, dtype):
    tol = 1e-2
    if dtype == "bf16":
        D = (out_bits.astype(np.uint32) << 16).view(np.float32)
        u = ref.astype(np.float32).view(np.uint32)
        u = u + np.uint32(0x7FFF) + ((u >> 16) & 1).astype(np.uint32)
        R = (u & np.uint32(0xFFFF0000)).view(np.float32)
    else:
        D = out_bits.view(np.float16).astype(np.float32)
        R = ref.astype(np.float32)
    return int((~(np.abs(D - R) <= tol + tol * np.abs(R))).sum())


def main(argv=None) -> int:
    a = _parse_args(argv)
    if not a.verify and not a.time:
        sys.modules["torch"] = (
            None  # keep torch's HIP runtime out of a profiled process
        )
    import numpy as np
    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

    from dispatch.grouped_convolution import (
        ConvGroupedRequest,
        _problem,
        dispatch_conv_grouped,
    )
    from kernels.common.conv_implicit_gemm_dgrad import (
        build_implicit_gemm_conv_dgrad,
        pack_sub_gemm_buffer,
    )

    pad = (a.Y - 1) // 2 if a.pad is None else a.pad
    req = ConvGroupedRequest(
        N=a.N,
        C=a.C,
        K=a.K,
        Hi=a.H,
        Wi=a.W,
        Y=a.Y,
        X=a.Y,
        G=a.G,
        pad_h=pad,
        pad_w=pad,
        dtype=a.dtype,
        arch=a.arch,
        direction="dgrad",
    )
    r = dispatch_conv_grouped(req)
    print("\n".join(r.explanation))
    print(f"spec: {r.spec}")

    rt = Runtime()
    ho, wo = a.H + 2 * pad - a.Y + 1, a.W + 2 * pad - a.Y + 1
    dY32, dYb = _rand_bits(np, (a.N, ho, wo, a.K), a.dtype, 1)
    W32, Wb = _rand_bits(np, (a.K, a.Y, a.Y, a.C // a.G), a.dtype, 2)
    nbytes = {"dY": dYb.nbytes, "W": Wb.nbytes, "dX": a.N * a.H * a.W * a.C * 2}
    calls = []
    if hasattr(r.spec, "launch_plan"):
        # Imported here so the driver still runs (on the igemm pick) against a
        # checkout that predates the direct-MFMA dgrad candidate.
        from kernels.common.conv_direct_grouped import direct_mfma_dgrad_stage_kernel

        plan = r.spec.launch_plan(req)
        nbytes.update(dict(plan.buffer_bytes))
        ptr = {role: rt.alloc(nb) for role, nb in nbytes.items()}
        for st in plan.stages:
            art = compile_kernel(
                direct_mfma_dgrad_stage_kernel(st, arch=a.arch), arch=a.arch
            )
            print(
                f"stage {st.role:10s} {art.kernel_name} grid={st.grid} block={st.block}"
            )
            if st.b is None:
                sig = _PREPASS_SIG
                vals = {
                    "A": ptr[st.a],
                    "D": ptr[st.d],
                    "A_bytes": nbytes[st.a],
                    "D_bytes": nbytes[st.d],
                }
            else:
                sig = conv_args_signature(a.dtype)
                vals = {
                    "A": ptr[st.a],
                    "B": ptr[st.b],
                    "D": ptr[st.d],
                    "A_bytes": nbytes[st.a],
                    "B_bytes": nbytes[st.b],
                    "D_bytes": nbytes[st.d],
                }
            launcher = KernelLauncher(
                hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
            )
            calls.append((launcher, vals, LaunchConfig(grid=st.grid, block=st.block)))
    elif not hasattr(r.spec, "to_dgrad_spec"):
        # Depthwise pick: one windowed kernel whose instance spec carries the
        # grid. Imported here for the same reason as the direct plan above.
        from kernels.common.conv_direct_grouped import (
            build_direct_depthwise_dgrad_windowed,
        )

        ptr = {role: rt.alloc(nb) for role, nb in nbytes.items()}
        art = compile_kernel(
            build_direct_depthwise_dgrad_windowed(r.spec.instance, arch=a.arch),
            arch=a.arch,
        )
        print(f"depthwise {art.kernel_name} grid={r.grid} block={r.block}")
        vals = {
            "A": ptr["dY"],
            "B": ptr["W"],
            "D": ptr["dX"],
            "A_bytes": nbytes["dY"],
            "B_bytes": nbytes["W"],
            "D_bytes": nbytes["dX"],
        }
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=conv_args_signature(a.dtype),
        )
        calls.append((launcher, vals, LaunchConfig(grid=r.grid, block=r.block)))
    else:
        inst = r.spec.to_dgrad_spec(_problem(req))
        if inst.needs_atomic:
            print("igemm pick needs a zeroed dX per launch; not supported here")
            return 2
        ptr = {role: rt.alloc(nb) for role, nb in nbytes.items()}
        sgs = inst.compute_sub_gemms()
        packed = pack_sub_gemm_buffer(sgs, inst.tile_m, inst.tile_n)
        buf = struct.pack(f"{len(packed)}i", *packed)
        sg = rt.alloc(len(buf))
        rt.memcpy_h2d(sg, (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf), len(buf))
        art = compile_kernel(
            build_implicit_gemm_conv_dgrad(inst, arch=a.arch), arch=a.arch
        )
        print(f"igemm {art.kernel_name} grid={r.grid} block={r.block}")
        sig = conv_args_signature(a.dtype) + [
            {"name": "sub_gemm_buf", "type": "ptr<i32, global>", "size_bytes": 8},
            {"name": "num_sub_gemms", "type": "i32", "size_bytes": 4},
        ]
        vals = {
            "A": ptr["dY"],
            "B": ptr["W"],
            "D": ptr["dX"],
            "A_bytes": nbytes["dY"],
            "B_bytes": nbytes["W"],
            "D_bytes": nbytes["dX"],
            "sub_gemm_buf": sg,
            "num_sub_gemms": len(sgs),
        }
        launcher = KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
        )
        calls.append((launcher, vals, LaunchConfig(grid=r.grid, block=r.block)))
    for role, arr in (("dY", dYb), ("W", Wb)):
        rt.memcpy_h2d(
            ptr[role],
            (ctypes.c_uint8 * arr.nbytes).from_address(arr.ctypes.data),
            arr.nbytes,
        )

    def run():
        for launcher, vals, cfg in calls:
            launcher(vals, config=cfg)

    rc = 0
    if a.verify:
        for role, nb in nbytes.items():
            if role not in ("dY", "W"):
                rt.memset(ptr[role], 0xFF, nb)
        run()
        synchronize_and_release(0)
        out = np.empty((a.N, a.H, a.W, a.C), dtype=np.uint16)
        rt.memcpy_d2h(
            (ctypes.c_uint8 * out.nbytes).from_address(out.ctypes.data),
            ptr["dX"],
            out.nbytes,
        )
        bad = _bad_count(np, out, _reference(np, dY32, W32, a, pad), a.dtype)
        print(f"VERIFY {'PASS' if bad == 0 else 'FAIL'} bad={bad}/{out.size}")
        rc = 0 if bad == 0 else 1
    if a.time:
        ms = [
            time_launches(run, warmup=3, iters=a.iters, stream=0) for _ in range(a.reps)
        ]
        print(
            f"TIME median_ms={statistics.median(ms):.5f} (reps={a.reps}, iters={a.iters})"
        )
    if a.loop:
        with no_fence():
            for _ in range(a.loop):
                run()
            rt.sync()
        print(f"LOOP done {a.loop}")
    for d in ptr.values():
        rt.free(d)
    return rc


if __name__ == "__main__":
    sys.exit(main())
