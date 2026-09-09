# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build, launch and time ONE dgrad implicit-GEMM config.

Both LDS-layout case studies end their capture command with
``-- python3 <single-config driver>``; this is that driver. The sweep driver
(``rocke.benchmark.benchmark_implicit_gemm_conv``) is the wrong tool for a trace:
it compiles thousands of kernels and rocprofv3 would have to decode every
dispatch. Here exactly one kernel runs, so ``--kernel-regex`` matches one thing.

Deliberately does NO numeric verification. torch's HIP runtime and rocke's fight
over the process HIP context, and timings come out multiples wrong when a verify
and a timing loop share a process -- verify in a separate run (the sweep driver's
``--verify``, or tests/instances/test_conv_dgrad_correctness.py).

``--print-name-only`` builds and compiles but never touches the GPU queue, which
is how you get the ``--kernel-regex`` for a capture without perturbing it.

Usage:
    python3 run_one_dgrad.py --kouter on --stride 2
    python3 run_one_dgrad.py --tile-m 128 --print-name-only

    ROCPROF_TRACE_DECODER_LIB=<dir with librocprof-trace-decoder.so> \\
    python3 ../../../../../dsl_docs/optimization/utilities/tools/wavescope/\\
capture_wavescope_trace.py --output-dir ./att_out \\
        --kernel-regex "rocke_trace_dgrad.*kouter" \\
        -- python3 run_one_dgrad.py --kouter on --stride 2 --warmup 1 --iters 2
"""

import argparse
import ctypes
import struct
import sys

_DEFAULT_CMD = (
    "./MIOpenDriver convbfp16 -n 8 -c 128 -H 32 -W 32 -k 128 -y 3 -x 3 "
    "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -in_layout=NHWC"
)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--dtype", default="bf16", choices=("fp16", "bf16", "fp32"))
    ap.add_argument(
        "--miopen-cmd",
        default=_DEFAULT_CMD,
        help="MIOpenDriver command string. -F 2 is backward-data (-F 4 is "
        "backward-weight); there are no per-dimension flags.",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="override -u/-v in --miopen-cmd. stride > 1 takes the tilde "
        "sub-GEMM decomposition, which is a different kernel shape entirely.",
    )
    # Defaults are the shipped dispatch geometry -- see _GFX950_TILE_* and
    # _GFX950_WARP_* in library/dispatch/grouped_convolution.py.
    ap.add_argument("--tile-m", type=int, default=64)
    ap.add_argument("--tile-n", type=int, default=64)
    ap.add_argument("--tile-k", type=int, default=64)
    ap.add_argument("--warp-m", type=int, default=2)
    ap.add_argument("--warp-n", type=int, default=2)
    ap.add_argument("--warp-tile-mn", type=int, default=32)
    ap.add_argument("--pipeline", default="mem")
    ap.add_argument("--epilogue", default="cshuffle")
    ap.add_argument("--split-k", type=int, default=1)
    ap.add_argument(
        "--kouter",
        choices=("on", "off"),
        default="on",
        help="force the K-outer B tile. The dispatch policy deduces this via "
        "DgradConvSpec.default_lds_k_outer, which answers the same way for "
        "every combo of a given shape -- so forcing it is the only way to A/B "
        "the two layouts.",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--name", default="rocke_trace_dgrad")
    ap.add_argument(
        "--print-name-only",
        action="store_true",
        help="build and compile, print the kernel name, exit without launching",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)

    import torch
    from rocke import compile_kernel
    from rocke.benchmark.benchmark_implicit_gemm_conv import parse_miopen_cmd
    from rocke.core.arch import ArchTarget
    from rocke.helpers.manifest import conv_args_signature
    from rocke.instances.common.conv_implicit_gemm import ConvDataSpec
    from rocke.instances.common.conv_implicit_gemm_dgrad import (
        DgradConvSpec,
        build_implicit_gemm_conv_dgrad,
        is_valid_dgrad_spec,
        pack_sub_gemm_buffer,
    )
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    cmd = a.miopen_cmd
    if a.stride is not None:
        import re

        cmd = re.sub(r"-u \d+", f"-u {a.stride}", cmd)
        cmd = re.sub(r"-v \d+", f"-v {a.stride}", cmd)
    p, _dt, _fwd = parse_miopen_cmd(cmd)

    tgt = ArchTarget.from_gfx(a.arch)
    family = "wmma" if tgt.wave_size == 32 else "mma"
    atom = tgt.mma.select_largest_k(
        family=family,
        a_dtype=a.dtype,
        b_dtype=a.dtype,
        c_dtype="fp32",
        m=a.warp_tile_mn,
        n=a.warp_tile_mn,
        k_max=a.tile_k,
    )
    if atom is None:
        print(
            f"no {family} atom for {a.warp_tile_mn} k_max={a.tile_k}", file=sys.stderr
        )
        return 2

    vec_a, vec_b, vec_c = DgradConvSpec.default_vector_sizes(p.cpg, p.kpg, a.dtype)
    spec = DgradConvSpec(
        problem=p,
        name=a.name,
        data=ConvDataSpec(dtype_a=a.dtype, dtype_b=a.dtype, dtype_d=a.dtype),
        tile_m=a.tile_m,
        tile_n=a.tile_n,
        tile_k=a.tile_k,
        warp_m=a.warp_m,
        warp_n=a.warp_n,
        warp_tile_m=a.warp_tile_mn,
        warp_tile_n=a.warp_tile_mn,
        warp_tile_k=atom.k,
        wave_size=tgt.wave_size,
        pipeline=a.pipeline,
        epilogue=a.epilogue,
        split_k=a.split_k,
        lds_k_outer=(a.kouter == "on"),
        vector_size_a=vec_a,
        vector_size_b=vec_b,
        vector_size_c=vec_c,
    )
    ok, why = is_valid_dgrad_spec(spec, a.arch)
    if not ok:
        print(f"invalid spec: {why}", file=sys.stderr)
        return 2
    spec.validate()

    kernel = build_implicit_gemm_conv_dgrad(spec, arch=a.arch)
    artifact = compile_kernel(kernel, arch=a.arch)
    sub_gemms = spec.compute_sub_gemms()
    print(f"KERNEL_NAME {artifact.kernel_name}", flush=True)
    print(
        f"config tile={a.tile_m}x{a.tile_n}x{a.tile_k} warp={a.warp_m}x{a.warp_n} "
        f"atom={a.warp_tile_mn}x{a.warp_tile_mn}x{atom.k} {a.pipeline}/{a.epilogue} "
        f"spk={a.split_k} kouter={a.kouter} sub_gemms={len(sub_gemms)} "
        f"needs_atomic={spec.needs_atomic}",
        flush=True,
    )
    if a.print_name_only:
        return 0

    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[a.dtype]
    torch.manual_seed(42)
    dY_t = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1, 1).to(torch_dtype)
    W_t = torch.empty(p.K, p.Y, p.X, p.cpg).uniform_(-1, 1).to(torch_dtype)
    dX_t = torch.empty(p.N, p.Hi, p.Wi, p.C, dtype=torch_dtype)

    def u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_buffer(
            t.detach().contiguous().view(torch.uint8).reshape(-1).numpy()
        )

    rt = Runtime()
    dY_dev = rt.alloc(dY_t.nbytes)
    W_dev = rt.alloc(W_t.nbytes)
    dX_dev = rt.alloc(dX_t.nbytes)
    rt.memcpy_h2d(dY_dev, u8(dY_t), dY_t.nbytes)
    rt.memcpy_h2d(W_dev, u8(W_t), W_t.nbytes)
    rt.memset(dX_dev, 0, dX_t.nbytes)

    buf_i32 = pack_sub_gemm_buffer(sub_gemms, spec.tile_m, spec.tile_n)
    buf_bytes = struct.pack(f"{len(buf_i32)}i", *buf_i32)
    sgbuf_dev = rt.alloc(len(buf_bytes))
    rt.memcpy_h2d(
        sgbuf_dev,
        (ctypes.c_uint8 * len(buf_bytes)).from_buffer_copy(buf_bytes),
        len(buf_bytes),
    )

    ext_sig = conv_args_signature(a.dtype) + [
        {"name": "sub_gemm_buf", "type": "ptr<i32, global>", "size_bytes": 8},
        {"name": "num_sub_gemms", "type": "i32", "size_bytes": 4},
    ]
    values = {
        "A": dY_dev,
        "B": W_dev,
        "D": dX_dev,
        "A_bytes": dY_t.nbytes,
        "B_bytes": W_t.nbytes,
        "D_bytes": dX_t.nbytes,
        "sub_gemm_buf": sgbuf_dev,
        "num_sub_gemms": len(sub_gemms),
    }
    # Conv group rides blockIdx.y; the tilde sub-GEMM geometry is
    # channel-independent so block_end is per-group.
    grid = (sub_gemms[-1].block_end, max(int(p.groups), 1), a.split_k)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=ext_sig
    )
    cfg = LaunchConfig(grid=grid, block=(spec.launch_block_size, 1, 1), stream=0)

    if spec.needs_atomic:

        def timed():
            rt.memset(dX_dev, 0, dX_t.nbytes)
            launcher(values, config=cfg)

    else:

        def timed():
            launcher(values, config=cfg)

    ms = time_launches(timed, warmup=a.warmup, iters=a.iters, stream=0)
    synchronize_and_release(0)
    print(
        f"grid={grid} block={spec.launch_block_size} ms={ms:.4f} "
        f"tflops={(float(p.flops) / ms) * 1e-9:.1f}",
        flush=True,
    )

    for d in (sgbuf_dev, dY_dev, W_dev, dX_dev):
        rt.free(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
