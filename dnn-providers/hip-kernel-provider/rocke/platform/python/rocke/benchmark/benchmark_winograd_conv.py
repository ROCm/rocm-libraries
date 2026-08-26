# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tile sweep benchmark for Winograd convolution (gfx942, gfx950).

Builds all valid combinations of Winograd block-tiling parameters for a given
3×3, stride=1 convolution problem, runs each configuration on GPU, and reports
the best configuration ranked by effective TFLOPS (relative to the naïve
direct convolution FLOP count).

Swept dimensions:
  out_tile       : 2 (F(2,3) — 4×4 transform domain)
                   4 (F(4,3) — 6×6 transform domain, higher FLOP reduction)
  block_c        : channels processed per block in data/filter transforms
  block_k        : output channels per block in filter/output transforms
  block_nhw      : (n, tile_h, tile_w) triples per block

Multi-pass pipeline:
  1. Data transform kernel   (B^T × input_patch × B)
  2. Filter transform kernel (G × filter × G^T)    — once, cached
  3. GEMM                    (batched element-wise × in xform domain)
  4. Output transform kernel (A^T × acc_tile × A)

Run:
  python benchmark_winograd_conv.py \\
      --N 8 --Hi 56 --Wi 56 --C 64 --K 64 \\
      --dtype fp16 --top 10

Shape / general parameters mirror benchmark_implicit_gemm_conv.py exactly;
  parameters only applicable to implicit-GEMM (tile_m/n/k, warp_m/n,
  pipeline, epilogue, split_k, direction, 3-D, groups, stride > 1, dilation
  > 1) are omitted.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import sys
from dataclasses import dataclass
from typing import List

os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

# ---------------------------------------------------------------------------
# Swept parameter grids
# ---------------------------------------------------------------------------

_OUT_TILES = (2, 4)
_BLOCK_C = (16, 32, 64, 128)
_BLOCK_K = (16, 32, 64, 128)
_BLOCK_NHW = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class Result:
    kernel_name: str
    out_tile: int
    block_c: int
    block_k: int
    block_nhw: int
    phase: str  # "data", "filter", "output", "total"
    ms: float
    tflops: float
    gbps: float
    passed: bool | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_combos(combos: list, frac: float, seed: int) -> list:
    import random

    n = max(1, round(len(combos) * frac))
    rng = random.Random(seed)
    return rng.sample(combos, min(n, len(combos)))


def _verify_result(
    *,
    rt,
    out_dev,
    out_t,
    ref_out,
    kernel_name: str,
    arch: str,
    u8,
    dump_fail: "str | None",
) -> tuple:
    import torch

    out_cpu = torch.empty_like(out_t)
    rt.memcpy_d2h(u8(out_cpu), out_dev, out_t.nbytes)

    if arch == "gfx1250":
        out_f32 = out_cpu.float()
        abs_diff = out_f32.sub(ref_out.cpu()).abs()
        ref_scale = ref_out.cpu().abs().max().clamp(min=1.0)
    else:
        out_f32 = out_cpu.float().cuda()
        abs_diff = out_f32.sub(ref_out).abs()
        ref_scale = ref_out.abs().max().clamp(min=1.0)

    rel_err = float(abs_diff.max() / ref_scale)
    tol = 5e-2
    status = "PASS" if rel_err < tol else f"FAIL(rel_err={rel_err:.2e})"
    print(f"  verify {kernel_name}: {status}", flush=True)

    if rel_err >= tol and dump_fail:
        import pathlib
        import numpy as np

        dump_dir = pathlib.Path(dump_fail)
        dump_dir.mkdir(parents=True, exist_ok=True)
        diff = out_f32.sub(ref_out)

        def _save(name, t):
            arr = t.cpu().numpy()
            np.savetxt(
                dump_dir / f"{kernel_name}_{name}.txt", arr.flatten(), fmt="%.6f"
            )

        _save("out", out_f32)
        _save("ref", ref_out)
        _save("diff", diff)
        max_idx = int(diff.abs().argmax())
        print(
            f"  [dump] saved to {dump_dir}/  max_diff={rel_err:.4e} at flat idx {max_idx}",
            flush=True,
        )
        return True, False

    return False, rel_err < tol


def _winograd_sig(kern) -> list:
    """Build a KernelLauncher-compatible signature from a KernelDef's params.

    Each param contributes a dict with "name", "type", and "size_bytes" so
    that ``pack_args`` can match values by name and pack them correctly.
    Uses ``p.type.name`` (e.g. ``"ptr<f16,global>"``, ``"i32"``) which is
    the same format ``conv_args_signature`` produces by hand.
    """
    _type_bytes = {"i32": 4, "i64": 8}

    def _size(type_name: str) -> int:
        if type_name.startswith("ptr"):
            return 8
        return _type_bytes.get(type_name, 4)

    return [
        {"name": p.name, "type": p.type.name, "size_bytes": _size(p.type.name)}
        for p in kern.params
    ]


def _compile_one_winograd(args_tuple):
    """Top-level picklable worker for ProcessPoolExecutor.

    Must be at module level so pickle can locate it by name.
    """
    k, arch = args_tuple
    from rocke import compile_kernel as _ck

    return k.name, _ck(k, arch=arch)


# ---------------------------------------------------------------------------
# MIOpen command parsing (winograd subset: 3×3, s=1, d=1, g=1, fp16 only)
# ---------------------------------------------------------------------------

_MIOPEN_DTYPE_MAP = {
    "conv": "fp32",
    "convfp16": "fp16",
    "convbfp16": "bf16",
    "convint8": "fp16",
}


def parse_miopen_cmd(cmd: str):
    """Parse a MIOpenDriver command into a ``(WinogradProblem, dtype)`` tuple.

    Raises ``ValueError`` for configs Winograd cannot handle:
    non-3×3 filter, stride > 1, dilation > 1, groups > 1, or non-fp16 dtype.
    """
    tokens = shlex.split(cmd)
    driver_kw = None
    driver_idx = None
    for i, t in enumerate(tokens):
        key = t.split("/")[-1].lower()
        if key in _MIOPEN_DTYPE_MAP:
            driver_kw = key
            driver_idx = i
            break
    if driver_kw is None:
        raise ValueError(
            f"No MIOpenDriver keyword found in command "
            f"(expected one of: {list(_MIOPEN_DTYPE_MAP)})"
        )
    dtype = _MIOPEN_DTYPE_MAP[driver_kw]
    if dtype != "fp16":
        raise ValueError(
            f"Winograd benchmark only supports fp16; got dtype={dtype!r} "
            f"from driver keyword {driver_kw!r}"
        )

    sub = argparse.ArgumentParser(add_help=False)
    sub.add_argument("-n", "--n", dest="N", type=int, default=1)
    sub.add_argument("-c", "--c", dest="C", type=int, default=1)
    sub.add_argument("-H", "--H", dest="Hi", type=int, default=1)
    sub.add_argument("-W", "--W", dest="Wi", type=int, default=1)
    sub.add_argument("-k", "--k", dest="K", type=int, default=1)
    sub.add_argument("-y", "--y", dest="Y", type=int, default=3)
    sub.add_argument("-x", "--x", dest="X", type=int, default=3)
    sub.add_argument("-p", "--p", dest="pH", type=int, default=1)
    sub.add_argument("-q", "--q", dest="pW", type=int, default=1)
    sub.add_argument("-u", "--u", dest="sH", type=int, default=1)
    sub.add_argument("-v", "--v", dest="sW", type=int, default=1)
    sub.add_argument("-l", "--l", dest="dH", type=int, default=1)
    sub.add_argument("-j", "--j", dest="dW", type=int, default=1)
    sub.add_argument("-g", "--g", dest="groups", type=int, default=1)
    sub.add_argument("-F", "--F", dest="forw", type=int, default=1)
    sub.add_argument(
        "-in_layout", "--in_layout", dest="in_layout", type=str, default="NHWC"
    )
    sub.add_argument("-m", "--m", dest="_mode", type=str, default="conv")
    sub.add_argument("-t", "--t", dest="_time", type=int, default=0)
    sub.add_argument("-V", "--V", dest="_verify", type=int, default=1)
    sub.add_argument("-_", "--_", dest="_spatial_dim", type=int, default=2)

    ma, _ = sub.parse_known_args(tokens[driver_idx + 1 :])

    if ma.Y != 3 or ma.X != 3:
        raise ValueError(f"Winograd requires a 3×3 filter; got Y={ma.Y} X={ma.X}")
    if ma.sH != 1 or ma.sW != 1:
        raise ValueError(f"Winograd requires stride=1; got sH={ma.sH} sW={ma.sW}")
    if ma.dH != 1 or ma.dW != 1:
        raise ValueError(f"Winograd requires dilation=1; got dH={ma.dH} dW={ma.dW}")
    if ma.groups != 1:
        raise ValueError(f"Winograd requires groups=1; got groups={ma.groups}")

    from rocke.instances.common.conv_winograd import WinogradProblem

    problem = WinogradProblem(
        N=ma.N, Hi=ma.Hi, Wi=ma.Wi, C=ma.C, K=ma.K, pH=ma.pH, pW=ma.pW
    )
    return problem, dtype


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Block-tile sweep benchmark for Winograd conv (3×3, stride=1)"
    )
    parser.add_argument(
        "--arch",
        default="gfx950",
        help="gfx target (gfx942, gfx950, ...) (default: gfx950)",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16"],
        help="data type — Winograd currently supports fp16 only (default: fp16)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="print top-N results ranked by total TFLOPS (default: 10)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="timed iterations (default: 10)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "number of parallel compile workers (default: 1, serial). "
            "Set to 0 to use os.cpu_count() workers."
        ),
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "randomly sample FRAC of the candidate combinations before sweeping "
            "(e.g. 0.1 for ~10%%). Uses --seed for reproducibility."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used by --sample (default: 0)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify output against torch reference (F.conv2d) before sweep",
    )
    parser.add_argument(
        "--dump-fail",
        default=None,
        metavar="PATH",
        dest="dump_fail",
        help=(
            "on the first verify FAIL, save output/ref/diff tensors to PATH/ "
            "and stop the sweep. Implies --verify."
        ),
    )
    parser.add_argument(
        "--debug-init",
        nargs="?",
        const=1.0,
        default=None,
        type=float,
        dest="debug_init",
        metavar="VALUE",
        help=(
            "initialise input and filter to a constant instead of random. "
            "Defaults to 1.0 when given without a value."
        ),
    )

    # Winograd-specific tiling sweep
    winograd_grp = parser.add_argument_group(
        "Winograd tiling", "Block-tile dimensions swept by the benchmark."
    )
    winograd_grp.add_argument(
        "--out-tile",
        type=int,
        default=None,
        choices=[2, 4],
        dest="out_tile",
        help=("fix the output tile size (2=F(2,3), 4=F(4,3)). " "Default: sweep both."),
    )
    winograd_grp.add_argument(
        "--block-c",
        type=int,
        default=None,
        dest="block_c",
        help="fix block_c (channels per block). Default: sweep %(choices)s."
        % {"choices": list(_BLOCK_C)},
    )
    winograd_grp.add_argument(
        "--block-k",
        type=int,
        default=None,
        dest="block_k",
        help="fix block_k (output channels per block). Default: sweep %(choices)s."
        % {"choices": list(_BLOCK_K)},
    )
    winograd_grp.add_argument(
        "--block-nhw",
        type=int,
        default=None,
        dest="block_nhw",
        help="fix block_nhw ((n,tile_h,tile_w) triples per block). Default: sweep %(choices)s."
        % {"choices": list(_BLOCK_NHW)},
    )

    # ConvProblem shape — identical subset to benchmark_implicit_gemm_conv.py
    conv = parser.add_argument_group(
        "WinogradProblem",
        "Convolution shape (stride=1, dilation=1, 3×3 filter assumed).",
    )
    conv.add_argument("--N", type=int, default=8, help="batch size")
    conv.add_argument("--Hi", type=int, default=56, help="input height")
    conv.add_argument("--Wi", type=int, default=56, help="input width")
    conv.add_argument("--C", type=int, default=64, help="input channels")
    conv.add_argument("--K", type=int, default=64, help="output channels / filters")
    conv.add_argument("--pH", type=int, default=1, help="vertical padding (0 or 1)")
    conv.add_argument("--pW", type=int, default=1, help="horizontal padding (0 or 1)")

    miopen_grp = parser.add_argument_group(
        "MIOpen input",
        "Load the conv problem from a MIOpenDriver command instead of explicit shape flags. "
        "Only 3×3 / stride=1 / dilation=1 / groups=1 / fp16 commands are accepted. "
        "When set, WinogradProblem and dtype are derived from the command; "
        "--dtype / shape flags are ignored.",
    )
    miopen_grp.add_argument(
        "--miopen-cmd",
        default=None,
        metavar="CMD",
        dest="miopen_cmd",
        help="MIOpenDriver command string, e.g. "
        '"./MIOpenDriver convfp16 -n 8 -c 64 -H 56 -W 56 -k 64 '
        '-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1"',
    )
    miopen_grp.add_argument(
        "--miopen-file",
        default=None,
        metavar="FILE",
        dest="miopen_file",
        help="Path to a file containing one MIOpenDriver command per line; "
        "the benchmark is run for each valid line (blank lines and # comments ignored).",
    )

    args = parser.parse_args()

    if args.dump_fail:
        args.verify = True

    import ctypes

    from rocke import compile_kernel
    from rocke.instances.common.conv_winograd import (
        WinogradConvSpec,
        WinogradProblem,
        build_winograd_data_transform,
        build_winograd_filter_transform,
        build_winograd_gemm,
        build_winograd_output_transform,
        is_valid_spec,
        winograd_data_transform_grid,
        winograd_filter_transform_grid,
        winograd_gemm_grid,
        winograd_output_transform_grid,
    )
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    arch = args.arch

    # Resolve problem(s): miopen-cmd / miopen-file override explicit shape flags.
    if args.miopen_cmd is not None:
        try:
            problem, _ = parse_miopen_cmd(args.miopen_cmd)
        except ValueError as e:
            print(f"error: --miopen-cmd: {e}", file=sys.stderr)
            return 2
        problems = [problem]
    elif args.miopen_file is not None:
        problems = []
        with open(args.miopen_file) as _f:
            for _lineno, _line in enumerate(_f, 1):
                _line = _line.strip()
                if not _line or _line.startswith("#"):
                    continue
                try:
                    _prob, _ = parse_miopen_cmd(_line)
                    problems.append(_prob)
                except ValueError as e:
                    print(
                        f"[warn] {args.miopen_file}:{_lineno}: skipping — {e}",
                        file=sys.stderr,
                    )
        if not problems:
            print(
                f"error: {args.miopen_file}: no valid Winograd-compatible cases found",
                file=sys.stderr,
            )
            return 2
    else:
        try:
            problem = WinogradProblem(
                N=args.N,
                Hi=args.Hi,
                Wi=args.Wi,
                C=args.C,
                K=args.K,
                pH=args.pH,
                pW=args.pW,
            )
        except ValueError as e:
            print(f"error: invalid problem: {e}", file=sys.stderr)
            return 2
        problems = [problem]

    def _sweep(problem):
        # Build sweep grid
        out_tiles = (args.out_tile,) if args.out_tile is not None else _OUT_TILES
        block_cs = (args.block_c,) if args.block_c is not None else _BLOCK_C
        block_ks = (args.block_k,) if args.block_k is not None else _BLOCK_K
        block_nhws = (args.block_nhw,) if args.block_nhw is not None else _BLOCK_NHW

        combos = list(itertools.product(out_tiles, block_cs, block_ks, block_nhws))

        if args.sample is not None:
            total = len(combos)
            combos = _sample_combos(combos, args.sample, args.seed)
            print(
                f"Sampling {len(combos)}/{total} combinations "
                f"({args.sample*100:.0f}%, seed={args.seed}).",
                flush=True,
            )

        p = problem
        print(
            f"Sweeping {len(combos)} combinations for {arch} {args.dtype} "
            f"N{p.N}H{p.Hi}W{p.Wi}C{p.C}K{p.K} pH{p.pH} pW{p.pW} ...",
            flush=True,
        )

        # -----------------------------------------------------------------------
        # Phase 1: validate and build IR for all combos
        # -----------------------------------------------------------------------
        valid_combos = []
        for combo in combos:
            out_tile, block_c, block_k, block_nhw = combo
            try:
                spec = WinogradConvSpec(
                    problem=problem,
                    out_tile=out_tile,
                    block_c=block_c,
                    block_k=block_k,
                    block_nhw=block_nhw,
                )
            except ValueError:
                continue
            ok, _ = is_valid_spec(spec, arch)
            if not ok:
                continue
            valid_combos.append((combo, spec))

        if not valid_combos:
            print("No valid configurations found.", file=sys.stderr)
            return 1

        print(f"Building IR for {len(valid_combos)} valid combos ...", flush=True)

        from rocke.core.arch import ArchTarget

        _wave = ArchTarget.from_gfx(arch).wave_size
        pending = []
        n_skipped = 0
        for combo, spec in valid_combos:
            try:
                kd = build_winograd_data_transform(spec, arch=arch)
                kf = build_winograd_filter_transform(spec, arch=arch)
                kg = build_winograd_gemm(spec, arch=arch)
                ko = build_winograd_output_transform(spec, arch=arch)
            except (ValueError, Exception):
                n_skipped += 1
                continue
            pending.append((combo, spec, kd, kf, kg, ko))

        if not pending:
            print("All configs failed IR build.", file=sys.stderr)
            return 1

        # -----------------------------------------------------------------------
        # Phase 2: compile
        # -----------------------------------------------------------------------
        def _compile_parallel(kernels, jobs):
            from concurrent.futures import ProcessPoolExecutor, as_completed

            unique = {k.name: k for k in kernels}
            if not unique:
                return {}

            artifact_map = {}
            if jobs == 1:
                for name, k in unique.items():
                    artifact_map[name] = compile_kernel(k, arch=arch)
                return artifact_map

            max_workers = os.cpu_count() if jobs == 0 else jobs
            work = [(k, arch) for k in unique.values()]
            print(
                f"Compiling {len(unique)} kernels with {max_workers} workers ...",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_compile_one_winograd, item): item[0].name
                    for item in work
                }
                done = 0
                for fut in as_completed(futures):
                    name, artifact = fut.result()
                    artifact_map[name] = artifact
                    done += 1
                    if done % max(1, len(unique) // 10) == 0 or done == len(unique):
                        print(f"  compiled {done}/{len(unique)}", flush=True)
            return artifact_map

        all_kernels = []
        for _, _, kd, kf, kg, ko in pending:
            all_kernels.extend([kd, kf, kg, ko])

        artifact_map = _compile_parallel(all_kernels, args.jobs)

        # -----------------------------------------------------------------------
        # Phase 3: GPU run
        # -----------------------------------------------------------------------
        import torch

        _torch_dtype = {"fp16": torch.float16}[args.dtype]
        torch.manual_seed(42)

        def _make(*shape):
            if args.debug_init is not None:
                return torch.full(shape, args.debug_init, dtype=_torch_dtype)
            return torch.empty(*shape, dtype=_torch_dtype).uniform_(-1.0, 1.0)

        A_t = _make(p.N, p.Hi, p.Wi, p.C)  # NHWC input
        W_t = _make(p.K, 3, 3, p.C)  # KYXC filter (3×3)
        D_t = torch.empty(p.N, p.Ho, p.Wo, p.K, dtype=_torch_dtype)

        flop = float(p.flops)
        bytes_xfer = float(A_t.nbytes + W_t.nbytes + D_t.nbytes)

        rt = Runtime()
        A_dev = rt.alloc(A_t.nbytes)
        W_dev = rt.alloc(W_t.nbytes)
        D_dev = rt.alloc(D_t.nbytes)
        rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
        rt.memcpy_h2d(W_dev, _u8(W_t), W_t.nbytes)
        rt.memset(D_dev, 0, D_t.nbytes)

        ref_out = None
        if args.verify:
            import torch.nn.functional as F

            A_nchw = A_t.float().permute(0, 3, 1, 2).contiguous()
            W_kcyx = W_t.float().permute(0, 3, 1, 2).contiguous()  # KYXC -> KCYX
            ref_nchw = F.conv2d(A_nchw, W_kcyx, padding=p.pH)
            ref_out = ref_nchw.permute(0, 2, 3, 1).contiguous().to(_torch_dtype)
            if arch != "gfx1250":
                ref_out = ref_out.cuda()
            print(
                f"Reference computed via torch F.conv2d "
                f"({tuple(ref_out.shape)}, {ref_out.dtype}).",
                flush=True,
            )

        results: List[Result] = []
        n_run = 0
        _INT32_MAX = 2**31 - 1

        for combo, spec, kd, kf, kg, ko in pending:
            out_tile, block_c, block_k, block_nhw = combo

            xs = spec.xform_size
            ntotal = p.N * spec.num_tiles
            dws_bytes = xs * xs * ntotal * p.C * 2
            fws_bytes = xs * xs * p.K * p.C * 2
            gws_bytes = xs * xs * ntotal * p.K * 2

            if max(dws_bytes, fws_bytes, gws_bytes) > _INT32_MAX:
                print(
                    f"[skip] out_tile={out_tile} bc={block_c} bk={block_k} bnhw={block_nhw}: "
                    "workspace exceeds int32 range",
                    flush=True,
                )
                n_skipped += 1
                continue

            DataWs_dev = rt.alloc(dws_bytes)
            FilterWs_dev = rt.alloc(fws_bytes)
            GemmWs_dev = rt.alloc(gws_bytes)
            rt.memset(DataWs_dev, 0, dws_bytes)
            rt.memset(FilterWs_dev, 0, fws_bytes)
            rt.memset(GemmWs_dev, 0, gws_bytes)

            loaded = {}
            skip = False
            for phase_name, kern in [
                ("data", kd),
                ("filter", kf),
                ("gemm", kg),
                ("output", ko),
            ]:
                artifact = artifact_map.get(kern.name)
                if artifact is None:
                    skip = True
                    break
                try:
                    sig = _winograd_sig(kern)
                    launcher = KernelLauncher(
                        hsaco=artifact.hsaco,
                        kernel_name=artifact.kernel_name,
                        signature=sig,
                    )
                    loaded[phase_name] = (launcher, artifact.kernel_name)
                except (HipError, Exception) as e:
                    print(
                        f"[skip] kernel load failed for {kern.name}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    skip = True
                    break

            if skip:
                rt.free(DataWs_dev)
                rt.free(FilterWs_dev)
                rt.free(GemmWs_dev)
                n_skipped += 1
                continue

            data_grid = winograd_data_transform_grid(spec)
            filter_grid = winograd_filter_transform_grid(spec)
            gemm_gx, gemm_gy, gemm_gz = winograd_gemm_grid(spec)
            output_grid = winograd_output_transform_grid(spec)

            block_data = (block_nhw * block_c, 1, 1)
            block_filt = (block_k * block_c, 1, 1)
            block_gemm = (spec.gemm_warp_m * spec.gemm_warp_n * _wave, 1, 1)
            block_output = (block_nhw * block_k, 1, 1)

            stride_a = ntotal * p.C
            stride_b = p.K * p.C
            stride_c = ntotal * p.K

            data_vals = {
                "A": A_dev,
                "A_bytes": A_t.nbytes,
                "DataWs": DataWs_dev,
                "DataWs_bytes": dws_bytes,
            }
            filter_vals = {
                "W": W_dev,
                "W_bytes": W_t.nbytes,
                "FilterWs": FilterWs_dev,
                "FilterWs_bytes": fws_bytes,
            }
            gemm_vals = {
                "A": DataWs_dev,
                "B": FilterWs_dev,
                "C": GemmWs_dev,
                "M": ntotal,
                "N": p.K,
                "K": p.C,
                "stride_a": stride_a,
                "stride_b": stride_b,
                "stride_c": stride_c,
            }
            output_vals = {
                "GemmWs": GemmWs_dev,
                "GemmWs_bytes": gws_bytes,
                "D": D_dev,
                "D_bytes": D_t.nbytes,
            }

            data_launcher, data_kname = loaded["data"]
            filter_launcher, filter_kname = loaded["filter"]
            gemm_launcher, gemm_kname = loaded["gemm"]
            output_launcher, output_kname = loaded["output"]

            cfg_data = LaunchConfig(grid=data_grid, block=block_data, fence=True)
            cfg_filter = LaunchConfig(grid=filter_grid, block=block_filt, fence=True)
            cfg_gemm = LaunchConfig(
                grid=(gemm_gx, gemm_gy, gemm_gz), block=block_gemm, fence=True
            )
            cfg_output = LaunchConfig(grid=output_grid, block=block_output, fence=True)

            stream = 0

            ms_data = time_launches(
                lambda: data_launcher(data_vals, config=cfg_data),
                warmup=args.warmup,
                iters=args.iters,
                stream=stream,
            )
            ms_filter = time_launches(
                lambda: filter_launcher(filter_vals, config=cfg_filter),
                warmup=args.warmup,
                iters=args.iters,
                stream=stream,
            )
            ms_gemm = time_launches(
                lambda: gemm_launcher(gemm_vals, config=cfg_gemm),
                warmup=args.warmup,
                iters=args.iters,
                stream=stream,
            )
            ms_output = time_launches(
                lambda: output_launcher(output_vals, config=cfg_output),
                warmup=args.warmup,
                iters=args.iters,
                stream=stream,
            )
            synchronize_and_release(stream)

            ms_total = ms_data + ms_filter + ms_gemm + ms_output
            tflops_total = (flop / ms_total) * 1e-9
            gbps_total = (bytes_xfer / ms_total) * 1e-6

            kernel_passed: bool | None = None
            if args.verify or args.dump_fail:
                stopped, kernel_passed = _verify_result(
                    rt=rt,
                    out_dev=D_dev,
                    out_t=D_t,
                    ref_out=ref_out,
                    kernel_name=f"winograd_output_f{out_tile}x3",
                    arch=arch,
                    u8=_u8,
                    dump_fail=args.dump_fail,
                )
                if stopped:
                    rt.free(DataWs_dev)
                    rt.free(FilterWs_dev)
                    rt.free(GemmWs_dev)
                    rt.free(A_dev)
                    rt.free(W_dev)
                    rt.free(D_dev)
                    return 1
                rt.memset(D_dev, 0, D_t.nbytes)

            n_run += 1
            results.append(
                Result(
                    kernel_name=f"winograd_f{out_tile}x3_bc{block_c}_bk{block_k}_bnhw{block_nhw}",
                    out_tile=out_tile,
                    block_c=block_c,
                    block_k=block_k,
                    block_nhw=block_nhw,
                    phase="total",
                    ms=ms_total,
                    tflops=tflops_total,
                    gbps=gbps_total,
                    passed=kernel_passed,
                )
            )

            print(
                f"[{n_run:4d}] f{out_tile}x3 bc={block_c:3d} bk={block_k:3d} bnhw={block_nhw} "
                f"data={ms_data:.2f}ms filt={ms_filter:.2f}ms "
                f"gemm={ms_gemm:.2f}ms out={ms_output:.2f}ms "
                f"total={ms_total:.2f}ms  {tflops_total:6.1f} TFLOPS",
                flush=True,
            )

            rt.free(DataWs_dev)
            rt.free(FilterWs_dev)
            rt.free(GemmWs_dev)

        rt.free(A_dev)
        rt.free(W_dev)
        rt.free(D_dev)

        print(f"\nSweep done: {n_run} run, {n_skipped} skipped.", flush=True)

        if not results:
            print("No valid configurations found.", file=sys.stderr)
            return 1

        results.sort(key=lambda r: r.tflops, reverse=True)
        top_n = min(args.top, len(results))
        show_verify = args.verify

        width = 96 if show_verify else 82
        print(f"\n{'='*width}")
        print(
            f"Top {top_n} configurations for {arch} {args.dtype} "
            f"N{p.N}H{p.Hi}W{p.Wi}C{p.C}K{p.K}"
        )
        print(f"{'='*width}")
        hdr = (
            f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  {'verify':>6}  config"
            if show_verify
            else f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
        )
        print(hdr)
        print("-" * width)
        for rank, r in enumerate(results[:top_n], 1):
            cfg_str = (
                f"f{r.out_tile}x3 bc={r.block_c} bk={r.block_k} bnhw={r.block_nhw}"
            )
            if show_verify:
                v = "PASS" if r.passed else "FAIL"
                print(
                    f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {v:>6}  {cfg_str}"
                )
            else:
                print(
                    f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {cfg_str}"
                )

        best = results[0]
        print(f"\nBest: {best.tflops:.1f} TFLOPS — {best.kernel_name}")
        return 0

    overall_rc = 0
    for problem in problems:
        rc = _sweep(problem)
        if rc != 0:
            overall_rc = rc
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
