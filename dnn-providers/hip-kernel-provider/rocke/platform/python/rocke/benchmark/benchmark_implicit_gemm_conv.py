# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tile/pipeline sweep benchmark for implicit-GEMM convolution (gfx950, gfx1250).

Supports both the forward pass (NHWC × KYXC → NHWK) and the backward-weight
(wgrad) pass (dY × X → dW).  Select with ``--direction fwd`` (default) or
``--direction wgrad``.

Builds every valid combination of tile / warp / pipeline / epilogue parameters,
runs each on GPU, and reports the best configuration ranked by TFLOPS.

Swept dimensions:
  tile_m, tile_n : 16, 32, 64, 128, 256
  tile_k         : 16, 32, 64, 128
  warp_m, warp_n : 1, 2, 4, 8
  warp_tile_m == warp_tile_n : 16, 32
  pipeline       : mem, compv3, compv4
  epilogue       : default, cshuffle

warp_tile_k is chosen as the largest valid K for the target MFMA atom
(same policy as bake_off_implicit_gemm.py).

Run (forward):
  python benchmark_implicit_gemm_conv.py \\
      --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --Y 3 --X 3 \\
      --dtype fp16 --top 10

Run (wgrad):
  python benchmark_implicit_gemm_conv.py \\
      --direction wgrad \\
      --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --Y 3 --X 3 \\
      --dtype fp16 --top 10

Shape / dtype parameters mirror bake_off_implicit_gemm.py exactly.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from dataclasses import dataclass
from typing import List

# Suppress the "fell back to Python lowerer" warning — expected in environments
# where the C++ engine extension is not built.
os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

# ---------------------------------------------------------------------------
# Swept parameter grids
# ---------------------------------------------------------------------------

_TILE_MN = (16, 32, 64, 128, 256)
_TILE_MN_GFX1250 = (16, 32, 64, 128, 256, 512)
_TILE_K = (16, 32, 64, 128)
_WARP_MN = (1, 2, 4, 8)
_WARP_MN_GFX1250 = (1, 2, 4, 8, 16)
_WARP_TILE_MN = (16, 32)
_PIPELINES = ("mem", "compv3", "compv4")
_EPILOGUES = ("default", "cshuffle")
# Split-K degrees swept when --split-k 0 (auto) is passed for wgrad.
_SPLIT_K_AUTO = (1, 2, 4, 8, 16, 32, 64, 128)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class Result:
    kernel_name: str
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_tile_mn: int
    warp_tile_k: int
    pipeline: str
    epilogue: str
    split_k: int
    ms: float
    tflops: float
    gbps: float
    passed: bool | None = None  # None when --verify was not requested


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_for_spec(spec, p):
    """Derive launch grid from spec and problem."""
    M = p.M
    N_gemm = p.N_gemm
    gx = (N_gemm + spec.tile_n - 1) // spec.tile_n
    gy = (M + spec.tile_m - 1) // spec.tile_m
    # grid_order="NM": x=N-tile, y=M-tile (mirrors bake_off_implicit_gemm)
    return (gx, gy, p.groups)


def _grid_for_wgrad_spec(spec, split_k: int):
    """Derive launch grid from wgrad spec and split-K degree."""
    tile_m, tile_n = spec.tile_m, spec.tile_n
    gx = (spec.wg_N + tile_n - 1) // tile_n
    gy = (spec.wg_M + tile_m - 1) // tile_m
    return (gx, gy, split_k)


def _sample_combos(combos: list, frac: float, seed: int) -> list:
    """Return a random subset of *combos* of size ceil(frac * len(combos))."""
    n = max(1, round(len(combos) * frac))
    rng = random.Random(seed)
    return rng.sample(combos, min(n, len(combos)))


def _verify_kernel(
    *,
    rt,
    launcher,
    values: dict,
    grid: tuple,
    block: tuple,
    out_dev,
    out_t,
    zero_init_out: bool,
    ref_out,
    kernel_name: str,
    dump_fail: "str | None",
    extra_tensors: "dict | None" = None,
    u8,
) -> bool:
    """Launch a kernel, compare against reference, optionally dump on failure.

    Parameters
    ----------
    zero_init_out:
        Zero the output buffer before launching (required for split-K atomic
        accumulation; not needed for direct-store kernels).
    dump_fail:
        Directory path to dump tensors into on first failure, or ``None`` to
        just print the error.
    extra_tensors:
        Additional {name: tensor} pairs saved alongside out/ref/diff when
        ``dump_fail`` is set (e.g. ``{"dY": dY_t, "X": X_t}`` for wgrad).

    Returns
    -------
    tuple[bool, bool]
        ``(stop, passed)`` — ``stop`` is ``True`` if a dump was triggered and
        the sweep should abort; ``passed`` is ``True`` if the kernel output
        matched the reference within tolerance.
    """
    import torch

    if zero_init_out:
        rt.memset(out_dev, 0, out_t.nbytes)

    from rocke.runtime.launcher import LaunchConfig

    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    out_cpu = torch.empty_like(out_t)
    rt.memcpy_d2h(u8(out_cpu), out_dev, out_t.nbytes)

    out_f32 = out_cpu.float().cuda()
    abs_diff = out_f32.sub(ref_out).abs()
    ref_scale = ref_out.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    # Peak-normalised relative error: max|out-ref| / max|ref|.
    # Caveat: a large relative error on a small-magnitude weight can be masked
    # by the global-max denominator, and 5e-2 is fairly loose for bf16
    # reductions over K_wg ~ 25k.  A mean/L2 relative check or a tighter bf16
    # bound would catch subtler reduction bugs -- revisit when verify is
    # re-enabled after the fwd fixes in #9824.
    tol = 5e-2 if out_t.dtype in (torch.float16, torch.bfloat16) else 1e-3
    err = rel_err
    status = "PASS" if err < tol else f"FAIL(rel_err={err:.2e})"
    print(f"  verify {kernel_name}: {status}", flush=True)

    if err >= tol and dump_fail:
        import pathlib
        import numpy as np

        dump_dir = pathlib.Path(dump_fail)
        dump_dir.mkdir(parents=True, exist_ok=True)
        diff = out_f32.sub(ref_out)

        def _save(name, t):
            arr = t.cpu().numpy()
            np.savetxt(
                dump_dir / f"{kernel_name}_{name}.txt",
                arr.flatten(),
                fmt="%.6f",
            )

        _save("out", out_f32)
        _save("ref", ref_out)
        _save("diff", diff)
        for name, tensor in (extra_tensors or {}).items():
            _save(name, tensor.float())

        max_idx = int(diff.argmax())
        unravel = np.unravel_index(max_idx, diff.shape)
        print(
            f"  [dump] saved to {dump_dir}/  "
            f"max_diff={err:.4e} at index {unravel} (flat {max_idx})\n"
            f"  [dump] out={float(out_f32.flatten()[max_idx]):.6f}  "
            f"ref={float(ref_out.flatten()[max_idx]):.6f}",
            flush=True,
        )
        return True, False  # dump triggered → stop the sweep; kernel failed

    return False, err < tol


# ---------------------------------------------------------------------------
# MIOpen driver command parser
# ---------------------------------------------------------------------------

_MIOPEN_DTYPE_MAP = {
    "conv": "fp32",
    "convfp16": "fp16",
    "convbfp16": "bf16",
    "convint8": "fp16",  # int8 not supported; fall back to fp16 and warn
}


def parse_miopen_cmd(cmd: str):
    """Parse a MIOpenDriver command string into a ``(ConvProblem, dtype)`` tuple.

    Accepts the full command line (including the binary path and driver name),
    e.g.::

        ./bin/MIOpenDriver convfp16 -n 8 -c 3 -H 224 -W 224 -k 64 \\
            -y 11 -x 11 -p 2 -q 2 -u 4 -v 4 -l 1 -j 1 -m conv -g 1 -F 1 \\
            -t 1 -in_layout=NHWC

    Only forward-pass (2-D NHWC) convolutions are supported; the function
    raises ``ValueError`` for unsupported cases (3-D, NCHW).
    Returns ``(problem, dtype)`` where ``dtype`` is ``"fp16"``, ``"bf16"``,
    or ``"fp32"``.
    """
    import shlex

    tokens = shlex.split(cmd)

    # Strip binary path (anything before the driver keyword).
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
    if driver_kw == "convint8":
        print(
            f"[warn] convint8 is not supported by this benchmark; " f"treating as fp16",
            file=sys.stderr,
        )

    # Re-parse the tokens after the driver keyword using argparse.
    sub = argparse.ArgumentParser(add_help=False)
    sub.add_argument("-n", "--n", dest="N", type=int, default=1)
    sub.add_argument("-c", "--c", dest="C", type=int, default=1)
    sub.add_argument("-H", "--H", dest="Hi", type=int, default=1)
    sub.add_argument("-W", "--W", dest="Wi", type=int, default=1)
    sub.add_argument("-k", "--k", dest="K", type=int, default=1)
    sub.add_argument("-y", "--y", dest="Y", type=int, default=1)
    sub.add_argument("-x", "--x", dest="X", type=int, default=1)
    sub.add_argument("-p", "--p", dest="pH", type=int, default=0)
    sub.add_argument("-q", "--q", dest="pW", type=int, default=0)
    sub.add_argument("-u", "--u", dest="sH", type=int, default=1)
    sub.add_argument("-v", "--v", dest="sW", type=int, default=1)
    sub.add_argument("-l", "--l", dest="dH", type=int, default=1)
    sub.add_argument("-j", "--j", dest="dW", type=int, default=1)
    sub.add_argument("-g", "--g", dest="groups", type=int, default=1)
    sub.add_argument("-F", "--F", dest="forw", type=int, default=1)
    sub.add_argument(
        "-in_layout", "--in_layout", dest="in_layout", type=str, default="NHWC"
    )
    # Ignored flags — consumed to avoid parse errors.
    sub.add_argument("-m", "--m", dest="_mode", type=str, default="conv")
    sub.add_argument("-t", "--t", dest="_time", type=int, default=0)
    sub.add_argument("-V", "--V", dest="_verify", type=int, default=1)
    sub.add_argument("-_", "--_", dest="_spatial_dim", type=int, default=2)

    miopen_args, _ = sub.parse_known_args(tokens[driver_idx + 1 :])

    layout = miopen_args.in_layout.upper()
    if layout not in ("NHWC", "NWC"):
        raise ValueError(
            f"Layout {layout!r} is not supported; only NHWC/NWC inputs are accepted"
        )

    from rocke.instances.common.conv_implicit_gemm import ConvProblem

    problem = ConvProblem(
        N=miopen_args.N,
        Hi=miopen_args.Hi,
        Wi=miopen_args.Wi,
        C=miopen_args.C,
        K=miopen_args.K,
        Y=miopen_args.Y,
        X=miopen_args.X,
        sH=miopen_args.sH,
        sW=miopen_args.sW,
        pH=miopen_args.pH,
        pW=miopen_args.pW,
        dH=miopen_args.dH,
        dW=miopen_args.dW,
        groups=miopen_args.groups,
    )
    return problem, dtype


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tile/pipeline sweep benchmark for implicit-GEMM conv (fwd + wgrad)"
    )
    parser.add_argument(
        "--direction",
        default="fwd",
        choices=["fwd", "wgrad"],
        help="convolution direction: forward (fwd) or backward-weight (wgrad) (default: fwd)",
    )
    parser.add_argument(
        "--arch",
        default="gfx950",
        help="gfx target (gfx942, gfx950, gfx1250, ...) (default: gfx950)",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="data type (default: fp16)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="print top-N results ranked by TFLOPS (default: 10)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="timed iterations (default: 10)"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "randomly sample FRAC of the candidate combinations before sweeping "
            "(e.g. 0.1 for ~10%%). Uses a fixed seed (--seed) for reproducibility."
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
        help="verify first valid kernel against torch reference before sweep",
    )
    parser.add_argument(
        "--dump-fail",
        default=None,
        metavar="PATH",
        dest="dump_fail",
        help=(
            "on the first verify FAIL, save kernel output, reference, and "
            "abs-diff tensors to PATH/{kernel_name}_{out,ref,diff}.txt and stop "
            "the sweep. Implies --verify."
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
            "initialise X and dY (wgrad) / A and B (fwd) to a constant value "
            "instead of random. Defaults to 1.0 when given without a value. "
            "With ones the expected dW[k,y,x,c] = N * valid_spatial_count(y,x) "
            "— a simple integer pattern easy to verify by eye or compare exactly."
        ),
    )
    parser.add_argument(
        "--split-k",
        type=int,
        default=-1,
        dest="split_k",
        metavar="N",
        help=(
            "wgrad split-K degree: "
            "0 = sweep all degrees in %(auto)s, "
            "1 = disabled, "
            ">1 = fixed degree, "
            "-1 = auto (CK formula per tile config)"
        ).replace("%(auto)s", str(list(_SPLIT_K_AUTO))),
    )

    miopen_grp = parser.add_argument_group(
        "MIOpen input",
        "Load the conv problem from a MIOpenDriver command instead of explicit shape flags. "
        "When set, ConvProblem and dtype are derived from the command; "
        "--dtype / shape flags are ignored.",
    )
    miopen_grp.add_argument(
        "--miopen-cmd",
        default=None,
        metavar="CMD",
        help="MIOpenDriver command string, e.g. "
        '"./MIOpenDriver convfp16 -n 8 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 '
        '-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -in_layout=NHWC"',
    )
    miopen_grp.add_argument(
        "--miopen-file",
        default=None,
        metavar="FILE",
        help="Path to a file containing one MIOpenDriver command per line; "
        "the benchmark is run once per line (blank lines and # comments ignored).",
    )

    conv = parser.add_argument_group("ConvProblem", "convolution shape parameters")
    conv.add_argument("--N", type=int, default=8, help="batch size")
    conv.add_argument("--Di", type=int, default=None, help="input depth (3-D only)")
    conv.add_argument("--Hi", type=int, default=56, help="input height")
    conv.add_argument("--Wi", type=int, default=56, help="input width")
    conv.add_argument("--C", type=int, default=64, help="input channels")
    conv.add_argument("--K", type=int, default=64, help="output channels / filters")
    conv.add_argument("--Z", type=int, default=None, help="filter depth (3-D only)")
    conv.add_argument("--Y", type=int, default=3, help="filter height")
    conv.add_argument("--X", type=int, default=3, help="filter width")
    conv.add_argument("--sD", type=int, default=None, help="depth stride (3-D only)")
    conv.add_argument("--sH", type=int, default=1, help="vertical stride")
    conv.add_argument("--sW", type=int, default=1, help="horizontal stride")
    conv.add_argument("--pD", type=int, default=None, help="depth padding (3-D only)")
    conv.add_argument("--pH", type=int, default=1, help="vertical padding")
    conv.add_argument("--pW", type=int, default=1, help="horizontal padding")
    conv.add_argument("--dD", type=int, default=None, help="depth dilation (3-D only)")
    conv.add_argument("--dH", type=int, default=1, help="vertical dilation")
    conv.add_argument("--dW", type=int, default=1, help="horizontal dilation")
    conv.add_argument(
        "--groups",
        "-g",
        type=int,
        default=1,
        help="number of conv groups; C and K must each be divisible by groups (default: 1)",
    )

    args = parser.parse_args()

    if args.miopen_cmd is None and args.miopen_file is None:
        if args.Di is not None and args.Z is None:
            print("--Z (filter depth) is required when --Di is set", file=sys.stderr)
            return 2
        if args.Z is not None and args.Di is None:
            print("--Di (input depth) is required when --Z is set", file=sys.stderr)
            return 2

    import ctypes

    from rocke import compile_kernel
    from rocke.core.arch import ArchTarget
    from rocke.instances.common.conv_implicit_gemm import (
        ConvDataSpec,
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
        is_valid_spec,
        is_valid_spec_for_problem,
    )
    from rocke.instances.common.conv_implicit_gemm_wgrad import (
        WgradConvSpec,
        build_implicit_gemm_conv_wgrad,
        is_valid_wgrad_spec,
    )
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    arch = args.arch
    target = ArchTarget.from_gfx(arch)

    # Build the list of (problem, dtype) cases to sweep.
    cases: list  # List[Tuple[ConvProblem, str]]
    if args.miopen_file is not None:
        path = args.miopen_file
        lines = open(path).readlines()
        cases = []
        for lineno, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                prob, dt = parse_miopen_cmd(line)
                cases.append((prob, dt))
            except ValueError as e:
                print(f"[warn] {path}:{lineno}: skipping — {e}", file=sys.stderr)
    elif args.miopen_cmd is not None:
        prob, dt = parse_miopen_cmd(args.miopen_cmd)
        cases = [(prob, dt)]
    else:
        problem = ConvProblem(
            N=args.N,
            Di=args.Di,
            Hi=args.Hi,
            Wi=args.Wi,
            C=args.C,
            K=args.K,
            Z=args.Z,
            Y=args.Y,
            X=args.X,
            sD=args.sD,
            sH=args.sH,
            sW=args.sW,
            pD=args.pD,
            pH=args.pH,
            pW=args.pW,
            dD=args.dD,
            dH=args.dH,
            dW=args.dW,
            groups=args.groups,
        )
        cases = [(problem, args.dtype)]

    all_rc = 0
    for case_idx, (problem, dtype) in enumerate(cases):
        if len(cases) > 1:
            print(f"\n{'#'*72}", flush=True)
            print(
                f"# Case {case_idx + 1}/{len(cases)}: {problem.short()} "
                f"dtype={dtype} direction={args.direction}",
                flush=True,
            )
            print(f"{'#'*72}", flush=True)

        _common = dict(
            args=args,
            problem=problem,
            dtype=dtype,
            arch=arch,
            target=target,
            compile_kernel=compile_kernel,
            ConvDataSpec=ConvDataSpec,
            synchronize_and_release=synchronize_and_release,
            time_launches=time_launches,
            Runtime=Runtime,
            KernelLauncher=KernelLauncher,
            LaunchConfig=LaunchConfig,
            u8=_u8,
        )

        if args.direction == "wgrad":
            rc = _run_wgrad_sweep(
                **_common,
                WgradConvSpec=WgradConvSpec,
                build_implicit_gemm_conv_wgrad=build_implicit_gemm_conv_wgrad,
                is_valid_wgrad_spec=is_valid_wgrad_spec,
            )
        else:
            rc = _run_sweep(
                **_common,
                ImplicitGemmConvSpec=ImplicitGemmConvSpec,
                build_implicit_gemm_conv=build_implicit_gemm_conv,
                is_valid_spec_for_problem=is_valid_spec_for_problem,
            )
        all_rc = all_rc or rc
    return all_rc


def _run_sweep(
    *,
    args,
    problem,
    dtype: str,
    arch: str,
    target,
    compile_kernel,
    ConvDataSpec,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    is_valid_spec_for_problem,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
    **_ignored,  # absorbs keys from _common not used by this sweep
) -> int:
    import torch
    from rocke.helpers.manifest import conv_args_signature

    _u8 = u8

    p = problem

    _torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    torch.manual_seed(42)

    def _make(*shape):
        return (
            torch.full(shape, args.debug_init)
            if args.debug_init is not None
            else torch.empty(*shape).uniform_(-1.0, 1.0)
        )

    if p.is_3d:
        _A_f32 = _make(p.N, p.Di, p.Hi, p.Wi, p.C)
        _B_f32 = _make(p.K, p.Z, p.Y, p.X, p.C)
        D_t = torch.empty(p.N, p.Do, p.Ho, p.Wo, p.K, dtype=_torch_dtype)
    else:
        _A_f32 = _make(p.N, p.Hi, p.Wi, p.C)
        _B_f32 = _make(p.K, p.Y, p.X, p.C)
        D_t = torch.empty(p.N, p.Ho, p.Wo, p.K, dtype=_torch_dtype)
    A_t = _A_f32.to(_torch_dtype)
    B_t = _B_f32.to(_torch_dtype)

    bytes_xfer = float(A_t.nbytes + B_t.nbytes + D_t.nbytes)
    flop = float(p.flops)

    sig = conv_args_signature(dtype)

    _mma_family = "wmma" if target.wave_size == 32 else "mma"

    # Early check: does the target have any MMA atom for this dtype?
    # A wave32/WMMA target may lack an atom for a given dtype (e.g. a future
    # target without fp32 WMMA), so bail with a clear message rather than
    # silently sweeping everything and reporting "No valid configurations".
    if (
        target.mma.select_largest_k(
            family=_mma_family,
            a_dtype=dtype,
            b_dtype=dtype,
            c_dtype="fp32",
            m=16,
            n=16,
        )
        is None
    ):
        print(
            f"error: {arch} has no {dtype} MMA atom — "
            f"{dtype} convolution is not supported on this target.",
            file=sys.stderr,
        )
        return 2

    _tile_mn = _TILE_MN_GFX1250 if arch == "gfx1250" else _TILE_MN
    _warp_mn = _WARP_MN_GFX1250 if arch == "gfx1250" else _WARP_MN
    combos = list(
        itertools.product(
            _tile_mn,
            _tile_mn,
            _TILE_K,
            _warp_mn,
            _warp_mn,
            _WARP_TILE_MN,
            _PIPELINES,
            _EPILOGUES,
        )
    )

    if args.sample is not None:
        total = len(combos)
        combos = _sample_combos(combos, args.sample, args.seed)
        print(
            f"Sampling {len(combos)}/{total} combinations "
            f"({args.sample*100:.0f}%, seed={args.seed}).",
            flush=True,
        )

    print(
        f"Sweeping {len(combos)} combinations for {arch} {dtype} {p.short()} ...",
        flush=True,
    )

    rt = Runtime()
    results: List[Result] = []
    n_built = 0
    n_skipped = 0

    # Upload inputs once; reuse across all kernels.
    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    ref_out: torch.Tensor | None = None
    if args.verify:
        from rocke.benchmark.conv_reference import (
            conv_reference,
            conv_reference_gfx1250,
        )

        if arch == "gfx1250" and not p.is_3d:
            ref_out = conv_reference_gfx1250(A_t, B_t, p, out_dtype=_torch_dtype).cuda()
            print(
                f"Reference computed via gfx1250 hand-written conv "
                f"({tuple(ref_out.shape)}, {ref_out.dtype}).",
                flush=True,
            )
        else:
            ref_out = conv_reference(A_t, B_t, p, out_dtype=_torch_dtype)
            print(
                f"Reference computed via torch ({tuple(ref_out.shape)}, {ref_out.dtype}).",
                flush=True,
            )

    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_tile_mn,
        pipeline,
        epilogue,
    ) in combos:
        atom = target.mma.select_largest_k(
            family=_mma_family,
            a_dtype=dtype,
            b_dtype=dtype,
            c_dtype="fp32",
            m=warp_tile_mn,
            n=warp_tile_mn,
            k_max=tile_k,
        )
        if atom is None:
            n_skipped += 1
            continue

        warp_tile_k = atom.k
        spec = ImplicitGemmConvSpec(
            problem=problem,
            name="rocke_bench_igemm_conv",
            data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_tile_m=warp_tile_mn,
            warp_tile_n=warp_tile_mn,
            warp_tile_k=warp_tile_k,
            wave_size=target.wave_size,
            pipeline=pipeline,
            epilogue=epilogue,
            groups=p.groups,
        )

        ok, _ = is_valid_spec_for_problem(spec, problem, arch)
        if not ok:
            n_skipped += 1
            continue

        try:
            kernel = build_implicit_gemm_conv(spec, arch=arch)
        except ValueError:
            n_skipped += 1
            continue

        artifact = compile_kernel(kernel, arch=arch)
        n_built += 1

        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
        grid = _grid_for_spec(spec, p)
        block = (spec.block_size, 1, 1)
        stream = 0

        values = {
            "A": A_dev,
            "B": B_dev,
            "D": D_dev,
            "A_bytes": A_t.nbytes,
            "B_bytes": B_t.nbytes,
            "D_bytes": D_t.nbytes,
        }
        cfg = LaunchConfig(grid=grid, block=block, stream=stream)

        # Verify every kernel against the pre-computed reference (when --verify).
        kernel_passed: bool | None = None
        if args.verify or args.dump_fail:
            stopped, kernel_passed = _verify_kernel(
                rt=rt,
                launcher=launcher,
                values=values,
                grid=grid,
                block=block,
                out_dev=D_dev,
                out_t=D_t,
                zero_init_out=False,
                ref_out=ref_out,
                kernel_name=artifact.kernel_name,
                dump_fail=args.dump_fail,
                u8=_u8,
            )
            if stopped:
                rt.free(A_dev)
                rt.free(B_dev)
                rt.free(D_dev)
                return 1
            rt.memset(D_dev, 0, D_t.nbytes)

        ms = time_launches(
            lambda: launcher(values, config=cfg),
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
        )
        synchronize_and_release(stream)

        cur_tflops = (flop / ms) * 1e-9
        cur_gbps = (bytes_xfer / ms) * 1e-6

        results.append(
            Result(
                kernel_name=artifact.kernel_name,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_mn=warp_tile_mn,
                warp_tile_k=warp_tile_k,
                pipeline=pipeline,
                epilogue=epilogue,
                split_k=1,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
                passed=kernel_passed,
            )
        )

        print(
            f"[{n_built:4d}] tile={tile_m}x{tile_n}x{tile_k} "
            f"warp={warp_m}x{warp_n} "
            f"atom={warp_tile_mn}x{warp_tile_mn}x{warp_tile_k} "
            f"{pipeline}/{epilogue:9s} "
            f"{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    # Free GPU buffers.
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)

    print(
        f"\nSweep done: {n_built} built, {n_skipped} skipped.",
        flush=True,
    )

    if not results:
        print("No valid configurations found.", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r.tflops, reverse=True)
    top_n = min(args.top, len(results))

    show_verify = args.verify
    width = 84 if show_verify else 72
    print(f"\n{'='*width}")
    print(f"Top {top_n} configurations for {arch} {dtype} {p.short()}")
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
            f"tile={r.tile_m}x{r.tile_n}x{r.tile_k} "
            f"warp={r.warp_m}x{r.warp_n} "
            f"atom={r.warp_tile_mn}x{r.warp_tile_mn}x{r.warp_tile_k} "
            f"{r.pipeline}/{r.epilogue}"
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


def _run_wgrad_sweep(
    *,
    args,
    problem,
    dtype: str,
    arch: str,
    target,
    compile_kernel,
    ConvDataSpec,
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
    is_valid_wgrad_spec,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
    **_ignored,
) -> int:
    """Sweep wgrad configurations and rank by TFLOPS.

    Wgrad GEMM dims:
        M    = K            (output channels — weight rows)
        N_wg = Y*X*C        (filter spatial × input channel — weight cols)
        K_wg = N*Ho*Wo      (output positions — reduction)

    Operands:
        A (dY): output gradient, shape (N, Ho, Wo, K)
        B (X):  input activations, shape (N, Hi, Wi, C)
        D (dW): weight gradient, shape (K, Y, X, C)

    Split-K (``--split-k``):
        1        — disabled (normal epilogue, z-grid = 1).
        >1       — fixed degree; dW is zero-initialised before each launch,
                   kernel atomic-adds partials, result is final dW.
        0 (auto) — sweep all degrees in _SPLIT_K_AUTO.
    """
    import torch
    from rocke.helpers.manifest import conv_args_signature

    _u8 = u8
    p = problem

    _torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    torch.manual_seed(42)

    def _make(*shape):
        return (
            torch.full(shape, args.debug_init)
            if args.debug_init is not None
            else torch.empty(*shape).uniform_(-1.0, 1.0)
        )

    if p.is_3d:
        _X_f32 = _make(p.N, p.Di, p.Hi, p.Wi, p.C)
        _dY_f32 = _make(p.N, p.Do, p.Ho, p.Wo, p.K)
        dW_t = torch.empty(p.K, p.Z, p.Y, p.X, p.C, dtype=_torch_dtype)
    else:
        _X_f32 = _make(p.N, p.Hi, p.Wi, p.C)
        _dY_f32 = _make(p.N, p.Ho, p.Wo, p.K)
        dW_t = torch.empty(p.K, p.Y, p.X, p.C, dtype=_torch_dtype)

    X_t = _X_f32.to(_torch_dtype)
    dY_t = _dY_f32.to(_torch_dtype)

    bytes_xfer = float(dY_t.nbytes + X_t.nbytes + dW_t.nbytes)
    flop = float(p.flops)

    sig = conv_args_signature(dtype)

    # split_k degrees to sweep:
    #   0   → sweep all degrees in _SPLIT_K_AUTO
    #  -1   → one combo per tile config, degree resolved by CK formula per build
    #  else → single fixed degree
    split_k_values = _SPLIT_K_AUTO if args.split_k == 0 else (args.split_k,)

    combos = list(
        itertools.product(
            _TILE_MN,
            _TILE_MN,
            _TILE_K,
            _WARP_MN,
            _WARP_MN,
            _WARP_TILE_MN,
            _PIPELINES,
            _EPILOGUES,
            split_k_values,
        )
    )

    if args.sample is not None:
        total = len(combos)
        combos = _sample_combos(combos, args.sample, args.seed)
        print(
            f"Sampling {len(combos)}/{total} wgrad combinations "
            f"({args.sample*100:.0f}%, seed={args.seed}).",
            flush=True,
        )

    _spk_label = {0: "sweep", -1: "auto(CK)"}.get(args.split_k, str(args.split_k))
    print(
        f"Sweeping {len(combos)} wgrad combinations for {arch} {dtype} {p.short()} "
        f"(split_k={_spk_label}) ...",
        flush=True,
    )

    rt = Runtime()
    results: List[Result] = []
    n_built = 0
    n_skipped = 0

    dY_dev = rt.alloc(dY_t.nbytes)
    X_dev = rt.alloc(X_t.nbytes)
    dW_dev = rt.alloc(dW_t.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY_t), dY_t.nbytes)
    rt.memcpy_h2d(X_dev, _u8(X_t), X_t.nbytes)
    rt.memset(dW_dev, 0, dW_t.nbytes)

    ref_out: torch.Tensor | None = None
    if args.verify or args.dump_fail:
        from rocke.benchmark.conv_reference import wgrad_reference

        ref_out = wgrad_reference(_X_f32, _dY_f32, p)
        print(
            f"Wgrad reference computed ({tuple(ref_out.shape)}, {ref_out.dtype}).",
            flush=True,
        )

    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_tile_mn,
        pipeline,
        epilogue,
        split_k,
    ) in combos:
        # split_k > 1 requires dtype_d in fp32/bf16/fp16 (atomic accumulation).
        # For split_k == 1 any dtype is fine; skip invalid split_k > 1 + cshuffle.
        if split_k > 1 and epilogue == "cshuffle":
            n_skipped += 1
            continue

        atom = target.mma.select_largest_k(
            a_dtype=dtype,
            b_dtype=dtype,
            c_dtype="fp32",
            m=warp_tile_mn,
            n=warp_tile_mn,
            k_max=tile_k,
        )
        if atom is None:
            n_skipped += 1
            continue

        warp_tile_k = atom.k
        # Resolve split_k=-1 once here so the concrete value is used for the
        # spec, the grid z-axis, the timing loop, and the result record.
        if split_k == -1:
            from rocke.helpers.split_k import select_split_k_wgrad

            resolved_split_k = select_split_k_wgrad(
                wg_M=problem.kpg,
                wg_N=problem.Y * problem.X * problem.cpg,
                wg_K=problem.N * problem.Ho * problem.Wo,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                arch=arch,
            ).split_k
        else:
            resolved_split_k = split_k

        spec = WgradConvSpec(
            problem=problem,
            name="rocke_bench_igemm_wgrad",
            data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_tile_m=warp_tile_mn,
            warp_tile_n=warp_tile_mn,
            warp_tile_k=warp_tile_k,
            pipeline=pipeline,
            epilogue=epilogue,
            split_k=resolved_split_k,
        )

        ok, reason = is_valid_wgrad_spec(spec, arch)
        if not ok:
            n_skipped += 1
            continue

        try:
            kernel = build_implicit_gemm_conv_wgrad(spec, arch=arch)
        except ValueError:
            n_skipped += 1
            continue

        artifact = compile_kernel(kernel, arch=arch)
        n_built += 1

        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
        grid = _grid_for_wgrad_spec(spec, resolved_split_k)
        block = (spec.block_size, 1, 1)
        stream = 0

        values = {
            "A": dY_dev,
            "B": X_dev,
            "D": dW_dev,
            "A_bytes": dY_t.nbytes,
            "B_bytes": X_t.nbytes,
            "D_bytes": dW_t.nbytes,
        }
        cfg = LaunchConfig(grid=grid, block=block, stream=stream)

        if args.verify or args.dump_fail:
            stopped, _ = _verify_kernel(
                rt=rt,
                launcher=launcher,
                values=values,
                grid=grid,
                block=block,
                out_dev=dW_dev,
                out_t=dW_t,
                # Split-K atomic accumulation requires a zero-init buffer.
                zero_init_out=(resolved_split_k > 1),
                ref_out=ref_out,
                kernel_name=artifact.kernel_name,
                dump_fail=args.dump_fail,
                extra_tensors={"dY": dY_t, "X": X_t},
                u8=_u8,
            )
            if stopped:
                rt.free(dY_dev)
                rt.free(X_dev)
                rt.free(dW_dev)
                return 1

        # For split_k > 1 the timed loop must zero-init dW before every launch
        # so the atomic accumulation starts from zero each iteration.
        if resolved_split_k > 1:

            def _launch_spk():
                rt.memset(dW_dev, 0, dW_t.nbytes)
                launcher(values, config=cfg)

            timed_fn = _launch_spk
        else:
            timed_fn = lambda: launcher(values, config=cfg)

        ms = time_launches(
            timed_fn,
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
        )
        synchronize_and_release(stream)

        cur_tflops = (flop / ms) * 1e-9
        cur_gbps = (bytes_xfer / ms) * 1e-6

        results.append(
            Result(
                kernel_name=artifact.kernel_name,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_mn=warp_tile_mn,
                warp_tile_k=warp_tile_k,
                pipeline=pipeline,
                epilogue=epilogue,
                split_k=resolved_split_k,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
            )
        )

        print(
            f"[{n_built:4d}] tile={tile_m}x{tile_n}x{tile_k} "
            f"warp={warp_m}x{warp_n} "
            f"atom={warp_tile_mn}x{warp_tile_mn}x{warp_tile_k} "
            f"{pipeline}/{epilogue:9s} spk{resolved_split_k:<3d} "
            f"{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    rt.free(dY_dev)
    rt.free(X_dev)
    rt.free(dW_dev)

    print(f"\nWgrad sweep done: {n_built} built, {n_skipped} skipped.", flush=True)

    if not results:
        print("No valid wgrad configurations found.", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r.tflops, reverse=True)
    top_n = min(args.top, len(results))

    print(f"\n{'='*72}")
    print(f"Top {top_n} wgrad configurations for {arch} {dtype} {p.short()}")
    print(f"{'='*72}")
    hdr = f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
    print(hdr)
    print("-" * 72)
    for rank, r in enumerate(results[:top_n], 1):
        cfg_str = (
            f"tile={r.tile_m}x{r.tile_n}x{r.tile_k} "
            f"warp={r.warp_m}x{r.warp_n} "
            f"atom={r.warp_tile_mn}x{r.warp_tile_mn}x{r.warp_tile_k} "
            f"{r.pipeline}/{r.epilogue} spk{r.split_k}"
        )
        print(f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {cfg_str}")

    best = results[0]
    print(f"\nBest: {best.tflops:.1f} TFLOPS — {best.kernel_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
