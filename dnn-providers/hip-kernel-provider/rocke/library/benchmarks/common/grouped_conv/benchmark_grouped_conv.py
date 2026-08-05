# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark for grouped convolution using the rocke dispatcher (fwd + wgrad).

Supports both the forward pass (NHWC × KYXC → NHWK) and the backward-weight
(wgrad) pass (dY × X → dW).  Select with ``--direction fwd`` (default) or
``--direction wgrad``.

Uses ``dispatch.grouped_convolution.conv_grouped_sweep_space`` to obtain the
dispatcher's candidate(s) for the given arch/dtype/shape, then compiles and
times each one on GPU, reporting TFLOPS and ms.

Run (forward):
  python benchmark_grouped_conv.py \\
      --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --Y 3 --X 3 \\
      --dtype fp16

Run (wgrad):
  python benchmark_grouped_conv.py \\
      --direction wgrad \\
      --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --Y 3 --X 3 \\
      --dtype fp16

  # Run from bench_cases_conv.json or bench_cases_conv_bwd.json:
  python benchmark_grouped_conv.py --json-file bench_cases_conv.json
  python benchmark_grouped_conv.py --json-file bench_cases_conv_bwd.json

Shape / dtype parameters mirror bake_off_implicit_gemm.py exactly.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

# Suppress the "fell back to Python lowerer" warning — expected in environments
# where the C++ engine extension is not built.
os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")


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
) -> tuple[bool, bool]:
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
        the benchmark should abort; ``passed`` is ``True`` if the kernel output
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
        return True, False  # dump triggered → abort; kernel failed

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


def parse_json_case(entry: dict):
    """Parse a single JSON benchmark-case entry into a ``(ConvProblem, dtype)`` tuple.

    Supports entries from bench_cases_conv.json and bench_cases_conv_bwd.json.
    Fields: N, Cin, Cout, H, W, Kh, Kw, stride (int or [sH,sW]), pad (int or [pH,pW]),
    dilation (int or [dH,dW]), groups, dtype, layout, Di/D/Kd for 3-D.

    Raises ``ValueError`` for unsupported layouts (NCHW) or unsupported op types.
    """
    op = entry.get("op", "conv2d")
    layout = entry.get("layout", "NHWC").upper()

    if op not in ("conv2d", "conv3d", "conv1d"):
        raise ValueError(f"op={op!r} is not supported")

    dtype = entry.get("dtype", "fp16")
    if dtype not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"dtype={dtype!r} is not supported (only fp16, bf16, fp32)")

    from rocke.instances.common.conv_implicit_gemm import ConvProblem

    def _scalar_or_pair(val, idx_h=0, idx_w=1):
        if isinstance(val, (list, tuple)):
            return int(val[idx_h]), int(val[idx_w])
        return int(val), int(val)

    if op == "conv1d":
        N = int(entry.get("N", 1))
        C = int(entry.get("Cin", entry.get("C", 1)))
        K = int(entry.get("Cout", entry.get("K", 1)))
        L = int(entry.get("L", 1))
        Kl = int(entry.get("Kl", 1))
        stride = int(entry.get("stride", 1))
        pad = int(entry.get("pad", 0))
        dilation = int(entry.get("dilation", 1))
        groups = int(entry.get("groups", 1))
        problem = ConvProblem(
            N=N,
            Hi=1,
            Wi=L,
            C=C,
            K=K,
            Y=1,
            X=Kl,
            sH=1,
            sW=stride,
            pH=0,
            pW=pad,
            dH=1,
            dW=dilation,
            groups=groups,
        )
        return problem, dtype

    if op == "conv3d":
        N = int(entry.get("N", 1))
        C = int(entry.get("Cin", entry.get("C", 1)))
        K = int(entry.get("Cout", entry.get("K", 1)))
        Di = int(entry.get("D", entry.get("Di", 1)))
        Hi = int(entry.get("H", entry.get("Hi", 1)))
        Wi = int(entry.get("W", entry.get("Wi", 1)))
        Kd = int(entry.get("Kd", entry.get("Z", 1)))
        Kh = int(entry.get("Kh", entry.get("Y", 1)))
        Kw = int(entry.get("Kw", entry.get("X", 1)))
        strides = entry.get("strides", entry.get("stride", 1))
        if isinstance(strides, (list, tuple)):
            sD, sH, sW = int(strides[0]), int(strides[1]), int(strides[2])
        else:
            sD = sH = sW = int(strides)
        pads_before = entry.get("pads_before", entry.get("pad", 0))
        if isinstance(pads_before, (list, tuple)):
            pD, pH, pW = int(pads_before[0]), int(pads_before[1]), int(pads_before[2])
        else:
            pD = pH = pW = int(pads_before)
        dilations = entry.get("dilations", entry.get("dilation", 1))
        if isinstance(dilations, (list, tuple)):
            dD, dH, dW = int(dilations[0]), int(dilations[1]), int(dilations[2])
        else:
            dD = dH = dW = int(dilations)
        groups = int(entry.get("groups", 1))
        problem = ConvProblem(
            N=N,
            Di=Di,
            Hi=Hi,
            Wi=Wi,
            C=C,
            K=K,
            Z=Kd,
            Y=Kh,
            X=Kw,
            sD=sD,
            sH=sH,
            sW=sW,
            pD=pD,
            pH=pH,
            pW=pW,
            dD=dD,
            dH=dH,
            dW=dW,
            groups=groups,
        )
        return problem, dtype

    # conv2d
    if layout not in ("NHWC", "NWC"):
        raise ValueError(
            f"layout={layout!r} is not supported; only NHWC/NWC inputs are accepted"
        )
    N = int(entry.get("N", 1))
    C = int(entry.get("Cin", entry.get("C", 1)))
    K = int(entry.get("Cout", entry.get("K", 1)))
    Hi = int(entry.get("H", entry.get("Hi", 1)))
    Wi = int(entry.get("W", entry.get("Wi", 1)))
    Kh = int(entry.get("Kh", entry.get("Y", 1)))
    Kw = int(entry.get("Kw", entry.get("X", 1)))

    stride = entry.get("stride", 1)
    sH, sW = _scalar_or_pair(stride)

    # Support both symmetric pad and pads_before/pads_after
    if "pads_before" in entry:
        pads_before = entry["pads_before"]
        pH = (
            int(pads_before[0])
            if isinstance(pads_before, (list, tuple))
            else int(pads_before)
        )
        pW = (
            int(pads_before[1])
            if isinstance(pads_before, (list, tuple))
            else int(pads_before)
        )
    else:
        pad = entry.get("pad", 0)
        pH, pW = _scalar_or_pair(pad)

    dilation = entry.get("dilation", 1)
    dH, dW = _scalar_or_pair(dilation)

    groups = int(entry.get("groups", 1))

    problem = ConvProblem(
        N=N,
        Hi=Hi,
        Wi=Wi,
        C=C,
        K=K,
        Y=Kh,
        X=Kw,
        sH=sH,
        sW=sW,
        pH=pH,
        pW=pW,
        dH=dH,
        dW=dW,
        groups=groups,
    )
    return problem, dtype


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
# CK Profiler comparison
# ---------------------------------------------------------------------------

_CK_DTYPE_STR = {"fp32": "fp32", "fp16": "fp16", "bf16": "bfp16"}


def _run_ckprofiler(
    problem, dtype: str, ckprofiler: str, converter_script: str, forw: int, verify: int
) -> None:
    """Delegate to convert_miopen_driver_to_profiler.py for the given ConvProblem.

    Builds a synthetic args namespace that mirrors what the converter script's
    argparse produces, then calls its init_const_args / run_ck_profiler functions
    directly — no logic is duplicated here.

    ``forw`` follows MIOpenDriver -F convention:
      0 fwd+bwd_data+bwd_weight   1 fwd   2 bwd_data   4 bwd_weight
      3 fwd+bwd_data   5 fwd+bwd_weight   6 bwd_data+bwd_weight
    """
    import importlib.util
    import types

    ck_dtype_str = _CK_DTYPE_STR.get(dtype)
    if ck_dtype_str is None:
        print(
            f"[ckprofiler] dtype={dtype!r} is not supported by ckProfiler; skipping",
            file=sys.stderr,
        )
        return

    # Load the converter script as a module without executing its __main__ block.
    spec = importlib.util.spec_from_file_location("_ck_converter", converter_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    p = problem

    # Build a namespace that matches the attributes the converter's argparse produces.
    args = types.SimpleNamespace(
        ck_profiler_cmd=ckprofiler,
        data_type=ck_dtype_str,
        in_layout="NHWC" if not p.is_3d else "NDHWC",
        spatial_dim=3 if p.is_3d else 2,
        batchsize=p.N,
        in_channels=p.C,
        out_channels=p.K,
        group_count=p.groups,
        in_h=p.Hi,
        in_w=p.Wi,
        fil_h=p.Y,
        fil_w=p.X,
        conv_stride_h=p.sH,
        conv_stride_w=p.sW,
        pad_h=p.pH,
        pad_w=p.pW,
        dilation_h=p.dH,
        dilation_w=p.dW,
        # 3-D fields
        in_d=p.Di if p.is_3d else 1,
        fil_d=p.Z if p.is_3d else 1,
        conv_stride_d=p.sD if p.is_3d else 1,
        pad_d=p.pD if p.is_3d else 0,
        dilation_d=p.dD if p.is_3d else 1,
        forw=forw,
        verify=verify,
        time=1,
        instance=-1,
        list_instances=False,
    )

    mod.init_const_args(args)
    mod.run_ck_profiler(args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark grouped convolution via the rocke dispatcher (fwd + wgrad)"
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
        "--warmup", type=int, default=3, help="warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="timed iterations (default: 10)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify kernel output against torch reference",
    )
    parser.add_argument(
        "--dump-fail",
        default=None,
        metavar="PATH",
        dest="dump_fail",
        help=(
            "on verify FAIL, save kernel output, reference, and "
            "abs-diff tensors to PATH/{kernel_name}_{out,ref,diff}.txt. Implies --verify."
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
    ck_grp = parser.add_argument_group(
        "ckProfiler comparison",
        "Run ckProfiler on the same conv problem(s) for side-by-side comparison "
        "via convert_miopen_driver_to_profiler.py. "
        "Uses --direction and --verify to control direction and verification. "
        "Requires a built ckProfiler binary.",
    )
    ck_grp.add_argument(
        "--ckprofiler",
        default=None,
        metavar="PATH",
        help="Path to the ckProfiler binary. When set, ckProfiler is run for each case "
        "before the rocke benchmark.",
    )
    ck_grp.add_argument(
        "--ckprofiler-script",
        default=None,
        metavar="PATH",
        help="Path to convert_miopen_driver_to_profiler.py (default: auto-detected from CK repo).",
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

    json_grp = parser.add_argument_group(
        "JSON input",
        "Load benchmark cases from a JSON file (bench_cases_conv.json or "
        "bench_cases_conv_bwd.json). Each entry in the array becomes one benchmark run. "
        "When set, --dtype / shape flags and MIOpen flags are ignored.",
    )
    json_grp.add_argument(
        "--json-file",
        default=None,
        metavar="FILE",
        help="Path to a JSON file containing an array of benchmark case objects, "
        "e.g. bench_cases_conv.json or bench_cases_conv_bwd.json. "
        "Supported fields: N, Cin, Cout, H, W, Kh, Kw, stride, pad, dilation, "
        "groups, dtype, layout, op (conv2d/conv3d/conv1d). "
        "Entries with unsupported dtypes or layouts are skipped with a warning.",
    )
    json_grp.add_argument(
        "--json-filter-suite",
        default=None,
        metavar="SUITE",
        help='Only run JSON entries whose "suite" field matches SUITE '
        '(e.g. "regular", "extended").',
    )
    json_grp.add_argument(
        "--json-filter-priority",
        default=None,
        metavar="PRIORITY",
        help='Only run JSON entries whose "priority" field matches PRIORITY '
        '(e.g. "P0", "P1").',
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

    if args.miopen_cmd is None and args.miopen_file is None and args.json_file is None:
        if args.Di is not None and args.Z is None:
            print("--Z (filter depth) is required when --Di is set", file=sys.stderr)
            return 2
        if args.Z is not None and args.Di is None:
            print("--Di (input depth) is required when --Z is set", file=sys.stderr)
            return 2

    import ctypes

    from rocke import compile_kernel
    from dispatch.grouped_convolution import (
        ConvGroupedRequest,
        conv_grouped_sweep_space,
    )
    from rocke.instances.common.conv_implicit_gemm import (
        ConvProblem,
        build_implicit_gemm_conv,
    )
    from rocke.instances.common.conv_implicit_gemm_wgrad import (
        build_implicit_gemm_conv_wgrad,
    )
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    arch = args.arch

    # Build the list of (problem, dtype) cases to run.
    cases: list  # List[Tuple[ConvProblem, str]]
    if args.json_file is not None:
        import json

        path = args.json_file
        with open(path) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            print(f"error: {path}: expected a JSON array at top level", file=sys.stderr)
            return 2
        cases = []
        for idx, entry in enumerate(entries):
            case_id = entry.get("case_id", f"#{idx + 1}")
            if args.json_filter_suite is not None:
                if entry.get("suite") != args.json_filter_suite:
                    continue
            if args.json_filter_priority is not None:
                if entry.get("priority") != args.json_filter_priority:
                    continue
            try:
                prob, dt = parse_json_case(entry)
                cases.append((prob, dt))
            except ValueError as e:
                print(f"[warn] {path} case {case_id}: skipping — {e}", file=sys.stderr)
        if not cases:
            print(
                f"error: {path}: no valid cases found "
                f"(check --json-filter-suite / --json-filter-priority)",
                file=sys.stderr,
            )
            return 2
    elif args.miopen_file is not None:
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

        if args.ckprofiler is not None:
            _ck_forw = {"fwd": 1, "wgrad": 4}[args.direction]
            _run_ckprofiler(
                problem=problem,
                dtype=dtype,
                ckprofiler=args.ckprofiler,
                converter_script=args.ckprofiler_script,
                forw=_ck_forw,
                verify=int(args.verify),
            )

        req_base = dict(
            N=problem.N,
            C=problem.C,
            K=problem.K,
            Hi=problem.Hi,
            Wi=problem.Wi,
            Y=problem.Y,
            X=problem.X,
            arch=arch,
            G=problem.groups,
            stride_h=problem.sH,
            stride_w=problem.sW,
            pad_h=problem.pH,
            pad_w=problem.pW,
            dilation_h=problem.dH,
            dilation_w=problem.dW,
            dtype=dtype,
            direction=args.direction,
        )
        if problem.is_3d:
            req_base.update(
                Di=problem.Di,
                Z=problem.Z,
                stride_d=problem.sD,
                pad_d=problem.pD,
                dilation_d=problem.dD,
            )

        if args.direction == "wgrad":
            rc = _run_wgrad(
                args=args,
                problem=problem,
                dtype=dtype,
                arch=arch,
                req_base=req_base,
                compile_kernel=compile_kernel,
                build_implicit_gemm_conv_wgrad=build_implicit_gemm_conv_wgrad,
                conv_grouped_sweep_space=conv_grouped_sweep_space,
                ConvGroupedRequest=ConvGroupedRequest,
                synchronize_and_release=synchronize_and_release,
                time_launches=time_launches,
                Runtime=Runtime,
                KernelLauncher=KernelLauncher,
                LaunchConfig=LaunchConfig,
                u8=_u8,
            )
        else:
            rc = _run_fwd(
                args=args,
                problem=problem,
                dtype=dtype,
                arch=arch,
                req_base=req_base,
                compile_kernel=compile_kernel,
                build_implicit_gemm_conv=build_implicit_gemm_conv,
                conv_grouped_sweep_space=conv_grouped_sweep_space,
                ConvGroupedRequest=ConvGroupedRequest,
                synchronize_and_release=synchronize_and_release,
                time_launches=time_launches,
                Runtime=Runtime,
                KernelLauncher=KernelLauncher,
                LaunchConfig=LaunchConfig,
                u8=_u8,
            )
        all_rc = all_rc or rc
    return all_rc


def _run_fwd(
    *,
    args,
    problem,
    dtype: str,
    arch: str,
    req_base: dict,
    compile_kernel,
    build_implicit_gemm_conv,
    conv_grouped_sweep_space,
    ConvGroupedRequest,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
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

    req = ConvGroupedRequest(**req_base)
    specs = list(conv_grouped_sweep_space(req))

    if not specs:
        print(
            f"No dispatcher candidates for {arch} {dtype} {p.short()}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Running {len(specs)} dispatcher candidate(s) for {arch} {dtype} {p.short()} ...",
        flush=True,
    )

    rt = Runtime()
    results: List[Result] = []

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

    for dspec in specs:
        instance_spec = dspec.to_fwd_spec(problem)

        try:
            kernel = build_implicit_gemm_conv(instance_spec, arch=arch)
        except ValueError:
            continue

        artifact = compile_kernel(kernel, arch=arch)

        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
        gm = (p.M + dspec.tile_m - 1) // dspec.tile_m
        gn = (p.N_gemm + dspec.tile_n - 1) // dspec.tile_n
        grid = (gn, gm, p.groups)
        block = (instance_spec.block_size, 1, 1)
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
                tile_m=dspec.tile_m,
                tile_n=dspec.tile_n,
                tile_k=dspec.tile_k,
                warp_m=dspec.warp_m,
                warp_n=dspec.warp_n,
                warp_tile_mn=dspec.warp_tile_mn,
                warp_tile_k=dspec.warp_tile_k,
                pipeline=dspec.pipeline,
                epilogue=dspec.epilogue,
                split_k=1,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
                passed=kernel_passed,
            )
        )

        print(
            f"  tile={dspec.tile_m}x{dspec.tile_n}x{dspec.tile_k} "
            f"warp={dspec.warp_m}x{dspec.warp_n} "
            f"atom={dspec.warp_tile_mn}x{dspec.warp_tile_mn}x{dspec.warp_tile_k} "
            f"{dspec.pipeline}/{dspec.epilogue:9s} "
            f"{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    # Free GPU buffers.
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)

    if not results:
        print("No valid configurations found.", file=sys.stderr)
        return 1

    show_verify = args.verify
    width = 84 if show_verify else 72
    print(f"\n{'='*width}")
    print(f"Results for {arch} {dtype} {p.short()}")
    print(f"{'='*width}")
    hdr = (
        f"{'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  {'verify':>6}  config"
        if show_verify
        else f"{'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
    )
    print(hdr)
    print("-" * width)
    for r in results:
        cfg_str = (
            f"tile={r.tile_m}x{r.tile_n}x{r.tile_k} "
            f"warp={r.warp_m}x{r.warp_n} "
            f"atom={r.warp_tile_mn}x{r.warp_tile_mn}x{r.warp_tile_k} "
            f"{r.pipeline}/{r.epilogue}"
        )
        if show_verify:
            v = "PASS" if r.passed else "FAIL"
            print(f"{r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {v:>6}  {cfg_str}")
        else:
            print(f"{r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {cfg_str}")
    return 0


def _run_wgrad(
    *,
    args,
    problem,
    dtype: str,
    arch: str,
    req_base: dict,
    compile_kernel,
    build_implicit_gemm_conv_wgrad,
    conv_grouped_sweep_space,
    ConvGroupedRequest,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
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

    req = ConvGroupedRequest(**req_base)
    specs = list(conv_grouped_sweep_space(req))

    if not specs:
        print(
            f"No dispatcher wgrad candidates for {arch} {dtype} {p.short()}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Running {len(specs)} dispatcher wgrad candidate(s) for {arch} {dtype} {p.short()} ...",
        flush=True,
    )

    rt = Runtime()
    results: List[Result] = []

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

    for dspec in specs:
        instance_spec = dspec.to_wgrad_spec(problem)

        try:
            kernel = build_implicit_gemm_conv_wgrad(instance_spec, arch=arch)
        except ValueError:
            continue

        artifact = compile_kernel(kernel, arch=arch)

        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
        wg_N = (p.Z if p.is_3d else 1) * p.Y * p.X * p.C
        gx = (wg_N + dspec.tile_n - 1) // dspec.tile_n
        gy = (p.K + dspec.tile_m - 1) // dspec.tile_m
        grid = (gx, gy, instance_spec.split_k)
        block = (instance_spec.block_size, 1, 1)
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
                zero_init_out=(instance_spec.split_k > 1),
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

        if instance_spec.split_k > 1:

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
                tile_m=dspec.tile_m,
                tile_n=dspec.tile_n,
                tile_k=dspec.tile_k,
                warp_m=dspec.warp_m,
                warp_n=dspec.warp_n,
                warp_tile_mn=dspec.warp_tile_mn,
                warp_tile_k=dspec.warp_tile_k,
                pipeline=dspec.pipeline,
                epilogue=dspec.epilogue,
                split_k=instance_spec.split_k,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
            )
        )

        print(
            f"  tile={dspec.tile_m}x{dspec.tile_n}x{dspec.tile_k} "
            f"warp={dspec.warp_m}x{dspec.warp_n} "
            f"atom={dspec.warp_tile_mn}x{dspec.warp_tile_mn}x{dspec.warp_tile_k} "
            f"{dspec.pipeline}/{dspec.epilogue:9s} spk{instance_spec.split_k:<3d} "
            f"{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    rt.free(dY_dev)
    rt.free(X_dev)
    rt.free(dW_dev)

    if not results:
        print("No valid wgrad configurations found.", file=sys.stderr)
        return 1

    print(f"\n{'='*72}")
    print(f"Results for {arch} {dtype} {p.short()} wgrad")
    print(f"{'='*72}")
    hdr = f"{'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
    print(hdr)
    print("-" * 72)
    for r in results:
        cfg_str = (
            f"tile={r.tile_m}x{r.tile_n}x{r.tile_k} "
            f"warp={r.warp_m}x{r.warp_n} "
            f"atom={r.warp_tile_mn}x{r.warp_tile_mn}x{r.warp_tile_k} "
            f"{r.pipeline}/{r.epilogue} spk{r.split_k}"
        )
        print(f"{r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {cfg_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
