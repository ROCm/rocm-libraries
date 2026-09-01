# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tile/pipeline sweep benchmark for implicit-GEMM convolution (gfx950, gfx1250).

Supports forward (NHWC × KYXC → NHWK), backward-weight (wgrad, dY × X → dW),
and backward-data (dgrad, dY × W → dX) directions.  Select with
``--direction fwd`` (default), ``--direction wgrad``, or ``--direction dgrad``.

Builds every valid combination of tile / warp / pipeline / epilogue parameters,
runs each on GPU, and reports the best configuration ranked by TFLOPS.

Swept dimensions:
  tile_m, tile_n : 16, 32, 64, 128, 256
  tile_k         : 16, 32, 64, 128
  warp_m, warp_n : 1, 2, 4, 8
  warp_tile_m == warp_tile_n : 16, 32
  pipeline       : mem, compv3, compv4, wavelet
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

  # Run from bench_cases_conv.json or bench_cases_conv_bwd.json:
  python benchmark_implicit_gemm_conv.py --json-file bench_cases_conv.json
  python benchmark_implicit_gemm_conv.py --json-file bench_cases_conv_bwd.json

Shape / dtype parameters mirror bake_off_implicit_gemm.py exactly.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import re
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
_TILE_K = (16, 32, 64)
_WARP_MN = (1, 2, 4, 8)
_WARP_MN_GFX1250 = (1, 2, 4, 8, 16)
_WARP_TILE_MN = (16, 32)
_PIPELINES = ("mem", "compv3", "compv4", "wavelet", "basic")
_EPILOGUES = ("default", "cshuffle")
# Split-K degrees swept when --split-k 0 (auto) is passed for wgrad.
_SPLIT_K_AUTO = (128, 64, 32, 16, 8, 4, 2, 1)


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
    vec_a: int = 1
    vec_b: int = 1
    vec_c: int = 1
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
    """Derive launch grid from wgrad spec and split-K degree.

    gx/gy tile the per-group GEMM (spec.wg_M/wg_N). The group index rides on
    block_id_z, giving z = groups * split_k (== split_k for the ungrouped
    groups==1 path).
    """
    tile_m, tile_n = spec.tile_m, spec.tile_n
    gx = (spec.wg_N + tile_n - 1) // tile_n
    gy = (spec.wg_M + tile_m - 1) // tile_m
    return (gx, gy, spec.problem.groups * split_k)


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
    arch: str,
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

    if arch == "gfx1250":
        out_f32 = out_cpu.float()
        abs_diff = out_f32.sub(ref_out.cpu()).abs()
        ref_scale = ref_out.cpu().abs().max().clamp(min=1.0)
    else:
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

    Only 2-D NHWC convolutions are supported; the function raises
    ``ValueError`` for unsupported cases (3-D, NCHW).
    Returns ``(problem, dtype, forw)`` where ``dtype`` is ``"fp16"``,
    ``"bf16"``, or ``"fp32"`` and ``forw`` is the raw MIOpenDriver ``-F``
    value (1=fwd, 2=bwd_data, 4=bwd_weight, 0=all).
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
    return problem, dtype, miopen_args.forw


# ---------------------------------------------------------------------------
# CK Profiler comparison
# ---------------------------------------------------------------------------

_CK_DTYPE_STR = {"fp32": "fp32", "fp16": "fp16", "bf16": "bfp16"}

# MIOpenDriver -F flag → rocke direction string.  Values 0/3/5/6 run multiple
# passes; we expand them at the call site.
_FORW_TO_DIRECTIONS: dict[int, list[str]] = {
    0: ["fwd", "dgrad", "wgrad"],
    1: ["fwd"],
    2: ["dgrad"],
    3: ["fwd", "dgrad"],
    4: ["wgrad"],
    5: ["fwd", "wgrad"],
    6: ["dgrad", "wgrad"],
}


def _run_ckprofiler(
    problem, dtype: str, ckprofiler: str, converter_script: str, forw: int, verify: int
) -> "list[dict]":
    """Delegate to convert_miopen_driver_to_profiler.py for the given ConvProblem.

    Builds a synthetic args namespace that mirrors what the converter script's
    argparse produces, then calls its init_const_args / run_ck_profiler functions
    directly — no logic is duplicated here.

    ``forw`` follows MIOpenDriver -F convention:
      0 fwd+bwd_data+bwd_weight   1 fwd   2 bwd_data   4 bwd_weight
      3 fwd+bwd_data   5 fwd+bwd_weight   6 bwd_data+bwd_weight

    Returns a list of dicts with keys ``direction``, ``ms``, ``tflops``, ``gbps``
    parsed from ckProfiler's "Best Perf:" output lines.
    """
    import importlib.util
    import subprocess
    import types

    ck_dtype_str = _CK_DTYPE_STR.get(dtype)
    if ck_dtype_str is None:
        print(
            f"[ckprofiler] dtype={dtype!r} is not supported by ckProfiler; skipping",
            file=sys.stderr,
        )
        return []

    # Load the converter script as a module without executing its __main__ block.
    spec = importlib.util.spec_from_file_location("_ck_converter", converter_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    p = problem

    # Build a namespace that matches the attributes the converter's argparse produces.
    ck_args = types.SimpleNamespace(
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

    # Patch run_ck_profiler_cmd to capture stdout while still printing it.
    # ckProfiler (grouped_conv) prints:
    #   Best configuration parameters:
    #   name: ...
    #   avg_time: <ms>
    #   tflops: <tflops>
    #   GB/s: <gbps>
    _best_cfg_re = re.compile(
        r"Best configuration parameters:.*?avg_time:\s*([\d.]+).*?tflops:\s*([\d.]+).*?GB/s:\s*([\d.]+)",
        re.DOTALL,
    )
    # Also accept the compact single-line format used by other CK profilers:
    #   Best Perf: <ms> ms, <tflops> TFlops, <gbps> GB/s
    _best_perf_re = re.compile(
        r"Best Perf:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*TFlops,\s*([\d.]+)\s*GB/s"
    )
    captured_rows: list[dict] = []
    _orig_run_cmd = mod.run_ck_profiler_cmd

    def _patched_run_cmd(cmd):
        print("ckProfiler command:")
        print(" ".join(cmd))
        result = subprocess.run(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = result.stdout + result.stderr
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        # Infer direction from the op name embedded in the command.
        direction = "fwd"
        for token in cmd:
            if "bwd_weight" in token:
                direction = "wgrad"
                break
            if "bwd_data" in token:
                direction = "bwd_data"
                break
        for m in _best_cfg_re.finditer(output):
            captured_rows.append(
                {
                    "direction": direction,
                    "ms": float(m.group(1)),
                    "tflops": float(m.group(2)),
                    "gbps": float(m.group(3)),
                }
            )
        for m in _best_perf_re.finditer(output):
            captured_rows.append(
                {
                    "direction": direction,
                    "ms": float(m.group(1)),
                    "tflops": float(m.group(2)),
                    "gbps": float(m.group(3)),
                }
            )

    mod.run_ck_profiler_cmd = _patched_run_cmd
    try:
        mod.init_const_args(ck_args)
        # init_const_args overwrites ck_profiler_cmd with a hardcoded relative
        # path ("../build/bin/ckProfiler"); restore the user-supplied binary.
        ck_args.ck_profiler_cmd = ckprofiler
        mod.run_ck_profiler(ck_args)
    finally:
        mod.run_ck_profiler_cmd = _orig_run_cmd

    return captured_rows


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
        choices=["fwd", "wgrad", "dgrad"],
        help="convolution direction: forward (fwd), backward-weight (wgrad), or backward-data (dgrad) (default: fwd)",
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
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "number of parallel compile workers (default: 1, serial). "
            "Uses multiprocessing (ProcessPoolExecutor) so each worker runs "
            "libamd_comgr in its own process, bypassing the GIL. "
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

    parser.add_argument(
        "--split-k-prune",
        type=float,
        default=None,
        dest="split_k_prune",
        metavar="PCT",
        help=(
            "Only effective with --split-k 0 (sweep). "
            "If a kernel's TFLOPS at a given split-K degree drops by at least PCT%% "
            "relative to its best split-K seen so far, skip all remaining split-K "
            "degrees for that tile/warp/pipeline/epilogue configuration. "
            "Example: --split-k-prune 15 prunes when performance drops by >=15%%."
        ),
    )

    parser.add_argument(
        "--csv",
        default=None,
        metavar="FILE",
        help=(
            "write combined rocke + ckProfiler results to FILE (CSV format). "
            "Each row is one rocke kernel; ck_tflops, ck_ms, ck_gbps, and "
            "speedup_rocke_vs_ck columns are populated when --ckprofiler is also set."
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
        "before the rocke sweep.",
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
        "bench_cases_conv_bwd.json). Each entry in the array becomes one sweep. "
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
    json_grp.add_argument(
        "--start-from-case",
        default=None,
        metavar="CASE_ID",
        help='Skip all JSON entries before the one whose "case_id" matches CASE_ID '
        '(e.g. "G0028"). The matching entry and all following entries are run.',
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
    from rocke.instances.common.conv_implicit_gemm_dgrad import (
        DgradConvSpec,
        build_implicit_gemm_conv_dgrad,
        is_valid_dgrad_spec,
    )
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    arch = args.arch
    target = ArchTarget.from_gfx(arch)

    # Build the list of (problem, dtype) cases to sweep.
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
        start_from = args.start_from_case
        reached_start = start_from is None
        for idx, entry in enumerate(entries):
            case_id = entry.get("case_id", f"#{idx + 1}")
            if not reached_start:
                if case_id == start_from:
                    reached_start = True
                else:
                    continue
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
                prob, dt, forw = parse_miopen_cmd(line)
                for _dir in _FORW_TO_DIRECTIONS.get(forw, ["fwd"]):
                    cases.append((prob, dt, _dir))
            except ValueError as e:
                print(f"[warn] {path}:{lineno}: skipping — {e}", file=sys.stderr)
    elif args.miopen_cmd is not None:
        prob, dt, forw = parse_miopen_cmd(args.miopen_cmd)
        cases = [(prob, dt, _dir) for _dir in _FORW_TO_DIRECTIONS.get(forw, ["fwd"])]
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
        cases = [(problem, args.dtype, args.direction)]

    _csv_fields = [
        "rank",
        "shape",
        "sH",
        "sW",
        "pH",
        "pW",
        "dH",
        "dW",
        "groups",
        "dtype",
        "direction",
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_tile_mn",
        "warp_tile_k",
        "pipeline",
        "epilogue",
        "split_k",
        "rocke_ms",
        "rocke_tflops",
        "rocke_gbps",
        "passed",
        "kernel_name",
        "ck_tflops",
        "ck_ms",
        "ck_gbps",
        "speedup_rocke_vs_ck",
    ]

    all_rc = 0
    # Maps (shape, dtype, direction) -> best CK tflops/ms/gbps for that case.
    # Populated before the rocke sweep so the CSV join is available immediately.
    ck_best: dict[tuple, dict] = {}

    _csv_file = None
    _csv_writer = None
    if args.csv is not None:
        _csv_file = open(args.csv, "w", newline="")
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
        _csv_writer.writeheader()
        _csv_file.flush()

    _INT32_MAX = 2**31 - 1

    n_csv_rows = 0
    try:
        for case_idx, (problem, dtype, direction) in enumerate(cases):
            _elem_bytes = 4 if dtype == "fp32" else 2
            _A_bytes = (
                problem.N
                * (problem.Di or 1)
                * problem.Hi
                * problem.Wi
                * problem.C
                * _elem_bytes
            )
            _B_bytes = (
                problem.K
                * (problem.Z or 1)
                * problem.Y
                * problem.X
                * problem.C
                * _elem_bytes
            )
            _D_bytes = problem.M * problem.K * _elem_bytes
            if max(_A_bytes, _B_bytes, _D_bytes) > _INT32_MAX:
                print(
                    f"[skip] {problem.short()}: tensor byte sizes "
                    f"(A={_A_bytes}, B={_B_bytes}, D={_D_bytes}) "
                    f"exceed int32 range — kernel buffer args would overflow",
                    flush=True,
                )
                continue
            if len(cases) > 1:
                print(f"\n{'#'*72}", flush=True)
                print(
                    f"# Case {case_idx + 1}/{len(cases)}: {problem.short()} "
                    f"dtype={dtype} direction={direction}",
                    flush=True,
                )
                print(f"{'#'*72}", flush=True)

            if args.ckprofiler is not None:
                _ck_forw = {"fwd": 1, "dgrad": 2, "wgrad": 4}[direction]
                ck_rows = _run_ckprofiler(
                    problem=problem,
                    dtype=dtype,
                    ckprofiler=args.ckprofiler,
                    converter_script=args.ckprofiler_script,
                    forw=_ck_forw,
                    verify=int(args.verify),
                )
                # Keep the best (highest tflops) CK result for this case.
                _key = (problem.short(), dtype, direction)
                for row in ck_rows:
                    if row["direction"] == direction:
                        prev = ck_best.get(_key)
                        if prev is None or row["tflops"] > prev["tflops"]:
                            ck_best[_key] = row

            _common = dict(
                args=args,
                problem=problem,
                dtype=dtype,
                arch=arch,
                target=target,
                compile_kernel=compile_kernel,
                jobs=args.jobs,
                ConvDataSpec=ConvDataSpec,
                synchronize_and_release=synchronize_and_release,
                time_launches=time_launches,
                Runtime=Runtime,
                KernelLauncher=KernelLauncher,
                LaunchConfig=LaunchConfig,
                u8=_u8,
                case_idx=case_idx,
            )

            if direction == "wgrad":
                rc, rocke_results = _run_wgrad_sweep(
                    **_common,
                    WgradConvSpec=WgradConvSpec,
                    build_implicit_gemm_conv_wgrad=build_implicit_gemm_conv_wgrad,
                    is_valid_wgrad_spec=is_valid_wgrad_spec,
                )
            elif direction == "dgrad":
                rc = _run_dgrad_sweep(
                    **_common,
                    DgradConvSpec=DgradConvSpec,
                    build_implicit_gemm_conv_dgrad=build_implicit_gemm_conv_dgrad,
                    is_valid_dgrad_spec=is_valid_dgrad_spec,
                )
                rocke_results = None
            else:
                rc, rocke_results = _run_sweep(
                    **_common,
                    ImplicitGemmConvSpec=ImplicitGemmConvSpec,
                    build_implicit_gemm_conv=build_implicit_gemm_conv,
                    is_valid_spec_for_problem=is_valid_spec_for_problem,
                )
            all_rc = all_rc or rc

            if _csv_writer is not None and rocke_results:
                _shape = problem.short()
                _key = (_shape, dtype, direction)
                _ck = ck_best.get(_key)
                for rank, r in enumerate(rocke_results[:5], 1):
                    speedup = (r.tflops / _ck["tflops"]) if _ck else None
                    _csv_writer.writerow(
                        {
                            "rank": rank,
                            "shape": _shape,
                            "sH": problem.sH,
                            "sW": problem.sW,
                            "pH": problem.pH,
                            "pW": problem.pW,
                            "dH": problem.dH,
                            "dW": problem.dW,
                            "groups": problem.groups,
                            "dtype": dtype,
                            "direction": direction,
                            "tile_m": r.tile_m,
                            "tile_n": r.tile_n,
                            "tile_k": r.tile_k,
                            "warp_m": r.warp_m,
                            "warp_n": r.warp_n,
                            "warp_tile_mn": r.warp_tile_mn,
                            "warp_tile_k": r.warp_tile_k,
                            "pipeline": r.pipeline,
                            "epilogue": r.epilogue,
                            "split_k": r.split_k,
                            "rocke_ms": r.ms,
                            "rocke_tflops": r.tflops,
                            "rocke_gbps": r.gbps,
                            "passed": r.passed,
                            "kernel_name": r.kernel_name,
                            "ck_tflops": _ck["tflops"] if _ck else "",
                            "ck_ms": _ck["ms"] if _ck else "",
                            "ck_gbps": _ck["gbps"] if _ck else "",
                            "speedup_rocke_vs_ck": (
                                f"{speedup:.4f}" if speedup is not None else ""
                            ),
                        }
                    )
                    n_csv_rows += 1
                _csv_file.flush()
    finally:
        if _csv_file is not None:
            _csv_file.close()
            if n_csv_rows:
                print(f"\nResults written to {args.csv} ({n_csv_rows} rows).")

    return all_rc


def _compile_one(args_tuple):
    """Top-level picklable worker for ProcessPoolExecutor.

    Receives (kernel, arch) and returns (kernel_name, artifact).  Must be
    defined at module level so pickle can locate it by name.
    """
    kernel, arch = args_tuple
    from rocke import compile_kernel as _compile_kernel

    artifact = _compile_kernel(kernel, arch=arch)
    return kernel.name, artifact


def _build_fwd_one(args_tuple):
    """Top-level picklable worker: validate + build IR for one fwd combo.

    Returns ``(combo, spec, kernel)`` on success, or ``None`` if the combo is
    invalid/unsupported.  Must live at module level for pickle.
    """
    combo, problem, dtype, arch, mma_family, wave_size = args_tuple
    tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline, epilogue = combo

    from rocke.core.arch import ArchTarget
    from rocke.instances.common.conv_implicit_gemm import (
        ConvDataSpec,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
        is_valid_spec_for_problem,
    )

    target = ArchTarget.from_gfx(arch)
    atom = target.mma.select_largest_k(
        family=mma_family,
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype="fp32",
        m=warp_tile_mn,
        n=warp_tile_mn,
        k_max=tile_k,
    )
    if atom is None:
        return None

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
        wave_size=wave_size,
        pipeline=pipeline,
        epilogue=epilogue,
        groups=problem.groups,
    )
    ok, _ = is_valid_spec_for_problem(spec, problem, arch)
    if not ok:
        return None
    try:
        kernel = build_implicit_gemm_conv(spec, arch=arch)
    except ValueError:
        return None
    return combo, spec, kernel


def _build_wgrad_one(args_tuple):
    """Top-level picklable worker: validate + build IR for one wgrad combo.

    Returns ``(combo, spec, resolved_split_k, kernel)`` on success, or ``None``.
    Must live at module level for pickle.
    """
    combo, problem, dtype, arch = args_tuple
    (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_tile_mn,
        pipeline,
        epilogue,
        split_k,
    ) = combo

    from rocke.core.arch import ArchTarget
    from rocke.instances.common.conv_implicit_gemm import ConvDataSpec
    from rocke.instances.common.conv_implicit_gemm_wgrad import (
        WgradConvSpec,
        build_implicit_gemm_conv_wgrad,
        is_valid_wgrad_spec,
    )

    target = ArchTarget.from_gfx(arch)
    _mma_family = "wmma" if target.wave_size == 32 else "mma"
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
        return None

    warp_tile_k = atom.k
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
        # split_k=0 (runtime) or split_k=1 (no-split): pass through as-is.
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
        wave_size=target.wave_size,
        pipeline=pipeline,
        epilogue=epilogue,
        split_k=resolved_split_k,
    )
    ok, _ = is_valid_wgrad_spec(spec, arch)
    if not ok:
        return None
    try:
        kernel = build_implicit_gemm_conv_wgrad(spec, arch=arch)
    except ValueError:
        return None
    return combo, spec, resolved_split_k, kernel


def _build_dgrad_one(args_tuple):
    """Top-level picklable worker: validate + build IR for one dgrad combo.

    Returns ``(combo, spec, resolved_split_k, kernel)`` on success, or ``None``.
    Must live at module level for pickle.
    """
    combo, problem, dtype, arch, vec_a, vec_b, vec_c = args_tuple
    (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_tile_mn,
        pipeline,
        epilogue,
        split_k,
    ) = combo

    if split_k > 1 and epilogue == "cshuffle":
        return None

    from rocke.core.arch import ArchTarget
    from rocke.instances.common.conv_implicit_gemm import ConvDataSpec
    from rocke.instances.common.conv_implicit_gemm_dgrad import (
        DgradConvSpec,
        build_implicit_gemm_conv_dgrad,
        is_valid_dgrad_spec,
    )

    target = ArchTarget.from_gfx(arch)
    _mma_family = "wmma" if target.wave_size == 32 else "mma"
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
        return None

    warp_tile_k = atom.k
    if split_k == -1:
        if _mma_family == "wmma":
            resolved_split_k = 1
        else:
            from rocke.helpers.split_k import select_split_k_wgrad

            resolved_split_k = select_split_k_wgrad(
                wg_M=problem.N * problem.Hi * problem.Wi,
                wg_N=problem.cpg,
                wg_K=problem.Y * problem.X * problem.kpg,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                arch=arch,
            ).split_k
    else:
        resolved_split_k = split_k

    spec = DgradConvSpec(
        problem=problem,
        name="rocke_bench_igemm_dgrad",
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
        split_k=resolved_split_k,
        vector_size_a=vec_a,
        vector_size_b=vec_b,
        vector_size_c=vec_c,
    )
    ok, _ = is_valid_dgrad_spec(spec, arch)
    if not ok:
        return None
    try:
        kernel = build_implicit_gemm_conv_dgrad(spec, arch=arch)
    except ValueError:
        return None
    return combo, spec, resolved_split_k, kernel


def _build_ir_parallel(work, worker_fn, jobs: int) -> list:
    """Run *worker_fn* over *work* items in parallel, returning non-None results.

    When *jobs* == 1 runs serially to avoid subprocess overhead.
    When *jobs* == 0 uses ``os.cpu_count()`` workers.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor

    if jobs == 1:
        return [r for item in work for r in [worker_fn(item)] if r is not None]

    max_workers = os.cpu_count() if jobs == 0 else jobs
    results = []
    n_killed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker_fn, item): i for i, item in enumerate(work)}
        done = 0
        total = len(work)
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except BrokenExecutor:
                # Worker process was killed (e.g. OOM); skip this item and keep going.
                n_killed += 1
                done += 1
                continue
            except Exception:
                done += 1
                continue
            if r is not None:
                results.append(r)
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                print(f"  IR built {done}/{total} ({len(results)} valid)", flush=True)
    if n_killed:
        print(
            f"  Warning: {n_killed} IR build worker(s) were killed (OOM?); "
            "results may be incomplete.",
            flush=True,
        )
    return results


def _compile_kernels_parallel(kernels, compile_kernel, arch: str, jobs: int) -> dict:
    """Compile *kernels* (a list of KernelDef) in parallel.

    Deduplicates by kernel name before submitting.  Returns a
    ``{name: KernelArtifact}`` dict covering every unique name in *kernels*.

    When *jobs* == 1 the compilation is serial (no subprocess overhead).
    When *jobs* == 0 the worker count defaults to ``os.cpu_count()``.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    unique: dict = {}
    for k in kernels:
        if k.name not in unique:
            unique[k.name] = k

    if not unique:
        return {}

    artifact_map: dict = {}

    if jobs == 1:
        for name, k in unique.items():
            artifact_map[name] = compile_kernel(k, arch=arch)
        return artifact_map

    max_workers = os.cpu_count() if jobs == 0 else jobs
    work = [(k, arch) for k in unique.values()]

    print(
        f"Compiling {len(unique)} unique kernels with {max_workers} workers ...",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compile_one, item): item[0].name for item in work}
        done = 0
        for fut in as_completed(futures):
            name, artifact = fut.result()
            artifact_map[name] = artifact
            done += 1
            if done % max(1, len(unique) // 10) == 0 or done == len(unique):
                print(f"  compiled {done}/{len(unique)}", flush=True)

    return artifact_map


def _run_sweep(
    *,
    args,
    problem,
    dtype: str,
    arch: str,
    target,
    compile_kernel,
    jobs: int = 1,
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
    case_idx: int = 0,
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

    # Weight is per-group: its channel extent is cpg = C / groups (== C when
    # groups == 1), matching make_b_descriptor (KYXC/KZYXC with C = cpg), the
    # grouped NumPy oracle, and torch's F.conv2d weight shape
    # [K, C/groups, Y, X]. Allocating full C here silently mismatched the kernel
    # and broke the grouped --verify reference.
    if p.is_3d:
        _A_f32 = _make(p.N, p.Di, p.Hi, p.Wi, p.C)
        _B_f32 = _make(p.K, p.Z, p.Y, p.X, p.cpg)
        D_t = torch.empty(p.N, p.Do, p.Ho, p.Wo, p.K, dtype=_torch_dtype)
    else:
        _A_f32 = _make(p.N, p.Hi, p.Wi, p.C)
        _B_f32 = _make(p.K, p.Y, p.X, p.cpg)
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
        return 2, []

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
        effective_seed = args.seed + case_idx
        combos = _sample_combos(combos, args.sample, effective_seed)
        print(
            f"Sampling {len(combos)}/{total} combinations "
            f"({args.sample*100:.0f}%, seed={effective_seed}).",
            flush=True,
        )

    print(
        f"Sweeping {len(combos)} combinations for {arch} {dtype} {p.short()} ...",
        flush=True,
    )

    # ---------------------------------------------------------------------------
    # Phase 1 – filter + IR build: validate every combo and build KernelDef IR.
    # ---------------------------------------------------------------------------
    if jobs != 1:
        print(f"Building IR for {len(combos)} combos in parallel ...", flush=True)
    work = [
        (combo, problem, dtype, arch, _mma_family, target.wave_size) for combo in combos
    ]
    pending = _build_ir_parallel(work, _build_fwd_one, jobs)
    n_skipped = len(combos) - len(pending)

    # ---------------------------------------------------------------------------
    # Phase 2 – compile: fan out compile_kernel across processes (or serial).
    # ---------------------------------------------------------------------------
    artifact_map = _compile_kernels_parallel(
        [k for _, _, k in pending], compile_kernel, arch, jobs
    )
    n_built = len(artifact_map)

    # ---------------------------------------------------------------------------
    # Phase 3 – GPU run: load modules and time each kernel serially.
    # ---------------------------------------------------------------------------
    from rocke.runtime.hip_module import HipError

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

    n_run = 0
    for combo, spec, kernel in pending:
        tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline, epilogue = combo
        warp_tile_k = spec.warp_tile_k
        artifact = artifact_map[kernel.name]

        try:
            launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=sig,
            )
        except HipError as e:
            n_skipped += 1
            print(
                f"[skip] kernel load failed for {artifact.kernel_name} "
                f"tile={tile_m}x{tile_n}x{tile_k} "
                f"warp={warp_m}x{warp_n} "
                f"atom={warp_tile_mn}x{warp_tile_mn}x{warp_tile_k} "
                f"{pipeline}/{epilogue}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

        grid = _grid_for_spec(spec, p)
        block = (spec.launch_block_size, 1, 1)
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
                arch=arch,
            )
            if stopped:
                rt.free(A_dev)
                rt.free(B_dev)
                rt.free(D_dev)
                return 1, []
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
        n_run += 1

        _va, _vb, _vc = ImplicitGemmConvSpec.default_vector_sizes(p.C, p.K, dtype)
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
                vec_a=_va,
                vec_b=_vb,
                vec_c=_vc,
                passed=kernel_passed,
            )
        )

        print(
            f"[{n_run:4d}] tile={tile_m}x{tile_n}x{tile_k} "
            f"warp={warp_m}x{warp_n} "
            f"atom={warp_tile_mn}x{warp_tile_mn}x{warp_tile_k} "
            f"{pipeline}/{epilogue:9s} "
            f"vec={_va}/{_vb}/{_vc} "
            f"{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    # Free GPU buffers.
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)

    print(
        f"\nSweep done: {n_built} compiled, {n_skipped} skipped.",
        flush=True,
    )

    if not results:
        print("No valid configurations found.", file=sys.stderr)
        return 1, []

    results.sort(key=lambda r: r.tflops, reverse=True)
    top_n = min(args.top, len(results))

    show_verify = args.verify
    width = 96 if show_verify else 84
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
            f"vec={r.vec_a}/{r.vec_b}/{r.vec_c} "
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
    return 0, results


def _run_wgrad_sweep(
    *,
    args,
    problem,
    dtype: str,
    arch: str,
    target,
    compile_kernel,
    jobs: int = 1,
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
    case_idx: int = 0,
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
    # Extended signature for runtime split-K kernels (split_k=0): adds ks i32 arg.
    sig_rt = sig + [{"name": "ks", "type": "i32", "size_bytes": 4}]

    # split_k degrees used to build kernel specs:
    #   0   → compile two variants: split_k=1 (no-atomic) and split_k=0 (runtime-atomic);
    #          actual degrees from _SPLIT_K_AUTO are swept at launch time via the ks arg.
    #  -1   → one combo per tile config, degree resolved by CK formula at build time.
    #  else → single fixed degree baked into the kernel.
    split_k_values = (0,) if args.split_k == 0 else (args.split_k,)

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
        effective_seed = args.seed + case_idx
        combos = _sample_combos(combos, args.sample, effective_seed)
        print(
            f"Sampling {len(combos)}/{total} wgrad combinations "
            f"({args.sample*100:.0f}%, seed={effective_seed}).",
            flush=True,
        )

    _spk_label = {0: "sweep", -1: "auto(CK)"}.get(args.split_k, str(args.split_k))
    print(
        f"Sweeping {len(combos)} wgrad combinations for {arch} {dtype} {p.short()} "
        f"(split_k={_spk_label}) ...",
        flush=True,
    )
    if p.is_pointwise:
        print("  (pointwise 1x1/s1/p0 — using explicit GEMM descriptors)", flush=True)

    # ---------------------------------------------------------------------------
    # Phase 1 – filter + IR build: validate every combo and build KernelDef IR.
    # ---------------------------------------------------------------------------
    if jobs != 1:
        print(f"Building IR for {len(combos)} wgrad combos in parallel ...", flush=True)
    work = [(combo, problem, dtype, arch) for combo in combos]
    pending = _build_ir_parallel(work, _build_wgrad_one, jobs)
    # _build_ir_parallel returns results in as_completed order (non-deterministic).
    # Re-sort: group by config (all combo dims except split_k), then split_k descending
    # so _SPLIT_K_AUTO order (128, 64, ..., 1) is preserved for --split-k-prune.
    pending.sort(key=lambda r: (r[0][:8], -r[2]))
    n_skipped = len(combos) - len(pending)

    # ---------------------------------------------------------------------------
    # Phase 2 – compile: fan out compile_kernel across processes (or serial).
    # ---------------------------------------------------------------------------
    artifact_map = _compile_kernels_parallel(
        [k for _, _, _, k in pending], compile_kernel, arch, jobs
    )
    n_built = len(artifact_map)

    # ---------------------------------------------------------------------------
    # Phase 3 – GPU run: load modules and time each kernel serially.
    # ---------------------------------------------------------------------------
    from rocke.runtime.hip_module import HipError

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
        if arch == "gfx1250":
            from rocke.benchmark.conv_reference import wgrad_reference_gfx1250

            ref_out = wgrad_reference_gfx1250(
                _X_f32, _dY_f32, p, out_dtype=_torch_dtype
            )
            print(
                f"Wgrad reference computed via gfx1250 hand-written wgrad "
                f"({tuple(ref_out.shape)}, {ref_out.dtype}).",
                flush=True,
            )
        else:
            from rocke.benchmark.conv_reference import wgrad_reference

            ref_out = wgrad_reference(_X_f32, _dY_f32, p)
            print(
                f"Wgrad reference computed ({tuple(ref_out.shape)}, {ref_out.dtype}).",
                flush=True,
            )

    _do_prune = args.split_k == 0 and args.split_k_prune is not None
    _prune_threshold = (args.split_k_prune or 0.0) / 100.0
    # Maps (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline, epilogue)
    # -> best TFLOPS seen so far across split-K degrees for that config.
    _best_tflops: dict = {}
    # Set of config keys that have been pruned (skip remaining split-K degrees).
    _pruned_configs: set = set()

    n_run = 0
    for combo, spec, resolved_split_k, kernel in pending:
        tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline, epilogue, _ = (
            combo
        )
        warp_tile_k = spec.warp_tile_k
        artifact = artifact_map[kernel.name]
        _is_rt = resolved_split_k == 0  # runtime split-K kernel

        if _do_prune:
            _cfg_key = (
                tile_m,
                tile_n,
                tile_k,
                warp_m,
                warp_n,
                warp_tile_mn,
                pipeline,
                epilogue,
            )
            if _cfg_key in _pruned_configs:
                n_skipped += 1
                continue

        _kernel_sig = sig_rt if _is_rt else sig
        try:
            launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=_kernel_sig,
            )
        except HipError as e:
            n_skipped += 1
            print(
                f"[skip] kernel load failed for {artifact.kernel_name} "
                f"tile={tile_m}x{tile_n}x{tile_k} "
                f"warp={warp_m}x{warp_n} "
                f"atom={warp_tile_mn}x{warp_tile_mn}x{warp_tile_k} "
                f"{pipeline}/{epilogue}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

        # For runtime kernels iterate over _SPLIT_K_AUTO degrees at launch time;
        # for fixed kernels there is exactly one degree (resolved_split_k itself).
        _rt_degrees = _SPLIT_K_AUTO if _is_rt else (resolved_split_k,)

        for _i, _launch_sk in enumerate(_rt_degrees):
            if _do_prune and _is_rt and _cfg_key in _pruned_configs:
                n_skipped += len(_rt_degrees) - _i
                break
            block = (spec.block_size, 1, 1)
            stream = 0

            if _is_rt:
                wg_K_padded = spec.wg_K_padded(split_k=_launch_sk)
                _ks_val = wg_K_padded // _launch_sk
                grid = _grid_for_wgrad_spec(spec, _launch_sk)
                values = {
                    "A": dY_dev,
                    "B": X_dev,
                    "D": dW_dev,
                    "A_bytes": dY_t.nbytes,
                    "B_bytes": X_t.nbytes,
                    "D_bytes": dW_t.nbytes,
                    "ks": _ks_val,
                }
                _is_atomic_launch = True
            else:
                grid = _grid_for_wgrad_spec(spec, _launch_sk)
                values = {
                    "A": dY_dev,
                    "B": X_dev,
                    "D": dW_dev,
                    "A_bytes": dY_t.nbytes,
                    "B_bytes": X_t.nbytes,
                    "D_bytes": dW_t.nbytes,
                }
                _is_atomic_launch = _launch_sk > 1

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
                    zero_init_out=_is_atomic_launch,
                    ref_out=ref_out,
                    kernel_name=artifact.kernel_name,
                    dump_fail=args.dump_fail,
                    extra_tensors={"dY": dY_t, "X": X_t},
                    u8=_u8,
                    arch=arch,
                )
                if stopped:
                    rt.free(dY_dev)
                    rt.free(X_dev)
                    rt.free(dW_dev)
                    return 1, []

            if _is_atomic_launch:
                _vals_snap = dict(values)
                _cfg_snap = cfg

                def _launch_spk(_v=_vals_snap, _c=_cfg_snap):
                    rt.memset(dW_dev, 0, dW_t.nbytes)
                    launcher(_v, config=_c)

                timed_fn = _launch_spk
            else:
                _vals_snap = dict(values)
                _cfg_snap = cfg
                timed_fn = lambda _v=_vals_snap, _c=_cfg_snap: launcher(_v, config=_c)

            ms = time_launches(
                timed_fn,
                warmup=args.warmup,
                iters=args.iters,
                stream=stream,
            )
            synchronize_and_release(stream)

            cur_tflops = (flop / ms) * 1e-9
            cur_gbps = (bytes_xfer / ms) * 1e-6
            n_run += 1

            _va, _vb, _vc = WgradConvSpec.default_vector_sizes(
                p.C, p.K, dtype, split_k=_launch_sk
            )
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
                    split_k=_launch_sk,
                    ms=ms,
                    tflops=cur_tflops,
                    gbps=cur_gbps,
                    vec_a=_va,
                    vec_b=_vb,
                    vec_c=_vc,
                )
            )

            _spk_label = f"spk{_launch_sk}rt" if _is_rt else f"spk{_launch_sk}"
            prune_marker = ""
            if _do_prune:
                _cfg_key = (
                    tile_m,
                    tile_n,
                    tile_k,
                    warp_m,
                    warp_n,
                    warp_tile_mn,
                    pipeline,
                    epilogue,
                )
                best = _best_tflops.get(_cfg_key)
                if best is None:
                    _best_tflops[_cfg_key] = cur_tflops
                else:
                    if cur_tflops > best:
                        _best_tflops[_cfg_key] = cur_tflops
                    elif (best - cur_tflops) / best >= _prune_threshold:
                        _pruned_configs.add(_cfg_key)
                        prune_marker = f"  [pruned: {cur_tflops:.1f} < {best:.1f} * {1 - _prune_threshold:.2f}]"

            print(
                f"[{n_run:4d}] tile={tile_m}x{tile_n}x{tile_k} "
                f"warp={warp_m}x{warp_n} "
                f"atom={warp_tile_mn}x{warp_tile_mn}x{warp_tile_k} "
                f"{pipeline}/{epilogue:9s} {_spk_label:<7s} "
                f"vec={_va}/{_vb}/{_vc} "
                f"{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms"
                f"{prune_marker}",
                flush=True,
            )

    rt.free(dY_dev)
    rt.free(X_dev)
    rt.free(dW_dev)

    print(f"\nWgrad sweep done: {n_built} compiled, {n_skipped} skipped.", flush=True)

    if not results:
        print("No valid wgrad configurations found.", file=sys.stderr)
        return 1, []

    results.sort(key=lambda r: r.tflops, reverse=True)
    top_n = min(args.top, len(results))

    print(f"\n{'='*84}")
    print(f"Top {top_n} wgrad configurations for {arch} {dtype} {p.short()}")
    print(f"{'='*84}")
    hdr = f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
    print(hdr)
    print("-" * 84)
    for rank, r in enumerate(results[:top_n], 1):
        cfg_str = (
            f"tile={r.tile_m}x{r.tile_n}x{r.tile_k} "
            f"warp={r.warp_m}x{r.warp_n} "
            f"atom={r.warp_tile_mn}x{r.warp_tile_mn}x{r.warp_tile_k} "
            f"vec={r.vec_a}/{r.vec_b}/{r.vec_c} "
            f"{r.pipeline}/{r.epilogue} spk{r.split_k}"
        )
        print(f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {cfg_str}")

    best = results[0]
    print(f"\nBest: {best.tflops:.1f} TFLOPS — {best.kernel_name}")
    return 0, results


def _run_dgrad_sweep(
    *,
    args,
    problem,
    dtype: str,
    arch: str,
    target,
    compile_kernel,
    jobs: int = 1,
    ConvDataSpec,
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    is_valid_dgrad_spec,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
    **_ignored,
) -> int:
    """Sweep dgrad configurations and rank by TFLOPS.

    Dgrad GEMM dims:
        M    = N*Hi*Wi      (input spatial positions)
        N_dg = C            (input channels)
        K_dg = Y*X*K        (filter spatial x output channels -- reduction)

    Operands:
        A (dY): output gradient, shape (N, Ho, Wo, K)
        B (W):  weights, shape (K, Y, X, C)
        D (dX): input gradient, shape (N, Hi, Wi, C)

    Split-K (``--split-k``):
        1        -- disabled (normal epilogue, z-grid = 1).
        >1       -- fixed degree; dX is zero-initialised before each launch,
                   kernel atomic-adds partials, result is final dX.
        0 (auto) -- sweep all degrees in _SPLIT_K_AUTO.
    """
    import ctypes
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

    _dY_f32 = _make(p.N, p.Ho, p.Wo, p.K)
    # Weight is stored per-group packed: KYXC with channel extent cpg = C/groups
    # (== C when groups == 1), matching the dgrad w_descriptor and the forward
    # make_b_descriptor.  dY/dX carry full K/C (the group rides the k_out/c index).
    _W_f32 = _make(p.K, p.Y, p.X, p.cpg)
    dX_t = torch.empty(p.N, p.Hi, p.Wi, p.C, dtype=_torch_dtype)

    dY_t = _dY_f32.to(_torch_dtype)
    W_t = _W_f32.to(_torch_dtype)

    bytes_xfer = float(dY_t.nbytes + W_t.nbytes + dX_t.nbytes)
    flop = float(p.flops)

    # Per-group channel runs (cpg/kpg) bound the vector widths for grouped dgrad
    # so loads/stores never straddle a group boundary (cpg==C, kpg==K ungrouped).
    vec_a, vec_b, vec_c = DgradConvSpec.default_vector_sizes(p.cpg, p.kpg, dtype)
    base_sig = conv_args_signature(dtype)
    ext_sig = base_sig + [
        {"name": "sub_gemm_buf", "type": "ptr<i32, global>", "size_bytes": 8},
        {"name": "num_sub_gemms", "type": "i32", "size_bytes": 4},
    ]

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
            f"Sampling {len(combos)}/{total} dgrad combinations "
            f"({args.sample*100:.0f}%, seed={args.seed}).",
            flush=True,
        )

    _spk_label = {0: "sweep", -1: "auto(CK)"}.get(args.split_k, str(args.split_k))
    print(
        f"Sweeping {len(combos)} dgrad combinations for {arch} {dtype} {p.short()} "
        f"(split_k={_spk_label}) ...",
        flush=True,
    )

    from rocke.instances.common.conv_implicit_gemm_dgrad import pack_sub_gemm_buffer
    import struct as _struct

    # ---- Phase 1: build + validate all specs, collect kernels (parallel) ----
    if jobs != 1:
        print(f"Building IR for {len(combos)} dgrad combos in parallel ...", flush=True)
    work = [(combo, problem, dtype, arch, vec_a, vec_b, vec_c) for combo in combos]
    pending = _build_ir_parallel(work, _build_dgrad_one, jobs)
    n_skipped = len(combos) - len(pending)

    # ---- Phase 2: compile in parallel ----
    artifact_map = _compile_kernels_parallel(
        [k for _, _, _, k in pending], compile_kernel, arch, jobs
    )
    n_built = len(artifact_map)
    n_skipped += len(pending) - n_built

    print(
        f"Compiled {n_built}/{len(pending)} dgrad kernels "
        f"({n_skipped} skipped total).",
        flush=True,
    )

    # ---- Phase 3: allocate GPU buffers, verify, benchmark ----
    rt = Runtime()
    results: List[Result] = []

    dY_dev = rt.alloc(dY_t.nbytes)
    W_dev = rt.alloc(W_t.nbytes)
    dX_dev = rt.alloc(dX_t.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY_t), dY_t.nbytes)
    rt.memcpy_h2d(W_dev, _u8(W_t), W_t.nbytes)
    rt.memset(dX_dev, 0, dX_t.nbytes)

    ref_out: torch.Tensor | None = None
    if args.verify or args.dump_fail:
        if arch == "gfx1250":
            from rocke.benchmark.conv_reference import dgrad_reference_gfx1250

            ref_out = dgrad_reference_gfx1250(
                _dY_f32, _W_f32, p, out_dtype=_torch_dtype
            )
            print(
                f"Dgrad reference computed via gfx1250 hand-written dgrad "
                f"({tuple(ref_out.shape)}, {ref_out.dtype}).",
                flush=True,
            )
        else:
            from rocke.benchmark.conv_reference import dgrad_reference

            ref_out = dgrad_reference(_dY_f32, _W_f32, p)
            print(
                f"Dgrad reference computed ({tuple(ref_out.shape)}, {ref_out.dtype}).",
                flush=True,
            )

    n_measured = 0
    for _combo, spec, resolved_split_k, kernel in pending:
        artifact = artifact_map.get(kernel.name)
        if artifact is None:
            continue

        sub_gemms = spec.compute_sub_gemms()
        buf_i32 = pack_sub_gemm_buffer(sub_gemms, spec.tile_m, spec.tile_n)
        buf_bytes = _struct.pack(f"{len(buf_i32)}i", *buf_i32)
        sgbuf_dev = rt.alloc(len(buf_bytes))
        rt.memcpy_h2d(
            sgbuf_dev,
            (ctypes.c_uint8 * len(buf_bytes)).from_buffer_copy(buf_bytes),
            len(buf_bytes),
        )
        flat_tiles = sub_gemms[-1].block_end
        # Conv group rides blockIdx.y (see conv_implicit_gemm_dgrad); the tilde
        # sub-GEMM geometry is channel-independent so flat_tiles is per-group.
        _groups = max(int(spec.problem.groups), 1)
        grid = (flat_tiles, _groups, resolved_split_k)
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

        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=ext_sig,
        )
        block = (spec.launch_block_size, 1, 1)
        stream = 0
        cfg = LaunchConfig(grid=grid, block=block, stream=stream)

        _zero_init = spec.needs_atomic

        if args.verify or args.dump_fail:
            stopped, _ = _verify_kernel(
                rt=rt,
                launcher=launcher,
                values=values,
                grid=grid,
                block=block,
                out_dev=dX_dev,
                out_t=dX_t,
                zero_init_out=_zero_init,
                ref_out=ref_out,
                kernel_name=artifact.kernel_name,
                dump_fail=args.dump_fail,
                extra_tensors={"dY": dY_t, "W": W_t},
                u8=_u8,
                arch=arch,
            )
            if stopped:
                rt.free(sgbuf_dev)
                rt.free(dY_dev)
                rt.free(W_dev)
                rt.free(dX_dev)
                return 1

        if _zero_init:

            def _launch_atomic():
                rt.memset(dX_dev, 0, dX_t.nbytes)
                launcher(values, config=cfg)

            timed_fn = _launch_atomic
        else:
            timed_fn = lambda: launcher(values, config=cfg)

        ms = time_launches(
            timed_fn, warmup=args.warmup, iters=args.iters, stream=stream
        )
        synchronize_and_release(stream)

        cur_tflops = (flop / ms) * 1e-9
        cur_gbps = (bytes_xfer / ms) * 1e-6
        n_measured += 1

        results.append(
            Result(
                kernel_name=artifact.kernel_name,
                tile_m=spec.tile_m,
                tile_n=spec.tile_n,
                tile_k=spec.tile_k,
                warp_m=spec.warp_m,
                warp_n=spec.warp_n,
                warp_tile_mn=spec.warp_tile_m,
                warp_tile_k=spec.warp_tile_k,
                pipeline=spec.pipeline,
                epilogue=spec.epilogue,
                split_k=resolved_split_k,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
                vec_a=vec_a,
                vec_b=vec_b,
                vec_c=vec_c,
            )
        )

        _lva = vec_a if resolved_split_k <= 1 else 1
        print(
            f"[{n_measured:4d}] tile={spec.tile_m}x{spec.tile_n}x{spec.tile_k} "
            f"warp={spec.warp_m}x{spec.warp_n} "
            f"atom={spec.warp_tile_m}x{spec.warp_tile_n}x{spec.warp_tile_k} "
            f"{spec.pipeline}/{spec.epilogue:9s} spk{resolved_split_k:<3d} "
            f"vec={_lva}/{vec_b}/{vec_c} "
            f"{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

        rt.free(sgbuf_dev)

    rt.free(dY_dev)
    rt.free(W_dev)
    rt.free(dX_dev)

    print(
        f"\nDgrad sweep done: {n_measured} measured, {n_skipped} skipped.", flush=True
    )

    if not results:
        print("No valid dgrad configurations found.", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r.tflops, reverse=True)
    top_n = min(args.top, len(results))

    print(f"\n{'='*80}")
    print(f"Top {top_n} dgrad configurations for {arch} {dtype} {p.short()}")
    print(f"{'='*80}")
    hdr = f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
    print(hdr)
    print("-" * 80)
    for rank, r in enumerate(results[:top_n], 1):
        _lva_r = r.vec_a if r.split_k <= 1 else 1
        cfg_str = (
            f"tile={r.tile_m}x{r.tile_n}x{r.tile_k} "
            f"warp={r.warp_m}x{r.warp_n} "
            f"atom={r.warp_tile_mn}x{r.warp_tile_mn}x{r.warp_tile_k} "
            f"{r.pipeline}/{r.epilogue} spk{r.split_k} "
            f"vec={_lva_r}/{r.vec_b}/{r.vec_c}"
        )
        print(f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {cfg_str}")

    best = results[0]
    print(f"\nBest: {best.tflops:.1f} TFLOPS -- {best.kernel_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
