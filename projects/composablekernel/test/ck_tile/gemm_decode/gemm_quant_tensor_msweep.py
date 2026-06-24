#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
r"""FP8 per-tensor MFMA ceiling side: M-sweep for CKTile's gemm_quant kernel.

The CKTile block-scale example `tile_example_gemm_quant` exposes a per-tensor
FP8 path (QuantType::TensorQuant) built on GemmConfigQuantDecode<fp8_t>: an
MFMA pipeline with M_Warp_Tile=16, i.e. a fixed 16-row MMA tile. That makes it
the natural *MFMA ceiling* for small-M FP8 per-tensor GEMM -- it cannot dip
below the cost of a 16-row tile no matter how small M is, which is exactly the
regime gemm_decode (warp-per-scalar) is built to beat.

The kernel hard-rejects M < 16 (IsSupportedArgument fails: it has no M-tile
padding, so M must be a multiple of M_Tile=16). That is precisely the ceiling
we want to chart: in the decode regime (M=1..8) this MFMA kernel cannot dispatch
below a full 16-row tile, so its real cost there is the M=16 tile cost. We
therefore round each requested M up to the next multiple of the tile (M=1..16 ->
16), launch the kernel at that padded size, cache by padded size (so the whole
decode sweep is one launch), and report that fixed time as the ceiling for each
M. TF/s and GB/s are re-derived at the *nominal* M under the same byte model as
bench_msweep_fp8.cpp (fp8 A,B = 1 B/elem, 2-byte C; per-tensor scales are
negligible) -- so the low small-M TF/s honestly reflects the wasted 15/16 of the
MMA tile. The kernel's own reported numbers are kept in the config column.

  /opt/venv/bin/python3 gemm_quant_tensor_msweep.py \
    --exe /path/to/build/bin/tile_example_gemm_quant \
    --N 8192 --K 7168 --mmax 8 --warmup 25 --repeat 200 \
    --csv-out /tmp/ck_gemm_quant_tensor_8192x7168.csv

Emits CSV columns: impl,M,N,K,time_us,tflops,gbytes_s,config
(impl = ck_gemm_quant_tensor)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

# "... : 0.0123 ms, 456.7 TFlops, 1234.5 GB/s," (scientific notation tolerated)
_PERF_RE = re.compile(
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*ms,\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*TFlops,\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*GB/s"
)
_VERIFY_RE = re.compile(r"CPU verification result is:\s*(correct|fail)")

# Where the example binary usually lands across the local CK checkouts.
_EXE_GLOBS = [
    "/home/AMD/samremes/dev/rocm-libraries/projects/composablekernel/build*/"
    "bin/tile_example_gemm_quant",
    "/home/AMD/samremes/dev/*/projects/composablekernel/build*/"
    "bin/tile_example_gemm_quant",
    "/home/AMD/samremes/dev/**/tile_example_gemm_quant",
]


def _find_exe(explicit: str) -> str:
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"--exe {explicit} does not exist")
        return explicit
    for pat in _EXE_GLOBS:
        hits = sorted(glob.glob(pat, recursive=True), key=os.path.getmtime, reverse=True)
        if hits:
            print(f"# auto-located exe: {hits[0]}", file=sys.stderr)
            return hits[0]
    sys.exit("tile_example_gemm_quant not found; pass --exe explicitly")


def _run_cell(exe, prec, M, N, K, warmup, repeat, flush_cache, verify, timeout):
    """Run one (M,N,K) cell; return (ms, kern_tflops, kern_gbps, verify_str)."""
    cmd = [
        exe,
        f"-m={M}", f"-n={N}", f"-k={K}",
        "-quant_mode=tensor", f"-prec={prec}",
        "-a_layout=R", "-b_layout=C", "-c_layout=R",
        f"-warmup={warmup}", f"-repeat={repeat}",
        f"-flush_cache={'true' if flush_cache else 'false'}",
        f"-v={1 if verify else 0}",
        "-split_k=1", "-init=0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + "\n" + proc.stderr
    perf = None
    for line in out.splitlines():
        m = _PERF_RE.search(line)
        if m:
            perf = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    vmatch = _VERIFY_RE.search(out)
    vstr = vmatch.group(1) if vmatch else ("n/a" if not verify else "missing")
    if perf is None:
        tail = "\n".join(out.strip().splitlines()[-12:])
        raise RuntimeError(f"no perf line (rc={proc.returncode}); tail:\n{tail}")
    return perf[0], perf[1], perf[2], vstr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="", help="Path to tile_example_gemm_quant.")
    ap.add_argument("--prec", choices=("fp8", "bf8"), default="fp8")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--K", type=int, default=7168)
    ap.add_argument("--mmax", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--flush-cache", action="store_true",
                    help="Run with cold L2 (default warm, matching gemm_decode sweep).")
    ap.add_argument("--tile-m", type=int, default=16,
                    help="Kernel M-tile granularity; M is rounded up to a multiple "
                         "of this before launch (kernel rejects M<tile).")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--csv-out", default="/tmp/ck_gemm_quant_tensor.csv")
    args = ap.parse_args()

    exe = _find_exe(args.exe)
    N, K, Mmax, tile_m = args.N, args.K, args.mmax, args.tile_m
    print(f"# CKTile gemm_quant TensorQuant M-sweep (MFMA M={tile_m} ceiling): "
          f"prec={args.prec} N={N} K={K} Mmax={Mmax} warmup={args.warmup} "
          f"repeat={args.repeat} flush_cache={args.flush_cache} exe={exe}",
          file=sys.stderr)

    def padded_m(M):
        return ((M + tile_m - 1) // tile_m) * tile_m

    # Cache by the padded (actually-launched) tile size: the kernel can't run
    # M<tile, and every M in a tile maps to the same launch/time.
    cache = {}  # padded_M -> (t_us, k_tf, k_gb) or None on failure

    def launch(pm, verify):
        ms, k_tf, k_gb, vstr = _run_cell(exe, args.prec, pm, N, K, args.warmup,
                                         args.repeat, args.flush_cache,
                                         verify=verify, timeout=args.timeout)
        return ms * 1000.0, k_tf, k_gb, vstr

    # One CPU-verified launch at the smallest tile we will use, as a sanity check.
    sm0 = padded_m(1)
    try:
        _, _, _, vstr = launch(sm0, verify=True)
        print(f"# verify M={sm0} (tile): CPU check = {vstr}", file=sys.stderr)
        if vstr == "fail":
            print("# WARNING: CKTile gemm_quant CPU verification FAILED", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - verification is best-effort
        print(f"# verify M={sm0} raised {type(e).__name__}: {e}", file=sys.stderr)

    rows = []
    for M in range(1, Mmax + 1):
        pm = padded_m(M)
        if pm not in cache:
            try:
                cache[pm] = launch(pm, verify=False)[:3]
            except Exception as e:  # noqa: BLE001
                print(f"#   M={M} (->{pm}): {type(e).__name__}: {e}", file=sys.stderr)
                cache[pm] = None
        if cache[pm] is None:
            continue
        t_us, k_tf, k_gb = cache[pm]
        # Unified byte model (matches bench_msweep_fp8.cpp): fp8 A,B + 2-byte C.
        # Computed at the *nominal* M so small-M TF/s reflects the wasted tile.
        tflops = 2.0 * M * N * K / (t_us * 1e-6) / 1e12
        gbps = (M * K + N * K + M * N * 2) / (t_us * 1e-6) / 1e9
        tile_tag = f"MFMA{tile_m}" + (f"=M{pm}tile" if pm != M else "")
        cfg = f"{args.prec}/tensor/{tile_tag}/flush{int(args.flush_cache)}/k{k_tf:.0f}TF"
        rows.append(("ck_gemm_quant_tensor", M, N, K, t_us, tflops, gbps, cfg))
        print(f"#   M={M:2d} (->{pm:2d})  {t_us:8.2f}us  {tflops:6.2f} TF/s  "
              f"{gbps:7.1f} GB/s  (kernel: {k_tf:.1f} TF/s {k_gb:.0f} GB/s)",
              file=sys.stderr)

    if not rows:
        print("# no gemm_quant cells succeeded; aborting", file=sys.stderr)
        return 1

    with open(args.csv_out, "w") as f:
        f.write("impl,M,N,K,time_us,tflops,gbytes_s,config\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.3f},{r[5]:.3f},"
                    f"{r[6]:.2f},{r[7]}\n")
    print(f"# wrote {len(rows)} rows -> {args.csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
