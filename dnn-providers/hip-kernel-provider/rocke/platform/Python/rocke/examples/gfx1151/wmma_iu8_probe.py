# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bit-exact verify of the RDNA3/3.5 native integer WMMA (iu8) lane map.

The deep-fusion port to native integer WMMA depends on the
``wmma_i32_16x16x16_iu8`` fragment ABI (A/B ``<4 x i32>``, acc/result
``<8 x i32>``). This probe pins those A/B/accumulator lane maps *before* the
kernel uses them: it builds the standalone iu8 GEMM
(:mod:`rocke.instances.gfx1151.wmma_gemm_iu8`) and verifies ``C = A @ B.T``
against an **integer** numpy reference, expecting **exact** ``max_abs_diff == 0``
(integer WMMA does no rounding). Random asymmetric A/B uniquely confirm the
row/col mapping — a row<->col swap transposes the result and fails.

Cross-build (this host has no gfx11 GPU; the Windows board runs it):

  # local: build hsaco + manifest into an output dir
  PYTHONPATH=Python python3 -m rocke.examples.gfx1151.wmma_iu8_probe \
      --arch gfx11-generic --m 64 --n 64 --k 64 --output-dir /tmp/iu8_probe --emit-only

  # board (after scp of the output dir to C:\\ck_test\\iu8_probe):
  set ROCKE_HIP_LIB=C:\\Windows\\System32\\amdhip64_7.dll
  python -m rocke.run_manifest C:\\ck_test\\iu8_probe\\<name>.hsaco \
      C:\\ck_test\\iu8_probe\\manifest.json --shape 64,64,64 --verify

``--verify`` (without ``--emit-only``) runs ``run_manifest`` locally, for the
day this is run directly on a gfx11 device.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from rocke.helpers import compile_kernel, make_gemm_manifest, write_artifact
from rocke.instances.gfx1151.wmma_gemm_iu8 import (
    WmmaGemmIu8Spec,
    build_wmma_gemm_iu8,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx11-generic")
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--emit-only",
        action="store_true",
        help="build hsaco + manifest and exit (no local launch); for board run",
    )
    args = p.parse_args()

    for d, name in ((args.m, "M"), (args.n, "N"), (args.k, "K")):
        if d % 16:
            raise SystemExit(
                f"{name}={d} must be a multiple of 16 (WMMA 16x16x16 tile)"
            )

    spec = WmmaGemmIu8Spec(name=f"wmma_iu8_probe_{args.arch}".replace("-", "_"))
    art = compile_kernel(build_wmma_gemm_iu8(spec, arch=args.arch), arch=args.arch)
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa})")

    out = Path(args.output_dir or f"/tmp/wmma_iu8_probe_{args.arch}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=16,
        block_n=16,
        block_k=16,
        threads_per_block=spec.block_size,  # 32 (wave32, one wave/block)
        default_shape=(args.m, args.n, args.k),
        grid_order="MN",  # kernel maps block_id.x -> M-tile
        atoms=["wmma_i32_16x16x16_iu8"],
        extra={"kind": "gemm_iu8"},
    )
    write_artifact(art, out, manifest)
    print(f"[{args.arch}] wrote artifact + manifest to {out}")

    if args.emit_only:
        print(
            "[emit-only] board: run "
            f"`python -m rocke.run_manifest {out / (art.kernel_name + '.hsaco')} "
            f"{out / 'manifest.json'} --shape {args.m},{args.n},{args.k} --verify`"
        )
        return 0

    cmd = [
        sys.executable,
        "-m",
        "rocke.run_manifest",
        str(out / f"{art.kernel_name}.hsaco"),
        str(out / "manifest.json"),
        "--shape",
        f"{args.m},{args.n},{args.k}",
        "--verify",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    sys.stdout.write(r.stdout)
    if r.returncode != 0 and "max_abs_diff" not in r.stdout:
        sys.stderr.write(r.stderr[-2000:])
        return r.returncode

    m = re.search(r"max_abs_diff=([0-9.eE+-]+)", r.stdout)
    if not m:
        print(f"[{args.arch}] FAIL: no verify output")
        return 1
    max_abs = float(m.group(1))
    ok = max_abs == 0.0
    print(
        f"[{args.arch}] iu8 WMMA lane-map probe {args.m}x{args.n}x{args.k}: "
        f"max_abs_diff={max_abs:.0f} -> "
        f"{'PASS (lane map confirmed, bit-exact)' if ok else 'FAIL (lane map wrong)'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
