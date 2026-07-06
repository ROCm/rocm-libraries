# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Empirically verify the gfx1250 (gfx1250) WMMA fragment lane map on hardware.

The gfx1250 WMMA fp16/bf16 atom is **16x16x32** (K=32), distinct from gfx1201's
16x16x16. A/B operands are ``<16 x half>`` per lane (the 32 K-elements split
across the two lane-halves, 16 each) and the accumulator is the gfx12
column-distributed ``<8 x float>``. The lane maps encoded in
``core/arch/target.py::_wmma_gfx1250_*`` and the epilogue in
``instances/gfx1250/wmma_gemm.py`` are a *hypothesis* until proven on silicon.

This probe builds that WMMA GEMM and runs ``rocke.run_manifest --verify``
against the numpy reference ``C = A @ B.T`` with **random (asymmetric)** A/B. A
row/col swap in the lane map transposes the result and fails verify, so a PASS
at multiple tile counts uniquely confirms the mapping.

Must run on a gfx1250 device:

  PYTHONPATH=Python python3 -m rocke.examples.gfx1250.wmma_probe --m 16 --n 16 --k 32
  PYTHONPATH=Python python3 -m rocke.examples.gfx1250.wmma_probe --m 64 --n 64 --k 64
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from rocke.helpers import compile_kernel, make_gemm_manifest, write_artifact
from rocke.instances.gfx1250.wmma_gemm import WmmaGemmSpec, build_wmma_gemm


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1250")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="max-abs-diff tolerance vs the fp32 numpy reference. WMMA "
        "accumulates in f32 in a different order than numpy, so a real fp "
        "GEMM is judged within tolerance (not bit-exact).",
    )
    args = p.parse_args()

    for d, name, mult in ((args.m, "M", 16), (args.n, "N", 16), (args.k, "K", 32)):
        if d % mult:
            raise SystemExit(f"{name}={d} must be a multiple of {mult}")

    spec = WmmaGemmSpec(name=f"wmma_probe_{args.arch}")
    art = compile_kernel(build_wmma_gemm(spec, arch=args.arch), arch=args.arch)
    print(
        f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa}) "
        f"total={art.timings.get('total', 0):.1f}ms"
    )

    out = Path(args.output_dir or f"/tmp/wmma_probe_{args.arch}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=16,
        block_n=16,
        block_k=32,
        threads_per_block=spec.block_size,  # 32 (wave32, one wave/block)
        default_shape=(args.m, args.n, args.k),
        grid_order="MN",
        atoms=["wmma_gfx1250_f32_16x16x32_f16"],
    )
    write_artifact(art, out, manifest)

    if args.no_verify:
        print(f"[{args.arch}] wrote artifact to {out} (verify skipped)")
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
    ok = max_abs <= args.tol
    print(
        f"[{args.arch}] WMMA lane-map probe {args.m}x{args.n}x{args.k}: "
        f"max_abs_diff={max_abs:.3e} tol={args.tol:.0e} -> "
        f"{'PASS (lane map confirmed)' if ok else 'FAIL (lane map wrong)'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
