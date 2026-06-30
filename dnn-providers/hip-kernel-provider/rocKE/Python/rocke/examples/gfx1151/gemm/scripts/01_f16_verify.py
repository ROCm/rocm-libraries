# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1151 WMMA GEMM on a Strix Halo node.

Builds the RDNA3.5 WMMA GEMM, writes a gemm manifest, and runs
``rocke.run_manifest --verify`` (numpy reference ``C = A @ B.T``, RCR f16).
Must run on a gfx1151 device (e.g. a gfx1151 SLURM node).

  python scripts/01_f16_verify.py --m 128 --n 128 --k 128
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # examples/gfx1151/gemm
_PYROOT = Path(__file__).resolve().parents[5]  # python root
sys.path.insert(0, str(_PYROOT))

from rocke.helpers import (
    compile_kernel,
    make_gemm_manifest,
    write_artifact,
)  # noqa: E402
from rocke.instances.gfx1151.wmma_gemm import (
    WmmaGemmSpec,
    build_wmma_gemm,
)  # noqa: E402


def _write_data(name: str, payload: dict) -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / f"{name}.json").write_text(json.dumps(payload, indent=2))


def _subprocess_env() -> dict:
    """Ensure the run_manifest child process can import rocke when this script
    is run as a file (not via ``-m`` with PYTHONPATH set)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PYROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1151")
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="max-abs-diff tolerance vs the fp32 numpy reference. "
        "WMMA accumulates in f32 in a different order than numpy, "
        "so exact (zero-tol) parity only holds by luck on integer "
        "inputs; a real fp GEMM is judged within tolerance.",
    )
    args = p.parse_args()

    for d, name in ((args.m, "M"), (args.n, "N"), (args.k, "K")):
        if d % 16:
            raise SystemExit(
                f"{name}={d} must be a multiple of 16 (WMMA 16x16x16 tile)"
            )

    spec = WmmaGemmSpec(name=f"wmma_gemm_{args.arch}")
    art = compile_kernel(build_wmma_gemm(spec, arch=args.arch), arch=args.arch)
    print(
        f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa}) "
        f"total={art.timings.get('total', 0):.1f}ms"
    )

    out = Path(args.output_dir or f"/tmp/wmma_gemm_verify_{args.arch}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=16,
        block_n=16,
        block_k=16,
        threads_per_block=spec.block_size,  # 32 (wave32, one wave/block)
        default_shape=(args.m, args.n, args.k),
        grid_order="MN",
        atoms=["wmma_f32_16x16x16_f16"],
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
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=180, env=_subprocess_env()
    )
    sys.stdout.write(r.stdout)
    if r.returncode != 0 and "max_abs_diff" not in r.stdout:
        # A real launch/runtime failure (not just a zero-tol mismatch).
        sys.stderr.write(r.stderr[-2000:])
        return r.returncode

    m = re.search(r"max_abs_diff=([0-9.eE+-]+)", r.stdout)
    if not m:
        print(f"[{args.arch}] FAIL: no verify output")
        return 1
    max_abs = float(m.group(1))
    ok = max_abs <= args.tol
    print(
        f"[{args.arch}] WMMA GEMM {args.m}x{args.n}x{args.k}: "
        f"max_abs_diff={max_abs:.3e} tol={args.tol:.0e} -> {'PASS' if ok else 'FAIL'}"
    )
    _write_data(
        "01_f16_verify",
        {
            "kernel": "wmma_gemm (f16 baseline)",
            "shape": {"M": args.m, "N": args.n, "K": args.k},
            "tol": args.tol,
            "max_abs_diff": max_abs,
            "pass": ok,
        },
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
