# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1151 ``MatMulNBits`` large-N kernel.

Builds the fp16-activation / packed-int4-weight large-N WMMA kernel, writes a
``matmul_nbits_fp16`` manifest, and runs ``rocke.run_manifest --verify`` (numpy
reference ``C = A @ dequant(B, scales)^T``). Must run on a gfx1151 device (e.g.
a gfx1151 SLURM node) for the verify step; ``--no-verify`` only builds and writes
the artifact so a remote node can run the numeric gate.

  python scripts/02_int4_matmul_nbits_verify.py --m 128 --n 4096 --k 4096
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
from rocke.instances import TileSpec  # noqa: E402
from rocke.instances.common.matmul_nbits import (
    MatMulNBitsSpec,
    build_matmul_nbits,
)  # noqa: E402
from rocke.instances.common._matmul_nbits_common import _scale_wire_dtype  # noqa: E402


def _write_data(name: str, payload: dict) -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / f"{name}.json").write_text(json.dumps(payload, indent=2))


def _subprocess_env() -> dict:
    """Ensure the run_manifest child can import rocke under file-run."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PYROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _large_n_spec(
    name: str, n: int, k: int, group: int, scale_dtype: str
) -> MatMulNBitsSpec:
    """The first tuned large-N geometry: 64x128x16, 2x2 warps, WMMA 16x16x16."""
    return MatMulNBitsSpec(
        name=name,
        N=n,
        K=k,
        tile=TileSpec(
            tile_m=64,
            tile_n=128,
            tile_k=16,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        group_size=group,
        scale_dtype=scale_dtype,
        family="large_n",
    )


def _args_signature(scale_wire: str) -> list:
    return [
        {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "B", "type": "ptr<i8, global>", "size_bytes": 8},
        {"name": "Scales", "type": f"ptr<{scale_wire}, global>", "size_bytes": 8},
        {"name": "C", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "M", "type": "i32", "size_bytes": 4},
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1151")
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--family", default="large_n")
    p.add_argument("--seq-len-tile", type=int, default=64)
    p.add_argument("--scale-dtype", default="fp16")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="max-abs-diff tolerance vs the fp32 numpy reference. WMMA "
        "accumulates in f32 in a different order than numpy, so a real fp "
        "matmul is judged within tolerance, not at zero-tol.",
    )
    args = p.parse_args()

    if args.family != "large_n":
        raise SystemExit(
            f"--family={args.family} not implemented yet (Milestone 6); "
            "only large_n is available"
        )

    spec = _large_n_spec(
        name=f"matmul_nbits_{args.arch}",
        n=args.n,
        k=args.k,
        group=args.group_size,
        scale_dtype=args.scale_dtype,
    )
    t = spec.tile
    if args.m % t.tile_m:
        raise SystemExit(
            f"M={args.m} must be a multiple of tile_m={t.tile_m} for the v1 "
            "large-N body (in-bounds A loads; partial-M tail is a follow-on)"
        )
    if args.n % t.tile_n:
        raise SystemExit(f"N={args.n} must be a multiple of tile_n={t.tile_n}")
    if args.k % t.tile_k:
        raise SystemExit(f"K={args.k} must be a multiple of tile_k={t.tile_k}")

    art = compile_kernel(build_matmul_nbits(spec, arch=args.arch), arch=args.arch)
    print(
        f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa}) "
        f"total={art.timings.get('total', 0):.1f}ms"
    )

    scale_wire = _scale_wire_dtype(args.scale_dtype)
    out = Path(args.output_dir or f"/tmp/matmul_nbits_verify_{args.arch}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=t.tile_m,
        block_n=t.tile_n,
        block_k=t.tile_k,
        threads_per_block=spec.block_size,
        default_shape=(args.m, args.n, args.k),
        grid_order="NM",
        args_signature=_args_signature(scale_wire),
        atoms=["wmma_f32_16x16x16_f16"],
        extra={
            "kind": "matmul_nbits_fp16",
            "N": args.n,
            "K": args.k,
            "group_size": args.group_size,
            "scale_dtype": scale_wire,
        },
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
        sys.stderr.write(r.stderr[-2000:])
        return r.returncode

    m = re.search(r"max_abs_diff=([0-9.eE+-]+)", r.stdout)
    if not m:
        print(f"[{args.arch}] FAIL: no verify output")
        return 1
    max_abs = float(m.group(1))
    ok = max_abs <= args.tol
    print(
        f"[{args.arch}] MatMulNBits {args.m}x{args.n}x{args.k} g{args.group_size}: "
        f"max_abs_diff={max_abs:.3e} tol={args.tol:.0e} -> {'PASS' if ok else 'FAIL'}"
    )
    _write_data(
        "02_int4_matmul_nbits_verify",
        {
            "kernel": "matmul_nbits large_n (int4 weight-only, W4A16)",
            "shape": {"M": args.m, "N": args.n, "K": args.k},
            "group_size": args.group_size,
            "scale_dtype": args.scale_dtype,
            "tol": args.tol,
            "max_abs_diff": max_abs,
            "pass": ok,
        },
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
