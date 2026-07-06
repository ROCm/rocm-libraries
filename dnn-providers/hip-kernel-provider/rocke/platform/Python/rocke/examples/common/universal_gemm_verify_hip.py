# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HIP-path (hipcc) Universal GEMM build + numeric verify.

Identical to ``universal_gemm_verify`` except the kernel is compiled through
the HIP-C++ -> hipcc backend (``compile_kernel_via_hipcc``) instead of the
LLVM-IR -> libamd_comgr backend. Used by the arch-hardening VERIFY pass to
confirm the HIP lowering path (lower_hip.py) is numerically correct on each
arch: MFMA builtins on CDNA (gfx942/gfx950) and WMMA C++ builtins on RDNA
wave32 (gfx1151).

  PYTHONPATH=Python python3 -m rocke.examples.common.universal_gemm_verify_hip --arch gfx950
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from rocke.core.arch import ArchTarget
from rocke.helpers import make_gemm_manifest, write_artifact
from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)
from rocke.instances import GemmPipelinePolicy


def _pick_atom(target: ArchTarget, dtype: str, want):
    family = "wmma" if target.wave_size == 32 else "mma"
    if want is not None:
        return want
    op = target.mma.select_largest_k(
        family=family, a_dtype=dtype, b_dtype=dtype, c_dtype="fp32", m=16, n=16
    )
    if op is None:
        raise SystemExit(f"no f16/bf16 16x16 {family} atom for {dtype} on {target.gfx}")
    return (op.m, op.n, op.k)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx950")
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--warp-tile", default=None)
    p.add_argument("--pipeline", default="mem")
    p.add_argument("--epilogue", default="default")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--tol", type=float, default=None)
    args = p.parse_args()

    target = ArchTarget.from_gfx(args.arch)
    want = None
    if args.warp_tile:
        want = tuple(int(x) for x in args.warp_tile.lower().split("x"))
    wtm, wtn, wtk = _pick_atom(target, args.dtype, want)

    tile = TileSpec(
        tile_m=2 * 2 * wtm,
        tile_n=2 * 2 * wtn,
        tile_k=max(32, wtk),
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=wtm,
        warp_tile_n=wtn,
        warp_tile_k=wtk,
    )
    trait = TraitSpec(
        pipeline=args.pipeline,
        scheduler="intrawave",
        epilogue=args.epilogue,
        pad_m=True,
        pad_n=True,
        pad_k=True,
    )
    data = DataSpec(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        dtype_acc="fp32",
        layout="RCR",
    )
    spec = UniversalGemmSpec(
        name=f"ugemm_hip_{args.arch}",
        tile=tile,
        trait=trait,
        data=data,
        wave_size=target.wave_size,
    )

    policy = GemmPipelinePolicy()
    res = policy.validate(target, spec)
    print(
        f"[{args.arch}] HIP-path atom={wtm}x{wtn}x{wtk} "
        f"tile={tile.tile_m}x{tile.tile_n}x{tile.tile_k} "
        f"wave={target.wave_size} validate -> {res.ok} ({res.reason})"
    )
    if not res.ok:
        return 2

    art = compile_kernel_via_hipcc(
        build_universal_gemm(spec, arch=args.arch), arch=args.arch
    )
    print(
        f"[{args.arch}] HIP-path built {art.kernel_name} ({art.hsaco_bytes} B, "
        f"isa={art.isa}) hipcc={art.timings.get('hipcc', 0):.0f}ms "
        f"total={art.timings.get('total', 0):.0f}ms"
    )

    out = Path(args.output_dir or f"/tmp/ugemm_verify_hip_{args.arch}")
    out.mkdir(parents=True, exist_ok=True)
    atom_family = "wmma" if target.wave_size == 32 else "mfma"
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=tile.tile_m,
        block_n=tile.tile_n,
        block_k=tile.tile_k,
        threads_per_block=spec.block_size,
        default_shape=(args.m, args.n, args.k),
        atoms=[f"{atom_family}_f32_{wtm}x{wtn}x{wtk}_{args.dtype}"],
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

    tol = (
        args.tol if args.tol is not None else (1e-2 if target.wave_size == 32 else 0.0)
    )
    m = re.search(r"max_abs_diff=([0-9.eE+-]+)", r.stdout)
    bad_m = re.search(r"bad=(\d+)/(\d+)", r.stdout)
    if m is None:
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-2000:])
        return r.returncode or 1
    max_abs = float(m.group(1))
    ok = max_abs <= tol
    bad_str = f" bad={bad_m.group(1)}/{bad_m.group(2)}" if bad_m else ""
    print(
        f"[{args.arch}] HIP-path GEMM {args.m}x{args.n}x{args.k}: "
        f"max_abs_diff={max_abs:.3e} tol={tol:.0e}{bad_str} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
