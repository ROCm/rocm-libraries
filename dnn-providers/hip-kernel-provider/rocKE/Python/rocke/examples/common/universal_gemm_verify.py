# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Arch-parametric Universal GEMM build + numeric verify (hybrid-layout example).

The reference per-arch verification harness for the multi-arch work: build a
Universal GEMM for ``--arch`` (e.g. ``gfx942`` or ``gfx950``), validate it with
the GEMM family policy, compile via ``compile_kernel(arch=...)``, write a
manifest, and run ``rocke.run_manifest --verify`` (numpy reference,
``ref = A @ B.T`` for RCR fp16/bf16).

Usage (from the composablekernel dir, on a node of the matching arch):
  PYTHONPATH=Python python3 -m rocke.examples.common.universal_gemm_verify --arch gfx942
  PYTHONPATH=Python python3 -m rocke.examples.common.universal_gemm_verify --arch gfx950 --m 512 --n 512 --k 512

If ``--warp-tile`` is omitted the harness asks the target's MMA catalog for the
largest-K legal atom for the chosen dtype, so the same command produces a
gfx942-legal kernel on gfx942 and a wider gfx950 kernel on gfx950.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from rocke.core.arch import ArchTarget
from rocke.helpers import compile_kernel, make_gemm_manifest, write_artifact
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)
from rocke.instances import GemmPipelinePolicy


def _pick_atom(target: ArchTarget, dtype: str, want):
    """Resolve the warp-tile atom: explicit (m,n,k) or catalog largest-K for m=n=16.

    The MMA family follows the target wave size: CDNA (wave64) -> MFMA, RDNA
    wave32 (gfx11xx) -> WMMA, so the same invocation picks the right atom for
    the arch under test.
    """
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
    p.add_argument("--arch", default="gfx950", help="gfx target (gfx942, gfx950, ...)")
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument(
        "--warp-tile",
        default=None,
        help="MFMA atom as MxNxK (e.g. 16x16x16); default = catalog largest-K",
    )
    p.add_argument("--pipeline", default="mem")
    p.add_argument("--epilogue", default="default")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument(
        "--tol",
        type=float,
        default=None,
        help="max-abs-diff tolerance vs the fp32 numpy reference. "
        "Default: 0 for CDNA/MFMA (which matches numpy's f32 "
        "accumulation order exactly on the integer test inputs) "
        "and 1e-2 for RDNA/WMMA (which accumulates in a different "
        "order, so exact parity does not hold).",
    )
    args = p.parse_args()

    target = ArchTarget.from_gfx(args.arch)
    want = None
    if args.warp_tile:
        want = tuple(int(x) for x in args.warp_tile.lower().split("x"))
    wtm, wtn, wtk = _pick_atom(target, args.dtype, want)

    # Choose a block tile that satisfies the GEMM geometry rules for this atom:
    # tile_m/n = 2*warp * warp_tile (2x2 warps), tile_k = max(32, wtk).
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
    # Wave size is an arch fact: CDNA is wave64, the RDNA wave32 targets
    # (gfx11xx) are wave32. It is baked into block_size, so it must match the
    # target the unified body resolves its MMA atom for.
    spec = UniversalGemmSpec(
        name=f"ugemm_{args.arch}",
        tile=tile,
        trait=trait,
        data=data,
        wave_size=target.wave_size,
    )

    policy = GemmPipelinePolicy()
    res = policy.validate(target, spec)
    print(
        f"[{args.arch}] atom={wtm}x{wtn}x{wtk} tile={tile.tile_m}x{tile.tile_n}x{tile.tile_k} "
        f"wave={target.wave_size} validate -> {res.ok} ({res.reason})"
    )
    if not res.ok:
        return 2

    art = compile_kernel(build_universal_gemm(spec, arch=args.arch), arch=args.arch)
    print(
        f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, "
        f"isa={art.isa}) total={art.timings.get('total', 0):.1f}ms"
    )

    out = Path(args.output_dir or f"/tmp/ugemm_verify_{args.arch}")
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

    # ``run_manifest --verify`` for GEMM uses an exact (``diff > 0``) bad-count
    # gate, which the MFMA path passes (it reproduces numpy's f32 accumulation
    # order on the integer test inputs). WMMA accumulates the f32 partials in a
    # different order, so a handful of elements drift by ~1 fp16 ULP; we judge
    # the WMMA result against the reported ``max_abs_diff`` and a tolerance
    # instead of the exact bad count (matching the standalone wmma_gemm_verify).
    tol = (
        args.tol if args.tol is not None else (1e-2 if target.wave_size == 32 else 0.0)
    )
    m = re.search(r"max_abs_diff=([0-9.eE+-]+)", r.stdout)
    if m is None:
        # No verify line -> a real launch/runtime failure, not a tolerance miss.
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-2000:])
        return r.returncode or 1
    max_abs = float(m.group(1))
    ok = max_abs <= tol
    print(
        f"[{args.arch}] GEMM {args.m}x{args.n}x{args.k}: "
        f"max_abs_diff={max_abs:.3e} tol={tol:.0e} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
