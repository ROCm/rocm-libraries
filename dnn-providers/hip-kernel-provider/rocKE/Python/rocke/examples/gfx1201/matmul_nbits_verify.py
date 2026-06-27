# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1201 (RDNA4) ``MatMulNBits`` kernels.

Same driver as ``examples/gfx1151/matmul_nbits_verify.py`` but defaults to the
gfx1201 arch and the gfx12 WMMA atom (``wmma_gfx12_f32_16x16x16_f16``). All
three families are buildable here, covering the full Qwen3.5-9B MatMulNBits set:

  * ``large_n`` — WMMA tiled body, N a multiple of tile_n=128 (attn/FFN proj).
  * ``skinny_n`` — same WMMA body with tile_n=32 (the N=32 linear-attn in_proj).
  * ``decode_gemv`` — scalar one-thread-per-column body for lm_head
    (N=248320, M=1); arch-agnostic, no WMMA.

The WMMA families share one source (``instances/common/_matmul_nbits_large_n.py``)
and branch the fragment ABI per arch, so this only differs from the gfx1151
driver in the manifest atom tag. Must run on a gfx1201 device (Navi 48) for the
verify step; ``--no-verify`` only builds + writes.

Real Qwen3.5-9B shapes (int4 / g32), pick via ``--family`` + ``--n/--k``:
  large_n:   N=4096/K=4096, 12288/4096, 4096/12288, 8192/4096, 1024/4096
  skinny_n:  N=32/K=4096
  decode_gemv: N=248320/K=4096 (M=1)

  PYTHONPATH=Python python3 -m rocke.examples.gfx1201.matmul_nbits_verify \
      --m 128 --n 4096 --k 4096
  PYTHONPATH=Python python3 -m rocke.examples.gfx1201.matmul_nbits_verify \
      --family skinny_n --m 128 --n 32 --k 4096
  PYTHONPATH=Python python3 -m rocke.examples.gfx1201.matmul_nbits_verify \
      --family decode_gemv --m 1 --n 248320 --k 4096
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from rocke.helpers import compile_kernel, make_gemm_manifest, write_artifact
from rocke.instances import TileSpec
from rocke.instances.common.matmul_nbits import MatMulNBitsSpec, build_matmul_nbits
from rocke.instances.common._matmul_nbits_common import _scale_wire_dtype

_WMMA_ATOM = {
    "gfx1151": "wmma_f32_16x16x16_f16",
    "gfx1201": "wmma_gfx12_f32_16x16x16_f16",
}

# Per-family tile geometry. large_n/skinny_n feed the WMMA atom; decode_gemv is a
# scalar body whose only structural field is block_size (warp_m*warp_n*wave).
_FAMILY_TILE = {
    # 64x128x16, 2x2 warps -> block_size 128.
    "large_n": TileSpec(
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
    # 64x32x16, 2x1 warps -> block_size 64. Narrow N tile for N=32.
    "skinny_n": TileSpec(
        tile_m=64,
        tile_n=32,
        tile_k=16,
        warp_m=2,
        warp_n=1,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
    ),
    # Scalar GEMV: 1x8 warps -> block_size 256 (one thread per output column).
    # tile_m/tile_n/tile_k are not load-bearing for the scalar body.
    "decode_gemv": TileSpec(
        tile_m=1,
        tile_n=256,
        tile_k=16,
        warp_m=1,
        warp_n=8,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
    ),
}


# Combined-opt large_n needs tile_k == group_size (32) so the scale is constant
# across the K tile (scale-on-accumulator). Same 64x128 output tile, 2x2 warps.
_OPT_TILE = TileSpec(
    tile_m=64,
    tile_n=128,
    tile_k=32,
    warp_m=2,
    warp_n=2,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=16,
)


def _spec(
    name: str,
    family: str,
    n: int,
    k: int,
    group: int,
    scale_dtype: str,
    optimized: bool = False,
) -> MatMulNBitsSpec:
    tile = _OPT_TILE if optimized else _FAMILY_TILE[family]
    return MatMulNBitsSpec(
        name=name,
        N=n,
        K=k,
        tile=tile,
        group_size=group,
        scale_dtype=scale_dtype,
        family=family,
        optimized=optimized,
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
    p.add_argument("--arch", default="gfx1201")
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument(
        "--family", default="large_n", choices=["large_n", "skinny_n", "decode_gemv"]
    )
    p.add_argument(
        "--opt",
        action="store_true",
        help="Build the combined-optimization large_n body "
        "(LDS-staged A + tile_k=group scale-on-acc + LDS "
        "epilogue transpose). Requires family=large_n.",
    )
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

    if args.arch not in _WMMA_ATOM:
        raise SystemExit(f"--arch={args.arch} not supported here ({list(_WMMA_ATOM)})")

    if args.opt and args.family != "large_n":
        raise SystemExit("--opt is only supported for --family large_n")

    spec = _spec(
        name=f"matmul_nbits_{args.arch}{'_opt' if args.opt else ''}",
        family=args.family,
        n=args.n,
        k=args.k,
        group=args.group_size,
        scale_dtype=args.scale_dtype,
        optimized=args.opt,
    )
    t = spec.tile

    # Geometry preconditions. The WMMA families load full tile_m A rows in
    # bounds (store is row-guarded) and require N a multiple of tile_n; the
    # scalar decode_gemv body guards both M and N per element.
    if args.k % t.tile_k:
        raise SystemExit(f"K={args.k} must be a multiple of tile_k={t.tile_k}")
    if args.family != "decode_gemv":
        if args.m % t.tile_m:
            raise SystemExit(
                f"M={args.m} must be a multiple of tile_m={t.tile_m} for the v1 "
                f"{args.family} body (in-bounds A loads; partial-M tail is a follow-on)"
            )
        if args.n % t.tile_n:
            raise SystemExit(f"N={args.n} must be a multiple of tile_n={t.tile_n}")

    art = compile_kernel(build_matmul_nbits(spec, arch=args.arch), arch=args.arch)
    print(
        f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa}) "
        f"total={art.timings.get('total', 0):.1f}ms"
    )

    # decode_gemv is scalar: one block of block_size threads covers block_size
    # output columns (block_m=1). The WMMA families launch one block per
    # tile_m x tile_n output tile. atoms[] is empty for the scalar body.
    if args.family == "decode_gemv":
        block_m, block_n = 1, spec.block_size
        atoms: list = []
    else:
        block_m, block_n = t.tile_m, t.tile_n
        atoms = [_WMMA_ATOM[args.arch]]

    scale_wire = _scale_wire_dtype(args.scale_dtype)
    out = Path(args.output_dir or f"/tmp/matmul_nbits_verify_{args.arch}_{args.family}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=block_m,
        block_n=block_n,
        block_k=t.tile_k,
        threads_per_block=spec.block_size,
        default_shape=(args.m, args.n, args.k),
        grid_order="NM",
        args_signature=_args_signature(scale_wire),
        atoms=atoms,
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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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
        f"[{args.arch}] MatMulNBits[{args.family}] {args.m}x{args.n}x{args.k} "
        f"g{args.group_size}: max_abs_diff={max_abs:.3e} tol={args.tol:.0e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
