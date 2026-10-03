# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build step-0 candidate kernels into per-kernel HSACO + manifest directories.

``step0_lever_sweep.py`` produces the measurement-ready candidate set; this
turns those candidates into artifacts a GPU host can run with
``python -m rocke.run_manifest <hsaco> <manifest> --verify``.

``universal_gemm_verify`` cannot drive these: it hardcodes a 2x2 warp grid and
derives the block tile from the atom, so it cannot express the swept geometry
or the LDS / occupancy levers.

Each candidate lands in ``<out>/<kernel_name>/`` holding one ``.hsaco`` and one
``manifest.json``, which is the layout ``run_manifest`` and the remote-test
orchestrator both expect.

Artifacts and manifests carry shapes and resource counts, never timings
(``platform/AGENTS.md`` compliance).

Usage::

    PYTHONPATH=platform/python python -m rocke.examples.gfx1250.gemm.build_candidates \\
        --candidates ~/rocke_step0/measurement_candidates.json \\
        --out ~/rocke_step0/artifacts --limit 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rocke.helpers import compile_kernel, make_gemm_manifest, write_artifact
from rocke.instances.common.gemm_universal import (
    DataSpec,
    mono_data_spec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)

ARCH = "gfx1250"
ATOM = (16, 16, 32)

# The simplest configuration that builds on the gfx1250 WMMA path. Everything
# swept is measured against this, since gfx1250 has no registered dispatcher
# candidate to beat (dispatch/gemm/fp16_rcr.py:232 omits it).
BASELINE: Dict[str, Any] = {
    "kernel_name": "baseline",
    "geometry": {
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 32,
        "warp_m": 2,
        "warp_n": 2,
    },
    "traits": {
        "pipeline": "mem",
        "waves_per_eu": None,
        "lds_k_pad": 0,
        "lds_swizzle": False,
        "pad_all": False,
    },
}


def spec_from_candidate(c: Dict[str, Any], name: str) -> UniversalGemmSpec:
    g, t = c["geometry"], c["traits"]
    pad = bool(t.get("pad_all", False))
    return UniversalGemmSpec(
        name=name,
        tile=TileSpec(
            tile_m=g["tile_m"],
            tile_n=g["tile_n"],
            tile_k=g["tile_k"],
            warp_m=g["warp_m"],
            warp_n=g["warp_n"],
            warp_k=1,  # DEFECTS['warp_k_dead'] -- pinned
            warp_tile_m=ATOM[0],
            warp_tile_n=ATOM[1],
            warp_tile_k=ATOM[2],
        ),
        trait=TraitSpec(
            pipeline=t.get("pipeline", "mem"),
            scheduler="intrawave",  # proven no-op on this path
            epilogue="default",  # only legal value on the WMMA path
            pad_m=pad,
            pad_n=pad,
            pad_k=pad,
            persistent=False,  # proven no-op on this path
            waves_per_eu=t.get("waves_per_eu"),
            lds_k_pad=t.get("lds_k_pad", 0),
            lds_swizzle=t.get("lds_swizzle", False),
            split_k=1,  # gated: CDNA-only
            wmma_async_lds=bool(t.get("wmma_async_lds", False)),
            tdm_lds=bool(t.get("tdm_lds", False)),
            tdm_scalarize=bool(t.get("tdm_scalarize", True)),
            tdm_prefetch=bool(t.get("tdm_prefetch", False)),
            tdm_prefetch_depth=int(t.get("tdm_prefetch_depth", 2)),
            tdm_split_barrier=bool(t.get("tdm_split_barrier", False)),
        ),
        data=(mono_data_spec(t["dtype"]) if t.get("dtype") else DataSpec()),
        wave_size=32,
    )


def build_one(
    c: Dict[str, Any], out_root: Path, shape: tuple, tag: str
) -> Dict[str, Any]:
    spec = spec_from_candidate(c, f"rocke_gemm_fp16_rcr_{ARCH}_{tag}")
    art = compile_kernel(build_universal_gemm(spec, arch=ARCH), arch=ARCH)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=spec.tile.tile_m,
        block_n=spec.tile.tile_n,
        block_k=spec.tile.tile_k,
        threads_per_block=spec.block_size,
        default_shape=shape,
        atoms=[f"wmma_f32_{ATOM[0]}x{ATOM[1]}x{ATOM[2]}_{spec.data.dtype_a}"],
    )
    dst = out_root / tag
    dst.mkdir(parents=True, exist_ok=True)
    write_artifact(art, dst, manifest)
    return {
        "tag": tag,
        "kernel_name": art.kernel_name,
        "dir": str(dst),
        "geometry": c["geometry"],
        "traits": c["traits"],
        "hsaco_bytes": art.hsaco_bytes,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="build only the first N")
    p.add_argument("--shape", default="4096,4096,4096")
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="build just the baseline config (end-to-end smoke)",
    )
    args = p.parse_args(argv)

    shape = tuple(int(x) for x in args.shape.split(","))
    doc = json.loads(args.candidates.read_text())
    cands = [BASELINE] if args.baseline_only else [BASELINE] + doc["candidates"]
    if args.limit is not None:
        cands = cands[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    built: List[Dict[str, Any]] = []
    for i, c in enumerate(cands):
        tag = "baseline" if c is BASELINE else f"cand{i:04d}"
        try:
            built.append(build_one(c, args.out, shape, tag))
        except Exception as exc:  # noqa: BLE001 - a build failure is data
            built.append({"tag": tag, "error": f"{type(exc).__name__}: {exc}"})

    index = args.out / "index.json"
    index.write_text(
        json.dumps(
            {"arch": ARCH, "shape": list(shape), "count": len(built), "built": built},
            indent=2,
        )
    )
    ok = sum(1 for b in built if "error" not in b)
    print(f"[build] {ok}/{len(built)} artifacts -> {args.out}")
    print(f"[build] index: {index}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
