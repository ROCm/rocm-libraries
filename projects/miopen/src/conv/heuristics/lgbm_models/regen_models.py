#!/usr/bin/env python3
"""
Regenerate Treelite --source-only C for the LGBM rank model.

Run from this directory (or anywhere; paths are resolved relative to the script):

    python3 regen_models.py [--source-dir ~/AutoResearchAllLGBM]

Outputs into ./rank/:
    main.c, tu0.c..tu7.c, quantize.c, header.h, recipe.json

These files are checked into the MIOpen tree. Only re-run when the source
LightGBM .txt model changes. Requires `treelite>=4` and `tl2cgen>=1`.

v17 (2026-06-16): HIP-only, rank-only, retrained on a delta data pull
(more gfx90a / gfx1100 coverage; solver vocab grew to 79). Same 41-feature
HIP-only schema as v16 -- every GPU feature is directly readable from
hipDeviceProp_t, so there is no embedded per-arch table. The only GPU
inputs are cu_count, wave_size, lds_size_per_workgroup_kb, l2_cache_total_kb,
boost_clock_mhz, vram_bytes (all hipDeviceProp_t) and gfx_id (gcnArchName).
This lets the model project to unseen architectures without curated data.
Source model is model_rank_v17_0616.txt.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import treelite
import tl2cgen

HERE = Path(__file__).resolve().parent

MODELS = [
    # (subdir, libname, source .txt model)
    # v17 HIP-only: 41 features. GPU inputs are exclusively hipDeviceProp_t
    # fields + gfx_id; no embedded per-arch table. Retrained on delta data.
    ("rank", "lgbm_rank", "model_rank_v17_0616.txt"),
]


def regen(source_dir: Path, subdir: str, libname: str, model_file: str) -> None:
    src = source_dir / model_file
    if not src.exists():
        sys.exit(f"missing source model: {src}")
    out_dir = HERE / subdir
    print(f"[regen] {model_file} -> {out_dir}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.iterdir():
        existing.unlink()

    model = treelite.frontend.load_lightgbm_model(str(src))
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / f"{libname}.zip"
        tl2cgen.export_srcpkg(
            model,
            toolchain="gcc",
            pkgpath=str(zip_path),
            libname=libname,
            params={"parallel_comp": 8, "quantize": 1},
            verbose=False,
        )
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td)
        # tl2cgen lays out as <td>/<libname>/{main.c, tuN.c, header.h, recipe.json, Makefile}
        srcpkg = Path(td) / libname
        for f in srcpkg.iterdir():
            if f.suffix in {".c", ".h", ".json"} or f.name == "Makefile":
                shutil.copy2(f, out_dir / f.name)
    print(f"  wrote: {sorted(p.name for p in out_dir.iterdir())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default=os.path.expanduser("~/AutoResearchAllLGBM"),
        help="directory containing model_*.txt files",
    )
    args = parser.parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    for subdir, libname, model_file in MODELS:
        regen(source_dir, subdir, libname, model_file)
    print("done.")


if __name__ == "__main__":
    main()
