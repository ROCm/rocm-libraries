#!/usr/bin/env python3
"""
Regenerate Treelite --source-only C for the LGBM rank model.

Run from this directory (or anywhere; paths are resolved relative to the script):

    python3 regen_models.py [--source-dir ~/AutoResearchAllLGBM]

Outputs into ./rank/:
    main.c, tu0.c..tu7.c, quantize.c, header.h, recipe.json

These files are checked into the MIOpen tree. Only re-run when the source
LightGBM .txt model changes. Requires `treelite>=4` and `tl2cgen>=1`.

v18 TunaNet+Align, expanded-data retrain (2026-07-21): HIP-only base
(41 feat) + 20 engineered derived features the C++ caller computes = 61
features total. The 13 tn_* GEMM-geometry features are MIOpen's existing
miopen::ai::common::EngineeredConvFeatures (reused verbatim); the 7 al_*
tile-alignment features are integer formulas on channels/output_channels/
batch. Still no embedded GPU data: base GPU inputs are the six
hipDeviceProp_t fields + gfx_id, and the derived features come from conv
dims + cu_count only. The expanded retrain adds 2 solvers (72 total:
+ConvHipImplicitGemmFwdXdlops/BwdXdlops) on 866k gold rows / 13 specs;
schema and gfx_id vocab unchanged. Source model is
model_rank_expanded_tnalign_t600.txt. See deploy/README_CPP_DERIVED.md
for the derived-feature spec.
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
    # v18 TunaNet+Align expanded retrain: 61 features (41 base + 13 tn_ + 7 al_),
    # 72 solvers. Derived features computed in C++ (no embedded GPU data).
    ("rank", "lgbm_rank", "model_rank_expanded_tnalign_t600.txt"),
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
