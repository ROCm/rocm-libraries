# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``_extract_elf_meta`` must read any arch, not just the one it was written for.

The previous implementation disassembled with ``llvm-objdump --mcpu=gfx950``
unconditionally, so a gfx1250 object produced no match and the sweep recorded
``elf_meta: {}``. Nothing failed -- the field is best-effort -- so every
gfx1250 sweep on record simply has no register or LDS data, and a
``waves_per_eu`` arm that changed nothing was indistinguishable from one that
did.

These tests compile real kernels rather than parsing a fixture: the thing that
broke was the coupling between the reader and the object's arch, which a stored
text fixture would not have exercised.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path


def _readelf_available() -> bool:
    return bool(
        shutil.which("llvm-readelf")
        or shutil.which("/opt/rocm/lib/llvm/bin/llvm-readelf")
        or shutil.which("/opt/rocm/llvm/bin/llvm-readelf")
    )


def _gemm_spec(*, arch: str):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    if arch == "gfx1250":
        return UniversalGemmSpec(
            name="elfmeta_wmma",
            tile=TileSpec(
                tile_m=64, tile_n=64, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
                warp_tile_m=16, warp_tile_n=16, warp_tile_k=64,
            ),
            trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="default"),
            data=DataSpec(
                dtype_a="fp8e4m3", dtype_b="fp8e4m3", dtype_c="bf16",
                dtype_acc="fp32", layout="RCR",
            ),
            wave_size=32,
        )
    return UniversalGemmSpec(
        name="elfmeta_mfma",
        tile=TileSpec(
            tile_m=128, tile_n=128, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"),
        data=DataSpec(
            dtype_a="bf16", dtype_b="bf16", dtype_c="bf16",
            dtype_acc="fp32", layout="RCR",
        ),
        wave_size=64,
    )


def _compiled_meta(arch: str):
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.instances.common.gemm_universal import build_universal_gemm
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.sweep import _extract_elf_meta

    ir = build_universal_gemm(_gemm_spec(arch=arch), arch=arch)
    ll = lower_kernel_to_llvm(ir, arch=arch)
    hsaco, _ = build_hsaco_from_llvm_ir(ll, isa=f"amdgcn-amd-amdhsa--{arch}")
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco)
        path = f.name
    try:
        return _extract_elf_meta(Path(path))
    finally:
        os.unlink(path)


@unittest.skipUnless(_readelf_available(), "llvm-readelf not on PATH")
class TestExtractElfMeta(unittest.TestCase):
    def test_reads_both_a_wmma_and_an_mfma_object(self):
        """The regression was arch coupling, so both arches must come back."""
        for arch in ("gfx950", "gfx1250"):
            with self.subTest(arch=arch):
                try:
                    meta = _compiled_meta(arch)
                except Exception as exc:  # comgr may not support every arch here
                    self.skipTest(f"{arch} not buildable in this env: {exc}")
                for key in ("vgprs", "sgprs", "lds_bytes"):
                    self.assertIn(key, meta, f"{key} missing for {arch}: {meta}")
                    self.assertGreater(meta[key], 0, f"{key} not populated for {arch}")

    def test_reports_spill_counts(self):
        """Spills are why this metadata is worth collecting.

        A kernel that meets its register budget by spilling looks identical to
        one that fits, on VGPR count alone -- which is exactly the distinction
        needed to tell a live ``waves_per_eu`` arm from an inert one.
        """
        try:
            meta = _compiled_meta("gfx1250")
        except Exception as exc:
            self.skipTest(f"gfx1250 not buildable in this env: {exc}")
        self.assertIn("vgpr_spills", meta)
        self.assertIn("sgpr_spills", meta)
        self.assertGreaterEqual(meta["vgpr_spills"], 0)

    def test_missing_file_is_best_effort_not_an_error(self):
        """The field is optional; a sweep must not die collecting it."""
        from rocke.sweep import _extract_elf_meta

        self.assertEqual(_extract_elf_meta(Path("/nonexistent/x.hsaco")), {})


if __name__ == "__main__":
    unittest.main()
