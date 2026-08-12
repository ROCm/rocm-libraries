# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Codegen test for ZGEMM (complex-double GEMM) on gfx1250 (wave32).
#
# gfx1250 has no f64 matrix instruction, so ZGEMM runs the VALU v_fma_f64 path
# (EnableMatrixInstruction=false, SIA0), like real DGEMM on this arch. One
# complex-double multiply is emulated as 4 v_fma_f64. This test compiles the
# kernel with --build-only (ISA pinned to gfx1250 in the yaml, so no GPU is
# needed) and asserts the emitted assembly contains those v_fma_f64 ops.
################################################################################

import pytest

pytestmark = pytest.mark.unit

from pathlib import Path

from Tensile import Tensile

# Reuse the common ZGEMM gfx1250 config (small enough to double as codegen input).
_CONFIG = (Path(__file__).parent / ".." / "common" / "gemm" / "gfx12"
           / "zgemm_gfx1250.yaml").resolve()


def _gfx1250_asm_supported() -> bool:
    import shutil, subprocess, tempfile, os
    asm = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    if not os.path.exists(asm):
        return False
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "t.s")
        Path(src).write_text("v_fma_f64 v[0:1], v[2:3], v[4:5], v[0:1]\n")
        r = subprocess.run(
            [asm, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx1250", "-c", src, "-o", os.path.join(d, "t.o")],
            capture_output=True,
        )
        return r.returncode == 0


@pytest.mark.skipif(not _gfx1250_asm_supported(),
                    reason="assembler does not support gfx1250")
def test_zgemm_gfx1250_codegen(tensile_args: list[str], tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    # Force gfx1250 last so it wins over any CI-provided --gpu-targets (this is a
    # gfx1250-only codegen check; other targets reject the complex VALU solutions).
    args = [
        str(_CONFIG), str(output_dir), "--build-only",
        "--global-parameters", "KeepBuildTmp=True",
        *tensile_args,
        "--gpu-targets", "gfx1250",
    ]
    Tensile.Tensile(args)

    asm_files = list(output_dir.rglob("*Z_B*.s"))
    assert asm_files, "no ZGEMM assembly generated"
    text = asm_files[0].read_text()

    # VALU f64 FMA (ZGEMM building block) — no matrix instruction on gfx1250.
    assert "v_fma_f64" in text
    assert "v_wmma_f64" not in text and "v_mfma_f64" not in text
    # 4 v_fma_f64 per complex element (real: 2, imag: 2).
    assert text.count("v_fma_f64") >= 4
