# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Codegen test for CGEMM (single-complex GEMM) on gfx1250 (WMMA, wave32).
#
# gfx1250 has no f64 matrix op and is WMMA-only. CGEMM is emulated as 4 real
# f32 matrix multiplies. Unlike MFMA (K=1), WMMA needs K=4 operands, so the
# interleaved complex load [r0,i0,r1,i1] is de-interleaved into planar
# [r0,r1]/[i0,i1] before the WMMA. This test compiles the kernel with
# --build-only (ISA pinned to gfx1250 in the yaml, so no GPU is needed) and
# asserts the emitted assembly contains that de-interleaved 4-WMMA pattern.
################################################################################

import pytest

pytestmark = pytest.mark.unit

from pathlib import Path

from Tensile import Tensile

# Reuse the common CGEMM gfx1250 config (small enough to double as codegen input).
_CONFIG = (Path(__file__).parent / ".." / "common" / "gemm" / "gfx12"
           / "cgemm_gfx1250.yaml").resolve()


def _gfx1250_asm_supported() -> bool:
    import shutil, subprocess, tempfile, os
    asm = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    if not os.path.exists(asm):
        return False
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "t.s")
        Path(src).write_text("v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[8:9], v[0:7]\n")
        r = subprocess.run(
            [asm, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx1250", "-c", src, "-o", os.path.join(d, "t.o")],
            capture_output=True,
        )
        return r.returncode == 0


@pytest.mark.skipif(not _gfx1250_asm_supported(),
                    reason="assembler does not support gfx1250 WMMA")
def test_cgemm_gfx1250_codegen(tensile_args: list[str], tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    # Force gfx1250 last so it wins over any CI-provided --gpu-targets (this is a
    # gfx1250-only codegen check; other targets reject the complex WMMA solutions).
    args = [
        str(_CONFIG), str(output_dir), "--build-only",
        "--global-parameters", "KeepBuildTmp=True",
        *tensile_args,
        "--gpu-targets", "gfx1250",
    ]
    Tensile.Tensile(args)

    asm_files = list(output_dir.rglob("*C_B*.s"))
    assert asm_files, "no CGEMM assembly generated"
    text = asm_files[0].read_text()

    # WMMA f32 matrix op (CGEMM real building block).
    assert "v_wmma_f32_16x16x4_f32" in text
    # De-interleave of the interleaved complex load into planar operands.
    assert "deint A re" in text and "deint A im" in text
    # 4-multiply complex emulation: real accumulator gets Ar*Br and an Ai*Bi
    # term, imaginary accumulator gets Ai*Br and Ar*Bi terms. The signs vary
    # with transpose/conjugate, so match the sign-agnostic operand pattern.
    assert "Cr += Ar*Br" in text
    assert "Ai*Bi" in text and "Ai*Br" in text and "Ar*Bi" in text
