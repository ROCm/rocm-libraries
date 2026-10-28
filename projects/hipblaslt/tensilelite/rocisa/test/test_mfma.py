# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Operand format modifiers belong to f8f6f4 opcodes, not legacy typed MFMA."""

import os
from pathlib import Path

import pytest
import rocisa
from rocisa.container import vgpr
from rocisa.enum import InstType
from rocisa.instruction import MFMAInstruction


FP8_TYPES = [
    (InstType.INST_F8, "fp8_fp8", "cbsz:0 blgp:0"),
    (InstType.INST_BF8, "bf8_bf8", "cbsz:1 blgp:1"),
    (InstType.INST_F8_BF8, "fp8_bf8", "cbsz:0 blgp:1"),
    (InstType.INST_BF8_F8, "bf8_fp8", "cbsz:1 blgp:0"),
]


@pytest.fixture
def use_isa():
    isa = rocisa.rocIsa.getInstance()
    previous_kernel = isa.getKernel()
    previous_indices = dict(isa.getVgprIdx())
    previous_msb = isa.getVgprMsb()
    assembler = str(Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/amdclang++")

    def select(target):
        isa.init(target, assembler, False)
        isa.setKernel(target, 64)

    yield select
    isa.setKernel(tuple(previous_kernel.isa), previous_kernel.wavefrontSize)
    for name, index in previous_indices.items():
        isa.setVgprIdx(name, index)
    isa.setVgprMsb(previous_msb)


def emit_mfma(inst_type, variant):
    registers = variant[0] * variant[1] // 64
    inputs = variant[0] * variant[2] // (64 * 4)
    return str(
        MFMAInstruction(
            instType=inst_type,
            accType=InstType.INST_F32,
            variant=variant,
            mfma1k=False,
            acc=vgpr(0, registers),
            a=vgpr(32, inputs),
            b=vgpr(64, inputs),
        )
    )


@pytest.mark.parametrize("target", [(9, 4, 2), (9, 5, 0)], ids=["gfx942", "gfx950"])
@pytest.mark.parametrize("variant", [[16, 16, 32, 1], [32, 32, 16, 1]])
@pytest.mark.parametrize("inst_type,suffix,modifiers", FP8_TYPES)
def test_typed_fp8_mfma_has_default_broadcast(
    use_isa, target, variant, inst_type, suffix, modifiers
):
    use_isa(target)
    text = emit_mfma(inst_type, variant)
    m, n, k, _ = variant
    assert text.startswith(f"v_mfma_f32_{m}x{n}x{k}_{suffix} ")
    assert "cbsz" not in text
    assert "blgp" not in text


@pytest.mark.parametrize("variant", [[16, 16, 128, 1], [32, 32, 64, 1]])
@pytest.mark.parametrize("inst_type,suffix,modifiers", FP8_TYPES)
def test_f8f6f4_mfma_retains_operand_formats(use_isa, variant, inst_type, suffix, modifiers):
    use_isa((9, 5, 0))
    text = emit_mfma(inst_type, variant)
    m, n, k, _ = variant
    assert text.startswith(f"v_mfma_f32_{m}x{n}x{k}_f8f6f4 ")
    assert modifiers in text
