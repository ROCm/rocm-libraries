# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
from rocisa.instruction import DSLoadB32, DSLoadU16, DSLoadU8

from Tensile.AsmMemoryInstruction import MemoryInstruction
from Tensile.Components.LocalRead import LocalReadMFMA
from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


def _writer(*, has_ecc_half=False, has_wmma_v1=True, lrvw_tile=2):
    writer = object.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(
        archCaps={"HasEccHalf": has_ecc_half},
        asmCaps={"HasLDSTrB128B16": False, "HasWMMA_V1": has_wmma_v1},
        bpr=4,
        convDTVA=False,
        convDTVB=False,
        lrvwTileA=lrvw_tile,
        lrvwTileB=lrvw_tile,
        packDTVA=False,
        packDTVB=False,
    )
    return writer


def _kernel(**overrides):
    kernel = {
        "ConvertAfterDS": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "EnableMatrixInstruction": True,
        "MIInputPerThreadA": 1,
        "MIInputPerThreadB": 1,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "ThreadTile0": 1,
        "ThreadTile1": 1,
        "UnrollMajorLDSA": False,
        "UnrollMajorLDSB": False,
        "UseF32XEmulation": False,
        "enableLDSTrA": False,
        "enableLDSTrB": False,
    }
    kernel.update(overrides)
    return kernel


def _tensor(tensor_char, instruction, **overrides):
    tensor = {
        "bpe": 2,
        "bpeDS": 2,
        "isM": False,
        "localReadInstruction": instruction,
        "tensorChar": tensor_char,
        "tile01Idx": 0 if tensor_char == "A" else 1,
        "tt": "ThreadTile0" if tensor_char == "A" else "ThreadTile1",
    }
    tensor.update(overrides)
    return tensor


@pytest.mark.parametrize("tensor_char", ["A", "B"])
def test_wmma_v1_d16_local_read_does_not_allocate_valu_pack(tensor_char):
    writer = _writer()
    d16 = MemoryInstruction(DSLoadU16, 1, 1, 1, 0.5)
    tensor = _tensor(tensor_char, d16)

    assert not writer.localReadNeedsPack(_kernel(), tensor)
    assert not writer.localReadNeedsValuPack(_kernel(), tensor)


@pytest.mark.parametrize(
    "writer_kwargs, kernel_overrides",
    [
        ({"has_ecc_half": True}, {}),
        ({"has_wmma_v1": False}, {}),
        ({}, {"EnableMatrixInstruction": False}),
    ],
)
def test_d16_local_read_uses_fallback_pack_rules(writer_kwargs, kernel_overrides):
    writer = _writer(**writer_kwargs)
    tensor = _tensor("B", MemoryInstruction(DSLoadU16, 1, 1, 1, 0.5))
    kernel = _kernel(**kernel_overrides)

    assert writer.localReadNeedsPack(kernel, tensor)
    assert writer.localReadNeedsValuPack(kernel, tensor)


def test_pack_allocation_helpers_are_ab_only():
    writer = _writer()
    tensor = _tensor("Metadata", MemoryInstruction(DSLoadU8, 1, 1, 1, 0.25))

    assert not writer.localReadNeedsPack(_kernel(), tensor)
    assert not writer.localReadNeedsValuPack(_kernel(), tensor)


def test_sub_dword_and_conversion_local_reads_allocate_valu_pack():
    writer = _writer()
    d8 = MemoryInstruction(DSLoadU8, 1, 1, 1, 0.25)
    d16 = MemoryInstruction(DSLoadU16, 1, 1, 1, 0.5)

    assert writer.localReadNeedsValuPack(_kernel(), _tensor("B", d8))
    assert writer.localReadNeedsValuPack(
        _kernel(ConvertAfterDS=True),
        _tensor("B", d16, bpe=1),
    )


def test_direct_to_vgpr_allocates_valu_pack_only_for_pack_or_conversion():
    writer = _writer()
    d16 = MemoryInstruction(DSLoadU16, 1, 1, 1, 0.5)
    kernel = _kernel(DirectToVgprB=True)
    tensor = _tensor("B", d16)

    assert not writer.localReadNeedsValuPack(kernel, tensor)
    writer.states.packDTVB = True
    assert writer.localReadNeedsValuPack(kernel, tensor)


@pytest.mark.parametrize(
    "instruction, bpe, lrvw_tile, expected",
    [
        (MemoryInstruction(DSLoadU16, 1, 1, 1, 0.5), 2, 2, 1),
        (MemoryInstruction(DSLoadB32, 1, 1, 1, 1), 2, 4, 1),
        (MemoryInstruction(DSLoadB32, 1, 1, 1, 1), 8, 2, 0),
        (MemoryInstruction(DSLoadU8, 1, 1, 1, 0.25), 0.5, 1, 2),
    ],
)
def test_local_read_element_stride_preserves_whole_and_partial_reads(
    instruction, bpe, lrvw_tile, expected
):
    kernel = _kernel()
    tensor = _tensor("B", instruction, bpe=bpe, bpeDS=bpe)

    assert (
        LocalReadMFMA.numElementsPerRead(
            kernel, tensor, instruction, 4, lrvw_tile
        )
        == expected
    )


def test_convert_after_ds_forces_one_element_only_when_width_changes():
    kernel = _kernel(ConvertAfterDS=True)
    d32 = MemoryInstruction(DSLoadB32, 1, 1, 1, 1)

    assert LocalReadMFMA.numElementsPerRead(
        kernel, _tensor("B", d32, bpe=1), d32, 4, 1
    ) == 1
    assert LocalReadMFMA.numElementsPerRead(
        kernel, _tensor("B", d32), d32, 4, 1
    ) == 2


def test_d16_vector_reads_retain_physical_instruction_count():
    writer = _writer()
    d16 = MemoryInstruction(DSLoadU16, 1, 1, 1, 0.5)
    tensor = _tensor("B", d16)
    kernel = _kernel(VectorWidthB=2)

    assert writer.adjustLocalReadInstructionCount(kernel, tensor, 8) == 8


def test_partial_element_reads_retain_physical_instruction_count():
    writer = _writer()
    d32 = MemoryInstruction(DSLoadB32, 1, 1, 1, 1)
    tensor = _tensor("B", d32, bpe=8, bpeDS=8)
    kernel = _kernel(VectorWidthB=2)

    assert writer.adjustLocalReadInstructionCount(kernel, tensor, 8) == 8


def test_regular_vector_reads_share_one_instruction():
    writer = _writer(has_wmma_v1=False)
    d32 = MemoryInstruction(DSLoadB32, 1, 1, 1, 1)
    tensor = _tensor("B", d32)
    kernel = _kernel(VectorWidthB=2)

    assert writer.adjustLocalReadInstructionCount(kernel, tensor, 8) == 4
