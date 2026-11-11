# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from rocisa import rocIsa
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("output_bytes, accumulation_bytes", [(2, 4), (4, 4), (4, 2)])
@pytest.mark.parametrize(
    "global_split_u, global_accumulation",
    [
        pytest.param(0, None, id="persistent-data-parallel"),
        pytest.param(1, None, id="no-workspace"),
        pytest.param(-1, "MultipleBuffer", id="automatic-gsu"),
        pytest.param(1, "MultipleBuffer", id="runtime-gsu"),
        pytest.param(4, "MultipleBuffer", id="fixed-gsu"),
        pytest.param(0, "PartialsBuffer", id="streamk"),
    ],
)
def test_remap_readback_buffers_do_not_overlap(
    output_bytes, accumulation_bytes, global_split_u, global_accumulation
):
    rocIsa.getInstance().setKernel((12, 5, 0), 32)
    kernel = dict(
        WavefrontSize=32, MIWaveGroup=[2, 2], MatrixInstM=16, MatrixInstN=16,
        MatrixInstBM=1, MatrixInstBN=1, MIOutputVectorWidth=4,
        MacroTile0=32, StoreRemapVectorWidth=4,
        GlobalSplitU=global_split_u, _GlobalAccumulation=global_accumulation,
    )
    writer = KernelWriterAssembly.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(
        kernel=kernel, bpeCexternal=output_bytes, bpeCexternalGSU1=output_bytes,
        bpeCinternal=accumulation_bytes, bpr=4, regCaps={"MaxSgpr": 102},
    )
    writer.vgprs = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.sgprBpeList = []
    writer.do = {"Sync": False}

    writer.storeRemapComputeStoreVgprs(kernel)
    buffers = writer.vgprs.storeRemapAS
    assert len(buffers) == 2
    next_temp = writer.vgprPool.checkOut(1, "next temporary")

    # Every readback must fit without overwriting another buffer or a later allocation.
    element_widths = (output_bytes, accumulation_bytes) if global_accumulation else (output_bytes,)
    for element_bytes in element_widths:
        registers_per_read = kernel["StoreRemapVectorWidth"] * element_bytes // 4
        occupied = [reg for base in buffers for reg in range(base, base + registers_per_read)]
        assert len(set(occupied)) == len(occupied)
        assert next_temp not in occupied

    if global_accumulation is None:
        # Final-output-only kernels must not reserve registers for wider partials.
        available_before = writer.vgprPool.available()
        for base in buffers:
            writer.vgprPool.checkIn(base)
        released = writer.vgprPool.available() - available_before
        assert released == len(buffers) * kernel["StoreRemapVectorWidth"] * output_bytes // 4


def _remapParams(gfx1250_iim):
  """A bf16 TN gfx1250 kernel, MT128x128 over 4 waves, with StoreRemapVectorWidth 4 and no TDMStoreInst."""
  from tensilelite.Common.Architectures import gfxToIsa
  from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

  isa, mi = gfxToIsa("gfx1250"), [16, 16, 32, 1, 1, 4, 4, 2, 2]
  problemType = {
    "OperationType": "GEMM", "DataType": "B", "MacDataTypeA": "B", "MacDataTypeB": "B",
    "DestDataType": "B", "ComputeDataType": "s", "HighPrecisionAccumulate": True,
    "TransposeA": True, "TransposeB": False, "UseBeta": True, "Batched": True, "MXBlockA": 0, "MXBlockB": 0,
  }
  params = {
    "ProblemType": problemType, "ISA": isa, "MatrixInstruction": mi, "WorkGroup": [16, 16, 1], "WavefrontSize": 32,
    "DepthU": 64, "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "ScheduleIterAlg": 4,
    "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1, "TransposeLDS": -1, "LdsPadA": -1, "LdsPadB": -1,
    "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1, "1LDSBuffer": 0, "VectorWidthA": 1, "VectorWidthB": 1,
    "StoreVectorWidth": -1, "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1, "LocalReadVectorWidth": -1,
    "SourceSwap": False, "ExpandPointerSwap": False, "GlobalSplitUAlgorithm": "MultipleBuffer", "TDMInst": 3,
    "LDSTrInst": False, "StreamK": 0, "StreamKForceDPOnly": 0, "PrefetchAcrossPersistent": 0, "UseSubtileImpl": False,
    "StoreRemapVectorWidth": 4, "DirectToVgprA": False, "DirectToVgprB": False, "DirectToVgprSparseMetadata": False,
    "WorkGroupMapping": 1, "TDMFuse": 0, "TDMSplit": False, "InitCIterWmma": 0,
  }
  params.update(matrixInstructionToMIParameters(mi, isa, params["WavefrontSize"], problemType, params["WorkGroup"],
                                                gfx1250_iim))
  return params


def test_store_remap_blocks_unchanged_without_tdm(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """Without TDMStoreInst every store block of a StoreRemapVectorWidth kernel remaps D through LDS: the remap is
  skipped only by TDM blocks."""
  import re
  from emit_test_helpers import emit_asm

  sol, asm = emit_asm(gfx1250_iim, assembler, _remapParams(gfx1250_iim))
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreRemapVectorWidth"] == 4 and "tensor_store_from_lds" not in asm
  blocks = re.split(r"^(label_GW_B\d_FD0_VW\d+_GSU1_(?:Then|NonEdge)):", asm, flags=re.M)
  names = blocks[1::2]
  assert names
  for name, text in zip(names, blocks[2::2]):
    body = text.split("label_GW_", 1)[0]
    assert "storeRemap lw" in body, name
