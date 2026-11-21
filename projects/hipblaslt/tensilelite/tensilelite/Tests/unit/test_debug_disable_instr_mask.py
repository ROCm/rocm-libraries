# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DebugDisableInstrMask: assembly filter, parameter registration and kernel naming."""

import copy

import pytest

from tensilelite.Common.DebugInstrMask import (DEBUG_INSTR_MASK_ALL,
                                               applyDebugDisableInstrMask,
                                               debugDisableInstrMaskConflict)
from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Common.ValidParameters import validParameters

pytestmark = pytest.mark.unit

ASM = """\
label_Loop:
  buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:0 // read A
  global_load_b128 v[0:3], v[4:5], off
  buffer_load_dword v4, s[8:11], 0 offen lds
  ds_store_b128 v5, v[0:3] offset:0
  ds_write_b64 v5, v[0:1]
  s_waitcnt lgkmcnt(0) vmcnt(0)
  ds_load_b128 v[8:11], v6 offset:256
  ds_read_b32 v8, v6
  ds_bpermute_b32 v8, v6, v7
  v_wmma_f32_16x16x16_bf16 v[16:23], v[8:15], v[0:7], v[16:23]
  v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]
  v_add_f32 v1, v2, v3 // v_wmma in a comment stays
  buffer_store_dwordx4 v[16:19], v20, s[12:15], 0 offen
  global_store_b32 v[4:5], v1, off
  buffer_atomic_add_f32 v1, v2, s[12:15], 0 offen
  s_cbranch_scc0 label_Loop
"""

CLASSES = {
  1: ["v_wmma_f32", "v_mfma_f32"],
  2: ["buffer_load_dwordx4", "global_load_b128", "buffer_load_dword v4"],
  4: ["buffer_store_dwordx4", "global_store_b32"],
  8: ["ds_load_b128", "ds_read_b32"],
  16: ["ds_store_b128", "ds_write_b64"],
}
UNTOUCHED = ["label_Loop:", "s_waitcnt", "ds_bpermute_b32", "v_add_f32", "buffer_atomic_add_f32", "s_cbranch_scc0"]


def _disabled(out, mnemonic):
  line = next(l for l in out.splitlines() if mnemonic in l)
  return line.lstrip().startswith("// DebugDisableInstrMask: ")


def test_mask_zero_is_identity():
  assert applyDebugDisableInstrMask(ASM, 0) is ASM


@pytest.mark.parametrize("mask", list(range(1, DEBUG_INSTR_MASK_ALL + 1)))
def test_each_combination_disables_exactly_its_classes(mask):
  out = applyDebugDisableInstrMask(ASM, mask)
  assert len(out.splitlines()) == len(ASM.splitlines())
  for bit, mnemonics in CLASSES.items():
    for m in mnemonics:
      assert _disabled(out, m) == bool(mask & bit), (mask, m)
  for m in UNTOUCHED:
    assert not _disabled(out, m), (mask, m)


def test_unknown_bits_rejected():
  with pytest.raises(ValueError):
    applyDebugDisableInstrMask(ASM, DEBUG_INSTR_MASK_ALL + 1)


def test_parameter_registered_with_off_default():
  assert defaultSolution["DebugDisableInstrMask"] == 0
  assert validParameters["DebugDisableInstrMask"] == list(range(DEBUG_INSTR_MASK_ALL + 1))


@pytest.mark.parametrize("streamK", [3, 4, 5])
@pytest.mark.parametrize("mask", [2, 4, 6, 31])
def test_streamk_fixup_rejects_global_mem_masks(streamK, mask):
  # Fixup spins on a global-memory flag; removing its load/store hangs.
  state = {"StreamK": streamK, "StreamKAtomic": 0, "DebugDisableInstrMask": mask}
  assert "Stream-K" in debugDisableInstrMaskConflict(state)


@pytest.mark.parametrize("state", [
  {"StreamK": 3, "StreamKAtomic": 0, "DebugDisableInstrMask": 1 | 8 | 16},  # no global mem bits
  {"StreamK": 3, "StreamKAtomic": 1, "DebugDisableInstrMask": 31},          # atomic SK: no fixup
  {"StreamK": 0, "DebugDisableInstrMask": 31},                              # not Stream-K
  {"StreamK": 3, "StreamKAtomic": 0, "DebugDisableInstrMask": 0},
])
def test_no_conflict(state):
  assert debugDisableInstrMaskConflict(state) is None


# --- End-to-end: Solution rejects the hanging combination (real gfx950 caps). ---

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


@pytest.fixture(scope="module")
def _gp_gfx950(gfx950_iim):
  from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters
  saved = [(d, copy.deepcopy(dict(d))) for d in (globalParameters, validParameters, defaultSolution)]
  defaultSolution.clear()
  defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
  assignGlobalParameters({}, gfx950_iim)
  yield
  for d, s in saved:
    d.clear()
    d.update(s)


def _solution(iim, assembler, mask, **overrides):
  from tensilelite.Common.Architectures import gfxToIsa
  from tensilelite.SolutionStructs.Solution import Solution
  from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

  isa = gfxToIsa("gfx950")
  mi = [16, 16, 32, 1, 1, 4, 4, 2, 2]  # MT 128x128
  problemType = {"OperationType": "GEMM", "DataType": "b", "DestDataType": "b",
                 "ComputeDataType": "s", "HighPrecisionAccumulate": True,
                 "TransposeA": True, "TransposeB": False, "UseBeta": True, "Batched": True}
  params = {"ProblemType": problemType, "InternalSupportParams": {"SupportUserGSU": True},
            "MatrixInstruction": mi, "WorkGroup": [16, 16, 1], "WavefrontSize": 64,
            "DepthU": 32, "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 0, "ScheduleIterAlg": 3, "StaggerU": 0,
            "TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid",
            "GlobalSplitU": 0, "WorkGroupMapping": 1, "MIArchVgpr": False,
            "UseSubtileImpl": False, "DebugDisableInstrMask": mask}
  params.update(overrides)
  params.update(matrixInstructionToMIParameters(mi, isa, 64, problemType, params["WorkGroup"], iim))
  return Solution(params, False, True, False, assembler, iim)


@pytest.mark.parametrize("mask", [0, 1, 8 | 16])
def test_solution_streamk_accepts_masks_without_global_mem(_gp_gfx950, gfx950_iim, assembler, capsys, mask):
  assert _solution(gfx950_iim, assembler, mask)["Valid"], capsys.readouterr().out


@pytest.mark.parametrize("mask", [2, 4])
def test_solution_streamk_rejects_global_mem_masks(_gp_gfx950, gfx950_iim, assembler, capsys, mask):
  assert not _solution(gfx950_iim, assembler, mask)["Valid"]
  assert "Stream-K fixup flag handshake" in capsys.readouterr().out


def test_solution_data_parallel_accepts_global_mem_masks(_gp_gfx950, gfx950_iim, assembler, capsys):
  sol = _solution(gfx950_iim, assembler, 6, TileProcessingStrategy="None", GlobalSplitU=1)
  assert sol["Valid"], capsys.readouterr().out
