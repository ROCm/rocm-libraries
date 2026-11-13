# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for TDMStoreInst: the store paths of each level, the NonEdge paths of level 3 included, solution
validation, the kernel-name token, the LDS layout, the store modifiers and the emitted TDM block: its C load and the cache
hints of C, its fp16 and narrow store vectors, and the waits around it after StinkyTofu."""

import re
from collections import Counter, defaultdict
from copy import deepcopy

import pytest
from rocisa.enum import CacheScope, NonVolatile, TemporalHint

from tensilelite.Common.DataType import DataType
from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.Components.TDMStore import STINKY_KEEP, TDM_STORE_PATHS, storesFinalD, tdmStoreBlocks, \
  tdmStoreLdsBytes, tdmStoreModifiers, tdmStoreRejectReason
from tensilelite.SolutionStructs.Naming import getKernelNameMin

from emit_test_helpers import assert_every_vector_preload_barriers, emit_asm
from toolchain_test_helpers import copy_isa_info_map

pytestmark = pytest.mark.unit

_LEVELS = [1, 2, 3]

# The ProblemType keys an override of _state may name; every other override goes to the solution.
_PROBLEM_TYPE_KEYS = {
  "DestDataType", "ComputeDataType", "HighPrecisionAccumulate", "FusedGemmA2A", "UseE", "Gradient",
  "UseGateResidual", "GroupedGemm", "UseBias", "UseScaleAlphaVec", "UseScaleAB", "UseScaleCD",
  "ActivationType", "OutputAmaxD",
}


def _state(**overrides):
  """The keys tdmStoreRejectReason and the ExecutionPolicy helpers it calls read, for a TDMStoreInst=2
  solution inside the supported configurations, with `overrides` applied."""
  problemType = {
    "DestDataType": DataType("b"),
    "ComputeDataType": DataType("s"),
    "HighPrecisionAccumulate": True,
    "FusedGemmA2A": False,
    "UseE": False,
    "Gradient": False,
    "UseGateResidual": False,
    "UseScaleCD": False,
    "OutputAmaxD": False,
    "GroupedGemm": False,
    "UseBias": 0,
    "UseScaleAlphaVec": 0,
  }
  state = {
    "TDMStoreInst": 2,
    "ISA": (12, 5, 0),
    "TileProcessingStrategy": "None",
    "WorkAssignment": "StaticGrid",
    "PrefetchAcrossPersistent": 0,
    "ReuseAcrossPersistent": 0,
    "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "AdaptiveGemmGSUA": 0,
    "UseSubtileImpl": False,
    "StoreRemapVectorWidth": 0,
    "CompactLoopStore": False,
    "MacroTile1": 256,
    "NumThreads": 128,
    "WavefrontSize": 32,
    "ProblemType": problemType,
  }
  for key, value in overrides.items():
    (problemType if key in _PROBLEM_TYPE_KEYS else state)[key] = value
  return state


@pytest.mark.parametrize("dest, hpa", [("b", True), ("h", True), ("f8", True), ("b8", True), ("s", True), ("s", False)])
def test_accepts_output_types(dest, hpa):
  """fp32 D, with or without HighPrecisionAccumulate, and bf16, fp16, fp8 and bf8 D converted from fp32 accumulators."""
  assert tdmStoreRejectReason(_state(DestDataType=DataType(dest), HighPrecisionAccumulate=hpa)) is None


@pytest.mark.parametrize("dest", ["d", "c"])
def test_accepts_eight_byte_d_from_accumulators_of_its_type(dest):
  """fp64 D from fp64 and complex float D from complex float accumulators, which the block stages as 8-byte elements
  from their register pairs."""
  assert tdmStoreRejectReason(_state(DestDataType=DataType(dest), ComputeDataType=DataType(dest),
                                    HighPrecisionAccumulate=False)) is None


def test_accepts_int8_d_from_int32_accumulators():
  """int8 D from int32 accumulators, which the block stages as 1-byte elements."""
  assert tdmStoreRejectReason(_state(DestDataType=DataType("i8"), ComputeDataType=DataType("i"))) is None


def test_accepts_int32_d_from_int32_accumulators():
  """int32 D from int32 accumulators, which the block stages as 4-byte elements like fp32 D."""
  assert tdmStoreRejectReason(_state(DestDataType=DataType("i"), ComputeDataType=DataType("i"))) is None


def test_accepts_streamk3():
  assert tdmStoreRejectReason(_state(TileProcessingStrategy="StreamK", GlobalSplitU=0)) is None


_GSU_ALGORITHMS = [("MultipleBuffer", 0), ("SingleBuffer", 0), ("MultipleBufferSingleKernel", 0), ("AtomicDest", 0),
                   ("MultipleBufferSingleKernel", 1)]


@pytest.mark.parametrize("aga", [0, 1])
def test_accepts_static_gsu_with_mbsk(aga):
  """GlobalSplitU>1 with MultipleBufferSingleKernel: the last workgroup of a tile stores the reduced tile to D."""
  assert tdmStoreRejectReason(_state(GlobalSplitU=2, GlobalSplitUAlgorithm="MultipleBufferSingleKernel",
                                     AdaptiveGemmGSUA=aga)) is None


@pytest.mark.parametrize("algorithm, aga", _GSU_ALGORITHMS)
def test_accepts_dynamic_gsu(algorithm, aga):
  """GlobalSplitU=-1 with every algorithm and AdaptiveGemmGSUA: its GSU == 1 branch stores D directly."""
  assert tdmStoreRejectReason(_state(GlobalSplitU=-1, GlobalSplitUAlgorithm=algorithm, AdaptiveGemmGSUA=aga,
                                     UseBias=1, UseScaleAlphaVec=1)) is None


@pytest.mark.parametrize("overrides", [
  {"AdaptiveGemm": 1},
  {"StoreRemapVectorWidth": -1},
  {"StoreRemapVectorWidth": 4},
  {"StoreRemapVectorWidth": 8},
  {"UseBias": 1},
  {"UseScaleAlphaVec": 1},
  {"UseScaleAlphaVec": 5},
  {"UseBias": 2},
  {"UseBias": 3},
  {"UseScaleAlphaVec": 2},
  {"UseScaleAlphaVec": 3},
  {"UseScaleAlphaVec": 6},
  {"UseScaleAlphaVec": 7},
  {"UseScaleAB": "Scalar"},
  {"UseScaleAB": "Vector"},
  {"UseScaleCD": True},
  {"GroupedGemm": True},
  {"ActivationType": "relu"},
  {"ActivationType": "hipblaslt_all"},
  {"UseBias": 1, "UseScaleAlphaVec": 1, "ActivationType": "hipblaslt_all",
   "TileProcessingStrategy": "StreamK", "GlobalSplitU": 0},
])
def test_accepts_supported_options(overrides):
  """AdaptiveGemm, StoreRemapVectorWidth (-1, which Solution resolves, 4 and 8), the
  epilogue vectors along M, N or either, also with the device-scalar bit of UseScaleAlphaVec, ScaleAB,
  ScaleCD, GroupedGemm and the activations."""
  assert tdmStoreRejectReason(_state(**overrides)) is None


def test_disabled_is_not_validated():
  assert tdmStoreRejectReason(_state(TDMStoreInst=0, ISA=(9, 5, 0), GlobalSplitU=4)) is None


_OUTPUT = "TDMStoreInst supports fp32 D, fp64, complex float and complex double D from accumulators of their type, " \
  "int32 and int8 D from int32 accumulators, and bf16, fp16, fp8 and bf8 D converted from fp32 accumulators"
_STREAMK = {"TileProcessingStrategy": "StreamK", "GlobalSplitU": 0}


@pytest.mark.parametrize("overrides, reason", [
  ({"DestDataType": DataType("d")}, _OUTPUT),
  ({"HighPrecisionAccumulate": False}, _OUTPUT),
  ({"ComputeDataType": DataType("h")}, _OUTPUT),
  ({"DestDataType": DataType("i")}, _OUTPUT),
  ({"DestDataType": DataType("i8")}, _OUTPUT),
  ({"DestDataType": DataType("d")}, _OUTPUT),
  ({"DestDataType": DataType("c")}, _OUTPUT),
  ({"DestDataType": DataType("z")}, _OUTPUT),
  ({"GlobalSplitU": 2}, "TDMStoreInst requires GlobalSplitU=1 or -1, or MultipleBufferSingleKernel"),
  ({"UseSubtileImpl": True}, "TDMStoreInst does not support UseSubtileImpl"),
  ({"FusedGemmA2A": True}, "TDMStoreInst does not support FusedGemmA2A"),
])
def test_rejects_unsupported(overrides, reason):
  assert tdmStoreRejectReason(_state(**overrides)) == reason


@pytest.mark.parametrize("overrides", [
  {"TileProcessingStrategy": "DataParallel", "GlobalSplitU": 0},
  dict(_STREAMK, WorkAssignment="DynamicWorkQueue"),
  dict(_STREAMK, WorkAssignment="Hybrid"),
  dict(_STREAMK, PrefetchAcrossPersistent=1),
  {"TileProcessingStrategy": "DataParallel", "GlobalSplitU": 0, "PrefetchAcrossPersistent": 1},
  {"TileProcessingStrategy": "DataParallel", "GlobalSplitU": 0, "ReuseAcrossPersistent": 1},
  {"TileProcessingStrategy": "DataParallel", "GlobalSplitU": 0, "ReuseAcrossPersistent": 1,
   "PrefetchAcrossPersistent": 1},
], ids=["dp", "sk4", "sk5", "sk3_pap", "dp_pap", "dp_rap", "dp_rap_pap"])
def test_accepts_persistent_policies(overrides):
  """Every persistent policy stores the tiles it completes directly to D: DataParallel, and
  StreamK with the dynamic work queue (StreamK 4) or the hybrid assignment (StreamK 5) as with the static grid,
  also with PrefetchAcrossPersistent and ReuseAcrossPersistent."""
  assert tdmStoreRejectReason(_state(**overrides)) is None


def test_levels_match_the_valid_values_and_add_one_path_each():
  assert list(TDM_STORE_PATHS) == validParameters["TDMStoreInst"]
  for level in _LEVELS:
    assert TDM_STORE_PATHS[level][:-1] == TDM_STORE_PATHS[level - 1]


def _blocks(level, edge=True, atomic=False, accumulation=None, vectorWidths=(8, 1), bpe=2, mpm=0):
  return tdmStoreBlocks({"TDMStoreInst": level, "_GlobalAccumulation": accumulation, "MbskPrefetchMethod": mpm},
                        edge, atomic, bpe, list(vectorWidths))


# The store blocks generateBetaModules gets with StoreVectorWidth 8: AdaptiveGemm=0 keeps the
# widest and the narrowest, AdaptiveGemm=1 every width.
_VECTOR_WIDTHS = {0: (8, 1), 1: (8, 4, 2, 1)}


@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("svw", [1, 2, 4, 8])
def test_each_level_picks_blocks_of_the_store_width_cascade(svw, adaptive):
  """A level picks the blocks of the store-width cascade that store through the TDM block: level 1 the
  masked blocks narrower than the store vector width and the narrowest one, level 2 every masked block,
  and of a store without edge blocks only level 3 the unmasked one. Level 2 adds the masked block of the
  store vector width, which at store vector width 1 is the narrowest one, already picked by level 1."""
  cascade = [svw, 1] if adaptive == 0 else [w for w in (8, 4, 2, 1) if w <= svw]
  edgeM = {w for w in cascade if w < svw} | {min(cascade)}
  expected = {0: set(), 1: edgeM, 2: edgeM | {svw}, 3: edgeM | {svw}}
  for level in [0] + _LEVELS:
    assert _blocks(level, vectorWidths=cascade) == expected[level], (level, cascade)
    assert _blocks(level, edge=False, vectorWidths=cascade[:1]) == ({svw} if level == 3 else set()), level
  assert (_blocks(1, vectorWidths=cascade) == _blocks(2, vectorWidths=cascade)) == (svw == 1)


@pytest.mark.parametrize("accumulation, atomic", [
  ("MultipleBuffer", False),
  ("SingleBuffer", True),
  (None, True),
])
def test_partial_and_atomic_stores_keep_buffer_store(accumulation, atomic):
  assert _blocks(3, atomic=atomic, accumulation=accumulation) == set()
  assert _blocks(3, edge=False, vectorWidths=(8,), atomic=atomic, accumulation=accumulation) == set()


@pytest.mark.parametrize("accumulation", [None, "PartialsBuffer", "MultipleBuffer", "MultipleBufferSingleKernel",
                                          "SingleBuffer"])
@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("mpm", [0, 1])
def test_stores_final_d_predicate(accumulation, atomic, mpm):
  """A block writes D's final elements unless it is atomic or a GSU workspace store, the reduced tile of
  MultipleBufferSingleKernel with MbskPrefetchMethod 0 only, and tdmStoreBlocks lets such a block, of 16-bit or
  fp32 elements, store through the TDM block."""
  final = not atomic and (accumulation in (None, "PartialsBuffer")
                          or (accumulation == "MultipleBufferSingleKernel" and mpm == 0))
  assert storesFinalD({"_GlobalAccumulation": accumulation, "MbskPrefetchMethod": mpm}, atomic) == final
  for bpe in (2, 4):
    assert bool(_blocks(3, atomic=atomic, accumulation=accumulation, bpe=bpe, mpm=mpm)) == final, bpe


@pytest.mark.parametrize("level", _LEVELS)
def test_fp32_final_store_uses_the_tdm_block(level):
  """A final store of fp32 D picks the blocks a 16-bit one picks."""
  assert _blocks(level, bpe=4) == _blocks(level)
  assert _blocks(level, edge=False, vectorWidths=(8,), bpe=4) == _blocks(level, edge=False, vectorWidths=(8,))


@pytest.mark.parametrize("accumulation, atomic", [("MultipleBuffer", False), (None, True)])
def test_gsu_partials_keep_buffer_store_at_fp32(accumulation, atomic):
  """GSU>1 partials, at the compute type, and atomic stores keep buffer_store."""
  assert _blocks(3, accumulation=accumulation, atomic=atomic, bpe=4) == set()


def test_streamk_final_store_uses_the_tdm_block():
  assert _blocks(2, accumulation="PartialsBuffer") == {8, 1}


def test_mbsk_reduced_tile_uses_the_tdm_block():
  """MultipleBufferSingleKernel stores the reduced tile to D: the blocks of a final store with MbskPrefetchMethod
  0, where one workgroup reduces the tile at a GSU of at most 2, and none with MbskPrefetchMethod 1."""
  assert _blocks(2, accumulation="MultipleBufferSingleKernel") == {8, 1}
  assert _blocks(2, accumulation="MultipleBufferSingleKernel", mpm=1) == set()


@pytest.mark.parametrize("dest, bpe", [("h", 2), ("s", 4)])
@pytest.mark.parametrize("level", [0] + _LEVELS)
def test_lds_bytes_of_the_staged_tile(level, dest, bpe):
  state = {"TDMStoreInst": level, "MacroTile0": 256, "MacroTile1": 128, "NumThreads": 128, "WavefrontSize": 32,
           "ProblemType": {"DestDataType": DataType(dest)}}
  assert tdmStoreLdsBytes(state) == (256 * 128 * bpe if level else 0)


@pytest.mark.parametrize("level", _LEVELS)
def test_lds_bytes_cover_the_columns_of_every_wave(level):
  """With MacroTile1 not a multiple of the waves, the staged tile holds MacroTile1 rounded up to a multiple of the
  waves: 128 columns over three waves are 3 x 43."""
  state = {"TDMStoreInst": level, "MacroTile0": 256, "MacroTile1": 128, "NumThreads": 96, "WavefrontSize": 32,
           "ProblemType": {"DestDataType": DataType("h")}}
  assert tdmStoreLdsBytes(state) == 256 * 129 * 2


def test_store_modifiers_follow_the_hints_of_d():
  """tensor_store_from_lds takes NonTemporalD, TemporalHintD and NonVolatileD as addStore decodes them.

  TH_WB and TH_LU share one value, so th is compared by value: whether the bound enum makes them
  one Python object depends on the rocisa backend.
  """
  caps = {"HasTHModifier": True}
  kernel = {"NonTemporalD": 2, "TemporalHintD": TemporalHint.TH_WB.value, "NonVolatileD": NonVolatile.NV.value}
  mods = tdmStoreModifiers(caps, kernel)
  assert (mods.scope, mods.th.value, mods.nv, mods.isStore) == \
    (CacheScope.SCOPE_DEV, TemporalHint.TH_WB.value, NonVolatile.NV, True)
  assert tdmStoreModifiers(caps, {"NonTemporalD": 0, "TemporalHintD": 0, "NonVolatileD": 0}) is None
  assert tdmStoreModifiers({"HasTHModifier": False}, kernel) is None


def _kernel(tdmStore):
  """The smallest kernel dict that exercises Naming._getName."""
  kernel = deepcopy(defaultSolution)
  kernel["ProblemType"] = {
    "OperationIdentifier": "Cijk_Alik_Bljk",
    "DataType": 0,
    "DestDataType": 0,
    "ComputeDataType": 0,
    "GroupedGemm": False,
    "UseBeta": False,
    "UseBias": 0,
  }
  kernel["MacroTile0"] = 256
  kernel["MacroTile1"] = 256
  kernel["DepthU"] = 64
  kernel["MatrixInstM"] = 16
  kernel["MatrixInstN"] = 16
  kernel["MatrixInstB"] = 1
  kernel["MatrixInstruction"] = [16, 16, 32, 1]
  kernel["MIWaveTile"] = [8, 8]
  kernel["WorkGroup"] = [32, 4, 1]
  kernel["ISA"] = (12, 5, 0)
  kernel["TDMStoreInst"] = tdmStore
  return kernel


@pytest.mark.parametrize("level", _LEVELS)
def test_kernel_name_token_only_when_enabled(level):
  """TDMStoreInst=0 keeps every existing kernel name; levels 1-3 add the TDMSI1, TDMSI2 or TDMSI3 token, which
  stays apart from the TDMS0 or TDMS1 of TDMSplit."""
  off = getKernelNameMin(_kernel(0), False)
  on = getKernelNameMin(_kernel(level), False)
  assert "TDMSI" not in off
  assert on.replace("_TDMSI%u" % level, "", 1) == off
  split, both = _kernel(0), _kernel(level)
  split["TDMSplit"] = both["TDMSplit"] = True
  split, both = getKernelNameMin(split, False), getKernelNameMin(both, False)
  assert "_TDMS1_" in split and "TDMSI" not in split
  assert "_TDMS1_" in both and "_TDMSI%u_" % level in both


def _params(gfx1250_iim, mi=None, **overrides):
  """A bf16 TN gfx1250 kernel with beta: by default MT256x256 over 4 waves, StoreVectorWidth 8."""
  from tensilelite.Common.Architectures import gfxToIsa
  from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

  isa = gfxToIsa("gfx1250")
  mi = mi or [16, 16, 32, 1, 1, 8, 8, 2, 2]
  problemType = {
    "OperationType": "GEMM", "DataType": "B", "MacDataTypeA": "B", "MacDataTypeB": "B",
    "DestDataType": "B", "ComputeDataType": "s", "HighPrecisionAccumulate": True,
    "TransposeA": True, "TransposeB": False, "UseBeta": True, "Batched": True,
    "MXBlockA": 0, "MXBlockB": 0,
  }
  problemType.update(overrides.pop("ProblemType", {}))
  params = {
    "ProblemType": problemType, "ISA": isa, "MatrixInstruction": mi,
    "WorkGroup": [16, 16, 1], "WavefrontSize": 32, "DepthU": 64,
    "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
    "ScheduleIterAlg": 4, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
    "TransposeLDS": -1, "LdsPadA": -1, "LdsPadB": -1,
    "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1, "1LDSBuffer": 0,
    "VectorWidthA": -1, "VectorWidthB": -1, "StoreVectorWidth": -1,
    "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1,
    "LocalReadVectorWidth": -1, "SourceSwap": False, "ExpandPointerSwap": False,
    "GlobalSplitUAlgorithm": "MultipleBuffer", "TDMInst": 3, "LDSTrInst": False,
    "StreamK": 0, "StreamKForceDPOnly": 0, "PrefetchAcrossPersistent": 0,
    "UseSubtileImpl": False, "StoreRemapVectorWidth": 0,
    "DirectToVgprA": False, "DirectToVgprB": False,
    "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 1,
    "TDMFuse": 0, "TDMSplit": False, "InitCIterWmma": 0,
  }
  params.update(overrides)
  params.update(matrixInstructionToMIParameters(
    mi, isa, params["WavefrontSize"], problemType, params["WorkGroup"], gfx1250_iim))
  return params


def _emit_asm(gfx1250_iim, assembler, mi=None, **overrides):
  """(solution, assembly text or None) of the kernel of _params. CPU-only; no GPU is touched."""
  return emit_asm(gfx1250_iim, assembler, _params(gfx1250_iim, mi, **overrides))


# The store blocks of a GlobalSplitU=1 kernel are labelled by write path: GSU1 stores D directly,
# MB copies the GSU>1 partials to the workspace and OptNLL_MB is the store of the optimized NLL.
_LABEL = re.compile(r"^label_GW_B(\d)_FD0_VW(\d+)_(\w+?)_(Then|NonEdge):")
_STORE = re.compile(r"^\s*buffer_store\S* ")
_STORE_D = re.compile(r"^\s*buffer_store\S* [^\n]*s\[sgprSrdD:")


def _stores(asm):
  """{(path, beta): Then vector widths, slots of the TDM stores of the Then blocks, NonEdge label, TDM stores
  and D buffer_stores of the NonEdge block, D buffer_store count, buffer_store count}."""
  paths = defaultdict(lambda: {"then": [], "tdm": [], "nonEdge": False, "nonEdgeTdm": 0, "nonEdgeBuffer": 0,
                               "buffer": 0, "stores": 0})
  block = part = None
  for line in asm.splitlines():
    m = _LABEL.match(line)
    if m:
      block, slot, part = paths[(m.group(3), m.group(1))], int(m.group(2)), m.group(4)
      if part == "Then":
        block["then"].append(slot)
      else:
        block["nonEdge"] = True
    elif block is not None and "tensor_store_from_lds " in line:
      if part == "Then":
        block["tdm"].append(slot)
      else:
        block["nonEdgeTdm"] += 1
    elif block is not None and _STORE.match(line):
      storeD = bool(_STORE_D.match(line))
      block["stores"] += 1
      block["buffer"] += storeD
      block["nonEdgeBuffer"] += storeD and part == "NonEdge"
  return paths


def _tdmBlock(asm, beta, slot, factorDim=0):
  """The GSU1 TDM block of `beta`, the Then block of vector width `slot` in the store tree of `factorDim`, up
  to its tensor_store_from_lds."""
  then = re.search(r"^label_GW_B%s_FD%u_VW%u_GSU1_Then:" % (beta, factorDim, slot), asm, re.M)
  assert then, "no GSU1 Then block of vector width %u for beta %s" % (slot, beta)
  return asm[then.end():asm.index("tensor_store_from_lds ", then.end())]


def _code(line):
  """The instruction of an assembly line, without its comment."""
  return line.partition("//")[0].strip()


def _outside(asm):
  """The instructions and labels of the kernel code outside the NonEdge blocks of the direct store and the
  Then block of the OptNLL store, each from its label to the next global-write label: labels renamed in
  order of appearance, without the instruction-prefetch hints StinkyTofu places by code offset."""
  lines, inside = [], False
  for line in map(_code, re.sub(r"\.amdgpu_metadata.*?\.end_amdgpu_metadata", "", asm, flags=re.S).splitlines()):
    if line.startswith("label_GW_"):
      inside = line.endswith(("_GSU1_NonEdge:", "_OptNLL_MB_Then:"))
    elif inside or line.startswith("s_prefetch_inst_pc_rel"):
      continue
    lines.append(line)
  names, code = {}, []
  for line in lines:
    if line and not line.startswith((".", "/*")) and (not line.endswith(":") or line.startswith("label_")):
      code.append(re.sub(r"\blabel_\w+", lambda m: names.setdefault(m.group(0), "L%u" % len(names)), line))
  return code


# Level 0 has no TDM block, so it is emitted with the default TDMInst only.
_LEVEL_TDM_INST = [(0, 3)] + [(level, tdmInst) for level in _LEVELS for tdmInst in (0, 3)]


@pytest.mark.parametrize("sourceSwap", [False, True])
@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("level, tdmInst", _LEVEL_TDM_INST)
def test_emitted_paths_of_each_level(gp_gfx1250, gfx1250_iim, assembler, capsys, level, tdmInst, adaptive,
                                     sourceSwap):
  """Each beta of the direct store keeps the store blocks of level 0 and stores the blocks its level picks
  through the TDM block, which reads C from the staged tile; the GSU>1 copy of the partials keeps buffer_store.
  Level 3 also stores the NonEdge path of each beta through the TDM block, so no block of the direct store
  keeps a buffer_store of D there.
  The block drains tensor_load_to_lds before it reuses the main-loop LDS only when A or B is a TDM load."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, TDMInst=tdmInst, AdaptiveGemm=adaptive,
                       SourceSwap=sourceSwap)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 8
  cascade = list(_VECTOR_WIDTHS[adaptive])
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["then"], reverse=True) == cascade
    assert sorted(direct["tdm"], reverse=True) == sorted(_blocks(level, vectorWidths=cascade), reverse=True)
    assert direct["nonEdge"] and direct["nonEdgeTdm"] == (level == 3), (beta, direct)
    assert bool(direct["nonEdgeBuffer"]) == bool(direct["buffer"]) == (level < 3), (beta, direct)
  partials = stores[("MB", "0")]
  assert sorted(partials["then"], reverse=True) == cascade and partials["nonEdge"] and not partials["tdm"]
  assert not partials["nonEdgeTdm"]
  if level:
    block = _tdmBlock(asm, "1", {1: 1, 2: 8, 3: 8}[level])
    assert not re.search(r"^\s*buffer_load\S* [^\n]*s\[sgprSrdC:", block, re.M)
    loads = list(re.finditer(r"^\s*tensor_load_to_lds ", block, re.M))
    assert len(loads) == 1 and re.search(r"^\s*ds_load\S* ", block[loads[0].end():], re.M)
    drain = re.search(r"^\s*s_wait_tensorcnt[^\n]*drain tensor_load_to_lds", asm, re.M)
    assert bool(drain) == (tdmInst == 3)


@pytest.mark.parametrize("dest", ["B", "H"])
@pytest.mark.parametrize("level", [0] + _LEVELS)
def test_optnll_store_goes_through_the_tdm_block_at_level_3(gp_gfx1250, gfx1250_iim, assembler, capsys, level, dest):
  """The OptNLL store is an unmasked full-tile store of D, a NonEdge path without edge blocks, so level 3
  stores it through the TDM block whatever the alignment of the columns, and levels 0-2 keep buffer_store."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, ProblemType=_dest(dest))
  assert sol.get("Valid"), capsys.readouterr().out
  optNLL = _stores(asm)[("OptNLL_MB", "0")]
  assert optNLL["then"] == [8] and not optNLL["nonEdge"]
  assert optNLL["tdm"] == ([8] if level == 3 else [])
  assert bool(optNLL["buffer"]) == (level < 3), optNLL


@pytest.mark.parametrize("adaptive", [0, 1])
def test_level_3_is_level_2_plus_the_nonedge_paths(gp_gfx1250, gfx1250_iim, assembler, capsys, adaptive):
  """No level adds, drops or moves a global-write label or a branch to one, so none checks the alignment of
  the columns, and a level-3 kernel equals its level-2 twin outside the NonEdge blocks of the direct store
  and the OptNLL store, which it stores through the TDM block."""
  asms = {}
  for level in [0] + _LEVELS:
    sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, AdaptiveGemm=adaptive)
    assert sol.get("Valid"), capsys.readouterr().out
    asms[level] = asm
  for level, asm in asms.items():
    assert _skeleton(asm) == _skeleton(asms[0]), level
  assert _outside(asms[3]) == _outside(asms[2])


@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("level", [0] + _LEVELS)
def test_sourceswap_store_vector_width_2(gp_gfx1250, gfx1250_iim, assembler, capsys, level, adaptive):
  """With SourceSwap the store vector width is VectorWidthA: MIWaveTile0=2 gives 2, whose only
  narrower path is VW1, so level 1 sends the VW1 path through the TDM block."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 2, 8, 4, 1], TDMStoreInst=level,
                       AdaptiveGemm=adaptive, SourceSwap=True)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 2
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["then"], reverse=True) == [2, 1]
    assert sorted(direct["tdm"], reverse=True) == {0: [], 1: [1], 2: [2, 1], 3: [2, 1]}[level]


_FP16 = {"DataType": "H", "MacDataTypeA": "H", "MacDataTypeB": "H", "DestDataType": "H"}


def _dest(dest, **problemType):
  """The ProblemType of bf16 ("B") or fp16 ("H") inputs and D, converted from fp32 accumulators."""
  return dict(_FP16 if dest == "H" else {}, **problemType)


def test_fp16_c_is_read_back_from_the_staged_tile(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """fp16 C comes back from the staged tile in 16-byte vectors at StoreVectorWidth 8."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=2, ProblemType=_FP16)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 8
  block = _tdmBlock(asm, "1", 8)
  load = re.search(r"^\s*tensor_load_to_lds ", block, re.M)
  assert load and re.search(r"^\s*ds_load_b128 ", block[load.end():], re.M)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_fp16_with_beta_at_store_vector_width_1(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """At StoreVectorWidth 1 every level stores the one masked block through the TDM block, which reads each
  C element back into its own VGPR: an even one with ds_load_u16, an odd one with ds_load_u16_d16_hi
  into the high half, where readInput puts it."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, StoreVectorWidth=1, ProblemType=_FP16)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 1
  loads = re.findall(r"^\s*(ds_load_\w+) v\S+, .*// load C element \((\d+), (\d+)\)", _tdmBlock(asm, "1", 1), re.M)
  assert {mnemonic for mnemonic, _, _ in loads} == {"ds_load_u16", "ds_load_u16_d16_hi"}
  for mnemonic, coord0, coord1 in loads:
    assert (mnemonic == "ds_load_u16_d16_hi") == (int(coord0) % 2 == 1), (mnemonic, coord0, coord1)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_fp16_d_stages_each_element_from_its_own_register(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """fp16 D converted from fp32 accumulators at StoreVectorWidth 1: every element, odd or even, is in the low half
  of its own VGPR, where the buffer_store path stores it from, so the TDM block stages each with one ds_store_b16
  of a single register, none with the d16_hi store."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, StoreVectorWidth=1, ProblemType=_FP16)
  assert sol.get("Valid"), capsys.readouterr().out
  staged = re.findall(r"^\s*(ds_store_\w+) v\S+, (v\S+) .*// stage D element \((\d+), (\d+)\)", _tdmBlock(asm, "0", 1),
                      re.M)
  assert staged and {mnemonic for mnemonic, _, _, _ in staged} == {"ds_store_b16"}
  assert all(re.fullmatch(r"v\[vgprValuC\+\d+(-\d+)?\]", src) for _, src, _, _ in staged), staged[:2]
  assert {int(coord0) % 2 for _, _, coord0, _ in staged} == {0, 1}


@pytest.mark.parametrize("level", _LEVELS)
def test_store_vector_width_4(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """At StoreVectorWidth 4 the TDM block stages 8-byte vectors in every block the level picks."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, StoreVectorWidth=4)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 4
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["tdm"]) == sorted(_blocks(level, vectorWidths=direct["then"]))
    for slot in direct["tdm"]:
      assert re.search(r"^\s*ds_store_b64 ", _tdmBlock(asm, beta, slot), re.M)


@pytest.mark.parametrize("level", _LEVELS)
def test_store_vector_width_16_splits_32_byte_vectors(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """bf16 at StoreVectorWidth 16 (VectorWidthA 2) puts 32-byte vectors through the TDM block, wider than any
  DS instruction, so it stages each D vector, and with beta != 0 reads each C vector back, with two
  16-byte DS ops."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, StoreVectorWidth=16, VectorWidthA=2)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 16
  for beta, tag, op in (("0", "stage D element", "ds_store_b128"), ("1", "load C element", "ds_load_b128")):
    lines = [line for line in _tdmBlock(asm, beta, {1: 1, 2: 16, 3: 16}[level]).splitlines() if tag in line]
    assert lines and {line.split()[0] for line in lines} == {op}, (beta, lines[:2])
    coords = Counter(re.search(r"%s \((\d+), (\d+)\)" % tag, line).groups() for line in lines)
    assert set(coords.values()) == {2}, (beta, lines[:4])


@pytest.mark.parametrize("level", _LEVELS)
def test_wide_vectors_split_into_16_byte_ds_ops(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """fp32 D from bf16 inputs at StoreVectorWidth 8 stages 32-byte D vectors, which dsStore writes as two 16-byte
  stores, and with beta != 0 reads each C vector back with two 16-byte DS loads. The staged tile is 4 bytes per
  element."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, StoreVectorWidth=8, ProblemType={"DestDataType": "S"})
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"] * 4
  for beta, tag, op in (("0", "stage D element", "ds_store_b128"), ("1", "load C element", "ds_load_b128")):
    lines = [line for line in _tdmBlock(asm, beta, {1: 1, 2: 8, 3: 8}[level]).splitlines() if tag in line]
    assert lines and {line.split()[0] for line in lines} == {op}, (beta, lines[:2])
    coords = Counter(re.search(r"%s \((\d+), (\d+)\)" % tag, line).groups() for line in lines)
    assert set(coords.values()) == {2}, (beta, lines[:4])


_INT32 = {"DataType": "I8", "MacDataTypeA": "I8", "MacDataTypeB": "I8", "DestDataType": "I",
          "ComputeDataType": "I"}


@pytest.mark.parametrize("level", _LEVELS)
def test_int32_d_stages_4_byte_elements(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """int32 D from int32 accumulators (i8 inputs) is staged like fp32 D: at store vector width 1 every D element is
  one 4-byte DS store into a staged tile of 4 bytes per element, and with beta != 0 every C element is read back
  with one 4-byte DS load."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 64, 1, 1, 2, 2, 2, 2], TDMStoreInst=level,
                       StoreVectorWidth=1, ProblemType=_INT32)
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"] * 4
  assert "tensor_store_from_lds" in asm
  for tag, op in (("stage D element", "ds_store_b32"), ("load C element", "ds_load_b32")):
    assert {line.split()[0] for line in asm.splitlines() if tag in line} == {op}, tag


@pytest.mark.parametrize("dest, compute", [("F8", "S"), ("B8", "S"), ("I8", "I")])
@pytest.mark.parametrize("level", _LEVELS)
def test_one_byte_d_stages_with_ds_store_b8_at_vw1(gp_gfx1250, gfx1250_iim, assembler, capsys, level, dest, compute):
  """fp8 and bf8 D from fp32 accumulators and int8 D from int32 accumulators at store vector width 1: every D
  element is one ds_store_b8 into a staged tile of 1 byte per element, and with beta != 0 every C element is read
  back with one ds_load_u8, into the low byte of its VGPR, where readInput's byte load puts it."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 64, 1, 1, 2, 2, 2, 2], TDMStoreInst=level,
                       StoreVectorWidth=1, ProblemType={"DataType": dest, "MacDataTypeA": dest, "MacDataTypeB": dest,
                                                       "DestDataType": dest, "ComputeDataType": compute})
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"]
  assert "tensor_store_from_lds" in asm
  for tag, op in (("stage D element", "ds_store_b8"), ("load C element", "ds_load_u8")):
    assert {line.split()[0] for line in asm.splitlines() if tag in line} == {op}, tag


def _fp64_params(**overrides):
  """An fp64 VALU kernel shaped like those of develop's f64_gfx1250.yaml: ThreadTile 4x4 over one wave."""
  from tensilelite.Common.Architectures import gfxToIsa
  params = {
    "ProblemType": {"OperationType": "GEMM", "DataType": "D", "DestDataType": "D", "ComputeDataType": "D",
                    "HighPrecisionAccumulate": False, "TransposeA": True, "TransposeB": False, "UseBeta": True,
                    "Batched": True},
    "ISA": gfxToIsa("gfx1250"), "KernelLanguage": "Assembly", "WavefrontSize": 32, "MatrixInstruction": [],
    "EnableMatrixInstruction": False, "ThreadTile": [4, 4], "WorkGroup": [8, 4, 1], "DepthU": 16,
    "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "ScheduleIterAlg": 4, "StaggerU": 0, "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer", "TDMInst": 0, "GlobalReadVectorWidthA": 1, "GlobalReadVectorWidthB": 1,
    "StoreVectorWidth": -1, "StoreRemapVectorWidth": 0, "WorkGroupMapping": 1, "TransposeLDS": -1, "LdsPadA": -1,
    "LdsPadB": -1, "1LDSBuffer": 0, "StreamK": 0, "StreamKForceDPOnly": 0,
  }
  params.update(overrides)
  return params


_COMPLEX = {"DataType": "C", "MacDataTypeA": "C", "MacDataTypeB": "C", "DestDataType": "C", "ComputeDataType": "C",
            "HighPrecisionAccumulate": False}


@pytest.mark.parametrize("kind", ["fp64", "complex"])
@pytest.mark.parametrize("level", _LEVELS)
def test_eight_byte_d_stages_two_registers_per_element(gp_gfx1250, gfx1250_iim, assembler, capsys, level, kind):
  """fp64 D on a VALU kernel and complex float D on a WMMA kernel: at store vector width 1 every D element is one
  ds_store_b64 of its register pair, ValuC + 2*sumIdx as addStore stores it, into a staged tile of 8 bytes per
  element, and with beta != 0 every C element is read back with one ds_load_b64."""
  if kind == "fp64":
    sol, asm = emit_asm(gfx1250_iim, assembler, _fp64_params(TDMStoreInst=level, StoreVectorWidth=1))
  else:
    sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 4, 1, 1, 2, 2, 2, 2], TDMStoreInst=level, DepthU=16,
                         StoreVectorWidth=1, ProblemType=_COMPLEX)
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"] * 8
  assert "tensor_store_from_lds" in asm
  for tag, op in (("stage D element", "ds_store_b64"), ("load C element", "ds_load_b64")):
    assert {line.split()[0] for line in asm.splitlines() if tag in line} == {op}, tag
  staged = [line for line in asm.splitlines() if "stage D element" in line]
  pairs = [re.search(r"v\[vgprValuC\+(\d+):vgprValuC\+(\d+)\+1\]", line) for line in staged]
  assert pairs and all(p for p in pairs), staged[:3]
  assert all(p.group(1) == p.group(2) and int(p.group(1)) % 2 == 0 for p in pairs), staged[:3]


_ZGEMM = {"OperationType": "GEMM", "DataType": "Z", "DestDataType": "Z", "ComputeDataType": "Z",
          "HighPrecisionAccumulate": False, "TransposeA": True, "TransposeB": False, "UseBeta": True, "Batched": True}


@pytest.mark.parametrize("level", _LEVELS)
def test_complex_double_descriptor_uses_8_byte_units(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """Complex double D on a VALU kernel: TDM data_size is at most 8 bytes, so every descriptor of D and C counts
  8-byte units, two per element, with data_size 3 and tensor_dim0, tile0 and the stride doubled. At store vector
  width 1 every D element is one ds_store_b128 of its four registers, ValuC + 4*sumIdx as addStore stores it, into a
  staged tile of 16 bytes per element, and with beta != 0 every C element is read back with one ds_load_b128."""
  sol, asm = emit_asm(gfx1250_iim, assembler, _fp64_params(TDMStoreInst=level, StoreVectorWidth=1, ProblemType=_ZGEMM))
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"] * 16
  descriptors = asm.count("tensor_dim0 = SizeI - wg0*MT0")
  assert descriptors and set(re.findall(r"Set data_size to (\d+)", asm)) == {"3"}
  assert re.findall(r"set tile0 to (\d+)", asm) == [str(2 * sol["MacroTile0"])] * descriptors
  assert asm.count("tensor_dim0 in 8-byte units") == descriptors
  assert len(re.findall(r"Stride[CD] in 8-byte units", asm)) == descriptors
  for tag, op in (("stage D element", "ds_store_b128"), ("load C element", "ds_load_b128")):
    assert {line.split()[0] for line in asm.splitlines() if tag in line} == {op}, tag
  staged = [re.search(r"v\[vgprValuC\+(\d+):vgprValuC\+(\d+)\+3\]", line) for line in asm.splitlines()
            if "stage D element" in line]
  assert staged and all(m and m.group(1) == m.group(2) and int(m.group(1)) % 4 == 0 for m in staged), staged[:3]


@pytest.mark.parametrize("algorithm, aga", _GSU_ALGORITHMS)
def test_dynamic_gsu_with_epilogue_vectors_at_level_3(gp_gfx1250, gfx1250_iim, assembler, capsys, algorithm, aga):
  """GlobalSplitU=-1 with bias and ScaleAlphaVec, for every algorithm and AdaptiveGemmGSUA: the GSU == 1 branch,
  the OptNLL store and, with MultipleBufferSingleKernel and MbskPrefetchMethod 0, the TDM copy of the reduced tile
  go through the TDM block, beside the buffer_store copy of the reduced tile; the partials and atomics of the
  GSU > 1 branch keep buffer_store or buffer atomics. AtomicDest runs without the vectors, which develop rejects with it (the
  epilogue would run per GSU slice), and with the even AssertFree0ElementMultiple of packed bf16 atomics."""
  extra = {"AssertFree0ElementMultiple": 2} if algorithm == "AtomicDest" else \
    {"ProblemType": {"UseBias": 1, "UseScaleAlphaVec": 1}}
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=3, GlobalSplitU=-1, GlobalSplitUAlgorithm=algorithm,
                       AdaptiveGemmGSUA=aga, **extra)
  assert sol.get("Valid"), capsys.readouterr().out
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["tdm"]) == sorted(direct["then"]) and direct["nonEdgeTdm"] == 1 and not direct["buffer"]
  optNLL = [path for path in stores if path[0].startswith("OptNLL")]
  assert optNLL and all(stores[path]["tdm"] == [8] for path in optNLL), optNLL
  reduced = [path for path in stores if path[0] == "MBSK"]
  copies = [path for path in stores if path[0] == "MBSKBuffer"]
  tdmReduced = sol["MbskPrefetchMethod"] == 0
  assert bool(reduced) == (algorithm == "MultipleBufferSingleKernel"), reduced
  assert bool(copies) == (bool(reduced) and tdmReduced), copies
  assert all((sorted(stores[path]["tdm"]) == sorted(stores[path]["then"]) and stores[path]["nonEdgeTdm"] == 1)
             == tdmReduced and not stores[path]["buffer"] for path in reduced), reduced
  assert not any(stores[path]["tdm"] or stores[path]["nonEdgeTdm"] for path in copies), copies
  partials = [path for path in stores if not path[0].startswith(("GSU1", "OptNLL", "MBSK"))]
  assert reduced or partials
  assert not any(stores[path]["tdm"] or stores[path]["nonEdgeTdm"] for path in partials), partials
  assert sum(len(stores[path]["tdm"]) + stores[path]["nonEdgeTdm"] for path in stores) == asm.count("tensor_store_from_lds ")
  assert reduced or any(stores[path]["stores"] for path in partials) or re.search(r"^\s*buffer_atomic\S* ", asm, re.M)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_mbsk_reduced_tile_at_each_level(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """Static GlobalSplitU=2 with MultipleBufferSingleKernel and MbskPrefetchMethod 0: the TDM copy of the reduced
  tile takes the TDM blocks the GSU == 1 store takes at the level, OptNLL_MBSK's being NonEdge, with its descriptor
  from SrdTD and D staged at the size of D; the buffer_store copy, which a GSU above 2 takes, stores every block
  with buffer_store to SrdTD, and one GSU check picks the copy. OptNLL_MBSK, which runs at GSU 1 only, without a
  reduction, keeps its store."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, GlobalSplitU=2,
                       GlobalSplitUAlgorithm="MultipleBufferSingleKernel", MbskPrefetchMethod=0)
  assert sol.get("Valid"), capsys.readouterr().out
  stores = _stores(asm)
  for beta in "01":
    reduced, direct, copy = stores[("MBSK", beta)], stores[("GSU1", beta)], stores[("MBSKBuffer", beta)]
    assert (reduced["tdm"], reduced["nonEdgeTdm"], reduced["buffer"]) == (direct["tdm"], direct["nonEdgeTdm"], 0)
    assert copy["then"] == reduced["then"] and copy["nonEdge"] and not copy["tdm"] and not copy["nonEdgeTdm"]
    assert copy["stores"] and not copy["buffer"]
  assert stores[("OptNLL_MBSK", "0")]["tdm"] == ([8] if level == 3 else [])
  assert not any(path[0] == "OptNLL_MBSKBuffer" for path in stores)
  assert asm.count("64-bit byte offset from SrdTD") == sum(len(stores[("MBSK", beta)]["tdm"])
                                                          + stores[("MBSK", beta)]["nonEdgeTdm"] for beta in "01")
  assert re.search(r"^\s*buffer_store\S* .*sgprSrdTD", asm, re.M)
  assert asm.count("GSU > 2: each wave reduces on its own") == 1
  assert {line.split()[0] for line in asm.splitlines() if "stage D element" in line} == {"ds_store_b128"}


def test_mbsk_reduced_tile_keeps_buffer_store_with_prefetch_method_1(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """MbskPrefetchMethod 1: each wave takes the last decrement of its own synchronizer slot and reduces on its own,
  so the reduced tile keeps buffer_store, with no TDM copy and no GSU check; the GSU == 1 store and OptNLL_MBSK,
  which runs at GSU 1 only, keep the block."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=3, GlobalSplitU=2,
                       GlobalSplitUAlgorithm="MultipleBufferSingleKernel", MbskPrefetchMethod=1)
  assert sol.get("Valid"), capsys.readouterr().out
  stores = _stores(asm)
  assert stores[("MBSK", "0")]["stores"] and not any(path[0] == "MBSKBuffer" for path in stores)
  tdm = [(path, stores[path]["tdm"], stores[path]["nonEdgeTdm"]) for path in stores
         if path[0] == "MBSK" and (stores[path]["tdm"] or stores[path]["nonEdgeTdm"])]
  assert not tdm, tdm
  assert stores[("OptNLL_MBSK", "0")]["tdm"] == [8]
  assert stores[("GSU1", "0")]["nonEdgeTdm"] == 1
  assert "64-bit byte offset from SrdTD" not in asm and "each wave reduces on its own" not in asm


_AUX_E = {"DataType": "B", "DestDataType": "B", "ComputeDataType": "s", "HighPrecisionAccumulate": True,
          "TransposeA": False, "TransposeB": False, "UseBeta": True, "Batched": True, "UseE": True, "UseBias": 1,
          "BiasDataTypeList": ["s", "B"], "Activation": True, "ActivationType": "hipblaslt_all", "UseScaleAlphaVec": 1,
          "SupportUserArgs": True}
_AUX_E_SOLUTION = dict(mi=[16, 16, 32, 1, 1, 8, 4, 2, 2], DepthU=64, VectorWidthA=1, VectorWidthB=1,
                       GlobalReadVectorWidthA=1, GlobalReadVectorWidthB=1, LocalReadVectorWidth=8, PrefetchGlobalRead=2,
                       PrefetchLocalRead=1, ScheduleIterAlg=4, TransposeLDS=-1, LDSTrInst=True, TDMInst=3,
                       ExpandPointerSwap=False, StoreVectorWidth=8, SourceSwap=False, ActivationFuncCall=True)


def _storeBlocks(asm):
  """{name: lines} of the store blocks, label_GW_*_Then and _NonEdge up to the next label_GW_ label."""
  blocks, cur = {}, None
  for line in asm.splitlines():
    m = re.match(r"label_GW_(\w+_(?:Then|NonEdge)):$", line)
    if line.startswith("label_GW_"):
      cur = blocks.setdefault(m.group(1), []) if m else None
    elif cur is not None:
      cur.append(line.strip())
  return blocks


def _sameRegs(lines):
  return [re.sub(r"\bv\[[^\]]*\]|\bv\d+\b|\bs\[\d+:\d+\]|\bs\d+\b", "R", l.split("//")[0].strip()) for l in lines]


@pytest.mark.parametrize("level", [1, 2, 3])
def test_e_kernel_tdm_blocks_keep_the_masked_side_effects(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """Develop's forward E kernel of bbs_bias_gelu_aux_cls_gfx1250.yaml: each TDM block stages D in place of its stores
  and keeps the E stores of the level-0 block; an edge TDM block is that edge block, with its vector width, its masks
  and its C loads, so no element outside D reaches E or the epilogue."""
  _, base = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=0, ProblemType=_AUX_E, **_AUX_E_SOLUTION)
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, ProblemType=_AUX_E, **_AUX_E_SOLUTION)
  assert base and sol.get("Valid"), capsys.readouterr().out
  ref, blocks = _storeBlocks(base), _storeBlocks(asm)
  tdm = {k: v for k, v in blocks.items() if any(l.startswith("tensor_store_from_lds ") for l in v)}
  assert tdm and set(tdm) <= set(ref)
  pick = lambda lines, tc, op: _sameRegs([l for l in lines if l.startswith(op) and "s[sgprSrd%s:" % tc in l])
  for k, v in tdm.items():
    assert not pick(v, "D", "buffer_store"), k
    assert pick(v, "E", "buffer_store") == pick(ref[k], "E", "buffer_store") and pick(v, "E", "buffer_store"), k
    if k.endswith("_Then"):
      assert pick(v, "C", "buffer_load") == pick(ref[k], "C", "buffer_load"), k
      assert sum("stage D element" in l for l in v) == len(pick(ref[k], "D", "buffer_store")), k


@pytest.mark.parametrize("level", [1, 2, 3])
def test_compact_loop_store_tdm_block_outside_the_loop(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """Develop's CompactLoopStore E kernel: each TDM block runs its stores in the CompactLoopStore loop. It sets up the
  staged tile (and loads C) before the loop label, stages every element inside the loop, steps the staged tile's base
  once an iteration when the loop runs more than once, and stores the tile once, after the loop exits; no D
  buffer_store is left in any TDM block."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, ProblemType=_AUX_E,
                       **dict(_AUX_E_SOLUTION, CompactLoopStore=True))
  assert sol.get("Valid"), capsys.readouterr().out
  blocks = _storeBlocks(asm)
  tdm = {k: v for k, v in blocks.items() if any(l.startswith("tensor_store_from_lds ") for l in v)}
  assert tdm
  iterated = []
  for k, v in tdm.items():
    assert not [l for l in v if l.startswith("buffer_store") and "s[sgprSrdD:" in l], k
    label = [n for n, l in enumerate(v) if re.match(r"label_CLS_?\d*:$", l)]
    assert len(label) == 1, k
    name = v[label[0]][:-1]
    count = [int(m.group(1), 0) for l in v for m in [re.match(r"s_mov_b32 s\[sgprCLSLoopCounter\], (\w+)", l)] if m]
    setup = min(n for n, l in enumerate(v) if "tile column = coord1 - wg1*MT1" in l)
    branch = max(n for n, l in enumerate(v) if name in l)
    stage = [n for n, l in enumerate(v) if "stage D element" in l]
    step = [n for n, l in enumerate(v) if "next iteration's rows of the staged tile" in l]
    store = [n for n, l in enumerate(v) if l.startswith("tensor_store_from_lds ")]
    assert len(count) == 1 and setup < label[0] < min(stage) and max(stage) < branch < min(store), k
    assert len(step) == int(count[0] > 1) and all(max(stage) < n < branch for n in step), k
    iterated.append(count[0] > 1)
  assert any(iterated), "no TDM block loops more than once"


def test_automatic_store_remap_vector_width(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """StoreRemapVectorWidth=-1 resolves to 0 with SourceSwap and to a remap width without it; TDMStoreInst
  takes both."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=2, SourceSwap=True, StoreRemapVectorWidth=-1)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreRemapVectorWidth"] == 0
  assert "tensor_store_from_lds " in asm
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 4, 4, 2, 2], TDMStoreInst=2, SourceSwap=False,
                       StoreRemapVectorWidth=-1, VectorWidthA=1, VectorWidthB=1)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreRemapVectorWidth"] > 0 and "tensor_store_from_lds " in asm


_REMAP = ("storeRemap lw", "storeRemap lr", "shift storeRemap coord1")


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize("srvw, waveTile", [(4, 4)])
def test_srvw_tdm_blocks_skip_the_remap(gp_gfx1250, gfx1250_iim, assembler, capsys, srvw, waveTile, level):
  """With StoreRemapVectorWidth 4 (MT128x128) a TDM block stages D itself: it has no remap local write,
  read-back or row shift, while the blocks that keep buffer_store still remap through LDS. An explicit remap
  width needs SourceSwap 0, and one wave's write must cover an M column."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, waveTile, waveTile, 2, 2], TDMStoreInst=level,
                       StoreRemapVectorWidth=srvw, SourceSwap=False, VectorWidthA=1, VectorWidthB=1)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreRemapVectorWidth"] == srvw
  stores = _stores(asm)
  for beta in "01":
    slots = sorted(set(stores[("GSU1", beta)]["tdm"]))
    assert slots, beta
    for slot in slots:
      block = _tdmBlock(asm, beta, slot)
      assert not any(tag in block for tag in _REMAP), (beta, slot)
  if level < 3:
    assert all(tag in asm for tag in _REMAP[:2])


def test_sav_device_scalar_bit_is_along_m(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """UseScaleAlphaVec 5 is the vector along M with the device-scalar bit, which only changes how the vector
  preload selects alpha, so a level-2 kernel stores through the TDM block as with UseScaleAlphaVec 1."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=2, ProblemType={"UseBias": 1, "UseScaleAlphaVec": 5})
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["ProblemType"]["UseScaleAlphaVec"] == 5 and "tensor_store_from_lds " in asm


@pytest.mark.parametrize("dest", ["B", "H"])
@pytest.mark.parametrize("useBias, factorDims", [(2, {"1"}), (3, {"0", "1"})], ids=["along_n", "along_m_or_n"])
def test_both_factor_dims_use_the_tdm_block(gp_gfx1250, gfx1250_iim, assembler, capsys, useBias, factorDims, dest):
  """Bias and ScaleAlphaVec along N (UseBias 2), or along M or N as the kernel argument picks (UseBias 3):
  generateBetaModules builds the store tree of each factor dim, and at level 3 each of them stores through
  the TDM block."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=3,
                       ProblemType=_dest(dest, UseBias=useBias, UseScaleAlphaVec=useBias))
  assert sol.get("Valid"), capsys.readouterr().out
  stored, factorDim = set(), None
  for line in asm.splitlines():
    m = re.match(r"label_GW_B\d_FD(\d)_", line)
    if m:
      factorDim = m.group(1)
    elif factorDim is not None and "tensor_store_from_lds " in line:
      stored.add(factorDim)
  assert factorDims <= stored, stored


_UNEVEN_MIS = [[16, 16, 32, 1, 1, 1, 3, 8, 1], [16, 16, 32, 1, 1, 1, 5, 8, 1]]


@pytest.mark.parametrize("mi", _UNEVEN_MIS, ids=["mt1_48", "mt1_80"])
@pytest.mark.parametrize("level", _LEVELS)
def test_mt1_not_a_multiple_of_the_waves(gp_gfx1250, gfx1250_iim, assembler, capsys, level, mi):
  """MacroTile1 48 and 80 over 32 waves (MIWaveGroup [8, 1] with LocalSplitU 4): each wave stores
  ceil(MacroTile1/32) columns, so the columns of the last waves reach past the tile, and at 80 columns one wave's
  columns straddle its end. Every TDM block ends tensor_dim1 at the column bound min(SizeJ, (wg1+1)*MT1),
  against which each wave tests its skip, and the staged tile is reserved for the columns of every wave. With
  MacroTile1 a multiple of the waves there is no column bound."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=mi, TDMStoreInst=level, WorkGroup=[16, 16, 4], DepthU=128)
  assert sol.get("Valid"), capsys.readouterr().out
  waves = sol["NumThreads"] // sol["WavefrontSize"]
  assert sol["MacroTile1"] % waves
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * -(-sol["MacroTile1"] // waves) * waves * 2
  assert sol["LdsNumBytes"] >= sol["_TDMStoreLdsOffset"] + tdmStoreLdsBytes(sol)
  blocks = len(re.findall(r"^label_TDMStore(?:LoadC)?Skip\w*:", asm, re.M))
  bounds = [line for line in asm.splitlines() if _code(line).startswith("s_min_u32") and "column bound" in line]
  skips = [line for line in asm.splitlines() if "at or past the column bound" in line]
  assert re.search(r"^\s*tensor_store_from_lds ", asm, re.M) and len(bounds) == len(skips) == blocks
  assert "tensor_dim1 = SizeJ - first column" not in asm
  if level == 3:
    _, even = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level)
    assert "column bound" not in even and "tensor_dim1 = SizeJ - first column" in even


@pytest.mark.parametrize("dest", ["B", "H"])
def test_scale_cd_scales_through_registers(gp_gfx1250, gfx1250_iim, assembler, capsys, dest):
  """ScaleC and ScaleD are scalars the kernel loads from its arguments into SGPRs: the TDM block of a level-2
  kernel multiplies its elements by ScaleD, as the buffer_store path does."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=2, ProblemType=_dest(dest, UseScaleCD=True))
  assert sol.get("Valid"), capsys.readouterr().out
  assert "AddressScaleC" in asm and "AddressScaleD" in asm
  assert re.search(r"^\s*v_\w*mul\w* [^\n]*sgprScaleD", _tdmBlock(asm, "1", 8), re.M)


@pytest.mark.parametrize("dest", ["B", "H"])
@pytest.mark.parametrize("level", _LEVELS)
def test_grouped_gemm_generates(gp_gfx1250, gfx1250_iim, assembler, capsys, level, dest):
  """GroupedGemm sets the sizes, strides and addresses of each group in the SGPRs the TDM descriptors read."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, ProblemType=_dest(dest, GroupedGemm=True))
  assert sol.get("Valid"), capsys.readouterr().out
  assert "tensor_store_from_lds " in asm


def _cacheTokens(line):
  """The th: and scope: tokens of an instruction line, as {name: value}."""
  return dict(re.findall(r"\b(th|scope):(\S+)", _code(line)))


def test_c_load_takes_the_hints_of_c(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """The tensor_load_to_lds of C carries the th: and scope: of the buffer_load of C on the other paths."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=2, TemporalHintC=3, NonTemporalC=2)
  assert sol.get("Valid"), capsys.readouterr().out
  load = re.search(r"^\s*tensor_load_to_lds [^\n]*", _tdmBlock(asm, "1", 8), re.M)
  bufferLoad = re.search(r"^\s*buffer_load\S* [^\n]*s\[sgprSrdC:[^\n]*", asm, re.M)
  assert load and bufferLoad
  expected = _cacheTokens(bufferLoad.group(0))
  assert "th" in expected
  assert _cacheTokens(load.group(0)) == expected


def test_requires_tdm_in_the_assembler(gp_gfx1250, gfx1250_iim, assembler, capsys):
  from tensilelite.Common.Architectures import gfxToIsa

  noTdm = copy_isa_info_map(gfx1250_iim)
  noTdm[gfxToIsa("gfx1250")].asmCaps["HasTDM"] = False
  sol, asm = _emit_asm(noTdm, assembler, TDMStoreInst=2, TDMInst=0)
  assert not sol.get("Valid") and asm is None
  assert "TDMStoreInst requires asmCaps HasTDM" in capsys.readouterr().out


def _nextLine(lines, start, prefixes):
  """The index of the first line after `start` whose instruction starts with one of `prefixes`, or None."""
  return next((i for i in range(start + 1, len(lines)) if _code(lines[i]).startswith(prefixes)), None)


@pytest.mark.parametrize("overMainLoop", [True, False])
def test_staged_tile_token(overMainLoop):
  """The staged tile's LDS accesses carry the token of the main-loop buffer it lies over, or the epilogue's
  token when it lies after the main-loop buffers."""
  from types import SimpleNamespace
  from tensilelite.Components.TDMStore import _stagedTileToken
  states = SimpleNamespace(memTokenLdsBuffer0=0, memTokenEpilogue=7)
  gwb = SimpleNamespace(kernel={"_TDMStoreOverMainLoop": overMainLoop}, parentWriter=SimpleNamespace(states=states))
  assert _stagedTileToken(gwb) == (0 if overMainLoop else 7)


_PAP = {"StreamK": 3, "StreamKForceDPOnly": 1, "PrefetchAcrossPersistent": 1}


@pytest.mark.parametrize("vectors", [False, True], ids=["plain", "vectors"])
def test_pap_stages_after_the_main_loop(gp_gfx1250, gfx1250_iim, assembler, capsys, vectors):
  """With PrefetchAcrossPersistent the next tile's loads write the main-loop buffers during the epilogue, so
  the staged tile lies after them and after the vectors, from a 256-byte boundary, and the setup has neither
  the TENSORcnt drain nor the barrier that hand the main-loop LDS to the staged tile. With the vectors, whose
  separate region has its own token, no TENSORcnt wait comes between the last WMMA of the tile and the
  first staging store. The tile is MT128x128: a staged MT256x256 tile takes a PAP kernel past MaxLDS."""
  extra = {"ProblemType": {"UseBias": 1, "UseScaleAlphaVec": 1}} if vectors else {}
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 4, 4, 2, 2], TDMStoreInst=2, **_PAP, **extra)
  assert sol.get("Valid"), capsys.readouterr().out
  offset, staged = sol["_TDMStoreLdsOffset"], tdmStoreLdsBytes(sol)
  assert not sol["_TDMStoreOverMainLoop"] and offset > 0 and offset % 256 == 0
  assert sol["LdsNumBytes"] >= offset + staged
  if vectors:
    assert sol["_SeparateEpilogueLds"] and offset >= sol["LdsOffsetBias"] + sol["MacroTile0"] * 4 * 2
  assert "drain tensor_load_to_lds before reusing its LDS" not in asm
  assert "all waves are done with the main-loop LDS" not in asm
  lines = asm.splitlines()
  first = next(i for i, line in enumerate(lines) if "stage D element" in line)
  last = max(i for i in range(first) if _code(lines[i]).startswith("v_wmma"))
  if vectors:
    assert not any(_code(line).startswith("s_wait_tensorcnt") for line in lines[last:first]), lines[last:first]


_RAP = dict(_PAP, ReuseAcrossPersistent=1, DepthU=256, AssertSummationElementMultiple=256, SourceSwap=True)
_FP8_FP32D = {"DataType": "F8", "MacDataTypeA": "F8", "MacDataTypeB": "F8", "DestDataType": "S"}


@pytest.mark.parametrize("pap", [0, 1])
@pytest.mark.parametrize("level", _LEVELS)
def test_rap_with_tdm_store_generates(gp_gfx1250, gfx1250_iim, assembler, capsys, level, pap):
  """ReuseAcrossPersistent keeps A in VGPRs from one DataParallel tile to the next: the RAP candidate, fp32 D
  from fp8 inputs at MT64x256, generates within the register budget at every level, with and without
  PrefetchAcrossPersistent."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 128, 1, 1, 2, 8, 2, 2], TDMStoreInst=level,
                       ProblemType=_FP8_FP32D, **dict(_RAP, PrefetchAcrossPersistent=pap))
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["ReuseAcrossPersistent"] and asm and "tensor_store_from_lds " in asm


def test_pap_oversized_tile_hits_max_lds(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """Under PrefetchAcrossPersistent the staged tile adds to the main-loop LDS, and develop's MaxLDS reject
  sees the sum: at DepthU 128 the main loop fits without TDMStoreInst and the sum does not."""
  sol, _ = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=2, DepthU=128, **_PAP)
  out = capsys.readouterr().out
  assert not sol.get("Valid") and "LDS" in out, out
  sol, _ = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=0, DepthU=128, **_PAP)
  assert sol.get("Valid"), capsys.readouterr().out


@pytest.mark.parametrize("streamk", [0, 3])
@pytest.mark.parametrize("level", [2, 3])
def test_tdm_waits_survive_stinkytofu(gp_gfx1250, gfx1250_iim, assembler, capsys, level, streamk):
  """In the emitted assembly the drain of tensor_load_to_lds still comes before the next barrier, and
  every tensor_store_from_lds is waited for before the next barrier or the end of the kernel."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, TDMInst=3, StreamK=streamk)
  assert sol.get("Valid"), capsys.readouterr().out
  lines = asm.splitlines()
  # Only the StinkyTofu pipeline writes STINKY_TOTAL_INST_BYTES; the kept staging wait shows its marker.
  assert "STINKY_TOTAL_INST_BYTES" in asm
  assert any(_code(line).startswith("s_wait_dscnt") and STINKY_KEEP in line for line in lines)
  drains = [i for i, line in enumerate(lines)
            if _code(line).startswith("s_wait_tensorcnt") and "drain tensor_load_to_lds" in line]
  assert drains
  for drain in drains:
    barrier = _nextLine(lines, drain, ("s_barrier",))
    assert barrier is not None and any("main-loop LDS" in line for line in lines[barrier:barrier + 3]), lines[drain]
  stores = [i for i, line in enumerate(lines) if _code(line).startswith("tensor_store_from_lds")]
  assert stores
  for store in stores:
    end = _nextLine(lines, store, ("s_barrier", "s_endpgm"))
    assert end is not None
    assert any(_code(line).startswith("s_wait_tensorcnt") for line in lines[store + 1:end]), lines[store]


_HANDOFF = ("hand compute LDS back to persistent TDM prefetch", "hand epilogue LDS back to persistent compute",
            "Sync before persistent re-entry")
_FLUSH_BARRIER = "every TDM store is done with the staged tile before the next tile writes LDS"


_MT128 = [16, 16, 32, 1, 1, 4, 4, 2, 2]


@pytest.mark.parametrize("mi, policy", [(None, {"StreamK": 3}), (None, {"StreamK": 3, "StreamKForceDPOnly": 1}),
                                        (None, {"StreamK": 4}), (None, {"StreamK": 3, "ClusterDim": [2, 1]}),
                                        (_MT128, _PAP)],
                         ids=["sk3", "dp", "sk4", "sk3_cluster", "dp_pap"])
def test_flush_has_no_duplicate_persistent_barrier(gp_gfx1250, gfx1250_iim, assembler, capsys, mi, policy):
  """With TDM loads the persistent tile handoff ends with a workgroup barrier before the next tile's work, so the
  TDM flush adds none: from the end of every store section with a TDM store to the handoff barrier nothing writes
  LDS. Without TDM
  loads a StaticGrid StreamK 3 handoff has no barrier, and the flush keeps its own. The PAP kernel and the
  control are MT128x128, like the PAP tests above."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=mi, TDMStoreInst=3, TDMInst=3, **policy)
  assert sol.get("Valid"), capsys.readouterr().out
  assert _FLUSH_BARRIER not in asm and "tensor_store_from_lds " in asm
  lines = asm.splitlines()
  ends = [i for i, line in enumerate(lines) if re.match(r"label_GW_End\w*:", line)]
  # The GlobalSplitU MultipleBuffer section ends at its own label and stores with buffer_store only.
  ends = [end for start, end in zip([0] + ends, ends) if any("tensor_store_from_lds " in x for x in lines[start:end])]
  assert ends
  for end in ends:
    handoff = next((i for i in range(end, len(lines)) if any(tag in lines[i] for tag in _HANDOFF)), None)
    assert handoff is not None, lines[end]
    writes = [code for code in map(_code, lines[end:handoff]) if code.startswith(("ds_store", "ds_write", "tensor_load_to_lds"))]
    assert not writes, (lines[end], writes[:2])
  csol, control = _emit_asm(gfx1250_iim, assembler, mi=_MT128, TDMStoreInst=3, TDMInst=0, StreamK=3)
  assert csol.get("Valid") and control, capsys.readouterr().out
  assert _FLUSH_BARRIER in control


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize("policy", [{"StreamK": 3, "StreamKForceDPOnly": 1}, {"StreamK": 4}], ids=["dp", "sk4"])
def test_persistent_kernels_stage_and_fence(gp_gfx1250, gfx1250_iim, assembler, capsys, level, policy):
  """DataParallel and StreamK 4 kernels store the tiles they complete through the TDM block, and after each
  TDM store's wait a barrier comes before the next tile's first tensor_load_to_lds."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, TDMInst=3, **policy)
  assert sol.get("Valid"), capsys.readouterr().out
  lines = asm.splitlines()
  stores = [i for i, line in enumerate(lines) if _code(line).startswith("tensor_store_from_lds")]
  assert stores
  for store in stores:
    wait = _nextLine(lines, store, ("s_wait_tensorcnt",))
    barrier = _nextLine(lines, wait, ("s_barrier",))
    load = _nextLine(lines, wait, ("tensor_load_to_lds",))
    assert barrier is not None and (load is None or barrier < load), lines[store]


@pytest.mark.parametrize("useBias, factorDim", [(1, 0), (2, 1)], ids=["along_m", "along_n"])
@pytest.mark.parametrize("streamk", [0, 3])
def test_staged_tile_goes_after_the_epilogue_vectors(gp_gfx1250, gfx1250_iim, assembler, capsys, streamk,
                                                     useBias, factorDim):
  """The bias and ScaleAlphaVec vectors, along M or N, keep LDS offset 0, as without TDMStoreInst, and the TDM
  block stages the D tile from a 256-byte boundary after them: its setup and the descriptors of the C load and
  the D store add that offset."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=2, StreamK=streamk,
                       ProblemType={"UseBias": useBias, "UseScaleAlphaVec": useBias})
  assert sol.get("Valid"), capsys.readouterr().out
  staged = sol["MacroTile0"] * sol["MacroTile1"] * 2
  assert tdmStoreLdsBytes(sol) == staged
  assert sol["LdsOffsetBias"] == sol["LdsOffsetBiasNonGSU"] == sol["LdsOffsetBiasGSU"] == 0
  offset = sol["_TDMStoreLdsOffset"]
  assert offset % 256 == 0 and offset >= 256 * 4 * 2
  assert sol["LdsNumBytes"] >= offset + staged
  assert "add lds offset" not in asm
  # The vectors overlap the main-loop LDS, so every preload barriers before its stores.
  assert_every_vector_preload_barriers(asm)
  for beta, descriptors in (("0", 1), ("1", 2)):
    slot = int(re.search(r"^label_GW_B%s_FD%u_VW(\d+)_GSU1_Then:" % (beta, factorDim), asm, re.M).group(1))
    adds = [_code(line) for line in _tdmBlock(asm, beta, slot, factorDim).splitlines()
            if "LDS offset of the staged tile" in line]
    assert sum(add.startswith("v_add") for add in adds) == 1, adds
    assert sum(add.startswith("s_add_u32") for add in adds) == descriptors, adds
    assert all(str(offset) in add or hex(offset) in add for add in adds), adds


_VECTORS = {"ProblemType": {"UseBias": 1, "UseScaleAlphaVec": 1}}


@pytest.mark.parametrize("overrides", [{}, dict(_VECTORS, GlobalSplitU=-1), dict(_VECTORS, StreamK=3)],
                         ids=["plain", "gsu-1_vectors", "sk3_vectors"])
def test_staged_tile_over_main_loop_property(gp_gfx1250, gfx1250_iim, assembler, capsys, overrides):
  """Without PrefetchAcrossPersistent the staged tile lies over the main-loop buffers (_TDMStoreOverMainLoop):
  it starts at 0, or at the first 256-byte boundary after the epilogue vectors, and the setup fences the
  main-loop LDS. TDMStoreInst 0 reserves no tile."""
  for level in (0, 2):
    sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, **deepcopy(overrides))
    assert sol.get("Valid"), capsys.readouterr().out
    offset = sol["_TDMStoreLdsOffset"]
    assert sol["_TDMStoreOverMainLoop"] == bool(level), level
    if not level:
      assert offset == 0 and tdmStoreLdsBytes(sol) == 0
      continue
    assert (offset > 0) == bool(overrides) and offset % 256 == 0 and offset >= sol["LdsOffsetBias"], offset
    assert sol["LdsNumBytes"] >= offset + tdmStoreLdsBytes(sol)
    assert "all waves are done with the main-loop LDS" in asm


@pytest.mark.parametrize("storeVectorWidth", [-1, 1])
@pytest.mark.parametrize("level", _LEVELS)
def test_epilogue_vectors_keep_lds_offset_0(gp_gfx1250, gfx1250_iim, assembler, capsys, level, storeVectorWidth):
  """No level adds an LDS offset to the addresses of the bias and ScaleAlphaVec vectors."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, StoreVectorWidth=storeVectorWidth,
                       ProblemType={"UseBias": 1, "UseScaleAlphaVec": 1})
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["LdsOffsetBias"] == 0 and sol["_TDMStoreLdsOffset"] > 0 and sol["_TDMStoreLdsOffset"] % 256 == 0
  assert not re.search(r"add (bias )?lds offset", asm)


def _skeleton(asm):
  """The global-write labels of a kernel in order, and how many of its instructions name each of them."""
  lines = [_code(line) for line in asm.splitlines()]
  names = [name for line in lines if line and not line.endswith(":") for name in re.findall(r"\blabel_GW_\w+", line)]
  return [line for line in lines if re.match(r"label_GW_\w+:$", line)], Counter(names)


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize("adaptive", [0, 1])
def test_store_vector_width_1_levels_1_and_2_store_the_one_masked_block(gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                                        adaptive, vectors):
  """At store vector width 1 the store-width cascade has one masked block, the narrowest and the one of the
  store vector width at once, so levels 1 and 2 store it through the TDM block and emit the same kernel
  apart from its name. Every level keeps the labels and branches of level 0, which send the M-edge,
  N-edge-only and corner tiles all to that block, and its NonEdge path keeps buffer_store until level 3
  stores it through the TDM block too. With bias and ScaleAlphaVec the vectors keep LDS offset 0."""
  kernels = {}
  for level in [0] + _LEVELS:
    sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=level, StoreVectorWidth=1, AdaptiveGemm=adaptive,
                         **({"ProblemType": {"UseBias": 1, "UseScaleAlphaVec": 1}} if vectors else {}))
    assert sol.get("Valid"), capsys.readouterr().out
    assert sol["StoreVectorWidth"] == 1
    kernels[level] = asm
  unnamed = lambda asm: [re.sub(r"_TDMSI[12](?=_)", "_TDMSI", _code(line)) for line in asm.splitlines()]
  assert unnamed(kernels[1]) == unnamed(kernels[2])
  for level, asm in kernels.items():
    assert _skeleton(asm) == _skeleton(kernels[0]), level
    stores = _stores(asm)
    lines = [code for code in map(_code, asm.splitlines())
             if code and not code.startswith(("s_prefetch_inst_pc_rel", "s_set_vgpr_msb"))]
    for beta in "01":
      direct = stores[("GSU1", beta)]
      assert direct["then"] == [1] and direct["nonEdge"], (level, beta, direct)
      assert direct["tdm"] == ([1] if level else []), (level, beta, direct)
      assert direct["nonEdgeTdm"] == (level == 3) and bool(direct["nonEdgeBuffer"]) == (level < 3), (level, beta, direct)
      end = lines.index("label_GW_B%s_FD0_VW1_GSU1_NonEdgeEnd:" % beta)
      assert lines[end + 1:end + 3] == ["label_GW_B%s_FD0_VW1_GSU1_%s:" % (beta, part) for part in ("Else", "Then")]
    assert "add lds offset" not in asm


def test_solution_rejects_with_the_reason(gp_gfx1250, gfx1250_iim, assembler, capsys):
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=1, GlobalSplitU=2)
  assert not sol.get("Valid") and asm is None
  assert "TDMStoreInst requires GlobalSplitU=1 or -1, or MultipleBufferSingleKernel" in capsys.readouterr().out


@pytest.mark.parametrize("dest", ["B", "H"])
@pytest.mark.parametrize("sourceSwap", [False, True])
@pytest.mark.parametrize("level", _LEVELS)
def test_local_split_u_stages_each_element_after_the_lsu_offset(gp_gfx1250, gfx1250_iim, assembler, capsys, level,
                                                                sourceSwap, dest):
  """With LocalSplitU=2, shaped like the LSU=2 kernels of the gfx1250 bf16 TN logic, each thread stages
  the elements it owns after the LDS reduction where the LSU global write would store them: the LSU
  offset goes into coord0 (coord1 with SourceSwap) after the reduction, the TDM setup reads that VGPR
  with nothing in the block writing it first, and the staging stores address the tile from the setup."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, [16, 16, 32, 1, 1, 4, 4, 1, 2], TDMStoreInst=level,
                       SourceSwap=sourceSwap, WorkGroup=[16, 4, 2], DepthU=128, ProblemType=_dest(dest))
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["LocalSplitU"] == 2 and sol["NumThreads"] == 128 and sol["StoreVectorWidth"] == 4
  operands = lambda line: [op.strip() for op in _code(line).partition(" ")[2].split(",")]
  lines = asm.splitlines()
  dim = 1 if sourceSwap else 0
  offsets = [i for i, line in enumerate(lines) if "coord%u += LSU offset%u" % (dim, dim) in line]
  assert len(offsets) == 1 and "valid offset coord%u is zero." % (1 - dim) in asm
  lsuCoord = operands(lines[offsets[0]])[0]
  assert max(i for i, line in enumerate(lines) if "post-lsu local write" in line) < offsets[0]
  slot = 1 if level == 1 else 4
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert slot in direct["tdm"] and sorted(direct["tdm"]) == sorted(_blocks(level, vectorWidths=direct["then"]))
    then = re.search(r"^label_GW_B%s_FD0_VW%u_GSU1_Then:" % (beta, slot), asm, re.M)
    assert asm.count("\n", 0, then.start()) > offsets[0]
    block = _tdmBlock(asm, beta, slot).splitlines()
    setup = next(i for i, line in enumerate(block) if "tile column = coord1 - wg1*MT1" in line)
    plus0 = next(i for i in range(setup, len(block)) if block[i].rstrip().endswith("+ coord0"))
    coords = {1: operands(block[setup])[1], 0: operands(block[plus0])[2]}
    assert coords[dim] == lsuCoord, (block[setup], block[plus0])
    assert not [line for line in block[:setup] if operands(line)[0] == lsuCoord
                and not _code(line).startswith(("s_", "v_cmp", "buffer_store", "ds_store", "tensor_"))]
    staging = [line for line in block if "stage D element" in line]
    assert staging and all(operands(line)[0] == operands(block[setup])[0] for line in staging)
