# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for TDMStoreInst: the store paths the switch picks, EdgeM, EdgeN and NonEdge, solution
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
from tensilelite.Components.GlobalWriteBatch import GlobalWriteBatchWriter
from tensilelite.Components.TDMStoreInst import DS_OFFSET_MAX, STINKY_KEEP, hasEpilogueSideEffects, \
  TDM_TILE_DIM_MAX, storesFinalD, tdmStoreBlocks, tdmStoreLdsBytes, tdmStoreLdsSplit, tdmStoreModifiers, \
  tdmStoreRejectReason, tdmStoreVectorRejectReason
from tensilelite.SolutionStructs.Naming import getKernelNameMin

from emit_test_helpers import assert_every_vector_preload_barriers, emit_asm
from toolchain_test_helpers import copy_isa_info_map

pytestmark = pytest.mark.unit

# The ProblemType keys an override of _state may name; every other override goes to the solution.
_PROBLEM_TYPE_KEYS = {
  "DestDataType", "ComputeDataType", "HighPrecisionAccumulate", "FusedA2A", "UseE", "Gradient",
  "UseGateResidual", "GroupedGemm", "UseBias", "UseScaleAlphaVec", "UseScaleAB", "UseScaleCD",
  "ActivationType", "OutputAmaxD", "OutputPeerD",
}


def _state(**overrides):
  """The keys tdmStoreRejectReason, tdmStoreVectorRejectReason and the ExecutionPolicy helpers they call read, for a
  TDMStoreInst=True solution inside the supported configurations, with `overrides` applied."""
  problemType = {
    "DestDataType": DataType("b"),
    "ComputeDataType": DataType("s"),
    "HighPrecisionAccumulate": True,
    "FusedA2A": "Disabled",
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
    "TDMStoreInst": True,
    "ISA": (12, 5, 0),
    "TileProcessingStrategy": "None",
    "WorkAssignment": "StaticGrid",
    "StreamKAtomic": 0,
    "PrefetchAcrossPersistent": 0,
    "ReuseAcrossPersistent": 0,
    "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "AdaptiveGemmGSUA": 0,
    "UseSubtileImpl": False,
    "StoreRemapVectorWidth": 0,
    "CompactLoopStore": False,
    "MacroTile0": 256,
    "MacroTile1": 256,
    "NumThreads": 128,
    "WavefrontSize": 32,
    "EnableMatrixInstruction": True,
    "_VectorStore": 1,
    "StoreVectorWidth": 8,
    "VectorWidthA": 1,
    "AdaptiveGemm": 0,
    "LocalSplitU": 1,
    "PackedC0IndicesX": [0],
    "PackedC1IndicesX": [1],
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
  assert tdmStoreRejectReason(_state(TDMStoreInst=False, ISA=(9, 5, 0), GlobalSplitU=4)) is None


_OUTPUT = "TDMStoreInst supports fp32 D, fp64, complex float and complex double D from accumulators of their type, " \
  "int32 and int8 D from int32 accumulators, and bf16, fp16, fp8 and bf8 D converted from fp32 accumulators"
_STREAMK = {"TileProcessingStrategy": "StreamK", "GlobalSplitU": 0}
_TILE = "TDMStoreInst requires the staged tile's tile_dim0 (data_size units) and tile_dim1 (columns per wave) to fit 16 bits"
_ZGEMM_STATE = {"DestDataType": DataType("z"), "ComputeDataType": DataType("z"), "HighPrecisionAccumulate": False}


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
  ({"FusedA2A": "GemmA2A"}, "TDMStoreInst does not support FusedA2A"),
  ({"OutputPeerD": True}, "TDMStoreInst does not support OutputPeerD"),
  (dict(_STREAMK, StreamKAtomic=1),
   "TDMStoreInst does not support StreamKAtomic: the kernel adds its tiles to D atomically instead of storing them"),
  ({"PackedC0IndicesX": [0, 2]}, "TDMStoreInst requires one free index per operand, not packed free indices"),
  ({"PackedC1IndicesX": [1, 3]}, "TDMStoreInst requires one free index per operand, not packed free indices"),
  ({"MacroTile0": TDM_TILE_DIM_MAX + 1}, _TILE),
  (dict(_ZGEMM_STATE, MacroTile0=TDM_TILE_DIM_MAX // 2 + 1), _TILE),
  ({"MacroTile1": 4 * TDM_TILE_DIM_MAX + 1}, _TILE),
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


@pytest.mark.parametrize("overrides", [
  {"MacroTile0": TDM_TILE_DIM_MAX},
  dict(_ZGEMM_STATE, MacroTile0=TDM_TILE_DIM_MAX // 2),
  {"MacroTile1": 4 * TDM_TILE_DIM_MAX},
], ids=["mt0", "mt0_complex_double", "columns"])
def test_accepts_the_largest_tile_the_descriptor_holds(overrides):
  """tile_dim0 holds MacroTile0 in data_size units, two per complex double element, and tile_dim1 the columns of a
  wave, MacroTile1 over the four waves rounded up: up to 0xFFFF each."""
  assert tdmStoreRejectReason(_state(**overrides)) is None


def test_tdm_descriptor_refuses_a_tile_the_16_bit_field_cannot_hold():
  """tile_dim0 and tile_dim1 are 16-bit and a tile_dim0 of 0 makes the transfer a NOP, so the descriptor builder
  refuses such a tile instead of masking it into another one."""
  from tensilelite.Components.TensorDataMover import TensorDataMoverStore
  tdm = TensorDataMoverStore([0])
  for setter in (tdm.setTensorTile0, tdm.setTensorTile1):
    for tile in (0, TDM_TILE_DIM_MAX + 1):
      with pytest.raises(AssertionError):
        setter(8, tile)


# The store vector widths of validParameters whose D vectors emitTDMStoreStaging would not write whole, by D type: 3
# and 6 for every type but complex double, whose 96 bytes at width 6 are three 32-byte stores.
_UNCOVERED = {"f8": {3, 6}, "b": {3, 6}, "s": {3, 6}, "d": {3, 6}, "z": {3}}


@pytest.mark.parametrize("dest", sorted(_UNCOVERED))
@pytest.mark.parametrize("svw", [w for w in validParameters["StoreVectorWidth"] if w > 0])
def test_rejects_d_vectors_the_staging_does_not_cover(dest, svw):
  """emitTDMStoreStaging writes a vector as stores of min(bytes, 32), so it covers vectors of 1, 2, 4, 8, 16 and
  multiples of 32 bytes only: 3 bytes of fp8 at width 3, 12 of bf16 at width 6, and 48 of fp64 at width 6 or of
  complex double at width 3 are rejected."""
  bps = svw * int(DataType(dest).numBytes())
  reason = tdmStoreVectorRejectReason(_state(DestDataType=DataType(dest), StoreVectorWidth=svw))
  if svw in _UNCOVERED[dest]:
    assert reason == "TDMStoreInst requires D vectors of 1, 2, 4, 8, 16 or a multiple of 32 bytes, not %u " \
                     "(vector width %u)" % (bps, svw)
  else:
    assert reason is None, (dest, svw)


@pytest.mark.parametrize("adaptive, lsu, rejected", [(1, 1, True), (0, 1, False), (1, 2, False)])
def test_side_effect_kernels_check_every_width_of_their_cascade(adaptive, lsu, rejected):
  """With per-element outputs beyond C and D every block stages at its own width: complex double at store vector width
  6 stages 96 bytes, which the staging covers, but AdaptiveGemm=1 also builds the block of width 3, whose 48 bytes it
  does not. AdaptiveGemm=0 and LocalSplitU > 1 keep only the widths 6 and 1; without E only width 6 stages."""
  state = _state(DestDataType=DataType("z"), StoreVectorWidth=6, UseE=True, AdaptiveGemm=adaptive, LocalSplitU=lsu)
  assert (tdmStoreVectorRejectReason(state) is not None) == rejected
  assert tdmStoreVectorRejectReason(dict(state, ProblemType=dict(state["ProblemType"], UseE=False))) is None


@pytest.mark.parametrize("overrides, rejected", [
  ({"_VectorStore": 0, "StoreVectorWidth": 6}, False),
  ({"EnableMatrixInstruction": False, "VectorWidthA": 3}, True),
  ({"EnableMatrixInstruction": False, "VectorWidthA": 4, "StoreVectorWidth": 6}, False),
  ({"TDMStoreInst": False, "StoreVectorWidth": 6}, False),
], ids=["no_vector_store", "valu_vwa3", "valu_vwa4", "off"])
def test_the_vector_check_follows_the_first_store_width(overrides, rejected):
  """The cascade starts at 1 without _VectorStore and at VectorWidthA on a VALU kernel, whatever StoreVectorWidth is;
  with TDMStoreInst off nothing is checked."""
  assert (tdmStoreVectorRejectReason(_state(**overrides)) is not None) == rejected


def test_the_switch_matches_the_valid_values_and_on_covers_every_path():
  """TDMStoreInst is False or True only, and True sends every store path of a final-D store through
  the TDM block, so the epilogue keeps one store path and splits only on beta / non-beta."""
  assert validParameters["TDMStoreInst"] == [False, True]
  # bool == int in python, so the list comparison alone passes against [0, 1] too. The validator
  # keys on type() rather than isinstance, so the type is the thing that has to be pinned.
  assert [type(v) for v in validParameters["TDMStoreInst"]] == [bool, bool]
  cascade = [8, 4, 2, 1]
  assert _blocks(False, vectorWidths=cascade) == _blocks(False, edge=False, vectorWidths=cascade) == set()
  assert _blocks(True, vectorWidths=cascade) == set(cascade)
  assert _blocks(True, edge=False, vectorWidths=cascade) == {8}


def _blocks(tdm, edge=True, atomic=False, accumulation=None, vectorWidths=(8, 1), mpm=0):
  return tdmStoreBlocks({"TDMStoreInst": tdm, "_GlobalAccumulation": accumulation, "MbskPrefetchMethod": mpm},
                        edge, atomic, list(vectorWidths))


# The store blocks generateBetaModules gets with StoreVectorWidth 8: AdaptiveGemm=0 keeps the
# widest and the narrowest, AdaptiveGemm=1 every width.
_VECTOR_WIDTHS = {0: (8, 1), 1: (8, 4, 2, 1)}


@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("svw", [1, 2, 4, 8])
def test_the_switch_picks_blocks_of_the_store_width_cascade(svw, adaptive):
  """TDMStoreInst on picks every block of the store-width cascade: with edge blocks the masked ones
  narrower than the store vector width, the narrowest one (EdgeM) and the masked block of the store
  vector width (EdgeN) - which together are the whole cascade - and without edge blocks the unmasked
  one (NonEdge). TDMStoreInst off picks none, so every block keeps its buffer_store."""
  cascade = [svw, 1] if adaptive == 0 else [w for w in (8, 4, 2, 1) if w <= svw]
  edgeM = {w for w in cascade if w < svw} | {min(cascade)}
  expected = {0: set(), 1: edgeM | {svw}}
  for tdm in (False, True):
    assert _blocks(tdm, vectorWidths=cascade) == expected[tdm], (tdm, cascade)
    assert _blocks(tdm, edge=False, vectorWidths=cascade[:1]) == ({svw} if tdm else set()), tdm
  # With edge blocks the picked set is the whole cascade, at every store vector width.
  assert _blocks(1, vectorWidths=cascade) == set(cascade)


@pytest.mark.parametrize("accumulation, atomic", [
  ("MultipleBuffer", False),
  ("SingleBuffer", True),
  (None, True),
])
def test_partial_and_atomic_stores_keep_buffer_store(accumulation, atomic):
  assert _blocks(1, atomic=atomic, accumulation=accumulation) == set()
  assert _blocks(1, edge=False, vectorWidths=(8,), atomic=atomic, accumulation=accumulation) == set()


@pytest.mark.parametrize("accumulation", [None, "PartialsBuffer", "MultipleBuffer", "MultipleBufferSingleKernel",
                                          "SingleBuffer"])
@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("mpm", [0, 1])
def test_stores_final_d_predicate(accumulation, atomic, mpm):
  """A block writes D's final elements unless it is atomic or a GSU workspace store, the reduced tile of
  MultipleBufferSingleKernel with MbskPrefetchMethod 0 only, and tdmStoreBlocks lets such a block store through the
  TDM block."""
  final = not atomic and (accumulation in (None, "PartialsBuffer")
                          or (accumulation == "MultipleBufferSingleKernel" and mpm == 0))
  assert storesFinalD({"_GlobalAccumulation": accumulation, "MbskPrefetchMethod": mpm}, atomic) == final
  assert bool(_blocks(True, atomic=atomic, accumulation=accumulation, mpm=mpm)) == final


def test_streamk_final_store_uses_the_tdm_block():
  assert _blocks(1, accumulation="PartialsBuffer") == {8, 1}


def test_mbsk_reduced_tile_uses_the_tdm_block():
  """MultipleBufferSingleKernel stores the reduced tile to D: the blocks of a final store with MbskPrefetchMethod
  0, where one workgroup reduces the tile at a GSU of at most 2, and none with MbskPrefetchMethod 1."""
  assert _blocks(1, accumulation="MultipleBufferSingleKernel") == {8, 1}
  assert _blocks(1, accumulation="MultipleBufferSingleKernel", mpm=1) == set()


@pytest.mark.parametrize("dest, bpe", [("h", 2), ("s", 4)])
@pytest.mark.parametrize("tdm", (0, 1))
def test_lds_bytes_of_the_staged_tile(tdm, dest, bpe):
  state = {"TDMStoreInst": tdm, "MacroTile0": 256, "MacroTile1": 128, "NumThreads": 128, "WavefrontSize": 32,
           "ProblemType": {"DestDataType": DataType(dest)}}
  assert tdmStoreLdsBytes(state) == (256 * 128 * bpe if tdm else 0)


def test_lds_bytes_cover_the_columns_of_every_wave():
  """With MacroTile1 not a multiple of the waves, the staged tile holds MacroTile1 rounded up to a multiple of the
  waves: 128 columns over three waves are 3 x 43."""
  tdm = 1
  state = {"TDMStoreInst": tdm, "MacroTile0": 256, "MacroTile1": 128, "NumThreads": 96, "WavefrontSize": 32,
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


def test_kernel_name_token_only_when_enabled():
  """TDMStoreInst=False keeps every existing kernel name; TDMStoreInst=True adds the TDMSI1 token, which
  stays apart from the TDMS0 or TDMS1 of TDMSplit."""
  tdm = 1
  off = getKernelNameMin(_kernel(0), False)
  on = getKernelNameMin(_kernel(tdm), False)
  assert "TDMSI" not in off
  assert on.replace("_TDMSI%u" % tdm, "", 1) == off
  split, both = _kernel(0), _kernel(tdm)
  split["TDMSplit"] = both["TDMSplit"] = True
  split, both = getKernelNameMin(split, False), getKernelNameMin(both, False)
  assert "_TDMS1_" in split and "TDMSI" not in split
  assert "_TDMS1_" in both and "_TDMSI%u_" % tdm in both


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


# Off has no TDM block, so it is emitted with the default TDMInst only. TDMInst is the load-side
# instruction count -- a different parameter despite the similar name -- and these pairs cross the two.
_TDM_TDMINST = [(False, 3)] + [(True, tdmInst) for tdmInst in (0, 3)]


@pytest.mark.parametrize("sourceSwap", [False, True])
@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("tdm, tdmInst", _TDM_TDMINST)
def test_emitted_paths_of_the_switch(gp_gfx1250, gfx1250_iim, assembler, capsys, tdm, tdmInst, adaptive,
                                     sourceSwap):
  """With the switch off every block keeps its buffer_store. With it on, every path of a final-D store --
  EdgeM, EdgeN and NonEdge -- stores through the TDM block, which reads C from the staged tile, so no block
  of the direct store keeps a buffer_store of D; the GSU>1 copy of the partials still keeps buffer_store.
  The block drains tensor_load_to_lds before it reuses the main-loop LDS only when A or B is a TDM load."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, TDMInst=tdmInst, AdaptiveGemm=adaptive,
                       SourceSwap=sourceSwap)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 8
  cascade = list(_VECTOR_WIDTHS[adaptive])
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["then"], reverse=True) == cascade
    assert sorted(direct["tdm"], reverse=True) == sorted(_blocks(tdm, vectorWidths=cascade), reverse=True)
    assert direct["nonEdge"] and direct["nonEdgeTdm"] == bool(tdm), (beta, direct)
    assert bool(direct["nonEdgeBuffer"]) == bool(direct["buffer"]) == (not tdm), (beta, direct)
  partials = stores[("MB", "0")]
  assert sorted(partials["then"], reverse=True) == cascade and partials["nonEdge"] and not partials["tdm"]
  assert not partials["nonEdgeTdm"]
  if tdm:
    block = _tdmBlock(asm, "1", {1: 1, 2: 8, 3: 8}[tdm])
    assert not re.search(r"^\s*buffer_load\S* [^\n]*s\[sgprSrdC:", block, re.M)
    loads = list(re.finditer(r"^\s*tensor_load_to_lds ", block, re.M))
    assert len(loads) == 1 and re.search(r"^\s*ds_load\S* ", block[loads[0].end():], re.M)
    drain = re.search(r"^\s*s_wait_tensorcnt[^\n]*drain tensor_load_to_lds", asm, re.M)
    assert bool(drain) == (tdmInst == 3)


@pytest.mark.parametrize("dest", ["B", "H"])
@pytest.mark.parametrize("tdm", (0, 1))
def test_optnll_store_goes_through_the_tdm_block(gp_gfx1250, gfx1250_iim, assembler, capsys, tdm, dest):
  """The OptNLL store is an unmasked full-tile store of D, a NonEdge path without edge blocks, so with
  TDMStoreInst on it goes through the TDM block whatever the alignment of the columns, and with it off
  it keeps buffer_store."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, ProblemType=_dest(dest))
  assert sol.get("Valid"), capsys.readouterr().out
  optNLL = _stores(asm)[("OptNLL_MB", "0")]
  assert optNLL["then"] == [8] and not optNLL["nonEdge"]
  assert optNLL["tdm"] == ([8] if tdm else [])
  assert bool(optNLL["buffer"]) == (not tdm), optNLL


@pytest.mark.parametrize("adaptive", [0, 1])
def test_turning_it_on_only_adds_the_tdm_blocks(gp_gfx1250, gfx1250_iim, assembler, capsys, adaptive):
  """TDMStoreInst adds, drops or moves no global-write label or branch to one, so it never checks the
  alignment of the columns, and a kernel with it on equals its switched-off twin outside the blocks it
  stores through the TDM block."""
  asms = {}
  for tdm in (False, True):
    sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, AdaptiveGemm=adaptive)
    assert sol.get("Valid"), capsys.readouterr().out
    asms[tdm] = asm
  for tdm, asm in asms.items():
    assert _skeleton(asm) == _skeleton(asms[0]), tdm
  # A byte-level differential of the untouched code is not available here: off and on move every store
  # block to the TDM block, so there is no region that has to stay identical. _skeleton above is the
  # property that does survive, and is the one this test exists for: the label and branch structure of
  # the global write is the same whether the switch is off or on.


@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("tdm", (0, 1))
def test_sourceswap_store_vector_width_2(gp_gfx1250, gfx1250_iim, assembler, capsys, tdm, adaptive):
  """With SourceSwap the store vector width is VectorWidthA: MIWaveTile0=2 gives 2, whose only
  narrower path is VW1, and TDMStoreInst sends both the VW2 and the VW1 path through the TDM block."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 2, 8, 4, 1], TDMStoreInst=tdm,
                       AdaptiveGemm=adaptive, SourceSwap=True)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 2
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["then"], reverse=True) == [2, 1]
    assert sorted(direct["tdm"], reverse=True) == ([2, 1] if tdm else [])


_FP16 = {"DataType": "H", "MacDataTypeA": "H", "MacDataTypeB": "H", "DestDataType": "H"}


def _dest(dest, **problemType):
  """The ProblemType of bf16 ("B") or fp16 ("H") inputs and D, converted from fp32 accumulators."""
  return dict(_FP16 if dest == "H" else {}, **problemType)


def test_fp16_c_is_read_back_from_the_staged_tile(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """fp16 C comes back from the staged tile in 16-byte vectors at StoreVectorWidth 8."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, ProblemType=_FP16)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 8
  block = _tdmBlock(asm, "1", 8)
  load = re.search(r"^\s*tensor_load_to_lds ", block, re.M)
  assert load and re.search(r"^\s*ds_load_b128 ", block[load.end():], re.M)


def test_fp16_with_beta_at_store_vector_width_1(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """At StoreVectorWidth 1 the switch stores the one masked block through the TDM block, which reads each
  C element back into its own VGPR: an even one with ds_load_u16, an odd one with ds_load_u16_d16_hi
  into the high half, where readInput puts it."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, StoreVectorWidth=1, ProblemType=_FP16)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 1
  loads = re.findall(r"^\s*(ds_load_\w+) v\S+, .*// load C element \((\d+), (\d+)\)", _tdmBlock(asm, "1", 1), re.M)
  assert {mnemonic for mnemonic, _, _ in loads} == {"ds_load_u16", "ds_load_u16_d16_hi"}
  for mnemonic, coord0, coord1 in loads:
    assert (mnemonic == "ds_load_u16_d16_hi") == (int(coord0) % 2 == 1), (mnemonic, coord0, coord1)


def test_fp16_d_stages_each_element_from_its_own_register(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """fp16 D converted from fp32 accumulators at StoreVectorWidth 1: every element, odd or even, is in the low half
  of its own VGPR, where the buffer_store path stores it from, so the TDM block stages each with one ds_store_b16
  of a single register, none with the d16_hi store."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, StoreVectorWidth=1, ProblemType=_FP16)
  assert sol.get("Valid"), capsys.readouterr().out
  staged = re.findall(r"^\s*(ds_store_\w+) v\S+, (v\S+) .*// stage D element \((\d+), (\d+)\)", _tdmBlock(asm, "0", 1),
                      re.M)
  assert staged and {mnemonic for mnemonic, _, _, _ in staged} == {"ds_store_b16"}
  assert all(re.fullmatch(r"v\[vgprValuC\+\d+(-\d+)?\]", src) for _, src, _, _ in staged), staged[:2]
  assert {int(coord0) % 2 for _, _, coord0, _ in staged} == {0, 1}


def test_store_vector_width_4(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """At StoreVectorWidth 4 the TDM block stages 8-byte vectors in every block the switch picks."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, StoreVectorWidth=4)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 4
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["tdm"]) == sorted(_blocks(tdm, vectorWidths=direct["then"]))
    for slot in direct["tdm"]:
      assert re.search(r"^\s*ds_store_b64 ", _tdmBlock(asm, beta, slot), re.M)


def test_store_vector_width_16_splits_32_byte_vectors(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """bf16 at StoreVectorWidth 16 (VectorWidthA 2) puts 32-byte vectors through the TDM block, wider than any
  DS instruction, so it stages each D vector, and with beta != 0 reads each C vector back, with two
  16-byte DS ops."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, StoreVectorWidth=16, VectorWidthA=2)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 16
  for beta, tag, op in (("0", "stage D element", "ds_store_b128"), ("1", "load C element", "ds_load_b128")):
    lines = [line for line in _tdmBlock(asm, beta, {1: 1, 2: 16, 3: 16}[tdm]).splitlines() if tag in line]
    assert lines and {line.split()[0] for line in lines} == {op}, (beta, lines[:2])
    coords = Counter(re.search(r"%s \((\d+), (\d+)\)" % tag, line).groups() for line in lines)
    assert set(coords.values()) == {2}, (beta, lines[:4])


def test_wide_vectors_split_into_16_byte_ds_ops(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """fp32 D from bf16 inputs at StoreVectorWidth 8 stages 32-byte D vectors, which dsStore writes as two 16-byte
  stores, and with beta != 0 reads each C vector back with two 16-byte DS loads. The staged tile is 4 bytes per
  element."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, StoreVectorWidth=8, ProblemType={"DestDataType": "S"})
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"] * 4
  for beta, tag, op in (("0", "stage D element", "ds_store_b128"), ("1", "load C element", "ds_load_b128")):
    lines = [line for line in _tdmBlock(asm, beta, {1: 1, 2: 8, 3: 8}[tdm]).splitlines() if tag in line]
    assert lines and {line.split()[0] for line in lines} == {op}, (beta, lines[:2])
    coords = Counter(re.search(r"%s \((\d+), (\d+)\)" % tag, line).groups() for line in lines)
    assert set(coords.values()) == {2}, (beta, lines[:4])


_INT32 = {"DataType": "I8", "MacDataTypeA": "I8", "MacDataTypeB": "I8", "DestDataType": "I",
          "ComputeDataType": "I"}


def test_int32_d_stages_4_byte_elements(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """int32 D from int32 accumulators (i8 inputs) is staged like fp32 D: at store vector width 1 every D element is
  one 4-byte DS store into a staged tile of 4 bytes per element, and with beta != 0 every C element is read back
  with one 4-byte DS load."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 64, 1, 1, 2, 2, 2, 2], TDMStoreInst=tdm,
                       StoreVectorWidth=1, ProblemType=_INT32)
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"] * 4
  assert "tensor_store_from_lds" in asm
  for tag, op in (("stage D element", "ds_store_b32"), ("load C element", "ds_load_b32")):
    assert {line.split()[0] for line in asm.splitlines() if tag in line} == {op}, tag


@pytest.mark.parametrize("dest, compute", [("F8", "S"), ("B8", "S"), ("I8", "I")])
def test_one_byte_d_stages_with_ds_store_b8_at_vw1(gp_gfx1250, gfx1250_iim, assembler, capsys, dest, compute):
  """fp8 and bf8 D from fp32 accumulators and int8 D from int32 accumulators at store vector width 1: every D
  element is one ds_store_b8 into a staged tile of 1 byte per element, and with beta != 0 every C element is read
  back with one ds_load_u8, into the low byte of its VGPR, where readInput's byte load puts it."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 64, 1, 1, 2, 2, 2, 2], TDMStoreInst=tdm,
                       StoreVectorWidth=1, ProblemType={"DataType": dest, "MacDataTypeA": dest, "MacDataTypeB": dest,
                                                       "DestDataType": dest, "ComputeDataType": compute})
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"]
  assert "tensor_store_from_lds" in asm
  for tag, op in (("stage D element", "ds_store_b8"), ("load C element", "ds_load_u8")):
    assert {line.split()[0] for line in asm.splitlines() if tag in line} == {op}, tag


def _fp64_params(**overrides):
  """An fp64 VALU kernel shaped like those of f64_gfx1250.yaml: ThreadTile 4x4 over one wave."""
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
def test_eight_byte_d_stages_two_registers_per_element(gp_gfx1250, gfx1250_iim, assembler, capsys, kind):
  """fp64 D on a VALU kernel and complex float D on a WMMA kernel: at store vector width 1 every D element is one
  ds_store_b64 of its register pair, ValuC + 2*sumIdx as addStore stores it, into a staged tile of 8 bytes per
  element, and with beta != 0 every C element is read back with one ds_load_b64."""
  tdm = 1
  if kind == "fp64":
    sol, asm = emit_asm(gfx1250_iim, assembler, _fp64_params(TDMStoreInst=tdm, StoreVectorWidth=1))
  else:
    sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 4, 1, 1, 2, 2, 2, 2], TDMStoreInst=tdm, DepthU=16,
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


def test_complex_double_descriptor_uses_8_byte_units(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """Complex double D on a VALU kernel: TDM data_size is at most 8 bytes, so every descriptor of D and C counts
  8-byte units, two per element, with data_size 3 and tensor_dim0, tile0 and the stride doubled. At store vector
  width 1 every D element is one ds_store_b128 of its four registers, ValuC + 4*sumIdx as addStore stores it, into a
  staged tile of 16 bytes per element, and with beta != 0 every C element is read back with one ds_load_b128."""
  tdm = 1
  sol, asm = emit_asm(gfx1250_iim, assembler, _fp64_params(TDMStoreInst=tdm, StoreVectorWidth=1, ProblemType=_ZGEMM))
  assert sol.get("Valid"), capsys.readouterr().out
  assert tdmStoreLdsBytes(sol) == sol["MacroTile0"] * sol["MacroTile1"] * 16
  descriptors = asm.count("tensor_dim0 = SizeI - wg0*MT0")
  assert descriptors and set(re.findall(r"Set data_size to (\d+)", asm)) == {"3"}
  assert re.findall(r"set tile0 to (\d+)", asm) == [str(2 * sol["MacroTile0"])] * descriptors
  assert asm.count("tensor_dim0 in 8-byte units") == descriptors
  assert asm.count("stride in data_size units") == descriptors
  for tag, op in (("stage D element", "ds_store_b128"), ("load C element", "ds_load_b128")):
    assert {line.split()[0] for line in asm.splitlines() if tag in line} == {op}, tag
  staged = [re.search(r"v\[vgprValuC\+(\d+):vgprValuC\+(\d+)\+3\]", line) for line in asm.splitlines()
            if "stage D element" in line]
  assert staged and all(m and m.group(1) == m.group(2) and int(m.group(1)) % 4 == 0 for m in staged), staged[:3]


def test_complex_double_stride_fills_the_48_bit_field(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """tensor_dim0_stride is 48 bits, group1+5 and the low half of group1+6. A complex double stride counts 8-byte units,
  twice the 32-bit StrideD or StrideC, so it can need bit 32: every descriptor of D and C doubles the stride into
  group1+5, and the next instruction writes its carry to group1+6."""
  tdm = 1
  sol, asm = emit_asm(gfx1250_iim, assembler, _fp64_params(TDMStoreInst=tdm, StoreVectorWidth=1, ProblemType=_ZGEMM))
  assert sol.get("Valid"), capsys.readouterr().out
  lines = asm.splitlines()
  doubles = [i for i, line in enumerate(lines) if "stride in data_size units" in line]
  assert doubles and len(doubles) == asm.count("tensor_dim0 = SizeI - wg0*MT0")
  for i in doubles:
    add = re.match(r"s_add_u32 s\[?(\d+)\]?, (s\[sgprStride[CD]\w*\]), (s\[sgprStride[CD]\w*\])$", _code(lines[i]))
    assert add and add.group(2) == add.group(3), lines[i]
    transfer = next(_code(line) for line in lines[i:]
                    if _code(line).startswith(("tensor_store_from_lds ", "tensor_load_to_lds ")))
    group1 = int(re.findall(r"s\[(\d+):\d+\]", transfer)[1])
    assert int(add.group(1)) == group1 + 5, (lines[i], transfer)
    carry = next(_code(line) for line in lines[i + 1:] if _code(line))
    assert re.match(r"s_addc_u32 s\[?%u\]?, 0, 0$" % (group1 + 6), carry), carry


@pytest.mark.parametrize("algorithm, aga", _GSU_ALGORITHMS)
def test_dynamic_gsu_with_epilogue_vectors(gp_gfx1250, gfx1250_iim, assembler, capsys, algorithm, aga):
  """GlobalSplitU=-1 with bias and ScaleAlphaVec, for every algorithm and AdaptiveGemmGSUA: the GSU == 1 branch,
  the OptNLL store and, with MultipleBufferSingleKernel and MbskPrefetchMethod 0, the TDM copy of the reduced tile
  go through the TDM block, beside the buffer_store copy of the reduced tile; the partials and atomics of the
  GSU > 1 branch keep buffer_store or buffer atomics. AtomicDest runs without the vectors, which develop rejects with it (the
  epilogue would run per GSU slice), and with the even AssertFree0ElementMultiple of packed bf16 atomics."""
  extra = {"AssertFree0ElementMultiple": 2} if algorithm == "AtomicDest" else \
    {"ProblemType": {"UseBias": 1, "UseScaleAlphaVec": 1}}
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, GlobalSplitU=-1, GlobalSplitUAlgorithm=algorithm,
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


def test_mbsk_reduced_tile_with_the_switch_on(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """Static GlobalSplitU=2 with MultipleBufferSingleKernel and MbskPrefetchMethod 0: the TDM copy of the reduced
  tile takes the TDM blocks the GSU == 1 store takes at the tdm, OptNLL_MBSK's being NonEdge, with its descriptor
  from SrdTD and D staged at the size of D; the buffer_store copy, which a GSU above 2 takes, stores every block
  with buffer_store to SrdTD, and one GSU check picks the copy. OptNLL_MBSK, which runs at GSU 1 only, without a
  reduction, keeps its store."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, GlobalSplitU=2,
                       GlobalSplitUAlgorithm="MultipleBufferSingleKernel", MbskPrefetchMethod=0)
  assert sol.get("Valid"), capsys.readouterr().out
  stores = _stores(asm)
  for beta in "01":
    reduced, direct, copy = stores[("MBSK", beta)], stores[("GSU1", beta)], stores[("MBSKBuffer", beta)]
    assert (reduced["tdm"], reduced["nonEdgeTdm"], reduced["buffer"]) == (direct["tdm"], direct["nonEdgeTdm"], 0)
    assert copy["then"] == reduced["then"] and copy["nonEdge"] and not copy["tdm"] and not copy["nonEdgeTdm"]
    assert copy["stores"] and not copy["buffer"]
  assert stores[("OptNLL_MBSK", "0")]["tdm"] == ([8] if tdm else [])
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
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, GlobalSplitU=2,
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


def test_e_kernel_tdm_blocks_keep_the_masked_side_effects(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """The forward E kernel of bbs_bias_gelu_aux_cls_gfx1250.yaml: each TDM block stages D in place of its stores
  and keeps the E stores of the switched-off block; an edge TDM block is that edge block, with its vector width, its masks
  and its C loads, so no element outside D reaches E or the epilogue."""
  tdm = 1
  _, base = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=False, ProblemType=_AUX_E, **_AUX_E_SOLUTION)
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, ProblemType=_AUX_E, **_AUX_E_SOLUTION)
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


@pytest.mark.parametrize("problem", ["aux_e", "pure_gemm"])
def test_compact_loop_store_tdm_block_outside_the_loop(gp_gfx1250, gfx1250_iim, assembler, capsys, problem):
  """The CompactLoopStore E kernel of bbs_bias_gelu_aux_cls_gfx1250.yaml, and a pure GEMM kernel with CompactLoopStore
  on: each TDM block runs its stores in the CompactLoopStore loop. It sets up the
  staged tile (and loads C) before the loop label, stages every element inside the loop, steps the staged tile's base
  once an iteration when the loop runs more than once, and stores the tile once, after the loop exits; no D
  buffer_store is left in any TDM block."""
  tdm = 1
  setup = dict(_AUX_E_SOLUTION, ProblemType=_AUX_E) if problem == "aux_e" else {}
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, **dict(setup, CompactLoopStore=True))
  assert sol.get("Valid") and sol["CompactLoopStore"], capsys.readouterr().out
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
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, SourceSwap=True, StoreRemapVectorWidth=-1)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreRemapVectorWidth"] == 0
  assert "tensor_store_from_lds " in asm
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 4, 4, 2, 2], TDMStoreInst=True, SourceSwap=False,
                       StoreRemapVectorWidth=-1, VectorWidthA=1, VectorWidthB=1)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreRemapVectorWidth"] > 0 and "tensor_store_from_lds " in asm


_REMAP = ("storeRemap lw", "storeRemap lr", "shift storeRemap coord1")


@pytest.mark.parametrize("srvw, waveTile", [(4, 4)])
def test_srvw_tdm_blocks_skip_the_remap(gp_gfx1250, gfx1250_iim, assembler, capsys, srvw, waveTile):
  """With StoreRemapVectorWidth 4 (MT128x128) a TDM block stages D itself: it has no remap local write,
  read-back or row shift, while the blocks that keep buffer_store still remap through LDS. An explicit remap
  width needs SourceSwap 0, and one wave's write must cover an M column."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, waveTile, waveTile, 2, 2], TDMStoreInst=tdm,
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
  if not tdm:
    assert all(tag in asm for tag in _REMAP[:2])


def test_sav_device_scalar_bit_is_along_m(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """UseScaleAlphaVec 5 is the vector along M with the device-scalar bit, which only changes how the vector
  preload selects alpha, so a TDMStoreInst kernel stores through the TDM block as with UseScaleAlphaVec 1."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, ProblemType={"UseBias": 1, "UseScaleAlphaVec": 5})
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["ProblemType"]["UseScaleAlphaVec"] == 5 and "tensor_store_from_lds " in asm


@pytest.mark.parametrize("dest", ["B", "H"])
@pytest.mark.parametrize("useBias, factorDims", [(2, {"1"}), (3, {"0", "1"})], ids=["along_n", "along_m_or_n"])
def test_both_factor_dims_use_the_tdm_block(gp_gfx1250, gfx1250_iim, assembler, capsys, useBias, factorDims, dest):
  """Bias and ScaleAlphaVec along N (UseBias 2), or along M or N as the kernel argument picks (UseBias 3):
  generateBetaModules builds the store tree of each factor dim, and with the switch on each of them stores through
  the TDM block."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True,
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
def test_mt1_not_a_multiple_of_the_waves(gp_gfx1250, gfx1250_iim, assembler, capsys, mi):
  """MacroTile1 48 and 80 over 32 waves (MIWaveGroup [8, 1] with LocalSplitU 4): each wave stores
  ceil(MacroTile1/32) columns, so the columns of the last waves reach past the tile, and at 80 columns one wave's
  columns straddle its end. Every TDM block ends tensor_dim1 at the column bound min(SizeJ, (wg1+1)*MT1),
  against which each wave tests its skip, and the staged tile is reserved for the columns of every wave. With
  MacroTile1 a multiple of the waves there is no column bound."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=mi, TDMStoreInst=tdm, WorkGroup=[16, 16, 4], DepthU=128)
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
  if tdm:
    _, even = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm)
    assert "column bound" not in even and "tensor_dim1 = SizeJ - first column" in even


@pytest.mark.parametrize("dest", ["B", "H"])
def test_scale_cd_scales_through_registers(gp_gfx1250, gfx1250_iim, assembler, capsys, dest):
  """ScaleC and ScaleD are scalars the kernel loads from its arguments into SGPRs: the TDM block of a TDMStoreInst
  kernel multiplies its elements by ScaleD, as the buffer_store path does."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, ProblemType=_dest(dest, UseScaleCD=True))
  assert sol.get("Valid"), capsys.readouterr().out
  assert "AddressScaleC" in asm and "AddressScaleD" in asm
  assert re.search(r"^\s*v_\w*mul\w* [^\n]*sgprScaleD", _tdmBlock(asm, "1", 8), re.M)


@pytest.mark.parametrize("dest", ["B", "H"])
def test_grouped_gemm_generates(gp_gfx1250, gfx1250_iim, assembler, capsys, dest):
  """GroupedGemm sets the sizes, strides and addresses of each group in the SGPRs the TDM descriptors read."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, ProblemType=_dest(dest, GroupedGemm=True))
  assert sol.get("Valid"), capsys.readouterr().out
  assert "tensor_store_from_lds " in asm


def _cacheTokens(line):
  """The th: and scope: tokens of an instruction line, as {name: value}."""
  return dict(re.findall(r"\b(th|scope):(\S+)", _code(line)))


def test_c_load_takes_the_hints_of_c(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """The tensor_load_to_lds of C carries the th: and scope: that the buffer_load of C carries with
  TDMStoreInst off. With it on every path stores through the TDM block, so no buffer_load of C
  survives in the same kernel to compare against; the reference comes from the switched-off twin, which
  is what this pins: the cache hints of C are preserved across the change of transport."""
  hints = dict(TemporalHintC=3, NonTemporalC=2)
  solOff, asmOff = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=False, **hints)
  assert solOff.get("Valid"), capsys.readouterr().out
  bufferLoad = re.search(r"^\s*buffer_load\S* [^\n]*s\[sgprSrdC:[^\n]*", asmOff, re.M)
  assert bufferLoad, "TDMStoreInst off must still read C with buffer_load"
  expected = _cacheTokens(bufferLoad.group(0))
  assert "th" in expected

  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, **hints)
  assert sol.get("Valid"), capsys.readouterr().out
  load = re.search(r"^\s*tensor_load_to_lds [^\n]*", _tdmBlock(asm, "1", 8), re.M)
  assert load, "the TDM block must load C with tensor_load_to_lds"
  assert _cacheTokens(load.group(0)) == expected


def test_requires_tdm_in_the_assembler(gp_gfx1250, gfx1250_iim, assembler, capsys):
  from tensilelite.Common.Architectures import gfxToIsa

  noTdm = copy_isa_info_map(gfx1250_iim)
  noTdm[gfxToIsa("gfx1250")].asmCaps["HasTDM"] = False
  sol, asm = _emit_asm(noTdm, assembler, TDMStoreInst=True, TDMInst=0)
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
  from tensilelite.Components.TDMStoreInst import _stagedTileToken
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
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 4, 4, 2, 2], TDMStoreInst=True, **_PAP, **extra)
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
def test_rap_with_tdm_store_generates(gp_gfx1250, gfx1250_iim, assembler, capsys, pap):
  """ReuseAcrossPersistent keeps A in VGPRs from one DataParallel tile to the next: the RAP candidate, fp32 D
  from fp8 inputs at MT64x256, generates within the register budget at every tdm, with and without
  PrefetchAcrossPersistent."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 128, 1, 1, 2, 8, 2, 2], TDMStoreInst=tdm,
                       ProblemType=_FP8_FP32D, **dict(_RAP, PrefetchAcrossPersistent=pap))
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["ReuseAcrossPersistent"] and asm and "tensor_store_from_lds " in asm


def test_pap_oversized_tile_hits_max_lds(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """Under PrefetchAcrossPersistent the staged tile adds to the main-loop LDS, and the MaxLDS reject
  sees the sum: at DepthU 128 the main loop fits without TDMStoreInst and the sum does not."""
  sol, _ = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, DepthU=128, **_PAP)
  out = capsys.readouterr().out
  assert not sol.get("Valid") and "LDS" in out, out
  sol, _ = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=False, DepthU=128, **_PAP)
  assert sol.get("Valid"), capsys.readouterr().out


@pytest.mark.parametrize("streamk", [0, 3])
@pytest.mark.parametrize("tdm", [1])
def test_tdm_waits_survive_stinkytofu(gp_gfx1250, gfx1250_iim, assembler, capsys, tdm, streamk):
  """In the emitted assembly the drain of tensor_load_to_lds still comes before the next barrier, and
  every tensor_store_from_lds is waited for before the next barrier or the end of the kernel."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, TDMInst=3, StreamK=streamk)
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
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=mi, TDMStoreInst=True, TDMInst=3, **policy)
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
  csol, control = _emit_asm(gfx1250_iim, assembler, mi=_MT128, TDMStoreInst=True, TDMInst=0, StreamK=3)
  assert csol.get("Valid") and control, capsys.readouterr().out
  assert _FLUSH_BARRIER in control


@pytest.mark.parametrize("streamk", [4, 5], ids=["sk4", "sk5"])
def test_queue_handoff_barrier_fences_the_flush(gp_gfx1250, gfx1250_iim, assembler, capsys, streamk):
  """Without TDM loads, epilogue vectors or a cluster only the queue fences the tile handoff:
  DynamicWorkQueue.closeLoop ends every tile with a workgroup barrier before its back-edge, and the next work
  item is checked for exhaustion after it. From the end of every store section with a TDM store to that barrier
  nothing writes LDS. A StreamK 4 kernel therefore has no flush barrier. A StreamK 5 kernel runs
  StaticGrid.closeLoop, which has no queue barrier, on its static arm, so its flush keeps a barrier that the
  dynamic arm branches around."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=_MT128, TDMStoreInst=True, TDMInst=0, StreamK=streamk)
  assert sol.get("Valid"), capsys.readouterr().out
  assert "tensor_store_from_lds " in asm
  lines = asm.splitlines()
  ends = [i for i, line in enumerate(lines) if re.match(r"label_GW_End\w*:", line)]
  ends = [end for start, end in zip([0] + ends, ends) if any("tensor_store_from_lds " in x for x in lines[start:end])]
  assert ends
  for end in ends:
    handoff = next((i for i in range(end, len(lines)) if any(tag in lines[i] for tag in _HANDOFF)), None)
    assert handoff is not None and "Sync before persistent re-entry" in lines[handoff], lines[end]
    writes = [code for code in map(_code, lines[end:handoff]) if code.startswith(("ds_store", "ds_write", "tensor_load_to_lds"))]
    assert not writes, (lines[end], writes[:2])
  flushes = [i for i, line in enumerate(lines) if _FLUSH_BARRIER in line]
  if streamk == 4:
    assert not flushes
    return
  assert flushes
  for flush in flushes:
    branch = next(i for i in range(flush, -1, -1) if _code(lines[i]).startswith("s_cbranch_scc0"))
    target = _code(lines[branch]).split()[1]
    assert "TDMStoreFlushDynamicArm" in target and "static assignment selected" in lines[branch - 1], lines[branch - 1:branch + 1]
    assert any(_code(line) == target + ":" for line in lines[flush + 1:flush + 4]), target


@pytest.mark.parametrize("policy", [{"StreamK": 3, "StreamKForceDPOnly": 1}, {"StreamK": 4}], ids=["dp", "sk4"])
def test_persistent_kernels_stage_and_fence(gp_gfx1250, gfx1250_iim, assembler, capsys, policy):
  """DataParallel and StreamK 4 kernels store the tiles they complete through the TDM block, and after each
  TDM store's wait a barrier comes before the next tile's first tensor_load_to_lds."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, TDMInst=3, **policy)
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
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, StreamK=streamk,
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
  main-loop LDS. TDMStoreInst off reserves no tile."""
  for tdm in (False, True):
    sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, **deepcopy(overrides))
    assert sol.get("Valid"), capsys.readouterr().out
    offset = sol["_TDMStoreLdsOffset"]
    assert sol["_TDMStoreOverMainLoop"] == bool(tdm), tdm
    if not tdm:
      assert offset == 0 and tdmStoreLdsBytes(sol) == 0
      continue
    assert (offset > 0) == bool(overrides) and offset % 256 == 0 and offset >= sol["LdsOffsetBias"], offset
    assert sol["LdsNumBytes"] >= offset + tdmStoreLdsBytes(sol)
    assert "all waves are done with the main-loop LDS" in asm


@pytest.mark.parametrize("storeVectorWidth", [-1, 1])
def test_epilogue_vectors_keep_lds_offset_0(gp_gfx1250, gfx1250_iim, assembler, capsys, storeVectorWidth):
  """Neither setting of the switch adds an LDS offset to the addresses of the bias and ScaleAlphaVec vectors."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, StoreVectorWidth=storeVectorWidth,
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
def test_store_vector_width_1_stores_the_one_masked_block(gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                                        adaptive, vectors):
  """At store vector width 1 the store-width cascade has one masked block, the narrowest and the one of the
  store vector width at once, so TDMStoreInst stores it through the TDM block. It keeps the labels and
  branches of the switched-off kernel, which send the M-edge, N-edge-only and corner tiles all to that block, and its
  NonEdge path goes through the TDM block as well. With bias and ScaleAlphaVec the vectors keep LDS
  offset 0."""
  kernels = {}
  for tdm in (False, True):
    sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=tdm, StoreVectorWidth=1, AdaptiveGemm=adaptive,
                         **({"ProblemType": {"UseBias": 1, "UseScaleAlphaVec": 1}} if vectors else {}))
    assert sol.get("Valid"), capsys.readouterr().out
    assert sol["StoreVectorWidth"] == 1
    kernels[tdm] = asm
  for tdm, asm in kernels.items():
    assert _skeleton(asm) == _skeleton(kernels[0]), tdm
    stores = _stores(asm)
    lines = [code for code in map(_code, asm.splitlines())
             if code and not code.startswith(("s_prefetch_inst_pc_rel", "s_set_vgpr_msb"))]
    for beta in "01":
      direct = stores[("GSU1", beta)]
      assert direct["then"] == [1] and direct["nonEdge"], (tdm, beta, direct)
      assert direct["tdm"] == ([1] if tdm else []), (tdm, beta, direct)
      assert direct["nonEdgeTdm"] == bool(tdm) and bool(direct["nonEdgeBuffer"]) == (not tdm), (tdm, beta, direct)
      end = lines.index("label_GW_B%s_FD0_VW1_GSU1_NonEdgeEnd:" % beta)
      assert lines[end + 1:end + 3] == ["label_GW_B%s_FD0_VW1_GSU1_%s:" % (beta, part) for part in ("Else", "Then")]
    assert "add lds offset" not in asm


def test_solution_checks_the_d_vectors_once_their_widths_are_final(gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                                  monkeypatch):
  """Solution asks tdmStoreVectorRejectReason after it resolves StoreVectorWidth -1 and _VectorStore, and rejects the
  solution with its reason."""
  import importlib
  # The package re-exports the Solution class under the module's name, so the module comes from sys.modules.
  solution = importlib.import_module("tensilelite.SolutionStructs.Solution")
  seen = []
  monkeypatch.setattr(solution, "tdmStoreVectorRejectReason",
                      lambda state: seen.append((state["StoreVectorWidth"], state["_VectorStore"])) or "vector check probe")
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True)
  assert not sol.get("Valid") and asm is None
  assert seen and set(seen) == {(8, 1)}, seen
  assert "vector check probe" in capsys.readouterr().out


def test_solution_rejects_with_the_reason(gp_gfx1250, gfx1250_iim, assembler, capsys):
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, GlobalSplitU=2)
  assert not sol.get("Valid") and asm is None
  assert "TDMStoreInst requires GlobalSplitU=1 or -1, or MultipleBufferSingleKernel" in capsys.readouterr().out


@pytest.mark.parametrize("dest", ["B", "H"])
@pytest.mark.parametrize("sourceSwap", [False, True])
def test_local_split_u_stages_each_element_after_the_lsu_offset(gp_gfx1250, gfx1250_iim, assembler, capsys, sourceSwap, dest):
  """With LocalSplitU=2, shaped like the LSU=2 kernels of the gfx1250 bf16 TN logic, each thread stages
  the elements it owns after the LDS reduction where the LSU global write would store them: the LSU
  offset goes into coord0 (coord1 with SourceSwap) after the reduction, the TDM setup reads that VGPR
  with nothing in the block writing it first, and the staging stores address the tile from the setup."""
  tdm = 1
  sol, asm = _emit_asm(gfx1250_iim, assembler, [16, 16, 32, 1, 1, 4, 4, 1, 2], TDMStoreInst=tdm,
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
  slot = 1 if tdm == 1 else 4
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert slot in direct["tdm"] and sorted(direct["tdm"]) == sorted(_blocks(tdm, vectorWidths=direct["then"]))
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


# The DS offset field is 16-bit unsigned. An element wider than one DS instruction is staged as
# offset + piece * width, so the piece that has to fit is the LAST one. Checking only the element's own
# offset let a vector starting near the top of a 64 KiB page emit offset:65536, which the assembler rejects
# with "expected a 16-bit unsigned offset".

@pytest.mark.parametrize("offset, span", [
  (0, 0), (0, 32), (DS_OFFSET_MAX, 0), (DS_OFFSET_MAX - 32, 32),
  (0xFFE0, 32), (0xFFF0, 16), (0x10000, 0), (0x1FFE0, 32), (0x20000, 32),
  (((128 - 1) * 256 + 255) * 4, 32),   # fp32 D over 128 columns per wave: the element alone passes 64 KiB
  (((128 - 1) * 256 + 255) * 8, 32),   # fp64 D over the same tile
])
def test_lds_split_keeps_every_piece_inside_the_ds_offset_field(offset, span):
  """Every piece the caller addresses fits the 16-bit field, and the fold plus the offset still name the same
  byte of the staged tile."""
  high, low = tdmStoreLdsSplit(offset, span)
  assert high + low == offset, "the split must address the same byte"
  assert low >= 0, "a DS offset is unsigned"
  assert low + span <= DS_OFFSET_MAX, "last piece %u does not fit the 16-bit DS offset field" % (low + span)


def test_lds_split_only_folds_when_it_has_to():
  """An element that fits with its span is addressed straight off the base register, with no extra VALU."""
  assert tdmStoreLdsSplit(0x1234, 32) == (0, 0x1234)
  assert tdmStoreLdsSplit(DS_OFFSET_MAX, 0) == (0, DS_OFFSET_MAX)
  assert tdmStoreLdsSplit(DS_OFFSET_MAX, 1)[0] != 0, "one byte past what fits does fold"


def test_lds_split_folds_whole_pages_while_the_span_still_fits():
  """The fold stays a multiple of 64 KiB where it can, so the DS offsets keep addressing within one page."""
  assert tdmStoreLdsSplit(0x10010, 16) == (0x10000, 0x10)


def test_lds_split_folds_the_whole_offset_when_the_page_remainder_cannot_hold_the_span():
  """At the top of a page the remainder has no room for the span, so the element's whole offset is folded and
  its pieces start at 0. Without a span the same offset still fits inline: it is the span that forces this."""
  assert tdmStoreLdsSplit(0xFFE0, 0) == (0, 0xFFE0)
  assert tdmStoreLdsSplit(0xFFE0, 32) == (0xFFE0, 0)


def test_lds_split_is_not_the_masking_form_that_emitted_offset_65536():
  """The original code folded `offset & ~0xFFFF` and returned `offset & 0xFFFF`, never consulting the span, so
  an element at 0x1FFE0 staged in 32-byte pieces emitted offset:65536. Pin that case."""
  offset, span = 0x1FFE0, 32
  assert (offset & 0xFFFF) + span > DS_OFFSET_MAX, "premise: the masking form overflows here"
  high, low = tdmStoreLdsSplit(offset, span)
  assert high + low == offset
  assert low + span <= DS_OFFSET_MAX


@pytest.mark.parametrize("policy", [{"StreamK": 0}, {"StreamK": 3}, dict(_PAP, mi=_MT128)], ids=["sk0", "sk3", "pap"])
def test_c_load_wait_is_kept_before_the_barrier(gp_gfx1250, gfx1250_iim, assembler, capsys, policy):
  """In every beta TDM block the tensor_load_to_lds of C is waited for, with a wait StinkyTofu keeps, before the
  barrier after which the threads read C from the staged tile."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, TDMInst=3, **policy)
  assert sol.get("Valid"), capsys.readouterr().out
  lines, loads, block = asm.splitlines(), [], None
  for i, line in enumerate(lines):
    if line.startswith("label_GW_"):
      block = re.match(r"label_GW_B1_\w+_(?:Then|NonEdge):$", line)
    elif block and _code(line).startswith("tensor_load_to_lds"):
      loads.append(i)
  assert loads, "no tensor_load_to_lds of C in a beta store block"
  for load in loads:
    barrier = _nextLine(lines, load, ("s_barrier",))
    assert barrier is not None, lines[load]
    waits = [line for line in lines[load + 1:barrier] if _code(line).startswith("s_wait_tensorcnt")]
    assert any(STINKY_KEEP in line for line in waits), (lines[load], waits)


@pytest.mark.parametrize("term", ["UseE", "Gradient", "UseGateResidual", "OutputAmaxD"])
def test_each_term_has_epilogue_side_effects(term):
  """Each per-element output beyond C and D makes a kernel one whose edge TDM blocks keep the edge block."""
  pt = {"UseE": False, "Gradient": False, "UseGateResidual": False, "OutputAmaxD": False}
  assert not hasEpilogueSideEffects(pt)
  assert hasEpilogueSideEffects(dict(pt, **{term: True}))


# ProblemType additions, over the bf16 kernel of _params, of per-element outputs beyond C and D other than E. AmaxD
# needs fp8 D, which that kernel does not support, so test_each_term_has_epilogue_side_effects covers its term.
_SIDE_EFFECTS = {
  "gate_residual": {"UseGateResidual": True},
  "gradient_bias": {"Gradient": True, "UseBias": 1, "BiasSrc": "D", "BiasDataTypeList": ["s"]},
}


@pytest.mark.parametrize("effect", sorted(_SIDE_EFFECTS))
def test_side_effect_kernels_keep_the_edge_blocks(gp_gfx1250, gfx1250_iim, assembler, capsys, effect):
  """With UseGateResidual or Gradient with UseBias each edge TDM block is the edge block of the kernel with the switch
  off, with its vector width and its C loads, and stages D in place of its D stores."""
  _, base = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=False, ProblemType=_SIDE_EFFECTS[effect])
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStoreInst=True, ProblemType=_SIDE_EFFECTS[effect])
  assert base and sol.get("Valid"), capsys.readouterr().out
  ref, blocks = _storeBlocks(base), _storeBlocks(asm)
  tdm = {k: v for k, v in blocks.items() if any(line.startswith("tensor_store_from_lds ") for line in v)}
  edges = [k for k in tdm if k.endswith("_Then")]
  assert edges and set(tdm) <= set(ref), sorted(tdm)
  pick = lambda lines, tc, op: _sameRegs([line for line in lines if line.startswith(op) and "s[sgprSrd%s:" % tc in line])
  for k in edges:
    assert not pick(tdm[k], "D", "buffer_store"), k
    assert sum("stage D element" in line for line in tdm[k]) == len(pick(ref[k], "D", "buffer_store")), k
    if k.startswith("B1_"):
      assert pick(tdm[k], "C", "buffer_load") == pick(ref[k], "C", "buffer_load") != [], k


@pytest.mark.parametrize("tdmInst", [1, 2])
def test_one_sided_tdm_loads_are_rejected(gp_gfx1250, gfx1250_iim, assembler, capsys, tdmInst):
  """TDMInst 1 and 2 load only A or only B with TDM, and Solution rejects both, so the TDM term of the handoff, which
  _handoffBarriersLds reads as A or B, only ever sees both."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=_MT128, TDMStoreInst=True, TDMInst=tdmInst, StreamK=3)
  assert not sol.get("Valid") and asm is None
  assert "Currently TDMA and TDMB must be enabled simultaneously" in capsys.readouterr().out


def test_lds_reject_names_the_staged_tile(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """The fp32 D tile of an MT256x512 kernel, staged over the main-loop buffers, takes more LDS than the kernel has, and
  the LDS reject says that TDMStoreInst stages it, with its size and offset."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 8, 16, 2, 2], TDMStoreInst=True,
                       ProblemType={"DestDataType": "s"})
  out = capsys.readouterr().out
  assert not sol.get("Valid") and asm is None, out
  why = "; TDMStoreInst stages a %u-byte D tile from LDS offset %u" % (tdmStoreLdsBytes(sol), sol["_TDMStoreLdsOffset"])
  assert re.search(r"Kernel Uses \d+ > \d+ bytes of LDS" + re.escape(why), out), out


def _dsBytes(module):
  """The bytes of each DS store and load of an emitted module, from its mnemonics."""
  return [int(m.group(1)) // 8 for line in str(module).splitlines()
          for m in [re.match(r"\s*ds_(?:store|load)_[bu](\d+)", line)] if m]


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("svw", [3, 6])
@pytest.mark.parametrize("dest", sorted(_UNCOVERED))
def test_store_vector_widths_3_and_6_are_staged_whole_or_rejected(gp_gfx1250, gfx1250_iim, dest, svw, beta):
  """At store vector widths 3 and 6, with beta 0 and 1: a D vector tdmStoreVectorRejectReason accepts is staged by
  emitTDMStoreStaging, and with beta read back by emitTDMStoreLoadCElement, in DS instructions whose bytes add up to
  the vector; a vector it rejects is one the staging does not write whole."""
  from types import SimpleNamespace
  from toolchain_test_helpers import use_isa
  from tensilelite.Components.TDMStoreInst import emitTDMStoreLoadCElement, emitTDMStoreStaging

  use_isa((12, 5, 0), 32)
  bpe = int(DataType(dest).numBytes())
  states = SimpleNamespace(bpeCexternalGSU1=bpe, bpr=4, c=SimpleNamespace(startVgprValu=0),
                           memTokenLdsBuffer0=0, memTokenEpilogue=0)
  gwb = SimpleNamespace(parentWriter=SimpleNamespace(states=states), gwvw=svw, tdmStoreLdsBase=10, tmpVgpr=11,
                        kernel={"MacroTile0": 256, "_TDMStoreOverMainLoop": True,
                                "ProblemType": {"DestDataType": DataType(dest)}})

  def whole(emit):
    try:
      return sum(_dsBytes(emit())) == bpe * svw
    except Exception:  # a width dsStore or dsLoad does not take
      return False

  accepted = tdmStoreVectorRejectReason(_state(DestDataType=DataType(dest), StoreVectorWidth=svw)) is None
  assert accepted == (svw not in _UNCOVERED[dest])
  assert whole(lambda: emitTDMStoreStaging(gwb, [(0, 0, 0)])) == accepted
  if beta and accepted:
    assert whole(lambda: emitTDMStoreLoadCElement(gwb, SimpleNamespace(coordOffset0=0, coordOffset1=0), 20, 0))


_FOLD = "LDS offset above 64 KiB"


def _instructions(module):
  """The instructions of an emitted module, without comments."""
  return [code for code in map(_code, str(module).splitlines()) if re.match(r"[a-z][a-z0-9_]* ", code)]


def _stagedAddresses(module, base, tmp):
  """(the LDS byte each DS store of an emitted module addresses, from its address VGPR, the fold last added to that
  VGPR and its DS offset; the number of folds)."""
  value, addresses, folds = {base: 0}, [], 0
  for line in str(module).splitlines():
    code = _code(line)
    if _FOLD in line:
      dst, src, high = [operand.strip() for operand in code.split(None, 1)[1].split(",")]
      assert (re.sub(r"\D", "", dst), re.sub(r"\D", "", src)) == (str(tmp), str(base)), line
      value[tmp], folds = int(high, 16), folds + 1
    elif code.startswith("ds_store"):
      addr = int(re.sub(r"\D", "", code.split(None, 1)[1].split(",")[0]))
      offset = re.search(r"offset:(\d+)", code)
      addresses.append(value[addr] + (int(offset.group(1)) if offset else 0))
  return addresses, folds


def test_staging_adds_each_fold_once_per_batch(gp_gfx1250, gfx1250_iim):
  """Only DS stores, which write no VGPR, run between the elements a batch stages, so an element whose offset needs the
  64 KiB fold already in tmpVgpr reuses it: with one batch each fold is added once per run of elements that share
  it, and every DS store addresses the byte it addresses when each element is staged on its own, with every other
  instruction the same. The C read-back is emitted between other code of the element loop, so it adds the fold of
  every element that needs one."""
  from types import SimpleNamespace
  from toolchain_test_helpers import use_isa
  from tensilelite.Components.TDMStoreInst import emitTDMStoreLoadCElement, emitTDMStoreStaging

  use_isa((12, 5, 0), 32)
  bpe, mt0, base, tmp = 4, 256, 10, 11
  states = SimpleNamespace(bpeCexternalGSU1=bpe, bpr=4, c=SimpleNamespace(startVgprValu=0),
                           memTokenLdsBuffer0=0, memTokenEpilogue=0)
  gwb = SimpleNamespace(parentWriter=SimpleNamespace(states=states), gwvw=1, tdmStoreLdsBase=base, tmpVgpr=tmp,
                        kernel={"MacroTile0": mt0, "_TDMStoreOverMainLoop": True,
                                "ProblemType": {"DestDataType": DataType("s")}})
  # Columns 62 and 63 lie below 64 KiB, 64 to 66 in the first fold and 130 in the second; then 63, which needs
  # none and leaves the second fold in tmpVgpr, and 64, which needs the first again.
  elements = [(i, row, column) for i, (row, column) in enumerate(
    [(0, 62), (255, 63), (0, 64), (7, 65), (255, 66), (0, 130), (9, 130), (1, 63), (2, 64)])]
  expected = [(column * mt0 + row) * bpe for _, row, column in elements]
  batch = emitTDMStoreStaging(gwb, elements)
  singles = [emitTDMStoreStaging(gwb, [element]) for element in elements]
  addresses, folds = _stagedAddresses(batch, base, tmp)
  assert addresses == [address for single in singles for address in _stagedAddresses(single, base, tmp)[0]] == expected
  assert (folds, sum(_stagedAddresses(single, base, tmp)[1] for single in singles)) == (3, 6)
  others = lambda module: [code for code in _instructions(module) if not code.startswith("v_add")]
  assert others(batch) == [code for single in singles for code in others(single)]
  loads = [emitTDMStoreLoadCElement(gwb, SimpleNamespace(coordOffset0=row, coordOffset1=column), 20, 0)
           for _, row, column in elements]
  assert [str(load).count(_FOLD) for load in loads] == [int(address > DS_OFFSET_MAX) for address in expected]
