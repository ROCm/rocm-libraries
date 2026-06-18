#!/usr/bin/env python3
################################################################################
# GPU roundtrip unit test for notLocalSplitUGlobalWrite* store-D path.
#
# Tests the actual KernelWriterAssembly.notLocalSplitUGlobalWriteIndices and
# notLocalSplitUGlobalWrite methods by:
#   1. Constructing a minimal KernelWriterAssembly instance
#   2. Allocating D-tile accvgpr registers and initializing them from host data
#   3. Calling the real store-D methods to emit kernel assembly
#   4. Running on GPU and verifying output matches reference
################################################################################

import itertools
import os
import struct
import sys
import numpy as np

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from unittest.mock import MagicMock

from rocisa.code import Module, TextBlock
from rocisa.container import accvgpr, vgpr, sgpr
from rocisa.instruction import SWaitCnt

from Tensile.Common.Types import DebugConfig
from Tensile.KernelWriter import CodeModules, StateValues, StateVgprs
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.KernelWriter import KernelWriter
from Tensile.KernelWriterModules import mapAcctoArchRegs
from Tensile.Components.Subtile.Kernel import TileInfo, CD_F32

from gpu_test_helpers import (
    TileConfig,
    GFX_TARGET,
    WAVESIZE,
    NUM_THREADS,
    create_writer,
    generate_kernel_asm,
    assemble_and_run,
    assemble_kernel,
    hip,
    hip_check,
    init_rocisa,
    requires_gpu,
)

pytestmark = requires_gpu

# ---------------------------------------------------------------------------
# Test configurations: (mt_a, mt_b, depth_u)
# ---------------------------------------------------------------------------
CONFIGS = [
    TileConfig(mt_a=128, mt_b=128, depth_u=64),
    TileConfig(mt_a=256, mt_b=128, depth_u=64),
    TileConfig(mt_a=128, mt_b=256, depth_u=64),
]

# ---------------------------------------------------------------------------
# BF16 odd-MIWaveTile[0] configs: exercise the orphan subtile store path.
#
# MIWaveGroup=[2,2] (4 waves, auto-selected for even mt_a//16), MatrixInstM=16:
#   MIWaveTile[0] = mt_a / (16 * 2)
#   mt_a=96  → MIWaveTile[0]=3 (odd) — last d0 element has no sba=1 partner
#   mt_a=160 → MIWaveTile[0]=5 (odd) — same
#
# These configs specifically target _emit16bitSubtileScalarStore, which handles
# the orphan (unpaired) subtile when MIWaveTile[0] is odd.
# ---------------------------------------------------------------------------
BF16_ODD_MIWT_CONFIGS = [
    TileConfig(mt_a=96,  mt_b=128, depth_u=64),   # MIWaveTile[0]=3
    TileConfig(mt_a=160, mt_b=128, depth_u=64),   # MIWaveTile[0]=5
]

# ---------------------------------------------------------------------------
# Partial-tile edge cases: (cfg, size_i, size_j)
#
# All three configs use MIWaveGroup=[2,2], MatrixInstM/N=16, so:
#   128×128: wave_rows=64,  wave_cols=64
#   256×128: wave_rows=128, wave_cols=64
#   128×256: wave_rows=64,  wave_cols=128
#
# Edge cases target:
#   - Boundary at 1 (single row/col)
#   - MT-1 (last row/col OOB)
#   - Exactly one MMA tile deep (16)
#   - Wave boundary (wave_rows or wave_cols)
#   - Non-aligned partial in both dims simultaneously
# ---------------------------------------------------------------------------
_128x128 = TileConfig(mt_a=128, mt_b=128, depth_u=64)
_256x128 = TileConfig(mt_a=256, mt_b=128, depth_u=64)
_128x256 = TileConfig(mt_a=128, mt_b=256, depth_u=64)

PARTIAL_CASES = [
    # 128×128 — vary M, full N
    (_128x128,   1, 128),   # single row
    (_128x128, 127, 128),   # last row OOB
    (_128x128,  16, 128),   # one MMA tile in M
    (_128x128,  64, 128),   # wave boundary in M
    # 128×128 — full M, vary N
    (_128x128, 128,   1),   # single col
    (_128x128, 128, 127),   # last col OOB
    (_128x128, 128,  16),   # one MMA tile in N
    (_128x128, 128,  64),   # wave boundary in N
    # 128×128 — partial both dims
    (_128x128,  64,  64),   # wave boundary in both
    (_128x128,  17,  23),   # non-aligned in both
    # 256×128 — wave_rows=128
    (_256x128,   1, 128),   # single row, tall MT
    (_256x128, 255, 128),   # last row OOB, tall MT
    (_256x128, 128, 128),   # wave boundary in M
    (_256x128,  17,  23),   # non-aligned in both
    # 128×256 — wave_cols=128
    (_128x256, 128,   1),   # single col, wide MT
    (_128x256, 128, 255),   # last col OOB, wide MT
    (_128x256, 128, 128),   # wave boundary in N
    (_128x256,  17,  23),   # non-aligned in both
]


def _mock_f32_dtype():
    """Mock DataType that reports 4 bytes and isSingle()=True, isHalf()=False."""
    m = MagicMock()
    m.numBytes.return_value = 4
    m.numRegisters.return_value = 1
    m.isSingle.return_value = True
    m.isHalf.return_value = False
    m.isInt32.return_value = False
    m.isInt8.return_value = False
    m.isBFloat16.return_value = False
    m.isComplex.return_value = False
    m.toEnum.return_value = 0
    return m


def _mock_bf16_dtype():
    """Mock DataType that reports 2 bytes and isBFloat16()=True."""
    m = MagicMock()
    m.numBytes.return_value = 2
    m.numRegisters.return_value = 0.5
    m.isSingle.return_value = False
    m.isHalf.return_value = False
    m.isInt32.return_value = False
    m.isInt8.return_value = False
    m.isBFloat16.return_value = True
    m.isComplex.return_value = False
    m.toEnum.return_value = 0
    return m


def _build_store_kernel(cfg, mi_wave_group=None, use_bf16=False):
    """Build kernel dict for the store-D test.

    Extends the minimal kernel from create_writer with all fields needed by
    notLocalSplitUGlobalWriteIndices / notLocalSplitUGlobalWrite for the
    simple case: GSU=0, BufferStore, no bias/activation/ScaleAB/StreamK.

    mi_wave_group: optional [m0, m1] override (default: derived from cfg tile size).
    use_bf16: if True, use bf16 DestDataType with HighPrecisionAccumulate=True.
    """
    from gpu_test_helpers import _mock_dtype, _create_kernel
    kernel = _create_kernel(cfg, mi_wave_group=mi_wave_group)

    from Tensile.Common.DataType import DataType
    f32_dtype = _mock_f32_dtype()
    # PackData component lookup matches on real DataType enum values, so use real
    # DataType for DestDataType and ComputeDataType when bf16 dest is requested.
    if use_bf16:
        dest_dtype = DataType('BFloat16')
        compute_dtype = DataType('Float')
    else:
        dest_dtype = f32_dtype
        compute_dtype = f32_dtype

    # Required by TileInfo('D', kernel)
    kernel["MatrixInstN"] = 16
    kernel["MatrixInstBM"] = 1
    kernel["MatrixInstBN"] = 1

    # MIWaveTile: per-wave tile counts
    # MIWaveTile[0] = MacroTile0 / (MatrixInstM * MIWaveGroup[0])
    # MIWaveTile[1] = MacroTile1 / (MatrixInstN * MIWaveGroup[1])
    miWaveGroup = kernel["MIWaveGroup"]
    kernel["MIWaveTile"] = [
        cfg.mt_a // (16 * miWaveGroup[0]),
        cfg.mt_b // (16 * miWaveGroup[1]),
    ]

    # MIRegPerOut: accvgpr registers per output element (1 for f32)
    kernel["MIRegPerOut"] = 1

    # MIOutputVectorWidth: contiguous output elements per thread per MFMA
    # For 16x16 MFMA with WavefrontSize=64: 16*16/64 = 4
    kernel["MIOutputVectorWidth"] = 4

    # MIArchVgpr: False = use real accvgprs on gfx950
    kernel["MIArchVgpr"] = False

    # Store control
    kernel["BufferStore"] = True
    kernel["GlobalSplitU"] = 0
    kernel["_GlobalAccumulation"] = None
    kernel["StreamK"] = 0
    kernel["CompactLoopStore"] = False
    kernel["LocalSplitU"] = 1
    kernel["StoreRemapVectorWidth"] = 0
    kernel["SourceSwap"] = False
    kernel["EnableMatrixInstruction"] = True
    kernel["AdaptiveGemm"] = 0
    kernel["AdaptiveGemmGSUA"] = 0
    kernel["ActivationFuncCall"] = False
    kernel["ISA"] = [9, 5, 0]
    kernel["KernelLanguage"] = "Assembly"
    kernel["NumThreads"] = NUM_THREADS

    kernel["VectorWidthA"] = 1
    kernel["VectorWidthB"] = 1

    # Target the widest store instruction possible for both dtypes, given VWA=VWB=1.
    # With VWA=VWB=1 and MIOutputVW=4, vectorWidth0 = VWA * MIOutputVW = 4.
    # SVW must be <= vectorWidth0 (=4) because NotLocalFullTileElements generates
    # elements by stepping vc0 in range(0, vectorWidth0, SVW) — SVW > vectorWidth0
    # collapses all elements to a single vc0=0 step and tries to span more accvgprs
    # than are allocated per element, producing wrong output.
    #   f32  dest (4 B/elem): SVW = min(16/4, 4) = 4 → buffer_store_dwordx4 (16 bytes) ✓
    #   bf16 dest (2 B/elem): SVW = min(16/2, 4) = 4 → buffer_store_dwordx2  (8 bytes) ✓
    bpe = int(dest_dtype.numBytes())
    mi_output_vw = kernel["MIOutputVectorWidth"]
    kernel["StoreVectorWidth"] = min(16 // bpe, mi_output_vw)
    kernel["_VectorStore"] = True

    # PackedC1IndicesX: [1] means the J-dim index is index 1
    kernel["PackedC1IndicesX"] = [1]

    # WorkGroup sgpr names (used symbolically)
    kernel["WorkGroup0"] = "WorkGroup0"
    kernel["WorkGroup1"] = "WorkGroup1"

    # LdsOffsetBias placeholders (accessed even when useBias=NONE via backup/restore)
    kernel["LdsOffsetBiasNonGSU"] = 0
    kernel["LdsOffsetBiasGSU"] = 0
    kernel["LdsNumBytes"] = 256  # min LDS granularity; avoids div-by-zero in getLdsLimitedOccupancy

    # Store occupancy / batch fields
    kernel["GlobalSplitUAlgorithm"] = "SingleBuffer"
    kernel["PackedC0IndicesX"] = [0]
    kernel["NumWaveSplitK"] = 1
    kernel["GroupLoadStore"] = False
    kernel["GlobalWriteVectorWidth"] = min(16 // bpe, mi_output_vw)
    kernel["NonTemporalD"] = 0
    kernel["NonTemporalC"] = 0
    kernel["NonTemporalE"] = 0
    kernel["StorePriorityOpt"] = False
    kernel["StoreSyncOpt"] = False
    kernel["AssertFree0ElementMultiple"] = 1
    kernel["NumElementsPerBatchStore"] = 0  # 0 = auto
    kernel["UseDotInstruction"] = False
    kernel["ActivationFused"] = False
    kernel["MbskPrefetchMethod"] = 0
    kernel["NumMbskPrefetchElements"] = 0
    kernel["InterleaveAlpha"] = False
    kernel["ExpertSchedulingMode"] = 0
    kernel["WorkGroupReduction"] = False

    # Override ComputeDataType: use real DataType for bf16 (needed by PackData component
    # lookup), or a proper f32 mock for the f32 path.
    kernel["ProblemType"]["ComputeDataType"] = compute_dtype

    # ProblemType: DataTypeE (needed by GlobalWriteBatch for UseE=False path)
    kernel["ProblemType"]["DataTypeE"] = _mock_f32_dtype()

    # ProblemType extras
    kernel["ProblemType"]["UseE"] = False
    kernel["ProblemType"]["UseScaleAB"] = ""
    kernel["ProblemType"]["UseScaleAlphaVec"] = False
    kernel["ProblemType"]["UseScaleCD"] = False
    kernel["ProblemType"]["OutputAmaxD"] = False
    kernel["ProblemType"]["UseBias"] = False
    kernel["ProblemType"]["UseBeta"] = False
    kernel["ProblemType"]["BiasSrc"] = "D"
    kernel["ProblemType"]["BiasDataTypeList"] = []
    kernel["ProblemType"]["Gradient"] = False
    kernel["ProblemType"]["WorkGroupReduction"] = False
    kernel["ProblemType"]["SparseA"] = False
    kernel["ProblemType"]["ActivationType"] = "none"
    kernel["ProblemType"]["ActivationNoGuard"] = True
    kernel["ProblemType"]["ActivationComputeDataType"] = compute_dtype
    kernel["ProblemType"]["HighPrecisionAccumulate"] = use_bf16
    kernel["ProblemType"]["StochasticRounding"] = False
    kernel["ProblemType"]["UseInitialStridesCD"] = False
    kernel["ProblemType"]["UseInitialStridesAB"] = False
    kernel["ProblemType"]["Fp16AltImpl"] = False
    kernel["ProblemType"]["Fp16AltImplRound"] = False
    kernel["ProblemType"]["MacDataTypeA"] = _mock_f32_dtype()
    kernel["ProblemType"]["MacDataTypeB"] = _mock_f32_dtype()
    kernel["ProblemType"]["DestDataType"] = dest_dtype
    kernel["ProblemType"]["DataType"] = f32_dtype  # input data type (A/B); always f32 accumulators
    kernel["ProblemType"]["Index0"] = 0
    kernel["ProblemType"]["Index1"] = 1
    kernel["ProblemType"]["OperationType"] = "GEMM"  # required by GlobalWriteComponents.find

    return kernel


def _build_kwa(kernel, writer, use_bf16=False):
    """Construct a minimal KernelWriterAssembly with real pools.

    Calls KernelWriter.__init__ with a mock Assembler and DebugConfig(),
    then wires in the pools and states from the mock writer.
    """
    mock_assembler = MagicMock()
    mock_assembler.rocm_version = MagicMock()
    mock_assembler.rocm_version.major = 6

    debug_config = DebugConfig()

    # Bypass __init__ of KernelWriterAssembly (it only adds globalread_gpr_record)
    # and use KernelWriter.__init__ directly
    kw = object.__new__(KernelWriterAssembly)
    KernelWriter.__init__(kw, mock_assembler, debug_config)

    # Add KernelWriterAssembly-only attribute
    from Tensile.KernelWriterAssembly import GlobalReadGprRecord
    kw.globalread_gpr_record = GlobalReadGprRecord()

    # Wire in real pools from the mock writer
    kw.sgprPool = writer.sgprPool
    kw.vgprPool = writer.vgprPool
    kw.agprPool = writer.agprPool
    kw.sgprs    = writer.sgprs

    # Initialize codes container
    kw.codes = CodeModules()

    # Set up states with the kernel
    # init_rocisa() must have been called by the test before _build_kwa.
    # Do NOT call setKernel here — it resets archCaps to {}.
    version = (9, 5, 0)

    kw.states = StateValues(version=version, kernel=kernel, kernelName="test_storeD")
    kw.vgprs  = StateVgprs()

    # Get caps from the rocisa singleton.
    # Always call init+setKernel here to ensure the singleton is properly
    # initialized regardless of what other imports may have done.
    import shutil
    from rocisa import rocIsa as rocIsaModule2
    ri2 = rocIsaModule2.getInstance()
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri2.init(version, asmpath)
    ri2.setKernel(version, WAVESIZE)
    kw.states.archCaps = ri2.getArchCaps()
    kw.states.asmCaps  = ri2.getAsmCaps()
    kw.states.regCaps  = ri2.getRegCaps()

    # Populate required states fields for the simple store case
    bpeDest = 2 if use_bf16 else 4
    kw.states.indexChars = list("IJKLMNOPQRSTUVWXYZ")
    kw.states.bpr = 4
    kw.states.bpeCexternal    = bpeDest  # 2 for bf16, 4 for f32
    kw.states.bpeCinternal    = 4        # always fp32 accumulators
    kw.states.bpeCexternalGSU1 = bpeDest
    kw.states.bpeE            = 4   # E matrix bytes per element (UseE=False, so unused)
    kw.states.maxLimitAgprs   = 256  # gfx950 supports 256 agprs
    kw.states.useBias         = kw.states.useBias  # already DataDirection.NONE
    kw.states.serializedStore = True
    kw.states.numStoreSgprToLoad  = 0
    kw.states.numStoreSgprToLoad2 = 0
    kw.states.doubleVgpr = False

    # c.numVgprValu: 0 for pure accvgpr path (no arch vgprs used for ValuC)
    kw.states.c.numVgprValu = 0
    kw.states.c.startVgprValu = 0

    return kw


def _allocate_d_tile(kernel, writer):
    """Allocate D-tile accvgprs via TileInfo.

    Returns (tileInfoD, agpr_base) where agpr_base is the first accvgpr index.
    """
    tileInfoD = TileInfo(CD_F32, 'D', writer, kernel)
    tileInfoD.allocVgprTileRegisters_legacy(writer, kernel)
    # Collect all allocated agpr indices (in order)
    agpr_indices = []
    for vtile in tileInfoD.vgprTiles:
        for reg in vtile:
            agpr_indices.append(reg)
    return tileInfoD, agpr_indices


def _sgpr_offset(writer_sgprs, name):
    """Return byte offset of a named sgpr in the sgprs dict (4 bytes per reg)."""
    return writer_sgprs[name] * 4


def _finalize_inner_asm(parts, kw):
    """Join ASM parts, appending deferred edge modules (after s_endpgm) if present.

    UseSubtileImpl defers edge store batches to kw.states.deferredEdgeModules.
    These are reached via PC-relative jumps (s_getpc/s_setpc), not fall-through,
    so they must be placed after an s_endpgm that terminates the main code path.
    Without this, execution falls through label_GW_End into the deferred code.
    """
    if hasattr(kw.states, 'deferredEdgeModules') and kw.states.deferredEdgeModules:
        deferred = "\n".join(str(m) for m in kw.states.deferredEdgeModules)
        kw.states.deferredEdgeModules = []
        parts.append("  s_waitcnt vmcnt(0)\n  s_endpgm\n")
        parts.append(deferred)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Kernel args layout for the store-D test:
#   s[0:1]  = kernarg ptr (hardware)
#   s[2:3]  = SrdD (output buffer address, 64-bit) → loaded from kernargs offset 0
#   s[4:5]  = SrdC (same as SrdD for this test)   → loaded from kernargs offset 8
#   s[6]    = WorkGroup0                           → always 0
#   s[7]    = WorkGroup1                           → always 0
#   s[8]    = StrideD1J (stride in elements)       → loaded from kernargs offset 24
#   s[9:12] = SrdD full descriptor (128-bit)       → SRD for buffer_store_dword*
#   s[13:14]= input buffer ptr                     → loaded from kernargs offset 32
#   s[15]   = StrideC1J                            → same as StrideD1J
# ---------------------------------------------------------------------------

def _build_sgprs_for_test(writer):
    """Allocate and name the sgprs needed by the store-D path.

    The store-D path uses:
      - SrdD[0:3]  (4 regs): buffer resource descriptor for D (must be 4-aligned)
      - SrdC[0:3]  (4 regs): buffer resource descriptor for C (same buffer)
      - WorkGroup0 (1 reg)
      - WorkGroup1 (1 reg)
      - StrideDJ   (1 reg): output stride in elements (D J dimension)
      - StrideCJ   (1 reg): same as StrideDJ
      - SizeI      (1 reg): number of rows (MacroTile0)
      - SizeJ      (1 reg): number of cols (MacroTile1)
      - NumWorkGroups0 (1 reg): = 1 for workgroup (0,0)
      - NumWorkGroups1 (1 reg): = 1 for workgroup (0,0)
      - Alpha      (1 reg): 1.0f for identity alpha
      - SrdInput[0:3] (4 regs): input buffer descriptor for loading accvgprs

    Layout:
      s[0:3] reserved: s[0:1] = kernarg ptr, s[2:3] padding for 4-alignment
      s[4:7]  = SrdD
      s[8:11] = SrdC
      s[12]   = WorkGroup0
      s[13]   = WorkGroup1
      s[14]   = StrideDJ
      s[15]   = StrideCJ
      s[16]   = SizeI
      s[17]   = SizeJ
      s[18]   = NumWorkGroups0
      s[19]   = NumWorkGroups1
      s[20]   = Alpha (1.0f = 0x3f800000)
      s[24:27] = SrdInput (4-aligned)
    """
    # s[0:3] reserved: s[0:1] = kernarg ptr, s[2:3] = padding to 4-align SrdD
    writer.sgprPool.checkOut(4, tag="_build_sgprs_for_test_sgprs")

    # SrdD: 4 sgprs for buffer descriptor (must be 4-aligned for buffer_store)
    srd_d_base = writer.sgprPool.checkOut(4, "SrdD")
    writer.sgprs["SrdD"] = srd_d_base           # s[4]; s[4:7] used as SRD

    # SrdC: 4 sgprs for C buffer descriptor (same as D for this test)
    srd_c_base = writer.sgprPool.checkOut(4, "SrdC")
    writer.sgprs["SrdC"] = srd_c_base           # s[8]; s[8:11] used as SRD

    # WorkGroup0, WorkGroup1
    wg0 = writer.sgprPool.checkOut(1, "WorkGroup0")
    writer.sgprs["WorkGroup0"] = wg0             # s[12]

    wg1 = writer.sgprPool.checkOut(1, "WorkGroup1")
    writer.sgprs["WorkGroup1"] = wg1             # s[13]

    # StrideDJ / StrideCJ (names used by the store-D code)
    strdj = writer.sgprPool.checkOut(1, "StrideDJ")
    writer.sgprs["StrideDJ"] = strdj             # s[14]

    strcj = writer.sgprPool.checkOut(1, "StrideCJ")
    writer.sgprs["StrideCJ"] = strcj             # s[15]

    # SizeI, SizeJ (used by bound checking in the store path)
    sizei = writer.sgprPool.checkOut(1, "SizeI")
    writer.sgprs["SizeI"] = sizei                # s[16]

    sizej = writer.sgprPool.checkOut(1, "SizeJ")
    writer.sgprs["SizeJ"] = sizej                # s[17]

    # NumWorkGroups0/1 (used by coord computation)
    nwg0 = writer.sgprPool.checkOut(1, "NumWorkGroups0")
    writer.sgprs["NumWorkGroups0"] = nwg0        # s[18]

    nwg1 = writer.sgprPool.checkOut(1, "NumWorkGroups1")
    writer.sgprs["NumWorkGroups1"] = nwg1        # s[19]

    # Alpha (2 regs: s[Alpha] and s[Alpha+1] both set to 1.0f)
    # VMulPKF32 uses sgpr("Alpha", 2) to do packed f32 multiply on 2 vgprs at once.
    # Both sgprs must contain 1.0f (0x3f800000) for identity alpha scaling.
    alpha = writer.sgprPool.checkOut(2, "Alpha")
    writer.sgprs["Alpha"] = alpha                # s[20:21]

    # Pad to 4-align SrdInput (s[22:23] = padding, SrdInput starts at s[24])
    writer.sgprPool.checkOut(2, tag="_build_sgprs_for_test_pad_to_4_align_SrdInput")

    # SrdInput: input buffer descriptor (4-aligned)
    srd_in_base = writer.sgprPool.checkOut(4, "SrdInput")
    writer.sgprs["SrdInput"] = srd_in_base       # s[24]; s[24:27] used as SRD

    # Subtile guard sgprs (used by UseSubtileImpl edge/non-edge dispatch)
    mGuard = writer.sgprPool.checkOut(1, "subtileMValidBlocks")
    writer.sgprs["subtileMValidBlocks"] = mGuard
    nGuard = writer.sgprPool.checkOut(1, "subtileNValidBlocks")
    writer.sgprs["subtileNValidBlocks"] = nGuard

    return writer.sgprs


def _build_prologue(sgprs, num_agprs, mt0, mt1, stride_d=None, use_input_buf=True):
    """Generate prologue that loads kernel args and initializes the SRDs.

    Kernarg layout (matches args descriptor in the test):
      offset  0: u64 input buffer ptr   (SrdInput base)
      offset  8: u64 output buffer ptr  (SrdD/SrdC base)
      offset 16: u32 size_i (SizeI = actual M dimension, used for bounds check)
      offset 20: u32 size_j (SizeJ = actual N dimension, used for bounds check)
      offset 24: u32 stride_d (StrideDJ = MacroTile0 = buffer leading dimension)

    stride_d defaults to mt0 (correct when size_i == mt0).

    output_bytes: if given, used as SrdD num_records so that the OOB-redirect address
    (BufferOOB label) falls outside the buffer and is silently discarded by the hardware.
    If None, falls back to 0xFFFFFFFF (max).

    sgpr layout:
      s[0:1]  = kernarg ptr (hardware)
      s[4:7]  = SrdD
      s[8:11] = SrdC
      s[12]   = WorkGroup0
      s[13]   = WorkGroup1
      s[14]   = StrideDJ
      s[15]   = StrideCJ
      s[16]   = SizeI
      s[17]   = SizeJ
      s[18]   = NumWorkGroups0
      s[19]   = NumWorkGroups1
      s[20]   = Alpha (1.0f)
      s[24:27]= SrdInput

    SRD format: [base_lo, base_hi, num_records, format_word]
    format_word = 0x20000 (stride=0, OOB_SELECT=2 = clamp)
    """
    from rocisa.instruction import SLoadB64, SLoadB32, SMovB32, SMulI32
    from rocisa.code import TextBlock

    if stride_d is None:
        stride_d = mt0

    m = Module("prologue")

    # --- Load pointers from kernargs ---
    # run_on_gpu builds kernargs as: [input_ptrs...] + [output_ptr] + [scalars...]
    # With use_input_buf=True, inputs=(input_arr,), scalars=(size_i, size_j, stride_d):
    #   kernarg[0:8]   = input_ptr
    #   kernarg[8:16]  = output_ptr
    #   kernarg[16:20] = size_i
    #   kernarg[20:24] = size_j
    #   kernarg[24:28] = stride_d
    # With use_input_buf=False, inputs=(), scalars=(size_i, size_j, stride_d):
    #   kernarg[0:8]   = output_ptr
    #   kernarg[8:12]  = size_i
    #   kernarg[12:16] = size_j
    #   kernarg[16:20] = stride_d
    if use_input_buf:
        m.add(SLoadB64(dst=sgpr(sgprs["SrdInput"], 2), base=sgpr(0, 2), soffset=0,
                       comment="load input ptr → SrdInput[0:1]"))
        m.add(SLoadB64(dst=sgpr(sgprs["SrdD"], 2),     base=sgpr(0, 2), soffset=8,
                       comment="load output ptr → SrdD[0:1]"))
        m.add(SLoadB32(dst=sgpr(sgprs["SizeI"]),        base=sgpr(0, 2), soffset=16,
                       comment="load size_i → SizeI"))
        m.add(SLoadB32(dst=sgpr(sgprs["SizeJ"]),        base=sgpr(0, 2), soffset=20,
                       comment="load size_j → SizeJ"))
        m.add(SLoadB32(dst=sgpr(sgprs["StrideDJ"]),     base=sgpr(0, 2), soffset=24,
                       comment="load stride_d → StrideDJ"))
    else:
        m.add(SLoadB64(dst=sgpr(sgprs["SrdD"], 2),  base=sgpr(0, 2), soffset=0,
                       comment="load output ptr → SrdD[0:1]"))
        m.add(SLoadB32(dst=sgpr(sgprs["SizeI"]),     base=sgpr(0, 2), soffset=8,
                       comment="load size_i → SizeI"))
        m.add(SLoadB32(dst=sgpr(sgprs["SizeJ"]),     base=sgpr(0, 2), soffset=12,
                       comment="load size_j → SizeJ"))
        m.add(SLoadB32(dst=sgpr(sgprs["StrideDJ"]),  base=sgpr(0, 2), soffset=16,
                       comment="load stride_d → StrideDJ"))
    m.add(SWaitCnt(dscnt=0, comment="wait for sloads"))

    srd_d   = sgprs["SrdD"]
    srd_c   = sgprs["SrdC"]
    srd_in  = sgprs["SrdInput"]

    # --- Build SrdD ---
    # Set num_records = BufferOOB = 0x80000000.  The store path redirects OOB elements
    # to byte offset 0x80000000 via v_mov_b32 vN, BufferOOB.  With num_records=0x80000000,
    # any store at offset >= 0x80000000 is out-of-bounds and silently suppressed.
    # This matches how real Tensile kernels set up the SRD post-loop.
    m.add(SMovB32(dst=sgpr(srd_d + 2), src="0x80000000", comment="SrdD num_records = BufferOOB (2GB)"))
    m.add(SMovB32(dst=sgpr(srd_d + 3), src="0x20000", comment="SrdD format word"))

    # --- Copy SrdD to SrdC (same buffer for this test) ---
    for i in range(4):
        m.add(SMovB32(dst=sgpr(srd_c + i), src=sgpr(srd_d + i),
                      comment=f"SrdC[{i}] = SrdD[{i}]"))

    # --- SrdInput ---
    m.add(SMovB32(dst=sgpr(srd_in + 2), src="0xFFFFFFFF",
                  comment="SrdInput num_records = max"))
    m.add(SMovB32(dst=sgpr(srd_in + 3), src="0x20000",
                  comment="SrdInput format word (OOB_SELECT=2)"))

    # --- Scalar constants ---
    # WorkGroup0/1 = 0 (single workgroup at (0,0))
    m.add(SMovB32(dst=sgpr(sgprs["WorkGroup0"]), src=0, comment="WorkGroup0 = 0"))
    m.add(SMovB32(dst=sgpr(sgprs["WorkGroup1"]), src=0, comment="WorkGroup1 = 0"))

    # StrideCJ = StrideDJ (loaded from kernargs above)
    m.add(SMovB32(dst=sgpr(sgprs["StrideCJ"]), src=sgpr(sgprs["StrideDJ"]),
                  comment="StrideCJ = StrideDJ"))

    # NumWorkGroups0/1 = 1 (only one workgroup in each dimension)
    m.add(SMovB32(dst=sgpr(sgprs["NumWorkGroups0"]), src=1, comment="NumWorkGroups0 = 1"))
    m.add(SMovB32(dst=sgpr(sgprs["NumWorkGroups1"]), src=1, comment="NumWorkGroups1 = 1"))

    # Alpha = 1.0f (0x3f800000) for identity alpha scaling.
    # Both Alpha and Alpha+1 must be 1.0f: VMulPKF32 uses sgpr("Alpha", 2).
    m.add(SMovB32(dst=sgpr(sgprs["Alpha"]),     src="0x3f800000", comment="Alpha[0] = 1.0f"))
    m.add(SMovB32(dst=sgpr(sgprs["Alpha"] + 1), src="0x3f800000", comment="Alpha[1] = 1.0f"))

    # Subtile guard SGPRs: compute numValidMBlocks / numValidNBlocks per wave.
    if "subtileMValidBlocks" in sgprs:
        import math
        miM = 16
        miWaveGroup0 = mt0 // (miM * (mt0 // (miM * max(1, mt0 // (miM * 4)))))
        miWaveGroup1 = num_agprs // (mt0 // miM * mt1 // 16) if mt1 > 16 else 1
        # Derive MIWaveGroup from tile: same logic as _create_kernel
        if ((mt0 // 16) % 2 == 0) and ((mt1 // 16) % 2 == 0):
            wg0, wg1 = 2, 2
        elif ((mt0 // 16) % 2 != 0) and ((mt1 // 16) % 4 == 0):
            wg0, wg1 = 1, 4
        elif ((mt0 // 16) % 4 == 0) and ((mt1 // 16) % 2 != 0):
            wg0, wg1 = 4, 1
        else:
            wg0, wg1 = 2, 2
        miwt0 = mt0 // (miM * wg0)
        miwt1 = mt1 // (16 * wg1)
        waveGroupM = miM * miwt0
        waveGroupN = 16 * miwt1
        log2_wg0 = int(math.log2(wg0))
        miMShift = int(math.log2(miM))
        mGuard = sgprs["subtileMValidBlocks"]
        nGuard = sgprs["subtileNValidBlocks"]
        m.add(TextBlock(
            f"  v_readfirstlane_b32 s{mGuard}, v0          // lane0 tid\n"
            f"  s_lshr_b32 s{mGuard}, s{mGuard}, 6         // waveId = tid >> 6\n"
            f"  s_lshr_b32 s{nGuard}, s{mGuard}, {log2_wg0} // waveIdN\n"
            f"  s_and_b32  s{mGuard}, s{mGuard}, {wg0-1}   // waveIdM\n"
            f"  s_mul_i32  s{mGuard}, s{mGuard}, {waveGroupM} // waveBase\n"
            f"  s_sub_u32  s{mGuard}, s{sgprs['SizeI']}, s{mGuard} // remainder\n"
            f"  s_cselect_b32 s{mGuard}, 0, s{mGuard}      // clamp to 0 if OOB\n"
            f"  s_add_u32  s{mGuard}, s{mGuard}, {miM-1}   // ceil\n"
            f"  s_lshr_b32 s{mGuard}, s{mGuard}, {miMShift} // >> {miMShift}\n"
            f"  s_min_u32  s{mGuard}, s{mGuard}, {miwt0}   // clamp to MIWaveTile[0]\n"
            f"  s_mul_i32  s{nGuard}, s{nGuard}, {waveGroupN} // waveBaseN\n"
            f"  s_sub_u32  s{nGuard}, s{sgprs['SizeJ']}, s{nGuard} // validN - waveBaseN\n"
            f"  s_cselect_b32 s{nGuard}, 0, s{nGuard}      // clamp to 0 if OOB\n"
            f"  s_min_u32  s{nGuard}, s{nGuard}, {waveGroupN} // min(validN_wave, waveGroupN)\n"
            f"  s_lshr_b32 s{nGuard}, s{nGuard}, 4          // numValid16NBlocks\n"
        ))

    return m



def _compute_reference(cfg, kernel, tileInfoD, size_i, size_j, rng=None):
    """Build the MT_a × MT_b column-major input buffer for the store-D roundtrip test.

    Buffer layout (stride MT_a, matching the init asm):
        flat[col * MT_a + row] = value  for row < size_i, col < size_j
        flat[col * MT_a + row] = f32(0xFFFFFFFF) sentinel  otherwise (OOB position)

    When rng is None (serial mode): value = float(col * size_i + row), giving
    contiguous indices 0..size_i*size_j-1 for valid positions.
    When rng is provided (random mode): value is drawn from Uniform[-9, 9] for each
    valid position.

    Returns:
        (input_bytes, expected_set, ref_arr):
            input_bytes  — raw bytes for the MT_a × MT_b float32 buffer (column-major).
            expected_set — set of valid values for serial mode; None for random mode.
            ref_arr      — numpy float32 array of the full MT_a × MT_b column-major
                           buffer (valid values + sentinel at OOB), used for
                           positional verification in random mode.
    """
    MT_a = kernel["MacroTile0"]
    MT_b = kernel["MacroTile1"]
    _SENTINEL = struct.unpack('f', struct.pack('I', 0xFFFFFFFF))[0]  # quiet NaN

    if rng is not None:
        valid_vals = rng.uniform(-9.0, 9.0, size=size_i * size_j).astype(np.float32)
    valid_idx = 0
    flat = []
    for col in range(MT_b):
        for row in range(MT_a):
            if row < size_i and col < size_j:
                if rng is not None:
                    flat.append(float(valid_vals[valid_idx]))
                    valid_idx += 1
                else:
                    flat.append(float(col * size_i + row))
            else:
                flat.append(_SENTINEL)

    ref_arr = np.array(flat, dtype=np.float32)
    input_bytes = ref_arr.tobytes()
    expected_set = None if rng is not None else {float(v) for v in range(size_i * size_j)}
    return input_bytes, expected_set, ref_arr


def _is_sentinel(v):
    """Return True if v has all-1 binary encoding (0xffffffff for f32, 0xffff for f16/bf16).

    For bf16 output the buffer is initialized to 0xFF bytes, so unwritten elements
    are 0xFFFF as uint16.  After zero-extending to float32 (<<16) the pattern becomes
    0xFFFF0000, which is a quiet NaN — detect this pattern too.
    """
    bits = struct.unpack('I', struct.pack('f', float(v)))[0]
    return bits == 0xFFFFFFFF or bits == 0x0000FFFF or bits == 0xFFFF0000


def _print_subtile_map(mt0, mt1, out, tileInfoD, label):
    """Print D output as a grid with one value per MMA tile (column-major reshape).

    mt0, mt1: buffer dimensions (rounded up to MT multiples).
    Unwritten (OOB) elements initialized to 0xffffffff/0xffff are shown as '-'.
    """
    tile_m = tileInfoD.mmaTileShape[0]
    tile_n = tileInfoD.mmaTileShape[1]
    out_2d = np.array(out, dtype=np.float32).reshape(mt0, mt1, order='F')
    nrows = mt0 // tile_m
    ncols = mt1 // tile_n
    print(f"\nD subtile map [{label}] — one value per {tile_m}x{tile_n} MMA tile "
          f"(rows=I-dim, cols=J-dim):")
    for r in range(nrows):
        print("  " + " ".join(
            "  -" if _is_sentinel(out_2d[r*tile_m, c*tile_n]) else f"{int(out_2d[r*tile_m, c*tile_n]):3d}"
            for c in range(ncols)))


def _print_full_matrix(out, size_i, size_j, label):
    """Print all size_i x size_j D values (column-major layout).

    Unwritten (OOB) elements initialized to 0xffffffff/0xffff are shown as '-'.
    """
    out_2d = np.array(out, dtype=np.float32).reshape(size_i, size_j, order='F')
    print(f"\nD matrix [{label}] — all {size_i}x{size_j} values (rows=I-dim, cols=J-dim):")
    for r in range(size_i):
        print("  " + " ".join(" -" if _is_sentinel(v) else f"{int(v):2d}" for v in out_2d[r]))


def _print_ref_matrix(ref_arr, size_i, size_j, label):
    """Print the reference C matrix (column-major layout).

    OOB positions (sentinel 0xFFFFFFFF) are shown as '-'.
    """
    ref_2d = ref_arr.reshape(size_i, size_j, order='F')
    print(f"\nC ref [{label}] — all {size_i}x{size_j} values (rows=I-dim, cols=J-dim):")
    for r in range(size_i):
        print("  " + " ".join(" -" if _is_sentinel(v) else f"{int(v):2d}" for v in ref_2d[r]))


def _print_store_instructions(store_write_asm, label):
    """Print all buffer_store_* instructions emitted by the store-D module.

    Each store instruction is printed together with the nearest preceding comment
    line (if any) so the reader can identify which element/batch it belongs to.

    Args:
        store_write_asm: Assembly string from str(store_write_mod).
        label:           Test case label for the header line.
    """
    import re
    lines = store_write_asm.splitlines()
    print(f"\n[store-insts] {label}")
    last_comment = ""
    inst_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            last_comment = stripped
        elif re.search(r'\bbuffer_store_\w+', stripped):
            if last_comment:
                print(f"  {last_comment}")
                last_comment = ""
            print(f"  {stripped}")
            inst_count += 1
    print(f"  ({inst_count} buffer_store instruction(s) total)")


def _verify_output(out, expected_set, full_tile=True):
    """Assert non-sentinel output values are consistent with expected_set.

    full_tile=True  (default): bidirectional check — no unexpected values AND
                               all expected values must appear (full macrotile written).
    full_tile=False:           one-directional check — no unexpected values only
                               (used when M < MT0 or N < MT1, so some accvgprs map
                               to OOB positions and are never written).
    """
    non_sentinel = [v for v in out if not _is_sentinel(v)]
    unexpected = [v for v in non_sentinel if v not in expected_set]
    assert not unexpected, \
        f"Found {len(unexpected)} unexpected output values: {unexpected[:10]}"
    if full_tile:
        missing = expected_set - set(non_sentinel)
        assert not missing, \
            f"Missing {len(missing)} expected values in output: {list(missing)[:10]}"


def _make_full_dref(mt0, mt1):
    """Build the full MT_a × MT_b reference matrix: D_ref[r][c] = c * mt0 + r (column-major)."""
    return [[c * mt0 + r for c in range(mt1)] for r in range(mt0)]


def _verify_oob_sentinel(out, round_mt0, round_mt1, size_i, size_j, label="f32"):
    """Assert every OOB position in the D buffer still holds the hardware sentinel.

    The GPU runner initialises the output buffer with 0xFF bytes before launching
    the kernel, so every element starts as:
      f32  → 0xFFFFFFFF (quiet NaN)
      bf16 → 0xFFFF per element → zero-extended to 0x0000FFFF as f32

    Any store that reaches an OOB address must be suppressed (by the SrdD
    BufferOOB redirect or the SrdC+2 gating).  If a store slips through it
    overwrites the sentinel and the assertion fires.

    Args:
        out:       Flat sequence of float values from the GPU output buffer.
        round_mt0: Buffer rows (leading dimension of the column-major buffer).
        round_mt1: Buffer cols.
        size_i:    Number of valid rows  (positions [0, size_i) × [0, size_j) are valid).
        size_j:    Number of valid cols.
        label:     "f32" or "bf16" — selects which sentinel bit pattern to expect.
    """
    D_u32 = np.array(out, dtype=np.float32).view(np.uint32).reshape(round_mt0, round_mt1, order='F')
    # f32:  buffer init is 0xFF bytes → each f32 element = 0xFFFFFFFF.
    # bf16: buffer init is 0xFF bytes → each bf16 element = 0xFFFF (uint16).
    #       The caller zero-extends via (uint16 << 16), so the sentinel in the
    #       'out' array is 0xFFFF0000 (not 0x0000FFFF).
    sentinel = np.uint32(0xFFFFFFFF) if label == "f32" else np.uint32(0xFFFF0000)
    errors = []
    for col in range(round_mt1):
        for row in range(round_mt0):
            if row < size_i and col < size_j:
                continue  # valid position — not checked here
            if D_u32[row, col] != sentinel:
                errors.append((row, col, hex(int(sentinel)), hex(int(D_u32[row, col]))))
    if errors:
        assert False, (
            f"OOB sentinel overwritten at {len(errors)} positions "
            f"(first 5 (row,col,expected,got): {errors[:5]})"
        )


def _verify_matrix_positions(out, round_mt0, round_mt1, size_i=None, size_j=None):
    """Verify the output exactly matches the serial column-major D_ref matrix,
    and that every OOB position still holds the f32 sentinel (0xFFFFFFFF).

    D_ref[row][col] = col * size_i + row  for row < size_i, col < size_j.

    size_i / size_j default to round_mt0 / round_mt1 (full-tile case).

    Args:
        out:        Flat tuple of float values from the GPU output buffer.
        round_mt0:  Buffer rows (leading dimension of the column-major buffer).
        round_mt1:  Buffer cols.
        size_i:     Valid M dimension (rows). Defaults to round_mt0.
        size_j:     Valid N dimension (cols). Defaults to round_mt1.
    """
    if size_i is None:
        size_i = round_mt0
    if size_j is None:
        size_j = round_mt1
    D = np.array(out, dtype=np.float32).reshape(round_mt0, round_mt1, order='F')
    errors = []
    for c in range(size_j):
        for r in range(size_i):
            expected = float(c * size_i + r)
            got = float(D[r, c])
            if got != expected:
                errors.append((r, c, expected, got))
    if errors:
        assert False, (
            f"Matrix position mismatch in {len(errors)} elements "
            f"(first 5 (row,col,expected,got): {errors[:5]})"
        )
    _verify_oob_sentinel(out, round_mt0, round_mt1, size_i, size_j, label="f32")


def _bf16_rne(f32_arr):
    """Convert a float32 numpy array to bf16 round-to-nearest-even, returned as float32.

    Matches the rounding performed by v_cvt_pk_bf16_f32 on CDNA3.
    """
    u32 = f32_arr.view(np.uint32)
    lsb       = (u32 >> 16) & np.uint32(1)
    round_bit = (u32 >> 15) & np.uint32(1)
    sticky    = (u32 & np.uint32(0x7FFF)) != np.uint32(0)
    round_up  = round_bit & (sticky | lsb)
    rounded   = (u32 + (round_up.astype(np.uint32) << np.uint32(16))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _verify_bf16_random_matrix_positions(out, ref_arr, round_mt0, round_mt1, size_i, size_j):
    """Verify bf16 D output matches bf16_rne(ref) at valid positions, and that OOB
    positions still hold the bf16 sentinel (0x0000FFFF when zero-extended to f32).

    ref_arr contains f32 random values (valid positions) and f32 sentinel (OOB).
    The GPU converts accvgprs to bf16 via v_cvt_pk_bf16_f32, so the expected value
    at each valid position is bf16_rne(ref_arr[r, c]).

    Args:
        out:      Flat sequence of bf16-upcast-to-f32 values from the GPU output.
        ref_arr:  numpy float32 array, shape (round_mt0 * round_mt1,), column-major.
        size_i:   Number of valid rows.
        size_j:   Number of valid cols.
    """
    ref_rne     = _bf16_rne(ref_arr)   # apply RNE to flat array (contiguous)
    D_u32       = np.array(out, dtype=np.float32).view(np.uint32).reshape(round_mt0, round_mt1, order='F')
    ref_rne_u32 = ref_rne.view(np.uint32).reshape(round_mt0, round_mt1, order='F')
    ref_rne_2d  = ref_rne.reshape(round_mt0, round_mt1, order='F')
    D_f32       = np.array(out, dtype=np.float32).reshape(round_mt0, round_mt1, order='F')

    errors = []
    for c in range(size_j):
        for r in range(size_i):
            if D_u32[r, c] != ref_rne_u32[r, c]:
                errors.append((r, c, float(ref_rne_2d[r, c]), float(D_f32[r, c])))
    if errors:
        assert False, (
            f"BF16 partial-tile mismatch in {len(errors)} valid positions "
            f"(first 5 (row,col,expected,got): {errors[:5]})"
        )
    _verify_oob_sentinel(out, round_mt0, round_mt1, size_i, size_j, label="bf16")


def _verify_bf16_matrix_positions(out, round_mt0, round_mt1, size_i=None, size_j=None):
    """Verify bf16 output matches the serial input matrix down-cast to bf16 (RNE),
    and that every OOB position still holds the bf16 sentinel (0x0000FFFF as f32).

    Input matrix: D_ref[row][col] = float(col * round_mt0 + row) (column-major).
    Each output element at a valid position should equal bf16_rne(D_ref[row][col]).

    Args:
        out:        Flat tuple of float values from the GPU output buffer (bf16 values
                    upcast to float32 by zero-extending the mantissa).
        round_mt0:  Number of rows (leading dimension of the column-major buffer).
        round_mt1:  Number of cols.
        size_i:     Valid row count (defaults to round_mt0 — full tile).
        size_j:     Valid col count (defaults to round_mt1 — full tile).
    """
    if size_i is None:
        size_i = round_mt0
    if size_j is None:
        size_j = round_mt1
    D = np.array(out, dtype=np.float32).reshape(round_mt0, round_mt1, order='F')
    # The init matrix uses col * size_i + row as the serial value (stride = size_i).
    ref_f32 = np.fromfunction(lambda r, c: c * size_i + r,
                              (round_mt0, round_mt1), dtype=np.float32)
    ref = _bf16_rne(ref_f32)
    errors = []
    for c in range(size_j):
        for r in range(size_i):
            if D[r, c] != ref[r, c]:
                errors.append((r, c, float(ref[r, c]), float(D[r, c])))
    if errors:
        assert False, (
            f"BF16 matrix position mismatch in {len(errors)} valid elements "
            f"(first 5 (row,col,expected,got): {errors[:5]})"
        )
    _verify_oob_sentinel(out, round_mt0, round_mt1, size_i, size_j, label="bf16")


def _inject_bf16_permute(store_asm: str, perm_addr_reg: int, perm_tmp_regs: tuple) -> str:
    """Inject XOR-16 ds_bpermute shuffle + exec-masked stores into generated bf16 store assembly.

    With the "contiguous by wave" M-row layout and SVW=4, the codegen emits per
    store element (one d0 value, 4 accvgprs → 2 bf16 dwords):
        v_cvt_pk_bf16_f32 vD+0, acc[0], acc[1]  // rows r+0, r+1  (packed in dword)
        v_cvt_pk_bf16_f32 vD+1, acc[2], acc[3]  // rows r+2, r+3  (packed in dword)
        buffer_store_dwordx2 v[vD:vD+1], vAddr, s[SrdD:SrdD+3], ...

    Within a d0 element, different lane-groups (lg0: lanes 0-15, lg1: lanes 16-31,
    lg2: 32-47, lg3: 48-63 within the wave) compute different 4-row sub-blocks:
        lg0 → rows r+0..r+3
        lg1 → rows r+4..r+7
        lg2 → rows r+8..r+11
        lg3 → rows r+12..r+15

    An XOR-16 ds_bpermute gives each lane the data from its partner (lane XOR 16),
    which is the adjacent lane-group's 4 rows.  Exec masking (exec_lo=0x0000FFFF,
    exec_hi=0x0000FFFF = lanes {0-15, 32-47}) suppresses duplicate writes from
    lg1 and lg3.

    Two exec-masked dwordx2 stores cover 8 consecutive rows per element:
        lanes {0-15,32-47}: dwordx2 at vAddr       → rows r+0..r+3
        lanes {0-15,32-47}: dwordx2 at vAddr+8     → rows r+4..r+7 (from partner)

    For each buffer_store_dwordx2 this function inserts:
        v_xor_b32 v{vPA}, 16, v0               // partner lane = lane XOR 16
        v_lshlrev_b32 v{vPA}, 2, v{vPA}        // byte address = partner_lane * 4
        ds_bpermute_b32 v{T0}, v{vPA}, v{D0}   // lg0 ← lg1 dword 0
        ds_bpermute_b32 v{T1}, v{vPA}, v{D1}   // lg0 ← lg1 dword 1
        s_waitcnt lgkmcnt(0)
        s_mov_b32 exec_lo, 0x0000ffff          // lanes {0-15, 32-47} only
        s_mov_b32 exec_hi, 0x0000ffff
        buffer_store_dwordx2 v[D0:D1], vAddr, ...   // lg0's rows r+0..r+3
        v_add_u32 v{vAddr}, 8, v{vAddr}        // advance by 4 bf16 rows = 8 bytes
        buffer_store_dwordx2 v[T0:T1], vAddr, ...   // lg1's rows r+4..r+7
        v_add_u32 v{vAddr}, -8, v{vAddr}       // restore addr
        s_mov_b32 exec_lo, -1                  // restore full exec
        s_mov_b32 exec_hi, -1

    Args:
        store_asm:      Generated store module assembly string.
        perm_addr_reg:  Scratch vgpr for ds_bpermute address ((lane XOR 16)*4).
        perm_tmp_regs:  2-tuple (or more) of scratch vgprs for ds_bpermute output.

    Returns:
        Modified assembly string.
    """
    import re

    vPA = perm_addr_reg
    t0, t1 = perm_tmp_regs[0], perm_tmp_regs[1]

    # Match: buffer_store_dwordx2 v[D0:D1], vAddr, s[SrdD:SrdD+3], ...
    # Only the non-edge dwordx2 path (sgprSrdD) — skip edge short stores.
    store_pat = re.compile(
        r'(buffer_store_dwordx2\s+v\[(\d+):(\d+)\],\s*v(\d+),\s*s\[sgprSrdD[^\n]+)'
    )

    result_parts = []
    pos = 0
    for m in store_pat.finditer(store_asm):
        d0 = int(m.group(2))
        d1 = int(m.group(3))
        v_addr = int(m.group(4))
        # Only rewrite dwordx2 stores (d1 == d0+1)
        if d1 != d0 + 1:
            continue
        # Extract the store suffix (offset, glc, slc, etc.) after the vgpr range.
        orig_store_line = m.group(0)
        range_str = f"v[{d0}:{d1}]"
        suffix = orig_store_line[orig_store_line.index(range_str) + len(range_str):]
        permute_and_store = (
            f"  // --- bf16 XOR-16 permute for contiguous 8-row store ---\n"
            f"  v_xor_b32 v{vPA}, 16, v0             // partner lane = lane XOR 16\n"
            f"  v_lshlrev_b32 v{vPA}, 2, v{vPA}       // byte addr = partner_lane * 4\n"
            f"  ds_bpermute_b32 v{t0}, v{vPA}, v{d0}  // lg0 ← lg1 dword 0\n"
            f"  ds_bpermute_b32 v{t1}, v{vPA}, v{d1}  // lg0 ← lg1 dword 1\n"
            f"  s_waitcnt lgkmcnt(0)                   // wait ds_bpermute\n"
            f"  s_mov_b32 exec_lo, 0x0000ffff          // lanes {{0-15, 32-47}} only\n"
            f"  s_mov_b32 exec_hi, 0x0000ffff\n"
            f"  buffer_store_dwordx2 v[{d0}:{d1}]{suffix}\n"
            f"  v_add_u32 v{v_addr}, 8, v{v_addr}      // advance by 4 bf16 rows = 8 bytes\n"
            f"  buffer_store_dwordx2 v[{t0}:{t1}]{suffix}\n"
            f"  v_add_u32 v{v_addr}, -8, v{v_addr}     // restore addr\n"
            f"  s_mov_b32 exec_lo, -1                   // restore full exec\n"
            f"  s_mov_b32 exec_hi, -1\n"
            f"  // --- end bf16 XOR-16 permute ---\n"
        )

        chunk = store_asm[pos:m.start()]
        result_parts.append(chunk)
        result_parts.append(permute_and_store)
        pos = m.end()  # skip the original store line

    result_parts.append(store_asm[pos:])
    return "".join(result_parts)


def _run_storeD(cfg, tmp_path, size_i, size_j, mi_wave_group=None,
                num_threads=None, dump_asm=False, dump_store_insts=False,
                asm_output_dir=None, use_bf16=False, init_mode="matrix"):
    """Assemble and run the store-D roundtrip for the given tile and wave configuration.

    Args:
        cfg:           TileConfig (mt_a, mt_b, depth_u).
        tmp_path:      pytest tmp_path for output files.
        size_i:        Actual M dimension. Elements with coord0 >= size_i are not written.
        size_j:        Actual N dimension. Elements with coord1 >= size_j are not written.
        mi_wave_group: Optional [m, n] MIWaveGroup override (default: auto from tile size).
        num_threads:   Total thread count (default: NUM_THREADS = WAVESIZE * NUM_WAVES).
        init_mode:     "matrix" — load from full MT_a×MT_b host matrix (default);
                       "wave_id" — write float(wave_id) to every accvgpr.

    Returns:
        (out, tileInfoD, expected_set, round_mt0, round_mt1): output float tuple (sized
        round_mt0 * round_mt1), TileInfo for D, expected value set (None for wave_id),
        and the rounded buffer dimensions.
    """
    if num_threads is None:
        num_threads = NUM_THREADS

    kernel = _build_store_kernel(cfg, mi_wave_group=mi_wave_group, use_bf16=use_bf16)
    kernel["NumThreads"] = num_threads
    kernel["UseSubtileImpl"] = True

    writer, _, _, _ = create_writer(cfg, mi_wave_group=mi_wave_group)

    sgprs = _build_sgprs_for_test(writer)
    tileInfoD, agpr_indices = _allocate_d_tile(kernel, writer)

    kw = _build_kwa(kernel, writer, use_bf16=use_bf16)
    kw.states.d.tileInfo = tileInfoD

    kw.states.subtileM32ValidBlocksSgpr = sgprs["subtileMValidBlocks"]
    kw.states.subtileN16ValidBlocksSgpr = sgprs["subtileNValidBlocks"]
    kw.sgprs["SubtileMGuard"] = sgprs["subtileMValidBlocks"]
    kw.sgprs["SubtileNGuard"] = sgprs["subtileNValidBlocks"]
    kw.states.subtileMBlockSize         = 16

    tmp_v = writer.vgprPool.checkOut(1, "tmp_init", preventOverflow=False)

    kw.codes.accVgprRead = mapAcctoArchRegs(kernel, kw.states.maxLimitAgprs, write=False)
    store_indices_mod = kw.notLocalSplitUGlobalWriteIndices(kernel)
    kw.states.c.startVgprValu = 0
    store_write_mod, _ = kw.notLocalSplitUGlobalWrite(kernel, tPA=None, tPB=None)

    store_write_asm = str(store_write_mod)

    # Round buffer dimensions up to the nearest macrotile multiple so the full
    # macrotile footprint is allocated and initialized to the sentinel value.
    round_mt0 = ((size_i + cfg.mt_a - 1) // cfg.mt_a) * cfg.mt_a
    round_mt1 = ((size_j + cfg.mt_b - 1) // cfg.mt_b) * cfg.mt_b
    stride_d = round_mt0

    if init_mode == "wave_id":
        prologue = _build_prologue(sgprs, len(agpr_indices), cfg.mt_a, cfg.mt_b,
                                   stride_d=stride_d, use_input_buf=False)
        init_mod = _build_accvgpr_init_wave_id_asm(agpr_indices, tmp_v)
        args = [
            ("output",   8, "global_buffer", "u8"),
            ("size_i",   4, "by_value",      "u32"),
            ("size_j",   4, "by_value",      "u32"),
            ("stride_d", 4, "by_value",      "u32"),
        ]
        run_inputs  = ()
        expected_set = None
    elif init_mode == "random":
        vaddr = writer.vgprPool.checkOut(1, "vaddr", preventOverflow=False)
        vtmp2 = writer.vgprPool.checkOut(1, "vtmp2", preventOverflow=False)
        prologue = _build_prologue(sgprs, len(agpr_indices), cfg.mt_a, cfg.mt_b,
                                   stride_d=stride_d, use_input_buf=True)
        init_mod = _build_accvgpr_init_matrix_asm(agpr_indices, kernel, tileInfoD, sgprs,
                                                  tmp_v, vaddr, vtmp2)
        args = [
            ("input",    8, "global_buffer", "u8"),
            ("output",   8, "global_buffer", "u8"),
            ("size_i",   4, "by_value",      "u32"),
            ("size_j",   4, "by_value",      "u32"),
            ("stride_d", 4, "by_value",      "u32"),
        ]
        input_bytes, _, ref_arr = _compute_reference(cfg, kernel, tileInfoD, size_i, size_j,
                                                     rng=np.random.default_rng())
        run_inputs = (np.frombuffer(input_bytes, dtype=np.float32),)
        expected_set = ref_arr  # numpy float32 array; caller uses for positional verification
    else:
        vaddr = writer.vgprPool.checkOut(1, "vaddr", preventOverflow=False)
        vtmp2 = writer.vgprPool.checkOut(1, "vtmp2", preventOverflow=False)
        prologue = _build_prologue(sgprs, len(agpr_indices), cfg.mt_a, cfg.mt_b,
                                   stride_d=stride_d, use_input_buf=True)
        init_mod = _build_accvgpr_init_matrix_asm(agpr_indices, kernel, tileInfoD, sgprs,
                                                  tmp_v, vaddr, vtmp2)
        args = [
            ("input",    8, "global_buffer", "u8"),
            ("output",   8, "global_buffer", "u8"),
            ("size_i",   4, "by_value",      "u32"),
            ("size_j",   4, "by_value",      "u32"),
            ("stride_d", 4, "by_value",      "u32"),
        ]
        input_bytes, expected_set, _ = _compute_reference(cfg, kernel, tileInfoD,
                                                          size_i, size_j)
        run_inputs = (np.frombuffer(input_bytes, dtype=np.float32),)

    init_asm = str(init_mod)
    parts = [
        ".set BufferOOB, 0x80000000\n",
        ".set vgprValuC, 0\n",
        str(prologue),
        init_asm,
        "  s_nop 3  // 4 NOPs: satisfy CDNA3 acc write→read latency\n",
        str(store_indices_mod),
        store_write_asm,
    ]
    inner_asm = _finalize_inner_asm(parts, kw)

    wg = mi_wave_group or kernel["MIWaveGroup"]
    label = f"{cfg.label}_wg{'x'.join(str(g) for g in wg)}_m{size_i}n{size_j}"

    full_asm = generate_kernel_asm(inner_asm, writer, args, lds_size=0, num_threads=num_threads)

    if asm_output_dir is not None:
        import pathlib
        out_dir = pathlib.Path(asm_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        asm_path = str(out_dir / f"test_{label}.s")
        with open(asm_path, "w") as f:
            f.write(full_asm)
        print(f"\n[asm-output-dir] Full ASM written to: {asm_path}")

    if dump_asm:
        print(f"\n[dump_asm] Store indices module:\n{str(store_indices_mod)}")
        print(f"\n[dump_asm] Store write module (first 200 lines):\n" +
              "\n".join(store_write_asm.splitlines()[:200]))

    if dump_store_insts:
        _print_store_instructions(store_write_asm, label)

    bpeDest = 2 if use_bf16 else 4
    num_elems = round_mt0 * round_mt1
    output_size = num_elems * bpeDest

    raw = assemble_and_run(
        full_asm, tmp_path, label,
        output_size=output_size,
        inputs=run_inputs,
        scalars=(size_i, size_j, stride_d),
        num_threads=num_threads,
    )

    if use_bf16:
        raw_u16 = np.frombuffer(raw[:output_size], dtype=np.uint16)
        raw_u32 = raw_u16.astype(np.uint32) << 16
        out = tuple(v for v in raw_u32.view(np.float32))
    else:
        out = struct.unpack(f"{num_elems}f", raw[:num_elems * 4])

    return out, tileInfoD, expected_set, round_mt0, round_mt1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.label)
def test_storeD_mfma_layout(cfg, use_bf16, tmp_path):
    """GPU roundtrip: init accvgprs from a full MT_a×MT_b column-major host matrix.

    The host matrix D_ref is laid out column-major: D_ref[row][col] = col * MT_a + row.
    Each thread uses the full MFMA macrotile layout (wave_id0/1, subtile, lane_group, reg_k)
    to load its assigned element into the correct accvgpr.

    After store-D, every output element must equal col * MT_a + row at its (row, col)
    position — confirming that the full (lane, accvgpr, wave, subtile) → (row, col)
    mapping is correct end-to-end.  Tested for both fp32 and bf16 output dtypes.
    """
    init_rocisa()
    out, tileInfoD, expected_set, round_mt0, round_mt1 = _run_storeD(
        cfg, tmp_path, cfg.mt_a, cfg.mt_b, use_bf16=use_bf16,
    )
    if use_bf16:
        _verify_bf16_matrix_positions(out, round_mt0, round_mt1)
    else:
        _verify_output(out, expected_set)
        _verify_matrix_positions(out, round_mt0, round_mt1)


@pytest.mark.parametrize("cfg", BF16_ODD_MIWT_CONFIGS, ids=lambda c: c.label)
def test_storeD_bf16_odd_miwt(cfg, tmp_path):
    """BF16 store-D roundtrip for configs where MIWaveTile[0] is odd.

    When MIWaveTile[0] is odd the last (even) tt0 element has no sba=1 partner
    and is handled by _emit16bitSubtileScalarStore (the "orphan" store path).
    This test specifically targets that code path, which is not exercised by
    CONFIGS (all of which produce even MIWaveTile[0] values).

    The orphan store must:
      - write to the correct N-column (lane_id & 15, NOT the lane-group index)
      - write to the correct M-rows (LG*4+{0..3}, contiguous in memory)
      - NOT corrupt neighboring elements
    """
    init_rocisa()
    out, tileInfoD, expected_set, round_mt0, round_mt1 = _run_storeD(
        cfg, tmp_path, cfg.mt_a, cfg.mt_b, use_bf16=True,
    )
    _verify_bf16_matrix_positions(out, round_mt0, round_mt1)


@pytest.mark.parametrize("use_bf16", [False, True], ids=["f32", "bf16"])
@pytest.mark.parametrize("cfg,size_i,size_j", PARTIAL_CASES,
                         ids=[f"{c.label}_m{si}n{sj}" for c, si, sj in PARTIAL_CASES])
def test_storeD_partial_tile(cfg, size_i, size_j, use_bf16, tmp_path):
    """Store-D roundtrip for partial-tile dimensions, both fp32 and bf16 output dtypes.

    Uses random init so each valid position carries a unique random value in [-9, 9]
    and OOB positions are loaded with the sentinel (0xFFFFFFFF).  After store-D:
      - fp32: every position (valid and OOB) must match the reference bitwise.
      - bf16: every valid position must equal bf16_rne(ref); OOB are not checked
              because the bf16 unwritten sentinel (0xFFFF → 0xFFFF0000) differs from
              the f32 ref sentinel (0xFFFFFFFF) in bit pattern.
    """
    init_rocisa()
    out, tileInfoD, ref_arr, round_mt0, round_mt1 = _run_storeD(
        cfg, tmp_path, size_i, size_j, init_mode="random", use_bf16=use_bf16,
    )
    if use_bf16:
        _verify_bf16_random_matrix_positions(out, ref_arr, round_mt0, round_mt1, size_i, size_j)
    else:
        _verify_random_matrix_positions(out, ref_arr, round_mt0, round_mt1)


def _verify_bf16_positions(out, round_mt0, round_mt1, kernel, check_ncol=True, check_wave=True):
    """Positional correctness checks for bf16 store output.

    Two independent checks:

    check_ncol (acc_index init — value = float(k), k = accvgpr position within MMA tile):
      MI16x16x32 with MIOutputVW=4: v_cvt_pk_bf16_f32 packs consecutive k-values into
      bf16 dwords, so k maps to M-row within each 4-row group (SVW=4, dwordx2).
      With XOR-16 permute + exec masking, 8 rows are written per element, but the
      k-to-M-row mapping is preserved: D[i, j] == float(i % 4) for every written element.

    check_wave (wave_id init — value = float(wave_id)):
      Each 16×16 MMA tile in D belongs to exactly one wave.  Every written
      (non-sentinel) element within a given MMA tile must hold the same wave_id
      value, and that value must be a valid wave index in [0, num_waves).
      (Does not require every element in a tile to be written — coverage gaps
      from SVW batching are acceptable.)
    """
    import numpy as np
    D = np.array(out, dtype=np.float32).reshape(round_mt0, round_mt1, order='F')

    if check_ncol:
        errors = []
        for i in range(round_mt0):
            exp = float(i % 4)
            for j in range(round_mt1):
                v = float(D[i, j])
                if _is_sentinel(v):
                    continue
                if v != exp:
                    errors.append((i, j, v, exp))
        assert not errors, (
            f"M-row routing wrong: {len(errors)} mismatches "
            f"(first 10: {errors[:10]})"
        )

    if check_wave:
        num_waves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
        valid_wave_ids = set(float(w) for w in range(num_waves))
        mma_m = kernel["MatrixInstM"]   # 16
        mma_n = kernel["MatrixInstN"]   # 16
        errors = []
        for bi in range(round_mt0 // mma_m):
            for bj in range(round_mt1 // mma_n):
                tile = D[bi*mma_m:(bi+1)*mma_m, bj*mma_n:(bj+1)*mma_n]
                written = [float(v) for v in tile.ravel() if not _is_sentinel(float(v))]
                if not written:
                    continue
                unique = set(written)
                if len(unique) != 1:
                    errors.append((bi, bj, f"mixed values in tile: {unique}"))
                elif unique.pop() not in valid_wave_ids:
                    errors.append((bi, bj, f"invalid wave_id {written[0]} in tile"))
        assert not errors, (
            f"Wave-tile uniformity wrong: {len(errors)} bad tiles "
            f"(first 5: {errors[:5]})"
        )


def _build_accvgpr_init_matrix_asm(agpr_indices, kernel, tileInfoD, sgprs, tmp_v, vaddr, vtmp2):
    """Init accvgprs from a column-major MT_a×MT_b host matrix using the MFMA output layout.

    Buffer layout: column-major float32, flat[col * MT_a + row] = D_ref[row][col].
    Each thread computes its (row, col) for each accvgpr using the full macrotile layout:

      lane_id    = tid % WAVESIZE
      wave_id    = tid // WAVESIZE
      wave_id0   = wave_id % MIWaveGroup[0]        (M dimension)
      wave_id1   = wave_id // MIWaveGroup[0]        (N dimension)
      col_in_wave = lane_id % mma_n                 (0..15)
      lane_group  = lane_id // mma_n                (0..3, gives 4-row groups)

      For vtile index v (sId0 = v // localSubtileGrid[1], sId1 = v % localSubtileGrid[1]):
        col = wave_id1 * wave_cols + sId1 * mma_n + col_in_wave
        row = wave_id0 * wave_rows + sId0 * mma_m + lane_group * 4 + reg_k

      column-major byte offset = (col * MT_a + row) * 4

    The dynamic base is computed once into vaddr; per-vtile static byte offsets are applied
    incrementally. vtiles are visited in allocation order: sId0 outer, sId1 inner.

    Requires three scratch vgprs: tmp_v, vaddr, vtmp2.
    """
    srd_in = sgprs["SrdInput"]

    MIWaveGroup = kernel["MIWaveGroup"]
    wg0, wg1    = MIWaveGroup
    MT_a        = kernel["MacroTile0"]
    mma_m       = kernel["MatrixInstM"]   # 16
    mma_n       = kernel["MatrixInstN"]   # 16
    regs_per_subtile = int(tileInfoD.mmaTileRegCount)  # 4

    # Dimensions of each wave's block and per-wave subtile grid.
    wave_rows = MT_a // wg0                              # rows owned by one wave
    wave_cols = kernel["MacroTile1"] // wg1              # cols owned by one wave
    local_sg0 = int(tileInfoD.localSubtileGrid[0])        # sId0 range
    local_sg1 = int(tileInfoD.localSubtileGrid[1])        # sId1 range

    stride_bytes = MT_a * 4   # bytes per column in the column-major matrix

    # Phase 1: compute the per-thread dynamic base byte offset into vaddr.
    #
    # col_byte_dyn = (wave_id1 * wave_cols + col_in_wave) * stride_bytes
    # row_byte_dyn = (wave_id0 * wave_rows + lane_group * 4) * 4
    # base_byte    = col_byte_dyn + row_byte_dyn
    #
    # Register usage: vaddr = col_byte_dyn (then base_byte), tmp_v = row scratch,
    # vtmp2 = col_in_wave and lane_group intermediates.
    log2_ws  = WAVESIZE.bit_length() - 1
    log2_wg0 = (wg0).bit_length() - 1   # assumes wg0 is power of 2

    m = Module("accvgpr_init_matrix")
    m.add(TextBlock(
        # vaddr = wave_id1 = tid >> (log2_ws + log2_wg0)
        f"  v_lshrrev_b32 v{vaddr}, {log2_ws + log2_wg0}, v0  // wave_id1\n"
        # vaddr = wave_id1 * wave_cols
        f"  v_mul_u32_u24 v{vaddr}, {wave_cols}, v{vaddr}      // wave_id1 * {wave_cols}\n"
        # vtmp2 = col_in_wave = lane_id & 15 = v0 & 15
        f"  v_and_b32 v{vtmp2}, 15, v0                          // col_in_wave\n"
        # vaddr = wave_id1 * wave_cols + col_in_wave
        f"  v_add_u32 v{vaddr}, v{vtmp2}, v{vaddr}              // + col_in_wave\n"
        # vaddr = col_byte_dyn = above * stride_bytes
        f"  v_mul_u32_u24 v{vaddr}, {stride_bytes}, v{vaddr}   // col_byte_dyn\n"
        # tmp_v = wave_id = tid >> log2_ws
        f"  v_lshrrev_b32 v{tmp_v}, {log2_ws}, v0               // wave_id\n"
        # tmp_v = wave_id0 = wave_id & (wg0-1)
        f"  v_and_b32 v{tmp_v}, {wg0 - 1}, v{tmp_v}             // wave_id0\n"
        # tmp_v = wave_id0 * wave_rows
        f"  v_mul_u32_u24 v{tmp_v}, {wave_rows}, v{tmp_v}       // wave_id0 * {wave_rows}\n"
        # vtmp2 = lane_group = (lane_id >> 4); lane_id = tid & (WAVESIZE-1), bits[5:4]
        f"  v_lshrrev_b32 v{vtmp2}, 4, v0                        // lane_id >> 4\n"
        f"  v_and_b32 v{vtmp2}, 3, v{vtmp2}                      // lane_group (0..3)\n"
        # vtmp2 = lane_group * 4  (each lane_group spans 4 rows)
        f"  v_lshlrev_b32 v{vtmp2}, 2, v{vtmp2}                  // lane_group * 4\n"
        # tmp_v = wave_id0 * wave_rows + lane_group * 4
        f"  v_add_u32 v{tmp_v}, v{vtmp2}, v{tmp_v}               // + lane_group*4\n"
        # tmp_v = row_byte_dyn = (wave_id0*wave_rows + lane_group*4) * 4
        f"  v_lshlrev_b32 v{tmp_v}, 2, v{tmp_v}                  // row_byte_dyn (*4)\n"
        # vaddr = base_byte = col_byte_dyn + row_byte_dyn
        f"  v_add_u32 v{vaddr}, v{tmp_v}, v{vaddr}               // base_byte\n"
    ))

    # Phase 2: walk vtiles in allocation order: sId1-outer, sId0-inner
    # (getLocalMMATileLinearId: i = sId1 * localMMATileGrid[0] + sId0)
    prev_static = 0
    num_vtiles = len(agpr_indices) // regs_per_subtile
    for vtile_idx in range(num_vtiles):
        sId0 = vtile_idx % local_sg0
        sId1 = vtile_idx // local_sg0
        # static per-vtile offset from the wave's base:
        #   col contribution: sId1 * mma_n * stride_bytes
        #   row contribution: sId0 * mma_m * 4
        static_off = sId1 * mma_n * stride_bytes + sId0 * mma_m * 4
        delta = static_off - prev_static
        if delta != 0:
            m.add(TextBlock(
                f"  v_add_u32 v{vaddr}, {delta}, v{vaddr}  // subtile ({sId0},{sId1})\n"
            ))
        prev_static = static_off
        base = vtile_idx * regs_per_subtile
        for k in range(regs_per_subtile):
            agpr = agpr_indices[base + k]
            # reg k within the subtile is at row_byte_dyn + k*4 (via imm offset)
            m.add(TextBlock(
                f"  buffer_load_dword v{tmp_v}, v{vaddr}, s[{srd_in}:{srd_in+3}], 0 offen offset:{k*4}\n"
                f"  s_waitcnt vmcnt(0)\n"
                f"  v_accvgpr_write_b32 a{agpr}, v{tmp_v}\n"
            ))
    return m


def _build_accvgpr_init_wave_id_asm(agpr_indices, tmp_v):
    """Init all accvgprs to float(wave_id).

    Every thread in the same wave writes the same value — useful for visually
    verifying which MMA tiles belong to which wave.
    """
    log2_ws = WAVESIZE.bit_length() - 1
    m = Module("accvgpr_init_wave_id")
    m.add(TextBlock(
        f"  v_lshrrev_b32 v{tmp_v}, {log2_ws}, v0  // wave_id = tid >> {log2_ws}\n"
        f"  v_cvt_f32_u32 v{tmp_v}, v{tmp_v}        // float(wave_id)\n"
    ))
    for agpr in agpr_indices:
        m.add(TextBlock(f"  v_accvgpr_write_b32 a{agpr}, v{tmp_v}\n"))
    return m


def _verify_random_matrix_positions(out, ref_arr, round_mt0, round_mt1):
    """Verify D output matches the random reference matrix at every position.

    Compares uint32 bit patterns so that the NaN sentinel (0xFFFFFFFF) compares
    equal to itself.  Both valid positions (should carry the stored random value)
    and OOB positions (should remain as sentinel) are checked.

    Args:
        out:        Flat tuple/sequence of float values from the GPU output buffer.
        ref_arr:    numpy float32 array of shape (round_mt0 * round_mt1,) in
                    column-major order — the reference C matrix built by
                    _compute_reference(..., rng=...).
        round_mt0:  Leading dimension (rows) of the column-major output buffer.
        round_mt1:  Number of columns of the output buffer.
    """
    D_u32   = np.array(out, dtype=np.float32).view(np.uint32).reshape(round_mt0, round_mt1, order='F')
    ref_u32 = ref_arr.view(np.uint32).reshape(round_mt0, round_mt1, order='F')
    bad = np.argwhere(D_u32 != ref_u32)
    if len(bad):
        ref_f32 = ref_arr.reshape(round_mt0, round_mt1, order='F')
        D_f32   = np.array(out, dtype=np.float32).reshape(round_mt0, round_mt1, order='F')
        errors  = [(int(r), int(c), float(ref_f32[r, c]), float(D_f32[r, c]))
                   for r, c in bad[:5]]
        assert False, (
            f"Random-init matrix mismatch in {len(bad)} positions "
            f"(first 5 (row,col,expected,got): {errors})"
        )


# ---------------------------------------------------------------------------
# Beta-path OOB guard test (UseSubtileImpl NonEdge).
#
# The SrdC+2 gating fix prevents OOB waves from reading garbage from C when
# beta != 0.  In the NonEdge path, M % MT != 0 but M % subtile_block == 0,
# so some waves are entirely OOB in M or N.  Without the fix those waves load
# from C and corrupt D; with it, SrdC+2 is set to 0 (hardware returns zero)
# for OOB elements before each beta C load.
#
# Test strategy: alpha=0, beta=1, accvgprs=0, C=serial values.
#   D_expected = 0*alpha + 1*C = C   at valid positions
#   D_expected = sentinel            at OOB positions (C load gated → 0 + 0 = 0,
#                                    but store is also OOB-redirected → sentinel)
#
# We verify:
#   1. Every valid (row,col) in D equals the corresponding C value.
#   2. No OOB position is corrupted (remains sentinel or is legitimately 0).
# ---------------------------------------------------------------------------

# MT=64×64, MIWaveTile=[2,2] → MIWaveGroup=[2,2], 4 waves.
# Wave layout: wave_rows=32 (MatrixInstM*MIWaveTile[0]=16*2), wave_cols=32.
#
# Guard values per (M, N) — each wave owns 2 d0 (M) and 2 d1 (N) steps:
#
#   numValidD1Steps  (M guard, per wave in M dim):
#     M=16  → wave-0: ceil(16/16)=1,  wave-1: 0        guard values {0,1}
#     M=32  → wave-0: 2,              wave-1: 0        guard values {0,2}
#     M=48  → wave-0: 2,              wave-1: ceil(16/16)=1  guard values {1,2}
#     M=64  → wave-0: 2,              wave-1: 2        guard values {2,2}
#
#   numValid16NBlocks (N guard, per wave in N dim):
#     N=16  → wave-0-N: 1,            wave-1-N: 0      guard values {0,1}
#     N=32  → wave-0-N: 2,            wave-1-N: 0      guard values {0,2}
#     N=48  → wave-0-N: 2,            wave-1-N: 1      guard values {1,2}
#     N=64  → wave-0-N: 2,            wave-1-N: 2      guard values {2,2}
#
# Cases cover every distinct guard value combination for M and N:
_BETA_CFG = TileConfig(mt_a=64, mt_b=64, depth_u=512)
BETA_OOB_CASES = [
    # --- M guard boundary cases ---
    (_BETA_CFG, 64, 64),   # M guard={2,2}, N guard={2,2}: baseline, all waves valid
    (_BETA_CFG, 32, 64),   # M guard={2,0}, N guard={2,2}: wave-1 fully OOB in M
    (_BETA_CFG, 16, 64),   # M guard={1,0}, N guard={2,2}: wave-0 partial (d0=0 only), wave-1 OOB
    (_BETA_CFG, 48, 64),   # M guard={2,1}, N guard={2,2}: wave-1 partial (d0=0 only)
    # --- N guard boundary cases ---
    (_BETA_CFG, 64, 32),   # M guard={2,2}, N guard={2,0}: wave-1-N fully OOB in N
    (_BETA_CFG, 64, 16),   # M guard={2,2}, N guard={1,0}: wave-0-N partial (d1=0 only), wave-1-N OOB
    (_BETA_CFG, 64, 48),   # M guard={2,2}, N guard={2,1}: wave-1-N partial (d1=0 only)
    # --- combined M+N guard cases ---
    (_BETA_CFG, 32, 32),   # M guard={2,0}, N guard={2,0}: wave-1 and wave-1-N fully OOB
    (_BETA_CFG, 48, 48),   # M guard={2,1}, N guard={2,1}: both dims partial at wave-1
    (_BETA_CFG, 16, 16),   # M guard={1,0}, N guard={1,0}: both dims partial at wave-0, wave-1 OOB
    (_BETA_CFG, 32, 48),   # M guard={2,0}, N guard={2,1}: M-OOB + N-partial corner
    (_BETA_CFG, 48, 32),   # M guard={2,1}, N guard={2,0}: M-partial + N-OOB corner
]


def _build_sgprs_for_beta_test(writer):
    """Extend the store-D sgpr layout with Beta and subtile guard SGPRs.

    Builds on _build_sgprs_for_test and adds:
      Beta       (1 reg, s[28]): beta scalar (1.0f)
      mGuard     (1 reg, s[29]): subtileM32ValidBlocksSgpr — numValidD1Steps
      nGuard     (1 reg, s[30]): subtileN16ValidBlocksSgpr — numValid16NBlocks

    Returns the sgprs dict (same reference as writer.sgprs).
    """
    sgprs = _build_sgprs_for_test(writer)

    beta = writer.sgprPool.checkOut(1, "Beta")
    writer.sgprs["Beta"] = beta

    mGuard = writer.sgprPool.checkOut(1, "subtileMValidBlocks")
    writer.sgprs["subtileMValidBlocks"] = mGuard

    nGuard = writer.sgprPool.checkOut(1, "subtileNValidBlocks")
    writer.sgprs["subtileNValidBlocks"] = nGuard

    return sgprs


def _build_beta_prologue(sgprs, mt0, mt1, size_i, size_j, mi_wave_tile, c_stride=None):
    """Prologue for the beta-path OOB guard test.

    Loads two buffers from kernargs:
      kernarg[0:8]   = C buffer ptr  (SrdC base)
      kernarg[8:16]  = D buffer ptr  (SrdD base)
      kernarg[16:20] = size_i
      kernarg[20:24] = size_j
      kernarg[24:28] = stride (= mt0, the buffer leading dimension)

    Computes the subtile M/N guard SGPRs statically: because WG=(0,0) and the
    geometry is fixed at test-build time, we precompute the values for each wave
    in Python and write them via s_cselect (conditioned on waveIdM and waveIdN).

    Guard semantics (matching _emitSubtileGuards):
      mGuard = numValidD1Steps  = min(ceil(max(validM-waveBase,0)/MatrixInstM), MIWaveTile[0])
      nGuard = numValid16NBlocks = min(max(validN-waveBaseN,0), waveGroupN) >> 4
      mOff   = uninitialized (GlobalWriteBatch fills it per d0 group via s_cselect)

    For WG0=WG1=0:
      validM = size_i,  validN = size_j
      waveBase(waveIdM) = waveIdM * waveGroupM  (waveGroupM = MatrixInstM * MIWaveTile[0])
      waveBaseN(waveIdN) = waveIdN * waveGroupN
    """
    from rocisa.instruction import SLoadB64, SLoadB32, SMovB32, SLShiftRightB32, SAndB32

    miM       = 16               # MatrixInstM
    miN       = 16               # MatrixInstN
    wg0, wg1  = 2, 2             # MIWaveGroup (4 waves)
    miwt0     = mi_wave_tile[0]  # MIWaveTile[0]
    miwt1     = mi_wave_tile[1]  # MIWaveTile[1]
    waveGroupM = miM * miwt0     # rows per wave in M
    waveGroupN = miN * miwt1     # cols per wave in N

    m = Module("beta_prologue")

    # Load pointers and scalars from kernargs.
    # kernargs layout: [C_ptr(8), D_ptr(8), size_i(4), size_j(4), stride(4)]
    srd_c = sgprs["SrdC"]
    srd_d = sgprs["SrdD"]
    m.add(SLoadB64(dst=sgpr(srd_c, 2), base=sgpr(0, 2), soffset=0,
                   comment="C ptr → SrdC[0:1]"))
    m.add(SLoadB64(dst=sgpr(srd_d, 2), base=sgpr(0, 2), soffset=8,
                   comment="D ptr → SrdD[0:1]"))
    m.add(SLoadB32(dst=sgpr(sgprs["SizeI"]),    base=sgpr(0, 2), soffset=16,
                   comment="size_i → SizeI"))
    m.add(SLoadB32(dst=sgpr(sgprs["SizeJ"]),    base=sgpr(0, 2), soffset=20,
                   comment="size_j → SizeJ"))
    m.add(SLoadB32(dst=sgpr(sgprs["StrideDJ"]), base=sgpr(0, 2), soffset=24,
                   comment="stride → StrideDJ"))
    m.add(SWaitCnt(dscnt=0, comment="wait sloads"))

    # Set up SrdC: num_records = BufferOOB (real kernel restores this; here we start open).
    m.add(SMovB32(dst=sgpr(srd_c + 2), src="0x80000000", comment="SrdC num_records = BufferOOB"))
    m.add(SMovB32(dst=sgpr(srd_c + 3), src="0x20000",    comment="SrdC format word"))

    # SrdD: same setup.
    m.add(SMovB32(dst=sgpr(srd_d + 2), src="0x80000000", comment="SrdD num_records = BufferOOB"))
    m.add(SMovB32(dst=sgpr(srd_d + 3), src="0x20000",    comment="SrdD format word"))

    # Scalar constants.
    m.add(SMovB32(dst=sgpr(sgprs["WorkGroup0"]), src=0,  comment="WorkGroup0 = 0"))
    m.add(SMovB32(dst=sgpr(sgprs["WorkGroup1"]), src=0,  comment="WorkGroup1 = 0"))
    if c_stride is not None:
        m.add(SMovB32(dst=sgpr(sgprs["StrideCJ"]), src=c_stride,
                      comment=f"StrideCJ = {c_stride} (tight C stride for page-fault test)"))
    else:
        m.add(SMovB32(dst=sgpr(sgprs["StrideCJ"]), src=sgpr(sgprs["StrideDJ"]), comment="StrideCJ = StrideDJ"))
    m.add(SMovB32(dst=sgpr(sgprs["NumWorkGroups0"]), src=1, comment="NumWorkGroups0 = 1"))
    m.add(SMovB32(dst=sgpr(sgprs["NumWorkGroups1"]), src=1, comment="NumWorkGroups1 = 1"))
    # Alpha = 0.0f: accvgprs are 0, so alpha*acc = 0 regardless; but set to 1.0f for clarity.
    # The test uses accvgprs=0, so D = alpha*0 + beta*C = beta*C = C (with beta=1).
    m.add(SMovB32(dst=sgpr(sgprs["Alpha"]),     src="0x3f800000", comment="Alpha = 1.0f"))
    m.add(SMovB32(dst=sgpr(sgprs["Alpha"] + 1), src="0x3f800000", comment="Alpha[1] = 1.0f"))
    # Beta = 1.0f.
    m.add(SMovB32(dst=sgpr(sgprs["Beta"]), src="0x3f800000", comment="Beta = 1.0f"))

    # Compute subtile M/N guard SGPRs using SGPR arithmetic (matches _emitSubtileGuards).
    # For WG=(0,0): validM = SizeI, validN = SizeJ.
    # waveId = tid >> 6; waveIdM = waveId & 1; waveIdN = waveId >> 1.
    # We use a scratch vgpr (v0 = tid) to read waveId into an SGPR via v_readfirstlane.
    mGuard = sgprs["subtileMValidBlocks"]
    nGuard = sgprs["subtileNValidBlocks"]

    # Emit the guard computation matching _emitSubtileGuards logic.
    # Use TextBlock since we need vgpr reads that are easier to express as raw asm.
    import math
    log2_wg0    = int(math.log2(wg0))
    miMShift    = int(math.log2(miM))

    m.add(TextBlock(
        # --- waveId extraction ---
        f"  v_readfirstlane_b32 s{mGuard}, v0          // lane0 tid\n"
        f"  s_lshr_b32 s{mGuard}, s{mGuard}, 6         // waveId = tid >> 6\n"
        f"  s_lshr_b32 s{nGuard}, s{mGuard}, {log2_wg0} // waveIdN = waveId >> {log2_wg0}\n"
        f"  s_and_b32  s{mGuard}, s{mGuard}, {wg0-1}   // waveIdM = waveId & {wg0-1}\n"
        # --- M guard: numValidD1Steps ---
        # waveBase = waveIdM * waveGroupM
        f"  s_mul_i32  s{mGuard}, s{mGuard}, {waveGroupM} // waveBase = waveIdM * {waveGroupM}\n"
        # remainder = max(validM - waveBase, 0)  using SizeI for validM (WG0=0)
        f"  s_sub_u32  s{mGuard}, s{sgprs['SizeI']}, s{mGuard} // remainder = SizeI - waveBase; SCC=1 if OOB\n"
        f"  s_cselect_b32 s{mGuard}, 0, s{mGuard}      // clamp to 0 if OOB\n"
        # ceil(remainder / miM)
        f"  s_add_u32  s{mGuard}, s{mGuard}, {miM-1}   // ceil: + {miM-1}\n"
        f"  s_lshr_b32 s{mGuard}, s{mGuard}, {miMShift} // numValidD1Steps = >> {miMShift}\n"
        # min(result, MIWaveTile[0])
        f"  s_min_u32  s{mGuard}, s{mGuard}, {miwt0}   // clamp to MIWaveTile[0]={miwt0}\n"
        # --- N guard: numValid16NBlocks ---
        # waveBaseN = waveIdN * waveGroupN
        f"  s_mul_i32  s{nGuard}, s{nGuard}, {waveGroupN} // waveBaseN = waveIdN * {waveGroupN}\n"
        # remainder = max(validN - waveBaseN, 0)  using SizeJ for validN (WG1=0)
        f"  s_sub_u32  s{nGuard}, s{sgprs['SizeJ']}, s{nGuard} // validN - waveBaseN; SCC=1 if OOB\n"
        f"  s_cselect_b32 s{nGuard}, 0, s{nGuard}      // clamp to 0 if OOB\n"
        # clamp to waveGroupN
        f"  s_sub_u32  s{sgprs['Alpha']+1}, s{nGuard}, {waveGroupN} // SCC=1 if < waveGroupN\n"
        f"  s_cselect_b32 s{nGuard}, s{nGuard}, {waveGroupN} // min(validN_wave, waveGroupN)\n"
        f"  s_lshr_b32 s{nGuard}, s{nGuard}, 4          // numValid16NBlocks = >> 4\n"
        # Restore Alpha[1] = 1.0f (reused as tmp above).
        f"  s_mov_b32  s{sgprs['Alpha']+1}, 0x3f800000  // restore Alpha[1] = 1.0f\n"
    ))

    return m


def _build_accvgpr_zero_asm(agpr_indices, tmp_v):
    """Zero-initialize all accvgprs (simulates a GEMM result of 0)."""
    m = Module("accvgpr_zero")
    m.add(TextBlock(f"  v_mov_b32 v{tmp_v}, 0           // zero\n"))
    for agpr in agpr_indices:
        m.add(TextBlock(f"  v_accvgpr_write_b32 a{agpr}, v{tmp_v}\n"))
    return m


def _run_storeD_beta(cfg, tmp_path, size_i, size_j, mi_wave_group=None, dump_asm=False):
    """Assemble and run the store-D beta-path test.

    Sets up UseSubtileImpl=1, UseBeta=True, alpha=1, beta=1, accvgprs=0.
    Loads C from a column-major buffer; D should equal C at valid (row,col)
    positions and remain sentinel at OOB positions (SrdC+2 gating prevents
    OOB C loads, and SrdD OOB-redirects the store).

    Returns:
        (out, c_arr, round_mt0, round_mt1):
            out       — flat float32 tuple from the D output buffer.
            c_arr     — numpy float32 array (column-major, size round_mt0*round_mt1)
                        used as the reference C values at valid positions.
            round_mt0 — buffer row count (= mt0 = 64 for this test)
            round_mt1 — buffer col count (= mt1 = 64)
    """
    if mi_wave_group is None:
        mi_wave_group = [2, 2]

    # Build kernel with UseSubtileImpl=1 and UseBeta=True.
    kernel = _build_store_kernel(cfg, mi_wave_group=mi_wave_group)
    kernel["UseSubtileImpl"] = True
    kernel["ProblemType"]["UseBeta"] = True
    kernel["NumThreads"] = NUM_THREADS

    writer, _, _, _ = create_writer(cfg, mi_wave_group=mi_wave_group)
    sgprs = _build_sgprs_for_beta_test(writer)

    tileInfoD, agpr_indices = _allocate_d_tile(kernel, writer)
    kw = _build_kwa(kernel, writer)
    kw.states.d.tileInfo = tileInfoD

    # Wire subtile guard SGPRs into kw.states so GlobalWriteBatch can use them.
    kw.states.subtileM32ValidBlocksSgpr = sgprs["subtileMValidBlocks"]
    kw.states.subtileN16ValidBlocksSgpr = sgprs["subtileNValidBlocks"]
    kw.sgprs["SubtileMGuard"] = sgprs["subtileMValidBlocks"]
    kw.sgprs["SubtileNGuard"] = sgprs["subtileNValidBlocks"]
    kw.states.subtileMBlockSize         = 16  # MatrixInstM for f32

    tmp_v = writer.vgprPool.checkOut(1, "tmp_v", preventOverflow=False)

    kw.codes.accVgprRead = mapAcctoArchRegs(kernel, kw.states.maxLimitAgprs, write=False)
    store_indices_mod = kw.notLocalSplitUGlobalWriteIndices(kernel)
    kw.states.c.startVgprValu = 0
    store_write_mod, _ = kw.notLocalSplitUGlobalWrite(kernel, tPA=None, tPB=None)

    round_mt0 = cfg.mt_a
    round_mt1 = cfg.mt_b
    stride_d  = round_mt0

    # C buffer: column-major, value at (row, col) = float(col * round_mt0 + row).
    # Positions outside (size_i, size_j) use the NaN sentinel so that any OOB
    # load that bypasses the guard would produce an obviously wrong value in D.
    _SENTINEL_F32 = struct.unpack('f', struct.pack('I', 0xFFFFFFFF))[0]
    c_flat = []
    for col in range(round_mt1):
        for row in range(round_mt0):
            if row < size_i and col < size_j:
                c_flat.append(float(col * round_mt0 + row))
            else:
                c_flat.append(_SENTINEL_F32)
    c_arr = np.array(c_flat, dtype=np.float32)

    prologue = _build_beta_prologue(sgprs, round_mt0, round_mt1, size_i, size_j,
                                    mi_wave_tile=kernel["MIWaveTile"])
    init_mod = _build_accvgpr_zero_asm(agpr_indices, tmp_v)

    store_write_asm = str(store_write_mod)
    parts = [
        ".set BufferOOB, 0x80000000\n",
        ".set vgprValuC, 0\n",
        str(prologue),
        str(init_mod),
        "  s_nop 3  // CDNA3 acc write→read latency\n",
        str(store_indices_mod),
        store_write_asm,
    ]
    inner_asm = _finalize_inner_asm(parts, kw)

    wg = mi_wave_group
    label = f"beta_{cfg.label}_wg{'x'.join(str(g) for g in wg)}_m{size_i}n{size_j}"

    args = [
        ("c_input",  8, "global_buffer", "u8"),
        ("output",   8, "global_buffer", "u8"),
        ("size_i",   4, "by_value",      "u32"),
        ("size_j",   4, "by_value",      "u32"),
        ("stride_d", 4, "by_value",      "u32"),
    ]

    full_asm = generate_kernel_asm(inner_asm, writer, args, lds_size=0, num_threads=NUM_THREADS)

    if dump_asm:
        print(f"\n[dump_asm] {label}")
        print(f"Store indices:\n{str(store_indices_mod)}")
        print(f"Store write (first 200 lines):\n" +
              "\n".join(store_write_asm.splitlines()[:200]))

    output_size = round_mt0 * round_mt1 * 4  # f32 bytes
    raw = assemble_and_run(
        full_asm, tmp_path, label,
        output_size=output_size,
        inputs=(c_arr,),
        scalars=(size_i, size_j, stride_d),
        num_threads=NUM_THREADS,
    )
    out = struct.unpack(f"{round_mt0 * round_mt1}f", raw[:output_size])
    return out, c_arr, round_mt0, round_mt1


@pytest.mark.parametrize("cfg,size_i,size_j", BETA_OOB_CASES,
                         ids=[f"{c.label}_m{si}n{sj}" for c, si, sj in BETA_OOB_CASES])
def test_storeD_beta_oob_guard(cfg, size_i, size_j, tmp_path):
    """Beta-path OOB guard: SrdC+2 gating prevents OOB waves from loading garbage from C.

    Geometry: MT=64×64, MIWaveTile=[2,2], UseSubtileImpl=1.
    Wave layout: 4 waves (MIWaveGroup=[2,2]), wave_rows=32, wave_cols=32.

    Scenario: accvgprs=0, alpha=1, beta=1.
      D = alpha*acc + beta*C = 0 + 1*C = C   at valid positions
      D = sentinel                            at OOB positions (store OOB-redirected)

    The SrdC+2 gating fix (GlobalWriteBatch.py) must ensure that OOB waves do
    NOT load C (which would produce garbage) and instead use 0, so that the
    final D value at valid positions is exactly C.

    Without the fix, OOB waves load garbage from C, then the store for valid
    elements of the same wave sees the corrupted acc value and writes wrong data.
    With the fix: SrdC+2=0 returns 0 for all C bytes of OOB waves, so D=C
    at valid positions and the OOB store is suppressed by the SrdD OOB redirect.
    """
    init_rocisa()
    out, c_arr, round_mt0, round_mt1 = _run_storeD_beta(cfg, tmp_path, size_i, size_j)

    D_u32   = np.array(out,   dtype=np.float32).view(np.uint32).reshape(round_mt0, round_mt1, order='F')
    ref_u32 = c_arr.view(np.uint32).reshape(round_mt0, round_mt1, order='F')

    D_f32 = np.array(out, dtype=np.float32).reshape(round_mt0, round_mt1, order='F')
    C_f32 = c_arr.reshape(round_mt0, round_mt1, order='F')

    errors = []
    for col in range(size_j):
        for row in range(size_i):
            if D_u32[row, col] != ref_u32[row, col]:
                errors.append((row, col, float(C_f32[row, col]), float(D_f32[row, col])))
    assert not errors, (
        f"Beta OOB guard mismatch at {len(errors)} valid positions "
        f"(first 5 (row,col,expected_C,got): {errors[:5]})"
    )
    _verify_oob_sentinel(out, round_mt0, round_mt1, size_i, size_j, label="f32")


# ---------------------------------------------------------------------------
# SrdC+2 C-load OOB guard test via page-fault detection
#
# Allocates the C buffer with tight stride (size_i) so that the last valid
# element sits exactly at a page boundary.  The next page is mprotect'd
# PROT_NONE and not registered with HIP, so it has no IOMMU mapping.
#
# If SrdC+2 gating is broken (OOB C loads go through), the OOB wave's load
# for the last valid N column hits the guard page → GPU IOMMU fault →
# hipDeviceSynchronize() returns a non-zero error code.
#
# If the fix is in place (SrdC+2=0 for OOB elements), the hardware returns
# 0 without touching memory and the kernel completes cleanly.
# ---------------------------------------------------------------------------

import ctypes
import mmap as mmap_module
import resource as resource_module

# Cases where OOB C loads reliably reach the guard page (verified empirically).
# For M=16/48 the per-thread MI layout keeps addresses within the allocated
# buffer even when SrdC+2 gating is disabled, so those cases can't trigger
# the hardware fault — they are covered by test_storeD_beta_oob_guard instead.
CLOAD_OOB_CASES = [
    (_BETA_CFG, 32, 64),   # wave-1-M fully OOB → last col hits guard
    (_BETA_CFG, 64, 32),   # wave-1-N fully OOB → last row hits guard
    (_BETA_CFG, 64, 16),   # wave-0-N partially OOB → hits guard
    (_BETA_CFG, 64, 48),   # wave-1-N partially OOB → hits guard
    (_BETA_CFG, 32, 32),   # M and N both OOB → hits guard
    (_BETA_CFG, 32, 48),   # M OOB + N partial → hits guard
]

# Cases where the per-thread MI address layout stays within the allocated
# buffer even without SrdC+2 gating.  These should never trigger a page fault
# regardless of whether the gate is present.
CLOAD_SAFE_CASES = [
    # Fully in-bounds baseline: M%MT0==0, N%MT1==0 → all waves valid.
    (_BETA_CFG, 64, 64),

    # M Edge-path (M%MT0 ∉ {0, 32}): edgeProtectCode handles masking, not SrdC+2.
    # The page-fault mechanism cannot distinguish broken SrdC+2 gating on these
    # cases — they are covered by test_storeD_beta_oob_guard (sentinel check).
    (_BETA_CFG, 16, 64),   # M=16: edge path for M
    (_BETA_CFG, 48, 64),   # M=48: edge path for M
    (_BETA_CFG, 48, 48),   # M=48, N=48: both edge path
    (_BETA_CFG, 16, 16),   # M=16, N=16: both edge path

    # NonEdge path (M%MT0==32 or N%MT1==32): SrdC+2 gate is emitted.
    # M/N does not divide MT; the OOB stride offset overshoots the single guard
    # page window (guard page = c_pages..c_pages+PAGE_SIZE), so no IOMMU fault
    # occurs even when SrdC+2 gating is absent.
    (_BETA_CFG, 32, 16),   # M%MT0==32 → NonEdge; N=16 small partial (N%MT1≠32)
    (_BETA_CFG, 16, 32),   # N%MT1==32 → NonEdge; M=16 edge (M%MT0≠32)
    (_BETA_CFG, 48, 32),   # N%MT1==32 → NonEdge; M=48 edge (M%MT0≠32)
]

CLOAD_OOB_CASES = [
    (_BETA_CFG, 32, 64),   # wave-1-M fully OOB → last col hits guard
    (_BETA_CFG, 64, 32),   # wave-1-N fully OOB → last row hits guard
    (_BETA_CFG, 64, 16),   # wave-0-N partially OOB → hits guard
    (_BETA_CFG, 64, 48),   # wave-1-N partially OOB → hits guard
    (_BETA_CFG, 32, 32),   # M and N both OOB → hits guard
    (_BETA_CFG, 32, 48),   # M OOB + N partial → hits guard
    # storeAlign8: M%8==0 (but M%16!=0) takes NonEdge with partial-LG exec mask.
    # C-load SrdC+2 gate operates at 16-row MMA tile granularity — verify no OOB.
    (_BETA_CFG, 8,  64),   # M=8:  1 valid LG in wave-0, wave-1 fully OOB
    (_BETA_CFG, 24, 64),   # M=24: partial within wave-0 (8 valid rows in 2nd tile)
    (_BETA_CFG, 40, 64),   # M=40: wave-0 full, wave-1 partial (8 valid rows)
    (_BETA_CFG, 56, 64),   # M=56: wave-0 full, wave-1 partial (24 valid rows)
    (_BETA_CFG, 8,  13),   # M=8 + arbitrary N=13: both M and N partial
    (_BETA_CFG, 40, 5),    # M=40 + N=5: partial M + very small N
    (_BETA_CFG, 64, 3),    # full M + tiny N=3: arbitrary N boundary
    (_BETA_CFG, 64, 13),   # full M + N=13: arbitrary N partial
    (_BETA_CFG, 8,  1),    # M=8 + N=1: minimal partial in both dims
]


def _run_storeD_cload_pagefault(cfg, tmp_path, size_i, size_j, mi_wave_group=None,
                                dump_asm=False):
    """Assemble and run the C-load page-fault OOB detection test.

    C buffer layout (tight stride = size_i):
      - Valid region: size_i × size_j f32 elements, column-major
      - Last valid byte sits exactly at a page boundary
      - The next page is PROT_NONE and has no HIP/IOMMU mapping

    An OOB C load (SrdC+2 not gated to 0) for the last N column will access
    the guard page → GPU IOMMU fault → non-zero hipDeviceSynchronize() result.

    Returns:
        (err_code, int): The raw hipDeviceSynchronize() return value.
                         0 = no fault (test should pass), non-zero = fault detected.
    """
    PAGE_SIZE = resource_module.getpagesize()

    # Initialize device first so hipHostRegister has a valid context even after
    # a hipDeviceReset() in a previous test iteration.
    hip_check(hip.hipInit(0))
    hip_check(hip.hipSetDevice(0))

    if mi_wave_group is None:
        mi_wave_group = [2, 2]

    kernel = _build_store_kernel(cfg, mi_wave_group=mi_wave_group)
    kernel["UseSubtileImpl"] = True
    kernel["ProblemType"]["UseBeta"] = True
    kernel["NumThreads"] = NUM_THREADS

    writer, _, _, _ = create_writer(cfg, mi_wave_group=mi_wave_group)
    sgprs = _build_sgprs_for_beta_test(writer)

    tileInfoD, agpr_indices = _allocate_d_tile(kernel, writer)
    kw = _build_kwa(kernel, writer)
    kw.states.d.tileInfo = tileInfoD

    kw.states.subtileM32ValidBlocksSgpr = sgprs["subtileMValidBlocks"]
    kw.states.subtileN16ValidBlocksSgpr = sgprs["subtileNValidBlocks"]
    kw.sgprs["SubtileMGuard"] = sgprs["subtileMValidBlocks"]
    kw.sgprs["SubtileNGuard"] = sgprs["subtileNValidBlocks"]
    kw.states.subtileMBlockSize         = 16

    tmp_v = writer.vgprPool.checkOut(1, "tmp_v", preventOverflow=False)

    kw.codes.accVgprRead = mapAcctoArchRegs(kernel, kw.states.maxLimitAgprs, write=False)
    store_indices_mod = kw.notLocalSplitUGlobalWriteIndices(kernel)
    kw.states.c.startVgprValu = 0
    store_write_mod, _ = kw.notLocalSplitUGlobalWrite(kernel, tPA=None, tPB=None)

    round_mt0 = cfg.mt_a
    round_mt1 = cfg.mt_b

    # Use tight stride (size_i) for C so OOB accesses in the last column
    # spill past the buffer end and into the guard page.
    prologue = _build_beta_prologue(sgprs, round_mt0, round_mt1, size_i, size_j,
                                    mi_wave_tile=kernel["MIWaveTile"],
                                    c_stride=size_i)
    init_mod = _build_accvgpr_zero_asm(agpr_indices, tmp_v)

    store_write_asm = str(store_write_mod)
    parts = [
        ".set BufferOOB, 0x80000000\n",
        ".set vgprValuC, 0\n",
        str(prologue),
        str(init_mod),
        "  s_nop 3\n",
        str(store_indices_mod),
        store_write_asm,
    ]
    inner_asm = _finalize_inner_asm(parts, kw)

    label = f"cload_pg_{cfg.label}_m{size_i}n{size_j}"
    args = [
        ("c_input",  8, "global_buffer", "u8"),
        ("output",   8, "global_buffer", "u8"),
        ("size_i",   4, "by_value",      "u32"),
        ("size_j",   4, "by_value",      "u32"),
        ("stride_d", 4, "by_value",      "u32"),
    ]

    full_asm = generate_kernel_asm(inner_asm, writer, args, lds_size=0, num_threads=NUM_THREADS)

    if dump_asm:
        print(f"\n[dump_asm] {label}")
        print(store_write_asm[:2000])

    # Assemble to code object.
    co_path = str(tmp_path / f"test_{label}.co")
    assemble_kernel(full_asm, co_path)

    # --- Guarded C buffer ---
    # Tight C layout: size_i × size_j f32, column-major, stride=size_i.
    valid_bytes = size_i * size_j * 4
    c_pages = ((valid_bytes + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    total   = c_pages + PAGE_SIZE  # valid region + 1 guard page

    mm = mmap_module.mmap(-1, total,
                          flags=mmap_module.MAP_SHARED | mmap_module.MAP_ANONYMOUS,
                          prot=mmap_module.PROT_READ | mmap_module.PROT_WRITE)
    libc    = ctypes.CDLL("libc.so.6", use_errno=True)
    buf_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))

    # Protect the guard page (CPU PROT_NONE; also not registered with HIP).
    ret = libc.mprotect(ctypes.c_void_p(buf_ptr + c_pages),
                        ctypes.c_size_t(PAGE_SIZE), ctypes.c_int(0))
    assert ret == 0, f"mprotect failed errno={ctypes.get_errno()}"

    # Register valid pages with HIP so the GPU can access them via IOMMU.
    # hipHostRegister expects a plain int for the host pointer.
    hip_check(hip.hipHostRegister(buf_ptr, c_pages, int(hip.hipHostRegisterDefault)))

    # Shift C data to end exactly at the page boundary.
    # data_ptr = buf_ptr + c_pages - valid_bytes → last byte at buf_ptr+c_pages-1.
    data_offset = c_pages - valid_bytes
    host_data_ptr = buf_ptr + data_offset

    # Fill valid C region (column-major, stride=size_i).
    c_np = np.arange(size_i * size_j, dtype=np.float32)
    ctypes.memmove(host_data_ptr, c_np.ctypes.data, valid_bytes)

    dev_c_ptr = hip_check(hip.hipHostGetDevicePointer(host_data_ptr, 0))

    # --- D output buffer (regular hipMalloc, round_mt0 stride) ---
    output_size = round_mt0 * round_mt1 * 4
    d_output    = hip_check(hip.hipMalloc(output_size))
    hip_check(hip.hipMemset(d_output, 0xff, output_size))

    # --- Launch kernel ---
    module = hip_check(hip.hipModuleLoad(co_path.encode()))
    kern   = hip_check(hip.hipModuleGetFunction(module, b"test_kernel"))

    stride_d = round_mt0

    class KernArgs(ctypes.Structure):
        _fields_ = [
            ("c_ptr",    ctypes.c_uint64),
            ("out_ptr",  ctypes.c_uint64),
            ("size_i",   ctypes.c_uint32),
            ("size_j",   ctypes.c_uint32),
            ("stride_d", ctypes.c_uint32),
        ]

    kargs      = KernArgs(int(dev_c_ptr), int(d_output),
                          size_i, size_j, stride_d)
    kargs_size = ctypes.c_size_t(ctypes.sizeof(kargs))
    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(0x01),
        ctypes.c_void_p(ctypes.addressof(kargs)),
        ctypes.c_void_p(0x02),
        ctypes.c_void_p(ctypes.addressof(kargs_size)),
        ctypes.c_void_p(0x03),
    )

    hip_check(hip.hipModuleLaunchKernel(kern, 1, 1, 1, NUM_THREADS, 1, 1,
                                        0, None, None, extra))

    # Do NOT use hip_check here: a page-fault returns non-zero and we want to
    # observe the error rather than raise immediately.
    sync_result = hip.hipDeviceSynchronize()
    if isinstance(sync_result, tuple):
        err_code = sync_result[0]
    else:
        err_code = sync_result

    # Cleanup.  On GPU fault the context is poisoned; hipDeviceReset recovers it
    # but invalidates all existing allocations, so skip further HIP calls.
    if err_code != 0:
        hip.hipDeviceReset()
        # hipHostUnregister would fail on the reset context; skip it.
        # The mmap will be closed below which releases the host memory.
    else:
        hip_check(hip.hipFree(d_output))
        hip_check(hip.hipModuleUnload(module))
        hip.hipHostUnregister(buf_ptr)  # best-effort; ignore return value
    mm.close()

    return err_code


@pytest.mark.parametrize("cfg,size_i,size_j", CLOAD_OOB_CASES,
                         ids=[f"{c.label}_m{si}n{sj}" for c, si, sj in CLOAD_OOB_CASES])
def test_storeD_cload_pagefault(cfg, size_i, size_j, tmp_path):
    """SrdC+2 gating: OOB C loads must not reach memory (verified via page-fault).

    C buffer is tight-stride (stride=size_i) with a guard page immediately
    after the last valid byte.  With the fix (SrdC+2=0 for OOB elements) no
    memory access crosses the boundary and the kernel completes cleanly.

    Without the fix (SrdC+2=BufferOOB for OOB elements), the last-column OOB
    load hits the guard page → GPU IOMMU fault → non-zero sync error.
    """
    init_rocisa()
    err = _run_storeD_cload_pagefault(cfg, tmp_path, size_i, size_j)
    assert err == 0, (
        f"GPU page fault detected (err={err}): OOB C load crossed guard page. "
        f"SrdC+2 gating is broken for size_i={size_i}, size_j={size_j}."
    )


@pytest.mark.parametrize("cfg,size_i,size_j", CLOAD_SAFE_CASES,
                         ids=[f"{c.label}_m{si}n{sj}" for c, si, sj in CLOAD_SAFE_CASES])
def test_storeD_cload_pagefault_safe(cfg, size_i, size_j, tmp_path):
    """Guard-page test for cases where MI layout keeps accesses in-bounds.

    These sizes never trigger a fault even without SrdC+2 gating, so the
    page-fault mechanism cannot detect a broken gate here.  The test verifies
    that the infrastructure itself doesn't spuriously fault on safe inputs.
    """
    init_rocisa()
    err = _run_storeD_cload_pagefault(cfg, tmp_path, size_i, size_j)
    assert err == 0, (
        f"Unexpected GPU fault (err={err}) on safe-case size_i={size_i}, size_j={size_j}. "
        "The guard page was hit even though all C accesses should be in-bounds."
    )


# ---------------------------------------------------------------------------
# CLI-driven test: pytest test_storeD_roundtrip.py --m 21 32 --n 23 32
#                                                   --mt0 16 32 --mt1 16 32
#                                                   --wave-config 1,1 2,2
# ---------------------------------------------------------------------------

def _parse_pairs(raw, option_name):
    """Parse a list of 'A,B' strings into [[A, B], ...] int pairs."""
    result = []
    for s in raw:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"{option_name} entries must be 'A,B', got: {s!r}")
        result.append([int(parts[0]), int(parts[1])])
    return result


def test_storeD_cli(request, tmp_path):
    """Store-D roundtrip for all combinations of (M,N), (MT0,MT1), and wave config pairs.

    Driven entirely by CLI options; skipped when none are provided.
    Each option takes a list of pairs; the test runs the Cartesian product.

    Example:
        pytest test_storeD_roundtrip.py -s \\
            --mn 23,17 32,32 \\
            --mt 16,16 32,32 \\
            --wave-config 1,1 2,2
    """
    mn_raw           = request.config.getoption("--mn",               default=None)
    mt_raw           = request.config.getoption("--mt",               default=None)
    wc_raw           = request.config.getoption("--wave-config",      default=None)
    dump_asm         = request.config.getoption("--dump-asm",         default=False)
    dump_store_insts = request.config.getoption("--dump-store-insts", default=False)
    asm_output_dir   = request.config.getoption("--asm-output-dir",   default=None)
    subtile_map      = request.config.getoption("--subtile-map",      default=False)
    init_mode        = request.config.getoption("--init-mode",        default="matrix")
    print_ref        = request.config.getoption("--print-ref",        default=False)
    dtype            = request.config.getoption("--dtype",            default="fp32")
    use_bf16         = (dtype == "bf16")

    mn_list      = _parse_pairs(mn_raw, "--mn")          if mn_raw else []
    mt_list      = _parse_pairs(mt_raw, "--mt")          if mt_raw else []
    wave_configs = _parse_pairs(wc_raw, "--wave-config") if wc_raw else []

    if not mn_list or not mt_list or not wave_configs:
        pytest.skip("specify --mn, --mt, and --wave-config to run")

    init_rocisa()

    for (mt0, mt1), wg, (size_i, size_j) in itertools.product(mt_list, wave_configs, mn_list):
        num_threads = wg[0] * wg[1] * WAVESIZE
        cfg = TileConfig(mt_a=mt0, mt_b=mt1, depth_u=64)
        label = f"MT={mt0}x{mt1} WG={wg} M={size_i} N={size_j} init={init_mode} dtype={dtype}"
        print(f"\n--- {label} ---")
        out, tileInfoD, expected_set, round_mt0, round_mt1 = _run_storeD(
            cfg, tmp_path, size_i, size_j,
            mi_wave_group=wg,
            num_threads=num_threads,
            dump_asm=dump_asm,
            dump_store_insts=dump_store_insts,
            asm_output_dir=asm_output_dir,
            use_bf16=use_bf16,
            init_mode=init_mode,
        )
        if init_mode == "matrix":
            if use_bf16:
                # For bf16, only check valid positions; OOB sentinel differs in bit pattern.
                _verify_bf16_matrix_positions(out, round_mt0, round_mt1, size_i=size_i, size_j=size_j)
            else:
                _verify_matrix_positions(out, round_mt0, round_mt1, size_i=size_i, size_j=size_j)
        elif init_mode == "random":
            if use_bf16:
                _verify_bf16_random_matrix_positions(out, expected_set, round_mt0, round_mt1, size_i, size_j)
            else:
                _verify_random_matrix_positions(out, expected_set, round_mt0, round_mt1)
        if print_ref and init_mode == "random":
            _print_ref_matrix(expected_set, round_mt0, round_mt1, label)
        if subtile_map:
            _print_subtile_map(round_mt0, round_mt1, out, tileInfoD, label)
        else:
            _print_full_matrix(out, round_mt0, round_mt1, label)
