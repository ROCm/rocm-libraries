################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import math
from functools import lru_cache

from .Architectures import SUPPORTED_ISA

################################################################################
# Enumerate Valid Solution Parameters
################################################################################

validThreadTileSides = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + list(
    range(20, 256, 4)
)
validThreadTiles = []
for i in validThreadTileSides:
    for j in validThreadTileSides:
        validThreadTiles.append([i, j])

validActivationFormats = ("NCHW", "NHWC", "CNHW", "NCDHW", "NDHWC", "CNDHW")
validWeightFormats = ("KCYX", "KYXC", "CKYX", "CYXK", "KCZYX", "CKZYX", "CZYXK")
validMacroTileSides = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    6,
    12,
    24,
    48,
    96,
    192,
    384,
    768,
]
validMacroTiles = []
validISA = [(0, 0, 0)]
validISA.extend(SUPPORTED_ISA)
for i in validMacroTileSides:
    for j in validMacroTileSides:
        validMacroTiles.append([i, j])
validTT = 32

@lru_cache
def makeValidWorkGroups():
    validWorkGroups = []
    for numThreads in range(32, 1025, 32):
        for nsg in [1, 2, 4, 8, 16, 32, 64, 96, 128, 256]:
            for sg0 in range(1, numThreads // nsg + 1):
                sg1 = numThreads // nsg // sg0
                if sg0 * sg1 * nsg == numThreads:
                    workGroup = [sg0, sg1, nsg]
                    validWorkGroups.append(workGroup)
    return validWorkGroups

def makeValidWMMA():
    return [[16, 16, 16, 1]]

@lru_cache
def makeValidMFMA():
    validMFMA = {}
    validMFMA["H"] = [[32, 32, 16, 1], [32, 32, 4, 2], [32, 32, 8, 1], [16, 16, 32, 1], [16, 16, 4, 4], [16, 16, 16, 1], [4, 4, 4, 16]]
    validMFMA["S"] = [[32, 32, 1, 2], [32, 32, 2, 1], [16, 16, 1, 4], [16, 16, 4, 1], [4, 4, 1, 16], [16, 16, 32, 1], [32, 32, 16, 1]]
    validMFMA["B"] = [[32, 32, 16, 1], [32, 32, 2, 2], [32, 32, 4, 1], [16, 16, 32, 1], [16, 16, 2, 4], [16, 16, 8, 1], [4, 4, 2, 16]]
    validMFMA["4xi8"] = [
        [32, 32, 4, 2],
        [32, 32, 8, 1],
        [16, 16, 4, 4],
        [16, 16, 16, 1],
        [4, 4, 4, 16],
        [32, 32, 16, 1],
        [16, 16, 32, 1],
    ]
    validMFMA["D"] = [[16, 16, 4, 1], [4, 4, 4, 4]]
    validMFMA["B1k"] = [[32, 32, 4, 2], [32, 32, 8, 1], [16, 16, 4, 4], [16, 16, 16, 1], [4, 4, 4, 16]]
    validMFMA["C"] = validMFMA["S"]
    validMFMA["Z"] = validMFMA["D"]
    validMFMA["I8"] = [
        [32, 32, 4, 2],
        [32, 32, 8, 1],
        [16, 16, 4, 4],
        [16, 16, 16, 1],
        [4, 4, 4, 16],
    ] + [[32, 32, 16, 1], [16, 16, 32, 1],] + [[16, 16, 64, 1],[32, 32, 32, 1]]
    validMFMA["X"] = [[32, 32, 4, 1], [16, 16, 8, 1], [16, 16, 16, 1], [16, 16, 32, 1], [32, 32, 16, 1]]
    validMFMA["F8"] = [[32, 32, 16, 1], [16, 16, 32, 1], [32, 32, 64, 1], [16, 16, 128, 1]]
    validMFMA["B8"] = validMFMA["F8"]
    validMFMA["F8B8"] = validMFMA["F8"]
    validMFMA["B8F8"] = validMFMA["F8"]
    validMFMA["F8N"] = [[32, 32, 16, 1], [16, 16, 32, 1]]
    validMFMA["B8N"] = validMFMA["F8N"]
    validMFMA["F8B8N"] = validMFMA["F8N"]
    validMFMA["B8F8N"] = validMFMA["F8N"]
    validMFMA["_format9"] = []

    for MFMA in [
        validMFMA["H"],
        validMFMA["S"],
        validMFMA["B"],
        validMFMA["D"],
        validMFMA["X"],
        validMFMA["F8N"],
        makeValidWMMA(),
    ]:
        for MI in MFMA:
            for bm in range(int(math.log(MI[3], 2)) + 1):
                for tt0 in range(1, validTT + 1):
                    for tt1 in range(1, validTT + 1):
                        for wave_m in range(3):
                            for wave_n in range(3):
                                validMFMA["_format9"].append(
                                    [MI[0], MI[1], MI[2], MI[3], 2**bm, tt0, tt1, 2**wave_m, 2**wave_n]
                                )
    return validMFMA

@lru_cache
def makeValidSMFMA():
    validSMFMA = {}
    validSMFMA["H"] = [[32, 32, 16, 1], [16, 16, 32, 1], [16, 16, 64, 1], [32, 32, 32, 1]]
    validSMFMA["B"] = [[32, 32, 16, 1], [16, 16, 32, 1], [16, 16, 64, 1], [32, 32, 32, 1]]
    validSMFMA["4xi8"] = [[32, 32, 32, 1], [16, 16, 64, 1], [16, 16, 128, 1], [32, 32, 64, 1]]
    validSMFMA["I8"] = validSMFMA["4xi8"]
    validSMFMA["F8"] = [[32, 32, 32, 1], [16, 16, 64, 1], [16, 16, 128, 1], [32, 32, 64, 1]]
    validSMFMA["B8"] = validSMFMA["F8"]
    validSMFMA["F8B8"] = validSMFMA["F8"]
    validSMFMA["B8F8"] = validSMFMA["F8"]
    validSMFMA["F8N"] = validSMFMA["F8"]
    validSMFMA["B8N"] = validSMFMA["F8"]
    validSMFMA["F8B8N"] = validSMFMA["F8N"]
    validSMFMA["B8F8N"] = validSMFMA["F8N"]
    validSMFMA["_format9"] = []
    for SMFMA in [validSMFMA["H"], validSMFMA["B"], validSMFMA["4xi8"], validSMFMA["F8N"]]:
        for MI in SMFMA:
            for bm in range(int(math.log(MI[3], 2)) + 1):
                for tt0 in range(1, validTT + 1):
                    for tt1 in range(1, validTT + 1):
                        for wave_m in range(3):
                            for wave_n in range(3):
                                validSMFMA["_format9"].append(
                                    [MI[0], MI[1], MI[2], MI[3], 2**bm, tt0, tt1, 2**wave_m, 2**wave_n]
                                )
    return validSMFMA

@lru_cache
def makeValidMatrixInstructions():
    mfma = makeValidMFMA()
    smfma = makeValidSMFMA()
    validMatrixInstructions = (
        [[], [-1]]
        + mfma["H"]
        + mfma["S"]
        + mfma["B"]
        + mfma["D"]
        + mfma["B1k"]
        + mfma["X"]
        + smfma["H"]
        + smfma["B"]
        + smfma["4xi8"]
    )
    return validMatrixInstructions + mfma["_format9"] + smfma["_format9"]


validParameters = { # we need to make sure this matches develop
    # 0: Global read is along parallel direction in thread level,
    #     each load instruction stride whole threads.
    #                         ----> perp
    #       | [w0,  w0,  w1,w1,w2,w2,w3,w3,  w0,  w0, w1,w1,w2,w2,w3,w3]
    #       | [ t0,t32] [                 ] [ t0,t32] [                ]
    #  para | [ t1,t33] [  wave 1,2,3     ] [ t1,t33] [ wave 1,2,3     ]
    #       | [ .., ..] [                 ] [ .., ..] [                ]
    #       | [t31,t63] [                 ] [t31,t63] [                ]
    #       V [-load_1]                      [-load_2]
    #
    # 1: Each wave load a block of memory,
    #     each load instruction stride 64 threads.
    #                         ----> perp
    #         [ w0, w0,  w0, w0, w1,w1,w1,w1, w2,w2,w2,w2, w3,w3,w3,w3]
    #       | [ t0,t32][ t0,t32]
    #  para | [ t1,t33][ t1,t33]
    #       | [ .., ..][ .., ..]
    #       | [t31,t63][t31,t63]
    #       V [-load_1][-load_2]
    #
    #
    # 2: Each load instruction spread threads evenly in the perp direction
    #                         ----> perp
    #       |  [w0, w1, w2, w3, w0, w1, w2, w3, w0, w1, w2, w3, w0, w1, w2, w3]
    #       |  [t0 ]           [t0 ]           [t32]           [t32]
    #  para |  [t1 ]           [t1 ]           [t33]           [t33]
    #       |  [.. ]           [.. ]           [.. ]           [.. ]
    #       |  [t31]           [t31]           [t63]           [t63]
    #       V [load_1]        [load_2]        [load_1]        [load_2]
    #
    "WaveSeparateGlobalReadA": [0, 1, 2],
    "WaveSeparateGlobalReadB": [0, 1, 2],
    # Add an unrolled loop and NGLL loop with swapped GRA and GRB order.
    # which may change the tlb thrashing behavior.
    "UnrollLoopSwapGlobalReadOrder": [0, 1],
    # PrefetchGlobalRead = 1:
    # Requires 2X LDS space, and VGPRs for buffering data on way into LDS
    #   prefetch / double-buffer reads from global memory -> vgprs -> lds.
    #
    # PrefetchGlobalRead = 2:
    # Do another prefetch while writing data from vgpr to lds.
    #   prefetch / double-buffer reads from global memory -> vgprs --> lds.
    #                                                              |-> prefetch reads
    "PrefetchGlobalRead": [0, 1, 2],
    # number of iteration prefetch local reads from lds to VGPRs buffer = PLR
    "PrefetchLocalRead": list(range(128 + 1)),
    # MatrixInstruction Only
    # If set ClusterLocalRead, each iteration dedicated vgprBuffer for localRead
    # So we can schedule these localReads to the front of the loop
    "ClusterLocalRead": [0, 1],
    # We use double LDS buffer when PrefetchGlobalRead.
    # While it reads data from LDS[0]/[1], it prefetch global data and writes to LDS[1]/[0]
    # If we can make sure all data are read from LDS to register before writing data to LDS, we can use 1 LDS buffer to save LDS memory.
    # this can help to generate Kernel that LDS usage originally exceed MaxLDS if using double LDS buffer,
    # or help to increase Occupancy.
    #     1 means: Force to use 1 LDS Buffer even with PrefetchGlobalRead
    #    -1 means: generator will use 1 LDS buffer only when LDS exceed MaxLDS
    # Use case:
    #    SIA2: 1LDSBuffer is set to 1 natively
    #    SIA3: 1LDSBuffer works only when PGR=True
    # TODO: optimize scheduling to support more cases.
    "1LDSBuffer": [-1, 0, 1],
    # Split the unroll summation into multiple sections and combine the sections
    # GSU applies only to the unroll summation dimension
    # Set to 0 to disable GSU, kernel code will be generated without GSU support
    # Set to -1 to choose GSU automatically in runtime, determined by function calculateAutoGSU
    "GlobalSplitU": list(range(-1, 1024 + 1)),
    # choose how to do GlobalSplitU
    # 1: use atomic operation to accumulate on one buffer
    # 2: each GSU group write to each own buffer and accumulate by another kernel
    # 3: each GSU group write to each own buffer and accumulate by same kernel
    "GlobalSplitUAlgorithm": ["SingleBuffer", "MultipleBuffer", "MultipleBufferSingleKernel"],
    # don't create a whole copy of the Unroll loop with loads removed - instead
    # use buffer limits to suppress global loads and ignore unnecessary ds_reads
    "SuppressNoLoadLoop": [False, True],
    # For PrefetchGlobalRead=1, create a second copy of the unroll loop with
    # the LDS pointer swaps expanded into inline constants for LDS read and write instructions
    # This eliminates 4 vector XOR instructions used for pointer swap
    "ExpandPointerSwap": [False, True],
    # Schedule global reads and global read increments into LocalRead iterations
    # Can reduce pressure on local read instruction dispatch queue
    # 0=perform global reads at start of instruction loop
    # 1=schedule into the local read instruction iterations
    "ScheduleGlobalRead": [0, 1],
    # Schedule local writes into LocalRead iterations.
    # Can reduce pressure on local read instruction dispatch queue
    "ScheduleLocalWrite": [0, 1],
    # Scheduling algorithm to use for each iteration:
    # 0 = minimal/no scheduling.  Global Read and increments, followed by local reads,
    # followed by local writes, followed by MACs
    "ScheduleIterAlg": [0, 1, 2, 3],
    # For MatrixInstruction and SIA3, number of GlobalReadInstruction between mfma
    # the purpose of this parameter is to control density of global read instruction scheduling
    # Scheduling global read back to back can have better memory efficiency
    # However, when full of vmem FIFO, it will block other instruction to be issued
    # Range from 0.01 to 32
    #         0.1 means 1 GR per 10 mfma
    #           5 means 5 GR per 1 mfma
    "GlobalReadPerMfma": [i / 100 for i in range(1, 3200)],
    #
    # For MatrixInstruction and SIA3, number of LocalWriteInstruction between mfma
    # the purpose of this parameter is to control density of local write instruction scheduling
    # In PGR1, we want to schedule local write more denser, so we can have more
    #          latency to hide global read
    # In PGR2, since LW is followed by GR, every LW has same whole loop latency
    #          to hide global read. We want to schedule LW less denser, can
    #          avoid full of vmem FIFO.
    # Range from 0.01 to 32
    #         0.1 means 1 LW per 10 mfma
    #           5 means 5 LW per 1 mfma
    # -1 will derived an optimized value internally
    # -2 will derived an optimized value and override LWPM silently (debug only, not recommended)
    "LocalWritePerMfma": [i / 100 for i in range(1, 3200)] + [-1],
    # Interleave alpha scale calculation with beta loads and address calcs - rather
    # than as a separate block of instructions
    "InterleaveAlpha": [0, 1],
    # Create a copy of NoLoadLoop which interleaves the stores with the final mac
    # calculation and may perform other optimizations
    # 0 = no interleave
    # 1 = interleave one stores after required macs have completed execution
    # 2 = interleave two stores after required macs have completed execution
    "OptNoLoadLoop": [0, 1, 2],
    "BufferLoad": [False, True],
    "BufferStore": [False, True],
    # Attempt to load directly from global memory into Vgpr.
    # Assembly only
    "DirectToVgprA": [False, True],
    "DirectToVgprB": [False, True],
    "DirectToVgprSparseMetadata": [False, True],
    # Attempt to load directly from global memory into LDS.
    # Assembly only
    # Requires BufferLoad, assembler support for lds modifier on buffer
    # loads (checked automatically), GlobalVectorWidth=1 (this is hw
    # requirement) and A/B must not require any transpose.
    # DirectToLds reduces load latency and eliminates the
    # G2L registers used to stage data.  Also replaces the
    # local write offset with an SGPR.
    # For an 8x8 TT with PrefetchGlobalRead=1 this can save 33 VGPRs.
    #    - Requirements for DirectToLds=1:
    #      GlobalReadVectorWidth = 1/2/4 (GRVW * bpe must be 4 for now)
    #      TransposeLDS = 1 for TLU=0 case
    # DirectToLds support for x1 only for now
    "DirectToLds": [False, True],
    # Load options:
    # (GRO = Global Read Offset)
    # BufferLoad=0:
    #  = Use flat instructions with 64 bit GRO for each load
    #    + supports sizes up to 2^64
    #    - uses many VGPR for addressing
    #    - uses execmask+compares for edge detection
    #    - generates extra LDS traffic (could convert flat->global load)
    # BufferLoad=1:
    #  = Use buffer load instructions with 32-bit offset
    #    + Less VGPRS (32b offset vs 64-bit) needed for addressing
    #    + Uses hardware buffer limit for edge detection
    #    - Limited range - the bot-right corner of macro-tile (plus padding=GRVW
    #        for shift-pointer, if ShiftPtr is required) must be within 2^32.
    #      ShiftPtrPad = MayShift ? GRWV*BPE : 0
    #      For TLU=1: Unroll*StrideA1 + ShiftPtrPad <= 2^32
    #      For TLU=0: MT*StrideA1 + ShiftPtrPad <= 2^32
    #      These conditions should be checked using Assert - TODO
    #  = UseSgprForGRO=1:
    #    + Attempt to use SGPR for Global Read Offsets.
    #    + Use one VGPR base GRO + many SGPR GRO rather than many VGPR GRO.
    #    + Each SGPR stores an offset from base GlobalReadOffset+0.
    #    - Requirements for UseSgprForGRO=1:
    #      - BufferLoad=1
    #      - Use appropriate Assert*ElementMultiple or GRVW=1 to eliminate need for ShifPtr
    #        (UseSgprForGRO does not support ShiftPtr since ShiftPtr needs to potentially shift GRO)
    #  = KernelWriterAssembly also supports 64-bit 2D buffer size (see use64bPbcLimit)
    #    - Requires 4 instructions to move scalar limit and a couple SGPR
    #    - Enabled by default.  If the overhead matters we can add asserts/YAML parm to specialize
    #  = UseInstOffsetForGRO=1:
    #    + Attempt to use Instruction offset for Global Read Offsets.
    #    + This feature avoid updating m0 for subsequent GRO(s) for directToLds feature
    #    - Requirements for UseInstOffsetForGRO=1:
    #      - BufferLoad=1
    #      - DirectToLds=1
    #  converting m0 update from LocalWriteAddrSGpr using  is usually win
    # -1 attempt to use a heuristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseInstOffsetForGRO": [-1, 0, 1],
    # Converting VGPR GRO into SGPR GRO is usually a win
    # However, the mode may exhaust all available SGPR, in particular for large unroll
    # -1 attempt to use a heuristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseSgprForGRO": [-1, 0, 1],
    # Use a 64-bit shadow limit register to allow buffers larger than 2^32 bytes
    "Use64bShadowLimit": [True, False],
    # Assertion properties
    # These provide information or assertions that the problem size meets certain requirements
    # for sizes or alignments.  The kernel generator can use this information to produce
    # a kernel which uses those assertions to produce a faster kernel.
    #
    # If modifying or adding Assertions also change ProblemProperties class in TensileTypes.h
    # Kernel generator will assume that the summation size is some multiple of the element size
    # and uses this to optimize the kernel.
    # This can result in more efficient kernels, but requires runtime checking to ensure the specified
    # summation value meets the requirements.
    # (Recommended AF1EM value is 8 for half, 4 for single, 2 for double)
    #
    # Optimizations enabled by AssertSummationElementMultiple>1:
    #  - If >=2 for half:
    #     - Tail loop loads can be vectorized 2X to use dword
    #     - Enables asm kernels on V20
    #     - Can use DirectToLds for both unroll and tail loops
    #  - Tail loop can be unrolled up to InnerUnroll amount if AssertSummationElementMultiple%InnerUnroll==0
    #
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertSummationElementMultiple": [1, 2, 4, 8, 16, 32, 64, 128],
    # Kernel generator will assume that the FreeIndex[0] size is some multiple of the element size
    # and uses this to optimize the kernel.
    # FreeIndex[0] is usually letter "I"
    # (Recommended AF0EM value is 8 for half, 4 for single, 2 for double)
    #
    # Optimizations enabled by AssertFree0ElementMultiple>1:
    # Load optimizations:
    #  - For TLU=1 matrix, if AF1WM>=GLVW then can enable UseSgprForGRO
    #      - Reduces registers used for address calculations
    #      - Removes address shift/unshift code
    #    - UseSgprForGRO will only be enabled if all matrices meet assertion requirements.
    #
    # Store Optimizations:
    #  - Can vectorize stores in edge tiles.  Vector width can be up to AF0EM.
    #   (since C matrix is always coalesced in Free0 index direction and this assertion guarantees the index element multiple)
    #
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertFree0ElementMultiple": [1, 2, 4, 8, 16],
    # Kernel generator will assume that the FreeIndex[1] size is some multiple of the element size
    # and uses this to optimize the kernel.
    # FreeIndex[1] is usually letter "J"
    # (Recommended AF1EM value is 8 for half, 4 for single, 2 for double)
    # Optimizations enabled by AssertFree1ElementMultiple>1:
    #  - See above AssertFree0ElementMultiple "Load optimizations"
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertFree1ElementMultiple": [1, 2, 4, 8, 16],
    # Assertions that require arithmetic intensity to be specified value.
    # Arithmetic intensity measures the ratio of computation to memory bandwidth required for a problem.
    # These predicates can be used to adjust solution selection compute-bound or memory-bound problems.
    "AssertAIGreaterThanEqual": -1,
    "AssertAILessThanEqual": -1,
    # Stagger the start summation position of the tiles.
    # Elements from the summation dimension are loaded at offsets rather than all starting at 0.
    # StaggerU is the max 'clicks' of StaggerUStride bytes where each wg starts ; see StaggerUMapping
    # for how the specific stagger for a given wg is determined.
    #
    # The tile assignment C are same as with StaggerOffset=0 ; the difference is the
    # order that the summation elements are added.
    # GRO will wrap back to the row start when the edge is reached.
    #
    # This can be effective for TLU=0 style matrices where the K dimension is a large power-of-2.
    # In this case the start of each row of the tile is separated by an exact power-of-2
    # which causes poor dram, cache, and tlb behavior.  V20 has 16 channels each 256 bytes wide.
    # StaggerU adjusts the start position in the summation (aka 'U') dimension
    # to avoid these conflicts.  Both A and B matrix start at the adjusted position.
    # If >0 specifies the offset in multiples of the macro-tile "unroll" dim
    #  - Higher values will spread traffic to more channels but provide less L2 re-use.
    #  - StaggerU and WorkGroupMapping interact and should be tuned together -
    #    The WGM controls how tiles are assigned in C matrix, while StaggerU controls where those
    #    tiles start reading their summation dim parms.
    #  - StaggerU requires BufferLoad==1 and is silently ignored if BufferLoad==0
    "StaggerU": [0, 2, 4, 8, 16, 32, 64],
    # Stride in bytes for each staggeru 'click'.
    # 256 is recommended since this is the width of memory channel (on gfx803,gfx900,gf906) - so
    # each click will start in a new memory channel and spread traffic among the 16 available channels.
    # For example StaggerUStride=256 and StaggerU=8 will use 8 unique starting points
    # in summation dimension, each offset by 256-bytes - provided the tensor dims are large
    # enough to support this.
    # StaggerUStride will be internally increased so it is an integer multiple of DepthU*BpeAB.
    # (the implementation requires this - the unroll iteration accesses data in steps of
    # DepthU*BPE
    "StaggerUStride": [-1, 16, 32, 64, 128, 256, 512, 1024, 2048],
    # How the tile assignment (wg0, wg1, wg2) controls the initial StaggerU offset:
    # 0: Use wg0
    # 1: Use wg1
    # 2: Use wg2
    # 3: Use wgSerial, wgSerial = wg0 + wg1 * nwg0 + wg2 * (nwg0 * nwg1)
    # 4: Debug mode, offset each tile max allowed StaggerU.  This just moves hotspot
    #    to a different bank since all workgroups still start at same point.
    "StaggerUMapping": [0, 1, 2, 3, 4],
    # GSU Workgroup Coalesced Ordering
    # False: {(wg0,wg1,wg2,wgn)|(wg0,wg1,wg2,wgn)|...|(wg0,wg1,wg2,wgn)}
    # True:  {(wg0,wg0,wg0)|(wg1,wg1,wg1)|(wg2,wg2,wg2)|...|(wgn,wgn,wgn)}
    "GlobalSplitUCoalesced": [False, True],
    # GSU Workgroup Mapping
    # False: wg issued order = {(wg0,wg1,wg2,wgn),(wg0,wg1,wg2,wgn)|...|(wg0,wg1,wg2,wgn)}
    #   -> workgroups do the summation by tile -> slower GR but faster GW
    # True:  wg issused oder = {(wg0,wg0,wg0)|(wg1,wg1,wg1)|(wg2,wg2,wg2)|...|(wgn,wgn,wgn)}
    #   -> workgroups split up the summation -> faster GR but slower GW
    "GlobalSplitUWorkGroupMappingRoundRobin": [False, True],
    # 0=don't use magic div (source only)
    # 1=magic div alg #1.  Slightly faster but limited range (if magic number is 2^32)
    # 2=magic div alg#2.  Slightly slower but handles all unsigned ints up to 2^32
    "MagicDivAlg": [0, 1, 2],
    # For Block Mapping type:
    # 0   : Use hardware-assigned wg number with no remapping.
    # N   : WG block width.  "Wrap" to a new wg1 "row" assignment after N WGs assigned in that row.
    # Tensor C always mapped with first free coord as fastest moving
    # (Elements in this dimension are sequential in memory.
    #
    # For 2D nonbatched Matrix this means index order is I, then J
    # For 2D batched Matrix this means index order is I, then J, then K.
    #
    # Then for 2D case:
    #   - If drawn in row-major format, I is the width and J is the height.
    #   - WGM determines dimensions of the box used to assign tiles from C
    #   - WGM is the height of the box (in the J dimension)
    #   - Given WGM, the box width (in I dim) is determined by number of CUs
    #   - The box always moves across matrixC in the fastest-moving "I" dim, then
    #     wraps to next J.  TODO - might be useful to change this?
    #
    # Examples for 2D matrix:
    # WGM=8:  on CU64 machine this is a square box
    # WGM=1:  Short/Fat - this will cover maximum width in I dimension of C.  This matches hardware assigned mapping.
    # WGM=64: Tall/Skinny - this will cover maximum width in J dimension of C.
    #
    # Formula for wgSerial:
    # wgSerial = wg0 + (wg1 % WorkGroupMapping) * nwg0
    "WorkGroupMapping": list(
        range(-1024, 1024 + 1)
    ),  # change a workgroup's id so that the all the workgroups on the gpu at a time are hitting L2 cache the best
    "WorkGroupMappingXCC": [
        -1,
         1,
         2,
         4,
         8,
         16,
         32,
    ],  # change a workgroup's id so that contiguous workgroup can map on same XCC
    # -1 : WorkGroupMappingXCCGroup will be set to CU_count at runtime. Please ensure that (CU_count % WGMXCC == 0).
    "WorkGroupMappingXCCGroup": list(
        range(-1, 1024)
    ),  # change a workgroup's id so that contiguous workgroup can map on same XCC, remap workgroup in a group of WGMXCCG.
    # Use Space-filling-algorithm to determine workgroup mapping
    # Parameter is provided as a list of integers. The length of the list indicates the level depth.
    # Each integer represents an ordering ID to denote which ordering to use at each level.
    # An empty array means use default (old) WGM algo
    #
    # With multi-level orderings the WGM values contains grid dims for each level encoded as 8bit values
    # WGM: (msb) [GridDimN_L2, GridDimM_L2, GridDimN_L1, GridDimM_L1] (lsb)
    #
    # Order IDS:
    # 0 : map workgroup IDs to C based on col-major order
    # 1 : map workgroup IDs to C based on row-major order
    # 2 : map workgroup IDs to C tiles based on Hilbert curve
    # 3 : map workgroup IDs to C tiles based on Morton Z-curve
    # 4 : map workgroup IDs to C tiles based on Morton (reverseN)-curve
    # 5 : map workgroup IDs to C tiles based on Morton U-curve
    #
    # Examples:
    # [0]: Single level ordering of C tiles using column major ordering
    # [0,1]: Double level ordering of C tiles using column major ordering, then row-major
    # [1,1,1]: Triple level ordering of C tiles using row major ordering at each level
    "SpaceFillingAlgo" : -1,
    # Used to specify the grid dims for each level when using SpaceFillingAlgo. This value is directly passed into the
    # WGM parameter. The format is
    # WGM: (msb) [GridDimN_L2, GridDimM_L2, GridDimN_L1, GridDimM_L1] (lsb)
    #
    # Examples:
    # 0x01020304
    #  - Level1 grid dim 4x3
    #  - Level2 grid dim 2x1 (if enabled, otherwise last 16 bit values are ignored)
    # 0x10010820
    #  - Level1 grid dim 32x8
    #  - Level2 grid dim 1x16 (if enabled, otherwise last 16 bit values are ignored)
    "SFCWGM" : -1,
    "MaxOccupancy": list(
        range(1, 40 + 1)
    ),  # wg / CU; if cache thrashing is hurting performance, this allocates extra lds to artificially limit occupancy
    "MaxLDS": [-1, 65536, 163840],
    "WorkGroup": makeValidWorkGroups(),  # ( wg0 x wg1 x LocalSplitU ) dimensions of the workgroup which will operate on a tile and share lds
    # ThreadTile: ( tt0 x tt1 ) dimensions of the C tile that each thread works on,
    # TT=4 and VW=4 means a thread will work on a tight 4x4 tile of C, where VW=1 means the tile will work on 16 spread out values
    # Generally, the VW determines the consecutive a WI will work on, then it will skip ahead SG0*VW elements to get to the next row of VGPR inputs
    "ThreadTile": validThreadTiles,
    "MacroTile": validMacroTiles,  # MT0 = wg0*tt0, MT1 = wg1*tt1
    "WavefrontSize": [32, 64],
    # MatrixInstruction: (M x N x K x B)
    # XDLOPS tile definition, only valid for gfx908, gfx90a
    # MxNxKxB specifies matrix instruction variants
    #  MxNxB determines the shape of the C tile each instruction worked on
    #      K determines the unroll depth
    # If empty, do not use these instructions
    #
    # Alternative format: (M x N x K x B x MIBlockM x WaveTileM x WaveTileN x WaveM x WaveN)
    # (Note: MxN means M-by-N in the following comments)
    # MIBlockM determines how many blocks along M dimension for multi-block MI variants. Concrete examples:
    #  - MI 16x16x1x4 (4-block variant) with MIBlockM=4 -> (16x16)*(4x1)=64x16 tile per instruction executed
    #  - MI 32x32x1x2 (2-block variant) with MIBlockM=1 -> (32x32)*(1x2)=32x64 tile per instruction executed
    # WaveTileM/N are dimensions of the C tile each wave works on, and is close to the concept of ThreadTile in classic VALU kernels
    #  - WT 4x1 -> each wave executes 4x1 matrix instructions on the C tile of total area (4*MITileM)x(1*MITileN)
    # WaveM/N are dimensions of waves spawned for one workgroup where each wave consists of 64 threads
    #  - Wave2x2 -> a total of 4 waves in one workgroup of shape 2x2
    # Putting it all together:
    #  - [32, 32, 1, 2,  1,  4, 1,  2, 2]
    #     ^^^^^^^^^^^^   ^   ^^^^   ^^^^
    #      MatrixInst  BlkM   WT    Wave
    #  - means (32x64) per MI * (4x1) per wave * (2x2) per workgroup = (32*4*2)x(64*1*2) = 256x128 macro tile
    # Tensile will ignore the parameters ThreadTile and WorkGroup when the alternative format is used
    # NOTE: MatrixInstruction is no longer validated through this structure, but is instead validated via the
    #   ``TensileLogic`` program.
    "MatrixInstruction": -1,
    # StoreRemap: Optimize MatrixInstruction store patterns to enhance performance.
    #             MI output data between each threads are along N dims.
    #             But global memory is along M dim continuous.
    #             That mean global write between each threads are not continuous.
    #             Therefore, store performance for MI instruction is poor.
    # How StoreRemap works in final store stage:
    #             1. Put all thread output data into LDS.
    #             2. All thread read data from LDS along M dims.
    #                (match global Memory continuous direction)
    #             3. All thread write out data into global memory.
    # 0:   Disable StoreRemap (default)
    # 1~8: Enable StoreRemap and set the global write vector width
    # Suggest optimum value: fp32 = [2,4], fp16 or bf16 = [4,8] (dwordx2 and dowrdx4)
    # -1:  Use dwordx2 if support SRVW, or set SRVW to 0
    "StoreRemapVectorWidth": [-1, 0, 1, 2, 4, 8],
    # SourceSwap: Optimizes MatrixInstruction store pattern by swapping mfma input order.
    "SourceSwap": [False, True],
    # Following parameters are designed for store scheduling.
    # (store stands for load from C (with beta) and store to C/D)
    #
    # we want to hide store behind unroll loop
    #   1. if we can launch 2 WorkGroups per CU (occupancy >= 2, large M/N)
    #   2. if there are remaining global memory bandwidth in unroll loop (compute bound kernel)
    #
    # we can hide store behind the other WG's loop by lowering priority of store
    #   priority of loop is the same as priority of store
    #     WG0: ???????????????\__
    #         |<-- loop --->|<-- store -->|end
    #
    #     WG1: ___________________________/????????????\__
    #         |<--------- loop ------------------->|<-- store -->|end
    #
    #   priority of loop is higher than priority of store
    #     WG0: ???????\____________________
    #         |<-- loop --->|<------ store ----->|end
    #
    #     WG1: _____________/?????\__________________
    #         |<------- loop -------->|<----- store ---->|end
    "StorePriorityOpt": [False, True],
    #
    # If we issue store in short period of time, kernel will become from compute bound to memory bound
    # 0 means issue instructions as many as possible if VGPR available
    "NumElementsPerBatchStore": list(range(-1, 256)),
    #
    # add sync after per batch store in order to store contiguous elements
    # add sleep after per batch store in order to distribute store over whole loops
    # NOTE: this parameter is highly depends on size_k
    # 0 means no sync and sleep
    "StoreSyncOpt": list(range(0, 256)),
    #
    # There are index or address calculation between global instructions.
    # issue global instruction b2b has better performance
    "GroupLoadStore": [False, True],
    # In order to remove the copying from Acc vgpr to Arch vgpr, only use Arch vgprs for v_mfma_xxx.
    # Only support for kernel whose totalVgpr counts less than 256 and gcn that has control bit ACC_CD.
    "MIArchVgpr": [False, True],
    # StreamK (SK) kernels divide work evenly among CUs by splitting along MT and K dimensions.
    # Total work units are calculated as (#MTs x #LoopIters) and divided among workgroups.
    # In most cases each workgroup will calculate a partial tile that are accumulated in a fixup step in the same kernel
    # 0 : Standard data-parallel kernel
    # 1 : Basic StreamK
    # 2 : Two-Tile StreamK (each WG completes an even number of sk iterations, followed by an even number of dp tiles)
    # 3 : Two-Tile StreamK with DP before SK tiles
    # StreamK kernels can adjust the number of CUs being used.
    # Using fewer sometimes increases overall throughput by allowing other kernels to run in parallel.
    # StreamK grid is controlled by setting these enviornment variables:
    # TENSILE_STREAMK_FIXED_GRID lets you override the default grid size with a specific number
    #   0 = override disabled (default)
    # TENSILE_STREAMK_FULL_TILES sets the number of full tiles to be included in stream-k work
    #   -1 = use prediction model for best performance (not yet implemented)
    #   0 = only remainder tiles run in stream-k
    #   1+ = remainder + 1 (or more) full grids of tiles run in stream-k (default=1)
    # TENSILE_STREAMK_DYNAMIC_GRID selects dynamic grid mode, which automatically limits the number of CUs used:
    #   0 = Off, always use all CUs.
    #   1 = Only reduce CUs for small problems to number of output tiles when num_tiles < CU count.
    #   2 = Also reduce CUs used for large sizes to improve data-parallel portion and reduce power.
    #   3 = Analytically predict the best grid-size by weighing the cost of the fix-up step and the cost of processing MACs (default).
    #       Note: dynamic grid coefficients currently apply to gfx942 variants
    #   4 = StreamK algorithm will behave as data parallel (Launch WGs = #CUs)
    #   5 = StreamK Algorithm will use Origami's "select_best_grid_size" function
    # TENSILE_STREAMK_DYNAMIC_WGM Enables Origami's analytical model-based WGM selection
    # TENSILE_STREAMK_MAX_CUS allows the user to manually set maximum number of CUs used, which could free up some CUs for
    #   other operations to run in parallel with gemm.
    # TENSILE_STREAMK_GRID_MULTIPLIER lets you set how many workgroups are created per CU being used.
    #   1 = 1 WG per CU (default), for example. 2 will launch WGs = 2 x CU count.
    # The priority of these environment variables is defined as follows:
    # TENSILE_STREAMK_FIXED_GRID > TENSILE_STREAMK_DYNAMIC_GRID > TENSILE_STREAMK_MAX_CUS > TENSILE_STREAMK_GRID_MULTIPLIER
    "StreamK": [0, 1, 2, 3],
    # Determines if StreamK kernel uses atomics
    # 0: uses workspace to store partial tiles, accumulate in deterministic fix-up step
    # 1: uses atomics to accumulate partial tiles
    "StreamKAtomic": [0, 1],
    # Enables XCC-based remapping of workgroups, set the value to the number of XCCs
    # for the device/configuration being used
    #  0: uses default workgroup assignment
    # 2+: remaps workgroups to be contiguous within an XCC for a given number of XCCs
    "StreamKXCCMapping": [0] + list(range(2, 9)),
    # Enables using a Tree-reduction for the fixup step of StreamK algorithm
    # 0: use linear reduction
    # 1: use tree reduction
    "StreamKFixupTreeReduction": [0, 1],
    # Debug settings for stream-k kernels to disable parts of the kernel
    #   Bit 0: Don't generate fixup code
    #   Bit 1: Don't generate write to partials code
    # Both parts can be disabled together
    #   0 = Debug mode off, generate full kernel
    #   1 = No fixup
    #   2 = No partials
    #   3 = Nofixup and no partials
    "DebugStreamK": [0, 1, 2, 3],
    # Controls desired width (#elements) for loads from global memory -> LDS.
    # and eliminates the pointer unshift logic
    # -1 : Set GlobalReadVectorWidth =  VectorWidth
    # NOTE: for input bpe=32, max GRVW is 4  (to fit dwordx4) (FP32), min GRVW is 1 (dword)
    #                 bpe=16, max GRVW is 8  (to fit dwordx4) (FP16), min GRVW is 2 (dword)
    #                 bpe=8,  max GRVW is 16 (to fit dwordx4) (INT8), min GRVW is 4 (dword)
    "GlobalReadVectorWidthA": [-2, -1, 1, 2, 3, 4, 6, 8, 16],
    "GlobalReadVectorWidthB": [-2, -1, 1, 2, 3, 4, 6, 8, 16],
    # Controls desired width (#elements) for loads from LDS -> VGPR.
    # -1 : Set LocalReadVectorWidth =  VectorWidth
    #  1 cannot be used for half type.
    # used in combination with TransposeLDS=True
    # in TransposeLDS=1 case, use wider load to fetch elements in summation dimension from LDS
    # helps optimizing instruction scheduling between MFMA and nonMFMA instructions
    # NOTE: for input bpe=32, max LRVW is 4  (to fit ds_read_b128) (FP32)
    #                 bpe=16, max LRVW is 8  (to fit ds_read_b128) (FP16)
    #                 bpe=8,  max LRVW is 16 (to fit ds_read_b128) (INT8)
    "LocalReadVectorWidth": [-1, 1, 2, 4, 8, 16],
    # threads should read/write/operate on this many contiguous elements from the C matrix.
    # If VW=4 then thread0 will process 4 consec C elements, then thread1 next 4, etc.
    # If the ThreadTile is > VectorWidth then thread0 will next operate on the 4 elements in C at (4*NumThreads)
    # Typically the load vector width and store vector width are directly related to the VW.
    # The global load width is closely related to the width of local stores so
    # GlobalReadVectorWidth also controls local write width.
    # Local read width also matches since VectorWidth consec elements must be read
    # Typically matching 16 bytes is good choice since the stores will be optimally coalesced with 16 bytes/WI.
    # Using a VW too large which results in >16bytes/thread isn't supported
    # For MFMA non SourceSwap: this parameter didn't take effect
    # -1 means set vw to largest localReadWidth according to MIWaveTile
    "VectorWidthA": [-1, 1, 2, 3, 4, 6, 8],
    "VectorWidthB": [-1, 1, 2, 3, 4, 6, 8],
    # If 0, store 1 element per instruction.
    # If 1, store vector-width elements per instruction.
    # if -1, store vector-wide elements per instruction unless PBD would not generate a valid kernel
    "VectorStore": [-1, 0, 1],
    # Controls desired width (#elements) for stores from reg to global memory.
    # When MatrixInstruciton == None, derived parameter gwvw takes precedence.
    # -1 : Set StoreVectorWidth = VectorWidth
    "StoreVectorWidth": [-1, 1, 2, 3, 4, 6, 8],
    # when loading all the data from global into lds requires multiple load instructions, these parameters govern which
    # loads will pull which rectangle of data from global into lds
    # NLC=1 means one load along the coalesced dimension, which results in the most coalescing possible
    # NLC=-1 looks for the largest number of reads along the coalesced dimension which results in the least ammount of coalescing;
    # however in this case the stride between one load and another is a static value, therefore buffer loads only need one set of registers
    # whereas the =1 case has a stride which is a multiple of a kernel argument and therefore needs one address per load in the perpendicular dimension
    "NumLoadsCoalescedA": list(range(-1, 64 + 1)),
    "NumLoadsCoalescedB": list(range(-1, 64 + 1)),
    # DepthU, LocalSplitU (which is the 3rd number in WorkGroup), and LoopUnroll are closely related
    # LoopUnroll=4 means there are 4 subiterations within the loop, 4 actual iterations written in the code.
    # LocalSplit=2 means the workgroup is split up into 2 subgroups, and each subgroup is doing different parts of the summation.
    # subgroup0 does k=0-3, 8-11... and subgroup1 does k=4-7, 12-15...
    # So, each iteration through the summation loop, which has 4 actual subiterations, does 8 summation iterations, because each subgroup did 4;
    # and when data is read from global memory the threads read 8 elements along the summation dimension.
    # DepthU = LoopUnroll * LocalSplitU = 4*2 in this case
    # it made more sense for the user to directly control LocalSplitU and DepthU, then derrive afterwards LoopUnroll=DepthU/LocalSplitU
    # -1 : Only allow GLVW=1
    # -2 : Only allow max(GLVWA,GLVWB) < VW ?
    # -3 : Only allow min(GLVWA,GLVWB) < VW ?
    "DepthU": [-1] + list(range(2, 1024 + 1, 1)),
    # integer amount of padding to put into LDS, in 2016 this didn't seem to help performance, profilers were showing that channel conflicts weren't really hurting
    # performance so this has been deprecated and probably doesn't work
    # -1 means use same padding as the VectorWidth if TLU=0 else 0.  (Padding only helps when transpose is required)
    # With MatrixInstruciton: -1 means max(GRVW,MIInput) if TLU=0
    "LdsPadA": [-1, 0, 1, 2, 3, 4, 8, 16, 32, 48, 64],
    "LdsPadB": [-1, 0, 1, 2, 3, 4, 8, 16, 32, 48, 64],
    "LdsPadMetadata": [-1, 0, 1, 2, 3, 4, 8],
    # Padding boundary for LDS. defines block-size for pad insertion. for every 'LdsBlockSizePerPad' bytes, LDS padding (pad value from LdsPad parameter)
    # is added (readOffset aware of the pad and adjusts offset value based on this parameter value).
    # Only support LdsBlockSizePerPad >= unrollDepth * BPE
    # 0 means disable LdsBlockSizePerPad
    "LdsBlockSizePerPadA": [-1, 0, 64, 128, 256, 512, 1024, 2048],
    "LdsBlockSizePerPadB": [-1, 0, 64, 128, 256, 512, 1024, 2048],
    "LdsBlockSizePerPadMetadata": [-1, 0, 64, 128, 256, 512, 1024, 2048],
    # Transpose LDS format. Local store in coalesced dimension , same as optimized global fetch dimension . applicable only in TLU=0 case for miSIMD(s)
    # -1 : keep LDS layout same as global fetch dimension for both A and B
    #      set TLDS = 1 for NN,TN,TT
    #      set TLDS = 0 for NT
    # 0  : coalesced dimension of lds is tile dimension
    # 1  : keep LDS layout same as global fetch dimension for both A and B for NN,TN,TT, but NT would be rejected
    # 2  : coalesced dimension of lds is unroll dimension for both A and B
    "TransposeLDS": [-1, 1, 0, 2],
    # add gls or slc after global memory read/writes to change caching, not caching the writes is promising and improved performance a tiny bit
    # 0: none, 1: glc, 2: slc, 3: glc slc
    # For gfx942, sets sc0/sc1/nt bits
    # 0: none, 1: sc0, 2: sc1, 3: sc0 sc1, 4: nt, 5: nt sc0, 6: nt sc1, 7: nt sc0 sc1
    "NonTemporalE": list(range(0, 8)),
    "NonTemporalD": list(range(0, 8)),
    "NonTemporalC": list(range(0, 8)),
    "NonTemporalA": list(range(0, 8)),
    "NonTemporalB": list(range(0, 8)),
    "NonTemporalWS": list(range(0, 8)),
    "NonTemporalMetadata": list(range(0, 8)),
    "NonTemporal": list(range(-1, 8)),
    # Group together unroll iterations inside the unroll loop.
    # For example, InnerUnroll=2 will fetch LDS for two unroll iterations
    "InnerUnroll": [1, 2, 4, 8, 16, 32, 64],
    # Enable CP preload kernel arguments feature
    # It can reduce time of loading kernel arguments by s_load.
    # It needs new complier and vbios to support this feature.
    "PreloadKernArgs": [False, True],
    # Kernels should be written in assembly or source
    # if assembly, ISA will determine architecture
    # if source, Runtime will determine language
    # later on, we'll relax this to inner kernel languages and outer kernel languages, such as inline asm embedded in ocl or in llvm
    "KernelLanguage": ["Assembly"],
    # We set validParams["ISA"] in multiple places
    "ISA": validISA,  # arch for assembly kernels
    # Name of the custom kernel located at `CUSTOM_KERNEL_PATH`.
    # a custom kernel is a user written assembly kernel with its associated configuration parameters included in a custom.config section
    # inside the yaml block between the --- and ... markers.  These parameters are only used for information purposes, not kernel generation.
    # Ex:
    # custom.config:
    #   ProblemType:
    #     OperationType: GEMM
    #     etc...
    #   ThreadTile: [8, 8]
    #   etc...
    #
    # Custom kernels can be included in a BenchmarkProblemSizeGroup by having their name (without file extension) listed under the "CustomKernels"
    # category alongside InitialSolutionParameters, BenchmarkCommonParameters, etc...
    "CustomKernelName": -1,
    # Will allow a kernel to be accepted even when checks determine it's not viable.
    # Intended for use with custom kernels which have confirmed to be correct
    "NoReject": [False, True],
    # Debug use only.
    "ActivationFused": [True],
    # True-  function call
    # False- inline
    "ActivationFuncCall": [False, True],
    # Alternative implementation for activation function
    # Currently only supports GSU == 1
    "ActivationAlt": [False, True],
    # Do workgroup reduction. Currently for DBias
    "WorkGroupReduction": [False],
    # 4:2 Structured Sparse A Matrix, 0=Non Sparse, 1=Sparse Matrix A, 2=Sparse Matrix B
    "Sparse": [0, 1, 2],
    # in mix mode F8 need to convert to F16, do this before(0) ds or after(1) ds
    "ConvertAfterDS": [False, True],
    # Force disable shadow init to release more sgpr in preloop
    "ForceDisableShadowInit": [False, True],
    # Enable LDS Transpose Instruction
    "LDSTrInst": [False, True],
    # False: Use LocalSplitU. Number of WorkGroup[2] WorkItems (wave or thread) will compute the same output elements (matrix D) along different
    #        unroll indices. The local sum from those WorkItems are reduced through LDS.
    # True:  Use WaveSplitK. Number of WorkGroup[2] threads in the same wave compute the same output elements (matrix D) along different unroll indices.
    #        The local sum from those threads are reduced through suffling using VALU instructions. Currently only support dot2 kernel.
    "WaveSplitK": [False, True],
    # Control mbsk reduction prefetch order
    # -1 : Select between 0/1 based on # store elements.
    # 0  : Fetch from workgroup dim -> elements dim. (default)
    # 1  : Fetch from elements dim -> workgroup dim. Has better prefetch pattern when # store elements is large.
    "MbskPrefetchMethod": [-1, 0, 1],
    "UseCustomMainLoopSchedule" : [0, 1]
}

newMIValidParameters = {
    "EnableF32XdlMathOp": [False, True],
    "UseF32XEmulation": [False, True],
    'EnableMatrixInstruction': [False, True],
    'ISA': -1,
    'MFMA_BF16_1K': [False, True],
    'MIBlock': -1,
    'MIInputPerThread': -1,
    'MIInputPerThreadA': -1,
    'MIInputPerThreadB': -1,
    'MIInputPerThreadMetadata': -1,
    'MIWaveGroup': -1,
    'MIWaveTile': -1,
    'MatrixInstM': -1,
    'MatrixInstN': -1,
    'MatrixInstK': -1,
    'MatrixInstB': -1,
    'MatrixInstBM': -1,
    'MatrixInstBN': -1,
    'MatrixInstruction': -1,
    'Sparse': -1,
    'ThreadTile': -1,
    'WavefrontSize': -1,
    'WorkGroup': -1,
}

def checkSpaceFillAlgoIsValid(name, value):
    if type(value) != list:
        msgBase = "Invalid parameter value: {} = {}\nMust be a list of values"
        raise Exception(msgBase.format(name, value))
    elif len(value) > 3:
        msgBase = "Invalid parameter value: {} = {}\nOnly 3 level ordering supported"
        raise Exception(msgBase.format(name, value))
    else:
        maxOrderID = 5
        for orderId in value:
            if orderId not in range(0,maxOrderID + 1):
                msgBase = "Invalid parameter value: {} = {}\nOrderID out of range"
                raise Exception(msgBase.format(name, value))

def checkSpaceFillAlgoWGMIsValid(name, value):
    if type(value) != list:
        msgBase = "Invalid parameter value: {} = {}\nMust be a nested list of values"
        raise Exception(msgBase.format(name, value))
    elif len(value) > 2:
        msgBase = "Invalid parameter value: {} = {}\nOnly 3 level ordering supported"
        raise Exception(msgBase.format(name, value))
    else:
        for pair in value:
            if len(pair) != 2:
                msgBase = "Invalid parameter value: {} = {}\nMust be exactly 2 values per level"
                raise Exception(msgBase.format(name, value))
            for dim in pair:
                if dim not in range(0,256):
                    msgBase = "Invalid parameter value: {} = {}\nGridDim {} out of range [0,256)"
                    raise Exception(msgBase.format(name, value, dim))


def checkParametersAreValid(param, validParams):
    """Ensures paramaters in params exist and have valid values as specified by validParames"""
    (name, values) = param
    if name == "ProblemSizes":
        return
    elif name == "InternalSupportParams":
        return

    if name not in validParams:
        raise Exception(
            "Invalid parameter name: {}\nValid parameters are {}.".format(
                name, sorted(validParameters.keys())
            )
        )

    for value in values:
        if validParams[name] != -1 and value not in validParams[name]:
            msgBase = "Invalid parameter value: {} = {}\nValid values for {} are {}{}."
            msgExt = (
                " (only first 32 combos printed)\nRefer to Common.py for more info"
                if len(validParams[name]) > 32
                else ""
            )
            raise Exception(msgBase.format(name, value, name, validParams[name][:32], msgExt))
        elif name == "SpaceFillingAlgo":
            checkSpaceFillAlgoIsValid(name, value)
        elif name == "SFCWGM":
            checkSpaceFillAlgoWGMIsValid(name, value)
