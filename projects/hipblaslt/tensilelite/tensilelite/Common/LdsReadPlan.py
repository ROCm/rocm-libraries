# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class LocalReadLoopPlan:

  numTilePerInst: int
  miInputPerThUnroll: int
  numVectorsPerTile: int
  numReadsPerVector: int
  numReadsPerUnroll: int
  numElementPerRead: int


def make_local_read_loop_plan(vectorWidth: int,
                              bpeDS: float,
                              bpe: float,
                              blockWidth: float,
                              unrollMajorLDS: bool,
                              miInputPerThread: int,
                              miWaveTile: int,
                              matrixInstT: int,
                              matrixInstTO: int,
                              lrvwTile: int,
                              convertAfterDS: bool,
                              useF32XEmulation: bool) -> LocalReadLoopPlan:
  numTilePerInst = matrixInstT // matrixInstTO
  miInputPerThUnroll = miInputPerThread // numTilePerInst
  tileBlockWidth = bpeDS / 4 if unrollMajorLDS else blockWidth
  unrollBlockWidth = blockWidth if unrollMajorLDS else bpeDS / 4
  numVectorsPerTile = miWaveTile // vectorWidth
  numReadsPerVector = int(
      (vectorWidth * bpeDS) / (tileBlockWidth * 4))
  numReadsPerUnroll = ceil(
      bpeDS * miInputPerThUnroll / (unrollBlockWidth * 4))
  if convertAfterDS and not useF32XEmulation:
    numElementPerRead = 1
  else:
    numElementPerRead = int(
        int(blockWidth * 4) // bpe // lrvwTile)
  return LocalReadLoopPlan(
      numTilePerInst, miInputPerThUnroll, numVectorsPerTile,
      numReadsPerVector, numReadsPerUnroll, numElementPerRead)


def gfx1250_lds_offset_bytes(*,
                             rIdx: int,
                             tiIdx: int,
                             tileOffsetElements: int,
                             localReadOffsetElements: int,
                             localReadVectorWidth: int,
                             numElementPerRead: int,
                             unrollStrideElements: int,
                             tileStrideElements: int,
                             matrixInstTO: int,
                             vectorWidth: int,
                             bpeDS: float,
                             unrollMajorLDS: bool,
                             isMXS: bool = False) -> int:
  if isMXS:
    incOffset = rIdx * numElementPerRead * unrollStrideElements
  elif unrollMajorLDS:
    incOffset = rIdx * numElementPerRead * unrollStrideElements * 2
    incOffset += tiIdx * matrixInstTO * vectorWidth * tileStrideElements
  else:
    incOffset = ((rIdx // localReadVectorWidth)
                 * unrollStrideElements * localReadVectorWidth)
    incOffset += rIdx * numElementPerRead * unrollStrideElements
  return int(
      (incOffset + tileOffsetElements + localReadOffsetElements) * bpeDS)


def force_lrvw_tile_one(macDataTypeBytes: float,
                        enableMatrixInstruction: bool,
                        miInputPerThread: int,
                        useF32XEmulation: bool,
                        useMFMAF32XEmulation: bool,
                        useCustomMainLoopSchedule: bool,
                        hasWMMAV3: bool) -> bool:
  wideType = macDataTypeBytes >= 4
  multiInputMI = enableMatrixInstruction and miInputPerThread > 1
  supportsWideXF32 = (useF32XEmulation
                      and (useMFMAF32XEmulation
                           or useCustomMainLoopSchedule
                           or hasWMMAV3))
  return wideType and multiInputMI and not supportsWideXF32


def effective_lrvw_tile(vectorWidth: int,
                        unrollMajorLDS: bool,
                        macDataTypeBytes: float,
                        enableMatrixInstruction: bool,
                        miInputPerThread: int,
                        useF32XEmulation: bool,
                        useMFMAF32XEmulation: bool,
                        useCustomMainLoopSchedule: bool,
                        hasWMMAV3: bool) -> int:
  forceOne = force_lrvw_tile_one(
      macDataTypeBytes, enableMatrixInstruction, miInputPerThread,
      useF32XEmulation, useMFMAF32XEmulation,
      useCustomMainLoopSchedule, hasWMMAV3)
  return vectorWidth if not unrollMajorLDS and not forceOne else 1


def effective_lrvw_unroll(localReadVectorWidth: int,
                          unrollMajorLDS: bool) -> int:
  return localReadVectorWidth if unrollMajorLDS else 1


def local_read_input_per_thread(localReadVectorWidth: int,
                                miInputPerThread: int,
                                inTailLoop: bool,
                                useF32XEmulation: bool = False,
                                matrixInstM: int = 16,
                                matrixInstK: int = 16) -> int:
  if useF32XEmulation and (matrixInstM, matrixInstK) != (16, 16):
    return 4
  return miInputPerThread if inTailLoop else localReadVectorWidth
