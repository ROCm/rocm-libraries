// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Pure C++ port of the read-only TileInfo construction + grid/index query
// layer used by the subtile path (Tensile/Components/Subtile/Kernel.py,
// TileInfo for the ABTilePair case).
//
// This header is intentionally free of any nanobind / Python dependency so the
// query math can be unit-tested and reasoned about as plain C++. The nanobind
// bindings live in src/main.cpp.
//
// SCOPE: only the *read-only* derived grids, load ratios, and grid/index query
// helpers are ported. No writer pool / register allocation, rocisa instruction
// emission, scale offset, tail, or main-loop orchestration is included. The
// geometry math itself is reused from subtile_geometry.hpp rather than
// duplicated here.

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "tensile_writer/subtile_geometry.hpp"

namespace tw::subtile {

// ---------------------------------------------------------------------------
// Data-only plans for the subtile emit leaves.
//
// These mirror exactly the instruction-shape arithmetic of
// SubtileGREmit.emitSingleBufferLoad and SubtileLREmit.emitSingleDsRead. They
// carry NO rocisa objects and NO writer register state (soffset/voff/dst VGPR
// indices stay on the Python side); only the integer offsets / strides and the
// per-instruction loop structure are computed here.
// ---------------------------------------------------------------------------
struct SingleBufferLoadPlan {
  // When loadRatioGR > 1 several local subtiles share one global read; only the
  // first subtile of each group emits. ``skip`` mirrors the early ``return
  // module`` (empty) in the Python leaf.
  bool skip;
  long grBaseId;
  long offsetK;             // MUBUF offset12, also subtracted from m0
  std::vector<long> m0Offsets;  // one entry per GR load within the subtile
};

struct DsReadEntry {
  long dstRegOffset;  // VGPR offset within the destination tile for this read
  long addrIdx;       // index into sharedVgprLROffset for this read
};

struct SingleDsReadPlan {
  long regsPerDsRead;
  long mfmaId;
  long offset;  // DS immediate offset (LDS byte position of the subtile)
  long numReadsForTile;
  std::vector<DsReadEntry> reads;
};

// ---------------------------------------------------------------------------
// GR / LR offset-assignment plans (B16 / TLU0 only).
//
// These hold the scalar offset-assignment *math* that the legacy
// SubtileGREmit.graTileAssignment / SubtileLREmit.lraTileAssignment functions
// derive inline before emitting the rocisa offset-calculation instructions.
// The plan carries NO rocisa objects and NO writer register state (VGPR/SGPR
// pool checkout/checkin and the shared offset registers stay on the Python
// side). Only integer/derived scalars used as immediates / shift amounts /
// comment values are computed here.
//
// SCOPE: row-major (TLU0) BF16 (bpe == 2). FP8 (bpe == 1, distinct swizzle)
// and the TLU1 column-major path are intentionally excluded; the Python
// emitter falls back to its native path for those.
// ---------------------------------------------------------------------------
struct GROffsetAssignPlan {
  long subIterKBytes;        // depthUBytes / localSubtileGrid[1]
  long loadWidth;            // gr.loadWidth
  long blockSize;            // subIterKBytes / loadWidth
  long numRowsPerLDSBanks;   // ldsRowBankSize / subIterKBytes
  long numRowsPerWave;       // waveSize / blockSize
  long partitionOffset;      // mmaTileShape[0] * localSubtileGrid[0]
  // Wave-partition expression selector keyed on loadRatioGR:
  //   1 -> loadRatioGR == 1.0, 0 -> 0.5, 2 -> 2.0, -1 -> unsupported.
  int partitionMode;
  long subtileSizeElems;     // subtileShape[0] * mmaTileShape[0]
  long grAdvanceOffset;      // ceil(subtileSizeElems * loadRatioGR)
  long bpeBits;              // int(8 * bpe)
  long grSubtileRowOffset;   // ceil(numGRPerSubtile * loadRatioGR * subtileSizeElems)
  long sStride;              // int(grSubtileRowOffset * bpe)
  long numGRPerSubtile;
  double loadRatioGR;
};

struct LROffsetAssignPlan {
  long subIterKBytes;
  long loadWidthLR;          // lr.loadWidth
  long loadWidthGR;          // gr.loadWidth (used by the wave-partition math)
  long blockSize;            // subIterKBytes / loadWidthLR
  long numRowsPerLDSBanks;   // ldsRowBankSize / subIterKBytes
  long miM;                  // mmaTileShape[0]
  long numMFMACols;          // int(mmaTileShape[1] * bpe) / loadWidthLR
  long partitionOffset;      // mmaTileShape[0] * localSubtileGrid[0]
  long sInterval;            // partitionOffset * subIterKBytes
  long mWavesM;              // MIWaveGroup[0] (supplied by caller)
  // Wave-partition selector keyed on loadRatioGR:
  //   -1 -> no partition (>= 2.0), 1 -> 1.0, 0 -> 0.5, -2 -> unsupported.
  int wavePartMode;
  double loadRatioGR;
};

// ---------------------------------------------------------------------------
// ABTileInfoQuery — read-only snapshot of the AB (ABTilePair) TileInfo state.
//
// Built from an *already materialized* ABGRGeometry (subtileCount/subtileStride
// set via forKernel) and its ABLRGeometry partner, plus the kernel-derived
// scalar fields TileInfo extracts in __init__:
//   macroTile, depthU, waveGroupSize, waveSize, numWaves.
//
// The constructor reproduces exactly the derived attributes TileInfo computes
// for the ABTilePair branch, then exposes the read-only properties and the
// grid/index query helpers.
// ---------------------------------------------------------------------------
struct ABTileInfoQuery {
  // Inputs
  ABGRGeometry gr;
  ABLRGeometry lr;
  long macroTile;
  long depthU;
  long waveGroupSize;
  long waveSize;
  long numWaves;

  // Derived grids (GR is the primary scheduler-facing grid)
  std::pair<long, long> globalMMATileGrid;
  std::pair<long, long> localMMATileGrid;
  std::pair<int, int> subtileShape;
  std::optional<int> subtileCount;
  std::optional<int> subtileStride;
  std::pair<double, double> globalSubtileGrid;
  std::pair<long, long> localSubtileGrid;
  double subtileSize;

  // GR cooperative load counts
  double loadRatioGR;

  // LR grid / load counts
  std::pair<int, int> lrSubtileShape;
  double lrSubtileSize;
  // Python keeps lrGlobalSubtileGrid as the raw (possibly fractional) float
  // grid from ABLRGeometry.globalSubtileGrid; mirror that exactly rather than
  // truncating to integers.
  std::pair<double, double> lrGlobalSubtileGrid;
  std::pair<long, long> lrLocalSubtileGrid;
  double loadRatioLR;

  ABTileInfoQuery(const ABGRGeometry& gr_, const ABLRGeometry& lr_,
                  long macroTile_, long depthU_, long waveGroupSize_,
                  long waveSize_, long numWaves_)
      : gr(gr_),
        lr(lr_),
        macroTile(macroTile_),
        depthU(depthU_),
        waveGroupSize(waveGroupSize_),
        waveSize(waveSize_),
        numWaves(numWaves_) {
    globalMMATileGrid = gr.globalMMATileGrid(macroTile, depthU);
    localMMATileGrid = gr.localMMATileGrid(macroTile, depthU, waveGroupSize);

    subtileShape = gr.subtileShape;
    subtileCount = gr.subtileCount;
    subtileStride = gr.subtileStride;
    globalSubtileGrid = gr.globalSubtileGrid(macroTile, depthU);
    // Python: int(localMMATileGrid[k] / subtileShape[k]) — truncating division
    // on non-negative operands.
    localSubtileGrid = {
        static_cast<long>(static_cast<double>(localMMATileGrid.first) /
                          subtileShape.first),
        static_cast<long>(static_cast<double>(localMMATileGrid.second) /
                          subtileShape.second)};
    subtileSize = gr.subtileSizeBytes();

    long grBytesPerLoad = gr.bytesPerLoad(numWaves);
    double globalGRTileSize =
        subtileSize * (subtileCount.has_value() ? *subtileCount : 1);
    loadRatioGR =
        globalGRTileSize != 0.0
            ? static_cast<double>(grBytesPerLoad) / globalGRTileSize
            : 0.0;

    lrSubtileShape = lr.subtileShape;
    lrSubtileSize = lr.subtileSizeBytes();
    lrGlobalSubtileGrid = lr.globalSubtileGrid(macroTile, depthU);
    // AB: LR iterates over the GR subtile grid.
    lrLocalSubtileGrid = localSubtileGrid;
    double lrBytesPerLoad = static_cast<double>(lr.loadWidth) * waveSize;
    loadRatioLR = lrSubtileSize != 0.0 ? lrBytesPerLoad / lrSubtileSize : 0.0;
  }

  // --- Read-only count properties (mirror TileInfo convenience accessors) ---

  // mmaTileLocalTotalCount
  long numMFMATiles() const {
    return localMMATileGrid.first * localMMATileGrid.second;
  }

  // grSubtileTotalCount
  long numGlobalSubtiles() const {
    return static_cast<long>(globalSubtileGrid.first * globalSubtileGrid.second);
  }

  long numLocalSubtiles() const {
    return localSubtileGrid.first * localSubtileGrid.second;
  }

  // --- Grid utility methods ---

  long getLocalSubtileLinearId(long sId0, long sId1) const {
    return sId1 * localSubtileGrid.first + sId0;
  }

  // --- Tile index mappings ---

  long grLoadIndexForSubtile(long sId0, long sId1, long loadIdx = 0) const {
    long linearId = getLocalSubtileLinearId(sId0, sId1);
    long baseGR = loadRatioGR != 0.0
                      ? static_cast<long>(
                            std::floor(static_cast<double>(linearId) / loadRatioGR))
                      : 0;
    return baseGR + loadIdx;
  }

  long lrTileIndexForSubtile(long sId0, long sId1, long mfmaId = 0) const {
    long linearId = sId1 * lrLocalSubtileGrid.first + sId0;
    long tilesPerSubtile =
        static_cast<long>(lrSubtileShape.first) * lrSubtileShape.second;
    return linearId * tilesPerSubtile + mfmaId;
  }

  std::vector<std::pair<long, long>> globalMmaTilesForSubtile(long sId0,
                                                              long sId1) const {
    long baseRow = sId0 * subtileShape.first;
    long baseCol = sId1 * subtileShape.second;
    return gr.subtileForMmaTile(baseRow, baseCol).mma_tiles;
  }

  std::vector<std::pair<long, long>> waveMmaTilesForSubtile(long sId0,
                                                            long sId1) const {
    long baseRow = sId0 * subtileShape.first;
    long baseCol = sId1 * subtileShape.second;
    std::vector<std::pair<long, long>> tiles;
    for (int m = 0; m < subtileShape.first; ++m) {
      for (int k = 0; k < subtileShape.second; ++k) {
        tiles.emplace_back(baseRow + m, baseCol + k);
      }
    }
    return tiles;
  }

  long grRegGroupForSubtileRow(long sId0) const {
    if (loadRatioGR >= 2.0) {
      return static_cast<long>(
          std::floor(static_cast<double>(sId0) / loadRatioGR));
    }
    return sId0;
  }

  // TileInfo.getSubtileShapeLinearId(k0, k1) = k1 * subtileShape[0] + k0.
  long getSubtileShapeLinearId(long k0, long k1) const {
    return k1 * subtileShape.first + k0;
  }

  // numGRPerSubtile = ceil(1 / loadRatioGR) (0 when loadRatioGR == 0).
  long numGRPerSubtile() const {
    if (loadRatioGR == 0.0) return 0;
    return static_cast<long>(std::ceil(1.0 / loadRatioGR));
  }

  // --- Emit-leaf plans (instruction shape only) ---

  // Pure port of SubtileGREmit.emitSingleBufferLoad's offset arithmetic.
  SingleBufferLoadPlan singleBufferLoadPlan(long sId0, long sId1) const {
    SingleBufferLoadPlan plan;
    long linearId = getLocalSubtileLinearId(sId0, sId1);
    plan.grBaseId =
        loadRatioGR != 0.0
            ? static_cast<long>(std::floor(static_cast<double>(linearId) /
                                           loadRatioGR))
            : 0;
    plan.skip = false;
    if (loadRatioGR > 1.0) {
      long firstInGroup =
          static_cast<long>(static_cast<double>(plan.grBaseId) * loadRatioGR);
      if (linearId != firstInGroup) {
        plan.skip = true;
        plan.offsetK = 0;
        return plan;
      }
    }

    // offsetK = sId1 * int(mmaTileShape[1] * subtileShape[1] * bpe)
    long offsetKUnit = static_cast<long>(static_cast<double>(gr.mmaTileShape.second) *
                                         subtileShape.second * gr.bpe);
    plan.offsetK = sId1 * offsetKUnit;

    long subtileOffset = static_cast<long>(std::ceil(loadRatioGR * subtileSize));
    long numGR = numGRPerSubtile();
    for (long i = 0; i < numGR; ++i) {
      double m0 = static_cast<double>(i) * subtileOffset +
                  (static_cast<double>(sId0) +
                   static_cast<double>(sId1) * globalSubtileGrid.first) *
                      subtileSize;
      plan.m0Offsets.push_back(static_cast<long>(m0));
    }
    return plan;
  }

  // Pure port of SubtileLREmit.emitSingleDsRead's offset / read arithmetic.
  // numRegs is the destination tile register count (Python register state).
  SingleDsReadPlan singleDsReadPlan(long sId0, long sId1, long subIterK,
                                    long numRegs) const {
    SingleDsReadPlan plan;
    plan.regsPerDsRead = lr.loadWidth / 4;
    plan.mfmaId = getSubtileShapeLinearId(subIterK, 0);
    long offsetStride = static_cast<long>(subtileSize);
    plan.offset = sId0 * offsetStride +
                  sId1 * static_cast<long>(globalSubtileGrid.first) *
                      offsetStride;
    plan.numReadsForTile =
        plan.regsPerDsRead != 0 ? numRegs / plan.regsPerDsRead : 0;
    for (long readIdx = 0; readIdx < plan.numReadsForTile; ++readIdx) {
      DsReadEntry e;
      e.dstRegOffset = readIdx * plan.regsPerDsRead;
      e.addrIdx = plan.mfmaId * plan.numReadsForTile + readIdx;
      plan.reads.push_back(e);
    }
    return plan;
  }

  // --- GR / LR offset-assignment math (B16 / TLU0) ---

  // depthUBytes / localSubtileGrid[1] — TileInfo.subIterKBytes for the AB case.
  long subIterKBytes() const {
    long depthUBytes = static_cast<long>(static_cast<double>(depthU) * gr.bpe);
    return depthUBytes / localSubtileGrid.second;
  }

  // Pure port of the scalar math in SubtileGREmit._graTileAssignment_legacy
  // (and its _grComputeRowPartition / _grComputeAllOffsets /
  // _grComputeSubtileOffsets / _grSwizzleColIds helpers) for one tensor.
  // ldsRowBankSize = archCaps["LDSBankCount"] * archCaps["LDSBankWidth"].
  GROffsetAssignPlan grOffsetAssignPlan(long ldsRowBankSize) const {
    GROffsetAssignPlan p;
    p.subIterKBytes = subIterKBytes();
    p.loadWidth = gr.loadWidth;
    p.blockSize = p.subIterKBytes / p.loadWidth;
    p.numRowsPerLDSBanks = ldsRowBankSize / p.subIterKBytes;
    p.numRowsPerWave = waveSize / p.blockSize;
    p.partitionOffset =
        static_cast<long>(gr.mmaTileShape.first) * localSubtileGrid.first;
    if (loadRatioGR == 1.0)
      p.partitionMode = 1;
    else if (loadRatioGR == 0.5)
      p.partitionMode = 0;
    else if (loadRatioGR == 2.0)
      p.partitionMode = 2;
    else
      p.partitionMode = -1;
    p.subtileSizeElems =
        static_cast<long>(subtileShape.first) * gr.mmaTileShape.first;
    p.grAdvanceOffset = static_cast<long>(
        std::ceil(static_cast<double>(p.subtileSizeElems) * loadRatioGR));
    p.bpeBits = static_cast<long>(8 * gr.bpe);
    p.numGRPerSubtile = numGRPerSubtile();
    p.grSubtileRowOffset = static_cast<long>(std::ceil(
        static_cast<double>(p.numGRPerSubtile) * loadRatioGR *
        static_cast<double>(p.subtileSizeElems)));
    p.sStride = static_cast<long>(
        static_cast<double>(p.grSubtileRowOffset) * gr.bpe);
    p.loadRatioGR = loadRatioGR;
    return p;
  }

  // Pure port of the scalar math in SubtileLREmit._lraTileAssignment_legacy
  // (and its _computeLROffset / _applyWavePartitionLROffset helpers) for one
  // tensor. mWavesM is kernel["MIWaveGroup"][0] (used identically for A and B
  // by the legacy wave-partition path).
  LROffsetAssignPlan lrOffsetAssignPlan(long ldsRowBankSize,
                                        long mWavesM) const {
    LROffsetAssignPlan p;
    p.subIterKBytes = subIterKBytes();
    p.loadWidthLR = lr.loadWidth;
    p.loadWidthGR = gr.loadWidth;
    p.blockSize = p.subIterKBytes / p.loadWidthLR;
    p.numRowsPerLDSBanks = ldsRowBankSize / p.subIterKBytes;
    p.miM = lr.mmaTileShape.first;
    p.numMFMACols =
        static_cast<long>(static_cast<double>(lr.mmaTileShape.second) * lr.bpe) /
        p.loadWidthLR;
    p.partitionOffset =
        static_cast<long>(lr.mmaTileShape.first) * localSubtileGrid.first;
    p.sInterval = p.partitionOffset * p.subIterKBytes;
    p.mWavesM = mWavesM;
    p.loadRatioGR = loadRatioGR;
    if (loadRatioGR >= 2.0)
      p.wavePartMode = -1;
    else if (loadRatioGR == 1.0)
      p.wavePartMode = 1;
    else if (loadRatioGR == 0.5)
      p.wavePartMode = 0;
    else
      p.wavePartMode = -2;
    return p;
  }
};

}  // namespace tw::subtile
