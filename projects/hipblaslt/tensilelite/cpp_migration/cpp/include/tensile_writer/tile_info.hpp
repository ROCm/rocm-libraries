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
};

}  // namespace tw::subtile
