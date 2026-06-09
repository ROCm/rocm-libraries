// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Pure C++ port of the geometry value/query layer from
// Tensile/Components/Subtile/SubtileGeometry.py.
//
// This header is intentionally free of any nanobind / Python dependency so the
// geometry math can be unit-tested and reasoned about as plain C++. The
// nanobind bindings live in src/subtile_geometry_bindings.cpp.
//
// Only *pure geometry math* is ported here. No writer state, register
// allocation, rocisa instruction emission, or main-loop logic is included.

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace tw::subtile {

// Python's // (floor division) and % differ from C++ truncation for negative
// operands. All geometry indices used here are non-negative, but we provide
// floor helpers so the port matches CPython semantics exactly regardless.
inline long floordiv(long a, long b) {
  long q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) {
    --q;
  }
  return q;
}

inline long floormod(long a, long b) {
  long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
    r += b;
  }
  return r;
}

// ---------------------------------------------------------------------------
// LoadShape
// ---------------------------------------------------------------------------
struct LoadShape {
  int m;
  int k;

  LoadShape(int m_ = 1, int k_ = 1) : m(m_), k(k_) {}

  bool operator==(const LoadShape& o) const { return m == o.m && k == o.k; }
};

// ---------------------------------------------------------------------------
// MMALayout — data-type independent MFMA/WMMA lane layout
// ---------------------------------------------------------------------------
struct MMALayout {
  int instM;
  int blocks;
  int vgprs;
  int waveSize;

  // Derived
  int contiguousLanes;
  int kGroups;
  int elementsPerLaneNonK;

  MMALayout(int instM_, int blocks_ = -1, int vgprs_ = -1, int waveSize_ = -1)
      : instM(instM_), blocks(blocks_), vgprs(vgprs_), waveSize(waveSize_) {
    contiguousLanes = instM;
    kGroups = static_cast<int>(
        floordiv(waveSize, static_cast<long>(contiguousLanes) * blocks));
    elementsPerLaneNonK = static_cast<int>(floordiv(instM, kGroups));
  }

  long inputBytesPerLane() const { return static_cast<long>(vgprs) * 4; }

  long tileSizeBytes(int instK, double elementBytes) const {
    return static_cast<long>(static_cast<double>(instM) * instK * elementBytes);
  }

  double regsPerTile(int instK, double elementBytes) const {
    return static_cast<double>(tileSizeBytes(instK, elementBytes)) / waveSize / 4.0;
  }
};

// ---------------------------------------------------------------------------
// MMAScaleLayout — data-type independent MX scale factor lane layout
// ---------------------------------------------------------------------------
struct MMAScaleLayout {
  int instM;
  int blocks;
  double vgprs;
  int mxBlock;
  int waveSize;

  // Derived
  int contiguousLanes;

  MMAScaleLayout(int instM_, int blocks_ = -1, double vgprs_ = -1,
                 int mxBlock_ = -1, int waveSize_ = -1)
      : instM(instM_),
        blocks(blocks_),
        vgprs(vgprs_),
        mxBlock(mxBlock_),
        waveSize(waveSize_) {
    contiguousLanes = instM;
  }
};

// ---------------------------------------------------------------------------
// ABInputBase — shared dtype params + derived attrs for A/B GR and LR
// ---------------------------------------------------------------------------
struct ABInputBase {
  MMALayout mmaLayout;
  int instK;
  double bpe;
  bool tlu;
  LoadShape loadShape;
  int loadWidth;

  // Derived
  std::pair<int, int> mmaTileShape;
  long mmaTileSize;
  double mmaTileRegCount;

  ABInputBase(const MMALayout& mmaLayout_, int instK_, double bpe_, bool tlu_,
              const LoadShape& loadShape_, int loadWidth_)
      : mmaLayout(mmaLayout_),
        instK(instK_),
        bpe(bpe_),
        tlu(tlu_),
        loadShape(loadShape_),
        loadWidth(loadWidth_) {
    int instM = mmaLayout.instM;
    mmaTileSize = static_cast<long>(static_cast<double>(instM) * instK * bpe);
    mmaTileShape = {instM, instK};
    mmaTileRegCount = static_cast<double>(mmaLayout.vgprs);
  }

  std::pair<long, long> globalMMATileGrid(long macroTile, long depthU) const {
    return {floordiv(macroTile, mmaTileShape.first),
            floordiv(depthU, mmaTileShape.second)};
  }

  std::pair<long, long> localMMATileGrid(long macroTile, long depthU,
                                         long waveGroupSize) const {
    auto glbl = globalMMATileGrid(macroTile, depthU);
    return {floordiv(glbl.first, waveGroupSize), glbl.second};
  }
};

// ---------------------------------------------------------------------------
// ABGRGeometry — A/B tile geometry for global reads
// ---------------------------------------------------------------------------
struct SubtileForMmaTileResult {
  std::pair<long, long> subtile_id;
  std::pair<int, int> block_shape;
  std::vector<std::pair<long, long>> mma_tiles;
};

struct ABGRGeometry : ABInputBase {
  std::pair<int, int> subtileShape;
  std::optional<int> subtileCount;
  std::optional<int> subtileStride;

  ABGRGeometry(const MMALayout& mmaLayout_, int instK_, double bpe_,
               const LoadShape& loadShape_,
               std::pair<int, int> subtileShape_ = {1, 1},
               std::optional<int> subtileCount_ = std::nullopt,
               std::optional<int> subtileStride_ = std::nullopt,
               bool tlu_ = false, int loadWidth_ = 16)
      : ABInputBase(mmaLayout_, instK_, bpe_, tlu_, loadShape_, loadWidth_),
        subtileShape(subtileShape_),
        subtileCount(subtileCount_),
        subtileStride(subtileStride_) {}

  std::pair<double, double> globalSubtileGrid(long macroTile,
                                              long depthU) const {
    auto glbl = globalMMATileGrid(macroTile, depthU);
    return {static_cast<double>(glbl.first) / subtileShape.first,
            static_cast<double>(glbl.second) / subtileShape.second};
  }

  double subtileSizeBytes() const {
    return static_cast<double>(subtileShape.first) * subtileShape.second *
           mmaTileSize;
  }

  long bytesPerLoad(long numWaves) const {
    long perLane = static_cast<long>(static_cast<double>(loadShape.m) *
                                     loadShape.k * bpe);
    return perLane * mmaLayout.waveSize * numWaves;
  }

  double loadsPerStrip(long numWaves) const {
    return subtileSizeBytes() / static_cast<double>(bytesPerLoad(numWaves));
  }

  std::pair<int, int> localGRGranularity(long numWaves) const {
    int bK = subtileShape.second;
    int bc = subtileCount.has_value() ? *subtileCount : 1;
    if (bc > 1) {
      return {1, bK};
    }
    double blocksPerLoad = bytesPerLoad(numWaves) / subtileSizeBytes();
    if (blocksPerLoad > 1) {
      return {static_cast<int>(blocksPerLoad), bK};
    }
    return {1, bK};
  }

  // Returns the (subtileCount, subtileStride) derived for a kernel config.
  // wg_m  = MIWaveGroup[0 if tc=='A' else 1]
  // mt_mma = MacroTile{tc} // mmaTileShape[0]
  std::pair<int, int> forKernelParams(int wg_m, long mt_mma) const {
    int bc = subtileCount.has_value() ? *subtileCount : wg_m;
    int bstride = subtileStride.has_value()
                      ? *subtileStride
                      : static_cast<int>(floordiv(mt_mma, bc));
    return {bc, bstride};
  }

  ABGRGeometry forKernel(int wg_m, long mt_mma) const {
    auto [bc, bstride] = forKernelParams(wg_m, mt_mma);
    ABGRGeometry out = *this;
    out.subtileCount = bc;
    out.subtileStride = bstride;
    return out;
  }

  SubtileForMmaTileResult subtileForMmaTile(long r, long c) const {
    // Caller must ensure subtileCount/subtileStride are materialized.
    int bM = subtileShape.first;
    int bK = subtileShape.second;
    int bc = *subtileCount;
    int bstr = *subtileStride;

    long subtile_k = floordiv(c, bK);
    std::vector<long> k_cols;
    for (long col = subtile_k * bK; col < (subtile_k + 1) * bK; ++col) {
      k_cols.push_back(col);
    }

    long subtile_m;
    std::vector<long> m_rows;
    if (bc == 1 || bstr == 0) {
      subtile_m = floordiv(r, bM);
      for (long row = subtile_m * bM; row < (subtile_m + 1) * bM; ++row) {
        m_rows.push_back(row);
      }
    } else {
      long bM_per_stride = floordiv(bstr, bM);
      long stride_group = floordiv(r, bstr);
      long major_group = floordiv(stride_group, bc);
      long within_major = floordiv(floormod(r, bstr), bM);
      subtile_m = major_group * bM_per_stride + within_major;
      for (int i = 0; i < bc; ++i) {
        long base = (major_group * bc + i) * bstr + within_major * bM;
        for (long row = base; row < base + bM; ++row) {
          m_rows.push_back(row);
        }
      }
    }

    SubtileForMmaTileResult res;
    res.subtile_id = {subtile_m, subtile_k};
    res.block_shape = {bM, bK};
    for (long row : m_rows) {
      for (long col : k_cols) {
        res.mma_tiles.emplace_back(row, col);
      }
    }
    return res;
  }
};

// ---------------------------------------------------------------------------
// ABLRGeometry — A/B tile geometry for local reads
// ---------------------------------------------------------------------------
struct ABLRGeometry : ABInputBase {
  std::pair<int, int> subtileShape;

  ABLRGeometry(const MMALayout& mmaLayout_, int instK_, double bpe_,
               const LoadShape& loadShape_,
               std::pair<int, int> subtileShape_ = {1, 1}, bool tlu_ = false,
               int loadWidth_ = 16)
      : ABInputBase(mmaLayout_, instK_, bpe_, tlu_, loadShape_, loadWidth_),
        subtileShape(subtileShape_) {}

  std::pair<double, double> globalSubtileGrid(long macroTile,
                                              long depthU) const {
    auto glbl = globalMMATileGrid(macroTile, depthU);
    return {static_cast<double>(glbl.first) / subtileShape.first,
            static_cast<double>(glbl.second) / subtileShape.second};
  }

  double subtileSizeBytes() const {
    return static_cast<double>(subtileShape.first) * subtileShape.second *
           mmaTileSize;
  }
};

// ---------------------------------------------------------------------------
// CDTileGeometry — output (C/D) tile geometry
// ---------------------------------------------------------------------------
struct CDTileGeometry {
  MMALayout mmaLayout;
  double bpe;
  LoadShape storeShape;

  // Derived
  std::pair<int, int> mmaTileShape;
  long mmaTileSize;
  double mmaTileRegCount;

  CDTileGeometry(const MMALayout& mmaLayout_, double bpe_,
                 const LoadShape& storeShape_ = LoadShape(1, 1))
      : mmaLayout(mmaLayout_), bpe(bpe_), storeShape(storeShape_) {
    int instM = mmaLayout.instM;
    mmaTileSize = static_cast<long>(static_cast<double>(instM) * instM * bpe);
    mmaTileShape = {instM, instM};
    mmaTileRegCount =
        static_cast<double>(mmaTileSize) / mmaLayout.waveSize / 4.0;
  }

  std::pair<long, long> globalMMATileGrid(long macroTile0,
                                          long macroTile1) const {
    int instM = mmaLayout.instM;
    return {floordiv(macroTile0, instM), floordiv(macroTile1, instM)};
  }

  std::pair<long, long> localMMATileGrid(long macroTile0, long macroTile1,
                                         std::pair<long, long> waveGroup) const {
    auto glbl = globalMMATileGrid(macroTile0, macroTile1);
    return {floordiv(glbl.first, waveGroup.first),
            floordiv(glbl.second, waveGroup.second)};
  }

  std::pair<double, double> globalSubtileGrid(
      long macroTile0, long macroTile1,
      std::pair<double, double> subtileShape) const {
    auto glbl = globalMMATileGrid(macroTile0, macroTile1);
    return {glbl.first / subtileShape.first,
            glbl.second / subtileShape.second};
  }

  std::pair<double, double> localSubtileGrid(
      long macroTile0, long macroTile1, std::pair<long, long> waveGroup,
      std::pair<double, double> subtileShape) const {
    auto locl = localMMATileGrid(macroTile0, macroTile1, waveGroup);
    return {locl.first / subtileShape.first,
            locl.second / subtileShape.second};
  }
};

// ---------------------------------------------------------------------------
// MXScale geometries
// ---------------------------------------------------------------------------
struct MXScaleInputBase {
  MMAScaleLayout scaleLayout;
  int instK;
  double bpe;
  int loadWidth;

  // Derived
  std::pair<int, int> mmaTileShape;
  long mmaTileSize;
  double mmaTileRegCount;

  MXScaleInputBase(const MMAScaleLayout& scaleLayout_, int instK_, double bpe_,
                   int loadWidth_)
      : scaleLayout(scaleLayout_),
        instK(instK_),
        bpe(bpe_),
        loadWidth(loadWidth_) {
    int instM = scaleLayout.instM;
    int instKScale = static_cast<int>(floordiv(instK, scaleLayout.mxBlock));
    mmaTileSize =
        static_cast<long>(static_cast<double>(instM) * instKScale * bpe);
    mmaTileShape = {instM, instKScale};
    mmaTileRegCount =
        static_cast<double>(mmaTileSize) / scaleLayout.waveSize / 4.0;
  }

  std::pair<long, long> globalMMATileGrid(long macroTile, long depthU) const {
    return {floordiv(macroTile, mmaTileShape.first), floordiv(depthU, instK)};
  }
};

struct MXScaleGRGeometry : MXScaleInputBase {
  std::optional<std::pair<int, int>> subtileShape;

  MXScaleGRGeometry(const MMAScaleLayout& scaleLayout_, int instK_, double bpe_,
                    int loadWidth_ = 16,
                    std::optional<std::pair<int, int>> subtileShape_ =
                        std::nullopt)
      : MXScaleInputBase(scaleLayout_, instK_, bpe_, loadWidth_),
        subtileShape(subtileShape_) {}

  // Returns the derived subtileShape (mt_mma, du_scale) for a kernel config,
  // or the pinned shape if already set.
  // mt_mma   = MacroTile{tc} // scaleLayout.instM
  // du_scale = _DepthU{tc}   // instK
  std::pair<int, int> forKernelShape(long macroTileVal, long depthUVal) const {
    if (subtileShape.has_value()) {
      return *subtileShape;
    }
    int instM = scaleLayout.instM;
    int mt_mma = static_cast<int>(floordiv(macroTileVal, instM));
    int du_scale = static_cast<int>(floordiv(depthUVal, instK));
    return {mt_mma, du_scale};
  }

  MXScaleGRGeometry forKernel(long macroTileVal, long depthUVal) const {
    if (subtileShape.has_value()) {
      return *this;
    }
    MXScaleGRGeometry out = *this;
    out.subtileShape = forKernelShape(macroTileVal, depthUVal);
    return out;
  }
};

struct MXScaleLRGeometry : MXScaleInputBase {
  std::pair<int, int> subtileShape;

  MXScaleLRGeometry(const MMAScaleLayout& scaleLayout_, int instK_, double bpe_,
                    int loadWidth_ = 16,
                    std::pair<int, int> subtileShape_ = {2, 2})
      : MXScaleInputBase(scaleLayout_, instK_, bpe_, loadWidth_),
        subtileShape(subtileShape_) {}

  std::pair<double, double> globalSubtileGrid(long macroTile,
                                              long depthU) const {
    auto glbl = globalMMATileGrid(macroTile, depthU);
    return {static_cast<double>(glbl.first) / subtileShape.first,
            static_cast<double>(glbl.second) / subtileShape.second};
  }

  double subtileSizeBytes() const {
    return static_cast<double>(subtileShape.first) * subtileShape.second *
           mmaTileSize;
  }
};

}  // namespace tw::subtile
