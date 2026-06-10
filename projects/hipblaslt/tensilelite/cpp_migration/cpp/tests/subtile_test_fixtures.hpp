// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared fixtures for the native C++ subtile gtest suite.
//
// These reconstruct, directly in C++, the pre-defined gfx950 geometry
// constants (AB_B16, AB_B8, …) and the kernel-derived TileInfo construction
// that the deleted Python ``*Cpp.py`` parity tests built through the Python
// facade. Because the writer-free subtile logic is now C++-only (the Python
// SubtileGeometry / TileInfo layers forward to it unconditionally), these
// tests exercise the C++ headers directly rather than comparing a Python shim
// against an independently-built C++ twin.
//
// The reference-formula helpers mirror the documented math the C++ query layer
// implements (the same golden oracle the Python parity tests used), so the
// query/plan methods are locked against an independent re-derivation rather
// than against themselves.

#pragma once

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensile_writer/emit_leaves.hpp"
#include "tensile_writer/subtile_geometry.hpp"
#include "tensile_writer/tile_info.hpp"

namespace tw_test {

using namespace tw::subtile;

// ---------------------------------------------------------------------------
// Pre-defined gfx950 layout constants (Kernel.py module-level constants).
// ---------------------------------------------------------------------------
inline MMALayout MFMA_16x16_1B_4K_4V() { return MMALayout(16, 1, 4, 64); }
inline MMALayout MFMA_16x16_1B_4K_8V() { return MMALayout(16, 1, 8, 64); }
inline MMALayout MFMA_16x16_1B_4N_4V() { return MMALayout(16, 1, 4, 64); }
inline MMAScaleLayout MFMA_SCALE_16x16_1B_MX32_8V() {
  return MMAScaleLayout(16, 1, 0.25, 32, 64);
}

// ---------------------------------------------------------------------------
// Pre-defined A/B GR + LR geometry pairs (Kernel.py: AB_B16, AB_B8, …).
// Built with the exact dtype params / subtile shapes / load shapes the Python
// ABTilePair constants use.
// ---------------------------------------------------------------------------
struct ABPair {
  ABGRGeometry gr;
  ABLRGeometry lr;
};

// _B16: instK=32, bpe=2 ; _B4: instK=128, bpe=0.5 ; _B8: instK=128, bpe=1.
inline ABPair AB_B16() {
  return {ABGRGeometry(MFMA_16x16_1B_4K_4V(), 32, 2.0, LoadShape(1, 8), {1, 2}),
          ABLRGeometry(MFMA_16x16_1B_4K_4V(), 32, 2.0, LoadShape(1, 8), {1, 2})};
}
inline ABPair AB_B4() {
  return {ABGRGeometry(MFMA_16x16_1B_4K_4V(), 128, 0.5, LoadShape(1, 32), {1, 2}),
          ABLRGeometry(MFMA_16x16_1B_4K_4V(), 128, 0.5, LoadShape(1, 32), {1, 2})};
}
inline ABPair AB_B8() {
  return {ABGRGeometry(MFMA_16x16_1B_4K_8V(), 128, 1.0, LoadShape(1, 16), {1, 1}),
          ABLRGeometry(MFMA_16x16_1B_4K_8V(), 128, 1.0, LoadShape(1, 16), {1, 1})};
}
inline ABPair AB_B4_2x2() {
  return {ABGRGeometry(MFMA_16x16_1B_4K_4V(), 128, 0.5, LoadShape(1, 32), {2, 2},
                       /*subtileCount=*/1, /*subtileStride=*/0),
          ABLRGeometry(MFMA_16x16_1B_4K_4V(), 128, 0.5, LoadShape(1, 32), {2, 2})};
}
inline ABPair AB_B16_2x2() {
  return {ABGRGeometry(MFMA_16x16_1B_4K_4V(), 32, 2.0, LoadShape(1, 8), {2, 2},
                       /*subtileCount=*/1, /*subtileStride=*/0),
          ABLRGeometry(MFMA_16x16_1B_4K_4V(), 32, 2.0, LoadShape(1, 8), {2, 2})};
}
inline ABPair AB_B16_TLU1() {
  return {ABGRGeometry(MFMA_16x16_1B_4K_4V(), 32, 2.0, LoadShape(8, 1), {8, 1},
                       /*subtileCount=*/1, /*subtileStride=*/0, /*tlu=*/true),
          ABLRGeometry(MFMA_16x16_1B_4K_4V(), 32, 2.0, LoadShape(8, 1), {8, 1},
                       /*tlu=*/true)};
}
inline ABPair AB_B16_TLU1_16x1() {
  return {ABGRGeometry(MFMA_16x16_1B_4K_4V(), 32, 2.0, LoadShape(16, 1), {16, 1},
                       /*subtileCount=*/1, /*subtileStride=*/0, /*tlu=*/true,
                       /*loadWidth=*/32),
          ABLRGeometry(MFMA_16x16_1B_4K_4V(), 32, 2.0, LoadShape(16, 1), {16, 1},
                       /*tlu=*/true, /*loadWidth=*/32)};
}

// ---------------------------------------------------------------------------
// MX scale geometry pairs (Kernel.py: MXSA_B4, …). _MXS_B4 / _MXS_B8 share the
// same params; A/B variants are identical too.
// ---------------------------------------------------------------------------
struct MXPair {
  MXScaleGRGeometry gr;
  MXScaleLRGeometry lr;
};

inline MXPair MXS_B4() {
  return {MXScaleGRGeometry(MFMA_SCALE_16x16_1B_MX32_8V(), 128, 1.0, 16),
          MXScaleLRGeometry(MFMA_SCALE_16x16_1B_MX32_8V(), 128, 1.0, 4)};
}
inline MXPair MXS_B8() { return MXS_B4(); }  // identical params

// ---------------------------------------------------------------------------
// C/D output geometry (Kernel.py: CD_F32).
// ---------------------------------------------------------------------------
inline CDTileGeometry CD_F32() {
  return CDTileGeometry(MFMA_16x16_1B_4N_4V(), 4.0, LoadShape(1, 4));
}

// ---------------------------------------------------------------------------
// Minimal kernel config (the fields TileInfo extracts for the AB query layer).
// ---------------------------------------------------------------------------
struct Kernel {
  long macroTileA;
  long macroTileB;
  long depthU;  // _DepthUA == _DepthUB for these cases
  std::array<long, 2> miWaveGroup{4, 1};
  long waveSize{64};
};

inline Kernel make_kernel(long macroTileA, long macroTileB, long depthU,
                          std::array<long, 2> waveGroup = {4, 1}) {
  Kernel k;
  k.macroTileA = macroTileA;
  k.macroTileB = macroTileB;
  k.depthU = depthU;
  k.miWaveGroup = waveGroup;
  k.waveSize = 64;
  return k;
}

// Build the C++ ABTileInfoQuery for one tensor component, exactly mirroring
// TileInfo.__init__ scalar extraction + gr.for_kernel materialization.
inline ABTileInfoQuery make_query(const ABPair& pair, const std::string& tc,
                                  const Kernel& k) {
  bool isA = (tc == "A");
  long macroTile = isA ? k.macroTileA : k.macroTileB;
  long depthU = k.depthU;
  long waveGroupSize = isA ? k.miWaveGroup[0] : k.miWaveGroup[1];
  long waveSize = k.waveSize;
  long numWaves = k.miWaveGroup[0] * k.miWaveGroup[1];

  long mt_mma = floordiv(macroTile, pair.gr.mmaTileShape.first);
  ABGRGeometry gr_cfg = pair.gr.forKernel(static_cast<int>(waveGroupSize), mt_mma);

  return ABTileInfoQuery(gr_cfg, pair.lr, macroTile, depthU, waveGroupSize,
                         waveSize, numWaves);
}

// ---------------------------------------------------------------------------
// Reference formulas — the documented math the C++ query/plan layer implements.
// Computed from the query's exposed construction state, mirroring the Python
// _ref_* oracle helpers in the deleted parity tests.
// ---------------------------------------------------------------------------
inline long ref_local_subtile_linear_id(const ABTileInfoQuery& q, long s0,
                                         long s1) {
  return s1 * q.localSubtileGrid.first + s0;
}

inline long ref_gr_load_index(const ABTileInfoQuery& q, long s0, long s1,
                              long loadIdx = 0) {
  long linearId = ref_local_subtile_linear_id(q, s0, s1);
  long baseGR =
      q.loadRatioGR != 0.0
          ? static_cast<long>(std::floor(static_cast<double>(linearId) /
                                          q.loadRatioGR))
          : 0;
  return baseGR + loadIdx;
}

inline long ref_lr_tile_index(const ABTileInfoQuery& q, long s0, long s1,
                              long mfmaId = 0) {
  long linearId = s1 * q.lrLocalSubtileGrid.first + s0;
  long tilesPerSubtile =
      static_cast<long>(q.lrSubtileShape.first) * q.lrSubtileShape.second;
  return linearId * tilesPerSubtile + mfmaId;
}

inline std::vector<std::pair<long, long>> ref_wave_mma_tiles(
    const ABTileInfoQuery& q, long s0, long s1) {
  long baseRow = s0 * q.subtileShape.first;
  long baseCol = s1 * q.subtileShape.second;
  std::vector<std::pair<long, long>> tiles;
  for (int m = 0; m < q.subtileShape.first; ++m)
    for (int kk = 0; kk < q.subtileShape.second; ++kk)
      tiles.emplace_back(baseRow + m, baseCol + kk);
  return tiles;
}

inline long ref_gr_reg_group(const ABTileInfoQuery& q, long s0) {
  if (q.loadRatioGR >= 2.0)
    return static_cast<long>(std::floor(static_cast<double>(s0) / q.loadRatioGR));
  return s0;
}

// Reference single-buffer-load plan (skip flag, offsetK, m0 offsets).
struct RefBufferLoadPlan {
  bool skip;
  long grBaseId;
  long offsetK;
  std::vector<long> m0Offsets;
};

inline RefBufferLoadPlan ref_single_buffer_load_plan(const ABTileInfoQuery& q,
                                                     long s0, long s1) {
  RefBufferLoadPlan p;
  long linearId = s1 * q.localSubtileGrid.first + s0;
  p.grBaseId = q.loadRatioGR != 0.0
                   ? static_cast<long>(std::floor(
                         static_cast<double>(linearId) / q.loadRatioGR))
                   : 0;
  p.skip = false;
  if (q.loadRatioGR > 1.0) {
    long firstInGroup =
        static_cast<long>(static_cast<double>(p.grBaseId) * q.loadRatioGR);
    if (linearId != firstInGroup) {
      p.skip = true;
      p.offsetK = 0;
      return p;
    }
  }
  long offsetKUnit = static_cast<long>(static_cast<double>(q.gr.mmaTileShape.second) *
                                       q.subtileShape.second * q.gr.bpe);
  p.offsetK = s1 * offsetKUnit;
  long subtileOffset =
      static_cast<long>(std::ceil(q.loadRatioGR * q.subtileSize));
  long numGR = q.numGRPerSubtile();
  for (long i = 0; i < numGR; ++i) {
    double m0 = static_cast<double>(i) * subtileOffset +
                (static_cast<double>(s0) +
                 static_cast<double>(s1) * q.globalSubtileGrid.first) *
                    q.subtileSize;
    p.m0Offsets.push_back(static_cast<long>(m0));
  }
  return p;
}

// Reference single-ds-read plan (DS offset, register stride, per-read map).
struct RefDsReadPlan {
  long regsPerDsRead;
  long mfmaId;
  long offset;
  long numReadsForTile;
  std::vector<std::pair<long, long>> reads;  // (dstRegOffset, addrIdx)
};

inline RefDsReadPlan ref_single_ds_read_plan(const ABTileInfoQuery& q, long s0,
                                             long s1, long subIterK,
                                             long numRegs) {
  RefDsReadPlan p;
  p.regsPerDsRead = q.lr.loadWidth / 4;
  p.mfmaId = q.getSubtileShapeLinearId(subIterK, 0);
  long offsetStride = static_cast<long>(q.subtileSize);
  p.offset = s0 * offsetStride +
             s1 * static_cast<long>(q.globalSubtileGrid.first) * offsetStride;
  p.numReadsForTile =
      p.regsPerDsRead != 0 ? numRegs / p.regsPerDsRead : 0;
  for (long r = 0; r < p.numReadsForTile; ++r)
    p.reads.emplace_back(r * p.regsPerDsRead, p.mfmaId * p.numReadsForTile + r);
  return p;
}

// ---------------------------------------------------------------------------
// Reference GR / LR offset-assignment plans — independent re-derivation of the
// documented scalar math (SubtileGREmit/SubtileLREmit legacy emit) from the
// query's primitive construction state. Locks grOffsetAssignPlan /
// lrOffsetAssignPlan against the formula, not against themselves. depthUBytes
// uses int(depthU * bpe) so fp4 (bpe 0.5) rounds the product rather than
// truncating bpe to 0 before the multiply.
// ---------------------------------------------------------------------------
struct RefGRPlan {
  long subIterKBytes, loadWidth, blockSize, numRowsPerLDSBanks, numRowsPerWave;
  long partitionOffset;
  int partitionMode;
  long subtileSizeElems, grAdvanceOffset, bpeBits, grSubtileRowOffset, sStride,
      numGRPerSubtile;
  double loadRatioGR;
  bool isFp8;
};

inline RefGRPlan ref_gr_offset_assign_plan(const ABTileInfoQuery& q, long lds) {
  RefGRPlan p;
  long depthUBytes = static_cast<long>(q.depthU * q.gr.bpe);
  p.subIterKBytes = depthUBytes / q.localSubtileGrid.second;
  p.loadWidth = q.gr.loadWidth;
  p.blockSize = p.subIterKBytes / p.loadWidth;
  p.numRowsPerLDSBanks = lds / p.subIterKBytes;
  p.numRowsPerWave = q.waveSize / p.blockSize;
  p.partitionOffset =
      static_cast<long>(q.gr.mmaTileShape.first) * q.localSubtileGrid.first;
  p.partitionMode = q.loadRatioGR == 1.0   ? 1
                    : q.loadRatioGR == 0.5 ? 0
                    : q.loadRatioGR == 2.0 ? 2
                                           : -1;
  p.subtileSizeElems =
      static_cast<long>(q.subtileShape.first) * q.gr.mmaTileShape.first;
  p.grAdvanceOffset = static_cast<long>(
      std::ceil(static_cast<double>(p.subtileSizeElems) * q.loadRatioGR));
  p.bpeBits = static_cast<long>(8 * q.gr.bpe);
  p.numGRPerSubtile = q.numGRPerSubtile();
  p.grSubtileRowOffset = static_cast<long>(
      std::ceil(static_cast<double>(p.numGRPerSubtile) * q.loadRatioGR *
                static_cast<double>(p.subtileSizeElems)));
  p.sStride =
      static_cast<long>(static_cast<double>(p.grSubtileRowOffset) * q.gr.bpe);
  p.loadRatioGR = q.loadRatioGR;
  p.isFp8 = (q.gr.bpe == 1.0);
  return p;
}

struct RefLRPlan {
  long subIterKBytes, loadWidthLR, loadWidthGR, blockSize, numRowsPerLDSBanks;
  long miM, numMFMACols, partitionOffset, sInterval, mWavesM;
  int wavePartMode;
  double loadRatioGR;
  bool isFp8;
};

inline RefLRPlan ref_lr_offset_assign_plan(const ABTileInfoQuery& q, long lds,
                                           long mWavesM) {
  RefLRPlan p;
  long depthUBytes = static_cast<long>(q.depthU * q.gr.bpe);
  p.subIterKBytes = depthUBytes / q.localSubtileGrid.second;
  p.loadWidthLR = q.lr.loadWidth;
  p.loadWidthGR = q.gr.loadWidth;
  p.blockSize = p.subIterKBytes / p.loadWidthLR;
  p.numRowsPerLDSBanks = lds / p.subIterKBytes;
  p.miM = q.lr.mmaTileShape.first;
  p.numMFMACols =
      static_cast<long>(static_cast<double>(q.lr.mmaTileShape.second) *
                        q.lr.bpe) /
      p.loadWidthLR;
  p.partitionOffset =
      static_cast<long>(q.lr.mmaTileShape.first) * q.localSubtileGrid.first;
  p.sInterval = p.partitionOffset * p.subIterKBytes;
  p.mWavesM = mWavesM;
  p.loadRatioGR = q.loadRatioGR;
  p.wavePartMode = q.loadRatioGR >= 2.0   ? -1
                   : q.loadRatioGR == 1.0 ? 1
                   : q.loadRatioGR == 0.5 ? 0
                                          : -2;
  p.isFp8 = (q.lr.bpe == 1.0);
  return p;
}

}  // namespace tw_test
