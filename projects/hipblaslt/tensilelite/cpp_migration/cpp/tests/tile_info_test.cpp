// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Native C++ port of Tensile/Tests/unit/test_tileInfoCpp.py.
//
// The Python file pinned the read-only TileInfo query values produced by the
// (C++-only) ABTileInfoQuery and checked the C++ snapshot's derived state
// against the Python TileInfo.__init__ state. With the query layer C++-only,
// these tests:
//   * lock each grid/index query method against the documented reference math
//     (golden oracle), and
//   * pin absolute gfx950 AB_B16 values, including the fractional
//     lrGlobalSubtileGrid case that previously exposed a narrowing bug.

#include <gtest/gtest.h>

#include "subtile_test_fixtures.hpp"

using namespace tw::subtile;
using namespace tw_test;

namespace {

// Each AB pair paired with a kernel whose macroTile/depthU yield a
// non-degenerate (strictly positive) subtile grid for both tc='A' (waveGroup 4)
// and tc='B' (waveGroup 1). Tighter-K (B4) and tall-M (TLU1/16x1) shapes need a
// larger depthU / MacroTileA than the AB_B16 base to tile cleanly.
struct NamedPair {
  std::string name;
  ABPair pair;
  Kernel kernel;
};

std::vector<NamedPair> query_pairs() {
  Kernel base = make_kernel(256, 128, 128);
  return {
      {"AB_B16", AB_B16(), base},
      {"AB_B8", AB_B8(), base},
      {"AB_B16_2x2", AB_B16_2x2(), base},
      {"AB_B4", AB_B4(), make_kernel(256, 128, 256)},
      {"AB_B4_2x2", AB_B4_2x2(), make_kernel(256, 128, 256)},
      {"AB_B16_TLU1", AB_B16_TLU1(), make_kernel(512, 128, 128)},
      {"AB_B16_TLU1_16x1", AB_B16_TLU1_16x1(), make_kernel(1024, 256, 128)},
  };
}

}  // namespace

// ---------------------------------------------------------------------------
// Query-method values must match the documented reference math over the grid.
// ---------------------------------------------------------------------------
TEST(TileInfoQueryValues, GridIndexQueries) {
  for (const auto& np : query_pairs()) {
    for (const std::string tc : {"A", "B"}) {
      ABTileInfoQuery q = make_query(np.pair, tc, np.kernel);
      long g0 = q.localSubtileGrid.first;
      long g1 = q.localSubtileGrid.second;
      SCOPED_TRACE(np.name + "." + tc);
      ASSERT_GT(g0, 0);
      ASSERT_GT(g1, 0);
      long lrTiles =
          static_cast<long>(q.lrSubtileShape.first) * q.lrSubtileShape.second;
      for (long s0 = 0; s0 < g0; ++s0) {
        for (long s1 = 0; s1 < g1; ++s1) {
          EXPECT_EQ(q.getLocalSubtileLinearId(s0, s1),
                    ref_local_subtile_linear_id(q, s0, s1));
          EXPECT_EQ(q.grLoadIndexForSubtile(s0, s1),
                    ref_gr_load_index(q, s0, s1));
          EXPECT_EQ(q.grLoadIndexForSubtile(s0, s1, 1),
                    ref_gr_load_index(q, s0, s1, 1));
          for (long mfmaId = 0; mfmaId < lrTiles; ++mfmaId) {
            EXPECT_EQ(q.lrTileIndexForSubtile(s0, s1, mfmaId),
                      ref_lr_tile_index(q, s0, s1, mfmaId));
          }
          EXPECT_EQ(q.waveMmaTilesForSubtile(s0, s1),
                    ref_wave_mma_tiles(q, s0, s1));
          // globalMmaTilesForSubtile must be non-empty and in-grid.
          auto glob = q.globalMmaTilesForSubtile(s0, s1);
          EXPECT_FALSE(glob.empty());
          for (auto [m, kk] : glob) {
            EXPECT_GE(m, 0);
            EXPECT_LT(m, q.globalMMATileGrid.first);
            EXPECT_GE(kk, 0);
            EXPECT_LT(kk, q.globalMMATileGrid.second);
          }
        }
        EXPECT_EQ(q.grRegGroupForSubtileRow(s0), ref_gr_reg_group(q, s0));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Derived snapshot state must match the documented construction formulas.
// ---------------------------------------------------------------------------
TEST(TileInfoSnapshot, DerivedStateSelfConsistent) {
  for (const auto& np : query_pairs()) {
    for (const std::string tc : {"A", "B"}) {
      ABTileInfoQuery q = make_query(np.pair, tc, np.kernel);
      SCOPED_TRACE(np.name + "." + tc);
      // localSubtileGrid = int(localMMATileGrid / subtileShape).
      EXPECT_EQ(q.localSubtileGrid.first,
                static_cast<long>(static_cast<double>(q.localMMATileGrid.first) /
                                  q.subtileShape.first));
      EXPECT_EQ(q.localSubtileGrid.second,
                static_cast<long>(static_cast<double>(q.localMMATileGrid.second) /
                                  q.subtileShape.second));
      // count properties.
      EXPECT_EQ(q.numMFMATiles(),
                q.localMMATileGrid.first * q.localMMATileGrid.second);
      EXPECT_EQ(q.numLocalSubtiles(),
                q.localSubtileGrid.first * q.localSubtileGrid.second);
      EXPECT_EQ(q.numGlobalSubtiles(),
                static_cast<long>(q.globalSubtileGrid.first *
                                  q.globalSubtileGrid.second));
      // AB: LR iterates over the GR subtile grid.
      EXPECT_EQ(q.lrLocalSubtileGrid, q.localSubtileGrid);
    }
  }
}

// lrGlobalSubtileGrid is a raw (possibly fractional) float grid; the C++
// snapshot must preserve the fraction rather than truncating to int. The
// 16-row TLU1 LR subtile over MacroTileB=128 yields a half-tile in M (8 MMA
// tiles / 16 rows = 0.5), which previously exposed a C++ narrowing divergence.
TEST(TileInfoSnapshot, FractionalLrGlobalSubtileGrid) {
  ABTileInfoQuery q =
      make_query(AB_B16_TLU1_16x1(), "B", make_kernel(512, 128, 128));
  EXPECT_DOUBLE_EQ(q.lrGlobalSubtileGrid.first, 0.5);
  EXPECT_DOUBLE_EQ(q.lrGlobalSubtileGrid.second, 4.0);
}

// ---------------------------------------------------------------------------
// Absolute-value pins for an AB_B16 gfx950 case.
// ---------------------------------------------------------------------------
TEST(TileInfoAbsolute, AbB16AKnownValues) {
  ABTileInfoQuery q = make_query(AB_B16(), "A", make_kernel(256, 128, 128));
  // MacroTileA=256, depthU=128, MMA tile (16,32), subtileShape (1,2),
  // waveGroupSize=4 -> localMMATileGrid=(4,4), localSubtileGrid=(4,2).
  EXPECT_EQ(q.localMMATileGrid.first, 4);
  EXPECT_EQ(q.localMMATileGrid.second, 4);
  EXPECT_EQ(q.localSubtileGrid.first, 4);
  EXPECT_EQ(q.localSubtileGrid.second, 2);
  EXPECT_DOUBLE_EQ(q.loadRatioGR, 0.5);
  // getLocalSubtileLinearId: sId1*localSubtileGrid[0] + sId0.
  EXPECT_EQ(q.getLocalSubtileLinearId(3, 1), 7);
  // loadRatioGR=0.5 < 2 -> grRegGroup is identity.
  EXPECT_EQ(q.grRegGroupForSubtileRow(3), 3);
  // baseGR = floor(linearId / 0.5) = linearId*2.
  EXPECT_EQ(q.grLoadIndexForSubtile(3, 1), 14);
  EXPECT_EQ(q.grLoadIndexForSubtile(3, 1, 1), 15);
  // globalMmaTilesForSubtile exact pin for subtile (1,0).
  std::vector<std::pair<long, long>> expected = {
      {1, 0}, {1, 1}, {5, 0}, {5, 1}, {9, 0}, {9, 1}, {13, 0}, {13, 1}};
  EXPECT_EQ(q.globalMmaTilesForSubtile(1, 0), expected);
}

// ---------------------------------------------------------------------------
// GR / LR offset-assignment plans (now C++-only for every AB geometry).
//
// These pin the scalar offset-assignment math the Python emitter consumes
// (SubtileGREmit.graTileAssignment / SubtileLREmit.lraTileAssignment) for
// BF16, FP4, FP8, the 2x2 tile-shape variants, and the column-major TLU1
// shapes — the geometries the deleted Python ``_legacy`` planners covered.
// The emitted rocisa strings for these same plans are locked end-to-end by
// Tensile/Tests/unit/test_subtileOffsetAssignCpp.py against a golden snapshot.
// ---------------------------------------------------------------------------

// gfx950 LDS row bank size = archCaps["LDSBankCount"](64) * LDSBankWidth(4).
static constexpr long kLdsRowBankSize = 256;

TEST(TileInfoOffsetAssignPlan, GrLrPlansMatchReference) {
  for (const auto& np : query_pairs()) {
    for (const std::string tc : {"A", "B"}) {
      ABTileInfoQuery q = make_query(np.pair, tc, np.kernel);
      SCOPED_TRACE(np.name + "." + tc);
      const long subK = q.subIterKBytes();
      if (subK <= 0) continue;  // degenerate K-tiling; offset assign undefined
      const long mWavesM = np.kernel.miWaveGroup[0];

      // GR divides waveSize by blockSize, so guard a positive blockSize.
      if (subK / q.gr.loadWidth > 0) {
        GROffsetAssignPlan g = q.grOffsetAssignPlan(kLdsRowBankSize);
        RefGRPlan r = ref_gr_offset_assign_plan(q, kLdsRowBankSize);
        EXPECT_EQ(g.subIterKBytes, r.subIterKBytes);
        EXPECT_EQ(g.loadWidth, r.loadWidth);
        EXPECT_EQ(g.blockSize, r.blockSize);
        EXPECT_EQ(g.numRowsPerLDSBanks, r.numRowsPerLDSBanks);
        EXPECT_EQ(g.numRowsPerWave, r.numRowsPerWave);
        EXPECT_EQ(g.partitionOffset, r.partitionOffset);
        EXPECT_EQ(g.partitionMode, r.partitionMode);
        EXPECT_EQ(g.subtileSizeElems, r.subtileSizeElems);
        EXPECT_EQ(g.grAdvanceOffset, r.grAdvanceOffset);
        EXPECT_EQ(g.bpeBits, r.bpeBits);
        EXPECT_EQ(g.grSubtileRowOffset, r.grSubtileRowOffset);
        EXPECT_EQ(g.sStride, r.sStride);
        EXPECT_EQ(g.numGRPerSubtile, r.numGRPerSubtile);
        EXPECT_DOUBLE_EQ(g.loadRatioGR, r.loadRatioGR);
        EXPECT_EQ(g.isFp8, r.isFp8);
      }

      LROffsetAssignPlan l = q.lrOffsetAssignPlan(kLdsRowBankSize, mWavesM);
      RefLRPlan rl = ref_lr_offset_assign_plan(q, kLdsRowBankSize, mWavesM);
      EXPECT_EQ(l.subIterKBytes, rl.subIterKBytes);
      EXPECT_EQ(l.loadWidthLR, rl.loadWidthLR);
      EXPECT_EQ(l.loadWidthGR, rl.loadWidthGR);
      EXPECT_EQ(l.blockSize, rl.blockSize);
      EXPECT_EQ(l.numRowsPerLDSBanks, rl.numRowsPerLDSBanks);
      EXPECT_EQ(l.miM, rl.miM);
      EXPECT_EQ(l.numMFMACols, rl.numMFMACols);
      EXPECT_EQ(l.partitionOffset, rl.partitionOffset);
      EXPECT_EQ(l.sInterval, rl.sInterval);
      EXPECT_EQ(l.mWavesM, rl.mWavesM);
      EXPECT_EQ(l.wavePartMode, rl.wavePartMode);
      EXPECT_DOUBLE_EQ(l.loadRatioGR, rl.loadRatioGR);
      EXPECT_EQ(l.isFp8, rl.isFp8);
    }
  }
}

// BF16 (bpe 2): row-major DPP pair-swap path; not the FP8 swizzle.
TEST(TileInfoOffsetAssignPlan, Bf16Selectors) {
  ABTileInfoQuery q = make_query(AB_B16(), "A", make_kernel(256, 128, 128));
  GROffsetAssignPlan g = q.grOffsetAssignPlan(kLdsRowBankSize);
  EXPECT_FALSE(g.isFp8);
  EXPECT_EQ(g.bpeBits, 16);            // int(8 * 2)
  EXPECT_EQ(g.partitionMode, 0);       // loadRatioGR == 0.5
  EXPECT_EQ(g.numGRPerSubtile, 2);     // ceil(1 / 0.5)
  EXPECT_EQ(q.lrOffsetAssignPlan(kLdsRowBankSize, 4).wavePartMode, 0);
  EXPECT_FALSE(q.lrOffsetAssignPlan(kLdsRowBankSize, 4).isFp8);
}

// FP4 (bpe 0.5): shares the BF16 swizzle. Regression guard for the
// depthUBytes = int(depthU * bpe) rounding (truncating bpe to 0 before the
// multiply zeroed subIterKBytes and caused a downstream divide-by-zero).
TEST(TileInfoOffsetAssignPlan, Fp4SubIterKBytesAndSelectors) {
  ABTileInfoQuery q = make_query(AB_B4(), "A", make_kernel(256, 128, 256));
  EXPECT_GT(q.subIterKBytes(), 0);
  GROffsetAssignPlan g = q.grOffsetAssignPlan(kLdsRowBankSize);
  EXPECT_GT(g.subIterKBytes, 0);
  EXPECT_FALSE(g.isFp8);               // fp4 is not the fp8 path
  EXPECT_EQ(g.bpeBits, 4);             // int(8 * 0.5)
  EXPECT_FALSE(q.lrOffsetAssignPlan(kLdsRowBankSize, 4).isFp8);
}

// FP8 (bpe 1): the plan selects the distinct block-swap swizzle / wave-rotation
// path for both GR and LR.
TEST(TileInfoOffsetAssignPlan, Fp8SelectorSet) {
  ABTileInfoQuery q = make_query(AB_B8(), "A", make_kernel(256, 128, 128));
  GROffsetAssignPlan g = q.grOffsetAssignPlan(kLdsRowBankSize);
  EXPECT_TRUE(g.isFp8);
  EXPECT_EQ(g.bpeBits, 8);             // int(8 * 1)
  EXPECT_TRUE(q.lrOffsetAssignPlan(kLdsRowBankSize, 4).isFp8);
}
