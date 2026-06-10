// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Native C++ port of Tensile/Tests/unit/test_subtileGeometryCpp.py.
//
// The Python file existed only to confirm the Python SubtileGeometry facade
// forwarded faithfully to this C++ layer and to pin absolute gfx950 values
// through that facade. With the geometry now C++-only, these tests exercise
// subtile_geometry.hpp directly: layout constants, A/B GR+LR queries, MX scale
// and C/D geometry, plus the absolute-value gfx950 pins.

#include <gtest/gtest.h>

#include "subtile_test_fixtures.hpp"

using namespace tw::subtile;
using namespace tw_test;

namespace {

// Macro tile / depthU sweep that stays divisible by every layout's MMA tile.
const std::vector<std::pair<long, long>> kMtDu = {
    {256, 128}, {128, 64}, {512, 256}, {64, 32}};

std::vector<ABPair> all_ab_pairs() {
  return {AB_B16(),     AB_B8(),          AB_B4(),
          AB_B4_2x2(),  AB_B16_2x2(),     AB_B16_TLU1(),
          AB_B16_TLU1_16x1()};
}

}  // namespace

// ---------------------------------------------------------------------------
// MMALayout / MMAScaleLayout constants — absolute pins + derived math.
// ---------------------------------------------------------------------------
TEST(SubtileGeometryLayout, MMALayoutKnownValues) {
  MMALayout layout = MFMA_16x16_1B_4K_4V();
  EXPECT_EQ(layout.instM, 16);
  EXPECT_EQ(layout.blocks, 1);
  EXPECT_EQ(layout.vgprs, 4);
  EXPECT_EQ(layout.waveSize, 64);
  EXPECT_EQ(layout.contiguousLanes, 16);
  EXPECT_EQ(layout.kGroups, 4);
  EXPECT_EQ(layout.elementsPerLaneNonK, 4);
  EXPECT_EQ(layout.inputBytesPerLane(), 16);
  EXPECT_EQ(layout.tileSizeBytes(32, 2.0), 1024);
  EXPECT_DOUBLE_EQ(layout.regsPerTile(32, 2.0), 4.0);
}

TEST(SubtileGeometryLayout, MMALayoutDerivedConsistency) {
  for (const MMALayout& layout :
       {MFMA_16x16_1B_4K_4V(), MFMA_16x16_1B_4K_8V(), MFMA_16x16_1B_4N_4V()}) {
    EXPECT_EQ(layout.contiguousLanes, layout.instM);
    EXPECT_EQ(layout.kGroups,
              floordiv(layout.waveSize,
                       static_cast<long>(layout.contiguousLanes) * layout.blocks));
    EXPECT_EQ(layout.elementsPerLaneNonK,
              floordiv(layout.instM, layout.kGroups));
    EXPECT_EQ(layout.inputBytesPerLane(), static_cast<long>(layout.vgprs) * 4);
    for (auto [instK, eb] :
         std::vector<std::pair<int, double>>{{32, 2.0}, {128, 0.5}, {128, 1.0}, {16, 16.0}}) {
      EXPECT_EQ(layout.tileSizeBytes(instK, eb),
                static_cast<long>(static_cast<double>(layout.instM) * instK * eb));
      EXPECT_DOUBLE_EQ(layout.regsPerTile(instK, eb),
                       static_cast<double>(layout.tileSizeBytes(instK, eb)) /
                           layout.waveSize / 4.0);
    }
  }
}

TEST(SubtileGeometryLayout, ScaleLayoutKnownValues) {
  MMAScaleLayout s = MFMA_SCALE_16x16_1B_MX32_8V();
  EXPECT_EQ(s.instM, 16);
  EXPECT_EQ(s.blocks, 1);
  EXPECT_EQ(s.mxBlock, 32);
  EXPECT_EQ(s.waveSize, 64);
  EXPECT_DOUBLE_EQ(s.vgprs, 0.25);
  EXPECT_EQ(s.contiguousLanes, 16);
}

// ---------------------------------------------------------------------------
// A/B GR + LR geometry: query methods run and stay self-consistent with the
// documented grid/byte formulas over the MT/DU sweep.
// ---------------------------------------------------------------------------
TEST(SubtileGeometryAB, GRQueries) {
  for (const ABPair& pair : all_ab_pairs()) {
    const ABGRGeometry& gr = pair.gr;
    for (auto [mt, du] : kMtDu) {
      auto glbl = gr.globalMMATileGrid(mt, du);
      EXPECT_EQ(glbl.first, floordiv(mt, gr.mmaTileShape.first));
      EXPECT_EQ(glbl.second, floordiv(du, gr.mmaTileShape.second));

      auto local = gr.localMMATileGrid(mt, du, 4);
      EXPECT_EQ(local.first, floordiv(glbl.first, 4));
      EXPECT_EQ(local.second, glbl.second);

      auto gsg = gr.globalSubtileGrid(mt, du);
      EXPECT_DOUBLE_EQ(gsg.first,
                       static_cast<double>(glbl.first) / gr.subtileShape.first);
      EXPECT_DOUBLE_EQ(gsg.second,
                       static_cast<double>(glbl.second) / gr.subtileShape.second);

      EXPECT_DOUBLE_EQ(gr.subtileSizeBytes(),
                       static_cast<double>(gr.subtileShape.first) *
                           gr.subtileShape.second * gr.mmaTileSize);
      EXPECT_GT(gr.bytesPerLoad(4), 0);
      EXPECT_GT(gr.loadsPerStrip(4), 0.0);
      auto gran = gr.localGRGranularity(4);
      EXPECT_GE(gran.first, 1);
      EXPECT_EQ(gran.second, gr.subtileShape.second);
    }
  }
}

TEST(SubtileGeometryAB, LRQueries) {
  for (const ABPair& pair : all_ab_pairs()) {
    const ABLRGeometry& lr = pair.lr;
    for (auto [mt, du] : kMtDu) {
      auto glbl = lr.globalMMATileGrid(mt, du);
      EXPECT_EQ(glbl.first, floordiv(mt, lr.mmaTileShape.first));
      EXPECT_EQ(glbl.second, floordiv(du, lr.mmaTileShape.second));
      auto gsg = lr.globalSubtileGrid(mt, du);
      EXPECT_DOUBLE_EQ(gsg.first,
                       static_cast<double>(glbl.first) / lr.subtileShape.first);
      EXPECT_DOUBLE_EQ(lr.subtileSizeBytes(),
                       static_cast<double>(lr.subtileShape.first) *
                           lr.subtileShape.second * lr.mmaTileSize);
    }
  }
}

// for_kernel derives (subtileCount, subtileStride) and subtileForMmaTile stays
// consistent across the materialized MMA tile grid.
TEST(SubtileGeometryAB, ForKernelAndSubtileForMmaTile) {
  Kernel k = make_kernel(256, 128, 128, {4, 1});
  for (const ABPair& pair : all_ab_pairs()) {
    for (const std::string tc : {"A", "B"}) {
      bool isA = (tc == "A");
      long mt = isA ? k.macroTileA : k.macroTileB;
      int wg_m = static_cast<int>(isA ? k.miWaveGroup[0] : k.miWaveGroup[1]);
      long mt_mma = floordiv(mt, pair.gr.mmaTileShape.first);
      ABGRGeometry fk = pair.gr.forKernel(wg_m, mt_mma);
      ASSERT_TRUE(fk.subtileCount.has_value());
      ASSERT_TRUE(fk.subtileStride.has_value());

      auto grid = fk.globalMMATileGrid(mt, isA ? k.depthU : k.depthU);
      for (long r = 0; r < grid.first; ++r) {
        for (long c = 0; c < grid.second; ++c) {
          auto res = fk.subtileForMmaTile(r, c);
          EXPECT_EQ(res.block_shape.first, fk.subtileShape.first);
          EXPECT_EQ(res.block_shape.second, fk.subtileShape.second);
          EXPECT_FALSE(res.mma_tiles.empty());
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// MX scale geometry — grid/byte formulas + for_kernel shape derivation.
// ---------------------------------------------------------------------------
TEST(SubtileGeometryMX, GRAndLRQueries) {
  for (const MXPair& pair : {MXS_B4(), MXS_B8()}) {
    for (auto [mt, du] : kMtDu) {
      auto grGlbl = pair.gr.globalMMATileGrid(mt, du);
      EXPECT_EQ(grGlbl.first, floordiv(mt, pair.gr.mmaTileShape.first));
      EXPECT_EQ(grGlbl.second, floordiv(du, pair.gr.instK));

      auto lrGlbl = pair.lr.globalMMATileGrid(mt, du);
      auto lrGsg = pair.lr.globalSubtileGrid(mt, du);
      EXPECT_DOUBLE_EQ(lrGsg.first, static_cast<double>(lrGlbl.first) /
                                        pair.lr.subtileShape.first);
      EXPECT_DOUBLE_EQ(lrGsg.second, static_cast<double>(lrGlbl.second) /
                                         pair.lr.subtileShape.second);
      EXPECT_DOUBLE_EQ(pair.lr.subtileSizeBytes(),
                       static_cast<double>(pair.lr.subtileShape.first) *
                           pair.lr.subtileShape.second * pair.lr.mmaTileSize);
    }
  }
}

TEST(SubtileGeometryMX, ForKernelShape) {
  MXPair pair = MXS_B4();
  // tc=A: MacroTileA=256, _DepthUA=128 -> (256//16, 128//128) = (16, 1).
  auto shapeA = pair.gr.forKernelShape(256, 128);
  EXPECT_EQ(shapeA.first, 16);
  EXPECT_EQ(shapeA.second, 1);
  // tc=B: MacroTileB=128, _DepthUB=128 -> (8, 1).
  auto shapeB = pair.gr.forKernelShape(128, 128);
  EXPECT_EQ(shapeB.first, 8);
  EXPECT_EQ(shapeB.second, 1);

  MXScaleGRGeometry fk = pair.gr.forKernel(256, 128);
  ASSERT_TRUE(fk.subtileShape.has_value());
  EXPECT_EQ(fk.subtileShape->first, 16);
  EXPECT_EQ(fk.subtileShape->second, 1);
}

TEST(SubtileGeometryMX, KnownValues) {
  MXPair pair = MXS_B4();
  // instKScale = 128 // 32 = 4; mmaTileSize = 16 * 4 * 1 = 64.
  EXPECT_EQ(pair.gr.mmaTileShape.first, 16);
  EXPECT_EQ(pair.gr.mmaTileShape.second, 4);
  EXPECT_EQ(pair.gr.mmaTileSize, 64);
  auto glbl = pair.gr.globalMMATileGrid(256, 128);
  EXPECT_EQ(glbl.first, 16);
  EXPECT_EQ(glbl.second, 1);
  // LR subtileShape (2,2): subtileSizeBytes = 2*2*64 = 256.
  EXPECT_DOUBLE_EQ(pair.lr.subtileSizeBytes(), 256.0);
}

// ---------------------------------------------------------------------------
// C/D output geometry.
// ---------------------------------------------------------------------------
TEST(SubtileGeometryCD, Queries) {
  CDTileGeometry cd = CD_F32();
  std::pair<long, long> wg = {2, 2};
  std::pair<double, double> ss = {1.0, 1.0};
  for (auto [mt0, mt1] :
       std::vector<std::pair<long, long>>{{256, 128}, {128, 128}, {64, 256}}) {
    auto glbl = cd.globalMMATileGrid(mt0, mt1);
    EXPECT_EQ(glbl.first, floordiv(mt0, cd.mmaLayout.instM));
    EXPECT_EQ(glbl.second, floordiv(mt1, cd.mmaLayout.instM));
    auto local = cd.localMMATileGrid(mt0, mt1, wg);
    EXPECT_EQ(local.first, floordiv(glbl.first, wg.first));
    EXPECT_EQ(local.second, floordiv(glbl.second, wg.second));
    auto gsg = cd.globalSubtileGrid(mt0, mt1, ss);
    EXPECT_DOUBLE_EQ(gsg.first, glbl.first / ss.first);
    EXPECT_DOUBLE_EQ(gsg.second, glbl.second / ss.second);
    auto lsg = cd.localSubtileGrid(mt0, mt1, wg, ss);
    EXPECT_DOUBLE_EQ(lsg.first, local.first / ss.first);
    EXPECT_DOUBLE_EQ(lsg.second, local.second / ss.second);
  }
}

TEST(SubtileGeometryCD, KnownValues) {
  CDTileGeometry cd = CD_F32();
  EXPECT_EQ(cd.mmaTileShape.first, 16);
  EXPECT_EQ(cd.mmaTileShape.second, 16);
  EXPECT_EQ(cd.mmaTileSize, 1024);  // 16 * 16 * 4
  auto glbl = cd.globalMMATileGrid(256, 128);
  EXPECT_EQ(glbl.first, 16);
  EXPECT_EQ(glbl.second, 8);
}

// ---------------------------------------------------------------------------
// Absolute-value pins for the AB_B16 gfx950 case (lock the C++ formulas).
// ---------------------------------------------------------------------------
TEST(SubtileGeometryAB, AbB16KnownValues) {
  ABGRGeometry gr = AB_B16().gr;
  EXPECT_EQ(gr.mmaTileShape.first, 16);
  EXPECT_EQ(gr.mmaTileShape.second, 32);
  EXPECT_EQ(gr.mmaTileSize, 1024);

  auto glbl = gr.globalMMATileGrid(256, 128);
  EXPECT_EQ(glbl.first, 16);
  EXPECT_EQ(glbl.second, 4);

  auto gsg = gr.globalSubtileGrid(256, 128);
  EXPECT_DOUBLE_EQ(gsg.first, 16.0);
  EXPECT_DOUBLE_EQ(gsg.second, 2.0);

  EXPECT_EQ(gr.bytesPerLoad(4), 4096);

  // for_kernel (MIWaveGroup[0]=4, MacroTileA=256): mt_mma=16 -> (count=4, stride=4).
  ABGRGeometry fk = gr.forKernel(4, 256 / gr.mmaTileShape.first);
  ASSERT_TRUE(fk.subtileCount.has_value());
  ASSERT_TRUE(fk.subtileStride.has_value());
  EXPECT_EQ(*fk.subtileCount, 4);
  EXPECT_EQ(*fk.subtileStride, 4);

  auto res = fk.subtileForMmaTile(5, 3);
  EXPECT_EQ(res.subtile_id.first, 1);
  EXPECT_EQ(res.subtile_id.second, 1);
  EXPECT_EQ(res.block_shape.first, 1);
  EXPECT_EQ(res.block_shape.second, 2);
  std::vector<std::pair<long, long>> expected = {
      {1, 2}, {1, 3}, {5, 2}, {5, 3}, {9, 2}, {9, 3}, {13, 2}, {13, 3}};
  EXPECT_EQ(res.mma_tiles, expected);
}
