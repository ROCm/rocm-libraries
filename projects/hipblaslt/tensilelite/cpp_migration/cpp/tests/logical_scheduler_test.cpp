// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Native C++ port of the value/config-layer portions of
// Tensile/Tests/unit/test_logicalSchedulerCpp.py.
//
// The deleted Python file compared the Python LogicalScheduler value/config
// primitives against the compiled nanobind extension. With those primitives
// now C++-only (logical_scheduler.hpp), these tests exercise the header
// directly and pin absolute golden values (the same pins the Python parity
// test carried), so the value layer is locked against an independent
// re-derivation rather than against a Python twin.
//
// Covered here: Pass enum values, fmt_mt, MFMATileRange, ReadGranularity,
// SchedulerConfig (partition normalization + candidate generation + the
// derived state pins), and the placement / before-chain op value types
// (str()/kind + the pass-populated list fields). The writer-free pass pipeline
// (place_LRs .. emit/build) is covered separately in
// logical_scheduler_passes_test.cpp.

#include <gtest/gtest.h>

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "tensile_writer/logical_scheduler.hpp"

using namespace tw::subtile::lsched;

namespace {

// Build a config from a compact spec and run __post_init__ (the C++ ctor does
// not auto-run it; the nanobind binding does).
struct ConfigSpec {
  int M = 0;
  int N = 0;
  int numSubIterK = 0;
  std::pair<int, int> lrA{1, 1};
  std::pair<int, int> lrB{1, 1};
  std::pair<int, int> grA{1, 2};
  std::pair<int, int> grB{1, 2};
  std::optional<std::pair<int, int>> lrSA, lrSB, grSA, grSB;
  PartitionSpec partM = 0;
  PartitionSpec partN = 0;
  int pgr = 2;
};

ReadGranularity rg(std::pair<int, int> p) { return ReadGranularity(p.first, p.second); }

SchedulerConfig makeConfig(const ConfigSpec& s) {
  SchedulerConfig c;
  c.numMFMATilesM = s.M;
  c.numMFMATilesN = s.N;
  c.numSubIterK = s.numSubIterK;
  c.lrA = rg(s.lrA);
  c.lrB = rg(s.lrB);
  c.grA = rg(s.grA);
  c.grB = rg(s.grB);
  if (s.lrSA) c.lrSA = rg(*s.lrSA);
  if (s.lrSB) c.lrSB = rg(*s.lrSB);
  if (s.grSA) c.grSA = rg(*s.grSA);
  if (s.grSB) c.grSB = rg(*s.grSB);
  c.partitionSizeM = s.partM;
  c.partitionSizeN = s.partN;
  c.pgr = s.pgr;
  c.post_init();
  return c;
}

// str()/kind for a before-chain op or a Dep's referenced placement.
std::string before_str(const BeforeOp& o) {
  return std::visit([](const auto& v) { return v.str(); }, o);
}
std::string before_kind(const BeforeOp& o) {
  return std::visit([](const auto& v) { return v.kind; }, o);
}
std::string dep_ref_kind(const Dep& d) {
  return std::visit([](const auto& v) { return v.kind; }, d.ref);
}
std::string dep_ref_tensor(const Dep& d) {
  return std::visit([](const auto& v) { return v.tensor; }, d.ref);
}
int dep_ref_partition(const Dep& d) {
  return std::visit([](const auto& v) { return v.partition; }, d.ref);
}

LRPlacement makeLR(const std::string& tensor = "A", int part = 0) {
  return LRPlacement(tensor, 0, MFMATileRange(0, 1, 0, 1), 0, part);
}
GRPlacement makeGR(const std::string& tensor = "A", int part = 0) {
  return GRPlacement(tensor, 0, MFMATileRange(0, 1, 0, 1), 0, part);
}
MFMAPlacement makeMFMA() {
  return MFMAPlacement(0, MFMATileRange(0, 1, 0, 1), MFMATileRange(0, 1, 0, 1));
}

}  // namespace

// ---------------------------------------------------------------------------
// Pass enum
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, PassEnumValues) {
  EXPECT_EQ(static_cast<int>(Pass::LR), 0);
  EXPECT_EQ(static_cast<int>(Pass::VGPR_TILES), 1);
  EXPECT_EQ(static_cast<int>(Pass::GR), 2);
  EXPECT_EQ(static_cast<int>(Pass::DEPS), 3);
  EXPECT_EQ(static_cast<int>(Pass::REMOVE_GR_DEPS), 4);
  EXPECT_EQ(static_cast<int>(Pass::REMOVE_LR_DEPS), 5);
  EXPECT_EQ(static_cast<int>(Pass::REMOVE_DEPS), 6);
  EXPECT_EQ(static_cast<int>(Pass::GR_INC), 7);
  EXPECT_EQ(static_cast<int>(Pass::GROUP_LR_GR), 8);
  EXPECT_EQ(static_cast<int>(Pass::REMOVE_WAIT_LR_SYNC), 9);
  EXPECT_EQ(static_cast<int>(Pass::EMIT), 10);
  EXPECT_EQ(static_cast<int>(Pass::BUILD), 11);
  EXPECT_EQ(static_cast<int>(Pass::POPULATE), 12);
}

// ---------------------------------------------------------------------------
// fmt_mt
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, FmtMt) {
  EXPECT_EQ(fmt_mt(0), "n");
  EXPECT_EQ(fmt_mt(1), "n+1");
  EXPECT_EQ(fmt_mt(2), "n+2");
  EXPECT_EQ(fmt_mt(3), "n+3");
  EXPECT_EQ(fmt_mt(7), "n+7");
  EXPECT_EQ(fmt_mt(15), "n+15");
}

// ---------------------------------------------------------------------------
// MFMATileRange
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, MFMATileRange) {
  struct Case {
    MFMATileRange r;
    std::vector<int> sk;
    std::vector<int> tiles;
    std::string fmtK;
    std::string fmtTiles;
  };
  std::vector<Case> cases = {
      {MFMATileRange(0, 1, 0, 4), {0}, {0, 1, 2, 3}, "[0]", "[0-3]"},
      {MFMATileRange(0, 2, 0, 8), {0, 1}, {0, 1, 2, 3, 4, 5, 6, 7}, "[0,1]", "[0-7]"},
      {MFMATileRange(1, 2, 4, 8), {1}, {4, 5, 6, 7}, "[1]", "[4-7]"},
      {MFMATileRange(2, 4, 0, 16), {2, 3}, {}, "[2,3]", "[0-15]"},
  };
  for (auto& c : cases) {
    EXPECT_EQ(c.r.subIterK_list(), c.sk);
    if (!c.tiles.empty()) EXPECT_EQ(c.r.tileId_list(), c.tiles);
    EXPECT_EQ(c.r.fmt_k(), c.fmtK);
    EXPECT_EQ(c.r.fmt_tiles(), c.fmtTiles);
  }
}

// ---------------------------------------------------------------------------
// ReadGranularity.tile_range — pinned against a simple independent re-derivation
// (all inputs are non-negative, so floor-div is integer division).
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, ReadGranularityTileRange) {
  for (auto [mn, k] : std::vector<std::pair<int, int>>{
           {1, 1}, {1, 2}, {2, 2}, {2, 1}, {4, 2}}) {
    for (auto [kk, t0, t1] : std::vector<std::tuple<int, int, int>>{
             {0, 0, 8}, {1, 0, 8}, {3, 1, 7}, {2, 3, 5}, {0, 5, 5}}) {
      MFMATileRange got = ReadGranularity(mn, k).tile_range(kk, t0, t1);
      int ks = (kk / k) * k;
      int ts = (t0 / mn) * mn;
      int te = ((t1 + mn - 1) / mn) * mn;
      SCOPED_TRACE("mn=" + std::to_string(mn) + " k=" + std::to_string(k) +
                   " kk=" + std::to_string(kk));
      EXPECT_EQ(got.subIterK_start, ks);
      EXPECT_EQ(got.subIterK_end, ks + k);
      EXPECT_EQ(got.tileId_start, ts);
      EXPECT_EQ(got.tileId_end, te);
    }
  }
}

// ---------------------------------------------------------------------------
// SchedulerConfig — derived state + absolute pins (the CONFIGS table).
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, SchedulerConfigDerivedState) {
  struct Expect {
    std::optional<std::vector<int>> partM, partN;
    std::optional<int> numPartitions;
    std::optional<bool> hasScale;
  };
  struct Named {
    std::string name;
    ConfigSpec spec;
    Expect exp;
  };
  std::vector<Named> configs = {
      {"no_scale_k1", {2, 2, 2}, {{{2}}, {{2}}, 1, false}},
      {"fp4_2x2",
       []() {
         ConfigSpec s{8, 8, 2};
         s.lrSA = {2, 2};
         s.lrSB = {2, 2};
         s.grSA = {2, 2};
         s.grSB = {2, 2};
         s.partM = 4;
         s.partN = 4;
         return s;
       }(),
       {{{4, 4}}, {{4, 4}}, 4, true}},
      {"bf16_10x1",
       []() {
         ConfigSpec s{10, 10, 2};
         s.partM = 1;
         s.partN = 10;
         return s;
       }(),
       {std::nullopt, std::nullopt, 10, false}},
      {"bf16_256x384_n6",
       []() { ConfigSpec s{8, 12, 2}; s.partN = 6; return s; }(),
       {std::nullopt, {{6, 6}}, std::nullopt, std::nullopt}},
      {"bf16_256x384_n4",
       []() { ConfigSpec s{8, 12, 2}; s.partN = 4; return s; }(),
       {std::nullopt, {{4, 4, 4}}, std::nullopt, std::nullopt}},
      {"bf16_256x384_n5",
       []() { ConfigSpec s{8, 12, 2}; s.partN = 5; return s; }(),
       {std::nullopt, {{5, 2, 5}}, std::nullopt, std::nullopt}},
      {"bf16_256x352_n4",
       []() { ConfigSpec s{8, 11, 2}; s.partN = 4; return s; }(),
       {std::nullopt, {{4, 3, 4}}, std::nullopt, std::nullopt}},
      {"bf16_256x352_n3",
       []() { ConfigSpec s{8, 11, 2}; s.partN = 3; return s; }(),
       {std::nullopt, {{3, 2, 3, 3}}, std::nullopt, std::nullopt}},
      {"bf16_256x368_n4",
       []() { ConfigSpec s{4, 23, 2}; s.partN = 4; return s; }(),
       {std::nullopt, {{4, 4, 3, 4, 4, 4}}, std::nullopt, std::nullopt}},
      {"bf16_256x368_n8",
       []() { ConfigSpec s{4, 23, 2}; s.partN = 8; return s; }(),
       {std::nullopt, {{8, 7, 8}}, std::nullopt, std::nullopt}},
      {"explicit_list_N",
       []() {
         ConfigSpec s{2, 12, 2};
         s.partN = std::vector<int>{5, 2, 5};
         return s;
       }(),
       {std::nullopt, {{5, 2, 5}}, 3, std::nullopt}},
      {"pgr0_single",
       []() { ConfigSpec s{2, 2, 1}; s.pgr = 0; return s; }(),
       {std::nullopt, std::nullopt, 1, std::nullopt}},
      {"pgr1_single",
       []() { ConfigSpec s{2, 2, 1}; s.pgr = 1; return s; }(),
       {std::nullopt, std::nullopt, 1, std::nullopt}},
  };
  for (auto& nc : configs) {
    SCOPED_TRACE(nc.name);
    SchedulerConfig c = makeConfig(nc.spec);
    // prefix is consistent with the partition sizes.
    EXPECT_EQ(c._prefixM, SchedulerConfig::build_prefix(c.partitionSizesM()));
    EXPECT_EQ(c._prefixN, SchedulerConfig::build_prefix(c.partitionSizesN()));
    if (nc.exp.partM) EXPECT_EQ(c.partitionSizesM(), *nc.exp.partM);
    if (nc.exp.partN) EXPECT_EQ(c.partitionSizesN(), *nc.exp.partN);
    if (nc.exp.numPartitions) EXPECT_EQ(c.numPartitions(), *nc.exp.numPartitions);
    if (nc.exp.hasScale) EXPECT_EQ(c.hasScale(), *nc.exp.hasScale);
    // plr / offsetPartition follow pgr.
    EXPECT_EQ(c.plr, c.pgr == 0 ? 0 : 1);
    EXPECT_EQ(c.offsetPartition, c.pgr >= 2 ? 1 : 0);
  }
}

TEST(LogicalSchedulerValue, SchedulerConfigErrors) {
  // pgr=0 with >1 partition.
  ConfigSpec bad{4, 4, 1};
  bad.partM = 2;
  bad.partN = 2;
  bad.pgr = 0;
  EXPECT_THROW(makeConfig(bad), std::invalid_argument);

  // Explicit list that does not sum to total.
  ConfigSpec badSum{2, 12, 2};
  badSum.partN = std::vector<int>{5, 2, 4};
  EXPECT_THROW(makeConfig(badSum), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// _normalize_partition_sizes
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, NormalizePartitionSizes) {
  struct Case {
    PartitionSpec spec;
    int total;
    int mn;
    std::vector<int> expected;
  };
  std::vector<Case> cases = {
      {0, 8, 1, {8}},
      {4, 8, 1, {4, 4}},
      {3, 12, 1, {3, 3, 3, 3}},
      {5, 12, 1, {5, 2, 5}},
      {4, 11, 1, {4, 3, 4}},
      {8, 23, 1, {8, 7, 8}},
      {4, 8, 2, {4, 4}},
      {3, 9, 1, {3, 3, 3}},
      {6, 6, 1, {6}},
      {0, 10, 1, {10}},
      {std::vector<int>{5, 2, 5}, 12, 1, {5, 2, 5}},
      {std::vector<int>{4, 4, 4}, 12, 1, {4, 4, 4}},
      {std::vector<int>{6, 6}, 12, 2, {6, 6}},
  };
  for (auto& c : cases) {
    std::vector<int> got =
        SchedulerConfig::normalize_partition_sizes(c.spec, c.total, "X", c.mn);
    EXPECT_EQ(got, c.expected);
  }
}

// ---------------------------------------------------------------------------
// get_partition_candidates
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, GetPartitionCandidates) {
  using P = std::pair<int, int>;
  struct Case {
    int M, N;
    std::vector<P> expected;
  };
  std::vector<Case> cases = {
      {4, 8, {{4, 8}, {4, 4}, {4, 3}, {4, 2}, {4, 1}}},
      {8, 4, {{8, 4}, {4, 4}, {3, 4}, {2, 4}, {1, 4}}},
      {8, 8, {{8, 8}, {8, 4}, {8, 3}, {8, 2}, {8, 1}}},
      {1, 10, {{1, 10}, {1, 5}, {1, 4}, {1, 3}, {1, 2}, {1, 1}}},
      {10, 1, {{10, 1}, {5, 1}, {4, 1}, {3, 1}, {2, 1}, {1, 1}}},
      {16, 4, {{16, 4}, {8, 4}, {7, 4}, {6, 4}, {5, 4}, {4, 4}, {3, 4}, {2, 4}, {1, 4}}},
      {3, 7, {{3, 7}, {3, 4}, {3, 3}, {3, 2}, {3, 1}}},
  };
  for (auto& c : cases) {
    SCOPED_TRACE("M=" + std::to_string(c.M) + " N=" + std::to_string(c.N));
    EXPECT_EQ(SchedulerConfig::get_partition_candidates(c.M, c.N), c.expected);
  }
}

// ---------------------------------------------------------------------------
// Placement / op value type string formatting + kind
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, PlacementStr) {
  EXPECT_EQ(MFMAPlacement(1, MFMATileRange(0, 2, 0, 4), MFMATileRange(0, 2, 4, 8)).str(),
            "MFMAs (MT n, subIterK 1  ) A : [0-3] , B : [4-7]");
  MFMATileRange tr(0, 2, 0, 8);
  EXPECT_EQ(LRPlacement("A", 0, tr, 3, 1).str(), "LR A  (MT n, subIterK [0,1]) [0-7]");
  EXPECT_EQ(LRPlacement("SA", 2, tr, 3, 1).str(), "LR SA (MT n+2, subIterK [0,1]) [0-7]");
  EXPECT_EQ(GRPlacement("B", 1, tr, 3, 1).str(), "GR B (MT n+1, subIterK [0,1]) ids [0-7]");
  EXPECT_EQ(GRPlacement("SB", 0, tr, 3, 1).str(), "GR SB (MT n, subIterK [0,1]) ids [0-7]");
}

TEST(LogicalSchedulerValue, OpStr) {
  EXPECT_EQ(WaitGRCounts(1, 0, 2, 0).str(), "A=1,SA=2");
  EXPECT_EQ(WaitGRCounts().str(), "0");
  EXPECT_EQ(WaitGRCounts(0, 3, 0, 4).str(), "B=3,SB=4");

  EXPECT_EQ(WaitGROp(WaitGRCounts(1, 2, 0, 0), true, true).str(), "wait_gr(A=1,B=2)");
  EXPECT_EQ(WaitGROp(std::nullopt, false, true).str(), "wait_gr");
  EXPECT_EQ(WaitGROp().str(), "wait_gr");

  EXPECT_EQ(WaitLROp(true).str(), "wait_lr_sync");
  EXPECT_EQ(WaitLROp(false).str(), "wait_lr");
  EXPECT_EQ(SyncOp().str(), "sync");
  EXPECT_EQ(MaskKOp(3).str(), "mask_k(k=3)");
  EXPECT_EQ(LRIncOp("A").str(), "lr_inc(A)");
  EXPECT_EQ(GRIncOp("SB").str(), "gr_inc(SB)");

  SkipOp skip("LoopCounter", 2, "NLL", false, "");
  EXPECT_EQ(skip.str(), "skip(LoopCounter:2:NLL)");
  EXPECT_EQ(skip.tensor(), "LoopCounter:2:NLL");

  EXPECT_EQ(InlineModuleOp("inline").str(), "inline(inline)");
  EXPECT_EQ(InlineModuleOp("preloop").str(), "inline(preloop)");
  EXPECT_EQ(InlineModuleOp().str(), "inline(inline)");
  EXPECT_EQ(InlineModuleOp().label, "inline");
}

TEST(LogicalSchedulerValue, OpKind) {
  EXPECT_EQ(makeMFMA().kind, "mfma");
  EXPECT_EQ(makeLR().kind, "lr");
  EXPECT_EQ(makeGR().kind, "gr");
  EXPECT_EQ(WaitGROp().kind, "wait_gr");
  EXPECT_EQ(WaitLROp().kind, "wait_lr");
  EXPECT_EQ(SyncOp().kind, "sync");
  EXPECT_EQ(MaskKOp().kind, "mask_k");
  EXPECT_EQ(LRIncOp("A").kind, "lr_inc");
  EXPECT_EQ(GRIncOp("A").kind, "gr_inc");
  EXPECT_EQ(SkipOp().kind, "skip");
  EXPECT_EQ(InlineModuleOp().kind, "inline");
}

// ---------------------------------------------------------------------------
// Pass-populated placement fields: default empty + round-trip
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, PlacementPassFieldsDefaultEmpty) {
  MFMAPlacement m = makeMFMA();
  LRPlacement lr = makeLR();
  GRPlacement gr = makeGR();
  EXPECT_TRUE(m.deps.empty() && m.preOps.empty() && m.postOps.empty());
  EXPECT_TRUE(lr.deps.empty() && lr.preOps.empty() && lr.postOps.empty());
  EXPECT_TRUE(gr.deps.empty() && gr.preOps.empty() && gr.postOps.empty());
  EXPECT_TRUE(m.vgpr_tile_maps.empty());
  EXPECT_TRUE(lr.vgpr_tile_map.empty());
  EXPECT_TRUE(MaskKOp(3).vgpr_tile_map.empty());
}

TEST(LogicalSchedulerValue, PlacementPrePostOpsRoundtrip) {
  LRPlacement lr = makeLR();
  lr.preOps = {WaitGROp(WaitGRCounts(1, 0, 0, 0), false, true), SyncOp()};
  lr.postOps = {LRIncOp("A")};
  std::vector<std::string> preStr, preKind, postStr;
  for (auto& o : lr.preOps) {
    preStr.push_back(before_str(o));
    preKind.push_back(before_kind(o));
  }
  for (auto& o : lr.postOps) postStr.push_back(before_str(o));
  EXPECT_EQ(preStr, (std::vector<std::string>{"wait_gr(A=1)", "sync"}));
  EXPECT_EQ(preKind, (std::vector<std::string>{"wait_gr", "sync"}));
  EXPECT_EQ(postStr, (std::vector<std::string>{"lr_inc(A)"}));
}

TEST(LogicalSchedulerValue, LRVgprTileMapRoundtrip) {
  LRPlacement lr = makeLR();
  lr.vgpr_tile_map = {{{0, 4}, {1, 5}}, {{2, 6}}};
  ASSERT_EQ(lr.vgpr_tile_map.size(), 2u);
  EXPECT_EQ(lr.vgpr_tile_map[0], (std::map<int, int>{{0, 4}, {1, 5}}));
  EXPECT_EQ(lr.vgpr_tile_map[1], (std::map<int, int>{{2, 6}}));
}

TEST(LogicalSchedulerValue, MFMAVgprTileMapsRoundtrip) {
  MFMAPlacement m = makeMFMA();
  m.vgpr_tile_maps = {{"A", {{{0, 1}}}}, {"B", {{{0, 2}}, {{1, 3}}}}};
  EXPECT_EQ(m.vgpr_tile_maps["A"].size(), 1u);
  EXPECT_EQ(m.vgpr_tile_maps["B"].size(), 2u);
  EXPECT_EQ(m.vgpr_tile_maps["B"][1], (std::map<int, int>{{1, 3}}));
}

// ---------------------------------------------------------------------------
// Dep
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, Dep) {
  for (int mtOffset : {0, -1, -2}) {
    Dep lrDep(makeLR("B", 1), mtOffset);
    EXPECT_EQ(lrDep.mt_offset, mtOffset);
    EXPECT_EQ(dep_ref_kind(lrDep), "lr");
    EXPECT_EQ(dep_ref_tensor(lrDep), "B");
    EXPECT_EQ(dep_ref_partition(lrDep), 1);

    Dep grDep(makeGR("B", 1), mtOffset);
    EXPECT_EQ(grDep.mt_offset, mtOffset);
    EXPECT_EQ(dep_ref_kind(grDep), "gr");
  }
  EXPECT_EQ(Dep(makeLR()).mt_offset, 0);  // default mt_offset
}

// ---------------------------------------------------------------------------
// SubIterKSlot
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, SubIterKSlot) {
  SubIterKSlot slot(2);
  EXPECT_EQ(slot.subIterK, 2);
  EXPECT_FALSE(slot.mfma.has_value());
  EXPECT_TRUE(slot.lrs.empty());
  EXPECT_TRUE(slot.grs.empty());

  slot.mfma = makeMFMA();
  slot.lrs = {makeLR("A"), makeLR("B")};
  slot.grs = {makeGR("A")};
  EXPECT_EQ(slot.mfma->kind, "mfma");
  EXPECT_EQ(slot.lrs[0].tensor, "A");
  EXPECT_EQ(slot.lrs[1].tensor, "B");
  EXPECT_EQ(slot.grs[0].tensor, "A");
}

// ---------------------------------------------------------------------------
// EmittedModule
// ---------------------------------------------------------------------------
TEST(LogicalSchedulerValue, EmittedModuleOpType) {
  std::vector<std::pair<Emittable, std::string>> sources = {
      {makeMFMA(), "mfma"},   {makeLR(), "lr"},        {makeGR(), "gr"},
      {WaitGROp(), "wait_gr"}, {SyncOp(), "sync"},     {LRIncOp("A"), "lr_inc"},
      {InlineModuleOp("x"), "inline"},
  };
  for (auto& [src, kind] : sources) {
    EmittedModule em(7, 3, src);
    EXPECT_EQ(em.opType(), kind);
    EXPECT_EQ(em.moduleId, 7);
    ASSERT_TRUE(em.before.has_value());
    EXPECT_EQ(*em.before, 3);
    ASSERT_TRUE(em.source.has_value());
    EXPECT_EQ(emittable_kind(*em.source), kind);
  }
}

TEST(LogicalSchedulerValue, EmittedModuleEmptySource) {
  EmittedModule em;
  EXPECT_EQ(em.opType(), "");
  EXPECT_EQ(em.moduleId, -1);
  EXPECT_FALSE(em.before.has_value());
  EXPECT_FALSE(em.source.has_value());
}
