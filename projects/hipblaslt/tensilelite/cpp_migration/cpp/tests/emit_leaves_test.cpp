// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Native C++ port of the writer-free portions of
// Tensile/Tests/unit/test_subtileEmitLeavesCpp.py.
//
// Ported here (no rocisa / writer dependency):
//   * the MFMA F8F6F4 instType *selection* mapping (emit_leaves.hpp), and
//   * the single-buffer-load / single-ds-read instruction-shape *plans*
//     (ABTileInfoQuery), locked against the documented reference math.
//
// NOT ported (kept in Python as rocisa integration): the
// emitMfmaInstruction asm-rendering test, which builds real rocisa Modules and
// thus exercises KernelWriter/rocisa behavior rather than the C++ value layer.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "subtile_test_fixtures.hpp"

using namespace tw::subtile;
using namespace tw::subtile::emit;
using namespace tw_test;

namespace {

// (formatA, formatB, sourceSwap) -> expected InstType member name. Mirrors the
// golden table _INST_TABLE x MFMA_CASES from the Python parity test. Formats:
// "f8" (F8), "bf8" (B8), "f4" (F4).
struct MfmaCase {
  std::string fmtA;
  std::string fmtB;
  bool swap;
};

const std::vector<MfmaCase> kMfmaCases = {
    {"f8", "f8", false},  {"f8", "f8", true},  {"bf8", "bf8", false},
    {"f8", "bf8", false}, {"f8", "bf8", true}, {"bf8", "f8", false},
    {"f4", "f4", false},  {"f8", "f4", false}, {"f8", "f4", true},
    {"f4", "f8", false},  {"bf8", "f4", false}, {"f4", "bf8", true},
};

std::string expected_inst_type(std::string a, std::string b, bool swap) {
  if (swap) std::swap(a, b);
  if (a == "f8" && b == "f8") return "INST_F8";
  if (a == "bf8" && b == "bf8") return "INST_BF8";
  if (a == "f4" && b == "f4") return "INST_F4";
  if (a == "f8" && b == "bf8") return "INST_F8_BF8";
  if (a == "bf8" && b == "f8") return "INST_BF8_F8";
  if (a == "f8" && b == "f4") return "INST_F8_F4";
  if (a == "f4" && b == "f8") return "INST_F4_F8";
  if (a == "bf8" && b == "f4") return "INST_B8_F4";
  if (a == "f4" && b == "bf8") return "INST_F4_B8";
  return "<unsupported>";
}

// Translate a format tag into the per-operand boolean predicates the C++ entry
// point consumes.
void predicates(const std::string& fmt, bool& isF8, bool& isBF8, bool& isF4) {
  isF8 = (fmt == "f8");
  isBF8 = (fmt == "bf8");
  isF4 = (fmt == "f4");
}

// Plan geometries with a DepthU chosen so the 128x128 / MIWaveGroup=[2,2]
// kernel builds a valid (coverage-satisfying) TileInfo, matching the Python
// PLAN_GEOMS table.
struct PlanGeom {
  std::string name;
  ABPair pair;
  long depthU;
};

std::vector<PlanGeom> plan_geoms() {
  return {
      {"AB_B16", AB_B16(), 64},
      {"AB_B8", AB_B8(), 128},
      {"AB_B4", AB_B4(), 256},
      {"AB_B16_2x2", AB_B16_2x2(), 64},
      {"AB_B4_2x2", AB_B4_2x2(), 256},
  };
}

}  // namespace

// ---------------------------------------------------------------------------
// MFMA instType selection (now C++-only).
// ---------------------------------------------------------------------------
TEST(EmitLeavesMfma, InstTypeMapping) {
  for (const MfmaCase& c : kMfmaCases) {
    bool aF8, aBF8, aF4, bF8, bBF8, bF4;
    predicates(c.fmtA, aF8, aBF8, aF4);
    predicates(c.fmtB, bF8, bBF8, bF4);
    SCOPED_TRACE(c.fmtA + "/" + c.fmtB + (c.swap ? "/swap" : ""));
    EXPECT_EQ(mfma_f8f6f4_inst_type(aF8, aBF8, aF4, bF8, bBF8, bF4, c.swap),
              expected_inst_type(c.fmtA, c.fmtB, c.swap));
  }
}

TEST(EmitLeavesMfma, UnsupportedRaises) {
  EXPECT_THROW(
      mfma_f8f6f4_inst_type(false, false, false, false, false, false, false),
      std::runtime_error);
}

// ---------------------------------------------------------------------------
// emitSingleBufferLoad / emitSingleDsRead — instruction-shape plan values.
// ---------------------------------------------------------------------------
TEST(EmitLeavesPlans, SingleBufferLoadPlanValues) {
  for (const PlanGeom& pg : plan_geoms()) {
    Kernel k = make_kernel(128, 128, pg.depthU, {2, 2});
    for (const std::string tc : {"A", "B"}) {
      ABTileInfoQuery q = make_query(pg.pair, tc, k);
      SCOPED_TRACE(pg.name + "/" + tc);
      for (long s0 = 0; s0 < q.localSubtileGrid.first; ++s0) {
        for (long s1 = 0; s1 < q.localSubtileGrid.second; ++s1) {
          SingleBufferLoadPlan got = q.singleBufferLoadPlan(s0, s1);
          RefBufferLoadPlan ref = ref_single_buffer_load_plan(q, s0, s1);
          EXPECT_EQ(got.skip, ref.skip);
          if (ref.skip) continue;
          EXPECT_EQ(got.grBaseId, ref.grBaseId);
          EXPECT_EQ(got.offsetK, ref.offsetK);
          EXPECT_EQ(got.m0Offsets, ref.m0Offsets);
        }
      }
    }
  }
}

TEST(EmitLeavesPlans, SingleDsReadPlanValues) {
  for (const PlanGeom& pg : plan_geoms()) {
    Kernel k = make_kernel(128, 128, pg.depthU, {2, 2});
    for (long numRegs : {4, 8}) {
      for (const std::string tc : {"A", "B"}) {
        ABTileInfoQuery q = make_query(pg.pair, tc, k);
        SCOPED_TRACE(pg.name + "/" + tc + "/nr=" + std::to_string(numRegs));
        for (long s0 = 0; s0 < q.localSubtileGrid.first; ++s0) {
          for (long s1 = 0; s1 < q.localSubtileGrid.second; ++s1) {
            for (long subIterK = 0; subIterK < q.lrSubtileShape.second;
                 ++subIterK) {
              SingleDsReadPlan got =
                  q.singleDsReadPlan(s0, s1, subIterK, numRegs);
              RefDsReadPlan ref =
                  ref_single_ds_read_plan(q, s0, s1, subIterK, numRegs);
              EXPECT_EQ(got.regsPerDsRead, ref.regsPerDsRead);
              EXPECT_EQ(got.mfmaId, ref.mfmaId);
              EXPECT_EQ(got.offset, ref.offset);
              EXPECT_EQ(got.numReadsForTile, ref.numReadsForTile);
              std::vector<std::pair<long, long>> gotReads;
              for (const DsReadEntry& e : got.reads)
                gotReads.emplace_back(e.dstRegOffset, e.addrIdx);
              EXPECT_EQ(gotReads, ref.reads);
            }
          }
        }
      }
    }
  }
}
