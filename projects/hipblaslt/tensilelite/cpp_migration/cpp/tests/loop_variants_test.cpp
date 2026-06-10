// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Native C++ coverage for the loop-variant schedule builders ported from the
// Python LogicalScheduler (build_ngll / build_nll / build_preloop /
// build_tailloop_pgr0). These lock the structural / before-dependency and
// placement semantics the Python wrappers now rely on verbatim:
//
//   * NGLL drops GR(MT 2) and zeroes wait_gr inflight counts (PGR<2 ⇒ empty).
//   * NLL drops all GR, LR(MT 1), gr_inc(PGR2) and orphaned sync/wait_lr
//     (PGR0 ⇒ empty), zeroing the remaining wait_gr counts.
//   * preloop builds the PGR1/PGR2 init sequence (GRs + LR + skip guards;
//     PGR2 adds gr_inc, MT1 GRs and the NGLL skip; PGR0 ⇒ empty).
//   * tail-PGR0 builds the flat per-subIterK groups (one preamble group then
//     numSubIterK groups, each with all partitions' MFMAs + one K-mask), with
//     the BF16 boundary-fixup InlineModuleOp inserted only when requested.
//
// The byte-identical Python-vs-C++ end-to-end emission is covered separately by
// the retained test_SubtileBasedLogicalScheduler.py integration suite.

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "scheduler_pass_configs.hpp"
#include "tensile_writer/logical_scheduler_passes.hpp"

namespace {

using tw::subtile::lsched::passes::LogicalScheduler;
using ModuleGrid = LogicalScheduler::ModuleGrid;
using FlatTileMaps = LogicalScheduler::FlatTileMaps;

// True for the [[[]]] degenerate grid emit() produces for the no-op PGR cases.
bool isEmptyGrid(const ModuleGrid& g) {
  return g.size() == 1 && g[0].size() == 1 && g[0][0].empty();
}

// Count modules of a given opType across every partition/subIterK group.
int countOp(const ModuleGrid& g, const std::string& op) {
  int n = 0;
  for (const auto& partition : g)
    for (const auto& group : partition)
      for (const auto& em : group)
        if (em.opType() == op) ++n;
  return n;
}

template <typename Fn>
int countIf(const ModuleGrid& g, Fn pred) {
  int n = 0;
  for (const auto& partition : g)
    for (const auto& group : partition)
      for (const auto& em : group)
        if (pred(em)) ++n;
  return n;
}

// Flat list of every module in the grid (preserving order).
std::vector<const tw::subtile::lsched::EmittedModule*> flatten(
    const ModuleGrid& g) {
  std::vector<const tw::subtile::lsched::EmittedModule*> out;
  for (const auto& partition : g)
    for (const auto& group : partition)
      for (const auto& em : group) out.push_back(&em);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// NGLL: no GR(MT 2), wait_gr counts zeroed; PGR<2 ⇒ empty grid.
// ---------------------------------------------------------------------------
TEST(LoopVariants, NgllDropsGrMt2AndZeroesWaitCounts) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.build_ngll();
    const ModuleGrid& g = sched.value_ngll();

    if (spec.pgr < 2) {
      EXPECT_TRUE(isEmptyGrid(g));
      continue;
    }

    // No GR at MT 2 survives; the prefetched (n+2) global loads are gone.
    int grMt2 = countIf(g, [](const auto& em) {
      return em.opType() == "gr" &&
             std::get<tw::subtile::lsched::GRPlacement>(*em.source).mtIteration ==
                 2;
    });
    EXPECT_EQ(grMt2, 0);

    // Every surviving wait_gr has zeroed inflight counts.
    int nonZeroWaits = countIf(g, [](const auto& em) {
      if (em.opType() != "wait_gr") return false;
      const auto& wg = std::get<tw::subtile::lsched::WaitGROp>(*em.source);
      if (!wg.wait_gr_counts.has_value()) return false;
      const auto& c = *wg.wait_gr_counts;
      return c.A || c.B || c.SA || c.SB;
    });
    EXPECT_EQ(nonZeroWaits, 0);
  }
}

// ---------------------------------------------------------------------------
// NLL: no GR at all, no LR(MT 1), no gr_inc(PGR2); PGR0 ⇒ empty grid.
// ---------------------------------------------------------------------------
TEST(LoopVariants, NllDropsLoadsAndIncrements) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.build_nll();
    const ModuleGrid& g = sched.value_nll();

    if (spec.pgr == 0) {
      EXPECT_TRUE(isEmptyGrid(g));
      continue;
    }

    EXPECT_EQ(countOp(g, "gr"), 0) << "NLL keeps no global reads";

    int lrMt1 = countIf(g, [](const auto& em) {
      return em.opType() == "lr" &&
             std::get<tw::subtile::lsched::LRPlacement>(*em.source).mtIteration ==
                 1;
    });
    EXPECT_EQ(lrMt1, 0) << "NLL keeps no next-MT local reads";

    if (spec.pgr == 2)
      EXPECT_EQ(countOp(g, "gr_inc"), 0) << "PGR2 NLL drops gr_inc";

    // Surviving wait_gr counts are zeroed (no loads remain in flight).
    int nonZeroWaits = countIf(g, [](const auto& em) {
      if (em.opType() != "wait_gr") return false;
      const auto& wg = std::get<tw::subtile::lsched::WaitGROp>(*em.source);
      if (!wg.wait_gr_counts.has_value()) return false;
      const auto& c = *wg.wait_gr_counts;
      return c.A || c.B || c.SA || c.SB;
    });
    EXPECT_EQ(nonZeroWaits, 0);
  }
}

// ---------------------------------------------------------------------------
// preloop: PGR0 ⇒ empty; PGR1 ⇒ GR+LR+skip(NLL); PGR2 ⇒ +gr_inc/MT1 GR/NGLL.
// ---------------------------------------------------------------------------
TEST(LoopVariants, PreloopInitSequence) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.build_preloop();
    const ModuleGrid& g = sched.value_preloop();

    if (spec.pgr == 0) {
      EXPECT_TRUE(isEmptyGrid(g));
      continue;
    }

    // Shape is exactly [1 partition][1 subIterK].
    ASSERT_EQ(g.size(), 1u);
    ASSERT_EQ(g[0].size(), 1u);

    // GR(MT 0) for every tensor, plus the entry LR + wait/sync.
    EXPECT_GE(countIf(g,
                      [](const auto& em) {
                        return em.opType() == "gr" &&
                               std::get<tw::subtile::lsched::GRPlacement>(
                                   *em.source)
                                       .mtIteration == 0;
                      }),
              (int)sched.tensors.size());
    EXPECT_GT(countOp(g, "lr"), 0);
    EXPECT_EQ(countOp(g, "wait_gr"), 1);
    EXPECT_GE(countOp(g, "sync"), 1);

    // Skip guards: NLL always; NGLL only for PGR2.
    auto countSkip = [&](const std::string& target) {
      return countIf(g, [&](const auto& em) {
        return em.opType() == "skip" &&
               std::get<tw::subtile::lsched::SkipOp>(*em.source).target == target;
      });
    };
    EXPECT_EQ(countSkip("NLL"), 1);

    if (spec.pgr == 1) {
      EXPECT_EQ(countOp(g, "gr_inc"), 0);
      EXPECT_EQ(countSkip("NGLL"), 0);
    } else {
      EXPECT_GT(countOp(g, "gr_inc"), 0);
      EXPECT_EQ(countSkip("NGLL"), 1);
      // PGR2 prefetches the next MT's first-partition tiles.
      EXPECT_GT(countIf(g,
                        [](const auto& em) {
                          return em.opType() == "gr" &&
                                 std::get<tw::subtile::lsched::GRPlacement>(
                                     *em.source)
                                         .mtIteration == 1;
                        }),
                0);
    }
  }
}

// ---------------------------------------------------------------------------
// tail-PGR0: one preamble group + numSubIterK per-k groups; each k group has
// all partitions' MFMAs and one K-mask; BF16 inserts the boundary fixup.
// ---------------------------------------------------------------------------
TEST(LoopVariants, TailLoopFlatGroups) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    for (bool bf16 : {false, true}) {
      LogicalScheduler sched(tw_test::makePassConfig(spec));
      sched.ensure(tw::subtile::lsched::Pass::EMIT);
      const int numP = sched.config.numPartitions();
      const int numK = sched.config.numSubIterK;
      const int miK = 16;

      // The Python wrapper supplies the flat tile layout; structural coverage
      // only needs one (empty) map per partition.
      FlatTileMaps tile_maps(numP);
      sched.build_tailloop_pgr0(tile_maps, bf16, miK);
      const ModuleGrid& g = sched.value_tailloop();

      // Single outer partition holding (1 preamble + numK) groups.
      ASSERT_EQ(g.size(), 1u);
      ASSERT_EQ(g[0].size(), (size_t)(numK + 1));

      const auto& preamble = g[0][0];
      int preGr = 0, preWaitGr = 0, preInline = 0;
      for (const auto& em : preamble) {
        if (em.opType() == "gr") ++preGr;
        if (em.opType() == "wait_gr") ++preWaitGr;
        if (em.opType() == "inline") ++preInline;
      }
      EXPECT_EQ(preGr, (int)sched.tensors.size());
      EXPECT_EQ(preWaitGr, 1);
      EXPECT_EQ(preInline, bf16 ? 1 : 0)
          << "BF16 boundary fixup present iff bf16";

      // Per-k groups: numP MFMAs, exactly one K-mask tagged with k, an early
      // exit on every k except the last.
      for (int k = 0; k < numK; ++k) {
        const auto& group = g[0][k + 1];
        int mfma = 0, mask = 0, skip = 0, waitLr = 0;
        for (const auto& em : group) {
          if (em.opType() == "mfma") ++mfma;
          if (em.opType() == "wait_lr") ++waitLr;
          if (em.opType() == "mask_k") {
            ++mask;
            EXPECT_EQ(
                std::get<tw::subtile::lsched::MaskKOp>(*em.source).subIterK, k);
          }
          if (em.opType() == "skip") ++skip;
        }
        EXPECT_EQ(mfma, numP) << "all partitions' MFMAs for k=" << k;
        EXPECT_EQ(mask, 1);
        EXPECT_EQ(waitLr, 1);
        EXPECT_EQ(skip, k == numK - 1 ? 0 : 1) << "early-exit except last k";
      }
    }
  }
}
