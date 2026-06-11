// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Structured C++ replacement for the text-snapshot pass-pipeline tests.
//
// The previous version (grc.97 / grc.145) tested the LogicalScheduler pass
// pipeline by comparing print_* output against committed golden strings
// (scheduler_pass_goldens.inc, 5854 lines). This version replaces the full
// string snapshots with compact structural invariants extracted directly from
// the in-memory pass model (LogicalScheduler::partitions / emitted).
//
// Coverage across all 17 passConfigs() (BF16 / fp8 / fp4 / gfx950,
// single/multi-partition, PGR=0/1/2):
//   * Slot structure after place_LRs: partition count, slot count, MFMA
//     presence.
//   * LR/GR placement: every tensor appears in at least one LR; GR
//     mtIteration matches PGR.
//   * Dependency kinds: MFMA deps → LR, LR deps → GR, GR deps → LR.
//   * Wait ops: wait_gr preOps on LRs present iff PGR≥1.
//   * VGPR metadata: tile_peaks keys, unroll_factor self-consistency,
//     PGR=0 forces unroll_factor=1.
//   * Emit structure: every slot non-empty.
//   * Determinism and build-vs-emit equivalence.
//
// Pinned field expectations for two key configs:
//   * no_scale_k1 (M=2×N=2, numK=2, PGR=2, no scale) — single-partition
//     baseline; fully pins LR/GR tensors, VGPR peaks, MFMA dep kinds + mt
//     offsets, MFMA/LR preOp presence, and emit opType sequence.
//   * fp4_2x2 (8×8 MFMA tiles, 4×4 partitions, scale, PGR=2) — exercises
//     multi-partition placement and VGPR unrolling; pins LR tensor
//     distribution per slot and tile_peaks.

#include <gtest/gtest.h>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "scheduler_pass_configs.hpp"
#include "tensile_writer/logical_scheduler_passes.hpp"

using tw::subtile::lsched::Pass;
using tw::subtile::lsched::passes::LogicalScheduler;
using tw::subtile::lsched::passes::PKind;
using tw::subtile::lsched::passes::Placement;

namespace {

// ─── Compact extractors from the internal pass model ────────────────────────

std::vector<std::string> lrTensors(const LogicalScheduler& s, int pi, int k) {
  std::vector<std::string> out;
  for (auto* lr : s.partitions[pi][k].lrs) out.push_back(lr->tensor);
  return out;
}

std::vector<std::string> grTensors(const LogicalScheduler& s, int pi, int k) {
  std::vector<std::string> out;
  for (auto* gr : s.partitions[pi][k].grs) out.push_back(gr->tensor);
  return out;
}

std::vector<int> lrMTs(const LogicalScheduler& s, int pi, int k) {
  std::vector<int> out;
  for (auto* lr : s.partitions[pi][k].lrs) out.push_back(lr->mtIteration);
  return out;
}

std::vector<int> grMTs(const LogicalScheduler& s, int pi, int k) {
  std::vector<int> out;
  for (auto* gr : s.partitions[pi][k].grs) out.push_back(gr->mtIteration);
  return out;
}

struct DepSummary {
  std::string kind;
  std::string tensor;
  int mt_offset;
  bool operator==(const DepSummary& o) const {
    return kind == o.kind && tensor == o.tensor && mt_offset == o.mt_offset;
  }
};

std::vector<DepSummary> placementDeps(const Placement* p) {
  std::vector<DepSummary> out;
  for (auto& dep : p->deps)
    out.push_back({dep.ref->kind == PKind::LR ? "lr" : "gr",
                   dep.ref->tensor, dep.mt_offset});
  return out;
}

std::vector<std::string> preOpKinds(const Placement* p) {
  std::vector<std::string> out;
  for (auto& op : p->preOps) out.push_back(op.kind);
  return out;
}

// opType sequence for emitted[pi][k] in moduleId order.
std::vector<std::string> emitTypes(const LogicalScheduler& s, int pi, int k) {
  std::vector<std::string> out;
  if (pi >= (int)s.emitted.size() || k >= (int)s.emitted[pi].size()) return out;
  for (auto& em : s.emitted[pi][k]) out.push_back(em.opType());
  return out;
}

// Number of before-links (em.before.has_value()) in emitted[pi][k].
int emitBeforeLinks(const LogicalScheduler& s, int pi, int k) {
  if (pi >= (int)s.emitted.size() || k >= (int)s.emitted[pi].size()) return 0;
  int n = 0;
  for (auto& em : s.emitted[pi][k])
    if (em.before.has_value()) ++n;
  return n;
}

LogicalScheduler makeSched(const std::string& name) {
  for (auto& s : tw_test::passConfigs())
    if (s.name == name) return LogicalScheduler(tw_test::makePassConfig(s));
  throw std::runtime_error("Config not found: " + name);
}

}  // namespace

// ═══ Universal structural invariants (all 17 passConfigs) ══════════════════

// After place_LRs: correct partition count, numSubIterK slots per partition,
// and exactly one MFMA per slot.
TEST(LogicalSchedulerPasses, SlotStructureAfterLRPass) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.ensure(Pass::LR);
    ASSERT_EQ((int)sched.partitions.size(), sched.config.numPartitions());
    for (int pi = 0; pi < (int)sched.partitions.size(); ++pi) {
      ASSERT_EQ((int)sched.partitions[pi].size(), sched.config.numSubIterK)
          << "pi=" << pi;
      for (auto& slot : sched.partitions[pi])
        EXPECT_NE(slot.mfma, nullptr)
            << "pi=" << pi << " k=" << slot.subIterK;
    }
  }
}

// After place_GRs: every tensor in config.tensors appears in at least one LR.
TEST(LogicalSchedulerPasses, LRCoversAllTensors) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.ensure(Pass::GR);
    std::set<std::string> seen;
    for (auto& pslots : sched.partitions)
      for (auto& slot : pslots)
        for (auto* lr : slot.lrs) seen.insert(lr->tensor);
    for (auto& t : sched.tensors)
      EXPECT_TRUE(seen.count(t)) << "tensor " << t << " has no LR";
  }
}

// PGR=0 → all GR mtIteration=0 (no prefetch).
// PGR≥1 → all GR mtIteration≥1.
TEST(LogicalSchedulerPasses, GRMtIterationMatchesPGR) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.ensure(Pass::GR);
    for (auto& pslots : sched.partitions)
      for (auto& slot : pslots)
        for (auto* gr : slot.grs) {
          if (sched.config.pgr == 0)
            EXPECT_EQ(gr->mtIteration, 0);
          else
            EXPECT_GE(gr->mtIteration, 1);
        }
  }
}

// After annotate_deps: MFMA deps ref LR; LR deps (if any) ref GR; GR deps
// ref LR.
TEST(LogicalSchedulerPasses, DepKindsAreCorrect) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.ensure(Pass::DEPS);
    for (auto& pslots : sched.partitions) {
      for (auto& slot : pslots) {
        if (slot.mfma)
          for (auto& dep : slot.mfma->deps)
            EXPECT_EQ(dep.ref->kind, PKind::LR) << "MFMA dep is not LR";
        for (auto* lr : slot.lrs)
          for (auto& dep : lr->deps)
            EXPECT_EQ(dep.ref->kind, PKind::GR) << "LR dep is not GR";
        for (auto* gr : slot.grs)
          for (auto& dep : gr->deps)
            EXPECT_EQ(dep.ref->kind, PKind::LR) << "GR dep is not LR";
      }
    }
  }
}

// After remove_cross_deps: every config produces at least one wait_gr preOp
// on some LR — GRs always complete before their dependent LRs can run, so
// a wait is required regardless of PGR. (PGR=0 fetches the same MT, so the
// dependency is intra-slot; PGR≥1 fetches ahead, crossing slot boundaries.
// Either way the wait_gr is always emitted.)
TEST(LogicalSchedulerPasses, WaitGRPresent) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.ensure(Pass::REMOVE_DEPS);
    int waitGrCount = 0;
    for (auto& pslots : sched.partitions)
      for (auto& slot : pslots)
        for (auto* lr : slot.lrs)
          for (auto& op : lr->preOps)
            if (op.kind == "wait_gr") ++waitGrCount;
    EXPECT_GT(waitGrCount, 0);
  }
}

// Every emit slot is non-empty and has the correct dimensions.
TEST(LogicalSchedulerPasses, EmitSlotNonEmpty) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.emit();
    ASSERT_EQ((int)sched.emitted.size(), sched.config.numPartitions());
    for (int pi = 0; pi < (int)sched.emitted.size(); ++pi) {
      ASSERT_EQ((int)sched.emitted[pi].size(), sched.config.numSubIterK);
      for (int k = 0; k < (int)sched.emitted[pi].size(); ++k)
        EXPECT_GT((int)sched.emitted[pi][k].size(), 0)
            << "pi=" << pi << " k=" << k;
    }
  }
}

// ═══ VGPR metadata (all configs) ══════════════════════════════════════════

// tile_peaks has an entry for every active tensor; self-consistent with
// needs_unrolling and unroll_factor.
TEST(LogicalSchedulerPasses, VgprMetadataConsistent) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.assign_vgpr_tiles();
    EXPECT_EQ(sched.needs_unrolling, sched.unroll_factor > 1);
    EXPECT_GE(sched.unroll_factor, 1);
    for (const auto& t : sched.tensors)
      EXPECT_TRUE(sched.tile_peaks.count(t)) << "missing tile_peak for " << t;
  }
}

// PGR=0 forces unroll_factor=1 (no double-buffering in the tile map).
TEST(LogicalSchedulerPasses, VgprUnrollFactorForcedOneForPGR0) {
  for (const auto& spec : tw_test::passConfigs()) {
    if (spec.pgr != 0) continue;
    SCOPED_TRACE(spec.name);
    LogicalScheduler sched(tw_test::makePassConfig(spec));
    sched.assign_vgpr_tiles();
    EXPECT_EQ(sched.unroll_factor, 1);
    EXPECT_FALSE(sched.needs_unrolling);
  }
}

// ═══ Determinism and equivalence (all configs) ════════════════════════════

TEST(LogicalSchedulerPasses, PassPipelineDeterministic) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler a(tw_test::makePassConfig(spec));
    LogicalScheduler b(tw_test::makePassConfig(spec));
    a.emit();
    b.emit();
    EXPECT_EQ(a.print_emit(), b.print_emit());
  }
}

TEST(LogicalSchedulerPasses, BuildMatchesEmit) {
  for (const auto& spec : tw_test::passConfigs()) {
    SCOPED_TRACE(spec.name);
    LogicalScheduler emitSched(tw_test::makePassConfig(spec));
    emitSched.emit();
    LogicalScheduler buildSched(tw_test::makePassConfig(spec));
    buildSched.build();
    EXPECT_EQ(buildSched.print_emit(), emitSched.print_emit());
  }
}

// ═══ Pinned: no_scale_k1 (M=2×N=2, numK=2, PGR=2, no scale) ═════════════

// place_LRs: slot 0 gets A+B (mt=0); slot 1 gets A+B (mt=1, next MT).
TEST(LogicalSchedulerPasses, LRSlots_NoScaleK1) {
  LogicalScheduler s = makeSched("no_scale_k1");
  s.ensure(Pass::LR);
  ASSERT_EQ((int)s.partitions.size(), 1);
  EXPECT_EQ(lrTensors(s, 0, 0), (std::vector<std::string>{"A", "B"}));
  EXPECT_EQ(lrMTs(s, 0, 0), (std::vector<int>{0, 0}));
  EXPECT_EQ(lrTensors(s, 0, 1), (std::vector<std::string>{"A", "B"}));
  EXPECT_EQ(lrMTs(s, 0, 1), (std::vector<int>{1, 1}));
}

// place_GRs (PGR=2): slot 0 prefetches A (mt=2); slot 1 prefetches B (mt=2).
TEST(LogicalSchedulerPasses, GRSlots_NoScaleK1) {
  LogicalScheduler s = makeSched("no_scale_k1");
  s.ensure(Pass::GR);
  EXPECT_EQ(grTensors(s, 0, 0), (std::vector<std::string>{"A"}));
  EXPECT_EQ(grMTs(s, 0, 0), (std::vector<int>{2}));
  EXPECT_EQ(grTensors(s, 0, 1), (std::vector<std::string>{"B"}));
  EXPECT_EQ(grMTs(s, 0, 1), (std::vector<int>{2}));
}

// assign_vgpr_tiles: 2 sets × 2 tile groups = 4 per tensor; no unrolling.
TEST(LogicalSchedulerPasses, VgprPeaks_NoScaleK1) {
  LogicalScheduler s = makeSched("no_scale_k1");
  s.assign_vgpr_tiles();
  EXPECT_EQ(s.unroll_factor, 1);
  EXPECT_FALSE(s.needs_unrolling);
  EXPECT_EQ(s.tile_peaks, (std::map<std::string, int>{{"A", 4}, {"B", 4}}));
}

// annotate_deps: MFMA(k=0) waits on LRs from the previous MT (mt_offset=-1);
// MFMA(k=1) waits on LRs from the current MT (mt_offset=0).
TEST(LogicalSchedulerPasses, MFMADepMtOffsets_NoScaleK1) {
  LogicalScheduler s = makeSched("no_scale_k1");
  s.ensure(Pass::DEPS);
  auto d0 = placementDeps(s.partitions[0][0].mfma);
  ASSERT_EQ((int)d0.size(), 2);
  EXPECT_EQ(d0[0], (DepSummary{"lr", "A", -1}));
  EXPECT_EQ(d0[1], (DepSummary{"lr", "B", -1}));

  auto d1 = placementDeps(s.partitions[0][1].mfma);
  ASSERT_EQ((int)d1.size(), 2);
  EXPECT_EQ(d1[0].kind, "lr");
  EXPECT_EQ(d1[0].mt_offset, 0);
  EXPECT_EQ(d1[1].kind, "lr");
  EXPECT_EQ(d1[1].mt_offset, 0);
}

// After group_lr_gr: both MFMAs have wait_lr; slot 0 LRs have no wait_gr
// (GR already guaranteed by the pipeline); slot 1's first LR has wait_gr.
TEST(LogicalSchedulerPasses, PreOpPresence_NoScaleK1) {
  LogicalScheduler s = makeSched("no_scale_k1");
  s.ensure(Pass::GROUP_LR_GR);

  // MFMAs always get wait_lr after remove_cross_deps.
  EXPECT_EQ(preOpKinds(s.partitions[0][0].mfma),
            (std::vector<std::string>{"wait_lr"}));
  EXPECT_EQ(preOpKinds(s.partitions[0][1].mfma),
            (std::vector<std::string>{"wait_lr"}));

  // Slot 0 LRs: no wait_gr (deps were removed as already-guaranteed).
  for (auto* lr : s.partitions[0][0].lrs)
    for (auto& op : lr->preOps)
      EXPECT_NE(op.kind, "wait_gr") << "unexpected wait_gr on slot-0 LR";

  // Slot 1 LRs: at least one wait_gr from the cross-slot GR dependency.
  bool anyWaitGr = false;
  for (auto* lr : s.partitions[0][1].lrs)
    for (auto& op : lr->preOps)
      if (op.kind == "wait_gr") anyWaitGr = true;
  EXPECT_TRUE(anyWaitGr);
}

// emit: pinned opType sequence and before-link counts for both slots.
// Slot 0: 8 modules — placement chain + wait_lr pair + sync + gr_inc.
// Slot 1: 10 modules — adds wait_gr, lr_inc pair before the gr_inc.
TEST(LogicalSchedulerPasses, EmitStructure_NoScaleK1) {
  LogicalScheduler s = makeSched("no_scale_k1");
  s.emit();

  EXPECT_EQ(emitTypes(s, 0, 0),
            (std::vector<std::string>{"mfma", "lr", "lr", "gr", "wait_lr",
                                      "wait_lr", "sync", "gr_inc"}));
  EXPECT_EQ(emitBeforeLinks(s, 0, 0), 6);

  EXPECT_EQ(emitTypes(s, 0, 1),
            (std::vector<std::string>{"mfma", "lr", "lr", "gr", "wait_lr",
                                      "wait_gr", "sync", "lr_inc", "lr_inc",
                                      "gr_inc"}));
  EXPECT_EQ(emitBeforeLinks(s, 0, 1), 7);
}

// ═══ Pinned: fp4_2x2 (8×8 tiles, 4 partitions, scale SA/SB, PGR=2) ════════

// place_LRs: each partition picks up the subset of tensors it needs.
// Partition 0 slot 0 loads the bulk (A, B, SA). Last partition slot 1
// wraps all (A, B, SB) for the next MT.
TEST(LogicalSchedulerPasses, LRSlots_Fp4_2x2) {
  LogicalScheduler s = makeSched("fp4_2x2");
  s.ensure(Pass::LR);
  ASSERT_EQ((int)s.partitions.size(), 4);
  EXPECT_EQ(lrTensors(s, 0, 0), (std::vector<std::string>{"A", "B", "SA"}));
  EXPECT_EQ(lrTensors(s, 0, 1), (std::vector<std::string>{"A"}));
  EXPECT_EQ(lrTensors(s, 1, 0), (std::vector<std::string>{"A", "SB"}));
  EXPECT_EQ(lrTensors(s, 1, 1), (std::vector<std::string>{"B"}));
  EXPECT_EQ(lrTensors(s, 2, 0), (std::vector<std::string>{"B"}));
  EXPECT_EQ(lrTensors(s, 2, 1), (std::vector<std::string>{}));
  EXPECT_EQ(lrTensors(s, 3, 0), (std::vector<std::string>{"SA"}));
  EXPECT_EQ(lrTensors(s, 3, 1), (std::vector<std::string>{"A", "B", "SB"}));
}

// assign_vgpr_tiles: 2 sets × 8 tile groups = 16 for A/B; scale tensors have
// mn=2 → 4 groups × 2 sets = 8; unroll_factor=2 (odd k-chunks for SA/SB).
TEST(LogicalSchedulerPasses, VgprPeaks_Fp4_2x2) {
  LogicalScheduler s = makeSched("fp4_2x2");
  s.assign_vgpr_tiles();
  EXPECT_EQ(s.unroll_factor, 2);
  EXPECT_TRUE(s.needs_unrolling);
  EXPECT_EQ(s.tile_peaks,
            (std::map<std::string, int>{
                {"A", 16}, {"B", 16}, {"SA", 8}, {"SB", 8}}));
}
