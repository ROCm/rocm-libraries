// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Native C++ port of the data-only portions of
// Tensile/Tests/unit/test_instructionSchedulerCpp.py.
//
// The deleted Python file exercised the subtile InstructionScheduler through
// the nanobind shim: it built real rocisa instructions, classified them into
// the data-only C++ model, ran the compiled slot-placement algorithm, and
// pinned the final emission order + vmcnt. Because the slot-placement
// algorithm is C++-only (instruction_scheduler.hpp), these tests exercise that
// header directly on the data-only ModuleRef / InstRef model.
//
// Ported here (no rocisa / writer dependency):
//   * the six golden slot-placement scenarios (order + vmcnt post-pass),
//   * the order-is-a-permutation and MFMA-relative-order invariants,
//   * the data-only schedule / vmcnt-post-pass unit cases.
//
// NOT ported (kept as Python rocisa integration in
// test_SubtileBasedLogicalScheduler.py): classifyInstruction's rocisa →
// InstKind mapping and the live-rocisa instructionSchedule rebuild, which
// exercise rocisa object behavior rather than the C++ algorithm.

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensile_writer/instruction_scheduler.hpp"

using namespace tw::subtile::insched;

namespace {

using K = InstKind;

// One tagged instruction: a data-only kind plus a label used (like the rocisa
// `comment` in the Python parity test) to make the emitted order identifiable.
struct TaggedInst {
  InstKind kind;
  std::string tag;
  long vlcnt = -1;
  bool adjust = true;
};

struct TaggedModule {
  int moduleId;
  std::string opType;
  std::optional<int> before;
  std::vector<TaggedInst> insts;
};

struct Built {
  std::vector<ModuleRef> modules;
  std::vector<std::vector<std::string>> tags;  // [moduleIdx][instIdx] -> tag
};

Built build(const std::vector<TaggedModule>& spec) {
  Built b;
  for (const auto& m : spec) {
    std::vector<InstRef> irs;
    std::vector<std::string> mtags;
    for (const auto& i : m.insts) {
      irs.push_back(InstRef(i.kind, i.vlcnt, i.adjust));
      mtags.push_back(i.tag);
    }
    b.modules.push_back(ModuleRef(m.moduleId, m.opType, m.before, std::move(irs)));
    b.tags.push_back(std::move(mtags));
  }
  return b;
}

using Sig = std::vector<std::pair<std::string, long>>;

// (tag, post-pass vlcnt) for every emitted instruction, in emission order.
// Mirrors test_instructionSchedulerCpp._signature; the Python golden's `None`
// vlcnt corresponds to the data model's -1 default.
Sig signature(const Built& b, const ScheduleResult& r) {
  Sig out;
  for (size_t i = 0; i < r.order.size(); ++i) {
    const InstId& ref = r.order[i];
    out.emplace_back(b.tags[ref.first][ref.second], r.vlcnt[i]);
  }
  return out;
}

// moduleId offset from list index (10 * index) so the idToIdx / before-link
// resolution is exercised, not just positional ids (mirrors Python `_mid`).
int mid(int i) { return 10 * i; }

Built scenario_rich_multi() {
  return build({
      {mid(0), "mfma", mid(1),
       {{K::Mfma, "mfma0"}, {K::Mfma, "mfma1"}, {K::Mfma, "mfma2"}, {K::Mfma, "mfma3"}}},
      {mid(1), "wait_lr", std::nullopt, {{K::WaitCnt, "prewait", -1}}},
      {mid(2), "lr", std::nullopt, {{K::LocalRead, "lrA"}, {K::LocalRead, "lrB"}}},
      {mid(3), "gr", std::nullopt,
       {{K::GlobalRead, "gr0"}, {K::GlobalRead, "gr1"}, {K::GlobalRead, "gr2"}}},
      {mid(4), "wait_gr", mid(5), {{K::WaitCnt, "wgr", 2}}},
      {mid(5), "lr_inc", std::nullopt, {{K::Other, "lrinc"}}},
      {mid(6), "m0", std::nullopt, {{K::M0Update, "m0a"}}},
  });
}

Built scenario_single_mfma() {
  return build({
      {mid(0), "mfma", mid(1), {{K::Mfma, "mfma0"}}},
      {mid(1), "wait_lr", std::nullopt, {{K::WaitCnt, "prewait", 1}}},
      {mid(2), "lr", std::nullopt, {{K::LocalRead, "lrA"}, {K::LocalRead, "lrB"}}},
      {mid(3), "gr", std::nullopt, {{K::GlobalRead, "gr0"}, {K::GlobalRead, "gr1"}}},
  });
}

Built scenario_ds_read_wait_gap() {
  return build({
      {mid(0), "mfma", std::nullopt,
       {{K::Mfma, "mfma0"}, {K::Mfma, "mfma1"}, {K::Mfma, "mfma2"},
        {K::Mfma, "mfma3"}, {K::Mfma, "mfma4"}, {K::Mfma, "mfma5"}}},
      {mid(1), "lr", std::nullopt,
       {{K::LocalRead, "lr0"}, {K::LocalRead, "lr1"}, {K::LocalRead, "lr2"},
        {K::LocalRead, "lr3"}}},
      {mid(2), "wait", std::nullopt, {{K::WaitCnt, "w0", 0}}},
      {mid(3), "gr", std::nullopt,
       {{K::GlobalRead, "g0"}, {K::GlobalRead, "g1"}, {K::GlobalRead, "g2"},
        {K::GlobalRead, "g3"}}},
  });
}

Built scenario_chained_path() {
  return build({
      {mid(0), "mfma", std::nullopt,
       {{K::Mfma, "mfma0"}, {K::Mfma, "mfma1"}, {K::Mfma, "mfma2"}}},
      {mid(1), "lr", std::nullopt, {{K::LocalRead, "lrA"}}},
      {mid(2), "gr", mid(1), {{K::GlobalRead, "gr0"}, {K::GlobalRead, "gr1"}}},
      {mid(3), "lr_inc", mid(2), {{K::Other, "inc"}}},
      {mid(4), "m0", std::nullopt, {{K::M0Update, "m0a"}}},
  });
}

Built scenario_two_mfma_minimal() {
  return build({
      {mid(0), "mfma", std::nullopt, {{K::Mfma, "mfma0"}, {K::Mfma, "mfma1"}}},
      {mid(1), "lr", std::nullopt, {{K::LocalRead, "lrA"}}},
      {mid(2), "gr", std::nullopt, {{K::GlobalRead, "gr0"}}},
  });
}

Built scenario_generic_other() {
  // The trailing Other (a rocisa Label in the Python test) carries an empty
  // tag, so it surfaces as ("", -1) and is placed generically.
  return build({
      {mid(0), "mfma", std::nullopt, {{K::Mfma, "mfma0"}, {K::Mfma, "mfma1"}}},
      {mid(1), "lr", std::nullopt, {{K::LocalRead, "lrA"}, {K::Other, ""}}},
      {mid(2), "gr", std::nullopt, {{K::GlobalRead, "gr0"}}},
  });
}

struct NamedScenario {
  std::string name;
  Built (*make)();
  Sig golden;
};

const std::vector<NamedScenario>& scenarios() {
  static const std::vector<NamedScenario> kScenarios = {
      {"rich_multi", scenario_rich_multi,
       {{"prewait", -1}, {"mfma0", -1}, {"lrA", -1}, {"gr0", -1}, {"lrB", -1},
        {"gr1", -1}, {"mfma1", -1}, {"gr2", -1}, {"mfma2", -1}, {"lrinc", -1},
        {"m0a", -1}, {"wgr", 5}, {"mfma3", -1}}},
      {"single_mfma", scenario_single_mfma,
       {{"prewait", 1}, {"mfma0", -1}, {"lrA", -1}, {"lrB", -1}, {"gr0", -1},
        {"gr1", -1}}},
      {"ds_read_wait_gap", scenario_ds_read_wait_gap,
       {{"mfma0", -1}, {"lr0", -1}, {"w0", 0}, {"g0", -1}, {"mfma1", -1},
        {"lr1", -1}, {"lr2", -1}, {"g1", -1}, {"mfma2", -1}, {"lr3", -1},
        {"g2", -1}, {"mfma3", -1}, {"g3", -1}, {"mfma4", -1}, {"mfma5", -1}}},
      {"chained_path", scenario_chained_path,
       {{"mfma0", -1}, {"lrA", -1}, {"m0a", -1}, {"gr0", -1}, {"mfma1", -1},
        {"gr1", -1}, {"inc", -1}, {"mfma2", -1}}},
      {"two_mfma_minimal", scenario_two_mfma_minimal,
       {{"mfma0", -1}, {"lrA", -1}, {"gr0", -1}, {"mfma1", -1}}},
      {"generic_other", scenario_generic_other,
       {{"mfma0", -1}, {"lrA", -1}, {"gr0", -1}, {"", -1}, {"mfma1", -1}}},
  };
  return kScenarios;
}

}  // namespace

// ---------------------------------------------------------------------------
// Order + vmcnt: the C++ scheduler matches the pinned golden signatures.
// ---------------------------------------------------------------------------
TEST(InstructionSchedulerGolden, MatchesGoldenSignatures) {
  for (const NamedScenario& sc : scenarios()) {
    SCOPED_TRACE(sc.name);
    Built b = sc.make();
    ScheduleResult r = schedule(b.modules);
    EXPECT_EQ(signature(b, r), sc.golden);
  }
}

TEST(InstructionSchedulerGolden, OrderIsPermutationOfInput) {
  for (const NamedScenario& sc : scenarios()) {
    SCOPED_TRACE(sc.name);
    Built b = sc.make();
    ScheduleResult r = schedule(b.modules);
    std::vector<std::string> emitted;
    for (const InstId& ref : r.order) emitted.push_back(b.tags[ref.first][ref.second]);
    std::vector<std::string> expected;
    for (const auto& mtags : b.tags)
      for (const auto& t : mtags) expected.push_back(t);
    std::sort(emitted.begin(), emitted.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(emitted, expected);
  }
}

TEST(InstructionSchedulerGolden, MfmaRelativeOrderPreserved) {
  for (const NamedScenario& sc : scenarios()) {
    SCOPED_TRACE(sc.name);
    Built b = sc.make();
    std::vector<std::string> mfmaIn;
    for (size_t mi = 0; mi < b.modules.size(); ++mi)
      for (size_t ii = 0; ii < b.modules[mi].instructions.size(); ++ii)
        if (b.modules[mi].instructions[ii].kind == K::Mfma)
          mfmaIn.push_back(b.tags[mi][ii]);
    ScheduleResult r = schedule(b.modules);
    std::vector<std::string> mfmaOut;
    for (size_t i = 0; i < r.order.size(); ++i)
      if (r.kinds[i] == K::Mfma) mfmaOut.push_back(b.tags[r.order[i].first][r.order[i].second]);
    EXPECT_EQ(mfmaOut, mfmaIn);
  }
}

TEST(InstructionSchedulerGolden, EmptyChainReturnsEmpty) {
  ScheduleResult r = schedule({});
  EXPECT_TRUE(r.order.empty());
  EXPECT_TRUE(r.kinds.empty());
}

// ---------------------------------------------------------------------------
// Data-only model unit cases (mirroring TestDataOnlyModel).
// ---------------------------------------------------------------------------
TEST(InstructionSchedulerDataOnly, ScheduleReturnsAllInstructions) {
  std::vector<ModuleRef> modules = {
      ModuleRef(0, "mfma", std::nullopt, {InstRef(K::Mfma), InstRef(K::Mfma)}),
      ModuleRef(1, "wait_lr", std::nullopt, {InstRef(K::WaitCnt, 3, true)}),
      ModuleRef(2, "lr", std::nullopt, {InstRef(K::LocalRead)}),
      ModuleRef(3, "gr", std::nullopt, {InstRef(K::GlobalRead)}),
  };
  ScheduleResult r = schedule(modules);
  EXPECT_EQ(r.order.size(), 5u);
  EXPECT_EQ(r.kinds.size(), 5u);
  // With no preMFMA dependency, the two MFMAs bracket the single interval.
  EXPECT_EQ(r.kinds.front(), K::Mfma);
  EXPECT_EQ(r.kinds.back(), K::Mfma);
}

TEST(InstructionSchedulerDataOnly, VmcntPostPassBumpsWaitcnt) {
  // Single MFMA so the order is deterministic: MFMA, then paths. The GR path
  // (2 buffer loads) precedes a trailing waitcnt path.
  std::vector<ModuleRef> modules = {
      ModuleRef(0, "mfma", std::nullopt, {InstRef(K::Mfma)}),
      ModuleRef(1, "gr", std::nullopt, {InstRef(K::GlobalRead), InstRef(K::GlobalRead)}),
      ModuleRef(2, "wait", std::nullopt, {InstRef(K::WaitCnt, 5, true)}),
  };
  ScheduleResult r = schedule(modules);
  int waitIdx = -1;
  int numWaits = 0;
  for (size_t i = 0; i < r.kinds.size(); ++i)
    if (r.kinds[i] == K::WaitCnt) {
      waitIdx = static_cast<int>(i);
      ++numWaits;
    }
  ASSERT_EQ(numWaits, 1);
  EXPECT_EQ(r.vlcnt[waitIdx], 5 + 2);
  std::vector<std::pair<int, long>> expected = {{waitIdx, 2}};
  EXPECT_EQ(r.vmcntAdjustments, expected);
}

TEST(InstructionSchedulerDataOnly, NoAdjustWhenFlagFalse) {
  std::vector<ModuleRef> modules = {
      ModuleRef(0, "mfma", std::nullopt, {InstRef(K::Mfma)}),
      ModuleRef(1, "gr", std::nullopt, {InstRef(K::GlobalRead)}),
      ModuleRef(2, "wait", std::nullopt, {InstRef(K::WaitCnt, 5, false)}),
  };
  ScheduleResult r = schedule(modules);
  EXPECT_TRUE(r.vmcntAdjustments.empty());
}

TEST(InstructionSchedulerDataOnly, EmptyChain) {
  EXPECT_TRUE(schedule({}).order.empty());
}

TEST(InstructionSchedulerDataOnly, MultipleMfmaModulesRaises) {
  std::vector<ModuleRef> modules = {
      ModuleRef(0, "mfma", std::nullopt, {InstRef(K::Mfma)}),
      ModuleRef(1, "mfma", std::nullopt, {InstRef(K::Mfma)}),
  };
  EXPECT_THROW(schedule(modules), std::invalid_argument);
}
