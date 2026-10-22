// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/WaitTracker.hpp>

#include "BuilderTestSupport.hpp"

#include "container.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;
using Waits = std::vector<std::string>;

class WaitTrackerTest : public BuilderTest {
protected:
  // The same registers serve every instruction, since only the text matters.
  RegisterRange vdst = builder.allocateVgpr(4);
  RegisterRange voffset = builder.allocateVgpr(1);
  RegisterRange rsrc = builder.allocateSgpr(4, 4);
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange spair = builder.allocateSgpr(2, 2);
  WaitTracker tracker{builder};

  WaitToken vectorLoad() {
    return tracker.issue(builder.create<rocisa::BufferLoadB128>(
        vdst.container(), voffset.container(), rsrc.container(), 0,
        rocisa::MUBUFModifiers(true, 0)));
  }

  WaitToken vectorStore() {
    return tracker.issue(builder.create<rocisa::BufferStoreB32>(
        vdst.sub(0, 1), voffset.container(), rsrc.container(), 0,
        rocisa::MUBUFModifiers(true, 0)));
  }

  WaitToken ldsLoad() {
    return tracker.issue(builder.create<rocisa::DSLoadB32>(
        vdst.sub(0, 1), voffset.container(), rocisa::DSModifiers(1, 0)));
  }

  WaitToken scalarLoad() {
    return tracker.issue(builder.create<rocisa::SLoadB32>(
        sdst.container(), spair.container(), 0));
  }

  std::vector<WaitToken> vectorLoads(int count) {
    std::vector<WaitToken> tokens;
    for (int i = 0; i < count; ++i)
      tokens.push_back(vectorLoad());
    return tokens;
  }

  // Each emitted wait instruction, without padding and comments.
  Waits waits() {
    Waits result;
    std::istringstream stream(body.toString());
    for (std::string current; std::getline(stream, current);) {
      std::size_t begin = current.find("s_waitcnt");
      if (begin == std::string::npos)
        continue;
      std::string inst = current.substr(begin, current.find("//") - begin);
      result.push_back(inst.substr(0, inst.find_last_not_of(" \t") + 1));
    }
    return result;
  }
};

TEST_F(WaitTrackerTest, StraightLineVectorLoads) {
  std::vector<WaitToken> t = vectorLoads(3);
  tracker.waitFor(t[0]);
  tracker.waitFor(t[1]);
  tracker.waitFor(t[0]);
  tracker.waitFor(t[2]);
  EXPECT_EQ(waits(), (Waits{"s_waitcnt vmcnt(2)", "s_waitcnt vmcnt(1)",
                            "s_waitcnt vmcnt(0)"}));
}

TEST_F(WaitTrackerTest, ElidesWaitsAlreadySatisfied) {
  std::vector<WaitToken> t = vectorLoads(3);
  tracker.waitFor(t[1]);
  tracker.waitFor(t[0]);
  EXPECT_EQ(waits(), (Waits{"s_waitcnt vmcnt(1)"}));
}

TEST_F(WaitTrackerTest, StoresCountOnVmcnt) {
  WaitToken t0 = vectorLoad();
  vectorStore();
  vectorLoad();
  tracker.waitFor(t0);
  EXPECT_EQ(waits(), (Waits{"s_waitcnt vmcnt(2)"}));
}

TEST_F(WaitTrackerTest, LdsIgnoresScalarLoads) {
  WaitToken a = scalarLoad();
  WaitToken d0 = ldsLoad();
  WaitToken b = scalarLoad();
  WaitToken d1 = ldsLoad();
  tracker.waitFor(d0);
  tracker.waitFor(a);
  tracker.waitFor(b);
  tracker.waitFor(d1);
  EXPECT_EQ(waits(), (Waits{"s_waitcnt lgkmcnt(1)", "s_waitcnt lgkmcnt(0)"}));
}

TEST_F(WaitTrackerTest, CombinesCountersInOneWait) {
  std::vector<WaitToken> v = vectorLoads(2);
  WaitToken d0 = ldsLoad();
  ldsLoad();
  std::vector<WaitToken> tokens{v[0], d0};
  tracker.waitFor(tokens);
  EXPECT_EQ(waits(), (Waits{"s_waitcnt lgkmcnt(1), vmcnt(1)"}));
}

TEST_F(WaitTrackerTest, CapsCountsAtTargetMaxima) {
  std::vector<WaitToken> v = vectorLoads(70);
  tracker.waitFor(v[0]);
  // Indices 0 to 6 are at distance >= 63 and complete.
  tracker.waitFor(v[6]);
  tracker.waitFor(v[7]);
  EXPECT_EQ(waits(), (Waits{"s_waitcnt vmcnt(63)", "s_waitcnt vmcnt(62)"}));

  std::vector<WaitToken> d;
  for (int i = 0; i < 20; ++i)
    d.push_back(ldsLoad());
  tracker.waitFor(d[0]);
  EXPECT_EQ(waits().back(), "s_waitcnt lgkmcnt(15)");
}

TEST_F(WaitTrackerTest, WaitAll) {
  vectorLoads(2);
  ldsLoad();
  scalarLoad();
  tracker.waitAll(WaitKind::vectorMemory);
  tracker.waitAll(WaitKind::vectorMemory);
  tracker.waitAll(WaitKind::scalarMemory);
  tracker.waitAll(WaitKind::lds);
  EXPECT_EQ(waits(), (Waits{"s_waitcnt vmcnt(0)", "s_waitcnt lgkmcnt(0)"}));
}

TEST_F(WaitTrackerTest, BranchArmsJoinConservatively) {
  std::vector<WaitToken> t = vectorLoads(4);
  WaitTracker::State before = tracker.snapshot();
  tracker.waitFor(t[1]);
  WaitTracker::State armA = tracker.snapshot();
  tracker.restore(before);
  vectorLoads(2);
  tracker.join(armA);
  // Arm A completed t0 while arm B has five operations after it.
  tracker.waitFor(t[0]);
  // The newest position holding t2 wins: distance 1 on arm A, and waiting for
  // the smaller count also completes t2 on arm B, where it has distance 3.
  tracker.waitFor(t[2]);
  EXPECT_EQ(waits(), (Waits{"s_waitcnt vmcnt(2)", "s_waitcnt vmcnt(5)",
                            "s_waitcnt vmcnt(1)"}));
}

TEST_F(WaitTrackerTest, ReferenceRingLoop) {
  constexpr int slotCount = 16;
  Waits expected;
  std::vector<WaitToken> scales = vectorLoads(4);
  std::vector<WaitToken> slots = vectorLoads(slotCount);
  for (WaitToken scale : scales)
    tracker.waitFor(scale);
  for (int count = 19; count >= 16; --count)
    expected.push_back("s_waitcnt vmcnt(" + std::to_string(count) + ")");

  WaitTracker::State entry = tracker.snapshot();
  tracker.beginLoop();
  for (int chunk = 0; chunk < slotCount / 2; ++chunk) {
    tracker.waitFor(slots[2 * chunk]);
    tracker.waitFor(slots[2 * chunk + 1]);
    slots[2 * chunk] = vectorLoad();
    slots[2 * chunk + 1] = vectorLoad();
    expected.push_back("s_waitcnt vmcnt(15)");
    expected.push_back("s_waitcnt vmcnt(14)");
  }
  EXPECT_NO_THROW(tracker.endLoop());
  tracker.join(entry);

  // The first remainder arm consumes four chunks and the other path skips it.
  WaitTracker::State afterLoop = tracker.snapshot();
  for (int slot = 0; slot < 8; ++slot) {
    tracker.waitFor(slots[slot]);
    expected.push_back("s_waitcnt vmcnt(" + std::to_string(15 - slot) + ")");
  }
  WaitTracker::State arm1 = tracker.snapshot();
  tracker.restore(afterLoop);
  tracker.join(arm1);

  for (int slot = 0; slot < slotCount; ++slot) {
    tracker.waitFor(slots[slot]);
    expected.push_back("s_waitcnt vmcnt(" + std::to_string(15 - slot) + ")");
  }
  EXPECT_EQ(waits(), expected);
}

TEST_F(WaitTrackerTest, RejectsLoopThatGrowsTheWindow) {
  std::vector<WaitToken> t = vectorLoads(16);
  tracker.beginLoop();
  // Waiting for t2 completes three positions, so four loads grow the window.
  tracker.waitFor(t[2]);
  vectorLoads(4);
  EXPECT_THROW(tracker.endLoop(), std::logic_error);
}

TEST_F(WaitTrackerTest, RejectsLoopThatShrinksTheWindow) {
  std::vector<WaitToken> t = vectorLoads(16);
  tracker.beginLoop();
  tracker.waitFor(t[2]);
  vectorLoad();
  EXPECT_THROW(tracker.endLoop(), std::logic_error);
}

TEST_F(WaitTrackerTest, RejectsLoopWithOutstandingScalarLoad) {
  tracker.beginLoop();
  scalarLoad();
  EXPECT_THROW(tracker.endLoop(), std::logic_error);
}

TEST_F(WaitTrackerTest, RejectsEndLoopWithoutBeginLoop) {
  EXPECT_THROW(tracker.endLoop(), std::logic_error);
}

TEST_F(WaitTrackerTest, AcceptsLoopThatDrainsWithinTheBody) {
  tracker.beginLoop();
  tracker.waitFor(vectorLoad());
  EXPECT_NO_THROW(tracker.endLoop());
  EXPECT_EQ(waits(), (Waits{"s_waitcnt vmcnt(0)"}));
}

TEST_F(WaitTrackerTest, RejectsInvalidInputs) {
  EXPECT_THROW(tracker.issue(nullptr), std::invalid_argument);
  EXPECT_THROW(tracker.issue(builder.create<rocisa::SNop>(0)),
               std::invalid_argument);
  EXPECT_THROW(tracker.issue(builder.create<rocisa::FlatLoadB32>(
                   vdst.container(), voffset.container())),
               std::invalid_argument);
  EXPECT_THROW(tracker.waitFor(WaitToken{}), std::invalid_argument);
}

} // namespace
