// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Profiler.h"
#include <chrono>
#include <gtest/gtest.h>

using namespace raceemulator;

namespace {
void busyWait() {
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start <
         std::chrono::microseconds(1)) {
  }
}
} // namespace

TEST(ProfilerTest, DisabledByDefault) {
  Profiler profiler;
  EXPECT_FALSE(profiler.isEnabled());
  { auto sw = profiler.scopedStopwatch("should_not_record"); }
  EXPECT_DOUBLE_EQ(profiler.getAccountedTime(), 0.0);
}

TEST(ProfilerTest, AccumulatesTimeAndCount) {
  Profiler profiler;
  profiler.setEnabled(true);
  for (int i = 0; i < 3; ++i) {
    auto sw = profiler.scopedStopwatch("work");
    busyWait();
  }
  EXPECT_GE(profiler.getAccountedTime(), 0.0);
  // Verify report contains the key and count.
  std::string report = profiler.reportStr(0.0);
  EXPECT_NE(report.find("work"), std::string::npos);
  EXPECT_NE(report.find("3"), std::string::npos);
}

TEST(ProfilerTest, MultipleKeysTrackedIndependently) {
  Profiler profiler;
  profiler.setEnabled(true);
  { auto sw = profiler.scopedStopwatch("alpha"); }
  { auto sw = profiler.scopedStopwatch("alpha"); }
  { auto sw = profiler.scopedStopwatch("beta"); }

  std::string report = profiler.reportStr(0.0);
  EXPECT_NE(report.find("alpha"), std::string::npos);
  EXPECT_NE(report.find("beta"), std::string::npos);
}

TEST(ProfilerTest, LazyKeyNotCalledWhenDisabled) {
  Profiler profiler;
  bool keyCalled = false;
  {
    auto sw = profiler.scopedStopwatch([&]() -> std::string_view {
      keyCalled = true;
      return "lazy_key";
    });
  }
  EXPECT_FALSE(keyCalled);
}

TEST(ProfilerTest, LazyKeyCalledWhenEnabled) {
  Profiler profiler;
  profiler.setEnabled(true);
  bool keyCalled = false;
  {
    auto sw = profiler.scopedStopwatch([&]() -> std::string_view {
      keyCalled = true;
      return "lazy_key";
    });
  }
  EXPECT_TRUE(keyCalled);
  EXPECT_GE(profiler.getAccountedTime(), 0.0);
}

TEST(ProfilerTest, Reset) {
  Profiler profiler;
  profiler.setEnabled(true);
  { auto sw = profiler.scopedStopwatch("x"); }
  EXPECT_GE(profiler.getAccountedTime(), 0.0);
  profiler.reset();
  EXPECT_DOUBLE_EQ(profiler.getAccountedTime(), 0.0);
}

TEST(ProfilerTest, ReportHeaders) {
  Profiler profiler;
  profiler.setEnabled(true);
  { auto sw = profiler.scopedStopwatch("test_scope"); }
  std::string report = profiler.reportStr(0.0);
  EXPECT_NE(report.find("Scope"), std::string::npos);
  EXPECT_NE(report.find("Time [s]"), std::string::npos);
  EXPECT_NE(report.find("Count"), std::string::npos);
  EXPECT_NE(report.find("Avg [us]"), std::string::npos);
  EXPECT_NE(report.find("Total"), std::string::npos);
}

TEST(ProfilerTest, EmptyReport) {
  Profiler profiler;
  std::string report = profiler.reportStr();
  EXPECT_NE(report.find("no profiling data"), std::string::npos);
}

TEST(ProfilerTest, NestedScopesPartitionTime) {
  Profiler profiler;
  profiler.setEnabled(true);
  {
    auto outer = profiler.scopedStopwatch("nest_outer");
    busyWait();
    {
      auto inner = profiler.scopedStopwatch("nest_inner");
      busyWait();
    }
    busyWait();
  }
  std::string report = profiler.reportStr(0.0);
  EXPECT_NE(report.find("nest_outer"), std::string::npos);
  EXPECT_NE(report.find("nest_inner"), std::string::npos);
  EXPECT_LE(profiler.getAccountedTime(), profiler.getWallTime());
}

TEST(ProfilerTest, NonLIFODestructionWorks) {
  // Simulates: create "first_scope", create "second_scope",
  // destroy "first_scope" (middle), destroy "second_scope".
  // Trace: [] -> ["first_scope"] -> ["first_scope","second_scope"]
  //        -> ["second_scope"] -> []
  Profiler profiler;
  profiler.setEnabled(true);
  {
    Profiler::ScopedStopwatch stashed;
    {
      auto sw1 = profiler.scopedStopwatch("nonlifo_first");
      busyWait();
      stashed = profiler.scopedStopwatch("nonlifo_second");
      // sw1 is destroyed here (middle removal), "second" continues running.
    }
    // "second" is still active via stashed.
    busyWait();
    // stashed ("second") destroyed here.
  }
  std::string report = profiler.reportStr(0.0);
  EXPECT_NE(report.find("nonlifo_first"), std::string::npos);
  EXPECT_NE(report.find("nonlifo_second"), std::string::npos);
  EXPECT_LE(profiler.getAccountedTime(), profiler.getWallTime());
}

TEST(ProfilerTest, DeeplyNestedScopes) {
  Profiler profiler;
  profiler.setEnabled(true);
  {
    auto s1 = profiler.scopedStopwatch("deep_level1");
    {
      auto s2 = profiler.scopedStopwatch("deep_level2");
      {
        auto s3 = profiler.scopedStopwatch("deep_level3");
        busyWait();
      }
    }
  }
  std::string report = profiler.reportStr(0.0);
  EXPECT_NE(report.find("deep_level1"), std::string::npos);
  EXPECT_NE(report.find("deep_level2"), std::string::npos);
  EXPECT_NE(report.find("deep_level3"), std::string::npos);
  EXPECT_LE(profiler.getAccountedTime(), profiler.getWallTime());
}

TEST(ProfilerTest, DepthTracking) {
  Profiler profiler;
  profiler.setEnabled(true);
  {
    auto s0 = profiler.scopedStopwatch("top");
    {
      auto s1 = profiler.scopedStopwatch("middle");
      {
        auto s2 = profiler.scopedStopwatch("bottom");
        busyWait();
      }
    }
  }
  // A sequential (non-nested) scope at the top level.
  { auto s = profiler.scopedStopwatch("also_top"); }

  // Created at depth 1 but moved out so it outlives its parent.
  {
    Profiler::ScopedStopwatch stashed;
    {
      auto outer = profiler.scopedStopwatch("outer_d0");
      stashed = profiler.scopedStopwatch("moved_d1"); // created at depth 1
      // outer destroyed here (middle removal)
    }
    // stashed ("moved_d1") is now the only active scope.
    busyWait();
    // stashed destroyed here — last to go, but depth is still 1.
  }

  EXPECT_EQ(profiler.getDepth("top"), 0);
  EXPECT_EQ(profiler.getDepth("middle"), 1);
  EXPECT_EQ(profiler.getDepth("bottom"), 2);
  EXPECT_EQ(profiler.getDepth("also_top"), 0);
  EXPECT_EQ(profiler.getDepth("outer_d0"), 0);
  EXPECT_EQ(profiler.getDepth("moved_d1"), 1);
  EXPECT_EQ(profiler.getDepth("nonexistent"), -1);
}

TEST(ProfilerTest, SequentialScopesDoNotThrow) {
  Profiler profiler;
  profiler.setEnabled(true);
  { auto sw = profiler.scopedStopwatch("first"); }
  { auto sw = profiler.scopedStopwatch("second"); }
  EXPECT_GE(profiler.getAccountedTime(), 0.0);
}

TEST(ProfilerTest, NoOpStopwatchMoveSemantics) {
  Profiler::ScopedStopwatch sw1;
  Profiler::ScopedStopwatch sw2 = std::move(sw1);
  // Both should be safe to destroy.
}
