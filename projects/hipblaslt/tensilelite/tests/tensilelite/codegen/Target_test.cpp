// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/Target.hpp>

#include "base.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using rocm::tensilelite::codegen::Target;

TEST(Target, RejectsUnsupportedArch) {
  EXPECT_THROW(Target::get("gfx942"), std::invalid_argument);
  EXPECT_THROW(Target::get(""), std::invalid_argument);
}

TEST(Target, Gfx950Properties) {
  Target target = Target::get("gfx950");
  EXPECT_EQ(target.isaVersion(), (IsaVersion{9, 5, 0}));
  EXPECT_EQ(target.wavefrontSize(), 64);
  EXPECT_EQ(target.archName(), "gfx950");
  EXPECT_EQ(target.isaName(), "amdgcn-amd-amdhsa--gfx950");
  EXPECT_EQ(target.maxVgpr(), 256);
  EXPECT_EQ(target.maxSgpr(), 102);
  EXPECT_EQ(target.maxAgpr(), 256);
  EXPECT_EQ(target.maxUnifiedVgpr(), 512);
  EXPECT_EQ(target.maxLdsSize(), 163840);
  EXPECT_EQ(target.maxVmcnt(), 63);
  EXPECT_EQ(target.maxLgkmcnt(), 15);
  EXPECT_EQ(target.xcdCount(), 8);
}

TEST(Target, SelectsCurrentThreadKernel) {
  Target::get("gfx950").selectForCurrentThread();
  rocisa::KernelInfo kernel = rocisa::rocIsa::getInstance().getKernel();
  EXPECT_EQ(kernel.isaVersion, (IsaVersion{9, 5, 0}));
  EXPECT_EQ(kernel.wavefront, 64);
}

TEST(Target, ConcurrentGetIsSafe) {
  constexpr int numThreads = 4;
  std::vector<int> maxVgprs(numThreads, 0);
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i)
    threads.emplace_back(
        [&maxVgprs, i] { maxVgprs[i] = Target::get("gfx950").maxVgpr(); });
  for (std::thread &thread : threads)
    thread.join();
  for (int maxVgpr : maxVgprs)
    EXPECT_EQ(maxVgpr, 256);
}

} // namespace
