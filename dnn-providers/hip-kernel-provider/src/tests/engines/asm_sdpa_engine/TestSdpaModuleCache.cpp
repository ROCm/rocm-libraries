// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/asm_sdpa_engine/plans/SdpaKernelUtils.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

namespace asm_sdpa_engine
{
namespace
{

// loadOrGetCachedModule() requires hipModuleLoad to succeed for cache-hit
// tests, which needs a GPU and a real .co file.  These tests verify the
// cache's error-path behavior (invalid paths) which works without a GPU.
// Full cache-hit validation is covered by the existing integration tests
// (TestSdpaFwdPlanBuilder, TestSdpaBwdPlanBuilder).
//
// Each test clears the HIP error state after intentionally failing
// hipModuleLoad so the HipErrorHandler listener doesn't flag them.

TEST(TestSdpaModuleCache, NullReturnedForInvalidPath)
{
    auto result = loadOrGetCachedModule("/nonexistent/path/to/kernel.co", "fakeFunction");
    EXPECT_EQ(result, nullptr);
    // Clear HIP error state left by the intentional hipModuleLoad failure
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST(TestSdpaModuleCache, InvalidPathNotCached)
{
    // First call with invalid path returns nullptr
    auto first = loadOrGetCachedModule("/another/invalid/path.co", "fakeKernel");
    EXPECT_EQ(first, nullptr);

    // Second call with same invalid path should also return nullptr (not a cached nullptr)
    auto second = loadOrGetCachedModule("/another/invalid/path.co", "fakeKernel");
    EXPECT_EQ(second, nullptr);

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST(TestSdpaModuleCache, DifferentInvalidPathsReturnNull)
{
    auto a = loadOrGetCachedModule("/invalid/path/a.co", "funcA");
    auto b = loadOrGetCachedModule("/invalid/path/b.co", "funcB");
    EXPECT_EQ(a, nullptr);
    EXPECT_EQ(b, nullptr);

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

} // namespace
} // namespace asm_sdpa_engine
