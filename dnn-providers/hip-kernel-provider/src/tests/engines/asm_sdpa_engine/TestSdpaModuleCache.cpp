// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/asm_sdpa_engine/plans/SdpaModuleCache.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

namespace asm_sdpa_engine
{
namespace
{

// Each test creates its own SdpaModuleCache instance, so tests are fully
// isolated from each other — no shared static state.
//
// Positive cache-hit tests (module loads successfully and is returned from
// cache) require a GPU and a real .co file.  These tests verify the cache's
// error-path behavior and bookkeeping (size/contains) which work without a GPU.
//
// Each test clears the HIP error state after intentionally failing
// hipModuleLoad so the HipErrorHandler listener doesn't flag them.

TEST(TestSdpaModuleCache, EmptyOnConstruction)
{
    SdpaModuleCache cache;
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestSdpaModuleCache, MakeKeyFormatsCorrectly)
{
    auto key = SdpaModuleCache::makeKey("/path/to/kernel.co", "myFunc");
    EXPECT_EQ(key, "/path/to/kernel.co::myFunc");
}

TEST(TestSdpaModuleCache, NullReturnedForInvalidPath)
{
    SdpaModuleCache cache;
    auto result = cache.getOrLoad("/nonexistent/path/to/kernel.co", "fakeFunction");
    EXPECT_EQ(result, nullptr);
    // Clear HIP error state left by the intentional hipModuleLoad failure
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST(TestSdpaModuleCache, InvalidPathNotCached)
{
    SdpaModuleCache cache;

    // First call with invalid path returns nullptr
    auto first = cache.getOrLoad("/another/invalid/path.co", "fakeKernel");
    EXPECT_EQ(first, nullptr);

    // Second call with same invalid path should also return nullptr (not a cached nullptr)
    auto second = cache.getOrLoad("/another/invalid/path.co", "fakeKernel");
    EXPECT_EQ(second, nullptr);

    // Failed loads must not be cached
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.contains("/another/invalid/path.co", "fakeKernel"));

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST(TestSdpaModuleCache, DifferentInvalidPathsReturnNull)
{
    SdpaModuleCache cache;

    auto a = cache.getOrLoad("/invalid/path/a.co", "funcA");
    auto b = cache.getOrLoad("/invalid/path/b.co", "funcB");
    EXPECT_EQ(a, nullptr);
    EXPECT_EQ(b, nullptr);

    // Neither failed load should be cached
    EXPECT_EQ(cache.size(), 0u);

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST(TestSdpaModuleCache, ContainsReturnsFalseForUnknownKey)
{
    SdpaModuleCache cache;
    EXPECT_FALSE(cache.contains("/does/not/exist.co", "noFunc"));
}

TEST(TestSdpaModuleCache, SeparateInstancesAreIsolated)
{
    SdpaModuleCache cacheA;
    SdpaModuleCache cacheB;

    // Operations on one cache should not affect the other
    cacheA.getOrLoad("/invalid/path.co", "func");
    EXPECT_EQ(cacheA.size(), 0u);
    EXPECT_EQ(cacheB.size(), 0u);

    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

} // namespace
} // namespace asm_sdpa_engine
