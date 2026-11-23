// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "compilation/HiprtcModuleCache.hpp"
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <gtest/gtest.h>

namespace hip_kernel_provider::compilation
{
namespace
{

// Each test creates its own HiprtcModuleCache instance, so tests are fully
// isolated from each other without shared static state.
//
// Negative tests verify the cache's error-path behavior and bookkeeping
// (size/contains), which work without a GPU.
TEST(TestHiprtcModuleCache, EmptyOnConstruction)
{
    const HiprtcModuleCache cache;
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestHiprtcModuleCache, MakeKeyFormatsCorrectly)
{
    auto key = HiprtcModuleCache::makeKey("/path/to/module.cpp", {"-O3"});
    EXPECT_EQ(key, "/path/to/module.cpp::-O3");
}

TEST(TestHiprtcModuleCache, NullReturnedForInvalidPath)
{
    HiprtcModuleCache cache;
    auto result = cache.getOrLoad("/path/to/nonexistant.cpp", {"-03"});
    EXPECT_EQ(result, nullptr);
}

TEST(TestHiprtcModuleCache, InvalidPathNotCached)
{
    HiprtcModuleCache cache;

    // First call with invalid path returns nullptr
    auto first = cache.getOrLoad("/path/to/nonexistant.cpp", {"-03"});
    EXPECT_EQ(first, nullptr);

    // Second call with same invalid path should also return nullptr (not a cached nullptr)
    auto second = cache.getOrLoad("/path/to/nonexistant.cpp", {"-03"});
    EXPECT_EQ(second, nullptr);

    // Failed loads must not be cached
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.contains("/path/to/nonexistant.cpp", {"-03"}));
}

TEST(TestHiprtcModuleCache, DifferentInvalidPathsReturnNull)
{
    HiprtcModuleCache cache;

    auto a = cache.getOrLoad("/path/to/nonexistant/a.cpp", {"-00"});
    auto b = cache.getOrLoad("/path/to/nonexistant/b.cpp", {"-03"});
    EXPECT_EQ(a, nullptr);
    EXPECT_EQ(b, nullptr);

    // Neither failed load should be cached
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestHiprtcModuleCache, ContainsReturnsFalseForUnknownKey)
{
    const HiprtcModuleCache cache;
    EXPECT_FALSE(cache.contains("/path/to/nonexistant.cpp", {"-03"}));
}

// Positive cache-hit tests (module loads successfully and is returned from
// cache) require a GPU and a real HIP .cpp file.
TEST(TestHiprtcModuleCache, ValidEntries)
{
    SKIP_IF_NO_DEVICES();

    HiprtcModuleCache cache;
    auto result = cache.getOrLoad("vector_add.cpp", {"-O3", "-DFLOAT=float"});
    EXPECT_NE(result, nullptr);

    EXPECT_TRUE(cache.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}));
    EXPECT_EQ(cache.size(), 1u);

    result = cache.getOrLoad("vector_add.cpp", {"-DFLOAT=float"});
    EXPECT_NE(result, nullptr);

    EXPECT_TRUE(cache.contains("vector_add.cpp", {"-DFLOAT=float"}));
    EXPECT_EQ(cache.size(), 2u);
}

TEST(TestHiprtcModuleCache, SeparateInstancesAreIsolated)
{
    SKIP_IF_NO_DEVICES();

    HiprtcModuleCache cacheA;
    HiprtcModuleCache cacheB;

    // Operations on one cache should not affect the other
    auto result = cacheA.getOrLoad("vector_add.cpp", {"-O3", "-DFLOAT=float"});
    EXPECT_NE(result, nullptr);

    EXPECT_EQ(cacheA.size(), 1u);
    EXPECT_EQ(cacheB.size(), 0u);

    result = cacheB.getOrLoad("vector_add.cpp", {"-DFLOAT=half"});
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(cacheA.size(), 1u);
    EXPECT_EQ(cacheB.size(), 1u);

    EXPECT_TRUE(cacheA.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}));
    EXPECT_FALSE(cacheB.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}));
    EXPECT_FALSE(cacheA.contains("vector_add.cpp", {"-DFLOAT=half"}));
    EXPECT_TRUE(cacheB.contains("vector_add.cpp", {"-DFLOAT=half"}));
}

} // namespace
} // namespace hip_kernel_provider::compilation
