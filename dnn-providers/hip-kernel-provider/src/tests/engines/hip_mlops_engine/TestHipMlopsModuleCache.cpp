// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/hip_mlops_engine/HipMlopsModuleCache.hpp"
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <string>
#include <vector>

namespace hip_kernel_provider
{
namespace
{

// Each test creates its own HipMlopsModuleCache instance, so tests are fully
// isolated from each other without shared static state.
//
// Negative tests verify the cache's error-path behavior and bookkeeping
// (size/contains), which work without a GPU.
TEST(TestHipMlopsModuleCache, EmptyOnConstruction)
{
    const HipMlopsModuleCache cache;
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestHipMlopsModuleCache, MakeKeyFormatsCorrectly)
{
    auto key = HipMlopsModuleCache::makeKey("/path/to/module.cpp", {"-O3"}, 2);
    EXPECT_EQ(key, "/path/to/module.cpp::-O3@2");
}

// Every device has its own HIP context, so a module loaded on one ordinal must not be
// handed to a dispatch on another of the same architecture.
TEST(TestHipMlopsModuleCache, MakeKeyDistinguishesTwoOrdinalsOfTheSameArch)
{
    const std::vector<std::string> options{"-O3", "--offload-arch=gfx942:sramecc+:xnack-"};
    EXPECT_NE(HipMlopsModuleCache::makeKey("module.cpp", options, 0),
              HipMlopsModuleCache::makeKey("module.cpp", options, 1));
}

// The binary is compiled for an architecture, not a device, so it is keyed without one.
TEST(TestHipMlopsBinaryCache, MakeKeyFormatsCorrectly)
{
    EXPECT_EQ(HipMlopsBinaryCache::makeKey("/path/to/module.cpp", {"-O3", "-DX=1"}),
              "/path/to/module.cpp::-O3::-DX=1");
}

TEST(TestHipMlopsBinaryCache, FailedCompilationIsNotCached)
{
    HipMlopsBinaryCache cache;
    EXPECT_EQ(cache.getOrLoad("/path/to/nonexistant.cpp", {"-03"}), nullptr);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestHipMlopsModuleCache, NullReturnedForInvalidPath)
{
    HipMlopsModuleCache cache;
    auto result = cache.getOrLoad("/path/to/nonexistant.cpp", {"-03"}, 0);
    EXPECT_EQ(result, nullptr);
}

TEST(TestHipMlopsModuleCache, InvalidPathNotCached)
{
    HipMlopsModuleCache cache;

    // First call with invalid path returns nullptr
    auto first = cache.getOrLoad("/path/to/nonexistant.cpp", {"-03"}, 0);
    EXPECT_EQ(first, nullptr);

    // Second call with same invalid path should also return nullptr (not a cached nullptr)
    auto second = cache.getOrLoad("/path/to/nonexistant.cpp", {"-03"}, 0);
    EXPECT_EQ(second, nullptr);

    // Failed loads must not be cached
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.contains("/path/to/nonexistant.cpp", {"-03"}, 0));
}

TEST(TestHipMlopsModuleCache, DifferentInvalidPathsReturnNull)
{
    HipMlopsModuleCache cache;

    auto a = cache.getOrLoad("/path/to/nonexistant/a.cpp", {"-00"}, 0);
    auto b = cache.getOrLoad("/path/to/nonexistant/b.cpp", {"-03"}, 0);
    EXPECT_EQ(a, nullptr);
    EXPECT_EQ(b, nullptr);

    // Neither failed load should be cached
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestHipMlopsModuleCache, ContainsReturnsFalseForUnknownKey)
{
    const HipMlopsModuleCache cache;
    EXPECT_FALSE(cache.contains("/path/to/nonexistant.cpp", {"-03"}, 0));
}

// Positive cache-hit tests (module loads successfully and is returned from
// cache) require a GPU and a real HIP .cpp file.
TEST(TestHipMlopsModuleCache, ValidEntries)
{
    SKIP_IF_NO_DEVICES();

    HipMlopsModuleCache cache;
    auto result = cache.getOrLoad("vector_add.cpp", {"-O3", "-DFLOAT=float"}, 0);
    EXPECT_NE(result, nullptr);

    EXPECT_TRUE(cache.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}, 0));
    EXPECT_EQ(cache.size(), 1u);

    result = cache.getOrLoad("vector_add.cpp", {"-DFLOAT=float"}, 0);
    EXPECT_NE(result, nullptr);

    EXPECT_TRUE(cache.contains("vector_add.cpp", {"-DFLOAT=float"}, 0));
    EXPECT_EQ(cache.size(), 2u);
}

TEST(TestHipMlopsModuleCache, LoadsOnTheRequestedDevice)
{
    SKIP_IF_NO_DEVICES();

    int deviceCount = 0;
    ASSERT_EQ(hipSuccess, hipGetDeviceCount(&deviceCount));
    const int lastDevice = deviceCount - 1;

    HipMlopsModuleCache cache;
    auto program = cache.getOrLoad("vector_add.cpp", {"-O3", "-DFLOAT=float"}, lastDevice);
    ASSERT_NE(program, nullptr);
    EXPECT_EQ(program->deviceOrdinal(), lastDevice);
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_TRUE(cache.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}, lastDevice));
}

TEST(TestHipMlopsModuleCache, SeparateInstancesAreIsolated)
{
    SKIP_IF_NO_DEVICES();

    HipMlopsModuleCache cacheA;
    HipMlopsModuleCache cacheB;

    // Operations on one cache should not affect the other
    auto result = cacheA.getOrLoad("vector_add.cpp", {"-O3", "-DFLOAT=float"}, 0);
    EXPECT_NE(result, nullptr);

    EXPECT_EQ(cacheA.size(), 1u);
    EXPECT_EQ(cacheB.size(), 0u);

    result = cacheB.getOrLoad("vector_add.cpp", {"-DFLOAT=half"}, 0);
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(cacheA.size(), 1u);
    EXPECT_EQ(cacheB.size(), 1u);

    EXPECT_TRUE(cacheA.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}, 0));
    EXPECT_FALSE(cacheB.contains("vector_add.cpp", {"-O3", "-DFLOAT=float"}, 0));
    EXPECT_FALSE(cacheA.contains("vector_add.cpp", {"-DFLOAT=half"}, 0));
    EXPECT_TRUE(cacheB.contains("vector_add.cpp", {"-DFLOAT=half"}, 0));
}

} // namespace
} // namespace hip_kernel_provider
