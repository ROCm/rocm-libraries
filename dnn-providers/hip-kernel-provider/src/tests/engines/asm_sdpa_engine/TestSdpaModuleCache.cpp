// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/asm_sdpa_engine/plans/SdpaModuleCache.hpp"
#include "tests/engines/asm_sdpa_engine/SdpaStagedArchive.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <string>

namespace asm_sdpa_engine
{
namespace
{

// Each test creates its own SdpaModuleCache instance, so tests are fully
// isolated from each other — no shared static state.
//
// Positive cache-hit tests (module loads successfully and is returned from
// cache) require a GPU and a real .kpack archive.  These tests verify the
// cache's error-path behavior and bookkeeping (size/contains) which work
// without a GPU.
//
// With kpack loading, invalid arch/tocKey combinations throw
// HipdnnPluginException rather than returning nullptr (kpack_open fails
// for a non-existent archive path).

TEST(TestSdpaModuleCache, EmptyOnConstruction)
{
    const SdpaModuleCache cache;
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestSdpaModuleCache, MakeKeyFormatsCorrectly)
{
    auto key = SdpaModuleCache::makeKey(
        "fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co", "gfx942", 0, "myFunc");
    EXPECT_EQ(key, "gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co@0::myFunc");
}

TEST(TestSdpaModuleCache, MakeKeyIncludesArchAndTocKey)
{
    auto key
        = SdpaModuleCache::makeKey("fmha_v3_bwd/bwd_hd128_odo_bf16.co", "gfx950", 3, "kernel_func");
    EXPECT_EQ(key, "gfx950/fmha_v3_bwd/bwd_hd128_odo_bf16.co@3::kernel_func");
}

// Every device has its own HIP context, so a module loaded on one ordinal must not be
// handed to a dispatch on another of the same architecture.
TEST(TestSdpaModuleCache, MakeKeyDistinguishesTwoOrdinalsOfTheSameArch)
{
    const std::string tocKey = "fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co";
    EXPECT_NE(SdpaModuleCache::makeKey(tocKey, "gfx942", 0, "myFunc"),
              SdpaModuleCache::makeKey(tocKey, "gfx942", 1, "myFunc"));
}

TEST(TestSdpaModuleCache, LoadThrowsForInvalidArch)
{
    SdpaModuleCache cache;
    // Invalid arch causes kpack_open to fail (no .kpack file for this arch)
    EXPECT_THROW(cache.getOrLoad("some/tocKey.co", "gfx_bogus_999", 0, "fakeFunction"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaModuleCache, ContainsReturnsFalseForUnknownKey)
{
    const SdpaModuleCache cache;
    EXPECT_FALSE(cache.contains("does/not/exist.co", "gfx942", 0, "noFunc"));
}

TEST(TestSdpaModuleCache, SeparateInstancesAreIsolated)
{
    SdpaModuleCache cacheA;
    const SdpaModuleCache cacheB;

    // Operations on one cache should not affect the other
    EXPECT_THROW(cacheA.getOrLoad("invalid.co", "gfx_bogus", 0, "func"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_EQ(cacheA.size(), 0u);
    EXPECT_EQ(cacheB.size(), 0u);
}

// The ODO kernel, which both supported architectures ship under one toc key.
constexpr const char* K_ODO_TOC_KEY = "fmha_v3_bwd/bwd_hd128_odo_bf16.co";
constexpr const char* K_ODO_SYMBOL = "_ZN5aiter23fmha_bwd_hd128_odo_bf16E";

// Device 0's feature-stripped architecture when its SDPA archive is staged, else empty.
std::string supportedArchOfDeviceZero()
{
    hipDeviceProp_t properties{};
    if(hipGetDeviceProperties(&properties, 0) != hipSuccess)
    {
        return {};
    }
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    return (arch == "gfx942" || arch == "gfx950") && sdpaArchiveStaged(arch) ? arch : std::string{};
}

TEST(TestGpuSdpaModuleCache, LoadsOnTheRequestedOrdinalAndRestoresTheCaller)
{
    SKIP_IF_NO_DEVICES();
    const std::string arch = supportedArchOfDeviceZero();
    if(arch.empty())
    {
        GTEST_SKIP()
            << "device 0 is not an architecture the SDPA archives cover, or its archive is not "
               "staged beside this executable";
    }

    int before = -1;
    ASSERT_EQ(hipSuccess, hipGetDevice(&before));

    SdpaModuleCache cache;
    const auto module = cache.getOrLoad(K_ODO_TOC_KEY, arch, 0, K_ODO_SYMBOL);
    ASSERT_NE(module, nullptr);
    EXPECT_EQ(module->deviceOrdinal(), 0);
    EXPECT_NE(module->function(), nullptr);

    int after = -1;
    ASSERT_EQ(hipSuccess, hipGetDevice(&after));
    EXPECT_EQ(before, after);
}

TEST(TestGpuSdpaModuleCache, ASecondOrdinalOfTheSameArchLoadsItsOwnModule)
{
    SKIP_IF_NO_DEVICES();
    const std::string arch = supportedArchOfDeviceZero();
    if(arch.empty())
    {
        GTEST_SKIP()
            << "device 0 is not an architecture the SDPA archives cover, or its archive is not "
               "staged beside this executable";
    }

    int devices = 0;
    ASSERT_EQ(hipSuccess, hipGetDeviceCount(&devices));
    int peer = -1;
    for(int ordinal = 1; ordinal < devices && peer < 0; ++ordinal)
    {
        hipDeviceProp_t properties{};
        if(hipGetDeviceProperties(&properties, ordinal) == hipSuccess
           && hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName) == arch)
        {
            peer = ordinal;
        }
    }
    if(peer < 0)
    {
        GTEST_SKIP() << "needs a second device of architecture " << arch;
    }

    SdpaModuleCache cache;
    const auto first = cache.getOrLoad(K_ODO_TOC_KEY, arch, 0, K_ODO_SYMBOL);
    const auto second = cache.getOrLoad(K_ODO_TOC_KEY, arch, peer, K_ODO_SYMBOL);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    // Not the first device's module handed to the second: each lives in its own device's
    // context.
    EXPECT_NE(first, second);
    EXPECT_EQ(first->deviceOrdinal(), 0);
    EXPECT_EQ(second->deviceOrdinal(), peer);
    EXPECT_EQ(cache.size(), 2u);
}

} // namespace
} // namespace asm_sdpa_engine
