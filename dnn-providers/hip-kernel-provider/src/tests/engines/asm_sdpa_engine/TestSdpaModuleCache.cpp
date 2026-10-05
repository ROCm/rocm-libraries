// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/asm_sdpa_engine/plans/SdpaModuleCache.hpp"

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

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
    auto key
        = SdpaModuleCache::makeKey("fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co", "gfx942", "myFunc");
    EXPECT_EQ(key, "gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co::myFunc");
}

TEST(TestSdpaModuleCache, MakeKeyIncludesArchAndTocKey)
{
    auto key
        = SdpaModuleCache::makeKey("fmha_v3_bwd/bwd_hd128_odo_bf16.co", "gfx950", "kernel_func");
    EXPECT_EQ(key, "gfx950/fmha_v3_bwd/bwd_hd128_odo_bf16.co::kernel_func");
}

TEST(TestSdpaModuleCache, LoadThrowsForInvalidArch)
{
    SdpaModuleCache cache;
    // Invalid arch causes kpack_open to fail (no .kpack file for this arch)
    EXPECT_THROW(cache.getOrLoad("some/tocKey.co", "gfx_bogus_999", "fakeFunction"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaModuleCache, ContainsReturnsFalseForUnknownKey)
{
    const SdpaModuleCache cache;
    EXPECT_FALSE(cache.contains("does/not/exist.co", "gfx942", "noFunc"));
}

TEST(TestSdpaModuleCache, SeparateInstancesAreIsolated)
{
    SdpaModuleCache cacheA;
    const SdpaModuleCache cacheB;

    // Operations on one cache should not affect the other
    EXPECT_THROW(cacheA.getOrLoad("invalid.co", "gfx_bogus", "func"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_EQ(cacheA.size(), 0u);
    EXPECT_EQ(cacheB.size(), 0u);
}

} // namespace
} // namespace asm_sdpa_engine
