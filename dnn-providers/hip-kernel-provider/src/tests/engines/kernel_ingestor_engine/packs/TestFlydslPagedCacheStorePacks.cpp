// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormPacks.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslPagedCacheStoreTestGraphs.hpp"

/**
 * @file TestFlydslPagedCacheStorePacks.cpp
 * @brief A census of what hipkernel:flydsl_paged_cache_store loads from the shard the census
 *        entry names. Census binary only: every case reads loadedSet(); none touches a device.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslPagedCacheStorePacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_PAGED_CACHE_STORE.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_PAGED_CACHE_STORE.graphMatcher);
    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

// Every x dtype into itself, on every shard; and into both FP8 formats of one encoding --
// FNUZ (gfx942) or OCP (gfx950) -- or into none (gfx90a). A shard with one encoding's E4M3
// and the other's E5M2 would hold an object built for the wrong hardware conversion.
TEST(TestFlydslPagedCacheStorePacks, ShipsEachXDtypeIntoItselfAndOneFp8Encoding)
{
    const auto& set = loadedSet(FLYDSL_PAGED_CACHE_STORE.engineName);

    std::set<std::string> pairs;
    std::set<std::string> fp8;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_DTYPE_FIELD);
            const auto* cache
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_CACHE_DTYPE_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(cache, nullptr) << kernel.name;
            EXPECT_TRUE(pairs.insert(*dtype + "->" + *cache).second) << kernel.name;
            if(*cache != *dtype)
            {
                fp8.insert(*cache);
            }
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
        }
    }

    for(const std::string dtype : {"bf16", "f16", "f32"})
    {
        EXPECT_EQ(pairs.count(std::string(dtype).append("->").append(dtype)), 1U) << dtype;
        for(const auto& format : fp8)
        {
            EXPECT_EQ(pairs.count(std::string(dtype).append("->").append(format)), 1U)
                << dtype << "->" << format;
        }
    }
    const std::set<std::string> ocp{"fp8_e4m3", "fp8_e5m2"};
    const std::set<std::string> fnuz{"fp8_e4m3_fnuz", "fp8_e5m2_fnuz"};
    EXPECT_TRUE(fp8.empty() || fp8 == ocp || fp8 == fnuz);
    EXPECT_EQ(pairs.size(), 3 + 3 * fp8.size());
}

// A runtime pass-by-value scale raises a graph's schema floor to 1.2.0, which this engine
// declares so launch() can resolve it.
TEST(TestFlydslPagedCacheStorePacks, UnderstandsRuntimeScaleGraphs)
{
    const auto& set = loadedSet(FLYDSL_PAGED_CACHE_STORE.engineName);
    const auto& runtimeScale
        = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(false, true, false, false, false);
    EXPECT_FALSE(set.engine.sdkVersion < runtimeScale) << set.engine.sdkVersion.str();
}

TEST(TestFlydslPagedCacheStorePacks, EveryKernelDeclaresTheTwentyTwoSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_PAGED_CACHE_STORE.engineName);
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            // Four pointers (x, slot_mapping, container, scale), six 64-bit strides, then
            // eleven int32 and the f32 scale: widest first, so no padding.
            const auto& signature = kernel.source.signature;
            ASSERT_EQ(signature.size(), 22U) << kernel.name;
            uint32_t offset = 0;
            for(size_t slot = 0; slot < signature.size(); ++slot)
            {
                const bool pointer = slot < 4;
                const uint32_t size = slot < 10 ? 8 : 4;
                EXPECT_EQ(signature[slot].kind, pointer ? "global_buffer" : "by_value")
                    << kernel.name << " slot " << slot;
                EXPECT_EQ(signature[slot].size, size) << kernel.name << " slot " << slot;
                EXPECT_EQ(signature[slot].offset, offset) << kernel.name << " slot " << slot;
                offset += size;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
