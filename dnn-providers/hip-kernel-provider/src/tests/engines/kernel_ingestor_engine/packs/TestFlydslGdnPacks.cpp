// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormPacks.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslGdnTestGraphs.hpp"

/**
 * @file TestFlydslGdnPacks.cpp
 * @brief A census of what hipkernel:flydsl_gdn actually loads, read from the shard the
 *        census entry names. Belongs to hip_kernel_provider_census_tests; registered with
 *        hkp_register_census_tests() in src/tests/CMakeLists.txt. Every case reads the
 *        staged set, none skips, none touches a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// The kernel's argument list: twelve pointers, 29 int32s and the f32 scale.
constexpr size_t POINTER_SLOTS = 12;
constexpr size_t VALUE_SLOTS = 30;

TEST(TestFlydslGdnPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_GDN.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_GDN.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslGdnPacks, ShipsEveryDtypeHeadDimPairAndStateFormOnce)
{
    const auto& set = loadedSet(FLYDSL_GDN.engineName);

    // (dtype, K, V, state_strided) per kernel; one of each, the vector-state object ranked
    // above the strided one, each V cut into 32-row slices.
    std::set<std::tuple<std::string, int64_t, int64_t, int64_t>> seen;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_GDN_DTYPE_FIELD);
            const auto* keyDim
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_GDN_HEAD_DIM_FIELD);
            const auto* valueDim
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_GDN_HEAD_DIM_V_FIELD);
            const auto* strided
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_GDN_STATE_STRIDED_FIELD);
            const auto* dvSplit
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_GDN_DV_SPLIT_FIELD);
            const auto* blockThreads
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_GDN_BLOCK_THREADS_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(keyDim, nullptr) << kernel.name;
            ASSERT_NE(valueDim, nullptr) << kernel.name;
            ASSERT_NE(strided, nullptr) << kernel.name;
            ASSERT_NE(dvSplit, nullptr) << kernel.name;
            ASSERT_NE(blockThreads, nullptr) << kernel.name;
            EXPECT_EQ(*blockThreads, 256) << kernel.name;
            EXPECT_EQ(*dvSplit * 32, *valueDim) << kernel.name;
            EXPECT_EQ(kernel.priority, *strided == 0 ? 100 : 10) << kernel.name;
            EXPECT_TRUE(seen.emplace(*dtype, *keyDim, *valueDim, *strided).second)
                << kernel.name << " duplicates another object";

            // Only ever kpack: pre-built code objects, no source to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }

    std::set<std::tuple<std::string, int64_t, int64_t, int64_t>> expected;
    for(const std::string dtype : {"bf16", "f16"})
    {
        for(const int64_t keyDim : {64, 128, 256})
        {
            for(const int64_t valueDim : {64, 128, 256})
            {
                expected.emplace(dtype, keyDim, valueDim, 0);
                expected.emplace(dtype, keyDim, valueDim, 1);
            }
        }
    }
    EXPECT_EQ(seen, expected);
}

TEST(TestFlydslGdnPacks, EveryKernelDeclaresTheFortyTwoSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_GDN.engineName);

    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ASSERT_EQ(kernel.source.signature.size(), POINTER_SLOTS + VALUE_SLOTS) << kernel.name;
            for(size_t slot = 0; slot < kernel.source.signature.size(); ++slot)
            {
                const auto& argument = kernel.source.signature[slot];
                const bool pointer = slot < POINTER_SLOTS;
                const auto offset
                    = pointer ? 8 * slot : 8 * POINTER_SLOTS + 4 * (slot - POINTER_SLOTS);
                EXPECT_EQ(argument.kind, pointer ? "global_buffer" : "by_value")
                    << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, pointer ? 8U : 4U) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, offset) << kernel.name << " slot " << slot;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
