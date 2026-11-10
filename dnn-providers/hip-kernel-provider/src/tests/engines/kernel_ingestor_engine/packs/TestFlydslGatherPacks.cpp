// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormPacks.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslGatherTestGraphs.hpp"

/**
 * @file TestFlydslGatherPacks.cpp
 * @brief A census of what hipkernel:flydsl_gather actually loads, read from the shard the
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

TEST(TestFlydslGatherPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_GATHER.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_GATHER.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslGatherPacks, ShipsEveryDtypeIndexDtypeAndAccessWidthOnce)
{
    const auto& set = loadedSet(FLYDSL_GATHER.engineName);

    // (dtype, index dtype, vector) per kernel; one of each, the vector object ranked above
    // the per-element one.
    std::set<std::tuple<std::string, std::string, bool>> seen;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_GATHER_DTYPE_FIELD);
            const auto* indexDtype = tryGetMetadataField<std::string>(
                kernel.metadata, FLYDSL_GATHER_INDEX_DTYPE_FIELD);
            const auto* vecBytes
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_GATHER_VEC_BYTES_FIELD);
            const auto* blockThreads
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_GATHER_BLOCK_THREADS_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(indexDtype, nullptr) << kernel.name;
            ASSERT_NE(vecBytes, nullptr) << kernel.name;
            ASSERT_NE(blockThreads, nullptr) << kernel.name;
            EXPECT_EQ(*blockThreads, 256) << kernel.name;

            const int64_t elementBytes = *dtype == "f32" ? 4 : 2;
            const bool vector = *vecBytes == 16;
            EXPECT_TRUE(vector || *vecBytes == elementBytes) << kernel.name;
            EXPECT_EQ(kernel.priority, vector ? 100 : 10) << kernel.name;
            EXPECT_TRUE(seen.emplace(*dtype, *indexDtype, vector).second)
                << kernel.name << " duplicates another object";

            // Only ever kpack: pre-built code objects, no source to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }

    std::set<std::tuple<std::string, std::string, bool>> expected;
    for(const std::string dtype : {"bf16", "f16", "f32"})
    {
        for(const std::string indexDtype : {"i32", "i64"})
        {
            expected.emplace(dtype, indexDtype, true);
            expected.emplace(dtype, indexDtype, false);
        }
    }
    EXPECT_EQ(seen, expected);
}

TEST(TestFlydslGatherPacks, EveryKernelDeclaresTheSixSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_GATHER.engineName);

    // x, indices and y as raw pointers, then num_indices, axis_extent and inner as int32s.
    static const std::array<std::pair<const char*, uint32_t>, 6> s_slots{{{"global_buffer", 8},
                                                                          {"global_buffer", 8},
                                                                          {"global_buffer", 8},
                                                                          {"by_value", 4},
                                                                          {"by_value", 4},
                                                                          {"by_value", 4}}};
    static const std::array<uint32_t, 6> s_offsets{0, 8, 16, 24, 28, 32};
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ASSERT_EQ(kernel.source.signature.size(), s_slots.size()) << kernel.name;
            for(size_t slot = 0; slot < s_slots.size(); ++slot)
            {
                const auto& argument = kernel.source.signature[slot];
                EXPECT_EQ(argument.kind, s_slots[slot].first) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, s_slots[slot].second) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, s_offsets[slot]) << kernel.name << " slot " << slot;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
