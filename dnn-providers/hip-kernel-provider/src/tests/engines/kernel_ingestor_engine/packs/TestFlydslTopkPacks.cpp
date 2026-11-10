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

#include "engines/kernel_ingestor_engine/packs/FlydslTopkTestGraphs.hpp"

/**
 * @file TestFlydslTopkPacks.cpp
 * @brief A census of what hipkernel:flydsl_topk actually loads, read from the shard the
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

TEST(TestFlydslTopkPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_TOPK.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_TOPK.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslTopkPacks, ShipsTheReductionAndTheRadixSelectForEveryDtypeAndIndexDtype)
{
    const auto& set = loadedSet(FLYDSL_TOPK.engineName);

    // (dtype, index dtype, k_max) per kernel; one of each, the k = 1 reduction (1024
    // threads) ranked above the radix select (256 threads).
    std::set<std::tuple<std::string, std::string, int64_t>> seen;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_TOPK_DTYPE_FIELD);
            const auto* indexDtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_TOPK_INDEX_DTYPE_FIELD);
            const auto* kMax
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_TOPK_K_MAX_FIELD);
            const auto* blockThreads
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_TOPK_BLOCK_THREADS_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(indexDtype, nullptr) << kernel.name;
            ASSERT_NE(kMax, nullptr) << kernel.name;
            ASSERT_NE(blockThreads, nullptr) << kernel.name;

            const bool reduction = *kMax == 1;
            EXPECT_TRUE(reduction || *kMax == FLYDSL_TOPK_GENERAL_K_MAX) << kernel.name;
            EXPECT_EQ(*blockThreads, reduction ? 1024 : 256) << kernel.name;
            EXPECT_EQ(kernel.priority, reduction ? 100 : 10) << kernel.name;
            EXPECT_TRUE(seen.emplace(*dtype, *indexDtype, *kMax).second)
                << kernel.name << " duplicates another object";

            // Only ever kpack: pre-built code objects, no source to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }

    std::set<std::tuple<std::string, std::string, int64_t>> expected;
    for(const std::string dtype : {"bf16", "f16", "f32"})
    {
        for(const std::string indexDtype : {"i32", "i64"})
        {
            expected.emplace(dtype, indexDtype, 1);
            expected.emplace(dtype, indexDtype, FLYDSL_TOPK_GENERAL_K_MAX);
        }
    }
    EXPECT_EQ(seen, expected);
}

TEST(TestFlydslTopkPacks, EveryKernelDeclaresTheTenSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_TOPK.engineName);

    // x, indices and values, each a pointer and a 16-byte 2-D descriptor, then k, largest,
    // write_values and vec_ok as int32s.
    static const std::array<std::pair<const char*, uint32_t>, 10> s_slots{{{"global_buffer", 8},
                                                                           {"by_value", 16},
                                                                           {"global_buffer", 8},
                                                                           {"by_value", 16},
                                                                           {"global_buffer", 8},
                                                                           {"by_value", 16},
                                                                           {"by_value", 4},
                                                                           {"by_value", 4},
                                                                           {"by_value", 4},
                                                                           {"by_value", 4}}};
    static const std::array<uint32_t, 10> s_offsets{0, 8, 24, 32, 48, 56, 72, 76, 80, 84};
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
