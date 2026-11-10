// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormPacks.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslMoeRoutingTestGraphs.hpp"

/**
 * @file TestFlydslMoeRoutingPacks.cpp
 * @brief A census of what hipkernel:flydsl_moe_routing actually loads, read from the shard
 *        the census entry names. Belongs to hip_kernel_provider_census_tests; registered
 *        with hkp_register_census_tests() in src/tests/CMakeLists.txt. Every case reads the
 *        staged set, none skips, none touches a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslMoeRoutingPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_MOE_ROUTING.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_MOE_ROUTING.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslMoeRoutingPacks, ShipsOneInt32ObjectNamingItsPrefixAndScatterKernels)
{
    const auto& set = loadedSet(FLYDSL_MOE_ROUTING.engineName);

    size_t objects = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++objects;
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_MOE_ROUTING_DTYPE_FIELD);
            const auto* eMax
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_MOE_ROUTING_E_MAX_FIELD);
            const auto* prefix = tryGetMetadataField<std::string>(
                kernel.metadata, FLYDSL_MOE_ROUTING_PREFIX_SYMBOL_FIELD);
            const auto* scatter = tryGetMetadataField<std::string>(
                kernel.metadata, FLYDSL_MOE_ROUTING_SCATTER_SYMBOL_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(eMax, nullptr) << kernel.name;
            ASSERT_NE(prefix, nullptr) << kernel.name;
            ASSERT_NE(scatter, nullptr) << kernel.name;
            EXPECT_EQ(*dtype, "i32") << kernel.name;
            EXPECT_EQ(*eMax, 8192) << kernel.name;
            EXPECT_EQ(*prefix, "moe_routing_prefix_kernel_1") << kernel.name;
            EXPECT_EQ(*scatter, "moe_routing_scatter_kernel_2") << kernel.name;
            EXPECT_EQ(kernel.priority, 100) << kernel.name;

            // Only ever kpack: pre-built code objects, no source to fall back to. The UKD
            // names the histogram kernel; the other two are loaded by symbol.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_EQ(kernel.source.symbol, "moe_routing_histogram_kernel_0") << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }
    EXPECT_EQ(objects, 1U);
}

TEST(TestFlydslMoeRoutingPacks, EveryKernelDeclaresTheHistogramKernelsSignature)
{
    const auto& set = loadedSet(FLYDSL_MOE_ROUTING.engineName);

    // TopkIds and Counts, each a pointer and its 4-byte 1-D layout, then n_pairs,
    // num_experts, span and num_blocks.
    static const std::array<std::pair<const char*, uint32_t>, 8> s_slots{{{"global_buffer", 8},
                                                                          {"by_value", 4},
                                                                          {"global_buffer", 8},
                                                                          {"by_value", 4},
                                                                          {"by_value", 4},
                                                                          {"by_value", 4},
                                                                          {"by_value", 4},
                                                                          {"by_value", 4}}};
    static const std::array<uint32_t, 8> s_offsets{0, 8, 16, 24, 28, 32, 36, 40};
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
