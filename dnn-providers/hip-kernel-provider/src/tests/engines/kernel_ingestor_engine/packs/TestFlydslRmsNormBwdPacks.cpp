// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormPacks.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormBwdTestGraphs.hpp"

/**
 * @file TestFlydslRmsNormBwdPacks.cpp
 * @brief A census of what hipkernel:flydsl_rmsnorm_bwd actually loads, read from the shard
 *        the census entry names. Belongs to the census binary, as the forward's does.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslRmsNormBwdPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_RMSNORM_BWD.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_RMSNORM_BWD.graphMatcher);
    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslRmsNormBwdPacks, ShipsTheGenericTiersPerFormAndBakedWidthsFor16Bit)
{
    const auto& set = loadedSet(FLYDSL_RMSNORM_BWD.engineName);

    // Per form: the generic tiers' n_max values, and how many baked widths.
    std::map<std::string, std::set<int64_t>> genericBounds;
    std::map<std::string, size_t> baked;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_DTYPE_FIELD);
            const auto* n = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_N_FIELD);
            const auto* nMax = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_N_MAX_FIELD);
            const auto* scaleF32
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SCALE_F32_FIELD);
            const auto* reduce
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_REDUCE_SYMBOL_FIELD);
            ASSERT_TRUE(dtype != nullptr && n != nullptr && nMax != nullptr && scaleF32 != nullptr
                        && reduce != nullptr)
                << kernel.name << " lacks a backward metadata field";
            // Every object carries the reduce kernel the dispatch loads by name.
            EXPECT_EQ(*reduce, FLYDSL_BWD_REDUCE_SYMBOL) << kernel.name;
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;

            const auto form = *dtype + (*scaleF32 != 0 ? "+f32" : "");
            if(*n == FLYDSL_GENERIC_N)
            {
                genericBounds[form].insert(*nMax);
            }
            else
            {
                EXPECT_EQ(*nMax, *n) << kernel.name;
                baked[form] += 1;
            }
        }
    }

    const std::set<int64_t> tiers{1024, 8192, 16384};
    for(const std::string form : {"bf16", "f16", "f32", "bf16+f32", "f16+f32"})
    {
        EXPECT_EQ(genericBounds[form], tiers) << form;
        if(form != "f32")
        {
            EXPECT_GT(baked[form], 0U) << form;
        }
    }
    EXPECT_EQ(baked.count("f32"), 0U);
}

TEST(TestFlydslRmsNormBwdPacks, EveryKernelDeclaresThePartialKernelsSignature)
{
    const auto& set = loadedSet(FLYDSL_RMSNORM_BWD.engineName);

    // Seven (pointer, layout) pairs -- Source, Gamma, DY, DResidualOut, Rstd, DX,
    // DWeightPartial; 2-D layouts are 16 bytes, Gamma's and Rstd's 4 -- then the row count.
    static const std::array<std::pair<const char*, uint32_t>, 15> s_slots{{{"global_buffer", 8},
                                                                           {"by_value", 16},
                                                                           {"global_buffer", 8},
                                                                           {"by_value", 4},
                                                                           {"global_buffer", 8},
                                                                           {"by_value", 16},
                                                                           {"global_buffer", 8},
                                                                           {"by_value", 16},
                                                                           {"global_buffer", 8},
                                                                           {"by_value", 4},
                                                                           {"global_buffer", 8},
                                                                           {"by_value", 16},
                                                                           {"global_buffer", 8},
                                                                           {"by_value", 16},
                                                                           {"by_value", 4}}};
    static const std::array<uint32_t, 15> s_offsets{
        0, 8, 24, 32, 40, 48, 64, 72, 88, 96, 104, 112, 128, 136, 152};

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
