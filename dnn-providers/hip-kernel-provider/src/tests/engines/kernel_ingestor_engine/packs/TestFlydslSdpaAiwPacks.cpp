// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaAiwPacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa_aiw (AOTriton's FlyDSL gfx1201 kernel's engine)
 *        actually loads, read from the shard the census entry names.
 *
 * Belongs to the census binary for the reason TestFlydslSdpaRdnaPacks.cpp does: every case
 * opens by reading loadedSet(), none skips, and none touches a device. Registered with
 * hkp_register_census_tests() in src/tests/CMakeLists.txt, over the gfx12 shards only.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslSdpaAiwPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_SDPA_AIW.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_SDPA_AIW.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslSdpaAiwPacks, ShipsEveryVariantOnce)
{
    const auto& set = loadedSet(FLYDSL_SDPA_AIW.engineName);

    // dtype x head_dim x head_dim_v x masked x has_bias x bias_f32 x head_dim_max x has_sink
    // is the whole instance key; the schedule fields beside it are what the kernel's tuning
    // table chose for each.
    std::set<std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>>
        classes;
    size_t kernels = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto field = [&kernel](const char* name) {
                const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, name);
                EXPECT_NE(value, nullptr) << kernel.name << " has no " << name;
                return value == nullptr ? int64_t{-1} : *value;
            };
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_SDPA_RDNA_DTYPE_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name;
            classes.emplace(*dtype,
                            field(FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD),
                            field(FLYDSL_SDPA_RDNA_HEAD_DIM_V_FIELD),
                            field(FLYDSL_SDPA_RDNA_CAUSAL_FIELD),
                            field(FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD),
                            field(FLYDSL_SDPA_AIW_BIAS_F32_FIELD),
                            field(FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD),
                            field(FLYDSL_SDPA_RDNA_HAS_SINK_FIELD));
            EXPECT_GT(field(FLYDSL_SDPA_RDNA_BLOCK_M_FIELD), 0) << kernel.name;
            EXPECT_GT(field(FLYDSL_SDPA_AIW_BLOCK_THREADS_FIELD), 0) << kernel.name;
            ++kernels;

            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }

    EXPECT_EQ(kernels, classes.size()) << "two objects claim one class";
    for(const char* dtype : {"bf16", "f16"})
    {
        for(const int64_t masked : {0, 1})
        {
            for(const int64_t headDim : {64, 128, 256})
            {
                for(const auto& [hasBias, f32] :
                    {std::pair<int64_t, int64_t>{0, 0}, {1, 0}, {1, 1}})
                {
                    EXPECT_EQ(
                        classes.count({dtype, headDim, headDim, masked, hasBias, f32, headDim, 0}),
                        1U)
                        << dtype << " head_dim=" << headDim << " masked=" << masked
                        << " bias=" << hasBias << " f32=" << f32;
                }
            }
            for(const int64_t headDim : {64, 128})
            {
                EXPECT_EQ(classes.count({dtype, headDim, headDim, masked, 0, 0, headDim, 1}), 1U)
                    << dtype << " sink head_dim=" << headDim << " masked=" << masked;
            }
            EXPECT_EQ(classes.count({dtype, 192, 128, masked, 0, 0, 192, 0}), 1U)
                << dtype << " MLA masked=" << masked;
            for(const int64_t tier : {64, 96, 128, 160, 224, 256})
            {
                EXPECT_EQ(classes.count({dtype, 0, 0, masked, 0, 0, tier, 0}), 1U)
                    << dtype << " padded tier " << tier << " masked=" << masked;
            }
        }
    }
}

TEST(TestFlydslSdpaAiwPacks, EveryKernelDeclaresTheFiftySlotSignature)
{
    const auto& set = loadedSet(FLYDSL_SDPA_AIW.engineName);

    // Q, K, V, B, O, LSE and four varlen pointers; six i32; two dropout pointers, an i64 and
    // two more pointers; an i32, an f32, four i32, the f32 scale; fifteen i64 strides; SINK,
    // its i64 stride, the mask's i64 key stride, num_head_v (i32) and the LSE's three i64
    // strides -- each at the next offset its size aligns to.
    std::vector<std::pair<std::string, uint32_t>> kinds;
    const auto add = [&kinds](const char* kind, uint32_t size, int count) {
        for(int i = 0; i < count; ++i)
        {
            kinds.emplace_back(kind, size);
        }
    };
    add("global_buffer", 8, 10);
    add("by_value", 4, 6);
    add("global_buffer", 8, 2);
    add("by_value", 8, 1);
    add("global_buffer", 8, 2);
    add("by_value", 4, 7);
    add("by_value", 8, 15);
    add("global_buffer", 8, 1);
    add("by_value", 8, 2);
    add("by_value", 4, 1);
    add("by_value", 8, 3);
    ASSERT_EQ(kinds.size(), 50U);
    std::vector<uint32_t> offsets;
    uint32_t offset = 0;
    for(const auto& [kind, size] : kinds)
    {
        offset = (offset + size - 1) / size * size;
        offsets.push_back(offset);
        offset += size;
    }

    size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ASSERT_EQ(kernel.source.signature.size(), kinds.size()) << kernel.name;
            for(size_t slot = 0; slot < kinds.size(); ++slot)
            {
                const auto& argument = kernel.source.signature[slot];
                EXPECT_EQ(argument.kind, kinds[slot].first) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, kinds[slot].second) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, offsets[slot]) << kernel.name << " slot " << slot;
            }
            ++checked;
        }
    }
    EXPECT_GT(checked, 0U);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
