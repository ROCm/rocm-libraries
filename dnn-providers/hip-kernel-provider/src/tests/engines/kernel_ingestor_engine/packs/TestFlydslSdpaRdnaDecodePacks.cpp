// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaRdnaDecodePacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa_rdna_decode (the RDNA split-KV decode
 *        kernel's engine) actually loads, read from the shard the census entry names.
 *
 * Belongs to the census binary for the reason TestFlydslSdpaRdnaPacks.cpp does: every case
 * opens by reading loadedSet(), none skips, and none touches a device. Registered with
 * hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslSdpaRdnaDecodePacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA_DECODE.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_SDPA_RDNA_DECODE.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslSdpaRdnaDecodePacks, ShipsEveryDecodeVariantOnce)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA_DECODE.engineName);

    // dtype x head_dim x causal x has_bias x bias_q x head_dim_max x has_sink is the whole
    // instance key, as it is for the prefill engine; each object also names the merge kernel
    // it launches beside its main one.
    std::set<std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> classes;
    size_t kernels = 0;
    std::set<std::tuple<std::string, int64_t, int64_t, int64_t>> narrowV;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_SDPA_RDNA_DTYPE_FIELD);
            const auto* headDim
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD);
            const auto* causal
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_CAUSAL_FIELD);
            const auto* hasBias
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD);
            const auto* biasQ
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_BIAS_Q_FIELD);
            const auto* headDimMax = tryGetMetadataField<int64_t>(
                kernel.metadata, FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD);
            const auto* hasSink
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HAS_SINK_FIELD);
            const auto* headDimV
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HEAD_DIM_V_FIELD);
            const auto* mergeSymbol = tryGetMetadataField<std::string>(
                kernel.metadata, FLYDSL_SDPA_RDNA_MERGE_SYMBOL_FIELD);
            ASSERT_TRUE(dtype != nullptr && headDim != nullptr && causal != nullptr
                        && hasBias != nullptr && biasQ != nullptr && headDimMax != nullptr
                        && hasSink != nullptr && headDimV != nullptr && mergeSymbol != nullptr)
                << kernel.name;
            EXPECT_FALSE(mergeSymbol->empty()) << kernel.name;
            classes.emplace(*dtype, *headDim, *causal, *hasBias, *biasQ, *headDimMax, *hasSink);
            if(*headDimV != *headDim)
            {
                narrowV.emplace(*dtype, *headDim, *headDimV, *causal);
            }
            ++kernels;

            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }

    EXPECT_EQ(kernels, classes.size()) << "two objects claim one class";
    // Absorbed multi-head latent attention (Q/K 576, V 512), both causal variants.
    EXPECT_EQ(narrowV.size(), 4U);
    for(const char* dtype : {"bf16", "f16"})
    {
        for(const int64_t causal : {0, 1})
        {
            EXPECT_EQ(narrowV.count({dtype, 576, 512, causal}), 1U)
                << dtype << " causal=" << causal;
            // DeepSeek-V4's 512/512.
            EXPECT_EQ(classes.count({dtype, 512, causal, 0, 0, 512, 0}), 1U)
                << dtype << " head_dim=512 causal=" << causal;
        }
        // The specialized head dims, with and without a bias.
        for(const int64_t headDim : {64, 96, 128, 256})
        {
            for(const int64_t causal : {0, 1})
            {
                // No bias, an f32 bias, and a bias in the query's dtype.
                for(const auto& [hasBias, biasQ] :
                    {std::pair<int64_t, int64_t>{0, 0}, {1, 0}, {1, 1}})
                {
                    EXPECT_EQ(classes.count({dtype, headDim, causal, hasBias, biasQ, headDim, 0}),
                              1U)
                        << dtype << " head_dim=" << headDim << " causal=" << causal
                        << " has_bias=" << hasBias << " bias_q=" << biasQ;
                }
            }
        }
        // Generic: head_dim read at runtime, up to 96 and up to 256, with and without a bias.
        for(const int64_t headDimMax : {96, 256})
        {
            for(const int64_t causal : {0, 1})
            {
                for(const auto& [hasBias, biasQ] :
                    {std::pair<int64_t, int64_t>{0, 0}, {1, 0}, {1, 1}})
                {
                    EXPECT_EQ(classes.count({dtype, 0, causal, hasBias, biasQ, headDimMax, 0}), 1U)
                        << dtype << " generic up to " << headDimMax << " causal=" << causal
                        << " has_bias=" << hasBias << " bias_q=" << biasQ;
                }
            }
        }
        // Attention sinks: gpt-oss's head dims, alone and with a bias in f32 or the query's
        // dtype (N1).
        for(const int64_t headDim : {64, 128})
        {
            for(const int64_t causal : {0, 1})
            {
                for(const auto& [hasBias, biasQ] :
                    {std::pair<int64_t, int64_t>{0, 0}, {1, 0}, {1, 1}})
                {
                    EXPECT_EQ(classes.count({dtype, headDim, causal, hasBias, biasQ, headDim, 1}),
                              1U)
                        << dtype << " sink head_dim=" << headDim << " causal=" << causal
                        << " has_bias=" << hasBias << " bias_q=" << biasQ;
                }
            }
        }
    }
}

TEST(TestFlydslSdpaRdnaDecodePacks, RanksTheNarrowestServingObjectFirst)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA_DECODE.engineName);

    // Within this engine: an object built for the exact head_dim above every generic tier,
    // and the narrower generic tier above the wider.
    struct Ranked
    {
        int64_t headDimMax;
        int64_t priority;
    };
    std::map<std::tuple<std::string, int64_t, bool, int64_t, int64_t>, std::vector<Ranked>>
        families;
    size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_SDPA_RDNA_DTYPE_FIELD);
            const auto* headDim
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD);
            const auto* causal
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_CAUSAL_FIELD);
            const auto* headDimMax = tryGetMetadataField<int64_t>(
                kernel.metadata, FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD);
            const auto* hasBias
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD);
            const auto* biasQ
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_BIAS_Q_FIELD);
            ASSERT_TRUE(dtype != nullptr && headDim != nullptr && causal != nullptr
                        && headDimMax != nullptr && hasBias != nullptr && biasQ != nullptr)
                << kernel.name;
            families[{*dtype, *causal, *headDim == 0, *hasBias, *biasQ}].push_back(
                {*headDimMax, static_cast<int64_t>(kernel.priority)});
            ++checked;
        }
    }
    ASSERT_GT(checked, 0U);

    for(const auto& [key, ranked] : families)
    {
        const auto& [dtype, causal, generic, hasBias, biasQ] = key;
        if(!generic)
        {
            continue;
        }
        auto tiers = ranked;
        std::sort(tiers.begin(), tiers.end(), [](const Ranked& a, const Ranked& b) {
            return a.headDimMax < b.headDimMax;
        });
        for(size_t i = 1; i < tiers.size(); ++i)
        {
            EXPECT_GT(tiers[i - 1].priority, tiers[i].priority)
                << dtype << " causal=" << causal << ": the tier up to " << tiers[i - 1].headDimMax
                << " must outrank the one up to " << tiers[i].headDimMax;
        }
        const auto exact = families.find({dtype, causal, false, hasBias, biasQ});
        ASSERT_NE(exact, families.end());
        int64_t lowestExact = std::numeric_limits<int64_t>::max();
        for(const auto& r : exact->second)
        {
            lowestExact = std::min(lowestExact, r.priority);
        }
        EXPECT_GT(lowestExact, tiers.front().priority)
            << dtype << " causal=" << causal
            << ": an exact head_dim object must outrank every generic tier";
    }
}

TEST(TestFlydslSdpaRdnaDecodePacks, EveryKernelDeclaresTheFortyFourSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA_DECODE.engineName);

    // Q, K, V, O, LSE and the split workspace's O and LSE pointers; nine i32 (seq_len_q,
    // seq_len_kv, num_heads, kv_group, right_bound, left_bound, align_bottom_right, flags,
    // num_splits) and the f32 scale from offset 56; fifteen i64 strides from 96; the bias
    // pointer at 216 and its four i64 strides; the i32 head_dim at 256; the sink pointer at
    // 264 and its i64 head stride at 272; the per-batch length and packed-sequence offset
    // pointers from 280 (kernel modification 26). The merge kernel
    // in the same object is launched by name and checked by the generator.
    constexpr size_t SLOTS = 44;
    size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            ASSERT_EQ(kernel.source.signature.size(), SLOTS) << kernel.name;
            for(size_t slot = 0; slot < SLOTS; ++slot)
            {
                const auto& argument = kernel.source.signature[slot];
                std::string kind = "by_value";
                uint32_t size = 8;
                uint32_t offset = 0;
                if(slot < 7)
                {
                    kind = "global_buffer";
                    offset = static_cast<uint32_t>(8 * slot);
                }
                else if(slot < 17)
                {
                    size = 4;
                    offset = static_cast<uint32_t>(56 + 4 * (slot - 7));
                }
                else
                {
                    kind = slot == 32 || slot == 38 || slot >= 40 ? "global_buffer" : "by_value";
                    size = slot == 37 ? 4 : 8;
                    offset = static_cast<uint32_t>(96 + 8 * (slot - 17));
                }
                EXPECT_EQ(argument.kind, kind) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, size) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, offset) << kernel.name << " slot " << slot;
            }
        }
    }
    EXPECT_GT(checked, 0U);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
