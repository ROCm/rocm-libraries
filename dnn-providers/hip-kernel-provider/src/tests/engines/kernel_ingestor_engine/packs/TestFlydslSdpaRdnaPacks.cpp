// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
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
 * @file TestFlydslSdpaRdnaPacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa_rdna actually loads, read from the shard the
 *        census entry names rather than from anything built in-process.
 *
 * Belongs to the census binary for the reason TestFlydslRmsNormPacks.cpp does: every case
 * opens by reading loadedSet(), none skips, and none touches a device, so an empty shard
 * fails here instead of passing silently. The seam suites -- matchers, bindings, dispatch
 * and the real launches -- are in TestFlydslSdpaRdnaEngine.cpp.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslSdpaRdnaPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_SDPA_RDNA.graphMatcher);

    // SdpaAttributes is its own union arm, so admitting the node type IS the operation
    // test: there is no graph-scoped matcher to do it a second time.
    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslSdpaRdnaPacks, ShipsEverySpecializedAndGenericVariantOnce)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA.engineName);

    // dtype x head_dim x causal x has_bias x bias_q x head_dim_max is the whole instance key
    // (a generic object's head_dim is 0, read at runtime up to its max; bias_q says the bias
    // is in the query's dtype rather than f32): everything else the
    // kernel needs is a runtime argument. A missing class would decline silently -- the graph
    // would simply plan on another engine -- and a duplicate would make the choice
    // between two objects for one class an accident of descriptor ids.
    std::set<std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> classes;
    size_t kernels = 0;
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
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(headDim, nullptr) << kernel.name;
            ASSERT_NE(causal, nullptr) << kernel.name;
            const auto* headDimMax = tryGetMetadataField<int64_t>(
                kernel.metadata, FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD);
            ASSERT_NE(hasBias, nullptr) << kernel.name;
            const auto* biasQ
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_BIAS_Q_FIELD);
            ASSERT_NE(biasQ, nullptr) << kernel.name;
            const auto* decode
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_DECODE_FIELD);
            ASSERT_NE(headDimMax, nullptr) << kernel.name;
            ASSERT_NE(decode, nullptr) << kernel.name;
            classes.emplace(*dtype, *headDim, *causal, *hasBias, *biasQ, *headDimMax, *decode);
            ++kernels;

            // Only ever kpack: pre-built flyDSL code objects with no source to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }

    EXPECT_EQ(kernels, classes.size()) << "two objects claim one class";
    for(const char* dtype : {"bf16", "f16"})
    {
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
        // The generic tiers: head_dim read at runtime, no bias, each up to its largest head.
        for(const int64_t headDimMax : {64, 96, 128, 160, 224, 256})
        {
            for(const int64_t causal : {0, 1})
            {
                EXPECT_EQ(classes.count({dtype, 0, causal, 0, 0, headDimMax, 0}), 1U)
                    << dtype << " generic head_dim up to " << headDimMax << " causal=" << causal;
            }
        }
        // The decode family: the specialized head dims, with and without a bias.
        for(const int64_t headDim : {64, 96, 128, 256})
        {
            for(const int64_t causal : {0, 1})
            {
                // No bias, an f32 bias, and a bias in the query's dtype.
                for(const auto& [hasBias, biasQ] :
                    {std::pair<int64_t, int64_t>{0, 0}, {1, 0}, {1, 1}})
                {
                    EXPECT_EQ(classes.count({dtype, headDim, causal, hasBias, biasQ, headDim, 1}),
                              1U)
                        << dtype << " decode head_dim=" << headDim << " causal=" << causal
                        << " has_bias=" << hasBias << " bias_q=" << biasQ;
                }
            }
        }
        // Generic decode: head_dim read at runtime, no bias, up to 96 and up to 256.
        for(const int64_t headDimMax : {96, 256})
        {
            for(const int64_t causal : {0, 1})
            {
                EXPECT_EQ(classes.count({dtype, 0, causal, 0, 0, headDimMax, 1}), 1U)
                    << dtype << " generic decode up to " << headDimMax << " causal=" << causal;
            }
        }
    }
}

/// Whether @p kernel is one of the decode family's objects.
template <typename Kernel>
bool isDecodeKernel(const Kernel& kernel)
{
    const auto* decode
        = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_DECODE_FIELD);
    return decode != nullptr && *decode == 1;
}

TEST(TestFlydslSdpaRdnaPacks, RanksTheNarrowestServingObjectFirst)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA.engineName);

    // Several objects can serve one graph and the highest priority wins, so the ordering
    // is part of the content: a decode object above every prefill one, an object built
    // for the exact head_dim above every generic tier, and among the generic tiers the
    // narrower above the wider -- equal priorities would leave the choice to descriptor
    // order. Per (dtype, causal): {decode, generic} -> [(head_dim_max, priority)].
    struct Ranked
    {
        int64_t headDimMax;
        int64_t priority;
    };
    std::map<std::tuple<std::string, int64_t, int64_t, bool>, std::vector<Ranked>> families;
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
            const auto* decode
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_DECODE_FIELD);
            ASSERT_TRUE(dtype != nullptr && headDim != nullptr && causal != nullptr
                        && headDimMax != nullptr && decode != nullptr)
                << kernel.name;
            families[{*dtype, *causal, *decode, *headDim == 0}].push_back(
                {*headDimMax, static_cast<int64_t>(kernel.priority)});
            ++checked;
        }
    }
    ASSERT_GT(checked, 0U);

    auto lowest = [](const std::vector<Ranked>& v) {
        int64_t p = std::numeric_limits<int64_t>::max();
        for(const auto& r : v)
        {
            p = std::min(p, r.priority);
        }
        return p;
    };
    auto highest = [](const std::vector<Ranked>& v) {
        int64_t p = std::numeric_limits<int64_t>::lowest();
        for(const auto& r : v)
        {
            p = std::max(p, r.priority);
        }
        return p;
    };
    for(const auto& [key, ranked] : families)
    {
        const auto& [dtype, causal, decode, generic] = key;
        if(generic)
        {
            auto tiers = ranked;
            std::sort(tiers.begin(), tiers.end(), [](const Ranked& a, const Ranked& b) {
                return a.headDimMax < b.headDimMax;
            });
            for(size_t i = 1; i < tiers.size(); ++i)
            {
                EXPECT_GT(tiers[i - 1].priority, tiers[i].priority)
                    << dtype << " causal=" << causal << " decode=" << decode << ": the tier up to "
                    << tiers[i - 1].headDimMax << " must outrank the one up to "
                    << tiers[i].headDimMax;
            }
            const auto exact = families.find({dtype, causal, decode, false});
            ASSERT_NE(exact, families.end());
            EXPECT_GT(lowest(exact->second), highest(ranked))
                << dtype << " causal=" << causal << " decode=" << decode
                << ": an exact head_dim object must outrank every generic tier";
        }
        if(decode == 1)
        {
            for(const bool prefillGeneric : {false, true})
            {
                const auto prefill = families.find({dtype, causal, 0, prefillGeneric});
                ASSERT_NE(prefill, families.end());
                EXPECT_GT(lowest(ranked), highest(prefill->second))
                    << dtype << " causal=" << causal << ": a decode object must outrank every "
                    << "prefill object";
            }
        }
    }
}

TEST(TestFlydslSdpaRdnaPacks, EveryPrefillKernelDeclaresTheThirtySixSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA.engineName);

    // Q, K, V, O, LSE pointers; eight i32 (seq_len_q, seq_len_kv, num_heads, kv_group,
    // right_bound, left_bound, align_bottom_right, lse_on); the f32 scale; fifteen i64
    // strides from offset 80; then the bias pointer at 200 and its four i64 strides from
    // 208, then the runtime head_dim (i32) at 240 and the query heads per V head (i32) at
    // 244 -- each appended so every earlier slot kept its offset. Most slots share a size
    // with a neighbour, so a permutation is invisible to the loader's kind-and-size check --
    // the semantic names in the manifest are the only record of which is which, and this
    // table is what a regenerated layout has to agree with.
    constexpr size_t SLOTS = 36;
    size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            if(isDecodeKernel(kernel))
            {
                continue;
            }
            ++checked;
            ASSERT_EQ(kernel.source.signature.size(), SLOTS) << kernel.name;
            for(size_t slot = 0; slot < SLOTS; ++slot)
            {
                const auto& argument = kernel.source.signature[slot];
                std::string kind = "by_value";
                uint32_t size = 8;
                uint32_t offset = 0;
                if(slot < 5)
                {
                    kind = "global_buffer";
                    offset = static_cast<uint32_t>(8 * slot);
                }
                else if(slot < 14)
                {
                    size = 4;
                    offset = static_cast<uint32_t>(40 + 4 * (slot - 5));
                }
                else if(slot < 29)
                {
                    offset = static_cast<uint32_t>(80 + 8 * (slot - 14));
                }
                else if(slot == 29)
                {
                    kind = "global_buffer";
                    offset = 200;
                }
                else if(slot < 34)
                {
                    offset = static_cast<uint32_t>(208 + 8 * (slot - 30));
                }
                else
                {
                    size = 4;
                    offset = static_cast<uint32_t>(240 + 4 * (slot - 34));
                }
                EXPECT_EQ(argument.kind, kind) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, size) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, offset) << kernel.name << " slot " << slot;
            }
        }
    }
    EXPECT_GT(checked, 0U);
}

TEST(TestFlydslSdpaRdnaPacks, EveryDecodeKernelDeclaresTheThirtyEightSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_SDPA_RDNA.engineName);

    // Q, K, V, O, LSE and the split workspace's O and LSE pointers; nine i32 (seq_len_q,
    // seq_len_kv, num_heads, kv_group, right_bound, left_bound, align_bottom_right, lse_on,
    // num_splits) and the f32 scale from offset 56; fifteen i64 strides from 96; the bias
    // pointer at 216 and its four i64 strides; the i32 head_dim at 256. The merge kernel
    // in the same object is launched by name and checked by the generator.
    constexpr size_t SLOTS = 38;
    size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            if(!isDecodeKernel(kernel))
            {
                continue;
            }
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
                    kind = slot == 32 ? "global_buffer" : "by_value";
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
