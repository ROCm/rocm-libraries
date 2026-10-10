// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaCdnaPacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa_cdna actually loads, read from the shard the
 *        census entry names rather than from anything built in-process.
 *
 * Every case opens by reading loadedSet(), none skips, and none touches a device, so an
 * empty gfx942/gfx950 shard fails here instead of passing silently. The seam suites --
 * matchers, bindings and the real launches -- are in TestFlydslSdpaCdnaEngine.cpp.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt, over the
 * architectures the CDNA objects ship for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* tryGetMetadataField(const MetadataValues& values, const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

constexpr const char* ENGINE = "hipkernel:flydsl_sdpa_cdna";
constexpr const char* GRAPH_MATCHER = "hipkernel.flydsl_sdpa_cdna.graph_match";

/// One tensor parameter: a pointer and FlyDSL's 40-byte rank-4 dynamic layout.
constexpr uint32_t POINTER_BYTES = 8;
constexpr uint32_t LAYOUT_BYTES = 40;
constexpr uint32_t I32_BYTES = 4;

TEST(TestFlydslSdpaCdnaPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(ENGINE);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, GRAPH_MATCHER);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslSdpaCdnaPacks, ShipsUniqueObjectsAndOneServeAllDefaultPerClass)
{
    const auto& set = loadedSet(ENGINE);

    // Everything an object was built with: a duplicate would make the choice between two
    // objects an accident of ids.
    using BuildKey = std::tuple<std::string,
                                std::string,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t>;
    std::set<BuildKey> builds;
    // Per graph class -- (dtype, head_dim, head_dim_v, causal, has_bias) -- how many objects serve every
    // shape of it at the default rank: exactly one, so the untuned pick never depends on
    // a key length or a diagonal, and every other object is a measured companion below it.
    using ClassKey = std::tuple<std::string, int64_t, int64_t, int64_t, int64_t>;
    std::map<ClassKey, size_t> defaults;
    const std::set<std::string> features = {"none", "rows", "seq_q"};
    size_t kernels = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto field = [&](const char* name) {
                const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, name);
                EXPECT_NE(value, nullptr) << kernel.name << " has no " << name;
                return value == nullptr ? int64_t{-1} : *value;
            };
            const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
            const auto* family = tryGetMetadataField<std::string>(kernel.metadata, "kernel");
            const auto* feature
                = tryGetMetadataField<std::string>(kernel.metadata, "prefer_feature");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(family, nullptr) << kernel.name;
            ASSERT_NE(feature, nullptr) << kernel.name;
            EXPECT_EQ(features.count(*feature), 1U) << kernel.name << ": " << *feature;

            const auto headDim = field("head_dim");
            const auto causal = field("causal");
            const auto kvAlign = field("kv_align");
            const auto anyDiagonal = field("any_diagonal");
            const auto hasBias = field("has_bias");
            const auto headDimV = field("head_dim_v");
            builds.emplace(*dtype,
                           *family,
                           headDim,
                           causal,
                           field("block_m"),
                           kvAlign,
                           anyDiagonal,
                           field("waves_per_eu"),
                           hasBias,
                           field("setprio"),
                           field("kv_gpfetch"),
                           headDimV,
                           field("qk_mask"),
                           field("split_kv"),
                           field("pack_gqa"),
                           field("dv_split"));
            auto& count = defaults[{*dtype, headDim, headDimV, causal, hasBias}];
            if(kvAlign == 0 && anyDiagonal == 1 && kernel.priority >= 100)
            {
                ++count;
            }
            ++kernels;

            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }
    EXPECT_GT(kernels, 0U);
    EXPECT_EQ(kernels, builds.size()) << "two objects share one build";
    for(const auto& [key, count] : defaults)
    {
        EXPECT_EQ(count, 1U) << std::get<0>(key) << " head_dim " << std::get<1>(key) << "/"
                             << std::get<2>(key) << " causal " << std::get<3>(key) << " bias "
                             << std::get<4>(key) << ": " << count << " serve-all default objects";
    }
}

TEST(TestFlydslSdpaCdnaPacks, EveryKernelDeclaresItsFamilysSignature)
{
    const auto& set = loadedSet(ENGINE);

    // Dual-wave: 12 tensors, then 8 i32, the f32 scale, the two head counts, lse_on,
    // diag_offset, seven operand strides and two bias strides. Generic: 5 tensors, 2 i32, 3 tensors, then 1 i32, the
    // f32 scale, the two head counts, the two token strides, lse_on, diag_offset and the
    // same seven strides, the window, the bias, its three strides, V's three strides and
    // the Q/K head dim, then the split workspace's two tensors and the split count. Every
    // scalar slot is 4 bytes.
    using Slot = std::pair<std::string, uint32_t>;
    const auto expected = [](const std::string& family) {
        std::vector<Slot> slots;
        const auto tensors = [&slots](int count) {
            for(int i = 0; i < count; ++i)
            {
                slots.emplace_back("global_buffer", POINTER_BYTES);
                slots.emplace_back("by_value", LAYOUT_BYTES);
            }
        };
        const auto scalars = [&slots](int count) {
            for(int i = 0; i < count; ++i)
            {
                slots.emplace_back("by_value", I32_BYTES);
            }
        };
        if(family == "dualwave")
        {
            tensors(12);
            scalars(22);
        }
        else
        {
            tensors(5);
            scalars(2);
            tensors(3);
            scalars(16);
            tensors(1);
            scalars(7);
            tensors(2);
            scalars(1);
        }
        return slots;
    };

    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* family = tryGetMetadataField<std::string>(kernel.metadata, "kernel");
            ASSERT_NE(family, nullptr) << kernel.name;
            const auto slots = expected(*family);
            const auto& signature = kernel.source.signature;
            ASSERT_EQ(signature.size(), slots.size()) << kernel.name;
            for(size_t i = 0; i < slots.size(); ++i)
            {
                EXPECT_EQ(signature[i].kind, slots[i].first) << kernel.name << " argument " << i;
                EXPECT_EQ(signature[i].size, slots[i].second) << kernel.name << " argument " << i;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
