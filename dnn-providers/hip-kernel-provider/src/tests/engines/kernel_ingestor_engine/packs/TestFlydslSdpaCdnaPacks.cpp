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

TEST(TestFlydslSdpaCdnaPacks, ShipsOneObjectPerClassAndOneFamilyPerHeadDim)
{
    const auto& set = loadedSet(ENGINE);

    // dtype x family x head_dim x causal x kv_align x any_diagonal is the whole instance
    // key today: a duplicate would make the choice between two objects an accident of ids.
    std::set<std::tuple<std::string, std::string, int64_t, int64_t, int64_t, int64_t>> classes;
    // Which kernel family serves each (dtype, head_dim): gfx950 ships the dual-wave
    // kernel beside the generic one, each for its own head dims.
    std::map<std::pair<std::string, int64_t>, std::set<std::string>> families;
    size_t kernels = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
            const auto* family = tryGetMetadataField<std::string>(kernel.metadata, "kernel");
            const auto* headDim = tryGetMetadataField<int64_t>(kernel.metadata, "head_dim");
            const auto* causal = tryGetMetadataField<int64_t>(kernel.metadata, "causal");
            const auto* kvAlign = tryGetMetadataField<int64_t>(kernel.metadata, "kv_align");
            const auto* anyDiagonal = tryGetMetadataField<int64_t>(kernel.metadata, "any_diagonal");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(family, nullptr) << kernel.name;
            ASSERT_NE(headDim, nullptr) << kernel.name;
            ASSERT_NE(causal, nullptr) << kernel.name;
            ASSERT_NE(kvAlign, nullptr) << kernel.name;
            ASSERT_NE(anyDiagonal, nullptr) << kernel.name;
            classes.emplace(*dtype, *family, *headDim, *causal, *kvAlign, *anyDiagonal);
            families[{*dtype, *headDim}].insert(*family);
            ++kernels;

            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }
    EXPECT_GT(kernels, 0U);
    EXPECT_EQ(kernels, classes.size()) << "two objects claim one class";
    for(const auto& [key, served] : families)
    {
        EXPECT_EQ(served.size(), 1U) << key.first << " head_dim " << key.second
                                     << " is served by more than one kernel family";
    }
}

TEST(TestFlydslSdpaCdnaPacks, EveryKernelDeclaresItsFamilysSignature)
{
    const auto& set = loadedSet(ENGINE);

    // Dual-wave: 12 tensors, then 8 i32, the f32 scale, the two head counts, lse_on and
    // diag_offset. Generic: 5 tensors, 2 i32, 3 tensors, then 1 i32, the f32 scale, the
    // two head counts, the two token strides, lse_on and diag_offset. Every scalar slot
    // is 4 bytes.
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
            scalars(13);
        }
        else
        {
            tensors(5);
            scalars(2);
            tensors(3);
            scalars(8);
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
