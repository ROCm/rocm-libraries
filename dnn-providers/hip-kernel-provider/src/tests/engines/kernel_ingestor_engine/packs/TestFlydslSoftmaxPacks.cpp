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
#include <tuple>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSoftmaxTestGraphs.hpp"

/**
 * @file TestFlydslSoftmaxPacks.cpp
 * @brief A census of what hipkernel:flydsl_softmax loads, read from the shard the census
 *        entry names. Every case opens by reading loadedSet(); see TestFlydslRmsNormPacks.cpp
 *        for why that is what qualifies a suite for the census binary.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

int64_t intField(const KernelDescriptor& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
    return value == nullptr ? -1 : *value;
}

TEST(TestFlydslSoftmaxPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_SOFTMAX.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_SOFTMAX.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslSoftmaxPacks, ShipsEveryDtypePairInEveryTierOnce)
{
    const auto& set = loadedSet(FLYDSL_SOFTMAX.engineName);

    // A tier: (N, n_max, vector or per-element, block_threads, threads_per_row).
    using Tier = std::tuple<int64_t, int64_t, bool, int64_t, int64_t>;
    const std::set<Tier> tiers{{0, 256, true, 256, 16},
                               {0, 2048, true, 256, 64},
                               {0, 16384, true, 256, 256},
                               {0, 0, true, 1024, 1024},
                               {0, 2048, false, 256, 64},
                               {0, 0, false, 256, 256},
                               {512, 512, true, 256, 64}};

    // Per arch: the packer emits one copy of the pack per architecture it ships for.
    std::map<std::string, std::map<std::pair<std::string, std::string>, std::multiset<Tier>>>
        perArch;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_DTYPE_FIELD);
            const auto* outDtype = tryGetMetadataField<std::string>(kernel.metadata, "out_dtype");
            ASSERT_NE(dtype, nullptr) << kernel.name << " declares no string dtype";
            ASSERT_NE(outDtype, nullptr) << kernel.name << " declares no string out_dtype";

            // Only ever kpack: pre-built flyDSL code objects with no source to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;

            // The vector width is 128 bits of p: 8 for 16-bit p, 4 for f32.
            const auto nAlign = intField(kernel, "n_align");
            const auto vector = *dtype == "f32" ? int64_t{4} : int64_t{8};
            EXPECT_TRUE(nAlign == 1 || nAlign == vector) << kernel.name;
            const Tier tier{intField(kernel, FLYDSL_N_FIELD),
                            intField(kernel, "n_max"),
                            nAlign > 1,
                            intField(kernel, FLYDSL_BLOCK_THREADS_FIELD),
                            intField(kernel, "threads_per_row")};

            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            for(const auto& arch : arches)
            {
                perArch[arch][{*dtype, *outDtype}].insert(tier);
            }
        }
    }

    ASSERT_FALSE(perArch.empty());
    const std::multiset<Tier> everyTierOnce(tiers.begin(), tiers.end());
    for(const auto& [arch, pairs] : perArch)
    {
        EXPECT_EQ(pairs.size(), 9U) << arch << ": every p dtype beside every s dtype";
        for(const auto& [pair, shipped] : pairs)
        {
            EXPECT_EQ(shipped, everyTierOnce) << arch << " " << pair.first << "->" << pair.second;
        }
    }
}

TEST(TestFlydslSoftmaxPacks, EveryKernelDeclaresTheNineSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_SOFTMAX.engineName);

    // p and s: a pointer and a 16-byte 2-D descriptor each; stats: a pointer and a 4-byte
    // 1-D descriptor; the row count, the stats switch and the stats scale, 4 bytes each.
    // If a regenerated object changes this, change flydslSoftmaxKernelSignature() and
    // launch() with it.
    constexpr std::array<std::tuple<const char*, uint32_t, uint32_t>, 9> EXPECTED{{
        {"global_buffer", 8, 0},
        {"by_value", 16, 8},
        {"global_buffer", 8, 24},
        {"by_value", 16, 32},
        {"global_buffer", 8, 48},
        {"by_value", 4, 56},
        {"by_value", 4, 60},
        {"by_value", 4, 64},
        {"by_value", 4, 68},
    }};
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& signature = kernel.source.signature;
            ASSERT_EQ(signature.size(), EXPECTED.size()) << kernel.name;
            for(size_t slot = 0; slot < signature.size(); ++slot)
            {
                const auto& [kind, size, offset] = EXPECTED[slot];
                EXPECT_EQ(signature[slot].kind, kind) << kernel.name << " slot " << slot;
                EXPECT_EQ(signature[slot].size, size) << kernel.name << " slot " << slot;
                EXPECT_EQ(signature[slot].offset, offset) << kernel.name << " slot " << slot;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
