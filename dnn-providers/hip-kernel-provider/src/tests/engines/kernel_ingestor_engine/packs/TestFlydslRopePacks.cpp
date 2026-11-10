// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormPacks.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRopeTestGraphs.hpp"

/**
 * @file TestFlydslRopePacks.cpp
 * @brief A census of what hipkernel:flydsl_rope loads, read from the shard the census
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

TEST(TestFlydslRopePacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_ROPE.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_ROPE.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslRopePacks, ShipsOneRuntimeShapeObjectPerDtype)
{
    const auto& set = loadedSet(FLYDSL_ROPE.engineName);

    // Per arch: the packer emits one copy of the pack per architecture it ships for, so
    // count per (arch, dtype).
    std::map<std::string, std::map<std::string, size_t>> perArch;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_DTYPE_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name << " declares no string dtype";
            const auto* blockThreads
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_BLOCK_THREADS_FIELD);
            ASSERT_NE(blockThreads, nullptr) << kernel.name << " declares no block_threads";
            EXPECT_EQ(*blockThreads, 64) << kernel.name;

            // Only ever kpack: pre-built flyDSL code objects with no source to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;

            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            for(const auto& arch : arches)
            {
                perArch[arch][*dtype] += 1;
            }
        }
    }

    ASSERT_FALSE(perArch.empty());
    const std::map<std::string, size_t> oneEach{{"bf16", 1}, {"f16", 1}, {"f32", 1}};
    for(const auto& [arch, counts] : perArch)
    {
        EXPECT_EQ(counts, oneEach) << arch;
    }
}

TEST(TestFlydslRopePacks, EveryKernelDeclaresTheNineteenSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_ROPE.engineName);

    // x, freqs and y pointers; the four int64 element strides of x, of freqs and of y;
    // head_dim, rope_dim and rope_offset (int32); output_scale (f32). If a regenerated
    // object changes this, change flydslRopeKernelSignature() and launch() with it.
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& signature = kernel.source.signature;
            ASSERT_EQ(signature.size(), 19U) << kernel.name;
            for(size_t slot = 0; slot < signature.size(); ++slot)
            {
                const auto& argument = signature[slot];
                const bool pointer = slot < 3;
                const bool stride = slot >= 3 && slot < 15;
                const uint32_t size = pointer || stride ? 8U : 4U;
                const uint32_t offset = slot < 15 ? static_cast<uint32_t>(8 * slot)
                                                  : static_cast<uint32_t>(120 + 4 * (slot - 15));
                EXPECT_EQ(argument.kind, pointer ? "global_buffer" : "by_value")
                    << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, size) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, offset) << kernel.name << " slot " << slot;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
