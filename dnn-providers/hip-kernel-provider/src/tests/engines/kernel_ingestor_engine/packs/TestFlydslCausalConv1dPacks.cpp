// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormPacks.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslCausalConv1dTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

/**
 * @file TestFlydslCausalConv1dPacks.cpp
 * @brief A census of what hipkernel:flydsl_causal_conv1d loads, read from the shard the
 *        census entry names. Every case opens by reading loadedSet(); see
 *        TestFlydslRmsNormPacks.cpp for why that is what qualifies a suite for the census
 *        binary.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslCausalConv1dPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_CAUSAL_CONV1D.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_CAUSAL_CONV1D.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslCausalConv1dPacks, ShipsEveryDtypeWidthActivationAndLaneCountOnce)
{
    const auto& set = loadedSet(FLYDSL_CAUSAL_CONV1D.engineName);

    // Per arch: the packer emits one copy of the pack per architecture it ships for.
    using Variant = std::tuple<std::string, int64_t, std::string, int64_t>;
    std::map<std::string, std::map<Variant, size_t>> perArch;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_DTYPE_FIELD);
            const auto* width = tryGetMetadataField<int64_t>(kernel.metadata, "width");
            const auto* activation
                = tryGetMetadataField<std::string>(kernel.metadata, "activation");
            const auto* lanes = tryGetMetadataField<int64_t>(kernel.metadata, "lanes");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(width, nullptr) << kernel.name;
            ASSERT_NE(activation, nullptr) << kernel.name;
            ASSERT_NE(lanes, nullptr) << kernel.name;
            // The 4-lane objects rank above the 1-lane ones that serve every other C.
            EXPECT_EQ(kernel.priority, *lanes == 4 ? 100 : 10) << kernel.name;

            // Only ever kpack: pre-built flyDSL code objects with no source to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;

            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            for(const auto& arch : arches)
            {
                perArch[arch][Variant{*dtype, *width, *activation, *lanes}] += 1;
            }
        }
    }

    ASSERT_FALSE(perArch.empty());
    std::map<Variant, size_t> oneEach;
    for(const auto* dtype : {"bf16", "f16", "f32"})
    {
        for(const int64_t width : {2, 3, 4})
        {
            for(const auto* activation : {"identity", "silu"})
            {
                for(const int64_t lanes : {1, 4})
                {
                    oneEach[Variant{dtype, width, activation, lanes}] = 1;
                }
            }
        }
    }
    for(const auto& [arch, counts] : perArch)
    {
        EXPECT_EQ(counts, oneEach) << arch;
    }
}

TEST(TestFlydslCausalConv1dPacks, EveryKernelDeclaresTheTwentySixSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_CAUSAL_CONV1D.engineName);

    // Eight addresses and eleven int64 element strides, all by value; then seven int32s
    // (T, N, C, the state rows, the two index widths, the flags). If a regenerated object
    // changes this, change flydslCausalConv1dKernelSignature() and launch() with it.
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& signature = kernel.source.signature;
            ASSERT_EQ(signature.size(), 26U) << kernel.name;
            for(size_t slot = 0; slot < signature.size(); ++slot)
            {
                const auto& argument = signature[slot];
                const bool wide = slot < 19;
                const uint32_t offset = wide ? static_cast<uint32_t>(8 * slot)
                                             : static_cast<uint32_t>(152 + 4 * (slot - 19));
                EXPECT_EQ(argument.kind, "by_value") << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, wide ? 8U : 4U) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, offset) << kernel.name << " slot " << slot;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
