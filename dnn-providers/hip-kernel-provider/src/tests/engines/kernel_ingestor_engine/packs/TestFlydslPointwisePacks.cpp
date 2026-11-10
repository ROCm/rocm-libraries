// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormPacks.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslPointwiseTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

/**
 * @file TestFlydslPointwisePacks.cpp
 * @brief A census of what hipkernel:flydsl_pointwise loads, read from the shard the census
 *        entry names. Census binary only: every case opens by reading loadedSet().
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslPointwisePacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_POINTWISE.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_POINTWISE.graphMatcher);
    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslPointwisePacks, ShipsOneObjectPerOperationAndTypeCombination)
{
    const auto& set = loadedSet(FLYDSL_POINTWISE.engineName);

    const std::array<std::string, 3> dtypes{"bf16", "f16", "f32"};
    std::set<std::tuple<std::string, std::string, std::string, std::string>> expected;
    for(const std::string operation : {"sigmoid", "swish", "exp", "softplus", "neg", "identity"})
    {
        for(const auto& in0 : dtypes)
        {
            for(const auto& out : dtypes)
            {
                expected.emplace(operation, in0, "none", out);
            }
        }
    }
    for(const std::string operation : {"add", "sub", "mul", "div"})
    {
        for(const auto& in0 : dtypes)
        {
            for(const auto& in1 : dtypes)
            {
                for(const auto& out : dtypes)
                {
                    expected.emplace(operation, in0, in1, out);
                }
            }
        }
    }

    // Per architecture: the packer emits one copy of the pack per arch it ships for.
    std::set<std::string> arches;
    for(const auto& pack : set.packs)
    {
        arches.insert(pack.arch.begin(), pack.arch.end());
    }
    for(const auto& arch : arches)
    {
        std::set<std::tuple<std::string, std::string, std::string, std::string>> shipped;
        for(const auto& pack : set.packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                const auto& kernelArches = kernel.arch.empty() ? pack.arch : kernel.arch;
                if(std::find(kernelArches.begin(), kernelArches.end(), arch) == kernelArches.end())
                {
                    continue;
                }
                const auto* operation = tryGetMetadataField<std::string>(
                    kernel.metadata, FLYDSL_POINTWISE_OPERATION_FIELD);
                const auto* in0 = tryGetMetadataField<std::string>(kernel.metadata,
                                                                   FLYDSL_POINTWISE_DTYPE_FIELD);
                const auto* in1 = tryGetMetadataField<std::string>(
                    kernel.metadata, FLYDSL_POINTWISE_IN1_DTYPE_FIELD);
                const auto* out = tryGetMetadataField<std::string>(
                    kernel.metadata, FLYDSL_POINTWISE_OUT_DTYPE_FIELD);
                ASSERT_TRUE(operation != nullptr && in0 != nullptr && in1 != nullptr
                            && out != nullptr)
                    << kernel.name << " lacks a string operation or dtype field";
                EXPECT_TRUE(shipped.emplace(*operation, *in0, *in1, *out).second)
                    << kernel.name << " duplicates another object on " << arch;

                // Pre-built objects only: there is no source to fall back to.
                EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
                EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
                EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
            }
        }
        EXPECT_EQ(shipped, expected) << arch;
    }
}

TEST(TestFlydslPointwisePacks, EveryKernelDeclaresTheTwentyOneSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_POINTWISE.engineName);

    // Out, A, B (pointers); n, vec, d1, d2, d3 (int32); the output's, A's and B's four
    // element strides (int64, padded to 8 after the int32s); param (f32).
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ASSERT_EQ(kernel.source.signature.size(), 21U) << kernel.name;
            for(size_t slot = 0; slot < 21; ++slot)
            {
                const auto& argument = kernel.source.signature[slot];
                uint32_t size = 0;
                uint32_t offset = 0;
                std::string kind = "by_value";
                if(slot < 3)
                {
                    kind = "global_buffer";
                    size = 8;
                    offset = static_cast<uint32_t>(8 * slot);
                }
                else if(slot < 8)
                {
                    size = 4;
                    offset = static_cast<uint32_t>(24 + 4 * (slot - 3));
                }
                else if(slot < 20)
                {
                    size = 8;
                    offset = static_cast<uint32_t>(48 + 8 * (slot - 8));
                }
                else
                {
                    size = 4;
                    offset = 144;
                }
                EXPECT_EQ(argument.kind, kind) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, size) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, offset) << kernel.name << " slot " << slot;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
