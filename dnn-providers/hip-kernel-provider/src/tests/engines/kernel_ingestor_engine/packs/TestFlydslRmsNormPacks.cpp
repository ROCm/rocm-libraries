// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

/**
 * @file TestFlydslRmsNormPacks.cpp
 * @brief A census of what hipkernel:flydsl_rmsnorm actually loads, read from the shard
 *        the census entry names rather than from anything built in-process.
 *
 * This file belongs to the census binary, hip_kernel_provider_census_tests, not to the
 * ordinary unit binary: every case below opens by reading loadedSet(), which needs a
 * descriptor shard and the architecture the census environment supplies. FlyDSL's other
 * suites -- matchers, scoring, dispatch and the real launches -- stay in
 * TestFlydslRmsNormEngine.cpp.
 *
 * These are the ones that qualify for a census entry. Every case here reads the
 * staged set, none skips, and none touches a device; a suite that built its graph
 * in-process instead would stay green against an EMPTY shard, which is the exact miss the
 * census exists to catch.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslRmsNormPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_RMSNORM.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_RMSNORM.graphMatcher);

    // RMSNormAttributes is its own union arm, so admitting the node type IS the operation
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

TEST(TestFlydslRmsNormPacks, ShipsTheGenericTierPerFormAndBakedTiersFor16Bit)
{
    const auto& set = loadedSet(FLYDSL_RMSNORM.engineName);

    std::map<std::string, size_t> generic;
    std::map<std::string, size_t> baked;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* n = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_N_FIELD);
            ASSERT_NE(n, nullptr) << kernel.name << " declares no integer N";
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_DTYPE_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name << " declares no string dtype";
            const auto* scaleF32
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SCALE_F32_FIELD);
            ASSERT_NE(scaleF32, nullptr) << kernel.name << " declares no integer scale_f32";
            const auto* storeRstd
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_STORE_RSTD_FIELD);
            ASSERT_NE(storeRstd, nullptr) << kernel.name << " declares no integer store_rstd";
            // The form: x's dtype, "+f32" when gamma is read as f32 beside it, and "+rstd"
            // for the training forward that also writes inv_rms.
            const auto form
                = *dtype + (*scaleF32 != 0 ? "+f32" : "") + (*storeRstd != 0 ? "+rstd" : "");
            (*n == FLYDSL_GENERIC_N ? generic : baked)[form] += 1;

            // Only ever kpack: these are pre-built flyDSL code objects, so there is no
            // source for the runtime compiler to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }

    // One sentinel kernel per form -- bf16, f16 and f32 with gamma in x's type, and bf16
    // and f16 with an f32 gamma, each for inference and for training -- and baked
    // instances beside each 16-bit form: a shard holding only the generic tier would still
    // plan, silently, at the slower tier. f32 has no baked tier: the kernel's fast path is
    // 16-bit only, so it would not be faster.
    std::map<std::string, size_t> oneEach;
    for(const std::string phase : {"", "+rstd"})
    {
        for(const std::string form : {"bf16", "f16", "f32", "bf16+f32", "f16+f32"})
        {
            oneEach[form + phase] = 1;
            if(form != "f32")
            {
                EXPECT_GT(baked[form + phase], 0U) << form + phase;
            }
        }
        EXPECT_EQ(baked.count("f32" + phase), 0U);
    }
    EXPECT_EQ(generic, oneEach);
}

// GenericPlanBuilder declines a graph whose schema floor is above the engine's, before
// the native matcher runs. A runtime pass-by-value epsilon raises the floor to 1.2.0,
// which this engine declares so that launch() can resolve it. A declared alignment other
// than the default raises it to 1.3.0, which stays above: such graphs are declined.
TEST(TestFlydslRmsNormPacks, UnderstandsRuntimeEpsilonGraphsButNotAlignedOnes)
{
    const auto& set = loadedSet(FLYDSL_RMSNORM.engineName);

    const auto& runtimeEpsilon
        = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(false, true, false, false, false);
    const auto& aligned
        = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(false, false, false, true, false);
    EXPECT_FALSE(set.engine.sdkVersion < runtimeEpsilon) << set.engine.sdkVersion.str();
    EXPECT_TRUE(set.engine.sdkVersion < aligned) << set.engine.sdkVersion.str();
}

TEST(TestFlydslRmsNormPacks, EveryKernelDeclaresTheNineSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_RMSNORM.engineName);

    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            // Four (pointer, descriptor) pairs: x/Desc2D, gamma/Desc1D, rstd/Desc1D,
            // y/Desc2D; then epsilon, an f32. Offsets are pinned alongside sizes because the two 4-byte
            // descriptors sit before 8-byte pointers and are padded out to their
            // alignment -- the slots sum to 72 bytes but span a segment of 80, and a
            // reader who assumed the sizes were contiguous would place y four bytes low.
            //
            // The third pair is VESTIGIAL: the instances are built store_rstd=False, so
            // nothing is written through it, and launch() fills it with gamma's pointer
            // and descriptor rather than leaving it null. A rebuild with store_rstd=True
            // would produce this same nine-slot signature and silently write the
            // statistic over gamma. That aliasing is invisible in the signature, so this
            // is where it gets caught: if a regenerated manifest ever changes this table,
            // re-read launch() before updating the expectation.
            static const std::array<std::pair<const char*, uint32_t>, 9> s_slots{
                {{"global_buffer", 8},
                 {"by_value", 16},
                 {"global_buffer", 8},
                 {"by_value", 4},
                 {"global_buffer", 8},
                 {"by_value", 4},
                 {"global_buffer", 8},
                 {"by_value", 16},
                 {"by_value", 4}}};
            static const std::array<uint32_t, 9> s_offsets{0, 8, 24, 32, 40, 48, 56, 64, 80};

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
