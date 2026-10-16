// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestGfx950AttentionUnified2DPacks.cpp
 * @brief A census of what hipkernel:Gfx950AttentionUnified2D actually loads, read through
 *        discoverDescriptorSets(), which runs the provider's real typed registration.
 *
 * This file belongs to hip_kernel_provider_census_tests, not to hip_kernel_provider_tests:
 * every case needs a descriptor shard and the census environment.
 * Registered via hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

struct ExpectedInventory
{
    std::set<std::string> packNames;
    std::set<std::string> kernelNames;
};

constexpr const char* ARCH_WILDCARD = "*";

const std::map<std::string, ExpectedInventory>& expectedByArch()
{
    static const std::map<std::string, ExpectedInventory> s_expected = {
        {"gfx950",
         ExpectedInventory{
             {"hipkernel:gfx950_attention_unified2_d"},
             {
                 "attention_unified_2d.bf16_d128_hq10_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq128_kv128_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq128_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq12_kv12_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq16_kv16_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq16_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq20_kv20_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq24_kv24_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq28_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq32_kv32_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq32_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq32_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq40_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq40_kv40_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq40_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq48_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq4_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq64_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq64_kv64_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq64_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq8_kv1_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq8_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq8_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq96_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d128_hq9_kv9_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq10_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq128_kv128_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq128_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq12_kv12_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq16_kv16_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq16_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq20_kv20_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq24_kv24_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq28_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq32_kv32_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq32_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq32_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq40_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq40_kv40_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq40_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq48_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq4_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq64_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq64_kv64_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq64_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq8_kv1_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq8_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq8_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq96_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d256_hq9_kv9_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq10_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq128_kv128_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq128_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq12_kv12_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq16_kv16_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq16_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq20_kv20_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq24_kv24_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq28_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq32_kv32_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq32_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq32_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq40_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq40_kv40_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq40_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq48_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq4_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq64_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq64_kv64_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq64_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq8_kv1_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq8_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq8_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq96_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.bf16_d64_hq9_kv9_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq10_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq128_kv128_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq128_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq12_kv12_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq16_kv16_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq16_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq20_kv20_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq24_kv24_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq28_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq32_kv32_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq32_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq32_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq40_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq40_kv40_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq40_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq48_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq4_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq64_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq64_kv64_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq64_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq8_kv1_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq8_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq8_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq96_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d128_hq9_kv9_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq10_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq128_kv128_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq128_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq12_kv12_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq16_kv16_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq16_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq20_kv20_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq24_kv24_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq28_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq32_kv32_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq32_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq32_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq40_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq40_kv40_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq40_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq48_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq4_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq64_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq64_kv64_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq64_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq8_kv1_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq8_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq8_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq96_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d256_hq9_kv9_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq10_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq128_kv128_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq128_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq12_kv12_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq16_kv16_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq16_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq20_kv20_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq24_kv24_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq28_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq32_kv32_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq32_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq32_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq40_kv10_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq40_kv40_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq40_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq48_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq4_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq64_kv4_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq64_kv64_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq64_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq8_kv1_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq8_kv2_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq8_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq96_kv8_bs16_t16_nw1_mw16.gfx950",
                 "attention_unified_2d.fp16_d64_hq9_kv9_bs16_t16_nw1_mw16.gfx950",
             }}},
    };
    return s_expected;
}

constexpr ingestor::KernelSourceKind EXPECTED_SOURCE_KIND = ingestor::KernelSourceKind::KPACK;

bool coversArch(const std::string& declared, const std::string& arch)
{
    return declared == ARCH_WILDCARD || declared == arch;
}

class TestGfx950AttentionUnified2DPacks : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _arch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");

        const auto& allExpected = expectedByArch();
        auto it = allExpected.find(_arch);
        if(it == allExpected.end())
        {
            it = allExpected.find(std::string(ARCH_WILDCARD));
        }
        if(it == allExpected.end())
        {
            GTEST_SKIP() << "No expected inventory for arch '" << _arch << "'.";
        }
        _expected = &it->second;

        const auto& allSets = discoverDescriptorSets();
        auto setIt = std::find_if(allSets.begin(), allSets.end(), [](const auto& s) {
            return s.engine.name == GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME;
        });
        if(setIt == allSets.end())
        {
            GTEST_SKIP() << "hipkernel:Gfx950AttentionUnified2D was not loaded.";
        }
        _set = &*setIt;
    }

    std::string _arch;
    const ExpectedInventory* _expected = nullptr;
    const ingestor::DescriptorSet* _set = nullptr;
};

TEST_F(TestGfx950AttentionUnified2DPacks, LoadsExactlyTheEmittedPacks)
{
    std::set<std::string> loaded;
    for(const auto& pack : _set->packs)
    {
        loaded.insert(pack.name);
    }
    EXPECT_EQ(loaded, _expected->packNames);
}

TEST_F(TestGfx950AttentionUnified2DPacks, LoadsExactlyTheEmittedKernels)
{
    std::set<std::string> loaded;
    for(const auto& pack : _set->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            loaded.insert(kernel.name);
        }
    }
    EXPECT_EQ(loaded, _expected->kernelNames);
}

TEST_F(TestGfx950AttentionUnified2DPacks, EveryLoadedDescriptorCoversTheRequestedArch)
{
    for(const auto& pack : _set->packs)
    {
        for(const auto& arch : pack.arch)
        {
            EXPECT_TRUE(coversArch(arch, _arch)) << "Pack '" << pack.name << "' declares arch '"
                                                 << arch << "', expected '" << _arch << "'";
        }
    }
}

TEST_F(TestGfx950AttentionUnified2DPacks, EveryKernelDeclaresTheEmittedRuntimeSourceKind)
{
    for(const auto& pack : _set->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            EXPECT_EQ(kernel.source.kind, EXPECTED_SOURCE_KIND)
                << "Kernel '" << kernel.name << "' has unexpected source kind";
        }
    }
}

TEST_F(TestGfx950AttentionUnified2DPacks, DeclaresTheConfiguredSdkVersion)
{
    using hipdnn_data_sdk::utilities::Version;
    EXPECT_EQ(_set->engine.sdkVersion, hipdnn_data_sdk::utilities::Version{std::string("1.0.0")});
}

TEST_F(TestGfx950AttentionUnified2DPacks, ExposesTheConfiguredKnobs)
{
    const std::set<std::string> expected{"tile_size", "num_warps", "block_m_per_warp"};
    const std::set<std::string> actual(_set->engine.knobs.begin(), _set->engine.knobs.end());
    EXPECT_EQ(actual, expected);
}

TEST_F(TestGfx950AttentionUnified2DPacks, RanksThroughItsRegisteredScoreSymbol)
{
    constexpr std::string_view EXPECTED_SCORE_SYMBOL
        = "hipkernel.gfx950_attention_unified2_d.score";
    ASSERT_TRUE(_set->heuristic.has_value());
    EXPECT_EQ(_set->heuristic->payload, std::string(EXPECTED_SCORE_SYMBOL));
    EXPECT_TRUE(ingestor::ScoreRegistry::isRegistered(std::string(EXPECTED_SCORE_SYMBOL)));
}

TEST_F(TestGfx950AttentionUnified2DPacks, CarriesNoGraphScopedMatcher)
{
    EXPECT_FALSE(_set->engine.graphMatchNativeSymbol.empty());
    const auto graphScoped
        = std::count_if(_set->matchers.begin(), _set->matchers.end(), [](const auto& matcher) {
              return matcher.scope == ingestor::MatchScope::GRAPH;
          });
    EXPECT_EQ(graphScoped, 0);
}

TEST_F(TestGfx950AttentionUnified2DPacks, CarriesExactlyOneSharedKernelScopedMatcher)
{
    const auto kernelScoped
        = std::count_if(_set->matchers.begin(), _set->matchers.end(), [](const auto& matcher) {
              return matcher.scope == ingestor::MatchScope::KERNEL;
          });
    EXPECT_EQ(kernelScoped, 1);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
