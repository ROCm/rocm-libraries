// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestHipConvFwdRdnaPacks.cpp
 * @brief A census of what hipkernel:hip_conv_fwd_rdna loads from one architecture's shard,
 *        read through discoverDescriptorSets(), which registers the provider's native
 *        symbols before validating and loading, so a pack whose symbols do not register
 *        drops out here rather than at first dispatch on a device.
 *
 * The expected inventory is the variant grid of descriptors/hip/conv_rdna/
 * generate_descriptors.py: fma_igemm for every dtype x layout x channel tile.
 *
 * Census binary only: every case needs the shard and HIPDNN_TEST_EXPECTED_ARCH that the
 * hkp_register_census_tests() entry supplies.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

constexpr const char* ENGINE_NAME = "hipkernel:hip_conv_fwd_rdna";

/// The (dtype, layout, tile) of every kernel the bundle ships, and its name.
std::set<std::tuple<std::string, std::string, int64_t>> expectedVariants()
{
    std::set<std::tuple<std::string, std::string, int64_t>> variants;
    for(const char* dtype : {"FLOAT", "HALF", "BFLOAT16"})
    {
        for(const char* layout : {"CHANNELS_FIRST", "CHANNELS_LAST"})
        {
            for(const int64_t tile : {32, 64, 128})
            {
                variants.emplace(dtype, layout, tile);
            }
        }
    }
    return variants;
}

class TestHipConvFwdRdnaPacks : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _arch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
        ASSERT_FALSE(_arch.empty())
            << "HIPDNN_TEST_EXPECTED_ARCH is unset, so this census has no expectation to "
               "check against.";

        const auto& sets = discoverDescriptorSets();
        const auto match = std::find_if(sets.begin(), sets.end(), [](const auto& set) {
            return set.engine.name == ENGINE_NAME;
        });
        ASSERT_NE(match, sets.end())
            << "descriptor set '" << ENGINE_NAME << "' did not load. Either its descriptors "
            << "were not packed for '" << _arch << "', or its pack failed to register its "
            << "native symbols and every descriptor naming one of them was dropped.";
        _set = &*match;
    }

    std::string _arch;
    const ingestor::DescriptorSet* _set = nullptr;
};

TEST_F(TestHipConvFwdRdnaPacks, LoadsOnePackShippingEveryVariantOnce)
{
    ASSERT_EQ(_set->packs.size(), 1U);
    const auto& pack = _set->packs.front();
    EXPECT_EQ(pack.name, ENGINE_NAME);

    std::set<std::tuple<std::string, std::string, int64_t>> loaded;
    for(const auto& kernel : pack.kernels)
    {
        EXPECT_EQ(std::get<std::string>(kernel.metadata.at("kernel_family")), "fma_igemm")
            << kernel.name;
        const auto variant = std::make_tuple(std::get<std::string>(kernel.metadata.at("dtype")),
                                             std::get<std::string>(kernel.metadata.at("layout")),
                                             std::get<int64_t>(kernel.metadata.at("tile_n")));
        EXPECT_TRUE(loaded.insert(variant).second) << "duplicate variant " << kernel.name;
    }
    // Set equality, not a count: a missing variant and an extra one cancel in a count.
    EXPECT_EQ(loaded, expectedVariants());
}

TEST_F(TestHipConvFwdRdnaPacks, EveryLoadedDescriptorCoversTheRequestedArch)
{
    for(const auto& pack : _set->packs)
    {
        EXPECT_TRUE(ingestor::archSupports(pack.arch, _arch))
            << "pack '" << pack.name << "' does not cover '" << _arch << "'";
        for(const auto& kernel : pack.kernels)
        {
            EXPECT_TRUE(ingestor::archSupports(kernel.arch, _arch))
                << "kernel '" << kernel.name << "' does not cover '" << _arch << "'";
        }
    }
}

TEST_F(TestHipConvFwdRdnaPacks, EveryKernelIsAPackedConvFwdFmaIgemmObject)
{
    for(const auto& pack : _set->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            EXPECT_EQ(kernel.source.kind, ingestor::KernelSourceKind::KPACK) << kernel.name;
            EXPECT_EQ(kernel.source.symbol, "ConvFwdFmaIgemm") << kernel.name;
        }
    }
}

TEST_F(TestHipConvFwdRdnaPacks, ExposesNoKnobs)
{
    EXPECT_TRUE(_set->engine.knobs.empty());
}

TEST_F(TestHipConvFwdRdnaPacks, RanksAndDispatchesThroughRegisteredSymbols)
{
    ASSERT_TRUE(_set->heuristic.has_value());
    EXPECT_EQ(_set->heuristic->nativeSymbol, "hipkernel.hip_conv_fwd_rdna.score");
    EXPECT_TRUE(ingestor::ScoreRegistry::isRegistered("hipkernel.hip_conv_fwd_rdna.score"));
    ASSERT_EQ(_set->dispatches.size(), 1U);
    EXPECT_EQ(_set->dispatches.front().dispatchSymbol, "hipkernel.hip_conv_fwd_rdna.dispatch");
}

/// A single-pack engine: its graph_match admits the node and validates the shape, and one
/// kernel-scoped matcher picks the kernel.
TEST_F(TestHipConvFwdRdnaPacks, CarriesOnlyOneKernelScopedMatcher)
{
    EXPECT_EQ(_set->engine.graphMatchNativeSymbol, "hipkernel.hip_conv_fwd_rdna.graph_match");
    const auto scoped = [this](ingestor::MatchScope scope) {
        return std::count_if(_set->matchers.begin(),
                             _set->matchers.end(),
                             [scope](const auto& matcher) { return matcher.scope == scope; });
    };
    EXPECT_EQ(scoped(ingestor::MatchScope::GRAPH), 0);
    EXPECT_EQ(scoped(ingestor::MatchScope::KERNEL), 1);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
