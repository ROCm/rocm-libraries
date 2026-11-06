// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdParityCases.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdVerdictTable.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdParityGraphs.cpp
 * @brief The host half of the split backward engine's parity cases (FlydslSdpaBwdParityCases.hpp),
 *        a census over the gfx11 shard the census entry names.
 *
 * On the architecture the entry names (HIPDNN_TEST_EXPECTED_ARCH, a gfx11 member) the split
 * graph matcher binds every served graph and each slot has a shipped object that admits it;
 * each declined graph is declined with its rule's cause. The suite is registered with
 * hkp_register_census_tests() in src/tests/CMakeLists.txt over the gfx11 architectures the
 * split ops ship content for. The suite is compiled into the census binary whatever
 * GPU_TARGETS holds, because a family job without gfx11 targets can supply the binary a gfx11
 * entry runs; it skips, with its reason, when the shard holds no gfx11 pack, which a census
 * entry reports as a failure. A shard that holds gfx11 packs but no object a graph needs
 * fails that case. The device half, which launches the same cases, is
 * TestFlydslSdpaBwdParity.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_parity;

constexpr std::array<FlydslSdpaBwdSplitSlot, 3> SLOT_ORDER{
    FlydslSdpaBwdSplitSlot::PREP, FlydslSdpaBwdSplitSlot::DKDV, FlydslSdpaBwdSplitSlot::DQ};

/// The objects the shard ships, per slot, as the engine's matchers read them.
std::array<std::vector<KernelDefinition>, 3> shippedKernels()
{
    std::array<std::vector<KernelDefinition>, 3> kernels;
    const auto& set = loadedSet(SPLIT_ENGINE_NAME);
    for(size_t index = 0; index < SLOT_ORDER.size(); ++index)
    {
        for(const auto& slot : set.slots)
        {
            if(slot.name != slotName(SLOT_ORDER[index]))
            {
                continue;
            }
            for(const auto& pack : slot.packs)
            {
                for(const auto& kernel : pack.kernels)
                {
                    kernels[index].push_back(definitionOf(pack, kernel));
                }
            }
        }
    }
    return kernels;
}

/// Whether the loaded split engine set holds a pack stamped for a gfx11 architecture.
bool shardHoldsGfx11Content()
{
    const auto* set = findLoadedSet(SPLIT_ENGINE_NAME);
    if(set == nullptr)
    {
        return false;
    }
    return std::any_of(set->slots.begin(), set->slots.end(), [](const auto& slot) {
        return std::any_of(slot.packs.begin(), slot.packs.end(), [](const auto& pack) {
            return std::any_of(pack.arch.begin(), pack.arch.end(), [](const std::string& arch) {
                return arch.rfind("gfx11", 0) == 0;
            });
        });
    });
}

class TestFlydslSdpaBwdParityGraphs : public ::testing::TestWithParam<size_t>
{
protected:
    void SetUp() override
    {
        _arch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
        ASSERT_FALSE(_arch.empty())
            << "HIPDNN_TEST_EXPECTED_ARCH is unset, so this census has no expectation to check "
               "against; the architecture is supplied by the census entry.";
        ASSERT_EQ(_arch.rfind("gfx11", 0), 0U)
            << "the parity cases follow the gfx11 verdict table; " << _arch
            << " is not a gfx11 architecture";

        if(!shardHoldsGfx11Content())
        {
            GTEST_SKIP() << "the loaded split engine set holds no gfx11 pack, so the gfx11 "
                            "parity cases have nothing to check";
        }
    }

    const std::string& arch() const
    {
        return _arch;
    }

private:
    std::string _arch;
};

TEST_P(TestFlydslSdpaBwdParityGraphs, IsServedOrDeclinedAsTheMatcherContractStates)
{
    const auto& parity = parityCases()[GetParam()];
    const GraphFixture fixture(buildSdpaBwdGraph(parity.spec), devicePropertiesFor(arch()));
    const auto context = fixture.context();
    const auto bound = matchesGraph(splitSymbols(FlydslSdpaBwdSplitSlot::PREP), context);

    if(!parity.rule.empty())
    {
        EXPECT_FALSE(bound.has_value())
            << parity.name << ": graph_match accepts a graph that " << parity.rule << " declines";
        const auto shared
            = flydsl_sdpa_bwd::matchSdpaBackward(context, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS);
        const auto* decline = std::get_if<flydsl_sdpa_bwd::Decline>(&shared);
        ASSERT_NE(decline, nullptr) << parity.name << ": the shared matcher serves the graph";
        EXPECT_EQ(decline->cause, parity.cause)
            << parity.rule << ": " << decline->cause << " (" << decline->detail << ")";
        return;
    }

    ASSERT_TRUE(bound.has_value()) << parity.name << ": graph_match declines the graph";
    const auto shipped = shippedKernels();
    for(size_t index = 0; index < SLOT_ORDER.size(); ++index)
    {
        const auto symbols = splitSymbols(SLOT_ORDER[index]);
        ASSERT_FALSE(shipped[index].empty())
            << parity.name << ": the shard for " << arch() << " ships no "
            << slotName(SLOT_ORDER[index]) << " object to match the graph against";
        const auto serving
            = std::count_if(shipped[index].begin(), shipped[index].end(), [&](const auto& kernel) {
                  return matchesKernel(symbols, context, kernel, *bound);
              });
        EXPECT_GE(serving, 1) << parity.name << ": no shipped " << slotName(SLOT_ORDER[index])
                              << " object serves the graph";
    }
}

INSTANTIATE_TEST_SUITE_P(, TestFlydslSdpaBwdParityGraphs, parityIndices(), parityName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
