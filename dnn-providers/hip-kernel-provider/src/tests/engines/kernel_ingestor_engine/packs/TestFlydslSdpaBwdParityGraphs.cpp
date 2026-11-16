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
 *        a census over the gfx11, gfx12, gfx90a or gfx942 shard the census entry names.
 *
 * On the architecture the entry names (HIPDNN_TEST_EXPECTED_ARCH, a gfx11 or gfx12 member,
 * gfx90a or gfx942) the split graph matcher binds every served graph and each slot has a shipped
 * object that admits it; each declined graph is declined with its rule's cause. The suite is
 * registered with hkp_register_census_tests() in src/tests/CMakeLists.txt over the gfx11, gfx12,
 * gfx90a and gfx942 architectures the split ops ship content for. The suite is compiled into
 * the census binary whatever GPU_TARGETS holds, because a family job without the entry's
 * targets can supply the binary the entry runs; it skips, with its reason, when the shard
 * holds no pack for the entry's family, which a census entry reports as a failure. A shard
 * that holds the family's packs but no object a graph needs fails that case. The gfx12 wide
 * head dim cases (gfx12WideParityCases()) are a suite of their own,
 * TestFlydslSdpaBwdParityGraphsGfx12Wide, registered for the gfx12 members only. The device
 * half, which launches the same cases, is TestFlydslSdpaBwdParity.cpp.
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

/// Whether @p arch is one the parity cases are held to: the gfx11 members share one verdict
/// table, as do the gfx12 members; gfx90a and gfx942 have tables of their own.
bool isParityArch(const std::string& arch)
{
    return arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0 || arch == "gfx90a"
           || arch == "gfx942";
}

/// Whether @p arch belongs to the family of @p entryArch: any gfx11 member, any gfx12 member,
/// or that exact architecture.
bool sameParityFamily(const std::string& entryArch, const std::string& arch)
{
    for(const char* family : {"gfx11", "gfx12"})
    {
        if(entryArch.rfind(family, 0) == 0)
        {
            return arch.rfind(family, 0) == 0;
        }
    }
    return arch == entryArch;
}

/// Whether the loaded split engine set holds a pack stamped for the architecture family of
/// @p entryArch (see sameParityFamily).
bool shardHoldsContentFor(const std::string& entryArch)
{
    const auto* set = findLoadedSet(SPLIT_ENGINE_NAME);
    if(set == nullptr)
    {
        return false;
    }
    return std::any_of(set->slots.begin(), set->slots.end(), [&](const auto& slot) {
        return std::any_of(slot.packs.begin(), slot.packs.end(), [&](const auto& pack) {
            return std::any_of(pack.arch.begin(), pack.arch.end(), [&](const std::string& arch) {
                return sameParityFamily(entryArch, arch);
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
        ASSERT_TRUE(isParityArch(_arch))
            << "the parity cases follow the gfx11, gfx12, gfx90a and gfx942 verdict tables; "
            << _arch << " is none of them";

        if(!shardHoldsContentFor(_arch))
        {
            GTEST_SKIP() << "the loaded split engine set holds no " << _arch
                         << " pack, so the parity cases have nothing to check";
        }
    }

    const std::string& arch() const
    {
        return _arch;
    }

private:
    std::string _arch;
};

/// @p parity checked against the loaded shard of @p arch: a served graph binds and every slot
/// has an object that admits it; a declined one is declined with its rule's cause.
void expectServedOrDeclined(const ParityCase& parity, const std::string& arch)
{
    const GraphFixture fixture(buildSdpaBwdGraph(parity.spec), devicePropertiesFor(arch));
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
            << parity.name << ": the shard for " << arch << " ships no "
            << slotName(SLOT_ORDER[index]) << " object to match the graph against";
        const auto serving
            = std::count_if(shipped[index].begin(), shipped[index].end(), [&](const auto& kernel) {
                  return matchesKernel(symbols, context, kernel, *bound);
              });
        EXPECT_GE(serving, 1) << parity.name << ": no shipped " << slotName(SLOT_ORDER[index])
                              << " object serves the graph";
    }
}

TEST_P(TestFlydslSdpaBwdParityGraphs, IsServedOrDeclinedAsTheMatcherContractStates)
{
    expectServedOrDeclined(parityCases()[GetParam()], arch());
}

INSTANTIATE_TEST_SUITE_P(, TestFlydslSdpaBwdParityGraphs, parityIndices(), parityName);

/// The head dims past 256 only the gfx12 catalog has rungs for (gfx12WideParityCases()): the
/// suite skips on any other architecture, and its census registration is for the gfx12 members.
class TestFlydslSdpaBwdParityGraphsGfx12Wide : public TestFlydslSdpaBwdParityGraphs
{
protected:
    void SetUp() override
    {
        TestFlydslSdpaBwdParityGraphs::SetUp();
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        if(arch().rfind("gfx12", 0) != 0)
        {
            GTEST_SKIP() << "the wide head dim cases follow the gfx12 catalog; " << arch()
                         << " is not a gfx12 architecture";
        }
    }
};

TEST_P(TestFlydslSdpaBwdParityGraphsGfx12Wide, IsServedOrDeclinedAsTheMatcherContractStates)
{
    expectServedOrDeclined(gfx12WideParityCases()[GetParam()], arch());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaBwdParityGraphsGfx12Wide,
                         gfx12WideParityIndices(),
                         gfx12WideParityName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
