// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/Container.hpp"
#include "core/Handle.hpp"
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"
#endif

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::core;

/// Engines the provider exposes: one per discovered descriptor set, and nothing else.
///
/// Read from the inventory rather than hardcoded. A literal count goes wrong the moment
/// a pack is added or an install drops one, and the failure it would report is a number
/// mismatch rather than the missing engine.
static uint32_t expectedEngines()
{
    uint32_t expected = 0;
#ifdef HIPDNN_ENGINE_HIP_FLASH2
    ++expected;
#endif
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    expected += static_cast<uint32_t>(
        hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets().size());
#endif
    return expected;
}

/// Upper bound for the fixed-size buffers below; only needs to be at least
/// expectedEngines().
constexpr uint32_t MAX_EXPECTED_ENGINES = 16;

TEST(TestContainer, ConstructsSuccessfully)
{
    const Container container;
}

TEST(TestContainer, CopyEngineIdsReturnsExpectedEngineCount)
{
    uint32_t numEngines = 0;
    auto totalEngines = Container::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, expectedEngines());
    EXPECT_EQ(numEngines, expectedEngines());
}

TEST(TestContainer, CopyEngineIdsWithBufferContainsEveryDescriptorEngine)
{
#ifndef HIPDNN_ENABLE_KERNEL_INGESTOR
    GTEST_SKIP();
#else
    std::array<int64_t, MAX_EXPECTED_ENGINES> engineIds = {};
    uint32_t numEngines = 0;
    auto totalEngines
        = Container::copyEngineIds(engineIds.data(), MAX_EXPECTED_ENGINES, numEngines);

    EXPECT_EQ(totalEngines, expectedEngines());
    EXPECT_EQ(numEngines, expectedEngines());

    // Advertising an id the constructor then fails to build is the failure this
    // guards: every set discovered must reach the copied-out list.
    for(const auto& set : hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets())
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(set.engine.name);
        EXPECT_NE(std::find(engineIds.begin(), engineIds.end(), engineId), engineIds.end())
            << "no engine id copied out for descriptor set '" << set.engine.name << "'";
    }
#endif
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
TEST(TestContainer, ExposesAnEngineForEveryDiscoveredDescriptorSet)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    // Names the ids rather than counting them. A count cannot tell a missing ingestor
    // engine from an extra one, and it cannot see the failure this is really guarding:
    // the pack table being dropped from a binary that links the provider as a static
    // archive, where an unreferenced member is discarded.
    const auto& sets = discoverDescriptorSets();

    std::vector<std::string> names;
    names.reserve(sets.size());
    for(const auto& set : sets)
    {
        names.push_back(set.engine.name);
    }
    std::sort(names.begin(), names.end());
    EXPECT_EQ(names,
              (std::vector<std::string>{"hipkernel:AsmSdpaBackward",
                                        "hipkernel:AsmSdpaForward",
                                        "hipkernel:Batchnorm",
                                        "hipkernel:ConvFwd",
                                        "hipkernel:LayernormForward",
                                        "hipkernel:Pointwise",
                                        "hipkernel:RMSnorm",
                                        "hipkernel:Resample"}));

    Container container;
    const auto allEngineIds = container.getEngineManager().getAllEngineIds();

    for(const auto& set : sets)
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(set.engine.name);
        EXPECT_NE(std::find(allEngineIds.begin(), allEngineIds.end(), engineId), allEngineIds.end())
            << "no engine for descriptor set '" << set.engine.name << "'";
    }
}
#endif

TEST(TestContainer, GetEngineManagerReturnsValidReference)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    (void)engineManager;
}

TEST(TestContainer, GetApplicableEngineIdsSdpaGraph)
{
    SKIP_IF_NO_DEVICES();
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    Handle handle;
    auto deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }
    Container container;
    auto& engineManager = container.getEngineManager();

    const std::vector<int64_t> dims{4, 8, 256, 128};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto graph = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     DataType::BFLOAT16,
                                                                     DataType::FLOAT);
    auto graphBuffer = graph.Release();

    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuffer.data(), graphBuffer.size());

    auto applicableEngines = engineManager.getApplicableEngineIds(handle, graphWrapper);

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    // The forward SDPA graph is served by the descriptor-backed engine, so the claim is
    // about the id its UED name hashes to rather than a compile-time constant.
    ASSERT_EQ(applicableEngines.size(), 1);
    EXPECT_EQ(applicableEngines.front(),
              hipdnn_data_sdk::utilities::engineNameToId("hipkernel:AsmSdpaForward"));
#else
    EXPECT_TRUE(applicableEngines.empty());
#endif
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
TEST(TestContainer, GetApplicableEngineIdsPointwiseAddGraph)
{
    // Applicability is device-resolved: with no device, matchers decline.
    SKIP_IF_NO_DEVICES();

    using namespace hip_kernel_provider::kernel_ingestor_engine;
    using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

    Handle handle;
    Container container;
    auto& engineManager = container.getEngineManager();

    const auto graph = buildPointwiseGraph();
    const auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graph.GetBufferPointer(), graph.GetSize());

    auto applicableEngines = engineManager.getApplicableEngineIds(handle, graphWrapper);

    EXPECT_NE(std::find(applicableEngines.begin(),
                        applicableEngines.end(),
                        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)),
              applicableEngines.end());
}
#endif

TEST(TestContainer, GetAllEngineIds)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    auto allEngines = engineManager.getAllEngineIds();

    ASSERT_EQ(allEngines.size(), expectedEngines());
}
