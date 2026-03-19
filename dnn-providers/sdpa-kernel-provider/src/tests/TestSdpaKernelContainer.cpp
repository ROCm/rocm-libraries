// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "SdpaKernelContainer.hpp"
#include "SdpaKernelEngine.hpp"

using namespace sdpa_kernel_provider;

TEST(TestSdpaKernelContainer, ConstructsSuccessfully)
{
    SdpaKernelContainer container;
}

TEST(TestSdpaKernelContainer, CopyEngineIdsReturnsZeroEngines)
{
    uint32_t numEngines = 1;
    auto totalEngines = SdpaKernelContainer::copyEngineIds(nullptr, 1, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
}

TEST(TestSdpaKernelContainer, CopyEngineIdsWithBufferReturnsOne)
{
    std::array<int64_t, 2> engineIds = {0, 0};
    uint32_t numEngines = 1;
    auto totalEngines = SdpaKernelContainer::copyEngineIds(engineIds.data(), 2, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
}

TEST(TestSdpaKernelContainer, GetEngineManagerReturnsValidReference)
{
    SdpaKernelContainer container;
    auto& engineManager = container.getEngineManager();

    // Engine manager should exist but have no engines
    (void)engineManager;
}

TEST(TestSdpaKernelContainer, GetApplicableEngineIdsSupportedGraph)
{
    SdpaKernelHandle handle;
    SdpaKernelContainer container;
    auto& engineManager = container.getEngineManager();

    auto graph = hipdnn_test_sdk::utilities::createValidSdpaFpropGraph();
    auto graphBuffer = graph.Release();

    auto graphWrapper = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(graphBuffer.data(),
                                                                            graphBuffer.size());

    auto applicableEngines = engineManager.getApplicableEngineIds(handle, graphWrapper);

    ASSERT_EQ(applicableEngines.size(), 1);
    EXPECT_EQ(applicableEngines.front(), SdpaKernelEngine::staticId());
}

TEST(TestSdpaKernelContainer, GetAllEngineIds)
{
    SdpaKernelHandle handle;
    SdpaKernelContainer container;
    auto& engineManager = container.getEngineManager();

    auto allEngines = engineManager.getAllEngineIds();

    ASSERT_EQ(allEngines.size(), 1);
    EXPECT_EQ(allEngines.front(), SdpaKernelEngine::staticId());
}
