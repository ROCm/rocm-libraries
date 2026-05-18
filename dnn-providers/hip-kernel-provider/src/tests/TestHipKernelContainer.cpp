// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <gtest/gtest.h>

#include "HipKernelContainer.hpp"
#include "HipKernelHandle.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hip_kernel_provider;

constexpr uint32_t EXPECTED_ENGINES = 0

#ifdef HIPDNN_ENGINE_ASM_SDPA
                                      + 1
#endif

#ifdef HIPDNN_ENGINE_HIP_KERNEL
                                      + 1
#endif

    // Add more blocks like this as more engines are implemented
    ;

TEST(TestHipKernelContainer, ConstructsSuccessfully)
{
    const HipKernelContainer container;
}

TEST(TestHipKernelContainer, CopyEngineIdsReturnsExpectedEngineCount)
{
    uint32_t numEngines = 0;
    auto totalEngines = HipKernelContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, EXPECTED_ENGINES);
    EXPECT_EQ(numEngines, EXPECTED_ENGINES);
}

TEST(TestHipKernelContainer, CopyEngineIdsWithBufferContainsHipKernelEngineId)
{
#ifndef HIPDNN_ENGINE_HIP_KERNEL
    GTEST_SKIP();
#else
    std::array<int64_t, EXPECTED_ENGINES> engineIds = {};
    uint32_t numEngines = 0;
    auto totalEngines
        = HipKernelContainer::copyEngineIds(engineIds.data(), EXPECTED_ENGINES, numEngines);

    EXPECT_EQ(totalEngines, EXPECTED_ENGINES);
    EXPECT_EQ(numEngines, EXPECTED_ENGINES);

    bool containsHipKernelEngine = false;
    for(const int64_t engine : engineIds)
    {
        containsHipKernelEngine |= (engine == hipdnn_data_sdk::utilities::HIP_KERNEL_ENGINE_ID);
    }
    EXPECT_EQ(containsHipKernelEngine, true);
#endif
}

TEST(TestHipKernelContainer, GetEngineManagerReturnsValidReference)
{
    HipKernelContainer container;
    auto& engineManager = container.getEngineManager();

    (void)engineManager;
}

TEST(TestHipKernelContainer, GetApplicableEngineIdsSdpaGraph)
{
    SKIP_IF_NO_DEVICES();
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    HipKernelHandle handle;
    auto deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }
    HipKernelContainer container;
    auto& engineManager = container.getEngineManager();

    const std::vector<int64_t> dims{4, 8, 256, 128};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto graph = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims, strides, dims, strides, dims, strides, dims, strides, DataType::BFLOAT16);
    auto graphBuffer = graph.Release();

    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuffer.data(), graphBuffer.size());

    auto applicableEngines = engineManager.getApplicableEngineIds(handle, graphWrapper);

#ifdef HIPDNN_ENGINE_ASM_SDPA
    ASSERT_EQ(applicableEngines.size(), 1);
    EXPECT_EQ(applicableEngines.front(), asm_sdpa_engine::AsmSdpaEngine::staticId());
#else
    EXPECT_TRUE(applicableEngines.empty());
#endif
}

TEST(TestHipKernelContainer, GetAllEngineIds)
{
    HipKernelContainer container;
    auto& engineManager = container.getEngineManager();

    auto allEngines = engineManager.getAllEngineIds();

    ASSERT_EQ(allEngines.size(), EXPECTED_ENGINES);
}
