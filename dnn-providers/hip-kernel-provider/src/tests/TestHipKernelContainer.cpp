// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <gtest/gtest.h>

#include "HipKernelContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

using namespace hip_kernel_provider;

TEST(TestHipKernelContainer, ConstructsSuccessfully)
{
    HipKernelContainer container;
}

TEST(TestHipKernelContainer, CopyEngineIdsReturnsOneEngine)
{
    uint32_t numEngines = 0;
    auto totalEngines = HipKernelContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
}

TEST(TestHipKernelContainer, CopyEngineIdsWithBufferReturnsHipKernelEngineId)
{
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;
    auto totalEngines = HipKernelContainer::copyEngineIds(engineIds.data(), 1, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_EQ(engineIds[0], hipdnn_data_sdk::utilities::HIP_KERNEL_ENGINE_ID);
}

TEST(TestHipKernelContainer, GetEngineManagerReturnsValidReference)
{
    HipKernelContainer container;
    auto& engineManager = container.getEngineManager();

    (void)engineManager;
}
