// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <gtest/gtest.h>

#include "SdpaKernelContainer.hpp"

using namespace sdpa_kernel_provider;

TEST(TestSdpaKernelContainer, ConstructsSuccessfully)
{
    SdpaKernelContainer container;
}

TEST(TestSdpaKernelContainer, CopyEngineIdsReturnsZeroEngines)
{
    uint32_t numEngines = 0;
    auto totalEngines = SdpaKernelContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, 0u);
    EXPECT_EQ(numEngines, 0u);
}

TEST(TestSdpaKernelContainer, CopyEngineIdsWithBufferReturnsZero)
{
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;
    auto totalEngines = SdpaKernelContainer::copyEngineIds(engineIds.data(), 1, numEngines);

    EXPECT_EQ(totalEngines, 0u);
    EXPECT_EQ(numEngines, 0u);
}

TEST(TestSdpaKernelContainer, GetEngineManagerReturnsValidReference)
{
    SdpaKernelContainer container;
    auto& engineManager = container.getEngineManager();

    // Engine manager should exist but have no engines
    (void)engineManager;
}
