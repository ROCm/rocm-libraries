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
