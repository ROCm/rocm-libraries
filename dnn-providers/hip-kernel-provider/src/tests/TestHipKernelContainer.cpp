// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <gtest/gtest.h>

#include "HipKernelContainer.hpp"

using namespace hip_kernel_provider;

TEST(TestHipKernelContainer, ConstructsSuccessfully)
{
    HipKernelContainer container;
}

TEST(TestHipKernelContainer, CopyEngineIdsReturnsZeroEngines)
{
    uint32_t numEngines = 0;
    auto totalEngines = HipKernelContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, 0u);
    EXPECT_EQ(numEngines, 0u);
}

TEST(TestHipKernelContainer, CopyEngineIdsWithBufferReturnsZero)
{
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;
    auto totalEngines = HipKernelContainer::copyEngineIds(engineIds.data(), 1, numEngines);

    EXPECT_EQ(totalEngines, 0u);
    EXPECT_EQ(numEngines, 0u);
}

TEST(TestHipKernelContainer, GetEngineManagerReturnsValidReference)
{
    HipKernelContainer container;
    auto& engineManager = container.getEngineManager();

    // Engine manager should exist but have no engines
    (void)engineManager;
}
