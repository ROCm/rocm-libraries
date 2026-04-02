// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <gtest/gtest.h>

#include "HipKernelContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

using namespace hip_kernel_provider;

#ifdef HIPDNN_ENGINE_ASM_SDPA
constexpr uint32_t EXPECTED_ENGINES = 2u;
#else
constexpr uint32_t EXPECTED_ENGINES = 1u;
#endif

TEST(TestHipKernelContainer, ConstructsSuccessfully)
{
    HipKernelContainer container;
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
    std::array<int64_t, EXPECTED_ENGINES> engineIds = {};
    uint32_t numEngines = 0;
    auto totalEngines
        = HipKernelContainer::copyEngineIds(engineIds.data(), EXPECTED_ENGINES, numEngines);

    EXPECT_EQ(totalEngines, EXPECTED_ENGINES);
    EXPECT_EQ(numEngines, EXPECTED_ENGINES);
    EXPECT_EQ(engineIds[0], hipdnn_data_sdk::utilities::HIP_KERNEL_ENGINE_ID);
}

TEST(TestHipKernelContainer, GetEngineManagerReturnsValidReference)
{
    HipKernelContainer container;
    auto& engineManager = container.getEngineManager();

    (void)engineManager;
}
