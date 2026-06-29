// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "RocKEContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

TEST(TestRocKEContainer, CopyEngineIdsReportsTotalWithoutCopy)
{
    uint32_t numEngines = 0;

    const auto totalEngines = rocke_client::RocKEContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
}

TEST(TestRocKEContainer, CopyEngineIdsCopiesRocKEEngineId)
{
    int64_t engineIds[1] = {0};
    uint32_t numEngines = 0;

    const auto totalEngines = rocke_client::RocKEContainer::copyEngineIds(engineIds, 1, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_EQ(engineIds[0], hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID);
}

TEST(TestRocKEContainer, EngineManagerContainsRocKEEngine)
{
    rocke_client::RocKEContainer container;

    const auto engineIds = container.getEngineManager().getAllEngineIds();

    ASSERT_EQ(engineIds.size(), 1u);
    EXPECT_EQ(engineIds[0], hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID);
}
