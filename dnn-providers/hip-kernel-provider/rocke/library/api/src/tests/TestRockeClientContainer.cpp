// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "RockeClientContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

TEST(TestRockeClientContainer, CopyEngineIdsReportsTotalWithoutCopy)
{
    uint32_t numEngines = 0;

    const auto totalEngines
        = rocke_client::RockeClientContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
}

TEST(TestRockeClientContainer, CopyEngineIdsCopiesRockeClientEngineId)
{
    int64_t engineIds[1] = {0};
    uint32_t numEngines = 0;

    const auto totalEngines
        = rocke_client::RockeClientContainer::copyEngineIds(engineIds, 1, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_EQ(engineIds[0], hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID);
}

TEST(TestRockeClientContainer, CopyEngineIdsHonorsZeroCapWithoutWriting)
{
    // maxEngines == 0 must report the total and never dereference engineIds.
    uint32_t numEngines = 99;

    const auto totalEngines
        = rocke_client::RockeClientContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, 1u);
    EXPECT_EQ(numEngines, 1u);
}

TEST(TestRockeClientContainer, EngineManagerContainsRockeClientEngine)
{
    rocke_client::RockeClientContainer container;

    const auto engineIds = container.getEngineManager().getAllEngineIds();

    ASSERT_EQ(engineIds.size(), 1u);
    EXPECT_EQ(engineIds[0], hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID);
}
