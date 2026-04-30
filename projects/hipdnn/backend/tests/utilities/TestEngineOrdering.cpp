// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_data_sdk/utilities/EngineNames.hpp"
#include "utilities/EngineOrdering.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend::utilities;
using namespace hipdnn_data_sdk::utilities;

TEST(TestEngineOrdering, EmptyVector)
{
    std::vector<int64_t> engineIds;
    EXPECT_NO_THROW(sortEngineIds(engineIds));
    EXPECT_TRUE(engineIds.empty());
}

TEST(TestEngineOrdering, SingleElement)
{
    std::vector<int64_t> engineIds = {MIOPEN_ENGINE_ID};
    sortEngineIds(engineIds);
    ASSERT_EQ(engineIds.size(), 1u);
    EXPECT_EQ(engineIds[0], MIOPEN_ENGINE_ID);
}

TEST(TestEngineOrdering, OnlyMiopenEngineDeterministic)
{
    std::vector<int64_t> engineIds = {MIOPEN_ENGINE_DETERMINISTIC_ID};
    sortEngineIds(engineIds);
    ASSERT_EQ(engineIds.size(), 1u);
    EXPECT_EQ(engineIds[0], MIOPEN_ENGINE_DETERMINISTIC_ID);
}

TEST(TestEngineOrdering, BothMiopenEngines)
{
    // Test with deterministic first (should be reordered)
    std::vector<int64_t> engineIds = {MIOPEN_ENGINE_DETERMINISTIC_ID, MIOPEN_ENGINE_ID};
    sortEngineIds(engineIds);
    ASSERT_EQ(engineIds.size(), 2u);
    EXPECT_EQ(engineIds[0], MIOPEN_ENGINE_ID);
    EXPECT_EQ(engineIds[1], MIOPEN_ENGINE_DETERMINISTIC_ID);

    // Test with correct order (should remain unchanged)
    engineIds = {MIOPEN_ENGINE_ID, MIOPEN_ENGINE_DETERMINISTIC_ID};
    sortEngineIds(engineIds);
    ASSERT_EQ(engineIds.size(), 2u);
    EXPECT_EQ(engineIds[0], MIOPEN_ENGINE_ID);
    EXPECT_EQ(engineIds[1], MIOPEN_ENGINE_DETERMINISTIC_ID);
}

TEST(TestEngineOrdering, MiopenEnginesWithOthers)
{
    // Create a fake "other" engine ID
    const int64_t otherEngine1 = HIPBLASLT_ENGINE_ID;
    const int64_t otherEngine2 = 999999; // Arbitrary other engine

    // Test: other, MIOPEN_ENGINE_DETERMINISTIC, MIOPEN_ENGINE, other
    std::vector<int64_t> engineIds
        = {otherEngine1, MIOPEN_ENGINE_DETERMINISTIC_ID, MIOPEN_ENGINE_ID, otherEngine2};
    sortEngineIds(engineIds);

    ASSERT_EQ(engineIds.size(), 4u);
    EXPECT_EQ(engineIds[0], MIOPEN_ENGINE_ID) << "MIOPEN_ENGINE should be first";
    // Middle two should be otherEngine1 and otherEngine2 in stable order
    EXPECT_EQ(engineIds[1], otherEngine1) << "Other engines should maintain stable order";
    EXPECT_EQ(engineIds[2], otherEngine2) << "Other engines should maintain stable order";
    EXPECT_EQ(engineIds[3], MIOPEN_ENGINE_DETERMINISTIC_ID)
        << "MIOPEN_ENGINE_DETERMINISTIC should be last";
}

TEST(TestEngineOrdering, OnlyOtherEngines)
{
    const int64_t otherEngine1 = HIPBLASLT_ENGINE_ID;
    const int64_t otherEngine2 = 888888;
    const int64_t otherEngine3 = 777777;

    std::vector<int64_t> engineIds = {otherEngine1, otherEngine2, otherEngine3};
    const std::vector<int64_t> originalOrder = engineIds;

    sortEngineIds(engineIds);

    // Order should be preserved (stable sort)
    ASSERT_EQ(engineIds.size(), 3u);
    EXPECT_EQ(engineIds, originalOrder) << "Other engines should maintain their original order";
}

TEST(TestEngineOrdering, ComplexScenario)
{
    const int64_t other1 = 111111;
    const int64_t other2 = 222222;
    const int64_t other3 = 333333;

    // Start with: other1, MIOPEN_ENGINE_DETERMINISTIC, other2, MIOPEN_ENGINE, other3
    std::vector<int64_t> engineIds
        = {other1, MIOPEN_ENGINE_DETERMINISTIC_ID, other2, MIOPEN_ENGINE_ID, other3};
    sortEngineIds(engineIds);

    ASSERT_EQ(engineIds.size(), 5u);
    EXPECT_EQ(engineIds[0], MIOPEN_ENGINE_ID) << "MIOPEN_ENGINE should be first";
    EXPECT_EQ(engineIds[1], other1) << "other1 should be second (stable order among others)";
    EXPECT_EQ(engineIds[2], other2) << "other2 should be third (stable order among others)";
    EXPECT_EQ(engineIds[3], other3) << "other3 should be fourth (stable order among others)";
    EXPECT_EQ(engineIds[4], MIOPEN_ENGINE_DETERMINISTIC_ID)
        << "MIOPEN_ENGINE_DETERMINISTIC should be last";
}

TEST(TestEngineOrdering, StableOrderPreservedForOthers)
{
    const int64_t other1 = 100;
    const int64_t other2 = 200;
    const int64_t other3 = 300;
    const int64_t other4 = 400;

    // Mix with MIOpen engines
    std::vector<int64_t> engineIds
        = {other1, other2, MIOPEN_ENGINE_DETERMINISTIC_ID, other3, MIOPEN_ENGINE_ID, other4};
    sortEngineIds(engineIds);

    ASSERT_EQ(engineIds.size(), 6u);
    EXPECT_EQ(engineIds[0], MIOPEN_ENGINE_ID);
    // The four "other" engines should be in their original relative order
    EXPECT_EQ(engineIds[1], other1);
    EXPECT_EQ(engineIds[2], other2);
    EXPECT_EQ(engineIds[3], other3);
    EXPECT_EQ(engineIds[4], other4);
    EXPECT_EQ(engineIds[5], MIOPEN_ENGINE_DETERMINISTIC_ID);
}
