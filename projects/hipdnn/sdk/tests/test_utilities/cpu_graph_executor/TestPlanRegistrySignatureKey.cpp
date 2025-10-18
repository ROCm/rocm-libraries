// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferenceSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanRegistrySignatureKey.hpp>
#include <unordered_map>

using namespace hipdnn_sdk::test_utilities;
using hipdnn_sdk::data_objects::DataType;

TEST(TestPlanRegistrySignatureKey, HashAndEqualityFwdInference)
{
    BatchnormFwdInferenceSignatureKey key1(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    BatchnormFwdInferenceSignatureKey key2(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    BatchnormFwdInferenceSignatureKey key3(DataType::HALF, DataType::HALF, DataType::HALF);

    PlanRegistrySignatureKey v1 = key1;
    PlanRegistrySignatureKey v2 = key2;
    PlanRegistrySignatureKey v3 = key3;

    PlanRegistrySignatureKeyHash hasher;
    PlanRegistrySignatureKeyEqual eq;

    EXPECT_EQ(hasher(v1), hasher(v2));
    EXPECT_TRUE(eq(v1, v2));
    EXPECT_FALSE(eq(v1, v3));
    EXPECT_NE(hasher(v1), hasher(v3));
}

TEST(TestPlanRegistrySignatureKey, HashAndEqualityBwd)
{
    BatchnormBwdSignatureKey key1(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    BatchnormBwdSignatureKey key2(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    BatchnormBwdSignatureKey key3(DataType::HALF, DataType::HALF, DataType::HALF);

    PlanRegistrySignatureKey v1 = key1;
    PlanRegistrySignatureKey v2 = key2;
    PlanRegistrySignatureKey v3 = key3;

    PlanRegistrySignatureKeyHash hasher;
    PlanRegistrySignatureKeyEqual eq;

    EXPECT_EQ(hasher(v1), hasher(v2));
    EXPECT_TRUE(eq(v1, v2));
    EXPECT_FALSE(eq(v1, v3));
    EXPECT_NE(hasher(v1), hasher(v3));
}

TEST(TestPlanRegistrySignatureKey, CrossTypeEquality)
{
    BatchnormFwdInferenceSignatureKey fwdKey(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    BatchnormBwdSignatureKey bwdKey(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    PlanRegistrySignatureKey vFwd = fwdKey;
    PlanRegistrySignatureKey vBwd = bwdKey;

    PlanRegistrySignatureKeyEqual eq;
    EXPECT_FALSE(eq(vFwd, vBwd));
}

TEST(TestPlanRegistrySignatureKey, UnorderedMapUsage)
{
    BatchnormFwdInferenceSignatureKey key1(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    BatchnormBwdSignatureKey key2(DataType::HALF, DataType::HALF, DataType::HALF);

    PlanRegistrySignatureKey v1 = key1;
    PlanRegistrySignatureKey v2 = key2;

    std::unordered_map<PlanRegistrySignatureKey,
                       int,
                       PlanRegistrySignatureKeyHash,
                       PlanRegistrySignatureKeyEqual>
        registry;
    registry[v1] = 42;
    registry[v2] = 24;

    EXPECT_EQ(registry[v1], 42);
    EXPECT_EQ(registry[v2], 24);
}
