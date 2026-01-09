// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/BatchnormFwdInferenceWithVariancePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/BatchnormFwdInferenceWithVarianceSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::data_objects;

TEST(TestBatchnormFwdInferenceWithVariancePlan, PlanBuilderMapContainsExpectedKeys)
{
    auto planBuilders = BatchnormFwdInferenceWithVarianceSignatureKey::getPlanBuilders();

    // Verify we have builders for common type combinations
    EXPECT_GT(planBuilders.size(), 0);

    // FP32 case
    BatchnormFwdInferenceWithVarianceSignatureKey fp32Key(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(fp32Key) != planBuilders.end());

    // FP16 case with FP32 params
    BatchnormFwdInferenceWithVarianceSignatureKey fp16Key(
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(fp16Key) != planBuilders.end());

    // BFP16 case with FP32 params
    BatchnormFwdInferenceWithVarianceSignatureKey bfp16Key(
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(bfp16Key) != planBuilders.end());
}

TEST(TestBatchnormFwdInferenceWithVariancePlan, SignatureKeyHashingWorks)
{
    BatchnormFwdInferenceWithVarianceSignatureKey key1(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    BatchnormFwdInferenceWithVarianceSignatureKey key2(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    BatchnormFwdInferenceWithVarianceSignatureKey key3(
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT);

    // Same keys should be equal
    EXPECT_TRUE(key1 == key2);
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    // Different keys should not be equal
    EXPECT_FALSE(key1 == key3);
    // Hash collision is possible but unlikely for these specific cases
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
}

TEST(TestBatchnormFwdInferenceWithVariancePlan, NodeTypeIsCorrect)
{
    BatchnormFwdInferenceWithVarianceSignatureKey key;
    EXPECT_EQ(key.nodeType, NodeAttributes::BatchnormInferenceAttributesVarianceExt);
}
