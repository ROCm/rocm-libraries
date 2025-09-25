// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferenceSignatureKey.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestBatchnormFwdInferenceSignatureKey, EqualityOperator)
{
    BatchnormFwdInferenceSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1.equal(key2));

    BatchnormFwdInferenceSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey key4{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key3.equal(key4));

    BatchnormFwdInferenceSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key5.equal(key6));

    BatchnormFwdInferenceSignatureKey key7{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey key8{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key7.equal(key8));

    BatchnormFwdInferenceSignatureKey key9{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey key10{DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE};
    EXPECT_FALSE(key9.equal(key10));
}

TEST(TestBatchnormFwdInferenceSignatureKey, HashFunction)
{
    BatchnormFwdInferenceSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    BatchnormFwdInferenceSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestBatchnormFwdInferenceSignatureKey, Copy)
{
    BatchnormFwdInferenceSignatureKey original{DataType::FLOAT, DataType::HALF, DataType::DOUBLE};
    BatchnormFwdInferenceSignatureKey copied{original};

    EXPECT_TRUE(original.equal(copied));
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
}
