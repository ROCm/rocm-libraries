// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdSignatureKey.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestBatchnormBwdSignatureKey, EqualityOperator)
{
    BatchnormBwdSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    BatchnormBwdSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key4{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    BatchnormBwdSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    BatchnormBwdSignatureKey key7{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key8{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    BatchnormBwdSignatureKey key9{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key10{DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestBatchnormBwdSignatureKey, HashFunction)
{
    BatchnormBwdSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    BatchnormBwdSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BatchnormBwdSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestBatchnormBwdSignatureKey, Copy)
{
    BatchnormBwdSignatureKey original{DataType::FLOAT, DataType::HALF, DataType::DOUBLE};
    BatchnormBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
}
