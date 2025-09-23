// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistryKey.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestBatchnormSignatureRegistryKey, EqualityOperator)
{
    BatchnormSignatureRegistryKey key1{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key2{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    EXPECT_EQ(key1, key2);

    BatchnormSignatureRegistryKey key3{.inputDataType = DataType::HALF,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key4{.inputDataType = DataType::HALF,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    EXPECT_EQ(key3, key4);

    BatchnormSignatureRegistryKey key5{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key6{.inputDataType = DataType::HALF,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    EXPECT_NE(key5, key6);

    BatchnormSignatureRegistryKey key7{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key8{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::HALF,
                                       .meanVarianceDataType = DataType::FLOAT};
    EXPECT_NE(key7, key8);

    BatchnormSignatureRegistryKey key9{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key10{.inputDataType = DataType::FLOAT,
                                        .scaleBiasDataType = DataType::FLOAT,
                                        .meanVarianceDataType = DataType::DOUBLE};
    EXPECT_NE(key9, key10);
}

TEST(TestBatchnormSignatureRegistryKey, HashFunction)
{
    BatchnormSignatureRegistryKey key1{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key2{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};

    BatchnormSignatureRegistryKeyHash hasher;
    EXPECT_EQ(hasher(key1), hasher(key2));

    BatchnormSignatureRegistryKey key3{.inputDataType = DataType::HALF,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key4{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::HALF,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key5{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::HALF};

    auto hash3 = hasher(key3);
    auto hash4 = hasher(key4);
    auto hash5 = hasher(key5);

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestBatchnormSignatureRegistryKey, Copy)
{
    // Test copy constructor
    BatchnormSignatureRegistryKey original{.inputDataType = DataType::FLOAT,
                                           .scaleBiasDataType = DataType::HALF,
                                           .meanVarianceDataType = DataType::DOUBLE};
    BatchnormSignatureRegistryKey copied{original};

    EXPECT_EQ(original, copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
}

TEST(TestBatchnormSignatureRegistryKey, UnorderedMapUsage)
{
    std::unordered_map<BatchnormSignatureRegistryKey, int, BatchnormSignatureRegistryKeyHash> map;

    BatchnormSignatureRegistryKey key1{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key2{.inputDataType = DataType::HALF,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};

    map[key1] = 100;
    map[key2] = 200;

    EXPECT_EQ(map[key1], 100);
    EXPECT_EQ(map[key2], 200);
    EXPECT_EQ(map.size(), 2u);

    // Test that same key overwrites
    BatchnormSignatureRegistryKey key3{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    map[key3] = 300;

    EXPECT_EQ(map[key1], 300);
    EXPECT_EQ(map.size(), 2u);
}

TEST(TestBatchnormSignatureRegistryKey, UnorderedSetUsage)
{
    std::unordered_set<BatchnormSignatureRegistryKey, BatchnormSignatureRegistryKeyHash> set;

    BatchnormSignatureRegistryKey key1{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key2{.inputDataType = DataType::HALF,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureRegistryKey key3{.inputDataType = DataType::FLOAT,
                                       .scaleBiasDataType = DataType::FLOAT,
                                       .meanVarianceDataType = DataType::FLOAT}; // Same as key1

    set.insert(key1);
    set.insert(key2);
    set.insert(key3);

    EXPECT_EQ(set.size(), 2u);
    EXPECT_TRUE(set.contains(key1));
    EXPECT_TRUE(set.contains(key2));
}
