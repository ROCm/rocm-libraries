// // Copyright © Advanced Micro Devices, Inc., or its affiliates.
// // SPDX-License-Identifier:  MIT

// #include <gtest/gtest.h>
// #include <unordered_map>
// #include <unordered_set>

// #include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferenceSignatureKey.hpp>

// using namespace hipdnn_sdk::test_utilities;
// using namespace hipdnn_sdk::data_objects;

// TEST(TestBatchnormSignatureRegistryKey, EqualityOperator)
// {
//     BatchnormFwdInferenceSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     EXPECT_EQ(key1, key2);

//     BatchnormFwdInferenceSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key4{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
//     EXPECT_EQ(key3, key4);

//     BatchnormFwdInferenceSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
//     EXPECT_NE(key5, key6);

//     BatchnormFwdInferenceSignatureKey key7{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key8{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
//     EXPECT_NE(key7, key8);

//     BatchnormFwdInferenceSignatureKey key9{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key10{DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE};
//     EXPECT_NE(key9, key10);
// }

// TEST(TestBatchnormSignatureRegistryKey, HashFunction)
// {
//     BatchnormFwdInferenceSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

//     BatchnormSignatureRegistryKeyHash hasher;
//     EXPECT_EQ(hasher(key1), hasher(key2));

//     BatchnormFwdInferenceSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF};

//     auto hash3 = hasher(key3);
//     auto hash4 = hasher(key4);
//     auto hash5 = hasher(key5);

//     EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
// }

// TEST(TestBatchnormSignatureRegistryKey, Copy)
// {
//     // Test copy constructor
//     BatchnormFwdInferenceSignatureKey original{DataType::FLOAT, DataType::HALF, DataType::DOUBLE};
//     BatchnormFwdInferenceSignatureKey copied{original};

//     EXPECT_EQ(original, copied);
//     EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
//     EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
//     EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
// }

// TEST(TestBatchnormSignatureRegistryKey, UnorderedMapUsage)
// {
//     std::unordered_map<BatchnormFwdInferenceSignatureKey, int, BatchnormSignatureRegistryKeyHash> map;

//     BatchnormFwdInferenceSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key2{DataType::HALF, DataType::FLOAT, DataType::FLOAT};

//     map[key1] = 100;
//     map[key2] = 200;

//     EXPECT_EQ(map[key1], 100);
//     EXPECT_EQ(map[key2], 200);
//     EXPECT_EQ(map.size(), 2u);

//     // Test that same key overwrites
//     BatchnormFwdInferenceSignatureKey key3{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     map[key3] = 300;

//     EXPECT_EQ(map[key1], 300);
//     EXPECT_EQ(map.size(), 2u);
// }

// TEST(TestBatchnormSignatureRegistryKey, UnorderedSetUsage)
// {
//     std::unordered_set<BatchnormFwdInferenceSignatureKey, BatchnormSignatureRegistryKeyHash> set;

//     BatchnormFwdInferenceSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key2{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
//     BatchnormFwdInferenceSignatureKey key3{
//         DataType::FLOAT, DataType::FLOAT, DataType::FLOAT}; // Same as key1

//     set.insert(key1);
//     set.insert(key2);
//     set.insert(key3);

//     EXPECT_EQ(set.size(), 2u);
//     EXPECT_TRUE(set.find(key1) != set.end());
//     EXPECT_TRUE(set.find(key2) != set.end());
// }
