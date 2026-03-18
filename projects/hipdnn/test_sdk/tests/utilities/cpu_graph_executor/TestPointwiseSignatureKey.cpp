// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwiseSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestPointwiseSignatureKey, EqualityOperator)
{
    PointwiseSignatureKey const key1{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key2{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    PointwiseSignatureKey const key3{
        PointwiseMode::RELU_FWD, DataType::HALF, DataType::FLOAT, DataType::HALF};
    PointwiseSignatureKey const key4{
        PointwiseMode::RELU_FWD, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    // Different operations
    PointwiseSignatureKey const key5{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key6{
        PointwiseMode::SUB, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    // Different input data types
    PointwiseSignatureKey const key7{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key8{
        PointwiseMode::ADD, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key7 == key8);

    // Different output data types
    PointwiseSignatureKey const key9{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key10{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestPointwiseSignatureKey, HashFunction)
{
    PointwiseSignatureKey const key1{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key2{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    PointwiseSignatureKey const key3{
        PointwiseMode::SUB, DataType::HALF, DataType::FLOAT, DataType::HALF};
    PointwiseSignatureKey const key4{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();

    EXPECT_TRUE(hash3 != hash4);

    // Test that different operations produce different hashes
    PointwiseSignatureKey const key5{
        PointwiseMode::RELU_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key6{
        PointwiseMode::SIGMOID_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash5 != hash6);
}

TEST(TestPointwiseSignatureKey, Copy)
{
    PointwiseSignatureKey const original{
        PointwiseMode::TANH_FWD, DataType::HALF, DataType::FLOAT, DataType::HALF};
    PointwiseSignatureKey const copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.operation, PointwiseMode::TANH_FWD);
    EXPECT_EQ(copied.inputDataType, DataType::HALF);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
    EXPECT_EQ(copied.outputDataType, DataType::HALF);
}

TEST(TestPointwiseSignatureKey, CreateFromNodeAndTensorMapUnary)
{
    PointwiseSignatureKey const expectedKey{
        PointwiseMode::RELU_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> const inputDims = {1, 3, 4, 4};
    std::vector<int64_t> const outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                   1,
                                   TensorLayout::NCHW);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(flatbufferGraph.data(),
                                                                         flatbufferGraph.size());

    PointwiseSignatureKey const keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    // Debug output to see the actual mismatch
    std::cout << "Expected key: operation=" << static_cast<int>(expectedKey.operation)
              << ", inputDataType=" << static_cast<int>(expectedKey.inputDataType)
              << ", computeDataType=" << static_cast<int>(expectedKey.computeDataType)
              << ", outputDataType=" << static_cast<int>(expectedKey.outputDataType)
              << ", input1DataType=" << static_cast<int>(expectedKey.input1DataType) << '\n';

    std::cout << "Actual key: operation=" << static_cast<int>(keyFromNode.operation)
              << ", inputDataType=" << static_cast<int>(keyFromNode.inputDataType)
              << ", computeDataType=" << static_cast<int>(keyFromNode.computeDataType)
              << ", outputDataType=" << static_cast<int>(keyFromNode.outputDataType)
              << ", input1DataType=" << static_cast<int>(keyFromNode.input1DataType) << '\n';

    EXPECT_TRUE(keyFromNode == expectedKey);
}

TEST(TestPointwiseSignatureKey, CreateFromNodeAndTensorMapBinary)
{
    PointwiseSignatureKey const expectedKey{
        PointwiseMode::ADD, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    std::vector<int64_t> const input1Dims = {1, 3, 2, 2};
    std::vector<int64_t> const input2Dims = {1, 3, 2, 2};
    std::vector<int64_t> const outputDims = {1, 3, 2, 2};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
                                    outputDims,
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD,
                                    1,
                                    TensorLayout::NCHW);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(flatbufferGraph.data(),
                                                                         flatbufferGraph.size());

    PointwiseSignatureKey const keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    // Debug output to see the actual mismatch
    std::cout << "Expected key: operation=" << static_cast<int>(expectedKey.operation)
              << ", inputDataType=" << static_cast<int>(expectedKey.inputDataType)
              << ", computeDataType=" << static_cast<int>(expectedKey.computeDataType)
              << ", outputDataType=" << static_cast<int>(expectedKey.outputDataType)
              << ", input1DataType=" << static_cast<int>(expectedKey.input1DataType) << '\n';

    std::cout << "Actual key: operation=" << static_cast<int>(keyFromNode.operation)
              << ", inputDataType=" << static_cast<int>(keyFromNode.inputDataType)
              << ", computeDataType=" << static_cast<int>(keyFromNode.computeDataType)
              << ", outputDataType=" << static_cast<int>(keyFromNode.outputDataType)
              << ", input1DataType=" << static_cast<int>(keyFromNode.input1DataType) << '\n';

    EXPECT_TRUE(keyFromNode == expectedKey);
}

TEST(TestPointwiseSignatureKey, UnorderedMapUsage)
{
    std::unordered_map<PointwiseSignatureKey, int, PointwiseSignatureKey> testMap;

    PointwiseSignatureKey const key1{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key2{
        PointwiseMode::SUB, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key3{
        PointwiseMode::ADD, DataType::HALF, DataType::FLOAT, DataType::HALF};

    testMap[key1] = 1;
    testMap[key2] = 2;
    testMap[key3] = 3;

    EXPECT_EQ(testMap.size(), 3);
    EXPECT_EQ(testMap[key1], 1);
    EXPECT_EQ(testMap[key2], 2);
    EXPECT_EQ(testMap[key3], 3);

    // Test that equal keys map to same value
    PointwiseSignatureKey const key1Copy{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_EQ(testMap[key1Copy], 1);
}

TEST(TestPointwiseSignatureKey, UnorderedSetUsage)
{
    std::unordered_set<PointwiseSignatureKey, PointwiseSignatureKey> testSet;

    PointwiseSignatureKey const key1{
        PointwiseMode::RELU_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key2{
        PointwiseMode::SIGMOID_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    PointwiseSignatureKey const key3{PointwiseMode::RELU_FWD,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT}; // Duplicate of key1

    testSet.insert(key1);
    testSet.insert(key2);
    testSet.insert(key3);

    EXPECT_EQ(testSet.size(), 2); // key3 should be treated as duplicate of key1

    EXPECT_TRUE(testSet.find(key1) != testSet.end());
    EXPECT_TRUE(testSet.find(key2) != testSet.end());
    EXPECT_TRUE(testSet.find(key3) != testSet.end()); // Should find key1 instead
}

TEST(TestPointwiseSignatureKey, DifferentOperationsAreDifferent)
{
    auto unaryModesBitset = hipdnn_data_sdk::utilities::getUnaryModesBitset();
    auto binaryModesBitset = hipdnn_data_sdk::utilities::getBinaryModesBitset();

    // Test that all supported operations create different keys
    std::unordered_set<PointwiseSignatureKey, PointwiseSignatureKey> uniqueKeys;

    // Add all unary operations
    size_t unaryCount = 0;
    for(size_t i = 0; i < unaryModesBitset.size(); ++i)
    {
        if(unaryModesBitset.test(i))
        {
            auto op = static_cast<PointwiseMode>(i);
            PointwiseSignatureKey const key{op, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
            uniqueKeys.insert(key);
            ++unaryCount;
        }
    }

    // Add all binary operations
    size_t binaryCount = 0;
    for(size_t i = 0; i < binaryModesBitset.size(); ++i)
    {
        if(binaryModesBitset.test(i))
        {
            auto op = static_cast<PointwiseMode>(i);
            PointwiseSignatureKey const key{op, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
            uniqueKeys.insert(key);
            ++binaryCount;
        }
    }

    size_t const totalOps = unaryCount + binaryCount;
    EXPECT_EQ(uniqueKeys.size(), totalOps);
}
