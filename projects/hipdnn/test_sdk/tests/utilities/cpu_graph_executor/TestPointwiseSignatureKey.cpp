// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwiseSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

namespace
{

using TensorMap = std::unordered_map<int64_t, const TensorAttributes*>;

// Builds standalone TensorAttributes FlatBuffers and exposes them as the
// const-pointer map consumed by PointwiseSignatureKey's node constructor.
// Each tensor is serialized into its own buffer kept alive for the lifetime
// of the builder so the pointers stored in the map remain valid.
class TensorMapBuilder
{
public:
    void add(int64_t uid)
    {
        auto fbb = std::make_unique<flatbuffers::FlatBufferBuilder>();
        const std::vector<int64_t> dims = {1};
        const std::vector<int64_t> strides = {1};
        auto offset
            = CreateTensorAttributesDirect(*fbb, uid, "t", DataType::FLOAT, &strides, &dims);
        fbb->Finish(offset);
        _map[uid] = flatbuffers::GetRoot<TensorAttributes>(fbb->GetBufferPointer());
        _builders.push_back(std::move(fbb));
    }

    // Registers a uid whose tensor pointer is absent from the map's backing
    // storage, matching a corrupted/dangling flatbuffer reference.
    void addNull(int64_t uid)
    {
        _map[uid] = nullptr;
    }

    const TensorMap& map() const
    {
        return _map;
    }

private:
    std::vector<std::unique_ptr<flatbuffers::FlatBufferBuilder>> _builders;
    TensorMap _map;
};

// Builds a standalone Node wrapping PointwiseAttributes for direct
// PointwiseSignatureKey construction tests, bypassing full graph
// serialization so malformed/missing tensor references can be exercised.
const Node& buildPointwiseNode(flatbuffers::FlatBufferBuilder& fbb,
                               PointwiseMode mode,
                               int64_t in0Uid,
                               flatbuffers::Optional<int64_t> in1Uid,
                               flatbuffers::Optional<int64_t> in2Uid,
                               int64_t out0Uid)
{
    auto pwAttrs = CreatePointwiseAttributes(fbb,
                                             mode,
                                             flatbuffers::nullopt,
                                             flatbuffers::nullopt,
                                             flatbuffers::nullopt,
                                             flatbuffers::nullopt,
                                             in0Uid,
                                             in1Uid,
                                             in2Uid,
                                             out0Uid);
    auto node
        = CreateNode(fbb, 0, DataType::FLOAT, NodeAttributes::PointwiseAttributes, pwAttrs.Union());
    fbb.Finish(node);
    return *flatbuffers::GetRoot<Node>(fbb.GetBufferPointer());
}

} // namespace

TEST(TestPointwiseSignatureKey, EqualityOperator)
{
    const PointwiseSignatureKey key1{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key2{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const PointwiseSignatureKey key3{
        PointwiseMode::RELU_FWD, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const PointwiseSignatureKey key4{
        PointwiseMode::RELU_FWD, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    // Different operations
    const PointwiseSignatureKey key5{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key6{
        PointwiseMode::SUB, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    // Different input data types
    const PointwiseSignatureKey key7{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key8{
        PointwiseMode::ADD, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key7 == key8);

    // Different output data types
    const PointwiseSignatureKey key9{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key10{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key9 == key10);

    // Different input1 data types (all other fields equal)
    const PointwiseSignatureKey key11{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key12{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key11 == key12);

    // Different input2 data types (all other fields, including input1, equal)
    const PointwiseSignatureKey key13{PointwiseMode::BINARY_SELECT,
                                      DataType::FLOAT,
                                      DataType::FLOAT,
                                      DataType::FLOAT,
                                      DataType::FLOAT,
                                      DataType::BOOLEAN};
    const PointwiseSignatureKey key14{PointwiseMode::BINARY_SELECT,
                                      DataType::FLOAT,
                                      DataType::FLOAT,
                                      DataType::FLOAT,
                                      DataType::FLOAT,
                                      DataType::UNSET};
    EXPECT_FALSE(key13 == key14);
}

TEST(TestPointwiseSignatureKey, HashFunction)
{
    const PointwiseSignatureKey key1{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key2{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const PointwiseSignatureKey key3{
        PointwiseMode::SUB, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const PointwiseSignatureKey key4{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();

    EXPECT_TRUE(hash3 != hash4);

    // Test that different operations produce different hashes
    const PointwiseSignatureKey key5{
        PointwiseMode::RELU_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key6{
        PointwiseMode::SIGMOID_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash5 != hash6);
}
TEST(TestPointwiseSignatureKey, HashSeparatesBooleanFields)
{
    const PointwiseSignatureKey input1Boolean{PointwiseMode::BINARY_SELECT,
                                              DataType::FLOAT,
                                              DataType::FLOAT,
                                              DataType::FLOAT,
                                              DataType::BOOLEAN,
                                              DataType::UNSET};
    const PointwiseSignatureKey input2Boolean{PointwiseMode::BINARY_SELECT,
                                              DataType::FLOAT,
                                              DataType::FLOAT,
                                              DataType::FLOAT,
                                              DataType::UNSET,
                                              DataType::BOOLEAN};

    EXPECT_NE(input1Boolean.hashSelf(), input2Boolean.hashSelf());
}

TEST(TestPointwiseSignatureKey, Copy)
{
    const PointwiseSignatureKey original{
        PointwiseMode::TANH_FWD, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const PointwiseSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.operation, PointwiseMode::TANH_FWD);
    EXPECT_EQ(copied.inputDataType, DataType::HALF);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
    EXPECT_EQ(copied.outputDataType, DataType::HALF);
}

TEST(TestPointwiseSignatureKey, CreateFromNodeAndTensorMapUnary)
{
    const PointwiseSignatureKey expectedKey{
        PointwiseMode::RELU_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                   1,
                                   TensorLayout::NCHW);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwiseSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

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
    const PointwiseSignatureKey expectedKey{
        PointwiseMode::ADD, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};

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
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwiseSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

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

    const PointwiseSignatureKey key1{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key2{
        PointwiseMode::SUB, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key3{
        PointwiseMode::ADD, DataType::HALF, DataType::FLOAT, DataType::HALF};

    testMap[key1] = 1;
    testMap[key2] = 2;
    testMap[key3] = 3;

    EXPECT_EQ(testMap.size(), 3);
    EXPECT_EQ(testMap[key1], 1);
    EXPECT_EQ(testMap[key2], 2);
    EXPECT_EQ(testMap[key3], 3);

    // Test that equal keys map to same value
    const PointwiseSignatureKey key1Copy{
        PointwiseMode::ADD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_EQ(testMap[key1Copy], 1);
}

TEST(TestPointwiseSignatureKey, UnorderedSetUsage)
{
    std::unordered_set<PointwiseSignatureKey, PointwiseSignatureKey> testSet;

    const PointwiseSignatureKey key1{
        PointwiseMode::RELU_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key2{
        PointwiseMode::SIGMOID_FWD, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const PointwiseSignatureKey key3{PointwiseMode::RELU_FWD,
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
    auto unaryModesBitset = hipdnn_flatbuffers_sdk::utilities::getUnaryModesBitset();
    auto binaryModesBitset = hipdnn_flatbuffers_sdk::utilities::getBinaryModesBitset();

    // Test that all supported operations create different keys
    std::unordered_set<PointwiseSignatureKey, PointwiseSignatureKey> uniqueKeys;

    // Add all unary operations
    size_t unaryCount = 0;
    for(size_t i = 0; i < unaryModesBitset.size(); ++i)
    {
        if(unaryModesBitset.test(i))
        {
            auto op = static_cast<PointwiseMode>(i);
            const PointwiseSignatureKey key{op, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
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
            const PointwiseSignatureKey key{op, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
            uniqueKeys.insert(key);
            ++binaryCount;
        }
    }

    const size_t totalOps = unaryCount + binaryCount;
    EXPECT_EQ(uniqueKeys.size(), totalOps);
}

TEST(TestPointwiseSignatureKey, CreateFromNodeAndTensorMapTernary)
{
    const PointwiseSignatureKey expectedKey{PointwiseMode::BINARY_SELECT,
                                            DataType::FLOAT,
                                            DataType::FLOAT,
                                            DataType::FLOAT,
                                            DataType::FLOAT,
                                            DataType::BOOLEAN};
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseTernaryGraph({1, 1, 1, 1},
                                     {1, 1, 1, 1},
                                     {1, 1, 1, 1},
                                     {1, 1, 1, 1},
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::BOOLEAN,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwiseSignatureKey keyFromNode(graphWrapper.getNode(0), graphWrapper.getTensorMap());

    EXPECT_EQ(keyFromNode, expectedKey);
    EXPECT_EQ(keyFromNode.input2DataType, DataType::BOOLEAN);
}

TEST(TestPointwiseSignatureKey, CreateFromNodeMissingSecondInputTensorThrows)
{
    flatbuffers::FlatBufferBuilder fbb;
    const Node& node = buildPointwiseNode(
        fbb, PointwiseMode::ADD, 1, flatbuffers::nullopt, flatbuffers::nullopt, 2);

    TensorMapBuilder tensors;
    tensors.add(1);
    tensors.add(2);

    EXPECT_THROW(PointwiseSignatureKey(node, tensors.map()), std::runtime_error);
}

TEST(TestPointwiseSignatureKey, CreateFromNodeSecondInputTensorNotInMapThrows)
{
    flatbuffers::FlatBufferBuilder fbb;
    const Node& node = buildPointwiseNode(
        fbb, PointwiseMode::ADD, 1, flatbuffers::Optional<int64_t>(99), flatbuffers::nullopt, 2);

    TensorMapBuilder tensors;
    tensors.add(1);
    tensors.add(2);
    tensors.addNull(99);

    EXPECT_THROW(PointwiseSignatureKey(node, tensors.map()), std::runtime_error);
}

TEST(TestPointwiseSignatureKey, CreateFromNodeMissingThirdInputTensorThrows)
{
    flatbuffers::FlatBufferBuilder fbb;
    const Node& node = buildPointwiseNode(fbb,
                                          PointwiseMode::BINARY_SELECT,
                                          1,
                                          flatbuffers::Optional<int64_t>(3),
                                          flatbuffers::nullopt,
                                          2);

    TensorMapBuilder tensors;
    tensors.add(1);
    tensors.add(2);
    tensors.add(3);

    EXPECT_THROW(PointwiseSignatureKey(node, tensors.map()), std::runtime_error);
}

TEST(TestPointwiseSignatureKey, CreateFromNodeThirdInputTensorNotInMapThrows)
{
    flatbuffers::FlatBufferBuilder fbb;
    const Node& node = buildPointwiseNode(fbb,
                                          PointwiseMode::BINARY_SELECT,
                                          1,
                                          flatbuffers::Optional<int64_t>(3),
                                          flatbuffers::Optional<int64_t>(99),
                                          2);

    TensorMapBuilder tensors;
    tensors.add(1);
    tensors.add(2);
    tensors.add(3);
    tensors.addNull(99);

    EXPECT_THROW(PointwiseSignatureKey(node, tensors.map()), std::runtime_error);
}
