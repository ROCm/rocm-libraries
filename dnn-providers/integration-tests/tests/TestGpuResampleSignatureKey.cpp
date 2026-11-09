// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/detail/GpuResampleSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

// =============================================================
// Test GpuResampleFwdSignatureKey
// =============================================================

TEST(TestGpuResampleFwdSignatureKey, EqualityOperator)
{
    const GpuResampleFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    const GpuResampleFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    EXPECT_TRUE(key1 == key2);

    const GpuResampleFwdSignatureKey key3{
        DataType::HALF, DataType::BFLOAT16, DataType::FLOAT, DataType::INT32};
    const GpuResampleFwdSignatureKey key4{
        DataType::HALF, DataType::BFLOAT16, DataType::FLOAT, DataType::INT32};
    EXPECT_TRUE(key3 == key4);

    const GpuResampleFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    const GpuResampleFwdSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    EXPECT_FALSE(key5 == key6);

    const GpuResampleFwdSignatureKey key7{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::INT32};
    const GpuResampleFwdSignatureKey key8{
        DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT, DataType::INT32};
    EXPECT_FALSE(key7 == key8);

    const GpuResampleFwdSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    const GpuResampleFwdSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::UNSET};
    EXPECT_FALSE(key9 == key10);

    const GpuResampleFwdSignatureKey key11{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    const GpuResampleFwdSignatureKey key12{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestGpuResampleFwdSignatureKey, HashFunction)
{
    const GpuResampleFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    const GpuResampleFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    std::vector<size_t> hashes;
    hashes.push_back((GpuResampleFwdSignatureKey{
                          DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::INT32})
                         .hashSelf());
    hashes.push_back((GpuResampleFwdSignatureKey{
                          DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT, DataType::INT32})
                         .hashSelf());
    hashes.push_back((GpuResampleFwdSignatureKey{
                          DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::INT32})
                         .hashSelf());
    hashes.push_back((GpuResampleFwdSignatureKey{
                          DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET})
                         .hashSelf());

    for(size_t i = 0; i < hashes.size(); ++i)
    {
        for(size_t j = i + 1; j < hashes.size(); ++j)
        {
            EXPECT_NE(hashes[i], hashes[j]);
        }
    }
}

TEST(TestGpuResampleFwdSignatureKey, Copy)
{
    const GpuResampleFwdSignatureKey original{
        DataType::BFLOAT16, DataType::HALF, DataType::FLOAT, DataType::INT32};
    const GpuResampleFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.yDataType, DataType::HALF);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
    EXPECT_EQ(copied.indexDataType, DataType::INT32);
}

TEST(TestGpuResampleFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdSignatureKey keyFromNode(
        graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type());
    const GpuResampleFwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};

    EXPECT_TRUE(keyFromNode == expectedKey);
}

TEST(TestGpuResampleFwdSignatureKey, ThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(GpuResampleFwdSignatureKey(
                     graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type()),
                 std::runtime_error);
}

TEST(TestGpuResampleFwdSignatureKey, ThrowsForMissingTensor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleFwdAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy[nodeAttributes->x_tensor_uid()] = nullptr;

    EXPECT_THROW(GpuResampleFwdSignatureKey(
                     graph.getNode(0), tensorMapCopy, graph.getNode(0).compute_data_type()),
                 std::runtime_error);
}

// =============================================================
// Test GpuResampleBwdSignatureKey
// =============================================================

TEST(TestGpuResampleBwdSignatureKey, EqualityOperator)
{
    const GpuResampleBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    const GpuResampleBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    EXPECT_TRUE(key1 == key2);

    const GpuResampleBwdSignatureKey key3{
        DataType::BFLOAT16, DataType::HALF, DataType::FLOAT, DataType::INT32};
    const GpuResampleBwdSignatureKey key4{
        DataType::BFLOAT16, DataType::HALF, DataType::FLOAT, DataType::INT32};
    EXPECT_TRUE(key3 == key4);

    const GpuResampleBwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    const GpuResampleBwdSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    EXPECT_FALSE(key5 == key6);

    const GpuResampleBwdSignatureKey key7{
        DataType::BFLOAT16, DataType::HALF, DataType::FLOAT, DataType::INT32};
    const GpuResampleBwdSignatureKey key8{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::INT32};
    EXPECT_FALSE(key7 == key8);

    const GpuResampleBwdSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    const GpuResampleBwdSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::UNSET};
    EXPECT_FALSE(key9 == key10);

    const GpuResampleBwdSignatureKey key11{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    const GpuResampleBwdSignatureKey key12{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestGpuResampleBwdSignatureKey, HashFunction)
{
    const GpuResampleBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    const GpuResampleBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    std::vector<size_t> hashes;
    hashes.push_back((GpuResampleBwdSignatureKey{
                          DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::INT32})
                         .hashSelf());
    hashes.push_back((GpuResampleBwdSignatureKey{
                          DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::INT32})
                         .hashSelf());
    hashes.push_back((GpuResampleBwdSignatureKey{
                          DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::INT32})
                         .hashSelf());
    hashes.push_back((GpuResampleBwdSignatureKey{
                          DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET})
                         .hashSelf());

    for(size_t i = 0; i < hashes.size(); ++i)
    {
        for(size_t j = i + 1; j < hashes.size(); ++j)
        {
            EXPECT_NE(hashes[i], hashes[j]);
        }
    }
}

TEST(TestGpuResampleBwdSignatureKey, Copy)
{
    const GpuResampleBwdSignatureKey original{
        DataType::HALF, DataType::BFLOAT16, DataType::FLOAT, DataType::INT32};
    const GpuResampleBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dyDataType, DataType::HALF);
    EXPECT_EQ(copied.dxDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
    EXPECT_EQ(copied.indexDataType, DataType::INT32);
}

TEST(TestGpuResampleBwdSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdSignatureKey keyFromNode(
        graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type());
    const GpuResampleBwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};

    EXPECT_TRUE(keyFromNode == expectedKey);
}

TEST(TestGpuResampleBwdSignatureKey, ThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(GpuResampleBwdSignatureKey(
                     graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type()),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdSignatureKey, ThrowsForMissingTensor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleBwdAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy[nodeAttributes->dx_tensor_uid()] = nullptr;

    EXPECT_THROW(GpuResampleBwdSignatureKey(
                     graph.getNode(0), tensorMapCopy, graph.getNode(0).compute_data_type()),
                 std::runtime_error);
}
