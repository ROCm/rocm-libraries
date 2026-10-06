// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/detail/GpuBatchnormBwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuBatchnormBwdSignatureKey, EqualityAndHash)
{
    const GpuBatchnormBwdSignatureKey key1{DataType::HALF,
                                           DataType::HALF,
                                           DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::HALF,
                                           DataType::FLOAT};
    const GpuBatchnormBwdSignatureKey key2{key1};
    const GpuBatchnormBwdSignatureKey different{DataType::FLOAT,
                                                DataType::HALF,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::HALF,
                                                DataType::FLOAT};

    EXPECT_EQ(key1, key2);
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());
    EXPECT_FALSE(key1 == different);
    EXPECT_NE(key1.hashSelf(), different.hashSelf());
}

TEST(TestGpuBatchnormBwdSignatureKey, CreateFromNodeWithSavedStatistics)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormBwdSignatureKey key(graph.getNode(0), graph.getTensorMap());
    const GpuBatchnormBwdSignatureKey expected{DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT};

    EXPECT_EQ(key, expected);
}

TEST(TestGpuBatchnormBwdSignatureKey, CreateFromNodeWithoutSavedStatistics)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {12, 4, 2, 1}, {2, 3, 2, 2}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormBwdSignatureKey key(graph.getNode(0), graph.getTensorMap());

    EXPECT_EQ(key.meanVarianceDataType, key.scaleBiasDataType);
}

TEST(TestGpuBatchnormBwdSignatureKey, RegistersSupportedComputeTypesOnly)
{
    const auto builders = GpuBatchnormBwdSignatureKey::getPlanBuilders();

    EXPECT_EQ(builders.size(), 7);
    const GpuBatchnormBwdSignatureKey supported{DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT};
    const GpuBatchnormBwdSignatureKey unsupported{DataType::HALF,
                                                  DataType::HALF,
                                                  DataType::HALF,
                                                  DataType::HALF,
                                                  DataType::HALF,
                                                  DataType::HALF};
    EXPECT_NE(builders.find(supported), builders.end());
    EXPECT_EQ(builders.find(unsupported), builders.end());
}
