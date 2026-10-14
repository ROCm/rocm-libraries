// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/detail/GpuReductionSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuReductionSignatureKey, EqualityOperator)
{
    const GpuReductionSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuReductionSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuReductionSignatureKey key3{DataType::HALF, DataType::HALF, DataType::DOUBLE};
    const GpuReductionSignatureKey key4{DataType::HALF, DataType::HALF, DataType::DOUBLE};
    EXPECT_TRUE(key3 == key4);

    const GpuReductionSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE};
    const GpuReductionSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::DOUBLE};
    EXPECT_FALSE(key5 == key6);

    const GpuReductionSignatureKey key7{DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT};
    const GpuReductionSignatureKey key8{DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    const GpuReductionSignatureKey key9{DataType::BFLOAT16, DataType::HALF, DataType::FLOAT};
    const GpuReductionSignatureKey key10{DataType::BFLOAT16, DataType::HALF, DataType::DOUBLE};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestGpuReductionSignatureKey, HashFunction)
{
    const GpuReductionSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuReductionSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuReductionSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::DOUBLE};
    const GpuReductionSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::DOUBLE};
    const GpuReductionSignatureKey key5{DataType::BFLOAT16, DataType::HALF, DataType::FLOAT};
    const GpuReductionSignatureKey key6{DataType::HALF, DataType::BFLOAT16, DataType::FLOAT};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash3 != hash6 && hash4 != hash5
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestGpuReductionSignatureKey, Copy)
{
    const GpuReductionSignatureKey original{DataType::HALF, DataType::FLOAT, DataType::DOUBLE};
    const GpuReductionSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::HALF);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::DOUBLE);
}

TEST(TestGpuReductionSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuReductionSignatureKey keyFromNode(
        graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type());
    const GpuReductionSignatureKey expectedKey{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
