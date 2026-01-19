// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_plugin_sdk;

TEST(TestMatmulCpuGraphExecutor, MatmulFloatExecutesAndMatchesReference)
{
    auto builder
        = createValidMatmulGraph({4, 8}, {8, 1}, {8, 5}, {5, 1}, {4, 5}, {5, 1}, DataType::FLOAT);

    GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    GraphTensorBundle tensorBundle(graphWrap.getTensorMap());
    unsigned int seed = getGlobalTestSeed();

    tensorBundle.randomizeTensor(1, -1.0f, 1.0f, seed);
    tensorBundle.randomizeTensor(2, -1.0f, 1.0f, seed + 1);

    // Build reference output C using the same shapes/strides as the graph's C tensor.
    auto cTensorAttrT = unpackTensorAttributes(*graphWrap.getTensorMap().at(3));
    hipdnn_data_sdk::utilities::Tensor<float> cRef(cTensorAttrT.dims, cTensorAttrT.strides);

    auto aTensorAttrT = unpackTensorAttributes(*graphWrap.getTensorMap().at(1));
    auto bTensorAttrT = unpackTensorAttributes(*graphWrap.getTensorMap().at(2));

    auto shallowA
        = createShallowTensor<float>(aTensorAttrT, tensorBundle.tensors.at(1)->rawHostData());
    auto shallowB
        = createShallowTensor<float>(bTensorAttrT, tensorBundle.tensors.at(2)->rawHostData());

    CpuFpReferenceMatmul::matmul<float, float, float, float>(*shallowA, *shallowB, cRef);

    auto variantPack = tensorBundle.toHostVariantPack();
    CpuReferenceGraphExecutor().execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);

    CpuFpReferenceValidation<float> validator(1e-5f, 1e-5f);
    EXPECT_TRUE(validator.allClose(cRef, *tensorBundle.tensors.at(3)));
}

TEST(TestMatmulCpuGraphExecutor, MatmulBatchBroadcastDivisibleExecutesAndMatchesReference)
{
    // Rank 3: batch dims broadcasting with divisibility.
    // A: [2, M=4, K=8]
    // B: [1, K=8, N=5] (broadcast along batch)
    // C: [2, M=4, N=5]
    auto builder = createValidMatmulGraph(
        {2, 4, 8}, {32, 8, 1}, {1, 8, 5}, {40, 5, 1}, {2, 4, 5}, {20, 5, 1}, DataType::FLOAT);

    GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    GraphTensorBundle tensorBundle(graphWrap.getTensorMap());
    unsigned int seed = getGlobalTestSeed();

    tensorBundle.randomizeTensor(1, -1.0f, 1.0f, seed);
    tensorBundle.randomizeTensor(2, -1.0f, 1.0f, seed + 1);

    // Reference output C
    auto cTensorAttrT = unpackTensorAttributes(*graphWrap.getTensorMap().at(3));
    hipdnn_data_sdk::utilities::Tensor<float> cRef(cTensorAttrT.dims, cTensorAttrT.strides);

    auto aTensorAttrT = unpackTensorAttributes(*graphWrap.getTensorMap().at(1));
    auto bTensorAttrT = unpackTensorAttributes(*graphWrap.getTensorMap().at(2));

    auto shallowA
        = createShallowTensor<float>(aTensorAttrT, tensorBundle.tensors.at(1)->rawHostData());
    auto shallowB
        = createShallowTensor<float>(bTensorAttrT, tensorBundle.tensors.at(2)->rawHostData());

    CpuFpReferenceMatmul::matmul<float, float, float, float>(*shallowA, *shallowB, cRef);

    auto variantPack = tensorBundle.toHostVariantPack();
    CpuReferenceGraphExecutor().execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);

    CpuFpReferenceValidation<float> validator(1e-5f, 1e-5f);
    EXPECT_TRUE(validator.allClose(cRef, *tensorBundle.tensors.at(3)));
}
