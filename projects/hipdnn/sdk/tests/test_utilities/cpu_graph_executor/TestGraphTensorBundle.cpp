// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "BatchnormGraphUtils.hpp"
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_plugin;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestGraphTensorBundle : public ::testing::Test
{
protected:
    std::unique_ptr<GraphWrapper> buildTestGraph(DataType inputDataType,
                                                 DataType scaleBiasDataType,
                                                 DataType meanVarianceDataType)
    {
        std::vector<int64_t> dims = {2, 3, 4, 4};
        auto graph = buildBatchnormFwdInferenceGraph(
            inputDataType, scaleBiasDataType, meanVarianceDataType, dims, TensorLayout::NCHW);

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
        _flatbufferData = std::move(flatbufferGraph);

        return std::make_unique<GraphWrapper>(_flatbufferData.data(), _flatbufferData.size());
    }

private:
    flatbuffers::DetachedBuffer _flatbufferData;
};

TEST_F(TestGraphTensorBundle, ConstructorCreatesAllNonVirtualTensors)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    // Should create tensors for all non-virtual entries in tensorMap
    EXPECT_EQ(bundle.tensors.size(), tensorMap.size());

    for(const auto& [uid, attr] : tensorMap)
    {
        EXPECT_TRUE(bundle.tensors.find(uid) != bundle.tensors.end());
    }
}

TEST_F(TestGraphTensorBundle, ConstructorSkipsVirtualTensors)
{
    std::vector<int64_t> dims = {2, 3, 4, 4};
    auto graph = buildBatchnormFwdInferenceGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NCHW, true);

    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
    auto& tensorMap = graphWrapper.getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    // Should have fewer tensors than tensorMap entries due to virtual tensors
    EXPECT_LT(bundle.tensors.size(), tensorMap.size());

    // Virtual tensors should not be in the bundle
    for(const auto& [uid, attr] : tensorMap)
    {
        if(attr->virtual_())
        {
            EXPECT_TRUE(bundle.tensors.find(uid) == bundle.tensors.end());
        }
        else
        {
            EXPECT_TRUE(bundle.tensors.find(uid) != bundle.tensors.end());
        }
    }
}

TEST_F(TestGraphTensorBundle, RandomizeTensorSucceeds)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    // Get the first tensor UID
    ASSERT_FALSE(bundle.tensors.empty());
    auto firstUid = bundle.tensors.begin()->first;

    // Randomize should not throw
    EXPECT_NO_THROW(bundle.randomizeTensor(firstUid, -1.0f, 1.0f, 42));
}

TEST_F(TestGraphTensorBundle, RandomizeTensorThrowsForInvalidUid)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    int64_t invalidUid = 99999;
    EXPECT_THROW(bundle.randomizeTensor(invalidUid, -1.0f, 1.0f, 42), std::runtime_error);
}

TEST_F(TestGraphTensorBundle, ToVariantPackReturnsCorrectMapping)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    auto variantPack = bundle.toHostVariantPack();

    EXPECT_EQ(variantPack.size(), bundle.tensors.size());

    for(const auto& [uid, tensorPtr] : bundle.tensors)
    {
        ASSERT_TRUE(variantPack.find(uid) != variantPack.end());
        EXPECT_EQ(variantPack[uid], tensorPtr->rawHostData());
    }
}

TEST_F(TestGraphTensorBundle, ConstructorHandlesDifferentDataTypes)
{
    auto graphWrapper = buildTestGraph(DataType::HALF, DataType::HALF, DataType::HALF);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    EXPECT_EQ(bundle.tensors.size(), tensorMap.size());
}

TEST_F(TestGraphTensorBundle, ConstructorHandlesBFloat16DataTypes)
{
    auto graphWrapper = buildTestGraph(DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    EXPECT_EQ(bundle.tensors.size(), tensorMap.size());
}

TEST_F(TestGraphTensorBundle, TensorsHaveCorrectDimensions)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    for(const auto& [uid, tensorPtr] : bundle.tensors)
    {
        auto attr = tensorMap.at(uid);
        auto dims = tensorPtr->dims();

        ASSERT_EQ(dims.size(), attr->dims()->size());
        for(size_t i = 0; i < dims.size(); ++i)
        {
            EXPECT_EQ(dims[i], attr->dims()->Get(static_cast<flatbuffers::uoffset_t>(i)));
        }
    }
}
