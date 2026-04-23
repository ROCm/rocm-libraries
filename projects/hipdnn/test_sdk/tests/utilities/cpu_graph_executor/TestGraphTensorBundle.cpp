// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "BatchnormGraphUtils.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestGraphTensorBundle : public ::testing::Test
{
protected:
    std::unique_ptr<GraphWrapper> buildTestGraph(DataType inputDataType,
                                                 DataType scaleBiasDataType,
                                                 DataType meanVarianceDataType)
    {
        const std::vector<int64_t> dims = {2, 3, 4, 4};
        auto graph = buildBatchnormFwdInferenceGraph(inputDataType,
                                                     scaleBiasDataType,
                                                     meanVarianceDataType,
                                                     meanVarianceDataType,
                                                     dims,
                                                     TensorLayout::NCHW);

        auto [serializedGraph, serErr] = graph->to_binary();
        if(serErr.is_bad())
        {
            throw std::runtime_error("Graph serialization failed: " + serErr.get_message());
        }
        _serializedData = std::move(serializedGraph);

        return std::make_unique<GraphWrapper>(_serializedData.data(), _serializedData.size());
    }

private:
    std::vector<uint8_t> _serializedData;
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
    const std::vector<int64_t> dims = {2, 3, 4, 4};
    auto graph = buildBatchnormFwdInferenceGraph(DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 dims,
                                                 TensorLayout::NCHW,
                                                 true);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
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

    const int64_t invalidUid = 99999;
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

    const GraphTensorBundle bundle(tensorMap);

    EXPECT_EQ(bundle.tensors.size(), tensorMap.size());
}

TEST_F(TestGraphTensorBundle, ConstructorHandlesBFloat16DataTypes)
{
    auto graphWrapper = buildTestGraph(DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
    auto& tensorMap = graphWrapper->getTensorMap();

    const GraphTensorBundle bundle(tensorMap);

    EXPECT_EQ(bundle.tensors.size(), tensorMap.size());
}

TEST_F(TestGraphTensorBundle, TensorsHaveCorrectDimensions)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    const GraphTensorBundle bundle(tensorMap);

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

TEST_F(TestGraphTensorBundle, ToDeviceVariantPackReturnsCorrectMapping)
{
    // Only this test in the suite touches device memory: rawDeviceData() lazily hipMallocs.
    SKIP_IF_NO_DEVICES();
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    auto variantPack = bundle.toDeviceVariantPack();

    EXPECT_EQ(variantPack.size(), bundle.tensors.size());

    for(const auto& [uid, tensorPtr] : bundle.tensors)
    {
        ASSERT_TRUE(variantPack.find(uid) != variantPack.end());
        EXPECT_EQ(variantPack[uid], tensorPtr->rawDeviceData());
    }
}

TEST_F(TestGraphTensorBundle, GetTensorReturnsCorrectTensor)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    ASSERT_FALSE(bundle.tensors.empty());
    auto firstUid = bundle.tensors.begin()->first;

    auto& tensor = bundle.getTensor(firstUid);
    EXPECT_EQ(&tensor, bundle.tensors.at(firstUid).get());
}

TEST_F(TestGraphTensorBundle, GetTensorThrowsForInvalidUid)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    const int64_t invalidUid = 99999;
    EXPECT_THROW(bundle.getTensor(invalidUid), std::runtime_error);
}

TEST_F(TestGraphTensorBundle, GetTensorConstReturnsCorrectTensor)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    ASSERT_FALSE(bundle.tensors.empty());
    auto firstUid = bundle.tensors.begin()->first;

    const auto& constBundle = bundle;
    const auto& tensor = constBundle.getTensor(firstUid);
    EXPECT_EQ(&tensor, bundle.tensors.at(firstUid).get());
}

TEST_F(TestGraphTensorBundle, GetTensorConstThrowsForInvalidUid)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    const GraphTensorBundle bundle(tensorMap);

    const auto& constBundle = bundle;
    const int64_t invalidUid = 99999;
    EXPECT_THROW(constBundle.getTensor(invalidUid), std::runtime_error);
}

TEST_F(TestGraphTensorBundle, IsOutputReturnsTrueForOutputTensors)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    // Manually mark some tensors as outputs
    ASSERT_GE(bundle.tensors.size(), 2U);
    auto it = bundle.tensors.begin();
    auto firstUid = it->first;
    ++it;
    auto secondUid = it->first;

    bundle.outputTensorIds.insert(firstUid);
    bundle.outputTensorIds.insert(secondUid);

    EXPECT_TRUE(bundle.isOutput(firstUid));
    EXPECT_TRUE(bundle.isOutput(secondUid));

    // Other tensors should not be outputs
    for(const auto& [uid, tensorPtr] : bundle.tensors)
    {
        if(uid != firstUid && uid != secondUid)
        {
            EXPECT_FALSE(bundle.isOutput(uid));
        }
    }
}

TEST_F(TestGraphTensorBundle, IsOutputReturnsFalseWhenNoOutputsSet)
{
    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    const GraphTensorBundle bundle(tensorMap);

    for(const auto& [uid, attr] : tensorMap)
    {
        EXPECT_FALSE(bundle.isOutput(uid));
    }
}

TEST_F(TestGraphTensorBundle, SentinelFillOutputTensorsFillsOnlyOutputs)
{
    using hipdnn_data_sdk::types::isnan;

    auto graphWrapper = buildTestGraph(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    auto& tensorMap = graphWrapper->getTensorMap();

    GraphTensorBundle bundle(tensorMap);

    // Fill all tensors with 1.0 first
    for(auto& [uid, tensor] : bundle.tensors)
    {
        tensor->fillTensorWithValue(1.0f);
    }

    // Mark one tensor as output and sentinel-fill
    auto firstUid = bundle.tensors.begin()->first;
    bundle.outputTensorIds.insert(firstUid);
    bundle.sentinelFillOutputTensors();

    // The output tensor should have NaN sentinel values
    auto& outputTensor = *bundle.tensors.at(firstUid);
    TensorView<float> outputView(outputTensor);
    EXPECT_TRUE(isnan(outputView.getHostValue({0})));

    // Non-output tensors should still have 1.0
    for(const auto& [uid, tensor] : bundle.tensors)
    {
        if(uid != firstUid)
        {
            TensorView<float> view(*tensor);
            EXPECT_FLOAT_EQ(static_cast<float>(view.getHostValue({0})), 1.0f);
        }
    }
}
