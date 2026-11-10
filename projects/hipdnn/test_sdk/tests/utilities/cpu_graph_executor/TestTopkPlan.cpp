// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/TopkPlan.hpp>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

// createValidTopkGraph(): x [2, 8] FLOAT (uid 1), indices [2, 3] INT32 (uid 2),
// values [2, 3] FLOAT (uid 3), k 3, largest false.
struct TopkGraphTensors
{
    Tensor<float> x{std::vector<int64_t>{2, 8}, generateStrides({2, 8})};
    Tensor<int32_t> indices{std::vector<int64_t>{2, 3}, generateStrides({2, 3})};
    Tensor<float> values{std::vector<int64_t>{2, 3}, generateStrides({2, 3})};

    TopkGraphTensors()
    {
        const float data[] = {5, 1, 3, 1, 7, 0, 2, 9, 4, 4, -2, 8, -2, 6, 3, 4};
        auto* xData = x.memory().hostData();
        for(size_t i = 0; i < 16; ++i)
        {
            xData[i] = data[i];
        }
    }

    std::unordered_map<int64_t, void*> variantPack()
    {
        return {{1, x.memory().hostData()},
                {2, indices.memory().hostData()},
                {3, values.memory().hostData()}};
    }

    // Smallest three per row, ties by lower position: row 0 picks 0 (5), 1 (1), 1 (3);
    // row 1 picks -2 (2), -2 (4), 3 (6).
    static std::vector<int32_t> expectedIndices()
    {
        return {5, 1, 3, 2, 4, 6};
    }

    static std::vector<float> expectedValues()
    {
        return {0, 1, 1, -2, -2, 3};
    }

    std::vector<int32_t> indexValues()
    {
        const auto* data = indices.memory().hostData();
        return {data, data + 6};
    }

    std::vector<float> valueValues()
    {
        const auto* data = values.memory().hostData();
        return {data, data + 6};
    }
};

TopkParams makeParams(TopkGraphTensors& tensors, bool withValues)
{
    TopkParams params;
    params.xTensor.uid = 1;
    params.xTensor.data_type = DataType::FLOAT;
    params.xTensor.dims = tensors.x.dims();
    params.xTensor.strides = tensors.x.strides();
    params.indicesTensor.uid = 2;
    params.indicesTensor.data_type = DataType::INT32;
    params.indicesTensor.dims = tensors.indices.dims();
    params.indicesTensor.strides = tensors.indices.strides();
    if(withValues)
    {
        TensorAttributesT valuesTensor;
        valuesTensor.uid = 3;
        valuesTensor.data_type = DataType::FLOAT;
        valuesTensor.dims = tensors.values.dims();
        valuesTensor.strides = tensors.values.strides();
        params.valuesTensor = valuesTensor;
    }
    params.k = 3;
    params.largest = false;
    return params;
}

} // namespace

TEST(TestTopkPlan, ExecutePlan)
{
    TopkGraphTensors tensors;

    TopkPlan<float, int32_t> patient(makeParams(tensors, true));
    EXPECT_EQ(patient.getOutputTensorIds(), (std::vector<int64_t>{2, 3}));

    patient.execute(tensors.variantPack());

    EXPECT_EQ(tensors.indexValues(), TopkGraphTensors::expectedIndices());
    EXPECT_EQ(tensors.valueValues(), TopkGraphTensors::expectedValues());
}

TEST(TestTopkPlan, ExecutePlanWithoutValues)
{
    TopkGraphTensors tensors;

    TopkPlan<float, int32_t> patient(makeParams(tensors, false));
    EXPECT_EQ(patient.getOutputTensorIds(), (std::vector<int64_t>{2}));

    patient.execute({{1, tensors.x.memory().hostData()}, {2, tensors.indices.memory().hostData()}});

    EXPECT_EQ(tensors.indexValues(), TopkGraphTensors::expectedIndices());
}

TEST(TestTopkPlanBuilder, IsApplicable)
{
    auto builder = createValidTopkGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const TopkPlanBuilder<DataType::FLOAT, DataType::INT32> floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Wrong index data type.
    const TopkPlanBuilder<DataType::FLOAT, DataType::INT64> int64PlanBuilder;
    EXPECT_FALSE(int64PlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Wrong data type.
    const TopkPlanBuilder<DataType::HALF, DataType::INT32> halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing values tensor in tensorMap.
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(3);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // No values tensor in the node.
    auto noValuesBuilder = createValidTopkGraph(false);
    const GraphWrapper noValuesWrap(noValuesBuilder.GetBufferPointer(), noValuesBuilder.GetSize());
    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(noValuesWrap.getNode(0), noValuesWrap.getTensorMap()));

    // Incompatible node type.
    auto otherBuilder = createValidGatherGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(otherGraphWrap.getNode(0), otherGraphWrap.getTensorMap()));
}

TEST(TestTopkPlanBuilder, BuildNodePlan)
{
    const TopkPlanBuilder<DataType::FLOAT, DataType::INT32> patient;

    auto builder = createValidTopkGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));
    ASSERT_NE((dynamic_cast<TopkPlan<float, int32_t>*>(builtPlan.get())), nullptr);
    EXPECT_EQ(builtPlan->getOutputTensorIds(), (std::vector<int64_t>{2, 3}));

    auto otherBuilder = createValidGatherGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_THROW(patient.buildNodePlan(otherGraphWrap, otherGraphWrap.getNode(0)),
                 std::runtime_error);
}

TEST(TestTopkPlan, CpuReferenceGraphExecutorRunsTopkGraph)
{
    TopkGraphTensors tensors;
    auto builder = createValidTopkGraph();

    CpuReferenceGraphExecutor executor;
    executor.execute(builder.GetBufferPointer(), builder.GetSize(), tensors.variantPack());

    EXPECT_EQ(tensors.indexValues(), TopkGraphTensors::expectedIndices());
    EXPECT_EQ(tensors.valueValues(), TopkGraphTensors::expectedValues());
}
