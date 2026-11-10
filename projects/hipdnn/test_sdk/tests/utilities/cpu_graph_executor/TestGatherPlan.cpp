// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/GatherPlan.hpp>

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

// createValidGatherGraph(): x [2, 5, 4] FLOAT (uid 1), indices [3] INT64 (uid 2), axis 1,
// y [2, 3, 4] FLOAT (uid 3).
struct GatherGraphTensors
{
    Tensor<float> x{std::vector<int64_t>{2, 5, 4}, generateStrides({2, 5, 4})};
    Tensor<int64_t> indices{std::vector<int64_t>{3}, std::vector<int64_t>{1}};
    Tensor<float> y{std::vector<int64_t>{2, 3, 4}, generateStrides({2, 3, 4})};

    GatherGraphTensors()
    {
        auto* xData = x.memory().hostData();
        for(int64_t i = 0; i < 40; ++i)
        {
            xData[i] = static_cast<float>(i);
        }
        auto* indexData = indices.memory().hostData();
        indexData[0] = 4;
        indexData[1] = 0;
        indexData[2] = 2;
    }

    std::unordered_map<int64_t, void*> variantPack()
    {
        return {{1, x.memory().hostData()},
                {2, indices.memory().hostData()},
                {3, y.memory().hostData()}};
    }

    // y[a, j, c] = x[a, indices[j], c] = 20 * a + 4 * indices[j] + c.
    static std::vector<float> expected()
    {
        std::vector<float> values;
        const int64_t gathered[] = {4, 0, 2};
        for(int64_t a = 0; a < 2; ++a)
        {
            for(const auto row : gathered)
            {
                for(int64_t c = 0; c < 4; ++c)
                {
                    values.push_back(static_cast<float>(20 * a + 4 * row + c));
                }
            }
        }
        return values;
    }

    std::vector<float> yValues()
    {
        const auto* yData = y.memory().hostData();
        return {yData, yData + 24};
    }
};

} // namespace

TEST(TestGatherPlan, ExecutePlan)
{
    GatherGraphTensors tensors;

    GatherParams params;
    params.xTensor.uid = 1;
    params.xTensor.data_type = DataType::FLOAT;
    params.xTensor.dims = tensors.x.dims();
    params.xTensor.strides = tensors.x.strides();
    params.indicesTensor.uid = 2;
    params.indicesTensor.data_type = DataType::INT64;
    params.indicesTensor.dims = tensors.indices.dims();
    params.indicesTensor.strides = tensors.indices.strides();
    params.yTensor.uid = 3;
    params.yTensor.data_type = DataType::FLOAT;
    params.yTensor.dims = tensors.y.dims();
    params.yTensor.strides = tensors.y.strides();
    params.axis = 1;

    GatherPlan<float, int64_t> patient(std::move(params));
    EXPECT_EQ(patient.getOutputTensorIds(), (std::vector<int64_t>{3}));

    patient.execute(tensors.variantPack());

    EXPECT_EQ(tensors.yValues(), GatherGraphTensors::expected());
}

TEST(TestGatherPlanBuilder, IsApplicable)
{
    auto builder = createValidGatherGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const GatherPlanBuilder<DataType::FLOAT, DataType::INT64> floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Wrong index data type.
    const GatherPlanBuilder<DataType::FLOAT, DataType::INT32> int32PlanBuilder;
    EXPECT_FALSE(int32PlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Wrong data type.
    const GatherPlanBuilder<DataType::HALF, DataType::INT64> halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor in tensorMap.
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // Incompatible node type.
    auto otherBuilder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(otherGraphWrap.getNode(0), otherGraphWrap.getTensorMap()));
}

TEST(TestGatherPlanBuilder, BuildNodePlan)
{
    const GatherPlanBuilder<DataType::FLOAT, DataType::INT64> patient;

    auto builder = createValidGatherGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));
    EXPECT_NE((dynamic_cast<GatherPlan<float, int64_t>*>(builtPlan.get())), nullptr);

    auto otherBuilder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_THROW(patient.buildNodePlan(otherGraphWrap, otherGraphWrap.getNode(0)),
                 std::runtime_error);
}

TEST(TestGatherPlan, CpuReferenceGraphExecutorRunsGatherGraph)
{
    GatherGraphTensors tensors;
    auto builder = createValidGatherGraph();

    CpuReferenceGraphExecutor executor;
    executor.execute(builder.GetBufferPointer(), builder.GetSize(), tensors.variantPack());

    EXPECT_EQ(tensors.yValues(), GatherGraphTensors::expected());
}
