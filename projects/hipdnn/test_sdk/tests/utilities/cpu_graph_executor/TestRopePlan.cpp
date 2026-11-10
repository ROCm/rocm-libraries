// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RopePlan.hpp>

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

constexpr float K_HALF_PI = 1.57079632679489661923f;

// createValidRopeGraph(): x [1, 2, 3, 8] FLOAT (uid 1), freqs [1, 1, 3, 4] FLOAT (uid 2),
// y [1, 2, 3, 8] FLOAT (uid 3), rope_dim 4, rope_offset_ext 0 (or absent), output_scale 0.5.
struct RopeGraphTensors
{
    Tensor<float> x{std::vector<int64_t>{1, 2, 3, 8}, generateStrides({1, 2, 3, 8})};
    Tensor<float> freqs{std::vector<int64_t>{1, 1, 3, 4}, generateStrides({1, 1, 3, 4})};
    Tensor<float> y{std::vector<int64_t>{1, 2, 3, 8}, generateStrides({1, 2, 3, 8})};

    // x[row, j] = j + 1 for every row; all angles pi/2.
    RopeGraphTensors()
    {
        auto* xData = x.memory().hostData();
        for(int64_t i = 0; i < 48; ++i)
        {
            xData[i] = static_cast<float>(i % 8 + 1);
        }
        auto* freqsData = freqs.memory().hostData();
        for(int64_t i = 0; i < 12; ++i)
        {
            freqsData[i] = K_HALF_PI;
        }
    }

    std::unordered_map<int64_t, void*> variantPack()
    {
        return {
            {1, x.memory().hostData()}, {2, freqs.memory().hostData()}, {3, y.memory().hostData()}};
    }

    // Leading segment [1, 2, 3, 4] -> [-3, -4, 1, 2], passthrough 5..8, all scaled by 0.5.
    static std::vector<float> expectedLeading()
    {
        return repeatRow({-1.5f, -2.0f, 0.5f, 1.0f, 2.5f, 3.0f, 3.5f, 4.0f});
    }

    // Trailing segment [5, 6, 7, 8] -> [-7, -8, 5, 6], passthrough 1..4, all scaled by 0.5.
    static std::vector<float> expectedTrailing()
    {
        return repeatRow({0.5f, 1.0f, 1.5f, 2.0f, -3.5f, -4.0f, 2.5f, 3.0f});
    }

    static std::vector<float> repeatRow(const std::vector<float>& row)
    {
        std::vector<float> values;
        for(int64_t r = 0; r < 6; ++r)
        {
            values.insert(values.end(), row.begin(), row.end());
        }
        return values;
    }

    void expectY(const std::vector<float>& expected)
    {
        const auto* yData = y.memory().hostData();
        ASSERT_EQ(expected.size(), 48U);
        for(size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(yData[i], expected[i], 1e-5f) << "at index " << i;
        }
    }
};

} // namespace

TEST(TestRopePlan, ExecutePlan)
{
    RopeGraphTensors tensors;

    RopeParams params;
    params.xTensor.uid = 1;
    params.xTensor.data_type = DataType::FLOAT;
    params.xTensor.dims = tensors.x.dims();
    params.xTensor.strides = tensors.x.strides();
    params.freqsTensor.uid = 2;
    params.freqsTensor.data_type = DataType::FLOAT;
    params.freqsTensor.dims = tensors.freqs.dims();
    params.freqsTensor.strides = tensors.freqs.strides();
    params.yTensor.uid = 3;
    params.yTensor.data_type = DataType::FLOAT;
    params.yTensor.dims = tensors.y.dims();
    params.yTensor.strides = tensors.y.strides();
    params.outputScale = 0.5f;
    params.ropeDim = 4;
    params.ropeOffset = 0;

    RopePlan<float> patient(std::move(params));
    EXPECT_EQ(patient.getOutputTensorIds(), (std::vector<int64_t>{3}));

    patient.execute(tensors.variantPack());

    tensors.expectY(RopeGraphTensors::expectedLeading());
}

TEST(TestRopePlanBuilder, IsApplicable)
{
    auto builder = createValidRopeGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const RopePlanBuilder<DataType::FLOAT> floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Wrong data type.
    const RopePlanBuilder<DataType::HALF> halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor in tensorMap.
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // Incompatible node type.
    auto otherBuilder = createValidGatherGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(otherGraphWrap.getNode(0), otherGraphWrap.getTensorMap()));
}

TEST(TestRopePlanBuilder, BuildNodePlan)
{
    const RopePlanBuilder<DataType::FLOAT> patient;

    auto builder = createValidRopeGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));
    EXPECT_NE(dynamic_cast<RopePlan<float>*>(builtPlan.get()), nullptr);

    auto otherBuilder = createValidGatherGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_THROW(patient.buildNodePlan(otherGraphWrap, otherGraphWrap.getNode(0)),
                 std::runtime_error);
}

TEST(TestRopePlan, CpuReferenceGraphExecutorRunsLeadingSegmentGraph)
{
    RopeGraphTensors tensors;
    auto builder = createValidRopeGraph();

    CpuReferenceGraphExecutor executor;
    executor.execute(builder.GetBufferPointer(), builder.GetSize(), tensors.variantPack());

    tensors.expectY(RopeGraphTensors::expectedLeading());
}

TEST(TestRopePlan, CpuReferenceGraphExecutorRunsTrailingSegmentGraph)
{
    RopeGraphTensors tensors;
    auto builder = createValidRopeGraph(false);

    CpuReferenceGraphExecutor executor;
    executor.execute(builder.GetBufferPointer(), builder.GetSize(), tensors.variantPack());

    tensors.expectY(RopeGraphTensors::expectedTrailing());
}
