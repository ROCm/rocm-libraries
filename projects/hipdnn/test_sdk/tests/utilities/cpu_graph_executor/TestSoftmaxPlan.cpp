// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SoftmaxPlan.hpp>

#include <cmath>
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

constexpr float K_TOLERANCE = 1e-6f;

// createValidSoftmaxGraph(): p [2, 3, 5] FLOAT (uid 1), s [2, 3, 5] (uid 2), stats [2, 3, 1]
// (uid 3) in log2 units. Every row holds p[j] = ln(j + 1), so exp(p - max) = (j + 1) / 5,
// sum = 3, s[j] = (j + 1) / 15 and stats = ln 5 + ln 3 = ln 15, i.e. log2(15) in log2 units.
struct SoftmaxGraphTensors
{
    Tensor<float> p{std::vector<int64_t>{2, 3, 5}, generateStrides({2, 3, 5})};
    Tensor<float> s{std::vector<int64_t>{2, 3, 5}, generateStrides({2, 3, 5})};
    Tensor<float> stats{std::vector<int64_t>{2, 3, 1}, generateStrides({2, 3, 1})};

    SoftmaxGraphTensors()
    {
        auto* pData = p.memory().hostData();
        for(int64_t row = 0; row < 6; ++row)
        {
            for(int64_t j = 0; j < 5; ++j)
            {
                pData[row * 5 + j] = std::log(static_cast<float>(j + 1));
            }
        }
    }

    std::unordered_map<int64_t, void*> variantPack(bool withStats = true)
    {
        std::unordered_map<int64_t, void*> pack{{1, p.memory().hostData()},
                                                {2, s.memory().hostData()}};
        if(withStats)
        {
            pack[3] = stats.memory().hostData();
        }
        return pack;
    }

    void expectS()
    {
        const auto* sData = s.memory().hostData();
        for(int64_t row = 0; row < 6; ++row)
        {
            for(int64_t j = 0; j < 5; ++j)
            {
                EXPECT_NEAR(sData[row * 5 + j], static_cast<float>(j + 1) / 15.0f, K_TOLERANCE);
            }
        }
    }

    void expectStatsLog2()
    {
        const auto* statsData = stats.memory().hostData();
        for(int64_t row = 0; row < 6; ++row)
        {
            EXPECT_NEAR(statsData[row], std::log2(15.0f), 1e-5f);
        }
    }
};

SoftmaxParams makeParams(const SoftmaxGraphTensors& tensors, bool withStats)
{
    SoftmaxParams params;
    params.pTensor.uid = 1;
    params.pTensor.data_type = DataType::FLOAT;
    params.pTensor.dims = tensors.p.dims();
    params.pTensor.strides = tensors.p.strides();
    params.sTensor.uid = 2;
    params.sTensor.data_type = DataType::FLOAT;
    params.sTensor.dims = tensors.s.dims();
    params.sTensor.strides = tensors.s.strides();
    if(withStats)
    {
        TensorAttributesT statsAttributes;
        statsAttributes.uid = 3;
        statsAttributes.data_type = DataType::FLOAT;
        statsAttributes.dims = tensors.stats.dims();
        statsAttributes.strides = tensors.stats.strides();
        params.statsTensor = statsAttributes;
    }
    params.statsUseLog2 = true;
    return params;
}

} // namespace

TEST(TestSoftmaxPlan, ExecutePlanWithStats)
{
    SoftmaxGraphTensors tensors;

    SoftmaxPlan<float, float, float> patient(makeParams(tensors, true));
    EXPECT_EQ(patient.getOutputTensorIds(), (std::vector<int64_t>{2, 3}));

    patient.execute(tensors.variantPack());

    tensors.expectS();
    tensors.expectStatsLog2();
}

TEST(TestSoftmaxPlan, ExecutePlanWithoutStats)
{
    SoftmaxGraphTensors tensors;

    SoftmaxPlan<float, float, float> patient(makeParams(tensors, false));
    EXPECT_EQ(patient.getOutputTensorIds(), (std::vector<int64_t>{2}));

    patient.execute(tensors.variantPack(false));

    tensors.expectS();
}

TEST(TestSoftmaxPlanBuilder, IsApplicable)
{
    auto builder = createValidSoftmaxGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const SoftmaxPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Wrong data type.
    const SoftmaxPlanBuilder<DataType::HALF, DataType::HALF, DataType::FLOAT> halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing required tensor in tensorMap.
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // Missing optional stats tensor referenced by the node.
    auto statsMissing = graphWrap.getTensorMap();
    statsMissing.erase(3);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), statsMissing));

    // Incompatible node type.
    auto otherBuilder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(otherGraphWrap.getNode(0), otherGraphWrap.getTensorMap()));
}

TEST(TestSoftmaxPlanBuilder, IsApplicableWithoutStats)
{
    auto builder = createValidSoftmaxGraph(false);
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const SoftmaxPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestSoftmaxPlanBuilder, BuildNodePlan)
{
    const SoftmaxPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto builder = createValidSoftmaxGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));
    auto* softmaxPlan = dynamic_cast<SoftmaxPlan<float, float, float>*>(builtPlan.get());
    ASSERT_NE(softmaxPlan, nullptr);
    EXPECT_EQ(softmaxPlan->getOutputTensorIds(), (std::vector<int64_t>{2, 3}));

    auto otherBuilder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_THROW(patient.buildNodePlan(otherGraphWrap, otherGraphWrap.getNode(0)),
                 std::runtime_error);
}

TEST(TestSoftmaxPlan, CpuReferenceGraphExecutorRunsSoftmaxGraph)
{
    SoftmaxGraphTensors tensors;
    auto builder = createValidSoftmaxGraph();

    CpuReferenceGraphExecutor executor;
    executor.execute(builder.GetBufferPointer(), builder.GetSize(), tensors.variantPack());

    tensors.expectS();
    tensors.expectStatsLog2();
}
