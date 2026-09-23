// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "PointwiseGraphUtils.hpp"
#include "ReductionGraphUtils.hpp"
#include "ReductionTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceReduction.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ReductionPlan.hpp>

using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;

class TestReductionPlan : public ::testing::Test
{
protected:
    static void initTensorValues(TensorAttributesT& tensorAttr,
                                 DataType dataType,
                                 const Tensor<float>& tensor,
                                 int64_t uid)
    {
        tensorAttr.data_type = dataType;
        tensorAttr.dims = tensor.dims();
        tensorAttr.strides = tensor.strides();
        tensorAttr.uid = uid;
    }
};

TEST_F(TestReductionPlan, ExecutePlan)
{
    // Test that reduction plan actually calls CpuFpReferenceReduction::reduce
    // (or at least produces the same results).

    const std::vector<int64_t> xDims = {2, 3, 4, 4};
    const std::vector<int64_t> yDims = {2, 3, 1, 1};

    const unsigned int seed = getGlobalTestSeed();
    ReductionTensorBundle<float> planTensorBundle(xDims, yDims, seed);
    ReductionTensorBundle<float> directTensorBundle(xDims, yDims, seed);

    ReductionParams params;
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.xTensor, 1);
    initTensorValues(params.yTensor, DataType::FLOAT, planTensorBundle.yTensor, 2);
    params.mode = ReductionMode::ADD;

    ReductionPlan<float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.xTensor.memory().hostData();
    variantPack[2] = planTensorBundle.yTensor.memory().hostData();

    CpuFpReferenceReduction::reduce<float, float, float>(
        directTensorBundle.xTensor, directTensorBundle.yTensor, ReductionMode::ADD);

    patient.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(1e-5f, 1e-5f);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.yTensor, planTensorBundle.yTensor));
}

TEST_F(TestReductionPlan, NotSetModeFailsSerialization)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 4};
    const std::vector<int64_t> yDims = {2, 3, 1, 1};

    const unsigned int seed = getGlobalTestSeed();
    ReductionTensorBundle<float> tensorBundle(xDims, yDims, seed);

    auto graphTuple = buildReductionGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, hipdnn_frontend::ReductionMode::NOT_SET);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    EXPECT_FALSE(serErr.is_good());
}

TEST(TestReductionPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 4};
    const std::vector<int64_t> yDims = {2, 3, 1, 1};

    ReductionTensorBundle<float> tensorBundle(xDims, yDims, getGlobalTestSeed());

    auto graphTuple = buildReductionGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());

    // Correct case
    const ReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Mismatched tensor data types
    const ReductionPlanBuilder<DataType::HALF, DataType::HALF, DataType::FLOAT> halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor in tensorMap
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(tensorMapCopy.begin());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // Incompatible node type
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graphPointwiseTuple = buildPointwiseUnaryGraph(dims,
                                                        dims,
                                                        DataType::FLOAT,
                                                        DataType::FLOAT,
                                                        DataType::FLOAT,
                                                        hipdnn_frontend::PointwiseMode::RELU_FWD,
                                                        1,
                                                        TensorLayout::NCHW);

    auto [serializedGraphPointwise, serErrPointwise]
        = std::get<0>(graphPointwiseTuple)->to_binary();
    ASSERT_TRUE(serErrPointwise.is_good()) << serErrPointwise.get_message();
    const GraphWrapper graphWrapPointwise(serializedGraphPointwise.data(),
                                          serializedGraphPointwise.size());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapPointwise.getNode(0),
                                               graphWrapPointwise.getTensorMap()));
}

TEST(TestReductionPlanBuilder, BuildNodePlan)
{
    const ReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    // Correct case
    {
        const std::vector<int64_t> xDims = {2, 3, 4, 4};
        const std::vector<int64_t> yDims = {2, 3, 1, 1};

        ReductionTensorBundle<float> tensorBundle(xDims, yDims, getGlobalTestSeed());

        auto graphTuple = buildReductionGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

        auto& graph = std::get<0>(graphTuple);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_NO_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)));
    }

    // Incompatible node type
    {
        const std::vector<int64_t> dims = {1, 3, 4, 4};
        auto graphTuple = buildPointwiseUnaryGraph(dims,
                                                   dims,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                                   1,
                                                   TensorLayout::NCHW);

        auto& graph = std::get<0>(graphTuple);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
    }
}

TEST(TestReductionPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 4};
    const std::vector<int64_t> yDims = {2, 3, 1, 1};

    ReductionTensorBundle<float> tensorBundle(xDims, yDims, getGlobalTestSeed());

    auto graphTuple = buildReductionGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());

    const ReductionPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<ReductionPlan<float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}
