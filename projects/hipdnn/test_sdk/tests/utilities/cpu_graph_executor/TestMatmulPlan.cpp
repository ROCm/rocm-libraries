// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "MatmulGraphUtils.hpp"
#include "MatmulTensorBundles.hpp"
#include "PointwiseGraphUtils.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MatmulPlan.hpp>

using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;

class TestMatmulPlan : public ::testing::Test
{
protected:
    static void
        initTensorValues(hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorAttr,
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

TEST_F(TestMatmulPlan, ExecutePlan)
{
    const std::vector<int64_t> aDims = {2, 3};
    const std::vector<int64_t> bDims = {3, 4};
    const std::vector<int64_t> cDims = {2, 4};

    const unsigned int seed = getGlobalTestSeed();
    MatmulTensorBundle<float> planTensorBundle(aDims, bDims, cDims, false, false, seed);
    MatmulTensorBundle<float> directTensorBundle(aDims, bDims, cDims, false, false, seed);

    MatmulParams params;
    initTensorValues(params.aTensor, DataType::FLOAT, planTensorBundle.aTensor, 1);
    initTensorValues(params.bTensor, DataType::FLOAT, planTensorBundle.bTensor, 2);
    initTensorValues(params.cTensor, DataType::FLOAT, planTensorBundle.cTensor, 3);

    MatmulPlan<float, float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.aTensor.memory().hostData();
    variantPack[2] = planTensorBundle.bTensor.memory().hostData();
    variantPack[3] = planTensorBundle.cTensor.memory().hostData();

    CpuFpReferenceMatmul::matmul<float, float, float, float>(
        directTensorBundle.aTensor, directTensorBundle.bTensor, directTensorBundle.cTensor);

    patient.execute(variantPack);

    auto tolerance
        = hipdnn_test_sdk::utilities::matmul::calculateMatmulTolerance<float, float, float>(
            planTensorBundle.aTensor, planTensorBundle.bTensor);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.cTensor, planTensorBundle.cTensor));
}

TEST(TestMatmulPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> aDims = {2, 2, 3};
    const std::vector<int64_t> bDims = {2, 3, 4};
    const std::vector<int64_t> cDims = {2, 2, 4};

    MatmulTensorBundle<float> tensorBundle(aDims, bDims, cDims, false, false, getGlobalTestSeed());

    auto graphTuple = buildMatmulGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());

    // Correct case
    const MatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Mismatched compute data type
    const MatmulPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missed tensor in tensorMap
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    const MatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // Incorrect tensor data types
    const MatmulPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT>
        mixedPlanBuilder;
    EXPECT_FALSE(mixedPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Uncompatible node type
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

TEST(TestMatmulPlanBuilder, BuildNodePlan)
{
    const MatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    // Correct case
    {
        const std::vector<int64_t> aDims = {2, 2, 3};
        const std::vector<int64_t> bDims = {2, 3, 4};
        const std::vector<int64_t> cDims = {2, 2, 4};

        MatmulTensorBundle<float> tensorBundle(
            aDims, bDims, cDims, false, false, getGlobalTestSeed());

        auto graphTuple = buildMatmulGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

        auto& graph = std::get<0>(graphTuple);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_NO_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)));
    }

    // Uncompatible node type
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

TEST(TestMatmulPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> aDims = {2, 2, 3};
    const std::vector<int64_t> bDims = {2, 3, 4};
    const std::vector<int64_t> cDims = {2, 2, 4};

    MatmulTensorBundle<float> tensorBundle(aDims, bDims, cDims, false, false, getGlobalTestSeed());

    auto graphTuple = buildMatmulGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());

    const MatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<MatmulPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}
