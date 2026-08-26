// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "MatmulGraphUtils.hpp"
#include "MatmulTensorBundles.hpp"
#include "MoeGroupedMatmulBwdGraphUtils.hpp"
#include "MoeGroupedMatmulBwdTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmulBwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulBwdPlan.hpp>

using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;

namespace
{

std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>, std::unordered_map<int64_t, void*>>
    buildIncompatibleGraph()
{
    const std::vector<int64_t> aDims = {1, 1, 4, 2};
    const std::vector<int64_t> bDims = {1, 1, 2, 3};
    const std::vector<int64_t> cDims = {1, 1, 4, 3};
    static MatmulTensorBundle<float> s_matmulTensorBundle(aDims, bDims, cDims, false, false, 1);
    return buildMatmulGraph(s_matmulTensorBundle, DataType::FLOAT, DataType::FLOAT);
}

} // namespace

class TestMoeGroupedMatmulBwdPlan : public ::testing::Test
{
protected:
    template <typename T>
    static void
        initTensorValues(hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorAttr,
                         DataType dataType,
                         const Tensor<T>& tensor,
                         int64_t uid)
    {
        tensorAttr.data_type = dataType;
        tensorAttr.dims = tensor.dims();
        tensorAttr.strides = tensor.strides();
        tensorAttr.uid = uid;
    }
};

TEST_F(TestMoeGroupedMatmulBwdPlan, ExecutePlan)
{
    constexpr int64_t EXPERTS = 2;
    constexpr int64_t HIDDEN_K = 3;
    constexpr int64_t OUTPUT_N = 4;
    constexpr int64_t TOKEN_ROWS = 8;

    const unsigned int seed = getGlobalTestSeed();
    MoeGroupedMatmulBwdTensorBundle<float> planTensorBundle(
        EXPERTS, HIDDEN_K, OUTPUT_N, TOKEN_ROWS, seed);
    MoeGroupedMatmulBwdTensorBundle<float> directTensorBundle(
        EXPERTS, HIDDEN_K, OUTPUT_N, TOKEN_ROWS, seed);

    MoeGroupedMatmulBwdParams params;
    initTensorValues(params.doutputTensor, DataType::FLOAT, planTensorBundle.doutputTensor, 1);
    initTensorValues(params.tokenTensor, DataType::FLOAT, planTensorBundle.tokenTensor, 2);
    initTensorValues(
        params.firstTokenOffsetTensor, DataType::INT32, planTensorBundle.firstTokenOffsetTensor, 3);
    initTensorValues(params.dweightTensor, DataType::FLOAT, planTensorBundle.dweightTensor, 4);

    MoeGroupedMatmulBwdPlan<float, float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.doutputTensor.memory().hostData();
    variantPack[2] = planTensorBundle.tokenTensor.memory().hostData();
    variantPack[3] = planTensorBundle.firstTokenOffsetTensor.memory().hostData();
    variantPack[4] = planTensorBundle.dweightTensor.memory().hostData();

    CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
        directTensorBundle.doutputTensor,
        directTensorBundle.tokenTensor,
        directTensorBundle.firstTokenOffsetTensor,
        directTensorBundle.dweightTensor);

    patient.execute(variantPack);

    const float tolerance = matmul::getTolerance<float>();
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.dweightTensor,
                                                planTensorBundle.dweightTensor));
}

TEST(TestMoeGroupedMatmulBwdPlanBuilder, IsApplicable)
{
    MoeGroupedMatmulBwdTensorBundle<float> tensorBundle(2, 3, 4, 8, getGlobalTestSeed());

    auto graphTuple = buildMoeGroupedMatmulBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());

    // Correct case
    const MoeGroupedMatmulBwdPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Mismatched compute data type
    const MoeGroupedMatmulBwdPlanBuilder<DataType::HALF,
                                         DataType::HALF,
                                         DataType::HALF,
                                         DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor in tensorMap
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // Incorrect tensor data types
    const MoeGroupedMatmulBwdPlanBuilder<DataType::HALF,
                                         DataType::HALF,
                                         DataType::HALF,
                                         DataType::FLOAT>
        mixedPlanBuilder;
    EXPECT_FALSE(mixedPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Incompatible node type
    auto [serializedMatmulGraph, matmulSerErr] = std::get<0>(buildIncompatibleGraph())->to_binary();
    ASSERT_TRUE(matmulSerErr.is_good()) << matmulSerErr.get_message();
    const GraphWrapper matmulGraphWrap(serializedMatmulGraph.data(), serializedMatmulGraph.size());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(matmulGraphWrap.getNode(0), matmulGraphWrap.getTensorMap()));
}

TEST(TestMoeGroupedMatmulBwdPlanBuilder, BuildNodePlan)
{
    const MoeGroupedMatmulBwdPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;

    // Correct case
    {
        MoeGroupedMatmulBwdTensorBundle<float> tensorBundle(2, 3, 4, 8, getGlobalTestSeed());
        auto graphTuple
            = buildMoeGroupedMatmulBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);
        auto& graph = std::get<0>(graphTuple);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_NO_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)));
    }

    // Incompatible node type
    {
        auto [serializedGraph, serErr] = std::get<0>(buildIncompatibleGraph())->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
    }
}

TEST(TestMoeGroupedMatmulBwdPlanBuilder, PlanConstruction)
{
    MoeGroupedMatmulBwdTensorBundle<float> tensorBundle(2, 3, 4, 8, getGlobalTestSeed());
    auto graphTuple = buildMoeGroupedMatmulBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());

    const MoeGroupedMatmulBwdPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<MoeGroupedMatmulBwdPlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}
