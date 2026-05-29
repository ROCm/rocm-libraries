// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "ConvolutionGraphUtils.hpp"
#include "ConvolutionTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionBwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestConvolutionBwdPlan : public ::testing::Test
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

TEST_F(TestConvolutionBwdPlan, ExecutePlan)
{
    const std::vector<int64_t> dxDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 2, 2};

    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    const unsigned int seed = getGlobalTestSeed();
    ConvolutionBwdTensorBundle<float> planTensorBundle(
        dxDims, wDims, dyDims, seed, TensorLayout::NHWC);
    ConvolutionBwdTensorBundle<float> directTensorBundle(
        dxDims, wDims, dyDims, seed, TensorLayout::NHWC);

    ConvolutionBwdParams params;
    initTensorValues(params.dxTensor, DataType::FLOAT, planTensorBundle.dxTensor, 1);
    initTensorValues(params.wTensor, DataType::FLOAT, planTensorBundle.wTensor, 2);
    initTensorValues(params.dyTensor, DataType::FLOAT, planTensorBundle.dyTensor, 3);
    params.stride = strides;
    params.dilation = dilation;
    params.prePadding = padding;
    params.postPadding = padding;

    ConvolutionBwdPlan<float, float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.dxTensor.memory().hostData();
    variantPack[2] = planTensorBundle.wTensor.memory().hostData();
    variantPack[3] = planTensorBundle.dyTensor.memory().hostData();

    CpuFpReferenceConvolution::dgrad<float, float, float, float>(directTensorBundle.dxTensor,
                                                                 directTensorBundle.wTensor,
                                                                 directTensorBundle.dyTensor,
                                                                 strides,
                                                                 dilation,
                                                                 padding);

    patient.execute(variantPack);

    // Calculate dynamic tolerance for conv dgrad
    // ConvolutionBwdTensorBundle initializes dy and w with range [-1.0, 1.0]
    auto tolerance = conv::calculateConvDgradTolerance<float, float, float>(-1.0,
                                                                            1.0, // dyMin, dyMax
                                                                            -1.0,
                                                                            1.0, // wMin, wMax
                                                                            wDims); // wDims

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);

    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dxTensor, planTensorBundle.dxTensor));
}

TEST(TestConvolutionBwdPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dxDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 2, 2};

    ConvolutionBwdTensorBundle<float> tensorBundle(dxDims, wDims, dyDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const ConvolutionBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<ConvolutionBwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestConvolutionBwdPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dxDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 2, 2};

    ConvolutionBwdTensorBundle<float> tensorBundle(dxDims, wDims, dyDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const ConvolutionBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    const ConvolutionBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::HALF,
                                    DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
