// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "ConvolutionGraphUtils.hpp"
#include "ConvolutionTensorBundles.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionBwdPlan.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_plugin;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestConvolutionBwdPlan : public ::testing::Test
{
protected:
    static void initTensorValues(hipdnn_sdk::data_objects::TensorAttributesT& tensorAttr,
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
    std::vector<int64_t> dxDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> dyDims = {1, 1, 2, 2};

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    unsigned int seed = getGlobalTestSeed();
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

    ConvolutionBwdPlan<float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.dxTensor.memory().hostData();
    variantPack[2] = planTensorBundle.wTensor.memory().hostData();
    variantPack[3] = planTensorBundle.dyTensor.memory().hostData();

    CpuFpReferenceConvolutionImpl<float, float>::convBwdData(directTensorBundle.dxTensor,
                                                             directTensorBundle.wTensor,
                                                             directTensorBundle.dyTensor,
                                                             strides,
                                                             dilation,
                                                             padding);

    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(conv::getToleranceBwd<float>(),
                                                           conv::getToleranceBwd<float>());

    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dxTensor, planTensorBundle.dxTensor));
}

TEST(TestConvolutionBwdPlanBuilder, PlanConstruction)
{
    std::vector<int64_t> dxDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> dyDims = {1, 1, 2, 2};

    ConvolutionBwdTensorBundle<float> tensorBundle(dxDims, wDims, dyDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    ConvolutionBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    bool result = dynamic_cast<ConvolutionBwdPlan<float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestConvolutionBwdPlanBuilder, IsApplicable)
{
    std::vector<int64_t> dxDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> dyDims = {1, 1, 2, 2};

    ConvolutionBwdTensorBundle<float> tensorBundle(dxDims, wDims, dyDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    ConvolutionBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT> floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    ConvolutionBwdPlanBuilder<DataType::FLOAT, DataType::HALF> badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
