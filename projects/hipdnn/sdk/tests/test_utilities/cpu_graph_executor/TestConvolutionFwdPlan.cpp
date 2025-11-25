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
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionFwdPlan.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_plugin;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestConvolutionFwdPlan : public ::testing::Test
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

TEST_F(TestConvolutionFwdPlan, ExecutePlan)
{
    std::vector<int64_t> xDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 2, 2};

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    unsigned int seed = getGlobalTestSeed();
    ConvolutionFwdTensorBundle<float> planTensorBundle(
        xDims, wDims, yDims, seed, TensorLayout::NHWC);
    ConvolutionFwdTensorBundle<float> directTensorBundle(
        xDims, wDims, yDims, seed, TensorLayout::NHWC);

    ConvolutionFwdParams params;
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.xTensor, 1);
    initTensorValues(params.wTensor, DataType::FLOAT, planTensorBundle.wTensor, 2);
    initTensorValues(params.yTensor, DataType::FLOAT, planTensorBundle.yTensor, 3);
    params.stride = strides;
    params.dilation = dilation;
    params.prePadding = padding;
    params.postPadding = padding;

    ConvolutionFwdPlan<float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.xTensor.memory().hostData();
    variantPack[2] = planTensorBundle.wTensor.memory().hostData();
    variantPack[3] = planTensorBundle.yTensor.memory().hostData();

    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(directTensorBundle.xTensor,
                                                                  directTensorBundle.wTensor,
                                                                  directTensorBundle.yTensor,
                                                                  strides,
                                                                  dilation,
                                                                  padding);

    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(conv::getToleranceFwd<float>(),
                                                           conv::getToleranceFwd<float>());

    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.yTensor, planTensorBundle.yTensor));
}

TEST(TestConvolutionFwdPlanBuilder, PlanConstruction)
{
    std::vector<int64_t> xDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 2, 2};

    ConvolutionFwdTensorBundle<float> tensorBundle(xDims, wDims, yDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionFwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    ConvolutionFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    bool result = dynamic_cast<ConvolutionFwdPlan<float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestConvolutionFwdPlanBuilder, IsApplicable)
{
    std::vector<int64_t> xDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 2, 2};

    ConvolutionFwdTensorBundle<float> tensorBundle(xDims, wDims, yDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionFwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    ConvolutionFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT> floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    ConvolutionFwdPlanBuilder<DataType::FLOAT, DataType::HALF> badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
