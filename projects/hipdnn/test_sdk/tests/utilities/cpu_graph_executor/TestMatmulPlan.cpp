// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "MatmulGraphUtils.hpp"
#include "MatmulTensorBundles.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/MatmulPlan.hpp>

using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_plugin_sdk;
using namespace ::testing;

class TestMatmulPlan : public ::testing::Test
{
protected:
    static void initTensorValues(hipdnn_data_sdk::data_objects::TensorAttributesT& tensorAttr,
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
    std::vector<int64_t> aDims = {2, 3};
    std::vector<int64_t> bDims = {3, 4};
    std::vector<int64_t> cDims = {2, 4};

    unsigned int seed = getGlobalTestSeed();
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

    CpuFpReferenceValidation<float> cpuRefOutputValidation(matmul::getTolerance<float>(),
                                                           matmul::getTolerance<float>());
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.cTensor, planTensorBundle.cTensor));
}

TEST(TestMatmulPlanBuilder, PlanConstruction)
{
    std::vector<int64_t> aDims = {2, 2, 3};
    std::vector<int64_t> bDims = {2, 3, 4};
    std::vector<int64_t> cDims = {2, 2, 4};

    MatmulTensorBundle<float> tensorBundle(aDims, bDims, cDims, false, false, getGlobalTestSeed());

    auto graphTuple = buildMatmulGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    GraphWrapper graphWrap(flatbufferGraph.data(), flatbufferGraph.size());

    MatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    bool result = dynamic_cast<MatmulPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestMatmulPlanBuilder, IsApplicable)
{
    std::vector<int64_t> aDims = {2, 2, 3};
    std::vector<int64_t> bDims = {2, 3, 4};
    std::vector<int64_t> cDims = {2, 2, 4};

    MatmulTensorBundle<float> tensorBundle(aDims, bDims, cDims, false, false, getGlobalTestSeed());

    auto graphTuple = buildMatmulGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    GraphWrapper graphWrap(flatbufferGraph.data(), flatbufferGraph.size());

    MatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    MatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT>
        badTypesPlanBuilder;

    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
