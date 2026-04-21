// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

TEST(TestSdpaFwdPlan, ExecutePlan)
{
    // [B=1, H=2, Sq=4, Skv=4, D=8] — standard MHA (numHeads == numKvHeads)
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    SdpaFwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                         *tensorMap.at(nodeAttributes->k_tensor_uid()),
                         *tensorMap.at(nodeAttributes->v_tensor_uid()),
                         *tensorMap.at(nodeAttributes->o_tensor_uid()),
                         std::nullopt,
                         false);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();

    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor);

    SdpaFwdPlan<float, float, float, float> patient(std::move(params));
    patient.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor));
}

TEST(TestSdpaFwdPlan, ExecutePlanWithCausalMask)
{
    // [B=1, H=2, Sq=4, Skv=4, D=8] with causal mask
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle, DataType::FLOAT, /*causalMask=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    SdpaFwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                         *tensorMap.at(nodeAttributes->k_tensor_uid()),
                         *tensorMap.at(nodeAttributes->v_tensor_uid()),
                         *tensorMap.at(nodeAttributes->o_tensor_uid()),
                         std::nullopt,
                         /*causalMask=*/true);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();

    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor,
                                                            std::nullopt,
                                                            noMask,
                                                            /*causalMask=*/true);

    SdpaFwdPlan<float, float, float, float> patient(std::move(params));
    patient.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor));
}

TEST(TestSdpaFwdPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<SdpaFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestSdpaFwdPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    // Correct data types: applicable
    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Mismatched data types: not applicable
    const SdpaFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(
        halfPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Missing tensor in map: not applicable
    auto tensorMapCopy = graphWrapper.getTensorMap();
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    tensorMapCopy.erase(nodeAttributes->k_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}
