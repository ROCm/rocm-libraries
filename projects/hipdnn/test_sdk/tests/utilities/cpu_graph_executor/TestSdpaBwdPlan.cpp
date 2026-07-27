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
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

TEST(TestSdpaBwdPlan, ExecutePlan)
{
    // [B=1, H=2, Sq=4, Skv=4, D=8] — standard MHA (numHeads == numKvHeads)
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaBwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaBwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    // O and Stats (LSE) must be the actual forward-pass output for Q/K/V:
    // the backward math uses O for the correction term D = sum(dO * O), and
    // SdpaBwdPlan always forwards Stats to backward() as the LSE tensor.
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward<float, float, float, float>(planTensorBundle.qTensor,
                                                            planTensorBundle.kTensor,
                                                            planTensorBundle.vTensor,
                                                            planTensorBundle.oTensor,
                                                            std::nullopt,
                                                            noMask,
                                                            -1,
                                                            -1,
                                                            true,
                                                            &planTensorBundle.statsTensor);
    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor,
                                                            std::nullopt,
                                                            noMask,
                                                            -1,
                                                            -1,
                                                            true,
                                                            &directTensorBundle.statsTensor);

    auto graphTuple = buildSdpaBwdGraph(planTensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto& variantPack = std::get<1>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const SdpaBwdPlanBuilder<DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT>
        builder;
    auto builtPlan = builder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));
    builtPlan->execute(variantPack);

    CpuFpReferenceSdpa::backward<float, float, float, float, float, float, float, float>(
        directTensorBundle.qTensor,
        directTensorBundle.kTensor,
        directTensorBundle.vTensor,
        directTensorBundle.oTensor,
        directTensorBundle.doTensor,
        directTensorBundle.dqTensor,
        directTensorBundle.dkTensor,
        directTensorBundle.dvTensor,
        std::nullopt,
        &directTensorBundle.statsTensor);

    const float tolerance = 1e-4f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dqTensor, planTensorBundle.dqTensor));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dkTensor, planTensorBundle.dkTensor));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dvTensor, planTensorBundle.dvTensor));
}

TEST(TestSdpaBwdPlan, ExecutePlanWithRuntimeScaleFromPack)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaBwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaBwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    // Pure runtime pass-by-value scale delivered through the variant pack.
    // O and Stats (LSE) must be the actual forward-pass output computed with
    // the same scale, since SdpaBwdPlan always forwards Stats as the LSE tensor.
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    float scaleHostValue = 0.25f;
    CpuFpReferenceSdpa::forward<float, float, float, float>(planTensorBundle.qTensor,
                                                            planTensorBundle.kTensor,
                                                            planTensorBundle.vTensor,
                                                            planTensorBundle.oTensor,
                                                            scaleHostValue,
                                                            noMask,
                                                            -1,
                                                            -1,
                                                            true,
                                                            &planTensorBundle.statsTensor);
    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor,
                                                            scaleHostValue,
                                                            noMask,
                                                            -1,
                                                            -1,
                                                            true,
                                                            &directTensorBundle.statsTensor);
    auto graphTuple = buildSdpaBwdGraph(
        planTensorBundle, DataType::FLOAT, /*runtimeScaleHostPtr=*/&scaleHostValue);
    auto& graph = std::get<0>(graphTuple);
    auto& variantPack = std::get<1>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaBackwardAttributes();
    ASSERT_TRUE(nodeAttributes->scale_tensor_uid().has_value());

    const SdpaBwdPlanBuilder<DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT>
        builder;
    auto builtPlan = builder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));
    builtPlan->execute(variantPack);

    // Direct reference with the same explicit scale value.
    CpuFpReferenceSdpa::backward<float, float, float, float, float, float, float, float>(
        directTensorBundle.qTensor,
        directTensorBundle.kTensor,
        directTensorBundle.vTensor,
        directTensorBundle.oTensor,
        directTensorBundle.doTensor,
        directTensorBundle.dqTensor,
        directTensorBundle.dkTensor,
        directTensorBundle.dvTensor,
        scaleHostValue,
        &directTensorBundle.statsTensor);

    const float tolerance = 1e-4f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dqTensor, planTensorBundle.dqTensor));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dkTensor, planTensorBundle.dkTensor));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dvTensor, planTensorBundle.dvTensor));
}
