// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace
{

using SdpaBwdPlanBuilderFp32 = SdpaBwdPlanBuilder<DataType::FLOAT,
                                                  DataType::FLOAT,
                                                  DataType::FLOAT,
                                                  DataType::FLOAT,
                                                  DataType::FLOAT,
                                                  DataType::FLOAT,
                                                  DataType::FLOAT,
                                                  DataType::FLOAT>;

struct MaskSpelling
{
    bool causalMask;
    bool causalMaskBottomRight;
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
    hipdnn_frontend::DiagonalAlignment alignment;
};

struct DeprecatedCausalMaskMergeCase
{
    const char* name;
    MaskSpelling deprecated;
    int64_t mergedLeftBound;
    int64_t mergedRightBound;
    hipdnn_frontend::DiagonalAlignment mergedAlignment;
};

class TestSdpaBwdPlanDeprecatedCausalMask : public TestWithParam<DeprecatedCausalMaskMergeCase>
{
};

void executeSdpaBwdPlan(SdpaBwdTensorBundle<float>& bundle, const MaskSpelling& mask)
{
    auto graphTuple = buildSdpaBwdGraph(bundle,
                                        DataType::FLOAT,
                                        /*runtimeScaleHostPtr=*/nullptr,
                                        mask.causalMask,
                                        mask.causalMaskBottomRight,
                                        mask.leftBound,
                                        mask.rightBound,
                                        mask.alignment);
    auto& graph = std::get<0>(graphTuple);
    auto& variantPack = std::get<1>(graphTuple);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());

    const SdpaBwdPlanBuilderFp32 builder;
    builder.buildNodePlan(wrapper, wrapper.getNode(0))->execute(variantPack);
}

} // namespace

TEST_P(TestSdpaBwdPlanDeprecatedCausalMask, MatchesMergedExplicitBounds)
{
    const auto& param = GetParam();
    // Sq != Skv so TOP_LEFT and BOTTOM_RIGHT alignments produce different gradients.
    const std::vector<int64_t> qDims = {1, 2, 2, 8};
    const std::vector<int64_t> kvDims = {1, 2, 4, 8};
    const unsigned int seed = getGlobalTestSeed();
    SdpaBwdTensorBundle<float> deprecatedBundle(qDims, kvDims, kvDims, seed);
    SdpaBwdTensorBundle<float> mergedBundle(qDims, kvDims, kvDims, seed);

    const bool mergedTopLeft
        = param.mergedAlignment == hipdnn_frontend::DiagonalAlignment::TOP_LEFT;
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    for(auto* bundle : {&deprecatedBundle, &mergedBundle})
    {
        CpuFpReferenceSdpa::forward<float, float, float, float>(bundle->qTensor,
                                                                bundle->kTensor,
                                                                bundle->vTensor,
                                                                bundle->oTensor,
                                                                std::nullopt,
                                                                noMask,
                                                                param.mergedLeftBound,
                                                                param.mergedRightBound,
                                                                mergedTopLeft,
                                                                &bundle->statsTensor);
    }

    executeSdpaBwdPlan(deprecatedBundle, param.deprecated);
    executeSdpaBwdPlan(
        mergedBundle,
        {false, false, param.mergedLeftBound, param.mergedRightBound, param.mergedAlignment});

    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    EXPECT_TRUE(exactValidation.allClose(deprecatedBundle.dqTensor, mergedBundle.dqTensor));
    EXPECT_TRUE(exactValidation.allClose(deprecatedBundle.dkTensor, mergedBundle.dkTensor));
    EXPECT_TRUE(exactValidation.allClose(deprecatedBundle.dvTensor, mergedBundle.dvTensor));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestSdpaBwdPlanDeprecatedCausalMask,
    Values(
        DeprecatedCausalMaskMergeCase{
            "Causal",
            {true, false, std::nullopt, std::nullopt, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            -1,
            0,
            hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
        DeprecatedCausalMaskMergeCase{
            "BottomRight",
            {false, true, std::nullopt, std::nullopt, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            -1,
            0,
            hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT},
        DeprecatedCausalMaskMergeCase{"CausalWithBottomRightAlignment",
                                      {true,
                                       false,
                                       std::nullopt,
                                       std::nullopt,
                                       hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT},
                                      -1,
                                      0,
                                      hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT},
        DeprecatedCausalMaskMergeCase{
            "CausalWithLeftBound",
            {true, false, 1, std::nullopt, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            1,
            0,
            hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
        DeprecatedCausalMaskMergeCase{
            "CausalWithUnboundedRight",
            {true, false, std::nullopt, -1, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            -1,
            0,
            hipdnn_frontend::DiagonalAlignment::TOP_LEFT}),
    [](const TestParamInfo<DeprecatedCausalMaskMergeCase>& info) {
        return std::string(info.param.name);
    });

TEST(TestSdpaBwdPlan, BothDeprecatedCausalMasksThrow)
{
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    SdpaBwdTensorBundle<float> bundle(dims, dims, dims, getGlobalTestSeed());

    auto graphTuple = buildSdpaBwdGraph(bundle,
                                        DataType::FLOAT,
                                        /*runtimeScaleHostPtr=*/nullptr,
                                        /*causalMask=*/true,
                                        /*causalMaskBottomRight=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());

    const SdpaBwdPlanBuilderFp32 builder;
    EXPECT_THROW(builder.buildNodePlan(wrapper, wrapper.getNode(0)), std::invalid_argument);
}
