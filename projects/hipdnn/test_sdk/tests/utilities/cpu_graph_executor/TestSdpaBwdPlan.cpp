// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

namespace
{

// Builds a minimal single-node SDPA-backward flatbuffer graph directly, for the mask cases the
// frontend cannot express (a mask tensor of dtype UNSET, or an attn_mask_tensor_uid with no
// tensor behind it). All tensors are packed float [1, 2, 4, 8] (stats is FLOAT [1, 2, 4, 1]);
// the q/k/v/o/dO/stats/dq/dk/dv uids are filled into `attrs` here.
// Fixed UIDs: Q=1, K=2, V=3, O=4, dO=5, Stats=6, dQ=7, dK=8, dV=9, Mask=10.
//
// When `maskDataType` is engaged (any dtype, including DataType::UNSET), a [1, 1, 4, 4] mask
// tensor of that dtype is added under uid 10 and attrs.attn_mask_tensor_uid is set to 10. When
// disengaged the mask is left as `attrs` has it, so a caller-set attn_mask_tensor_uid dangles.
struct RawDbiasSpec
{
    DataType dataType;
    std::vector<int64_t> dims;
};

// When `dBias` is engaged, a dBias output tensor with that dtype and dims is added under uid 11
// and attrs.dbias_tensor_uid is set to 11.
flatbuffers::FlatBufferBuilder
    makeRawSdpaBwdGraph(SdpaBackwardAttributesT attrs,
                        std::optional<DataType> maskDataType = std::nullopt,
                        const std::optional<RawDbiasSpec>& dBias = std::nullopt,
                        const std::vector<int64_t>& statsDims = {1, 2, 4, 1},
                        DataType statsDataType = DataType::FLOAT)
{
    constexpr int64_t DBIAS_UID = 11;
    constexpr int64_t Q_UID = 1;
    constexpr int64_t K_UID = 2;
    constexpr int64_t V_UID = 3;
    constexpr int64_t O_UID = 4;
    constexpr int64_t DO_UID = 5;
    constexpr int64_t STATS_UID = 6;
    constexpr int64_t DQ_UID = 7;
    constexpr int64_t DK_UID = 8;
    constexpr int64_t DV_UID = 9;
    constexpr int64_t MASK_UID = 10;
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const auto statsStrides = hipdnn_data_sdk::utilities::generateStrides(statsDims);
    const std::vector<int64_t> maskDims = {1, 1, 4, 4};
    const auto maskStrides = hipdnn_data_sdk::utilities::generateStrides(maskDims);

    attrs.q_tensor_uid = Q_UID;
    attrs.k_tensor_uid = K_UID;
    attrs.v_tensor_uid = V_UID;
    attrs.o_tensor_uid = O_UID;
    attrs.do_tensor_uid = DO_UID;
    attrs.stats_tensor_uid = STATS_UID;
    attrs.dq_tensor_uid = DQ_UID;
    attrs.dk_tensor_uid = DK_UID;
    attrs.dv_tensor_uid = DV_UID;
    if(maskDataType.has_value())
    {
        attrs.attn_mask_tensor_uid = MASK_UID;
    }

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const auto addTensor = [&](int64_t uid,
                               const char* name,
                               DataType dataType,
                               const std::vector<int64_t>& tensorStrides,
                               const std::vector<int64_t>& tensorDims) {
        tensors.push_back(CreateTensorAttributesDirect(
            builder, uid, name, dataType, &tensorStrides, &tensorDims));
    };
    addTensor(Q_UID, "Q", DataType::FLOAT, strides, dims);
    addTensor(K_UID, "K", DataType::FLOAT, strides, dims);
    addTensor(V_UID, "V", DataType::FLOAT, strides, dims);
    addTensor(O_UID, "O", DataType::FLOAT, strides, dims);
    addTensor(DO_UID, "dO", DataType::FLOAT, strides, dims);
    addTensor(STATS_UID, "Stats", statsDataType, statsStrides, statsDims);
    addTensor(DQ_UID, "dQ", DataType::FLOAT, strides, dims);
    addTensor(DK_UID, "dK", DataType::FLOAT, strides, dims);
    addTensor(DV_UID, "dV", DataType::FLOAT, strides, dims);
    if(maskDataType.has_value())
    {
        addTensor(MASK_UID, "Mask", maskDataType.value(), maskStrides, maskDims);
    }
    std::vector<int64_t> dBiasStrides;
    if(dBias.has_value())
    {
        attrs.dbias_tensor_uid = DBIAS_UID;
        dBiasStrides = hipdnn_data_sdk::utilities::generateStrides(dBias->dims);
        addTensor(DBIAS_UID, "dBias", dBias->dataType, dBiasStrides, dBias->dims);
    }

    auto sdpaAttrs = CreateSdpaBackwardAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaBackwardAttributes,
                                     sdpaAttrs.Union()));
    auto graph = CreateGraphDirect(builder,
                                   "SdpaBwdRawGraph",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);
    builder.Finish(graph);
    return builder;
}

} // namespace

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

namespace
{

template <class MaskT>
std::vector<uint8_t> serializeSdpaBwdGraph(SdpaBwdTensorBundle<float, MaskT>& bundle,
                                           std::unordered_map<int64_t, void*>* variantPack
                                           = nullptr)
{
    auto graphTuple = buildSdpaBwdGraph(bundle, DataType::FLOAT);
    if(variantPack != nullptr)
    {
        *variantPack = std::get<1>(graphTuple);
    }
    auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    return serializedGraph;
}

// Runs the bundle's graph through the CPU reference executor (key -> registry -> plan).
template <class MaskT>
void executeSdpaBwd(SdpaBwdTensorBundle<float, MaskT>& bundle)
{
    std::unordered_map<int64_t, void*> variantPack;
    auto serializedGraph = serializeSdpaBwdGraph(bundle, &variantPack);
    CpuReferenceGraphExecutor executor;
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);
}

} // namespace

template <class MaskT>
class SdpaBwdPlanMaskDtype : public Test
{
};

using SupportedMaskTypes
    = Types<float, hipdnn_data_sdk::types::half, hipdnn_data_sdk::types::bfloat16>;
TYPED_TEST_SUITE(SdpaBwdPlanMaskDtype, SupportedMaskTypes, );

// A half/bfloat16 mask must give exactly the gradients of the same graph with a float mask
// holding the same values. Reading the mask buffer as float instead gives different gradients.
TYPED_TEST(SdpaBwdPlanMaskDtype, GradientsMatchFloatMask)
{
    const unsigned int seed = getGlobalTestSeed();

    SdpaBwdTensorBundle<float, float> floatMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS, seed);
    emplaceNonUniformMask(floatMaskBundle);
    ASSERT_NO_FATAL_FAILURE(executeSdpaBwd(floatMaskBundle));

    SdpaBwdTensorBundle<float, TypeParam> typedMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS, seed);
    emplaceNonUniformMask(typedMaskBundle);
    ASSERT_NO_FATAL_FAILURE(executeSdpaBwd(typedMaskBundle));

    const CpuFpReferenceValidation<float> exact(EXACT_TOLERANCE, EXACT_TOLERANCE);
    EXPECT_TRUE(exact.allClose(floatMaskBundle.dqTensor, typedMaskBundle.dqTensor));
    EXPECT_TRUE(exact.allClose(floatMaskBundle.dkTensor, typedMaskBundle.dkTensor));
    EXPECT_TRUE(exact.allClose(floatMaskBundle.dvTensor, typedMaskBundle.dvTensor));

    // The mask must actually influence the gradients, otherwise the comparison is vacuous.
    SdpaBwdTensorBundle<float, float> noMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS, seed);
    ASSERT_NO_FATAL_FAILURE(executeSdpaBwd(noMaskBundle));
    EXPECT_FALSE(exact.allClose(noMaskBundle.dvTensor, typedMaskBundle.dvTensor));
}

TYPED_TEST(SdpaBwdPlanMaskDtype, ExecutorAcceptsGraph)
{
    SdpaBwdTensorBundle<float, TypeParam> bundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    emplaceNonUniformMask(bundle);
    auto serializedGraph = serializeSdpaBwdGraph(bundle);

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
}

TEST(TestSdpaBwdPlanMask, ExecutorAcceptsGraphWithoutMask)
{
    SdpaBwdTensorBundle<float> bundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    auto serializedGraph = serializeSdpaBwdGraph(bundle);

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
}

TEST(TestSdpaBwdPlanMask, UnsupportedMaskDtypeHasNoPlan)
{
    SdpaBwdTensorBundle<float, int32_t> bundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    bundle.maskTensor.emplace(std::vector<int64_t>{1, 2, 4, 6});
    std::unordered_map<int64_t, void*> variantPack;
    auto serializedGraph = serializeSdpaBwdGraph(bundle, &variantPack);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));

    try
    {
        executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);
        FAIL() << "An INT32 mask must not resolve to a plan";
    }
    catch(const CpuReferenceNotApplicableError& e)
    {
        FAIL() << "Expected the registry error naming the mask dtype, got not-applicable: "
               << e.what();
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("mask=INT32"), std::string::npos) << e.what();
    }
}

TEST(TestSdpaBwdPlanMask, MaskTensorWithUnsetDtypeIsNotApplicable)
{
    auto graphBuilder = makeRawSdpaBwdGraph(SdpaBackwardAttributesT{}, DataType::UNSET);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
    EXPECT_THROW(executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), {}),
                 CpuReferenceNotApplicableError);
}

// Attributes the backward reference does not implement must make the graph not applicable rather
// than be ignored: an ignored attribute lets a provider that also ignores it pass validation.
struct UnsupportedBwdAttributeCase
{
    const char* name;
    void (*apply)(SdpaBackwardAttributesT&);
};

class TestSdpaBwdPlanUnsupportedAttribute : public TestWithParam<UnsupportedBwdAttributeCase>
{
};

TEST(TestSdpaBwdPlan, RawBaselineGraphIsApplicable)
{
    auto graphBuilder = makeRawSdpaBwdGraph(SdpaBackwardAttributesT{});

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

TEST_P(TestSdpaBwdPlanUnsupportedAttribute, IsNotApplicable)
{
    SdpaBackwardAttributesT attrs;
    GetParam().apply(attrs);
    auto graphBuilder = makeRawSdpaBwdGraph(attrs);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
    EXPECT_THROW(executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), {}),
                 CpuReferenceNotApplicableError);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestSdpaBwdPlanUnsupportedAttribute,
    Values(UnsupportedBwdAttributeCase{"AlibiMask",
                                       [](SdpaBackwardAttributesT& a) { a.alibi_mask = true; }},
           // cuDNN: padding_mask requires both seq_len tensors.
           UnsupportedBwdAttributeCase{"PaddingMaskWithoutSeqLens",
                                       [](SdpaBackwardAttributesT& a) { a.padding_mask = true; }},
           // Only the probability is set: no seed/offset/mask tensors accompany it.
           UnsupportedBwdAttributeCase{
               "DropoutProbability",
               [](SdpaBackwardAttributesT& a) { a.dropout_probability = 0.1F; }},
           // dBias has the bias tensor's shape, so it cannot be produced without a bias.
           UnsupportedBwdAttributeCase{
               "DbiasWithoutMask", [](SdpaBackwardAttributesT& a) { a.dbias_tensor_uid = 11; }}),
    [](const TestParamInfo<UnsupportedBwdAttributeCase>& info) { return info.param.name; });

// dBias must have exactly the mask's shape and dtype (cuDNN: the bias gradient is reduced back to
// the bias tensor's shape). The raw graph's mask is FLOAT [1, 1, 4, 4].
TEST(TestSdpaBwdPlanDbias, MatchingShapeAndDtypeIsApplicable)
{
    auto graphBuilder = makeRawSdpaBwdGraph(
        SdpaBackwardAttributesT{}, DataType::FLOAT, RawDbiasSpec{DataType::FLOAT, {1, 1, 4, 4}});

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

TEST(TestSdpaBwdPlanDbias, DtypeMismatchIsNotApplicable)
{
    auto graphBuilder = makeRawSdpaBwdGraph(
        SdpaBackwardAttributesT{}, DataType::FLOAT, RawDbiasSpec{DataType::HALF, {1, 1, 4, 4}});

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

TEST(TestSdpaBwdPlanDbias, ShapeMismatchIsNotApplicable)
{
    auto graphBuilder = makeRawSdpaBwdGraph(
        SdpaBackwardAttributesT{}, DataType::FLOAT, RawDbiasSpec{DataType::FLOAT, {1, 2, 4, 4}});

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

// The plan writes dBias, and it equals the reference called directly.
TEST(TestSdpaBwdPlanDbias, ExecuteMatchesDirectReference)
{
    auto graphBuilder = makeRawSdpaBwdGraph(
        SdpaBackwardAttributesT{}, DataType::FLOAT, RawDbiasSpec{DataType::FLOAT, {1, 1, 4, 4}});

    const std::vector<int64_t> dims = {1, 2, 4, 8};
    hipdnn_data_sdk::utilities::Tensor<float> q(dims);
    hipdnn_data_sdk::utilities::Tensor<float> k(dims);
    hipdnn_data_sdk::utilities::Tensor<float> v(dims);
    hipdnn_data_sdk::utilities::Tensor<float> o(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dO(dims);
    hipdnn_data_sdk::utilities::Tensor<float> stats({1, 2, 4, 1});
    hipdnn_data_sdk::utilities::Tensor<float> mask({1, 1, 4, 4});
    q.fillWithRandomValues(-1.0F, 1.0F, 91);
    k.fillWithRandomValues(-1.0F, 1.0F, 92);
    v.fillWithRandomValues(-1.0F, 1.0F, 93);
    dO.fillWithRandomValues(-1.0F, 1.0F, 94);
    mask.fillWithRandomValues(-1.0F, 1.0F, 95);
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, &mask, -1, -1, true, &stats);

    hipdnn_data_sdk::utilities::Tensor<float> dQ(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dK(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dV(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dBias({1, 1, 4, 4});
    dBias.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    const std::unordered_map<int64_t, void*> variantPack = {{1, q.memory().hostData()},
                                                            {2, k.memory().hostData()},
                                                            {3, v.memory().hostData()},
                                                            {4, o.memory().hostData()},
                                                            {5, dO.memory().hostData()},
                                                            {6, stats.memory().hostData()},
                                                            {7, dQ.memory().hostData()},
                                                            {8, dK.memory().hostData()},
                                                            {9, dV.memory().hostData()},
                                                            {10, mask.memory().hostData()},
                                                            {11, dBias.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);

    hipdnn_data_sdk::utilities::Tensor<float> dQRef(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dKRef(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dVRef(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dBiasRef({1, 1, 4, 4});
    CpuFpReferenceSdpa::backward(
        q, k, v, o, dO, dQRef, dKRef, dVRef, std::nullopt, &stats, &mask, -1, -1, true, &dBiasRef);

    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    EXPECT_TRUE(exactValidation.allClose(dBias, dBiasRef));
    EXPECT_TRUE(exactValidation.allClose(dQ, dQRef));
}

// Stats may be rank-3 [B, H, Sq] (the checked-in backward bundles use it) or rank-4
// [B, H, Sq, 1]; both give identical gradients. Stats must be FLOAT.
TEST(TestSdpaBwdPlanStats, Rank3StatsEqualRank4)
{
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    hipdnn_data_sdk::utilities::Tensor<float> q(dims);
    hipdnn_data_sdk::utilities::Tensor<float> k(dims);
    hipdnn_data_sdk::utilities::Tensor<float> v(dims);
    hipdnn_data_sdk::utilities::Tensor<float> o(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dO(dims);
    hipdnn_data_sdk::utilities::Tensor<float> stats({1, 2, 4, 1});
    q.fillWithRandomValues(-1.0F, 1.0F, 61);
    k.fillWithRandomValues(-1.0F, 1.0F, 62);
    v.fillWithRandomValues(-1.0F, 1.0F, 63);
    dO.fillWithRandomValues(-1.0F, 1.0F, 64);
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, -1, -1, true, &stats);

    std::vector<hipdnn_data_sdk::utilities::Tensor<float>> grads;
    for(const auto& statsDims : {std::vector<int64_t>{1, 2, 4, 1}, std::vector<int64_t>{1, 2, 4}})
    {
        auto graphBuilder
            = makeRawSdpaBwdGraph(SdpaBackwardAttributesT{}, std::nullopt, std::nullopt, statsDims);
        grads.emplace_back(dims);
        grads.emplace_back(dims);
        grads.emplace_back(dims);
        const auto n = grads.size();
        const std::unordered_map<int64_t, void*> variantPack
            = {{1, q.memory().hostData()},
               {2, k.memory().hostData()},
               {3, v.memory().hostData()},
               {4, o.memory().hostData()},
               {5, dO.memory().hostData()},
               {6, stats.memory().hostData()},
               {7, grads[n - 3].memory().hostData()},
               {8, grads[n - 2].memory().hostData()},
               {9, grads[n - 1].memory().hostData()}};
        CpuReferenceGraphExecutor executor;
        ASSERT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
        executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);
    }
    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    for(size_t i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(exactValidation.allClose(grads[i], grads[i + 3])) << "gradient " << i;
    }
}

TEST(TestSdpaBwdPlanStats, NonFloatOrBadShapeIsNotApplicable)
{
    CpuReferenceGraphExecutor executor;
    for(const auto& [statsDims, type] :
        {std::pair{std::vector<int64_t>{1, 2, 4, 1}, DataType::HALF},
         std::pair{std::vector<int64_t>{1, 2, 4, 2}, DataType::FLOAT},
         std::pair{std::vector<int64_t>{8}, DataType::FLOAT}})
    {
        auto graphBuilder = makeRawSdpaBwdGraph(
            SdpaBackwardAttributesT{}, std::nullopt, std::nullopt, statsDims, type);
        EXPECT_FALSE(
            executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
    }
}

TEST(TestSdpaBwdPlanMask, DanglingMaskUidIsNotApplicable)
{
    SdpaBackwardAttributesT attrs;
    attrs.attn_mask_tensor_uid = 99;
    auto graphBuilder = makeRawSdpaBwdGraph(attrs);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
    EXPECT_THROW(executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), {}),
                 std::out_of_range);
}

// A builder claims only graphs whose mask dtype is exactly its own, so the no-mask builder never
// reads a mask buffer as its placeholder float type.
TEST(TestSdpaBwdPlanMask, BuilderAcceptsOnlyMatchingMaskDtype)
{
    using hipdnn_data_sdk::types::half;

    SdpaBwdTensorBundle<float, half> halfMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    emplaceNonUniformMask(halfMaskBundle);
    auto halfMaskGraph = serializeSdpaBwdGraph(halfMaskBundle);
    const GraphWrapper halfMaskWrapper(halfMaskGraph.data(), halfMaskGraph.size());

    SdpaBwdTensorBundle<float, float> floatMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    emplaceNonUniformMask(floatMaskBundle);
    auto floatMaskGraph = serializeSdpaBwdGraph(floatMaskBundle);
    const GraphWrapper floatMaskWrapper(floatMaskGraph.data(), floatMaskGraph.size());

    SdpaBwdTensorBundle<float, float> noMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    auto noMaskGraph = serializeSdpaBwdGraph(noMaskBundle);
    const GraphWrapper noMaskWrapper(noMaskGraph.data(), noMaskGraph.size());

    const SdpaBwdPlanBuilder<DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::HALF>
        halfMaskBuilder;
    EXPECT_TRUE(
        halfMaskBuilder.isApplicable(halfMaskWrapper.getNode(0), halfMaskWrapper.getTensorMap()));
    EXPECT_FALSE(
        halfMaskBuilder.isApplicable(floatMaskWrapper.getNode(0), floatMaskWrapper.getTensorMap()));
    EXPECT_FALSE(
        halfMaskBuilder.isApplicable(noMaskWrapper.getNode(0), noMaskWrapper.getTensorMap()));

    const SdpaBwdPlanBuilder<DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT>
        noMaskBuilder;
    EXPECT_TRUE(noMaskBuilder.isApplicable(noMaskWrapper.getNode(0), noMaskWrapper.getTensorMap()));
    EXPECT_FALSE(
        noMaskBuilder.isApplicable(halfMaskWrapper.getNode(0), halfMaskWrapper.getTensorMap()));
    EXPECT_FALSE(
        noMaskBuilder.isApplicable(floatMaskWrapper.getNode(0), floatMaskWrapper.getTensorMap()));
}
